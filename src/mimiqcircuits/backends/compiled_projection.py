#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Compiled form of a projection circuit.

:func:`~mimiqcircuits.backends.measure_analysis.extract_projection`
returns the projection as a :class:`Circuit`, and
:func:`~mimiqcircuits.backends.measure_analysis.evaluate_projection`
interprets it one instruction at a time, for one shot at a time. That
is the reference semantics, and it costs one interpreted loop per shot.

:class:`CompiledProjection` compiles the same circuit once, into a
straight-line program over classical bits, and runs it over a whole
block of shots. Two tiers:

- :attr:`Tier.IDENTITY`. Bit ``b`` reads qubit ``b`` for every bit and
  the sample is exactly as wide as the register, so the samples are
  already the answer and are returned unchanged. This is the common
  case: ``measure(range, range)``, ``measure_all()``, and any circuit
  with no classical register, since ``extract_projection`` synthesises
  the identity mapping for those.
- :attr:`Tier.PROGRAM`. Everything else. The block becomes one
  ``(shots, num_bits)`` array and each operation is a single vector
  operation over its columns, so a permuted read-out, a reset-folded
  constant and a ``Xor`` tail all cost the same handful of them.

The tier depends on the width of the samples as well as on the
circuit, because a ``Measure`` reading past the end of a sample
contributes a zero rather than a qubit. It is therefore chosen the
first time a block of a given width is evaluated, and cached.

Semantics are those of ``evaluate_projection``, which stays the
reference implementation the compiled form is tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mimiqcircuits as mc

from bitarray import bitarray, frozenbitarray


__all__ = ["CompiledProjection", "Tier"]


# Rough ceiling on the unpacked block a single vectorised pass holds, in
# bytes. Blocks wider or deeper than this are evaluated in chunks, so a
# million-shot run costs bounded memory rather than a gigabyte.
_BLOCK_BYTES = 8_000_000


# ── the straight-line program ───────────────────────────────────────────────
#
# Steps are tuples tagged by their first element. Destinations are
# classical bits, and so are sources, except in a gather whose sources
# are qubits of the sample.

_GATHER = "gather"   # (dests, sites)  cstate[dests] = sample[sites]
_CONST = "const"     # (dests, values) cstate[dests] = values
_NOT = "not"         # (dest,)         cstate[dest] = !cstate[dest]
_AND = "and"         # (dest, srcs)    cstate[dest] = all(cstate[srcs])
_OR = "or"           # (dest, srcs)    cstate[dest] = any(cstate[srcs])
_XOR = "xor"         # (dest, srcs)    cstate[dest] = parity(cstate[srcs])

_FUSABLE = (_GATHER, _CONST)

# The reduction each of And / Or / Xor / ParityCheck applies to its
# sources, and the value it yields for an empty source list. The empty
# case mirrors `all(())`, `any(())` and `sum(()) % 2`.
_REDUCERS = {
    _AND: ("bitwise_and", 1),
    _OR: ("bitwise_or", 0),
    _XOR: ("bitwise_xor", 0),
}


class Tier:
    """Names of the two evaluation strategies, in increasing generality."""

    IDENTITY = "identity"
    PROGRAM = "program"


@dataclass
class _Plan:
    """A program specialised to one sample width."""

    tier: str
    steps: List[tuple]


@dataclass
class CompiledProjection:
    """A projection circuit compiled for block evaluation.

    Build one with :meth:`from_circuit` and evaluate a whole block of
    shots with :meth:`evaluate_batch`. An instance is reusable across
    blocks and holds no per-shot state.

    Attributes
    ----------
    num_bits : int
        Width of the classical bitstring the projection produces,
        equal to ``projection.num_bits()``.
    steps : list[tuple]
        The straight-line program, before specialisation to a sample
        width. Internal representation, subject to change.
    """

    num_bits: int
    steps: List[tuple] = field(default_factory=list)
    _plans: dict = field(default_factory=dict, repr=False, compare=False)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_circuit(cls, projection: "mc.Circuit") -> "CompiledProjection":
        """Compile `projection` for block evaluation.

        Arguments:
            projection (Circuit): the projection circuit returned by
                :func:`extract_projection`, in the frame its samples
                will arrive in. Qubit remapping must already have been
                applied, otherwise the tier keys off the wrong frame
                and a permuted read-out is mistaken for an identity
                one.

        Returns:
            CompiledProjection: the compiled form, of width
            ``projection.num_bits()``.

        Raises:
            ValueError: if `projection` holds an operation outside the
                classical language, that is anything other than
                ``Measure``, ``SetBit0``, ``SetBit1``, ``Not``,
                ``And``, ``Or``, ``Xor`` and ``ParityCheck``.
        """
        steps: List[tuple] = []
        for inst in projection.instructions:
            _emit(steps, inst)
        return cls(num_bits=projection.num_bits(), steps=_fuse(steps))

    # ── evaluation ──────────────────────────────────────────────────────────

    def evaluate_batch(self, samples: Sequence["mc.BitString"]) -> List["mc.BitString"]:
        """Evaluate the projection for every shot in `samples`.

        Arguments:
            samples (Sequence[BitString]): one computational-basis
                outcome per shot, all of the same width. Not quantum
                states: each is a single shot already drawn from one.

        Returns:
            list[BitString]: one classical bitstring per shot, of
            length :attr:`num_bits`, in the order the samples came in.

        On the identity tier the input bitstrings are returned as they
        are, without copying, so a caller that keeps using the samples
        afterwards must copy them itself.
        """
        if not samples:
            return []

        width = len(samples[0].bits)
        plan = self._plans.get(width)
        if plan is None:
            plan = _specialise(self.steps, self.num_bits, width)
            self._plans[width] = plan

        if plan.tier == Tier.IDENTITY:
            return list(samples)

        chunk = max(1024, _BLOCK_BYTES // max(width, self.num_bits, 1))
        if len(samples) <= chunk:
            return self._run(plan, samples)
        out: List["mc.BitString"] = []
        for start in range(0, len(samples), chunk):
            out.extend(self._run(plan, samples[start:start + chunk]))
        return out

    def evaluate(self, sample: "mc.BitString") -> "mc.BitString":
        """Evaluate the projection for a single shot.

        Provided for parity with
        :func:`~mimiqcircuits.backends.measure_analysis.evaluate_projection`.
        :meth:`evaluate_batch` is the fast path and the one drivers
        should call.
        """
        return self.evaluate_batch([sample])[0]

    # ── the program tier ────────────────────────────────────────────────────

    def _run(self, plan: _Plan, samples) -> List["mc.BitString"]:
        import numpy as np

        nb = self.num_bits
        cstates = np.zeros((len(samples), nb), dtype=np.uint8)
        block = None

        for step in plan.steps:
            kind = step[0]
            if kind == _GATHER:
                if block is None:
                    block = _samples_to_array(samples)
                cstates[:, step[1]] = block[:, step[2]]
            elif kind == _CONST:
                cstates[:, step[1]] = step[2]
            elif kind == _NOT:
                cstates[:, step[1]] ^= 1
            else:
                name, empty = _REDUCERS[kind]
                srcs = step[2]
                if not srcs:
                    cstates[:, step[1]] = empty
                    continue
                # Fancy indexing copies, so a destination that is also
                # one of its own sources reads the old value, matching
                # the reference evaluator.
                cstates[:, step[1]] = getattr(np, name).reduce(
                    cstates[:, srcs], axis=1,
                )

        return _array_to_bitstrings(cstates, nb)


# ── compilation helpers ─────────────────────────────────────────────────────


def _emit(steps: List[tuple], inst) -> None:
    """Append the step for one projection instruction."""
    op = inst.operation
    bits = inst.bits

    if isinstance(op, mc.Measure):
        steps.append((_GATHER, [bits[0]], [inst.qubits[0]]))
    elif isinstance(op, mc.SetBit0):
        steps.append((_CONST, [bits[0]], [0]))
    elif isinstance(op, mc.SetBit1):
        steps.append((_CONST, [bits[0]], [1]))
    elif isinstance(op, mc.Not):
        steps.append((_NOT, bits[0]))
    elif isinstance(op, mc.And):
        steps.append((_AND, bits[0], list(bits[1:])))
    elif isinstance(op, mc.Or):
        steps.append((_OR, bits[0], list(bits[1:])))
    elif isinstance(op, (mc.Xor, mc.ParityCheck)):
        steps.append((_XOR, bits[0], list(bits[1:])))
    else:
        raise ValueError(
            f"CompiledProjection: unsupported instruction {type(op).__name__}"
        )


def _fuse(steps: List[tuple]) -> List[tuple]:
    """Merge neighbouring gathers, and neighbouring constants, into one
    step each. A projection is mostly a run of ``Measure``, so this is
    what collapses the program tier to a handful of vector operations.
    """
    out: List[tuple] = []
    for step in steps:
        kind = step[0]
        if kind not in _FUSABLE:
            out.append(step)
            continue
        if out and out[-1][0] == kind:
            out[-1][1].extend(step[1])
            out[-1][2].extend(step[2])
            continue
        out.append((kind, list(step[1]), list(step[2])))
    return out


def _specialise(steps: List[tuple], num_bits: int, width: int) -> _Plan:
    """Resolve `steps` against a known sample width and pick a tier.

    A ``Measure`` reading past the end of the sample cannot read a
    qubit, so it becomes a constant zero, which is what leaving the bit
    untouched amounts to on a register that starts at zero.
    """
    resolved: List[tuple] = []
    for step in steps:
        if step[0] != _GATHER:
            resolved.append(step)
            continue
        dests, sites, dead = [], [], []
        for b, q in zip(step[1], step[2]):
            if 0 <= q < width:
                dests.append(b)
                sites.append(q)
            else:
                dead.append(b)
        if dests:
            resolved.append((_GATHER, dests, sites))
        if dead:
            resolved.append((_CONST, dead, [0] * len(dead)))
    resolved = _fuse(resolved)

    if width == num_bits and _is_identity_gather(resolved, num_bits):
        return _Plan(tier=Tier.IDENTITY, steps=resolved)
    return _Plan(tier=Tier.PROGRAM, steps=resolved)


def _is_identity_gather(steps: List[tuple], num_bits: int) -> bool:
    """True when `steps` is the single gather that copies qubit ``b``
    into bit ``b`` for every bit of the register."""
    if len(steps) != 1 or steps[0][0] != _GATHER:
        return False
    dests, sites = steps[0][1], steps[0][2]
    return (len(dests) == num_bits
            and dests == sites
            and sorted(dests) == list(range(num_bits)))


# ── block conversions ───────────────────────────────────────────────────────


def _samples_to_array(samples):
    """Stack a block of samples into a ``(shots, 8 * nbytes)`` array of
    0/1 bytes. Every sample has the same width, so one buffer join and
    one unpack cover the block. The columns past the sample width are
    padding and no step ever reads them.

    A `bitarray` carries its own bit order, and the two lay the same bits
    out in opposite directions within each byte, so a little-endian buffer
    read as big-endian comes back reversed. `BitString` does not normalise
    what it is handed, so a sample can hold either; one is converted here
    rather than assumed away. Everything this package produces is already
    big-endian, so the check costs one attribute read per shot and the
    conversion never runs.
    """
    import numpy as np

    raw = b"".join(_big_endian(s.bits).tobytes() for s in samples)
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(len(samples), -1)
    return np.unpackbits(packed, axis=1, bitorder="big")


def _big_endian(bits):
    """`bits` laid out most-significant-bit first, converting only if it
    is not already."""
    return bits if bits.endian == "big" else bitarray(bits, endian="big")


def _array_to_bitstrings(cstates, num_bits: int) -> List["mc.BitString"]:
    """Turn a ``(shots, num_bits)`` array of 0/1 bytes into one
    :class:`BitString` per row."""
    import numpy as np

    if num_bits == 0:
        return [mc.BitString(0) for _ in range(len(cstates))]

    packed = np.packbits(cstates, axis=1, bitorder="big")
    stride = packed.shape[1]
    raw = packed.tobytes()
    out = []
    for off in range(0, len(raw), stride):
        # Explicitly big-endian, to match `packbits` above rather than
        # inherit whatever `bitarray`'s default is in the caller's process.
        bits = bitarray(endian="big")
        bits.frombytes(raw[off:off + stride])
        del bits[num_bits:]
        out.append(mc.BitString(frozenbitarray(bits)))
    return out
