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

"""Final-block analysis. Reverse-scan a `Circuit`, absorb its trailing
"projection block" (Measures, Resets, post-measure X gates) into a
small `projection_circuit` of classical-bit instructions, and return
the remaining `quantum_circuit` plus that projection.

The two-circuit decomposition is the post-evolution analog of the
old ``MeasureInfo`` enum:

    quantum_circuit, projection_circuit = extract_projection(c)
    # evolve through quantum_circuit once
    # for each shot:
    #     sample = quantum_state.sample()
    #     cstate = evaluate_projection(projection_circuit, sample)

`projection_circuit` only contains classical-bit operations:

- ``Measure(q, b)``               — ``cstate[b] = sample[q]``
- ``SetBit0(b)`` / ``SetBit1(b)`` — bit is classically known
- ``Not(b)``                      — ``cstate[b] = !cstate[b]``
- ``And`` / ``Or`` / ``Xor`` / ``ParityCheck`` — ``bits[0]`` is the
  target, the rest are read

Evaluate it one shot at a time with `evaluate_projection`, or a whole
block at a time with `CompiledProjection`, which is the same semantics
compiled once instead of interpreted per shot.

Mirrors the Julia `AbstractQCSs.extract_projection`. The two ports
must stay behavioural-parity — if one is fixed, fix the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from bitarray import bitarray

import mimiqcircuits as mc


__all__ = [
    "extract_projection",
    "evaluate_projection",
    "needs_trajectories",
    "needs_loss_sampling",
    "any_mixed_unitary",
    "remap_projection_qubits",
]


def needs_trajectories(circuit: "mc.Circuit") -> bool:
    """Return True if `circuit` still contains any non-unitary op
    that requires per-shot evolution. Operations that don't touch
    qubits (Amplitude on a z-register, Tick, …) or that declare
    themselves unitary (Gates, AbstractAnnotation, ExpectationValue, …)
    are ignored.

    Mirrors the Julia `AbstractQCSs.needs_trajectories`:
    ``num_qubits(op) != 0 && !isunitary(op)``.
    """
    for inst in circuit.instructions:
        op = inst.operation
        if op.num_qubits == 0:
            continue
        if op.isunitary():
            continue
        return True
    return False


def needs_loss_sampling(circuit: "mc.Circuit") -> bool:
    """Return True if `circuit` contains loss operations (`Loss`, `Reload`,
    `Check`, `MeasureCheck`) that must be resolved into primitives before the
    simulator runs."""
    try:
        from mimiqcircuits.operations.losschannel import (
            Loss,
            Reload,
            Check,
            MeasureCheck,
        )
    except ImportError:
        return False
    for inst in circuit.instructions:
        if isinstance(inst.operation, (Loss, Reload, Check, MeasureCheck)):
            return True
    return False


def any_mixed_unitary(circuit: "mc.Circuit") -> bool:
    """Return True if `circuit` contains a mixed-unitary
    :class:`krauschannel` whose `ismixedunitary()` is true. Used as
    the default predicate for the per-trajectory recompile decision
    in :meth:`LocalBackend.recompile_per_trajectory`.

    Mirrors the Julia `AbstractQCSs.any_mixed_unitary`.
    """
    for inst in circuit.instructions:
        op = inst.operation
        if isinstance(op, mc.krauschannel) and op.ismixedunitary():
            return True
    return False


def remap_projection_qubits(projection: "mc.Circuit",
                            qubit_order: list[int],
                            do_remap: bool) -> "mc.Circuit":
    """Rewrite every `Measure(q, b)` instruction in `projection` so
    that ``q → qubit_order[q]``. Used by the driver when the pass
    pipeline reordered the qubits and the projection was synthesised
    in the reordered frame.

    Returns a new `Circuit`; the input is not mutated.
    """
    if not do_remap:
        return projection
    out = mc.Circuit()
    for inst in projection.instructions:
        op = inst.operation
        if isinstance(op, mc.Measure):
            q_old = inst.qubits[0]
            b = inst.bits[0]
            out.push(op, qubit_order[q_old], b)
        else:
            out.push(op, *inst.qubits, *inst.bits, *inst.zvars)
    return out


# ── public API ──────────────────────────────────────────────────────────────


def extract_projection(circuit: "mc.Circuit") -> Tuple["mc.Circuit", "mc.Circuit"]:
    """Reverse-scan `circuit`, absorbing every trailing operation that
    does not affect the qubit observables of the post-evolution state
    into a `projection_circuit` of classical-bit instructions. The
    remaining operations are returned as `quantum_circuit` — the part
    the simulator must evolve.

    Absorbed operations:

    - Trailing ``Measure(q, b)`` / ``MeasureReset(q, b)`` → ``Measure(q, b)``.
    - Trailing classical logic: ``Not``, ``And``, ``Or``, ``Xor``,
      ``ParityCheck``, ``SetBit0``, ``SetBit1``. These read other
      classical bits, so what comes out is a straight-line program over
      the register rather than a per-bit map, emitted in source order.
      A logic op is absorbed only when no operation left in
      ``quantum_circuit`` touches any of its bits, otherwise moving it
      past that operation would change what it reads.
    - Trailing ``GateID`` — dropped as a no-op. ``GateX`` / ``GateY``
      / ``GateZ`` are NOT absorbed: ``Y`` and ``Z`` carry phases that
      would corrupt amplitude lookups, and ``X`` absorbed alone would
      require XOR-ing the qubit-flip pattern into every user-supplied
      ``bitstrings`` entry for amplitudes to stay consistent. Keeping
      all Paulis in ``quantum_circuit`` means the projection only
      contains phase-free transformations and amplitude lookups need
      no compensation.
    - ``Reset(q)`` whose qubit has captured pending bits → those bits
      become classical constants (``SetBit0`` / ``SetBit1`` based on
      absorbed X parity). A ``Reset`` whose qubit has no captured
      bits is harmless and dropped.

    Anything that cannot be absorbed (other gates, Kraus channels,
    ``IfStatement``, ``Amplitude``, ``ExpectationValue``, …) blocks
    its qubits and survives into ``quantum_circuit``.

    If the source has no classical register, the projection is
    synthesised over an identity bit↔qubit mapping (bit ``i`` mirrors
    qubit ``i``, length = ``circuit.num_qubits()``).

    A write to a classical bit that a later absorbed operation
    overwrites without reading is dropped from the projection, so
    reusing a classical bit does not cost a per-shot write that nothing
    observes.
    """
    nq = circuit.num_qubits()
    nm = circuit.num_bits()
    nb_eff = nq if nm == 0 else nm

    qstates: List[_QubitTailState] = [_QubitTailState() for _ in range(nq)]
    if nm == 0:
        # No classical register: synthesise bit q ← qubit q. These come
        # from no instruction, so they sort ahead of everything real.
        for q in range(nq):
            qstates[q].pending.append((-1, q))

    # A bit is blocked once an operation that stays in `quantum_circuit`
    # touches it. Nothing absorbed may then read or write it, because
    # the projection runs after everything that was kept.
    bit_blocked = [False] * nb_eff
    classical: List[Tuple[int, "mc.Operation", tuple]] = []
    insts = list(circuit.instructions)
    last_kept_idx = -1

    for i in range(len(insts) - 1, -1, -1):
        inst = insts[i]
        op = inst.operation
        qs = inst.qubits
        bs = inst.bits

        # Case 1: writing op (Measure, MeasureReset, Not, Xor, IfStatement, ...).
        if _is_writing_op(op):
            if isinstance(op, (mc.Measure, mc.MeasureReset)) and \
               _try_absorb_measurement(i, op, qs[0], bs[0], qstates, bit_blocked):
                continue

            if isinstance(op, _CLASSICAL_OPS) and \
               _try_absorb_classical(i, op, bs, bit_blocked, nb_eff, classical):
                continue

            last_kept_idx = max(last_kept_idx, i)
            for q in qs:
                qstates[q].blocked = True
            for b in bs:
                if 0 <= b < nb_eff:
                    bit_blocked[b] = True
            continue

        # Case 2: trailing X (absorbable, phase-free) or ID (no-op).
        if qs and _try_absorb_gate(op, qs[0], qstates):
            continue

        # Case 3: Reset on a qubit with captured pending bits → promote.
        if isinstance(op, mc.Reset) and qs:
            q = qs[0]
            if not qstates[q].blocked:
                _const_promote(qstates[q])
                continue

        # Case 4: zero-qubit op or idle qubits → drop, but only when
        # the op is unitary. A Kraus channel on an idle qubit can
        # still mix probabilities on entangled measured qubits, so
        # non-unitary ops must survive into `quantum_circuit`.
        all_idle = True
        for q in qs:
            if qstates[q].pending or qstates[q].blocked:
                all_idle = False
                break
        if all_idle and op.isunitary():
            continue

        # Case 5: non-absorbable — block its qubits and keep it.
        last_kept_idx = max(last_kept_idx, i)
        _block_qubits(qstates, qs)

    for q in range(nq):
        _finalise_pending(qstates[q], q)

    # Build quantum_circuit from the surviving prefix. Those instructions
    # came out of a valid circuit, so hand the slice to the constructor
    # instead of re-validating each one through `push`.
    kept = insts[: last_kept_idx + 1] if last_kept_idx >= 0 else []
    quantum_circuit = mc.Circuit(kept)

    # Any captured measurement whose qubit never appears in
    # `quantum_circuit` is provably reading |0⟩ — const-promote.
    circuit_qubits: set[int] = set()
    for inst in kept:
        circuit_qubits.update(inst.qubits)
    for q in range(nq):
        if q not in circuit_qubits:
            _force_const_promote(qstates[q])

    # Absorbed operations keep their source order, since the logic ops
    # read bits the measurements write.
    entries = [pd.entry() for q in range(nq) for pd in qstates[q].done]
    entries.extend((i, op, (), bits) for i, op, bits in classical)
    entries.sort(key=lambda e: e[0])

    projection_circuit = mc.Circuit()
    for _, op, qubits, bits in _drop_dead_writes(entries):
        projection_circuit.push(op, *qubits, *bits)

    return quantum_circuit, projection_circuit


def evaluate_projection(projection: "mc.Circuit",
                        sample: "mc.BitString") -> "mc.BitString":
    """Run the projection circuit for one shot.

    Arguments:
        projection (Circuit): the projection circuit returned by
            :func:`extract_projection`.
        sample (BitString): one computational-basis outcome, one bit
            per qubit of the evolved state. Not a quantum state: it is
            a single shot already drawn from it.

    Returns:
        BitString: the classical register after the projection, of
        length ``projection.num_bits()``.

    Raises:
        ValueError: if `projection` holds an instruction outside the
            classical language listed below.

    The classical language is ``Measure``, ``SetBit0``, ``SetBit1``,
    ``Not``, ``And``, ``Or``, ``Xor`` and ``ParityCheck``. The logic
    ops take their target as the first bit and read the rest, and the
    whole right-hand side is evaluated before the target is written,
    so a target that is also one of its own operands reads its old
    value.

    A ``Measure(q, b)`` whose qubit falls outside `sample` leaves bit
    ``b`` untouched, which is how a projection that references a qubit
    the state never spanned reads back as zero.

    This is the single-shot reference implementation of the projection
    semantics. :class:`~mimiqcircuits.backends.compiled_projection.CompiledProjection`
    is the fast path, and is tested against this.
    """
    nb = projection.num_bits()
    # Accumulate in a mutable bitarray and freeze once: BitString is
    # immutable, so writing bits through it reallocates per write.
    cstate = bitarray(nb)
    cstate.setall(0)
    src = sample.bits
    nq_sample = len(src)
    for inst in projection.instructions:
        op = inst.operation
        if isinstance(op, mc.Measure):
            q = inst.qubits[0]
            b = inst.bits[0]
            if 0 <= q < nq_sample:
                cstate[b] = src[q]
        elif isinstance(op, mc.Not):
            b = inst.bits[0]
            cstate[b] = not cstate[b]
        elif isinstance(op, mc.SetBit0):
            cstate[inst.bits[0]] = 0
        elif isinstance(op, mc.SetBit1):
            cstate[inst.bits[0]] = 1
        elif isinstance(op, mc.And):
            bits = inst.bits
            cstate[bits[0]] = int(all(cstate[b] for b in bits[1:]))
        elif isinstance(op, mc.Or):
            bits = inst.bits
            cstate[bits[0]] = int(any(cstate[b] for b in bits[1:]))
        elif isinstance(op, (mc.Xor, mc.ParityCheck)):
            bits = inst.bits
            cstate[bits[0]] = sum(cstate[b] for b in bits[1:]) % 2
        else:
            raise ValueError(
                f"evaluate_projection: unsupported instruction {type(op).__name__}"
            )
    return mc.BitString(cstate)


# ── internal types and helpers ───────────────────────────────────────────────


# The classical logic the projection can carry. `bits[0]` is the target
# of each and the rest are read, which is what `_projection_deps` relies
# on. `Measure` is absorbed by its own path, since it reads a qubit.
_CLASSICAL_OPS = (
    mc.SetBit0, mc.SetBit1, mc.Not, mc.And, mc.Or, mc.Xor, mc.ParityCheck,
)


@dataclass
class _PendingDone:
    index: int                 # position in the source circuit, for ordering
    bit: int
    direct_qubit: int          # -1 sentinel for "classical constant"
    const_value: int           # 0 / 1, only consulted when direct_qubit == -1

    def entry(self) -> Tuple[int, "mc.Operation", tuple, tuple]:
        """This record as an ``(index, operation, qubits, bits)`` entry."""
        if self.direct_qubit >= 0:
            return (self.index, mc.Measure(), (self.direct_qubit,), (self.bit,))
        op = mc.SetBit1() if self.const_value else mc.SetBit0()
        return (self.index, op, (), (self.bit,))


@dataclass
class _QubitTailState:
    blocked: bool = False
    # (source index, bit index) pairs, so the projection can be emitted
    # in source order once the whole scan is done.
    pending: List[Tuple[int, int]] = field(default_factory=list)
    done: List[_PendingDone] = field(default_factory=list)


def _try_absorb_measurement(i: int, op, q: int, b: int,
                            qstates: List[_QubitTailState],
                            bit_blocked: List[bool]) -> bool:
    qs = qstates[q]
    if not bit_blocked[b] and not qs.blocked:
        if isinstance(op, mc.MeasureReset):
            # Any bits captured *later* in forward time read a |0⟩
            # register; const-promote them.
            _const_promote(qs)
        qs.pending.append((i, b))
        return True
    return False


def _try_absorb_classical(i: int, op, bits, bit_blocked: List[bool],
                          nb_eff: int, classical: List[tuple]) -> bool:
    """Absorb a classical logic operation into the projection.

    It touches no qubit, so the only thing that can stop it is an
    operation left in `quantum_circuit` that touches one of its bits:
    the projection runs after all of those, so moving it there would
    change what it reads or what reads it.
    """
    for b in bits:
        if 0 <= b < nb_eff and bit_blocked[b]:
            return False
    classical.append((i, op, tuple(bits)))
    return True


def _try_absorb_gate(op, q: int, qstates: List[_QubitTailState]) -> bool:
    """Absorb only `GateID` (true no-op). All non-trivial Paulis
    (X, Y, Z) stay in `quantum_circuit` so amplitude lookups need
    no compensation; the projection circuit is then guaranteed
    phase-free.
    """
    qs = qstates[q]
    if qs.blocked:
        return False
    if isinstance(op, mc.GateID):
        return True
    return False


def _const_promote(qs: _QubitTailState) -> None:
    for i, b in qs.pending:
        qs.done.append(
            _PendingDone(index=i, bit=b, direct_qubit=-1, const_value=0)
        )
    qs.pending.clear()


def _finalise_pending(qs: _QubitTailState, q: int) -> None:
    for i, b in qs.pending:
        qs.done.append(
            _PendingDone(index=i, bit=b, direct_qubit=q, const_value=0)
        )
    qs.pending.clear()


def _block_qubits(qstates: List[_QubitTailState], qubits) -> None:
    for q in qubits:
        qs = qstates[q]
        if not qs.blocked:
            _finalise_pending(qs, q)
            qs.blocked = True


def _force_const_promote(qs: _QubitTailState) -> None:
    for i, pd in enumerate(qs.done):
        if pd.direct_qubit >= 0:
            qs.done[i] = _PendingDone(
                index=pd.index, bit=pd.bit, direct_qubit=-1, const_value=0,
            )


def _is_writing_op(op) -> bool:
    """True if `op` writes a classical bit or a z-variable.

    The op's own footprint comes first. Wrappers such as
    ``ExpectationValue``, ``PairMeasure`` and ``IfStatement`` write a bit
    or a z-var while wrapping an operator that writes nothing, so
    unwrapping them first would report no write and let the projection
    absorb operations that must not move past them.
    """
    if op.num_bits > 0 or op.num_zvars > 0:
        return True
    if op.iswrapper():
        return _is_writing_op(op.op)
    return False


def _projection_deps(op, bits) -> Tuple[int, tuple]:
    """The ``(target, operands)`` of one projection operation.

    Every operation the projection can hold writes ``bits[0]``.
    ``Not`` reads it back, the logic ops read ``bits[1:]``, and
    ``Measure`` / ``SetBit0`` / ``SetBit1`` read no classical bit at
    all.
    """
    if isinstance(op, mc.Not):
        return bits[0], (bits[0],)
    if isinstance(op, (mc.And, mc.Or, mc.Xor, mc.ParityCheck)):
        return bits[0], tuple(bits[1:])
    return bits[0], ()


def _drop_dead_writes(entries: List[tuple]) -> List[tuple]:
    """Remove writes that a later entry overwrites without reading.

    Every bit of the projection is part of its result, so a write is
    dead only when another entry rewrites it first. Ordinary
    projections write each bit once and nothing is dropped; this pays
    off on circuits that reuse classical bits.
    """
    live = {bits[0] for _, _, _, bits in entries}
    kept = []
    for entry in reversed(entries):
        _, op, _, bits = entry
        target, operands = _projection_deps(op, bits)
        if target not in live:
            continue
        live.discard(target)
        live.update(operands)
        kept.append(entry)
    kept.reverse()
    return kept
