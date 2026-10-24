#
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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
"""Pauli-based computation pass."""

from itertools import product

import numpy as np
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.decomposition.abstract import DecompositionBasis
from mimiqcircuits.decomposition.basis import CliffordTBasis
from mimiqcircuits.decomposition.decompose import _to_basis
from mimiqcircuits.matrices import pauli_string_matrix


def pauli_based_computation(circuit, basis=None, magic_states=False):
    r"""Rewrite a circuit in Pauli-based computation form.

    The result is a sequence of :math:`\pi/8` Pauli product rotations
    (:class:`RPauli` with angle :math:`\pm\pi/4`) and Pauli product
    measurements (:class:`MeasurePauli`), with no Clifford gates left.

    The circuit is first decomposed with ``basis``, which must produce
    Clifford+T gates, the measurements and resets listed below, and the
    conditional operations described below. Those measurements, resets and
    conditional Pauli products are kept whole, whatever ``basis`` does with
    them. Each Clifford :math:`C` is then moved past every later T gate and
    measurement, conjugating them: a T gate
    on qubit :math:`q` becomes the rotation
    :math:`\exp(-i \frac{\pi}{8} C^\dagger Z_q C)`, and a measurement of
    :math:`Z_q` becomes a measurement of :math:`C^\dagger Z_q C`. When the
    conjugated observable has a minus sign, the rotation angle is negated, or
    the measured bit is flipped with :class:`Not`.

    The Cliffords left after the last measurement do not change any outcome
    and are dropped. The result therefore reproduces the measurement
    statistics of ``circuit``, not its final quantum state, and each rotation
    is exact up to a global phase. Classical operations and annotations are
    kept in place; barriers and identity gates are dropped.

    Measurements, resets and conditional operations:
        - :class:`Measure`, :class:`MeasureX`, :class:`MeasureY`,
          :class:`MeasureXX`, :class:`MeasureYY`, :class:`MeasureZZ` and
          :class:`MeasurePauli` measure a Pauli product :math:`P` and become
          a ``MeasurePauli`` of :math:`C^\dagger P C`.
        - A reset in the Z, X or Y basis (:class:`Reset`, :class:`ResetX`,
          :class:`ResetY`) measures :math:`Z`, :math:`X` or :math:`Y` into a
          new bit, then applies, if the outcome is 1, the Pauli that maps the
          :math:`-1` eigenstate to the :math:`+1` one: :math:`X` for the Z and
          Y bases, :math:`Z` for the X basis. The measure-and-reset
          operations (:class:`MeasureReset`, :class:`MeasureResetX`,
          :class:`MeasureResetY`) do the same with the bit they are given.
        - An :class:`IfStatement` whose body is made only of Pauli operators
          applies one Pauli product :math:`P` when its condition holds. The
          body can be ``GateX``, ``GateY``, ``GateZ``, ``GateID``, a
          :class:`PauliString`, or a :class:`Block`, :class:`Parallel`,
          :class:`Repeat` or :class:`GateCall` holding only such operations,
          at any depth. It becomes one ``IfStatement`` of the ``PauliString``
          :math:`C^\dagger P C` with the same condition. The phase of
          :math:`P` is dropped, as it only multiplies one classical branch.
        - A conditional T or T† gate becomes a conditional ``RPauli``
          rotation with the same condition.

        Any other conditional operation is refused. After a conditional
        Clifford gate, the accumulated Clifford would depend on the condition
        bits, so the later rotations and measurements could not be computed
        before the circuit runs.

    Magic-state injection:
        With ``magic_states=True`` the rotations are removed as well. Each
        :math:`\pi/8` rotation on :math:`P` consumes a magic state
        :math:`T\ket{+}`, prepared with :class:`PrepareStateT` on one
        ancilla qubit that is reused for every rotation:

        1. :class:`MeasurePauli` of :math:`P \otimes Z` on the data qubits and
           the ancilla, with outcome :math:`m_1`.
        2. :class:`MeasurePauli` of :math:`X` on the ancilla if
           :math:`m_1 = 0`, of :math:`Y` if :math:`m_1 = 1` (swapped for a
           negative angle), with outcome :math:`m_2`. The choice is made by
           two :class:`IfStatement` operations.

        When :math:`m_2 = 1` the rotation is applied together with a Pauli
        byproduct :math:`P`. Byproducts are not corrected with gates. Instead,
        their product :math:`F` is tracked in classical bits, the Pauli frame:
        one bit for the X part and one for the Z part of :math:`F` on each
        data qubit. When :math:`m_2 = 1`, :class:`Xor` operations multiply the
        frame by :math:`P`. Each later :math:`m_1` or measurement outcome
        whose Pauli anticommutes with :math:`F` is flipped in place, again
        with ``Xor``, so the bits hold the outcomes of the ideal circuit.

        A conditional Pauli, including the correction of a reset, is not
        applied either: it multiplies the frame when its condition holds,
        with the same ``Xor`` operations. A condition other than a single bit
        equal to 1 is first evaluated into one extra bit with
        :class:`SetBit1` and 3-bit :class:`And` operations; each condition
        bit compared with 0 is flipped with :class:`Not` while it is read and
        flipped back right after. A conditional T or T† gate is refused,
        since a magic state cannot be injected conditionally.

        Every classical operation added is therefore a 3-bit ``Xor`` or
        ``And``, a ``Not`` or a ``SetBit1``, whatever the size of the
        circuit. The result contains only Pauli product measurements,
        magic-state preparations and classical logic.

    Bit layout:
        Qubits and bits are counted from 0. With :math:`n_b` the number of
        bits of the decomposed circuit and :math:`r` its number of resets,
        the :math:`j`-th reset (counting from 0) writes its outcome to bit
        :math:`n_b + j`. With magic states, let :math:`n'_b = n_b + r`,
        :math:`n` be the number of qubits and :math:`t` the number of T and
        T† gates of the decomposed circuit. The ancilla is qubit :math:`n`.
        The :math:`k`-th rotation (counting from 0) writes :math:`m_1` to bit
        :math:`n'_b + 2k` and :math:`m_2` to bit :math:`n'_b + 2k + 1`. The
        frame bits of qubit :math:`q` follow: the X part at bit
        :math:`n'_b + 2t + q` and the Z part at bit :math:`n'_b + 2t + n + q`.
        The condition of a conditional Pauli, when it needs its own bit, goes
        to bit :math:`n'_b + 2t + 2n`.

    Args:
        circuit (Circuit): the circuit to rewrite.
        basis (DecompositionBasis, optional): the basis used for the first
            decomposition. Defaults to :class:`CliffordTBasis`.
        magic_states (bool, optional): replace every rotation by magic-state
            injection. Defaults to ``False``.

    Returns:
        Circuit: a new circuit made of :class:`RPauli`, :class:`MeasurePauli`,
        conditional ``PauliString`` and ``RPauli`` operations, classical
        operations and annotations. With ``magic_states=True``,
        :class:`PrepareStateT`, conditional ``MeasurePauli`` operations and
        classical logic replace the rotations and conditional Paulis.

    Raises:
        ValueError: if the decomposed circuit contains anything other than
            Clifford+T gates, the measurements and resets listed above, the
            conditional operations listed above, classical operations and
            annotations: for example a non-Clifford gate, a conditional
            Clifford gate or a conditional measurement. With
            ``magic_states=True``, a conditional T or T† gate also raises a
            ``ValueError``.
        DecompositionError: if ``basis`` cannot decompose an operation, such
            as a noise channel.

    Examples:

        Prepare a Bell pair, apply a T gate to qubit 1 and measure both
        qubits:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateCX(), 0, 1)
        >>> _ = c.push(GateT(), 1)
        >>> _ = c.push(Measure(), range(2), range(2))
        >>> pauli_based_computation(c)
        2-qubit, 2-bit circuit with 3 instructions:
        ├── R("XZ", (1/4)*pi) @ q[0:1]
        ├── MP(X) @ q[0], c[0]
        └── MP(XZ) @ q[0:1], c[1]
        <BLANKLINE>

        The Clifford before the T gate is :math:`C = \mathrm{CX}_{01} H_0`, so
        the T gate becomes a :math:`\pi/8` rotation on
        :math:`C^\dagger Z_1 C = X_0 Z_1`. The two measurements of :math:`Z_0`
        and :math:`Z_1` become measurements of :math:`X_0` and
        :math:`X_0 Z_1`, and write to the same bits as before.

        With magic states, the rotation becomes an injection on the ancilla
        qubit 2, whose outcomes :math:`m_1` and :math:`m_2` go to the new bits
        2 and 3:

        >>> pauli_based_computation(c, magic_states=True)
        3-qubit, 8-bit circuit with 8 instructions:
        ├── PrepareStateT @ q[2]
        ├── MP(XZZ) @ q[0:2], c[2]
        ├── IF(c==0) MP(X) @ q[2], c[3], condition[2]
        ├── IF(c==1) MP(Y) @ q[2], c[3], condition[2]
        ├── c[4] ^= c[3]
        ├── c[7] ^= c[3]
        ├── MP(X) @ q[0], c[0]
        └── MP(XZ) @ q[0:1], c[1]
        <BLANKLINE>

        The ancilla is measured in X or Y depending on :math:`m_1` (bit 2).
        The byproduct :math:`X_0 Z_1` goes into the frame: bit 4 holds its X
        part on qubit 0 and bit 7 its Z part on qubit 1. The final
        measurements commute with it, so their bits are not corrected.

        A conditional Pauli and a reset. The Hadamard gate before the
        conditional :math:`Z_1` turns it into a conditional :math:`X_1`, and
        the reset of qubit 0 after its Hadamard gate measures :math:`X_0`
        into the new bit 3, then applies :math:`Z_0` if the outcome is 1:

        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(Measure(), 0, 0)
        >>> _ = c.push(GateH(), 1)
        >>> _ = c.push(IfStatement(GateZ(), BitString("1")), 1, 0)
        >>> _ = c.push(Reset(), 0)
        >>> _ = c.push(Measure(), range(2), range(1, 3))
        >>> pauli_based_computation(c)
        2-qubit, 4-bit circuit with 6 instructions:
        ├── MP(X) @ q[0], c[0]
        ├── IF(c==1) X @ q[1], condition[0]
        ├── MP(X) @ q[0], c[3]
        ├── IF(c==1) Z @ q[0], condition[3]
        ├── MP(X) @ q[0], c[1]
        └── MP(X) @ q[1], c[2]
        <BLANKLINE>

    References:
        - D. Litinski, "A Game of Surface Codes: Large-Scale Quantum Computing
          with Lattice Surgery", Quantum 3, 128 (2019).
        - S. Bravyi, G. Smith, J. A. Smolin, "Trading Classical and Quantum
          Computational Resources", Phys. Rev. X 6, 021043 (2016).

    See Also:
        :class:`MeasurePauli`, :class:`PrepareStateT`, :class:`RPauli`,
        :class:`CliffordTBasis`
    """
    if basis is None:
        basis = CliffordTBasis()

    c = mc.decompose(circuit, basis=_PBCBasis(_to_basis(basis)))
    n = c.num_qubits()

    # Each reset measures into its own new bit, after the bits of the circuit.
    nb = c.num_bits()
    resetbit = nb
    nb += sum(isinstance(inst.operation, _PBC_RESETS) for inst in c)

    # Magic-state injection: one reused ancilla after the data qubits. Rotation
    # k writes its outcomes to bits nb+2k and nb+2k+1, and the Pauli frame
    # follows the last rotation's bits: fx[q] at nb+2t+q and fz[q] at nb+2t+n+q.
    # The bit after the frame holds the condition of a conditional Pauli.
    t = sum(isinstance(inst.operation, (mc.GateT, mc.GateTDG)) for inst in c)
    anc = n
    frame = _PauliFrame(nb + 2 * t, n)
    condbit = nb + 2 * t + 2 * n
    rot = 0

    # Heisenberg tableau of the Cliffords seen so far, C: tx[q] = C†XqC, tz[q] = C†ZqC.
    tx = [(1 << q, 0, 0) for q in range(n)]
    tz = [(0, 1 << q, 0) for q in range(n)]
    cache = {}
    out = mc.Circuit()

    def conditional_pauli(Q, cbits, pattern):
        # A Pauli P applied when the bits cbits match pattern. Moving it before
        # the Cliffords turns it into Q = C†PC, whose sign does not matter.
        x, z, _ = Q
        if not (x or z):
            return

        if not magic_states:
            qs, P, _ = _pauli_string(Q)
            out.push(mc.IfStatement(mc.PauliString(P), pattern), *qs, *cbits)
            return

        # With magic states, Q goes into the Pauli frame, multiplied in when the
        # condition holds. A condition other than a single bit equal to 1 is
        # first reduced to condbit with 3-bit And; a bit expected to be 0 is
        # flipped just for the time it is read.
        if len(cbits) == 1 and pattern[0]:
            frame.update(out, Q, cbits[0])
            return
        out.push(mc.SetBit1(), condbit)
        for b, v in zip(cbits, pattern):
            if not v:
                out.push(mc.Not(), b)
            out.push(mc.And(), condbit, condbit, b)
            if not v:
                out.push(mc.Not(), b)
        frame.update(out, Q, condbit)

    for inst in c:
        op = inst.operation
        m = _pauli_measurement(op)

        if isinstance(op, (mc.GateT, mc.GateTDG)) and magic_states:
            pauli = tz[inst.qubits[0]]
            qs, P, sign = _pauli_string(pauli)
            m1, m2 = nb + 2 * rot, nb + 2 * rot + 1
            rot += 1

            out.push(mc.PrepareStateT(), anc)
            out.push(mc.MeasurePauli(P + "Z"), *qs, anc, m1)

            # A byproduct that anticommutes with P flips m1.
            frame.correct(out, pauli, m1)

            # The X measurement completes exp(-iπ/8·P) when the corrected m1 is 0
            # and the Y measurement when it is 1; the opposite rotation, from a
            # T† or a negative sign, swaps the two.
            positive = isinstance(op, mc.GateT) == (sign > 0)
            bx, by = ("0", "1") if positive else ("1", "0")
            out.push(
                mc.IfStatement(mc.MeasurePauli("X"), mc.BitString(bx)), anc, m2, m1
            )
            out.push(
                mc.IfStatement(mc.MeasurePauli("Y"), mc.BitString(by)), anc, m2, m1
            )

            # Outcome 1 leaves the Pauli byproduct P on the data.
            frame.update(out, pauli, m2)

        elif isinstance(op, (mc.GateT, mc.GateTDG)):
            qs, P, sign = _pauli_string(tz[inst.qubits[0]])
            positive = isinstance(op, mc.GateT) == (sign > 0)
            theta = se.pi / 4 if positive else -se.pi / 4
            out.push(mc.RPauli(mc.PauliString(P), theta), *qs)

        elif m is not None:
            # A reset measures into a new bit, a measure-and-reset into its own
            # bit; both then apply, if the outcome is 1, the Pauli that maps the
            # -1 eigenstate back to the +1 one.
            observable, correction = m
            qs = inst.qubits
            if isinstance(op, _PBC_RESETS):
                b = resetbit
                resetbit += 1
            else:
                b = inst.bits[0]
            pauli = _through_tableau(1, observable, qs, tx, tz)
            mqs, P, sign = _pauli_string(pauli)
            out.push(mc.MeasurePauli(P), *mqs, b)
            frame.correct(out, pauli, b)
            if sign < 0:
                out.push(mc.Not(), b)
            if correction is not None:
                Q = _through_tableau(1, ((0, correction),), qs, tx, tz)
                conditional_pauli(Q, (b,), mc.BitString("1"))

        elif isinstance(op, (mc.GateID, mc.Barrier)):
            continue

        elif isinstance(op, mc.Gate):
            # Appending G to C gives (GC)†P(GC) = C†(G†PG)C: map the local image
            # G†PG through the current tableau. Reading the image of every
            # generator also checks that G is a Clifford.
            qs = list(inst.qubits)
            k = len(qs)
            images = []
            for i in range(k):
                row = []
                for p in ("X", "Z"):
                    s, img = _cached_conjugate(cache, op, i, p)
                    row.append(_through_tableau(s, img, qs, tx, tz))
                images.append(row)
            for i, q in enumerate(qs):
                tx[q], tz[q] = images[i]

        elif isinstance(op, (mc.AbstractClassical, mc.AbstractAnnotation)):
            out.push(inst)

        elif isinstance(op, mc.IfStatement):
            body = op.op
            pattern = op.bitstring
            cbits = inst.bits[body.num_bits :]
            local_pauli = _as_pauli(body)
            is_t = isinstance(body, (mc.GateT, mc.GateTDG))

            if local_pauli is not None:
                pairs = _pauli_pairs(*local_pauli)
                Q = _through_tableau(1, pairs, inst.qubits, tx, tz)
                conditional_pauli(Q, cbits, pattern)

            elif is_t and not magic_states:
                qs, P, sign = _pauli_string(tz[inst.qubits[0]])
                positive = isinstance(body, mc.GateT) == (sign > 0)
                theta = se.pi / 4 if positive else -se.pi / 4
                rotation = mc.RPauli(mc.PauliString(P), theta)
                out.push(mc.IfStatement(rotation, pattern), *qs, *cbits)

            elif is_t:
                raise ValueError(
                    "Pauli-based computation with magic_states=True does not "
                    f"support a classically controlled {body} gate: a magic "
                    "state cannot be injected conditionally. Use "
                    "magic_states=False to keep it as a conditional rotation."
                )

            elif isinstance(body, mc.Gate):
                raise ValueError(
                    "Pauli-based computation does not support a classically "
                    f"controlled {body} gate: the Clifford accumulated after "
                    "it would depend on the condition bits, so the later rotations "
                    "and measurements cannot be computed statically. Only "
                    "conditional Pauli operators and conditional T or T† gates "
                    "are supported."
                )

            else:
                raise ValueError(
                    "Pauli-based computation does not support a classically "
                    f"controlled {body.name}. Only conditional Pauli operators "
                    "and conditional T or T† gates are supported."
                )

        else:
            raise ValueError(f"Pauli-based computation does not support {op.name}.")

    return out


class _PBCBasis(DecompositionBasis):
    """Decomposes with ``basis``, but keeps whole the operations that the pass
    handles directly: Pauli measurements and resets, which need no expansion
    into Clifford gates, and conditional Pauli products, which ``basis`` could
    split into several conditional gates."""

    def __init__(self, basis):
        self.basis = basis

    def isterminal(self, op):
        if _pauli_measurement(op) is not None:
            return True
        if isinstance(op, mc.IfStatement) and _as_pauli(op.op) is not None:
            return True
        return self.basis.isterminal(op)

    def decompose(self, op, qubits, bits, zvars):
        return self.basis.decompose(op, qubits, bits, zvars)


_PBC_RESETS = (mc.Reset, mc.ResetZ, mc.ResetX, mc.ResetY)

# The Pauli measured by a measurement or a reset, as (qubit, letter) pairs over
# its qubits, and the letter of the Pauli that maps the -1 eigenstate to the +1
# one, applied after a reset when the outcome is 1 (None for a measurement).
_PAULI_MEASUREMENTS = {
    mc.Measure: (((0, "Z"),), None),
    mc.MeasureZ: (((0, "Z"),), None),
    mc.MeasureX: (((0, "X"),), None),
    mc.MeasureY: (((0, "Y"),), None),
    mc.MeasureXX: (((0, "X"), (1, "X")), None),
    mc.MeasureYY: (((0, "Y"), (1, "Y")), None),
    mc.MeasureZZ: (((0, "Z"), (1, "Z")), None),
    mc.Reset: (((0, "Z"),), "X"),
    mc.ResetZ: (((0, "Z"),), "X"),
    mc.MeasureReset: (((0, "Z"),), "X"),
    mc.MeasureResetZ: (((0, "Z"),), "X"),
    mc.ResetX: (((0, "X"),), "Z"),
    mc.MeasureResetX: (((0, "X"),), "Z"),
    mc.ResetY: (((0, "Y"),), "X"),
    mc.MeasureResetY: (((0, "Y"),), "X"),
}


def _pauli_measurement(op):
    if isinstance(op, mc.MeasurePauli):
        return tuple((i, p) for i, p in enumerate(op.pstring()) if p != "I"), None
    return _PAULI_MEASUREMENTS.get(type(op))


def _as_pauli(op):
    """The Pauli product applied by an operation made only of Pauli operators,
    as its X and Z bit sets over the operation's qubits, or None for any other
    operation. The phase is dropped."""
    if isinstance(op, mc.GateID):
        return 0, 0
    if isinstance(op, mc.GateX):
        return 1, 0
    if isinstance(op, mc.GateY):
        return 1, 1
    if isinstance(op, mc.GateZ):
        return 0, 1
    if isinstance(op, mc.PauliString):
        x = sum(1 << q for q, p in enumerate(op.pauli) if p in "XY")
        z = sum(1 << q for q, p in enumerate(op.pauli) if p in "YZ")
        return x, z
    if isinstance(op, mc.Parallel):
        p = _as_pauli(op.op)
        if p is None:
            return None
        w = op.op.num_qubits
        x = sum(p[0] << (w * i) for i in range(op.num_repeats))
        z = sum(p[1] << (w * i) for i in range(op.num_repeats))
        return x, z
    if isinstance(op, mc.Repeat):
        p = _as_pauli(op.op)
        if p is None:
            return None
        return p if op.repeats % 2 == 1 else (0, 0)
    if isinstance(op, mc.Block):
        if op.num_bits or op.num_zvars:
            return None
        return _as_pauli_instructions(op.instructions)
    if isinstance(op, mc.GateCall):
        return _as_pauli_instructions(op._decl.circuit)
    return None


def _as_pauli_instructions(instructions):
    x = z = 0
    for inst in instructions:
        p = _as_pauli(inst.operation)
        if p is None:
            return None
        for i, q in enumerate(inst.qubits):
            x ^= ((p[0] >> i) & 1) << q
            z ^= ((p[1] >> i) & 1) << q
    return x, z


def _pauli_pairs(x, z):
    """The non-identity (qubit, letter) pairs of the Pauli with bit sets x and z."""
    qs, P, _ = _pauli_string((x, z, 0))
    return tuple(zip(qs, P))


class _PauliFrame:
    """Pauli frame of the injection: the product of the byproducts left so far,
    stored as its X part in bits fx[q] = offset+q and Z part in fz[q] =
    offset+n+q. Every update and correction is a 3-bit Xor, so the arity of
    the operations emitted does not grow with the circuit. ``livex`` and
    ``livez`` mark the bits ever written; the others are known to be 0 and are
    skipped."""

    def __init__(self, offset, n):
        self.offset = offset
        self.n = n
        self.livex = 0
        self.livez = 0

    def correct(self, out, pauli, bit):
        # A measurement of `pauli` flips when the frame anticommutes with it,
        # that is for every qubit where the X part of `pauli` meets fz or its
        # Z part meets fx.
        x, z, _ = pauli
        for q in range(self.n):
            if (x >> q) & 1 and (self.livez >> q) & 1:
                out.push(mc.Xor(), bit, bit, self.offset + self.n + q)
            if (z >> q) & 1 and (self.livex >> q) & 1:
                out.push(mc.Xor(), bit, bit, self.offset + q)

    def update(self, out, pauli, bit):
        # Multiplies the frame by `pauli` when `bit` is 1.
        x, z, _ = pauli
        for q in range(self.n):
            if (x >> q) & 1:
                fx = self.offset + q
                out.push(mc.Xor(), fx, fx, bit)
            if (z >> q) & 1:
                fz = self.offset + self.n + q
                out.push(mc.Xor(), fz, fz, bit)
        self.livex |= x
        self.livez |= z


def _cached_conjugate(cache, op, i, p):
    # Operations hash by their attributes, so one holding an unhashable value
    # (a numpy array, say) is conjugated without the cache.
    key = (op, i, p)
    try:
        hash(key)
    except TypeError:
        return _clifford_conjugate(op, i, p)
    if key not in cache:
        cache[key] = _clifford_conjugate(op, i, p)
    return cache[key]


# A Pauli operator is (x, z, r) = iʳ·X^x·Z^z, with X^x Z^z the qubit-wise
# product, x and z bit sets over the qubits and r mod 4.


def _pauli_mul(a, b):
    # Moving Z^z₁ past X^x₂ costs (-1)^(z₁·x₂).
    x1, z1, r1 = a
    x2, z2, r2 = b
    return (x1 ^ x2, z1 ^ z2, (r1 + r2 + 2 * (z1 & x2).bit_count()) & 3)


def _through_tableau(s, img, qs, tx, tz):
    """s·img rewritten through the tableau, with img a Pauli given as (i,
    letter) pairs acting on the qubits qs[i]."""
    acc = (0, 0, 0 if s > 0 else 2)
    for i, ch in img:
        q = qs[i]
        if ch == "X":
            acc = _pauli_mul(acc, tx[q])
        elif ch == "Z":
            acc = _pauli_mul(acc, tz[q])
        elif ch == "Y":
            # Y = i·X·Z
            x, z, r = _pauli_mul(_pauli_mul(acc, tx[q]), tz[q])
            acc = (x, z, (r + 1) & 3)
    return acc


def _pauli_string(pauli):
    """The support, the letters on it and the ±1 sign of a Hermitian Pauli,
    using X·Z = -i·Y."""
    x, z, r = pauli
    qs, letters = [], []
    rest = x | z
    while rest:
        low = rest & -rest
        qs.append(low.bit_length() - 1)
        letters.append("Y" if x & z & low else "X" if x & low else "Z")
        rest ^= low
    r = (r - (x & z).bit_count()) & 3
    return qs, "".join(letters), 1 if r == 0 else -1


def _clifford_conjugate(g, i, p):
    """C†PC for a k-qubit Clifford C and the generator P = X or Z on its qubit
    i, read off the Pauli basis. The image is returned as a sign and (qubit,
    letter) pairs, qubits counted within the gate."""
    try:
        U = np.array(g.matrix().tolist(), dtype=complex)
    except (TypeError, RuntimeError):
        U = None
    if U is not None:
        k = g.num_qubits
        loc = "".join(p if j == i else "I" for j in range(k))
        R = U.conj().T @ pauli_string_matrix(loc) @ U

        for chars in product("IXYZ", repeat=k):
            c = np.trace(pauli_string_matrix(chars) @ R) / 2**k
            if np.isclose(abs(c.real), 1, atol=1e-8):
                img = tuple((j, ch) for j, ch in enumerate(chars) if ch != "I")
                return int(round(c.real)), img

    raise ValueError(f"Pauli-based computation needs Clifford+T gates, got {g.name}.")


__all__ = ["pauli_based_computation"]
