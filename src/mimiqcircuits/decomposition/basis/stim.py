#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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
"""Stim stabilizer simulator decomposition basis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


# Terminality per operation type, for the types where the type alone decides
# it. Filled on first use and shared by all instances.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


def _decompose_ecr(qubits):
    """``GateECR`` as ``S``, ``SX``, ``CX`` and ``X``, equal up to a global phase."""
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.gates.standard.cpauli import GateCX
    from mimiqcircuits.operations.gates.standard.pauli import GateX
    from mimiqcircuits.operations.gates.standard.s import GateS
    from mimiqcircuits.operations.gates.standard.sx import GateSX

    a, b = qubits
    circ = Circuit()
    circ.push(GateS(), a)
    circ.push(GateSX(), b)
    circ.push(GateCX(), a, b)
    circ.push(GateX(), a)
    return circ


class StimBasis(DecompositionBasis):
    """Decomposition basis targeting the gate set of the Stim stabilizer simulator.

    `Stim <https://github.com/quantumlib/Stim>`_ simulates stabilizer
    circuits: Clifford gates, Pauli measurements and resets, Pauli noise and
    quantum error correction annotations. Gates are rewritten toward that set
    by the first rule that matches, in this order:

    1. A ``GateU`` whose angles are all zero (a pure global phase) is
       dropped.
    2. :class:`ZYZRewrite`: ``GateU`` to ``RZ*RY*RZ``, and ``GateRX`` to
       ``GateRY`` between ``S`` and ``S†``.
    3. :class:`SpecialAngleRewrite` with ``only_cliffords=True``: rotations
       by multiples of ``pi/2`` to Clifford gates.
    4. ``GateECR`` to ``S``, ``SX``, ``CX`` and ``X``, equal up to a global
       phase. Stim has no ECR instruction, and the canonical decomposition
       goes through non-Clifford ``RZX(+-pi/4)``.
    5. :class:`CanonicalRewrite`: any other operation.

    A ``GateRX``, ``GateRY`` or ``GateRZ`` by any other angle is not a
    Clifford gate and raises :class:`DecompositionError`; so does any
    non-Clifford gate that reduces to one, such as ``GateT``. Use
    :class:`CliffordTBasis` for such circuits.

    The terminal operations are:

    - single-qubit Cliffords: ``GateID``, ``GateX``, ``GateY``, ``GateZ``,
      ``GateH``, ``GateHXY``, ``GateHYZ``, ``GateS``, ``GateSDG``,
      ``GateSX``, ``GateSXDG``, ``GateSY`` and ``GateSYDG``;
    - two-qubit Cliffords: ``GateCX``, ``GateCY``, ``GateCZ``, ``GateSWAP``,
      ``GateISWAP``, ``GateISWAPDG`` and ``GateDCX``;
    - Pauli rotations by ``+-pi/2``: ``GateRXX``, ``GateRYY``, ``GateRZZ``,
      and ``RPauli`` with a non-identity Pauli string. Angles are compared
      modulo ``2*pi``, since a rotation by ``3*pi/2`` equals one by
      ``-pi/2`` up to a global phase. Other angles are decomposed;
    - measurements in any Pauli basis: ``Measure``, ``MeasureX``,
      ``MeasureY``, ``MeasureXX``, ``MeasureYY``, ``MeasureZZ`` and
      ``MeasurePauli``;
    - resets and measure-and-resets in the Z, X and Y bases: ``Reset``,
      ``ResetX``, ``ResetY``, ``MeasureReset``, ``MeasureResetX`` and
      ``MeasureResetY``;
    - Pauli noise: ``PauliX``, ``PauliY``, ``PauliZ``, ``Depolarizing1``,
      ``Depolarizing2`` and ``PauliNoise``;
    - ``Barrier`` and annotations, including ``Detector``,
      ``ObservableInclude`` and ``QubitCoordinates``;
    - feedback: an ``IfStatement`` of ``GateX``, ``GateY`` or ``GateZ``
      conditioned on a single bit being ``1``.

    The same Pauli conditioned on a bit being ``0`` becomes the Pauli
    followed by the Pauli conditioned on ``1``. Any other conditional
    operation has its body decomposed while staying conditional, and raises
    :class:`DecompositionError` if that does not lead to conditional Paulis.

    The terminal operations correspond to Stim instructions as follows.

    =========================== =============================================================================
    MIMIQ Operation             Stim Operation
    =========================== =============================================================================
    ``GateID``                  ``I``
    ``GateX``                   ``X``
    ``GateY``                   ``Y``
    ``GateZ``                   ``Z``
    ``GateH``                   ``H``
    ``GateHXY``                 ``H_XY``
    ``GateHYZ``                 ``H_YZ``
    ``GateS``                   ``S``
    ``GateSDG``                 ``S_DAG``
    ``GateSX``                  ``SQRT_X``
    ``GateSXDG``                ``SQRT_X_DAG``
    ``GateSY``                  ``SQRT_Y``
    ``GateSYDG``                ``SQRT_Y_DAG``
    ``GateCX``                  ``CNOT`` / ``CX``
    ``GateCY``                  ``CY``
    ``GateCZ``                  ``CZ``
    ``GateSWAP``                ``SWAP``
    ``GateISWAP``               ``ISWAP``
    ``GateISWAPDG``             ``ISWAP_DAG``
    ``GateDCX``                 ``SWAPCX``
    ``GateRXX(+-pi/2)``         ``SQRT_XX`` / ``SQRT_XX_DAG``
    ``GateRYY(+-pi/2)``         ``SQRT_YY`` / ``SQRT_YY_DAG``
    ``GateRZZ(+-pi/2)``         ``SQRT_ZZ`` / ``SQRT_ZZ_DAG``
    ``RPauli(P, +-pi/2)``       ``SPP`` / ``SPP_DAG``
    ``IfStatement(GateX(), 1)`` ``CX rec[-k] q``
    ``IfStatement(GateY(), 1)`` ``CY rec[-k] q``
    ``IfStatement(GateZ(), 1)`` ``CZ rec[-k] q``
    ``Measure``                 ``M``
    ``MeasureX``                ``MX``
    ``MeasureY``                ``MY``
    ``MeasureXX``               ``MXX``
    ``MeasureYY``               ``MYY``
    ``MeasureZZ``               ``MZZ``
    ``MeasurePauli``            ``MPP``
    ``Reset``                   ``R``
    ``ResetX``                  ``RX``
    ``ResetY``                  ``RY``
    ``MeasureReset``            ``MR``
    ``MeasureResetX``           ``MRX``
    ``MeasureResetY``           ``MRY``
    ``PauliX``                  ``X_ERROR``
    ``PauliY``                  ``Y_ERROR``
    ``PauliZ``                  ``Z_ERROR``
    ``Depolarizing1``           ``DEPOLARIZE1``
    ``Depolarizing2``           ``DEPOLARIZE2``
    ``PauliNoise``              ``PAULI_CHANNEL_1``, ``PAULI_CHANNEL_2`` or ``E`` / ``ELSE_CORRELATED_ERROR``
    ``Detector``                ``DETECTOR``
    ``ObservableInclude``       ``OBSERVABLE_INCLUDE``
    ``QubitCoordinates``        ``QUBIT_COORDS``
    =========================== =============================================================================

    ``GateECR`` is terminal as a Clifford gate but has no Stim instruction of
    its own.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> c = Circuit()
        >>> c.push(GateRY(pi/2), 0)
        1-qubit circuit with 1 instruction:
        └── RY((1/2)*pi) @ q[0]
        <BLANKLINE>
        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── RY((1/2)*pi) @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
        >>> # RY(pi/2) is the Clifford SY; CX is already a Stim gate
        >>> decompose(c, StimBasis())
        2-qubit circuit with 2 instructions:
        ├── SY @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>

        >>> # a T gate is not a Clifford gate and is refused
        >>> c = Circuit()
        >>> c.push(GateT(), 0)
        1-qubit circuit with 1 instruction:
        └── T @ q[0]
        <BLANKLINE>
        >>> decompose(c, StimBasis())
        Traceback (most recent call last):
            ...
        mimiqcircuits.decomposition.abstract.DecompositionError: Rotation RZ cannot be decomposed into the Stim basis. Only rotations with angles that are multiples of pi/2 are Clifford gates.

    See Also:
        :class:`CliffordTBasis`, :class:`QASMBasis`, :class:`CanonicalBasis`
    """

    def isterminal(self, op: Operation) -> bool:
        """Check whether an operation is terminal in the Stim basis.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if ``op`` is one of the terminal operations listed in the
            class description.
        """
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        from mimiqcircuits.operations.annotations import (
            AbstractAnnotation,
            Detector,
            ObservableInclude,
            QubitCoordinates,
        )
        from mimiqcircuits.operations.barrier import Barrier
        from mimiqcircuits.operations.gates.generalized.rpauli import RPauli
        from mimiqcircuits.operations.gates.standard.cpauli import (
            GateCX,
            GateCY,
            GateCZ,
        )
        from mimiqcircuits.operations.gates.standard.dcx import GateDCX
        from mimiqcircuits.operations.gates.standard.hadamard import (
            GateH,
            GateHXY,
            GateHYZ,
        )
        from mimiqcircuits.operations.gates.standard.id import GateID
        from mimiqcircuits.operations.gates.standard.interactions import (
            GateRXX,
            GateRYY,
            GateRZZ,
        )
        from mimiqcircuits.operations.gates.standard.iswap import GateISWAP, GateISWAPDG
        from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
        from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
        from mimiqcircuits.operations.gates.standard.swap import GateSWAP
        from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
        from mimiqcircuits.operations.gates.standard.sy import GateSY, GateSYDG
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.measure import Measure, MeasureX, MeasureY
        from mimiqcircuits.operations.measurepauli import MeasurePauli
        from mimiqcircuits.operations.measurereset import (
            MeasureReset,
            MeasureResetX,
            MeasureResetY,
        )
        from mimiqcircuits.operations.noisechannel.standards.depolarizing import (
            Depolarizing,
        )
        from mimiqcircuits.operations.noisechannel.standards.pauli import (
            PauliNoise,
            PauliX,
            PauliY,
            PauliZ,
        )
        from mimiqcircuits.operations.pairmeasure import MeasureXX, MeasureYY, MeasureZZ
        from mimiqcircuits.operations.reset import Reset, ResetX, ResetY

        # These are decided by their values, so they never reach the cache.
        # Pauli rotations by +-pi/2 are Stim's SQRT_XX, SQRT_YY, SQRT_ZZ, SPP.
        if isinstance(op, (GateRXX, GateRYY, GateRZZ)):
            return _is_half_pi(op.theta)
        if isinstance(op, RPauli):
            return any(p != "I" for p in str(op.pauli)) and _is_half_pi(op.theta)
        # Depolarizing1 and Depolarizing2 build a Depolarizing.
        if isinstance(op, Depolarizing):
            return op.num_qubits in (1, 2)
        # Feedback: Stim's CX/CY/CZ with a measurement record as control.
        if isinstance(op, IfStatement):
            return isinstance(op.op, (GateX, GateY, GateZ)) and _is_one(
                op.bitstring
            )

        known = isinstance(
            op,
            (
                # Single-qubit Cliffords
                GateID,
                GateX,
                GateY,
                GateZ,
                GateH,
                GateHXY,
                GateHYZ,
                GateS,
                GateSDG,
                GateSX,
                GateSXDG,
                GateSY,
                GateSYDG,
                # Two-qubit Cliffords
                GateCX,
                GateCY,
                GateCZ,
                GateSWAP,
                GateISWAP,
                GateISWAPDG,
                GateDCX,
                # Measurements, resets and measure-resets in all Pauli bases
                Measure,
                MeasureX,
                MeasureY,
                MeasureXX,
                MeasureYY,
                MeasureZZ,
                MeasurePauli,
                Reset,
                ResetX,
                ResetY,
                MeasureReset,
                MeasureResetX,
                MeasureResetY,
                Barrier,
                # Pauli noise
                PauliX,
                PauliY,
                PauliZ,
                PauliNoise,
                # QEC and generic annotations
                Detector,
                ObservableInclude,
                QubitCoordinates,
                AbstractAnnotation,
            ),
        )
        _TERMINAL_BY_TYPE[t] = known
        return known

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite a non-terminal operation one step toward the Stim gate set.

        Applies the first matching rule of the list in the class description.
        A conditional operation is handled as described there.

        Args:
            op: The operation to rewrite.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If ``op`` is a rotation by an angle that is
                not a multiple of ``pi/2``, a conditional operation Stim
                cannot express, or has no built-in decomposition.
        """
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.rotations import (
            GateRX,
            GateRY,
            GateRZ,
        )
        from mimiqcircuits.operations.gates.standard.u import GateU
        from mimiqcircuits.operations.gates.standard.ecr import GateECR
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.decomposition.rules import (
            CanonicalRewrite,
            SpecialAngleRewrite,
            ZYZRewrite,
        )
        from mimiqcircuits.decomposition.rules.zyz import _iszero

        if isinstance(op, IfStatement):
            return self._decompose_ifstatement(op, qubits, bits, zvars)

        # A bare GateU with zero angles is only a global phase, so it expands
        # to nothing; ZYZRewrite refuses it. Under a Control the phase is a
        # relative phase; that case is a Control, not a GateU, and goes to
        # CanonicalRewrite.
        if isinstance(op, GateU) and all(
            _iszero(p) for p in (op.theta, op.phi, op.lmbda)
        ):
            return Circuit()

        # Priority 1: GateU -> ZYZ decomposition (RZ * RY * RZ), and GateRX
        zyz = ZYZRewrite()
        if zyz.matches(op):
            return zyz.decompose_step(op, qubits, bits, zvars)

        # Priority 2: Special angles (Clifford-only) - handles rotations at k*pi/2
        special_angle_clifford = SpecialAngleRewrite(only_cliffords=True)
        if special_angle_clifford.matches(op):
            return special_angle_clifford.decompose_step(op, qubits, bits, zvars)

        # Explicitly reject non-Clifford rotations
        # This prevents infinite loops (RZ -> U -> ZYZ -> RZ) and gives clear errors
        if isinstance(op, (GateRX, GateRY, GateRZ)):
            raise DecompositionError(
                f"Rotation {op.name} cannot be decomposed into the Stim basis. "
                "Only rotations with angles that are multiples of pi/2 are Clifford gates."
            )

        if isinstance(op, GateECR):
            return _decompose_ecr(qubits)

        # Priority 3: Fallback to canonical decomposition
        canonical = CanonicalRewrite()
        if canonical.matches(op):
            return canonical.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} cannot be decomposed into the Stim basis."
        )

    def _decompose_ifstatement(self, op, qubits, bits, zvars):
        from mimiqcircuits.bitstrings import BitString
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.decomposition.rules import SpecialAngleRewrite
        from mimiqcircuits.operations.ifstatement import _decompose_ifstatement

        inner = op.op
        condition = op.bitstring

        if len(condition) != 1:
            raise DecompositionError(
                f"IfStatement with a {len(condition)}-bit condition cannot be decomposed "
                "into the Stim basis. Stim only conditions X, Y and Z gates on a single "
                "measurement record."
            )

        # P if c == 0 equals P, then P if c == 1, since P^2 = I.
        if isinstance(inner, (GateX, GateY, GateZ)):
            circ = Circuit()
            circ.push(inner, *qubits)
            circ.push(IfStatement(inner, BitString("1")), *qubits, *bits)
            return circ

        if self.isterminal(inner):
            raise DecompositionError(
                f"IfStatement of {inner} cannot be decomposed into the Stim basis. "
                "Stim only conditions X, Y and Z gates on a single measurement record."
            )

        # Rotations by multiples of pi go straight to the Pauli they equal; the
        # general pipeline would pass through gates that cannot be conditioned.
        special_angle_clifford = SpecialAngleRewrite(only_cliffords=True)
        if special_angle_clifford.matches(inner):
            return _decompose_ifstatement(
                special_angle_clifford, op, qubits, bits, zvars
            )

        return _decompose_ifstatement(self, op, qubits, bits, zvars)


def _is_half_pi(theta) -> bool:
    from mimiqcircuits.decomposition.rules.special_angle import _get_pi_factor

    return _get_pi_factor(theta) in (2, 6)


def _is_one(bitstring) -> bool:
    return len(bitstring) == 1 and bitstring[0]


__all__ = ["StimBasis"]
