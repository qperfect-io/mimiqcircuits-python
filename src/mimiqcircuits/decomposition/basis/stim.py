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
    from mimiqcircuits import Circuit, Operation


# Terminality per operation type, for the types where the type alone decides
# it. Filled on first use and shared by all instances.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


class StimBasis(DecompositionBasis):
    """Decomposition basis targeting the Stim stabilizer simulator gate set.

    `Stim <https://github.com/quantumlib/Stim>`_ is a high-performance stabilizer
    circuit simulator by Craig Gidney. It supports only **Clifford gates**,
    stabilizer operations, noise channels, and quantum error correction annotations.

    Decomposition Pipeline
    ----------------------

    The decomposition follows this priority order:

    1. A ``GateU`` whose angles are all zero (a pure global phase) is dropped.
    2. **ZYZRewrite**: ``GateU`` -> ``RZ * RY * RZ``
    3. **SpecialAngleRewrite** (Clifford-only): Rotations at ``k*pi/2`` -> Clifford gates
    4. **CanonicalRewrite**: Fallback for other gates

    Non-Clifford rotations (angles not multiples of pi/2) will raise a
    :class:`DecompositionError`.

    Terminal Operations
    -------------------

    **Single-qubit Clifford gates:**
        - ``GateID``
        - Pauli: ``GateX``, ``GateY``, ``GateZ``
        - Hadamard: ``GateH``, ``GateHXY``, ``GateHYZ``
        - Phase gates: ``GateS``, ``GateSDG``
        - Square root gates: ``GateSX``, ``GateSXDG``, ``GateSY``, ``GateSYDG``

    **Two-qubit Clifford gates:**
        - Controlled Pauli: ``GateCX``, ``GateCY``, ``GateCZ``
        - Swap family: ``GateSWAP``, ``GateISWAP``, ``ISWAP†``
        - ``GateDCX``, ``GateECR``

    **Pauli rotations by +-pi/2:**
        - ``GateRXX``, ``GateRYY``, ``GateRZZ`` with angle +-pi/2
        - ``RPauli`` with angle +-pi/2 and a non-identity Pauli string

    Angles are compared modulo 2*pi, since a rotation by 3*pi/2 equals a
    rotation by -pi/2 up to a global phase. Other angles are decomposed.

    **Feedback:**
        - ``IfStatement`` of ``GateX``, ``GateY`` or ``GateZ`` conditioned on a
          single bit being ``1``. The same gate conditioned on the bit being
          ``0`` is decomposed into the gate followed by the gate conditioned on
          ``1``. Any other conditional operation has its body decomposed while
          staying conditional, and raises a :class:`DecompositionError` if that
          does not lead to a conditional Pauli.

    **Measurements (all Pauli bases):**
        - ``Measure``, ``MeasureX``, ``MeasureY``
        - Two-qubit parities: ``MeasureXX``, ``MeasureYY``, ``MeasureZZ``
        - Pauli products: ``MeasurePauli``

    **Resets (all Pauli bases):**
        - ``Reset``, ``ResetX``, ``ResetY``

    **Measure-and-reset (all Pauli bases):**
        - ``MeasureReset``, ``MeasureResetX``, ``MeasureResetY``

    **Noise channels:**
        - ``PauliX``, ``PauliY``, ``PauliZ``
        - ``Depolarizing1``, ``Depolarizing2``
        - ``PauliNoise``

    **QEC annotations:**
        - ``Detector``, ``ObservableInclude``, ``QubitCoordinates``
        - ``Barrier``

    Stim Gate Mapping
    -----------------

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
    ``ISWAP†``                  ``ISWAP_DAG``
    ``GateDCX``                 ``DCX`` (custom)
    ``GateECR``                 ``ECR`` (custom)
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

    Example:
        >>> from mimiqcircuits import Circuit, GateH, GateS, GateCX
        >>> from mimiqcircuits.decomposition import decompose, StimBasis
        >>> from symengine import pi
        >>> # Decompose Clifford circuit to Stim-compatible gates
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateS(), 0)
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── S @ q[0]
        <BLANKLINE>
        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── S @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
        >>> decomposed = decompose(c, StimBasis())

        >>> # RZ(pi/2) = S is Clifford, so this works
        >>> from mimiqcircuits import GateRZ
        >>> c2 = Circuit()
        >>> c2.push(GateRZ(pi/2), 0)
        1-qubit circuit with 1 instruction:
        └── RZ((1/2)*pi) @ q[0]
        <BLANKLINE>
        >>> decomposed2 = decompose(c2, StimBasis())

    Note:
        Non-Clifford rotations will raise a DecompositionError::
            >>> from mimiqcircuits import *
            >>> c3 = Circuit()
            >>> c3.push(GateRZ(pi/4), 0)  # T gate is non-Clifford
            1-qubit circuit with 1 instruction:
            └── RZ((1/4)*pi) @ q[0]
            <BLANKLINE>
            >>> decompose(c3, StimBasis())  # raises DecompositionError doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            mimiqcircuits.decomposition.abstract.DecompositionError: Rotation RZ cannot be decomposed into the Stim basis. Only rotations with angles that are multiples of pi/2 are Clifford gates.


    See Also:
        :class:`CliffordTBasis`, :class:`QASMBasis`, :class:`CanonicalBasis`
    """

    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in the Stim basis.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is in the Stim Clifford gate set.
        """
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        import mimiqcircuits as mc

        # These are decided by their values, so they never reach the cache.
        # Pauli rotations by +-pi/2 are Stim's SQRT_XX, SQRT_YY, SQRT_ZZ, SPP.
        if isinstance(op, (mc.GateRXX, mc.GateRYY, mc.GateRZZ)):
            return _is_half_pi(op.theta)
        if isinstance(op, mc.RPauli):
            return any(p != "I" for p in str(op.pauli)) and _is_half_pi(op.theta)
        # Depolarizing1 and Depolarizing2 build a Depolarizing.
        if isinstance(op, mc.Depolarizing):
            return op.num_qubits in (1, 2)
        # Feedback: Stim's CX/CY/CZ with a measurement record as control.
        if isinstance(op, mc.IfStatement):
            return isinstance(op.op, (mc.GateX, mc.GateY, mc.GateZ)) and _is_one(
                op.bitstring
            )

        known = isinstance(
            op,
            (
                # Single-qubit Cliffords
                mc.GateID,
                mc.GateX,
                mc.GateY,
                mc.GateZ,
                mc.GateH,
                mc.GateHXY,
                mc.GateHYZ,
                mc.GateS,
                mc.GateSDG,
                mc.GateSX,
                mc.GateSXDG,
                mc.GateSY,
                mc.GateSYDG,
                # Two-qubit Cliffords
                mc.GateCX,
                mc.GateCY,
                mc.GateCZ,
                mc.GateSWAP,
                mc.GateISWAP,
                mc.GateISWAPDG,
                mc.GateDCX,
                mc.GateECR,
                # Measurements, resets and measure-resets in all Pauli bases
                mc.Measure,
                mc.MeasureX,
                mc.MeasureY,
                mc.MeasureXX,
                mc.MeasureYY,
                mc.MeasureZZ,
                mc.MeasurePauli,
                mc.Reset,
                mc.ResetX,
                mc.ResetY,
                mc.MeasureReset,
                mc.MeasureResetX,
                mc.MeasureResetY,
                mc.Barrier,
                # Pauli noise
                mc.PauliX,
                mc.PauliY,
                mc.PauliZ,
                mc.PauliNoise,
                # QEC and generic annotations
                mc.Detector,
                mc.ObservableInclude,
                mc.QubitCoordinates,
                mc.AbstractAnnotation,
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
        """Decompose a non-terminal operation toward the Stim Clifford gate set.

        Applies rules in priority order:
            1. Drops a GateU whose angles are all zero (a global phase)
            2. ZYZRewrite for GateU
            3. SpecialAngleRewrite (Clifford-only) for rotations at k*pi/2
            4. CanonicalRewrite as fallback

        Non-Clifford rotations (angles not multiples of pi/2) will raise an error.

        Args:
            op: The operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices for the operation.
            zvars: Z-variable indices for the operation.

        Returns:
            A Circuit containing the decomposed instructions.

        Raises:
            DecompositionError: If the operation cannot be decomposed to Clifford gates.
        """
        import mimiqcircuits as mc
        from mimiqcircuits.decomposition.rules import (
            CanonicalRewrite,
            SpecialAngleRewrite,
            ZYZRewrite,
        )
        from mimiqcircuits.decomposition.rules.zyz import _iszero

        if isinstance(op, mc.IfStatement):
            return self._decompose_ifstatement(op, qubits, bits, zvars)

        # A bare GateU with zero angles is only a global phase, so it expands
        # to nothing; ZYZRewrite refuses it. Under a Control the phase is a
        # relative phase; that case is a Control, not a GateU, and goes to
        # CanonicalRewrite.
        if isinstance(op, mc.GateU) and all(
            _iszero(p) for p in (op.theta, op.phi, op.lmbda)
        ):
            return mc.Circuit()

        # Priority 1: GateU -> ZYZ decomposition (RZ * RY * RZ)
        zyz = ZYZRewrite()
        if zyz.matches(op):
            return zyz.decompose_step(op, qubits, bits, zvars)

        # Priority 2: Special angles (Clifford-only) - handles rotations at k*pi/2
        special_angle_clifford = SpecialAngleRewrite(only_cliffords=True)
        if special_angle_clifford.matches(op):
            return special_angle_clifford.decompose_step(op, qubits, bits, zvars)

        # Explicitly reject non-Clifford rotations
        # This prevents infinite loops (RZ -> U -> ZYZ -> RZ) and gives clear errors
        if isinstance(op, (mc.GateRX, mc.GateRY, mc.GateRZ)):
            raise DecompositionError(
                f"Rotation {op.name} cannot be decomposed into the Stim basis. "
                "Only rotations with angles that are multiples of pi/2 are Clifford gates."
            )

        # Priority 3: Fallback to canonical decomposition
        canonical = CanonicalRewrite()
        if canonical.matches(op):
            return canonical.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} cannot be decomposed into the Stim basis."
        )

    def _decompose_ifstatement(self, op, qubits, bits, zvars):
        import mimiqcircuits as mc
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
        if isinstance(inner, (mc.GateX, mc.GateY, mc.GateZ)):
            circ = mc.Circuit()
            circ.push(inner, *qubits)
            circ.push(mc.IfStatement(inner, mc.BitString("1")), *qubits, *bits)
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
