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


class StimBasis(DecompositionBasis):
    """Decomposition basis targeting the Stim stabilizer simulator gate set.

    `Stim <https://github.com/quantumlib/Stim>`_ is a high-performance stabilizer
    circuit simulator by Craig Gidney. It supports only **Clifford gates**,
    stabilizer operations, noise channels, and quantum error correction annotations.

    Decomposition Pipeline
    ----------------------

    The decomposition follows this priority order:

    1. **ZYZRewrite**: ``GateU`` -> ``RZ * RY * RZ``
    2. **SpecialAngleRewrite** (Clifford-only): Rotations at ``k*pi/2`` -> Clifford gates
    3. **CanonicalRewrite**: Fallback for other gates

    Non-Clifford rotations (angles not multiples of pi/2) will raise a
    :class:`DecompositionError`.

    Terminal Operations
    -------------------

    **Single-qubit Clifford gates:**
        - ``GateID``
        - Pauli: ``GateX``, ``GateY``, ``GateZ``
        - Hadamard: ``GateH``
        - Phase gates: ``GateS``, ``GateSDG``
        - Square root gates: ``GateSX``, ``GateSXDG``, ``GateSY``, ``GateSYDG``

    **Two-qubit Clifford gates:**
        - Controlled Pauli: ``GateCX``, ``GateCY``, ``GateCZ``
        - Swap family: ``GateSWAP``, ``GateISWAP``, ``ISWAP†``
        - ``GateDCX``, ``GateECR``

    **Measurements (all Pauli bases):**
        - ``Measure``, ``MeasureX``, ``MeasureY``

    **Resets (all Pauli bases):**
        - ``Reset``, ``ResetX``, ``ResetY``

    **Measure-and-reset (all Pauli bases):**
        - ``MeasureReset``, ``MeasureResetX``, ``MeasureResetY``

    **Noise channels:**
        - ``PauliX``, ``PauliY``, ``PauliZ``
        - ``Depolarizing1``, ``Depolarizing2``

    **QEC annotations:**
        - ``Detector``, ``ObservableInclude``, ``QubitCoordinates``
        - ``Barrier``

    Stim Gate Mapping
    -----------------

    ==================== =====================
    MIMIQ Operation      Stim Operation
    ==================== =====================
    ``GateID``           ``I``
    ``GateX``            ``X``
    ``GateY``            ``Y``
    ``GateZ``            ``Z``
    ``GateH``            ``H``
    ``GateS``            ``S``
    ``GateSDG``          ``S_DAG``
    ``GateSX``           ``SQRT_X``
    ``GateSXDG``         ``SQRT_X_DAG``
    ``GateSY``           ``SQRT_Y``
    ``GateSYDG``         ``SQRT_Y_DAG``
    ``GateCX``           ``CNOT`` / ``CX``
    ``GateCY``           ``CY``
    ``GateCZ``           ``CZ``
    ``GateSWAP``         ``SWAP``
    ``GateISWAP``        ``ISWAP``
    ``ISWAP†``           ``ISWAP_DAG``
    ``GateDCX``          ``DCX`` (custom)
    ``GateECR``          ``ECR`` (custom)
    ``Measure``          ``M``
    ``MeasureX``         ``MX``
    ``MeasureY``         ``MY``
    ``Reset``            ``R``
    ``ResetX``           ``RX``
    ``ResetY``           ``RY``
    ``MeasureReset``     ``MR``
    ``MeasureResetX``    ``MRX``
    ``MeasureResetY``    ``MRY``
    ``Detector``         ``DETECTOR``
    ``ObservableInclude`` ``OBSERVABLE_INCLUDE``
    ``QubitCoordinates`` ``QUBIT_COORDS``
    ==================== =====================

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
        import mimiqcircuits as mc

        # === Single-qubit Clifford gates ===

        # Identity
        if isinstance(op, mc.GateID):
            return True

        # Pauli gates
        if isinstance(op, (mc.GateX, mc.GateY, mc.GateZ)):
            return True

        # Hadamard
        if isinstance(op, mc.GateH):
            return True

        # Phase gates: S, Sdg
        if isinstance(op, (mc.GateS, mc.GateSDG)):
            return True

        # Square root gates: SX, SXdg
        if isinstance(op, (mc.GateSX, mc.GateSXDG)):
            return True

        # SY, SYdg
        if isinstance(op, (mc.GateSY, mc.GateSYDG)):
            return True

        # === Two-qubit Clifford gates ===

        # Controlled Pauli: CX, CY, CZ
        if isinstance(op, (mc.GateCX, mc.GateCY, mc.GateCZ)):
            return True

        # Swap family
        if isinstance(op, mc.GateSWAP):
            return True
        if isinstance(op, (mc.GateISWAP, mc.GateISWAPDG)):
            return True

        # DCX and ECR
        if isinstance(op, (mc.GateDCX, mc.GateECR)):
            return True

        # === Measurements (all Pauli bases) ===

        if isinstance(op, (mc.Measure, mc.MeasureX, mc.MeasureY)):
            return True

        # === Resets (all Pauli bases) ===

        if isinstance(op, (mc.Reset, mc.ResetX, mc.ResetY)):
            return True

        # === Measure-and-reset (all Pauli bases) ===

        if isinstance(op, (mc.MeasureReset, mc.MeasureResetX, mc.MeasureResetY)):
            return True

        # === Barriers ===

        if isinstance(op, mc.Barrier):
            return True

        # === Noise channels (Stim supports Pauli noise natively) ===

        # Pauli error channels
        if isinstance(op, (mc.PauliX, mc.PauliY, mc.PauliZ)):
            return True

        # Depolarizing noise
        if isinstance(op, (mc.Depolarizing1, mc.Depolarizing2)):
            return True

        # === QEC Annotations ===

        if isinstance(op, (mc.Detector, mc.ObservableInclude, mc.QubitCoordinates)):
            return True

        # Generic annotations
        if isinstance(op, mc.AbstractAnnotation):
            return True

        return False

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose a non-terminal operation toward the Stim Clifford gate set.

        Applies rules in priority order:
            1. ZYZRewrite for GateU
            2. SpecialAngleRewrite (Clifford-only) for rotations at k*pi/2
            3. CanonicalRewrite as fallback

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


__all__ = ["StimBasis"]
