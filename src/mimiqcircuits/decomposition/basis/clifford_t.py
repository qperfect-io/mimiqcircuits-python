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
"""Clifford+T decomposition basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


@dataclass(frozen=True)
class CliffordTBasis(DecompositionBasis):
    """Decomposition basis targeting the Clifford+T universal gate set.

    The Clifford+T gate set is fault-tolerant and widely used in error-corrected
    quantum computing. It consists of:

    Single-qubit Clifford gates:
        - Pauli: X, Y, Z
        - Hadamard: H
        - Phase: S, Sdg
        - Square roots: SX, SXdg, SY, SYdg

    T gates (non-Clifford, enables universality):
        - T, Tdg

    Two-qubit Clifford gates:
        - CX (CNOT), CY, CZ
        - SWAP, iSWAP
        - DCX, ECR

    Plus measurement, reset, barriers, classical ops, and annotations.

    The decomposition pipeline applies rules in priority order:
        1. SpecialAngleRewrite: Rotations with angles k*pi/4 -> exact Clifford+T
        2. ToffoliToCliffordTRewrite: CCX -> explicit Clifford+T
        3. ZYZRewrite: GateU -> RZ*RY*RZ
        4. ToZRotationRewrite: RX, RY -> RZ + Cliffords
        5. SolovayKitaevRewrite: Arbitrary rotations -> approximate Clifford+T
        6. CanonicalRewrite: Fallback for other gates

    Args:
        sk_depth: Solovay-Kitaev recursion depth (default: 3).
                  Higher = better precision, more gates.

    Example:
        >>> from mimiqcircuits import Circuit, GateH, GateT, GateRZ
        >>> from mimiqcircuits.decomposition import decompose, CliffordTBasis
        >>> from symengine import pi
        >>> c = Circuit()
        >>> c.push(GateRZ(pi/4), 0)  # Special angle -> exact T gate
        1-qubit circuit with 1 instruction:
        └── RZ((1/4)*pi) @ q[0]
        <BLANKLINE>
        >>> decomposed = decompose(c, CliffordTBasis())

        >>> # Arbitrary angle -> Solovay-Kitaev approximation
        >>> c2 = Circuit()
        >>> c2.push(GateRZ(0.123), 0)
        1-qubit circuit with 1 instruction:
        └── RZ(0.123) @ q[0]
        <BLANKLINE>
        >>> decomposed2 = decompose(c2, CliffordTBasis(sk_depth=0))
    """

    sk_depth: int = 3

    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in the Clifford+T basis.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is in the Clifford+T gate set.
        """
        import mimiqcircuits as mc

        # Identity
        if isinstance(op, mc.GateID):
            return True

        # Pauli gates (single-qubit Clifford)
        if isinstance(op, (mc.GateX, mc.GateY, mc.GateZ)):
            return True

        # Hadamard (single-qubit Clifford)
        if isinstance(op, mc.GateH):
            return True

        # Phase gates (single-qubit Clifford): S, S†
        if isinstance(op, (mc.GateS, mc.GateSDG)):
            return True

        # Square root gates (single-qubit Clifford): SX, SX†, SY, SY†
        if isinstance(op, (mc.GateSX, mc.GateSXDG)):
            return True
        if isinstance(op, (mc.GateSY, mc.GateSYDG)):
            return True

        # T gates (non-Clifford, enables universality): T, T†
        if isinstance(op, (mc.GateT, mc.GateTDG)):
            return True

        # Two-qubit Clifford gates: CX, CY, CZ
        if isinstance(op, (mc.GateCX, mc.GateCY, mc.GateCZ)):
            return True
        if isinstance(op, (mc.GateSWAP, mc.GateISWAP, mc.GateISWAPDG)):
            return True
        if isinstance(op, (mc.GateDCX, mc.GateECR)):
            return True

        # Measurement, reset, barriers
        if isinstance(op, (mc.AbstractMeasurement, mc.Reset, mc.MeasureReset)):
            return True
        if isinstance(op, mc.Barrier):
            return True

        # Classical operations
        if isinstance(op, mc.AbstractClassical):
            return True

        # Annotations
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
        """Decompose a non-terminal operation toward Clifford+T.

        Applies rules in priority order:
            1. SpecialAngleRewrite for special angles (exact)
            2. SolovayKitaevRewrite for arbitrary single-qubit rotations (approximate)
            3. ZYZRewrite for GateU
            4. ToZRotationRewrite for RX/RY
            5. CanonicalRewrite as fallback

        Args:
            op: The operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices for the operation.
            zvars: Z-variable indices for the operation.

        Returns:
            A Circuit containing the decomposed instructions.

        Raises:
            DecompositionError: If the operation cannot be decomposed.
        """
        from mimiqcircuits.decomposition.rules import (
            CanonicalRewrite,
            SolovayKitaevRewrite,
            SpecialAngleRewrite,
            ToZRotationRewrite,
            ToffoliToCliffordTRewrite,
            ZYZRewrite,
        )

        # Try rules in priority order
        # 1. Special angles first (exact decomposition)
        special_angle = SpecialAngleRewrite()
        if special_angle.matches(op):
            return special_angle.decompose_step(op, qubits, bits, zvars)

        # 2. Optimal Toffoli decomposition
        toffoli = ToffoliToCliffordTRewrite()
        if toffoli.matches(op):
            return toffoli.decompose_step(op, qubits, bits, zvars)

        # 3. ZYZ for GateU (will be further decomposed)
        zyz = ZYZRewrite()
        if zyz.matches(op):
            return zyz.decompose_step(op, qubits, bits, zvars)

        # 4. ToZRotation for RX/RY
        to_z = ToZRotationRewrite()
        if to_z.matches(op):
            return to_z.decompose_step(op, qubits, bits, zvars)

        # 5. Solovay-Kitaev for arbitrary rotations (approximate)
        sk = SolovayKitaevRewrite(depth=self.sk_depth)
        if sk.matches(op):
            return sk.decompose_step(op, qubits, bits, zvars)

        # 6. Canonical fallback
        canonical = CanonicalRewrite()
        if canonical.matches(op):
            return canonical.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} cannot be decomposed to Clifford+T basis."
        )


__all__ = ["CliffordTBasis"]
