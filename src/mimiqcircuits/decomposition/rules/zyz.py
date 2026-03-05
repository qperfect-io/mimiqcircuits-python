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
"""ZYZ Euler angle decomposition rewrite rule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


def _iszero(val) -> bool:
    """Check if a value is zero (handling symbolic expressions)."""
    import symengine as se
    import sympy as sp

    if isinstance(val, (se.Basic, sp.Basic)):
        try:
            return complex(val) == 0
        except (TypeError, ValueError):
            return False
    return val == 0


class ZYZRewrite(RewriteRule):
    """Rewrite rule for ZYZ Euler angle decomposition.

    Decomposes single-qubit unitaries into the ZYZ Euler angle form:
    ``RZ(phi) * RY(theta) * RZ(lambda)``

    This is a standard decomposition for arbitrary single-qubit gates,
    producing only Z and Y rotations (plus a global phase for GateU).

    Supported operations:
        - ``GateU(theta, phi, lambda, gamma)`` -> ``RZ(lambda) * RY(theta) * RZ(phi) * Phase(gamma)``
        - ``GateRX(theta)`` -> ``S * RY(theta) * Sdg``

    Identity rotations (angle = 0) are omitted from the output.

    Example:
        >>> from mimiqcircuits import GateU
        >>> from mimiqcircuits.decomposition import ZYZRewrite
        >>> from symengine import pi
        >>> rule = ZYZRewrite()
        >>> circ = rule.decompose_step(GateU(pi/2, pi/4, pi/3), [0], [], [])
    """

    def matches(self, op: Operation) -> bool:
        """Check if this rule can decompose the operation.

        Matches GateU (unless it's identity) and GateRX.

        Args:
            op: The operation to check.

        Returns:
            True if this rule can decompose the operation.
        """
        import mimiqcircuits as mc

        if isinstance(op, mc.GateU):
            # Match unless it's the identity (all angles zero)
            return not (
                _iszero(op.theta) and _iszero(op.phi) and _iszero(op.lmbda)
            )
        if isinstance(op, mc.GateRX):
            return True
        return False

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Apply ZYZ decomposition to the operation.

        Args:
            op: The operation to decompose (GateU or GateRX).
            qubits: Qubit indices for the operation.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A Circuit containing the decomposed instructions.

        Raises:
            DecompositionError: If the operation is not supported.
        """
        import mimiqcircuits as mc

        if not self.matches(op):
            raise DecompositionError(
                f"ZYZRewrite cannot decompose {type(op).__name__}"
            )

        circ = mc.Circuit()
        q = qubits[0]

        if isinstance(op, mc.GateU):
            # GateU(theta, phi, lambda, gamma) = e^{i*gamma} * RZ(phi) * RY(theta) * RZ(lambda)
            # Emit gates right-to-left (matrix multiplication order)
            lmbda = op.lmbda
            theta = op.theta
            phi = op.phi
            gamma = op.gamma

            if not _iszero(lmbda):
                circ.push(mc.GateRZ(lmbda), q)
            if not _iszero(theta):
                circ.push(mc.GateRY(theta), q)
            if not _iszero(phi):
                circ.push(mc.GateRZ(phi), q)
            if not _iszero(gamma):
                circ.push(mc.GateU(0, 0, 0, gamma), q)

        elif isinstance(op, mc.GateRX):
            # RX(theta) = S * RY(theta) * Sdg
            # This converts X-rotation to Y-rotation via S conjugation
            circ.push(mc.GateS(), q)
            circ.push(mc.GateRY(op.theta), q)
            circ.push(mc.GateSDG(), q)

        return circ


__all__ = ["ZYZRewrite"]
