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
"""Rewrite rule for converting X and Y rotations to Z rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


class ToZRotationRewrite(RewriteRule):
    """Rewrite rule that converts RX and RY rotations to RZ + Clifford gates.

    This is useful for gate sets where only Z-rotations are available as
    the parametric rotation (e.g., some hardware native gate sets).

    Transformations:
        - ``RX(theta)`` -> ``H * RZ(theta) * H``
        - ``RY(theta)`` -> ``Sdg * H * RZ(theta) * H * S``

    Example:
        >>> from mimiqcircuits import GateRX, GateRY
        >>> from mimiqcircuits.decomposition import ToZRotationRewrite
        >>> from symengine import pi
        >>> rule = ToZRotationRewrite()
        >>> circ = rule.decompose_step(GateRX(pi/3), [0], [], [])
    """

    def matches(self, op: Operation) -> bool:
        """Check if this rule can decompose the operation.

        Args:
            op: The operation to check.

        Returns:
            True for GateRX and GateRY operations.
        """
        import mimiqcircuits as mc

        return isinstance(op, (mc.GateRX, mc.GateRY))

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Convert RX or RY to RZ + Clifford gates.

        Args:
            op: The rotation operation (GateRX or GateRY).
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
                f"ToZRotationRewrite cannot decompose {type(op).__name__}"
            )

        circ = mc.Circuit()
        q = qubits[0]

        if isinstance(op, mc.GateRX):
            # RX(theta) = H * RZ(theta) * H
            circ.push(mc.GateH(), q)
            circ.push(mc.GateRZ(op.theta), q)
            circ.push(mc.GateH(), q)

        elif isinstance(op, mc.GateRY):
            # RY(theta) = Sdg * H * RZ(theta) * H * S
            circ.push(mc.GateSDG(), q)
            circ.push(mc.GateH(), q)
            circ.push(mc.GateRZ(op.theta), q)
            circ.push(mc.GateH(), q)
            circ.push(mc.GateS(), q)

        return circ


__all__ = ["ToZRotationRewrite"]
