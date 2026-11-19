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
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


class ToZRotationRewrite(RewriteRule):
    """Rewrite rule that turns ``GateRX`` and ``GateRY`` into ``GateRZ`` and Cliffords.

    Use it for gate sets whose only parametric rotation is ``GateRZ``. The
    rewrite is exact, keeps the angle as it is (symbolic angles included),
    and listed in circuit order is:

    - ``GateRX(theta)`` to ``H``, ``RZ(theta)``, ``H``;
    - ``GateRY(theta)`` to ``S†``, ``H``, ``RZ(theta)``, ``H``, ``S``.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> rule = ToZRotationRewrite()
        >>> # an X rotation is a Z rotation conjugated by Hadamard gates
        >>> rule.decompose_step(GateRX(pi/3), [0], [], [])
        1-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── RZ((1/3)*pi) @ q[0]
        └── H @ q[0]
        <BLANKLINE>
    """

    def matches(self, op: Operation) -> bool:
        """Check whether an operation is an X or Y rotation.

        Args:
            op: The operation to check.

        Returns:
            ``True`` for ``GateRX`` and ``GateRY``.
        """
        from mimiqcircuits.operations.gates.standard.rotations import GateRX, GateRY

        return isinstance(op, (GateRX, GateRY))

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite an X or Y rotation as a Z rotation between Cliffords.

        Args:
            op: The ``GateRX`` or ``GateRY`` to rewrite.
            qubits: Qubit index the rotation acts on.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If ``op`` is not a ``GateRX`` or ``GateRY``.
        """
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.hadamard import GateH
        from mimiqcircuits.operations.gates.standard.rotations import (
            GateRX,
            GateRY,
            GateRZ,
        )
        from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG

        if not self.matches(op):
            raise DecompositionError(
                f"ToZRotationRewrite cannot decompose {type(op).__name__}"
            )

        circ = Circuit()
        q = qubits[0]

        if isinstance(op, GateRX):
            # RX(theta) = H * RZ(theta) * H
            circ.push(GateH(), q)
            circ.push(GateRZ(op.theta), q)
            circ.push(GateH(), q)

        elif isinstance(op, GateRY):
            # RY(theta) = Sdg * H * RZ(theta) * H * S
            circ.push(GateSDG(), q)
            circ.push(GateH(), q)
            circ.push(GateRZ(op.theta), q)
            circ.push(GateH(), q)
            circ.push(GateS(), q)

        return circ


__all__ = ["ToZRotationRewrite"]
