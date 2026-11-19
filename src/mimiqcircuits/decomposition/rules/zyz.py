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
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


def _iszero(val) -> bool:
    """Check if a value is zero (handling symbolic expressions)."""
    from mimiqcircuits._sympycheck import _is_basic

    if _is_basic(val):
        try:
            return complex(val) == 0
        except (TypeError, ValueError, RuntimeError):
            return False
    return val == 0


class ZYZRewrite(RewriteRule):
    r"""Rewrite rule for the ZYZ Euler decomposition of single-qubit gates.

    Writes a single-qubit gate as Z and Y rotations, the Euler form
    :math:`R_Z(\phi) R_Y(\theta) R_Z(\lambda)`.

    Supported Operations:
        - ``GateU(theta, phi, lambda, gamma)``, except the identity (all of
          ``theta``, ``phi`` and ``lambda`` zero), to ``RZ(lambda)``,
          ``RY(theta)``, ``RZ(phi)`` in circuit order, followed by
          ``GateU(0, 0, 0, gamma)`` for the phase ``gamma``. Rotations by a
          zero angle are left out. The result equals the gate up to a
          global phase.
        - ``GateRX(theta)`` to ``S``, ``RY(theta)``, ``S†`` in circuit
          order, which is exact.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> rule = ZYZRewrite()
        >>> # the Euler angles of GateU become Z and Y rotations
        >>> rule.decompose_step(GateU(pi/2, pi/4, pi/3), [0], [], [])
        1-qubit circuit with 3 instructions:
        ├── RZ((1/3)*pi) @ q[0]
        ├── RY((1/2)*pi) @ q[0]
        └── RZ((1/4)*pi) @ q[0]
        <BLANKLINE>
    """

    def matches(self, op: Operation) -> bool:
        """Check whether an operation is a ``GateU`` or ``GateRX``.

        A ``GateU`` whose ``theta``, ``phi`` and ``lambda`` are all zero is
        not matched, since it is only a phase.

        Args:
            op: The operation to check.

        Returns:
            ``True`` for ``GateRX`` and for a non-identity ``GateU``.
        """
        from mimiqcircuits.operations.gates.standard.rotations import GateRX
        from mimiqcircuits.operations.gates.standard.u import GateU

        if isinstance(op, GateU):
            # Match unless it's the identity (all angles zero)
            return not (
                _iszero(op.theta) and _iszero(op.phi) and _iszero(op.lmbda)
            )
        if isinstance(op, GateRX):
            return True
        return False

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite a ``GateU`` or ``GateRX`` as Z and Y rotations.

        Args:
            op: The ``GateU`` or ``GateRX`` to rewrite.
            qubits: Qubit index the gate acts on.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If :meth:`matches` rejects ``op``.
        """
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.rotations import (
            GateRX,
            GateRY,
            GateRZ,
        )
        from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
        from mimiqcircuits.operations.gates.standard.u import GateU

        if not self.matches(op):
            raise DecompositionError(
                f"ZYZRewrite cannot decompose {type(op).__name__}"
            )

        circ = Circuit()
        q = qubits[0]

        if isinstance(op, GateU):
            # GateU(theta, phi, lambda, gamma) = e^{i*gamma} * RZ(phi) * RY(theta) * RZ(lambda)
            # Emit gates right-to-left (matrix multiplication order)
            lmbda = op.lmbda
            theta = op.theta
            phi = op.phi
            gamma = op.gamma

            if not _iszero(lmbda):
                circ.push(GateRZ(lmbda), q)
            if not _iszero(theta):
                circ.push(GateRY(theta), q)
            if not _iszero(phi):
                circ.push(GateRZ(phi), q)
            if not _iszero(gamma):
                circ.push(GateU(0, 0, 0, gamma), q)

        elif isinstance(op, GateRX):
            # RX(theta) = S * RY(theta) * Sdg
            # This converts X-rotation to Y-rotation via S conjugation
            circ.push(GateS(), q)
            circ.push(GateRY(op.theta), q)
            circ.push(GateSDG(), q)

        return circ


__all__ = ["ZYZRewrite"]
