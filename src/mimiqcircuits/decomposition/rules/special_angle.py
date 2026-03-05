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
"""Special angle rewrite rule for Clifford+T decomposition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


def _get_pi_factor(val) -> int | None:
    """Check if value is a multiple of pi/4 and return the factor k in {0,1,...,7}.

    Args:
        val: The angle value to check.

    Returns:
        The factor k if val = k * pi/4, or None if not a special angle.
    """
    import symengine as se
    import sympy as sp

    # Check for symbolic values
    if isinstance(val, (se.Basic, sp.Basic)):
        try:
            val = complex(val).real
        except (TypeError, ValueError):
            return None

    # Check for multiples of pi/4
    factor = val / (math.pi / 4)
    k = round(factor)
    if math.isclose(val, k * (math.pi / 4), abs_tol=1e-10):
        return k % 8
    return None


@dataclass(frozen=True, slots=True)
class SpecialAngleRewrite(RewriteRule):
    """Rewrite rule for decomposing rotations with special angles to Clifford+T.

    Decomposes single-qubit rotations whose angle is a multiple of pi/4 into
    explicit Clifford or Clifford+T gates.

    This rule only matches rotations with angles k*pi/4 (or k*pi/2 in Clifford-only
    mode). Generic rotations with arbitrary angles are not matched.

    Args:
        only_cliffords: If True, only match angles k*pi/2 (Clifford gates only).
                       If False (default), match angles k*pi/4 (Clifford+T).

    Supported operations:
        - ``GateRX(theta)`` where theta = k*pi/4
        - ``GateRY(theta)`` where theta = k*pi/4
        - ``GateRZ(lambda)`` where lambda = k*pi/4

    Example:
        >>> from mimiqcircuits import GateRZ
        >>> from mimiqcircuits.decomposition import SpecialAngleRewrite
        >>> from symengine import pi
        >>> rule = SpecialAngleRewrite()
        >>> rule.matches(GateRZ(pi/4))  # True, decomposes to T
        True
        >>> rule.matches(GateRZ(0.123))  # False, not a special angle
        False
    """

    only_cliffords: bool = False

    def matches(self, op: Operation) -> bool:
        """Check if this rule can decompose the operation.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is a rotation with a special angle.
        """
        import mimiqcircuits as mc

        if isinstance(op, mc.GateRX):
            k = _get_pi_factor(op.theta)
        elif isinstance(op, mc.GateRY):
            k = _get_pi_factor(op.theta)
        elif isinstance(op, mc.GateRZ):
            k = _get_pi_factor(op.lmbda)
        else:
            return False

        if k is None:
            return False

        # In Clifford-only mode, only match even multiples (k*pi/2)
        if self.only_cliffords:
            return k % 2 == 0

        return True

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose the rotation to Clifford+T gates.

        Args:
            op: The rotation operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A Circuit containing the decomposed instructions.

        Raises:
            DecompositionError: If the operation cannot be decomposed.
        """
        import mimiqcircuits as mc

        circ = mc.Circuit()
        q = qubits[0]

        if isinstance(op, mc.GateRZ):
            k = _get_pi_factor(op.lmbda)
            if k is None:
                raise DecompositionError(
                    f"SpecialAngleRewrite: RZ angle {op.lmbda} is not a multiple of pi/4"
                )
            self._decompose_rz(circ, q, k)

        elif isinstance(op, mc.GateRX):
            k = _get_pi_factor(op.theta)
            if k is None:
                raise DecompositionError(
                    f"SpecialAngleRewrite: RX angle {op.theta} is not a multiple of pi/4"
                )
            self._decompose_rx(circ, q, k)

        elif isinstance(op, mc.GateRY):
            k = _get_pi_factor(op.theta)
            if k is None:
                raise DecompositionError(
                    f"SpecialAngleRewrite: RY angle {op.theta} is not a multiple of pi/4"
                )
            self._decompose_ry(circ, q, k)

        else:
            raise DecompositionError(
                f"SpecialAngleRewrite cannot decompose {type(op).__name__}"
            )

        return circ

    def _decompose_rz(self, circ: Circuit, q: int, k: int) -> None:
        """Decompose RZ(k*pi/4) to Clifford+T gates."""
        import mimiqcircuits as mc

        # RZ decomposition table for k in {0,1,...,7}:
        #   k=0: identity
        #   k=1: T
        #   k=2: S
        #   k=3: S*T
        #   k=4: Z
        #   k=5: Z*T = S*S*T
        #   k=6: Sdg = S*S*S
        #   k=7: Tdg
        match k:
            case 0:
                pass  # identity
            case 1:
                circ.push(mc.GateT(), q)
            case 2:
                circ.push(mc.GateS(), q)
            case 3:
                circ.push(mc.GateS(), q)
                circ.push(mc.GateT(), q)
            case 4:
                circ.push(mc.GateZ(), q)
            case 5:
                circ.push(mc.GateZ(), q)
                circ.push(mc.GateT(), q)
            case 6:
                circ.push(mc.GateSDG(), q)
            case 7:
                circ.push(mc.GateTDG(), q)

    def _decompose_rx(self, circ: Circuit, q: int, k: int) -> None:
        """Decompose RX(k*pi/4) to Clifford+T gates."""
        import mimiqcircuits as mc

        # TX = RX(pi/4) = H*T*H
        def push_tx():
            circ.push(mc.GateH(), q)
            circ.push(mc.GateT(), q)
            circ.push(mc.GateH(), q)

        # TXdg = RX(-pi/4) = H*Tdg*H
        def push_txdg():
            circ.push(mc.GateH(), q)
            circ.push(mc.GateTDG(), q)
            circ.push(mc.GateH(), q)

        # RX decomposition table for k in {0,1,...,7}:
        #   k=0: identity
        #   k=1: TX
        #   k=2: SX
        #   k=3: SX*TX
        #   k=4: X
        #   k=5: X*TX
        #   k=6: SXdg
        #   k=7: TXdg
        match k:
            case 0:
                pass  # identity
            case 1:
                push_tx()
            case 2:
                circ.push(mc.GateSX(), q)
            case 3:
                circ.push(mc.GateSX(), q)
                push_tx()
            case 4:
                circ.push(mc.GateX(), q)
            case 5:
                circ.push(mc.GateX(), q)
                push_tx()
            case 6:
                circ.push(mc.GateSXDG(), q)
            case 7:
                push_txdg()

    def _decompose_ry(self, circ: Circuit, q: int, k: int) -> None:
        """Decompose RY(k*pi/4) to Clifford+T gates."""
        import mimiqcircuits as mc

        # TY = RY(pi/4) = Sdg*H*T*H*S
        def push_ty():
            circ.push(mc.GateSDG(), q)
            circ.push(mc.GateH(), q)
            circ.push(mc.GateT(), q)
            circ.push(mc.GateH(), q)
            circ.push(mc.GateS(), q)

        # TYdg = RY(-pi/4) = Sdg*H*Tdg*H*S
        def push_tydg():
            circ.push(mc.GateSDG(), q)
            circ.push(mc.GateH(), q)
            circ.push(mc.GateTDG(), q)
            circ.push(mc.GateH(), q)
            circ.push(mc.GateS(), q)

        # RY decomposition table for k in {0,1,...,7}:
        #   k=0: identity
        #   k=1: TY
        #   k=2: SY
        #   k=3: SY*TY
        #   k=4: Y
        #   k=5: Y*TY
        #   k=6: SYdg
        #   k=7: TYdg
        match k:
            case 0:
                pass  # identity
            case 1:
                push_ty()
            case 2:
                circ.push(mc.GateSY(), q)
            case 3:
                circ.push(mc.GateSY(), q)
                push_ty()
            case 4:
                circ.push(mc.GateY(), q)
            case 5:
                circ.push(mc.GateY(), q)
                push_ty()
            case 6:
                circ.push(mc.GateSYDG(), q)
            case 7:
                push_tydg()


__all__ = ["SpecialAngleRewrite"]
