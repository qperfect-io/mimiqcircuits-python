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
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


def _get_pi_factor(val) -> int | None:
    """Check if value is a multiple of pi/4 and return the factor k in {0,1,...,7}.

    Args:
        val: The angle value to check.

    Returns:
        The factor k if val = k * pi/4, or None if not a special angle.
    """
    from mimiqcircuits._sympycheck import _is_basic

    # Check for symbolic values
    if _is_basic(val):
        try:
            val = complex(val).real
        except (TypeError, ValueError, RuntimeError):
            return None

    # Check for multiples of pi/4
    factor = val / (math.pi / 4)
    if not math.isfinite(factor):
        return None
    k = round(factor)
    if math.isclose(val, k * (math.pi / 4), abs_tol=1e-10):
        return k % 8
    return None


@dataclass(frozen=True, slots=True)
class SpecialAngleRewrite(RewriteRule):
    """Rewrite rule that turns rotations by multiples of ``pi/4`` into Clifford+T gates.

    A rotation by ``k*pi/4`` equals, up to a global phase, a short product of
    Clifford gates and at most one ``T`` (``T†``, or ``T`` conjugated into the
    X or Y axis). This rule writes that product out exactly. For example
    ``RZ(pi/4)`` becomes ``T``, ``RZ(pi/2)`` becomes ``S`` and ``RX(pi)``
    becomes ``X``. Angles are compared to a multiple of ``pi/4`` with a
    relative tolerance of ``1e-9`` (absolute ``1e-10`` near zero), modulo
    ``2*pi``. Other angles are not matched.

    Args:
        only_cliffords: If ``True``, match only angles ``k*pi/2``, whose
            rewrite uses Clifford gates only. Defaults to ``False``, which
            matches every ``k*pi/4``.

    Supported Operations:
        - ``GateRX(theta)`` with ``theta = k*pi/4``
        - ``GateRY(theta)`` with ``theta = k*pi/4``
        - ``GateRZ(lambda)`` with ``lambda = k*pi/4``

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> rule = SpecialAngleRewrite()
        >>> # 3*pi/4 is a special angle: RZ(3*pi/4) is S followed by T
        >>> rule.decompose_step(GateRZ(3*pi/4), [0], [], [])
        1-qubit circuit with 2 instructions:
        ├── S @ q[0]
        └── T @ q[0]
        <BLANKLINE>
        >>> # an arbitrary angle is not matched
        >>> rule.matches(GateRZ(0.123))
        False
        >>> # pi/4 needs a T gate, which the Clifford-only mode refuses
        >>> SpecialAngleRewrite(only_cliffords=True).matches(GateRZ(pi/4))
        False
    """

    only_cliffords: bool = False

    def matches(self, op: Operation) -> bool:
        """Check whether an operation is a rotation by a special angle.

        Args:
            op: The operation to check.

        Returns:
            ``True`` for a ``GateRX``, ``GateRY`` or ``GateRZ`` whose angle is
            a multiple of ``pi/4``, or of ``pi/2`` when ``only_cliffords`` is
            set.
        """
        from mimiqcircuits.operations.gates.standard.rotations import (
            GateRX,
            GateRY,
            GateRZ,
        )

        if isinstance(op, GateRX):
            k = _get_pi_factor(op.theta)
        elif isinstance(op, GateRY):
            k = _get_pi_factor(op.theta)
        elif isinstance(op, GateRZ):
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
        """Rewrite a rotation by a multiple of ``pi/4`` as Clifford+T gates.

        The result equals the rotation up to a global phase, and is empty for
        a multiple of ``2*pi``.

        Args:
            op: The ``GateRX``, ``GateRY`` or ``GateRZ`` to rewrite.
            qubits: Qubit index the rotation acts on.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If :meth:`matches` refuses ``op``.
        """
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.rotations import (
            GateRX,
            GateRY,
            GateRZ,
        )

        if not self.matches(op):
            raise DecompositionError(f"SpecialAngleRewrite cannot decompose {op}")

        circ = Circuit()
        q = qubits[0]

        if isinstance(op, GateRZ):
            k = _get_pi_factor(op.lmbda)
            if k is None:
                raise DecompositionError(
                    f"SpecialAngleRewrite: RZ angle {op.lmbda} is not a multiple of pi/4"
                )
            self._decompose_rz(circ, q, k)

        elif isinstance(op, GateRX):
            k = _get_pi_factor(op.theta)
            if k is None:
                raise DecompositionError(
                    f"SpecialAngleRewrite: RX angle {op.theta} is not a multiple of pi/4"
                )
            self._decompose_rx(circ, q, k)

        elif isinstance(op, GateRY):
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
        from mimiqcircuits.operations.gates.standard.pauli import GateZ
        from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
        from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG

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
                circ.push(GateT(), q)
            case 2:
                circ.push(GateS(), q)
            case 3:
                circ.push(GateS(), q)
                circ.push(GateT(), q)
            case 4:
                circ.push(GateZ(), q)
            case 5:
                circ.push(GateZ(), q)
                circ.push(GateT(), q)
            case 6:
                circ.push(GateSDG(), q)
            case 7:
                circ.push(GateTDG(), q)

    def _decompose_rx(self, circ: Circuit, q: int, k: int) -> None:
        """Decompose RX(k*pi/4) to Clifford+T gates."""
        from mimiqcircuits.operations.gates.standard.hadamard import GateH
        from mimiqcircuits.operations.gates.standard.pauli import GateX
        from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
        from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG

        # TX = RX(pi/4) = H*T*H
        def push_tx():
            circ.push(GateH(), q)
            circ.push(GateT(), q)
            circ.push(GateH(), q)

        # TXdg = RX(-pi/4) = H*Tdg*H
        def push_txdg():
            circ.push(GateH(), q)
            circ.push(GateTDG(), q)
            circ.push(GateH(), q)

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
                circ.push(GateSX(), q)
            case 3:
                circ.push(GateSX(), q)
                push_tx()
            case 4:
                circ.push(GateX(), q)
            case 5:
                circ.push(GateX(), q)
                push_tx()
            case 6:
                circ.push(GateSXDG(), q)
            case 7:
                push_txdg()

    def _decompose_ry(self, circ: Circuit, q: int, k: int) -> None:
        """Decompose RY(k*pi/4) to Clifford+T gates."""
        from mimiqcircuits.operations.gates.standard.hadamard import GateH
        from mimiqcircuits.operations.gates.standard.pauli import GateY
        from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
        from mimiqcircuits.operations.gates.standard.sy import GateSY, GateSYDG
        from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG

        # TY = RY(pi/4) = Sdg*H*T*H*S
        def push_ty():
            circ.push(GateSDG(), q)
            circ.push(GateH(), q)
            circ.push(GateT(), q)
            circ.push(GateH(), q)
            circ.push(GateS(), q)

        # TYdg = RY(-pi/4) = Sdg*H*Tdg*H*S
        def push_tydg():
            circ.push(GateSDG(), q)
            circ.push(GateH(), q)
            circ.push(GateTDG(), q)
            circ.push(GateH(), q)
            circ.push(GateS(), q)

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
                circ.push(GateSY(), q)
            case 3:
                circ.push(GateSY(), q)
                push_ty()
            case 4:
                circ.push(GateY(), q)
            case 5:
                circ.push(GateY(), q)
                push_ty()
            case 6:
                circ.push(GateSYDG(), q)
            case 7:
                push_tydg()


__all__ = ["SpecialAngleRewrite"]
