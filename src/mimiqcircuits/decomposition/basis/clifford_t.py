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

from mimiqcircuits.decomposition.abstract import (
    DecompositionBasis,
    DecompositionError,
    RewriteRule,
)

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


@dataclass(frozen=True, init=False)
class CliffordTBasis(DecompositionBasis):
    """Decomposition basis targeting the Clifford+T gate set.

    Clifford+T is the standard universal gate set of error-corrected quantum
    computing. The terminal operations of this basis are those of
    :class:`StimBasis` plus ``GateT`` and ``GateTDG``, so the two bases always
    keep the same Clifford and stabilizer operations:

    - single-qubit Cliffords: ``GateID``, ``GateX``, ``GateY``, ``GateZ``,
      ``GateH``, ``GateHXY``, ``GateHYZ``, ``GateS``, ``GateSDG``, ``GateSX``,
      ``GateSXDG``, ``GateSY``, ``GateSYDG``;
    - two-qubit Cliffords: ``GateCX``, ``GateCY``, ``GateCZ``, ``GateSWAP``,
      ``GateISWAP``, ``GateISWAPDG``, ``GateDCX``, ``GateECR``;
    - Pauli rotations by ``+-pi/2``: ``GateRXX``, ``GateRYY``, ``GateRZZ``,
      and ``RPauli`` with a non-identity Pauli string;
    - the non-Clifford ``GateT`` and ``GateTDG``;
    - measurements, resets and measure-and-resets in the X, Y and Z bases,
      ``MeasureXX``, ``MeasureYY``, ``MeasureZZ`` and ``MeasurePauli``;
    - Pauli noise: ``PauliX``, ``PauliY``, ``PauliZ``, ``Depolarizing1``,
      ``Depolarizing2`` and ``PauliNoise``;
    - barriers and annotations;
    - classical operations, which :class:`StimBasis` does not keep;
    - an ``IfStatement`` whose body is terminal in this basis, with any
      condition. :class:`StimBasis` only keeps a conditional ``GateX``,
      ``GateY`` or ``GateZ`` on a single bit.

    Non-terminal operations are rewritten by the first rule that matches, in
    this order:

    1. :class:`SpecialAngleRewrite`: rotations by multiples of ``pi/4`` to
       exact Clifford+T.
    2. :class:`ToffoliToCliffordTRewrite`: ``GateCCX`` to Clifford+T with 7 T
       gates.
    3. A ``GateU`` whose angles are all zero (a pure global phase) is
       dropped.
    4. :class:`ZYZRewrite`: ``GateU`` to ``RZ*RY*RZ``.
    5. :class:`ToZRotationRewrite`: ``GateRX``, ``GateRY`` to ``GateRZ`` and
       Cliffords.
    6. The ``rotations`` rule (default :class:`GridSynthRewrite`): remaining
       ``GateRZ`` to an approximate Clifford+T sequence.
    7. :class:`CanonicalRewrite`: any other operation, toward ``GateU`` and
       ``GateCX``.

    The ``rotations`` rule only sees angles that step 1 could not decompose
    exactly. The body of an ``IfStatement`` goes through the same pipeline
    and stays conditional. No T-count optimization is performed on the
    result.

    Args:
        sk_depth: Use :class:`SolovayKitaevRewrite` with this recursion depth
            as the ``rotations`` rule.
        rotations: Rule that synthesizes the remaining Z-rotations. The
            default ``GridSynthRewrite()`` approximates each rotation to
            operator-norm error ``1e-10``. The rule must match ``GateRZ`` with
            a numeric angle and emit only gates of this basis.

    Raises:
        ValueError: If both ``sk_depth`` and ``rotations`` are given, or if
            ``rotations`` does not match ``GateRZ``.

    Decomposing an operation that no rule handles, such as a noise channel
    other than the Pauli ones above, raises a :class:`DecompositionError`.

    Example:
        >>> from mimiqcircuits import Circuit, GateH, GateT, GateRZ
        >>> from mimiqcircuits.decomposition import decompose, CliffordTBasis
        >>> from mimiqcircuits.decomposition import GridSynthRewrite
        >>> from symengine import pi
        >>> c = Circuit()
        >>> c.push(GateRZ(pi/4), 0)  # special angle, decomposed exactly
        1-qubit circuit with 1 instruction:
        └── RZ((1/4)*pi) @ q[0]
        <BLANKLINE>
        >>> decomposed = decompose(c, CliffordTBasis())

        >>> # An arbitrary angle is approximated to 1e-10 by gridsynth
        >>> c2 = Circuit()
        >>> c2.push(GateRZ(0.123), 0)
        1-qubit circuit with 1 instruction:
        └── RZ(0.123) @ q[0]
        <BLANKLINE>
        >>> decomposed2 = decompose(c2, CliffordTBasis())

        >>> # A looser error bound per rotation gives fewer T gates
        >>> basis = CliffordTBasis(rotations=GridSynthRewrite(epsilon=1e-6))
        >>> decomposed3 = decompose(c2, basis)

        >>> # Solovay-Kitaev instead of gridsynth
        >>> decomposed4 = decompose(c2, CliffordTBasis(sk_depth=0))
    """

    rotations: RewriteRule

    def __init__(
        self,
        sk_depth: int | None = None,
        rotations: RewriteRule | None = None,
    ):
        import mimiqcircuits as mc
        from mimiqcircuits.decomposition.rules import (
            GridSynthRewrite,
            SolovayKitaevRewrite,
        )

        if sk_depth is not None and rotations is not None:
            raise ValueError("give either sk_depth or rotations, not both")
        if rotations is None:
            if sk_depth is None:
                rotations = GridSynthRewrite()
            else:
                rotations = SolovayKitaevRewrite(depth=sk_depth)
        if not rotations.matches(mc.GateRZ(0.1)):
            raise ValueError(
                f"rotations rule {type(rotations).__name__} does not match GateRZ"
            )
        object.__setattr__(self, "rotations", rotations)

    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in the Clifford+T basis.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is terminal in :class:`StimBasis`, is
            ``GateT``, ``GateTDG`` or a classical operation, or is an
            ``IfStatement`` whose body is terminal in this basis.
        """
        import mimiqcircuits as mc
        from mimiqcircuits.decomposition.basis.stim import StimBasis

        if isinstance(op, (mc.GateT, mc.GateTDG)):
            return True

        # Classical operations touch no qubit, so they need no decomposition
        # here even though Stim cannot run them.
        if isinstance(op, mc.AbstractClassical):
            return True

        # Clifford+T is a gate set, so any conditional gate of it is terminal,
        # not only the single-bit conditional Paulis that Stim accepts.
        if isinstance(op, mc.IfStatement):
            return self.isterminal(op.get_operation())

        # Delegating to StimBasis keeps the two terminal sets from drifting
        # apart.
        return StimBasis().isterminal(op)

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose a non-terminal operation toward Clifford+T.

        Applies the rules listed in the class docstring, in that order.

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
        import mimiqcircuits as mc
        from mimiqcircuits.decomposition.rules import (
            CanonicalRewrite,
            SpecialAngleRewrite,
            ToZRotationRewrite,
            ToffoliToCliffordTRewrite,
            ZYZRewrite,
        )
        from mimiqcircuits.decomposition.rules.zyz import _iszero
        from mimiqcircuits.operations.ifstatement import _decompose_ifstatement

        # The body of a conditional goes through this pipeline too, so that,
        # for example, a conditional rotation by a multiple of pi/4 stays exact.
        if isinstance(op, mc.IfStatement):
            return _decompose_ifstatement(self, op, qubits, bits, zvars)

        # Try rules in priority order
        # 1. Special angles first (exact decomposition)
        special_angle = SpecialAngleRewrite()
        if special_angle.matches(op):
            return special_angle.decompose_step(op, qubits, bits, zvars)

        # 2. Optimal Toffoli decomposition
        toffoli = ToffoliToCliffordTRewrite()
        if toffoli.matches(op):
            return toffoli.decompose_step(op, qubits, bits, zvars)

        # A bare GateU with zero angles is only a global phase, so it expands
        # to nothing. ZYZRewrite refuses it, and a rotations rule that matches
        # only GateRZ cannot absorb it. Under a Control the phase is a relative
        # phase; that case is a Control, not a GateU, and goes to
        # CanonicalRewrite.
        if isinstance(op, mc.GateU) and all(
            _iszero(p) for p in (op.theta, op.phi, op.lmbda)
        ):
            return mc.Circuit()

        # 3. ZYZ for GateU (will be further decomposed)
        zyz = ZYZRewrite()
        if zyz.matches(op):
            return zyz.decompose_step(op, qubits, bits, zvars)

        # 4. ToZRotation for RX/RY
        to_z = ToZRotationRewrite()
        if to_z.matches(op):
            return to_z.decompose_step(op, qubits, bits, zvars)

        # 5. Arbitrary Z-rotations (approximate)
        if self.rotations.matches(op):
            return self.rotations.decompose_step(op, qubits, bits, zvars)

        # 6. Canonical fallback
        canonical = CanonicalRewrite()
        if canonical.matches(op):
            return canonical.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} cannot be decomposed to Clifford+T basis."
        )


__all__ = ["CliffordTBasis"]
