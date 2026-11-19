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
    _AncillaOptions,
)

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


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
    2. ``GateECR`` to ``S``, ``SX``, ``CX`` and ``X``, as in
       :class:`StimBasis`, with no T gate.
    3. :class:`ToffoliToCliffordTRewrite`: ``GateCCX`` to Clifford+T with 7 T
       gates.
    4. A ``GateU`` whose angles are all zero (a pure global phase) is
       dropped.
    5. :class:`ZYZRewrite`: ``GateU`` to ``RZ*RY*RZ``, and ``GateRX`` to
       ``GateRY`` between ``S`` and ``S†``.
    6. :class:`ToZRotationRewrite`: ``GateRY`` to ``GateRZ`` and Cliffords.
    7. The ``rotations`` rule (default :class:`GridSynthRewrite`): remaining
       ``GateRZ`` to an approximate Clifford+T sequence.
    8. :class:`CanonicalRewrite`: any other operation, toward ``GateU`` and
       ``GateCX``. A ``GateRX``, ``GateRY`` or ``GateRZ`` that reaches this
       point has a symbolic angle and raises :class:`DecompositionError`.

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
        borrow: Let decompositions borrow qubits of the circuit that the
            operation does not act on. A borrowed qubit can be in any state;
            the decomposition leaves it as it found it.
        max_ancillas: Let decompositions add up to this many clean ancillas,
            numbered after the qubits of the circuit and marked with
            :class:`AncillaAlloc` and :class:`AncillaFree`. Ancillas freed by
            one operation are reused by the next.
        allow_measurement: With ``max_ancillas``, let decompositions undo
            steps by measuring an ancilla into one extra bit, numbered after
            the bits of the circuit. A circuit without bits is refused with a
            ``ValueError``. With a clean ancilla, a ``GateCCX`` then costs
            4 T instead of 7.

    The three ancilla options are off by default, and only change how
    multi-controlled X gates are decomposed. Circuits whose width is unknown,
    such as a lone operation, get no ancillas.

    Raises:
        ValueError: If both ``sk_depth`` and ``rotations`` are given, or if
            ``rotations`` does not match ``GateRZ``.

    Decomposing an operation that no rule handles, such as a noise channel
    other than the Pauli ones above, raises a :class:`DecompositionError`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> c = Circuit()
        >>> c.push(GateRZ(pi/4), 0)
        1-qubit circuit with 1 instruction:
        └── RZ((1/4)*pi) @ q[0]
        <BLANKLINE>
        >>> # a special angle is decomposed exactly
        >>> decompose(c, CliffordTBasis())
        1-qubit circuit with 1 instruction:
        └── T @ q[0]
        <BLANKLINE>

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
        >>> def tcount(c):
        ...     return sum(isinstance(inst.operation, (GateT, GateTDG)) for inst in c)
        >>> tcount(decomposed3) < tcount(decomposed2)
        True

        >>> # Solovay-Kitaev instead of gridsynth
        >>> decomposed4 = decompose(c2, CliffordTBasis(sk_depth=0))
    """

    rotations: RewriteRule
    ancillas: _AncillaOptions

    def __init__(
        self,
        sk_depth: int | None = None,
        rotations: RewriteRule | None = None,
        borrow: bool = False,
        max_ancillas: int = 0,
        allow_measurement: bool = False,
    ):
        from mimiqcircuits.operations.gates.standard.rotations import GateRZ
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
        if not rotations.matches(GateRZ(0.1)):
            raise ValueError(
                f"rotations rule {type(rotations).__name__} does not match GateRZ"
            )
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(
            self,
            "ancillas",
            _AncillaOptions(
                borrow=borrow,
                max_ancillas=max_ancillas,
                allow_measurement=allow_measurement,
            ),
        )

    def _ancillaoptions(self):
        return self.ancillas

    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in the Clifford+T basis.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is terminal in :class:`StimBasis`, is
            ``GateT``, ``GateTDG`` or a classical operation, or is an
            ``IfStatement`` whose body is terminal in this basis.
        """
        from mimiqcircuits.classical.abstract_classical import AbstractClassical
        from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.decomposition.basis.stim import StimBasis

        if isinstance(op, (GateT, GateTDG)):
            return True

        # Classical operations touch no qubit, so they need no decomposition
        # here even though Stim cannot run them.
        if isinstance(op, AbstractClassical):
            return True

        # Clifford+T is a gate set, so any conditional gate of it is terminal,
        # not only the single-bit conditional Paulis that Stim accepts.
        if isinstance(op, IfStatement):
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
        """Rewrite a non-terminal operation one step toward Clifford+T.

        Applies the first matching rule of the list in the class description.

        Args:
            op: The operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices for the operation.
            zvars: Z-variable indices for the operation.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If no rule applies to ``op``.
        """
        from mimiqcircuits.decomposition.abstract import _NO_CONTEXT

        return self._decompose_cliffordt(op, qubits, bits, zvars, _NO_CONTEXT)

    def decompose_ctx(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx,
    ) -> Circuit:
        """Decompose with a context, which only some operations use.

        Operations without a rule that uses the context go to
        :meth:`decompose`, so an override of it keeps being called.
        """
        from mimiqcircuits.decomposition.abstract import _uses_context

        if _uses_context(op):
            return self._decompose_cliffordt(op, qubits, bits, zvars, ctx)
        return self.decompose(op, qubits, bits, zvars)

    def _decompose_cliffordt(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx,
    ) -> Circuit:
        """The rewrite of :meth:`decompose`, with a context."""
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.u import GateU
        from mimiqcircuits.operations.gates.standard.rotations import (
            GateRX,
            GateRY,
            GateRZ,
        )
        from mimiqcircuits.operations.gates.standard.ecr import GateECR
        from mimiqcircuits.decomposition.basis.stim import _decompose_ecr
        from mimiqcircuits.operations.ifstatement import IfStatement
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
        if isinstance(op, IfStatement):
            return _decompose_ifstatement(self, op, qubits, bits, zvars, ctx)

        # Try rules in priority order
        # 1. Special angles first (exact decomposition)
        special_angle = SpecialAngleRewrite()
        if special_angle.matches(op):
            return special_angle.decompose_step(op, qubits, bits, zvars)

        # ECR is Clifford; the canonical rule would spend two T gates on it.
        if isinstance(op, GateECR):
            return _decompose_ecr(qubits)

        # 2. Optimal Toffoli decomposition without measurements; with them,
        # CanonicalRewrite uses 4 T
        toffoli = ToffoliToCliffordTRewrite()
        if toffoli.matches(op) and not (ctx.can_measure() and ctx.available() > 0):
            return toffoli.decompose_step(op, qubits, bits, zvars)

        # A bare GateU with zero angles is only a global phase, so it expands
        # to nothing. ZYZRewrite refuses it, and a rotations rule that matches
        # only GateRZ cannot absorb it. Under a Control the phase is a relative
        # phase; that case is a Control, not a GateU, and goes to
        # CanonicalRewrite.
        if isinstance(op, GateU) and all(
            _iszero(p) for p in (op.theta, op.phi, op.lmbda)
        ):
            return Circuit()

        # 3. ZYZ for GateU and GateRX (will be further decomposed)
        zyz = ZYZRewrite()
        if zyz.matches(op):
            return zyz.decompose_step(op, qubits, bits, zvars)

        # 4. ToZRotation for RY (RX was taken by ZYZ)
        to_z = ToZRotationRewrite()
        if to_z.matches(op):
            return to_z.decompose_step(op, qubits, bits, zvars)

        # 5. Arbitrary Z-rotations (approximate)
        if self.rotations.matches(op):
            return self.rotations.decompose_step(op, qubits, bits, zvars)

        # A rotation left here has a symbolic angle that no rule can
        # approximate; the canonical rule would cycle RZ -> U -> RZ forever.
        if isinstance(op, (GateRX, GateRY, GateRZ)):
            raise DecompositionError(
                f"Rotation {op} has a symbolic angle and cannot be decomposed "
                "to the Clifford+T basis. Evaluate it first."
            )

        # 6. Canonical fallback
        canonical = CanonicalRewrite()
        if canonical.matches(op):
            return canonical.decompose_step_ctx(op, qubits, bits, zvars, ctx)

        raise DecompositionError(
            f"Operation {op.name} cannot be decomposed to Clifford+T basis."
        )


__all__ = ["CliffordTBasis"]
