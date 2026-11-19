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
"""Canonical rewrite rule for decomposing operations to GateU and GateCX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


class CanonicalRewrite(RewriteRule):
    """Rewrite rule that applies the built-in decomposition of each operation.

    Every operation class defines one decomposition step in terms of simpler
    operations; this rule applies it. Repeated, the steps end at ``GateU``,
    ``GateCX`` and the non-unitary primitives, which is what
    :class:`CanonicalBasis` does. One step can stop earlier: ``GateCCX``, for
    example, becomes ``GateH``, ``GateT``, ``GateTDG`` and ``GateCX``.

    It is the default rule of :func:`decompose_step` and the fallback of the
    other bases.

    Examples:
        >>> from mimiqcircuits import *
        >>> rule = CanonicalRewrite()
        >>> # H has a built-in decomposition into GateU
        >>> rule.matches(GateH())
        True
        >>> rule.decompose_step(GateH(), [0], [], [])
        1-qubit circuit with 1 instruction:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
        >>> # GateU is a primitive, so there is nothing to rewrite
        >>> rule.matches(GateU(0.1, 0.2, 0.3))
        False
    """

    def matches(self, op: Operation) -> bool:
        """Check whether an operation has a built-in decomposition.

        Matches every operation whose ``supports_canonical_rewrite()`` is
        true. The answer does not depend on any basis: an operation that
        :class:`CanonicalBasis` keeps terminal (such as ``ResetX``) still
        matches, so other bases can expand it. Primitives without a
        decomposition (``GateU``, ``GateCX``, ``Measure``, ``Reset``,
        ``Barrier``, ...) do not match.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if ``op`` has a built-in decomposition.
        """
        return op.supports_canonical_rewrite()

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Apply one step of the built-in decomposition of an operation.

        No check is made that :meth:`matches` accepts ``op``. A primitive
        returns a circuit holding just itself.

        Args:
            op: The operation to decompose.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the decomposition.
        """
        from mimiqcircuits.circuit import Circuit

        circ = Circuit()
        op._decompose(circ, list(qubits), list(bits), list(zvars))
        return circ

    def decompose_step_ctx(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx,
    ) -> Circuit:
        from mimiqcircuits.circuit import Circuit

        circ = Circuit()
        op._decompose_ctx(circ, list(qubits), list(bits), list(zvars), ctx)
        return circ


__all__ = ["CanonicalRewrite"]
