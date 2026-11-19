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
"""RuleBasis: Wrapper to use a RewriteRule as a DecompositionBasis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


@dataclass(frozen=True, slots=True)
class RuleBasis(DecompositionBasis):
    """Decomposition basis that applies one rewrite rule until it no longer matches.

    An operation is terminal when the rule does not match it. When the rule
    does not match an ``IfStatement``, it is applied to the body, which stays
    conditional. :func:`decompose` and :func:`eachdecomposed` wrap a rule
    passed in place of a basis in a ``RuleBasis``.

    Args:
        rule: The rewrite rule to apply.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateRX(pi/2), 0)
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── RX((1/2)*pi) @ q[0]
        <BLANKLINE>
        >>> # rewrite every X and Y rotation into Z rotations; H is kept
        >>> decompose(c, RuleBasis(ToZRotationRewrite()))
        1-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── H @ q[0]
        ├── RZ((1/2)*pi) @ q[0]
        └── H @ q[0]
        <BLANKLINE>
    """

    rule: RewriteRule

    def isterminal(self, op: Operation) -> bool:
        """Check whether the rule leaves an operation alone.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if the rule does not match ``op``. For an
            ``IfStatement`` the rule does not match, the answer for its body.
        """
        from mimiqcircuits.operations.ifstatement import IfStatement

        # A rule that does not handle conditionals itself is applied to their
        # body.
        if isinstance(op, IfStatement) and not self.rule.matches(op):
            return self.isterminal(op.get_operation())
        return not self.rule.matches(op)

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Apply the rule once to an operation.

        Args:
            op: The operation to rewrite.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite. For an
            ``IfStatement`` the rule does not match, the rewrite of its body,
            with each instruction made conditional.

        Raises:
            DecompositionError: If the rule cannot rewrite ``op``.
        """
        from mimiqcircuits.decomposition.abstract import _NO_CONTEXT

        return self.decompose_ctx(op, qubits, bits, zvars, _NO_CONTEXT)

    def decompose_ctx(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx,
    ) -> Circuit:
        """Apply the rule once, with a context, as :meth:`decompose` does."""
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.ifstatement import _decompose_ifstatement

        if isinstance(op, IfStatement) and not self.rule.matches(op):
            return _decompose_ifstatement(self, op, qubits, bits, zvars, ctx)
        return self.rule.decompose_step_ctx(op, qubits, bits, zvars, ctx)

    def __repr__(self) -> str:
        return f"RuleBasis({self.rule!r})"


__all__ = ["RuleBasis"]
