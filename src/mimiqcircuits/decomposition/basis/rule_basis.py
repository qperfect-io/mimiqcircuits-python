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
    from mimiqcircuits import Circuit, Operation


@dataclass(frozen=True, slots=True)
class RuleBasis(DecompositionBasis):
    """A decomposition basis that recursively applies a single rewrite rule.

    This allows any :class:`RewriteRule` to be used directly as a decomposition
    basis. The rule is applied recursively until no more operations match.

    An operation is terminal if the rule does not match it.

    Args:
        rule: The rewrite rule to apply.

    Example:
        >>> from mimiqcircuits import Circuit, GateH
        >>> from mimiqcircuits.decomposition import decompose, RuleBasis, ZYZRewrite
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> # Decompose using ZYZ rewrite rule
        >>> decomposed = decompose(c, RuleBasis(ZYZRewrite()))
    """

    rule: RewriteRule

    def isterminal(self, op: Operation) -> bool:
        """Check if the operation is terminal (rule does not match).

        Args:
            op: The operation to check.

        Returns:
            True if the rule does not match this operation.
        """
        return not self.rule.matches(op)

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Apply the rewrite rule to decompose the operation.

        Args:
            op: The operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices for the operation.
            zvars: Z-variable indices for the operation.

        Returns:
            A Circuit containing the decomposed instructions.
        """
        return self.rule.decompose_step(op, qubits, bits, zvars)

    def __repr__(self) -> str:
        return f"RuleBasis({self.rule!r})"


__all__ = ["RuleBasis"]
