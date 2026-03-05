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
    from mimiqcircuits import Circuit, Operation


class CanonicalRewrite(RewriteRule):
    """Rewrite rule that decomposes operations using their built-in decomposition.

    This rule delegates to each operation's ``_decompose`` method, which typically
    reduces gates to a minimal set of primitives (GateU, GateCX, etc.).

    The canonical decomposition is the default decomposition defined by each gate
    class. Most gates decompose down to GateU (arbitrary single-qubit) and GateCX
    (two-qubit entangling gate).

    Example:
        >>> from mimiqcircuits import GateH
        >>> from mimiqcircuits.decomposition import CanonicalRewrite
        >>> rule = CanonicalRewrite()
        >>> rule.matches(GateH())
        True
        >>> circ = rule.decompose_step(GateH(), [0], [], [])
        >>> print(circ)
        1-qubit circuit with 1 instruction:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
    """

    def matches(self, op: Operation) -> bool:
        """Check if this rule can decompose the operation.

        Matches operations that are non-terminal in the canonical basis.

        This rule applies to operations that are not already in canonical form
        (e.g., GateU, GateCX, measurement/reset/barrier, and conditionals
        whose inner operation is already canonical) do not match.

        Args:
            op: The operation to check.

        Returns:
            True if the operation should be rewritten by canonical decomposition.
        """
        from mimiqcircuits.decomposition.basis import CanonicalBasis

        return not CanonicalBasis().isterminal(op)

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose the operation using its built-in decomposition.

        Args:
            op: The operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices for the operation.
            zvars: Z-variable indices for the operation.

        Returns:
            A Circuit containing the decomposed instructions.
        """
        import mimiqcircuits as mc

        circ = mc.Circuit()
        op._decompose(circ, list(qubits), list(bits), list(zvars))
        return circ


__all__ = ["CanonicalRewrite"]
