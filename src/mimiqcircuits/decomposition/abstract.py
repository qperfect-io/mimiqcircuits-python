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
"""Abstract base classes for the decomposition framework.

This module defines the core abstractions for quantum circuit decomposition:

- :class:`RewriteRule`: Single-step transformation rules
- :class:`DecompositionBasis`: Recursive decomposition to a target gate set
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


class DecompositionError(Exception):
    """Raised when an operation cannot be decomposed."""

    pass


class RewriteRule(ABC):
    """Abstract base class for single-step rewrite rules.

    A RewriteRule defines *how* to transform an operation into a sequence of
    simpler operations, but not *when* to stop. For recursive decomposition
    to a target set of operations, see :class:`DecompositionBasis`.

    Subclasses must implement:
        - :meth:`matches`: Return True if this rule can transform the operation
        - :meth:`decompose_step`: Apply the transformation

    Example:
        >>> class MyRewrite(RewriteRule):
        ...     def matches(self, op: Operation) -> bool:
        ...         return isinstance(op, GateToffoli)
        ...
        ...     def decompose_step(self, op, qubits, bits, zvars):
        ...         from mimiqcircuits import Circuit
        ...         circ = Circuit()
        ...         # ... decomposition logic ...
        ...         return circ
    """

    @abstractmethod
    def matches(self, op: Operation) -> bool:
        """Check if this rule can transform the given operation.

        Args:
            op: The operation to check.

        Returns:
            True if this rule can decompose the operation.
        """
        ...

    @abstractmethod
    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Apply this rule to transform an operation.

        This method should only be called if :meth:`matches` returns True.

        Args:
            op: The operation to transform.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices for the operation.
            zvars: Z-variable indices for the operation.

        Returns:
            A Circuit containing the decomposed instructions.

        Raises:
            DecompositionError: If the operation cannot be decomposed.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DecompositionBasis(ABC):
    """Abstract base class for decomposition targets.

    A DecompositionBasis combines two concerns:
        1. **What is terminal**: Operations that should not be decomposed further.
        2. **How to decompose**: The transformation logic for non-terminal operations.

    Subclasses must implement:
        - :meth:`isterminal`: Return True if the operation is in the target set
        - :meth:`decompose`: Decompose non-terminal operations

    Example:
        >>> class CliffordT(DecompositionBasis):
        ...     def isterminal(self, op: Operation) -> bool:
        ...         return isinstance(op, (GateH, GateS, GateT, GateCX))
        ...
        ...     def decompose(self, op, qubits, bits, zvars):
        ...         # Use a rewrite rule to decompose
        ...         return CanonicalRewrite().decompose_step(op, qubits, bits, zvars)
    """

    @abstractmethod
    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in this basis.

        Terminal operations are not decomposed further - they represent the
        target instruction set that decomposition reduces circuits to.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is terminal.
        """
        ...

    @abstractmethod
    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose a non-terminal operation.

        This method should only be called for non-terminal operations
        (i.e., when :meth:`isterminal` returns False).

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
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


__all__ = ["RewriteRule", "DecompositionBasis", "DecompositionError"]
