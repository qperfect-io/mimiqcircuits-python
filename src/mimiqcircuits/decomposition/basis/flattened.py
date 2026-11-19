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
"""Flattened decomposition basis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


# Terminality per operation type, for the types where the type alone decides
# it. Filled on first use and shared by all instances.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


class FlattenedBasis(DecompositionBasis):
    """Decomposition basis that expands containers and keeps every other operation.

    ``GateCall``, ``Block`` and ``BlockCall``, and the ``Inverse`` of a
    ``GateCall``, are expanded with :class:`FlattenContainers`
    until none is left, at any depth. An ``IfStatement`` whose body is a
    container becomes one conditional instruction per instruction of the
    body. Gates are not decomposed, and that includes a ``GateCall`` wrapped
    in ``Control``, ``Power``, ``Parallel`` or ``Repeat``, which is kept.

    Examples:
        >>> from mimiqcircuits import *
        >>> # a declared gate that holds a Bell-pair preparation
        >>> bell = GateDecl("bell", (), Circuit().push(GateH(), 0).push(GateCX(), 0, 1))
        >>> c = Circuit()
        >>> c.push(bell(), 0, 1)
        2-qubit circuit with 1 instruction:
        └── bell() @ q[0:1]
        <BLANKLINE>
        >>> c.push(GateSWAP(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── bell() @ q[0:1]
        └── SWAP @ q[0:1]
        <BLANKLINE>
        >>> # the call is expanded; SWAP is a gate and is kept
        >>> decompose(c, FlattenedBasis())
        2-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        └── SWAP @ q[0:1]
        <BLANKLINE>
    """

    def isterminal(self, op: Operation) -> bool:
        """Check whether an operation is kept, that is not a container.

        Args:
            op: The operation to check.

        Returns:
            ``False`` for the containers listed in the class description and
            for an ``IfStatement`` whose body is one, ``True`` otherwise.
        """
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        from mimiqcircuits.blockdecl import BlockCall
        from mimiqcircuits.gatedecl import GateCall
        from mimiqcircuits.operations.block import Block
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.inverse import Inverse

        # An inverse is decided by what it wraps, so it is never cached.
        if isinstance(op, Inverse):
            return not isinstance(op.op, (GateCall, Block, BlockCall))
        # A conditional container is flattened into conditional instructions;
        # it is decided by its body, so it is never cached either.
        if isinstance(op, IfStatement):
            return self.isterminal(op.get_operation())
        known = not isinstance(op, (GateCall, Block, BlockCall))
        _TERMINAL_BY_TYPE[t] = known
        return known

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Expand one level of a container.

        Args:
            op: The container, or the ``IfStatement`` of one, to expand.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the contents of
            ``op``, each made conditional if ``op`` is an ``IfStatement``.

        Raises:
            DecompositionError: If ``op`` is not a container.
        """
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.decomposition.rules import FlattenContainers
        from mimiqcircuits.operations.ifstatement import _decompose_ifstatement

        if isinstance(op, IfStatement):
            return _decompose_ifstatement(self, op, qubits, bits, zvars)

        rule = FlattenContainers()
        if rule.matches(op):
            return rule.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} matches FlattenedBasis but cannot be decomposed."
        )


__all__ = ["FlattenedBasis"]
