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
    from mimiqcircuits import Circuit, Operation


# Terminality per operation type, for the types where the type alone decides
# it. Filled on first use and shared by all instances.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


class FlattenedBasis(DecompositionBasis):
    """Flatten container operations while leaving all other operations untouched."""

    def isterminal(self, op: Operation) -> bool:
        """Containers are non-terminal; all other operations are terminal."""
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        import mimiqcircuits as mc

        # An inverse is decided by what it wraps, so it is never cached.
        if isinstance(op, mc.Inverse):
            return not isinstance(op.op, (mc.GateCall, mc.Block))
        # A conditional container is flattened into conditional instructions;
        # it is decided by its body, so it is never cached either.
        if isinstance(op, mc.IfStatement):
            return self.isterminal(op.get_operation())
        known = not isinstance(op, (mc.GateCall, mc.Block))
        _TERMINAL_BY_TYPE[t] = known
        return known

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose one container expansion step using FlattenContainers."""
        import mimiqcircuits as mc
        from mimiqcircuits.decomposition.rules import FlattenContainers
        from mimiqcircuits.operations.ifstatement import _decompose_ifstatement

        if isinstance(op, mc.IfStatement):
            return _decompose_ifstatement(self, op, qubits, bits, zvars)

        rule = FlattenContainers()
        if rule.matches(op):
            return rule.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} matches FlattenedBasis but cannot be decomposed."
        )


__all__ = ["FlattenedBasis"]
