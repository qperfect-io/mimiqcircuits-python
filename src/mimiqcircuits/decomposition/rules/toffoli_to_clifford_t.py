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
"""Rewrite rule for decomposing Toffoli into Clifford+T."""

from __future__ import annotations


from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


class ToffoliToCliffordTRewrite(RewriteRule):
    """Decompose ``GateCCX`` into an explicit Clifford+T sequence."""

    def matches(self, op: Operation) -> bool:
        """Return True for Toffoli (CCX) gates."""
        import mimiqcircuits as mc

        return isinstance(op, mc.GateCCX)

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Apply the standard 7-T Toffoli decomposition."""
        import mimiqcircuits as mc

        if not self.matches(op):
            raise DecompositionError(
                f"ToffoliToCliffordTRewrite cannot decompose {type(op).__name__}"
            )

        if len(qubits) != 3:
            raise DecompositionError(
                "ToffoliToCliffordTRewrite expects exactly 3 target qubits."
            )

        c1, c2, t = qubits
        circ = mc.Circuit()

        circ.push(mc.GateH(), t)
        circ.push(mc.GateCX(), c2, t)
        circ.push(mc.GateTDG(), t)
        circ.push(mc.GateCX(), c1, t)
        circ.push(mc.GateT(), t)
        circ.push(mc.GateCX(), c2, t)
        circ.push(mc.GateTDG(), t)
        circ.push(mc.GateCX(), c1, t)
        circ.push(mc.GateT(), c2)
        circ.push(mc.GateT(), t)
        circ.push(mc.GateH(), t)
        circ.push(mc.GateCX(), c1, c2)
        circ.push(mc.GateT(), c1)
        circ.push(mc.GateTDG(), c2)
        circ.push(mc.GateCX(), c1, c2)

        return circ


__all__ = ["ToffoliToCliffordTRewrite"]
