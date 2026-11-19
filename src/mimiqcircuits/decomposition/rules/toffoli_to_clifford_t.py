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
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


class ToffoliToCliffordTRewrite(RewriteRule):
    """Rewrite rule that turns a Toffoli gate into Clifford+T gates.

    Replaces ``GateCCX`` by the standard exact circuit of 15 gates: 7 ``T``
    or ``T†``, 2 ``H`` and 6 ``CX``. No ancilla is used.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateCCX(), 0, 1, 2)
        3-qubit circuit with 1 instruction:
        └── C₂X @ q[0:1], q[2]
        <BLANKLINE>
        >>> # count the T and T† gates of the rewrite
        >>> ct = decompose_step(c, ToffoliToCliffordTRewrite())
        >>> sum(isinstance(inst.operation, (GateT, GateTDG)) for inst in ct)
        7
    """

    def matches(self, op: Operation) -> bool:
        """Check whether an operation is a Toffoli gate.

        Args:
            op: The operation to check.

        Returns:
            ``True`` for ``GateCCX``.
        """
        from mimiqcircuits.operations.gates.standard.cnx import GateCCX

        return isinstance(op, GateCCX)

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite a Toffoli gate as 15 Clifford+T gates.

        Args:
            op: The ``GateCCX`` to rewrite.
            qubits: The two control qubits, then the target.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If ``op`` is not a ``GateCCX`` or
                ``qubits`` does not hold exactly 3 indices.
        """
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.gates.standard.cpauli import GateCX
        from mimiqcircuits.operations.gates.standard.hadamard import GateH
        from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG

        if not self.matches(op):
            raise DecompositionError(
                f"ToffoliToCliffordTRewrite cannot decompose {type(op).__name__}"
            )

        if len(qubits) != 3:
            raise DecompositionError(
                "ToffoliToCliffordTRewrite expects exactly 3 target qubits."
            )

        c1, c2, t = qubits
        circ = Circuit()

        circ.push(GateH(), t)
        circ.push(GateCX(), c2, t)
        circ.push(GateTDG(), t)
        circ.push(GateCX(), c1, t)
        circ.push(GateT(), t)
        circ.push(GateCX(), c2, t)
        circ.push(GateTDG(), t)
        circ.push(GateCX(), c1, t)
        circ.push(GateT(), c2)
        circ.push(GateT(), t)
        circ.push(GateH(), t)
        circ.push(GateCX(), c1, c2)
        circ.push(GateT(), c1)
        circ.push(GateTDG(), c2)
        circ.push(GateCX(), c1, c2)

        return circ


__all__ = ["ToffoliToCliffordTRewrite"]
