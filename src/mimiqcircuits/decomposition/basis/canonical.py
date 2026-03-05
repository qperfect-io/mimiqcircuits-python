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
"""Canonical decomposition basis targeting GateU and GateCX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


class CanonicalBasis(DecompositionBasis):
    """Decomposition basis targeting the canonical primitive gate set.

    The canonical basis decomposes all operations to a minimal set of primitives:
        - ``GateU``: Arbitrary single-qubit unitary
        - ``GateCX``: Two-qubit CNOT gate
        - ``Measure``, ``Reset``: Measurement and reset operations
        - ``Barrier``: Scheduling barriers
        - Classical operations, annotations, noise channels

    This is the default basis used by :func:`decompose`.

    Example:
        >>> from mimiqcircuits import Circuit, GateH, GateCCX
        >>> from mimiqcircuits.decomposition import decompose, CanonicalBasis
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateCCX(), 0, 1, 2)
        3-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── C₂X @ q[0:1], q[2]
        <BLANKLINE>
        >>> decomposed = decompose(c, CanonicalBasis())
        >>> # All gates are now GateU or GateCX
    """

    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in the canonical basis.

        Terminal operations include:
            - Core gates: GateU, GateCX
            - Measurement/reset: Measure, Reset and variants
            - Control flow: Barrier, IfStatement (if inner op is terminal)
            - Classical operations, annotations
            - Noise channels, expectation values
            - Z-register operations: Add, Multiply, Pow
            - Delay (timing information)

        Args:
            op: The operation to check.

        Returns:
            True if the operation is terminal.
        """
        import mimiqcircuits as mc

        # Core gates
        if isinstance(op, mc.GateU):
            return True
        if isinstance(op, mc.GateCX):
            return True

        # Measurement and reset operations
        if isinstance(op, (mc.AbstractMeasurement, mc.Reset)):
            return True
        if isinstance(op, (mc.MeasureReset,)):
            return True

        # Control flow
        if isinstance(op, mc.Barrier):
            return True

        # IfStatement: terminal if inner operation is terminal
        if isinstance(op, mc.IfStatement):
            return self.isterminal(op.get_operation())

        # Classical operations
        if isinstance(op, mc.AbstractClassical):
            return True

        # Annotations
        if isinstance(op, mc.AbstractAnnotation):
            return True

        # Noise channels
        if isinstance(op, mc.krauschannel):
            return True

        # Expectation values and entanglement measures
        if isinstance(op, (mc.ExpectationValue, mc.Amplitude)):
            return True
        if isinstance(op, (mc.SchmidtRank, mc.VonNeumannEntropy, mc.BondDim)):
            return True

        # Z-register operations
        if isinstance(op, (mc.Add, mc.Multiply, mc.Pow)):
            return True

        # Delay (timing information)
        if isinstance(op, mc.Delay):
            return True

        return False

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose a non-terminal operation using canonical rewrite.

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
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        rule = CanonicalRewrite()
        if rule.matches(op):
            return rule.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} is not supported by CanonicalBasis "
            "and cannot be decomposed."
        )


__all__ = ["CanonicalBasis"]
