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
"""OpenQASM 2.0 decomposition basis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


class QASMBasis(DecompositionBasis):
    """Decomposition basis targeting the OpenQASM 2.0 gate library.

    This basis includes all gates defined in ``qelib1.inc`` (the standard QASM 2.0
    library) plus common extensions, making it suitable for exporting circuits to
    QASM format or running on QASM-compatible backends.

    Terminal Operations
    -------------------

    The following gate categories are terminal (not decomposed further):

    **Fundamental gates (OpenQASM 2.0 built-in):**
        - ``GateU``, ``GateCX`` — the universal basis for QASM 2.0

    **Single-qubit gates from qelib1.inc:**
        - Legacy U gates: ``GateU3``, ``GateU2``, ``GateU1``
        - Pauli: ``GateID``, ``GateX``, ``GateY``, ``GateZ``
        - Rotations: ``GateRX``, ``GateRY``, ``GateRZ``
        - Phase gates: ``GateS``, ``GateSDG``, ``GateT``, ``GateTDG``
        - Hadamard: ``GateH``
        - Square roots: ``GateSX``, ``GateSXDG``
        - Phase: ``GateP``

    **Two-qubit gates from qelib1.inc:**
        - ``GateCY``, ``GateCZ``
        - ``GateSWAP``
        - ``GateCH``
        - ``GateCRX``, ``GateCRY``, ``GateCRZ``
        - ``GateCU``, ``GateCP``

    **Non-unitary operations:**
        - ``Measure``, ``Reset``
        - ``Barrier``
        - Classical operations, annotations

    Example:
        >>> from mimiqcircuits import Circuit, GateH, GateCCX
        >>> from mimiqcircuits.decomposition import decompose, QASMBasis
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
        >>> # Decompose to QASM-compatible gates
        >>> qasm_circuit = decompose(c, QASMBasis())

    See Also:
        :class:`CanonicalBasis`, :class:`CliffordTBasis`, :class:`StimBasis`
    """

    def isterminal(self, op: Operation) -> bool:
        """Check if an operation is terminal in the OpenQASM 2.0 basis.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is in the QASM gate set.
        """
        import mimiqcircuits as mc

        # === Fundamental gates (QASM 2.0 built-in) ===
        if isinstance(op, mc.GateU):
            return True
        if isinstance(op, mc.GateCX):
            return True

        # === Single-qubit gates from qelib1.inc ===

        # Legacy U gates
        if isinstance(op, (mc.GateU1, mc.GateU2, mc.GateU3)):
            return True

        # Pauli gates
        if isinstance(op, mc.GateID):
            return True
        if isinstance(op, (mc.GateX, mc.GateY, mc.GateZ)):
            return True

        # Rotation gates
        if isinstance(op, (mc.GateRX, mc.GateRY, mc.GateRZ)):
            return True

        # Phase gates: S, Sdg, T, Tdg
        if isinstance(op, (mc.GateS, mc.GateSDG, mc.GateT, mc.GateTDG)):
            return True

        # Hadamard
        if isinstance(op, mc.GateH):
            return True

        # Square root gates: SX, SXdg
        if isinstance(op, (mc.GateSX, mc.GateSXDG)):
            return True

        # Phase gate
        if isinstance(op, mc.GateP):
            return True

        # === Two-qubit gates from qelib1.inc ===

        # Controlled Pauli: CY, CZ
        if isinstance(op, (mc.GateCY, mc.GateCZ)):
            return True

        # SWAP
        if isinstance(op, mc.GateSWAP):
            return True

        # Controlled Hadamard
        if isinstance(op, mc.GateCH):
            return True

        # Ising coupling gates
        if isinstance(op, (mc.GateRXX, mc.GateRZZ)):
            return True

        # Multi-controlled qelib1 gates
        if isinstance(op, (mc.GateCCX, mc.GateCSWAP, mc.GateC3X)):
            return True

        # Controlled rotations (CRX, CRY, CRZ are parametric Control gates)
        # Check if it's a controlled rotation by checking inner gate type
        if isinstance(op, mc.Control) and op.num_controls == 1:
            inner = op.op
            if isinstance(inner, (mc.GateRX, mc.GateRY, mc.GateRZ)):
                return True
            # Controlled U and phase
            if isinstance(inner, (mc.GateU, mc.GateP)):
                return True

        # GateCall (custom gate declarations)
        if isinstance(op, mc.GateCall):
            return all(
                self.isterminal(inst.get_operation()) for inst in op.decl.circuit
            )

        # === Non-unitary operations ===

        # Measurement and reset
        if isinstance(op, (mc.AbstractMeasurement, mc.Reset)):
            return True
        if isinstance(op, mc.MeasureReset):
            return True

        # Barrier
        if isinstance(op, mc.Barrier):
            return True

        # IfStatement: terminal if inner operation is terminal
        if isinstance(op, mc.IfStatement):
            inner_op = op.get_operation()
            return inner_op is not None and self.isterminal(inner_op)

        # Classical operations
        if isinstance(op, mc.AbstractClassical):
            return True

        # Annotations
        if isinstance(op, mc.AbstractAnnotation):
            return True

        # Noise channels (for completeness)
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

        # Delay
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
            f"Operation {op.name} cannot be decomposed into the OpenQASM 2.0 basis."
        )


__all__ = ["QASMBasis"]
