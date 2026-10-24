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

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import DecompositionBasis, DecompositionError

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation


# Terminality per operation type, for the types where the type alone decides
# it. Filled on first use and shared by all instances.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


@lru_cache(maxsize=None)
def _qasm_terminal_types():
    import mimiqcircuits as mc

    return (
        # OpenQASM 2.0 built-ins
        mc.GateU,
        mc.GateCX,
        # qelib1.inc, single qubit
        mc.GateU3,
        mc.GateU2,
        mc.GateU1,
        mc.GateID,
        mc.GateX,
        mc.GateY,
        mc.GateZ,
        mc.GateH,
        mc.GateS,
        mc.GateSDG,
        mc.GateT,
        mc.GateTDG,
        mc.GateRX,
        mc.GateRY,
        mc.GateRZ,
        # qelib1.inc, two qubits
        mc.GateCY,
        mc.GateCZ,
        mc.GateCH,
        mc.GateCRX,
        mc.GateCRY,
        mc.GateCRZ,
        mc.GateSWAP,
        mc.GateRXX,
        mc.GateRZZ,
        # qelib1.inc, three and four qubits
        mc.GateCCX,
        mc.GateCSWAP,
        mc.GateC3X,
        mc.Barrier,
    )


class QASMBasis(DecompositionBasis):
    """Decomposition basis targeting the OpenQASM 2.0 gate library.

    The terminal gates are the OpenQASM 2.0 built-ins and the gates of the
    ``qelib1.inc`` include file that have a direct MIMIQ counterpart, so a
    decomposed circuit can be written as OpenQASM 2.0 and read back by any
    parser that ships the standard ``qelib1.inc``.

    Terminal Operations
    -------------------

    **Fundamental gates (OpenQASM 2.0 built-in):**
        - ``GateU``, ``GateCX``

    **Single-qubit gates from qelib1.inc:**
        - Legacy U gates: ``GateU3``, ``GateU2``, ``GateU1``
        - Pauli: ``GateID``, ``GateX``, ``GateY``, ``GateZ``
        - Hadamard: ``GateH``
        - Phase: ``GateS``, ``GateSDG``, ``GateT``, ``GateTDG``
        - Rotations: ``GateRX``, ``GateRY``, ``GateRZ``

    **Two-qubit gates from qelib1.inc:**
        - Controlled Paulis and Hadamard: ``GateCY``, ``GateCZ``, ``GateCH``
        - Controlled rotations: ``GateCRX``, ``GateCRY``, ``GateCRZ``
        - ``GateSWAP``
        - Ising couplings: ``GateRXX``, ``GateRZZ``

    **Three- and four-qubit gates from qelib1.inc:**
        - ``GateCCX``, ``GateCSWAP``, ``GateC3X``

    **Non-unitary operations:**
        - ``Measure``, ``Reset``, ``Barrier``
        - ``IfStatement`` whose inner operation is terminal

    **Custom gates:**
        - ``GateCall`` whose declaration contains only terminal operations

    Every other operation is decomposed through :class:`CanonicalRewrite`.
    Operations that OpenQASM 2.0 cannot express and that have no decomposition
    into the gates above (noise channels, classical operations, annotations,
    Z-register operations, ...) raise :class:`DecompositionError`.

    ``MeasureReset`` decomposes into ``Measure`` followed by ``Reset``. This
    leaves the same state and bit as its canonical decomposition, a
    measurement and an ``X`` conditioned on its outcome, and unlike that
    conditional it can be written for a circuit of any number of bits.
    OpenQASM 2.0 has no loops, so ``WhileStatement`` raises
    :class:`DecompositionError`.

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
        # Terminality depends only on the operation's type, except for gate
        # calls and conditionals, which are decided by what they wrap. Caching the type-only answers skips the chain of
        # isinstance checks, each of which goes through ABCMeta.
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        import mimiqcircuits as mc

        # Measure and Reset are matched exactly: their X/Y-basis variants are
        # not part of OpenQASM 2.0 and must be decomposed.
        if type(op) in (mc.Measure, mc.Reset):
            return True

        if isinstance(op, mc.GateCall):
            return all(
                self.isterminal(inst.get_operation()) for inst in op.decl.circuit
            )

        if isinstance(op, mc.IfStatement):
            inner_op = op.get_operation()
            return inner_op is not None and self.isterminal(inner_op)

        known = isinstance(op, _qasm_terminal_types())
        _TERMINAL_BY_TYPE[t] = known
        return known

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
        import mimiqcircuits as mc
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        if type(op) is mc.MeasureReset:
            c = mc.Circuit()
            c.push(mc.Measure(), qubits[0], bits[0])
            c.push(mc.Reset(), qubits[0])
            return c

        if isinstance(op, mc.WhileStatement):
            raise DecompositionError(
                "WhileStatement cannot be decomposed into the OpenQASM 2.0 basis, "
                "which has no loops."
            )

        rule = CanonicalRewrite()
        if rule.matches(op):
            return rule.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} cannot be decomposed into the OpenQASM 2.0 basis."
        )


__all__ = ["QASMBasis"]
