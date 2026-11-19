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

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import (
    DecompositionBasis,
    DecompositionError,
    _AncillaOptions,
)

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


# Terminality per operation type, for the types where the type alone decides
# it. Filled on first use and shared by all instances.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


@lru_cache(maxsize=None)
def _qasm_terminal_types():
    from mimiqcircuits.operations.barrier import Barrier
    from mimiqcircuits.operations.gates.standard.chadamard import GateCH
    from mimiqcircuits.operations.gates.standard.cnx import GateC3X, GateCCX
    from mimiqcircuits.operations.gates.standard.cpauli import GateCX, GateCY, GateCZ
    from mimiqcircuits.operations.gates.standard.crotations import (
        GateCRX,
        GateCRY,
        GateCRZ,
    )
    from mimiqcircuits.operations.gates.standard.cswap import GateCSWAP
    from mimiqcircuits.operations.gates.standard.deprecated import (
        GateU1,
        GateU2,
        GateU3,
    )
    from mimiqcircuits.operations.gates.standard.hadamard import GateH
    from mimiqcircuits.operations.gates.standard.id import GateID
    from mimiqcircuits.operations.gates.standard.interactions import GateRXX, GateRZZ
    from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
    from mimiqcircuits.operations.gates.standard.rotations import GateRX, GateRY, GateRZ
    from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
    from mimiqcircuits.operations.gates.standard.swap import GateSWAP
    from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
    from mimiqcircuits.operations.gates.standard.u import GateU

    return (
        # OpenQASM 2.0 built-ins
        GateU,
        GateCX,
        # qelib1.inc, single qubit
        GateU3,
        GateU2,
        GateU1,
        GateID,
        GateX,
        GateY,
        GateZ,
        GateH,
        GateS,
        GateSDG,
        GateT,
        GateTDG,
        GateRX,
        GateRY,
        GateRZ,
        # qelib1.inc, two qubits
        GateCY,
        GateCZ,
        GateCH,
        GateCRX,
        GateCRY,
        GateCRZ,
        GateSWAP,
        GateRXX,
        GateRZZ,
        # qelib1.inc, three and four qubits
        GateCCX,
        GateCSWAP,
        GateC3X,
        Barrier,
    )


@dataclass(frozen=True, init=False)
class QASMBasis(DecompositionBasis):
    """Decomposition basis targeting the OpenQASM 2.0 gate library.

    The terminal gates are the OpenQASM 2.0 built-ins and the gates of the
    ``qelib1.inc`` include file that have a direct MIMIQ counterpart, so a
    decomposed circuit can be written as OpenQASM 2.0 and read back by any
    parser that ships the standard ``qelib1.inc``.

    The terminal operations are:

    - the OpenQASM 2.0 built-ins ``GateU`` and ``GateCX``;
    - the single-qubit gates of ``qelib1.inc``: ``GateU3``, ``GateU2``,
      ``GateU1``, ``GateID``, ``GateX``, ``GateY``, ``GateZ``, ``GateH``,
      ``GateS``, ``GateSDG``, ``GateT``, ``GateTDG``, ``GateRX``, ``GateRY``
      and ``GateRZ``;
    - the two-qubit gates of ``qelib1.inc``: ``GateCY``, ``GateCZ``,
      ``GateCH``, ``GateCRX``, ``GateCRY``, ``GateCRZ``, ``GateSWAP``,
      ``GateRXX`` and ``GateRZZ``;
    - the three- and four-qubit gates of ``qelib1.inc``: ``GateCCX``,
      ``GateCSWAP`` and ``GateC3X``;
    - ``Measure``, ``Reset`` and ``Barrier``, but not the X and Y variants of
      ``Measure`` and ``Reset``;
    - annotations, such as :class:`AncillaAlloc`, which the OpenQASM writer
      drops;
    - an ``IfStatement`` whose body is terminal;
    - a ``GateCall`` whose declaration holds only terminal operations.

    Every other operation is decomposed through :class:`CanonicalRewrite`.
    Operations that OpenQASM 2.0 cannot express and that have no decomposition
    into the gates above (noise channels, classical operations, Z-register
    operations, ...) raise :class:`DecompositionError`.

    ``MeasureReset`` decomposes into ``Measure`` followed by ``Reset``. This
    leaves the same state and bit as its canonical decomposition, a
    measurement and an ``X`` conditioned on its outcome, and unlike that
    conditional it can be written for a circuit of any number of bits.
    OpenQASM 2.0 has no loops, so ``WhileStatement`` raises
    :class:`DecompositionError`.

    Args:
        borrow: Let decompositions borrow qubits of the circuit that the
            operation does not act on. A borrowed qubit can be in any state;
            the decomposition leaves it as it found it.
        max_ancillas: Let decompositions add up to this many clean ancillas,
            numbered after the qubits of the circuit and marked with
            :class:`AncillaAlloc` and :class:`AncillaFree`. Ancillas freed by
            one operation are reused by the next.
        allow_measurement: With ``max_ancillas``, let decompositions undo
            steps by measuring an ancilla into one extra bit, numbered after
            the bits of the circuit. A circuit without bits is refused with a
            ``ValueError``.

    The three ancilla options are off by default, and only change how
    multi-controlled X gates are decomposed. Circuits whose width is unknown,
    such as a lone operation, get no ancillas.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateCCX(), 0, 1, 2)
        3-qubit circuit with 1 instruction:
        └── C₂X @ q[0:1], q[2]
        <BLANKLINE>
        >>> c.push(GateSX(), 0)
        3-qubit circuit with 2 instructions:
        ├── C₂X @ q[0:1], q[2]
        └── SX @ q[0]
        <BLANKLINE>
        >>> c.push(MeasureReset(), 0, 0)
        3-qubit, 1-bit circuit with 3 instructions:
        ├── C₂X @ q[0:1], q[2]
        ├── SX @ q[0]
        └── MR @ q[0], c[0]
        <BLANKLINE>
        >>> # CCX is in qelib1.inc and is kept; SX is not and is rewritten;
        >>> # MeasureReset becomes a measurement and a reset
        >>> decompose(c, QASMBasis())
        3-qubit, 1-bit circuit with 7 instructions:
        ├── C₂X @ q[0:1], q[2]
        ├── S† @ q[0]
        ├── H @ q[0]
        ├── S† @ q[0]
        ├── U(0, 0, 0, (1/4)*pi) @ q[0]
        ├── M @ q[0], c[0]
        └── Reset @ q[0]
        <BLANKLINE>

    See Also:
        :class:`CanonicalBasis`, :class:`CliffordTBasis`, :class:`StimBasis`
    """

    ancillas: _AncillaOptions

    def __init__(
        self,
        borrow: bool = False,
        max_ancillas: int = 0,
        allow_measurement: bool = False,
    ):
        object.__setattr__(
            self,
            "ancillas",
            _AncillaOptions(
                borrow=borrow,
                max_ancillas=max_ancillas,
                allow_measurement=allow_measurement,
            ),
        )

    def _ancillaoptions(self):
        return self.ancillas


    def isterminal(self, op: Operation) -> bool:
        """Check whether an operation is terminal in the OpenQASM 2.0 basis.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if ``op`` is one of the terminal operations listed in the
            class description.
        """
        # Terminality depends only on the operation's type, except for gate
        # calls and conditionals, which are decided by what they wrap. Caching
        # the type-only answers skips the chain of isinstance checks, each of
        # which goes through ABCMeta.
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        from mimiqcircuits.gatedecl import GateCall
        from mimiqcircuits.operations.annotations import AbstractAnnotation
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.measure import Measure
        from mimiqcircuits.operations.reset import Reset

        # Measure and Reset are matched exactly: their X/Y-basis variants are
        # not part of OpenQASM 2.0 and must be decomposed.
        if type(op) in (Measure, Reset):
            return True
        # Annotations carry no gate. The OpenQASM writer drops them.
        if isinstance(op, AbstractAnnotation):
            return True

        if isinstance(op, GateCall):
            return all(
                self.isterminal(inst.get_operation()) for inst in op.decl.circuit
            )

        if isinstance(op, IfStatement):
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
        """Rewrite a non-terminal operation one step toward OpenQASM 2.0.

        ``MeasureReset`` becomes ``Measure`` and ``Reset``; every other
        operation goes through :class:`CanonicalRewrite`.

        Args:
            op: The operation to rewrite.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with the rewrite.

        Raises:
            DecompositionError: If ``op`` is a ``WhileStatement`` or has no
                built-in decomposition.
        """
        from mimiqcircuits.circuit import Circuit
        from mimiqcircuits.operations.measure import Measure
        from mimiqcircuits.operations.measurereset import MeasureReset
        from mimiqcircuits.operations.reset import Reset
        from mimiqcircuits.operations.whilestatement import WhileStatement
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        if type(op) is MeasureReset:
            c = Circuit()
            c.push(Measure(), qubits[0], bits[0])
            c.push(Reset(), qubits[0])
            return c

        if isinstance(op, WhileStatement):
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


    def decompose_ctx(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
        ctx,
    ) -> Circuit:
        """Decompose with a context, which only some operations use.

        Operations without a rule that uses the context go to
        :meth:`decompose`, so an override of it keeps being called.
        """
        from mimiqcircuits.decomposition.abstract import _uses_context
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        rule = CanonicalRewrite()
        if _uses_context(op) and rule.matches(op):
            return rule.decompose_step_ctx(op, qubits, bits, zvars, ctx)
        return self.decompose(op, qubits, bits, zvars)

__all__ = ["QASMBasis"]
