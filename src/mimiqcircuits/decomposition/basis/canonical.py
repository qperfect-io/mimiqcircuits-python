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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits.decomposition.abstract import (
    DecompositionBasis,
    DecompositionError,
    _AncillaOptions,
)

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.operation import Operation


# Terminality per operation type, filled on first use. Shared by all
# instances: the canonical gate set does not depend on the basis's state.
_TERMINAL_BY_TYPE: dict[type, bool] = {}


@dataclass(frozen=True, init=False)
class CanonicalBasis(DecompositionBasis):
    """Decomposition basis targeting ``GateU`` and ``GateCX``.

    Every gate is rewritten into ``GateU`` (an arbitrary single-qubit gate)
    and ``GateCX``, with the built-in decomposition of each operation
    (:class:`CanonicalRewrite`). The non-unitary primitives are kept as they
    are; :meth:`isterminal` lists them. This is the default basis of
    :func:`decompose`.

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
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateSWAP(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── SWAP @ q[0:1]
        <BLANKLINE>
        >>> c.push(Measure(), 0, 0)
        2-qubit, 1-bit circuit with 3 instructions:
        ├── H @ q[0]
        ├── SWAP @ q[0:1]
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> # gates become GateU and GateCX; the measurement is kept
        >>> decompose(c, CanonicalBasis())
        2-qubit, 1-bit circuit with 5 instructions:
        ├── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        ├── CX @ q[0], q[1]
        ├── CX @ q[1], q[0]
        ├── CX @ q[0], q[1]
        └── M @ q[0], c[0]
        <BLANKLINE>
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
        """Check whether an operation is terminal in the canonical basis.

        The terminal operations are ``GateU`` and ``GateCX``; ``Measure``;
        ``Reset``, ``ResetX`` and ``ResetY``; ``Loss``, ``Reload``, ``Check``
        and ``MeasureCheck``; ``Barrier`` and ``Delay``; classical operations
        and annotations; noise channels and ``ReadoutErr``;
        ``ExpectationValue``, ``Amplitude``, ``SchmidtRank``,
        ``VonNeumannEntropy`` and ``BondDim``; and the z-register operations
        ``Add``, ``Multiply`` and ``Pow``. An ``IfStatement`` or
        ``WhileStatement`` is terminal when its body is.

        Args:
            op: The operation to check.

        Returns:
            ``True`` if ``op`` is terminal.
        """
        # Terminality depends only on the operation's type, except for the
        # conditionals, which defer to their body. Caching by type skips the
        # chain of isinstance checks, each of which goes through ABCMeta.
        t = type(op)
        known = _TERMINAL_BY_TYPE.get(t)
        if known is not None:
            return known

        from mimiqcircuits.classical.abstract_classical import AbstractClassical
        from mimiqcircuits.complex import Add, Multiply, Pow
        from mimiqcircuits.operations.amplitude import Amplitude
        from mimiqcircuits.operations.annotations import AbstractAnnotation
        from mimiqcircuits.operations.barrier import Barrier
        from mimiqcircuits.operations.entanglement import (
            BondDim,
            SchmidtRank,
            VonNeumannEntropy,
        )
        from mimiqcircuits.operations.expectationvalue import ExpectationValue
        from mimiqcircuits.operations.gates.delay import Delay
        from mimiqcircuits.operations.gates.standard.cpauli import GateCX
        from mimiqcircuits.operations.gates.standard.u import GateU
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.krauschannel import krauschannel
        from mimiqcircuits.operations.losschannel import (
            Check,
            Loss,
            MeasureCheck,
            Reload,
        )
        from mimiqcircuits.operations.measure import Measure
        from mimiqcircuits.operations.readouterr import ReadoutErr
        from mimiqcircuits.operations.reset import Reset
        from mimiqcircuits.operations.whilestatement import WhileStatement

        # IfStatement, WhileStatement: terminal if the inner operation is
        if isinstance(op, (IfStatement, WhileStatement)):
            return self.isterminal(op.get_operation())

        known = isinstance(
            op,
            (
                # Core gates
                GateU,
                GateCX,
                # Measurement, reset and Loss operations
                Measure,
                Reset,
                Loss,
                Reload,
                Check,
                MeasureCheck,
                # Control flow
                Barrier,
                # Classical operations
                AbstractClassical,
                # Annotations
                AbstractAnnotation,
                # Noise channels
                krauschannel,
                ReadoutErr,
                # Expectation values and entanglement measures
                ExpectationValue,
                Amplitude,
                SchmidtRank,
                VonNeumannEntropy,
                BondDim,
                # Z-register operations
                Add,
                Multiply,
                Pow,
                # Delay (timing information)
                Delay,
            ),
        )
        _TERMINAL_BY_TYPE[t] = known
        return known

    def decompose(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Rewrite a non-terminal operation with its built-in decomposition.

        Args:
            op: The operation to rewrite.
            qubits: Qubit indices the operation acts on.
            bits: Classical bit indices the operation acts on.
            zvars: Z-variable indices the operation acts on.

        Returns:
            A new :class:`~mimiqcircuits.Circuit` with one decomposition step.

        Raises:
            DecompositionError: If ``op`` has no built-in decomposition.
        """
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        rule = CanonicalRewrite()
        if rule.matches(op):
            return rule.decompose_step(op, qubits, bits, zvars)

        raise DecompositionError(
            f"Operation {op.name} is not supported by CanonicalBasis "
            "and cannot be decomposed."
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

__all__ = ["CanonicalBasis"]
