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
"""Conditional execution (IfStatement)."""

from mimiqcircuits.operations.operation import Operation
import mimiqcircuits as mc
from mimiqcircuits.instruction import _find_unit_range, _string_with_square


class IfStatement(Operation):
    """Conditional operation

    This operation applies a specific operation if a given classical bit condition is met.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(IfStatement(GateX(), BitString('1')), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── IF(c==1) X @ q[0], condition[0]
        <BLANKLINE>
        >>> IfStatement(Parallel(4,GateH()), BitString("01"))
        IF (c==01) ⨷ ⁴ H
    """

    _name = "If"

    _num_qubits = None

    _num_bits = None
    _num_cregs = 1

    _num_zvars = None

    _op = None
    _bitstring = None

    def __init__(self, operation, bitstring: mc.BitString):
        if isinstance(operation, type) and issubclass(operation, Operation):
            op = operation()
        elif isinstance(operation, Operation):
            op = operation
        else:
            raise ValueError("Operation must be an Operation object or type.")

        if not isinstance(bitstring, mc.BitString):
            raise ValueError("bitstring must be a BitString object.")

        super().__init__()

        self._num_qubits = op.num_qubits
        self._num_qregs = op._num_qregs
        self._num_zvars = op.num_zvars
        self._num_zregs = op._num_zregs

        self._num_bits = op._num_bits + len(bitstring)
        self._cregsizes = [self._num_bits]
        self._qregsizes = [self._num_qubits]
        self._zregsizes = [self._num_zvars]

        self._op = op
        self._bitstring = bitstring

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    @property
    def bitstring(self):
        return self._bitstring

    def iswrapper(self):
        return True

    def getparams(self):
        return self.op.getparams()

    def inverse(self):
        raise NotImplementedError("Inverse not implemented for IfStatement")

    def power(self, power):
        raise NotImplementedError("Power not implemented for IfStatement")

    def control(self, num_controls):
        raise NotImplementedError("Control not implemented for IfStatement")

    def __str__(self):
        return f"IF (c=={self.bitstring.to01()}) {self.op}"

    def format_with_targets(self, qubits, bits, zvars):
        """Nested conditional formatting (explicit IF(c==..) chain)."""
        op = self.op
        chain = [self.bitstring.to01()]

        while isinstance(op, mc.IfStatement):
            chain.append(op.get_bitstring().to01())
            op = op.op
        if_chain = " ".join(f"IF(c=={bs})" for bs in chain)

        nb_op = getattr(op, "num_bits", 0)
        parts = []
        if qubits:
            parts.append("q" + _string_with_square(qubits, ","))

        if bits:
            if nb_op > 0:
                op_bits = bits[:nb_op]
                parts.append("c" + _string_with_square(op_bits, ","))
            cond_bits = bits[nb_op:]
            if cond_bits:
                cond_bits = _find_unit_range(cond_bits)
                parts.append("condition" + _string_with_square(cond_bits, ","))

        if zvars:
            parts.append("z" + _string_with_square(zvars, ","))

        return f"{if_chain} {op} @ {', '.join(parts)}"

    def evaluate(self, d):
        return IfStatement(self.op.evaluate(d), self.bitstring)

    def get_operation(self):
        return self.op

    def get_bitstring(self):
        return self.bitstring

    def asciiwidth(self, qubits, bits, zvars):
        val = self.get_bitstring()
        gw = self.op.asciiwidth(qubits, [], [])
        bstr = _string_with_square(_find_unit_range(bits), ",")
        iw = len(f"c{bstr}==" + val.to01()) + 2

        return max(gw, iw)

    def _decompose(self, circuit, qtargets, ctargets, ztargets):
        op = self.op
        bs_outer = self.get_bitstring()
        nb_op = op.num_bits

        # Extract condition bits for the outer IfStatement
        cond_bits = list(ctargets[nb_op : nb_op + len(bs_outer)])

        # Decompose the inner operation (nested IfStatement will be handled recursively)
        decomposed = op.decompose()
        instructions = (
            decomposed.instructions
            if hasattr(decomposed, "instructions")
            else decomposed
        )

        for inst in instructions:
            inner_op = inst.operation
            # nested IfStatement
            if isinstance(inner_op, mc.IfStatement):
                bs_inner = inner_op.get_bitstring()
                # inner condition first, then outer condition
                merged_bits = bs_inner.bits + bs_outer.bits
                merged_bs = mc.BitString(merged_bits)
                nested = mc.IfStatement(inner_op.op, merged_bs)
                nested._decompose(circuit, qtargets, ctargets, ztargets)
                continue

            q = [qtargets[i] for i in inst.get_qubits()]
            b = [ctargets[i] for i in inst.get_bits()]
            z = [ztargets[i] for i in inst.get_zvars()]
            # inner op bits first, then condition bits
            targeted_bits = b + cond_bits

            conditional = mc.IfStatement(inner_op, bs_outer)
            circuit.push(conditional, *q, *targeted_bits, *z)

        return circuit


__all__ = ["IfStatement"]
