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
"""Conditional loop (WhileStatement)."""

from mimiqcircuits.operations.operation import Operation
import mimiqcircuits as mc
from mimiqcircuits.instruction import _find_unit_range, _string_with_square


class WhileStatement(Operation):
    """Conditional loop operation.

    Repeatedly applies the provided operation while the classical register
    matches the specified ``BitString``.

    ``WhileStatement`` is the loop counterpart of :class:`IfStatement`. The body
    operation ``op`` is executed once for every iteration in which the
    classical register's state equals the ``BitString``. For the loop to
    terminate, ``op`` must mutate at least one of the bits used in the
    condition; ``WhileStatement`` allows body bits to alias condition bits,
    which is the only way to make progress.

    There is no built-in iteration cap — a non-terminating circuit is the
    user's responsibility, mirroring how every classical language treats
    ``while``.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(WhileStatement(Not(), BitString('1')), 0, 0)
        1-bit circuit with 1 instruction:
        └── WHILE(c==1) ! @ c[0], condition[0]
        <BLANKLINE>

    See Also:
        - :class:`IfStatement`
    """

    _name = "While"

    _num_qubits = None

    _num_bits = None
    _num_cregs = 1

    _num_zvars = None

    _op = None
    _bitstring = None

    # WhileStatement's classical-target layout is [op_bits..., condition_bits...].
    # Termination requires the body to mutate a condition bit, so aliasing
    # between body and condition is essential, not optional.
    _allow_bit_aliasing = True
    _allow_zvar_aliasing = True

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
        raise NotImplementedError("Inverse not implemented for WhileStatement")

    def power(self, power):
        raise NotImplementedError("Power not implemented for WhileStatement")

    def control(self, num_controls):
        raise NotImplementedError("Control not implemented for WhileStatement")

    def __str__(self):
        return f"WHILE(c=={self.bitstring.to01()}) {self.op}"

    def format_with_targets(self, qubits, bits, zvars):
        op = self.op
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

        return f"WHILE(c=={self.bitstring.to01()}) {op} @ {', '.join(parts)}"

    def evaluate(self, d):
        return WhileStatement(self.op.evaluate(d), self.bitstring)

    def get_operation(self):
        return self.op

    def get_bitstring(self):
        return self.bitstring

    def asciiwidth(self, qubits, bits, zvars):
        val = self.get_bitstring()
        gw = self.op.asciiwidth(qubits, [], [])
        bstr = _string_with_square(_find_unit_range(bits), ",")
        iw = len(f"while c{bstr}==" + val.to01()) + 2

        return max(gw, iw)

    def __eq__(self, other):
        if not isinstance(other, WhileStatement):
            return False
        return self.op == other.op and self.bitstring == other.bitstring

    def __hash__(self):
        return hash((type(self), self.op, self.bitstring))

    def _decompose(self, circuit, qtargets, ctargets, ztargets):
        op = self.op
        bs = self.get_bitstring()
        nb_op = op.num_bits

        target_bits = list(ctargets[:nb_op])
        condition_bits = list(ctargets[nb_op : nb_op + len(bs)])

        # Decompose the inner body. The loop boundary itself is opaque — we
        # never unroll, but we may rewrite the body.
        decomposed = op.decompose()
        instructions = (
            decomposed.instructions
            if hasattr(decomposed, "instructions")
            else decomposed
        )

        if len(instructions) == 1:
            only_inst = instructions[0]
            inner_op = only_inst.operation
            q = [qtargets[i] for i in only_inst.get_qubits()]
            b = [target_bits[i] for i in only_inst.get_bits()]
            z = [ztargets[i] for i in only_inst.get_zvars()]
            new_bits = b + condition_bits
            circuit.push(WhileStatement(inner_op, bs), *q, *new_bits, *z)
        else:
            block = mc.Block(instructions)
            new_bits = target_bits + condition_bits
            circuit.push(
                WhileStatement(block, bs),
                *qtargets,
                *new_bits,
                *ztargets,
            )

        return circuit


__all__ = ["WhileStatement"]
