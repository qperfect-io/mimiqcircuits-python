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
from mimiqcircuits.bitstrings import BitString
from mimiqcircuits.operations.block import Block
from mimiqcircuits.instruction import _find_unit_range, _string_with_square


class WhileStatement(Operation):
    """Operation applied repeatedly while classical bits match a bitstring.

    ``WhileStatement(op, bs)`` is the loop counterpart of
    :class:`IfStatement`. Before each iteration the condition bits are
    compared with ``bs``; while they match, ``op`` is applied again.

    The instruction takes the same targets as an :class:`IfStatement`:
    qubits of ``op``, classical bits of ``op``, ``len(bs)`` condition bits,
    then z-variables of ``op``. For the loop to end, ``op`` must change at
    least one condition bit, so the bits of ``op`` may overlap with the
    condition bits. There is no limit on the number of iterations: a loop
    whose condition never changes does not terminate.

    Args:
        operation: The loop body, or an operation class built with no
            arguments.
        bitstring: The values the condition bits must hold for the loop to
            continue.

    Raises:
        ValueError: If ``operation`` is not an operation or ``bitstring`` is
            not a :class:`BitString`.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> # measure and reset qubit 0 as long as bit 0 holds 1: the
        >>> # measurement writes bit 0, which is also the condition bit, and
        >>> # after the reset it reads 0, which ends the loop
        >>> c.push(WhileStatement(MeasureReset(), BitString('1')), 0, 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── WHILE(c==1) MR @ q[0], c[0], condition[0]
        <BLANKLINE>
        >>> # a classical body: flipping bit 1 while it is 1 runs at most once
        >>> c.push(WhileStatement(Not(), BitString('1')), 1, 1)
        1-qubit, 2-bit circuit with 2 instructions:
        ├── WHILE(c==1) MR @ q[0], c[0], condition[0]
        └── WHILE(c==1) ! @ c[1], condition[1]
        <BLANKLINE>

    See Also:
        :class:`IfStatement`
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

    def __init__(self, operation, bitstring: BitString):
        if isinstance(operation, type) and issubclass(operation, Operation):
            op = operation()
        elif isinstance(operation, Operation):
            op = operation
        else:
            raise ValueError("Operation must be an Operation object or type.")

        if not isinstance(bitstring, BitString):
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
        """The loop body. Read-only."""
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    @property
    def bitstring(self):
        """The values the condition bits must hold, as a :class:`BitString`."""
        return self._bitstring

    def iswrapper(self):
        return True

    def getparams(self):
        """Return the parameters of the loop body."""
        return self.op.getparams()

    def inverse(self):
        """Not supported for loops.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Inverse not implemented for WhileStatement")

    def power(self, power):
        """Not supported for loops.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Power not implemented for WhileStatement")

    def control(self, num_controls):
        """Not supported for loops.

        Raises:
            NotImplementedError: Always.
        """
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
        if not self.is_symbolic():
            return self
        return WhileStatement(self.op.evaluate(d), self.bitstring)

    def get_operation(self):
        """Return the loop body, :attr:`op`."""
        return self.op

    def get_bitstring(self):
        """Return the condition, :attr:`bitstring`."""
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
            block = Block(op.num_qubits, op.num_bits, op.num_zvars, list(instructions))
            new_bits = target_bits + condition_bits
            circuit.push(
                WhileStatement(block, bs),
                *qtargets,
                *new_bits,
                *ztargets,
            )

        return circuit


__all__ = ["WhileStatement"]
