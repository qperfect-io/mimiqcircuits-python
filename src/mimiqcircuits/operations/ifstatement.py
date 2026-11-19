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
from mimiqcircuits.bitstrings import BitString
from mimiqcircuits.instruction import _find_unit_range, _string_with_square


class IfStatement(Operation):
    """Operation applied only when classical bits match a bitstring.

    ``IfStatement(op, bs)`` applies ``op`` when the condition bits hold the
    values in ``bs``, the first condition bit compared with the first
    character of ``bs``, and so on. Otherwise it does nothing.

    The instruction takes the targets of ``op`` followed by the condition
    bits: qubits of ``op``, classical bits of ``op``, ``len(bs)`` condition
    bits, then z-variables of ``op``. The condition bits may overlap with the
    bits of ``op``.

    Args:
        operation: The operation to apply, or an operation class built with
            no arguments.
        bitstring: The values the condition bits must hold.

    Raises:
        ValueError: If ``operation`` is not an operation or ``bitstring`` is
            not a :class:`BitString`.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> # measure qubit 0 into bit 0
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> # flip qubit 1 only if bit 0 is 1
        >>> c.push(IfStatement(GateX(), BitString('1')), 1, 0)
        2-qubit, 1-bit circuit with 2 instructions:
        ├── M @ q[0], c[0]
        └── IF(c==1) X @ q[1], condition[0]
        <BLANKLINE>
        >>> # measure qubit 1 into bit 3 if bits 1 and 2 are 0 and 1:
        >>> # the bit of the measurement comes before the condition bits
        >>> c.push(IfStatement(Measure(), BitString('01')), 1, 3, 1, 2)
        2-qubit, 4-bit circuit with 3 instructions:
        ├── M @ q[0], c[0]
        ├── IF(c==1) X @ q[1], condition[0]
        └── IF(c==01) M @ q[1], c[3], condition[1:2]
        <BLANKLINE>
    """

    _name = "If"

    _num_qubits = None

    _num_bits = None
    _num_cregs = 1

    _num_zvars = None

    _op = None
    _bitstring = None

    # IfStatement's classical-target layout is [op_bits..., condition_bits...];
    # the body operation may read or write a bit it tests against (for example
    # ``IfStatement(Not(), bs"1")`` toggling the same bit).
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
        """The conditioned operation. Read-only."""
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

    def supports_canonical_rewrite(self):
        return self.op.supports_canonical_rewrite()

    def getparams(self):
        """Return the parameters of the conditioned operation."""
        return self.op.getparams()

    def inverse(self):
        """Not supported for conditional operations.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Inverse not implemented for IfStatement")

    def power(self, power):
        """Not supported for conditional operations.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Power not implemented for IfStatement")

    def control(self, num_controls):
        """Not supported for conditional operations.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Control not implemented for IfStatement")

    def __str__(self):
        return f"IF (c=={self.bitstring.to01()}) {self.op}"

    def format_with_targets(self, qubits, bits, zvars):
        """Nested conditional formatting (explicit IF(c==..) chain)."""
        op = self.op
        chain = [self.bitstring.to01()]

        while isinstance(op, IfStatement):
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
        if not self.is_symbolic():
            return self
        return IfStatement(self.op.evaluate(d), self.bitstring)

    def get_operation(self):
        """Return the conditioned operation, :attr:`op`."""
        return self.op

    def get_bitstring(self):
        """Return the condition, :attr:`bitstring`."""
        return self.bitstring

    def asciiwidth(self, qubits, bits, zvars):
        val = self.get_bitstring()
        gw = self.op.asciiwidth(qubits, [], [])
        bstr = _string_with_square(_find_unit_range(bits), ",")
        iw = len(f"c{bstr}==" + val.to01()) + 2

        return max(gw, iw)

    def _decompose(self, circuit, qtargets, ctargets, ztargets):
        return self._decompose_ctx(circuit, qtargets, ctargets, ztargets, None)

    def _decompose_ctx(self, circuit, qtargets, ctargets, ztargets, ctx):
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        decomposed = _decompose_ifstatement(
            CanonicalRewrite(), self, qtargets, ctargets, ztargets, ctx
        )
        for inst in decomposed:
            circuit.push(inst.operation, *inst.qubits, *inst.bits, *inst.zvars)
        return circuit


def _decompose_ifstatement(basis, op, qubits, bits, zvars, ctx=None):
    """One decomposition step of the body of an IfStatement, kept conditional.

    Decomposes the body of ``op`` once with ``basis`` (a ``DecompositionBasis``
    or a ``RewriteRule``) and conditions every resulting instruction on the
    same bits. Bases reuse it so that a conditional gate follows the same rules
    as the bare gate. Nested conditions are merged into one IfStatement.
    """
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.decomposition.abstract import _NO_CONTEXT, DecompositionBasis

    inner = op.op
    condition = op.bitstring
    nb = inner.num_bits
    target_bits = list(bits[:nb])
    condition_bits = list(bits[nb:])

    # Each piece is conditioned in turn, so none of them may measure.
    ctx = _NO_CONTEXT if ctx is None else ctx.unitary()
    if isinstance(basis, DecompositionBasis):
        decomposed = basis.decompose_ctx(inner, qubits, target_bits, zvars, ctx)
    else:
        decomposed = basis.decompose_step_ctx(inner, qubits, target_bits, zvars, ctx)

    from mimiqcircuits.operations.annotations import AncillaAlloc, AncillaFree

    circ = Circuit()
    for inst in decomposed:
        body = inst.operation
        if isinstance(body, (AncillaAlloc, AncillaFree)):
            # The ancilla is in |0> before and after, whether the body runs or not.
            circ.push(body, *inst.qubits)
            continue
        if isinstance(body, IfStatement):
            # Flatten the if statements, the topmost condition goes to the back.
            merged = BitString(list(body.bitstring) + list(condition))
            body = IfStatement(body.op, merged)
        else:
            body = IfStatement(body, condition)
        circ.push(body, *inst.qubits, *inst.bits, *condition_bits, *inst.zvars)
    return circ


__all__ = ["IfStatement"]
