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

    # IfStatement's classical-target layout is [op_bits..., condition_bits...];
    # the body operation may read or write a bit it tests against (for example
    # ``IfStatement(Not(), bs"1")`` toggling the same bit).
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

    def supports_canonical_rewrite(self):
        return self.op.supports_canonical_rewrite()

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
        if not self.is_symbolic():
            return self
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
        from mimiqcircuits.decomposition.rules import CanonicalRewrite

        decomposed = _decompose_ifstatement(
            CanonicalRewrite(), self, qtargets, ctargets, ztargets
        )
        for inst in decomposed:
            circuit.push(inst.operation, *inst.qubits, *inst.bits, *inst.zvars)
        return circuit


def _decompose_ifstatement(basis, op, qubits, bits, zvars):
    """One decomposition step of the body of an IfStatement, kept conditional.

    Decomposes the body of ``op`` once with ``basis`` (a ``DecompositionBasis``
    or a ``RewriteRule``) and conditions every resulting instruction on the
    same bits. Bases reuse it so that a conditional gate follows the same rules
    as the bare gate. Nested conditions are merged into one IfStatement.
    """
    from mimiqcircuits.decomposition.abstract import DecompositionBasis

    inner = op.op
    condition = op.bitstring
    nb = inner.num_bits
    target_bits = list(bits[:nb])
    condition_bits = list(bits[nb:])

    if isinstance(basis, DecompositionBasis):
        decomposed = basis.decompose(inner, qubits, target_bits, zvars)
    else:
        decomposed = basis.decompose_step(inner, qubits, target_bits, zvars)

    circ = mc.Circuit()
    for inst in decomposed:
        body = inst.operation
        if isinstance(body, mc.IfStatement):
            # Flatten the if statements, the topmost condition goes to the back.
            merged = mc.BitString(list(body.bitstring) + list(condition))
            body = mc.IfStatement(body.op, merged)
        else:
            body = mc.IfStatement(body, condition)
        circ.push(body, *inst.qubits, *inst.bits, *condition_bits, *inst.zvars)
    return circ


__all__ = ["IfStatement"]
