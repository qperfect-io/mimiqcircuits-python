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

from mimiqcircuits.operations.operation import Operation
import mimiqcircuits as mc
from mimiqcircuits.canvas import _find_unit_range, _string_with_square


class IfStatement(Operation):
    """Conditional operation

    This operation applies a specific operation if a given classical bit condition is met.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(IfStatement(GateX(), BitString('1')), 0, 0)
        1-qubit, 1-bit circuit with 1 instructions:
        └── IF (c==1) X @ q[0], c[0]
        <BLANKLINE>
    """

    _name = "If"

    _num_qubits = None

    _num_bits = None
    _num_cregs = 1

    _num_zvars = 0

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
        self._num_qregs = op.num_qregs
        self._qregsizes = op.qregsizes

        self._num_bits = len(bitstring)
        self._cregsizes = [self._num_bits]

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
        decomposed_insts = self.op.decompose()
        bs = self.get_bitstring()
        for inst in decomposed_insts:
            conditional_operation = IfStatement(inst.operation, bs)
            targeted_qubits = [qtargets[i] for i in inst.get_qubits()]
            circuit.push(conditional_operation, *targeted_qubits, *ctargets, *ztargets)

        return circuit


__all__ = ["IfStatement"]
