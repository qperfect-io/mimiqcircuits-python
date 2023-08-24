#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
from mimiqcircuits.operation import Operation
import mimiqcircuits.json_utils as ju
from mimiqcircuits.bitstates import BitState


class IfStatement(Operation):
    """Conditional operation

    This operation applies a specific operation if a given classical bit condition is met.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(IfStatement(GateX(), BitState(1)), 0,0)
        >>> c.push(IfStatement(GateCX(), BitState(2,[0,1])), 1,3,1,2)
            4-qubit circuit with 2 instructions:
             ├── If(X, bs0) @ q0, c0
             └── If(CX, bs11) @ q1, q3, c1, c2
    """
    _name = 'If'
    _num_qubits = None
    _num_bits = None
    _op = None
    _val = None

    def __init__(self, op: Operation, val: BitState):
        if not isinstance(op, Operation):
            raise ValueError("op must be an Operation")

        if not isinstance(val, BitState) or len(val) < 1:
            raise ValueError("val must be a non-empty BitState")

        super().__init__()
        self._num_qubits = op.num_qubits
        self._num_bits = len(val)
        self._op = op
        self._val = val

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        raise ValueError("Cannot set val. Read only parameter.")

    def inverse(self):
        return IfStatement(self.op.inverse(), self.val)

    @staticmethod
    def from_json(d):
        if d['name'] != "If":
            raise ValueError("Invalid json for IfStatement")

        if 'op' not in d or 'val' not in d:
            raise ValueError("Invalid json for IfStatement")

        op = ju.operation_from_json(d['op'])
        val = BitState(d['val'])

        return IfStatement(op, val)

    def to_json(self):
        j = super().to_json()
        j['op'] = self.op.to_json()
        j['val'] = str(self.val.bits.to01())
        return j

    def __str__(self):
        return f'If({self.op}, bs{self.val.bits.to01()})'


# export operations
__all__ = ['IfStatement']
