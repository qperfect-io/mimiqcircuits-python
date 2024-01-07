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

from mimiqcircuits.operations.operation import Operation


class IfStatement(Operation):
    """Conditional operation

    This operation applies a specific operation if a given classical bit condition is met.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(IfStatement(GateX(), 1,1), 0,0)
        init if -> operation= X
        1-qubit circuit with 1 instructions:
        └── If(X, 1) @ q[0], c[0]
        <BLANKLINE>
    """
    _name = 'If'

    _num_qubits = None

    _num_bits = None
    _num_cregs = 1

    _op = None
    _val = None

    def __init__(self, operation, num_bits, val, *args, **kwargs):
        print("init if -> operation=", operation)
        if isinstance(operation, type) and issubclass(operation, Operation):
            op = operation(*args, **kwargs)
        elif isinstance(operation, Operation):
            op = operation
        else:
            raise ValueError("Operation must be an Operation object or type.")

        if not isinstance(val, int) or val < 0:
            raise ValueError("val must be a positive integer")

        super().__init__()

        self._num_qubits = op.num_qubits
        self._num_qregs = op.num_qregs
        self._qregsizes = op.qregsizes

        self._num_bits = num_bits
        self._num_cregs = 1
        self._cregsizes = [num_bits, ]

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

    def iswrapper(self):
        return True

    def inverse(self):
        raise NotImplementedError("Inverse not implemented for IfStatement")

    def power(self, power):
        raise NotImplementedError("Power not implemented for IfStatement")

    def control(self, num_controls):
        raise NotImplementedError("Control not implemented for IfStatement")

    def __str__(self):
        return f'If({self.op}, {self.val})'

    def evaluate(self, d):
        return IfStatement(self.op.evaluate(d), self.num_bits, self.val)


# export operations
__all__ = ['IfStatement']
