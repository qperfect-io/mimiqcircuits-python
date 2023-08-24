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
from mimiqcircuits.gates import Gate
import mimiqcircuits.json_utils as ju
import numpy as np
from functools import reduce


class Parallel(Operation):
    """Parallel operation

    This is a composite operation that an operation in in parallel to
    on multiple targets.

    Examples:
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Parallel(3,GateX()),1,2,3)
        4-qubit circuit with 1 instructions:
         └── Parallel(3, X) @ q1, q2, q3
    """
    _name = 'Parallel'
    _num_qubits = None
    _num_bits = 0
    _num_repeats = None
    _op = None

    def __init__(self, num_repeats, op: Operation):
        if not isinstance(op, Gate):
            raise ValueError("op must be an Operation")

        if self.num_bits != 0:
            raise ValueError(
                "Parallel operations cannot act on classical bits.")

        if num_repeats < 2:
            raise ValueError(
                f"Parallel operations must have at least two repeats. If one, just use {op}.")

        super().__init__()
        self._num_qubits = op.num_qubits * num_repeats
        self._num_repeats = num_repeats
        self._op = op

    def matrix(self):
        op_matrix = self.op.matrix()
        return reduce(np.kron, [op_matrix] * (self._num_repeats))

    @property
    def num_repeats(self):
        return self._num_repeats

    @num_repeats.setter
    def num_repeats(self, value):
        raise ValueError('Cannot set num_repeats. Read only parameter.')

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    def inverse(self):
        return Parallel(self.num_repeats, self.op.inverse())

    @staticmethod
    def from_json(d):
        if d['name'] != "Parallel":
            raise ValueError("Invalid json for Parallel")

        if 'op' not in d or 'repeats' not in d:
            raise ValueError("Invalid json for Parallel")

        op = ju.operation_from_json(d['op'])
        repeats = d['repeats']

        return Parallel(repeats, op)

    def to_json(self):
        j = super().to_json()
        j['op'] = self.op.to_json()
        j['repeats'] = self.num_repeats
        return j

    def __str__(self):
        return f'Parallel({self.num_repeats}, {self.op})'


# export operations
__all__ = ['Parallel']
