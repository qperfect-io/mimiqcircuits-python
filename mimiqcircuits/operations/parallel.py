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

import numpy as np
from functools import reduce
import mimiqcircuits as mc
import symengine as se
import sympy as sp


class Parallel(mc.Operation):
    """Parallel operation

    This is a composite operation that applies multiple gates in parallel to the circuit at once.

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

    def __init__(self, num_repeats, op: mc.Operation):
        if not isinstance(op, mc.Operation):
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
        op_matrix = se.Matrix(sp.simplify(self.op.matrix()))
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

    def iswrapper(self):
        return True

    def inverse(self):
        return Parallel(self.num_repeats, self.op.inverse())

    def _decompose(self, circ, qubits, bits):
        for i in range(self.num_repeats):
            q = [q[j] for j in range(
                i * self.op.num_qubits, (i + 1) * self.op.num_qubits)]
            circ.push(self.op, *qubits)
        return circ

    def __str__(self):
        return f'Parallel({self.num_repeats}, {self.op})'

    def evaluate(self, param_dict):
        if not isinstance(self.op, (mc.Gate)):
            new_parallel = Parallel(self.num_repeats, self.op)
            if hasattr(new_parallel.op.op, 'evaluate'):
                new_parallel._op = new_parallel.op.op.evaluate(param_dict)
        else:
            new_parallel = Parallel(self.num_repeats, self.op)
            if hasattr(new_parallel._op, 'evaluate'):
                new_parallel._op = new_parallel._op.evaluate(param_dict)

        return new_parallel


# export operations
__all__ = ['Parallel']
