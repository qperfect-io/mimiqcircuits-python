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
import mimiqcircuits as mc
import numpy as np


class Control(Operation):
    """Control operation.

    A Control is an special operation that applies multi-control gates to the Circuit at once.

    Examples:
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Control(3,GateX()),1,2,3,4) 
        5-qubit circuit with 1 instructions:
            └── Control(3, X) @ q1, q2, q3, q4
    """
    _name = 'Control'
    _num_qubits = None
    _num_bits = 0
    _num_controls = None
    _op = None

    def __init__(self, *args):
        if len(args) == 1:
            num_controls = 1
            op = args[0]
        elif len(args) == 2:
            num_controls = args[0]
            op = args[1]

        if not isinstance(op, mc.Gate):
            raise TypeError(
                f"Cannot control {op.__class__.__name__} operation. It must be an unitary operation")

        if self.num_bits != 0:
            raise ValueError(
                "Controlled operations cannot act on classical bits.")

        if num_controls < 1:
            raise ValueError(
                "Controlled operations must have at least one control.")

        super().__init__()
        self._num_qubits = op.num_qubits + num_controls
        self._num_controls = num_controls
        self._op = op

    def matrix(self):
        op_matrix = self.op.matrix()
        Mdim = 2 ** self.op.num_qubits
        Ldim = 2 ** (self.op.num_qubits + self.num_controls)
        mat = np.zeros((Ldim, Ldim), dtype=op_matrix.dtype)
        mat[Ldim - Mdim:, Ldim - Mdim:] = op_matrix
        for i in range(0, Ldim - Mdim):
            mat[i, i] = 1.0
        return mat

    @property
    def num_controls(self):
        return self._num_controls

    @num_controls.setter
    def num_controls(self, value):
        raise ValueError('Cannot set num_controls. Read only parameter.')

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    def inverse(self):
        return Control(self.num_controls, self.op.inverse())

    @staticmethod
    def from_json(d):
        if d['name'] != "Control":
            raise ValueError("Invalid json for Control")

        if 'op' not in d or 'controls' not in d or 'N' not in d:
            raise ValueError("Invalid json for Control")

        op = ju.operation_from_json(d['op'])
        controls = d['controls']

        return Control(controls, op)

    def to_json(self):
        j = super().to_json()
        j['op'] = self.op.to_json()
        j['controls'] = self.num_controls
        return j

    def __str__(self):
        controls_subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        index = str(self.num_controls).translate(controls_subscript)
        control_str = f'C{index}{self.op}'
        if self.num_controls == 1:
            return f'C{self.op}'
        op_name = self.op._name
        if op_name.startswith('Control'):
            x = str(self.num_controls+1).translate(controls_subscript)
            y = f'{self.op}'.replace("C₁", "")
            return f'C{x}{y}'

        else:
            return f'{control_str}'


# export operations
__all__ = ['Control']
