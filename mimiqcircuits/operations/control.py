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
import mimiqcircuits.operations.gates.gate as mcg
import mimiqcircuits.operations.decompositions.control as ctrldecomp
import mimiqcircuits.matrices as matrices
import mimiqcircuits as mc
from symengine import *
import sympy as sp
import numpy as np


class Control(Operation):
    """Control operation.

    A Control is an special operation that applies multi-control gates to the Circuit at once.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Control(3,GateX()),1,2,3,4)
        5-qubit circuit with 1 instructions:
            └── Control(3, X) @ q1, q2, q3, q4
    """
    _name = 'Control'

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0

    _num_controls = None

    _op = None

    def __init__(self, num_controls, operation, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, Operation):
            op = operation(*args, **kwargs)
        elif isinstance(operation, Operation):
            op = operation
        else:
            raise TypeError("Operation must be an Operation object or type.")

        if isinstance(op, mc.Barrier) or isinstance(op, mc.Reset):
            raise TypeError("Barriers cannot be controlled operations.")

        if op.num_bits != 0:
            raise TypeError(
                "Power operation cannot act on classical bits.")

        if num_controls < 1:
            raise ValueError(
                "Controlled operations must have at least one control.")

        super().__init__()

        # TODO: check for possible problems when doing this, since we are not
        # using explicitly the operation given.
        if isinstance(op, Control):
            self._num_qubits = op.op.num_qubits + op.num_controls + num_controls
            self._num_controls = op.num_controls + num_controls
            self._op = op.op
            self._parnames = op.op.parnames
            self._qregsizes = op.op.qregsizes
            self._num_qregs = op.op.num_qregs
        else:
            self._num_qubits = op.num_qubits + num_controls
            self._num_controls = num_controls
            self._op = op
            self._qregsizes = op.qregsizes
            self._num_qregs = op.num_qregs
            self._parnames = op.parnames

    def matrix(self):
        op_matrix = sp.Matrix(self.op.matrix().tolist())
        Mdim = 2 ** self.op.num_qubits
        Ldim = 2 ** (self.op.num_qubits + self.num_controls)
        mat = np.zeros((Ldim, Ldim), dtype=object)
        mat[Ldim - Mdim:, Ldim - Mdim:] = op_matrix
        for i in range(0, Ldim - Mdim):
            mat[i, i] = 1
        return Matrix(mat.tolist())

    @property
    def num_controls(self):
        return self._num_controls

    @num_controls.setter
    def num_controls(self, value):
        raise ValueError('Cannot set num_controls. Read only parameter.')

    @property
    def num_targets(self):
        return self.num_qubits - self.num_controls

    @num_targets.setter
    def num_targets(self, value):
        raise ValueError('Cannot set num_targets. Read only parameter.')

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    def inverse(self):
        return Control(self.num_controls, self.op.inverse())

    def control(self, num_controls):
        return Control(self.num_controls + num_controls, self.op)

    def power(self, p):
        return Control(self.num_controls, self.op.power(p))

    def iswrapper(self):
        return True

    def __str__(self):
        controls_subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        ctext = ""
        if self.num_controls > 1:
            ctext = str(self.num_controls).translate(controls_subscript)

        if isinstance(self.op, mcg.Gate):
            return f"C{ctext}{str(self.op)}"
        else:
            return f"C{ctext}{str(self.op)}"

    def evaluate(self, param_dict):
        if not isinstance(self.op, (mcg.Gate)):
            new_control = Control(self.num_controls, self.op)
            if hasattr(new_control.op.op, 'evaluate'):
                new_control._op = new_control.op.op.evaluate(param_dict)
        else:
            new_control = Control(self.num_controls, self.op)
            if hasattr(new_control._op, 'evaluate'):
                new_control._op = new_control._op.evaluate(param_dict)

        return new_control

    def _decompose(self, circ, qubits, bits):
        return ctrldecomp.control_decompose(circ, self.op, qubits[:-1], qubits[-1])
