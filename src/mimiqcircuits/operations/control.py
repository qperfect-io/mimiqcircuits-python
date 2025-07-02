#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2032-2024 QPerfect. All Rights Reserved.
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

import mimiqcircuits as mc
import mimiqcircuits.operations.decompositions.control as ctrldecomp
from mimiqcircuits.printutils import print_wrapped_parens
import symengine as se
import sympy as sp
import mimiqcircuits.lazy as lz
from mimiqcircuits.operations.gates.gate import Gate


class Control(Gate):
    """Control operation.

    A Control is a special operation that applies multi-control gates to the Circuit at once.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Control(3,GateX()),1,2,3,4)
        5-qubit circuit with 1 instructions:
        └── C₃X @ q[1,2,3], q[4]
        <BLANKLINE>
        >>> Control(2, GateX()).matrix()
        [1.0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1.0, 0, 0, 0, 0, 0, 0]
        [0, 0, 1.0, 0, 0, 0, 0, 0]
        [0, 0, 0, 1.0, 0, 0, 0, 0]
        [0, 0, 0, 0, 1.0, 0, 0, 0]
        [0, 0, 0, 0, 0, 1.0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 1.0]
        [0, 0, 0, 0, 0, 0, 1.0, 0]
        <BLANKLINE>
    """

    _name = "Control"

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0
    _num_qregs = 2
    _num_controls = None

    _op = None

    def __init__(self, num_controls, operation, *args, **kwargs):
        if isinstance(operation, type) and issubclass(operation, mc.Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, mc.Gate):
            op = operation
        else:
            raise TypeError("Operation must be an Gate object or type.")

        if op.num_bits != 0:
            raise TypeError("Power operation cannot act on classical bits.")

        if num_controls < 1:
            raise ValueError("Controlled operations must have at least one control.")

        super().__init__()

        # TODO: check for possible problems when doing this, since we are not
        # using explicitly the operation given.
        if isinstance(op, Control):
            self._num_qubits = op.op.num_qubits + op.num_controls + num_controls
            self._num_controls = op.num_controls + num_controls
            self._op = op.op
            self._qregsizes = [self._num_controls]
            self._qregsizes.extend(op.op.qregsizes)

        else:
            self._num_qubits = op.num_qubits + num_controls
            self._num_controls = num_controls
            self._op = op
            self._qregsizes = [num_controls]
            self._qregsizes.extend(op.qregsizes)

    def _matrix(self):
        Mdim = 2**self.op.num_qubits
        Ldim = 2 ** (self.op.num_qubits + self.num_controls)
        mat = se.zeros(Ldim, Ldim)
        mat[Ldim - Mdim :, Ldim - Mdim :] = self.op.matrix()
        for i in range(0, Ldim - Mdim):
            mat[i, i] = 1
        return se.Matrix(sp.simplify(sp.Matrix(mat).evalf()))

    @property
    def num_controls(self):
        return self._num_controls

    @num_controls.setter
    def num_controls(self, value):
        raise ValueError("Cannot set num_controls. Read only parameter.")

    @property
    def num_targets(self):
        return self.num_qubits - self.num_controls

    @num_targets.setter
    def num_targets(self, value):
        raise ValueError("Cannot set num_targets. Read only parameter.")

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op):
        raise ValueError("Cannot set op. Read only parameter.")

    def inverse(self):
        return Control(self.num_controls, self.op.inverse())

    def getparams(self):
        return self.op.getparams()

    def get_operation(self):
        return self.op

    def control(self, *args):
        if len(args) == 0:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return Control(self.num_controls + num_controls, self.op)
        else:
            raise ValueError("Invalid number of arguments.")

    def _power(self, pwr):
        return Control(self.num_controls, self.op.power(pwr))

    def power(self, *args):
        if len(args) == 0:
            return lz.power(self)
        elif len(args) == 1:
            p = args[0]
            return self._power(p)
        else:
            raise ValueError("Invalid number of arguments.")

    def parallel(self, *args):
        if len(args) == 0:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return mc.Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments.")

    def __pow__(self, p):
        return self.power(p)

    def iswrapper(self):
        return True

    def __str__(self):
        controls_subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        ctext = ""
        if self.num_controls > 1:
            ctext = str(self.num_controls).translate(controls_subscript)

        return f"C{ctext}{print_wrapped_parens(self.op)}"

    def evaluate(self, d):
        ncontrol = self.num_controls
        return self.op.evaluate(d).control(ncontrol)

    def _decompose(self, circ, qubits, bits, zvars):
        decompose_map = {
            (2, mc.GateX): mc.GateCCX._decompose,
            (1, mc.GateX): mc.GateCX._decompose,
            (1, mc.GateY): mc.GateCY._decompose,
            (1, mc.GateZ): mc.GateCZ._decompose,
            (3, mc.GateX): mc.GateC3X._decompose,
            (1, mc.GateP): mc.GateCP._decompose,
            (2, mc.GateP): mc.GateCCP._decompose,
            (1, mc.GateRX): mc.GateCRX._decompose,
            (1, mc.GateRY): mc.GateCRY._decompose,
            (1, mc.GateRZ): mc.GateCRZ._decompose,
            (1, mc.GateSWAP): mc.GateCSWAP._decompose,
            (1, mc.GateU): mc.GateCU._decompose,
            (1, mc.GateH): mc.GateCH._decompose,
            (1, mc.GateSX): mc.GateCSX._decompose,
            (1, mc.GateS): mc.GateCS._decompose,
            (1, mc.GateSXDG): mc.GateCSXDG._decompose,
            (1, mc.GateSDG): mc.GateCSDG._decompose,
        }

        controls = qubits[: self.num_controls]
        targets = qubits[self.num_controls :]
        key = (self.num_controls, type(self.op))
        if key in decompose_map:
            return decompose_map[key](self, circ, qubits, bits, zvars)

        if self.num_controls == 1 or self.num_targets != 1:
            newcirc = self.op._decompose(mc.Circuit(), targets, bits, zvars)

            for inst in newcirc:
                inst_controls = list(controls)
                inst_targets = inst.get_qubits()
                circ.push(
                    Control(self.num_controls, inst.operation),
                    *inst_controls,
                    *inst_targets,
                )
            return circ

        else:
            return ctrldecomp.control_decompose(circ, self.op, controls, targets[0])


__all__ = ["Control"]
