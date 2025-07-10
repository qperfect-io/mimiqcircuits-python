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


class MeasureZZ(Operation):
    r"""MeasureZZ operation.

    The MeasureZZ operation measures the joint parity of two qubits in the Z-basis.
    This is achieved by applying a controlled-X (CX) gate, measuring the target qubit,
    and then applying another CX gate to undo the entanglement. The measurement result
    indicates whether the qubits are in the same or different states in the Z-basis.

    See Also:
        :class:`MeasureXX`: Measure the joint parity of two qubits in the X-basis.
        :class:`MeasureYY`: Measure the joint parity of two qubits in the Y-basis.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureZZ(), 0, 1, 0)
        2-qubit, 1-bit circuit with 1 instructions:
        └── MZZ @ q[0,1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit, 1-bit circuit with 3 instructions:
        ├── CX @ q[0], q[1]
        ├── M @ q[1], c[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """

    _name = "MZZ"
    _num_bits = 1
    _num_qubits = 2
    _num_cregs = 1
    _qregsizes = [2]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureZZ is not inversible")

    def power(self, p):
        raise TypeError("MeasureZZ^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureZZ is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def _decompose(self, circ, qubits, bits, zvars):
        q1, q2 = qubits
        c = bits[0]
        circ.push(mc.GateCX(), q1, q2)
        circ.push(mc.Measure(), q2, c)
        circ.push(mc.GateCX(), q1, q2)
        return circ


class MeasureXX(Operation):
    r"""MeasureXX operation.

    The MeasureXX operation measures the joint parity of two qubits in the X-basis, determining whether
    the qubits are in the same or different states within this basis. The operation begins by applying a
    controlled-X (CX) gate between the two qubits to entangle them.
    Following this, a Hadamard (H) gate is applied to the first qubit, rotating it into the X-basis.
    The second qubit, designated as the target, is then measured to extract the parity information.
    After the measurement, the Hadamard gate is applied again to the first qubit to reverse the rotation,
    and a second controlled-X (CX) gate is applied to disentangle the qubits, restoring the system to its original state.
    Through this sequence, the MeasureXX operation efficiently captures the parity relationship of the qubits in the X-basis.

    A result of `0` indicates that the qubits are in the same state, while a
    result of `1` indicates that they are in different states.


    See Also:
        :class:`MeasureZZ`: Measure the joint parity of two qubits in the Z-basis.
        :class:`MeasureYY`: Measure the joint parity of two qubits in the Y-basis.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureXX(), 0, 1, 0)
        2-qubit, 1-bit circuit with 1 instructions:
        └── MXX @ q[0,1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit, 1-bit circuit with 5 instructions:
        ├── CX @ q[0], q[1]
        ├── H @ q[0]
        ├── M @ q[1], c[0]
        ├── H @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """

    _name = "MXX"
    _num_bits = 1
    _num_qubits = 2
    _num_cregs = 1
    _qregsizes = [2]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureXX is not inversible")

    def power(self, p):
        raise TypeError("MeasureXX^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureXX is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def _decompose(self, circ, qubits, bits, zvars):
        q1, q2 = qubits
        c = bits[0]
        circ.push(mc.GateCX(), q1, q2)
        circ.push(mc.GateH(), q1)
        circ.push(mc.Measure(), q2, c)
        circ.push(mc.GateH(), q1)
        circ.push(mc.GateCX(), q1, q2)
        return circ


class MeasureYY(Operation):
    r"""MeasureYY operation.

    The MeasureYY operation measures the joint parity of two qubits in the Y-basis,
    determining whether they are in the same or different states in this basis.
    This is achieved by first applying an S gate (a π/2 phase shift) to both qubits,
    followed by a controlled-X (CX) gate. A Hadamard gate (H) is then applied to the
    first qubit, and the second qubit is measured. To restore the system, a Z gate is applied
    to the first qubit, followed by another Hadamard gate, another CX gate, and finally
    another S gate to both qubits. The measurement result reflects whether the qubits are in
    the same or different states in the Y-basis.

     See Also:
        :class:`MeasureZZ`: Measure the joint parity of two qubits in the Z-basis.
        :class:`MeasureXX`: Measure the joint parity of two qubits in the X-basis.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureYY(), 0, 1, 0)
        2-qubit, 1-bit circuit with 1 instructions:
        └── MYY @ q[0,1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit, 1-bit circuit with 10 instructions:
        ├── S @ q[0]
        ├── S @ q[1]
        ├── CX @ q[0], q[1]
        ├── H @ q[0]
        ├── M @ q[0], c[0]
        ├── Z @ q[0]
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        ├── S @ q[0]
        └── S @ q[1]
        <BLANKLINE>
    """

    _name = "MYY"
    _num_bits = 1
    _num_qubits = 2
    _num_cregs = 1
    _qregsizes = [2]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureYY is not inversible")

    def power(self, p):
        raise TypeError("MeasureYY^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureYY is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def _decompose(self, circ, qubits, bits, zvars):
        q1, q2 = qubits
        c = bits[0]
        circ.push(mc.GateS(), qubits)
        circ.push(mc.GateCX(), q1, q2)
        circ.push(mc.GateH(), q1)
        circ.push(mc.Measure(), q1, c)
        circ.push(mc.GateZ(), q1)
        circ.push(mc.GateH(), q1)
        circ.push(mc.GateCX(), q1, q2)
        circ.push(mc.GateS(), qubits)
        return circ
