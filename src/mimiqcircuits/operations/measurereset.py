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
import mimiqcircuits as mc
from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.measure import AbstractMeasurement


class MeasureReset(AbstractMeasurement):
    """MeasureReset operation.

    This operation measures a qubit q, stores the value in a classical bit c,
    then applies a X operation to the qubit if the measured value is 1, effectively
    resetting the qubit to the :math:`\\ket{0}` state.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureReset(), 1, 0)
        2-qubit, 1-bit circuit with 1 instructions:
        └── MR @ q[1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit, 1-bit circuit with 2 instructions:
        ├── M @ q[1], c[0]
        └── IF (c==1) X @ q[1], c[0]
        <BLANKLINE>
    """

    _name = "MR"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureReset is not inversible")

    def power(self, p):
        raise TypeError("MeasureReset^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureReset is not defined.")

    def iswrapper(self):
        return False

    def _decompose(self, circ, qubits, bits, zvars):
        circ.push(mc.Measure(), qubits, bits)
        circ.push(mc.IfStatement(mc.GateX(), bitstring=mc.BitString("1")), qubits, bits)

        return circ

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


class MeasureResetX(AbstractMeasurement):
    """MeasureResetX operation.

    The MeasureResetX operation first applies a Hadamard gate (H) to the qubit,
    performs a measurement and reset operation similar to the MeasureReset operation,
    and then applies another Hadamard gate. This sequence effectively measures the
    qubit in the X-basis and resets it to the :math:`|+>` state.

    See Also:
        :class:`MeasureReset`: The standard measurement and reset operation in the Z-basis.
        :class:`MeasureResetY`: Similar operation that measures and resets the qubit in the Y-basis.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureResetX(), 1, 0)
        2-qubit, 1-bit circuit with 1 instructions:
        └── MRX @ q[1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit, 1-bit circuit with 3 instructions:
        ├── H @ q[1]
        ├── MR @ q[1], c[0]
        └── H @ q[1]
        <BLANKLINE>
    """

    _name = "MRX"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureResetX is not inversible")

    def power(self, p):
        raise TypeError("MeasureResetX^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureResetX is not defined.")

    def iswrapper(self):
        return False

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        c = bits[0]
        circ.push(mc.GateH(), q)
        circ.push(MeasureReset(), q, c)
        circ.push(mc.GateH(), q)
        return circ

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


class MeasureResetY(AbstractMeasurement):
    """MeasureResetY operation.

    The MeasureResetY operation applies (HYZ) gate to
    the qubit, performs a MeasureReset operation,
    and then applies another HYZ gate.
    This sequence effectively measures the qubit in
    the Y-basis.

    See Also:
        :class:`MeasureReset`: The standard measurement and reset operation in the Z-basis.
        :class:`MeasureResetX`: Similar operation that measures and resets the qubit in the X-basis.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureResetY(), 1, 0)
        2-qubit, 1-bit circuit with 1 instructions:
        └── MRY @ q[1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        2-qubit, 1-bit circuit with 3 instructions:
        ├── HYZ @ q[1]
        ├── MR @ q[1], c[0]
        └── HYZ @ q[1]
        <BLANKLINE>
    """

    _name = "MRY"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureResetY is not inversible")

    def power(self, p):
        raise TypeError("MeasureResetY^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureResetY is not defined.")

    def iswrapper(self):
        return False

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        c = bits[0]
        circ.push(mc.GateHYZ(), q)
        circ.push(MeasureReset(), q, c)
        circ.push(mc.GateHYZ(), q)
        return circ

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


class MeasureResetZ(AbstractMeasurement):
    r"""This class  acting as an alias for :class:`MeasureReset`."""

    def __new__(self):
        return MeasureReset()


# export operations
__all__ = ["MeasureReset", "MeasureResetX", "MeasureResetY", "MeasureResetZ"]
