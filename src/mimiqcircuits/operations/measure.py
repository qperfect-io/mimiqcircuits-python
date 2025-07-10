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


class AbstractMeasurement(Operation):

    _name = "AbstractMeasurement"

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


class Measure(AbstractMeasurement):
    """Measure operation.

    This operation performs a measurement in the computational basis (Z-basis) and stores the result in a classical register.

    The measurement projects the quantum state onto either the :math:`|0⟩` or :math:`|1⟩` state, corresponding to the classical bit values of 0 and 1, respectively.

    .. warning::
        `Measure` is non-reversible.

    Examples:

        Adding Measure operation to the Circuit (The qubits (first arg) and the bits (second arg) can be: range, list, tuple, set or int)

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Measure(),0,0)
        1-qubit, 1-bit circuit with 1 instructions:
        └── M @ q[0], c[0]
        <BLANKLINE>

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Measure(), range(0,3), range(0,3))
        3-qubit, 3-bit circuit with 3 instructions:
        ├── M @ q[0], c[0]
        ├── M @ q[1], c[1]
        └── M @ q[2], c[2]
        <BLANKLINE>
    """

    _name = "M"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("Measure is not inversible")

    def power(self, p):
        raise TypeError("Measure^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Measure is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self


class MeasureX(AbstractMeasurement):
    r"""Single qubit measurement operation in the X-basis.

    This operation projects the quantum state onto the X-basis and stores the result of such
    measurement in a classical register. The measurement is performed by first applying a Hadamard
    gate to rotate the qubit's state into the computational basis, performing a standard Z-basis
    measurement, and then applying another Hadamard gate to return the qubit to its original basis.

    .. warning::
        `MeasureX` is non-reversible.

    See Also:
        :class:`Measure`, :class:`MeasureY`, :class:`MeasureZ`, :class:`Operation`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> MeasureX()
        MX

        >>> c = Circuit()
        >>> c.push(MeasureX(), 2, 1)
        3-qubit, 2-bit circuit with 1 instructions:
        └── MX @ q[2], c[1]
        <BLANKLINE>

        >>> c.push(MeasureX(), 3, 4)
        4-qubit, 5-bit circuit with 2 instructions:
        ├── MX @ q[2], c[1]
        └── MX @ q[3], c[4]
        <BLANKLINE>
    """

    _name = "MX"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureX is not inversible")

    def power(self, p):
        raise TypeError("MeasureX^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureX is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        b = bits[0]
        circ.push(mc.GateH(), q)
        circ.push(Measure(), q, b)
        circ.push(mc.GateH(), q)
        return circ


class MeasureY(AbstractMeasurement):
    r"""Single qubit measurement operation in the Y-basis.

    This operation projects the quantum state onto the Y-basis and stores the result of such
    measurement in a classical register. The measurement is performed by first applying a :math:`R_y(\pi/2)`
    rotation gate to rotate the qubit's state into the computational basis, performing a standard Z-basis
    measurement, and then applying another :math:`R_y(-\pi/2)` rotation to return the qubit to its original basis.

    .. warning::
        `MeasureY` is non-reversible.

    See Also:
        :class:`Measure`, :class:`MeasureX`, :class:`MeasureZ`, :class:`Operation`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> MeasureY()
        MY

        >>> c = Circuit()
        >>> c.push(MeasureY(), 2, 1)
        3-qubit, 2-bit circuit with 1 instructions:
        └── MY @ q[2], c[1]
        <BLANKLINE>

        >>> c.push(MeasureY(), 3, 4)
        4-qubit, 5-bit circuit with 2 instructions:
        ├── MY @ q[2], c[1]
        └── MY @ q[3], c[4]
        <BLANKLINE>
    """

    _name = "MY"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]
    _num_zvars = 0

    def inverse(self):
        raise TypeError("MeasureY is not inversible")

    def power(self, p):
        raise TypeError("MeasureY^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureY is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def _decompose(self, circ, qubits, bits, zvars):
        q = qubits[0]
        b = bits[0]
        circ.push(mc.GateHYZ(), q)
        circ.push(Measure(), q, b)
        circ.push(mc.GateHYZ(), q)
        return circ


class MeasureZ(AbstractMeasurement):
    r"""Single qubit measurement operation in the Z-basis.

    This class returns an instance of the :class:`Measure` operation, effectively acting as an alias for Z-basis measurement.

    .. warning::
        `MeasureZ` is non-reversible.

    See Also:
        :class:`Measure`, :class:`MeasureX`, :class:`MeasureY`, :class:`Operation`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> MeasureZ()
        M

        >>> c = Circuit()
        >>> c.push(MeasureZ(), 0, 0)
        1-qubit, 1-bit circuit with 1 instructions:
        └── M @ q[0], c[0]
        <BLANKLINE>

        >>> c.push(MeasureZ(), 1, 2)
        2-qubit, 3-bit circuit with 2 instructions:
        ├── M @ q[0], c[0]
        └── M @ q[1], c[2]
        <BLANKLINE>
    """

    def __new__(self):
        return Measure()


# export operations
__all__ = ["Measure", "MeasureX", "MeasureY", "MeasureZ"]
