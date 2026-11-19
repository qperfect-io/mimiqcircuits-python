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
"""Measurement operations."""

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.gates.standard.hadamard import GateH, GateHYZ


class AbstractMeasurement(Operation):
    """Base class for measurement operations, such as :class:`Measure` and
    :class:`MeasureXX`.

    Noise model rules for readout errors apply to every subclass.
    """

    _name = "AbstractMeasurement"

    def iswrapper(self):
        return False

    def supports_canonical_rewrite(self):
        return True

    def __str__(self):
        return self._name

    def asciiwidth(self, qubits, bits, zvars):
        return len(str(self)) + 2


class Measure(AbstractMeasurement):
    """Single-qubit measurement in the computational (Z) basis.

    Projects the qubit onto :math:`|0⟩` or :math:`|1⟩` and writes the
    outcome, 0 or 1, to a classical bit. The instruction takes one qubit and
    one bit.

    Warning:
        Measurements are not reversible: ``inverse``, ``power`` and
        ``control`` raise ``TypeError``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # measure qubit 0 into bit 0
        >>> c = Circuit()
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── M @ q[0], c[0]
        <BLANKLINE>

        Targets can be ranges, lists or tuples; one instruction is added per
        qubit-bit pair:

        >>> c = Circuit()
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

    def supports_canonical_rewrite(self):
        # A primitive: the other measurements decompose into it.
        return False

    def power(self, p):
        raise TypeError("Measure^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Measure is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self


class MeasureX(AbstractMeasurement):
    r"""Single-qubit measurement in the X basis.

    Projects the qubit onto :math:`|+⟩` or :math:`|-⟩` and writes 0 or 1 to
    a classical bit. It decomposes into a Hadamard gate, a Z-basis
    :class:`Measure` and a second Hadamard gate, so the qubit is left in the
    measured X eigenstate.

    Warning:
        Measurements are not reversible: ``inverse``, ``power`` and
        ``control`` raise ``TypeError``.

    See Also:
        :class:`Measure`, :class:`MeasureY`, :class:`MeasureZ`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> MeasureX()
        MX

        >>> # measure qubit 2 into bit 1

        >>> c = Circuit()
        >>> c.push(MeasureX(), 2, 1)
        3-qubit, 2-bit circuit with 1 instruction:
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
        circ.push(GateH(), q)
        circ.push(Measure(), q, b)
        circ.push(GateH(), q)
        return circ


class MeasureY(AbstractMeasurement):
    r"""Single-qubit measurement in the Y basis.

    Projects the qubit onto :math:`|{+i}⟩` or :math:`|{-i}⟩` and writes 0 or
    1 to a classical bit. It decomposes into a :class:`GateHYZ`, a Z-basis
    :class:`Measure` and a second :class:`GateHYZ`, so the qubit is left in
    the measured Y eigenstate.

    Warning:
        Measurements are not reversible: ``inverse``, ``power`` and
        ``control`` raise ``TypeError``.

    See Also:
        :class:`Measure`, :class:`MeasureX`, :class:`MeasureZ`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> MeasureY()
        MY

        >>> c = Circuit()
        >>> c.push(MeasureY(), 2, 1)
        3-qubit, 2-bit circuit with 1 instruction:
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
        circ.push(GateHYZ(), q)
        circ.push(Measure(), q, b)
        circ.push(GateHYZ(), q)
        return circ


class MeasureZ(AbstractMeasurement):
    r"""Single-qubit measurement in the Z basis, an alias of :class:`Measure`.

    ``MeasureZ()`` returns a :class:`Measure` instance, not a ``MeasureZ``.

    See Also:
        :class:`Measure`, :class:`MeasureX`, :class:`MeasureY`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> # the constructor returns a plain Measure
        >>> MeasureZ()
        M

        >>> c = Circuit()
        >>> c.push(MeasureZ(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
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
