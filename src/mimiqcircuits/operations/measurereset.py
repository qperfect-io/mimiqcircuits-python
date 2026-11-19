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
"""Measure and reset operations."""

from mimiqcircuits.bitstrings import BitString
from mimiqcircuits.operations.gates.standard.hadamard import GateH, GateHYZ
from mimiqcircuits.operations.gates.standard.pauli import GateX
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.measure import Measure
from mimiqcircuits.operations.measure import AbstractMeasurement


class MeasureReset(AbstractMeasurement):
    r"""Measure a qubit in the Z basis, then reset it to :math:`|0⟩`.

    Writes the outcome to a classical bit and then applies :class:`GateX`
    to the qubit if the outcome is 1, which leaves it in :math:`|0⟩`. It
    decomposes into :class:`Measure` followed by an :class:`IfStatement` of
    :class:`GateX` conditioned on that bit. The instruction takes one qubit
    and one bit.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`MeasureResetX`, :class:`MeasureResetY`, :class:`Reset`

    Examples:
        >>> from mimiqcircuits import *
        >>> # measure qubit 1 into bit 0 and leave it in |0>
        >>> c = Circuit()
        >>> c.push(MeasureReset(), 1, 0)
        2-qubit, 1-bit circuit with 1 instruction:
        └── MR @ q[1], c[0]
        <BLANKLINE>
        >>> # the same as a measurement followed by a conditional X
        >>> MeasureReset().decompose()
        1-qubit, 1-bit circuit with 2 instructions:
        ├── M @ q[0], c[0]
        └── IF(c==1) X @ q[0], condition[0]
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
        circ.push(Measure(), qubits, bits)
        circ.push(IfStatement(GateX(), bitstring=BitString("1")), qubits, bits)

        return circ

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


class MeasureResetX(AbstractMeasurement):
    r"""Measure a qubit in the X basis, then reset it to :math:`|+⟩`.

    Writes 0 for :math:`|+⟩` and 1 for :math:`|-⟩` to a classical bit and
    leaves the qubit in :math:`|+⟩`. It decomposes into a Hadamard gate, a
    :class:`MeasureReset` and a second Hadamard gate. The instruction takes
    one qubit and one bit.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`MeasureReset`, :class:`MeasureResetY`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureResetX(), 1, 0)
        2-qubit, 1-bit circuit with 1 instruction:
        └── MRX @ q[1], c[0]
        <BLANKLINE>
        >>> # rotate to the Z basis, measure and reset, rotate back
        >>> MeasureResetX().decompose()
        1-qubit, 1-bit circuit with 3 instructions:
        ├── H @ q[0]
        ├── MR @ q[0], c[0]
        └── H @ q[0]
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
        circ.push(GateH(), q)
        circ.push(MeasureReset(), q, c)
        circ.push(GateH(), q)
        return circ

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


class MeasureResetY(AbstractMeasurement):
    r"""Measure a qubit in the Y basis, then reset it to :math:`|{+i}⟩`.

    Writes 0 for :math:`|{+i}⟩` and 1 for :math:`|{-i}⟩` to a classical bit
    and leaves the qubit in :math:`|{+i}⟩`. It decomposes into a
    :class:`GateHYZ`, a :class:`MeasureReset` and a second
    :class:`GateHYZ`. The instruction takes one qubit and one bit.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`MeasureReset`, :class:`MeasureResetX`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureResetY(), 1, 0)
        2-qubit, 1-bit circuit with 1 instruction:
        └── MRY @ q[1], c[0]
        <BLANKLINE>
        >>> # rotate to the Z basis, measure and reset, rotate back
        >>> MeasureResetY().decompose()
        1-qubit, 1-bit circuit with 3 instructions:
        ├── HYZ @ q[0]
        ├── MR @ q[0], c[0]
        └── HYZ @ q[0]
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
        circ.push(GateHYZ(), q)
        circ.push(MeasureReset(), q, c)
        circ.push(GateHYZ(), q)
        return circ

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


class MeasureResetZ(AbstractMeasurement):
    r"""Alias of :class:`MeasureReset`.

    ``MeasureResetZ()`` returns a :class:`MeasureReset` instance.

    Examples:
        >>> from mimiqcircuits import *
        >>> MeasureResetZ()
        MR
    """

    def __new__(self):
        return MeasureReset()


# export operations
__all__ = ["MeasureReset", "MeasureResetX", "MeasureResetY", "MeasureResetZ"]
