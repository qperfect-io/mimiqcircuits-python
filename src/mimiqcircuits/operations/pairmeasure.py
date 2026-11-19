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
"""Pair measurement operations (ZZ, XX, YY)."""

from mimiqcircuits.operations.measure import AbstractMeasurement
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.measure import Measure


class MeasureZZ(AbstractMeasurement):
    r"""Measure the parity of two qubits in the Z basis.

    Measures the two-qubit Pauli operator :math:`Z \otimes Z` and writes 0
    for eigenvalue +1 (the qubits agree in the Z basis) or 1 for eigenvalue
    -1 (they differ). Only the parity is measured, so superpositions within
    each parity sector are kept. The instruction takes two qubits and one
    bit.

    It decomposes into a CX gate, a measurement of the second qubit, and a
    second CX gate that undoes the first.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`MeasureXX`, :class:`MeasureYY`, :class:`MeasurePauli`

    Examples:
        >>> from mimiqcircuits import *
        >>> # measure the ZZ parity of qubits 0 and 1 into bit 0
        >>> c = Circuit()
        >>> c.push(MeasureZZ(), 0, 1, 0)
        2-qubit, 1-bit circuit with 1 instruction:
        └── MZZ @ q[0:1], c[0]
        <BLANKLINE>
        >>> MeasureZZ().decompose()
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
        circ.push(GateCX(), q1, q2)
        circ.push(Measure(), q2, c)
        circ.push(GateCX(), q1, q2)
        return circ


class MeasureXX(AbstractMeasurement):
    r"""Measure the parity of two qubits in the X basis.

    Measures the two-qubit Pauli operator :math:`X \otimes X` and writes 0
    for eigenvalue +1 or 1 for eigenvalue -1. The instruction takes two
    qubits and one bit.

    It decomposes into a CX gate from the first qubit to the second and a
    Hadamard gate on the first, which together map :math:`X \otimes X` to
    :math:`Z` on the first qubit, then a measurement of that qubit, then the
    same two gates in reverse order. The qubits are left in the
    post-measurement state of the :math:`X \otimes X` measurement.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`MeasureZZ`, :class:`MeasureYY`, :class:`MeasurePauli`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureXX(), 0, 1, 0)
        2-qubit, 1-bit circuit with 1 instruction:
        └── MXX @ q[0:1], c[0]
        <BLANKLINE>
        >>> MeasureXX().decompose()
        2-qubit, 1-bit circuit with 5 instructions:
        ├── CX @ q[0], q[1]
        ├── H @ q[0]
        ├── M @ q[0], c[0]
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
        circ.push(GateCX(), q1, q2)
        circ.push(GateH(), q1)
        circ.push(Measure(), q1, c)
        circ.push(GateH(), q1)
        circ.push(GateCX(), q1, q2)
        return circ


class MeasureYY(AbstractMeasurement):
    r"""Measure the parity of two qubits in the Y basis.

    Measures the two-qubit Pauli operator :math:`Y \otimes Y` and writes 0
    for eigenvalue +1 or 1 for eigenvalue -1. The instruction takes two
    qubits and one bit.

    It decomposes into an S gate on both qubits, a CX gate from the first
    qubit to the second and a Hadamard gate on the first, which together map
    :math:`Y \otimes Y` to :math:`Z` on the first qubit, then a measurement
    of that qubit, then the inverse gates in reverse order. The qubits are
    left in the post-measurement state of the :math:`Y \otimes Y`
    measurement.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`MeasureZZ`, :class:`MeasureXX`, :class:`MeasurePauli`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureYY(), 0, 1, 0)
        2-qubit, 1-bit circuit with 1 instruction:
        └── MYY @ q[0:1], c[0]
        <BLANKLINE>
        >>> MeasureYY().decompose()
        2-qubit, 1-bit circuit with 9 instructions:
        ├── S @ q[0]
        ├── S @ q[1]
        ├── CX @ q[0], q[1]
        ├── H @ q[0]
        ├── M @ q[0], c[0]
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        ├── S† @ q[0]
        └── S† @ q[1]
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
        circ.push(GateS(), qubits)
        circ.push(GateCX(), q1, q2)
        circ.push(GateH(), q1)
        circ.push(Measure(), q1, c)
        circ.push(GateH(), q1)
        circ.push(GateCX(), q1, q2)
        circ.push(GateSDG(), qubits)
        return circ
