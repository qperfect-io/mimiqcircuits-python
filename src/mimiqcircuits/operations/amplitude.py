#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
"""Amplitude operation."""

from mimiqcircuits.bitstrings import BitString
from mimiqcircuits.operations.operation import Operation


class Amplitude(Operation):
    """Record the amplitude of a computational basis state.

    ``Amplitude(bs)`` reads the complex amplitude of the basis state ``bs``
    in the current quantum state and stores it in a z-variable. It does not
    change the state. The instruction takes only the target z-variable.

    Args:
        bs (BitString): The basis state whose amplitude is recorded.

    See Also:
        :class:`BitString`, :class:`ExpectationValue`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> # prepare a Bell state
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
        >>> # store the amplitude of |11> in z-variable 0
        >>> c.push(Amplitude(BitString("11")), 0)
        2-qubit, 1-zvar circuit with 3 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        └── Amplitude(bs"11") @ z[0]
        <BLANKLINE>
    """

    _name = "Amplitude"
    _num_zvars = 1
    _num_bits = 0
    _num_zregs = 1
    _num_cregs = 0
    _num_qregs = 0
    _num_qubits = 0

    def __init__(self, bs: BitString):
        self.bs = bs

        super().__init__()
        self._zregsizes = [1]

    @property
    def zregsizes(self):
        return self._zregsizes

    def iswrapper(self):
        return False

    def _decompose(self, circ, qubits, bits, zvars):
        return circ.push(self, *qubits, *bits, *zvars)

    def inverse(self):
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Cannot inverse an Amplitude operation.")

    def _power(self, _):
        raise NotImplementedError("Cannot elevate an Amplitude operation to any power.")

    def __str__(self):
        return f'{self._name}(bs"{self.bs.to01()}")'

    @staticmethod
    def isunitary():
        return True


__all__ = ["Amplitude"]
