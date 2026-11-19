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
"""Reset operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.gates.standard.hadamard import GateH, GateHYZ
from mimiqcircuits.operations.operators.projectors import Projector0
from mimiqcircuits.operations.operators.sigmas import SigmaMinus
from mimiqcircuits.operations.operators.custom import Operator
import numpy as np

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit


def _check_normaliz_length(normaliz, expected_length):
    if len(normaliz) != expected_length:
        raise ValueError(f"normaliz must be a list of length {expected_length}.")


class Reset(krauschannel):
    r"""Reset a qubit to :math:`|0⟩`.

    Puts the qubit in :math:`|0⟩` whatever its state, without recording an
    outcome. It is the Kraus channel with operators
    :math:`|0⟩⟨0|` and :math:`|0⟩⟨1|`.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`ResetX`, :class:`ResetY`, :class:`ResetZ`,
        :class:`MeasureReset`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Reset(), 0)
        1-qubit circuit with 1 instruction:
        └── Reset @ q[0]
        <BLANKLINE>

        Targets can be ranges, lists or tuples; one instruction is added per
        qubit:

        >>> c = Circuit()
        >>> c.push(Reset(),(0,1,2))
        3-qubit circuit with 3 instructions:
        ├── Reset @ q[0]
        ├── Reset @ q[1]
        └── Reset @ q[2]
        <BLANKLINE>
        >>> # the two Kraus operators: project on |0>, or move |1> to |0>
        >>> Reset().krausoperators()
        [P₀(1.0), SigmaMinus(1.0)]
    """

    _name = "Reset"
    _num_qubits = 1
    _num_bits = 0
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("Reset is not inversible")

    def power(self, pwr):
        raise TypeError("Reset^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Reset is not defined.")

    def iswrapper(self):
        return False

    def _matrix(self):
        raise AttributeError("Reset has no attribute 'matrix'.")

    def asciiwidth(self, qubits, bits, zvars):
        return 5

    def krausoperators(self, normaliz=None):
        """Return the Kraus operators :math:`|0⟩⟨0|` and :math:`|0⟩⟨1|`.

        Args:
            normaliz (list): Two numbers; each operator is divided by the
                matching entry. Defaults to ``[1, 1]``.

        Returns:
            list: ``[Projector0(...), SigmaMinus(...)]``.

        Raises:
            ValueError: If ``normaliz`` does not have length 2.
        """
        if normaliz is None:
            normaliz = [1, 1]
        _check_normaliz_length(normaliz, 2)
        return [Projector0(1 / normaliz[0]), SigmaMinus(1 / normaliz[1])]

    def __str__(self):
        return self._name


class ResetX(krauschannel):
    r"""Reset a qubit to :math:`|+⟩`.

    Decomposes into a Hadamard gate, a :class:`Reset` and a second Hadamard
    gate: the reset puts the qubit in :math:`|0⟩` and the last Hadamard
    turns it into :math:`|+⟩`.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`Reset`, :class:`ResetY`, :class:`ResetZ`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(ResetX(), 0)
        1-qubit circuit with 1 instruction:
        └── ResetX @ q[0]
        <BLANKLINE>
        >>> ResetX().decompose()
        1-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── Reset @ q[0]
        └── H @ q[0]
        <BLANKLINE>
    """

    _name = "ResetX"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("ResetX is not inversible")

    def power(self, pwr):
        raise TypeError("ResetX^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled ResetX is not defined.")

    def iswrapper(self):
        return False

    def supports_canonical_rewrite(self):
        return True

    def _matrix(self):
        raise AttributeError("ResetX has no attribute 'matrix'.")

    def krausoperators(self):
        """Return the two Kraus operators of the reset, as :class:`Operator`."""
        s = 1 / np.sqrt(2)
        return [
            Operator(np.array([[s, 0], [1 * s, 0]])),
            Operator(np.array([[0, s], [0, 1 * s]])),
        ]

    def __str__(self):
        return self._name

    def _decompose(self, circ: Circuit, qtargets, bits=None, zvars=None):
        q = qtargets[0]
        circ.push(GateH(), q)
        circ.push(Reset(), q)
        circ.push(GateH(), q)
        return circ


class ResetY(krauschannel):
    r"""Reset a qubit to :math:`|{+i}⟩`.

    Decomposes into a :class:`GateHYZ`, a :class:`Reset` and a second
    :class:`GateHYZ`: the reset puts the qubit in :math:`|0⟩` and the last
    gate turns it into :math:`|{+i}⟩`.

    Warning:
        Not reversible: ``inverse``, ``power`` and ``control`` raise
        ``TypeError``.

    See Also:
        :class:`Reset`, :class:`ResetX`, :class:`ResetZ`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(ResetY(), 0)
        1-qubit circuit with 1 instruction:
        └── ResetY @ q[0]
        <BLANKLINE>
        >>> ResetY().decompose()
        1-qubit circuit with 3 instructions:
        ├── HYZ @ q[0]
        ├── Reset @ q[0]
        └── HYZ @ q[0]
        <BLANKLINE>
    """

    _name = "ResetY"
    _num_qubits = 1
    _num_bits = 0
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("ResetY is not inversible")

    def power(self, pwr):
        raise TypeError("ResetY^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled ResetY is not defined.")

    def iswrapper(self):
        return False

    def supports_canonical_rewrite(self):
        return True

    def _matrix(self):
        raise AttributeError("ResetY has no attribute 'matrix'.")

    def krausoperators(self):
        """Return the two Kraus operators of the reset, as :class:`Operator`."""
        s = 1 / np.sqrt(2)
        return [
            Operator(np.array([[s, 0], [1j * s, 0]])),
            Operator(np.array([[0, s], [0, 1j * s]])),
        ]

    def __str__(self):
        return self._name

    def _decompose(self, circ: Circuit, qtargets, bits=None, zvars=None):
        q = qtargets[0]
        circ.push(GateHYZ(), q)
        circ.push(Reset(), q)
        circ.push(GateHYZ(), q)
        return circ


class ResetZ(krauschannel):
    r"""Reset a qubit to :math:`|0⟩`, an alias of :class:`Reset`.

    ``ResetZ()`` returns a :class:`Reset` instance.

    See Also:
        :class:`Reset`, :class:`ResetX`, :class:`ResetY`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(ResetZ(), 0)
        1-qubit circuit with 1 instruction:
        └── Reset @ q[0]
        <BLANKLINE>
    """

    def __new__(self):
        return Reset()


# export operations
__all__ = ["Reset", "ResetX", "ResetY", "ResetZ"]
