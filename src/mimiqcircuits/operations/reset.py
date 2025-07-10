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

from mimiqcircuits.operations.krauschannel import krauschannel
import mimiqcircuits as mc


def _check_normaliz_length(normaliz, expected_length):
    if len(normaliz) != expected_length:
        raise ValueError(f"normaliz must be a list of length {expected_length}.")


class Reset(krauschannel):
    """Reset operation.

    Quantum operation that resets the status of one qubit to the :math:`\\ket{0}` state.

    .. warning::
        This operation is non-reversible.

    Examples:
        Adding Reset operation to the Circuit (The args can be: range, list, tuple, set or int)

        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Reset(), 0)
        1-qubit circuit with 1 instructions:
        └── Reset @ q[0]
        <BLANKLINE>
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(Reset(),(0,1,2))
        3-qubit circuit with 3 instructions:
        ├── Reset @ q[0]
        ├── Reset @ q[1]
        └── Reset @ q[2]
        <BLANKLINE>

    .. seealso::
        :class:`ResetX`, :class:`ResetY`, :class:`ResetZ` - Variants of the Reset operation along different axes.
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

    def __str__(self):
        return f"{self._name}"

    def asciiwidth(self, qubits, bits, zvars):
        return 5

    def krausoperators(self, normaliz=None):
        if normaliz is None:
            normaliz = [1, 1]
        _check_normaliz_length(normaliz, 2)
        return [mc.Projector0(1 / normaliz[0]), mc.SigmaMinus(1 / normaliz[1])]

    def __str__(self):
        return self._name


class ResetX(krauschannel):
    """ResetX operation.

    This operation is performed by applying a Hadamard gate (H) to the qubit, which transforms the
    qubit from the computational basis (:math:`\\ket{0}` or :math:`\\ket{1}`) to the superposition
    basis (:math:`\\ket{+}` or :math:`\\ket{-}`). After applying the Hadamard gate, a standard Reset
    operation is performed, resetting the qubit to the :math:`\\ket{0}` state. Finally, another Hadamard
    gate is applied to convert the :math:`\\ket{0}` state back into the :math:`\\ket{+}` state.

    .. warning::
        This operation is non-reversible.

    Examples:
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(ResetX(), 0)
        1-qubit circuit with 1 instructions:
        └── ResetX @ q[0]
        <BLANKLINE>
        >>> c.decompose()
        1-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── Reset @ q[0]
        └── H @ q[0]
        <BLANKLINE>

    .. seealso::
        :class:`Reset`, :class:`ResetY`, :class:`ResetZ` - Variants of the Reset operation along different axes.
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

    def _matrix(self):
        raise AttributeError("ResetX has no attribute 'matrix'.")

    def __str__(self):
        return self._name

    def _decompose(self, circ: mc.Circuit, qtargets, bits=None, zvars=None):
        q = qtargets[0]
        circ.push(mc.GateH(), q)
        circ.push(Reset(), q)
        circ.push(mc.GateH(), q)
        return circ


class ResetY(krauschannel):
    """ResetY operation.

    The `ResetY` operation works by first rotating the qubit's state so that the Y-axis aligns with the Z-axis,
    applying a standard reset (which resets the qubit to the :math:`\\ket{0}` state), and then rotating the qubit
    back to its original basis.

    .. warning::
        This operation is non-reversible.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(ResetY(), 0)
        1-qubit circuit with 1 instructions:
        └── ResetY @ q[0]
        <BLANKLINE>
        >>> c.decompose()
        1-qubit circuit with 3 instructions:
        ├── HYZ @ q[0]
        ├── Reset @ q[0]
        └── HYZ @ q[0]
        <BLANKLINE>

    .. seealso::
        :class:`Reset`, :class:`ResetX`, :class:`ResetZ` - Variants of the Reset operation along different axes.
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

    def _matrix(self):
        raise AttributeError("ResetY has no attribute 'matrix'.")

    def __str__(self):
        return self._name

    def _decompose(self, circ: mc.Circuit, qtargets, bits=None, zvars=None):
        q = qtargets[0]
        circ.push(mc.GateHYZ(), q)
        circ.push(Reset(), q)
        circ.push(mc.GateHYZ(), q)
        return circ


class ResetZ(krauschannel):
    """ResetZ operation.

    Quantum operation that resets the status of one qubit to the :math:`\\ket{0}` state along the Z-axis.
    This operation is an alias of the :class:`Reset` operation.

    .. warning::
        This operation is non-reversible.

    Examples:
        >>> from mimiqcircuits import *
        >>> c= Circuit()
        >>> c.push(ResetZ(), 0)
        1-qubit circuit with 1 instructions:
        └── Reset @ q[0]
        <BLANKLINE>
        >>> c.decompose()
        1-qubit circuit with 1 instructions:
        └── Reset @ q[0]
        <BLANKLINE>

    .. seealso::
        :class:`Reset`, :class:`ResetX`, :class:`ResetY` - Variants of the Reset operation along different axes.
    """

    def __new__(self):
        return Reset()


# export operations
__all__ = ["Reset", "ResetX", "ResetY", "ResetZ"]
