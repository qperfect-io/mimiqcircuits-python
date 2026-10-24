#
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
"""Magic-state preparation."""

import numpy as np

import mimiqcircuits as mc
from mimiqcircuits.operations.krauschannel import krauschannel


class PrepareStateT(krauschannel):
    r"""Prepare one qubit in the magic state.

    Prepares the magic state
    :math:`T\ket{+} = (\ket{0} + e^{i\pi/4}\ket{1})/\sqrt{2}`,
    whatever the previous state of the qubit.

    A magic state is the resource consumed by a :math:`\pi/8` Pauli rotation
    when the rotation is carried out with Pauli product measurements (see
    :func:`pauli_based_computation`). Fault-tolerant architectures produce
    these states separately, by distillation, so a dedicated operation lets
    them be counted and scheduled.

    The operation is a Kraus channel with operators :math:`T\ket{+}\bra{0}` and
    :math:`T\ket{+}\bra{1}`, and is equivalent to :class:`Reset`,
    :class:`GateH`, :class:`GateT`. It cannot be inverted, raised to a power
    or controlled.

    Examples:
        >>> from mimiqcircuits import *
        >>> PrepareStateT()
        PrepareStateT
        >>> c = Circuit()
        >>> c.push(PrepareStateT(), 0)
        1-qubit circuit with 1 instruction:
        └── PrepareStateT @ q[0]
        <BLANKLINE>

        The decomposition resets the qubit to :math:`\ket{0}`, then applies
        ``H`` and ``T``:

        >>> PrepareStateT().decompose()
        1-qubit circuit with 3 instructions:
        ├── Reset @ q[0]
        ├── H @ q[0]
        └── T @ q[0]
        <BLANKLINE>

    .. seealso::
        :class:`Reset`, :class:`ResetX`, :class:`MeasurePauli`
    """

    _name = "PrepareStateT"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("PrepareStateT is not inversible")

    def power(self, pwr):
        raise TypeError("PrepareStateT^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled PrepareStateT is not defined.")

    def iswrapper(self):
        return False

    def supports_canonical_rewrite(self):
        return True

    def _matrix(self):
        raise AttributeError("PrepareStateT has no attribute 'matrix'.")

    def __str__(self):
        return self._name

    def krausoperators(self):
        # T|+⟩⟨0| and T|+⟩⟨1|: whatever the input, the output is T|+⟩.
        m = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
        return [
            mc.Operator(np.outer(m, [1, 0])),
            mc.Operator(np.outer(m, [0, 1])),
        ]

    def _decompose(self, circ, qtargets, bits=None, zvars=None):
        q = qtargets[0]
        circ.push(mc.Reset(), q)
        circ.push(mc.GateH(), q)
        circ.push(mc.GateT(), q)
        return circ


# Deprecated alias, kept so existing code keeps working.
PrepareMagic = PrepareStateT

__all__ = ["PrepareStateT", "PrepareMagic"]
