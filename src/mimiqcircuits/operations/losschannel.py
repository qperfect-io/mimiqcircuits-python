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
"""Loss-channel operations."""

from __future__ import annotations

import warnings

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.symbolics import UndefinedValue, unwrapvalue


def _validate_probability(p):
    try:
        value = unwrapvalue(p)
    except UndefinedValue:
        return

    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError("Loss probability p must be real.")
        value = value.real

    if not (0 <= value <= 1):
        raise ValueError("Loss probability p must be between 0 and 1.")


class Loss(Operation):
    """Loss operation.

    A qubit-loss event: the qubit is lost with probability ``p``. ``Loss()`` is
    ``Loss(1.0)``, a certain loss. It replaces both ``LossErr`` and
    ``QubitLoss``.

    The probability is resolved by :meth:`mimiqcircuits.Circuit.sample_losses`,
    which turns each ``Loss(p)`` into ``Loss(1.0)`` or removes it, and a circuit
    is turned into a runnable, loss-free form by
    :meth:`mimiqcircuits.Circuit.resolve_losses`.

    .. warning::
        This operation is non-reversible.

    Args:
        p: The loss probability, real and between 0 and 1. Defaults to 1.0.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Loss(0.1), 0)
        1-qubit circuit with 1 instruction:
        └── Loss(0.1) @ q[0]
        <BLANKLINE>
        >>> c.push(Loss(), 1)
        2-qubit circuit with 2 instructions:
        ├── Loss(0.1) @ q[0]
        └── Loss(1.0) @ q[1]
        <BLANKLINE>

    See Also:
        :class:`Reload`, :class:`Check`, :class:`MeasureCheck`
    """

    _name = "Loss"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _num_qregs = 1
    _qregsizes = [1]
    _parnames = ("p",)

    def __init__(self, p=1.0):
        self.p = p
        _validate_probability(p)
        super().__init__()

    def evaluate(self, d={}):
        evaluated_p = self.p.subs(d) if hasattr(self.p, "subs") else self.p

        try:
            numeric_p = unwrapvalue(evaluated_p)
        except UndefinedValue:
            return Loss(evaluated_p)

        if isinstance(numeric_p, complex):
            if numeric_p.imag != 0:
                raise ValueError("Loss probability p must be real after evaluation.")
            numeric_p = numeric_p.real

        return Loss(numeric_p)

    def inverse(self):
        raise TypeError("Loss is not inversible")

    def power(self, p):
        raise TypeError("Loss^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Loss is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self._name}({self.p})"


class Reload(Operation):
    """Reload operation.

    Re-initialise a qubit to :math:`|0\\rangle` and mark it present. By default
    a reload always resets, regardless of whether the qubit was lost. It
    replaces ``QubitReload``.

    .. warning::
        This operation is non-reversible.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Reload(), 0)
        1-qubit circuit with 1 instruction:
        └── Reload @ q[0]
        <BLANKLINE>

    See Also:
        :class:`Loss`, :class:`Reset`
    """

    _name = "Reload"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _num_qregs = 1
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("Reload is not inversible")

    def power(self, p):
        raise TypeError("Reload^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Reload is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


class Check(Operation):
    """Check operation.

    Record whether a qubit is present into a classical bit (1 present, 0 lost).
    Like :class:`Measure` records a qubit's value, ``Check`` records its
    presence; it does not touch the quantum state. It replaces ``CheckLoss``.

    .. warning::
        This operation is non-reversible.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Check(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── Check @ q[0], c[0]
        <BLANKLINE>

    See Also:
        :class:`MeasureCheck`, :class:`Loss`, :class:`Reload`
    """

    _name = "Check"
    _num_qubits = 1
    _num_bits = 1
    _num_zvars = 0
    _num_qregs = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]

    def inverse(self):
        raise TypeError("Check is not inversible")

    def power(self, p):
        raise TypeError("Check^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled Check is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


class MeasureCheck(Operation):
    """MeasureCheck operation.

    Measure a qubit if it is present, and record its presence. The first bit is
    the measurement result (0 if lost), the second is the presence (1 present,
    0 lost). The state is collapsed only when the qubit is present. It replaces
    ``MeasureCheckLoss``.

    .. warning::
        This operation is non-reversible.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureCheck(), 0, 0, 1)
        1-qubit, 2-bit circuit with 1 instruction:
        └── MeasureCheck @ q[0], c[0:1]
        <BLANKLINE>

    See Also:
        :class:`Check`, :class:`Measure`, :class:`Loss`
    """

    _name = "MeasureCheck"
    _num_qubits = 1
    _num_bits = 2
    _num_zvars = 0
    _num_qregs = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [2]

    def inverse(self):
        raise TypeError("MeasureCheck is not inversible")

    def power(self, p):
        raise TypeError("MeasureCheck^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureCheck is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


# Deprecated loss operations. The old names map onto the current ones.
def LossErr(p):
    warnings.warn("LossErr is deprecated; use Loss(p).", DeprecationWarning, stacklevel=2)
    return Loss(p)


def QubitLoss():
    warnings.warn("QubitLoss is deprecated; use Loss().", DeprecationWarning, stacklevel=2)
    return Loss(1.0)


def QubitReload():
    warnings.warn("QubitReload is deprecated; use Reload().", DeprecationWarning, stacklevel=2)
    return Reload()


def CheckLoss():
    warnings.warn("CheckLoss is deprecated; use Check().", DeprecationWarning, stacklevel=2)
    return Check()


def MeasureCheckLoss():
    warnings.warn("MeasureCheckLoss is deprecated; use MeasureCheck().", DeprecationWarning, stacklevel=2)
    return MeasureCheck()
