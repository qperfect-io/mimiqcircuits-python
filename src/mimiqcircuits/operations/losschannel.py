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
    """One-qubit loss event: the qubit is lost with probability ``p``.

    ``Loss()`` is ``Loss(1.0)``, a certain loss. A lost qubit leaves the
    computation: gates on it are dropped, a measurement of it reads 0, and
    :class:`Check` reports it as missing, until a :class:`Reload` brings it
    back.

    The probability is resolved by :meth:`Circuit.sample_losses`, which turns
    each ``Loss(p)`` into ``Loss(1.0)`` or removes it.
    :meth:`Circuit.resolve_losses` then gives a runnable, loss-free circuit;
    a :class:`LossModel` decides what happens to gates acting on both lost and
    present qubits.

    Loss cannot be inverted, raised to a power or controlled: :meth:`inverse`,
    :meth:`power` and :meth:`control` raise :class:`TypeError`.

    Args:
        p: Loss probability, real and in :math:`[0, 1]` (default ``1.0``).
            May be symbolic; it must be numeric before sampling.

    Raises:
        ValueError: If a numeric ``p`` is not real or is outside
            :math:`[0, 1]`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # qubit 0 is lost with probability 0.1
        >>> c = Circuit()
        >>> c.push(Loss(0.1), 0)
        1-qubit circuit with 1 instruction:
        └── Loss(0.1) @ q[0]
        <BLANKLINE>
        >>> # qubit 1 is lost for sure
        >>> c.push(Loss(), 1)
        2-qubit circuit with 2 instructions:
        ├── Loss(0.1) @ q[0]
        └── Loss(1.0) @ q[1]
        <BLANKLINE>

    See Also:
        :class:`Reload`, :class:`Check`, :class:`MeasureCheck`,
        :func:`sample_losses`, :func:`resolve_losses`
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
        """Return a copy with symbols in ``p`` substituted by values.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            Loss: A new operation; ``p`` becomes a number when it is fully
            numeric.

        Raises:
            ValueError: If the substituted ``p`` is not real or is outside
                :math:`[0, 1]`.
        """
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
    """Reload a qubit: reset it to :math:`|0\\rangle` and mark it present.

    A reload always resets the qubit, whether or not it was lost. After
    :func:`lower_losses` it becomes a :class:`Reset` followed by a
    :class:`Reloaded` marker.

    Reload is not reversible: :meth:`inverse`, :meth:`power` and
    :meth:`control` raise :class:`TypeError`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # bring qubit 0 back in |0>, lost or not
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
    """Record whether a qubit is present into a classical bit.

    The bit is set to 1 if the qubit is present and to 0 if it is lost. Like
    :class:`Measure` records a qubit's value, ``Check`` records its presence;
    it does not touch the quantum state. It targets one qubit and one bit.

    Check cannot be inverted, raised to a power or controlled: those methods
    raise :class:`TypeError`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # write the presence of qubit 0 into bit 0
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
    """Measure a qubit if it is present, and record its presence.

    It targets one qubit and two bits. The first bit gets the Z-basis
    measurement result (0 if the qubit is lost), the second the presence (1
    present, 0 lost). The state collapses only when the qubit is present.

    MeasureCheck cannot be inverted, raised to a power or controlled: those
    methods raise :class:`TypeError`.

    Examples:
        >>> from mimiqcircuits import *
        >>> # outcome of qubit 0 into bit 0, its presence into bit 1
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
