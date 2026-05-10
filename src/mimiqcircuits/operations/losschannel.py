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


class QubitLoss(Operation):
    """QubitLoss operation.

    This operation deterministically marks a qubit as lost. It is useful for
    representing qubit-loss events explicitly in a circuit.

    .. warning::
        This operation is non-reversible.

    Examples:
    
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(QubitLoss(), 1)
        2-qubit circuit with 1 instruction:
        └── QubitLoss @ q[1]
        <BLANKLINE>

    See Also:
        :class:`QubitReload`, :class:`LossErr`, :class:`CheckLoss`,
        :class:`MeasureCheckLoss`
    """

    _name = "QubitLoss"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _num_qregs = 1
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("QubitLoss is not inversible")

    def power(self, p):
        raise TypeError("QubitLoss^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled QubitLoss is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


class QubitReload(Operation):
    """QubitReload operation.

    This operation reloads a qubit that was previously marked as lost. It can
    be used to model qubit reintroduction after a loss event.

    .. warning::
        This operation is non-reversible.

    Examples:
    
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(QubitReload(), 1)
        2-qubit circuit with 1 instruction:
        └── QubitReload @ q[1]
        <BLANKLINE>

    See Also:
        :class:`QubitLoss`, :class:`LossErr`, :class:`CheckLoss`,
        :class:`MeasureCheckLoss`
    """

    _name = "QubitReload"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _num_qregs = 1
    _qregsizes = [1]

    def inverse(self):
        raise TypeError("QubitReload is not inversible")

    def power(self, p):
        raise TypeError("QubitReload^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled QubitReload is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


class LossErr(Operation):
    """LossErr operation.

    This operation models a probabilistic qubit-loss event with probability
    ``p``. During simulation or sampling, it can be resolved into a concrete
    qubit-loss event according to the specified probability.

    .. warning::
        This operation is non-reversible.

    Args:
        p: The probability of a qubit-loss event. It must be real and between
            0 and 1.

    Examples:
    
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(LossErr(0.5), 0)
        1-qubit circuit with 1 instruction:
        └── LossErr(0.5) @ q[0]
        <BLANKLINE>
        
        Seeded sampling makes stochastic loss reproducible:

        >>> import random
        >>> c = Circuit()
        >>> c.push(LossErr(0.5), 1)
        2-qubit circuit with 1 instruction:
        └── LossErr(0.5) @ q[1]
        <BLANKLINE>
        
        >>> c.push(GateH(), 1)
        2-qubit circuit with 2 instructions:
        ├── LossErr(0.5) @ q[1]
        └── H @ q[1]
        <BLANKLINE>
        
        >>> c.sample_losses(random.Random(0))
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
        
        >>> c.sample_losses(random.Random(7))
        2-qubit circuit with 1 instruction:
        └── QubitLoss @ q[1]
        <BLANKLINE>

        Symbolic probabilities must be evaluated before sampling:

        >>> from symengine import Symbol
        >>> p = Symbol("p")
        >>> c = Circuit()
        >>> c.push(LossErr(p), 0)
        1-qubit circuit with 1 instruction:
        └── LossErr(p) @ q[0]
        <BLANKLINE>
        
        >>> c = c.evaluate({p: 0.5})
        >>> c.sample_losses(random.Random(7))
        1-qubit circuit with 1 instruction:
        └── QubitLoss @ q[0]
        <BLANKLINE>

        Loss-model rules can rewrite gates that touch surviving qubits:

        >>> c = Circuit()
        >>> c.push(QubitLoss(), 1)
        2-qubit circuit with 1 instruction:
        └── QubitLoss @ q[1]
        <BLANKLINE>
        
        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── QubitLoss @ q[1]
        └── CX @ q[0], q[1]
        <BLANKLINE>
        
        >>> lm = LossModel().add_replace(GateCX(), Depolarizing1(0.2))
        >>> c.sample_losses(lm)
        2-qubit circuit with 2 instructions:
        ├── QubitLoss @ q[1]
        └── Depolarizing(0.2) @ q[0]
        <BLANKLINE>
        

    See Also:
        :class:`QubitLoss`, :class:`QubitReload`, :class:`CheckLoss`,
        :class:`MeasureCheckLoss`, :class:`mimiqcircuits.lossmodel.LossModel`
    """

    _name = "LossErr"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _num_qregs = 1
    _qregsizes = [1]
    _parnames = ("p",)

    def __init__(self, p):
        self.p = p
        _validate_probability(p)
        super().__init__()

    def evaluate(self, d={}):
        evaluated_p = self.p.subs(d) if hasattr(self.p, "subs") else self.p

        try:
            numeric_p = unwrapvalue(evaluated_p)
        except UndefinedValue:
            return LossErr(evaluated_p)

        if isinstance(numeric_p, complex):
            if numeric_p.imag != 0:
                raise ValueError("Loss probability p must be real after evaluation.")
            numeric_p = numeric_p.real

        return LossErr(numeric_p)

    def inverse(self):
        raise TypeError("LossErr is not inversible")

    def power(self, p):
        raise TypeError("LossErr^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled LossErr is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self._name}({self.p})"


class CheckLoss(Operation):
    """CheckLoss operation.

    This operation checks whether a qubit is present or lost, and stores the
    result in a classical bit.

    .. warning::
        This operation is non-reversible.

    Examples:
    
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(CheckLoss(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── CL @ q[0], c[0]
        <BLANKLINE>

    See Also:
        :class:`QubitLoss`, :class:`QubitReload`, :class:`LossErr`,
        :class:`MeasureCheckLoss`
    """

    _name = "CL"
    _num_qubits = 1
    _num_bits = 1
    _num_zvars = 0
    _num_qregs = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]

    def inverse(self):
        raise TypeError("CheckLoss is not inversible")

    def power(self, p):
        raise TypeError("CheckLoss^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled CheckLoss is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


class MeasureCheckLoss(Operation):
    """MeasureCheckLoss operation.

    This operation measures a qubit and reports both the measurement result and
    whether the qubit is present or lost.

    .. warning::
        This operation is non-reversible.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureCheckLoss(), 0, 0, 1)
        1-qubit, 2-bit circuit with 1 instruction:
        └── MCL @ q[0], c[0:1]
        <BLANKLINE>

    See Also:
        :class:`QubitLoss`, :class:`QubitReload`, :class:`LossErr`,
        :class:`CheckLoss`
    """

    _name = "MCL"
    _num_qubits = 1
    _num_bits = 2
    _num_zvars = 0
    _num_qregs = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [2]

    def inverse(self):
        raise TypeError("MeasureCheckLoss is not inversible")

    def power(self, p):
        raise TypeError("MeasureCheckLoss^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureCheckLoss is not defined.")

    def iswrapper(self):
        return False

    def __str__(self):
        return self._name


__all__ = [
    "QubitLoss",
    "QubitReload",
    "LossErr",
    "CheckLoss",
    "MeasureCheckLoss",
]
