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
"""Expectation value operation."""

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.operator import AbstractOperator
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString


class ExpectationValue(Operation):
    r"""Compute the expectation value of an operator and store it in a z-variable.

    For a pure state :math:`|\psi\rangle` the expectation value of an operator
    :math:`O` is

    .. math::
        \langle O \rangle = \langle \psi | O | \psi \rangle,

    and for a density matrix :math:`\rho` it is

    .. math::
        \langle O \rangle = \mathrm{Tr}(\rho O).

    When a noisy circuit is simulated with quantum trajectories, the value is
    computed on the pure state of each trajectory.

    The operator can be a gate or a non-unitary operator such as
    :class:`SigmaPlus` or :class:`Projector0`. It must act on one or two
    qubits, except for a :class:`PauliString`, which can act on any number.
    The instruction takes the operator's qubits followed by one z-variable.
    The quantum state is not changed.

    Args:
        op: The operator to measure.

    Raises:
        TypeError: If ``op`` is not an :class:`AbstractOperator`.
        ValueError: If ``op`` acts on more than two qubits and is not a
            :class:`PauliString`.

    See Also:
        :class:`AbstractOperator`, :class:`Gate`, :class:`Amplitude`

    Examples:
        >>> from mimiqcircuits import *
        >>> ExpectationValue(GateX())
        ⟨X⟩

        >>> # <X> on qubit 1, stored in z-variable 1
        >>> c = Circuit()
        >>> c.push(ExpectationValue(GateX()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instruction:
        └── ⟨X⟩ @ q[1], z[1]
        <BLANKLINE>

        >>> # non-unitary operators work too
        >>> c.push(ExpectationValue(SigmaPlus()), 1, 2)
        2-qubit, 3-zvar circuit with 2 instructions:
        ├── ⟨X⟩ @ q[1], z[1]
        └── ⟨SigmaPlus(1)⟩ @ q[1], z[2]
        <BLANKLINE>

        >>> # a Pauli string can span any number of qubits
        >>> c.push(ExpectationValue(PauliString("XYZ")), 0, 1, 2, 3)
        3-qubit, 4-zvar circuit with 3 instructions:
        ├── ⟨X⟩ @ q[1], z[1]
        ├── ⟨SigmaPlus(1)⟩ @ q[1], z[2]
        └── ⟨XYZ⟩ @ q[0:2], z[3]
        <BLANKLINE>
    """

    _name = "ExpectationValue"
    _num_zvars = 1
    _num_bits = 0
    _num_zregs = 1
    _num_cregs = 0
    _num_qregs = 1

    def __init__(self, op: AbstractOperator):
        self.op = op

        if not isinstance(op, AbstractOperator):
            raise TypeError(f"cannot get Expectation Value of {op.__class__.__name__}.")

        if not isinstance(op, PauliString) and not (1 <= op._num_qubits <= 2):
            raise ValueError(
                "ExpectationValue only supports 1- or 2-qubit operators unless the operator is a PauliString."
            )

        super().__init__()

        self._num_qubits = self.op._num_qubits
        self._qregsizes = [self._num_qubits]
        self._zregsizes = [1]
        self._parnames = tuple(op._parnames)

    def opname(self):
        """Return ``"ExpectationValue"``."""
        return self._name

    @property
    def qregsizes(self):
        return (self._num_qubits,)

    @property
    def cregsizes(self):
        return ()

    @property
    def zregsizes(self):
        return self._zregsizes

    def inverse(self):
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Cannot inverse an ExpectationValue operation.")

    def power(self, _):
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Cannot elevate an ExpectationValue operation to any power."
        )

    @staticmethod
    def isunitary():
        return True

    def __str__(self):
        return f"⟨{self.op}⟩"

    def iswrapper(self):
        # ``get_operation()`` returns the observable.
        return True

    def _decompose(self, circ, qubits, bits, zvars):
        return circ.push(self, *qubits, *bits, *zvars)

    def asciiwidth(self, qubits, bits=[], zvars=[]):
        """Return the width of the operation in ASCII circuit drawings."""
        return len(self.__str__()) + 8

    def get_operation(self):
        """Return the measured operator, ``op``."""
        return self.op

    def getparams(self):
        """Return the parameters of the measured operator."""
        return self.op.getparams()

    def evaluate(self, d):
        if not self.is_symbolic():
            return self
        return ExpectationValue(self.op.evaluate(d))


__all__ = ["ExpectationValue"]
