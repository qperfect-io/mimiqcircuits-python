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
from mimiqcircuits import Operation
from mimiqcircuits.operations.operator import AbstractOperator
import mimiqcircuits as mc


class ExpectationValue(Operation):
    r"""Operation to compute and store the expectation value of an Operator in a z-register.

    An expectation value for a pure state :math:`| \psi \rangle` is defined as:

    **Expectation Value for Pure State**

    .. math::
        \langle O \rangle = \langle \psi | O | \psi \rangle

    where :math:`O` is an operator. With respect to a density matrix :math:`\rho`, it's given by:

    **Expectation Value for Density Matrix**

    .. math::
        \langle O \rangle = \mathrm{Tr}(\rho O).

    However, when using quantum trajectories to solve noisy circuits, the expectation
    value is computed with respect to the pure state of each trajectory.

    The argument `op` can be any gate or non-unitary operator.

    .. note::
        ExpectationValue is currently restricted to one and two qubit operators.

    See Also:
        :class:`AbstractOperator`, :class:`Gate`

    Examples:

        In `push!`, the first argument corresponds to the qubit, and the second to the z-register.

        >>> from mimiqcircuits import *
        >>> ExpectationValue(GateX())
        ⟨X⟩

        >>> c = Circuit()
        >>> c.push(ExpectationValue(GateX()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨X⟩ @ q[1], z[1]
        <BLANKLINE>

        >>> c.push(ExpectationValue(SigmaPlus()), 1, 2)
        2-qubit, 3-zvar circuit with 2 instructions:
        ├── ⟨X⟩ @ q[1], z[1]
        └── ⟨SigmaPlus(1)⟩ @ q[1], z[2]
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

        if not isinstance(op, mc.PauliString) and not (1 <= op._num_qubits <= 2):
            raise ValueError(
                "ExpectationValue only supports 1- or 2-qubit operators unless the operator is a PauliString."
            )

        super().__init__()

        self._num_qubits = self.op._num_qubits
        self._qregsizes = [self._num_qubits]
        self._zregsizes = [1]
        self._parnames = tuple(op._parnames)

    def opname(self):
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
        raise NotImplementedError("Cannot inverse an ExpectationValue operation.")

    def power(self, _):
        raise NotImplementedError(
            "Cannot elevate an ExpectationValue operation to any power."
        )

    @staticmethod
    def isunitary():
        return True

    def __str__(self):
        return f"⟨{self.op}⟩"

    def iswrapper(self):
        return False

    def _decompose(self, circ, qubits, bits, zvars):
        return circ.push(self, *qubits, *bits, *zvars)

    def asciiwidth(self, qubits, bits=[], zvars=[]):
        """Calculate the width for ASCII drawing."""
        return len(self.__str__()) + 8

    def get_operation(self):
        return self.op

    def getparams(self):
        return self.op.getparams()

    def evaluate(self, d):
        return ExpectationValue(self.op.evaluate(d))


__all__ = ["ExpectationValue"]
