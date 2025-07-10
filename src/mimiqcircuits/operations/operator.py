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

from abc import abstractmethod
import sympy as sp
import symengine as se
from mimiqcircuits import Operation


class AbstractOperator(Operation):
    r"""Supertype for all N-qubit operators.

    Note that objects of type `AbstractOperator` do not need to be unitary.

    Operators can be used to define Kraus channels (noise) (see :class:`krauschannel`),
    or to compute expectation values (see :class:`ExpectationValue`). However,
    they will return an error if directly applied to states.

    See Also:
        :func:`matrix`, :func:`isunitary`
    """

    _name = None
    _num_qubits = None
    _qregsizes = None
    _num_bits = 0
    _parnames = ()
    _num_cregs = 0
    _num_qregs = 1
    _num_zvars = 0

    _cregsizes = None

    def opname(self):
        return self.__class__.__name__

    @abstractmethod
    def _matrix(self):
        """Return the matrix representation of the operator.

        This method should be overridden in subclasses to provide the specific
        matrix corresponding to the operator.

        Returns:
            symengine.Matrix or sympy.Matrix: The matrix representation of the operator.
        """
        pass

    def iswrapper(self):
        """Check if the operator is a wrapper around another operator.

        This method should be overridden in subclasses to return `True` if the
        operator is acting as a wrapper around another operation or object, and
        `False` otherwise.

        Returns:
            bool: Always returns `False` in the base class. Subclasses should
            override this method to provide the appropriate logic.
        """
        return False

    def __str__(self):
        return f"{self.opname()}"

    @staticmethod
    def isunitary():
        """Check if the object is unitary.

        By default, this method returns `False` unless explicitly overridden in a subclass.
        """
        return False

    def matrix(self):
        """Compute the matrix representation of the operator.

        This method returns a symengine Matrix object representing the operator.
        It simplifies the matrix expression and evaluates it to a floating-point precision.

        Returns:
            symengine.Matrix: The matrix representation of the operator.
        """
        return se.Matrix(sp.simplify(sp.Matrix((self._matrix().tolist())).evalf()))

    def inverse(self):
        """Raise an error, as non-unitary operators cannot be inverted.

        This method is not implemented for non-unitary operators and will raise
        a `NotImplementedError` if called.

        Raises:
            NotImplementedError: If the method is called.
        """
        raise NotImplementedError("Cannot invert non-unitary operator")

    def power(self, n):
        """Raise an error, as powers of non-unitary operators are not supported.

        This method is not implemented for non-unitary operators and will raise
        a `NotImplementedError` if called.

        Args:
            n (int): The exponent to which the operator would be raised.

        Raises:
            NotImplementedError: If the method is called.
        """
        raise NotImplementedError("Cannot take power of non-unitary operator")

    def unwrappedmatrix(self):
        """Compute the matrix representation with all parameters evaluated.

        This method returns the matrix representation of the operator with all
        symbolic parameters substituted with their numerical values. If any parameter
        cannot be evaluated to a numerical value, a `ValueError` is raised.

        Returns:
            symengine.Matrix: The evaluated matrix representation of the operator.

        Raises:
            ValueError: If a parameter cannot be evaluated to a numerical value.
        """
        if self.numparams() == 0:
            return self.matrix()

        params = self.getparams()
        for param in params:
            value = sp.N(param)
            if isinstance(value, sp.Basic) and not value.is_Number:
                raise ValueError(f"Unexpected symbolic expression in {self}")
        return self.matrix()

    def evaluate(self, d):
        """
        Substitute the symbolic parameters of the operator with numerical values.

        This method evaluates the operator's symbolic parameters using the values
        provided in the dictionary `d`. If the operator has no parameters, it returns
        the same instance. Otherwise, it creates a new instance of the operator with
        updated numerical parameters.

        Parameters:
            d (dict): A dictionary where keys are symbolic parameter names and values are values for substitution.

        Example:
            >>> from symengine import *
            >>> from mimiqcircuits import *
            >>> theta = symbols('theta')
            >>> op = GateRX(theta)
            >>> evaluated_op = op.evaluate({'theta': 0.5})
            >>> print(evaluated_op)
            RX(0.5)
        """
        if len(self.parnames) == 0:
            return self

        params = self.getparams()

        for i in range(len(params)):
            if isinstance(params[i], (int, float)):
                continue

            params[i] = params[i].subs(d)

        return type(self)(*params)


__all__ = ["AbstractOperator"]
