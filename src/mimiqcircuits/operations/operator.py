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
"""AbstractOperator base class."""

from abc import abstractmethod
import numbers

import numpy as np
import sympy as sp
import symengine as se

from mimiqcircuits import Operation


class UnexpectedSymbolics(ValueError):
    """Raised when :func:`unwrappedmatrix` hits a parameter that can't be
    evaluated to a number (Julia parity with ``UnexpectedSymbolics``)."""

    def __init__(self, op):
        super().__init__(
            f"Unexpected symbolic expression in {op}. "
            "Try to evaluate or define all symbols."
        )


def _try_unwrap(p):
    """Try to reduce ``p`` to a plain Python number.

    Returns ``(value, True)`` on success and ``(p, False)`` when ``p`` is
    genuinely symbolic. Numeric SymEngine/SymPy expressions like
    ``symengine.pi`` or ``2*pi/3`` round-trip through ``complex`` to a
    Python ``float``/``int``/``complex``.
    """
    if isinstance(p, bool):
        return int(p), True
    if isinstance(p, numbers.Integral):
        return int(p), True
    if isinstance(p, numbers.Real):
        return float(p), True
    if isinstance(p, numbers.Complex):
        return complex(p), True
    if isinstance(p, np.generic):
        return p.item(), True
    if isinstance(p, (se.Basic, sp.Basic)):
        if getattr(p, "free_symbols", None):
            return p, False
        try:
            c = complex(p)
        except (TypeError, ValueError):
            return p, False
        if c.imag == 0.0:
            r = c.real
            if r.is_integer():
                return int(r), True
            return r, True
        return c, True
    return p, False


_SE_ZERO = se.Integer(0)


def _smart_evalf(x):
    """Evaluate a SymEngine expression to a float/complex, preserving
    symbolic terms and keeping integer zeros as integers.

    Also collapses ``a + 0.0*I`` → ``a`` (real) and normalizes IEEE-754
    signed zeros to ``0.0``. Together these two rules make the output of
    matrix inversion and other algebraic operations reproducible and
    clean, which the old ``sp.simplify`` used to achieve indirectly.
    """
    if getattr(x, "free_symbols", None):
        return x
    if isinstance(x, se.Integer) and int(x) == 0:
        return x
    try:
        v = x.evalf()
    except AttributeError:
        return x
    # Canonicalize numeric results: fold away exact-zero imaginary parts
    # and normalize -0.0 → 0.0. ``complex(v)`` handles RealDouble,
    # ComplexDouble, Integer, Rational, and the symengine Float families.
    try:
        c = complex(v)
    except (TypeError, ValueError):
        return v
    real = c.real + 0.0  # IEEE-754 quirk: this turns -0.0 into 0.0
    imag = c.imag + 0.0
    if imag == 0.0:
        return _SE_ZERO if real == 0.0 else se.RealDouble(real)
    # Keep an explicit real part so ``0.0 + b*I`` still reads naturally
    # for users comparing with the documented matrix forms. Build
    # ``I*b + a`` rather than ``a + I*b`` — SymEngine's evaluator otherwise
    # re-introduces a signed zero into the real part when ``b`` is
    # negative.
    return se.I * se.RealDouble(imag) + se.RealDouble(real)


def _as_se_matrix(m):
    """Normalize the output of a ``_matrix()`` override to ``se.Matrix``.

    Some overrides build their matrix as a ``numpy.ndarray`` (generalized
    gates doing ``np.kron``) or as a ``sympy.Matrix`` (e.g. ``Power``
    using ``sympy``'s matrix power). Convert both into SymEngine so the
    downstream ``applyfunc``/display path is uniform.
    """
    if isinstance(m, np.ndarray):
        return se.Matrix(m.tolist())
    if isinstance(m, sp.Matrix):
        return se.Matrix(m.tolist())
    return m


def _to_ndarray(m):
    """Convert a SymEngine (or already-NumPy) matrix to ``complex128``."""
    if isinstance(m, np.ndarray):
        return m.astype(np.complex128, copy=False)
    return np.asarray(m.tolist(), dtype=np.complex128)


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

    def supports_canonical_rewrite(self):
        """Only gate-like operators participate in canonical rewrite by default."""
        import mimiqcircuits as mc

        return isinstance(self, mc.Gate)

    def matrix(self):
        """Return the SymEngine matrix representation of the operator.

        Constant (parameter-free) operators are cached at class level, so
        repeated calls on, say, ``GateX()`` are O(1). Numeric parametric
        operators build a SymEngine matrix through :meth:`_matrix` and then
        evaluate each non-zero entry to a floating-point scalar. Genuinely
        symbolic operators return the raw SymEngine expression untouched —
        the old ``sympy.simplify`` step is deliberately gone; it was the
        main hot spot and produced no information the caller couldn't
        recover with their own simplification pass.

        Returns:
            symengine.Matrix: The matrix representation.
        """
        # Wrappers (Control, Inverse, Power, Parallel, RescaledGate) have
        # no _parnames of their own but carry state through the inner op,
        # so we can only cache when both are empty.
        if not self._parnames and not self.iswrapper():
            cls = type(self)
            cached = cls.__dict__.get("_MATRIX_CACHE")
            if cached is None:
                cached = _as_se_matrix(self._matrix()).applyfunc(_smart_evalf)
                setattr(cls, "_MATRIX_CACHE", cached)
            return cached

        # applyfunc(_smart_evalf) is cheap on symbolic entries (a
        # free_symbols check) and converts any literal numeric entry
        # (``1`` → ``1.0``) the same way the old sympy pipeline did —
        # without paying for sympy.
        return _as_se_matrix(self._matrix()).applyfunc(_smart_evalf)

    def _matrix_numeric(self, *params):
        """Build the matrix for fully-numeric parameters as a NumPy array.

        Default implementation: build the SymEngine matrix through
        :meth:`_matrix` (applied to a freshly-constructed instance with the
        unwrapped parameters) and cast to ``complex128``. Gates where this
        is still too slow should override this to construct the matrix with
        native ``math``/``cmath`` operations — see
        :mod:`mimiqcircuits.numerics`.
        """
        if params:
            return _to_ndarray(type(self)(*params)._matrix())
        return _to_ndarray(self._matrix())

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
        """Numeric matrix representation as a ``numpy.ndarray[complex128]``.

        This is the Python analogue of Julia's
        ``unwrappedmatrix(::AbstractOperator)``: simulators and anything
        else that wants raw numbers (e.g. TensorWeaver) should call this
        instead of :meth:`matrix`. It bypasses SymEngine entirely for gates
        that override :meth:`_matrix_numeric` — see
        :mod:`mimiqcircuits.numerics` — and otherwise falls back to
        building the SymEngine matrix and casting it.

        Constant gates cache the result at class level.

        Raises:
            UnexpectedSymbolics: If any parameter is still symbolic.

        Returns:
            numpy.ndarray: ``complex128`` matrix.
        """
        if not self._parnames and not self.iswrapper():
            cls = type(self)
            cached = cls.__dict__.get("_UNWRAPPED_CACHE")
            if cached is None:
                cached = self._matrix_numeric()
                cached.flags.writeable = False
                setattr(cls, "_UNWRAPPED_CACHE", cached)
            return cached

        unwrapped = []
        for p in self.getparams():
            v, ok = _try_unwrap(p)
            if not ok:
                raise UnexpectedSymbolics(self)
            unwrapped.append(v)
        return self._matrix_numeric(*unwrapped)

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
