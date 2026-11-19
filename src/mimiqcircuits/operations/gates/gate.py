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
"""Abstract Gate base class."""

from mimiqcircuits.operations.operator import AbstractOperator
import mimiqcircuits.lazy as lz

from mimiqcircuits.operations.repeat import Repeat
from mimiqcircuits.operations.repeat import repeat


class Gate(AbstractOperator):
    """Base class for unitary gates.

    A gate acts on qubits only, has a unitary matrix, and can be inverted,
    raised to a power, controlled, repeated and run in parallel. Subclasses
    provide the matrix and the number of qubits.

    :meth:`power`, :meth:`control`, :meth:`parallel` and :meth:`repeat`
    called without their count return a :class:`LazyExpr` that takes it
    later.

    Examples:
        >>> from mimiqcircuits import *
        >>> GateX().control(2)
        C₂X
        >>> GateX().power(0.5)
        SX
        >>> GateH().parallel(3)
        ⨷ ³ H
        >>> # without the count, a lazy expression is returned
        >>> GateX().control()
        lazy control(?, X)
    """

    _name = None

    _num_qubits = None

    _num_bits = 0
    _num_cregs = 0
    _cregsizes = ()

    def _matrix(self):
        raise ValueError(
            "This is a parrent class of simple gates and does not have matrix"
        )

    def iswrapper(self):
        """Return ``False``: a plain gate does not wrap another operation."""
        return False

    def rescale(self, scale):
        """Return a :class:`RescaledGate` wrapping this gate with factor ``scale``."""
        from mimiqcircuits.operations.rescaledgates import RescaledGate
        return RescaledGate(self, scale)

    def rescale_inplace(self, scale):
        """Raise, as a gate cannot be rescaled in place. Use :meth:`rescale`."""
        raise TypeError("Cannot rescale a gate in place. Use `rescale` instead.")

    def inverse(self):
        """Return the inverse of the gate.

        Gates with a known inverse return it directly (for example
        ``GateS().inverse()`` is ``GateSDG()``); otherwise the gate is wrapped
        in :class:`Inverse`.
        """
        from mimiqcircuits.operations.inverse import Inverse
        return Inverse(self)

    def _power(self, pwr):
        from mimiqcircuits.operations.power import Power
        return Power(self, pwr)

    def power(self, *args):
        """Return the gate raised to a power.

        Gates with a known power return it directly (for example
        ``GateZ().power(0.5)`` is ``GateS()``, and ``GateRX(a).power(p)`` is
        ``GateRX(p * a)``); otherwise the gate is wrapped in :class:`Power`.

        Args:
            *args: The exponent. Without it, return a :class:`LazyExpr` that
                takes the exponent later.

        Raises:
            ValueError: If more than one argument is given.
            TypeError: If the exponent is not a real number.
        """
        if len(args) == 0:
            return lz.power(self)
        elif len(args) == 1:
            # Imported here: `power` imports this module.
            from mimiqcircuits.operations.power import _normalize_exponent

            return self._power(_normalize_exponent(args[0]))
        else:
            raise ValueError("Invalid number of arguments.")

    def __pow__(self, pwr):
        return self.power(pwr)

    def _control(self, num_controls):
        from mimiqcircuits.operations.control import Control
        return Control(num_controls, self)

    def control(self, *args):
        """Return the gate controlled by ``num_controls`` qubits.

        Gates with a named controlled form return it directly (for example
        ``GateX().control(1)`` is ``GateCX()``); otherwise the gate is
        wrapped in :class:`Control`. The controls are the first qubits.

        Args:
            *args: The number of controls. Without it, return a
                :class:`LazyExpr` that takes the number later.

        Raises:
            ValueError: If more than one argument is given.
        """
        if len(args) == 0:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return self._control(num_controls)
        else:
            raise ValueError("Invalid number of arguments.")

    def _parallel(self, num_repeats):
        from mimiqcircuits.operations.parallel import Parallel
        return Parallel(num_repeats, self)

    def parallel(self, *args):
        """Return a :class:`Parallel` applying the gate to separate qubit groups.

        Args:
            *args: The number of repetitions. Without it, return a
                :class:`LazyExpr` that takes the number later.

        Raises:
            ValueError: If more than one argument is given.
        """
        if len(args) == 0:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return self._parallel(num_repeats)
        else:
            raise ValueError("Invalid number of arguments.")

    def _repeat(self, num_repeats):
        return Repeat(num_repeats, self)

    def repeat(self, *args):
        """Return a :class:`Repeat` applying the gate several times in sequence on the same qubits.

        Args:
            *args: The number of repetitions. Without it, return a
                :class:`LazyExpr` that takes the number later.

        Raises:
            ValueError: If more than one argument is given.
        """
        if len(args) == 0:
            return repeat(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return self._repeat(num_repeats)
        else:
            raise ValueError("Invalid number of arguments.")

    def __str__(self):
        pars = ""
        if len(self.parnames) != 0:
            pars += "("
            pars += ", ".join([f"{getattr(self, pn)}" for pn in self.parnames])
            pars += ")"
        return self.name + pars

    def __repr__(self):
        return str(self)

    def evaluate(self, d):
        """Substitute symbols in the gate parameters and return the result.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            Gate: ``self`` if nothing changed, otherwise a new gate of the
            same type with the substituted parameters.
        """
        if len(self.parnames) == 0:
            return self

        params = self.getparams()
        newparams = [
            p if isinstance(p, (int, float)) else p.subs(d) for p in params
        ]

        if newparams == params:
            return self

        return type(self)(*newparams)

    def gettypekey(self):
        """Return the key used to look up registered decompositions: the gate class."""
        return type(self)

    @staticmethod
    def isunitary():
        """Return ``True``: gates are unitary."""
        return True


__all__ = ["Gate"]
