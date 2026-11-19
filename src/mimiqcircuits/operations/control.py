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
"""Control operation."""

import numpy as np
import symengine as se

import mimiqcircuits.lazy as lz
import mimiqcircuits.operations.decompositions.control as ctrldecomp
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.printutils import print_wrapped_parens
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, Dict

if TYPE_CHECKING:
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.parallel import Parallel

_control_decomposition_registry = {}

# Registry for canonical Control subclasses: (num_controls, inner_gate_type) -> subclass
_control_canonical_types: Dict[Tuple[int, Type], Type] = {}


def register_control_decomposition(num_controls: int, gate_type):
    """Decorator that registers a custom decomposition for a controlled gate.

    The decorated function is used by :meth:`Control.decompose` (and circuit
    decomposition) for every ``Control`` with ``num_controls`` controls whose
    target gate has type key ``gate_type``. It is called as
    ``func(op, circ, qubits, bits, zvars)``, must push the decomposition of
    ``op`` on ``qubits`` to ``circ``, and must return ``circ``.

    Args:
        num_controls: Number of control qubits.
        gate_type: Type key of the target gate. For plain gates this is the
            gate class, for example ``GateH``; for wrapped gates it is the
            value of their ``gettypekey()``.

    Returns:
        The decorator. It registers the function and returns it unchanged.
    """

    def decorator(decomp_func):
        key = (num_controls, gate_type)
        _control_decomposition_registry[key] = decomp_func
        return decomp_func

    return decorator


def canonical_control(num_controls: int, inner_gate_type: Type):
    """Decorator that registers a named subclass of :class:`Control`.

    After registration, ``Control(num_controls, g)`` with ``g`` an instance
    of ``inner_gate_type`` builds an instance of the decorated class instead
    of a plain ``Control``. This is how named gates such as :class:`GateCX`
    (``Control(1, GateX())``) are defined, and it makes ``isinstance`` checks
    work whichever way the gate was built.

    Registering a class for a pair that already has one replaces it for the
    rest of the session.

    Args:
        num_controls: Number of control qubits.
        inner_gate_type: Class of the target gate.

    Returns:
        The decorator. It registers the class and returns it unchanged.

    Examples:
        >>> from mimiqcircuits import *
        >>> # GateCX is registered for one control on GateX, so building
        >>> # the generic form gives a GateCX
        >>> isinstance(Control(1, GateX()), GateCX)
        True
    """

    def decorator(cls: Type) -> Type:
        _control_canonical_types[(num_controls, inner_gate_type)] = cls
        return cls

    return decorator


class Control(Gate):
    r"""Gate controlled by one or more qubits.

    ``Control(n, gate)`` applies ``gate`` to the target qubits only when all
    ``n`` control qubits are in state :math:`|1\rangle`. The control qubits
    come first in the target list, followed by the qubits of ``gate``.

    The constructor takes the number of controls and either a gate instance
    or a gate class followed by its constructor arguments, so
    ``Control(2, GateRX, 0.5)`` equals ``Control(2, GateRX(0.5))``. A
    controlled ``Control`` is flattened: ``Control(1, Control(2, GateX()))``
    is ``Control(3, GateX())``.

    When a named subclass is registered for the combination (see
    :func:`canonical_control`), the constructor returns an instance of that
    subclass: ``Control(1, GateX())`` is a :class:`GateCX`.

    The ``control`` method of a gate is the usual way to build one, for
    example ``GateH().control(2)``.

    Args:
        num_controls: Number of control qubits, at least 1.
        operation: The gate to control, or a gate class.
        *args: Constructor arguments when ``operation`` is a class.
        **kwargs: Constructor keyword arguments when ``operation`` is a class.

    Raises:
        TypeError: If ``operation`` is not a gate or gate class, or acts on
            classical bits or z-variables.
        ValueError: If ``num_controls`` is less than 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> # the generic form returns the named gate when one exists
        >>> isinstance(Control(1, GateX()), GateCX)
        True
        >>> # a Toffoli-like gate with three controls: q[1:3] control, q[4] is the target
        >>> c = Circuit()
        >>> c.push(Control(3,GateX()),1,2,3,4)
        5-qubit circuit with 1 instruction:
        └── C₃X @ q[1:3], q[4]
        <BLANKLINE>
        >>> # the matrix is the identity except for the last block, which is X
        >>> Control(2, GateX()).matrix()
        [1.0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1.0, 0, 0, 0, 0, 0, 0]
        [0, 0, 1.0, 0, 0, 0, 0, 0]
        [0, 0, 0, 1.0, 0, 0, 0, 0]
        [0, 0, 0, 0, 1.0, 0, 0, 0]
        [0, 0, 0, 0, 0, 1.0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 1.0]
        [0, 0, 0, 0, 0, 0, 1.0, 0]
        <BLANKLINE>
    """

    _name = "Control"

    def __new__(cls, num_controls: int = None, operation: Union[Type[Gate], Gate] = None, *args, **kwargs):
        """Create a Control instance, returning canonical subclass if registered.

        If a canonical subclass is registered for (num_controls, type(operation)),
        an instance of that subclass is returned instead of a plain Control.

        Args:
            num_controls: Number of control qubits
            operation: Gate operation to control or gate class to instantiate
            *args: Arguments to pass to operation constructor if a class is provided
            **kwargs: Keyword arguments to pass to operation constructor

        Returns:
            Instance of Control or a registered canonical subclass
        """
        # Only intercept direct Control() calls, not subclass calls
        if cls is Control:
            if num_controls is None or operation is None:
                # Let __init__ handle the error
                return object.__new__(cls)

            # Resolve the operation to get its type
            if isinstance(operation, type) and issubclass(operation, Gate):
                inner_type = operation
            elif isinstance(operation, Gate):
                inner_type = type(operation)
            else:
                # Let __init__ handle the error
                return object.__new__(cls)

            # __init__ flattens nested controls, so the canonical subclass
            # must be looked up for the flattened operation.
            if isinstance(operation, Control):
                num_controls = num_controls + operation.num_controls
                inner_type = type(operation.op)

            # Check for canonical subclass
            key = (num_controls, inner_type)
            canonical_cls = _control_canonical_types.get(key)
            if canonical_cls is not None:
                return object.__new__(canonical_cls)

        return object.__new__(cls)

    def __init__(
        self, num_controls: int, operation: Union[Type[Gate], Gate], *args, **kwargs
    ):
        """Initialize a Control gate.

        Args:
            num_controls: Number of control qubits
            operation: Gate operation to control or gate class to instantiate
            *args: Arguments to pass to operation constructor if a class is provided
            **kwargs: Keyword arguments to pass to operation constructor if a class is provided

        Raises:
            TypeError: If operation is not a Gate object or Gate class
            ValueError: If num_controls is less than 1
        """
        # Instantiate the operation if a class was provided
        if isinstance(operation, type) and issubclass(operation, Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, Gate):
            op = operation
        else:
            raise TypeError("Operation must be a Gate object or Gate class.")

        if op.num_bits != 0:
            raise TypeError("Control operation cannot act on classical bits.")

        if op.num_zvars != 0:
            raise TypeError("Control operation cannot act on z-register variables.")

        if num_controls < 1:
            raise ValueError("Controlled operations must have at least one control.")

        super().__init__()

        # Handle nested Control operations by flattening them
        if isinstance(op, Control):
            self._num_controls = op.num_controls + num_controls
            self._op = op.op
            self._num_qubits = self._op.num_qubits + self._num_controls
            self._qregsizes = [self._num_controls] + self._op.qregsizes
        else:
            self._num_controls = num_controls
            self._op = op
            self._num_qubits = op.num_qubits + num_controls
            self._qregsizes = [num_controls] + op.qregsizes

        # Standard initialization for Gate class
        self._num_bits = 0
        self._num_zvars = 0
        self._num_cregs = 0
        self._num_zregs = 0
        self._num_qregs = len(self._qregsizes)

    def _matrix(self):
        """SymEngine matrix representation of the controlled gate."""
        target_dim = 2**self.op.num_qubits
        total_dim = 2 ** (self.op.num_qubits + self.num_controls)

        full = se.zeros(total_dim, total_dim)
        for i in range(total_dim - target_dim):
            full[i, i] = 1
        full[total_dim - target_dim :, total_dim - target_dim :] = self.op.matrix()
        return full

    def _matrix_numeric(self, *params):
        """NumPy fast path — combine the inner op's numeric matrix with an
        identity in the control subspace without touching SymEngine."""
        inner = self.op.unwrappedmatrix()
        target_dim = inner.shape[0]
        total_dim = 2**self.num_controls * target_dim
        out = np.eye(total_dim, dtype=np.complex128)
        out[total_dim - target_dim :, total_dim - target_dim :] = inner
        return out

    @property
    def num_controls(self) -> int:
        """Number of control qubits. Read-only."""
        return self._num_controls

    @property
    def num_targets(self) -> int:
        """Number of target qubits, that is the qubits of :attr:`op`."""
        return self.num_qubits - self.num_controls

    @property
    def op(self) -> Gate:
        """The controlled gate. Read-only."""
        return self._op

    def inverse(self) -> "Control":
        """Return the inverse, the same controls applied to the inverse gate.

        Returns:
            Control: ``Control(num_controls, op.inverse())``.
        """
        return Control(self.num_controls, self.op.inverse())

    def getparams(self) -> Any:
        """Return the parameters of the controlled gate.

        Returns:
            list: The value of ``op.getparams()``.
        """
        return self.op.getparams()

    def get_operation(self):
        """Return the controlled gate, :attr:`op`."""
        return self.op

    def control(self, *args) -> Union[Any, "Control"]:
        """Add more control qubits.

        Args:
            *args: The number of extra controls. When omitted, a lazy
                expression is returned that takes the number later.

        Returns:
            Control: The same gate with ``num_controls + n`` controls, or a
            lazy expression when called without arguments.

        Raises:
            ValueError: If more than one argument is given.

        Examples:
            >>> from mimiqcircuits import *
            >>> Control(2, GateH()).control(1)
            C₃H
        """
        if not args:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return Control(self.num_controls + num_controls, self.op)
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 1.")

    def _power(self, power: Any) -> "Control":
        """Return the same controls applied to ``op`` raised to ``power``."""
        return Control(self.num_controls, self.op.power(power))

    def power(self, *args) -> Union[Any, "Control"]:
        """Raise the controlled gate to a power.

        A controlled gate raised to ``p`` is the same controls applied to the
        gate raised to ``p``.

        Args:
            *args: The exponent. When omitted, a lazy expression is returned
                that takes the exponent later.

        Returns:
            Control: ``Control(num_controls, op.power(p))``, or a lazy
            expression when called without arguments.

        Raises:
            ValueError: If more than one argument is given.

        Examples:
            >>> from mimiqcircuits import *
            >>> Control(2, GateX()).power(0.5)
            C₂SX
        """
        if not args:
            return lz.power(self)
        elif len(args) == 1:
            return self._power(args[0])
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 1.")

    def parallel(self, *args) -> Union[Any, "Parallel"]:
        """Return copies of this gate applied in parallel on separate qubits.

        Args:
            *args: The number of copies. When omitted, a lazy expression is
                returned that takes the number later.

        Returns:
            Parallel: ``Parallel(n, self)``, or a lazy expression when called
            without arguments.

        Raises:
            ValueError: If more than one argument is given.
        """
        from mimiqcircuits.operations.parallel import Parallel
        if not args:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 1.")

    def __pow__(self, power: Any) -> "Control":
        """Return ``self.power(power)``."""
        return self.power(power)

    def iswrapper(self) -> bool:
        """Return ``True``: a ``Control`` wraps another gate."""
        return True

    def __str__(self) -> str:
        """Return ``C`` followed by the gate, with the number of controls as a
        subscript when there is more than one, for example ``C₂X``."""
        # Create subscript numbers for the control count
        controls_subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

        # Only show the number if more than 1 control
        ctext = ""
        if self.num_controls > 1:
            ctext = str(self.num_controls).translate(controls_subscript)

        return f"C{ctext}{print_wrapped_parens(self.op)}"

    def evaluate(self, d: Any) -> "Control":
        """Substitute values for the symbols in the controlled gate.

        Args:
            d: Map from symbols (or their names) to values.

        Returns:
            Control: The controlled gate with the symbols substituted, or
            ``self`` when it has no symbolic parameters.
        """
        if not self.is_symbolic():
            return self
        return self.op.evaluate(d).control(self.num_controls)

    def gettypekey(self) -> Tuple:
        """Return the key used to look up registered decompositions.

        Returns:
            tuple: ``(Control, num_controls, op.gettypekey())``.
        """
        return (Control, self.num_controls, self.op.gettypekey())

    def _decompose(
        self, circ: "Circuit", qubits: list, bits: list, zvars: list
    ) -> "Circuit":
        """Decompose this controlled operation into simpler gates.

        This method uses the decomposition registry if available, otherwise
        applies different decomposition strategies based on the operation.

        Args:
            circ: Circuit to add decomposed operations to
            qubits: Qubits to apply the operation to
            bits: Classical bits for the operation
            zvars: Variables for parametric gates

        Returns:
            mc.Circuit: Circuit with decomposed operations
        """
        from mimiqcircuits.circuit import Circuit
        # Check if there's a specialized decomposition in the registry
        key = self.gettypekey()[1:]
        if key in _control_decomposition_registry:
            return _control_decomposition_registry[key](self, circ, qubits, bits, zvars)

        # Split qubits into controls and targets
        controls = qubits[: self.num_controls]
        targets = qubits[self.num_controls :]

        # Handle simple cases with single control or multi-qubit targets
        if self.num_controls == 1 or self.num_targets != 1:
            # Decompose the target operation
            newcirc = self.op._decompose(Circuit(), targets, bits, zvars)

            # Apply controls to each instruction in the decomposition
            for inst in newcirc:
                inst_controls = list(controls)
                inst_targets = inst.get_qubits()
                circ.push(
                    Control(self.num_controls, inst.operation),
                    *inst_controls,
                    *inst_targets,
                )
            return circ
        else:
            # Use specialized decomposition for multi-control single-target gates
            return ctrldecomp.control_decompose(circ, self.op, controls, targets[0])

    def _decompose_ctx(self, circ, qubits, bits, zvars, ctx):
        # A multi-controlled X can use qubits outside its targets. In order of
        # preference:
        #
        # - clean ancillas and measurements: a ladder of ANDs computed with 4 T
        #   each and uncomputed by measurement with none (Jones 2013, Gidney
        #   2018), also for the Toffoli;
        # - clean ancillas: an AND ladder of relative-phase Toffolis (Maslov
        #   2016) computes the AND of the first controls into them, so 4 T per
        #   ancilla and step instead of a full Toffoli;
        # - borrowed qubits: one gives a linear count of Toffolis (Lemma 7.3 of
        #   Barenco et al. 1995), n-2 give 4(n-2) (Lemma 7.2).
        #
        # GateC3X is included: its ancilla-free form needs pi/8 phases, which
        # have no exact Clifford+T decomposition.
        from mimiqcircuits.decomposition.abstract import _borrowable
        from mimiqcircuits.operations.gates.standard.pauli import GateX

        n = self.num_controls
        if isinstance(self.op, GateX) and n >= 2 and ctx.can_measure():
            clean = ctx.take_clean(n - 1)
            if clean:
                return ctrldecomp.cnx_measured_ladder(
                    circ, list(qubits[:n]), qubits[n], clean, ctx.bit
                )
        if isinstance(self.op, GateX) and n >= 3:
            controls = list(qubits[:n])
            target = qubits[n]
            clean = ctx.take_clean(n - 2)
            if clean:
                return ctrldecomp.cnx_ladder(circ, controls, target, clean)
            borrowed = _borrowable(ctx, qubits, n - 2)
            if borrowed:
                return ctrldecomp.cx_decompose(circ, controls, target, borrowed)
        return self._decompose(circ, qubits, bits, zvars)


__all__ = ["Control", "canonical_control"]
