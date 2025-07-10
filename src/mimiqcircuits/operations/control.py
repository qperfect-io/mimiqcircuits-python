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

import symengine as se
import sympy as sp

import mimiqcircuits as mc
import mimiqcircuits.lazy as lz
import mimiqcircuits.operations.decompositions.control as ctrldecomp
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.printutils import print_wrapped_parens
from typing import Union, Type, Tuple, Any

_control_decomposition_registry = {}


def register_control_decomposition(num_controls: int, gate_type):
    """Decorator to register a decomposition function for a specific gate type.

    Args:
        num_controls: Number of control qubits
        gate_type: Type of the gate to decompose

    Returns:
        Decorator function that registers the decomposition
    """

    def decorator(decomp_func):
        key = (num_controls, gate_type)
        _control_decomposition_registry[key] = decomp_func
        return decomp_func

    return decorator


class Control(Gate):
    """Control operation that applies multi-control gates to a Circuit.

    A Control is a special operation that wraps another gate with multiple
    control qubits.

    Attributes:
        num_controls: Number of control qubits
        op: The target gate operation

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Control(3,GateX()),1,2,3,4)
        5-qubit circuit with 1 instructions:
        └── C₃X @ q[1,2,3], q[4]
        <BLANKLINE>
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
        if isinstance(operation, type) and issubclass(operation, mc.Gate):
            op = operation(*args, **kwargs)
        elif isinstance(operation, mc.Gate):
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
        """Compute the matrix representation of the controlled gate.

        Returns:
            symengine.Matrix: Matrix representation of the controlled gate
        """
        target_dim = 2**self.op.num_qubits
        total_dim = 2 ** (self.op.num_qubits + self.num_controls)

        # Create identity matrix with target operation in bottom-right corner
        matrix = se.zeros(total_dim, total_dim)
        matrix[total_dim - target_dim :, total_dim - target_dim :] = self.op.matrix()

        # Set diagonal elements to 1 for the identity part
        for i in range(0, total_dim - target_dim):
            matrix[i, i] = 1

        return se.Matrix(sp.simplify(sp.Matrix(matrix).evalf()))

    @property
    def num_controls(self) -> int:
        """Number of control qubits."""
        return self._num_controls

    @property
    def num_targets(self) -> int:
        """Number of target qubits."""
        return self.num_qubits - self.num_controls

    @property
    def op(self) -> Gate:
        """Target gate operation."""
        return self._op

    def inverse(self) -> "Control":
        """Return the inverse of this controlled operation.

        Returns:
            Control: New Control operation with inverted target gate
        """
        return Control(self.num_controls, self.op.inverse())

    def getparams(self) -> Any:
        """Get parameters from the wrapped operation.

        Returns:
            Any: Parameters of the wrapped operation
        """
        return self.op.getparams()

    def get_operation(self):
        return self.op

    def control(self, *args) -> Union[Any, "Control"]:
        """Create a new controlled operation with additional controls.

        Args:
            *args: Optional number of additional controls

        Returns:
            Union[lazy.LazyValue, Control]: Lazy computation or new Control operation

        Raises:
            ValueError: If invalid number of arguments
        """
        if not args:
            return lz.control(self)
        elif len(args) == 1:
            num_controls = args[0]
            return Control(self.num_controls + num_controls, self.op)
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 1.")

    def _power(self, power: Any) -> "Control":
        """Internal method to compute the power of this operation.

        Args:
            power: Power to raise the operation to

        Returns:
            Control: New Control operation with powered target gate
        """
        return Control(self.num_controls, self.op.power(power))

    def power(self, *args) -> Union[Any, "Control"]:
        """Create a new operation raised to a power.

        Args:
            *args: Optional power value

        Returns:
            Union[lazy.LazyValue, Control]: Lazy computation or new Control operation

        Raises:
            ValueError: If invalid number of arguments
        """
        if not args:
            return lz.power(self)
        elif len(args) == 1:
            return self._power(args[0])
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 1.")

    def parallel(self, *args) -> Union[Any, "mc.Parallel"]:
        """Create parallel copies of this operation.

        Args:
            *args: Optional number of parallel copies

        Returns:
            Union[lazy.LazyValue, Parallel]: Lazy computation or new Parallel operation

        Raises:
            ValueError: If invalid number of arguments
        """
        if not args:
            return lz.parallel(self)
        elif len(args) == 1:
            num_repeats = args[0]
            return mc.Parallel(num_repeats, self)
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 1.")

    def __pow__(self, power: Any) -> "Control":
        """Implement the power operator.

        Args:
            power: Power to raise the operation to

        Returns:
            Control: New Control operation with powered target gate
        """
        return self.power(power)

    def iswrapper(self) -> bool:
        """Check if this operation is a wrapper.

        Returns:
            bool: Always True for Control operations
        """
        return True

    def __str__(self) -> str:
        """Get string representation of the Control operation.

        Returns:
            str: String representation with subscript for number of controls
        """
        # Create subscript numbers for the control count
        controls_subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

        # Only show the number if more than 1 control
        ctext = ""
        if self.num_controls > 1:
            ctext = str(self.num_controls).translate(controls_subscript)

        return f"C{ctext}{print_wrapped_parens(self.op)}"

    def evaluate(self, d: Any) -> "Control":
        """Evaluate the operation with given parameters.

        Args:
            d: Parameters for evaluation

        Returns:
            Any: Evaluated operation with controls
        """
        return self.op.evaluate(d).control(self.num_controls)

    def gettypekey(self) -> Tuple:
        """Get a tuple that uniquely identifies this operation type.

        Returns:
            Tuple: Type key for the operation
        """
        return (Control, self.num_controls, self.op.gettypekey())

    def _decompose(
        self, circ: "mc.Circuit", qubits: list, bits: list, zvars: list
    ) -> "mc.Circuit":
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
            newcirc = self.op._decompose(mc.Circuit(), targets, bits, zvars)

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


__all__ = ["Control"]
