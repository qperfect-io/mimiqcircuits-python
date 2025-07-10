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

import mimiqcircuits as mc
from mimiqcircuits.lazy import LazyExpr, LazyArg


class AbstractAnnotation(mc.Operation):
    r"""AbstractAnnotation
    ----------------------

    Abstract base class for annotations in quantum circuits.

    This class is used as a base for defining various annotation types, such as `Detector`, `QubitCoordinates`,
    `ShiftCoordinates`,`Tick`, `ObservableInclude`, which provide metadata or structural information for quantum circuits.
    """

    pass


class Detector(AbstractAnnotation):
    r"""Detector operation.
    ------------------------

    An annotation class representing a detector in a quantum circuit.

    The `Detector` operation monitors the parity of measurement results over `N` classical bits, where the parity should remain deterministic under ideal, noiseless execution.
    The Detector ensures that the combined parity (even or odd) of a set of measurement results is consistent.
    If noise or errors disrupt the circuit, the Detector can identify this through unexpected changes in parity, signaling potential measurement errors.
    This functionality helps in error detection by revealing inconsistencies caused by disturbances.

    See Also:
        :class:`QubitCoordinates`
        :class:`ShiftCoordinates`
        :class:`ObservableInclude`
        :class:`Tick`

    Examples:

        Adding Detector operation to a Circuit (applied to classical bits):

        >>> from mimiqcircuits import *
        >>> op = Detector([0.1,0.9])
        >>> op
        lazy Detector(?, [0.1, 0.9])

        Fill the Lazy argument by calling the number of bits

        >>> op_filled = op(1)
        >>> op_filled
        Detector(0.1, 0.9)
        >>> op_filled.num_bits
        1

        Getting the nots

        >>> op_filled.get_notes()
        [0.1, 0.9]

        Define a new Operation

        >>> op = Detector(1, [0.5, 1.0])
        >>> op.get_notes()
        [0.5, 1.0]

        Add to the Circuit

        >>> c = Circuit()
        >>> c.push(Detector(1), 0)
        1-bit circuit with 1 instructions:
        └── Detector() @ c[0]
        <BLANKLINE>
        >>> c.push(Detector(1, [0.5, 1.0]), 1)
        2-bit circuit with 2 instructions:
        ├── Detector() @ c[0]
        └── Detector(0.5, 1.0) @ c[1]
        <BLANKLINE>
    """

    _name = "Detector"
    _num_zvars = 0
    _num_qubits = 0
    _num_cregs = 1
    _num_qregs = 0
    _num_zregs = 0

    def __new__(cls, *args):
        if len(args) == 0:
            return object.__new__(cls)
        elif len(args) == 1 and isinstance(args[0], int):
            return object.__new__(cls)
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], list):
            return object.__new__(cls)
        elif len(args) == 1 and isinstance(args[0], list):
            return LazyExpr(cls, LazyArg(), args[0])
        elif len(args) > 1 and all(isinstance(arg, int) for arg in args):
            return object.__new__(cls)
        elif len(args) > 1 and all(isinstance(arg, float) for arg in args):
            return LazyExpr(cls, LazyArg(), args)

        else:
            raise ValueError("Incorrect Detector arguments")

    def __init__(self, *args):
        if len(args) == 0:
            N = 1
            notes = []
        elif len(args) == 1 and isinstance(args[0], int):
            N = args[0]
            notes = []
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], list):
            N, notes = args
        elif len(args) > 1 and all(isinstance(arg, int) for arg in args):
            N, notes = args[0], list(args[1:])
        else:
            raise ValueError("Incorrect Detector arguments")

        if not isinstance(N, int) or N <= 0:
            raise ValueError("Detectors should be applied to at least 1 classical bit.")

        self._num_bits = N
        self.notes = [float(note) for note in (notes or [])]
        self._cregsizes = [N]

    @staticmethod
    def opname():
        return "Detector"

    def get_notes(self):
        return self.notes

    def iswrapper(self):
        return False

    def __str__(self):
        sep = "," if getattr(self, "_compact", False) else ", "
        notes_str = sep.join(map(str, self.get_notes()))
        return f"{self.opname()}({notes_str})"


class QubitCoordinates(AbstractAnnotation):
    r"""QubitCoordinates operation for specifying qubit positions.

    An annotation class used to specify the spatial location of a qubit in a quantum circuit.
    Coordinates do not affect simulation results but are useful for visualizing and organizing qubit layouts within the circuit.

    See Also:
        :class:`Detector`
        :class:`ShiftCoordinates`
        :class:`ObservableInclude`
        :class:`Tick`


    Examples:

        Adding QubitCoordinates to a circuit:

        >>> from mimiqcircuits import *
        >>> QubitCoordinates([0.5, 0.75, 1.0])
        QubitCoordinates(0.5, 0.75, 1.0)
        >>> op = QubitCoordinates([0.5, 0.75, 1.0])
        >>> op.get_notes()
        [0.5, 0.75, 1.0]
        >>> c= Circuit()
        >>> c.push(QubitCoordinates([0.2, 0.3]), 0)
        1-qubit circuit with 1 instructions:
        └── QubitCoordinates(0.2, 0.3) @ q[0]
        <BLANKLINE>
    """

    _name = "QubitCoordinates"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 1
    _num_zregs = 0
    _num_bits = 0

    def __new__(cls, *args):
        if len(args) == 0:
            return LazyExpr(cls, LazyArg(), *args)
        if len(args) == 1 and isinstance(args[0], list):
            return object.__new__(cls)
        else:
            raise ValueError("Should provide a list of float")

    def __init__(self, coordinates: list):
        self._num_qubits = 1
        self.coordinates = [float(coord) for coord in (coordinates or [])]
        self._qregsizes = [1]

    @staticmethod
    def opname():
        return "QubitCoordinates"

    def get_notes(self):
        return self.coordinates

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}({', '.join(map(str, self.get_notes()))})"


class ShiftCoordinates(AbstractAnnotation):
    r"""ShiftCoordinates operation for shifting the coordinates of qubits.

    An annotation class used to apply a shift to the spatial coordinates of subsequent qubit or detector annotations in a quantum circuit.
    `ShiftCoordinates` accumulates offsets that adjust the position of related circuit components, aiding in visualization without affecting the simulation.

    See Also:
        :class:`Detector`
        :class:`QubitCoordinates`
        :class:`ObservableInclude`
        :class:`Tick`


    Examples:

        Adding ShiftCoordinates to represent spatial shifts in a circuit:

        >>> from mimiqcircuits import *
        >>> ShiftCoordinates(0.1, 0.2, 0.3)
        ShiftCoordinates(0.1, 0.2, 0.3)
        >>> ShiftCoordinates([0.4, 0.5])
        ShiftCoordinates(0.4, 0.5)
        >>> ShiftCoordinates(0.4, 0.5)
        ShiftCoordinates(0.4, 0.5)
        >>> op = ShiftCoordinates(0.1, 0.9)
        >>> op.get_notes()
        [0.1, 0.9]
        >>> c = Circuit()
        >>> c.push(ShiftCoordinates(0.4, 0.2))
         circuit with 1 instructions:
        └── ShiftCoordinates(0.4, 0.2)
        <BLANKLINE>
    """

    _name = "ShiftCoordinates"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 0
    _num_zregs = 0

    def __new__(cls, *args):
        if len(args) == 0:
            return LazyExpr(cls, LazyArg(), *args)
        elif len(args) == 1 and isinstance(args[0], list):
            return object.__new__(cls)

        elif all(isinstance(arg, (int, float)) for arg in args):
            return object.__new__(cls)
        else:
            raise ValueError("Should provide a list of float")

    def __init__(self, *coordinates):
        self._num_qubits = 0
        self._num_bits = 0
        self._num_zvars = 0
        # If a list is provided as a single argument, use it directly
        if len(coordinates) == 1 and isinstance(coordinates[0], list):
            self.coordinates = [float(coord) for coord in coordinates[0]]
        # Otherwise, assume multiple float arguments are provided
        else:
            self.coordinates = [float(coord) for coord in coordinates]

    @staticmethod
    def opname():
        return "ShiftCoordinates"

    def get_notes(self):
        return self.coordinates

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}({', '.join(map(str, self.get_notes()))})"


class ObservableInclude(AbstractAnnotation):
    r"""ObservableInclude operation for including observable index.

    An annotation class for adding measurement records to a specified logical observable within a quantum circuit.
    Observables are sets of measurements expected to produce a deterministic result, used to track specific logical qubit states across operations.

    The ObservableInclude class tags a group of measurement records as a logical observable,
    representing a consistent, predictable result under noiseless conditions.
    This grouping allows for tracking the state of logical qubits across circuit operations, which is crucial for error correction.
    Logical observables monitor encoded qubit states by combining multiple measurements, providing robustness against noise
    and helping to identify any deviations that indicate potential errors.

    See Also:
        :class:`Detector`
        :class:`QubitCoordinates`
        :class:`ShiftCoordinates`
        :class:`Tick`

    Examples:

        Adding ObservableInclude to a circuit:

        >>> from mimiqcircuits import *
        >>> ObservableInclude(2)
        ObservableInclude()
        >>> c= Circuit()
        >>> c.push(ObservableInclude(1), 0)
        1-bit circuit with 1 instructions:
        └── ObservableInclude() @ c[0]
        <BLANKLINE>
    """

    _name = "ObservableInclude"
    _num_zvars = 0
    _num_qubits = 0
    _num_cregs = 1
    _num_qregs = 0
    _num_zregs = 0

    def __new__(cls, *args):
        if len(args) == 0:
            return object.__new__(cls)
        elif len(args) == 1 and isinstance(args[0], int):
            return object.__new__(cls)
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], list):
            return object.__new__(cls)
        elif len(args) == 1 and isinstance(args[0], list):
            return LazyExpr(cls, LazyArg(), args[0])
        elif len(args) > 1 and all(isinstance(arg, int) for arg in args):
            return object.__new__(cls)
        else:
            raise ValueError("Incorrect ObservableInclude arguments")

    def __init__(self, *args):
        if len(args) == 0:
            N = 1
            notes = []
        elif len(args) == 1 and isinstance(args[0], int):
            N = args[0]
            notes = []
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], list):
            N, notes = args
        elif len(args) > 1 and all(isinstance(arg, int) for arg in args):
            N, notes = args[0], list(args[1:])
        else:
            raise ValueError("Incorrect ObservableInclude arguments")

        if not isinstance(N, int) or N <= 0:
            raise ValueError(
                "ObservableIncludes should be applied to at least 1 classical bit."
            )

        self._num_bits = N
        self.notes = [int(note) for note in notes]
        self._cregsizes = [N]

    @staticmethod
    def opname():
        return "ObservableInclude"

    def get_notes(self):
        return self.notes

    def iswrapper(self):
        return False

    def __str__(self):
        sep = "," if getattr(self, "_compact", False) else ", "
        notes_str = sep.join(map(str, self.get_notes()))
        return f"{self.opname()}({notes_str})"


class Tick(AbstractAnnotation):
    """
    An annotation class representing a timing marker or layer boundary in a quantum circuit.
    `Tick` does not affect simulation but provides structure by separating operations into distinct time steps,
    which is useful for visualization and analysis.

    See Also:
        :class:`Detector`
        :class:`QubitCoordinates`
        :class:`ShiftCoordinates`
        :class:`ObservableInclude`

    Examples:
        >>> from mimiqcircuits import *
        >>> tick = Tick()
        >>> print(tick)
        Tick

        >>> c = Circuit()
        >>> c.push(tick)
         circuit with 1 instructions:
        └── Tick
        <BLANKLINE>
    """

    _name = "Tick"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 0
    _num_zregs = 0
    _num_bits = 0
    _num_qubits = 0

    @staticmethod
    def opname():
        return "Tick"

    def get_notes(self):
        return []

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}"

    def __repr__(self):
        return self.__str__()


__all__ = ["Detector", "QubitCoordinates", "ShiftCoordinates", "ObservableInclude"]
