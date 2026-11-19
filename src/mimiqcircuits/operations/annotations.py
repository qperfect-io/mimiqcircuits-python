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
"""Circuit annotations (Detector, QubitCoordinates, etc.)."""

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.lazy import LazyExpr, LazyArg


class AbstractAnnotation(Operation):
    r"""Base class for circuit annotations.

    Annotations carry metadata and do not change the quantum state or the
    classical bits. They describe the circuit for error-correction tools,
    visualization or export, for example to Stim. Subclasses include
    :class:`Detector`, :class:`ObservableInclude`,
    :class:`QubitCoordinates`, :class:`ShiftCoordinates`, :class:`Tick`,
    :class:`Lost` and :class:`Reloaded`.

    Every annotation returns its numeric data from ``get_notes()``.
    """

    def iswrapper(self):
        return False

    def isunitary(self):
        return True


class Detector(AbstractAnnotation):
    r"""Annotation marking a set of measurement bits whose parity is deterministic.

    A detector covers ``N`` classical bits holding measurement results. In a
    noiseless run their parity is always the same, so a change in parity
    signals an error. Detectors are used by error-correction tools and in
    Stim export; they do not change the circuit's state.

    The notes are optional numbers attached to the detector, usually its
    coordinates.

    The constructor accepts:

    - ``Detector()``: one bit, no notes.
    - ``Detector(N)``: ``N`` bits, no notes.
    - ``Detector(N, notes)``: ``N`` bits with a list of notes.
    - ``Detector(N, n1, n2, ...)``: ``N`` bits with integer notes.
    - ``Detector(notes)``: a lazy expression that takes ``N`` later.

    Raises:
        ValueError: If the arguments match none of the forms above, or ``N``
            is less than 1.

    See Also:
        :class:`ObservableInclude`, :class:`QubitCoordinates`,
        :class:`ShiftCoordinates`, :class:`Tick`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> # measure two qubits
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> c.push(Measure(), 1, 1)
        2-qubit, 2-bit circuit with 2 instructions:
        ├── M @ q[0], c[0]
        └── M @ q[1], c[1]
        <BLANKLINE>
        >>> # declare that the parity of bits 0 and 1 is deterministic,
        >>> # with coordinates (1.0, 0.5) as notes
        >>> c.push(Detector(2, [1.0, 0.5]), 0, 1)
        2-qubit, 2-bit circuit with 3 instructions:
        ├── M @ q[0], c[0]
        ├── M @ q[1], c[1]
        └── Detector(1.0, 0.5) @ c[0:1]
        <BLANKLINE>

        Passing only the notes gives a lazy detector; call it with the
        number of bits:

        >>> op = Detector([0.1, 0.9])
        >>> op
        lazy Detector(?, [0.1, 0.9])
        >>> op(1).get_notes()
        [0.1, 0.9]
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
            return LazyExpr(cls, LazyArg(), list(args))

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
        """Return ``"Detector"``."""
        return "Detector"

    def get_notes(self):
        """Return the notes attached to the annotation, as a list of floats."""
        return self.notes

    def iswrapper(self):
        return False

    def __str__(self):
        sep = "," if getattr(self, "_compact", False) else ", "
        notes_str = sep.join(map(str, self.get_notes()))
        return f"{self.opname()}({notes_str})"


class QubitCoordinates(AbstractAnnotation):
    r"""Annotation giving the spatial coordinates of a qubit.

    Applied to one qubit. The coordinates do not affect simulation; they
    describe the physical layout for visualization and export.

    ``QubitCoordinates(coords)`` takes a list of numbers, stored as floats.
    ``QubitCoordinates()`` returns a lazy expression that takes the list
    later.

    Args:
        coordinates (list): The coordinates of the qubit.

    Raises:
        ValueError: If the argument is not a single list.

    See Also:
        :class:`Detector`, :class:`ObservableInclude`,
        :class:`ShiftCoordinates`, :class:`Tick`

    Examples:
        >>> from mimiqcircuits import *
        >>> QubitCoordinates([0.5, 0.75, 1.0])
        QubitCoordinates(0.5, 0.75, 1.0)
        >>> op = QubitCoordinates([0.5, 0.75, 1.0])
        >>> op.get_notes()
        [0.5, 0.75, 1.0]
        >>> # place qubit 0 at (0.2, 0.3)
        >>> c = Circuit()
        >>> c.push(QubitCoordinates([0.2, 0.3]), 0)
        1-qubit circuit with 1 instruction:
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

    # See Barrier.__getnewargs__: copies must not go through the lazy form.
    def __getnewargs__(self):
        return (list(self.coordinates),)

    def __init__(self, coordinates: list):
        self._num_qubits = 1
        self.coordinates = [float(coord) for coord in (coordinates or [])]
        self._qregsizes = [1]

    @staticmethod
    def opname():
        """Return ``"QubitCoordinates"``."""
        return "QubitCoordinates"

    def get_notes(self):
        """Return the coordinates of the annotation, as a list of floats."""
        return self.coordinates

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}({', '.join(map(str, self.get_notes()))})"


class ShiftCoordinates(AbstractAnnotation):
    r"""Annotation shifting the coordinates of later annotations.

    Takes no qubits or bits. The offsets are added to the coordinates of
    the qubit and detector annotations that follow it, and successive
    shifts accumulate. It does not affect simulation.

    The offsets can be given as separate numbers or as one list.
    ``ShiftCoordinates()`` returns a lazy expression that takes the list of
    offsets later.

    Args:
        *coordinates: The offsets, as numbers or a single list.

    Raises:
        ValueError: If the arguments are neither numbers nor a single list.

    See Also:
        :class:`Detector`, :class:`ObservableInclude`,
        :class:`QubitCoordinates`, :class:`Tick`

    Examples:
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
        circuit with 1 instruction:
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

    # See Barrier.__getnewargs__: copies must not go through the lazy form.
    def __getnewargs__(self):
        return (list(self.coordinates),)

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
        """Return ``"ShiftCoordinates"``."""
        return "ShiftCoordinates"

    def get_notes(self):
        """Return the coordinates of the annotation, as a list of floats."""
        return self.coordinates

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}({', '.join(map(str, self.get_notes()))})"


class ObservableInclude(AbstractAnnotation):
    r"""Annotation adding measurement bits to a logical observable.

    A logical observable is a set of measurement results whose combined
    parity is deterministic in a noiseless run and encodes the value of a
    logical qubit. ``ObservableInclude`` adds the ``N`` bits it is applied
    to into an observable. The notes are integers, usually the index of the
    observable. It does not change the circuit's state.

    The constructor accepts:

    - ``ObservableInclude()``: one bit, no notes.
    - ``ObservableInclude(N)``: ``N`` bits, no notes.
    - ``ObservableInclude(N, notes)``: ``N`` bits with a list of notes.
    - ``ObservableInclude(N, n1, n2, ...)``: ``N`` bits with integer notes.
    - ``ObservableInclude(notes)``: a lazy expression that takes ``N``
      later.

    Raises:
        ValueError: If the arguments match none of the forms above, or ``N``
            is less than 1.

    See Also:
        :class:`Detector`, :class:`QubitCoordinates`,
        :class:`ShiftCoordinates`, :class:`Tick`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 1 instruction:
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> # include bit 0 in observable 0
        >>> c.push(ObservableInclude(1, [0]), 0)
        1-qubit, 1-bit circuit with 2 instructions:
        ├── M @ q[0], c[0]
        └── ObservableInclude(0) @ c[0]
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
        """Return ``"ObservableInclude"``."""
        return "ObservableInclude"

    def get_notes(self):
        """Return the notes attached to the annotation, as a list of ints."""
        return self.notes

    def iswrapper(self):
        return False

    def __str__(self):
        sep = "," if getattr(self, "_compact", False) else ", "
        notes_str = sep.join(map(str, self.get_notes()))
        return f"{self.opname()}({notes_str})"


class Tick(AbstractAnnotation):
    """Annotation marking the boundary between two time steps.

    Takes no qubits or bits. It separates the circuit into layers for
    visualization, analysis and export, and does not affect simulation.

    See Also:
        :class:`Detector`, :class:`ObservableInclude`,
        :class:`QubitCoordinates`, :class:`ShiftCoordinates`

    Examples:
        >>> from mimiqcircuits import *
        >>> tick = Tick()
        >>> print(tick)
        Tick

        >>> c = Circuit()
        >>> c.push(Tick())
        circuit with 1 instruction:
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
        """Return ``"Tick"``."""
        return "Tick"

    def get_notes(self):
        """Return the notes of the annotation; this annotation has none, so the list is empty."""
        return []

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}"

    def __repr__(self):
        return self.__str__()


__all__ = ["Detector", "QubitCoordinates", "ShiftCoordinates", "ObservableInclude"]


class Lost(AbstractAnnotation):
    """Marker recording that a qubit was lost at this point.

    ``Lost`` has no effect on execution. It is emitted when a circuit's losses
    are resolved, so the resolved circuit still shows where losses landed, and
    it may be added by hand for documentation.

    See Also:
        :class:`Reloaded`, :class:`mimiqcircuits.Loss`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Lost(), 0)
        1-qubit circuit with 1 instruction:
        └── Lost @ q[0]
        <BLANKLINE>
    """

    _name = "Lost"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 1
    _num_zregs = 0
    _num_bits = 0
    _num_qubits = 1
    _qregsizes = [1]

    @staticmethod
    def opname():
        """Return ``"Lost"``."""
        return "Lost"

    def get_notes(self):
        """Return the notes of the annotation; this annotation has none, so the list is empty."""
        return []

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}"


class Reloaded(AbstractAnnotation):
    """Marker recording that a qubit was reloaded at this point.

    ``Reloaded`` has no effect on execution. It is emitted next to the
    :class:`Reset` that performs a reload, and may be added by hand for
    documentation.

    See Also:
        :class:`Lost`, :class:`mimiqcircuits.Reload`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Reloaded(), 0)
        1-qubit circuit with 1 instruction:
        └── Reloaded @ q[0]
        <BLANKLINE>
    """

    _name = "Reloaded"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 1
    _num_zregs = 0
    _num_bits = 0
    _num_qubits = 1
    _qregsizes = [1]

    @staticmethod
    def opname():
        """Return ``"Reloaded"``."""
        return "Reloaded"

    def get_notes(self):
        """Return the notes of the annotation; this annotation has none, so the list is empty."""
        return []

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}"


class AncillaAlloc(AbstractAnnotation):
    """A marker recording that a qubit is in ``|0>`` here and is used as a
    clean ancilla from here on.

    ``AncillaAlloc`` has no effect on execution. :func:`decompose` emits it,
    with :class:`AncillaFree`, around every use of a clean ancilla. The two
    markers bracket a window in which the qubit holds intermediate values and
    is returned to ``|0>``. Passes that keep the markers must keep them true:
    nothing outside the decomposition that placed them may act on the qubit
    inside the window.

    See Also:
        :class:`AncillaFree`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(AncillaAlloc(), 0)
        1-qubit circuit with 1 instruction:
        └── AncillaAlloc @ q[0]
        <BLANKLINE>
    """

    _name = "AncillaAlloc"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 1
    _num_zregs = 0
    _num_bits = 0
    _num_qubits = 1
    _qregsizes = [1]

    @staticmethod
    def opname():
        return "AncillaAlloc"

    def get_notes(self):
        """Return the notes of the annotation; this annotation has none, so the list is empty."""
        return []

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}"


class AncillaFree(AbstractAnnotation):
    """A marker recording that a clean ancilla is back in ``|0>`` here and
    is free for reuse.

    ``AncillaFree`` has no effect on execution. It closes the window opened
    by an :class:`AncillaAlloc` on the same qubit.

    See Also:
        :class:`AncillaAlloc`

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(AncillaFree(), 0)
        1-qubit circuit with 1 instruction:
        └── AncillaFree @ q[0]
        <BLANKLINE>
    """

    _name = "AncillaFree"
    _num_zvars = 0
    _num_cregs = 0
    _num_qregs = 1
    _num_zregs = 0
    _num_bits = 0
    _num_qubits = 1
    _qregsizes = [1]

    @staticmethod
    def opname():
        return "AncillaFree"

    def get_notes(self):
        """Return the notes of the annotation; this annotation has none, so the list is empty."""
        return []

    def iswrapper(self):
        return False

    def __str__(self):
        return f"{self.opname()}"
