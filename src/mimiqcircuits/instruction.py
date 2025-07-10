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

from mimiqcircuits.operations.operation import Operation
import mimiqcircuits as mc
import copy
import numpy as np
from mimiqcircuits.matrices import reorder_qubits_matrix


def _allunique(lst):
    seen = set()
    for element in lst:
        if element in seen:
            return False
        seen.add(element)
    return True


class Instruction:
    """Initializes an instruction of a quantum circuit.

    Args:
        operation (Operation): The operation applied by the instruction.
        qubits (tuple of int): The qubits to apply the quantum operation to.
        bits (tuple of int): The classical bits to apply the quantum operation to.

    Raises:
        TypeError: If operation is not a subclass of Gate or qubits is not a tuple.
        ValueError: If qubits contains less than 1 or more than 2 elements.

    Examples:

        >>> from mimiqcircuits import *
        >>> Instruction(GateX(),(0,),())
        X @ q[0]
        >>> Instruction(Barrier(4),(0,1,2,3),())
        Barrier @ q[0,1,2,3]

    """

    _operation = None
    _qubits = None
    _bits = None
    _zvars = None

    def __init__(self, operation, qubits=None, bits=None, zvars=None):
        if qubits is None:
            qubits = tuple()

        if bits is None:
            bits = tuple()

        if zvars is None:
            zvars = tuple()

        if not isinstance(qubits, tuple):
            raise TypeError(
                f"Target qubits should be given in a tuple of integers. Given {qubits} of type {type(qubits)}."
            )

        if not isinstance(bits, tuple):
            raise TypeError(
                f"Target bits should be given in a tuple of integers. Given {bits} of type {type(bits)}."
            )

        if not isinstance(zvars, tuple):
            raise TypeError(
                f"Target z-variables should be given in a tuple of integers. Given {zvars} of type {type(zvars)}."
            )

        if not isinstance(operation, Operation):
            raise TypeError(
                f"Operation must be a subclass of Operation. Given {operation} of type f{type(operation)}"
            )

        if not _allunique(qubits):
            raise ValueError("Duplicated qubit target in instruction")

        if not _allunique(bits):
            raise ValueError("Duplicated classical bit target in instruction")

        if not _allunique(zvars):
            raise ValueError("Duplicated z-variables target in instruction")

        for qi in qubits:
            if qi < 0:
                raise ValueError("Qubit target index cannot be negative")

        for bi in bits:
            if bi < 0:
                raise ValueError("Bit target index cannot be negative")

        for z in zvars:
            if z < 0:
                raise ValueError("Z-variable target index cannot be negative")

        if len(qubits) != operation.num_qubits:
            raise ValueError(
                f"Wrong number of target qubits for operation {operation} wanted  {operation.num_qubits}, given {len(qubits)}"
            )

        if len(bits) != operation.num_bits:
            raise ValueError(
                f"Wrong number of target bits for operation {operation} wanted  {operation.num_bits}, given {len(bits)}"
            )

        if len(zvars) != operation.num_zvars:
            raise ValueError(
                f"Wrong number of z-targets for operation {operation} wanted  {operation.num_zvars}, given {len(zvars)}"
            )

        self._operation = operation
        self._qubits = qubits
        self._bits = bits
        self._zvars = zvars

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, _):
        raise AttributeError("operation is a read-only attribute")

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, _):
        raise AttributeError("qubits is a read-only attribute")

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, _):
        raise AttributeError("bits is a read-only attribute")

    @property
    def zvars(self):
        return self._zvars

    @zvars.setter
    def zvars(self, _):
        raise AttributeError("zvars is a read-only attribute")

    def __repr__(self):
        return str(self)

    def get_operation(self):
        return self._operation

    def get_qubits(self):
        return self._qubits

    def get_bits(self):
        return self._bits

    def get_zvars(self):
        return self._zvars

    def num_qubits(self):
        return len(self._qubits)

    def num_bits(self):
        return len(self._bits)

    def num_zvars(self):
        return len(self._zvars)

    def get_operation(self):
        return self.operation

    def getparams(self):
        return self.operation.getparams()

    def listvars(self):
        return self.operation.listvars()

    def asciiwidth(self):
        return self.operation.asciiwidth(self._qubits, self._bits)

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return False
        return (
            (self.operation == other.operation)
            and (self.qubits == other.qubits)
            and (self.bits == other.bits)
            and (self.zvars == other.zvars)
        )

    def inverse(self):
        return Instruction(self.operation.inverse(), self.qubits, self.bits)

    def copy(self):
        """Creates a shallow copy of the instruction.
            To create a full copy use deepcopy() instead.

        Returns:
            Instruction: A new Instruction object containing references to the same attributes as the original circuit
        """
        return copy.copy(self)

    def deepcopy(self):
        """Creates a copy of the object and for all its attributes

        Returns:
            Instruction: A new Instruction object fully identical the original circuit
        """
        return copy.deepcopy(self)

    def _decompose(self, circ):
        return self.operation._decompose(circ, self.qubits, self.bits, self.zvars)

    def decompose(self):
        return self._decompose(mc.Circuit())

    def evaluate(self, d):
        return Instruction(
            self.operation.evaluate(d), self.qubits, self.bits, self.zvars
        )

    def matrix(self, nq: int = None):
        """
        Return the matrix of this instruction, expanded and reordered to act on `nq` qubits if specified.
        """
        op = self.get_operation()
        qubits = self.get_qubits()

        if nq is None:
            if len(qubits) == 0:
                raise ValueError("Cannot infer nq: instruction has no qubit targets.")
            nq = max(qubits) + 1

        M = op.matrix()
        if hasattr(op, "numparams") and op.numparams == 0:
            return reorder_qubits_matrix(M.copy(), qubits, nq)
        return reorder_qubits_matrix(M, qubits, nq)

    def __str__(self):
        compact = False
        if hasattr(self.operation, "format_with_targets"):
            # Custom rendering — do not append @ targets
            return self.operation.format_with_targets(
                self.qubits, self.bits, self.zvars
            )

        op = str(self.operation)
        nq = len(self.qubits)
        nb = len(self.bits)
        nz = len(self.zvars)
        space = "" if compact else " "
        targets = ""
        if nq != 0 or nb != 0 or nz != 0:
            targets = f"{space}@{space}"

            if nq != 0:
                q_partition = _partition(
                    self.get_qubits(), np.cumsum(self.operation.qregsizes)
                )
                q_targets = f",{space}".join(
                    f"q{_string_with_square(_find_unit_range(x), ',')}"
                    for x in q_partition
                )
                targets += f",".join(q_targets.split(","))

            if nb != 0:
                if nq != 0:  # If qubits exist, add a separator before classical bits
                    targets += f", "
                c_partition = _partition(
                    self.get_bits(), np.cumsum(self.operation.cregsizes)
                )
                c_targets = "".join(  # Concatenate classical bits without extra commas
                    f"c{_string_with_square(x, ',')}" for x in c_partition
                )
                targets += c_targets

            if nz != 0:
                if (
                    nq != 0 or nb != 0
                ):  # Add a separator before z-vars only if qubits or bits exist
                    targets += f", "
                z_partition = _partition(
                    self.get_zvars(), np.cumsum(self.operation.zregsizes)
                )
                z_targets = "".join(  # Concatenate z-vars without extra commas
                    f"z{_string_with_square(x, ',')}" for x in z_partition
                )
                targets += z_targets

        return f"{op}{targets}"


def _partition(arr, indices):
    vec = list(arr)
    partitions = [vec[: indices[0]]]

    for i in range(1, len(indices)):
        partitions.append(vec[indices[i - 1] : indices[i]])

    return partitions


def _string_with_square(arr, sep):
    return (
        "["
        + sep.join(
            map(lambda e: sep.join(map(str, e)) if isinstance(e, list) else str(e), arr)
        )
        + "]"
    )


def _find_unit_range(arr):
    if len(arr) < 2:
        return arr

    narr = []
    rangestart = arr[0]
    rangestop = arr[0]

    for v in arr[1:]:
        if v == rangestop + 1:
            rangestop = v
        elif rangestart == rangestop:
            narr.append(rangestart)
            rangestart = v
            rangestop = v
        else:
            narr.append(list(range(rangestart, rangestop + 1)))
            rangestart = v
            rangestop = v

    if rangestart == rangestop:
        narr.append(rangestart)
    else:
        narr.append(list(range(rangestart, rangestop + 1)))

    return narr


__all__ = ["Instruction"]
