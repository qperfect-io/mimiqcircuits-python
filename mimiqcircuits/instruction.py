#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
    """
    _operation = None
    _qubits = None
    _bits = None

    def __init__(self, operation, qubits=None, bits=None):

        if qubits is None:
            qubits = tuple()

        if bits is None:
            bits = tuple()

        if (not isinstance(qubits, tuple)):
            raise TypeError(
                f"Target qubits should be given in a tuple of integers. Given {qubits} of type {type(qubits)}.")

        if (not isinstance(bits, tuple)):
            raise TypeError(
                f"Target bits should be given in a tuple of integers. Given {bits} of type {type(bits)}.")

        if not isinstance(operation, Operation):
            raise TypeError(
                f"Operation must be a subclass of Operation. Given {operation} of type f{type(operation)}")

        if not _allunique(qubits):
            raise ValueError("Duplicated qubit target in instruction")

        if not _allunique(bits):
            raise ValueError("Duplicated classical bit target in instruction")

        for qi in qubits:
            if qi < 0:
                raise ValueError("Qubit target index cannot be negative")

        for bi in bits:
            if bi < 0:
                raise ValueError("Bit target index cannot be negative")

        if len(qubits) != operation.num_qubits:
            raise ValueError(
                f"Wrong number of target qubits for operation {operation} wanted  {operation.num_qubits}, given {len(qubits)}")

        if len(bits) != operation.num_bits:
            raise ValueError(
                f"Wrong number of target bits for operation {operation} wanted  {operation.num_bits}, given {len(bits)}")

        self._operation = operation
        self._qubits = qubits
        self._bits = bits

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

    def __str__(self):
        base = f'{self.operation}'
        qtargets = ', '.join(map(lambda q: f'q{q}', self.qubits))
        ctargets = ', '.join(map(lambda q: f'c{q}', self.bits))
        targets = ', '.join(filter(lambda x: x != '', [qtargets, ctargets]))
        return base + ' @ ' + targets

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return False
        return (self.operation == other.operation) and (self.qubits == other.qubits) and (self.bits == other.bits)

    def inverse(self):
        return Instruction(self.operation.inverse(), self.qubits, self.bits)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def _decompose(self, circ):
        return self.operation._decompose(circ, self.qubits, self.bits)

    def decompose(self):
        return self._decompose(mc.Circuit())

    def evaluate(self, d):
        return Instruction(self.operation.evaluate(d), self.qubits, self.bits)
