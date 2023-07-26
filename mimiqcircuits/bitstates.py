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

from bitarray import bitarray
from multimethod import multimethod
from mimiqcircuits import *
from mimiqcircuits.circuit import Circuit


def _helper_bitarr_to_int(arr):
    res = 0
    for bit in arr:
        res = (res << 1) | bit
    return res


# Function to convert a bitarray to an integer
def bitarr_to_int(arr):
    if not isinstance(arr, bitarray):
        raise TypeError("Input must be a bitarray.")
    num_bits = len(arr)
    if num_bits > 0:
        return _helper_bitarr_to_int(arr)
    else:
        raise ValueError("Input array is empty.")


# Function to convert an integer to a bitarray of given size (pad)
def int_to_bitarr(x, pad):
    if not isinstance(x, int):
        raise TypeError("Input must be an integer.")
    if x < 0:
        raise ValueError("Input integer must be non-negative.")
    binary_str = bin(x)[2:].zfill(pad)

    return bitarray(binary_str)


class BitState:
    """
    :: class BitState

    Represents a quantum bit state with arbitrary number of qubits.

    Args:
        bits (Union[int, str, Circuit, bitarray]): The input to initialize the BitState. It can be
            an integer, a binary string, a Circuit, or a bitarray.
        init (Optional[List[int]]): A list of non-zero qubit indices to set in the BitState. Default is None.

    Examples:

        >>> bs = BitState(16, [1, 2, 3, 4])
        >>> print(bs)
        >>> 16-qubit BitState with 4 non-zero qubits:
            ├── |0111100000000000⟩
            └── non-zero qubits: [1, 2, 3, 4]

        Setting the value of a qubit
        >>> bs = BitState(12, [1, 2, 3, 5])
        >>> print(bs)
        >>> 12-qubit BitState with 4 non-zero qubits:
            ├── |011101000000⟩
            └── non-zero qubits: [1, 2, 3, 5]

        >>> bs[0] = 1
        >>> print(bs)
        >>> 12-qubit BitState with 3 non-zero qubits:
            ├── |111010000000⟩
            └── non-zero qubits: [0, 1, 2, 3, 5]

        Creating a 9-qubit Circuit and adding a GateX on qubit 8
        >>> c = Circuit()
        >>> c.push(GateX(), 8)
        >>> print(c)

        >>> 9-qubit circuit with 1 gates
            └── X @ q8

        >>> print(c.num_qubits())
            9

        Creating a 4-qubit BitState from the Circuit 'c' with specific non-zero qubits
        >>> bs = BitState(c, [1, 2, 3, 4])
        >>> print(bs)
        >>> 9-qubit BitState with 4 non-zero qubits:
            ├── |011110000⟩
            └── non-zero qubits: [1, 2, 3, 4]

        Converting the BitState to an integer
        >>> bs = BitState(16, [1,2,3,4])
        >>> print(bs)
        >>> 16-qubit BitState with 4 non-zero qubits:
            ├── |0111100000000000⟩
            └── non-zero qubits: [1, 2, 3, 4]
        >>> v = bs.tointeger()
        >>> print(v)
            30

        Creating a 16-qubit BitState from an integer value with big-endian representation
        >>> bs = BitState.fromint(16, 30, 'big')
        >>> print(bs)
        >>> 16-qubit BitState with 4 non-zero qubits:
            16-qubit BitState with 4 non-zero qubits:
            ├── |0111100000000000⟩
            └── non-zero qubits: [1, 2, 3, 4]

        Comparing two BitState objects for equality
        >>> bs1 = BitState('10101')
        >>> bs2 = BitState('10101')
        >>> print(bs1 == bs2) 
            True 

        Converting to bitarray
        >>> bs = BitState('111001')
        >>> bits = bs.bits
        >>> print(bits)
            bitarray('111001') 

        Converting the BitState to a binary string with default big-endianess representation
        >>> bs = BitState('1101001')
        >>> binary_str = bs.to01()
        >>> print(binary_str)
            1101001

        Converting the BitState to a binary string with little-endianess representation (by default endianess='big')
        >>> bs = BitState('1101001')
        >>> binary_str = bs.to01(endianess='little')
        >>> print(binary_str)
            1001011

        Converting the BitState to an index value
        >>> bs = BitState('1010')
        >>> index = bs.bitarr_to_index()
        >>> print(index)
            5
        Converting to BitState from string
        >>> bitstring = "11001"
        >>> bs_from_string = BitState.fromstring(bitstring)
        >>> print(bs_from_string)
        >>> 5-qubit BitState with 3 non-zero qubits:
            ├── |11001⟩
            └── non-zero qubits: [0, 1, 4]

        Example using fromnonzeros method also
        >>> num_qubits = 5
        >>> nonzeros = [1, 3]
        >>> bit_state_from_nonzeros = BitState.fromnonzeros(num_qubits, nonzeros)
        >>> print(bit_state_from_nonzeros)
        >>> 5-qubit BitState with 2 non-zero qubits:
            ├── |01010⟩
            └── non-zero qubits: [1, 3]

        Example using fromstring method
        >>> bitstring = "11001"
        >>> bs_from_string = BitState.fromstring(bitstring)
        >>> print(bs_from_string)
        >>> 5-qubit BitState with 3 non-zero qubits:
            ├── |11001⟩
            └── non-zero qubits: [0, 1, 4]
        Example using the zeros method
        >>> bs = BitState('1010')
        >>> z= bs.zeros()
        >>> print(z)
        >>> [1, 3]
    """

    def __init__(self, num_qubits, init=None):
        if isinstance(num_qubits, int):
            self.bits = bitarray(num_qubits)
            if init is None:
                self.bits.setall(0)
            else:
                if isinstance(init, list):
                    self.bits.setall(0)
                    for idx in init:
                        if 0 <= idx <= num_qubits:  # Adjust 1-based index to 0-based index
                            self.bits[idx] = 1  # Set the bit to 1
                        else:
                            raise ValueError("Invalid index in the 'init' list.")
                else:
                    self.bits.setall(init)
        elif isinstance(num_qubits, str):
            self.bits = bitarray(num_qubits)
        elif isinstance(num_qubits, Circuit):
            num_qubits = num_qubits.num_qubits()
            self.bits = bitarray(num_qubits)
            self.bits.setall(0)

            if init is not None:
                for idx in init:
                    if 1 <= idx <= num_qubits:  # Adjust 1-based index to 0-based index
                        self.bits[idx] = 1  # Set the bit to 1
                    else:
                        raise ValueError("Invalid index in the 'init' list.")
        else:
            self.bits = bitarray(num_qubits)

    @staticmethod
    def fromnonzeros(num_qubits: int, nonzeros: list):
        x = BitState(num_qubits)
        for i in range(num_qubits):
            x[i] = (i in nonzeros)
        return x

    @staticmethod
    def fromfunction(num_qubits: int, f: type[lambda: None]):
        x = BitState(num_qubits)
        for i in range(num_qubits):
            x[i] = f(i)
        return x

    @staticmethod
    def fromstring(bitstring: str):
        return BitState(bitstring)

    def nonzeros(self):
        return [i for i, bit in enumerate(self.bits) if bit]

    def zeros(self):
        return [i for i, bit in enumerate(self.bits) if not bit]

    def tointeger(self):
        return bitarr_to_int(self.bits[::-1])

    @staticmethod
    def fromint(num_qubits: int, integer: int, endian: str = 'little'):
        bitstring = bin(integer)[2:]
        padded_bitstring = bitstring.zfill(num_qubits)

        if endian == 'little':
            padded_bitstring = padded_bitstring
        elif endian != 'big':
            raise ValueError("endian must be either 'big' or 'little'")

        return BitState(padded_bitstring[::-1])

    def to01(self, endianess='big'):
        if endianess == 'big':
            return ''.join(map(str, self.bits))
        elif endianess == 'little':
            return ''.join(map(str, reversed(self.bits)))
        else:
            raise ValueError("Invalid endianess. Must be 'big' or 'little'.")

    def bitarr_to_index(self):
        int_val = bitarr_to_int(self.bits[::-1])
        return int_val

    def __eq__(self, other):
        if not isinstance(other, BitState):
            return False
        return self.bits == other.bits

    def __str__(self):
        s = f"{len(self.bits)}-qubit BitState"
        nz = self.nonzeros()
        if len(nz) != 0:
            s += f" with {len(nz)} non-zero qubits:\n"
            s += f"├── |{self.bits.to01()}⟩\n"
            s += f"└── non-zero qubits: {nz}"
        else:
            s += f":\n└── |{self.bits.to01()}⟩"
        return s

    def num_qubits(self):
        return len(self.bits)

    def __repr__(self):
        return f"BitState('{self.to01()}')"

    def __len__(self):
        return len(self.bits)

    def __hash__(self):
        return hash(self.bits.to01())

    def __iter__(self):
        return iter(self.bits)

    def __getitem__(self, index):
        if type(index) is slice:
            return BitState(self.bits[index])
        return self.bits[index]

    def __setitem__(self, index, value):
        if 0 <= index < len(self.bits):
            self.bits[index] = 1 if value else 0
        else:
            raise ValueError("Invalid qubit index.")


__all__ = ["BitState"]
