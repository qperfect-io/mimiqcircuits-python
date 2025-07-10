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

from bitarray import bitarray, frozenbitarray


def _helper_bitvec_to_int(arr):
    res = 0
    for bit in arr:
        res = (res << 1) | bit
    return res


# Function to convert a bitarray to an integer
def bitvec_to_int(arr):
    if not isinstance(arr, bitarray) or not isinstance(arr, frozenbitarray):
        raise TypeError("Input must be a bitarray or BitVector.")
    num_bits = len(arr)
    if num_bits > 0:
        return _helper_bitvec_to_int(arr)
    else:
        raise ValueError("Input array is empty.")


# Function to convert an integer to a bitarray of given size (pad)
def int_to_bitvec(x, pad, frozen=True):
    if not isinstance(x, int):
        raise TypeError("Input must be an integer.")
    if x < 0:
        raise ValueError("Input integer must be non-negative.")

    binary_str = bin(x)[2:].zfill(pad)

    if frozen:
        return frozenbitarray(binary_str)
    else:
        return bitarray(binary_str)


class BitString:
    """BitString for the quantum states.

    Representation of the quantum state of a quantum register with definite
    values for each qubit.

    Examples:

        Initialization:

        >>> from mimiqcircuits import *
        >>> from bitarray import bitarray
        >>> BitString(16) # number of qubits
        bs"0000000000000000"
        >>> BitString('10101') # binary string
        bs"10101"
        >>> BitString([1,0,0,0,1]) # binary string
        bs"10001"
        >>> BitString((1,0,0,0,1)) # binary string
        bs"10001"

        >>> BitString(bitarray('101010')) # bitarray
        bs"101010"

        Other initializations:

        >>> BitString.fromnonzeros(16, [1, 3, 5, 7, 9, 11, 13, 15])
        bs"0101010101010101"
        >>> BitString.fromfunction(16, lambda i: i % 2 == 1)
        bs"0101010101010101"
        >>> BitString.fromstring('10101')
        bs"10101"
        >>> BitString.fromint(16, 21)
        bs"1010100000000000"
        >>> BitString.fromint(16, 21, 'little')
        bs"0000000000010101"

        Accessing the bits:

        >>> bs = BitString(16)
        >>> bs[0] # get the 0th bit
        0
        >>> bs[0:4] # get the first 4 bits
        bs"0000"

        Bitwise operations:

        >>> bs1 = BitString('10101')
        >>> bs2 = BitString('11100')
        >>> bs1 | bs2 # OR
        bs"11101"
        >>> bs1 & bs2 # AND
        bs"10100"
        >>> bs1 ^ bs2 # XOR
        bs"01001"
        >>> ~bs1 # NOT
        bs"01010"
        >>> bs1 << 2 # left shift
        bs"10100"
        >>> bs1 >> 2 # right shift
        bs"00101"

        Other operations:

        >>> bs1 + bs2 # concatenation
        bs"1010111100"
        >>> bs1 * 2 # repetition
        bs"1010110101"
    """

    _bits = bitarray(0)

    def __init__(self, arg, indices=None):
        """Initialize the BitString.

        If the number of qubits is given, then the BitString is initialized to
        the all-zero state.
        If a binary string or a bitarray is given, then the BitString is the
        corresponding state.

        Examples:
            >>> BitString(16) # number of qubits
            bs"0000000000000000"
            >>> BitString('10101') # binary string
            bs"10101"
            >>> BitString(bitarray('101010')) # bitarray
            bs"101010"
        """
        if isinstance(arg, int):
            if indices is None:
                bitstring = arg * "0"
            else:
                if not all(0 <= i < arg for i in indices):
                    raise ValueError(
                        "Index out of range for the given number of qubits."
                    )
                bitstring = "".join("1" if i in indices else "0" for i in range(arg))
        elif isinstance(arg, (str, bitarray, frozenbitarray, list, tuple)):
            bitstring = arg
        else:
            raise TypeError(
                "Invalid input type. Expected 'str', 'int', 'bitarray', 'list', 'tuple', 'frozenbitarray'"
            )

        self._bits = frozenbitarray(bitstring)

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, bits):
        raise AttributeError("Cannot set the bits of a BitString.")

    @staticmethod
    def fromnonzeros(num_qubits: int, nonzeros: list):
        """Initialize a BitString with specific non-zero qubits.

        Arguments:
            num_qubits (int): The number of qubits in the BitString.
            nonzeros (list): A list of non-zero qubit indices to set in the BitString.

        Returns:
            A BitString with the specified non-zero qubits.
        """
        bitstring = ""

        if not all(0 <= i < num_qubits for i in nonzeros):
            raise ValueError("Invalid nonzero index.")

        for i in range(num_qubits):
            bitstring += "1" if i in nonzeros else "0"

        return BitString(bitstring)

    @staticmethod
    def fromfunction(num_qubits: int, f: type[lambda: None]):
        """Initialize a BitString from a function.

        Arguments:
            num_qubits (int): The number of qubits in the BitString.
            f (function): A function that takes an integer and returns a boolean.

        Returns:
            A BitString.
        """
        bitstring = ""
        for i in range(num_qubits):
            bitstring += "1" if f(i) else "0"
        return BitString(bitstring)

    @staticmethod
    def fromstring(bitstring: str):
        """Initialize a BitString from a string.

        Arguments:
            bitstring (str): The string representation of the BitString.

        Returns:
            A BitString.
        """
        return BitString(bitstring)

    @staticmethod
    def fromint(num_qubits: int, integer: int, endianess: str = "big"):
        """Initialize a BitString from an integer.

        Arguments:
            num_qubits (int): The number of qubits in the BitString.
            integer (int): The integer value of the BitString.
            endianess (str): The endianess of the integer. Default is 'big'.

        Returns:
            A BitString.
        """
        if len(bin(integer)[2:]) > num_qubits:
            raise ValueError("Integer is too large for the given number of qubits.")

        if endianess == "little":
            bitstring = bin(integer)[2:].zfill(num_qubits)[::-1]
        elif endianess == "big":
            bitstring = bin(integer)[2:].zfill(num_qubits)
        else:
            raise ValueError("endian must be either 'big' or 'little'")

        return BitString(bitstring[::-1])

    def num_qubits(self):
        """Return the number of qubits in the BitString."""
        return len(self.bits)

    def nonzeros(self):
        """Return the indices of the non-zero qubits."""
        return [i for i, bit in enumerate(self.bits) if bit]

    def zeros(self):
        """Return the indices of the zero qubits."""
        return [i for i, bit in enumerate(self.bits) if not bit]

    def tointeger(self, endianess: str = "big"):
        """Return the integer value of the BitString.

        Arguments:
            endianess (str): The endianess of the integer. Default is 'big'.

        Returns:
            The integer value of the BitString.
        """
        if endianess == "big":
            return bitvec_to_int(self.bits[::-1])
        elif endianess == "little":
            return bitvec_to_int(self.bits)
        else:
            raise ValueError("Invalid endianess. Must be 'big' or 'little'.")

    def to01(self, endianess="big"):
        """Return the binary string representation of the BitString.

        Arguments:
            endianess (str): The endianess of the integer. Default is 'big'

        Retruns:
            The binary string representation of the BitString.
        """
        if endianess == "big":
            return "".join(map(str, self.bits))
        elif endianess == "little":
            return "".join(map(str, reversed(self.bits)))
        else:
            raise ValueError("Invalid endianess. Must be 'big' or 'little'.")

    def toindex(self, endianess: str = "big"):
        """Return the integer index of the BitString.

        Arguments:
            endianess (str): The endianess of the integer. Default is 'big'.

        Returns:
            The integer index of the BitString.
        """
        if endianess == "big":
            return bitvec_to_int(self.bits[::-1])
        elif endianess == "little":
            return bitvec_to_int(self.bits)
        else:
            raise ValueError("endian must be either 'big' or 'little'")

    def __eq__(self, other):
        if not isinstance(other, BitString):
            return False
        return self.bits == other.bits

    def __str__(self):
        s = f"{len(self.bits)}-qubit BitString"
        nz = self.nonzeros()
        if len(nz) != 0:
            s += f" with {len(nz)} non-zero qubits:\n"
            s += f"├── |{self.bits.to01()}⟩\n"
            s += f"└── non-zero qubits: {nz}"
        else:
            s += f":\n└── |{self.bits.to01()}⟩"
        return s

    def __repr__(self):
        return "bs" + '"' + self.to01() + '"'

    def __len__(self):
        return len(self.bits)

    def __hash__(self):
        return hash(self.bits.to01())

    def __iter__(self):
        return iter(self.bits)

    def __getitem__(self, index):
        if type(index) is slice:
            return BitString(self.bits[index])
        return self.bits[index]

    def __setitem__(self, index, value):
        """Set a specific bit by creating a new frozenbitarray."""
        if not isinstance(value, (bool, int)) or value not in (0, 1):
            raise ValueError("Value must be a boolean or an integer (0 or 1).")

        # Convert frozenbitarray to a mutable bitarray
        temp_bits = self._bits.tolist()
        temp_bits[index] = value

        # Recreate the frozenbitarray with the updated bits
        self._bits = frozenbitarray(temp_bits)

    def __or__(self, other):
        if not isinstance(other, BitString):
            raise TypeError("BitString can only be ORed with another BitString.")
        return BitString(self.bits | other.bits)

    def __and__(self, other):
        if not isinstance(other, BitString):
            raise TypeError("BitString can only be ANDed with another BitString.")
        return BitString(self.bits & other.bits)

    def __xor__(self, other):
        if not isinstance(other, BitString):
            raise TypeError("BitString can only be XORed with another BitString.")
        return BitString(self.bits ^ other.bits)

    def __invert__(self):
        return BitString(~self.bits)

    def __lshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("BitString can only be left shifted by an integer.")
        return BitString(self.bits << other)

    def __rshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("BitString can only be right shifted by an integer.")
        return BitString(self.bits >> other)

    def __add__(self, other):
        if not isinstance(other, BitString):
            raise TypeError("BitString can only be added with another BitString.")
        return BitString(self.bits + other.bits)

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("BitString can only be multiplied by an integer.")
        return BitString(self.bits * other)


__all__ = ["BitString"]
