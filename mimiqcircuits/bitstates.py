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


class BitState(bitarray):

    @staticmethod
    def zeros(num_qubits: int):
        x = BitState(num_qubits)
        x.setall(False)
        return x

    @staticmethod
    def fromnonzeros(num_qubits: int, nonzeros: list):
        x = BitState(num_qubits)
        for i in range(num_qubits):
            x[i] = (i in nonzeros)
        return x

    @staticmethod
    def fromstring(bitstring: str):
        return BitState(bitstring)

    @staticmethod
    def fromfunction(num_qubits: int, f: type(lambda: None)):
        x = BitState(num_qubits)
        for i in range(num_qubits):
            x[i] = f(i)
        return x

    @staticmethod
    def fromint(num_qubits: int, integer: int, endian: str = 'little'):
        bitstring = bin(integer)[2:]
        padded_bitstring = bitstring.zfill(num_qubits)

        if endian == 'little':
            padded_bitstring = padded_bitstring[::-1]
        elif endian != 'big':
            raise ValueError("endian must be either 'big' or 'little'")

        return BitState(padded_bitstring)

    def nonzeros(self):
        return [i for i in range(self.num_qubits()) if self[i]]

    def tointeger(self):
        return int(self.to01()[::-1], 2)

    def __str__(self):
        s = f"{self.num_qubits()}-qubit BitState"
        nz = self.nonzeros()
        if len(nz) != 0:
            s += f" with {len(nz)} non-zero qubits:\n"
            s += f"├── |{self.to01()}⟩\n"
            s += f"└── non-zero qubits: {nz}"
        else:
            s += f":\n└── |{self.to01()}⟩"
        return s

    def __repr__(self):
        return f"BitState('{self.to01()}')"

    def num_qubits(self):
        return len(self)

    def __hash__(self):
        return hash(self.to01())


__all__ = ["BitState"]
