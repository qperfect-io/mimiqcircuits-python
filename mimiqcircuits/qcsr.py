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

from typing import Type, Tuple, List, TypeVar, BinaryIO
import struct
from bitarray import bitarray
import numpy as np

QCSR_MAGIC = b"\x51\x43\x53\x52\x00\x00\x00\x00"
QCSR_EXTENSION = ".qcsr"
QCSR_VERSION = 1

T = TypeVar('T')


QCSR_DTYPE_BOOL = 1
QCSR_DTYPE_CHAR = 2
QCSR_DTYPE_UINT8 = 3
QCSR_DTYPE_UINT16 = 4
QCSR_DTYPE_UINT32 = 5
QCSR_DTYPE_UINT64 = 6
QCSR_DTYPE_INT8 = 7
QCSR_DTYPE_INT16 = 8
QCSR_DTYPE_INT32 = 9
QCSR_DTYPE_INT64 = 10
QCSR_DTYPE_FLOAT32 = 11
QCSR_DTYPE_FLOAT64 = 12
QCSR_DTYPE_COMPLEX64 = 13
QCSR_DTYPE_COMPLEX128 = 14


def dtype_to_type(dtype_id: int):
    return {
        QCSR_DTYPE_BOOL: bool,
        QCSR_DTYPE_CHAR: str,
        QCSR_DTYPE_UINT8: np.uint8,
        QCSR_DTYPE_UINT16: np.uint16,
        QCSR_DTYPE_UINT32: np.uint32,
        QCSR_DTYPE_UINT64: np.uint64,
        QCSR_DTYPE_INT8: np.int8,
        QCSR_DTYPE_INT16: np.int16,
        QCSR_DTYPE_INT32: np.int32,
        QCSR_DTYPE_INT64: np.int64,
        QCSR_DTYPE_FLOAT32: np.float32,
        QCSR_DTYPE_FLOAT64: np.float64,
        QCSR_DTYPE_COMPLEX64: np.complex64,
        QCSR_DTYPE_COMPLEX128: np.complex128,
    }[dtype_id]


def type_to_dtype(t: Type):
    return {
        bool: QCSR_DTYPE_BOOL,
        str: QCSR_DTYPE_CHAR,
        np.uint8: QCSR_DTYPE_UINT8,
        np.uint16: QCSR_DTYPE_UINT16,
        np.uint32: QCSR_DTYPE_UINT32,
        np.uint64: QCSR_DTYPE_UINT64,
        np.int8: QCSR_DTYPE_INT8,
        np.int16: QCSR_DTYPE_INT16,
        np.int32: QCSR_DTYPE_INT32,
        np.int64: QCSR_DTYPE_INT64,
        np.float32: QCSR_DTYPE_FLOAT32,
        np.float64: QCSR_DTYPE_FLOAT64,
        np.complex64: QCSR_DTYPE_COMPLEX64,
        np.complex128: QCSR_DTYPE_COMPLEX128,
        int: QCSR_DTYPE_INT64,
        float: QCSR_DTYPE_FLOAT64,
        complex: QCSR_DTYPE_COMPLEX128,
    }[t]


def _typecode(x):
    if x == bool:
        return '?'
    if x == np.uint8:
        return 'B'
    if x == np.uint16:
        return 'H'
    if x == np.uint32:
        return 'I'
    if x == np.uint64:
        return 'Q'
    if x == np.int8:
        return 'b'
    if x == np.int16:
        return 'h'
    if x == np.int32:
        return 'i'
    if x == np.int64:
        return 'q'
    if x == np.float32:
        return 'f'
    if x == np.float64:
        return 'd'
    if x == np.complex64:
        return 'f'
    if x == np.complex128:
        return 'd'
    elif x == int:
        return 'q'
    elif x == float:
        return 'd'
    elif x == complex:
        return 'd'
    else:
        raise TypeError(f'unhandled type: {type(x)}')


def _write_littleendian(f: BinaryIO, v, T: Type) -> int:
    if T == complex or T == np.complex64 or T == np.complex128:
        return f.write(
            struct.pack('<' + _typecode(T) + _typecode(T), v.real, v.imag)
        )

    f.write(struct.pack('<' + _typecode(T), v))
    return struct.calcsize(_typecode(T))


def _read_littleendian(f: BinaryIO, T: Type):
    if T == complex or T == np.complex64 or T == np.complex128:

        b = f.read(2 * struct.calcsize(_typecode(T)))
        if not b:
            raise EOFError('Unexpected EOF while reading complex value')

        s = struct.unpack('<' + _typecode(T) + _typecode(T), b)
        return T(s[0] + 1j * s[1])

    b = f.read(struct.calcsize(_typecode(T)))
    if not b:
        raise EOFError('Unexpected EOF while reading complex value')
    return struct.unpack('<' + _typecode(T), b)[0]


def _read_littleendian_array(f: BinaryIO, n: int, T: Type) -> List:
    return [_read_littleendian(f, T) for _ in range(n)]


class QcsrFile:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()

    def writeheader(self):
        f = self.file

        # 8 bytes for magic
        f.write(QCSR_MAGIC)

        # version and 4 reserved bytes (total 8 bytes)
        _write_littleendian(f, QCSR_VERSION, np.uint32)
        f.write(b"\x00\x00\x00\x00")

        # 8 reserved bytes
        f.write(b"\x00\x00\x00\x00")
        f.write(b"\x00\x00\x00\x00")

        # 8 reserved bytes
        f.write(b"\x00\x00\x00\x00")
        f.write(b"\x00\x00\x00\x00")

    def readheader(self):
        f = self.file
        magic = f.read(8)

        if magic != QCSR_MAGIC:
            raise ValueError('Wrong file')

        version = _read_littleendian(f, np.uint32)

        if version > QCSR_VERSION:
            raise ValueError('Non compatible file version')

        # 4 reserved bytes
        f.read(4)

        # 8 reseverd bytes
        f.read(8)

        # 8 reseverd bytes
        f.read(8)

        return magic, version

    def readone(self):

        # write the chunk header
        n = _read_littleendian(self.file, np.uint64)
        dtype = _read_littleendian(self.file, np.uint8)

        # reserved bytes to align the header
        _ = self.file.read(7)

        # actual data
        bs = bitarray(_read_littleendian_array(self.file, n, bool))
        d = _read_littleendian(self.file, dtype_to_type(dtype))

        return bs, d

    def read(self, n=None):
        if self.file is None:
            self.__enter__()

        res = []

        self.readheader()
        if n is None:
            while True:
                try:
                    res.append(self.readone())
                except EOFError:
                    break

        else:
            for i in range(n):
                res.append(self.readone())

        return res

    def write(self, data: List[Tuple[bitarray, T]]):
        if self.file is None:
            self.__enter__()

        self.writeheader()

        for bs, d in data:
            n = len(bs)
            dtype = type_to_dtype(type(d))

            _write_littleendian(self.file, n, np.uint64)
            _write_littleendian(self.file, dtype, np.uint8)

            self.file.write(b'\x00\x00\x00\x00\x00\x00\x00')

            for b in bs:
                _write_littleendian(self.file, b, bool)

            _write_littleendian(self.file, d, type(d))
