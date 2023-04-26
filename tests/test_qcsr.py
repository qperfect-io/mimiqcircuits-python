
import mimiqcircuits.qcsr as qcsr
from bitarray import bitarray
import numpy as np
import random
import tempfile
import os


TYPES = [
    bool, int, float, complex, np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64, np.float32, np.float64,
    np.complex64, np.complex128
]


def rand_type():
    return TYPES[random.randint(0, len(TYPES) - 1)]


def rand_bool():
    return bool(random.randint(0, 1))


def rand_bitarray(bslen):
    return bitarray((rand_bool() for _ in range(bslen)))


def rand_number(T):
    if T == bool:
        return rand_bool()
    if T == int:
        return T(random.randint(-2**63, 2**63 - 1))
    if T == float or T == np.float32 or T == np.float64:
        return random.random()
    if T == complex or T == np.complex64 or T == np.complex128:
        return T(complex(random.random(), random.random()))
    if T == np.uint8:
        return T(random.randint(0, 2**8 - 1))
    if T == np.uint16:
        return T(random.randint(0, 2**16 - 1))
    if T == np.uint32:
        return T(random.randint(0, 2**32 - 1))
    if T == np.uint64:
        return T(random.randint(0, 2**64 - 1))
    if T == np.int8:
        return T(random.randint(-2**7, 2**7 - 1))
    if T == np.int16:
        return T(random.randint(-2**15, 2**15 - 1))
    if T == np.int32:
        return T(random.randint(-2**31, 2**31 - 1))
    if T == np.int64:
        return T(random.randint(-2**63, 2**63 - 1))

    raise TypeError(f'unhandled type: {T}')


def test_qcsr():
    N = 2**10
    L = 300

    data = []

    for _ in range(N):
        bslen = np.random.randint(0, L)
        bs = rand_bitarray(bslen)
        d = rand_number(rand_type())
        data.append((bs, d))

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test.qcsr')

        with qcsr.QcsrFile(filename, "wb") as io:
            io.write(data)

        with qcsr.QcsrFile(filename, "rb") as io:
            data2 = io.read()

    for (d1, d2) in zip(data, data2):
        assert d1[0] == d2[0]
        assert d1[1] == d2[1]
