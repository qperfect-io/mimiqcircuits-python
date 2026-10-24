"""Wire layout of the protobuf ``BitVector``.

The layout is shared with MimiqCircuitsBase.jl (``bitarr_to_bytes`` /
``bytes_to_bitarr``), so files written by either language must decode in the
other. These tests pin the bytes, not just the round trip.
"""

import random

import numpy as np
import pytest
from bitarray import bitarray, frozenbitarray

import mimiqcircuits as mc
from mimiqcircuits.proto.bitvector import fromproto_bitvector, toproto_bitvector
from mimiqcircuits.proto.qcsrproto import bitvec_to_bytes, bytes_to_bitvec


def reference_bitvec_to_bytes(bits):
    # Per-bit transcription of the Julia `bitarr_to_bytes`, the definition
    # of the layout the fast path must reproduce.
    b = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            if i + j >= len(bits):
                break
            val += int(bits[i + j]) << j
        b.append(val)
    return bytes(b)


# Bytes the per-bit encoding produces for short inputs.
KNOWN = [
    ("", ""),
    ("1", "01"),
    ("0000001", "40"),
    ("10000000", "01"),
    ("100000001", "0101"),
    ("111111111", "ff01"),
    ("0110100111", "9603"),
    ("1" + "0" * 63 + "1", "010000000000000001"),
    ("10" * 50, "55555555555555555555555505"),
]

LENGTHS = [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 100, 127, 128, 129, 1000]


def _cases():
    rng = random.Random(1234)
    for n in LENGTHS:
        yield "0" * n
        yield "1" * n
        for _ in range(3):
            yield "".join(rng.choice("01") for _ in range(n))


def _inputs(s):
    # Every form a bit vector reaches the proto layer in. The big-endian
    # bitarray has a different buffer for the same bits, so it catches an
    # encoder that copies the buffer instead of the bit sequence. The bytes
    # form catches one that reads a bytes argument as a packed buffer.
    bits = [int(c) for c in s]
    return [
        mc.BitString(s),
        mc.BitString(frozenbitarray(s, endian="big")),
        frozenbitarray(s, endian="little"),
        frozenbitarray(s, endian="big"),
        bitarray(s, endian="little"),
        bitarray(s, endian="big"),
        s,
        bits,
        [bool(b) for b in bits],
        tuple(bits),
        bytes(bits),
        np.array(bits, dtype=np.int64),
        np.array(bits, dtype=np.uint8),
        np.array(bits, dtype=bool),
    ]


@pytest.mark.parametrize("s,hexbytes", KNOWN)
def test_known_bytes(s, hexbytes):
    for v in _inputs(s):
        assert bitvec_to_bytes(v).hex() == hexbytes
    decoded = bytes_to_bitvec(bytes.fromhex(hexbytes), len(s))
    assert decoded.to01() == s


@pytest.mark.parametrize("s", list(_cases()))
def test_matches_reference_layout(s):
    expected = reference_bitvec_to_bytes([int(c) for c in s])
    assert len(expected) == (len(s) + 7) // 8
    for v in _inputs(s):
        assert bitvec_to_bytes(v) == expected

    decoded = bytes_to_bitvec(expected, len(s))
    assert decoded.to01() == s
    assert decoded.endian == "little"
    assert isinstance(decoded, frozenbitarray)


@pytest.mark.parametrize(
    "bad,exc",
    [
        ([0, 2], ValueError),
        (b"\x02", ValueError),
        (np.array([1, 2]), ValueError),
        ([0.0, 1.0], TypeError),
        (np.array([0.0, 1.0]), TypeError),
        ("102", ValueError),
    ],
)
def test_rejects_non_bits(bad, exc):
    with pytest.raises(exc):
        bitvec_to_bytes(bad)


def test_decode_without_length_keeps_padding():
    assert bytes_to_bitvec(b"\x05").to01() == "10100000"


@pytest.mark.parametrize("s", list(_cases()))
def test_bitvector_message_roundtrip(s):
    msg = toproto_bitvector(mc.BitString(s))
    assert msg.len == len(s)
    assert msg.data == reference_bitvec_to_bytes([int(c) for c in s])
    assert fromproto_bitvector(msg).to01() == s

