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
from mimiqcircuits.proto import qcsresults_pb2

import mimiqcircuits.qcsresults as mq

from mimiqcircuits.bitstrings import BitString
from bitarray import bitarray, frozenbitarray


def bitvec_to_bytes(bv):
    """Pack a bit vector into bytes, the layout of a protobuf ``BitVector``.

    Bit ``i`` goes to byte ``i // 8`` at position ``i % 8`` counted from the
    least significant bit, and the unused high bits of the last byte are zero.

    Accepts a ``BitString``, a ``bitarray`` of either bit order, a string of
    ``"0"`` and ``"1"`` characters, a numpy array, or any other sequence of
    0, 1 and booleans. ``bytes`` and ``bytearray`` count as sequences of
    integers, one bit per element, not as a packed buffer. Returns ``bytes``.

    Raises ``ValueError`` for an element other than 0 or 1, and ``TypeError``
    for an element that is not an integer or boolean (for example a float).
    """
    if isinstance(bv, BitString):
        bv = bv.bits
    if isinstance(bv, bitarray):
        # A little-endian buffer already is the wire layout. Converting a
        # big-endian one copies bit by bit, keeping the sequence, not the buffer.
        if bv.endian != "little":
            bv = bitarray(bv, endian="little")
    elif isinstance(bv, str):
        bv = bitarray(bv, endian="little")
    else:
        # bitarray reads a bytes-like argument as a packed buffer, so every
        # other input goes in as a list. numpy's tolist() turns numpy scalars,
        # which bitarray refuses, into Python ints and bools.
        items = bv.tolist() if hasattr(bv, "tolist") else list(bv)
        bv = bitarray(items, endian="little")
    return bv.tobytes()


def bytes_to_bitvec(b, n=None):
    """Unpack the bytes of a protobuf ``BitVector`` into a little-endian
    ``frozenbitarray``, the inverse of :func:`bitvec_to_bytes`.

    With ``n`` given, keeps only the first ``n`` bits, dropping the padding of
    the last byte.
    """
    bv = bitarray(endian="little")
    bv.frombytes(b)
    if n is not None:
        del bv[n:]
    return frozenbitarray(bv)


def toproto_qcsr(s):
    from mimiqcircuits.proto.bitvector import toproto_bitvector

    qcs_results = qcsresults_pb2.QCSResults()
    qcs_results.simulator = s.simulator
    qcs_results.version = s.version
    qcs_results.fidelities.extend(s.fidelities)
    qcs_results.avggateerrors.extend(s.avggateerrors)

    qcs_results.cstates.extend([toproto_bitvector(cstate) for cstate in s.cstates])

    for zstate in s.zstates:
        qcs_results.zstates.extend([toproto_complexvector(zstate)])

    for key, value in s.timings.items():
        qcs_results.timings[key] = value

    amplitude_entries = [
        toproto_amplitude(key, value) for key, value in s.amplitudes.items()
    ]
    qcs_results.amplitudes.extend(amplitude_entries)

    return qcs_results


def fromproto_qcsr(s):
    from mimiqcircuits.proto.bitvector import fromproto_bitvector

    qcs_results = mq.QCSResults(
        s.simulator,
        s.version,
        s.fidelities,
        s.avggateerrors,
        [BitString(fromproto_bitvector(cstate)) for cstate in s.cstates],
        [fromproto_complexvector(zstate) for zstate in s.zstates],
        amplitudes={
            key: value
            for amplitude_entry in s.amplitudes
            for key, value in [fromproto_amplitude(amplitude_entry)]
        },
        timings=dict(s.timings),
    )

    return qcs_results


def toproto_amplitude(key, value):
    from mimiqcircuits.proto.bitvector import toproto_bitvector

    return qcsresults_pb2.AmplitudeEntry(
        key=toproto_bitvector(key),
        val=toproto_complexdouble(value),
    )


def fromproto_amplitude(amplitude_entry):
    from mimiqcircuits.proto.bitvector import fromproto_bitvector

    key = BitString(fromproto_bitvector(amplitude_entry.key))
    value = fromproto_complexdouble(amplitude_entry.val)
    return key, value


def toproto_complexvector(v):
    q = [toproto_complexdouble(x) for x in v]
    return qcsresults_pb2.ComplexVector(data=q)


def fromproto_complexvector(v):
    q = [fromproto_complexdouble(x) for x in v.data]
    return q


def toproto_complexdouble(v):
    return qcsresults_pb2.ComplexDouble(real=v.real, imag=v.imag)


def fromproto_complexdouble(v):
    return complex(v.real, v.imag)


__all__ = ["toproto_qcsr", "fromproto_qcsr"]
