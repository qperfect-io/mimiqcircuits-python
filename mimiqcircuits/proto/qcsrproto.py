from mimiqcircuits.proto import qcsresults_pb

import mimiqcircuits.qcsresults as mq

from mimiqcircuits.bitstrings import BitString
from bitarray import frozenbitarray


def bitvec_to_bytes(bv):
    b = bytearray()
    for i in range(0, len(bv), 8):
        val = 0
        for j in range(8):
            if i+j >= len(bv):
                break
            val += bv[i + j] << j
        b.append(val)
    return b


def bytes_to_bitvec(b, n=None):
    bitstring = ''
    for val in b:
        for i in range(8):
            bitstring += str((val >> i) & 1)
    if n is not None:
        return frozenbitarray(bitstring[:n])
    else:
        return frozenbitarray(bitstring)


def toproto_qcsr(s):
    qcs_results = qcsresults_pb.QCSResults()
    qcs_results.simulator = s.simulator
    qcs_results.version = s.version
    qcs_results.fidelities.extend(s.fidelities)
    qcs_results.avggateerrors.extend(s.avggateerrors)

    for cstate in s.cstates:
        qcs_results.cstates.extend([toproto_bitvector(cstate)])

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
    qcs_results = mq.QCSResults(
        s.simulator,
        s.version,
        s.fidelities,
        s.avggateerrors,
        [fromproto_bitvector(cstate) for cstate in s.cstates],
        [fromproto_complexvector(zstate) for zstate in s.zstates],
        amplitudes={
            key: value
            for amplitude_entry in s.amplitudes
            for key, value in [fromproto_amplitude(amplitude_entry)]
        },
        timings=dict(s.timings)
    )

    return qcs_results


def toproto_amplitude(key, value):
    key_bit_array = frozenbitarray(key)
    entry = qcsresults_pb.AmplitudeEntry(
        key=qcsresults_pb.BitVector(
            len=len(key_bit_array), data=bytes(key_bit_array)),
        val=toproto_complexdouble(value)
    )
    return entry


def fromproto_amplitude(amplitude_entry):
    key = BitString(fromproto_bitvector(amplitude_entry.key))
    value = fromproto_complexdouble(amplitude_entry.val)
    return key, value


def toproto_complexvector(v):
    q = [toproto_complexdouble(x) for x in v]
    return qcsresults_pb.ComplexVector(data=q)


def fromproto_complexvector(v):
    q = [fromproto_complexdouble(x) for x in v.data]
    return q


def toproto_complexdouble(v):
    return qcsresults_pb.ComplexDouble(real=v.real, imag=v.imag)


def fromproto_complexdouble(v):
    return complex(v.real, v.imag)


def toproto_bitvector(bv):
    return qcsresults_pb.BitVector(
        len=len(bv), data=bytes(bitvec_to_bytes(bv))
    )


def fromproto_bitvector(bv):
    return bytes_to_bitvec(bv.data, bv.len)


__all__ = ['toproto_qcsr', 'fromproto_qcsr']
