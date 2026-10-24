"""The rich reprs Jupyter reaches for.

A notebook calls ``_repr_html_`` before falling back to ``__repr__``. When
it raises, IPython prints the traceback into the cell and then renders the
plain repr anyway — so the cell shows a traceback next to a correct result
and nothing actually fails. These tests call the formatter directly so the
failure is a test failure instead.
"""

import pytest
from bitarray import bitarray, frozenbitarray

from mimiqcircuits import BitString, QCSResults
from mimiqcircuits.bitstrings import bitvec_to_int


def _results():
    return QCSResults(
        simulator="Test",
        version="0.0.0",
        fidelities=[1.0],
        avggateerrors=[0.0],
        cstates=[BitString.fromstring("01"), BitString.fromstring("11")],
        amplitudes={BitString.fromstring("01"): 0.5 + 0.0j},
        timings={"total": 1e-3},
    )


def test_repr_html_renders():
    html = _results()._repr_html_()
    assert html.startswith("<table>") and html.endswith("</table>")
    # Samples and amplitudes both index a BitString the same way, over
    # `bits` as written: "01" -> 0x1, "11" -> 0x3.
    assert ">0x1</td>" in html and ">0x3</td>" in html
    assert ">01</td>" in html


def test_repr_html_survives_an_empty_result():
    QCSResults()._repr_html_()


def test_bitvec_to_int_accepts_both_bitarray_flavours():
    # `frozenbitarray` subclasses `bitarray`; the guard used to reject the
    # mutable one, which took `_repr_html_` and `plothistogram` with it.
    assert bitvec_to_int(bitarray("1011")) == 11
    assert bitvec_to_int(frozenbitarray("1011")) == 11


def test_bitvec_to_int_rejects_a_bitstring():
    # A BitString is not a bit vector — callers must pass `.bits`.
    with pytest.raises(TypeError):
        bitvec_to_int(BitString.fromstring("1011"))
    with pytest.raises(ValueError):
        bitvec_to_int(bitarray())


def _roundtrip(results):
    from mimiqcircuits.proto.qcsrproto import fromproto_qcsr, toproto_qcsr

    return fromproto_qcsr(toproto_qcsr(results))


def test_proto_roundtrip_keeps_cstates_a_bitstring():
    # `fromproto_bitvector` returns a raw frozenbitarray; cstates must be
    # wrapped like amplitude keys already are, or remote results carry a
    # different type than local ones and every `.bits` call site breaks.
    rt = _roundtrip(_results())
    assert all(isinstance(cs, BitString) for cs in rt.cstates)
    assert [cs.to01() for cs in rt.cstates] == ["01", "11"]


def test_repr_html_renders_after_a_proto_roundtrip():
    assert _roundtrip(_results())._repr_html_() == _results()._repr_html_()


def test_histogram_survives_a_proto_roundtrip():
    local, remote = _results().histogram(), _roundtrip(_results()).histogram()
    assert remote == local == {BitString.fromstring("01"): 1, BitString.fromstring("11"): 1}
    # Same type on both paths means an amplitude key now finds its histogram
    # entry — it did not when cstates were frozenbitarrays.
    for key in _results().amplitudes:
        assert key in remote
