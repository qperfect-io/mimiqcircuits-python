import pytest
from symengine import pi, Symbol, eye
import mimiqcircuits as mc
from mimiqcircuits.operations.gates.generalized.rnz import GateRNZ
import io


def test_gate_rnz_construction():
    g = GateRNZ(2, pi / 2)
    assert g.num_qubits == 2
    assert g.theta == pi / 2

    # Lazy cases
    g1 = GateRNZ()
    assert isinstance(g1, mc.LazyExpr)

    g2 = GateRNZ(3)
    assert isinstance(g2, mc.LazyExpr)


def test_gate_rnz_invalid_args():
    with pytest.raises(ValueError):
        GateRNZ("2", pi / 2)
    with pytest.raises(ValueError):
        GateRNZ(-1, pi / 2)
    with pytest.raises(ValueError):
        GateRNZ(2, pi / 2, 1)


def test_gate_rnz_inverse():
    g = GateRNZ(3, pi / 4)
    g_inv = g.inverse()
    assert g_inv.num_qubits == 3
    assert g_inv.theta == -pi / 4


def test_gate_rnz_power():
    g = GateRNZ(2, pi / 3)
    g_pow = g.power(2)
    assert g_pow.num_qubits == 2
    assert g_pow.theta == 2 * pi / 3


def test_gate_rnz_matrix():
    g = GateRNZ(2, pi / 2)
    mat = g.matrix()
    assert mat.shape == (4, 4)


def test_gate_rnz_decompose():
    g = GateRNZ(3, pi / 2)
    c = mc.Circuit()
    c.push(g, 0, 1, 2)
    decomposed = g.decompose()
    assert len(decomposed) == 5
    assert isinstance(decomposed[2].operation, mc.GateRZ)


def test_gate_rnz_evaluate():
    θ = Symbol("θ")
    g = GateRNZ(2, θ)
    g_eval = g.evaluate({θ: pi})
    assert isinstance(g_eval, GateRNZ)
    assert g_eval.theta == pi


def test_gate_rnz_proto_io_bytesio():
    x = Symbol("x")
    c = mc.Circuit()
    c.push(GateRNZ(1, 1.23343), 0)
    c.push(GateRNZ(4, 2.718), *range(4))
    c.push(GateRNZ(1, x), 0)

    buf = io.BytesIO()
    c.saveproto(buf)

    buf.seek(0)

    # Deserialize
    loaded = mc.Circuit.loadproto(buf)

    assert isinstance(loaded, mc.Circuit)
    assert len(loaded) == len(c)

    for orig, restored in zip(c, loaded):
        assert type(orig.operation) == type(restored.operation)
        assert orig.qubits == restored.qubits
