import pytest
import mimiqcircuits as mc
from mimiqcircuits.operations.gates.generalized.rpauli import RPauli
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString
from symengine import pi, Symbol, Matrix
import numpy as np


def test_rpauli_str_and_construction():
    p = PauliString("XYI")
    theta = pi / 2
    gate = RPauli(p, theta)

    assert gate.num_qubits == 3
    assert gate.theta == theta
    assert gate.pauli == p
    assert str(gate) == 'R("XYI", (1/2)*pi)'


def test_rpauli_identity_detection():
    g_zero = RPauli(PauliString("XYZ"), 0)
    assert g_zero.isidentity() is True

    g_id_only = RPauli(PauliString("III"), pi)
    assert g_id_only.isidentity() is True

    g_id_zero = RPauli(PauliString("III"), 0)
    assert g_id_zero.isidentity() is True


def test_rpauli_isunitary_and_wrapper():
    g = RPauli(PauliString("XYZ"), pi / 3)
    assert g.isunitary()
    assert g.iswrapper()


def test_rpauli_matrix_numeric():
    g = RPauli(PauliString("X"), pi)
    mat = g.matrix()
    assert isinstance(mat, Matrix)
    assert mat.shape == (2, 2)


def test_rpauli_matrix_symbolic():
    x = Symbol("x")
    g = RPauli(PauliString("Z"), x)
    mat = g.matrix()
    assert isinstance(mat, Matrix)
    assert mat.shape == (2, 2)


def test_rpauli_decomposition():
    g = RPauli(PauliString("YX"), pi / 2)
    c = mc.Circuit()
    c.push(g, 0, 1)

    d = g.decompose()
    assert isinstance(d, mc.Circuit)
    assert len(d) > 0
    assert any(isinstance(inst.operation, mc.GateRNZ) for inst in d)


def test_rpauli_decomposition_identity_path():
    g = RPauli(PauliString("III"), pi)
    c = mc.Circuit()
    c.push(g, 0, 1, 2)
    d = g.decompose()
    assert isinstance(d, mc.Circuit)
    assert any(isinstance(inst.operation, mc.GateU) for inst in d)


def test_rpauli_invalid_decompose_length_mismatch():
    g = RPauli(PauliString("XY"), pi / 2)
    with pytest.raises(ValueError):
        g._decompose(mc.Circuit(), [0])


def test_rpauli_evaluate_symbolic():
    x = Symbol("x")
    g = RPauli(PauliString("XY"), x)
    g_eval = g.evaluate({x: pi})
    assert isinstance(g_eval, RPauli)
    assert g_eval.theta == pi
    assert g_eval.pauli == g.pauli


def test_rpauli_proto_conversion_io():
    x = Symbol("x")
    c = mc.Circuit()
    c.push(RPauli(PauliString("XYZ"), pi), 0, 1, 2)
    c.push(RPauli(PauliString("X"), x), 0)

    import io

    buf = io.BytesIO()
    c.saveproto(buf)
    buf.seek(0)

    c_loaded = mc.Circuit.loadproto(buf)
    assert isinstance(c_loaded, mc.Circuit)
    assert len(c_loaded) == len(c)
    for inst1, inst2 in zip(c, c_loaded):
        assert type(inst1.operation) == type(inst2.operation)
        assert inst1.qubits == inst2.qubits
