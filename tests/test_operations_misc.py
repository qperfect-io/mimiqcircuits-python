import numpy as np
import pytest
import mimiqcircuits as mc
import symengine as se


def test_diffusion_one_qubit_decomposes():
    c = mc.Diffusion(1).decompose()
    assert [type(i.operation) for i in c] == [mc.GateRY, mc.GateZ, mc.GateRY]


def test_customdiagonal_checks_numeric_entries_with_symbols():
    x = se.Symbol("x")
    with pytest.raises(ValueError):
        mc.GateCustomDiagonal([2, x])
    mc.GateCustomDiagonal([1j, x])


def test_lazy_expr_rejects_extra_arguments():
    with pytest.raises(TypeError):
        mc.ShiftCoordinates()(0.1, 0.2)
    assert mc.ShiftCoordinates()([0.1, 0.2]).get_notes() == [0.1, 0.2]
    with pytest.raises(TypeError):
        mc.control(mc.GateX())(1, 2)


def test_kronecker_mixed_inputs():
    from mimiqcircuits.matrices import kronecker

    k = kronecker(np.eye(2), se.Matrix([[0, 1], [1, 0]]))
    assert isinstance(k, se.Matrix)
    assert np.allclose(np.array(k.tolist(), dtype=float), np.kron(np.eye(2), [[0, 1], [1, 0]]))
