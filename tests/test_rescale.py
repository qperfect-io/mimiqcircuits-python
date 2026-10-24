import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc


@pytest.mark.parametrize(
    "op",
    [
        mc.Projector0(0.5),
        mc.SigmaPlus(0.5),
        mc.DiagonalOp(0.5, 0.25),
        mc.Operator(se.Matrix([[0.5, 0], [0, 0.25]])),
        mc.LossyOperator(se.Matrix([[0.5, 0], [0, 0.25]])),
    ],
)
def test_rescale_operator(op):
    def numeric(o):
        return np.array(o.matrix().tolist(), dtype=complex)

    expected = 2 * numeric(op)

    scaled = op.rescale(2)
    assert type(scaled) is type(op)
    assert np.allclose(numeric(scaled), expected)

    assert op.rescale_inplace(2) is op
    assert np.allclose(numeric(op), expected)


def test_rescale_keeps_lossy_mask():
    op = mc.LossyOperator(se.Matrix([[1, 0], [0, 1]]), [True])
    assert op.rescale(0.5).lossy == op.lossy


def test_rescale_gate():
    rg = mc.GateH().rescale(0.5)
    assert isinstance(rg, mc.RescaledGate)
    assert rg.get_scale() == 0.5

    assert rg.rescale_inplace(0.5) is rg
    assert rg.get_scale() == 0.25

    with pytest.raises(TypeError):
        mc.GateH().rescale_inplace(0.5)
