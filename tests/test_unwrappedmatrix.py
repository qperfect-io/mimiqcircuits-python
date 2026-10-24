"""``unwrappedmatrix`` returns a ``complex128`` NumPy array for every
numeric operator and raises ``UnexpectedSymbolics`` for a symbolic one."""

import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.operations.operator import UnexpectedSymbolics


def _as_array(m):
    return np.asarray(m.tolist(), dtype=complex)


@pytest.mark.parametrize(
    "op",
    [
        mc.Operator(np.array([[1, 0], [0, 0.5]])),
        mc.LossyOperator(np.array([[1, 0], [0, 0.5]]), [1]),
    ],
)
def test_matrix_operators(op):
    m = op.unwrappedmatrix()
    assert isinstance(m, np.ndarray) and m.dtype == np.complex128
    assert np.allclose(m, _as_array(op.matrix()))

    # Writing into the result must not change the operator.
    m[0, 0] = 9
    assert op.unwrappedmatrix()[0, 0] == 1


@pytest.mark.parametrize("cls", [mc.Operator, lambda m: mc.LossyOperator(m, [1])])
def test_matrix_operators_reject_symbols(cls):
    x = se.Symbol("x")
    with pytest.raises(UnexpectedSymbolics):
        cls(se.Matrix([[x, 0], [0, 1]])).unwrappedmatrix()


# Gates whose constructor takes more than their parameters (the qubit count,
# the Pauli string), directly and inside wrappers.
@pytest.mark.parametrize(
    "gate",
    [
        mc.GateRNZ(3, 0.2),
        mc.RPauli(mc.PauliString("XZ"), 0.3),
        mc.Control(1, mc.GateRNZ(2, 0.3)),
        mc.Inverse(mc.RPauli(mc.PauliString("Y"), 0.3)),
    ],
)
def test_generalized_gates(gate):
    m = gate.unwrappedmatrix()
    assert m.dtype == np.complex128
    assert np.allclose(m, _as_array(gate.matrix()))
