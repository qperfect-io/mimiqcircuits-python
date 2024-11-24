import pytest
import mimiqcircuits as mc
import sympy as sp
import symengine as se
import random
from numpy import pi


def test_init():
    # Test initialization of the Power operation
    op = mc.GateX()
    Power_op = mc.Power(op, 1)
    assert Power_op.num_qubits == op.num_qubits
    assert Power_op.num_bits == 0
    assert Power_op.exponent == 1
    assert Power_op.op == op

    Power_op = mc.Power(op, 2)
    assert Power_op.num_qubits == op.num_qubits
    assert Power_op.num_bits == 0
    assert Power_op.exponent == 2
    assert Power_op.op == op

    # Check for invalid input types
    with pytest.raises(ValueError):
        mc.Power(2, 0)
    with pytest.raises(ValueError):
        mc.Power(2, mc.BitString(0))


def test_matrix():
    # Test the matrix computation of the Power operation
    op = mc.GateX()
    Power_op = mc.Power(op, 2)
    sp_matrix = sp.Matrix(op.matrix().tolist())
    expected_matrix = se.Matrix((sp_matrix**2).tolist())
    assert Power_op.matrix() == expected_matrix


def test_decompose():
    # Test decomposition of the Power operation into a circuit
    op = mc.GateX()
    Power_op = mc.Power(op, 1 / 2)
    circuit = Power_op.decompose()
    assert len(circuit) == 1
    assert Power_op.exponent == 1 / 2


def test_inverse():
    # Test the inverse of the Power operation
    op = mc.GateX()
    powerop = op.power(1 / 4)
    inverse_op = powerop.inverse()
    assert inverse_op.op.exponent == 1 / 4
    assert mc.Power(mc.GateX(), 1 / 4).op == op


def is_close(matrix1, matrix2, tol=1e-7):
    matrix1 = sp.Matrix(matrix1).evalf()
    matrix2 = sp.Matrix(matrix2).evalf()

    diff_matrix = matrix1 - matrix2

    for entry in diff_matrix:
        abs_value = sp.Abs(entry).evalf()
        if abs_value > tol:
            return False
    return True


def test_poweru():
    for _ in range(100):
        pwr = 100 * (random.random() - 0.4)
        g = mc.GateU(
            random.random() * 4 * pi,
            random.random() * 2,
            random.random() * 2 * pi,
            random.random() * 2 * pi,
        )
        assert is_close(
            sp.Matrix(g.matrix()).evalf() ** pwr,
            sp.Matrix(g.power(pwr).matrix().tolist()),
        )
