import pytest
import mimiqcircuits as mc
import sympy as sp
import symengine as se


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
        mc.Power(2, mc.BitState(0))


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
    Power_op = mc.Power(op, 1/2)
    circuit = Power_op.decompose()
    assert len(circuit) == 1
    assert str(circuit) == '1-qubit circuit with 1 instructions:\n└── X^(1/2) @ q0'


def test_inverse():
    # Test the inverse of the Power operation
    op = mc.GateX()
    powerop = op.power(1/4)
    inverse_op = powerop.inverse()
    assert inverse_op.op.exponent == 1/4
    assert str(inverse_op) == '(X^(1/4))†'
