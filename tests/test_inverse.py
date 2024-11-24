import pytest
import mimiqcircuits as mc


def test_inverse_init():
    # Test initialization of the Inverse operation
    op = mc.GateX()
    inverse_op = mc.Inverse(op)
    assert inverse_op.num_qubits == op.num_qubits
    assert inverse_op.num_bits == op.num_bits
    assert inverse_op.op == op

    # Check for invalid input types
    with pytest.raises(ValueError):
        mc.Inverse(0)
    with pytest.raises(ValueError):
        mc.Inverse(mc.BitString(0))


def test_inverse_str():
    # Test the string representation of the Inverse operation
    op = mc.GateX()
    inverse_op = mc.Inverse(op)
    assert str(inverse_op) == f"{str(op)}†"


def test_inverse_inverse():
    # Test the inverse of the Inverse operation
    op = mc.GateX()
    inverse_op = mc.Inverse(op)
    inverse_inverse_op = inverse_op.inverse()
    assert inverse_inverse_op == op


def test_inverse_matrix():
    # Test the matrix computation of the Inverse operation
    op = mc.GateX()
    inverse_op = mc.Inverse(op)
    expected_matrix = op.matrix().inv()
    assert inverse_op.matrix() == expected_matrix
