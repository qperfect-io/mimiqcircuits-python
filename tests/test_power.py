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
    Power_op = mc.Power(op, 1 / 8)
    circuit = Power_op.decompose()
    assert len(circuit) == 1
    assert Power_op.exponent == 1 / 8


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


def test_power_of_control_moves_inside_the_controls():
    # (C-U)^p = C-(U^p), so the power is taken on the target gate only.
    circ = mc.Power(mc.GateCCX(), 0.5).decompose()
    assert len(circ) == 1
    assert circ[0].operation == mc.Control(2, mc.GateSX())
    assert circ[0].qubits == (0, 1, 2)


def test_power_of_phasegradient_scales_the_phases():
    circ = mc.Power(mc.PhaseGradient(2), 0.5).decompose()
    assert [inst.qubits for inst in circ] == [(1,), (0,)]
    assert [float(inst.operation.lmbda) for inst in circ] == pytest.approx(
        [pi / 4, pi / 2]
    )


def test_power_without_single_gate_decomposition_raises():
    from mimiqcircuits.decomposition import DecompositionError

    # The canonical decomposition of ECR has several gates, so its square
    # root cannot be taken gate by gate.
    with pytest.raises(DecompositionError):
        mc.Power(mc.GateECR(), 0.5).decompose()


@pytest.mark.parametrize(
    "gate", [mc.GateU(0, 0, 0.3), mc.GateU(0, 0, 0.3, 0.2), mc.GateU(0.3, 0.7, 1.1)]
)
def test_gateu_power_matches_matrix_power(gate):
    import numpy as np
    from scipy.linalg import fractional_matrix_power

    # A diagonal GateU used to give NaN angles, since phi and lambda were
    # read from its zero off-diagonal entries.
    m = np.array(gate.matrix().tolist(), dtype=complex)
    got = np.array(gate._power(0.5).matrix().tolist(), dtype=complex)
    assert np.allclose(got, fractional_matrix_power(m, 0.5), atol=1e-10)


@pytest.mark.parametrize(
    "build",
    [
        lambda x: mc.Power(mc.GateH(), x),
        lambda x: mc.Power(mc.Power(mc.GateH(), 0.5), x),
        lambda x: mc.Power(mc.GateH(), 0.5).power(x),
        lambda x: mc.GateH().power(x),
        lambda x: mc.GateRX(0.1).power(x),
        lambda x: mc.Inverse(mc.GateH()).power(x),
        lambda x: mc.Parallel(2, mc.GateH()).power(x),
        lambda x: mc.Power(mc.GateH(), 1j),
    ],
)
def test_power_rejects_non_real_exponents(build):
    with pytest.raises(TypeError, match="real number"):
        build(se.Symbol("x"))


def test_power_accepts_numeric_symbolic_exponents():
    assert mc.Power(mc.GateH(), se.Rational(1, 2)) == mc.Power(mc.GateH(), 0.5)
