import pytest
import mimiqcircuits as mc
import numpy as np
import symengine as se


def test_control_initialization():
    # Test initialization with a non-Control operation
    gate_x = mc.GateX()
    cgate_x = mc.Control(3, gate_x)
    assert cgate_x.num_controls == 3
    assert cgate_x.num_qubits == 4
    assert isinstance(cgate_x.op, mc.GateX)

    # Test initialization with a Control operation
    gate_cx = mc.Control(2, mc.GateCX())
    cgate_cx = mc.Control(1, gate_cx)
    assert cgate_cx.num_controls == 4
    assert cgate_cx.num_qubits == 5
    assert isinstance(cgate_cx.op, mc.GateX)


def test_control_init():
    op = mc.GateX()
    # Controlled operations must have at least one control.
    with pytest.raises(ValueError):
        mc.Control(0, op)

    with pytest.raises(ValueError):
        mc.Control(-1, op)

    cgate = mc.Control(1, mc.GateX())
    assert cgate.num_qubits == 2
    assert cgate.num_bits == 0
    assert isinstance(cgate.op, mc.GateX)

    cgate = mc.Control(1, mc.GateSWAP())
    assert cgate.num_qubits == 3
    assert cgate.num_bits == 0
    assert isinstance(cgate.op, mc.GateSWAP)

    cgate = mc.Control(3, mc.GateH())
    assert cgate.num_qubits == 4
    assert cgate.num_bits == 0
    assert isinstance(cgate.op, mc.GateH)

    cgate = mc.Control(2, mc.GateSWAP())
    assert cgate.num_qubits == 4
    assert cgate.num_bits == 0
    assert cgate.op == mc.GateSWAP()

    # no operations on classical qubits are allowed to be controlled
    with pytest.raises(TypeError):
        mc.Control(2, mc.Measure())
    with pytest.raises(TypeError):
        mc.Control(1, mc.Measure())

    # barriers should not be controlled
    with pytest.raises(TypeError):
        mc.Control(2, mc.Barrier(4))
    with pytest.raises(TypeError):
        mc.Control(2, mc.Barrier(1))
    with pytest.raises(TypeError):
        mc.Control(1, mc.Barrier(1))

    with pytest.raises(TypeError):
        mc.Control(1, mc.BitState(1))

    # resets are not allowed to be controlled
    with pytest.raises(TypeError):
        mc.Control(1, mc.Reset())
    with pytest.raises(TypeError):
        mc.Control(1, mc.Reset())

# Function for testing the equality of matrices for symengine


def is_close(matrix1, matrix2, tol=1e-15):
    diff_matrix = matrix1 - matrix2
    max_diff = max(abs(entry) for entry in diff_matrix)
    return max_diff < tol


def test_control_inverse():
    for nc in [1, 2, 3]:
        for gate in [mc.GateX(), mc.GateS(), mc.GateDCX(), mc.GateRX(0.1), mc.GateXXplusYY(1.12, 2.3)]:
            cgate = mc.Control(nc, gate)
            nq = 2**(cgate.num_qubits)
            product_matrix = cgate.inverse().matrix() * cgate.matrix()
            identity_matrix = se.eye(nq)
            assert is_close(product_matrix, identity_matrix)
            assert cgate.op.inverse().matrix() == cgate.inverse().op.matrix()


def test_control_circuit_instruction():
    c = mc.Circuit()
    c.push(mc.Control(3, mc.GateX()), 1, 2, 3, 4)
    assert len(c) == 1
    assert str(
        c) == '5-qubit circuit with 1 instructions:\n└── C₃X @ q1, q2, q3, q4'


def test_decompose():
    # Test decompose for a Control gate with one control
    control = mc.Control(1, mc.GateX())
    circuit = control.decompose()
    assert len(circuit) == 1
    assert str(circuit[0]) == 'CX @ q0, q1'


def test_control_matrix():
    gates = [mc.Control(5, mc.GateCP(np.pi)).matrix(),
             mc.Control(6, mc.GateP(np.pi)).matrix()]

    assert gates[0] == gates[1]

    control_gate = mc.Control(1, mc.GateX())
    c_gate = mc.GateCX()
    g1 = control_gate.matrix()
    g2 = c_gate.matrix()

    assert control_gate.num_qubits == c_gate.num_qubits == 2
    assert g1 == g2
