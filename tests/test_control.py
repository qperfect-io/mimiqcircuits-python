import pytest
import mimiqcircuits as mc
import numpy as np


def test_control_init():
    op = mc.GateX()
    # Controlled operations must have at least one control.
    with pytest.raises(ValueError):
        mc.Control(0, op)

    with pytest.raises(ValueError):
        mc.Control(-1, op)

    cgate = mc.Control(mc.GateX())
    assert cgate.num_qubits == 2
    assert cgate.num_bits == 0
    assert isinstance(cgate.op, mc.GateX)

    cgate = mc.Control(mc.GateSWAP())
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
        mc.Control(mc.Measure())

    # barriers should not be controlled
    with pytest.raises(TypeError):
        mc.Control(2, mc.Barrier(4))
    with pytest.raises(TypeError):
        mc.Control(2, mc.Barrier(1))
    with pytest.raises(TypeError):
        mc.Control(mc.Barrier(1))

    with pytest.raises(TypeError):
        mc.Control(mc.BitState(1))

    # resets are not allowed to be controlled
    with pytest.raises(TypeError):
        mc.Control(1, mc.Reset())
    with pytest.raises(TypeError):
        mc.Control(mc.Reset())


def test_control_inverse():
    for nc in [1, 2, 3]:
        for gate in [mc.GateX(), mc.GateS(), mc.GateDCX(), mc.GateRX(0.1), mc.GateXXplusYY(1.12, 2.3)]:
            cgate = mc.Control(nc, gate)
            nq = 2**(cgate.num_qubits)
            product_matrix = np.dot(cgate.inverse().matrix(), cgate.matrix())
            identity_matrix = np.eye(nq, nq)
            assert np.allclose(product_matrix, identity_matrix)
            assert np.all(cgate.op.inverse().matrix() ==
                          cgate.inverse().op.matrix())


def test_control_to_json():
    control_gate = mc.Control(3, mc.GateX())
    json_data = control_gate.to_json()
    assert json_data['name'] == 'Control'
    assert json_data['controls'] == control_gate.num_controls


def test_control_from_json():
    control_gate = mc.Control(3, mc.GateX())
    json_data = control_gate.to_json()
    reconstructed_gate = mc.Control.from_json(json_data)
    assert reconstructed_gate.num_controls == control_gate.num_controls
    assert str(reconstructed_gate.op) == str(control_gate.op)


def test_control_circuit_instruction():
    c = mc.Circuit()
    c.push(mc.Control(3, mc.GateX()), 1, 2, 3, 4)
    assert len(c) == 1
    assert str(
        c) == '5-qubit circuit with 1 instructions:\n └── C₃X @ q1, q2, q3, q4'
