import pytest
import mimiqcircuits as mc

def test_if_statement_init():
    gate = mc.GateX()
    bit_state = mc.BitState(1)
    if_statement = mc.IfStatement(gate, bit_state)
    assert if_statement.op == gate
    assert if_statement.val == bit_state
    assert if_statement.num_qubits == gate.num_qubits
    assert if_statement.num_bits == len(bit_state)

def test_if_statement_inverse():
    gate_x = mc.GateX()
    bit_state = mc.BitState(1)
    if_statement = mc.IfStatement(gate_x, bit_state)
    inverse = if_statement.inverse()
    assert inverse.op == gate_x.inverse()
    assert inverse.val == bit_state

def test_if_statement_to_json():
    gate_x = mc.GateX()
    bit_state = mc.BitState(1)
    if_statement = mc.IfStatement(gate_x, bit_state)
    json_data = if_statement.to_json()
    assert json_data['name'] == 'If'
    assert json_data['op'] == gate_x.to_json()
    assert json_data['val'] == str(bit_state.bits.to01())

def test_if_statement_from_json():
    gate_cx = mc.GateCX()
    bit_state = mc.BitState(2, [0, 1])
    if_statement = mc.IfStatement(gate_cx, bit_state)
    json_data = if_statement.to_json()
    reconstructed_statement = mc.IfStatement.from_json(json_data)
    assert str(reconstructed_statement) == str(if_statement)

def test_if_statement_circuit_instruction():
    circuit = mc.Circuit()
    gate_x = mc.GateX()
    bit_state = mc.BitState(1)
    if_statement = mc.IfStatement(gate_x, bit_state)
    circuit.push(if_statement, 0, 0)
    assert len(circuit) == 1
    assert str(circuit) == '1-qubit circuit with 1 instructions:\n └── If(X, bs0) @ q0, c0'
