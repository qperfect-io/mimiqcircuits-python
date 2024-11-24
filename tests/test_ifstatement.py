import pytest
import mimiqcircuits as mc


def test_if_statement_init():
    gate = mc.GateX()
    val = mc.BitString(2)
    if_statement = mc.IfStatement(gate, val)
    assert if_statement.op == gate
    assert if_statement.bitstring == val
    assert if_statement.num_qubits == gate.num_qubits
    assert if_statement.num_bits == len(val)


def test_if_statement_circuit_instruction():
    circuit = mc.Circuit()
    gate_x = mc.GateX()
    val = mc.BitString(2)
    if_statement = mc.IfStatement(gate_x, val)
    circuit.push(if_statement, 0, 1, 2)
    assert len(circuit) == 1
    assert circuit[0].operation.op.num_qubits == 1
