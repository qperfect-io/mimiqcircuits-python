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
    assert circuit[0].operation.bitstring == val
    assert circuit[0].qubits == (0,)
    assert circuit[0].bits == (1, 2)
    assert circuit.num_qubits() == 1
    assert circuit.num_bits() == 3
    assert circuit.num_zvars() == 0


def test_if_statement_block():
    c = mc.Circuit()
    c.push(mc.IfStatement(mc.GateX(), mc.BitString("1")), 0, 0)
    c.push(
        mc.IfStatement(mc.Parallel(4, mc.GateH()), mc.BitString("01")), 1, 2, 3, 4, 0, 1
    )
    c.push(mc.Measure(), 0, 0)
    c.push(mc.ExpectationValue(mc.GateX()), 0, 0)
    block = mc.Block(c)
    main = mc.Circuit()
    main.push(mc.IfStatement(block, mc.BitString("0")), 0, 1, 2, 3, 4, 0, 1, 2, 0)
    assert len(main) == 1
    assert main.num_qubits() == 5
    assert main.num_bits() == 3
    assert main.num_zvars() == 1
    assert main[0].operation.op.num_qubits == 5
    assert main[0].operation.bitstring == mc.BitString("0")
