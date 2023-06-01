import mimiqcircuits as mc
import numpy as np
import pytest


def test_Circuit():
    c = mc.Circuit()

    assert c.num_qubits() == 0
    assert c.empty()

    c.add_gate(mc.GateCX(), 3, 2)

    assert not c.empty()
    assert c.num_qubits() == 4
    assert len(c) == 1

    c.add_gate(mc.GateCX(), 0, 7)
    assert c.num_qubits() == 8
    assert len(c) == 2

    for gc in c:
        assert gc.operation == mc.GateCX()

    c = mc.Circuit()
    c.add_gate(mc.GateR(np.pi, np.pi), 0)
    c.remove_gate(0)
    assert c == mc.Circuit()
    assert c.empty()

    c = mc.Circuit()
    assert c.num_qubits() == 0
    assert c.empty()

    c.add_gate(mc.GateH(), 0)
    c.add_gate(mc.GateCX(), 0, 1)
    c.add_gate(mc.GateX(), 2)

    assert not c.empty()
    assert c.num_qubits() == 3
    assert len(c) == 3

    gates = [gc.operation for gc in c]
    assert gates == [mc.GateH(), mc.GateCX(), mc.GateX()]
    c.remove_gate(1)
    assert len(c) == 2

    c = mc.Circuit()
    c.add_gate(mc.GateRZ(np.pi / 2), 0)
    c.add_gate(mc.GateRY(np.pi), 1)
    c.add_gate(mc.GateCX(), 0, 1)
    c.add_gate(mc.GateR(np.pi, np.pi), 1)
    assert c.num_qubits() == 2
    assert len(c) == 4
    c.remove_gate(2)
    assert len(c) == 3
    gates = [gc.operation for gc in c]
    assert gates == [
        mc.GateRZ(np.pi / 2), mc.GateRY(np.pi), mc.GateR(np.pi, np.pi)]

    circuit = mc.Circuit()
    circuit.add_gate(mc.GateX(), 0)
    circuit.add_gate(mc.GateCH(), 0, 1)
    assert circuit.depth() == 1

    # should not have negative targets
    with pytest.raises(ValueError):
        c = mc.Circuit()
        circuit.add_gate(mc.GateCH(), -1, 2)

    with pytest.raises(ValueError):
        c = mc.Circuit()
        circuit.add_gate(mc.GateCH(), 1, -2)

    # should not add a gate with two equal qubits
    with pytest.raises(ValueError):
        c = mc.Circuit()
        circuit.add_gate(mc.GateCH(), 1, 1)

    # instruction should only get a tuple
    with pytest.raises(TypeError):
        mc.Instruction(mc.GateCH(), [0, 1, 2])

    # should not build instructions with gates and classical bits
    with pytest.raises(ValueError):
        mc.Instruction(mc.GateCH(), (1, 0), (0, 1))
