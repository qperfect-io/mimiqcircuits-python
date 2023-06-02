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
    c.remove(0)
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
    c.remove(1)
    assert len(c) == 2

    c = mc.Circuit()
    c.add_gate(mc.GateRZ(np.pi / 2), 0)
    c.add_gate(mc.GateRY(np.pi), 1)
    c.add_gate(mc.GateCX(), 0, 1)
    c.add_gate(mc.GateR(np.pi, np.pi), 1)
    assert c.num_qubits() == 2
    assert len(c) == 4
    c.remove(2)
    assert len(c) == 3
    gates = [gc.operation for gc in c]
    assert gates == [
        mc.GateRZ(np.pi / 2), mc.GateRY(np.pi), mc.GateR(np.pi, np.pi)]

    circuit = mc.Circuit()
    circuit.add_gate(mc.GateX(), 0)
    circuit.add_gate(mc.GateCH(), 0, 1)
    assert circuit.depth() == 2

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


def test_emptyCircuit():
    c = mc.Circuit()
    assert c.empty()
    assert len(c) == 0

    assert c.num_qubits() == 0
    assert c.num_bits() == 0

    # try to iterate over an empty circuit
    l = 0
    for g in c:
        l += 1

    assert l == 0


def test_constructCircuitFromInstructions():
    insts = [
        mc.Instruction(mc.GateH(), (0,)),
        mc.Instruction(mc.GateX(), (1,)),
        mc.Instruction(mc.GateCX(), (0, 1))
    ]

    c = mc.Circuit(insts)

    assert len(c) == 3
    for x in zip(c, insts):
        assert x[0] == x[1]

    assert c.num_qubits() == 2
    assert c.num_bits() == 0


def test_failconstructCircuitFromInstructions_1():
    insts = [
        mc.Instruction(mc.GateH(), (0,)),
        mc.GateX(),
        mc.Instruction(mc.GateCX(), (0, 1))
    ]

    with pytest.raises(TypeError):
        c = mc.Circuit(insts)


def test_failconstructCircuitFromInstructions_1():
    insts = mc.Instruction(mc.GateH(), (0,))

    with pytest.raises(TypeError):
        c = mc.Circuit(insts)


def test_addingGates():
    c = mc.Circuit()

    c.add_gate(mc.GateH(), 0)

    assert len(c) == 1
    assert c[-1] == mc.Instruction(mc.GateH(), (0,))
    assert c.num_qubits() == 1

    c.add_gate(mc.GateCX(), 23, 34)

    assert len(c) == 2
    assert c[-1] == mc.Instruction(mc.GateCX(), (23, 34))
    assert c.num_qubits() == 35
    assert c.num_bits() == 0

    c.add_gate(mc.GateRZ(np.pi / 2), 0)
    c.add_gate(mc.GateRY(np.pi), 2)
    c.add_gate(mc.GateCX(), 0, 2)
    c.add_gate(mc.GateR(np.pi, np.pi), 1)

    assert c.num_qubits() == 35
    assert c.num_bits() == 0
    assert len(c) == 6
    assert isinstance(c[-1].operation, mc.GateR)

    # check that nothing else changed by adding
    assert c[0] == mc.Instruction(mc.GateH(), (0,))
    assert c[1] == mc.Instruction(mc.GateCX(), (23, 34))

    # should not have negative targets
    with pytest.raises(ValueError):
        c.add_gate(mc.GateCH(), -1, 2)

    with pytest.raises(ValueError):
        c.add_gate(mc.GateCH(), 1, -2)

    # should not add a gate with two equal qubits
    with pytest.raises(ValueError):
        c.add_gate(mc.GateCH(), 1, 1)


def test_remove():
    c = mc.Circuit()

    for i in range(4):
        c.add_gate(mc.GateH(), i)

    c.add_barrier()
    c.add_gate(mc.GateCX(), 0, 1)
    c.add_gate(mc.GateCX(), 1, 2)


def test_circuitDepth():
    c = mc.Circuit()
    assert c.depth() == 0

    c.add_gate(mc.GateH(), 0)
    assert c.depth() == 1

    c.add_gate(mc.GateH(), 23)
    assert c.depth() == 1

    # barrier should not increase the depth
    c. add_barrier()
    assert c.depth() == 1

    c.add_gate(mc.GateH(), 22)
    assert c.depth() == 1

    c.add_gate(mc.GateH(), 23)
    assert c.depth() == 2

    # barrier should not increase the depth
    # no matter how many qubits it has
    c.add_barrier(0, 1, 2, 3)
    assert c.depth() == 2

    c.add_gate(mc.GateH(), 2)
    assert c.depth() == 2

    c.add_gate(mc.GateCX(), 2, 3)
    assert c.depth() == 2

    c.add_gate(mc.GateH(), 2)
    assert c.depth() == 3


def test_appendCircuitToCircuit():
    c = mc.Circuit()
    c.add_gate(mc.GateH(), 0)
    c.add_gate(mc.GateH(), 1)
    c.add_barrier()
    c.add_gate(mc.GateCX(), 0, 1)

    assert c.num_qubits() == 2
    assert len(c) == 4
    assert c.depth() == 2

    c2 = mc.Circuit()
    c2.add_gate(mc.GateT(), 1)
    c2.add_gate(mc.GateT(), 2)
    c2.add_gate(mc.GateT(), 3)
    c2.add_barrier()
    c2.add_gate(mc.GateCH(), 1, 2)

    assert c2.num_qubits() == 4
    assert len(c2) == 5
    assert c2.depth() == 2

    c.append(c2)

    assert c.num_qubits() == 4
    assert c.depth() == 4

    assert c[0].operation == mc.GateH()
    assert c[4].operation == mc.GateCX()
    assert c[5].operation == mc.GateT()
    assert c[-1].operation == mc.GateCH()


def test_appendInstructionsToCircuit():
    c = mc.Circuit()
    c.add_gate(mc.GateH(), 0)
    c.add_gate(mc.GateH(), 1)

    assert c.num_qubits() == 2
    assert len(c) == 2

    to_add = [mc.Instruction(mc.Barrier(), (0, 1)),
              mc.Instruction(mc.GateT(), (2,)),
              mc.Instruction(mc.GateCX(), (0,1)),
              mc.Instruction(mc.GateCH(), (0,2))]

    c.append_instructions(to_add)

    assert c.num_qubits() == 3
    assert len(c) == 2 + len(to_add)
    assert c[-1] == to_add[-1]


def test_inverseCircuit():
    c = mc.Circuit()
    c.add_gate(mc.GateH(), 0)
    c.add_gate(mc.GateSX(), 0)
    c.add_gate(mc.GateX(), 0)

    c2 = c.inverse()

    assert c2[-1] == mc.Instruction(mc.GateH(), (0,))
    assert c2[0] == mc.Instruction(mc.GateX(), (0,))
    assert c2[2] == mc.Instruction(mc.GateSXDG(), (0,))

