import mimiqcircuits as mc


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
        assert gc.gate == mc.GateCX()
