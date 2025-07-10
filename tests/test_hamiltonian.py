import pytest
import mimiqcircuits as mc
import tempfile
import os


def test_hamiltonian_term_single_qubit():
    p = mc.PauliString("X")
    term = mc.HamiltonianTerm(1.5, p, (1,))
    assert term.get_coefficient() == 1.5
    assert term.get_operation() == p
    assert term.get_qubits() == (1,)


def test_hamiltonian_term_multi_qubit():
    p = mc.PauliString("XY")
    term = mc.HamiltonianTerm(2.0, p, (1, 2))
    assert term.get_coefficient() == 2.0
    assert term.get_operation() == p
    assert term.get_qubits() == (1, 2)


def test_hamiltonian_term_proto_roundtrip():
    from mimiqcircuits.proto.hamiltonianproto import (
        toproto_hamiltonian_term,
        fromproto_hamiltonian_term,
    )

    p = mc.PauliString("XY")
    term = mc.HamiltonianTerm(2.0, p, (1, 2))
    proto = toproto_hamiltonian_term(term)
    restored = fromproto_hamiltonian_term(proto)
    assert isinstance(restored, mc.HamiltonianTerm)
    assert restored.get_operation() == p
    assert restored.get_coefficient() == 2.0
    assert restored.get_qubits() == (1, 2)


def test_hamiltonian_basic_usage():
    h = mc.Hamiltonian()
    assert len(h) == 0
    assert h.num_qubits() == 0

    h.push(1.0, mc.PauliString("X"), 0)
    h.push(2.0, mc.PauliString("Y"), 1)
    assert len(h) == 2
    assert h.num_qubits() == 2


def test_hamiltonian_indexing_iteration():
    h = mc.Hamiltonian()
    terms = [
        mc.HamiltonianTerm(1.0, mc.PauliString("X"), (0,)),
        mc.HamiltonianTerm(2.0, mc.PauliString("Y"), (1,)),
        mc.HamiltonianTerm(3.0, mc.PauliString("Z"), (2,)),
    ]
    for t in terms:
        h.add_terms(t)

    assert h[0] == terms[0]
    assert list(h) == terms


def test_hamiltonian_proto_roundtrip():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("Z"), 0)
    h.push(-0.5, mc.PauliString("XY"), 1, 2)

    from mimiqcircuits.proto.hamiltonianproto import (
        toproto_hamiltonian,
        fromproto_hamiltonian,
    )

    proto = toproto_hamiltonian(h)
    restored = fromproto_hamiltonian(proto)

    assert isinstance(restored, mc.Hamiltonian)
    assert len(restored) == len(h)
    assert restored[1].get_qubits() == (1, 2)
    assert restored[1].get_coefficient() == -0.5


def test_hamiltonian_save_load_proto():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("X"), 0)
    h.push(0.75, mc.PauliString("ZZ"), 1, 2)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pb") as tmp:
        path = tmp.name
    h.saveproto(path)
    assert os.path.isfile(path)

    loaded = mc.Hamiltonian.loadproto(path)
    assert isinstance(loaded, mc.Hamiltonian)
    assert len(loaded) == 2
    os.remove(path)


def test_push_expval_basics():
    h = mc.Hamiltonian()
    for i in range(4):
        h.push(1.45, mc.PauliString("Z"), i)
        for j in range(i + 1, 4):
            h.push(1.12, mc.PauliString("XX"), i, j)
            h.push(1.23, mc.PauliString("YY"), i, j)
            h.push(1.34, mc.PauliString("ZZ"), i, j)

    c = mc.Circuit()
    c.push_expval(h, 0, 1, 2, 3)

    assert any(isinstance(inst.operation, mc.ExpectationValue) for inst in c)
    assert any(isinstance(inst.operation, mc.Multiply) for inst in c)
    assert any(isinstance(inst.operation, mc.Add) for inst in c)


def test_push_expval_qubit_mismatch():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("ZZ"), 0, 1)
    c = mc.Circuit()
    with pytest.raises(ValueError):
        c.push_expval(h, 0)  # Wrong number of qubits


def test_push_lietrotter():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("XX"), 0, 1)
    h.push(0.5, mc.PauliString("Y"), 0)
    c = mc.Circuit()
    c.push_lietrotter(h, (0, 1), t=1.0, steps=2)

    assert len(c) == 2
    for inst in c:
        assert isinstance(inst.operation, mc.GateCall)
        assert inst.operation._decl.name == "trotter"


def test_push_lietrotter_qubit_mismatch():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("XX"), 0, 1)
    with pytest.raises(ValueError):
        mc.Circuit().push_lietrotter(h, (0,), t=1.0, steps=1)


def test_push_suzukitrotter():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("XX"), 0, 1)
    h.push(0.5, mc.PauliString("Y"), 0)
    c = mc.Circuit()
    c.push_suzukitrotter(h, (0, 1), t=1.0, steps=2, order=2)

    assert len(c) == 2
    for inst in c:
        assert isinstance(inst.operation, mc.GateCall)
        assert "suzukitrotter" in inst.operation._decl.name


def test_push_suzuki_order_and_qubit_errors():
    h = mc.Hamiltonian()
    h.push(1.0, mc.PauliString("XX"), 0, 1)

    with pytest.raises(ValueError):
        mc.Circuit().push_suzukitrotter(h, (0, 1), t=1.0, steps=2, order=3)

    with pytest.raises(ValueError):
        mc.Circuit().push_suzukitrotter(h, (0,), t=1.0, steps=2, order=2)
