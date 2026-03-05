import numpy as np
import pytest
import mimiqcircuits as mc


def test_gc_1q_decomp():
    U = np.array([[0, 1], [1, 0]], dtype=complex)
    g = mc.GateCustom(U)

    dec = g.decompose()
    assert len(dec) == 1
    assert isinstance(dec[0].operation, mc.GateU)
    assert dec[0].qubits == (0,)


def test_gc_2q_decomp():
    U = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dtype=complex,
    )
    g = mc.GateCustom(U)

    dec = g.decompose()
    assert len(dec) > 0
    assert all(not isinstance(inst.operation, mc.GateCustom) for inst in dec)
    assert all(q in (0, 1) for inst in dec for q in inst.qubits)


def test_gc_map_decomp():
    U = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dtype=complex,
    )
    g = mc.GateCustom(U)

    c = mc.Circuit()
    c.push(g, 2, 0)
    dec = c.decompose()

    assert len(dec) > 0
    assert all(not isinstance(inst.operation, mc.GateCustom) for inst in dec)
    used_qubits = {q for inst in dec for q in inst.qubits}
    assert used_qubits.issubset({0, 2})
    assert used_qubits == {0, 2}


def test_gc_bad_arity():
    g = mc.GateCustom(np.eye(4, dtype=complex))
    with pytest.raises(ValueError):
        g._decompose(mc.Circuit(), (0,), (), ())
