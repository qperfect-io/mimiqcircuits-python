import pytest
import mimiqcircuits as mc


def test_parallel_init():
    op = mc.GateX()
    with pytest.raises(ValueError):
        # Parallel operations must have at least two repeats.
        mc.Parallel(1, op)


def test_parallel_inverse():
    parallel_gate = mc.Parallel(3, mc.GateX())
    inverse = parallel_gate.inverse()
    assert inverse.num_repeats == parallel_gate.num_repeats
    assert str(inverse.op) == "X"  # Inverse of GateX() is itself


def test_parallel_circuit_instruction():
    c = mc.Circuit()
    parallel_gate = mc.Parallel(3, mc.GateX())
    c.push(parallel_gate, 1, 2, 3)
    assert len(c) == 1
    assert c[0].operation.op == mc.GateX()
