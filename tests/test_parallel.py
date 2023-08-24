import pytest
import mimiqcircuits as mc

def test_parallel_init():
    op = mc.GateX()
    with pytest.raises(ValueError):
        mc.Parallel(1, op)  # Parallel operations must have at least two repeats.

def test_parallel_inverse():
    parallel_gate = mc.Parallel(3, mc.GateX())
    inverse = parallel_gate.inverse()
    assert inverse.num_repeats == parallel_gate.num_repeats
    assert str(inverse.op) == 'X'  # Inverse of GateX() is itself

def test_parallel_to_json():
    parallel_gate = mc.Parallel(3, mc.GateX())
    json_data = parallel_gate.to_json()
    assert json_data['name'] == 'Parallel'
    assert json_data['repeats'] == parallel_gate.num_repeats

def test_parallel_from_json():
    parallel_gate = mc.Parallel(3, mc.GateX())
    json_data = parallel_gate.to_json()
    reconstructed_gate = mc.Parallel.from_json(json_data)
    assert reconstructed_gate.num_repeats == parallel_gate.num_repeats
    assert str(reconstructed_gate.op) == str(parallel_gate.op)

def test_parallel_circuit_instruction():
    c = mc.Circuit()
    parallel_gate = mc.Parallel(3, mc.GateX())
    c.push(parallel_gate, 1, 2, 3)
    assert len(c) == 1
    assert str(c) == '4-qubit circuit with 1 instructions:\n └── Parallel(3, X) @ q1, q2, q3'

