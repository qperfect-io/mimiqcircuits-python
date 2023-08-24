import pytest
import mimiqcircuits.barrier as mb

def test_barrier_inverse_returns_self():
    barrier = mb.Barrier()
    assert barrier.inverse() is barrier

def test_barrier_from_json_valid_json():
    num_qubits = 2 
    json_data = {"name": "Barrier", "num_qubits": num_qubits}
    barrier = mb.Barrier.from_json(json_data)
    assert isinstance(barrier, mb.Barrier)
    assert barrier._num_qubits == num_qubits

def test_barrier_from_json_invalid_json():
    json_data = {"name": "InvalidBarrier"}
    with pytest.raises(ValueError):
        mb.Barrier.from_json(json_data)
