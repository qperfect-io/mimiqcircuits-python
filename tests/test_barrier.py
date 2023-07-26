import pytest
import mimiqcircuits.barrier as mb

def test_barrier_inverse_returns_self():
    barrier = mb.Barrier()
    assert barrier.inverse() is barrier

def test_barrier_from_json_valid_json():
    json_data = {"name": "Barrier"}
    barrier = mb.Barrier.from_json(json_data)
    assert isinstance(barrier, mb.Barrier)

def test_barrier_from_json_invalid_json():
    json_data = {"name": "InvalidBarrier"}
    with pytest.raises(ValueError):
        mb.Barrier.from_json(json_data)
