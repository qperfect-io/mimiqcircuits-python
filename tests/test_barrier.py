import pytest
import mimiqcircuits as mc


def test_barrier_inverse_returns_self():
    barrier = mc.Barrier(1)
    assert barrier.inverse() is barrier
