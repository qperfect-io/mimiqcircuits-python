import copy
import pickle

import pytest

import mimiqcircuits as mc
from mimiqcircuits.addnoise import add_noise

# These operations return a LazyExpr from __new__ when called without
# arguments, which is how copy, deepcopy and pickle rebuild objects.
OPERATIONS = [
    mc.Barrier(1),
    mc.Barrier(3),
    mc.QubitCoordinates([1.0, 2.0]),
    mc.QubitCoordinates([]),
    mc.ShiftCoordinates([0.5, -1.0]),
    mc.ShiftCoordinates(2.0),
]


@pytest.mark.parametrize("op", OPERATIONS, ids=repr)
@pytest.mark.parametrize(
    "roundtrip",
    [copy.copy, copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_roundtrip_keeps_operation(op, roundtrip):
    new = roundtrip(op)
    assert type(new) is type(op)
    assert new == op


def test_add_noise_on_circuit_with_barrier():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Barrier(1), 0)
    noisy = add_noise(c, mc.GateH(), mc.Depolarizing(1, 0.1))
    assert [type(i.operation) for i in noisy] == [mc.GateH, mc.Depolarizing, mc.Barrier]
    assert "Barrier" in str(c.deepcopy())
