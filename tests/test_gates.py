import numpy as np

from mimiqcircuits.gates import GateX, GateY, GateZ
from mimiqcircuits.gates import GateCX, GateCY, GateCZ


def _checknpgate(gatetype, n, invtype=None):
    if invtype is None:
        invtype = gatetype
    assert isinstance(gatetype().inverse(), invtype)
    assert isinstance(gatetype().inverse().inverse(), gatetype)
    assert gatetype().num_qubits == n
    assert isinstance(gatetype().matrix(), np.ndarray)


def test_GateX():
    _checknpgate(GateX, 1)


def test_GateY():
    _checknpgate(GateY, 1)


def test_GateZ():
    _checknpgate(GateZ, 1)


def test_GateCX():
    _checknpgate(GateCX, 2)


def test_GateCY():
    _checknpgate(GateCY, 2)


def test_GateCZ():
    _checknpgate(GateCZ, 2)
