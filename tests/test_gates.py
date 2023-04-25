import numpy as np
from mimiqcircuits.gates import *


def _checknpgate(gatetype, n, invtype=None):
    if invtype is None:
        invtype = gatetype
    assert isinstance(gatetype().inverse(), invtype)
    assert isinstance(gatetype().inverse().inverse(), gatetype)
    assert gatetype().num_qubits == n
    assert isinstance(gatetype().matrix(), np.ndarray)


def _check_param_gate(gatetype, n, params=None, invtype=None):
    if invtype is None:
        invtype = gatetype
    gate = gatetype(*params) if params is not None else gatetype()
    assert isinstance(gate.inverse(), invtype)
    assert isinstance(gate.inverse().inverse(), gatetype)
    assert gate.num_qubits == n
    assert isinstance(gate.matrix(), np.ndarray)


theta = 0.5333 * np.pi
lmbda = 0.123 * np.pi
phi = 1.739 * np.pi
gamma = 1.333 * np.pi


def test_GateX():
    _checknpgate(GateX, 1)


def test_GateY():
    _checknpgate(GateY, 1)


def test_GateZ():
    _checknpgate(GateZ, 1)


def test_GateH():
    _checknpgate(GateH, 1)


def test_GateID():
    _checknpgate(GateID, 1)


def test_GateT():
    _checknpgate(GateT, 1,GateTDG)


def test_GateTDG():
    _checknpgate(GateTDG, 1,GateT)


def test_GateS():
    _checknpgate(GateS, 1,GateSDG)


def test_GateSDG():
    _checknpgate(GateSDG, 1,GateS)
    

def test_GateSX():
    _checknpgate(GateSX, 1,GateSXDG)


def test_GateSXDG():
    _checknpgate(GateSXDG, 1,GateSX)


def test_GateCX():
    _checknpgate(GateCX, 2)


def test_GateCY():
    _checknpgate(GateCY, 2)


def test_GateCZ():
    _checknpgate(GateCZ, 2)


def test_GateCH():
    _checknpgate(GateCH, 2)


def test_GateSWAP():
    _checknpgate(GateSWAP, 2)


def test_GateISWAP():
    _checknpgate(GateISWAP, 2, GateISWAPDG)
    

def test_GateISWAPDG():
    _checknpgate(GateISWAPDG, 2, GateISWAP)


def test_GateR():
    _check_param_gate(GateR, 1, [theta, phi])


def test_GateRX():
    _check_param_gate(GateRX, 1, [theta])


def test_GateRY():
    _check_param_gate(GateRY, 1, [theta])


def test_GateRZ():
    _check_param_gate(GateRZ, 1, [theta])


def test_GateP():
    _check_param_gate(GateP, 1, [lmbda])


def test_GateU():
    _check_param_gate(GateU, 1, [theta, phi, lmbda])


def test_GateCU():
    _check_param_gate(GateCU, 2, [theta, phi, lmbda, gamma])


def test_GateCP():
    _check_param_gate(GateCP, 2, [theta])


def test_GateCR():
    _check_param_gate(GateCR, 2, [theta, phi])


def test_GateCRX():
    _check_param_gate(GateCRX, 2, [theta])


def test_GateCRY():
    _check_param_gate(GateCRY, 2, [theta])


def test_GateCRZ():
    _check_param_gate(GateCRZ, 2, [theta])
