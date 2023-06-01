import numpy as np
import mimiqcircuits.gates as mg

def _checknpgate(gatetype, n, invtype=None):
    if invtype is None:
        invtype = gatetype
    assert isinstance(gatetype().inverse(), invtype)
    assert isinstance(gatetype().inverse().inverse(), gatetype)
    assert gatetype().num_qubits == n
    assert isinstance(gatetype().matrix(), np.ndarray)
    m = gatetype().matrix()
    mi = gatetype().inverse().matrix()
    id = np.dot(m, mi)
    assert id.shape[0] == id.shape[1]
    assert np.allclose(id, np.eye(id.shape[0]))



def _check_param_gate(gatetype, n, params=None, invtype=None):
    if invtype is None:
        invtype = gatetype
    gate = gatetype(*params) if params is not None else gatetype()
    assert isinstance(gate.inverse(), invtype)
    assert isinstance(gate.inverse().inverse(), gatetype)
    assert gate.num_qubits == n
    assert isinstance(gate.matrix(), np.ndarray)
    m = gate.matrix()
    mi = gate.inverse().matrix()
    id = np.dot(m, mi)
    assert id.shape[0] == id.shape[1]
    assert np.allclose(id, np.eye(id.shape[0]))


theta = 0.5333 * np.pi
lmbda = 0.123 * np.pi
phi = 1.739 * np.pi
beta = 1.785 * np.pi
gamma = 1.333 * np.pi


def test_GateX():
    _checknpgate(mg.GateX, 1)


def test_GateY():
    _checknpgate(mg.GateY, 1)


def test_GateZ():
    _checknpgate(mg.GateZ, 1)


def test_GateH():
    _checknpgate(mg.GateH, 1)


def test_GateID():
    _checknpgate(mg.GateID, 1)


def test_GateT():
    _checknpgate(mg.GateT, 1, mg.GateTDG)


def test_GateTDG():
    _checknpgate(mg.GateTDG, 1,mg.GateT)


def test_GateS():
    _checknpgate(mg.GateS, 1, mg.GateSDG)


def test_GateSDG():
    _checknpgate(mg.GateSDG, 1, mg.GateS)
    

def test_GateSX():
    _checknpgate(mg.GateSX, 1, mg.GateSXDG)


def test_GateSXDG():
    _checknpgate(mg.GateSXDG, 1, mg.GateSX)


def test_GateCX():
    _checknpgate(mg.GateCX, 2)


def test_GateCY():
    _checknpgate(mg.GateCY, 2)


def test_GateCZ():
    _checknpgate(mg.GateCZ, 2)


def test_GateCH():
    _checknpgate(mg.GateCH, 2)


def test_GateSWAP():
    _checknpgate(mg.GateSWAP, 2)


def test_GateISWAP():
    _checknpgate(mg.GateISWAP, 2, mg.GateISWAPDG)
    

def test_GateISWAPDG():
    _checknpgate(mg.GateISWAPDG, 2, mg.GateISWAP)


def test_GateCS():
    _checknpgate(mg.GateCS, 2, mg.GateCSDG)


def test_GateCSDG():
    _checknpgate(mg.GateCSDG, 2, mg.GateCS)


def test_GateCSX():
    _checknpgate(mg.GateCSX, 2, mg.GateCSXDG)


def test_GateCSXDG():
    _checknpgate(mg.GateCSXDG, 2, mg.GateCSX)


def test_GateECR():
    _checknpgate(mg.GateECR, 2)


def test_GateDCX():
    _checknpgate(mg.GateDCX, 2, mg.GateDCXDG)


def test_GateDCXDG():
    _checknpgate(mg.GateDCX, 2, mg.GateDCXDG)


def test_GateR():
    _check_param_gate(mg.GateR, 1, [theta, phi])


def test_GateRX():
    _check_param_gate(mg.GateRX, 1, [theta])


def test_GateRY():
    _check_param_gate(mg.GateRY, 1, [theta])


def test_GateRZ():
    _check_param_gate(mg.GateRZ, 1, [theta])


def test_GateP():
    _check_param_gate(mg.GateP, 1, [lmbda])


def test_GateU():
    _check_param_gate(mg.GateU, 1, [theta, phi, lmbda])


def test_GateCU():
    _check_param_gate(mg.GateCU, 2, [theta, phi, lmbda, gamma])


def test_GateCP():
    _check_param_gate(mg.GateCP, 2, [theta])


def test_GateCR():
    _check_param_gate(mg.GateCR, 2, [theta, phi])


def test_GateCRX():
    _check_param_gate(mg.GateCRX, 2, [theta])


def test_GateCRY():
    _check_param_gate(mg.GateCRY, 2, [theta])


def test_GateCRZ():
    _check_param_gate(mg.GateCRZ, 2, [theta])


def test_GateU1():
    _check_param_gate(mg.GateU1, 1, [lmbda])


def test_GateU2():
    _check_param_gate(mg.GateU2, 1, [phi, lmbda], mg.GateU2DG)


def test_GateU2DG():
    _check_param_gate(mg.GateU2DG, 1, [phi, lmbda], mg.GateU2)


def test_GateU3():
    _check_param_gate(mg.GateU3, 1, [theta, phi, lmbda])


def test_GateRZZ():
    _check_param_gate(mg.GateRZZ, 2, [theta])


def test_GateRXX():
    _check_param_gate(mg.GateRXX, 2, [theta])


def test_GateRYY():
    _check_param_gate(mg.GateRYY, 2, [theta])


def test_GateRZX():
    _check_param_gate(mg.GateRZX, 2, [theta])


def test_GateRXZ():
    _check_param_gate(mg.GateRXZ, 2, [theta])


def test_GateXXplusYY():
    _check_param_gate(mg.GateXXplusYY, 2, [theta, beta])


def test_GateXXminusYY():
    _check_param_gate(mg.GateXXminusYY, 2, [theta, beta])


def _check_custom_gate(N):
    M = 2**N
    mat = np.random.rand(M, M)
    gate = mg.GateCustom(mat)

    assert isinstance(gate, mg.GateCustom)
    assert np.array_equal(gate.matrix, mat)


def test_GateCustom():
    N = 2
    _check_custom_gate(N)


def test_cis():
    x = 0.5
    result = mg.cis(x)
    expected = np.exp(1j * 0.5)
    assert np.isclose(result, expected)


def test_decomplex():
    m = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    result = mg._decomplex(m)
    expected = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    assert np.allclose(result, expected)


def test_cispi():
    x = 0.5
    result = mg.cispi(x)
    expected = np.exp(1j * np.pi * 0.5)
    assert np.isclose(result, expected)


def test_pmatrixpi():
    lmbda = 0.5
    result = mg.pmatrixpi(lmbda)
    expected = np.array([[1, 0], [0, np.exp(1j * np.pi * lmbda)]])
    assert np.allclose(result, expected)


def test_pmatrix():
    lmbda = 0.5
    result = mg.pmatrix(lmbda)
    expected = np.array([[1, 0], [0, np.exp(1j * lmbda)]])
    assert np.allclose(result, expected)


def test_ctrl_fs():
    mat = np.array([[1, 2], [3, 4]])
    result = mg.ctrl_fs(mat)
    expected = np.dot(mg.ctrl(mat), mg.ctrl2(mat))
    assert np.allclose(result, expected)


def test_ctrl_sf():
    mat = np.array([[1, 2], [3, 4]])
    result = mg.ctrl_sf(mat)
    expected = np.dot(mg.ctrl2(mat), mg.ctrl(mat))
    assert np.allclose(result, expected)


def test_ctrl():
    mat = np.array([[1, 2], [3, 4]])
    dim = mat.shape[0]
    result = mg.ctrl(mat)
    expected = np.block([[np.identity(dim), np.zeros(mat.shape)], [np.zeros(mat.shape), mat]])
    assert np.allclose(result, expected)


def test_ctrl2():
    mat = np.array([[1, 2], [3, 4]])
    result = mg.ctrl2(mat)
    expected = np.block([[1, 0, 0, 0], [0, mat[0, 0], 0, mat[0, 1]], [0, 0, 1, 0], [0, mat[1, 0], 0, mat[1, 1]]])
    assert np.allclose(result, expected)


def test_gphase():
    result = mg.gphase(lmbda)
    expected = mg.cis(lmbda)
    assert np.isclose(result, expected)


def test_gphasepi():
    result = mg.gphasepi(lmbda)
    expected = mg.cispi(lmbda)
    assert np.isclose(result, expected)


def test_umatrix():
    cis = mg.cis
    result = mg.umatrix(theta, phi, lmbda, gamma)
    expected = np.array([
        [cis(gamma) * np.cos(theta/2), -cis(lmbda + gamma) * np.sin(theta/2)],
        [cis(phi + gamma) * np.sin(theta/2), cis(phi + lmbda + gamma) * np.cos(theta/2)] ])
    assert np.allclose(result, expected)


def test_umatrixpi():
    cispi = mg.cispi
    result = mg.umatrixpi(theta, phi, lmbda, gamma)
    expected = np.array([[cispi(gamma) * np.cos((theta/2) * np.pi), -cispi(lmbda + gamma) * np.sin((theta/2) * np.pi)],
        [ cispi(phi + gamma) * np.sin((theta/2) * np.pi),
            cispi(phi + lmbda + gamma) * np.cos((theta/2) * np.pi)]])
    assert np.allclose(result, expected)


def test_rmatrix():
    result = mg.rmatrix(theta, phi)
    expected = np.array([
        [np.cos(theta/2), -1j * mg.cis(-phi) * np.sin(theta/2)],
        [-1j * mg.cis(phi) * np.sin(theta/2), np.cos(theta/2)]])
    assert np.allclose(result, expected)


def test_rmatrixpi():
    result = mg.rmatrixpi(theta, phi)
    expected = np.array([
        [np.cos(theta/2 * np.pi), -1j * mg.cispi(-phi) * np.sin(theta/2 * np.pi)],
        [-1j * mg.cispi(phi) * np.sin(theta/2 * np.pi), np.cos(theta/2 * np.pi)]
    ])
    assert np.allclose(result, expected)


def test_rxmatrixpi():
    result = mg.rxmatrixpi(theta)
    expected = mg.umatrixpi(theta, -1/2, 1/2)
    assert np.allclose(result, expected)


def test_rxmatrix():
    result = mg.rxmatrix(theta)
    expected = mg.rxmatrixpi(theta / np.pi)
    assert np.allclose(result, expected)


def test_rymatrixpi():
    result = mg.rymatrixpi(theta)
    expected = mg.umatrixpi(theta, 0, 0)
    assert np.allclose(result, expected)


def test_rymatrix():
    result = mg.rymatrix(theta)
    expected = mg.rymatrixpi(theta / np.pi)
    assert np.allclose(result, expected)


def test_rzmatrixpi():
    result = mg.rzmatrixpi(lmbda)
    expected = mg.gphasepi(-lmbda / 2) * mg.umatrixpi(0, 0, lmbda)
    assert np.allclose(result, expected)


def test_rzmatrix():
    result = mg.rzmatrix(lmbda)
    expected = mg.rzmatrixpi(lmbda / np.pi)
    assert np.allclose(result, expected)
    