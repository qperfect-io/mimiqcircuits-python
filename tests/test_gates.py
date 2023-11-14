import numpy as np
import mimiqcircuits as mc
import symengine as se


# Function for testing the equality of matrices for symengine
def is_close(matrix1, matrix2, tol=1e-15):
    diff_matrix = matrix1 - matrix2
    max_diff = max(abs(entry) for entry in diff_matrix)
    return max_diff < tol


def _checknpgate(gatetype, n, invtype=None):
    gatetype_instance = gatetype()
    invtype_instance = invtype() if invtype else gatetype_instance.inverse()
    assert isinstance(gatetype_instance.inverse(), type(invtype_instance))
    assert isinstance(gatetype_instance.inverse().inverse(),
                      type(gatetype_instance))
    assert gatetype_instance.num_qubits == n
    m = gatetype_instance.matrix()
    mi = gatetype_instance.inverse().matrix()
    id = m * mi
    assert id.rows == id.cols
    # identity_matrix = se.eye(id.rows)
    # assert is_close(identity_matrix, id)


def _check_param_gate(gatetype, n, params=None, invtype=None):
    if params is not None:
        gatetype_instance = gatetype(*params)
    else:
        gatetype_instance = gatetype()
    if invtype is not None:
        invtype_instance = invtype(*params) if params else invtype()
    else:
        invtype_instance = gatetype_instance.inverse()
    assert isinstance(gatetype_instance.inverse(), type(invtype_instance))
    assert isinstance(gatetype_instance.inverse().inverse(),
                      type(gatetype_instance))
    assert gatetype_instance.num_qubits == n
    m = gatetype_instance.matrix()
    mi = gatetype_instance.inverse().matrix()
    id = m * mi
    assert id.rows == id.cols
    # identity_matrix = se.eye(id.rows)
    # assert is_close(identity_matrix, id)


theta = 0.5333 * np.pi
lmbda = 0.123 * np.pi
phi = 1.739 * np.pi
beta = 1.785 * np.pi
gamma = 1.333 * np.pi


def test_GateX():
    _checknpgate(mc.GateX, 1)


def test_GateY():
    _checknpgate(mc.GateY, 1)


def test_GateZ():
    _checknpgate(mc.GateZ, 1)


def test_GateH():
    _checknpgate(mc.GateH, 1)


def test_GateID():
    _checknpgate(mc.GateID, 1)


def test_GateT():
    _checknpgate(mc.GateT, 1, mc.GateTDG)


def test_GateTDG():
    _checknpgate(mc.GateTDG, 1, mc.GateT)


def test_GateS():
    _checknpgate(mc.GateS, 1, mc.GateSDG)


def test_GateSDG():
    _checknpgate(mc.GateSDG, 1, mc.GateS)


def test_GateSX():
    _checknpgate(lambda: mc.Power(mc.GateX(), 1/2), 1,
                 lambda: mc.Inverse(mc.GateSX()))


def test_GateSXDG():
    _checknpgate(mc.GateSXDG, 1, lambda: mc.GateSX())


def test_GateCX():
    _checknpgate(lambda: mc.Control(1, mc.GateX()), 2,
                 lambda: (mc.Control(1, mc.GateX()).inverse()))


def test_GateCY():
    _checknpgate(lambda: mc.Control(1, mc.GateY()), 2,
                 lambda: (mc.Control(1, mc.GateY()).inverse()))


def test_GateCZ():
    _checknpgate(lambda: mc.Control(1, mc.GateZ()), 2,
                 lambda: (mc.Control(1, mc.GateZ()).inverse()))


def test_GateCH():
    _checknpgate(lambda: mc.Control(1, mc.GateH()), 2,
                 lambda: (mc.Control(1, mc.GateH()).inverse()))


def test_GateCS():
    _checknpgate(lambda: mc.Control(1, mc.GateS()), 2,
                 lambda: (mc.Control(1, mc.GateSDG())))


def test_GateCSDG():
    _checknpgate(lambda: mc.Control(1, mc.GateS()), 2,
                 lambda: (mc.Control(1, mc.GateSDG())))


def test_GateCSX():
    _checknpgate(lambda: mc.Control(1, mc.GateSX()), 2,
                 lambda: (mc.Control(1, mc.GateSXDG())))


def test_GateCSXDG():
    _checknpgate(lambda: mc.Control(1, mc.GateSXDG()),
                 2, lambda: (mc.Control(1, mc.GateSX())))


def test_GateSWAP():
    _checknpgate(mc.GateSWAP, 2)


def test_GateISWAP():
    _checknpgate(mc.GateISWAP, 2, lambda: mc.Inverse(mc.GateISWAP()))


def test_GateECR():
    _checknpgate(mc.GateECR, 2)


def test_GateDCX():
    _checknpgate(mc.GateDCX, 2, lambda: mc.Inverse(mc.GateDCX()))


def test_GateR():
    _check_param_gate(mc.GateR, 1, [theta, phi])


def test_GateRX():
    _check_param_gate(mc.GateRX, 1, [theta])


def test_GateRY():
    _check_param_gate(mc.GateRY, 1, [theta])


def test_GateRZ():
    _check_param_gate(mc.GateRZ, 1, [lmbda])


def test_GateP():
    _check_param_gate(mc.GateP, 1, [lmbda])


def test_GateU():
    _check_param_gate(mc.GateU, 1, [theta, phi, lmbda])


def test_GateCU():
    _check_param_gate(lambda theta, phi, lamda, gamma: mc.Control(1, mc.GateU(theta, phi, lamda)), 2, [
                      theta, phi, lmbda, gamma], lambda theta, phi, lamda, gamma: mc.Control(1, mc.GateU(theta, phi, lamda).inverse()))


def test_GateCP():
    _check_param_gate(lambda lmbda: mc.Control(1, mc.GateP(lmbda)), 2, [
                      lmbda], lambda lmbda: mc.Control(1, mc.GateP(lmbda).inverse()))


def test_GateCR():
    _check_param_gate(lambda theta, phi: mc.Control(1, mc.GateR(theta, phi)), 2, [
                      theta, phi], lambda theta, phi: mc.Control(1, mc.GateR(theta, phi).inverse()))


def test_GateCRX():
    _check_param_gate(lambda theta: mc.Control(1, mc.GateRX(theta)), 2, [
                      theta], lambda theta: mc.Control(1, mc.GateRX(theta).inverse()))


def test_GateCRY():
    _check_param_gate(lambda theta: mc.Control(1, mc.GateRY(theta)), 2, [
                      theta], lambda theta: mc.Control(1, mc.GateRY(theta).inverse()))


def test_GateCRZ():
    _check_param_gate(lambda theta: mc.Control(1, mc.GateRZ(theta)), 2, [
                      theta], lambda theta: mc.Control(1, mc.GateRZ(theta).inverse()))


def test_GateU1():
    _check_param_gate(mc.GateU1, 1, [lmbda])


def test_GateU2():
    _check_param_gate(mc.GateU2, 1, [phi, lmbda])


def test_GateU3():
    _check_param_gate(mc.GateU3, 1, [theta, phi, lmbda])


def test_GateRZZ():
    _check_param_gate(mc.GateRZZ, 2, [theta])


def test_GateRXX():
    _check_param_gate(mc.GateRXX, 2, [theta])


def test_GateRYY():
    _check_param_gate(mc.GateRYY, 2, [theta])


def test_GateRZX():
    _check_param_gate(mc.GateRZX, 2, [theta])


def test_GateXXplusYY():
    _check_param_gate(mc.GateXXplusYY, 2, [theta, beta])


def test_GateXXminusYY():
    _check_param_gate(mc.GateXXminusYY, 2, [theta, beta])


def _check_custom_gate(N):
    M = 2**N
    mat = np.random.rand(M, M)
    gate = mc.GateCustom(mat)

    assert isinstance(gate, mc.GateCustom)
    assert np.array_equal(gate.matrix, mat)


def test_GateCustom():
    N = 2
    _check_custom_gate(N)
