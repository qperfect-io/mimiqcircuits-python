import numpy as np
import pytest
import mimiqcircuits as mc
import symengine as se
from scipy.linalg import qr


# Function for testing the equality of matrices for symengine
def is_close(matrix1, matrix2, tol=1e-15):
    diff_matrix = matrix1 - matrix2
    max_diff = max(abs(entry) for entry in diff_matrix)
    return max_diff < tol


def _checknpgate(gatetype, n, invtype=None):
    gatetype_instance = gatetype()
    invtype_instance = invtype() if invtype else gatetype_instance.inverse()
    assert isinstance(gatetype_instance.inverse(), type(invtype_instance))
    assert isinstance(gatetype_instance.inverse().inverse(), type(gatetype_instance))
    assert gatetype_instance.num_qubits == n
    m = gatetype_instance.matrix()
    mi = gatetype_instance.inverse().matrix()
    id = m * mi
    assert id.rows == id.cols
    identity_matrix = se.eye(id.rows)
    assert is_close(identity_matrix, id)


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
    assert isinstance(gatetype_instance.inverse().inverse(), type(gatetype_instance))
    assert gatetype_instance.num_qubits == n
    m = gatetype_instance.matrix()
    mi = gatetype_instance.inverse().matrix()
    id = m * mi
    assert id.rows == id.cols
    identity_matrix = se.eye(id.rows)
    assert is_close(identity_matrix, id)


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
    _checknpgate(
        lambda: mc.Power(mc.GateX(), 1 / 2), 1, lambda: mc.Inverse(mc.GateSX())
    )


def test_GateSXDG():
    _checknpgate(mc.GateSXDG, 1, lambda: mc.GateSX())


def test_GateCX():
    _checknpgate(
        lambda: mc.Control(1, mc.GateX()),
        2,
        lambda: (mc.Control(1, mc.GateX()).inverse()),
    )


def test_GateCY():
    _checknpgate(
        lambda: mc.Control(1, mc.GateY()),
        2,
        lambda: (mc.Control(1, mc.GateY()).inverse()),
    )


def test_GateCZ():
    _checknpgate(
        lambda: mc.Control(1, mc.GateZ()),
        2,
        lambda: (mc.Control(1, mc.GateZ()).inverse()),
    )


def test_GateCH():
    _checknpgate(
        lambda: mc.Control(1, mc.GateH()),
        2,
        lambda: (mc.Control(1, mc.GateH()).inverse()),
    )


def test_GateCS():
    _checknpgate(
        lambda: mc.Control(1, mc.GateS()), 2, lambda: (mc.Control(1, mc.GateSDG()))
    )


def test_GateCSDG():
    _checknpgate(
        lambda: mc.Control(1, mc.GateS()), 2, lambda: (mc.Control(1, mc.GateSDG()))
    )


def test_GateCSX():
    _checknpgate(
        lambda: mc.Control(1, mc.GateSX()), 2, lambda: (mc.Control(1, mc.GateSXDG()))
    )


def test_GateCSXDG():
    _checknpgate(
        lambda: mc.Control(1, mc.GateSXDG()), 2, lambda: (mc.Control(1, mc.GateSX()))
    )


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
    _check_param_gate(
        lambda theta, phi, lamda, gamma: mc.Control(1, mc.GateU(theta, phi, lamda)),
        2,
        [theta, phi, lmbda, gamma],
        lambda theta, phi, lamda, gamma: mc.Control(
            1, mc.GateU(theta, phi, lamda).inverse()
        ),
    )


def test_GateCP():
    _check_param_gate(
        lambda lmbda: mc.Control(1, mc.GateP(lmbda)),
        2,
        [lmbda],
        lambda lmbda: mc.Control(1, mc.GateP(lmbda).inverse()),
    )


def test_GateCR():
    _check_param_gate(
        lambda theta, phi: mc.Control(1, mc.GateR(theta, phi)),
        2,
        [theta, phi],
        lambda theta, phi: mc.Control(1, mc.GateR(theta, phi).inverse()),
    )


def test_GateCRX():
    _check_param_gate(
        lambda theta: mc.Control(1, mc.GateRX(theta)),
        2,
        [theta],
        lambda theta: mc.Control(1, mc.GateRX(theta).inverse()),
    )


def test_GateCRY():
    _check_param_gate(
        lambda theta: mc.Control(1, mc.GateRY(theta)),
        2,
        [theta],
        lambda theta: mc.Control(1, mc.GateRY(theta).inverse()),
    )


def test_GateCRZ():
    _check_param_gate(
        lambda theta: mc.Control(1, mc.GateRZ(theta)),
        2,
        [theta],
        lambda theta: mc.Control(1, mc.GateRZ(theta).inverse()),
    )


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

    # Generate a random unitary matrix using SciPy's qr decomposition
    mat, _ = qr(np.random.rand(M, M) + 1j * np.random.rand(M, M))
    sym_mat = se.Matrix(mat.tolist())

    gate = mc.GateCustom(sym_mat)

    assert isinstance(gate, mc.GateCustom)
    assert is_close(gate.matrix(), sym_mat)


def test_GateCustom():
    # the qubit count used to be derived with a shift that only happened to be
    # right for 2 and 3 qubits, so anything wider was rejected
    for N in range(1, 6):
        _check_custom_gate(N)
        assert mc.GateCustom(np.eye(2**N, dtype=complex)).num_qubits == N


def test_GateCustomDiagonal():
    d = np.exp(2j * np.pi * np.random.rand(8))
    gate = mc.GateCustomDiagonal(d)

    assert gate.num_qubits == 3
    assert np.allclose(gate.unwrappeddiagonal(), d)
    assert np.allclose(gate.unwrappedmatrix(), np.diag(d))
    assert np.allclose(
        np.array(gate.matrix().tolist(), dtype=complex), np.diag(d)
    )
    assert np.allclose(gate.inverse().unwrappeddiagonal(), np.conj(d))

    # equal whatever numeric type the entries came in as
    assert mc.GateCustomDiagonal([1, -1]) == mc.GateCustomDiagonal([1.0, -1.0])
    assert mc.GateCustomDiagonal([1, -1]) != mc.GateCustomDiagonal([1, 1])

    # only phases are unitary, and only 2**N of them
    with pytest.raises(ValueError):
        mc.GateCustomDiagonal([1, 2])
    with pytest.raises(ValueError):
        mc.GateCustomDiagonal([1, 1, 1])


def test_GateCustomDiagonal_decomposes_into_parity_rotations():
    # the rewrite is exact, global phase included — which the dense QSD route
    # applied to a diagonal matrix is not
    from mimiqcircuits.matrices import reorder_qubits_matrix

    d = np.exp(2j * np.pi * np.array([0.1, 0.4, 0.7, 0.9]))
    c = mc.Circuit()
    c.push(mc.GateCustomDiagonal(d), 0, 1)

    u = np.eye(4, dtype=complex)
    for inst in c.decompose():
        m = reorder_qubits_matrix(inst.operation.matrix(), list(inst.qubits), 2)
        u = np.array([[complex(m[i, j]) for j in range(4)] for i in range(4)]) @ u
    assert np.allclose(u, np.diag(d), atol=1e-10)


def test_GateCustomDiagonal_symbolic_entries():
    x = se.Symbol("x")
    gate = mc.GateCustomDiagonal([1, se.exp(se.I * x)])

    with pytest.raises(TypeError):
        gate.unwrappeddiagonal()

    assert gate.evaluate({x: 0}) == mc.GateCustomDiagonal([1, 1])


def test_GateCustom_matrix_is_a_method():
    # __init__ used to assign `self.matrix`, which shadowed the method of the
    # same name: `matrix()` was dead code and `Instruction.matrix()` raised
    # "MutableDenseMatrix object is not callable" for every fused block.
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    gate = mc.GateCustom(X)

    assert callable(gate.matrix)
    assert np.allclose(np.array(gate.matrix().tolist(), dtype=complex), X)

    embedded = mc.Instruction(gate, (0,)).matrix(2)
    assert np.allclose(
        np.array(embedded.tolist(), dtype=complex), np.kron(X, np.eye(2))
    )


def test_GateCustom_does_not_share_a_class_level_matrix_cache():
    # AbstractOperator.matrix()/unwrappedmatrix() memoise on the class for
    # parameter-free operators. GateCustom must keep its own overrides, or every
    # instance would hand back whichever matrix was built first.
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    a, b = mc.GateCustom(X), mc.GateCustom(Z)
    assert np.allclose(a.unwrappedmatrix(), X)
    assert np.allclose(b.unwrappedmatrix(), Z)
    assert np.allclose(np.array(a.matrix().tolist(), dtype=complex), X)
    assert np.allclose(np.array(b.matrix().tolist(), dtype=complex), Z)


def test_GateCustom_unitarity_tolerance():
    # numeric matrices are checked with NumPy; it has to agree with the
    # element-wise 1e-8 tolerance the symbolic check used, on both sides of it
    eye = np.eye(4, dtype=complex)
    assert mc.GateCustom(eye).num_qubits == 2

    near = eye.copy()
    near[0, 0] = 1 + 1e-10
    assert mc.GateCustom(near).num_qubits == 2

    far = eye.copy()
    far[0, 0] = 1 + 1e-4
    with pytest.raises(ValueError):
        mc.GateCustom(far)

    with pytest.raises(ValueError):
        mc.GateCustom(np.array([[1, 1], [0, 1]], dtype=complex))


def test_GateCustom_wide_matrix_is_cheap():
    # the unitarity check used to run as a dense symbolic product, which made an
    # 8-qubit block take seconds to construct
    assert mc.GateCustom(np.eye(256, dtype=complex)).num_qubits == 8


def test_GateCustom_rejects_bad_sizes():
    for rows in (1, 3, 6):
        with pytest.raises(ValueError):
            mc.GateCustom(np.eye(rows, dtype=complex))
