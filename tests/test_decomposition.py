import pytest
from symengine import Matrix, eye, zeros
from mimiqcircuits import *
from functools import reduce


# Parameters
alpha = 1.233
beta = 2.53
theta = 9.0927
gamma = 8894.444


@pytest.mark.parametrize(
    "class_instance",
    [
        GateH(),
        GateZ(),
        GateY(),
        GateT(),
        GateX(),
        GateS(),
        GateP(1.1),
        GateSX(),
        GateID(),
        GateU(alpha, beta, gamma, theta),
        GateU3(beta, alpha, gamma),
        GateU2(gamma, theta),
        GateU1(theta),
        GateR(alpha, beta),
        GateRX(alpha),
        GateRY(theta),
        GateRZ(beta),
    ],
)
def test_decomposition_oneq(class_instance, tolerance=1e-3):
    decomposition = class_instance.decompose().instructions

    # Extract matrices from instructions
    matrices = reversed(
        [
            Matrix(instruction.operation.matrix().tolist())
            for instruction in decomposition
        ]
    )

    # Perform matrix multiplication using map-reduce
    result_matrix = reduce(lambda x, y: x * y, matrices)

    main_matrix = Matrix(class_instance.matrix().tolist())

    # Check element-wise equality within tolerance
    for i in range(main_matrix.rows):
        for j in range(main_matrix.cols):
            assert abs(main_matrix[i, j] - result_matrix[i, j]) < tolerance


@pytest.mark.parametrize(
    "class_instance",
    [
        GateSWAP(),
        GateCH(),
        GateECR(),
        GateCP(alpha),
        GateXXminusYY(alpha, beta),
        GateXXplusYY(alpha, theta),
        GateCRX(theta),
        GateCRY(beta),
        GateCRZ(beta),
        GateRXX(alpha),
        GateISWAP(),
        GateDCX(),
        GateRYY(beta),
        GateRZX(beta),
    ],
)
def test_decomposition_twoq(class_instance, tolerance=1e-7):
    decomposition = class_instance.decompose().instructions

    I_2 = eye(2)

    matrices = []
    for instruction in reversed(decomposition):
        gate_matrix = Matrix(instruction.operation.matrix())

        # Check if the gate matrix is a 2x2 matrix (single-qubit gate in a two-qubit operation)
        if gate_matrix.rows == 2:
            if instruction.qubits[0] == 0:
                gate_matrix = kronecker_product(gate_matrix, I_2)
            else:
                gate_matrix = kronecker_product(I_2, gate_matrix)

        # Check for two-qubit operations
        elif gate_matrix.rows == 4:
            if instruction.qubits != (0, 1):
                # Adjust matrix to account for qubit order
                gate_matrix = GateSWAP().matrix() * gate_matrix * GateSWAP().matrix()

        matrices.append(gate_matrix)

    result_matrix = reduce(lambda x, y: x * y, matrices)

    main_matrix = Matrix(class_instance.matrix().tolist())

    for i in range(main_matrix.rows):
        for j in range(main_matrix.cols):
            assert abs(main_matrix[i, j] - result_matrix[i, j]) < tolerance


def kronecker_product(A, B):
    """
    Compute the Kronecker product of two matrices A and B using SymEngine.
    """
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    C = zeros(a_rows * b_rows, a_cols * b_cols)

    for i in range(a_rows):
        for j in range(a_cols):
            C[i * b_rows : (i + 1) * b_rows, j * b_cols : (j + 1) * b_cols] = (
                A[i, j] * B
            )
    return C


def _unitary(circ, n):
    import numpy as np

    u = np.eye(2**n, dtype=complex)
    for inst in circ:
        u = np.array(inst.matrix(n).tolist(), dtype=complex) @ u
    return u


def _oracle_diagonal(nx, a, b, c, d, qubits):
    # In `matrix` qubit q holds bit n-1-q of the basis index; each register is little-endian.
    n = len(qubits)

    def reg(k, qs):
        return sum(((k >> (n - 1 - q)) & 1) << i for i, q in enumerate(qs))

    diag = []
    for k in range(2**n):
        x, y = reg(k, qubits[:nx]), reg(k, qubits[nx:])
        diag.append(-1 if a * x * y + b * x + c * y + d == 0 else 1)
    return diag


@pytest.mark.parametrize("nx, ny", [(0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize(
    "coefs",
    [(1, 0, 0, -2), (0, 1, -1, 0), (1, -1, -1, 1), (0, 0, 1, -1), (0, 1, 0, 0), (0, 0, 0, 0), (1, 2, 3, 4)],
)
def test_polynomialoracle_decomposition(nx, ny, coefs):
    import numpy as np
    from mimiqcircuits.decomposition import CanonicalRewrite

    n = nx + ny
    circ = CanonicalRewrite().decompose_step(PolynomialOracle(nx, ny, *coefs), range(n), [], [])
    assert circ.num_qubits() == n
    want = np.diag(_oracle_diagonal(nx, *coefs, list(range(n))))
    assert np.allclose(_unitary(circ, n), want)


def test_polynomialoracle_full_decomposition_keeps_width():
    import numpy as np
    from mimiqcircuits.decomposition import decompose

    circ = decompose(Circuit().push(PolynomialOracle(1, 2, 0, 1, 0, 0), 0, 1, 2))
    assert circ.num_qubits() == 3
    want = np.diag(_oracle_diagonal(1, 0, 1, 0, 0, [0, 1, 2]))
    assert np.allclose(_unitary(circ, circ.num_qubits()), want)


def test_polynomialoracle_without_solutions_keeps_width():
    from mimiqcircuits.decomposition import CanonicalRewrite

    # f = xy + 2x + 3y + 4 >= 4: no solutions, so a single GateID on the last qubit
    circ = CanonicalRewrite().decompose_step(PolynomialOracle(5, 5, 1, 2, 3, 4), range(10), [], [])
    assert len(circ) == 1
    assert isinstance(circ[0].get_operation(), GateID)
    assert circ.num_qubits() == 10


def test_polynomialoracle_full_decomposition_on_scattered_qubits():
    import numpy as np
    from mimiqcircuits.decomposition import decompose

    qubits = [2, 3, 0, 1]
    c = decompose(Circuit().push(PolynomialOracle(2, 2, 1, 0, 0, -2), *qubits))
    u = _unitary(c, 4)
    want = np.diag(_oracle_diagonal(2, 1, 0, 0, -2, qubits))
    assert np.allclose(u / u[0, 0], want)  # equal up to a global phase


def test_stim_basis_decomposes_ecr_into_cliffords():
    import numpy as np
    import mimiqcircuits as mc

    c = mc.decompose(mc.Circuit().push(mc.GateECR(), 0, 1), mc.StimBasis())
    assert all(not isinstance(i.operation, mc.GateECR) for i in c)

    m = lambda g: np.array(g.matrix().tolist(), dtype=complex)
    u = np.eye(4, dtype=complex)
    for inst in c:
        g = m(inst.operation)
        if inst.operation.num_qubits == 1:
            g = np.kron(g, np.eye(2)) if inst.qubits == (0,) else np.kron(np.eye(2), g)
        u = g @ u
    ecr = m(mc.GateECR())
    # Equal up to a global phase.
    k = np.argmax(abs(u))
    phase = ecr.flat[k] / u.flat[k]
    assert np.allclose(ecr, phase * u)


def test_symbolic_rotations_raise_decomposition_error():
    import symengine as se
    import mimiqcircuits as mc
    from mimiqcircuits.decomposition import DecompositionError

    x = se.Symbol("x")
    with pytest.raises(DecompositionError):
        mc.decompose(mc.GateRZ(x), mc.CliffordTBasis())
    with pytest.raises(DecompositionError):
        mc.decompose(mc.GateU(x, 0.1, 0.2), mc.StimBasis())


def test_special_angle_step_honours_only_cliffords():
    import numpy as np
    import mimiqcircuits as mc
    from mimiqcircuits.decomposition import DecompositionError
    from mimiqcircuits.decomposition.rules import SpecialAngleRewrite

    with pytest.raises(DecompositionError):
        SpecialAngleRewrite(only_cliffords=True).decompose_step(mc.GateRZ(np.pi / 4), [0], [], [])


def test_clifford_t_basis_decomposes_ecr_without_t_gates():
    import mimiqcircuits as mc

    c = mc.decompose(mc.Circuit().push(mc.GateECR(), 0, 1), mc.CliffordTBasis())
    assert len(c) == 4
    assert not any(isinstance(i.operation, (mc.GateT, mc.GateTDG)) for i in c)


def test_diffusion_is_its_own_inverse():
    d = Diffusion(3)
    assert d.inverse() is d
