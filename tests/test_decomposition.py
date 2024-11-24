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
