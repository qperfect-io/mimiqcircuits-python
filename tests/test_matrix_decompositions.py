import numpy as np
import pytest

import mimiqcircuits as mc
from mimiqcircuits.operations.decompositions.matrix_decompositions.csd import (
    _csd_decomposition,
)
from mimiqcircuits.operations.decompositions.matrix_decompositions.qsd import (
    _qsd_decomposition,
)
from mimiqcircuits.operations.decompositions.matrix_decompositions.zyz import (
    _zyz_decomposition,
)

######### Helper functions for testing #########
def rand_u(dim, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    phases = np.where(np.abs(d) > 0, d / np.abs(d), 1.0)
    return Q * phases


def to_np(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix.astype(complex)
    return np.array(matrix.tolist(), dtype=complex)


def blkdiag(A, B):
    ar, ac = A.shape
    br, bc = B.shape
    return np.block(
        [
            [A, np.zeros((ar, bc), dtype=complex)],
            [np.zeros((br, ac), dtype=complex), B],
        ]
    )


def i2b(index, nq):
    return [(index >> (nq - 1 - i)) & 1 for i in range(nq)]


def b2i(bits):
    out = 0
    for b in bits:
        out = (out << 1) | b
    return out


def expand_inst(inst, nq):
    op = to_np(inst.operation.matrix())
    qubits = tuple(inst.qubits)
    k = len(qubits)
    dim = 2**nq
    full = np.zeros((dim, dim), dtype=complex)

    for col in range(dim):
        bits_in = i2b(col, nq)
        col_sub = b2i([bits_in[q] for q in qubits])

        for row_sub in range(2**k):
            amp = op[row_sub, col_sub]
            if np.isclose(amp, 0.0):
                continue

            bits_out = bits_in.copy()
            row_sub_bits = i2b(row_sub, k)
            for i, q in enumerate(qubits):
                bits_out[q] = row_sub_bits[i]

            row = b2i(bits_out)
            full[row, col] += amp

    return full


def circ_mat(circ, nq):
    total = np.eye(2**nq, dtype=complex)
    for inst in circ:
        total = expand_inst(inst, nq) @ total
    return total


def assert_phase_eq(actual, expected, atol=1e-8):
    overlap = np.vdot(expected.reshape(-1), actual.reshape(-1))
    if np.isclose(abs(overlap), 0.0, atol=atol):
        np.testing.assert_allclose(actual, expected, atol=atol)
        return

    phase = overlap / abs(overlap)
    np.testing.assert_allclose(actual, phase * expected, atol=atol)

########## Test cases#########

def test_zyz():
    with pytest.raises(ValueError):
        _zyz_decomposition(np.eye(4, dtype=complex))

    for seed in (0, 1, 2):
        U = rand_u(2, seed)
        theta, phi, lam, gamma = _zyz_decomposition(U)
        rec = to_np(mc.GateU(theta, phi, lam, gamma).matrix())
        np.testing.assert_allclose(rec, U, atol=1e-8)

    U0 = np.array(
        [[np.exp(1j * 0.37), 0.0], [0.0, np.exp(1j * (0.37 - 0.91))]], dtype=complex
    )
    theta, phi, lam, gamma = _zyz_decomposition(U0)
    assert np.isclose(theta, 0.0, atol=1e-10)
    assert np.isclose(phi, 0.0, atol=1e-10)
    np.testing.assert_allclose(to_np(mc.GateU(theta, phi, lam, gamma).matrix()), U0, atol=1e-8)

    Upi = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    theta, phi, lam, gamma = _zyz_decomposition(Upi)
    assert np.isclose(theta, np.pi, atol=1e-10)
    assert np.isclose(phi, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        to_np(mc.GateU(theta, phi, lam, gamma).matrix()), Upi, atol=1e-8
    )


def test_csd():
    with pytest.raises(ValueError):
        _csd_decomposition(np.eye(3, dtype=complex))

    U = rand_u(4, 7)
    L0, L1, R0, R1, theta = _csd_decomposition(U)

    L0 = to_np(L0)
    L1 = to_np(L1)
    R0 = to_np(R0)
    R1 = to_np(R1)
    theta = np.array(theta.tolist(), dtype=float).reshape(-1)

    C = np.diag(np.cos(theta))
    S = np.diag(np.sin(theta))
    rec = blkdiag(L0, L1) @ np.block([[C, -S], [S, C]]) @ blkdiag(R0, R1)
    np.testing.assert_allclose(rec, U, atol=1e-6)


def test_qsd():
    U1 = rand_u(2, 9)
    c1, p1 = _qsd_decomposition(U1)
    assert p1 == 0.0
    assert len(c1) == 1
    assert isinstance(c1[0].operation, mc.GateU)
    np.testing.assert_allclose(circ_mat(c1, 1), U1, atol=1e-8)

    U2 = rand_u(4, 10)
    c2, p2 = _qsd_decomposition(U2)
    assert_phase_eq(circ_mat(c2, 2), np.exp(-1j * p2) * U2, atol=1e-6)

    U3 = rand_u(8, 11)
    c3, p3 = _qsd_decomposition(U3)
    assert_phase_eq(circ_mat(c3, 3), np.exp(-1j * p3) * U3, atol=2e-5)
