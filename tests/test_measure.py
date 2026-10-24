import pytest
import mimiqcircuits as mc


def test_measure_inverse_raises_error():
    measure = mc.Measure()
    with pytest.raises(TypeError):
        measure.inverse()


# Pair measurements: the canonical decomposition must act as the ideal
# projective measurement of the two-qubit Pauli operator, both in its outcome
# probabilities and in the state it leaves behind.

import numpy as np
from mimiqcircuits.decomposition import decompose, CanonicalBasis

_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _apply(state, mat, qubits):
    # state has one axis per qubit; mat acts on `qubits` with the first qubit
    # as the most significant index, matching the gate matrix convention.
    k = len(qubits)
    t = np.asarray(mat, dtype=complex).reshape([2] * (2 * k))
    state = np.tensordot(t, state, axes=(list(range(k, 2 * k)), qubits))
    return np.moveaxis(state, list(range(k)), qubits)


def _branches(circuit, state):
    # Exact simulation that follows both outcomes of every Z measurement.
    # Returns a dict {outcome bit: (probability, normalized state)}.
    branches = [(1.0, state, None)]
    for inst in circuit:
        op, qs = inst.operation, list(inst.qubits)
        new = []
        for p, psi, bit in branches:
            if isinstance(op, mc.Measure):
                for b in (0, 1):
                    proj = np.diag([1.0 - b, float(b)]).astype(complex)
                    phi = _apply(psi, proj, qs)
                    pb = np.vdot(phi, phi).real
                    if pb > 1e-12:
                        new.append((p * pb, phi / np.sqrt(pb), b))
            else:
                mat = np.array(op.matrix().tolist(), dtype=complex)
                new.append((p, _apply(psi, mat, qs), bit))
        branches = new
    out = {}
    for p, psi, b in branches:
        assert b not in out
        out[b] = (p, psi)
    return out


@pytest.mark.parametrize(
    "op, pauli",
    [(mc.MeasureXX(), "X"), (mc.MeasureYY(), "Y"), (mc.MeasureZZ(), "Z")],
)
def test_pair_measure_decomposition_is_projective(op, pauli):
    rng = np.random.default_rng(1234)
    pp = np.kron(_PAULI[pauli], _PAULI[pauli])
    decomposed = decompose(mc.Circuit().push(op, 0, 1, 0), CanonicalBasis())
    for _ in range(10):
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        v /= np.linalg.norm(v)
        got = _branches(decomposed, v.reshape(2, 2))
        for b, sign in ((0, 1), (1, -1)):
            # Outcome 0 is the +1 eigenspace, outcome 1 the -1 eigenspace.
            phi = (np.eye(4) + sign * pp) @ v / 2
            pb = np.vdot(phi, phi).real
            p, psi = got[b]
            assert p == pytest.approx(pb, abs=1e-10)
            # Equal up to a global phase.
            overlap = abs(np.vdot(phi / np.sqrt(pb), psi.reshape(4)))
            assert overlap == pytest.approx(1.0, abs=1e-10)
