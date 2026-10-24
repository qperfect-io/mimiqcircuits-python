"""Characterization tests for operator matrices and their embedding.

``Instruction.matrix`` and ``reorder_qubits_matrix`` take a NumPy-indexed
shortcut for numeric SymEngine matrices, and operators fully determined by
their type cache their matrix on the class. Both must return exactly what the
plain SymEngine construction returns: the same entries, of the same SymEngine
types, with the same signed zeros. The reference implementations below are the
plain constructions the fast paths must match.
"""

import random

import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.matrices import (
    _index_permutation,
    kronecker,
    reorder_qubits_matrix,
)
from mimiqcircuits.operations.operator import (
    AbstractOperator,
    _as_se_matrix,
    _smart_evalf,
)


def _reference_matrix(op):
    """Uncached ``AbstractOperator.matrix``, or the operator's own override."""
    if type(op).matrix is not AbstractOperator.matrix:
        return op.matrix()
    return _as_se_matrix(op._matrix()).applyfunc(_smart_evalf)


def _reference_reorder(M, qubits, nq):
    """Pad with identities, then permute entry by entry, all in SymEngine."""
    fullqubits = list(qubits) + [q for q in range(nq) if q not in qubits]
    ident = _reference_matrix(mc.GateID())

    fullM = M
    for _ in range(nq - len(qubits)):
        fullM = kronecker(fullM, ident)

    if fullqubits == sorted(fullqubits):
        return fullM

    qperm = [(nq - 1) - i for i in reversed(np.argsort(fullqubits))]
    dim = 2**nq
    perm = _index_permutation(qperm, nq)
    reordered = se.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            reordered[i, j] = fullM[perm[i], perm[j]]
    return reordered


def _assert_identical(A, B):
    assert type(A) is type(B)
    assert (A.rows, A.cols) == (B.rows, B.cols)
    for a, b in zip(A, B):
        # str tells signed zeros and exact from floating zeros apart.
        assert type(a) is type(b) and str(a) == str(b), (a, b)


def _build(cls):
    if cls is mc.GateCustom:
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        return cls(q)
    if cls is mc.GateCustomDiagonal:
        return cls([1, 1j, -1, np.exp(0.3j)])
    # Controlled parametric gates carry their parameters on the inner gate.
    nargs = {mc.GateCU: 4, mc.GateCP: 1, mc.GateCRX: 1, mc.GateCRY: 1}.get(
        cls, 1 if cls in (mc.GateCRZ, mc.GateCCP) else len(cls._parnames)
    )
    return cls(*[0.37 * (i + 1) for i in range(nargs)])


GATES = [_build(cls) for group in mc.SIMPLEGATES_GROUPS.values() for cls in group]


def _placements(nqubits, rng):
    """Sorted, reversed and random target placements into up to 6 qubits."""
    out = []
    for nq in sorted({nqubits, min(6, nqubits + 2), 6}):
        qubits = rng.sample(range(nq), nqubits)
        out.append((sorted(qubits), nq))
        out.append((sorted(qubits, reverse=True), nq))
        out.append((qubits, nq))
    return out


@pytest.mark.parametrize("op", GATES, ids=lambda g: type(g).__name__)
def test_matrix_matches_uncached_construction(op):
    ref = _reference_matrix(op)
    _assert_identical(op.matrix(), ref)
    _assert_identical(op.matrix(), ref)  # served from the class cache


@pytest.mark.parametrize("op", GATES, ids=lambda g: type(g).__name__)
def test_instruction_matrix_matches_reference(op):
    rng = random.Random(type(op).__name__)
    M = _reference_matrix(op)
    for qubits, nq in _placements(op.num_qubits, rng):
        got = mc.Instruction(op, tuple(qubits)).matrix(nq)
        _assert_identical(got, _reference_reorder(M, qubits, nq))


def _symbolic_gates():
    x, y, z = se.symbols("x y z")
    return [
        mc.GateRX(x),
        mc.GateU(x, y, z),
        mc.GateCP(x),
        mc.GateCRY(y),
        mc.GateRZZ(z),
        mc.Control(2, mc.GateRY(x)),
    ]


@pytest.mark.parametrize("op", _symbolic_gates(), ids=str)
def test_symbolic_instruction_matrix_matches_reference(op):
    rng = random.Random(str(op))
    M = _reference_matrix(op)
    _assert_identical(op.matrix(), M)
    for qubits, nq in _placements(op.num_qubits, rng)[:6]:
        got = mc.Instruction(op, tuple(qubits)).matrix(nq)
        _assert_identical(got, _reference_reorder(M, qubits, nq))


@pytest.mark.parametrize(
    "M",
    [
        se.Matrix([[1, 2], [3, 4]]),
        se.Matrix([[0, se.Rational(1, 2)], [se.RealDouble(-0.0), -1]]),
        se.Matrix([[0, se.I], [-se.I, 0]]),  # exact complex: symbolic path
        se.Matrix([[se.ComplexDouble(-0.0 - 1j), 0], [0, se.ComplexDouble(1j)]]),
        se.Matrix(
            [
                [se.RealMPFR("0.1", 100), se.RealMPFR("-0", 100)],
                [se.Rational(1, 3), se.RealMPFR("-2.5", 100)],
            ]
        ),
    ],
    ids=["integer", "rational", "exact-complex", "complex-double", "mpfr"],
)
def test_reorder_raw_matrices_matches_reference(M):
    for qubits, nq in [([0], 1), ([0], 3), ([2], 3), ([1], 4)]:
        _assert_identical(
            reorder_qubits_matrix(M, qubits, nq), _reference_reorder(M, qubits, nq)
        )


def test_cached_matrix_does_not_leak_mutation():
    for g in (mc.GateH(), mc.GateT(), mc.GateCX()):
        ref = _reference_matrix(g)
        m = g.matrix()
        m[0, 0] = se.Symbol("junk")
        _assert_identical(g.matrix(), ref)

        e = mc.Instruction(g, tuple(range(g.num_qubits))).matrix()
        e[0, 0] = se.Symbol("junk")
        _assert_identical(g.matrix(), ref)

        u = g.unwrappedmatrix()
        assert not u.flags.writeable
        with pytest.raises(ValueError):
            u[0, 0] = 5


def test_named_wrappers_share_a_class_cache():
    # Named wrappers ignore constructor arguments, so any instance is the gate.
    assert mc.GateCX()._hascachedmatrix()
    assert mc.GateCX(3, mc.GateH())._hascachedmatrix()
    assert mc.GateT()._hascachedmatrix()
    assert mc.GateSDG()._hascachedmatrix()


def test_parametric_wrappers_are_not_cached_by_class():
    # Generic wrappers resolve to a named gate when one exists (``Power(Z,
    # 1/4)`` is ``GateT``), so these are built to stay generic.
    for g in (
        mc.Power(mc.GateX(), 0.3),
        mc.Control(2, mc.GateH()),
        mc.Inverse(mc.GateRX(0.2)),
        mc.GateCP(0.1),
        mc.GateCRX(0.2),
    ):
        assert not g._hascachedmatrix()

    a = mc.GateCP(0.1).unwrappedmatrix()
    b = mc.GateCP(0.2).unwrappedmatrix()
    assert not np.allclose(a, b)
    _assert_identical(mc.GateCP(0.2).matrix(), _reference_matrix(mc.GateCP(0.2)))
    _assert_identical(
        mc.Power(mc.GateX(), 0.3).matrix(),
        _reference_matrix(mc.Power(mc.GateX(), 0.3)),
    )


def test_cached_unwrappedmatrix_matches_uncached():
    for op in GATES:
        if op._hascachedmatrix():
            np.testing.assert_array_equal(op.unwrappedmatrix(), op._matrix_numeric())
