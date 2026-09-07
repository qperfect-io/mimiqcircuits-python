#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import itertools
from random import Random

import numpy as np

import mimiqcircuits as mc
from mimiqcircuits.matrices import (
    _apply_local,
    _index_permutation,
    reorder_qubits_matrix,
)


def _gate_matrix(op):
    # SymEngine matrix of a gate, for an independent reference unitary that
    # does not share the implementation's NumPy path.
    return op.matrix()


def _np(m):
    return np.array(
        [[complex(m[i, j]) for j in range(m.cols)] for i in range(m.rows)],
        dtype=complex,
    )


def _circuit_unitary(circuit, nq):
    """Unitary of a circuit's gate content, folded in circuit order.

    Non-gate ops are skipped: fusion only regroups unitaries and never reorders
    non-commuting gates, so this product is invariant under `fuse_circuit`.
    """
    u = np.eye(2**nq, dtype=complex)
    for inst in circuit:
        if not isinstance(inst.operation, mc.Gate):
            continue
        e = _np(reorder_qubits_matrix(_gate_matrix(inst.operation), list(inst.qubits), nq))
        u = e @ u
    return u


def test_two_singles_fuse():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 0)
    f = mc.fuse_circuit(c)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustom)
    assert f[0].operation.num_qubits == 1


def test_two_cx_fuse():
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 0, 1)
    f = mc.fuse_circuit(c)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustom)


def test_single_plus_two_qubit():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    f = mc.fuse_circuit(c)
    assert len(f) == 1
    assert tuple(f[0].qubits) == (0, 1)


def test_rotations_then_cx():
    c = mc.Circuit()
    c.push(mc.GateRX(0.3), 0)
    c.push(mc.GateRZ(0.5), 0)
    c.push(mc.GateCX(), 0, 1)
    f = mc.fuse_circuit(c, 2)
    assert len(f) == 1


def test_no_singleton_demotion():
    # a lone gate between two boundaries stays as itself, never GateCustom
    c = mc.Circuit()
    c.push(mc.Barrier(1), 0)
    c.push(mc.GateH(), 0)
    c.push(mc.Barrier(1), 0)
    f = mc.fuse_circuit(c)
    assert any(isinstance(i.operation, mc.GateH) for i in f)
    assert not any(isinstance(i.operation, mc.GateCustom) for i in f)


def test_max_support_one_passes_two_qubit_gate():
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    f = mc.fuse_circuit(c, 1)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCX)


def test_no_fusion_across_measure():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.GateH(), 0)
    f = mc.fuse_circuit(c)
    assert sum(isinstance(i.operation, mc.GateH) for i in f) == 2
    assert not any(isinstance(i.operation, mc.GateCustom) for i in f)
    assert sum(isinstance(i.operation, mc.Measure) for i in f) == 1


def test_boundary_clearing_one_wire_stays_convex():
    # a boundary that ends only one of a cluster's wires must still keep a later
    # gate on the other wire from fusing back across it
    c = mc.Circuit()
    c.push(mc.GateH(), 1)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 1, 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateH(), 1)
    f = mc.fuse_circuit(c)
    assert np.allclose(_circuit_unitary(c, 2), _circuit_unitary(f, 2), atol=1e-9)


def test_non_contiguous_support():
    # a 1-qubit gate sharing a cluster with a gate on higher/lower wires must be
    # embedded on its own wire, not the lowest one
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 2)
    c.push(mc.GateH(), 2)
    f = mc.fuse_circuit(c, 2)
    assert np.allclose(_circuit_unitary(c, 3), _circuit_unitary(f, 3), atol=1e-9)


def test_empty_circuit():
    assert len(mc.fuse_circuit(mc.Circuit())) == 0


def test_random_equivalence():
    rng = Random(20260711)
    g1 = [mc.GateH, mc.GateT, mc.GateX, mc.GateS,
          lambda: mc.GateRX(0.37), lambda: mc.GateRZ(1.1)]
    g2 = [mc.GateCX, mc.GateCZ, mc.GateSWAP]

    def randcirc(nq, depth):
        c = mc.Circuit()
        for _ in range(depth):
            r = rng.random()
            if r < 0.5:
                c.push(rng.choice(g1)(), rng.randrange(nq))
            elif r < 0.85 and nq >= 2:
                a = rng.randrange(nq)
                b = rng.choice([x for x in range(nq) if x != a])
                c.push(rng.choice(g2)(), a, b)
            elif r < 0.93:
                c.push(mc.Barrier(1), rng.randrange(nq))
            elif r < 0.97:
                q = rng.randrange(nq)
                c.push(mc.Measure(), q, q)
            else:
                c.push(mc.Reset(), rng.randrange(nq))
        return c

    for n in (1, 2, 3, 4, 5):
        for _ in range(60):
            nq = rng.randint(1, 5)
            c = randcirc(nq, rng.randint(2, 16))
            f = mc.fuse_circuit(c, n)
            assert len(f) <= len(c)
            assert np.allclose(_circuit_unitary(c, nq), _circuit_unitary(f, nq), atol=1e-9)


def test_diagonal_run_fuses_into_customdiagonal():
    c = mc.Circuit()
    c.push(mc.GateP(0.1), 0)
    c.push(mc.GateCZ(), 0, 1)
    c.push(mc.GateRZ(0.3), 1)
    f = mc.fuse_circuit(c)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustomDiagonal)
    assert np.allclose(_circuit_unitary(c, 2), _circuit_unitary(f, 2), atol=1e-12)


def test_one_dense_gate_makes_the_block_dense():
    c = mc.Circuit()
    c.push(mc.GateP(0.1), 0)
    c.push(mc.GateH(), 0)
    c.push(mc.GateCZ(), 0, 1)
    f = mc.fuse_circuit(c)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustom)
    assert np.allclose(_circuit_unitary(c, 2), _circuit_unitary(f, 2), atol=1e-12)


def _diagonal_chain():
    c = mc.Circuit()
    for q in range(4):
        c.push(mc.GateP(0.1 * (q + 1)), q)
    c.push(mc.GateCZ(), 0, 1)
    c.push(mc.GateCZ(), 1, 2)
    c.push(mc.GateCZ(), 2, 3)
    return c


def test_max_diagonal_support_widens_diagonal_runs():
    c = _diagonal_chain()
    assert len(mc.fuse_circuit(c, 2)) > 1

    f = mc.fuse_circuit(c, 2, 4)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustomDiagonal)
    assert tuple(f[0].qubits) == (0, 1, 2, 3)
    assert np.allclose(_circuit_unitary(c, 4), _circuit_unitary(f, 4), atol=1e-12)


def test_customdiagonal_member_with_unsorted_targets():
    c = mc.Circuit()
    c.push(mc.GateCustomDiagonal([1, 1j, -1, -1j]), 1, 0)
    c.push(mc.GateCustomDiagonal([1, np.exp(1.1j)]), 0)
    f = mc.fuse_circuit(c)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustomDiagonal)
    assert np.allclose(_circuit_unitary(c, 2), _circuit_unitary(f, 2), atol=1e-12)


def test_narrow_diagonal_budget_still_fuses_densely():
    c = mc.Circuit()
    c.push(mc.GateP(0.1), 0)
    c.push(mc.GateCZ(), 0, 1)
    f = mc.fuse_circuit(c, 2, 1)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustom)
    assert np.allclose(_circuit_unitary(c, 2), _circuit_unitary(f, 2), atol=1e-12)


def test_random_equivalence_with_diagonal_budget():
    rng = Random(20260818)
    gd = [mc.GateZ, mc.GateT, lambda: mc.GateP(rng.random()),
          lambda: mc.GateRZ(rng.random())]

    for _ in range(60):
        nq = rng.randint(2, 5)
        c = mc.Circuit()
        for _ in range(rng.randint(2, 20)):
            r = rng.random()
            a = rng.randrange(nq)
            b = rng.choice([x for x in range(nq) if x != a])
            if r < 0.4:
                c.push(rng.choice(gd)(), a)
            elif r < 0.7:
                c.push(mc.GateCZ(), a, b)
            elif r < 0.85:
                c.push(mc.GateH(), a)
            else:
                c.push(mc.GateCX(), a, b)
        for mds in range(1, 6):
            f = mc.fuse_circuit(c, 2, mds)
            assert len(f) <= len(c)
            assert np.allclose(
                _circuit_unitary(c, nq), _circuit_unitary(f, nq), atol=1e-9
            )


def test_pass_entry_point():
    from mimiqcircuits.backends import PassContext

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    p = mc.FusePass(2)
    assert p.spec().name == "fuse_gates"
    out, result = p.apply(PassContext(), c)
    assert result.qubit_permutation is None
    assert len(out) <= len(c)


def test_qubit_threshold_skips_small_circuits():
    from mimiqcircuits.backends import PassContext

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 0)  # H·H = I — fuses to a single block when it runs

    p = mc.FusePass(qubit_threshold=0)
    assert dict(p.spec().parameters)["qubit_threshold"].value == 0
    fused, _ = p.apply(PassContext(), c)
    assert len(fused) == 1

    # Below the threshold the circuit passes through untouched.
    pth = mc.FusePass(qubit_threshold=2)  # c has a single qubit
    assert dict(pth.spec().parameters)["qubit_threshold"].value == 2
    skipped, _ = pth.apply(PassContext(), c)
    assert len(skipped) == 2


def test_fuse_wider_than_three_qubits():
    # a cluster wider than 3 qubits used to be rejected by GateCustom
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    for t in (1, 2, 3):
        c.push(mc.GateCX(), 0, t)
    f = mc.fuse_circuit(c, 4)
    assert len(f) == 1
    assert isinstance(f[0].operation, mc.GateCustom)
    assert f[0].operation.num_qubits == 4


def _brickwall(nq, layers):
    c = mc.Circuit()
    for l in range(layers):
        for q in range(l % 2, nq - 1, 2):
            c.push(mc.GateH(), q)
            c.push(mc.GateCX(), q, q + 1)
    return c


def test_clusters_merge_past_two_qubits():
    # Brick-wall entangling layers: after the first layer every wire is owned,
    # so each later gate bridges two clusters. Fusion used to refuse every such
    # bridge, which pinned the output at the max_support=2 result no matter how
    # wide the budget was.
    nq = 5
    c = _brickwall(nq, 4)
    counts = [len(mc.fuse_circuit(c, k)) for k in range(2, nq + 1)]
    assert counts == sorted(counts, reverse=True)  # wider budget never fuses worse
    assert counts[-1] < counts[0]                  # ... and here it fuses better

    # a cluster wider than two qubits has to actually be emitted
    assert any(i.operation.num_qubits > 2 for i in mc.fuse_circuit(c, 4))

    ref = _circuit_unitary(c, nq)
    for k in (2, nq):
        assert np.allclose(ref, _circuit_unitary(mc.fuse_circuit(c, k), nq), atol=1e-9)


def test_merging_never_closes_a_cycle():
    # g1 and g3 both look mergeable at g4, but g2 sits between them: fusing the
    # two into one block would need g2 to run both after and before it.
    c = mc.Circuit()
    for a, b in ((0, 1), (1, 2), (2, 3), (0, 3)):
        c.push(mc.GateCX(), a, b)
    ref = _circuit_unitary(c, 4)
    for k in (2, 3, 4):
        assert np.allclose(ref, _circuit_unitary(mc.fuse_circuit(c, k), 4), atol=1e-9)


def _slow_index_permutation(qperm, nq):
    """Obvious O(2**nq) form of `_index_permutation`, for reference."""
    ints = [
        mc.BitString(
            "".join(str(mc.BitString.fromint(nq, i)[q]) for q in qperm)
        ).tointeger()
        for i in range(2**nq)
    ]
    return np.argsort(ints)


def _slow_embed(g, targets, nq):
    """Embedding by identity padding and reindexing.

    The independent reference for `_apply_local` and for the NumPy path of
    `reorder_qubits_matrix`, both of which now contract instead of padding.
    """
    fullqubits = list(targets) + [q for q in range(nq) if q not in targets]
    fullM = g
    for _ in range(nq - len(targets)):
        fullM = np.kron(fullM, np.eye(2))
    if fullqubits == sorted(fullqubits):
        return fullM
    qperm = [(nq - 1) - i for i in reversed(np.argsort(fullqubits))]
    perm = _slow_index_permutation(qperm, nq)
    return fullM[np.ix_(perm, perm)]


def test_index_permutation_matches_obvious_form():
    for nq in range(1, 6):
        for qperm in itertools.permutations(range(nq)):
            assert np.array_equal(
                _index_permutation(list(qperm), nq),
                _slow_index_permutation(list(qperm), nq),
            )


def test_reorder_qubits_matrix_matches_padding_route():
    """Contracting against an identity must reproduce the padding route exactly.

    Each entry of that product is a sum with at most one non-zero term, so the
    agreement is bit-for-bit and not merely to tolerance.
    """
    rng = np.random.default_rng(20260818)
    for nq in range(1, 5):
        for ng in range(1, nq + 1):
            g = rng.normal(size=(2**ng, 2**ng)) + 1j * rng.normal(size=(2**ng, 2**ng))
            for targets in itertools.permutations(range(nq), ng):
                targets = list(targets)
                assert np.array_equal(
                    reorder_qubits_matrix(g, targets, nq), _slow_embed(g, targets, nq)
                )


def test_apply_local_matches_embedding():
    """Contracting a gate in place must equal embedding it and multiplying."""
    rng = np.random.default_rng(20260819)
    for nq in range(1, 5):
        # Small integers are exact in float64, so any difference here is a logic
        # error rather than a difference in summation order.
        u = rng.integers(-4, 5, (2**nq, 2**nq)).astype(complex)
        for ng in range(1, nq + 1):
            g = rng.integers(-4, 5, (2**ng, 2**ng)).astype(complex)
            for targets in itertools.permutations(range(nq), ng):
                targets = list(targets)
                assert np.array_equal(
                    _apply_local(u, g, targets, nq),
                    _slow_embed(g, targets, nq) @ u,
                )
