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

from random import Random

import numpy as np

import mimiqcircuits as mc
from mimiqcircuits.matrices import reorder_qubits_matrix


def _gate_matrix(op):
    # SymEngine matrix of a gate, for an independent reference unitary that
    # does not share the implementation's NumPy path. GateCustom keeps its
    # matrix in an attribute rather than a method, so accept either form.
    m = op.matrix
    return m() if callable(m) else m


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

    for n in (1, 2, 3):
        for _ in range(60):
            nq = rng.randint(1, 5)
            c = randcirc(nq, rng.randint(2, 16))
            f = mc.fuse_circuit(c, n)
            assert len(f) <= len(c)
            assert np.allclose(_circuit_unitary(c, nq), _circuit_unitary(f, nq), atol=1e-9)


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
