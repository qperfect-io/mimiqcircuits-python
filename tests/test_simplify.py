#
# Copyright © 2026 QPerfect. All Rights Reserved.
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
"""Tests for the circuit simplification passes."""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits import _core
from mimiqcircuits.simplify import _CODES

# --- helpers -----------------------------------------------------------------


def _apply(u, m, qubits, nq):
    """``m`` on ``qubits`` applied after ``u``, both on ``nq`` qubits."""
    k = len(qubits)
    t = u.reshape((2,) * nq + (2**nq,))
    g = m.reshape((2,) * (2 * k))
    t = np.tensordot(g, t, axes=(list(range(k, 2 * k)), list(qubits)))
    t = np.moveaxis(t, list(range(k)), list(qubits))
    return t.reshape(2**nq, 2**nq)


def _unitary(c, nq):
    """Unitary of the gates of ``c``; other operations are skipped."""
    u = np.eye(2**nq, dtype=complex)
    for inst in c:
        if isinstance(inst.operation, mc.Gate):
            m = np.asarray(inst.operation.unwrappedmatrix(), dtype=complex)
            u = _apply(u, m, inst.qubits, nq)
    return u


def _n2q(c):
    return sum(len(inst.qubits) == 2 for inst in c)


def _random_circuit(rng, nq, n, wide=False):
    c = mc.Circuit()
    for _ in range(n):
        r = rng.random()
        qs = rng.sample(range(nq), nq)
        if r < 0.3:
            g = rng.choice([mc.GateCX(), mc.GateCZ(), mc.GateSWAP(), mc.GateCRZ(0.3), mc.GateRZZ(-0.3)])
            c.push(g, qs[0], qs[1])
        elif wide and nq >= 3 and r < 0.4:
            c.push(mc.GateCCX(), *qs[:3])
        elif r < 0.65:
            g = rng.choice([mc.GateH(), mc.GateX(), mc.GateZ(), mc.GateS(), mc.GateSDG(), mc.GateT(), mc.GateTDG(), mc.GateSX()])
            c.push(g, qs[0])
        else:
            g = rng.choice([mc.GateRZ, mc.GateRX, mc.GateP])(rng.choice([0.0, 0.25, -0.25, 0.5]))
            c.push(g, qs[0])
    return c


PASSES = [
    mc.remove_identities,
    mc.cancel_inverses,
    mc.merge_rotations,
    mc.resynthesize_1q,
    mc.resynthesize_2q,
    mc.resynthesize_linear,
    mc.simplify,
]


def test_created_gate_matrices():
    """The native code's matrices for the gates it creates match the gates."""
    params = {0: [0.3, 0.7, -1.1, 0.4], 1: [0.9], 2: [0.9], 3: [0.9], 4: [0.9]}
    for code, cls in enumerate(_CODES):
        p = params.get(code, [])
        native = np.array([complex(*z) for z in _core.simplify_code_matrix(code, p)])
        expected = np.asarray(cls(*p).unwrappedmatrix(), dtype=complex).reshape(-1)
        assert np.allclose(native, expected, atol=1e-14), cls.__name__


# --- parity with MimiqCircuitsBase.jl -----------------------------------------
#
# tests/data/simplify_julia_reference.json is written by
# scripts/simplify_julia_reference.jl. remove_identities, cancel_inverses,
# merge_rotations and resynthesize_linear are deterministic and must agree gate
# for gate. resynthesize_1q agrees gate for gate up to how an angle near ±π is
# written. resynthesize_2q picks one of several equivalent KAK factorizations,
# depending on the last bits of the matrices, so its local gates may differ;
# simplify inherits that.

_NAMES = {
    "H": mc.GateH, "X": mc.GateX, "Y": mc.GateY, "Z": mc.GateZ,
    "S": mc.GateS, "SDG": mc.GateSDG, "T": mc.GateT, "TDG": mc.GateTDG,
    "SX": mc.GateSX, "SXDG": mc.GateSXDG, "SY": mc.GateSY, "SYDG": mc.GateSYDG,
    "RX": mc.GateRX, "RY": mc.GateRY, "RZ": mc.GateRZ, "P": mc.GateP, "U": mc.GateU,
    "CX": mc.GateCX, "CZ": mc.GateCZ, "SWAP": mc.GateSWAP, "CCX": mc.GateCCX,
    "C3X": mc.GateC3X, "CRZ": mc.GateCRZ, "RZZ": mc.GateRZZ,
}

_REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "simplify_julia_reference.json").read_text()
)


def _decode(entries):
    c = mc.Circuit()
    for d in entries:
        if d["name"] == "M":
            c.push(mc.Measure(), *d["qubits"], *d["bits"])
        else:
            c.push(_NAMES[d["name"]](*d["params"]), *d["qubits"])
    return c


def _structure(c):
    return [(type(inst.operation), tuple(inst.qubits)) for inst in c]


def _matrix(op):
    return np.asarray(op.unwrappedmatrix(), dtype=complex)


@pytest.mark.parametrize("k", range(len(_REFERENCE)))
def test_julia_parity(k):
    case = _REFERENCE[k]
    c = _decode(case["circuit"])
    nq = max(q for inst in c for q in inst.qubits) + 1
    for name, entries in case["results"].items():
        ref = _decode(entries)
        out = getattr(mc, name)(c)
        if name == "simplify":
            assert _n2q(out) == _n2q(ref)
            assert abs(len(out) - len(ref)) <= 1
        else:
            assert _structure(out) == _structure(ref), name
        if name in ("remove_identities", "cancel_inverses", "merge_rotations", "resynthesize_linear"):
            for a, b in zip(out, ref):
                assert np.allclose(a.operation.getparams(), b.operation.getparams(), atol=1e-9), name
        elif name == "resynthesize_1q":
            for a, b in zip(out, ref):
                if isinstance(a.operation, mc.Gate):
                    assert np.allclose(_matrix(a.operation), _matrix(b.operation), atol=1e-9)
        assert np.allclose(_unitary(out, nq), _unitary(ref, nq), atol=1e-8), name


# --- behaviour ----------------------------------------------------------------


def test_remove_identities():
    th = se.Symbol("th")
    c = mc.Circuit()
    c.push(mc.GateID(), 0)
    c.push(mc.GateRZ(0), 0)
    c.push(mc.GateU(0, 0, 0), 1)
    c.push(mc.GateCRX(0.0), 0, 1)
    c.push(mc.GateRX(2 * math.pi), 0)  # -I: the identity only up to a phase
    c.push(mc.Delay(0.1), 0)  # marks idle time for noise
    c.push(mc.GateRZ(th), 0)
    d = mc.remove_identities(c)
    assert [type(i.operation) for i in d] == [mc.GateRX, mc.Delay, mc.GateRZ]


def test_kept_instructions_are_the_same_objects():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.Measure(), 1, 0)
    d = mc.cancel_inverses(c)
    assert len(d) == 2
    assert d[0] is c[0] and d[1] is c[3]


@pytest.mark.parametrize(
    "a, b",
    [
        (mc.GateH(), mc.GateH()),
        (mc.GateS(), mc.GateSDG()),
        (mc.GateT(), mc.GateTDG()),
        (mc.GateSX(), mc.GateSXDG()),
        (mc.GateRX(0.3), mc.GateRX(-0.3)),
    ],
)
def test_cancel_adjacent(a, b):
    c = mc.Circuit()
    c.push(a, 0)
    c.push(b, 0)
    assert len(mc.cancel_inverses(c)) == 0


def test_cancel_written_differently():
    for g in (mc.GateSWAP(), mc.GateCZ()):
        c = mc.Circuit()
        c.push(g, 0, 1)
        c.push(g, 1, 0)
        assert len(mc.cancel_inverses(c)) == 0
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 1, 0)
    assert len(mc.cancel_inverses(c)) == 2


@pytest.mark.parametrize(
    "g, q, through",
    [
        (mc.GateZ(), 0, True),
        (mc.GateRZ(0.4), 0, True),
        (mc.GateX(), 1, True),
        (mc.GateH(), 2, True),
        (mc.GateX(), 0, False),
        (mc.GateH(), 1, False),
        (mc.GateZ(), 1, False),
    ],
)
def test_cancel_across(g, q, through):
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(g, q)
    c.push(mc.GateCX(), 0, 1)
    assert len(mc.cancel_inverses(c)) == (1 if through else 3)


def test_cancel_blocked_by_non_unitary():
    for push in (
        lambda c: c.push(mc.Measure(), 0, 0),
        lambda c: c.push(mc.Reset(), 0),
        lambda c: c.push(mc.Barrier(1), 0),
        lambda c: c.push(mc.IfStatement(mc.GateX(), mc.BitString("1")), 0, 0),
    ):
        c = mc.Circuit()
        c.push(mc.GateX(), 0)
        push(c)
        c.push(mc.GateX(), 0)
        assert len(mc.cancel_inverses(c)) == 3


def test_global_observable_synchronises():
    c = mc.Circuit()
    c.push(mc.GateH(), 1)
    c.push(mc.Amplitude(mc.BitString("00")), 0)
    c.push(mc.GateH(), 1)
    assert len(mc.cancel_inverses(c)) == 3


def test_multicontrolled_x_rules():
    c = mc.Circuit()
    c.push(mc.GateCCX(), 0, 1, 6)
    c.push(mc.GateCCX(), 0, 2, 7)  # shares a control: commutes, wider than four qubits
    c.push(mc.GateCCX(), 0, 1, 6)
    assert len(mc.cancel_inverses(c)) == 1

    c = mc.Circuit()
    c.push(mc.Control(5, mc.GateX()), *range(6))
    c.push(mc.GateCZ(), 0, 1)
    c.push(mc.GateT(), 2)
    c.push(mc.GateX(), 5)
    c.push(mc.Control(4, mc.GateX()), 1, 2, 3, 4, 6)
    c.push(mc.Control(5, mc.GateX()), *range(6))
    assert len(mc.cancel_inverses(c)) == 4

    c = mc.Circuit()
    c.push(mc.GateCCX(), 0, 1, 2)
    c.push(mc.GateCCX(), 1, 0, 2)
    assert len(mc.cancel_inverses(c)) == 0

    for g, qs in [(mc.GateCCX(), (2, 3, 4)), (mc.GateZ(), (2,)), (mc.GateRX(0.2), (0,))]:
        c = mc.Circuit()
        c.push(mc.GateCCX(), 0, 1, 2)
        c.push(g, *qs)
        c.push(mc.GateCCX(), 0, 1, 2)
        assert len(mc.cancel_inverses(c)) == 3


@pytest.mark.parametrize(
    "gates, expected",
    [
        ((mc.GateT(), mc.GateT()), mc.GateS()),
        ((mc.GateS(), mc.GateS()), mc.GateZ()),
        ((mc.GateS(), mc.GateZ()), mc.GateSDG()),
        ((mc.GateSX(), mc.GateSX()), mc.GateX()),
        ((mc.GateX(), mc.GateSX()), mc.GateSXDG()),
        ((mc.GateSY(), mc.GateSY()), mc.GateY()),
        ((mc.GateRZ(0.1), mc.GateRZ(0.2)), mc.GateRZ(0.30000000000000004)),
        ((mc.GateP(0.1), mc.GateP(0.2)), mc.GateP(0.30000000000000004)),
    ],
)
def test_merge(gates, expected):
    c = mc.Circuit()
    for g in gates:
        c.push(g, 0)
    d = mc.merge_rotations(c)
    assert len(d) == 1
    assert d[0].operation == expected


def test_merge_families():
    c = mc.Circuit()
    c.push(mc.GateT(), 0)
    c.push(mc.GateS(), 0)
    d = mc.merge_rotations(c)
    assert isinstance(d[0].operation, mc.GateP)
    assert np.allclose(_unitary(d, 1), _unitary(c, 1))

    c = mc.Circuit()
    for g in (mc.GateT(), mc.GateT(), mc.GateS(), mc.GateZ()):
        c.push(g, 0)
    assert len(mc.merge_rotations(c)) == 0

    # RZ and P differ by a global phase, so they do not merge
    c = mc.Circuit()
    c.push(mc.GateRZ(0.3), 0)
    c.push(mc.GateP(0.2), 0)
    assert len(mc.merge_rotations(c)) == 2


def test_merge_two_qubit_and_controlled():
    for g in (mc.GateRXX, mc.GateRZZ, mc.GateCRZ, mc.GateCP):
        c = mc.Circuit()
        c.push(g(0.1), 0, 1)
        c.push(g(0.2), 0, 1)
        d = mc.merge_rotations(c)
        assert len(d) == 1
        assert type(d[0].operation) is g
        assert np.allclose(_unitary(d, 2), _unitary(c, 2))
    c = mc.Circuit()
    c.push(mc.GateRZX(0.1), 0, 1)
    c.push(mc.GateRZX(0.2), 1, 0)
    assert len(mc.merge_rotations(c)) == 2


def test_merge_symbolic():
    th, ph = se.Symbol("th"), se.Symbol("ph")
    c = mc.Circuit()
    c.push(mc.GateRZ(th), 0)
    c.push(mc.GateCZ(), 0, 1)  # diagonal: symbolic gates commute through it
    c.push(mc.GateRZ(ph), 0)
    d = mc.merge_rotations(c)
    assert len(d) == 2
    assert d[0].operation == mc.GateRZ(th + ph)

    c = mc.Circuit()
    c.push(mc.GateRZ(th), 0)
    c.push(mc.GateRZ(0.5), 0)
    c.push(mc.GateRZ(-th - 0.5), 0)
    assert len(mc.merge_rotations(c)) == 0

    c = mc.Circuit()
    c.push(mc.GateRZ(th), 0)
    c.push(mc.GateRZ(-th), 0)
    assert len(mc.cancel_inverses(c)) == 0


def test_resynthesize_1q():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateS(), 0)
    c.push(mc.GateH(), 0)
    c.push(mc.GateX(), 1)
    d = mc.resynthesize_1q(c)
    assert [type(i.operation) for i in d] == [mc.GateU, mc.GateX]
    assert np.allclose(_unitary(d, 2), _unitary(c, 2))

    c = mc.Circuit()
    for g in (mc.GateH(), mc.GateZ(), mc.GateH(), mc.GateX()):
        c.push(g, 0)
    assert len(mc.resynthesize_1q(c)) == 0

    th = se.Symbol("th")
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.GateH(), 0)
    c.push(mc.GateRZ(th), 0)
    c.push(mc.GateH(), 0)
    assert len(mc.resynthesize_1q(c)) == 7


def test_resynthesize_2q():
    c = mc.Circuit()
    for _ in range(4):
        c.push(mc.GateCX(), 0, 1)
        c.push(mc.GateRY(0.3), 1)
        c.push(mc.GateRZ(0.2), 0)
    d = mc.resynthesize_2q(c)
    assert _n2q(d) <= 3
    assert np.allclose(_unitary(d, 2), _unitary(c, 2))

    # a SWAP written as three CX is already minimal
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 1, 0)
    c.push(mc.GateCX(), 0, 1)
    assert _structure(mc.resynthesize_2q(c)) == _structure(c)


def test_resynthesize_linear():
    c = mc.Circuit()
    for q in range(4):
        c.push(mc.GateCX(), q, q + 1)
    for q in reversed(range(4)):
        c.push(mc.GateCX(), q, q + 1)
    assert len(mc.resynthesize_linear(c)) == 0

    c = mc.Circuit()
    for _ in range(2):
        for q in range(3):
            c.push(mc.GateCX(), q, q + 1)
    d = mc.resynthesize_linear(c)
    assert _n2q(d) == 2
    assert np.allclose(_unitary(d, 4), _unitary(c, 4))

    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 0, 1)
    d = mc.resynthesize_linear(c)
    assert [type(i.operation) for i in d] == [mc.GateX]


def test_simplify():
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateRZ(0.2), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateRZ(0.3), 0)
    d = mc.simplify(c)
    assert len(d) == 1
    assert d[0].operation == mc.GateRZ(0.5)

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateS(), 0)
    c.push(mc.GateH(), 0)
    assert len(mc.simplify(c, passes=(mc.cancel_inverses,))) == 3
    assert len(mc.simplify(c)) == 1


def test_simplify_with_custom_pass():
    """Passes that are not this module's run through the Python loop."""
    seen = []

    def spy(c):
        seen.append(len(c))
        return c

    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateH(), 0)
    d = mc.simplify(c, passes=(spy, mc.cancel_inverses))
    assert len(d) == 1
    # The second round runs `spy` again, since `cancel_inverses` changed the
    # circuit after it, and then stops, since nothing improved.
    assert seen == [3, 1]


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_keep_their_unitary(seed):
    rng = random.Random(seed)
    nq = rng.randint(2, 5)
    c = _random_circuit(rng, nq, rng.randint(5, 50), wide=seed % 2 == 1)
    u = _unitary(c, nq)
    for p in PASSES:
        d = p(c)
        assert np.allclose(_unitary(d, nq), u, atol=1e-8), p.__name__
    assert _n2q(mc.simplify(c)) <= _n2q(c)


def test_simplify_pass():
    from mimiqcircuits.backends.passes import PassContext, PInt, PList, PSym

    p = mc.SimplifyPass()
    s = p.spec()
    assert s.name == "simplify"
    assert dict(s.parameters) == {"maxiter": PInt(10)}

    # The passes are named, so the spec can describe them.
    p = mc.SimplifyPass(passes=["cancel_inverses", "merge_rotations"], maxiter=3)
    assert dict(p.spec().parameters)["passes"] == PList(
        (PSym("cancel_inverses"), PSym("merge_rotations"))
    )
    with pytest.raises(ValueError):
        mc.SimplifyPass(passes=["fuse"])
    with pytest.raises(ValueError):
        mc.SimplifyPass(maxiter=0)

    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateRZ(0.2), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateRZ(0.3), 0)
    c.push(mc.Measure(), 1, 0)
    out, result = mc.SimplifyPass().apply(PassContext(), c)
    assert [type(i.operation) for i in out] == [mc.GateRZ, mc.Measure]
    assert out[0].operation == mc.GateRZ(0.5)
    assert result.qubit_permutation is None

    # Only the named passes run.
    c = mc.Circuit()
    c.push(mc.GateRZ(0.2), 0)
    c.push(mc.GateRZ(0.3), 0)
    out, _ = mc.SimplifyPass(passes=["cancel_inverses"]).apply(PassContext(), c)
    assert len(out) == 2
    out, _ = mc.SimplifyPass(passes=["merge_rotations"]).apply(PassContext(), c)
    assert len(out) == 1
