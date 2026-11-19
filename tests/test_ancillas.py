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
"""Tests for decompositions that use qubits outside their targets."""

import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.operations.decompositions.control import cx_decompose


def _unitary(circuit):
    n = circuit.num_qubits()
    U = np.eye(2**n, dtype=complex)
    for inst in circuit:
        qs = list(inst.qubits)
        k = len(qs)
        M = np.array(
            [[complex(x) for x in row] for row in inst.operation.matrix().tolist()]
        )
        psi = np.moveaxis(U.reshape([2] * n + [2**n]), qs, list(range(k)))
        psi = np.tensordot(
            M.reshape([2] * (2 * k)), psi, axes=(range(k, 2 * k), range(k))
        )
        U = np.moveaxis(psi, list(range(k)), qs).reshape(2**n, 2**n)
    return U


def _count(c, T):
    return sum(isinstance(inst.operation, T) for inst in c)


def _tcount(c):
    return _count(c, (mc.GateT, mc.GateTDG))


def _cnx_circuit(n, extra):
    """A C_n(X) on qubits 0..n in a circuit `extra` qubits wider."""
    c = mc.Circuit()
    c.push(mc.Control(n, mc.GateX()), *range(n + 1))
    if extra > 0:
        c.push(mc.GateID(), n + extra)
    return c


def test_defaults_leave_the_output_unchanged():
    c = _cnx_circuit(4, 2)
    assert mc.decompose(c) == mc.decompose(c, mc.CanonicalBasis(borrow=False))
    assert mc.CanonicalBasis() == mc.CanonicalBasis(borrow=False)
    assert mc.CliffordTBasis(borrow=True) != mc.CliffordTBasis()


@pytest.mark.parametrize(
    "n,extra", [(n, extra) for n in range(3, 6) for extra in range(1, n - 1)]
)
def test_cnx_keeps_its_unitary(n, extra):
    c = _cnx_circuit(n, extra)
    b = mc.decompose(c, mc.CanonicalBasis(borrow=True))
    assert np.allclose(_unitary(b), _unitary(c), atol=1e-8)
    w = mc.decompose(c, mc.CanonicalBasis(borrow=True), wrap=True)
    assert np.allclose(_unitary(mc.decompose(w)), _unitary(c), atol=1e-8)


def test_gate_counts_per_basis():
    for n in range(4, 8):
        c = _cnx_circuit(n, 1)
        assert _count(
            mc.decompose(c, mc.CanonicalBasis(borrow=True)), mc.GateCX
        ) < _count(mc.decompose(c, mc.CanonicalBasis()), mc.GateCX)
        assert _count(mc.decompose(c, mc.QASMBasis(borrow=True)), mc.GateCCX) <= 8 * (
            n - 2
        )
        assert _count(
            mc.decompose(c, mc.RuleBasis(mc.CanonicalRewrite())), mc.GateCX
        ) == _count(mc.decompose(c, mc.CanonicalBasis()), mc.GateCX)

    # Lemma 7.2: n-2 borrowed qubits give 4(n-2) Toffolis, 7 T each.
    # QASMBasis keeps C3X, so its counts start at n = 4.
    for n in range(3, 8):
        c = _cnx_circuit(n, n - 2)
        if n >= 4:
            ccx = _count(mc.decompose(c, mc.QASMBasis(borrow=True)), mc.GateCCX)
            assert ccx == 4 * (n - 2)
        assert _tcount(mc.decompose(c, mc.CliffordTBasis(borrow=True))) == 28 * (n - 2)


def test_nearest_qubits_are_borrowed_first():
    c = mc.Circuit()
    c.push(mc.GateID(), 0)
    c.push(mc.Control(4, mc.GateX()), *range(3, 8))
    c.push(mc.GateID(), 11)
    b = mc.decompose(c, mc.QASMBasis(borrow=True))
    used = {q for inst in b for q in inst.qubits} - {0, 11, 3, 4, 5, 6, 7}
    assert used == {2, 8}


def test_no_width_no_borrowing():
    op = mc.Control(4, mc.GateX())
    assert mc.decompose(op, mc.CanonicalBasis(borrow=True)) == mc.decompose(op)
    inst = mc.Instruction(op, (2, 3, 4, 5, 6))
    b = mc.decompose(inst, mc.CanonicalBasis(borrow=True))
    assert {q for i in b for q in i.qubits} == {2, 3, 4, 5, 6}


def test_conditional_gates_borrow_too():
    c = mc.Circuit()
    c.push(mc.IfStatement(mc.Control(4, mc.GateX()), mc.BitString("1")), *range(5), 0)
    c.push(mc.GateID(), 6)
    b = mc.decompose(c, mc.QASMBasis(borrow=True))
    assert all(isinstance(i.operation, (mc.IfStatement, mc.GateID)) for i in b)
    assert _count(b, mc.IfStatement) == 8


def test_wrap():
    c = mc.Circuit()
    c.push(mc.Control(4, mc.GateX()), *range(5))
    c.push(mc.Control(4, mc.GateX()), *range(2, 7))
    c.push(mc.GateID(), 8)
    w = mc.decompose(c, mc.CanonicalBasis(borrow=True), wrap=True)
    # The body uses the two borrowed qubits as targets 5 and 6.
    assert len(w[0].qubits) == 7
    assert tuple(w[0].qubits[:5]) == (0, 1, 2, 3, 4)
    assert tuple(w[1].qubits[:5]) == (2, 3, 4, 5, 6)
    assert not set(w[1].qubits[5:]) & set(range(2, 7))
    # Both calls share one declaration.
    assert w[0].operation.decl is w[1].operation.decl
    assert np.allclose(_unitary(mc.decompose(w)), _unitary(c), atol=1e-8)


def test_losses_refuse_borrowing():
    c = mc.Circuit()
    c.push(mc.Loss(), 5)
    c.append(_cnx_circuit(4, 1))
    with pytest.raises(ValueError):
        mc.decompose(c, mc.CanonicalBasis(borrow=True))
    r = mc.Circuit()
    r.push(mc.IfStatement(mc.Reload(), mc.BitString("1")), 5, 0)
    r.append(_cnx_circuit(4, 1))
    with pytest.raises(ValueError):
        mc.decompose(r, mc.CanonicalBasis(borrow=True))
    assert isinstance(mc.decompose(c), mc.Circuit)

    # Losses nested in other operations count too.
    body = mc.Circuit()
    body.push(mc.Loss(), 0)
    lossy = mc.Block(body)
    nested = [
        mc.Repeat(2, lossy),
        mc.BlockDecl("lossy", [], body)(),
        mc.WhileStatement(lossy, mc.BitString("1")),
    ]
    for op in nested:
        n = mc.Circuit()
        n.push(op, 5, *([0] if op.num_bits > 0 else []))
        n.append(_cnx_circuit(4, 1))
        with pytest.raises(ValueError):
            mc.decompose(n, mc.CanonicalBasis(borrow=True))


def test_cnx_without_free_qubits():
    # The ancilla-free branch of the Barenco construction.
    b = cx_decompose(mc.Circuit(), [0, 1, 2, 3], 4, [])
    c = mc.Circuit()
    c.push(mc.Control(4, mc.GateX()), *range(5))
    assert np.allclose(_unitary(mc.decompose(b)), _unitary(c), atol=1e-8)


def test_overridden_decompose_is_still_called():
    class MyBasis(mc.CanonicalBasis):
        def decompose(self, op, qubits, bits, zvars):
            if isinstance(op, mc.GateH):
                c = mc.Circuit()
                c.push(mc.GateU(np.pi / 2, 0, np.pi), *qubits)
                c.push(mc.GateU(0, 0, 0), *qubits)
                return c
            return super().decompose(op, qubits, bits, zvars)

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateID(), 4)
    assert len(mc.decompose(c, MyBasis(borrow=True))) == 3


def _roundtrip(c):
    from mimiqcircuits.proto.circuitproto import fromproto_circuit, toproto_circuit

    return fromproto_circuit(toproto_circuit(c))


def _and_body():
    body = mc.Circuit()
    body.push(mc.GateCCX(), 0, 1, 2)
    body.push(mc.GateCX(), 2, 3)
    body.push(mc.GateCCX(), 0, 1, 2)
    return body


def test_markers():
    c = mc.Circuit()
    c.push(mc.AncillaAlloc(), 2)
    c.push(mc.GateX(), 2)
    c.push(mc.GateX(), 2)
    c.push(mc.AncillaFree(), 2)
    assert str(mc.AncillaAlloc()) == "AncillaAlloc"
    assert _roundtrip(c) == c
    for basis in (
        mc.CanonicalBasis(),
        mc.CliffordTBasis(),
        mc.QASMBasis(),
        mc.StimBasis(),
        mc.FlattenedBasis(),
    ):
        assert basis.isterminal(mc.AncillaAlloc())
        assert basis.isterminal(mc.AncillaFree())


def test_qasm_drops_annotations():
    from mimiqcircuits.qasm import dumps

    c = mc.Circuit()
    c.push(mc.AncillaAlloc(), 2)
    c.push(mc.GateH(), 0)
    c.push(mc.Lost(), 1)
    c.push(mc.AncillaFree(), 2)
    out = dumps(c)
    assert "qreg q[3];" in out
    assert "Ancilla" not in out and "Lost" not in out


def test_pauli_based_computation_drops_the_markers():
    c = mc.Circuit()
    c.push(mc.AncillaAlloc(), 1)
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateT(), 1)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.AncillaFree(), 1)
    c.push(mc.Tick())
    pbc = mc.pauli_based_computation(c)
    assert not any(
        isinstance(i.operation, (mc.AncillaAlloc, mc.AncillaFree)) for i in pbc
    )
    assert any(isinstance(i.operation, mc.Tick) for i in pbc)


def test_gatedecl_num_ancillas():
    d = mc.GateDecl("and", (), _and_body(), num_ancillas=1)
    assert d.num_ancillas == 1
    assert d().num_ancillas == 1
    assert mc.GateDecl("and", (), _and_body()).num_ancillas == 0
    assert mc.GateH().num_ancillas == 0
    with pytest.raises(ValueError):
        mc.GateDecl("and", (), _and_body(), num_ancillas=5)
    with pytest.raises(ValueError):
        mc.GateDecl("and", (), _and_body(), num_ancillas=-1)
    c = mc.Circuit()
    c.push(d(), 0, 1, 2, 3)
    assert _roundtrip(c)[0].operation.num_ancillas == 1


def test_block_num_ancillas():
    b = mc.Block(_and_body(), num_ancillas=1)
    assert b.num_ancillas == 1
    assert b != mc.Block(_and_body())
    assert mc.Block(4, 0, 0, list(_and_body()), num_ancillas=2).num_ancillas == 2
    theta = se.Symbol("theta")
    sym = mc.Block(2, 0, 0, [mc.Instruction(mc.GateRX(theta), (0,))], num_ancillas=1)
    assert sym.evaluate({theta: 0.1}).num_ancillas == 1
    with pytest.raises(ValueError):
        mc.Block(_and_body(), num_ancillas=5)
    c = mc.Circuit()
    c.push(b, 0, 1, 2, 3)
    assert _roundtrip(c)[0].operation.num_ancillas == 1


def test_blockdecl_num_ancillas():
    d = mc.BlockDecl("and", (), _and_body(), num_ancillas=1)
    assert d.num_ancillas == 1
    assert d().num_ancillas == 1
    assert d != mc.BlockDecl("and", (), _and_body())
    assert hash(d) != hash(mc.BlockDecl("and", (), _and_body()))
    b = mc.Block(_and_body(), num_ancillas=2)
    assert mc.BlockDecl("and", (), b).num_ancillas == 2
    c = mc.Circuit()
    c.push(d(), 0, 1, 2, 3)
    assert _roundtrip(c)[0].operation.num_ancillas == 1


def _acts_as(d, c, n):
    """`d` acts on qubits 0..n-1 as `c` does, with the other qubits in |0>.

    For every input with the qubits from n on in |0>, `d` returns them to
    |0>. Qubit 0 is the most significant bit of a matrix index, so these
    inputs are x * 2^(m-n).
    """
    strip = mc.Circuit(
        [i for i in d if not isinstance(i.operation, mc.AbstractAnnotation)]
    )
    m = max(d.num_qubits(), n)
    if strip.num_qubits() < m:
        strip.push(mc.GateID(), m - 1)
    ref = mc.Circuit(list(c))
    if ref.num_qubits() < n:
        ref.push(mc.GateID(), n - 1)
    V = _unitary(strip)
    U = _unitary(ref)
    sel = [x * 2 ** (m - n) for x in range(2**n)]
    sub = V[np.ix_(sel, sel)]
    return np.allclose(sub, U, atol=1e-8)


def _cnx(n, *qubits):
    c = mc.Circuit()
    c.push(mc.Control(n, mc.GateX()), *qubits)
    return c


@pytest.mark.parametrize("n,m", [(n, m) for n in range(3, 6) for m in range(1, n - 1)])
def test_clean_cnx(n, m):
    c = _cnx(n, *range(n + 1))
    d = mc.decompose(c, mc.CanonicalBasis(max_ancillas=m))
    assert mc.ancillas(d) == list(range(n + 1, n + 1 + m))
    mc.check_ancillas(d)
    assert _acts_as(d, c, n + 1)


def test_clean_t_counts():
    for n in range(3, 8):
        c = _cnx(n, *range(n + 1))
        # n-2 relative-phase Toffolis twice (4 T each) and one Toffoli.
        d = mc.decompose(c, mc.CliffordTBasis(max_ancillas=n - 2))
        assert _tcount(d) == 8 * (n - 2) + 7
        # A larger budget does not change the result.
        d = mc.decompose(c, mc.CliffordTBasis(max_ancillas=n))
        assert _tcount(d) == 8 * (n - 2) + 7


def test_freed_ancillas_are_reused():
    c = _cnx(4, *range(5))
    c.append(_cnx(4, *range(1, 6)))
    d = mc.decompose(c, mc.CanonicalBasis(max_ancillas=2))
    assert mc.ancillas(d) == [6, 7]
    assert d.num_qubits() == 8
    assert _acts_as(d, c, 6)


def test_markers_of_the_input_are_read():
    # max_ancillas bounds the new qubits; freed ones of the input come on top.
    c = mc.Circuit()
    c.push(mc.AncillaAlloc(), 4)
    c.push(mc.GateX(), 4)
    c.push(mc.GateX(), 4)
    c.push(mc.AncillaFree(), 4)
    c.append(_cnx(3, *range(4)))
    d = mc.decompose(c, mc.CanonicalBasis(max_ancillas=1))
    # Qubit 4 is free again after its window, and is used before a new one.
    assert mc.ancillas(d) == [4]
    assert d.num_qubits() == 5

    # Touched after its window, it is no longer known to be in |0>.
    c.push(mc.GateH(), 4)
    c.append(_cnx(3, *range(4)))
    d = mc.decompose(c, mc.CanonicalBasis(max_ancillas=1))
    assert mc.ancillas(d) == [4, 5]

    # Nested windows: the qubit is free only once both are closed.
    n = mc.Circuit()
    n.push(mc.AncillaAlloc(), 4)
    n.push(mc.AncillaAlloc(), 4)
    n.push(mc.AncillaFree(), 4)
    n.append(_cnx(3, *range(4)))
    n.push(mc.AncillaFree(), 4)
    n.append(_cnx(3, *range(4)))
    d = mc.decompose(n, mc.CanonicalBasis(max_ancillas=1))

    def allocs(q):
        return sum(
            isinstance(i.operation, mc.AncillaAlloc) and i.qubits[0] == q for i in d
        )

    assert allocs(5) == 1
    assert allocs(4) == 3


def test_clean_before_borrowed():
    c = _cnx(4, *range(5))
    c.push(mc.GateID(), 6)
    d = mc.decompose(c, mc.CanonicalBasis(borrow=True, max_ancillas=2))
    assert mc.ancillas(d) == [7, 8]
    assert not any(5 in i.qubits for i in d)


def test_clean_conditional():
    c = mc.Circuit()
    c.push(mc.IfStatement(mc.Control(4, mc.GateX()), mc.BitString("1")), *range(5), 0)
    d = mc.decompose(c, mc.QASMBasis(max_ancillas=2))
    markers = [
        i for i in d if isinstance(i.operation, (mc.AncillaAlloc, mc.AncillaFree))
    ]
    assert len(markers) == 4
    assert all(
        isinstance(i.operation, (mc.IfStatement, mc.AncillaAlloc, mc.AncillaFree))
        for i in d
    )
    mc.check_ancillas(d)


def test_clean_wrap():
    c = _cnx(4, *range(5))
    c.append(_cnx(4, *range(1, 6)))
    w = mc.decompose(c, mc.CanonicalBasis(max_ancillas=2), wrap=True)
    mc.check_ancillas(w)
    calls = [i for i in w if isinstance(i.operation, mc.GateCall)]
    assert len(calls) == 2
    assert all(i.operation.num_ancillas == 2 for i in calls)
    assert tuple(calls[0].qubits) == (0, 1, 2, 3, 4, 6, 7)
    assert calls[0].operation.decl is calls[1].operation.decl
    # The declaration body has no markers.
    assert not any(
        isinstance(i.operation, mc.AbstractAnnotation)
        for i in calls[0].operation.decl.circuit
    )
    assert _acts_as(mc.decompose(w), c, 6)

    # Borrowed and clean qubits together: data, borrowed, clean.
    c = _cnx(5, *range(6))
    c.push(mc.GateID(), 6)
    w = mc.decompose(c, mc.CanonicalBasis(borrow=True, max_ancillas=1), wrap=True)
    (call,) = [i for i in w if i.operation.num_ancillas > 0]
    assert call.qubits[-1] == 7
    assert _acts_as(mc.decompose(w), c, 7)


def test_merge_ancilla_windows():
    c = _cnx(4, *range(5))
    c.append(_cnx(4, *range(5)))
    w = mc.decompose(c, mc.CanonicalBasis(max_ancillas=2), wrap=True)
    m = mc.merge_ancilla_windows(w)
    assert len(m) == len(w) - 4
    assert isinstance(m[2].operation, mc.GateCall)
    assert isinstance(m[3].operation, mc.GateCall)
    mc.check_ancillas(m)

    # Something on the wire in between keeps the windows apart.
    k = mc.Circuit()
    k.push(mc.AncillaFree(), 0)
    k.push(mc.GateX(), 0)
    k.push(mc.AncillaAlloc(), 0)
    assert len(mc.merge_ancilla_windows(k)) == 3


def test_check_ancillas():
    with pytest.raises(ValueError):
        mc.check_ancillas(mc.Circuit().push(mc.AncillaFree(), 0))
    with pytest.raises(ValueError):
        mc.check_ancillas(mc.Circuit().push(mc.AncillaAlloc(), 0))
    body = mc.Circuit().push(mc.GateCCX(), 0, 1, 2)
    decl = mc.GateDecl("and", (), body, num_ancillas=1)
    with pytest.raises(ValueError):
        mc.check_ancillas(mc.Circuit().push(decl(), 0, 1, 2))
    ok = mc.Circuit()
    ok.push(mc.AncillaAlloc(), 2)
    ok.push(decl(), 0, 1, 2)
    ok.push(mc.AncillaFree(), 2)
    mc.check_ancillas(ok)


@pytest.mark.parametrize("seed", range(20))
def test_clean_random_circuits(seed):
    rng = np.random.default_rng(seed)
    n = 5
    c = mc.Circuit()
    for _ in range(4):
        k = int(rng.integers(2, 5))
        qs = [int(q) for q in rng.permutation(n)[: k + 1]]
        c.push(mc.Control(k, mc.GateX()), *qs)
        c.push(mc.GateH(), int(rng.integers(n)))
    m = int(rng.integers(1, 3))
    basis = mc.CanonicalBasis(borrow=bool(rng.integers(2)), max_ancillas=m)
    d = mc.decompose(c, basis, wrap=bool(rng.integers(2)))
    mc.check_ancillas(d)
    assert d.num_qubits() <= n + m
    # Flattening again reads the markers and keeps the width.
    f = mc.decompose(d, basis)
    mc.check_ancillas(f)
    assert f.num_qubits() == d.num_qubits()
    assert _acts_as(f, c, n)


def _coherent(d):
    """Replace each measured uncompute of an AND by its coherent equivalent.

    The measured uncompute (MeasureReset, then a CZ conditioned on the bit)
    and a CZ controlled by the ancilla followed by H on it both return the
    ancilla to |0> with the same state on the other qubits.
    """
    out = mc.Circuit()
    insts = list(d)
    i = 0
    while i < len(insts):
        op = insts[i].operation
        if isinstance(op, mc.MeasureReset):
            t = insts[i].qubits[0]
            cond = insts[i + 1]
            assert isinstance(cond.operation, mc.IfStatement)
            assert isinstance(cond.operation.op, mc.GateCZ)
            a, b = cond.qubits
            out.push(mc.Control(2, mc.GateZ()), t, a, b)
            out.push(mc.GateH(), t)
            i += 2
        else:
            if not isinstance(op, mc.AbstractClassical):
                out.append(mc.Circuit([insts[i]]))
            i += 1
    return out


def _cnx_with_bit(n):
    """A C_n(X) on qubits 0..n and one bit, which measuring rules need."""
    c = _cnx(n, *range(n + 1))
    c.push(mc.SetBit0(), 0)
    return c


def test_measured_t_counts():
    for n in range(2, 7):
        basis = mc.CliffordTBasis(max_ancillas=n - 1, allow_measurement=True)
        d = mc.decompose(_cnx_with_bit(n), basis)
        # n-1 ANDs with 4 T each, none to undo them.
        assert _tcount(d) == 4 * (n - 1)
        assert _count(d, mc.MeasureReset) == n - 1
        mc.check_ancillas(d)


@pytest.mark.parametrize("n,m", [(n, m) for n in range(2, 5) for m in range(1, n)])
def test_measured_cnx(n, m):
    c = _cnx_with_bit(n)
    basis = mc.CliffordTBasis(max_ancillas=m, allow_measurement=True)
    d = mc.decompose(c, basis)
    assert mc.ancillas(d) == list(range(n + 1, n + 1 + m))
    # Measurements write one bit, after those of the circuit.
    assert all(
        tuple(i.bits) == (1,) for i in d if isinstance(i.operation, mc.MeasureReset)
    )
    assert _acts_as(_coherent(d), _cnx(n, *range(n + 1)), n + 1)


def test_measured_refused_without_bits():
    c = mc.Circuit().push(mc.GateCCX(), 0, 1, 2)
    with pytest.raises(ValueError):
        mc.decompose(c, mc.CliffordTBasis(max_ancillas=1, allow_measurement=True))
    assert isinstance(mc.decompose(c, mc.CliffordTBasis()), mc.Circuit)


def test_measured_conditional_stays_unitary():
    c = mc.Circuit()
    c.push(mc.IfStatement(mc.Control(3, mc.GateX()), mc.BitString("1")), *range(4), 0)
    basis = mc.CliffordTBasis(max_ancillas=2, allow_measurement=True)
    d = mc.decompose(c, basis)
    assert not any(isinstance(i.operation, (mc.MeasureReset, mc.Measure)) for i in d)
    mc.check_ancillas(d)


def test_measured_wrap():
    c = _cnx_with_bit(3)
    basis = mc.CliffordTBasis(max_ancillas=2, allow_measurement=True)
    w = mc.decompose(c, basis, wrap=True)
    mc.check_ancillas(w)
    calls = [i for i in w if isinstance(i.operation, mc.BlockCall)]
    assert len(calls) == 1
    assert calls[0].operation.num_ancillas == 2
    assert tuple(calls[0].bits) == (1,)
    f = mc.decompose(w, basis)
    assert _acts_as(_coherent(f), _cnx(3, *range(4)), 4)

    # The same gate twice shares one cached declaration.
    c.push(mc.Control(3, mc.GateX()), *range(4))
    w = mc.decompose(c, basis, wrap=True)
    calls = [i for i in w if isinstance(i.operation, mc.BlockCall)]
    assert len(calls) == 2
    assert calls[0].operation.decl is calls[1].operation.decl

    # A cache shared with a basis that may not measure does not hand back
    # the measuring declaration.
    cache = {}
    list(mc.DecomposeIterator(c, basis, wrap=True, cache=cache))
    unitary = mc.CliffordTBasis(max_ancillas=2)
    u = mc.Circuit(list(mc.DecomposeIterator(c, unitary, wrap=True, cache=cache)))
    assert not any(isinstance(i.operation, mc.BlockCall) for i in u)


def test_pauli_based_computation_gets_no_ancillas():
    c = mc.Circuit()
    c.push(mc.GateCCX(), 0, 1, 2)
    c.push(mc.Measure(), 2, 0)
    basis = mc.CliffordTBasis(max_ancillas=2, allow_measurement=True)
    pbc = mc.pauli_based_computation(c, basis=basis)
    assert pbc.num_qubits() <= 4
