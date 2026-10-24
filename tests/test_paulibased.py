import random
from functools import reduce

import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.matrices import pauli_string_matrix
from mimiqcircuits.proto.circuitproto import fromproto_circuit, toproto_circuit


def _apply_gate(U, qs, psi, n):
    # Qubit 0 is the most significant factor, as in PauliString.matrix.
    k = len(qs)
    t = psi.reshape([2] * n)
    t = np.tensordot(U.reshape([2] * (2 * k)), t, axes=(list(range(k, 2 * k)), qs))
    t = np.moveaxis(t, list(range(k)), qs)
    return t.reshape(-1)


_MEASUREMENT_BASIS = {mc.Measure: "Z", mc.MeasureX: "X", mc.MeasureY: "Y"}


def _embed(K, q, n):
    # Qubit 0 is the most significant factor.
    return reduce(np.kron, [np.eye(2**q), K, np.eye(2 ** (n - q - 1))])


def _distribution(c, n, nb, psi0, keep=None):
    """Exact outcome distribution of a circuit made of unitary gates,
    Measure, MeasureX, MeasureY, MeasurePauli, Reset, Not, SetBit0, SetBit1,
    And, Xor, PrepareStateT and IfStatement, by following every
    measurement and Kraus branch. Only the first ``keep`` bits are reported."""

    def apply(op, qs, bs, branch):
        psi, bits, p = branch
        if isinstance(op, mc.IfStatement):
            nop = op.op.num_bits
            cond = bs[nop:]
            if any(bits[b] != op.bitstring[i] for i, b in enumerate(cond)):
                return [branch]
            return apply(op.op, qs, bs[:nop], branch)
        if type(op) in _MEASUREMENT_BASIS or isinstance(op, mc.MeasurePauli):
            s = _MEASUREMENT_BASIS.get(type(op)) or op.pstring()
            Ppsi = _apply_gate(pauli_string_matrix(s), qs, psi, n)
            out = []
            for sgn, val in ((1, False), (-1, True)):
                phi = (psi + sgn * Ppsi) / 2
                w = np.vdot(phi, phi).real
                if w <= 1e-14:
                    continue
                nbits = list(bits)
                nbits[bs[0]] = val
                out.append((phi / np.sqrt(w), nbits, p * w))
            return out
        if isinstance(op, mc.Reset):
            # Project on Z = ±1 and flip the -1 branch back to |0⟩.
            Zpsi = _apply_gate(pauli_string_matrix("Z"), qs, psi, n)
            out = []
            for sgn, flip in ((1, False), (-1, True)):
                phi = (psi + sgn * Zpsi) / 2
                w = np.vdot(phi, phi).real
                if w <= 1e-14:
                    continue
                if flip:
                    phi = _apply_gate(pauli_string_matrix("X"), qs, phi, n)
                out.append((phi / np.sqrt(w), bits, p * w))
            return out
        if isinstance(op, mc.PrepareStateT):
            out = []
            for K in op.krausmatrices():
                K = np.array(K.tolist(), dtype=complex)
                phi = _embed(K, qs[0], n) @ psi
                w = np.vdot(phi, phi).real
                if w > 1e-14:
                    out.append((phi / np.sqrt(w), bits, p * w))
            return out
        if isinstance(op, mc.Not):
            nbits = list(bits)
            nbits[bs[0]] = not nbits[bs[0]]
            return [(psi, nbits, p)]
        if isinstance(op, (mc.SetBit0, mc.SetBit1)):
            nbits = list(bits)
            nbits[bs[0]] = isinstance(op, mc.SetBit1)
            return [(psi, nbits, p)]
        if isinstance(op, mc.And):
            nbits = list(bits)
            nbits[bs[0]] = all(bits[b] for b in bs[1:])
            return [(psi, nbits, p)]
        if isinstance(op, mc.Xor):
            nbits = list(bits)
            nbits[bs[0]] = sum(bits[b] for b in bs[1:]) % 2 == 1
            return [(psi, nbits, p)]
        U = np.array(op.matrix().tolist(), dtype=complex)
        return [(_apply_gate(U, qs, psi, n), bits, p)]

    branches = [(psi0, [False] * nb, 1.0)]
    for inst in c:
        qs, bs = list(inst.qubits), list(inst.bits)
        branches = [nxt for br in branches for nxt in apply(inst.operation, qs, bs, br)]

    keep = nb if keep is None else keep
    dist = {}
    for _, bits, p in branches:
        key = tuple(bits[:keep])
        dist[key] = dist.get(key, 0.0) + p
    return dist


def _same_distribution(a, b, atol=1e-9):
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= atol for k in set(a) | set(b))


def _randstate(rng, n):
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return psi / np.linalg.norm(psi)


class TestMeasurePauli:
    def test_construction(self):
        with pytest.raises(ValueError):
            mc.MeasurePauli("III")
        with pytest.raises(ValueError):
            mc.MeasurePauli("XA")
        with pytest.raises(TypeError, match="PauliString or a str"):
            mc.MeasurePauli(123)
        m = mc.MeasurePauli("XIZ")
        assert m.num_qubits == 3
        assert m.num_bits == 1
        assert m.pstring() == "XIZ"
        assert mc.MeasurePauli("XY") == mc.MeasurePauli(mc.PauliString("XY"))
        assert mc.MeasurePauli("XY") != mc.MeasurePauli("YX")
        with pytest.raises(TypeError):
            mc.MeasurePauli("X").inverse()

    def test_str(self):
        assert str(mc.MeasurePauli("XIZ")) == "MP(XIZ)"
        assert repr(mc.MeasurePauli("XIZ")) == "MP(XIZ)"

    def test_proto_roundtrip(self):
        c = mc.Circuit()
        c.push(mc.MeasurePauli("XIYZ"), 3, 1, 0, 2, 1)
        restored = fromproto_circuit(toproto_circuit(c))
        assert len(restored) == 1
        inst = restored.instructions[0]
        assert inst.operation == mc.MeasurePauli("XIYZ")
        assert inst.qubits == (3, 1, 0, 2)
        assert inst.bits == (1,)

    def test_saveload_proto(self, tmp_path):
        c = mc.Circuit()
        c.push(mc.MeasurePauli("XIYZ"), 3, 1, 0, 2, 1)
        fname = str(tmp_path / "mp.pb")
        c.saveproto(fname)
        restored = mc.Circuit.loadproto(fname)
        assert restored.instructions[0].operation == mc.MeasurePauli("XIYZ")
        assert restored.instructions[0].qubits == (3, 1, 0, 2)

    def test_decompose_widest_target(self):
        # The unmeasured widest target is kept so the decomposition spans it.
        c = mc.Circuit().push(mc.MeasurePauli("XZI"), 0, 1, 2, 0)
        d = c.decompose(basis=mc.CanonicalBasis())
        assert d.num_qubits() == 3
        first = mc.decompose_step(c).instructions[0]
        assert isinstance(first.operation, mc.GateID)
        assert first.qubits == (2,)

    @pytest.mark.parametrize(
        "seed, s",
        enumerate(["Z", "X", "Y", "XZ", "ZX", "YY", "XIZ", "IYI", "ZXYI", "IIXY"]),
    )
    def test_decomposition_statistics(self, seed, s):
        # The canonical decomposition must give the same outcome statistics,
        # whatever the order of the target qubits.
        rng = np.random.default_rng(seed)
        n = len(s)
        qs = list(rng.permutation(n))
        m = mc.Circuit().push(mc.MeasurePauli(s), *qs, 0)
        psi = _randstate(rng, n)
        ref = _distribution(m, n, 1, psi)
        dec = _distribution(mc.decompose(m), n, 1, psi)
        assert _same_distribution(ref, dec)

    @pytest.mark.parametrize(
        "basis", [mc.StimBasis(), mc.QASMBasis(), mc.CliffordTBasis()]
    )
    def test_decompose_clifford_bases(self, basis):
        # The Y rotation uses SDG, H and S, so the QASM basis decomposes it; the
        # Stim and Clifford+T bases keep it as MPP.
        rng = np.random.default_rng(11)
        m = mc.Circuit().push(mc.MeasurePauli("XYZ"), 0, 1, 2, 0)
        psi = _randstate(rng, 3)
        d = mc.decompose(m, basis=basis)
        assert _same_distribution(
            _distribution(m, 3, 1, psi), _distribution(d, 3, 1, psi)
        )

    def test_decompose_y(self):
        d = mc.decompose_step(mc.Circuit().push(mc.MeasurePauli("Y"), 0, 0))
        assert [i.operation for i in d] == [
            mc.GateSDG(),
            mc.GateH(),
            mc.Measure(),
            mc.GateH(),
            mc.GateS(),
        ]


class TestPauliBasedComputation:
    def test_small_example(self):
        c = mc.Circuit()
        c.push(mc.GateH(), 0)
        c.push(mc.GateCX(), 0, 1)
        c.push(mc.GateT(), 1)
        c.push(mc.Measure(), range(2), range(2))
        p = mc.pauli_based_computation(c)
        assert len(p) == 3
        r = p.instructions[0].operation
        assert isinstance(r, mc.RPauli)
        assert str(r.pauli) == "XZ"
        assert float(r.theta) == pytest.approx(np.pi / 4)
        assert p.instructions[0].qubits == (0, 1)
        assert p.instructions[1].operation == mc.MeasurePauli("X")
        assert p.instructions[1].qubits == (0,)
        assert p.instructions[2].operation == mc.MeasurePauli("XZ")
        assert p.instructions[2].qubits == (0, 1)
        assert p.instructions[2].bits == (1,)

    def test_circuit_method(self):
        c = mc.Circuit()
        c.push(mc.GateH(), 0)
        c.push(mc.Measure(), 0, 0)
        p = c.pauli_based_computation()
        assert [i.operation for i in p] == [mc.MeasurePauli("X")]

    def test_negated_observable_flips_bit(self):
        c = mc.Circuit()
        c.push(mc.GateX(), 0)
        c.push(mc.Measure(), 0, 0)
        p = mc.pauli_based_computation(c)
        assert [i.operation for i in p] == [mc.MeasurePauli("Z"), mc.Not()]

    def test_negative_angle(self):
        c = mc.Circuit()
        c.push(mc.GateX(), 0)
        c.push(mc.GateT(), 0)
        p = mc.pauli_based_computation(c)
        r = p.instructions[0].operation
        assert str(r.pauli) == "Z"
        assert float(r.theta) == pytest.approx(-np.pi / 4)

    def test_classical_and_barriers(self):
        c = mc.Circuit()
        c.push(mc.GateH(), 0)
        c.push(mc.Barrier(2), 0, 1)
        c.push(mc.GateID(), 1)
        c.push(mc.Measure(), 0, 0)
        c.push(mc.Not(), 1)
        p = mc.pauli_based_computation(c)
        assert [i.operation for i in p] == [mc.MeasurePauli("X"), mc.Not()]

    def test_errors(self):
        with pytest.raises(ValueError):
            mc.pauli_based_computation(
                mc.Circuit().push(mc.GateRX(0.3), 0), basis=mc.CanonicalBasis()
            )

    def test_random_clifford_t(self):
        # Random Clifford+T circuits with mid-circuit measurements: the
        # rewritten circuit has no Clifford left and reproduces the outcome
        # statistics on a random input state.
        gates1 = [
            mc.GateH(),
            mc.GateS(),
            mc.GateSDG(),
            mc.GateT(),
            mc.GateTDG(),
            mc.GateX(),
            mc.GateY(),
            mc.GateZ(),
            mc.GateSX(),
            mc.GateSYDG(),
        ]
        gates2 = [
            mc.GateCX(),
            mc.GateCZ(),
            mc.GateCY(),
            mc.GateSWAP(),
            mc.GateISWAP(),
            mc.GateECR(),
            mc.GateDCX(),
        ]
        rnd = random.Random(42)
        rng = np.random.default_rng(42)
        n, nb = 3, 5
        for _ in range(20):
            c = mc.Circuit()
            b = -1
            for _ in range(30):
                r = rnd.random()
                if r < 0.5:
                    c.push(rnd.choice(gates1), rnd.randrange(n))
                elif r < 0.9:
                    c.push(rnd.choice(gates2), *rnd.sample(range(n), 2))
                elif b + 1 < nb - n:
                    b += 1
                    meas = rnd.choice([mc.Measure(), mc.MeasureX(), mc.MeasureY()])
                    c.push(meas, rnd.randrange(n), b)
            c.push(mc.GateCCX(), *rnd.sample(range(n), n))
            b += 1
            c.push(mc.MeasurePauli("XYZ"), *range(n), b)
            for q in range(nb - 1 - b):
                c.push(mc.Measure(), q % n, b + 1 + q)

            p = mc.pauli_based_computation(c)
            ops = [i.operation for i in p]
            assert all(isinstance(o, (mc.RPauli, mc.MeasurePauli, mc.Not)) for o in ops)
            assert all(
                abs(abs(float(o.theta)) - np.pi / 4) < 1e-12
                for o in ops
                if isinstance(o, mc.RPauli)
            )

            psi = _randstate(rng, n)
            ref = mc.decompose(c, basis=mc.CliffordTBasis())
            assert _same_distribution(
                _distribution(ref, n, nb, psi), _distribution(p, n, nb, psi)
            )

            # The angles are exact multiples of pi and survive a proto round
            # trip unchanged.
            assert all(
                o.theta in (se.pi / 4, -se.pi / 4)
                for o in ops
                if isinstance(o, mc.RPauli)
            )
            assert fromproto_circuit(toproto_circuit(p)) == p


class TestPrepareStateT:
    m = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)

    def test_kraus(self):
        Ks = [
            np.array(K.tolist(), dtype=complex)
            for K in mc.PrepareStateT().krausmatrices()
        ]
        assert np.allclose(sum(K.conj().T @ K for K in Ks), np.eye(2))
        # Any input state ends up in |m⟩⟨m|.
        rng = np.random.default_rng(3)
        for _ in range(5):
            psi = _randstate(rng, 1)
            rho = sum(K @ np.outer(psi, psi.conj()) @ K.conj().T for K in Ks)
            assert np.allclose(rho, np.outer(self.m, self.m.conj()))

    def test_decomposition(self):
        # The canonical decomposition, Reset then H then T, prepares the same state.
        d = mc.PrepareStateT().decompose()
        assert [i.operation for i in d] == [mc.Reset(), mc.GateH(), mc.GateT()]
        TH = np.array(
            (mc.GateT().matrix() * mc.GateH().matrix()).tolist(), dtype=complex
        )
        assert np.allclose(TH @ [1, 0], self.m)
        with pytest.raises(TypeError):
            mc.PrepareStateT().inverse()

    def test_str(self):
        assert str(mc.PrepareStateT()) == "PrepareStateT"

    def test_proto_roundtrip(self, tmp_path):
        c = mc.Circuit().push(mc.PrepareStateT(), 2)
        restored = fromproto_circuit(toproto_circuit(c))
        assert restored.instructions[0].operation == mc.PrepareStateT()
        assert restored.instructions[0].qubits == (2,)
        fname = str(tmp_path / "magic.pb")
        c.saveproto(fname)
        assert mc.Circuit.loadproto(fname) == c


class TestPauliBasedComputationMagicStates:
    def test_small_example(self):
        # Each π/8 rotation becomes PrepareStateT, a P⊗Z measurement and an X
        # or Y measurement of the ancilla chosen by the corrected outcome.
        c = mc.Circuit()
        c.push(mc.GateH(), 0)
        c.push(mc.GateCX(), 0, 1)
        c.push(mc.GateT(), 1)
        c.push(mc.Measure(), range(2), range(2))
        p = mc.pauli_based_computation(c, magic_states=True)
        assert p.num_qubits() == 3
        assert sum(isinstance(i.operation, mc.PrepareStateT) for i in p) == 1
        assert not any(
            isinstance(i.operation, (mc.RPauli, mc.GateT, mc.GateTDG)) for i in p
        )
        assert p.instructions[1].operation == mc.MeasurePauli("XZZ")
        assert p.instructions[1].qubits == (0, 1, 2)
        assert p.instructions[1].bits == (2,)
        ifx, ify = p.instructions[2], p.instructions[3]
        assert ifx.operation.op == mc.MeasurePauli("X")
        assert ifx.operation.bitstring == mc.BitString("0")
        assert ify.operation.op == mc.MeasurePauli("Y")
        assert ify.operation.bitstring == mc.BitString("1")
        assert ifx.qubits == (2,) and ifx.bits == (3, 2)

    def test_no_t_gates(self):
        c = mc.Circuit()
        c.push(mc.GateH(), 0)
        c.push(mc.Measure(), 0, 0)
        assert c.pauli_based_computation(
            magic_states=True
        ) == mc.pauli_based_computation(c)

    def test_random_clifford_t(self):
        # Random Clifford+T circuits with mid-circuit measurements: the
        # injected circuit reproduces the statistics of the original bits.
        # Kept small, as every injection doubles the branches three times.
        gates1 = [
            mc.GateH(),
            mc.GateS(),
            mc.GateSDG(),
            mc.GateT(),
            mc.GateTDG(),
            mc.GateX(),
            mc.GateY(),
            mc.GateZ(),
            mc.GateSX(),
        ]
        gates2 = [mc.GateCX(), mc.GateCZ(), mc.GateCY(), mc.GateSWAP(), mc.GateISWAP()]
        rnd = random.Random(7)
        rng = np.random.default_rng(7)
        n, nb = 3, 4
        ntested = 0
        for _ in range(40):
            c = mc.Circuit()
            b = 0
            for _ in range(14):
                r = rnd.random()
                if r < 0.55:
                    c.push(rnd.choice(gates1), rnd.randrange(n))
                elif r < 0.92:
                    c.push(rnd.choice(gates2), *rnd.sample(range(n), 2))
                elif b < nb - n:
                    meas = rnd.choice([mc.Measure(), mc.MeasureX(), mc.MeasureY()])
                    c.push(meas, rnd.randrange(n), b)
                    b += 1
            for q in range(nb - b):
                c.push(mc.Measure(), q % n, b + q)
            nt = sum(isinstance(i.operation, (mc.GateT, mc.GateTDG)) for i in c)
            if not 1 <= nt <= 4:
                continue
            ntested += 1

            p = mc.pauli_based_computation(c, magic_states=True)
            assert not any(
                isinstance(i.operation, (mc.RPauli, mc.GateT, mc.GateTDG)) for i in p
            )

            psi = _randstate(rng, n)
            psianc = np.kron(psi, [1.0, 0.0])
            ref = mc.decompose(c, basis=mc.CliffordTBasis())
            assert _same_distribution(
                _distribution(ref, n, nb, psi),
                _distribution(p, n + 1, p.num_bits(), psianc, keep=nb),
            )
            assert fromproto_circuit(toproto_circuit(p)) == p
        assert ntested >= 15

    def test_bounded_arity(self):
        # The frame is updated and read with 3-bit Xor only, so the classical
        # operations keep the same arities however long the circuit is.
        rnd = random.Random(11)
        n = 12
        c = mc.Circuit()
        for _ in range(600):
            if rnd.random() < 0.6:
                g = rnd.choice([mc.GateH(), mc.GateS(), mc.GateT(), mc.GateTDG()])
                c.push(g, rnd.randrange(n))
            else:
                c.push(mc.GateCX(), *rnd.sample(range(n), 2))
        c.push(mc.Measure(), range(n), range(n))
        p = mc.pauli_based_computation(c, magic_states=True)
        classical = [
            i.operation for i in p if isinstance(i.operation, mc.AbstractClassical)
        ]
        assert sum(isinstance(o, mc.Xor) for o in classical) > 100
        assert {o.num_bits for o in classical} <= {1, 3}
        assert all(isinstance(o, (mc.Xor, mc.Not)) for o in classical)
        assert fromproto_circuit(toproto_circuit(p)) == p


_NATIVE_MEASUREMENTS = (
    mc.MeasureX,
    mc.MeasureY,
    mc.MeasureXX,
    mc.MeasureYY,
    mc.MeasureZZ,
    mc.MeasurePauli,
    mc.ResetX,
    mc.ResetY,
    mc.MeasureReset,
    mc.MeasureResetX,
    mc.MeasureResetY,
)


def _expand_measurement(op, qubits, bits, zvars):
    """One step of the expansion of a measurement or reset into Clifford
    gates around Measure, Reset or MeasureReset. MeasureXX and MeasureYY are
    written out here, undoing the basis change exactly after the measurement."""
    c = mc.Circuit()
    if isinstance(op, (mc.MeasureXX, mc.MeasureYY)):
        q1, q2 = qubits
        if isinstance(op, mc.MeasureYY):
            c.push(mc.GateS(), q1)
            c.push(mc.GateS(), q2)
        c.push(mc.GateCX(), q1, q2)
        c.push(mc.GateH(), q1)
        c.push(mc.Measure(), q1, bits[0])
        c.push(mc.GateH(), q1)
        c.push(mc.GateCX(), q1, q2)
        if isinstance(op, mc.MeasureYY):
            c.push(mc.GateSDG(), q1)
            c.push(mc.GateSDG(), q2)
        return c
    return op._decompose(c, qubits, bits, zvars)


class _ReferenceBasis(mc.DecompositionBasis):
    """CliffordTBasis, with every measurement and reset expanded as by
    ``_expand_measurement``, so that the simulator can run the result."""

    def isterminal(self, op):
        if isinstance(op, _NATIVE_MEASUREMENTS):
            return False
        return mc.CliffordTBasis().isterminal(op)

    def decompose(self, op, qubits, bits, zvars):
        if isinstance(op, _NATIVE_MEASUREMENTS):
            return _expand_measurement(op, qubits, bits, zvars)
        return mc.CliffordTBasis().decompose(op, qubits, bits, zvars)


class _ExpandMeasurements(mc.DecompositionBasis):
    """Expands measurements and resets as ``_ReferenceBasis`` does, and keeps
    everything else, conditional operations included, unchanged."""

    def isterminal(self, op):
        return not isinstance(op, _NATIVE_MEASUREMENTS)

    def decompose(self, op, qubits, bits, zvars):
        return _expand_measurement(op, qubits, bits, zvars)


def _random_pauli_body(rnd):
    """A random operation made of Pauli operators only, and its number of qubits."""
    r = rnd.randint(1, 6)
    if r == 1:
        return rnd.choice([mc.GateX(), mc.GateY(), mc.GateZ()]), 1
    if r == 2:
        return mc.PauliString("".join(rnd.choice("IXYZ") for _ in range(3))), 3
    if r == 4:
        return mc.Parallel(2, rnd.choice([mc.GateX(), mc.GateY(), mc.GateZ()])), 2
    if r == 5:
        return mc.Repeat(rnd.randint(2, 3), mc.PauliString("XZ")), 2
    k = rnd.randint(2, 3)
    body = mc.Circuit().push(mc.GateID(), k - 1)
    for _ in range(3):
        g = rnd.choice([mc.GateX(), mc.GateY(), mc.GateZ(), mc.GateID()])
        body.push(g, rnd.randrange(k))
    if r == 3:
        return mc.Block(body), k
    return mc.GateDecl("paulis", (), body)(), k


def _random_conditional_circuit(rnd, nsteps, nmid, condt=False, maxt=100):
    """A random 3-qubit circuit with mid-circuit measurements (some native, some
    that also reset), resets, and conditional Paulis on the measured bits,
    followed by a measurement of every qubit into the last 3 bits. With
    ``condt``, some conditional bodies are T or T† gates."""
    n = 3
    gates1 = [
        mc.GateH(),
        mc.GateS(),
        mc.GateSDG(),
        mc.GateT(),
        mc.GateTDG(),
        mc.GateX(),
        mc.GateY(),
        mc.GateSX(),
    ]
    gates2 = [mc.GateCX(), mc.GateCZ(), mc.GateCY(), mc.GateSWAP(), mc.GateISWAP()]
    measures1 = [
        mc.Measure(),
        mc.MeasureX(),
        mc.MeasureY(),
        mc.MeasureReset(),
        mc.MeasureResetX(),
        mc.MeasureResetY(),
    ]
    measures2 = [mc.MeasureXX(), mc.MeasureYY(), mc.MeasureZZ(), mc.MeasurePauli("XY")]
    c = mc.Circuit()
    b = 0
    nt = 0
    for _ in range(nsteps):
        r = rnd.random()
        if r < 0.3:
            g = rnd.choice(gates1)
            if isinstance(g, (mc.GateT, mc.GateTDG)):
                if nt >= maxt:
                    continue
                nt += 1
            c.push(g, rnd.randrange(n))
        elif r < 0.45:
            c.push(rnd.choice(gates2), *rnd.sample(range(n), 2))
        elif r < 0.6 and b < nmid:
            if rnd.random() < 0.7:
                c.push(rnd.choice(measures1), rnd.randrange(n), b)
            else:
                c.push(rnd.choice(measures2), *rnd.sample(range(n), 2), b)
            b += 1
        elif r < 0.68:
            c.push(rnd.choice([mc.Reset(), mc.ResetX(), mc.ResetY()]), rnd.randrange(n))
        elif b > 0:
            k = rnd.randint(1, min(b, 3))
            cbits = rnd.sample(range(b), k)
            pattern = mc.BitString("".join(rnd.choice("01") for _ in range(k)))
            if condt and rnd.random() < 0.25:
                body, w = rnd.choice([mc.GateT(), mc.GateTDG()]), 1
            else:
                body, w = _random_pauli_body(rnd)
            c.push(mc.IfStatement(body, pattern), *rnd.sample(range(n), w), *cbits)
    c.push(mc.Measure(), range(n), range(b, b + n))
    return c


class TestPauliBasedComputationConditionals:
    def _xzy(self):
        body = mc.Circuit()
        body.push(mc.GateX(), 0)
        body.push(mc.GateZ(), 1)
        body.push(mc.GateY(), 2)
        return body

    @pytest.mark.parametrize("kind", ["block", "gatecall"])
    def test_pauli_product_stays_whole(self, kind):
        # A conditional Pauli product stays one conditional Pauli string,
        # however its body is written.
        body = self._xzy()
        op = mc.Block(body) if kind == "block" else mc.GateDecl("xzy", (), body)()
        c = mc.Circuit()
        c.push(mc.Measure(), 0, 0)
        c.push(mc.GateH(), 1)
        c.push(mc.IfStatement(op, mc.BitString("1")), 0, 1, 2, 0)
        p = mc.pauli_based_computation(c)
        ifs = [i for i in p if isinstance(i.operation, mc.IfStatement)]
        assert len(ifs) == 1
        assert ifs[0].operation == mc.IfStatement(
            mc.PauliString("XXY"), mc.BitString("1")
        )
        assert ifs[0].qubits == (0, 1, 2)
        assert ifs[0].bits == (0,)

    def test_identity_product(self):
        # A product equal to the identity leaves nothing behind.
        c = mc.Circuit()
        c.push(mc.Measure(), 0, 0)
        c.push(
            mc.IfStatement(mc.Repeat(2, mc.PauliString("XZ")), mc.BitString("1")),
            0,
            1,
            0,
        )
        assert mc.pauli_based_computation(c) == mc.Circuit().push(
            mc.MeasurePauli("Z"), 0, 0
        )

    def test_resets(self):
        # A reset measures into a new bit after the circuit's bits, then flips
        # the qubit back when the outcome is 1. MeasureReset uses the given bit.
        one = mc.BitString("1")
        p = mc.pauli_based_computation(mc.Circuit().push(mc.ResetX(), 0))
        assert [i.operation for i in p] == [
            mc.MeasurePauli("X"),
            mc.IfStatement(mc.PauliString("Z"), one),
        ]
        assert [i.bits for i in p] == [(0,), (0,)]
        c = mc.Circuit()
        c.push(mc.Measure(), 0, 0)
        c.push(mc.Reset(), 0)
        c.push(mc.MeasureResetY(), 0, 1)
        p = mc.pauli_based_computation(c)
        assert [i.operation for i in p] == [
            mc.MeasurePauli("Z"),
            mc.MeasurePauli("Z"),
            mc.IfStatement(mc.PauliString("X"), one),
            mc.MeasurePauli("Y"),
            mc.IfStatement(mc.PauliString("X"), one),
        ]
        assert [i.bits for i in p] == [(0,), (2,), (2,), (1,), (1,)]

    def test_magic_condition(self):
        # With magic states, a condition on several bits is reduced to the bit
        # after the frame, and bits compared with 0 are restored afterwards.
        c = mc.Circuit()
        c.push(mc.Measure(), range(2), range(2))
        c.push(mc.IfStatement(mc.GateX(), mc.BitString("01")), 0, 0, 1)
        p = mc.pauli_based_computation(c, magic_states=True)
        ops = [i.operation for i in p][2:]
        assert ops == [mc.SetBit1(), mc.Not(), mc.And(), mc.Not(), mc.And(), mc.Xor()]
        bits = [i.bits for i in p][2:]
        assert bits == [(6,), (0,), (6, 6, 0), (0,), (6, 6, 1), (2, 2, 6)]

    def test_conditional_t_and_refusals(self):
        # A conditional T becomes a conditional rotation; a conditional
        # Clifford and a conditional magic-state injection are refused.
        c = mc.Circuit()
        c.push(mc.Measure(), 0, 0)
        c.push(mc.GateH(), 1)
        c.push(mc.IfStatement(mc.GateTDG(), mc.BitString("0")), 1, 0)
        p = mc.pauli_based_computation(c)
        assert p.instructions[1].operation == mc.IfStatement(
            mc.RPauli(mc.PauliString("X"), -se.pi / 4), mc.BitString("0")
        )
        with pytest.raises(
            ValueError,
            match="Clifford accumulated after it would depend on the condition bits",
        ):
            mc.pauli_based_computation(
                mc.Circuit().push(mc.IfStatement(mc.GateH(), mc.BitString("1")), 0, 0)
            )
        with pytest.raises(ValueError):
            mc.pauli_based_computation(
                mc.Circuit().push(
                    mc.IfStatement(mc.GateCX(), mc.BitString("1")), 0, 1, 0
                )
            )
        with pytest.raises(
            ValueError, match="magic state cannot be injected conditionally"
        ):
            mc.pauli_based_computation(c, magic_states=True)
        with pytest.raises(ValueError):
            mc.pauli_based_computation(
                mc.Circuit().push(
                    mc.IfStatement(mc.Measure(), mc.BitString("1")), 0, 0, 1
                )
            )

    @pytest.mark.parametrize("seed", range(30))
    def test_random_plain(self, seed):
        # Random circuits with conditionals and resets reproduce the statistics
        # of the original bits.
        rnd = random.Random(seed)
        rng = np.random.default_rng(seed)
        c = _random_conditional_circuit(rnd, 24, 4, condt=seed % 2 == 1)
        nb = c.num_bits()
        psi = _randstate(rng, 3)
        ref = _distribution(mc.decompose(c, basis=_ReferenceBasis()), 3, nb, psi)

        p = mc.pauli_based_computation(c)
        allowed = (mc.RPauli, mc.MeasurePauli, mc.Not, mc.IfStatement)
        assert all(isinstance(i.operation, allowed) for i in p)
        assert _same_distribution(ref, _distribution(p, 3, p.num_bits(), psi, keep=nb))
        assert fromproto_circuit(toproto_circuit(p)) == p

        # The pass reads measurements and resets directly; expanding them into
        # Clifford gates first gives the same circuit.
        expanded = mc.decompose(c, basis=_ExpandMeasurements())
        assert mc.pauli_based_computation(expanded) == p

    @pytest.mark.parametrize("seed", range(30))
    def test_random_magic(self, seed):
        rnd = random.Random(100 + seed)
        rng = np.random.default_rng(100 + seed)
        c = _random_conditional_circuit(rnd, 18, 3, maxt=2)
        if not any(isinstance(i.operation, (mc.GateT, mc.GateTDG)) for i in c):
            c.push(mc.GateT(), 0)
            c.push(mc.Measure(), 0, c.num_bits())
        nb = c.num_bits()
        psi = _randstate(rng, 3)
        ref = _distribution(mc.decompose(c, basis=_ReferenceBasis()), 3, nb, psi)

        p = mc.pauli_based_computation(c, magic_states=True)
        assert not any(
            isinstance(i.operation, (mc.RPauli, mc.GateT, mc.GateTDG)) for i in p
        )
        psianc = np.kron(psi, [1.0, 0.0])
        assert _same_distribution(
            ref, _distribution(p, 4, p.num_bits(), psianc, keep=nb)
        )
        expanded = mc.decompose(c, basis=_ExpandMeasurements())
        assert mc.pauli_based_computation(expanded, magic_states=True) == p
        assert fromproto_circuit(toproto_circuit(p)) == p
