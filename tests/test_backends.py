#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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

"""Tests for the mimiqcircuits.backends ABCs and ADTs: capabilities,
fidelity, compiled circuits, passes, Backend / LocalBackend /
RemoteBackend. Mirrors the corresponding AbstractQCSs.jl test surface.
"""

from __future__ import annotations

import math
import random

import pytest

from mimiqcircuits.backends import (
    # capabilities
    Capability,
    CAPABILITY_VOCABULARY,
    Limits,
    Topology,
    AllToAll,
    CouplingMap,
    LinearChain,
    AdmissionResult,
    Admissible,
    Marginal,
    Inadmissible,
    is_admissible,
    UnsupportedCapabilityError,
    # fidelity
    Fidelity,
    ExactFidelity,
    UnknownFidelity,
    TruncationLowerBound,
    LowerBoundPerStep,
    EstimatedFidelity,
    as_lower_bound,
    as_expected,
    as_interval,
    # compiled
    CompiledCircuit,
    DefaultCompiledCircuit,
    CompiledParametricCircuit,
    CompileMetadata,
    UnboundSymbolicError,
    # passes
    PassParam,
    PStr,
    PInt,
    PFloat,
    PBool,
    PSym,
    PList,
    PDict,
    to_pass_param,
    PassSpec,
    PassResult,
    PassContext,
    AbstractPass,
    PassPipeline,
    apply_passes,
    invert_perm,
    UnacceptedPassError,
    RemotePassOrderError,
    # backend
    Backend,
    LocalBackend,
    RemoteBackend,
    RNGs,
    # rng helpers
    normalize_seed,
    derive_grid_seeds,
)


# ──────────────────────────────────────────────────────────────────────────
# Capabilities
# ──────────────────────────────────────────────────────────────────────────


def test_capability_vocabulary_has_core_entries():
    assert "amplitude" in CAPABILITY_VOCABULARY
    assert "sampling" in CAPABILITY_VOCABULARY
    assert "feed_forward" in CAPABILITY_VOCABULARY
    assert "loss" in CAPABILITY_VOCABULARY
    assert "shared_prefix_batch" in CAPABILITY_VOCABULARY
    assert len(CAPABILITY_VOCABULARY) >= 20


def test_limits_defaults_are_none():
    lim = Limits()
    assert lim.max_qubits is None
    assert lim.max_bond_dim is None


def test_topology_defaults():
    assert isinstance(AllToAll(), Topology)
    cm = CouplingMap(edges=((1, 2), (2, 3)))
    assert len(cm.edges) == 2
    lc = LinearChain(n=8)
    assert lc.n == 8


def test_admission_predicates():
    assert is_admissible(Admissible())
    assert is_admissible(Marginal(warning="close to limit"))
    assert not is_admissible(Inadmissible(reason="too big"))


def test_unsupported_capability_error_message():
    err = UnsupportedCapabilityError("Foo", "feed_forward", "no IfStatement")
    msg = str(err)
    assert "Foo" in msg
    assert "feed_forward" in msg
    assert "no IfStatement" in msg


# ──────────────────────────────────────────────────────────────────────────
# Default can_handle admission gates
# ──────────────────────────────────────────────────────────────────────────


class _AdmissionBackend(Backend):
    """Minimal concrete Backend with no capabilities declared, so the
    default can_handle gates can be exercised against undeclared caps."""
    def __init__(self, caps=None):
        self._caps = caps or set()

    @property
    def name(self): return "Admission"

    @property
    def version(self): return "0.0.1"

    def capabilities(self): return self._caps

    def execute(self, circuit, *, nsamples=1000, **kw):
        return None


def test_can_handle_rejects_noise_when_not_declared():
    import mimiqcircuits as mc
    b = _AdmissionBackend()
    c = mc.Circuit()
    c.push(mc.Depolarizing1(0.1), 0)
    res = b.can_handle(c)
    assert isinstance(res, Inadmissible)
    assert "noise" in res.reason


def test_can_handle_accepts_noise_when_declared():
    import mimiqcircuits as mc
    b = _AdmissionBackend(caps={"noise"})
    c = mc.Circuit()
    c.push(mc.Depolarizing1(0.1), 0)
    assert isinstance(b.can_handle(c), Admissible)


def test_can_handle_rejects_parametric_when_not_declared():
    import mimiqcircuits as mc
    from symengine import symbols
    b = _AdmissionBackend()
    theta = symbols("theta")
    c = mc.Circuit()
    c.push(mc.GateRX(theta), 0)
    res = b.can_handle(c)
    assert isinstance(res, Inadmissible)
    assert "parametric" in res.reason


def test_can_handle_accepts_parametric_when_declared():
    import mimiqcircuits as mc
    from symengine import symbols
    b = _AdmissionBackend(caps={"parametric"})
    theta = symbols("theta")
    c = mc.Circuit()
    c.push(mc.GateRX(theta), 0)
    assert isinstance(b.can_handle(c), Admissible)


def test_can_handle_finds_kraus_nested_in_ifstatement():
    # A noise channel nested inside an IfStatement body must still
    # trip the :noise gate (matches Julia's _op_has_kraus recursion).
    import mimiqcircuits as mc
    b = _AdmissionBackend()
    inner = mc.Depolarizing1(0.1)
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.IfStatement(inner, mc.BitString("1")), 0, 0)
    res = b.can_handle(c)
    assert isinstance(res, Inadmissible)
    assert "noise" in res.reason


# ──────────────────────────────────────────────────────────────────────────
# Fidelity
# ──────────────────────────────────────────────────────────────────────────


def test_exact_fidelity_reductions():
    f = ExactFidelity()
    assert as_lower_bound(f) == 1.0
    assert as_expected(f) == 1.0
    assert as_interval(f) == (1.0, 1.0)


def test_unknown_fidelity_is_nan():
    f = UnknownFidelity()
    assert math.isnan(as_lower_bound(f))
    assert math.isnan(as_expected(f))
    lo, hi = as_interval(f)
    assert math.isnan(lo) and math.isnan(hi)


def test_truncation_lower_bound():
    f = TruncationLowerBound(value=0.97)
    assert as_lower_bound(f) == 0.97
    assert as_expected(f) == 0.97


def test_lower_bound_per_step_product():
    f = LowerBoundPerStep(values=(0.99, 0.98, 0.97))
    assert as_lower_bound(f) == pytest.approx(0.99 * 0.98 * 0.97)
    # clamping
    f2 = LowerBoundPerStep(values=(0.5, 1.2, -0.1))
    assert as_lower_bound(f2) == 0.0


def test_estimated_fidelity_interval():
    f = EstimatedFidelity(mean=0.90, stderr=0.02)
    assert as_expected(f) == 0.90
    assert as_lower_bound(f) == pytest.approx(0.90 - 3 * 0.02)
    assert as_interval(f) == (0.88, 0.92)
    # clipped at 0
    f2 = EstimatedFidelity(mean=0.01, stderr=0.10)
    assert as_lower_bound(f2) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# CompiledCircuit
# ──────────────────────────────────────────────────────────────────────────


def test_default_compiled_circuit_metadata():
    cc = DefaultCompiledCircuit(_source="dummy")
    assert cc.source == "dummy"
    assert isinstance(cc.metadata, CompileMetadata)
    assert cc.metadata.qubit_permutation is None


def test_compiled_parametric_circuit():
    cc = CompiledParametricCircuit(_source="symbolic")
    assert isinstance(cc, CompiledCircuit)
    assert cc.source == "symbolic"


def test_unbound_symbolic_error_message():
    err = UnboundSymbolicError("FooBackend")
    msg = str(err)
    assert "FooBackend" in msg
    assert "parametric" in msg


# ──────────────────────────────────────────────────────────────────────────
# Passes
# ──────────────────────────────────────────────────────────────────────────


def test_passparam_equality_is_tag_sensitive():
    assert PStr("x") == PStr("x")
    assert PSym("x") != PStr("x")
    assert PInt(1) != PBool(True)
    assert PFloat(1.0) != PInt(1)


def test_to_pass_param_coerces_python_natives():
    assert isinstance(to_pass_param("hi"), PStr)
    # bool BEFORE int (otherwise True coerces to PInt; this is a real footgun)
    assert isinstance(to_pass_param(True), PBool)
    assert isinstance(to_pass_param(3), PInt)
    assert isinstance(to_pass_param(1.5), PFloat)
    assert isinstance(to_pass_param([1, 2]), PList)
    assert isinstance(to_pass_param({"k": 1}), PDict)


def test_passspec_from_dict_roundtrips():
    s = PassSpec.from_dict("reorder", {"method": "greedy", "alpha": 2.0})
    # Tuple-of-pairs storage for hashability
    params = dict(s.parameters)
    assert params["method"] == PStr("greedy")
    assert params["alpha"] == PFloat(2.0)


# ── A trivial pass and backend for pipeline tests ─────────────────────────


class _RecordPass(AbstractPass):
    def __init__(self, name: str, perm=None):
        self.name = name
        self.perm = perm

    def spec(self) -> PassSpec:
        return PassSpec(self.name)

    def apply(self, ctx, c):
        return c, PassResult(qubit_permutation=self.perm, metadata={"ran": True})


class _FakeBackend(Backend):
    def __init__(self, name="Fake", caps=None, rejects=None, delegates=None):
        self._name = name
        self._caps = caps or set()
        self._rejects = rejects or set()
        self._delegates = delegates or set()

    @property
    def name(self): return self._name

    @property
    def version(self): return "0.0.1"

    def capabilities(self): return self._caps

    def accepts_pass(self, p):
        return p.spec().name not in self._rejects

    def delegates_pass(self, p):
        return p.spec().name in self._delegates

    def execute(self, circuit, *, nsamples=1000, **kw):
        return None  # not exercised in these tests


def test_apply_passes_runs_all_and_records_results():
    b = _FakeBackend()
    pipe = PassPipeline(passes=[_RecordPass("a"), _RecordPass("b")])
    out, perm, results = apply_passes(pipe, PassContext(backend=b), "circ")
    assert perm is None
    assert len(results) == 2
    assert all(r.metadata.get("ran") for r in results)


def test_apply_passes_composes_permutations():
    b = _FakeBackend()
    pipe = PassPipeline(passes=[
        _RecordPass("p1", perm=[3, 1, 2]),
        _RecordPass("p2", perm=[2, 3, 1]),
    ])
    _, composed, _ = apply_passes(pipe, PassContext(backend=b), "circ")
    # composed[i] = new[prev[i]]; prev=[3,1,2], new=[2,3,1]
    # → [new[3], new[1], new[2]] = [1, 2, 3]
    assert composed == [1, 2, 3]


def test_invert_perm_round_trip():
    assert invert_perm([2, 3, 1]) == [3, 1, 2]
    assert invert_perm(invert_perm([4, 1, 3, 2])) == [4, 1, 3, 2]


def test_unaccepted_pass_raises():
    b = _FakeBackend(rejects={"reject_me"})
    pipe = PassPipeline(passes=[_RecordPass("reject_me")])
    with pytest.raises(UnacceptedPassError):
        apply_passes(pipe, PassContext(backend=b), "circ")


def test_delegates_pass_skips_apply_with_marker():
    b = _FakeBackend(delegates={"native"})
    pipe = PassPipeline(passes=[_RecordPass("native")])
    _, _, results = apply_passes(pipe, PassContext(backend=b), "circ")
    assert len(results) == 1
    assert results[0].metadata.get("delegated") is True
    assert "ran" not in results[0].metadata


def test_remote_pass_order_error_when_strict_and_unhonored():
    class _RemoteFake(RemoteBackend):
        @property
        def name(self): return "FakeRemote"
        @property
        def version(self): return "0"
        def capabilities(self): return set()
        def submit(self, *a, **kw): raise AssertionError("should not reach")

    r = _RemoteFake()
    pipe = PassPipeline(passes=[_RecordPass("anything")])
    with pytest.raises(RemotePassOrderError):
        r.execute("circuit", passes=pipe)


def test_remote_pass_order_skips_check_when_pipeline_empty():
    class _RemoteFake(RemoteBackend):
        @property
        def name(self): return "FakeRemote"
        @property
        def version(self): return "0"
        def capabilities(self): return set()

        def submit(self, *a, **kw):
            class _Job:
                def wait(self): return "OK"
            return _Job()

    r = _RemoteFake()
    # No passes → no strict-order check → execute proceeds and returns the job's wait()
    out = r.execute("circuit")
    assert out == "OK"


# ──────────────────────────────────────────────────────────────────────────
# RNGs
# ──────────────────────────────────────────────────────────────────────────


def test_rngs_from_seed_deterministic():
    r1 = RNGs.from_seed(42)
    r2 = RNGs.from_seed(42)
    assert r1.shot.random() == r2.shot.random()
    assert r1.noise.random() == r2.noise.random()
    assert r1.trajectory.random() == r2.trajectory.random()
    assert r1.pass_.random() == r2.pass_.random()


def test_rngs_streams_are_independent():
    r = RNGs.from_seed(7)
    # different streams have different first draws (probabilistic, but
    # masks differ deterministically so this holds)
    a = r.shot.random()
    b = r.noise.random()
    assert a != b


# ──────────────────────────────────────────────────────────────────────────
# normalize_seed / derive_grid_seeds
# ──────────────────────────────────────────────────────────────────────────


def test_normalize_seed_both_raises():
    with pytest.raises(TypeError):
        normalize_seed(42, RNGs.from_seed(7))


def test_normalize_seed_int_rngs_passthrough():
    assert normalize_seed(None, 123) == 123


def test_normalize_seed_seed_passthrough():
    assert normalize_seed(7, None) == 7
    assert normalize_seed(None, None) is None


def test_normalize_seed_rngs_bundle_xor_fold():
    seed = normalize_seed(None, RNGs.from_seed(42))
    assert isinstance(seed, int)
    assert 0 <= seed < (1 << 63)


def test_derive_grid_seeds_distinct():
    seeds = derive_grid_seeds(42, 4)
    assert len(set(seeds)) == 4
    # Same input twice → same output (determinism).
    assert seeds == derive_grid_seeds(42, 4)


def test_derive_grid_seeds_none_base():
    assert derive_grid_seeds(None, 3) == [None, None, None]


def test_derive_grid_seeds_zero_base_is_well_mixed():
    """SplitMix64 regression guard: the previous LCG produced
    base-seed-independent outputs at base=0. The new mixer must give
    different seeds across base in {0, 1, 2}."""
    a = derive_grid_seeds(0, 4)
    b = derive_grid_seeds(1, 4)
    c = derive_grid_seeds(2, 4)
    assert a != b != c != a
    # All outputs in range.
    for seq in (a, b, c):
        for s in seq:
            assert 0 <= s < (1 << 63)


# ──────────────────────────────────────────────────────────────────────────
# LocalBackend.execute driver — routing, projection, loss sampling
# ──────────────────────────────────────────────────────────────────────────


class _MockState:
    """Stub State that holds a fixed sample bitstring and records
    `classical_bits` / `complex_values` reads. Just enough surface
    for the LocalBackend.execute driver tests."""

    def __init__(self, nq, nb, nz, fixed_bits=None):
        import mimiqcircuits as mc
        self._nq = nq
        self._nb = nb
        self._nz = nz
        self._fixed = fixed_bits if fixed_bits is not None else [False] * nq
        # In-state classical and z registers — these flip when an
        # in-circuit Measure / Amplitude actually fires (trajectory
        # mode). The mock backend doesn't actually evolve, so they
        # stay at their default.
        self._cstate = mc.BitString(nb) if nb > 0 else mc.BitString(0)
        self._zstate = [complex(0)] * nz

    @property
    def num_qubits(self): return self._nq
    @property
    def num_bits(self): return self._nb
    @property
    def num_zvars(self): return self._nz

    def sample(self, nsamples, rng=None, *, seed=None):
        import mimiqcircuits as mc
        return [mc.BitString(self._fixed) for _ in range(nsamples)]

    def amplitude(self, bs):
        return 1.0 + 0j

    @property
    def classical_bits(self):
        return self._cstate

    @property
    def complex_values(self):
        return list(self._zstate)


class _MockBackend(LocalBackend):
    """Records every hook call. `fixed_bits` is what the state will
    return from `sample()` — lets tests inspect what the projection
    circuit does to the raw samples."""

    def __init__(self, fixed_bits=None):
        self._fixed = fixed_bits
        self.compile_calls = 0
        self.evolve_calls = 0
        self.build_state_calls = 0
        self.prepare_calls = 0

    @property
    def name(self): return "Mock"
    @property
    def version(self): return "0.0.1"
    def capabilities(self):
        return {"sampling", "amplitude", "midcircuit_measure",
                "midcircuit_reset", "feed_forward"}

    def build_state(self, nq, nb=0, nz=0, **kwargs):
        self.build_state_calls += 1
        bits = self._fixed if self._fixed is not None else [False] * nq
        return _MockState(nq, nb, nz, bits)

    def compile(self, circuit):
        self.compile_calls += 1
        return DefaultCompiledCircuit(circuit)

    def prepare_trajectory(self, compiled, rng):
        self.prepare_calls += 1
        return compiled

    def evolve(self, state, compiled, *, rng=None, callback=None, stopped=None):
        self.evolve_calls += 1
        return state, 1.0


def test_localbackend_execute_routes_reset_between_measures_to_sampling():
    """The OP's regression: Reset between two Measures must route
    to single-evolve + N-sample, not N independent trajectories."""
    import mimiqcircuits as mc

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateX(), 1)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Reset(), 0)
    c.push(mc.Measure(), 1, 1)

    backend = _MockBackend(fixed_bits=[True, True])  # sample = bs"11"
    res = backend.execute(c, nsamples=200)

    # Single evolve, 200 samples, all projecting to bs"11" (since the
    # mock state always returns the same bits).
    assert backend.evolve_calls == 1
    assert backend.compile_calls == 1
    assert backend.build_state_calls == 1
    assert len(res.cstates) == 200
    assert len(res.fidelities) == 1
    assert all(list(bs) == [True, True] for bs in res.cstates)


def test_localbackend_execute_routes_reset_cx_to_trajectories():
    """Reset+CX between Measures is genuinely non-absorbable — the
    driver must take the trajectory branch."""
    import mimiqcircuits as mc

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Reset(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.Measure(), 1, 1)

    backend = _MockBackend(fixed_bits=[False, False])
    res = backend.execute(c, nsamples=5)

    # Trajectory mode: one evolve per shot, one fidelity per shot.
    assert backend.evolve_calls == 5
    assert backend.build_state_calls == 5
    assert len(res.fidelities) == 5


def test_localbackend_execute_implicit_measure_all_when_no_bits():
    """A circuit with no classical register synthesises an implicit
    identity bit↔qubit mapping. Each shot's cstate must be the
    sampled qubit values."""
    import mimiqcircuits as mc

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateX(), 1)

    backend = _MockBackend(fixed_bits=[True, True])
    res = backend.execute(c, nsamples=4)

    assert len(res.cstates) == 4
    # All bits come straight from the fixed sample.
    for bs in res.cstates:
        assert list(bs) == [True, True]


def test_localbackend_amplitude_op_writes_zstate():
    """Amplitude observation flows through an in-circuit
    :class:`Amplitude` op into ``results.zstates``. The mock state's
    ``complex_values`` accessor returns whatever its ``_zstate``
    holds; the driver only forwards it. Assert the wiring: the
    z-register is sized by the circuit and the per-shot zstate
    snapshot lands in ``results.zstates``.
    """
    import mimiqcircuits as mc

    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.Amplitude(mc.BitString("00")), 0)
    c.push(mc.Amplitude(mc.BitString("11")), 1)

    backend = _MockBackend(fixed_bits=[True, True])
    res = backend.execute(c, nsamples=10)

    # Sampling-mode: one zstate snapshot of length nz==2.
    assert len(res.zstates) == 1
    assert len(res.zstates[0]) == 2


def test_localbackend_execute_batch_dispatch():
    """A list of circuits returns a list of results, one per input."""
    import mimiqcircuits as mc

    c1 = mc.Circuit()
    c1.push(mc.GateH(), 0)
    c2 = mc.Circuit()
    c2.push(mc.GateX(), 0)

    backend = _MockBackend(fixed_bits=[False])
    out = backend.execute([c1, c2], nsamples=3)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(len(r.cstates) == 3 for r in out)


def test_localbackend_recompile_per_trajectory_default():
    """Default predicate fires on mixed-unitary Kraus."""
    import mimiqcircuits as mc

    backend = _MockBackend()

    mixed = mc.Circuit()
    mixed.push(mc.GateH(), 0)
    mixed.push(mc.Depolarizing1(0.1), 0)   # mixed-unitary Kraus
    assert backend.recompile_per_trajectory(mixed) is True

    plain = mc.Circuit()
    plain.push(mc.GateH(), 0)
    plain.push(mc.GateCX(), 0, 1)
    assert backend.recompile_per_trajectory(plain) is False
