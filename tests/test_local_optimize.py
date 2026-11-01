"""LocalBackend.optimize on a toy one-qubit backend whose cost is cos(θ)."""

import math

import pytest
from symengine import symbols

import mimiqcircuits as mc
from mimiqcircuits.backends import LocalBackend, DefaultCompiledCircuit

theta = symbols("theta")


class _OptState:
    """One qubit after `RX(θ)`, with ⟨Z⟩ = cos(θ) written to z[0]."""

    def __init__(self, nq, nb, nz):
        self.theta = 0.0
        self._nq, self._nb, self._nz = nq, nb, nz

    @property
    def num_qubits(self): return self._nq
    @property
    def num_bits(self): return self._nb
    @property
    def num_zvars(self): return self._nz
    @property
    def classical_bits(self): return mc.BitString(self._nb)
    @property
    def complex_values(self): return [complex(math.cos(self.theta))] * self._nz

    def sample(self, nsamples, rng=None, *, seed=None):
        return [mc.BitString(self._nq) for _ in range(nsamples)]

    def amplitude(self, bs):
        return 1.0 + 0j


class _OptBackend(LocalBackend):
    @property
    def name(self): return "Opt"
    @property
    def version(self): return "0.0.1"
    def capabilities(self): return {"sampling", "classical_bits"}

    def build_state(self, nq, nb=0, nz=0, **kwargs):
        return _OptState(nq, nb, nz)

    def compile(self, circuit):
        return DefaultCompiledCircuit(circuit)

    def evolve(self, state, compiled, *, rng=None, callback=None, stopped=None):
        for inst in compiled.source:
            if isinstance(inst.operation, mc.GateRX):
                state.theta += float(inst.operation.theta)
        return state, 1.0


@pytest.fixture
def circuit():
    # The z-register only needs to exist; the backend writes the cost into it.
    return mc.Circuit().push(mc.GateRX(theta), 0).push(mc.Amplitude(mc.BitString(1)), 0)


@pytest.mark.parametrize(
    "optimizer", ["BFGS", "LBFGS", "CG", "NELDERMEAD", "NEWTON", "COBYLA"]
)
def test_optimizers_reach_minimum(circuit, optimizer):
    res = _OptBackend().optimize(
        circuit, {theta: 2.0}, optimizer=optimizer, nsamples=1, seed=1
    )
    assert isinstance(res, mc.OptimizationResults)
    assert res.get_best().cost == pytest.approx(-1, abs=1e-4)
    assert res.history == []


def test_cmaes(circuit):
    pytest.importorskip("cma")
    res = _OptBackend().optimize(
        circuit, {theta: 2.0}, optimizer="CMAES", maxiters=200, nsamples=1, seed=1
    )
    assert res.get_best().cost == pytest.approx(-1, abs=1e-3)


def test_history_and_best(circuit):
    res = _OptBackend().optimize(
        circuit, {theta: 2.0}, maxiters=20, return_history=True, nsamples=1, seed=1
    )
    assert 0 < len(res.history) <= 25
    assert res.get_best().cost == min(r.cost for r in res.history)
    assert set(res.get_best().parameters) == {theta}


def test_experiment(circuit):
    ex = mc.OptimizationExperiment(circuit, {theta: 2.0}, optimizer="NELDERMEAD")
    res = _OptBackend().optimize(ex, nsamples=1, seed=1)
    assert res.get_best().cost == pytest.approx(-1, abs=1e-4)

    with pytest.raises(TypeError):
        _OptBackend().optimize(ex, {theta: 1.0})


def test_string_keys(circuit):
    res = _OptBackend().optimize(circuit, {"theta": 2.0}, optimizer="NELDERMEAD",
                                 nsamples=1, seed=1)
    assert res.get_best().cost == pytest.approx(-1, abs=1e-4)


def test_execute_keywords_are_forwarded(circuit):
    backend = _OptBackend()
    res = backend.optimize(circuit, {theta: 2.0}, maxiters=5, nsamples=3, seed=1,
                           fuse=False, remove_swaps=False, canonicaldecompose=False)
    assert len(res.get_best().results.cstates) == 3
    with pytest.raises(TypeError):
        backend.optimize(circuit, {theta: 2.0}, method="BFGS")


def test_rejects_bad_arguments(circuit):
    with pytest.raises(ValueError, match="Unknown optimizer"):
        _OptBackend().optimize(circuit, {theta: 2.0}, optimizer="ADAM")
    with pytest.raises(ValueError, match="zregister"):
        _OptBackend().optimize(circuit, {theta: 2.0}, zregister=3)
    ex = mc.OptimizationExperiment(circuit, {theta: 2.0})
    with pytest.raises(TypeError, match="optimizer is read from"):
        _OptBackend().optimize(ex, optimizer="BFGS")


def test_list_of_experiments(circuit):
    exs = [mc.OptimizationExperiment(circuit, {theta: v}, optimizer="NELDERMEAD")
           for v in (2.0, 4.0)]
    res = _OptBackend().optimize(exs, nsamples=1, seed=1)
    assert isinstance(res, list) and len(res) == 2
    assert all(r.get_best().cost == pytest.approx(-1, abs=1e-4) for r in res)
