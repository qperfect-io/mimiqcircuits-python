# Runtime loss (a LossyOperator-bearing Kraus branch) is resolved live by the
# shared evolve_with_loss driver, which LocalBackend.execute routes to for any
# backend that opts in via uses_loss_driver. These tests drive that full path
# against a minimal single-qubit basis-state mock backend.

import random

import numpy as np

import mimiqcircuits as mc
from mimiqcircuits.backends.backend import LocalBackend, State, LossState
from mimiqcircuits.backends.compiled import DefaultCompiledCircuit
from mimiqcircuits.operations.operators.lossyoperator import LossyOperator


def _to_complex_matrix(op):
    m = op.unwrappedmatrix()
    return np.array(
        [[complex(m[r, c]) for c in range(m.cols)] for r in range(m.rows)]
    )


class _LossMockState(State):
    """A 1-qubit-per-wire basis-state stub: each qubit is |0> or |1>, enough
    to exercise the driver's loss bookkeeping end to end."""

    def __init__(self, nq, nb, nz):
        self._nq, self._nb, self._nz = nq, nb, nz
        self.qubits = [False] * nq
        self._cbits = [False] * nb
        self._loss = LossState(nq)

    @property
    def num_qubits(self):
        return self._nq

    @property
    def num_bits(self):
        return self._nb

    @property
    def num_zvars(self):
        return self._nz

    def amplitude(self, bs):
        return 1.0 + 0j

    def sample(self, nsamples, rng=None, *, seed=None):
        return [mc.BitString(self.qubits) for _ in range(nsamples)]

    @property
    def classical_bits(self):
        return mc.BitString(self._cbits)

    @property
    def complex_values(self):
        return []

    def get_loss_state(self):
        return self._loss


class _LossMockBackend(LocalBackend):
    @property
    def name(self):
        return "LossMock"

    @property
    def version(self):
        return "0.0.1"

    def capabilities(self):
        return {"sampling", "loss", "midcircuit_measure", "midcircuit_reset",
                "classical_bits"}

    def uses_loss_driver(self):
        return True

    def build_state(self, nq, nb=0, nz=0, **kwargs):
        return _LossMockState(nq, nb, nz)

    def compile(self, circuit):
        return DefaultCompiledCircuit(circuit)

    def evolve(self, state, compiled, *, rng=None, callback=None, stopped=None):
        raise AssertionError("runtime-loss circuits must use evolve_with_loss")

    def apply_instruction(self, state, inst, *, rng=None):
        op = inst.get_operation()
        qs = inst.get_qubits()
        bs = inst.get_bits()
        if isinstance(op, mc.GateX):
            state.qubits[qs[0]] = not state.qubits[qs[0]]
        elif isinstance(op, mc.Reset):
            state.qubits[qs[0]] = False
        elif isinstance(op, mc.Measure):
            state._cbits[bs[0]] = state.qubits[qs[0]]
        elif isinstance(op, mc.SetBit0):
            state._cbits[bs[0]] = False
        elif isinstance(op, mc.SetBit1):
            state._cbits[bs[0]] = True
        return 1.0

    def sample_kraus(self, state, channel, targets, *, rng=None):
        # 1-qubit basis-state sampler: branch probability is the squared norm
        # of the operator's column for the qubit's current basis value.
        col = 1 if state.qubits[targets[0]] else 0
        ops = channel.krausoperators()
        probs = []
        for o in ops:
            m = _to_complex_matrix(o)
            probs.append(float(np.sum(np.abs(m[:, col]) ** 2)))
        r = (rng or random.Random()).random()
        cum = 0.0
        chosen = ops[-1]
        for o, p in zip(ops, probs):
            cum += p
            if r <= cum:
                chosen = o
                break
        return 1.0, chosen


_CERTAIN_LOSS = mc.Kraus([
    mc.Operator(np.zeros((2, 2))),
    LossyOperator(np.eye(2)),
])
_SURVIVES = mc.Kraus([
    mc.Operator(np.eye(2)),
    LossyOperator(np.zeros((2, 2))),
])


def test_runtime_loss_routes_to_driver():
    backend = _LossMockBackend()
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(_CERTAIN_LOSS, 0)
    assert backend.uses_loss_driver()
    # Should not raise (evolve would assert); routing must pick the driver.
    backend.execute(c, nsamples=4, seed=1)


def test_lost_qubit_reads_zero_and_check_reflects_loss():
    backend = _LossMockBackend()
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(_CERTAIN_LOSS, 0)     # qubit 0 leaks for sure
    c.push(mc.GateX(), 0)        # dropped: target lost
    c.push(mc.Check(), 0, 0)     # present? -> lost -> c[0] = 0
    c.push(mc.Measure(), 0, 1)   # lost measurement -> c[1] = 0

    res = backend.execute(c, nsamples=8, seed=1)
    assert all(s[0] == 0 and s[1] == 0 for s in res.cstates)


def test_surviving_qubit_keeps_state():
    backend = _LossMockBackend()
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(_SURVIVES, 0)
    c.push(mc.Check(), 0, 0)     # present -> c[0] = 1
    c.push(mc.Measure(), 0, 1)   # |1> -> c[1] = 1

    res = backend.execute(c, nsamples=8, seed=2)
    assert all(s[0] == 1 and s[1] == 1 for s in res.cstates)


def test_reload_restores_a_lost_qubit():
    backend = _LossMockBackend()
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(_CERTAIN_LOSS, 0)     # lost
    c.push(mc.Reload(), 0)       # reset to |0>, present again
    c.push(mc.Check(), 0, 0)     # present -> c[0] = 1
    c.push(mc.Measure(), 0, 1)   # |0> -> c[1] = 0

    res = backend.execute(c, nsamples=8, seed=3)
    assert all(s[0] == 1 and s[1] == 0 for s in res.cstates)


def test_probabilistic_loss_splits_population():
    backend = _LossMockBackend()
    half = mc.Kraus([
        mc.Operator(np.array([[1.0, 0.0], [0.0, np.sqrt(0.5)]])),
        LossyOperator(np.array([[0.0, np.sqrt(0.5)], [0.0, 0.0]])),
    ])
    c = mc.Circuit()
    c.push(mc.GateX(), 0)              # |1>
    c.push(half, 0)
    c.push(mc.MeasureCheck(), 0, 0, 1)  # value bit c[0], presence bit c[1]

    res = backend.execute(c, nsamples=2000, seed=4)
    # presence 0 => lost, value forced 0; presence 1 => survived as |1>.
    assert all(
        (s[1] == 0 and s[0] == 0) or (s[1] == 1 and s[0] == 1)
        for s in res.cstates
    )
    lost_frac = sum(1 for s in res.cstates if s[1] == 0) / len(res.cstates)
    assert 0.4 < lost_frac < 0.6


def test_apply_segment_batches_present_runs():
    # A compressing backend overrides apply_segment to receive each maximal
    # all-present run as one batch (so it can fuse it). The driver must hand it
    # the run between two loss events, not split it into single instructions.
    segments = []

    class _SegBackend(_LossMockBackend):
        def apply_segment(self, state, insts, *, rng=None):
            segments.append([type(i.get_operation()).__name__ for i in insts])
            return super().apply_segment(state, insts, rng=rng)

    backend = _SegBackend()
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(mc.GateX(), 0)        # two present gates -> one batch
    c.push(_SURVIVES, 0)         # event (Kraus); qubit survives
    c.push(mc.GateX(), 0)        # trailing present gate ...
    c.push(mc.Measure(), 0, 0)   # ... plus a present measure -> one batch
    backend.execute(c, nsamples=1, seed=1)

    assert segments == [["GateX", "GateX"], ["GateX", "Measure"]]


def test_lossmodel_rewrite_matches_apply_helper():
    # The extracted decision core and the in-place helper must agree.
    from mimiqcircuits.lossmodel import lossmodel_rewrite, _apply_lossmodel_rules

    model = mc.LossModel()
    inst = mc.Instruction(mc.GateCX(), (0, 1), (), ())
    lost = {0: True}

    out = mc.Circuit()
    _apply_lossmodel_rules(out, inst, model, lost, random.Random(0))
    rewritten = lossmodel_rewrite(inst, lost, model, random.Random(0))
    assert [i.get_operation() for i in out] == [i.get_operation() for i in rewritten]
