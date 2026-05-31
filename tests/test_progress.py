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
"""Locks the progress wiring in ``LocalBackend.execute``: the sampling
path draws compression and execution bars; the trajectory path draws a
trajectory bar and mutes the per-shot execution detail; the default
``NoProgress`` sink leaves the user callback untouched."""

import mimiqcircuits as mc
from mimiqcircuits.backends import (
    LocalBackend,
    DefaultCompiledCircuit,
    Progress,
    Stage,
)


class _RecStage(Stage):
    def __init__(self, rec, name):
        self._rec = rec
        self._name = name

    def step(self, n=1):
        self._rec.steps[self._name] += n

    def finish(self):
        self._rec.finished.append(self._name)


class _RecProgress(Progress):
    """Records every stage the driver opens, with step counts."""

    def __init__(self):
        self.opened = []
        self.steps = {}
        self.finished = []

    def stage(self, name, *, total=None):
        self.opened.append((name, total))
        self.steps[name] = 0
        return _RecStage(self, name)

    def names(self):
        return [n for n, _ in self.opened]


class _MockState:
    def __init__(self, nq, nb, nz):
        self._nq, self._nb, self._nz = nq, nb, nz
        self._cstate = mc.BitString(nb)

    @property
    def num_qubits(self):
        return self._nq

    @property
    def num_bits(self):
        return self._nb

    @property
    def num_zvars(self):
        return self._nz

    def sample(self, nsamples, rng=None, *, seed=None):
        return [mc.BitString(self._nq) for _ in range(nsamples)]

    def amplitude(self, bs):
        return 1.0 + 0j

    @property
    def classical_bits(self):
        return self._cstate

    @property
    def complex_values(self):
        return []


class _PgBackend(LocalBackend):
    """Reports ``nsteps`` per-step callbacks per evolution and emits a
    compression stage of ``nsteps`` increments."""

    def __init__(self, nsteps=4):
        self.nsteps = nsteps

    @property
    def name(self):
        return "Pg"

    @property
    def version(self):
        return "0.0.1"

    def capabilities(self):
        return {"sampling", "amplitude", "midcircuit_measure",
                "midcircuit_reset", "feed_forward", "classical_bits"}

    def build_state(self, nq, nb=0, nz=0, **kwargs):
        return _MockState(nq, nb, nz)

    def compile(self, circuit):
        return DefaultCompiledCircuit(circuit)

    def compile_progress(self, circuit, progress):
        st = progress.stage("compression", total=self.nsteps)
        for _ in range(self.nsteps):
            st.step()
        st.finish()
        return self.compile(circuit)

    def prepare_trajectory(self, compiled, rng):
        return compiled

    def recompile_per_trajectory(self, circuit):
        return False

    def evolve(self, state, compiled, *, rng=None, callback=None, stopped=None):
        for i in range(1, self.nsteps + 1):
            if callback is not None:
                callback(i, self.nsteps)
        return state, 1.0


def _bare():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    return c


def _midcircuit():
    # Reset+CX between measures: genuinely non-absorbable, forces the
    # trajectory branch.
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Reset(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.Measure(), 1, 1)
    return c


def test_sampling_path_draws_compression_and_execution_bars():
    rec = _RecProgress()
    _PgBackend(4).execute(_bare(), nsamples=10, progress=rec)
    assert ("compression", 4) in rec.opened
    assert ("execute", 4) in rec.opened       # total taken from the callback
    assert rec.steps["execute"] == 4
    assert "trajectories" not in rec.names()
    assert "execute" in rec.finished


def test_trajectory_path_draws_trajectory_bar_and_mutes_execution():
    rec = _RecProgress()
    _PgBackend(4).execute(_midcircuit(), nsamples=6, progress=rec)
    assert ("trajectories", 6) in rec.opened
    assert rec.steps["trajectories"] == 6
    assert "compression" in rec.names()       # compiled once before the loop
    assert "execute" not in rec.names()       # per-shot detail suppressed
    assert "trajectories" in rec.finished


def test_default_progress_off_leaves_callback_unwrapped():
    calls = []
    _PgBackend(3).execute(
        _bare(), nsamples=2, callback=lambda i, total: calls.append(i)
    )
    assert calls == [1, 2, 3]                  # one per reported step


def test_progress_true_runs_end_to_end():
    res = _PgBackend(2).execute(_bare(), nsamples=5, progress=True)
    assert len(res.cstates) == 5
