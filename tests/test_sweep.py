#
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

import math
import random

import numpy as np
import pytest
from symengine import symbols

import mimiqcircuits as mc
from mimiqcircuits import ParameterTable, SweepExperiment, gridtable
from mimiqcircuits.backends import DefaultCompiledCircuit, LocalBackend

theta, phi, psi = symbols("theta phi psi")


# ── ParameterTable ──────────────────────────────────────────────────────────


def test_constructors_agree():
    m = [[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]]
    tables = [
        ParameterTable([theta, phi], m),
        ParameterTable(["theta", "phi"], m),
        ParameterTable.from_columns({theta: [0.0, 0.5, 1.0], phi: [1.0, 2.0, 3.0]}),
        ParameterTable.from_dicts(
            [{theta: 0.0, phi: 1.0}, {"theta": 0.5, "phi": 2.0}, {phi: 3.0, theta: 1.0}]
        ),
    ]
    for t in tables:
        assert len(t) == 3
        assert t.listvars() == [phi, theta] or t.listvars() == [theta, phi]
        cols = [t.listvars().index(s) for s in (theta, phi)]
        assert t.to_numpy()[:, cols].tolist() == m


def test_points():
    t = ParameterTable.from_columns({theta: [0.0, 0.5], phi: [1, 2]})
    assert t[1] == {theta: 0.5, phi: 2.0}
    assert list(t) == [t[0], t[1]]
    assert mc.Circuit().push(mc.GateRX(theta), 0).evaluate(t[1])[0].operation == mc.GateRX(0.5)


def test_values_are_read_only_and_exact():
    t = ParameterTable.from_dicts([{theta: mc.symbolics.se.pi}, {theta: 0.5}])
    assert t.to_numpy()[:, 0].tolist() == [math.pi, 0.5]
    with pytest.raises(ValueError):
        t._values[0, 0] = 1.0
    copy = t.to_numpy()
    copy[0, 0] = 1.0
    assert t[0][theta] == math.pi


@pytest.mark.parametrize(
    "build",
    [
        lambda: ParameterTable([theta, phi], np.zeros((3, 1))),
        lambda: ParameterTable.from_columns({theta: [0, 1], phi: [0]}),
        lambda: ParameterTable([theta, theta], np.zeros((1, 2))),
        lambda: ParameterTable([2 * theta], np.zeros((1, 1))),
        lambda: ParameterTable.from_dicts([{theta: 0}, {phi: 0}]),
        lambda: ParameterTable.from_dicts([{theta: 0}, {theta: 0, phi: 1}]),
        lambda: ParameterTable.from_dicts([]),
        lambda: ParameterTable.from_columns({}),
        lambda: gridtable({}),
    ],
)
def test_invalid_tables(build):
    with pytest.raises(ValueError):
        build()


def test_product():
    t = ParameterTable.from_columns({theta: [0.0, 1.0]}) * ParameterTable.from_columns(
        {phi: [0.0, 2.0, 4.0]}
    )
    assert t.listvars() == [theta, phi]
    # The first table varies slowest.
    assert t.to_numpy().tolist() == [
        [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [1.0, 0.0], [1.0, 2.0], [1.0, 4.0]
    ]
    with pytest.raises(ValueError):
        t * ParameterTable.from_columns({theta: [0.0]})


def test_gridtable_is_the_product_of_its_columns():
    g = gridtable({theta: [0.0, 1.0], phi: [0.0, 2.0, 4.0], psi: [5.0, 6.0]})
    p = (
        ParameterTable.from_columns({theta: [0.0, 1.0]})
        * ParameterTable.from_columns({phi: [0.0, 2.0, 4.0]})
        * ParameterTable.from_columns({psi: [5.0, 6.0]})
    )
    assert len(g) == 12
    assert g.listvars() == p.listvars()
    assert g.to_numpy().tolist() == p.to_numpy().tolist()


def test_hcat_pairs_points():
    t = ParameterTable.from_columns({theta: [0.0, 1.0]}).hcat(
        ParameterTable.from_columns({phi: [5.0, 6.0]}),
        ParameterTable.from_columns({psi: [7.0, 8.0]}),
    )
    assert t.listvars() == [theta, phi, psi]
    assert t.to_numpy().tolist() == [[0.0, 5.0, 7.0], [1.0, 6.0, 8.0]]
    with pytest.raises(ValueError):
        ParameterTable.from_columns({theta: [0.0]}).hcat(
            ParameterTable.from_columns({phi: [0.0, 1.0]})
        )
    with pytest.raises(ValueError):
        ParameterTable.from_columns({theta: [0.0]}).hcat(
            ParameterTable.from_columns({theta: [1.0]})
        )


def test_vcat_appends_points():
    a = ParameterTable.from_columns({theta: [0.0], phi: [1.0]})
    b = ParameterTable.from_columns({phi: [3.0], theta: [2.0]})  # other order
    t = a.vcat(b, a)
    assert t.listvars() == [theta, phi]
    assert t.to_numpy().tolist() == [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        a.vcat(ParameterTable.from_columns({theta: [0.0]}))


def test_display():
    t = ParameterTable.from_columns({theta: [0.0, 0.5]})
    assert t._short() == "ParameterTable(2 points over (theta))"
    long = repr(ParameterTable.from_columns({theta: [float(i) for i in range(20)]}))
    assert "⋮" in long
    assert long.count("\n") == 11


# ── SweepExperiment ─────────────────────────────────────────────────────────


def _rx_ry():
    return mc.Circuit().push(mc.GateRX(theta), 0).push(mc.GateRY(phi), 0)


def test_points_are_evaluated_circuits():
    c = _rx_ry()
    ex = SweepExperiment(c, gridtable({theta: [0.0, math.pi], phi: [0.0, math.pi / 2]}))
    assert len(ex) == 4
    assert ex[1] == c.evaluate({theta: 0.0, phi: math.pi / 2})
    assert not ex[3].is_symbolic()
    assert list(ex) == [ex[i] for i in range(4)]
    assert ex.num_qubits() == 1


def test_table_must_bind_exactly_the_circuit_symbols():
    c = _rx_ry()
    with pytest.raises(ValueError, match="no values for phi"):
        SweepExperiment(c, ParameterTable.from_columns({theta: [0.0]}))
    with pytest.raises(ValueError, match="psi"):
        SweepExperiment(
            c, ParameterTable.from_columns({theta: [0.0], phi: [0.0], psi: [0.0]})
        )
    with pytest.raises(ValueError, match="symbolic"):
        SweepExperiment(mc.Circuit().push(mc.GateH(), 0),
                        ParameterTable.from_columns({theta: [0.0]}))
    with pytest.raises(TypeError):
        SweepExperiment(c, [{theta: 0.0, phi: 0.0}])


# ── execute ─────────────────────────────────────────────────────────────────


class _RXState:
    """One qubit after `RX(theta)`: samples 1 with probability sin²(θ/2)."""

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
    def complex_values(self): return [0j] * self._nz

    def sample(self, nsamples, rng=None, *, seed=None):
        rng = rng if rng is not None else random.Random(seed)
        p = math.sin(self.theta / 2) ** 2
        return [mc.BitString([rng.random() < p]) for _ in range(nsamples)]

    def amplitude(self, bs):
        return 1.0 + 0j


class _RXBackend(LocalBackend):
    @property
    def name(self): return "RX"
    @property
    def version(self): return "0.0.1"
    def capabilities(self): return {"sampling", "classical_bits"}

    def build_state(self, nq, nb=0, nz=0, **kwargs):
        return _RXState(nq, nb, nz)

    def compile(self, circuit):
        return DefaultCompiledCircuit(circuit)

    def evolve(self, state, compiled, *, rng=None, callback=None, stopped=None):
        for inst in compiled.source:
            if isinstance(inst.operation, mc.GateRX):
                state.theta += float(inst.operation.theta)
        return state, 1.0


def _ones(res):
    return sum(bool(bs[0]) for bs in res.cstates)


@pytest.fixture
def sweep():
    c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.Measure(), 0, 0)
    # Points 2 and 3 bind the same angle, so only their seeds differ.
    return SweepExperiment(
        c, ParameterTable.from_columns({theta: [0.0, math.pi, math.pi / 2, math.pi / 2]})
    )


def test_execute_returns_one_result_per_point(sweep):
    res = _RXBackend().execute(sweep, nsamples=200, seed=1)
    assert isinstance(res, list) and len(res) == 4
    assert all(len(r.cstates) == 200 for r in res)
    assert _ones(res[0]) == 0
    assert _ones(res[1]) == 200


def test_execute_is_reproducible_and_points_are_independent(sweep):
    b = _RXBackend()
    a = b.execute(sweep, nsamples=200, seed=7)
    again = b.execute(sweep, nsamples=200, seed=7)
    assert [r.cstates for r in a] == [r.cstates for r in again]
    assert a[2].cstates != a[3].cstates


def test_execute_matches_param_grid(sweep):
    b = _RXBackend()
    swept = b.execute(sweep, nsamples=100, seed=3)
    grid = b.execute(sweep.circuit, nsamples=100, seed=3, param_grid=list(sweep.table))
    assert [r.cstates for r in swept] == [r.cstates for r in grid]


def test_execute_refuses_a_sweep_with_param_grid(sweep):
    with pytest.raises(TypeError):
        _RXBackend().execute(sweep, param_grid=[{theta: 0.0}])


def test_execute_empty_table_gives_no_results():
    c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.Measure(), 0, 0)
    ex = SweepExperiment(c, ParameterTable([theta], np.zeros((0, 1))))
    assert _RXBackend().execute(ex, nsamples=10) == []


def test_execute_with_the_same_rng_advances_it(sweep):
    b = _RXBackend()
    rng = random.Random(5)
    first = b.execute(sweep, nsamples=200, rng=rng)
    second = b.execute(sweep, nsamples=200, rng=rng)
    assert [r.cstates for r in first] != [r.cstates for r in second]


def test_slicing():
    t = gridtable({theta: [0.0, 1.0, 2.0], phi: [0.5]})
    assert t[0:2] == ParameterTable([theta, phi], [[0.0, 0.5], [1.0, 0.5]])
    assert t[::-1].listvars() == [theta, phi]
    assert len(t[5:]) == 0

    c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.GateRY(phi), 0)
    ex = SweepExperiment(c, t)
    assert ex[1:] == SweepExperiment(c, t[1:])


def test_index_list():
    t = gridtable({theta: [0.0, 1.0, 2.0], phi: [0.5]})
    assert t[[0, 2]] == ParameterTable([theta, phi], [[0.0, 0.5], [2.0, 0.5]])
    assert t[np.array([2])] == t[2:3]
    assert t[np.int64(1)] == {theta: 1.0, phi: 0.5}

    c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.GateRY(phi), 0)
    ex = SweepExperiment(c, t)
    assert ex[[0, 2]] == SweepExperiment(c, t[[0, 2]])


def test_equality():
    a = ParameterTable.from_columns({theta: [0.0, 1.0]})
    b = ParameterTable(["theta"], [[0.0], [1.0]])
    assert a == b and hash(a) == hash(b)
    assert a != ParameterTable.from_columns({theta: [0.0, 2.0]})
    assert a != ParameterTable.from_columns({phi: [0.0, 1.0]})

    c = mc.Circuit().push(mc.GateRX(theta), 0)
    assert SweepExperiment(c, a) == SweepExperiment(c, b)
    assert SweepExperiment(c, a) != SweepExperiment(c, a[0:1])
    with pytest.raises(TypeError):
        hash(SweepExperiment(c, a))


def test_from_dicts_refuses_a_symbol_given_twice():
    with pytest.raises(ValueError):
        ParameterTable.from_dicts([{theta: 0.0, "theta": 1.0}])
