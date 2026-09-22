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

"""Absorption of the trailing classical block by `extract_projection`.

The property that matters is that a circuit ending in measurements plus
classical logic stays on the sampling path: one evolution and a
projection, rather than one evolution per shot. `needs_trajectories` on
the returned `quantum_circuit` is what decides that, so every case
asserts it.
"""

import pytest

import mimiqcircuits as mc
from mimiqcircuits.backends.measure_analysis import (
    evaluate_projection,
    extract_projection,
    needs_trajectories,
)


def ghz(nq, tail=None):
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    for q in range(nq - 1):
        c.push(mc.GateCX(), q, q + 1)
    c.push(mc.Measure(), range(nq), range(nq))
    if tail is not None:
        tail(c)
    return c


def ops(circuit):
    return [
        (type(i.operation).__name__, tuple(i.qubits), tuple(i.bits))
        for i in circuit.instructions
    ]


# ── the classical language is absorbed ──────────────────────────────────────


@pytest.mark.parametrize("tail", [
    lambda c: c.push(mc.Not(), 0),
    lambda c: c.push(mc.SetBit0(), 0),
    lambda c: c.push(mc.SetBit1(), 0),
    lambda c: c.push(mc.And(3), 3, 0, 1),
    lambda c: c.push(mc.Or(3), 3, 0, 1),
    lambda c: c.push(mc.Xor(3), 3, 0, 1),
    lambda c: c.push(mc.ParityCheck(3), 3, 0, 1),
])
def test_a_classical_tail_stays_on_the_sampling_path(tail):
    quantum, _ = extract_projection(ghz(4, tail))
    assert not needs_trajectories(quantum)
    assert all(op.num_bits == 0 for op in
               (i.operation for i in quantum.instructions))


def test_a_classical_tail_is_evaluated_after_the_measurements():
    quantum, projection = extract_projection(ghz(4, lambda c: c.push(mc.Xor(3), 3, 0, 1)))
    assert not needs_trajectories(quantum)
    # Bit 3's measurement is dead: the Xor overwrites it without reading.
    assert ops(projection) == [
        ("Measure", (0,), (0,)),
        ("Measure", (1,), (1,)),
        ("Measure", (2,), (2,)),
        ("Xor", (), (3, 0, 1)),
    ]
    sample = mc.BitString("1101")
    assert evaluate_projection(projection, sample).to01() == "1100"


def test_a_long_classical_tail_keeps_its_source_order():
    def tail(c):
        c.push(mc.Not(), 0)
        c.push(mc.And(3), 4, 0, 1)
        c.push(mc.Or(3), 5, 4, 2)
        c.push(mc.Not(), 5)

    quantum, projection = extract_projection(ghz(6, tail))
    assert not needs_trajectories(quantum)
    assert ops(projection)[-4:] == [
        ("Not", (), (0,)),
        ("And", (), (4, 0, 1)),
        ("Or", (), (5, 4, 2)),
        ("Not", (), (5,)),
    ]


def test_a_classical_op_between_measurements_keeps_its_place():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 1)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Xor(3), 2, 0, 1)
    c.push(mc.Measure(), 1, 1)
    quantum, projection = extract_projection(c)
    assert not needs_trajectories(quantum)
    assert ops(projection) == [
        ("Measure", (0,), (0,)),
        ("Xor", (), (2, 0, 1)),
        ("Measure", (1,), (1,)),
    ]


# ── what blocks absorption ──────────────────────────────────────────────────


def test_a_classical_op_whose_bit_a_kept_operation_writes_is_not_absorbed():
    """A kept `MeasureZZ` writes bit 0 during evolution, so a `Not` on
    bit 0 cannot move past it into the projection."""
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Not(), 0)
    c.push(mc.MeasureZZ(), 0, 1, 0)
    quantum, projection = extract_projection(c)
    assert ("Not", (), (0,)) in ops(quantum)
    assert ("Not", (), (0,)) not in ops(projection)


def test_a_wrapper_that_writes_a_bit_blocks_absorption():
    """`IfStatement` wraps a gate that writes nothing while reading a bit
    itself, so the wrapper's own footprint is what counts. Reading the
    inner operator's instead would let the projection absorb the `Not` and
    the `Measure` that feed the condition."""
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Not(), 0)
    c.push(mc.IfStatement(mc.GateX(), mc.BitString("1")), 1, 0)
    quantum, projection = extract_projection(c)
    assert ("Not", (), (0,)) in ops(quantum)
    assert ops(projection) == []


# ── liveness ────────────────────────────────────────────────────────────────


def test_an_overwritten_measurement_is_dropped():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 1)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Measure(), 1, 0)
    quantum, projection = extract_projection(c)
    assert not needs_trajectories(quantum)
    assert ops(projection) == [("Measure", (1,), (0,))]


def test_a_write_that_a_later_operation_reads_is_kept():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Not(), 0)
    c.push(mc.SetBit1(), 0)
    _, projection = extract_projection(c)
    # `Not` reads bit 0, so the measurement survives; `SetBit1` then
    # overwrites both without reading, so only it is left.
    assert ops(projection) == [("SetBit1", (), (0,))]


def test_nothing_is_dropped_from_an_ordinary_projection():
    _, projection = extract_projection(ghz(5))
    assert ops(projection) == [("Measure", (q,), (q,)) for q in range(5)]


# ── the shapes that were already supported ──────────────────────────────────


def test_a_plain_measured_circuit_is_unchanged():
    quantum, projection = extract_projection(ghz(4))
    assert not needs_trajectories(quantum)
    assert len(quantum.instructions) == 4
    assert ops(projection) == [("Measure", (q,), (q,)) for q in range(4)]


def test_a_circuit_with_no_classical_register_gets_the_identity_mapping():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    _, projection = extract_projection(c)
    assert ops(projection) == [("Measure", (0,), (0,)), ("Measure", (1,), (1,))]


def test_a_reset_folds_its_captured_bits_into_constants():
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(mc.Reset(), 0)
    c.push(mc.Measure(), 0, 0)
    _, projection = extract_projection(c)
    assert ops(projection) == [("SetBit0", (), (0,))]


def test_mid_circuit_measurement_still_needs_trajectories():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.Measure(), 1, 1)
    quantum, _ = extract_projection(c)
    assert needs_trajectories(quantum)
