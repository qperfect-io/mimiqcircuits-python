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

"""`CompiledProjection` against `evaluate_projection`, the reference.

Every case here asserts bit-for-bit equality with the single-shot
evaluator, so the two can never drift. The tier each case lands on is
asserted as well: a case that silently fell back to the general tier
would still pass the equality check while losing the point of the
exercise.
"""

import random

import pytest

import mimiqcircuits as mc
from bitarray import bitarray, frozenbitarray
from mimiqcircuits.backends.compiled_projection import CompiledProjection, Tier
from mimiqcircuits.backends.measure_analysis import (
    evaluate_projection,
    extract_projection,
)


def samples(width, nshots, seed=1234):
    rng = random.Random(seed)
    return [
        mc.BitString([rng.randrange(2) for _ in range(width)])
        for _ in range(nshots)
    ]


def assert_matches_reference(projection, block, tier=None):
    """Compare the compiled projection with the reference evaluator over
    `block`, and check it picked the tier we expect."""
    compiled = CompiledProjection.from_circuit(projection)
    got = compiled.evaluate_batch(block)
    expected = [evaluate_projection(projection, s) for s in block]
    assert [b.to01() for b in got] == [b.to01() for b in expected]
    if tier is not None:
        assert compiled._plans[len(block[0].bits)].tier == tier
    return got


# ── the tiers ───────────────────────────────────────────────────────────────


def test_identity_mapping_takes_the_identity_tier():
    proj = mc.Circuit()
    for q in range(8):
        proj.push(mc.Measure(), q, q)
    block = samples(8, 32)
    got = assert_matches_reference(proj, block, Tier.IDENTITY)
    # The identity tier hands the samples straight back.
    assert all(a is b for a, b in zip(got, block))


def test_shuffled_clbit_mapping():
    order = [3, 0, 7, 1, 6, 2, 5, 4]
    proj = mc.Circuit()
    for b, q in enumerate(order):
        proj.push(mc.Measure(), q, b)
    assert_matches_reference(proj, samples(8, 32), Tier.PROGRAM)


def test_identity_mapping_on_a_wider_state_is_not_the_identity_tier():
    """A state wider than the register still has to drop the extra
    qubits, so the samples are not the answer as they stand."""
    proj = mc.Circuit()
    for q in range(4):
        proj.push(mc.Measure(), q, q)
    assert_matches_reference(proj, samples(8, 32), Tier.PROGRAM)


def test_cstate_narrower_than_the_register():
    proj = mc.Circuit()
    for b, q in enumerate((1, 3)):
        proj.push(mc.Measure(), q, b)
    got = assert_matches_reference(proj, samples(6, 16), Tier.PROGRAM)
    assert all(len(b) == 2 for b in got)


def test_measure_past_the_end_of_the_sample_reads_zero():
    proj = mc.Circuit()
    proj.push(mc.Measure(), 0, 0)
    proj.push(mc.Measure(), 9, 1)
    got = assert_matches_reference(proj, samples(4, 16), Tier.PROGRAM)
    assert all(b[1] == 0 for b in got)


def test_empty_projection():
    compiled = CompiledProjection.from_circuit(mc.Circuit())
    assert compiled.evaluate_batch([]) == []
    got = compiled.evaluate_batch(samples(4, 3))
    assert [b.to01() for b in got] == ["", "", ""]


# ── the classical language ──────────────────────────────────────────────────


def test_reset_folded_constants():
    proj = mc.Circuit()
    proj.push(mc.Measure(), 0, 0)
    proj.push(mc.SetBit0(), 1)
    proj.push(mc.SetBit1(), 2)
    proj.push(mc.Measure(), 3, 3)
    got = assert_matches_reference(proj, samples(4, 16), Tier.PROGRAM)
    assert all(b[1] == 0 and b[2] == 1 for b in got)


def test_not_tail():
    proj = mc.Circuit()
    for q in range(4):
        proj.push(mc.Measure(), q, q)
    proj.push(mc.Not(), 0)
    assert_matches_reference(proj, samples(4, 16), Tier.PROGRAM)


@pytest.mark.parametrize("op", [mc.And(3), mc.Or(3), mc.Xor(3), mc.ParityCheck(3)])
def test_logic_tails(op):
    proj = mc.Circuit()
    for q in range(4):
        proj.push(mc.Measure(), q, q)
    proj.push(op, 3, 1, 2)
    assert_matches_reference(proj, samples(4, 32), Tier.PROGRAM)


def test_combined_tail():
    proj = mc.Circuit()
    for q in range(6):
        proj.push(mc.Measure(), q, q)
    proj.push(mc.Not(), 0)
    proj.push(mc.And(3), 1, 0, 2)
    proj.push(mc.Or(3), 3, 1, 4)
    proj.push(mc.Xor(3), 4, 3, 5)
    proj.push(mc.ParityCheck(4), 5, 0, 1, 2)
    proj.push(mc.Not(), 5)
    assert_matches_reference(proj, samples(6, 64), Tier.PROGRAM)


# ParityCheck is absent: it is the one logic op that does not declare
# `allow_bit_aliasing`, so a circuit cannot name its target among its
# operands in the first place.
@pytest.mark.parametrize("op", [mc.And(3), mc.Or(3), mc.Xor(3)])
def test_target_that_is_also_an_operand_reads_its_old_value(op):
    proj = mc.Circuit()
    for q in range(3):
        proj.push(mc.Measure(), q, q)
    proj.push(op, 0, 0, 1)
    assert_matches_reference(proj, samples(3, 32), Tier.PROGRAM)


def test_unsupported_operation_is_rejected():
    proj = mc.Circuit()
    proj.push(mc.GateX(), 0)
    with pytest.raises(ValueError, match="unsupported instruction"):
        CompiledProjection.from_circuit(proj)


# ── plumbing ────────────────────────────────────────────────────────────────


def test_blocks_wider_than_one_chunk_match_the_reference():
    """The evaluator splits a long block into chunks, so the seam
    between two chunks has to be invisible."""
    proj = mc.Circuit()
    for b, q in enumerate((2, 0, 3, 1)):
        proj.push(mc.Measure(), q, b)
    compiled = CompiledProjection.from_circuit(proj)
    block = samples(4, 5000)
    got = compiled.evaluate_batch(block)
    expected = [evaluate_projection(proj, s) for s in block]
    assert [b.to01() for b in got] == [b.to01() for b in expected]


def test_a_little_endian_sample_reads_the_same():
    """`BitString` keeps whatever bit order it is handed, and the two lay
    the same bits out in opposite directions within a byte. A sample built
    little-endian must still project to the same result."""
    proj = mc.Circuit()
    for b, q in enumerate((2, 0, 1)):
        proj.push(mc.Measure(), q, b)

    big = samples(3, 16)
    little = [
        mc.BitString(frozenbitarray(bitarray(s.to01(), endian="little")))
        for s in big
    ]
    assert [s.bits.endian for s in little] == ["little"] * len(little)
    assert [s.to01() for s in little] == [s.to01() for s in big]

    compiled = CompiledProjection.from_circuit(proj)
    from_big = [b.to01() for b in compiled.evaluate_batch(big)]
    from_little = [b.to01() for b in compiled.evaluate_batch(little)]
    expected = [evaluate_projection(proj, s).to01() for s in big]
    assert from_little == from_big == expected


def test_single_shot_evaluate_matches_the_reference():
    proj = mc.Circuit()
    for b, q in enumerate((1, 0)):
        proj.push(mc.Measure(), q, b)
    proj.push(mc.Not(), 1)
    compiled = CompiledProjection.from_circuit(proj)
    for s in samples(2, 8):
        assert compiled.evaluate(s).to01() == evaluate_projection(proj, s).to01()


def test_a_compiled_projection_is_reusable_across_widths():
    """The tier depends on the sample width, so a projection reused
    against a different width has to re-plan rather than reuse a plan
    that no longer holds."""
    proj = mc.Circuit()
    proj.push(mc.Measure(), 0, 0)
    proj.push(mc.Measure(), 1, 1)
    compiled = CompiledProjection.from_circuit(proj)

    narrow = samples(2, 8, seed=1)
    wide = samples(5, 8, seed=2)
    for block in (narrow, wide, narrow):
        got = compiled.evaluate_batch(block)
        expected = [evaluate_projection(proj, s) for s in block]
        assert [b.to01() for b in got] == [b.to01() for b in expected]

    assert compiled._plans[2].tier == Tier.IDENTITY
    assert compiled._plans[5].tier == Tier.PROGRAM


def test_extracted_projection_of_a_measured_circuit():
    """The shape `extract_projection` actually produces, end to end."""
    circuit = mc.Circuit()
    circuit.push(mc.GateH(), 0)
    for q in range(3):
        circuit.push(mc.GateCX(), q, q + 1)
    circuit.push(mc.Measure(), range(4), range(4))
    _, proj = extract_projection(circuit)
    assert_matches_reference(proj, samples(4, 32), Tier.IDENTITY)
