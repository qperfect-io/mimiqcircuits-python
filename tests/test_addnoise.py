"""Behaviour of add_noise and the decorate_on_match_* family.

The expected outputs pin the order in which decorations are placed, for every
combination of before/after, single/parallel and match-by-instance/type.
"""

import pytest

import mimiqcircuits as mc
from mimiqcircuits.addnoise import (
    add_noise,
    add_noise_inplace,
    decorate,
    decorate_inplace,
    decorate_on_match_parallel,
    decorate_on_match_parallel_inplace,
    decorate_on_match_single,
    decorate_on_match_single_inplace,
)

D = mc.Depolarizing(1, 0.1)


def strs(c):
    return [str(inst) for inst in c]


def base_circuit():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateH(), 1)
    c.push(mc.GateH(), 2)
    c.push(mc.Measure(), 0, 0)
    c.push(mc.GateRX(0.2), 1)
    c.push(mc.GateH(), 0)
    return c


@pytest.mark.parametrize(
    "before,parallel,expected",
    [
        (
            False,
            False,
            [
                "H @ q[0]",
                "Depolarizing(0.1) @ q[0]",
                "CX @ q[0], q[1]",
                "H @ q[1]",
                "Depolarizing(0.1) @ q[1]",
                "H @ q[2]",
                "Depolarizing(0.1) @ q[2]",
                "M @ q[0], c[0]",
                "RX(0.2) @ q[1]",
                "H @ q[0]",
                "Depolarizing(0.1) @ q[0]",
            ],
        ),
        (
            True,
            False,
            [
                "Depolarizing(0.1) @ q[0]",
                "H @ q[0]",
                "CX @ q[0], q[1]",
                "Depolarizing(0.1) @ q[1]",
                "H @ q[1]",
                "Depolarizing(0.1) @ q[2]",
                "H @ q[2]",
                "M @ q[0], c[0]",
                "RX(0.2) @ q[1]",
                "Depolarizing(0.1) @ q[0]",
                "H @ q[0]",
            ],
        ),
        (
            False,
            True,
            [
                "H @ q[0]",
                "Depolarizing(0.1) @ q[0]",
                "CX @ q[0], q[1]",
                "H @ q[1]",
                "H @ q[2]",
                "Depolarizing(0.1) @ q[1]",
                "Depolarizing(0.1) @ q[2]",
                "M @ q[0], c[0]",
                "RX(0.2) @ q[1]",
                "H @ q[0]",
                "Depolarizing(0.1) @ q[0]",
            ],
        ),
        (
            True,
            True,
            [
                "Depolarizing(0.1) @ q[0]",
                "H @ q[0]",
                "CX @ q[0], q[1]",
                "Depolarizing(0.1) @ q[1]",
                "Depolarizing(0.1) @ q[2]",
                "H @ q[1]",
                "H @ q[2]",
                "M @ q[0], c[0]",
                "RX(0.2) @ q[1]",
                "Depolarizing(0.1) @ q[0]",
                "H @ q[0]",
            ],
        ),
    ],
)
@pytest.mark.parametrize("g", [mc.GateH(), mc.GateH])
def test_add_noise_placement(g, before, parallel, expected):
    c = base_circuit()
    original = strs(c)

    out = add_noise(c, g, D, before=before, parallel=parallel)
    assert strs(out) == expected
    assert strs(c) == original  # the non-mutating form leaves its input alone

    r = add_noise_inplace(c, g, D, before=before, parallel=parallel)
    assert r is c
    assert strs(c) == expected


def test_add_noise_matches_instance_parameters():
    c = mc.Circuit()
    c.push(mc.GateRX(0.1), 0)
    c.push(mc.GateRX(0.2), 0)
    assert strs(add_noise(c, mc.GateRX(0.2), mc.GateX())) == [
        "RX(0.1) @ q[0]",
        "RX(0.2) @ q[0]",
        "X @ q[0]",
    ]
    assert strs(add_noise(c, mc.GateRX, mc.GateX(), before=True)) == [
        "X @ q[0]",
        "RX(0.1) @ q[0]",
        "X @ q[0]",
        "RX(0.2) @ q[0]",
    ]


def test_add_noise_two_qubit_parallel_blocks():
    # A block ends at the first CZ that shares a qubit with it.
    c = mc.Circuit()
    c.push(mc.GateCZ(), 1, 2)
    c.push(mc.GateCZ(), 1, 3)
    c.push(mc.GateCZ(), 4, 5)
    c.push(mc.GateCZ(), 2, 3)
    assert strs(add_noise(c, mc.GateCZ, mc.Depolarizing(2, 0.1), parallel=True)) == [
        "CZ @ q[1], q[2]",
        "Depolarizing(0.1) @ q[1:2]",
        "CZ @ q[1], q[3]",
        "CZ @ q[4], q[5]",
        "Depolarizing(0.1) @ q[1,3]",
        "Depolarizing(0.1) @ q[4:5]",
        "CZ @ q[2], q[3]",
        "Depolarizing(0.1) @ q[2:3]",
    ]


def test_add_noise_measurements_by_abstract_type():
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.MeasureReset(), 1, 1)
    c.push(mc.GateH(), 0)
    c.push(mc.Measure(), 2, 2)
    expected = [
        "PauliX(0.1) @ q[0]",
        "PauliX(0.1) @ q[1]",
        "M @ q[0], c[0]",
        "MR @ q[1], c[1]",
        "H @ q[0]",
        "PauliX(0.1) @ q[2]",
        "M @ q[2], c[2]",
    ]
    noise = mc.PauliX(0.1)
    out = add_noise(c, mc.AbstractMeasurement, noise, before=True, parallel=True)
    assert strs(out) == expected


def test_add_noise_wrong_arity_raises():
    c = mc.Circuit().push(mc.GateCX(), 0, 1)
    with pytest.raises(ValueError, match="Wrong number of target"):
        add_noise(c, mc.GateCX(), D)
    with pytest.raises(TypeError):
        add_noise(c, mc.GateCX(), mc.Measure())
    with pytest.raises(TypeError):
        add_noise(c, "CX", D)


def measure_circuit():
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Measure(), 1, 1)
    c.push(mc.Measure(), 2, 1)
    c.push(mc.GateX(), 0)
    return c


def is_measure(inst):
    return isinstance(inst.operation, mc.AbstractMeasurement)


@pytest.mark.parametrize(
    "parallel,before,expected",
    [
        (
            False,
            False,
            [
                "M @ q[0], c[0]",
                "Reset @ q[0]",
                "M @ q[1], c[1]",
                "Reset @ q[1]",
                "M @ q[2], c[1]",
                "Reset @ q[2]",
                "X @ q[0]",
            ],
        ),
        (
            True,
            False,
            [
                # the third measurement writes c[1] again, which ends the block
                "M @ q[0], c[0]",
                "M @ q[1], c[1]",
                "Reset @ q[0]",
                "Reset @ q[1]",
                "M @ q[2], c[1]",
                "Reset @ q[2]",
                "X @ q[0]",
            ],
        ),
        (
            True,
            True,
            [
                "Reset @ q[0]",
                "Reset @ q[1]",
                "M @ q[0], c[0]",
                "M @ q[1], c[1]",
                "Reset @ q[2]",
                "M @ q[2], c[1]",
                "X @ q[0]",
            ],
        ),
    ],
)
def test_decorate_measurements(parallel, before, expected):
    def gen(inst):
        return mc.Instruction(mc.Reset(), inst.qubits)

    c = measure_circuit()
    assert strs(decorate(c, is_measure, gen, before=before, parallel=parallel)) == expected
    f = decorate_on_match_parallel if parallel else decorate_on_match_single
    assert strs(f(c, is_measure, gen, before=before)) == expected

    r = decorate_inplace(c, is_measure, gen, before=before, parallel=parallel)
    assert r is c
    assert strs(c) == expected


def test_decorate_parallel_block_ends_on_generated_overlap():
    # The decoration of H@q[0] lands on q[1], so H@q[1] cannot join its block.
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 1)
    c.push(mc.GateH(), 3)

    def gen(inst):
        return mc.Instruction(mc.GateZ(), (inst.qubits[0] + 1,))

    r = decorate_on_match_parallel_inplace(c, lambda i: True, gen)
    assert r is c
    assert strs(c) == [
        "H @ q[0]",
        "Z @ q[1]",
        "H @ q[1]",
        "H @ q[3]",
        "Z @ q[2]",
        "Z @ q[4]",
    ]


def test_decorate_generator_output_forms():
    c = mc.Circuit().push(mc.GateH(), 0).push(mc.GateCX(), 0, 1)

    # (operation, qubits, bits)
    out = decorate_on_match_single(
        c, lambda i: i.operation == mc.GateH(), lambda i: (mc.Measure(), i.qubits, [3])
    )
    assert strs(out) == ["H @ q[0]", "M @ q[0], c[3]", "CX @ q[0], q[1]"]

    # (operation, *targets) and the Barrier class
    out = decorate_on_match_single(
        c, lambda i: True, lambda i: (mc.Barrier, *i.qubits), before=True
    )
    assert strs(out) == [
        "Barrier @ q[0]",
        "H @ q[0]",
        "Barrier @ q[0:1]",
        "CX @ q[0], q[1]",
    ]

    r = decorate_on_match_single_inplace(c, lambda i: True, lambda i: (mc.GateZ(), 1))
    assert r is c
    assert strs(c) == ["H @ q[0]", "Z @ q[1]", "CX @ q[0], q[1]", "Z @ q[1]"]

    with pytest.raises(TypeError, match="Generator must return"):
        decorate_on_match_single(c, lambda i: True, lambda i: mc.GateZ())


def test_decorate_empty_and_unmatched():
    assert len(add_noise(mc.Circuit(), mc.GateH(), D)) == 0
    c = mc.Circuit().push(mc.GateX(), 0)
    assert strs(add_noise(c, mc.GateH(), D, parallel=True)) == ["X @ q[0]"]


def test_circuit_methods_delegate():
    c = mc.Circuit().push(mc.GateH(), [0, 1])
    assert strs(c.decorate_on_match_single(mc.GateH(), D)) == [
        "H @ q[0]",
        "Depolarizing(0.1) @ q[0]",
        "H @ q[1]",
        "Depolarizing(0.1) @ q[1]",
    ]
    assert len(c) == 2
    r = c.decorate_on_match_parallel(mc.GateH(), D, before=True)
    assert r is c
    assert strs(c) == [
        "Depolarizing(0.1) @ q[0]",
        "Depolarizing(0.1) @ q[1]",
        "H @ q[0]",
        "H @ q[1]",
    ]


def test_decorate_invalidates_cached_resources():
    c = mc.Circuit().push(mc.GateH(), 0)
    assert c.num_bits() == 0
    decorate_on_match_single_inplace(c, lambda i: True, lambda i: (mc.Measure(), [0], [2]))
    assert c.num_bits() == 3


def test_insert_circuit():
    c = mc.Circuit().push(mc.GateX(), 0).push(mc.GateY(), 0)
    sub = mc.Circuit().push(mc.GateH(), 1).push(mc.GateS(), 1)
    c.insert(1, sub)
    assert strs(c) == ["X @ q[0]", "H @ q[1]", "S @ q[1]", "Y @ q[0]"]
    c.insert(10, sub)
    assert strs(c)[-2:] == ["H @ q[1]", "S @ q[1]"]
    c.insert(0, c)
    assert len(c) == 12


def test_insert_circuit_negative_index():
    c = mc.Circuit().push(mc.GateX(), 0).push(mc.GateY(), 0).push(mc.GateZ(), 0)
    c.insert(-1, mc.Circuit().push(mc.GateH(), 1).push(mc.GateS(), 1))
    assert strs(c) == ["X @ q[0]", "Y @ q[0]", "H @ q[1]", "S @ q[1]", "Z @ q[0]"]


@pytest.mark.parametrize(
    "before,expected",
    [
        (False, ["H @ q[0]", "Z @ q[0]", "S @ q[0]", "X @ q[0]", "Z @ q[0]", "S @ q[0]"]),
        (True, ["Z @ q[0]", "S @ q[0]", "H @ q[0]", "Z @ q[0]", "S @ q[0]", "X @ q[0]"]),
    ],
)
def test_decorate_with_circuit_generator(before, expected):
    # Each match gets the whole generated circuit once; its instructions are
    # not matched again even though they satisfy the matcher.
    c = mc.Circuit().push(mc.GateH(), 0).push(mc.GateX(), 0)
    deco = mc.Circuit().push(mc.GateZ(), 0).push(mc.GateS(), 0)
    decorate_on_match_single_inplace(c, lambda i: True, lambda i: (deco,), before=before)
    assert strs(c) == expected


def test_insert_rejects_non_integer_index():
    # A slice would read None as "the whole list" and replace the circuit.
    c = mc.Circuit().push(mc.GateX(), 0).push(mc.GateY(), 0)
    sub = mc.Circuit().push(mc.GateH(), 1)
    for index in (None, 1.0):
        with pytest.raises(TypeError):
            c.insert(index, sub)
        with pytest.raises(TypeError):
            c.insert(index, mc.GateH(), 0)
    assert strs(c) == ["X @ q[0]", "Y @ q[0]"]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("before", [False, True])
def test_raising_generator_leaves_circuit_untouched(before, parallel):
    c = base_circuit()
    expected = strs(c)
    calls = []

    def generator(inst):
        # Fail on the second match, after one decoration was generated.
        calls.append(inst)
        if len(calls) == 2:
            raise RuntimeError("generator failed")
        return (D, inst.qubits)

    with pytest.raises(RuntimeError, match="generator failed"):
        decorate_inplace(
            c,
            lambda i: i.operation == mc.GateH(),
            generator,
            before=before,
            parallel=parallel,
        )
    assert len(calls) == 2
    assert strs(c) == expected


def conditional_circuit():
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.IfStatement(mc.GateX(), mc.BitString("1")), 1, 0)
    c.push(mc.IfStatement(mc.GateZ(), mc.BitString("1")), 2, 0)
    c.push(mc.Reset(), 0)
    c.push(mc.Reset(), 1)
    c.push(mc.GateH(), 1)
    return c


IF_X = "IF(c==1) X @ q[1], condition[0]"
IF_Z = "IF(c==1) Z @ q[2], condition[0]"


@pytest.mark.parametrize(
    "g,before,parallel,expected",
    [
        (
            mc.IfStatement,
            False,
            False,
            ["M @ q[0], c[0]", IF_X, "Depolarizing(0.1) @ q[1]", IF_Z,
             "Depolarizing(0.1) @ q[2]", "Reset @ q[0]", "Reset @ q[1]", "H @ q[1]"],
        ),
        # The two conditionals read the same bit, so parallel mode does not
        # group them into one block.
        (
            mc.IfStatement,
            True,
            True,
            ["M @ q[0], c[0]", "Depolarizing(0.1) @ q[1]", IF_X,
             "Depolarizing(0.1) @ q[2]", IF_Z, "Reset @ q[0]", "Reset @ q[1]", "H @ q[1]"],
        ),
        (
            mc.Reset,
            False,
            True,
            ["M @ q[0], c[0]", IF_X, IF_Z, "Reset @ q[0]", "Reset @ q[1]",
             "Depolarizing(0.1) @ q[0]", "Depolarizing(0.1) @ q[1]", "H @ q[1]"],
        ),
        (
            mc.Reset,
            True,
            False,
            ["M @ q[0], c[0]", IF_X, IF_Z, "Depolarizing(0.1) @ q[0]", "Reset @ q[0]",
             "Depolarizing(0.1) @ q[1]", "Reset @ q[1]", "H @ q[1]"],
        ),
    ],
)
def test_add_noise_by_type_on_ifstatement_and_reset(g, before, parallel, expected):
    c = conditional_circuit()
    add_noise_inplace(c, g, D, before=before, parallel=parallel)
    assert strs(c) == expected


@pytest.mark.parametrize(
    "before,expected",
    [
        (
            False,
            ["H @ q[0]", "H @ q[1]", "Z @ q[0]", "S @ q[0]", "Z @ q[0]", "S @ q[0]",
             "X @ q[0]", "Z @ q[0]", "S @ q[0]"],
        ),
        (
            True,
            ["Z @ q[0]", "S @ q[0]", "Z @ q[0]", "S @ q[0]", "H @ q[0]", "H @ q[1]",
             "Z @ q[0]", "S @ q[0]", "X @ q[0]"],
        ),
    ],
)
def test_decorate_parallel_with_circuit_generator(before, expected):
    # The H block gets one copy of the generated circuit per matched H, all
    # placed together on one side of the block.
    c = mc.Circuit().push(mc.GateH(), [0, 1]).push(mc.GateX(), 0)
    deco = mc.Circuit().push(mc.GateZ(), 0).push(mc.GateS(), 0)
    decorate_on_match_parallel_inplace(
        c, lambda i: True, lambda i: (deco,), before=before
    )
    assert strs(c) == expected
