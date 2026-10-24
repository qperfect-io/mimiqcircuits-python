import pytest
import numpy as np
from symengine import symbols
import mimiqcircuits as mc


@pytest.fixture
def symbolic_circuit():
    x, y = symbols("x y")
    c = mc.Circuit()
    c.push(mc.GateXXplusYY(x, y), 1, 2)
    return c, x, y


def test_listvars_only_symbolics(symbolic_circuit):
    c, x, y = symbolic_circuit
    vars_list = c.listvars()
    assert set(vars_list) == {x, y}, "listvars() should return all symbolic variables"


def test_getparams_contains_symbolics(symbolic_circuit):
    c, x, y = symbolic_circuit
    params = c.getparams()
    assert x in params and y in params, "getparams() must include symbolic variables"


def test_listvars_getparams(symbolic_circuit):
    c, x, y = symbolic_circuit

    assert set(c.listvars()) == set(c.getparams())
    assert set(c.listvars()) == {x, y}

    # Add a numeric parameter
    c.push(mc.GateRX(1.0), 1)
    assert set(c.listvars()) == {x, y}

    # getparams includes numeric, so they are now different
    assert set(c.listvars()) != set(c.getparams())


def test_listvars_vs_getparams_with_numeric():
    x, y, z = symbols("x y z")

    c = mc.Circuit()
    c.push(mc.GateXXplusYY(x + 2, y**2), 1, 2)
    c.push(mc.GateRX(1.0), 1)
    c.push(mc.GateRY(z), 2)

    assert set(c.listvars()) == {
        x,
        y,
        z,
    }, "listvars() should only return symbolic variables"

    all_params = c.getparams()
    assert x + 2 in all_params and y**2 in all_params and z in all_params
    assert any(
        p == 1.0 for p in all_params
    ), "getparams() should include numerical constants"

    assert set(c.listvars()) != set(
        all_params
    ), "listvars() and getparams() should differ with mixed parameter types"


x = symbols("x")


def _block():
    return mc.Block(mc.Circuit().push(mc.GateRX(x), 0))


@pytest.mark.parametrize(
    "op",
    [
        _block(),
        mc.Parallel(2, mc.GateRX(x)),
        mc.Control(1, mc.GateRX(x)),
        mc.Inverse(mc.GateRX(x)),
        mc.Power(mc.GateRX(x), 0.5),
        mc.Repeat(2, _block()),
        mc.IfStatement(_block(), mc.BitString("1")),
        mc.WhileStatement(_block(), mc.BitString("1")),
        mc.Control(1, mc.Parallel(2, mc.GateRX(x))),
    ],
)
def test_listvars_descends_into_wrappers(op):
    assert op.listvars() == [x]
    assert op.is_symbolic()


def test_sweep_over_a_symbol_inside_a_block():
    c = mc.Circuit().push(_block(), 0)
    ex = mc.SweepExperiment(c, mc.ParameterTable.from_columns({x: [0.1, 0.2]}))
    assert len(ex) == 2


@pytest.mark.parametrize(
    "wrap, simplified",
    [
        (mc.Inverse, mc.GateRX(-0.3)),
        (lambda g: mc.Power(g, 0.5), mc.GateRX(0.15)),
        (lambda g: mc.Control(1, g), mc.GateCRX(0.3)),
        (lambda g: mc.Parallel(2, g), mc.Parallel(2, mc.GateRX(0.3))),
        (lambda g: mc.Repeat(2, g), mc.Repeat(2, mc.GateRX(0.3))),
        (
            lambda g: mc.IfStatement(g, mc.BitString("1")),
            mc.IfStatement(mc.GateRX(0.3), mc.BitString("1")),
        ),
    ],
)
def test_evaluate_rebuilds_only_symbolic_wrappers(wrap, simplified):
    # Free symbols: rebuilt through the simplifying builders.
    assert wrap(mc.GateRX(x)).evaluate({x: 0.3}) == simplified
    # No free symbols: returned untouched.
    op = wrap(mc.GateRX(0.3))
    assert op.evaluate({}) is op


def test_evaluate_leaves_numeric_containers_untouched():
    numeric = mc.Block(mc.Circuit().push(mc.GateRX(0.3), 0))
    for op in [
        numeric,
        mc.WhileStatement(numeric, mc.BitString("1")),
        mc.Inverse(mc.GateCustom(np.array([[0, 1], [1, 0]]))),
    ]:
        assert op.evaluate({}) is op
    evaluated = mc.Inverse(mc.GateRX(0.3)).evaluate({})
    assert evaluated == mc.Inverse(mc.GateRX(0.3))
