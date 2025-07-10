import pytest
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
