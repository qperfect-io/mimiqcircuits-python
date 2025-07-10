import pytest
import mimiqcircuits as mc


def test_pow_operation_basic():
    op = mc.Pow(3)
    assert op.exponent == 3
    assert op._num_zvars == 1
    assert op._num_zregs == 1
    assert op._zregsizes == [1]
    assert str(op) == "Pow(3)"
    assert op.format_with_targets([], [], [2]) == "z[2] = z[2]^3"


def test_pow_inverse():
    op = mc.Pow(4)
    inv = op.inverse()
    assert isinstance(inv, mc.Pow)
    assert inv.exponent == -4


def test_pow_noop_warning():
    with pytest.warns(UserWarning, match="Pow\\(1\\) will be equivalent to a no-op."):
        mc.Pow(1)


def test_add_operation_basic():
    op = mc.Add(3, c=5.0)
    assert op._num_zvars == 3
    assert op.term == 5.0
    assert op._zregsizes == [3]
    assert str(op) == "Add(3, c=5.0)"
    assert op.format_with_targets([], [], [0, 1, 2]) == "z[0] += 5.0 + z[1] + z[2]"


def test_add_zero_warning():
    with pytest.warns(
        UserWarning, match="Add\\(1; c=0.0\\) will be equivalent to a no-op."
    ):
        mc.Add(1, 0.0)


def test_add_invalid_argument():
    with pytest.raises(ValueError):
        mc.Add(0)


def test_multiply_operation_basic():
    op = mc.Multiply(2, c=3.0)
    assert op._num_zvars == 2
    assert op.factor == 3.0
    assert op._zregsizes == [2]
    assert str(op) == "Multiply(2, c=3.0)"
    assert op.format_with_targets([], [], [0, 1]) == "z[0] *= 3.0 * z[1]"


def test_multiply_noop_warning():
    with pytest.warns(
        UserWarning, match="Multiply\\(1; c=1.0\\) will be equivalent to a no-op."
    ):
        mc.Multiply(1, 1.0)


def test_multiply_invalid_argument():
    with pytest.raises(ValueError):
        mc.Multiply(0)
