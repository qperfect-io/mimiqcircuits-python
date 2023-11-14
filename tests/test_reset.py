import pytest
import mimiqcircuits as mc


def test_reset_inverse_raises_error():
    reset = mc.Reset()
    with pytest.raises(TypeError):
        reset.inverse()
