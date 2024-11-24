import pytest
import mimiqcircuits as mc


def test_measure_inverse_raises_error():
    measure = mc.Measure()
    with pytest.raises(TypeError):
        measure.inverse()
