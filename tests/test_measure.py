import pytest
import mimiqcircuits as mc


def test_measure_inverse_raises_error():
    measure = mc.Measure()
    with pytest.raises(TypeError):
        measure.inverse()


def test_measure_from_json_valid_json():
    json_data = {"name": "Measure"}
    measure = mc.Measure.from_json(json_data)
    assert isinstance(measure, mc.Measure)


def test_measure_from_json_invalid_json():
    json_data = {"name": "InvalidMeasure"}
    with pytest.raises(ValueError):
        mc.Measure.from_json(json_data)
