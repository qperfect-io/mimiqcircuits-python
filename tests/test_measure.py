import pytest
import mimiqcircuits.measure as mm


def test_measure_inverse_raises_error():
    measure = mm.Measure()
    with pytest.raises(TypeError):
        measure.inverse()

def test_measure_from_json_valid_json():
    json_data = {"name": "Measure"}
    measure = mm.Measure.from_json(json_data)
    assert isinstance(measure, mm.Measure)

def test_measure_from_json_invalid_json():
    json_data = {"name": "InvalidMeasure"}
    with pytest.raises(ValueError):
        mm.Measure.from_json(json_data)