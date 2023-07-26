import pytest
import mimiqcircuits.reset as mr


def test_reset_inverse_raises_error():
    reset = mr.Reset()
    with pytest.raises(TypeError):
        reset.inverse()

def test_reset_from_json_valid_json():
    json_data = {"name": "Reset"}
    reset = mr.Reset.from_json(json_data)
    assert isinstance(reset, mr.Reset)

def test_reset_from_json_invalid_json():
    json_data = {"name": "InvalidReset"}
    with pytest.raises(ValueError):
        mr.Reset.from_json(json_data)