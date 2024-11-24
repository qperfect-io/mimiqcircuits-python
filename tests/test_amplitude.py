import pytest
import mimiqcircuits as mc


def test_amplitude_initialization():
    bitstring = mc.BitString(2)
    amp = mc.Amplitude(bitstring)
    assert amp.bs == bitstring  # Changed from amp.bitstring to amp.bs
    assert amp.qregsizes is None
    assert amp.cregsizes is None
    assert amp.zregsizes == [1]  # Remains the same


def test_amplitude_inverse_not_implemented():
    bitstring = mc.BitString(2)
    amp = mc.Amplitude(bitstring)
    with pytest.raises(
        NotImplementedError, match="Cannot inverse an Amplitude operation."
    ):
        amp.inverse()


def test_amplitude_power_not_implemented():
    b = mc.BitString(2)
    amp = mc.Amplitude(b)
    with pytest.raises(
        NotImplementedError, match="Cannot elevate an Amplitude operation to any power."
    ):
        amp._power(2)


def test_amplitude_decompose():
    bitstring = mc.BitString(2)
    amp = mc.Amplitude(bitstring)
    circ = mc.Circuit()
    zvars = [0]
    new_circ = amp._decompose(circ, [], [], zvars)
    assert len(new_circ) == 1
    assert new_circ[0].operation == amp
