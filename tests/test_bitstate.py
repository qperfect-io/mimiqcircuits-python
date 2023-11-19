import pytest
from mimiqcircuits import BitString
from bitarray import frozenbitarray, bitarray


def test_init_int():
    """Test BitString initialization from integer.
    """
    for i in [0, 1, 2, 3, 5, 9, 16, 32, 123, 12324]:
        assert len(BitString(i)) == i
        for j in range(i):
            assert BitString(i)[j] == 0


def test_init_str():
    """Test BitString initialization from string.
    """
    for s in ['', '1',  '0', '11', '10', '10111011', '00000000']:
        assert BitString(s).to01() == s


def test_init_bitarray():
    """Test BitString initialization from bitarray.
    """
    for s in ['', '1',  '0', '11', '10', '10111011', '00000000']:
        assert BitString(frozenbitarray(s)).to01() == s
        assert BitString(bitarray(s)).to01() == s


def test_init_list():
    """Test BitString initialization from list.
    """
    for s in [[], [1], [0], [1, 0], [1, 1], [1, 0, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]:
        bs = BitString(s)
        for j in range(len(s)):
            assert bs[j] == s[j]

    with pytest.raises(ValueError):
        BitString([1, 0, 2])


def test_init_tuple():
    """Test BitString initialization from tuple.
    """
    for s in [(), (1,), (0,), (1, 0), (1, 1), (1, 0, 1, 1, 1, 0, 1, 1), (0, 0, 0, 0, 0)]:
        bs = BitString(s)
        for j in range(len(s)):
            assert bs[j] == s[j]

    with pytest.raises(ValueError):
        BitString((1, 2, 0))


def test_init_other():
    """Test BitString initialization from other types.
    Should raise TypeError for any case that is not of the ones above.
    """
    with pytest.raises(TypeError):
        BitString(1.0)

    with pytest.raises(TypeError):
        BitString(1+1j)

    with pytest.raises(TypeError):
        BitString({1, 0, 1})

    with pytest.raises(TypeError):
        BitString({1: 0, 2: 1})

    with pytest.raises(TypeError):
        BitString(None)


def test_init_fromnonzeros():
    """Test BitString initialization from nonzeros.
    """
    assert BitString.fromnonzeros(0, []) == BitString(0)
    assert BitString.fromnonzeros(1, []) == BitString(1)
    assert BitString.fromnonzeros(1, [0]) == BitString('1')
    assert BitString.fromnonzeros(2, [1]) == BitString('01')
    assert BitString.fromnonzeros(3, [0, 2]) == BitString('101')

    with pytest.raises(ValueError):
        BitString.fromnonzeros(2, [0, 2])


def test_init_fromfunction():
    """Test BitString initialization from function.
    """
    assert BitString.fromfunction(0, lambda x: 0) == BitString(0)
    assert BitString.fromfunction(3, lambda x: 0) == BitString('000')
    assert BitString.fromfunction(3, lambda x: 1) == BitString('111')
    assert BitString.fromfunction(6, lambda x: x %
                                  2 == 0) == BitString('101010')


def test_int_conversion():
    """Test BitString to int conversion.
    """
    for i in [0, 1, 2, 3, 5, 9, 16, 32, 123, 12324]:
        assert BitString.fromint(31, i).tointeger() == i


def test_zeros_nonzeros():
    """Test BitString zeros and nonzeros methods.
    """
    assert BitString(0).zeros() == []
    assert BitString(0).nonzeros() == []
    assert BitString(3).zeros() == [0, 1, 2]
    assert BitString(3).nonzeros() == []
    assert BitString('101').zeros() == [1]
    assert BitString('101').nonzeros() == [0, 2]


def test_num_qubits():
    """Test BitString num_qubits method.
    """
    assert BitString(0).num_qubits() == 0
    assert BitString(1).num_qubits() == 1
    assert BitString(2).num_qubits() == 2
    assert BitString(1234).num_qubits() == 1234


def test_len():
    """Test BitString len method.
    """
    assert len(BitString(0)) == 0
    assert len(BitString(1)) == 1
    assert len(BitString(2)) == 2
    assert len(BitString(1234)) == 1234


def test_bit_operations():
    """Test BitString bit operations.
    """
    assert BitString('0') & BitString('0') == BitString('0')
    assert BitString('0') & BitString('1') == BitString('0')
    assert BitString('1') & BitString('0') == BitString('0')
    assert BitString('1') & BitString('1') == BitString('1')
    assert BitString('0') | BitString('0') == BitString('0')
    assert BitString('0') | BitString('1') == BitString('1')
    assert BitString('1') | BitString('0') == BitString('1')
    assert BitString('1') | BitString('1') == BitString('1')
    assert BitString('0') ^ BitString('0') == BitString('0')
    assert BitString('0') ^ BitString('1') == BitString('1')
    assert BitString('1') ^ BitString('0') == BitString('1')
    assert BitString('1') ^ BitString('1') == BitString('0')
    assert ~BitString('0') == BitString('1')
    assert ~BitString('1') == BitString('0')

    assert BitString('110101001') & BitString(
        '101011101') == BitString('100001001')
    assert BitString('110101001') | BitString(
        '101011101') == BitString('111111101')
    assert BitString('110101001') ^ BitString(
        '101011101') == BitString('011110100')
    assert ~BitString('110101001') == BitString('001010110')


def test_repeat_concatenate():
    """Test BitString repeat and concatenate methods.
    """
    assert BitString('0') * 3 == BitString('000')
    assert BitString('10') * 4 == BitString('10101010')
    assert BitString('10011') + BitString('101') == BitString('10011101')


def test_bitshifts():
    """Test BitString bitshifts.
    """
    assert BitString('00110') << 2 == BitString('11000')
    assert BitString('00110') >> 2 == BitString('00001')
