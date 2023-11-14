import pytest
from mimiqcircuits import BitState
from bitarray import frozenbitarray, bitarray


def test_init_int():
    """Test BitState initialization from integer.
    """
    for i in [0, 1, 2, 3, 5, 9, 16, 32, 123, 12324]:
        assert len(BitState(i)) == i
        for j in range(i):
            assert BitState(i)[j] == 0


def test_init_str():
    """Test BitState initialization from string.
    """
    for s in ['', '1',  '0', '11', '10', '10111011', '00000000']:
        assert BitState(s).to01() == s


def test_init_bitarray():
    """Test BitState initialization from bitarray.
    """
    for s in ['', '1',  '0', '11', '10', '10111011', '00000000']:
        assert BitState(frozenbitarray(s)).to01() == s
        assert BitState(bitarray(s)).to01() == s


def test_init_list():
    """Test BitState initialization from list.
    """
    for s in [[], [1], [0], [1, 0], [1, 1], [1, 0, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]:
        bs = BitState(s)
        for j in range(len(s)):
            assert bs[j] == s[j]

    with pytest.raises(ValueError):
        BitState([1, 0, 2])


def test_init_tuple():
    """Test BitState initialization from tuple.
    """
    for s in [(), (1,), (0,), (1, 0), (1, 1), (1, 0, 1, 1, 1, 0, 1, 1), (0, 0, 0, 0, 0)]:
        bs = BitState(s)
        for j in range(len(s)):
            assert bs[j] == s[j]

    with pytest.raises(ValueError):
        BitState((1, 2, 0))


def test_init_other():
    """Test BitState initialization from other types.
    Should raise TypeError for any case that is not of the ones above.
    """
    with pytest.raises(TypeError):
        BitState(1.0)

    with pytest.raises(TypeError):
        BitState(1+1j)

    with pytest.raises(TypeError):
        BitState({1, 0, 1})

    with pytest.raises(TypeError):
        BitState({1: 0, 2: 1})

    with pytest.raises(TypeError):
        BitState(None)


def test_init_fromnonzeros():
    """Test BitState initialization from nonzeros.
    """
    assert BitState.fromnonzeros(0, []) == BitState(0)
    assert BitState.fromnonzeros(1, []) == BitState(1)
    assert BitState.fromnonzeros(1, [0]) == BitState('1')
    assert BitState.fromnonzeros(2, [1]) == BitState('01')
    assert BitState.fromnonzeros(3, [0, 2]) == BitState('101')

    with pytest.raises(ValueError):
        BitState.fromnonzeros(2, [0, 2])


def test_init_fromfunction():
    """Test BitState initialization from function.
    """
    assert BitState.fromfunction(0, lambda x: 0) == BitState(0)
    assert BitState.fromfunction(3, lambda x: 0) == BitState('000')
    assert BitState.fromfunction(3, lambda x: 1) == BitState('111')
    assert BitState.fromfunction(6, lambda x: x % 2 == 0) == BitState('101010')


def test_int_conversion():
    """Test BitState to int conversion.
    """
    for i in [0, 1, 2, 3, 5, 9, 16, 32, 123, 12324]:
        assert BitState.fromint(31, i).tointeger() == i


def test_zeros_nonzeros():
    """Test BitState zeros and nonzeros methods.
    """
    assert BitState(0).zeros() == []
    assert BitState(0).nonzeros() == []
    assert BitState(3).zeros() == [0, 1, 2]
    assert BitState(3).nonzeros() == []
    assert BitState('101').zeros() == [1]
    assert BitState('101').nonzeros() == [0, 2]


def test_num_qubits():
    """Test BitState num_qubits method.
    """
    assert BitState(0).num_qubits() == 0
    assert BitState(1).num_qubits() == 1
    assert BitState(2).num_qubits() == 2
    assert BitState(1234).num_qubits() == 1234


def test_len():
    """Test BitState len method.
    """
    assert len(BitState(0)) == 0
    assert len(BitState(1)) == 1
    assert len(BitState(2)) == 2
    assert len(BitState(1234)) == 1234


def test_bit_operations():
    """Test BitState bit operations.
    """
    assert BitState('0') & BitState('0') == BitState('0')
    assert BitState('0') & BitState('1') == BitState('0')
    assert BitState('1') & BitState('0') == BitState('0')
    assert BitState('1') & BitState('1') == BitState('1')
    assert BitState('0') | BitState('0') == BitState('0')
    assert BitState('0') | BitState('1') == BitState('1')
    assert BitState('1') | BitState('0') == BitState('1')
    assert BitState('1') | BitState('1') == BitState('1')
    assert BitState('0') ^ BitState('0') == BitState('0')
    assert BitState('0') ^ BitState('1') == BitState('1')
    assert BitState('1') ^ BitState('0') == BitState('1')
    assert BitState('1') ^ BitState('1') == BitState('0')
    assert ~BitState('0') == BitState('1')
    assert ~BitState('1') == BitState('0')

    assert BitState('110101001') & BitState(
        '101011101') == BitState('100001001')
    assert BitState('110101001') | BitState(
        '101011101') == BitState('111111101')
    assert BitState('110101001') ^ BitState(
        '101011101') == BitState('011110100')
    assert ~BitState('110101001') == BitState('001010110')


def test_repeat_concatenate():
    """Test BitState repeat and concatenate methods.
    """
    assert BitState('0') * 3 == BitState('000')
    assert BitState('10') * 4 == BitState('10101010')
    assert BitState('10011') + BitState('101') == BitState('10011101')


def test_bitshifts():
    """Test BitState bitshifts.
    """
    assert BitState('00110') << 2 == BitState('11000')
    assert BitState('00110') >> 2 == BitState('00001')
