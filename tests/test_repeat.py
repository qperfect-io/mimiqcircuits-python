import pytest
import mimiqcircuits as mc
import io


def test_repeat_construction():
    r1 = mc.Repeat(13, mc.GateX())
    assert r1.num_qubits == 1
    assert r1.num_bits == 0
    assert r1.num_zvars == 0
    assert r1.repeats == 13

    with pytest.raises(ValueError):
        mc.Repeat(-1, mc.GateX())

    r2 = mc.Repeat(12, mc.Control(3, mc.GateX()))
    assert r2.num_qubits == 4
    assert r2.num_bits == 0
    assert r2.num_zvars == 0
    assert r2.repeats == 12

    r3 = mc.Repeat(11, mc.Measure())
    assert r3.num_qubits == 1
    assert r3.num_bits == 1
    assert r3.num_zvars == 0
    assert r3.repeats == 11


def test_repeat_protobuf_save_load():
    c = mc.Circuit()
    c.push(mc.Repeat(3, mc.GateX()), 1)
    c.push(mc.Repeat(3, mc.GateX()), 4)
    c.push(mc.Repeat(3, mc.GateCH()), 4, 5)
    c.push(mc.Repeat(3, mc.GateRZZ(3.21)), 3, 4)

    b = mc.Block(3, 2, 1)
    b.push(mc.GateH(), 0)
    b.push(mc.GateCX(), 0, 1)
    b.push(mc.Measure(), 1, 1)
    b.push(mc.ExpectationValue(mc.PauliString("ZZ")), 0, 2, 0)
    b.push(mc.MeasureZZ(), 0, 2, 1)

    c.push(mc.Repeat(3, b), 0, 1, 2, 0, 1, 0)

    buffer = io.BytesIO()
    c.saveproto(buffer)
    buffer.seek(0)
    loaded = mc.Circuit.loadproto(buffer)

    assert isinstance(loaded, mc.Circuit)
    assert len(loaded) == len(c)

    for i in range(len(c)):
        assert isinstance(loaded[i].operation, mc.Repeat)
        assert loaded[i].operation.repeats == 3
