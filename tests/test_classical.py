import pytest
from mimiqcircuits import Circuit, Not, SetBit0, SetBit1, And, Or, Xor, ParityCheck
from mimiqcircuits.proto import circuit_pb2
from mimiqcircuits.proto.circuitproto import fromproto_generalized_operation
import io


def test_not_basic_repr_inverse():
    op = Not()
    assert op.name == "!"
    assert repr(op) == "!"
    assert isinstance(op.inverse(), Not)
    assert not op.iswrapper()
    assert op.get_operation() is op


def test_setbits0_basic_repr_inverse():
    op = SetBit0()
    assert op.name == "set0"
    assert repr(op) == "set0"
    inv = op.inverse()
    assert isinstance(inv, SetBit1)
    assert not op.iswrapper()
    assert op.get_operation() is op


def test_setbits1_basic_repr_inverse():
    op = SetBit1()
    assert op.name == "set1"
    assert repr(op) == "set1"
    inv = op.inverse()
    assert isinstance(inv, SetBit0)
    assert not op.iswrapper()
    assert op.get_operation() is op


@pytest.mark.parametrize("opclass", [Not, SetBit0, SetBit1])
def test_proto_roundtrip_io_simple(opclass):
    op = opclass()
    c = Circuit().push(op, 0)
    buf = io.BytesIO()
    c.saveproto(buf)
    buf.seek(0)

    loaded = Circuit.loadproto(buf)
    assert isinstance(loaded, Circuit)
    assert len(loaded.instructions) == 1
    inst = loaded.instructions[0]
    assert isinstance(inst.get_operation(), opclass)


@pytest.mark.parametrize(
    "op", [And(), Or(), Xor(), And(5), Or(4), Xor(6)]
)
def test_proto_roundtrip_io_multibits(op):
    c = Circuit().push(op, *range(op.num_bits))
    buf = io.BytesIO()
    c.saveproto(buf)
    buf.seek(0)

    loaded = Circuit.loadproto(buf)
    assert isinstance(loaded, Circuit)
    assert len(loaded.instructions) == 1
    inst = loaded.instructions[0]
    assert isinstance(inst.get_operation(), type(op))
    assert inst.get_operation().num_bits == op.num_bits


def test_paritycheck_is_deprecated_alias_of_xor():
    with pytest.warns(DeprecationWarning):
        op = ParityCheck(5)
    assert op == Xor(5)


def test_paritycheck_proto_loads_as_xor():
    pb = circuit_pb2.GeneralizedOperation(
        mtype=circuit_pb2.GeneralizedOperationType.ParityCheck, numbits=4
    )
    assert fromproto_generalized_operation(pb) == Xor(4)
