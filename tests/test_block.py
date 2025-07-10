import pytest
import mimiqcircuits as mc
import io


def test_block_construction():
    # Empty block
    b1 = mc.Block()
    assert b1._nq == 0
    assert b1._nc == 0
    assert b1._nz == 0
    assert len(b1) == 0

    # Block with specific dimensions
    b2 = mc.Block(2, 1, 1)
    assert b2._nq == 2
    assert b2._nc == 1
    assert b2._nz == 1
    assert len(b2) == 0

    # Block with named parameters
    b3 = mc.Block(3, 2, 1)
    assert b3._nq == 3
    assert b3._nc == 2
    assert b3._nz == 1

    # Block from a circuit
    c = mc.Circuit()
    c.push(mc.GateH(), 1)
    c.push(mc.GateCX(), 1, 2)
    b4 = mc.Block(c)
    assert b4._nq == 3
    assert b4._nc == 0
    assert b4._nz == 0
    assert len(b4) == 2

    # Block from instruction list
    insts = [
        mc.Instruction(mc.GateH(), (1,), (), ()),
        mc.Instruction(mc.GateCX(), (1, 2), (), ()),
    ]
    b5 = mc.Block(insts)
    assert b5._nq == 3
    assert b5._nc == 0
    assert b5._nz == 0
    assert len(b5) == 2


def test_block_operations():
    b = mc.Block(2, 1, 0)
    b.push(mc.GateH(), 1)
    b.push(mc.GateCX(), 0, 1)
    b.push(mc.Measure(), 0, 0)

    assert len(b) == 3
    assert b[0] == mc.Instruction(mc.GateH(), (1,), (), ())
    assert list(b) == list(b.instructions)

    ops = [inst.operation for inst in b]
    assert ops == [mc.GateH(), mc.GateCX(), mc.Measure()]

    b_slice = b[0:2]
    assert len(b_slice) == 2
    assert b_slice.instructions[0].operation == mc.GateH()

    with pytest.raises(ValueError):
        b.push(mc.GateH(), 3)

    with pytest.raises(ValueError):
        b.push(mc.Measure(), 1, 2)


def test_block_in_circuit():
    b = mc.Block(2, 0, 0)
    b.push(mc.GateH(), 0)
    b.push(mc.GateCX(), 0, 1)

    c = mc.Circuit()
    c.push(b, 0, 1)
    c1 = c.decompose()

    assert len(c1) == 2
    assert c1[0].operation == mc.GateH()
    assert c1[0].qubits == (0,)
    assert c1[1].operation == mc.GateCX()
    assert c1[1].qubits == (0, 1)

    # Nested block
    b_inner = mc.Block(1, 0, 0)
    b_inner.push(mc.GateX(), 0)
    b_outer = mc.Block(2, 0, 0)
    b_outer.push(mc.GateH(), 1)
    b_outer.push(b_inner, 0)

    c2 = b_outer.decompose()
    assert len(c2) == 2
    assert isinstance(c2[0].operation, (mc.GateH))
    assert isinstance(c2[1].operation, (mc.Block))


def test_block_protobuf_serialization():
    # Create a block with operations
    b = mc.Block(3, 1, 0)
    b.push(mc.GateH(), 1)
    b.push(mc.GateCX(), 1, 2)
    b.push(mc.Measure(), 2, 0)

    # Create a circuit containing the block
    c = mc.Circuit()
    c.push(b, 3, 4, 5, 0)  # 3 qubits: 3,4,5 and bit 0

    # Serialize to protobuf
    buffer = io.BytesIO()
    c.saveproto(buffer)
    buffer.seek(0)

    # Deserialize from protobuf
    restored_c = mc.Circuit.loadproto(buffer)

    assert len(restored_c) == len(c)
    inst = restored_c[0]
    assert isinstance(inst.operation, mc.Block)
    assert inst.qubits == (3, 4, 5)
    assert inst.bits == (0,)

    # Validate block contents
    block = inst.operation
    assert isinstance(block[0].operation, mc.GateH)
    assert isinstance(block[2].operation, mc.Measure)

    # Test nested block serialization
    inner = mc.Block(1, 0, 0)
    inner.push(mc.GateX(), 0)

    outer = mc.Block(2, 0, 0)
    outer.push(mc.GateH(), 1)
    outer.push(inner, 0)

    circuit_nested = mc.Circuit()
    circuit_nested.push(outer, 1, 2)

    nested_buf = io.BytesIO()
    circuit_nested.saveproto(nested_buf)
    nested_buf.seek(0)

    restored_nested = mc.Circuit.loadproto(nested_buf)
    assert len(restored_nested) == 1
    nested_inst = restored_nested[0]
    assert isinstance(nested_inst.operation, mc.Block)

    restored_outer = nested_inst.operation
    assert isinstance(restored_outer[0].operation, mc.GateH)
    assert isinstance(restored_outer[1].operation, mc.Block)
    assert isinstance(restored_outer[1].decompose()[0].operation, mc.GateX)
