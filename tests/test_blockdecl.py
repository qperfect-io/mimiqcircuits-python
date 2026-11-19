import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.proto.circuitproto import fromproto_circuit, toproto_circuit
from mimiqcircuits.proto.circuitproto import fromproto_operation, toproto_operation

theta, phi = se.symbols("theta phi")


@mc.blockdecl("syn")
def syn(a):
    return (
        mc.Circuit()
        .push(mc.GateRX(a), 0)
        .push(mc.GateCX(), 0, 1)
        .push(mc.Measure(), 1, 0)
    )


def test_construction():
    c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.Measure(), 1, 0)
    decl = mc.BlockDecl("syn", (theta,), c)
    assert decl.name == "syn"
    assert (decl.num_qubits, decl.num_bits, decl.num_zvars) == (2, 1, 0)
    assert len(decl) == 2
    assert decl == mc.BlockDecl("syn", (theta,), list(c))

    # a Block keeps its declared sizes
    b = mc.Block(3, 2, 1).push(mc.GateH(), 0)
    decl = mc.BlockDecl("wide", (), b)
    assert (decl.num_qubits, decl.num_bits, decl.num_zvars) == (3, 2, 1)

    assert len(mc.BlockDecl("empty", (), [])) == 0


def test_validation():
    c = mc.Circuit().push(mc.GateRX(theta), 0)
    with pytest.raises(TypeError):
        mc.BlockDecl("bad", (theta + 1,), c)
    with pytest.raises(ValueError):
        mc.BlockDecl("bad", (theta, theta), c)
    # theta is used in the body but is not an argument
    with pytest.raises(ValueError):
        mc.BlockDecl("bad", (), c)
    with pytest.raises(ValueError):
        mc.BlockDecl("bad", (phi,), c)


def test_call():
    call = syn(0.1)
    assert isinstance(call, mc.BlockCall)
    assert call.decl is syn
    assert call.num_qubits == 2 and call.num_bits == 1
    assert not call.isunitary()
    assert call == syn(0.1)
    assert call != syn(0.2)
    with pytest.raises(ValueError):
        syn()
    with pytest.raises(ValueError):
        syn(0.1, 0.2)
    assert str(call) == "syn(0.1)"

    rot = mc.BlockDecl("rot", (theta,), mc.Circuit().push(mc.GateRX(theta), 0))
    assert rot(0.1).isunitary()
    prep = mc.BlockDecl("prep", (), mc.Circuit().push(mc.GateH(), 0))
    assert str(prep()) == "prep"


def test_decomposition():
    c = mc.Circuit().push(syn(0.1), 0, 1, 0).push(syn(0.2), 3, 2, 1)
    assert c.num_qubits() == 4
    assert c.num_bits() == 2

    expected = (
        mc.Circuit()
        .push(mc.GateRX(0.1), 0)
        .push(mc.GateCX(), 0, 1)
        .push(mc.Measure(), 1, 0)
        .push(mc.GateRX(0.2), 3)
        .push(mc.GateCX(), 3, 2)
        .push(mc.Measure(), 2, 1)
    )
    assert mc.decompose_step(c) == expected
    assert mc.decompose(c, basis=mc.FlattenedBasis()) == expected
    assert mc.decompose_step(c, rule=mc.FlattenContainers()) == expected

    # z-variables are remapped too
    ev = mc.BlockDecl(
        "ev", (), mc.Circuit().push(mc.ExpectationValue(mc.GateZ()), 0, 0)
    )
    c = mc.Circuit().push(ev(), 1, 2)
    assert mc.decompose_step(c)[0].zvars == (2,)

    # calls nested in a declaration
    outer = mc.BlockDecl("outer", (phi,), mc.Circuit().push(syn(2 * phi), 0, 1, 0))
    c = mc.Circuit().push(outer(0.1), 0, 1, 0)
    flat = mc.Circuit().push(syn(0.2), 0, 1, 0)
    assert mc.decompose(c, basis=mc.FlattenedBasis()) == mc.decompose(
        flat, basis=mc.FlattenedBasis()
    )


def test_symbolics():
    call = syn(2 * theta)
    assert call.is_symbolic()
    assert call.listvars() == [theta]
    assert not syn(0.1).is_symbolic()
    assert syn(0.1).listvars() == []

    c = mc.Circuit().push(call, 0, 1, 0)
    assert c.listvars() == [theta]

    evaluated = call.evaluate({theta: 0.5})
    assert evaluated == syn(1.0)
    assert evaluated.decl is syn
    assert c.evaluate({theta: 0.5})[0].operation.decl is syn


def test_gatecall_symbolics():
    rot = mc.GateDecl("rot", (theta,), mc.Circuit().push(mc.GateRX(theta), 0))
    assert rot(theta).is_symbolic()
    assert not rot(0.1).is_symbolic()
    assert set(rot(theta + phi).listvars()) == {theta, phi}


def test_proto():
    c = mc.Circuit()
    c.push(syn(0.1), 0, 1, 0)
    c.push(syn(theta), 2, 3, 1)
    c.push(mc.GateH(), 0)

    pb = toproto_circuit(c)
    # both calls refer to a single stored declaration
    assert len(pb.decls) == 1

    restored = fromproto_circuit(pb)
    assert restored == c
    assert restored[0].operation.decl is restored[1].operation.decl

    # outside a circuit the declaration is written inline
    call = syn(0.3)
    assert fromproto_operation(toproto_operation(call)) == call

    # nested declarations are stored before the ones that use them
    body = mc.Circuit().push(syn(phi), 0, 1, 0).push(syn(2 * phi), 1, 0, 0)
    outer = mc.BlockDecl("outer", (phi,), body)
    c = mc.Circuit().push(outer(0.4), 0, 1, 0)
    pb = toproto_circuit(c)
    assert len(pb.decls) == 2
    restored = fromproto_circuit(pb)
    assert restored == c
    inner = restored[0].operation.decl
    assert inner[0].operation.decl is inner[1].operation.decl


def test_noise_model():
    model = mc.NoiseModel(
        [mc.OperationInstanceNoise(mc.GateH(), mc.AmplitudeDamping(0.01))]
    )
    prep = mc.BlockDecl(
        "prep", (), mc.Circuit().push(mc.GateH(), 0).push(mc.Measure(), 0, 0)
    )
    c = mc.Circuit().push(prep(), 0, 0)

    noisy = mc.apply_noise_model(c, model)
    assert len(noisy) == 1
    inner = noisy[0].operation
    assert isinstance(inner, mc.Block)
    assert [i.operation for i in inner] == [
        mc.GateH(),
        mc.AmplitudeDamping(0.01),
        mc.Measure(),
    ]
