import os
import tempfile

import mimiqcircuits as mc
from mimiqcircuits.proto.circuitproto import toproto_operation, fromproto_operation


def test_loss_operation_proto_roundtrip():
    operations = [
        mc.Loss(),
        mc.Reload(),
        mc.Check(),
        mc.MeasureCheck(),
        mc.Loss(0.25),
    ]

    for operation in operations:
        restored = fromproto_operation(toproto_operation(operation))
        assert type(restored) is type(operation)
        assert restored == operation


def test_circuit_with_loss_operations_proto_roundtrip():
    circuit = mc.Circuit()
    circuit.push(mc.Loss(0.5), 0)
    circuit.push(mc.Loss(), 1)
    circuit.push(mc.Check(), 0, 0)
    circuit.push(mc.MeasureCheck(), 1, 1, 2)
    circuit.push(mc.Reload(), 1)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "loss_circuit.pb")
        circuit.saveproto(path)
        restored = mc.Circuit.loadproto(path)

    assert restored == circuit


def test_circuit_with_loss_annotations_proto_roundtrip():
    circuit = mc.Circuit()
    circuit.push(mc.Lost(), 0)
    circuit.push(mc.Reloaded(), 1)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "loss_annotations_circuit.pb")
        circuit.saveproto(path)
        restored = mc.Circuit.loadproto(path)

    assert restored == circuit
    assert type(restored[0].operation) is mc.Lost
    assert type(restored[1].operation) is mc.Reloaded


def test_lossmodel_proto_roundtrip():
    model = mc.LossModel(
        [
            mc.DropRule(),
            mc.DropRule(mc.GateSWAP()),
            mc.ReplaceRule(mc.GateCX(), mc.Depolarizing1(0.2)),
            mc.DecorateRule(mc.GateCZ(), mc.GateX(), before=True),
        ],
        name="Loss Proto",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lossmodel.pb")
        model.saveproto(path)
        restored = mc.LossModel.loadproto(path)

    assert isinstance(restored, mc.LossModel)
    assert restored.name == "Loss Proto"
    assert len(restored.rules) == len(model.rules)
    for orig, new in zip(model.rules, restored.rules):
        assert type(orig) is type(new)
        assert orig == new


def test_legacy_qubitloss_decodes_to_loss():
    from mimiqcircuits.proto import circuit_pb2 as pb

    # Pre-redesign circuits encoded certain loss as the parameterless QubitLoss
    # (OperationType tag 16). New decoders fold it into Loss() so those circuits
    # keep loading; the encoder only ever emits the Loss message.
    legacy = pb.Operation(simpleoperation=pb.SimpleOperation(mtype=16))
    assert fromproto_operation(legacy) == mc.Loss()
    assert toproto_operation(mc.Loss()).WhichOneof("operation") == "loss"
