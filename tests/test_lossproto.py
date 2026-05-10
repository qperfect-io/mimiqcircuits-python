import os
import tempfile

import mimiqcircuits as mc
from mimiqcircuits.proto.circuitproto import toproto_operation, fromproto_operation


def test_loss_operation_proto_roundtrip():
    operations = [
        mc.QubitLoss(),
        mc.QubitReload(),
        mc.CheckLoss(),
        mc.MeasureCheckLoss(),
        mc.LossErr(0.25),
    ]

    for operation in operations:
        restored = fromproto_operation(toproto_operation(operation))
        assert type(restored) is type(operation)
        assert restored == operation


def test_circuit_with_loss_operations_proto_roundtrip():
    circuit = mc.Circuit()
    circuit.push(mc.LossErr(0.5), 0)
    circuit.push(mc.QubitLoss(), 1)
    circuit.push(mc.CheckLoss(), 0, 0)
    circuit.push(mc.MeasureCheckLoss(), 1, 1, 2)
    circuit.push(mc.QubitReload(), 1)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "loss_circuit.pb")
        circuit.saveproto(path)
        restored = mc.Circuit.loadproto(path)

    assert restored == circuit


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
