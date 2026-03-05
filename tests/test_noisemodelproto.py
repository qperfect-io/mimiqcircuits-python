# ------------------------------------------------------------
#  Protobuf Roundtrip and Integration Tests
# ------------------------------------------------------------
import tempfile
import os
from mimiqcircuits import *


def make_composite_noisemodel():
    """Helper to build a NoiseModel containing every rule type."""
    nm = NoiseModel(
        [
            # Readout rules
            GlobalReadoutNoise(ReadoutErr(0.01, 0.02)),
            ExactQubitReadoutNoise([1, 2], ReadoutErr(0.05, 0.1)),
            SetQubitReadoutNoise([1, 3], ReadoutErr(0.1, 0.05)),
            OperationInstanceNoise(GateH(), PauliX(0.3)),
            ExactOperationInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.02)),
            SetOperationInstanceQubitNoise(GateX(), [1], PauliY(0.1)),
            # Gate instance rules
            OperationInstanceNoise(GateRX(3.14 / 2), PauliX(0.1)),
            ExactOperationInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.03)),
            SetOperationInstanceQubitNoise(GateRZ(3.14 / 2), [1], PauliZ(0.05)),
            IdleNoise(PauliZ(0.1)),
            SetIdleQubitNoise(PauliZ(0.2), [1, 2]),
            # --- Custom noise rule ---
            # CustomNoiseRule(
            #     matcher=lambda inst: False,
            #     generator=lambda inst: Instruction(PauliX(0.02), inst.get_qubits()),
            #     priority_val=80,
            #     before=False,
            # ),
        ],
        name="Composite NoiseModel",
    )
    return nm


def test_noisemodel_proto_roundtrip():
    nm = make_composite_noisemodel()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nm.pb")
        nm.saveproto(path)
        loaded = nm.loadproto(path)

        assert isinstance(loaded, NoiseModel)
        assert loaded.name == "Composite NoiseModel"
        assert len(loaded.rules) == len(nm.rules)
        # Type check for each rule
        for orig, rest in zip(nm.rules, loaded.rules):
            assert type(orig) is type(rest)


def test_noisemodel_proto_roundtrip_operation_replace():
    nm = NoiseModel(
        [OperationInstanceNoise(GateH(), AmplitudeDamping(0.01), replace=True)],
        name="Replace Model",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nm_replace.pb")
        nm.saveproto(path)
        loaded = nm.loadproto(path)

        assert isinstance(loaded, NoiseModel)
        assert loaded.name == "Replace Model"
        assert len(loaded.rules) == 1
        assert isinstance(loaded.rules[0], OperationInstanceNoise)
        assert loaded.rules[0].replaces()


def test_noisemodel_proto_roundtrip_exact_set_operation_replace():
    nm = NoiseModel(
        [
            ExactOperationInstanceQubitNoise(
                GateH(), [1], AmplitudeDamping(0.01), replace=True
            ),
            SetOperationInstanceQubitNoise(
                GateH(), [1, 3], AmplitudeDamping(0.02), replace=True
            ),
        ],
        name="Replace Exact/Set",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nm_replace_exact_set.pb")
        nm.saveproto(path)
        loaded = nm.loadproto(path)

        assert loaded.name == "Replace Exact/Set"
        assert isinstance(loaded.rules[0], ExactOperationInstanceQubitNoise)
        assert isinstance(loaded.rules[1], SetOperationInstanceQubitNoise)
        assert loaded.rules[0].replaces()
        assert loaded.rules[1].replaces()


def test_noisemodel_proto_roundtrip_measurement_and_reset_operation_rules():
    nm = NoiseModel(
        [
            OperationInstanceNoise(Measure(), PauliX(0.2), before=True),
            ExactOperationInstanceQubitNoise(Measure(), [1], PauliZ(0.15), before=True),
            SetOperationInstanceQubitNoise(Measure(), [1, 3], PauliY(0.05), before=True),
            OperationInstanceNoise(Reset(), Depolarizing(1, 0.02)),
            ExactOperationInstanceQubitNoise(Reset(), [2], PauliX(0.1)),
            SetOperationInstanceQubitNoise(Reset(), [1, 2], PauliZ(0.2)),
        ],
        name="Measurement and Reset",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nm_measure_reset.pb")
        nm.saveproto(path)
        loaded = nm.loadproto(path)

        assert loaded.name == "Measurement and Reset"
        assert len(loaded.rules) == 6
        assert sum(isinstance(r, OperationInstanceNoise) for r in loaded.rules) == 2
        assert (
            sum(isinstance(r, ExactOperationInstanceQubitNoise) for r in loaded.rules)
            == 2
        )
        assert (
            sum(isinstance(r, SetOperationInstanceQubitNoise) for r in loaded.rules)
            == 2
        )

        measure_rules = [r for r in loaded.rules if isinstance(r.operation, Measure)]
        reset_rules = [r for r in loaded.rules if isinstance(r.operation, Reset)]
        assert len(measure_rules) == 3
        assert len(reset_rules) == 3
        assert all(r.before() for r in measure_rules)
        assert all(not r.before() for r in reset_rules)


def test_noisemodel_proto_roundtrip_idle_symbolic_relation():
    import symengine as se

    t = se.Symbol("t")
    nm = NoiseModel(
        [
            IdleNoise((t, AmplitudeDamping(0.001 * t))),
            SetIdleQubitNoise((t, AmplitudeDamping(0.002 * t)), [1, 2]),
        ],
        name="Idle Symbolic",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nm_idle_symbolic.pb")
        nm.saveproto(path)
        loaded = nm.loadproto(path)

        assert loaded.name == "Idle Symbolic"
        assert sum(isinstance(r, IdleNoise) for r in loaded.rules) == 1
        assert sum(isinstance(r, SetIdleQubitNoise) for r in loaded.rules) == 1
        idle_rule = next(r for r in loaded.rules if isinstance(r, IdleNoise))
        set_idle_rule = next(r for r in loaded.rules if isinstance(r, SetIdleQubitNoise))
        assert isinstance(idle_rule.relation, tuple)
        assert isinstance(set_idle_rule.relation, tuple)
        var0, _ = idle_rule.relation
        var1, _ = set_idle_rule.relation
        assert str(var0) == "t"
        assert str(var1) == "t"


# def test_noisemodel_proto_customnoise_safe():
#     nm = NoiseModel(
#         [
#             CustomNoiseRule(
#                 matcher=lambda inst: True,
#                 generator=lambda inst: Instruction(PauliX(0.01), inst.get_qubits()),
#                 priority_val=50,
#                 before=False,
#             )
#         ],
#         name="CustomModel",
#     )

#     with tempfile.TemporaryDirectory() as tmp:
#         path = os.path.join(tmp, "nm_custom.pb")
#         nm.saveproto(path)
#         loaded = NoiseModel.loadproto(path)

#         assert isinstance(loaded, NoiseModel)
#         assert loaded.name == "CustomModel"
#         assert len(loaded.rules) == 1
#         assert isinstance(loaded.rules[0], CustomNoiseRule)
