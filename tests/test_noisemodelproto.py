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
            GateInstanceNoise(GateH(), PauliX(0.3)),
            ExactGateInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.02)),
            SetGateInstanceQubitNoise(GateX(), [1], PauliY(0.1)),
            # Gate instance rules
            GateInstanceNoise(GateRX(3.14 / 2), PauliX(0.1)),
            ExactGateInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.03)),
            SetGateInstanceQubitNoise(GateRZ(3.14 / 2), [1], PauliZ(0.05)),
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
