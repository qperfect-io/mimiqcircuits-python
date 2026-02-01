import pytest
from mimiqcircuits import *
from mimiqcircuits.noisemodel import NoiseModel


def test_abstract_noise_rule_interface():
    class DummyRule(AbstractNoiseRule):
        pass

    dummy = DummyRule()
    assert dummy.priority() == 100
    assert dummy.before() is False

    inst = Instruction(GateH(), (1,))
    with pytest.raises(NotImplementedError):
        dummy.matches(inst)
    with pytest.raises(NotImplementedError):
        dummy.apply_rule(inst)


def test_global_readout_noise():
    rule = GlobalReadoutNoise(ReadoutErr(0.01, 0.02))

    for q in range(1, 6):
        assert rule.matches(Instruction(Measure(), (q,), (1,)))

    assert rule.apply_rule(Instruction(Measure(), (1,), (1,))) == Instruction(
        ReadoutErr(0.01, 0.02), (), (1,)
    )


def test_gate_instance_noise():
    rule = GateInstanceNoise(GateR(1, 1), AmplitudeDamping(0.01))
    assert rule.matches(Instruction(GateR(1, 1), (1,)))
    assert not rule.matches(Instruction(GateRX(3.14), (1,)))

    assert rule.apply_rule(Instruction(GateR(1, 1), (1,))) == Instruction(
        AmplitudeDamping(0.01), (1,)
    )

    import symengine as se

    theta = se.Symbol("theta")
    rule_sym = GateInstanceNoise(GateRX(theta), AmplitudeDamping(0.01 * theta))
    assert rule_sym.apply_rule(Instruction(GateRX(3.14), (1,))) == Instruction(
        AmplitudeDamping(0.01 * 3.14), (1,)
    )


def test_exact_gate_instance_qubit_noise():
    rule = ExactGateInstanceQubitNoise(GateCX(), (1, 2), Depolarizing(2, 0.01))
    assert rule.matches(Instruction(GateCX(), (1, 2)))
    assert not rule.matches(Instruction(GateCX(), (2, 1)))


def test_set_gate_instance_qubit_noise():
    rule = SetGateInstanceQubitNoise(GateCX(), [1, 2, 3], Depolarizing(2, 0.01))

    assert rule.matches(Instruction(GateCX(), (1, 2)))
    assert rule.matches(Instruction(GateCX(), (2, 3)))
    assert not rule.matches(Instruction(GateCX(), (4, 5)))
    assert not rule.matches(Instruction(GateH(), (1,)))


def test_exact_qubit_readout_noise():
    rule = ExactQubitReadoutNoise([1], ReadoutErr(0.1, 0.2))
    assert rule.matches(Instruction(Measure(), (1,), (1,)))
    assert not rule.matches(Instruction(Measure(), (2,), (1,)))


def test_set_qubit_readout_noise():
    rule = SetQubitReadoutNoise([1, 2, 3], ReadoutErr(0.1, 0.2))

    assert rule.matches(Instruction(Measure(), (1,), (1,)))
    assert rule.matches(Instruction(Measure(), (2,), (1,)))
    assert not rule.matches(Instruction(Measure(), (4,), (1,)))


def test_idle_noise():
    import symengine as se

    t = se.Symbol("t")
    rule = IdleNoise((t, AmplitudeDamping(0.0001 * t)))

    assert rule.matches(Instruction(Delay(0.1), (1,)))
    assert not rule.matches(Instruction(GateH(), (1,)))

    out = rule.apply_rule(Instruction(Delay(0.1), (1,)))
    assert out == Instruction(AmplitudeDamping(0.0001 * 0.1), (1,))


def test_set_idle_noise():
    import symengine as se

    t = se.Symbol("t")
    rule = SetIdleQubitNoise((t, AmplitudeDamping(0.001 * t)), [1])

    assert rule.matches(Instruction(Delay(0.2), (1,)))
    assert not rule.matches(Instruction(Delay(0.2), (2,)))

    out = rule.apply_rule(Instruction(Delay(0.2), (1,)))
    assert out == Instruction(AmplitudeDamping(0.001 * 0.2), (1,))


def test_custom_noise_rule():
    rule = CustomNoiseRule(
        lambda inst: isinstance(inst.get_operation(), GateH),
        lambda inst: Instruction(AmplitudeDamping(0.01), inst.get_qubits()),
    )

    inst = Instruction(GateH(), (1,))
    assert rule.matches(inst)
    assert rule.apply_rule(inst) == Instruction(AmplitudeDamping(0.01), (1,))


def test_apply_noise_model_basic():
    c = Circuit()
    c.push(GateH(), 1)
    c.push(GateCX(), 1, 2)
    c.push(Measure(), 1, 1)

    model = NoiseModel(
        [
            GateInstanceNoise(GateH(), AmplitudeDamping(0.01)),
            GateInstanceNoise(GateCX(), Depolarizing(2, 0.01)),
            GlobalReadoutNoise(ReadoutErr(0.01, 0.02)),
        ]
    )

    noisy = apply_noise_model(c, model)

    assert len(noisy.instructions) == 6
    assert isinstance(noisy.instructions[1].get_operation(), AmplitudeDamping)
    assert isinstance(noisy.instructions[3].get_operation(), Depolarizing)
    assert isinstance(noisy.instructions[5].get_operation(), ReadoutErr)


def test_before_flag_behavior():
    c = Circuit()
    c.push(GateH(), 1)

    model = NoiseModel(
        [GateInstanceNoise(GateH(), AmplitudeDamping(0.01), before=True)]
    )

    noisy = apply_noise_model(c, model)
    ops = [inst.get_operation() for inst in noisy.instructions]

    assert isinstance(ops[0], AmplitudeDamping)
    assert isinstance(ops[1], GateH)


def test_priority_ordering():
    c = Circuit()
    c.push(GateCX(), 1, 2)

    model = NoiseModel(
        [
            GateInstanceNoise(GateCX(), Depolarizing(2, 0.01)),
            ExactGateInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.02)),
        ]
    )

    noisy = apply_noise_model(c, model)
    assert len(noisy.instructions) == 2
    assert noisy.instructions[1].get_operation().getparam("p") == 0.02


def test_complex_noise_model():
    c = Circuit()
    c.push(GateH(), 1)
    c.push(GateH(), 2)
    c.push(GateCX(), 1, 2)
    c.push(GateCX(), 2, 1)
    c.push(GateRX(3.14 / 2), 1)
    c.push(Measure(), 1, 1)

    model = NoiseModel(
        [
            ExactGateInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.01)),
            ExactGateInstanceQubitNoise(GateCX(), [2, 1], Depolarizing(2, 0.02)),
            GateInstanceNoise(GateH(), AmplitudeDamping(0.001)),
            GateInstanceNoise(GateRX(3.14 / 2), AmplitudeDamping(0.002)),
            ExactQubitReadoutNoise([1], ReadoutErr(0.01, 0.02)),
        ],
        name="Complex Model",
    )

    noisy = apply_noise_model(c, model)
    assert len(noisy.instructions) == 12

    ops = [inst.get_operation() for inst in noisy.instructions]
    assert isinstance(ops[1], AmplitudeDamping)
    assert isinstance(ops[3], AmplitudeDamping)
    assert isinstance(ops[5], Depolarizing)
    assert isinstance(ops[7], Depolarizing)
    assert isinstance(ops[9], AmplitudeDamping)
    assert isinstance(ops[11], ReadoutErr)


def test_add_helpers():
    nm = NoiseModel()

    nm.add_readout_noise(ReadoutErr(0.01, 0.02))
    nm.add_readout_noise(ReadoutErr(0.03, 0.04), qubits=[1, 3])
    nm.add_readout_noise(ReadoutErr(0.05, 0.06), qubits=[2, 1], exact=True)
    assert len(nm.rules) == 3

    nm.add_gate_noise(GateH(), AmplitudeDamping(0.01))
    nm.add_gate_noise(GateCX(), Depolarizing(2, 0.01), qubits=[1, 2, 3])
    nm.add_gate_noise(GateCX(), Depolarizing(2, 0.02), qubits=[1, 2], exact=True)

    nm.add_idle_noise(AmplitudeDamping(0.1))
    nm.add_idle_noise(AmplitudeDamping(0.2), qubits=[1])
    assert len(nm.rules) == 8
