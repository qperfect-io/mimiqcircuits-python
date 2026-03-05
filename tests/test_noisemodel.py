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
    rule = OperationInstanceNoise(GateR(1, 1), AmplitudeDamping(0.01))
    assert rule.matches(Instruction(GateR(1, 1), (1,)))
    assert not rule.matches(Instruction(GateRX(3.14), (1,)))

    assert rule.apply_rule(Instruction(GateR(1, 1), (1,))) == Instruction(
        AmplitudeDamping(0.01), (1,)
    )

    import symengine as se

    theta = se.Symbol("theta")
    rule_sym = OperationInstanceNoise(GateRX(theta), AmplitudeDamping(0.01 * theta))
    assert rule_sym.apply_rule(Instruction(GateRX(3.14), (1,))) == Instruction(
        AmplitudeDamping(0.01 * 3.14), (1,)
    )


def test_exact_gate_instance_qubit_noise():
    rule = ExactOperationInstanceQubitNoise(GateCX(), (1, 2), Depolarizing(2, 0.01))
    assert rule.matches(Instruction(GateCX(), (1, 2)))
    assert not rule.matches(Instruction(GateCX(), (2, 1)))


def test_set_gate_instance_qubit_noise():
    rule = SetOperationInstanceQubitNoise(GateCX(), [1, 2, 3], Depolarizing(2, 0.01))

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
            OperationInstanceNoise(GateH(), AmplitudeDamping(0.01)),
            OperationInstanceNoise(GateCX(), Depolarizing(2, 0.01)),
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
        [OperationInstanceNoise(GateH(), AmplitudeDamping(0.01), before=True)]
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
            OperationInstanceNoise(GateCX(), Depolarizing(2, 0.01)),
            ExactOperationInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.02)),
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
            ExactOperationInstanceQubitNoise(GateCX(), [1, 2], Depolarizing(2, 0.01)),
            ExactOperationInstanceQubitNoise(GateCX(), [2, 1], Depolarizing(2, 0.02)),
            OperationInstanceNoise(GateH(), AmplitudeDamping(0.001)),
            OperationInstanceNoise(GateRX(3.14 / 2), AmplitudeDamping(0.002)),
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

    nm.add_operation_noise(GateH(), AmplitudeDamping(0.01))
    nm.add_operation_noise(GateCX(), Depolarizing(2, 0.01), qubits=[1, 2, 3])
    nm.add_operation_noise(GateCX(), Depolarizing(2, 0.02), qubits=[1, 2], exact=True)

    nm.add_idle_noise(AmplitudeDamping(0.1))
    nm.add_idle_noise(AmplitudeDamping(0.2), qubits=[1])
    assert len(nm.rules) == 8


def test_add_operation_noise_replace():
    c = Circuit()
    c.push(GateH(), 0)

    model = NoiseModel()
    model.add_operation_noise(GateH(), AmplitudeDamping(0.01), replace=True)

    noisy = apply_noise_model(c, model)
    assert len(noisy.instructions) == 1
    assert isinstance(noisy.instructions[0].get_operation(), AmplitudeDamping)


def test_apply_noise_model_recurses_into_block():
    inner = Circuit()
    inner.push(GateH(), 0)
    block = Block(inner)

    c = Circuit()
    c.push(block, 0)

    model = NoiseModel([OperationInstanceNoise(GateH(), AmplitudeDamping(0.01))])
    noisy = apply_noise_model(c, model)

    assert len(noisy.instructions) == 1
    rewritten_block = noisy.instructions[0].get_operation()
    assert isinstance(rewritten_block, Block)
    assert len(rewritten_block.instructions) == 2
    assert isinstance(rewritten_block.instructions[0].get_operation(), GateH)
    assert isinstance(rewritten_block.instructions[1].get_operation(), AmplitudeDamping)


def test_apply_noise_model_recurses_into_ifstatement():
    c = Circuit()
    c.push(IfStatement(GateH(), BitString("1")), 0, 0)

    model = NoiseModel([OperationInstanceNoise(GateH(), AmplitudeDamping(0.01))])
    noisy = apply_noise_model(c, model)

    assert len(noisy.instructions) == 1
    wrapped = noisy.instructions[0].get_operation()
    assert isinstance(wrapped, IfStatement)
    inner = wrapped.get_operation()
    assert isinstance(inner, Block)
    assert len(inner.instructions) == 2
    assert isinstance(inner.instructions[0].get_operation(), GateH)
    assert isinstance(inner.instructions[1].get_operation(), AmplitudeDamping)


def test_apply_noise_model_recurses_into_gatecall():
    inner = Circuit()
    inner.push(GateH(), 0)
    decl = GateDecl("Inner", (), inner)

    c = Circuit()
    c.push(GateCall(decl, ()), 0)

    model = NoiseModel([OperationInstanceNoise(GateH(), AmplitudeDamping(0.01))])
    noisy = apply_noise_model(c, model)

    assert len(noisy.instructions) == 1
    rewritten = noisy.instructions[0].get_operation()
    assert isinstance(rewritten, Block)
    assert len(rewritten.instructions) == 2
    assert isinstance(rewritten.instructions[0].get_operation(), GateH)
    assert isinstance(rewritten.instructions[1].get_operation(), AmplitudeDamping)


def test_set_qubit_readout_noise_two_qubit_measurement():
    rule = SetQubitReadoutNoise([1, 3, 5], ReadoutErr(0.01, 0.02))

    inst_ok = Instruction(MeasureZZ(), (1, 3), (7,))
    inst_bad = Instruction(MeasureZZ(), (2, 3), (7,))

    assert rule.matches(inst_ok)
    assert not rule.matches(inst_bad)
    assert rule.apply_rule(inst_ok) == Instruction(ReadoutErr(0.01, 0.02), (), (7,))


def test_operation_instance_noise_construction_validation():
    with pytest.raises(ValueError):
        OperationInstanceNoise(GateH(), Depolarizing(2, 0.01))

    with pytest.raises(ValueError):
        OperationInstanceNoise(AmplitudeDamping(0.1), PauliX(0.01))

    with pytest.raises(ValueError):
        OperationInstanceNoise(
            GateH(), AmplitudeDamping(0.01), before=True, replace=True
        )

    assert isinstance(
        OperationInstanceNoise(Repeat(2, GateH()), AmplitudeDamping(0.01)),
        OperationInstanceNoise,
    )
    assert isinstance(
        OperationInstanceNoise(
            IfStatement(GateH(), BitString("1")), AmplitudeDamping(0.01)
        ),
        OperationInstanceNoise,
    )


def test_operation_instance_noise_symbolic_and_constant_expressions():
    import symengine as se

    p = se.Symbol("p")
    q = se.Symbol("q")

    with pytest.raises(ValueError):
        OperationInstanceNoise(GateRX(p + 3.0), AmplitudeDamping(p))

    # Constant symbolic numeric expression is allowed.
    rule_const = OperationInstanceNoise(GateRX(se.pi / 2), AmplitudeDamping(0.01))
    assert rule_const.matches(Instruction(GateRX(se.pi / 2), (1,)))

    rule_sym = OperationInstanceNoise(GateRX(p), AmplitudeDamping(0.01 * p / se.pi))
    assert rule_sym.matches(Instruction(GateRX(0.5), (1,)))
    out = rule_sym.apply_rule(Instruction(GateRX(0.5), (1,)))
    assert isinstance(out.get_operation(), AmplitudeDamping)
    assert float(out.get_operation().getparams()[0]) == pytest.approx(
        0.01 * 0.5 / float(se.pi)
    )
    assert out.get_qubits() == (1,)

    # Symbolic instruction cannot be used as substitution source.
    with pytest.raises(ValueError):
        rule_sym.apply_rule(Instruction(GateRX(q), (1,)))


def test_exact_operation_instance_qubit_noise_validation():
    with pytest.raises(ValueError):
        ExactOperationInstanceQubitNoise(GateH(), [], AmplitudeDamping(0.01))

    with pytest.raises(ValueError):
        ExactOperationInstanceQubitNoise(GateCX(), [1], Depolarizing(2, 0.01))

    with pytest.raises(ValueError):
        ExactOperationInstanceQubitNoise(GateCX(), [1, 1], Depolarizing(2, 0.01))

    with pytest.raises(ValueError):
        ExactOperationInstanceQubitNoise(
            GateH(), [1], AmplitudeDamping(0.01), before=True, replace=True
        )


def test_set_operation_instance_qubit_noise_validation():
    with pytest.raises(ValueError):
        SetOperationInstanceQubitNoise(GateH(), [], AmplitudeDamping(0.01))

    with pytest.raises(ValueError):
        SetOperationInstanceQubitNoise(GateCX(), [1], Depolarizing(2, 0.01))

    with pytest.raises(ValueError):
        SetOperationInstanceQubitNoise(GateH(), [1, 1], AmplitudeDamping(0.01))

    with pytest.raises(ValueError):
        SetOperationInstanceQubitNoise(
            GateH(), [1], AmplitudeDamping(0.01), before=True, replace=True
        )


def test_noisemodel_rules_sorted_on_construction_and_add():
    r1 = OperationInstanceNoise(GateH(), AmplitudeDamping(0.01))
    r2 = IdleNoise(AmplitudeDamping(0.0001))
    r3 = ExactOperationInstanceQubitNoise(GateH(), [1], AmplitudeDamping(0.02))

    model = NoiseModel([r1, r2, r3], name="sorted")
    priorities = [r.priority() for r in model.rules]
    assert priorities == sorted(priorities)

    model2 = NoiseModel()
    model2.add_rule(r2).add_rule(r1).add_rule(r3)
    priorities2 = [r.priority() for r in model2.rules]
    assert priorities2 == sorted(priorities2)


def test_apply_noise_model_recurses_into_parallel_and_repeat():
    model = NoiseModel([OperationInstanceNoise(GateH(), AmplitudeDamping(0.01))])

    c_parallel = Circuit().push(Parallel(2, GateH()), 0, 1)
    noisy_parallel = apply_noise_model(c_parallel, model)
    op_parallel = noisy_parallel.instructions[0].get_operation()
    assert isinstance(op_parallel, Block)
    assert len(op_parallel.instructions) == 4

    c_repeat = Circuit().push(Repeat(2, GateH()), 0)
    noisy_repeat = apply_noise_model(c_repeat, model)
    op_repeat = noisy_repeat.instructions[0].get_operation()
    assert isinstance(op_repeat, Block)
    assert len(op_repeat.instructions) == 4
