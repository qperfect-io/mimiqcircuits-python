import pytest
import symengine as se

from mimiqcircuits import *


class DummyRng:
    def __init__(self, *values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def _has_loss_ops(circuit):
    return any(
        isinstance(inst.get_operation(), (Loss, Reload, Check, MeasureCheck))
        for inst in circuit
    )


def test_sample_mixedunitaries_lossy():
    # Branch 2 (probability 1) loses local qubit 1; mapped to qubit 5.
    much = MixedUnitary([0.0, 1.0], [GateID(), GateX()], lossy=[[], [1]])
    c = Circuit()
    c.push(much, 5)
    s = c.sample_mixedunitaries(rng=DummyRng(0.5))
    assert isinstance(s.instructions[0].get_operation(), GateX)
    assert s.instructions[0].qubits == (5,)
    assert s.instructions[1].get_operation() == Loss(1.0)
    assert s.instructions[1].qubits == (5,)

    # A lossy identity branch still emits the loss with ids=False.
    much0 = MixedUnitary([1.0, 0.0], [GateID(), GateX()], lossy=[[1], []])
    c0 = Circuit()
    c0.push(much0, 3)
    s0 = c0.sample_mixedunitaries(rng=DummyRng(0.5))
    assert len(s0.instructions) == 1
    assert s0.instructions[0].get_operation() == Loss(1.0)
    assert s0.instructions[0].qubits == (3,)

    # Two-qubit branch: local lossy qubit 2 maps to the second target.
    much2 = MixedUnitary([1.0], [GateCX()], lossy=[[2]])
    c2 = Circuit()
    c2.push(much2, 3, 5)
    s2 = c2.sample_mixedunitaries(rng=DummyRng(0.5))
    assert isinstance(s2.instructions[0].get_operation(), GateCX)
    assert s2.instructions[1].get_operation() == Loss(1.0)
    assert s2.instructions[1].qubits == (5,)


def test_loss_operations_basic():
    with pytest.raises(ValueError):
        Loss(1.1)

    assert Loss() == Loss(1.0)

    theta = se.Symbol("theta")
    assert Loss(theta).evaluate({theta: 0.25}) == Loss(0.25)

    assert str(Loss(0.1)) == "Loss(0.1)"
    assert str(Reload()) == "Reload"
    assert str(Check()) == "Check"
    assert str(MeasureCheck()) == "MeasureCheck"
    assert MeasureCheck().num_bits == 2


def test_deprecated_aliases_map_to_new_ops():
    with pytest.warns(DeprecationWarning):
        assert LossErr(0.2) == Loss(0.2)
    with pytest.warns(DeprecationWarning):
        assert QubitLoss() == Loss(1.0)
    with pytest.warns(DeprecationWarning):
        assert isinstance(QubitReload(), Reload)
    with pytest.warns(DeprecationWarning):
        assert isinstance(CheckLoss(), Check)
    with pytest.warns(DeprecationWarning):
        assert isinstance(MeasureCheckLoss(), MeasureCheck)


def test_sample_losses_resolves_randomness_only():
    # a certain loss is kept as Loss(1.0); everything else passes through
    c = Circuit()
    c.push(Loss(), 0)
    c.push(GateH(), 0)
    c.push(Check(), 0, 0)
    sampled = sample_losses(c)

    expected = Circuit()
    expected.push(Loss(1.0), 0)
    expected.push(GateH(), 0)
    expected.push(Check(), 0, 0)
    assert sampled == expected

    # an impossible loss is dropped
    zero = sample_losses(Circuit().push(Loss(0.0), 0))
    assert len(zero) == 0


def test_sample_losses_symbolic_requires_evaluation():
    p = se.Symbol("p")
    c = Circuit()
    c.push(Loss(p), 0)

    with pytest.raises(ValueError, match="Use evaluate\\(\\)"):
        sample_losses(c)

    sampled = sample_losses(c.evaluate({p: 0.5}), rng=DummyRng(0.4))
    expected = Circuit()
    expected.push(Loss(1.0), 0)
    assert sampled == expected


def test_lower_losses_bookkeeping():
    c = Circuit()
    c.push(Loss(), 0)
    c.push(Reload(), 0)
    c.push(Check(), 0, 0)
    c.push(MeasureCheck(), 1, 1, 2)
    lowered = lower_losses(c)

    assert isinstance(lowered[0].get_operation(), Lost)
    assert isinstance(lowered[1].get_operation(), Reset)
    assert isinstance(lowered[2].get_operation(), Reloaded)
    # q0 present again after reload -> Check writes 1
    assert isinstance(lowered[3].get_operation(), SetBit1)
    # q1 never lost -> MeasureCheck measures and marks present
    assert isinstance(lowered[4].get_operation(), Measure)
    assert isinstance(lowered[5].get_operation(), SetBit1)
    assert not _has_loss_ops(lowered)


def test_lower_losses_measurement_on_lost_reads_zero():
    c = Circuit()
    c.push(Loss(), 0)
    c.push(Measure(), 0, 0)
    lowered = lower_losses(c)
    assert isinstance(lowered[-1].get_operation(), SetBit0)


@pytest.mark.parametrize("meas", [MeasureX, MeasureY, MeasureReset])
def test_lower_losses_single_qubit_measurement_on_lost_reads_zero(meas):
    # any single-qubit measurement on a lost qubit must still write its bit
    c = Circuit()
    c.push(Loss(), 0)
    c.push(meas(), 0, 0)
    lowered = lower_losses(c)
    assert isinstance(lowered[-1].get_operation(), SetBit0)


def test_resolve_losses_replace_rule_filters_lost_qubits():
    c = Circuit()
    c.push(Loss(), 1)
    c.push(GateCX(), 0, 1)
    model = LossModel().add_replace(GateCX(), Depolarizing1(0.2))

    expected = Circuit()
    expected.push(Lost(), 1)
    expected.push(Depolarizing1(0.2), 0)

    assert resolve_losses(c, lossmodel=model) == expected
    assert c.resolve_losses(lossmodel=model) == expected
    assert c.resolve_losses(model) == expected
    assert resolve_losses(c, model) == expected


def test_resolve_losses_drop_priority_wins_over_replace():
    c = Circuit()
    c.push(Loss(), 1)
    c.push(GateSWAP(), 0, 1)
    model = LossModel([ReplaceRule(GateSWAP(), GateX()), DropRule(GateSWAP())])

    expected = Circuit()
    expected.push(Lost(), 1)

    assert resolve_losses(c, lossmodel=model) == expected
    assert isinstance(model.rules[0], DropRule)


def test_resolve_losses_custom_rule_receives_lost_map():
    c = Circuit()
    c.push(Loss(), 1)
    c.push(GateCX(), 0, 1)
    model = LossModel(
        [
            CustomRule(
                lambda inst: isinstance(inst.get_operation(), GateCX),
                lambda inst, lost: [
                    Instruction(GateZ(), (q,))
                    for q in inst.get_qubits()
                    if not lost.get(q, False)
                ],
            )
        ]
    )

    expected = Circuit()
    expected.push(Lost(), 1)
    expected.push(GateZ(), 0)
    assert resolve_losses(c, lossmodel=model) == expected


def test_decorate_rule_instruction_mapping_filters_lost_targets():
    c = Circuit()
    c.push(Loss(), 1)
    c.push(GateCX(), 0, 1)
    model = LossModel(
        [
            DecorateRule(
                GateCX(),
                [Instruction(GateX(), (0,)), Instruction(GateZ(), (1,))],
                before=True,
            )
        ]
    )

    expected = Circuit()
    expected.push(Lost(), 1)
    expected.push(GateX(), 0)
    assert resolve_losses(c, lossmodel=model) == expected
