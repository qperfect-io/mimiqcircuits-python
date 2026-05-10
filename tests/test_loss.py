import pytest
import symengine as se

from mimiqcircuits import *


class DummyRng:
    def __init__(self, *values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def test_loss_operations_basic():
    with pytest.raises(ValueError):
        LossErr(1.1)

    theta = se.Symbol("theta")
    assert LossErr(theta).evaluate({theta: 0.25}) == LossErr(0.25)

    assert str(QubitLoss()) == "QubitLoss"
    assert str(QubitReload()) == "QubitReload"
    assert str(CheckLoss()) == "CL"
    assert str(MeasureCheckLoss()) == "MCL"
    assert MeasureCheckLoss().num_bits == 2


def test_sample_losses_replace_rule_filters_lost_qubits():
    c = Circuit()
    c.push(QubitLoss(), 1)
    c.push(GateCX(), 0, 1)

    model = LossModel().add_replace(GateCX(), Depolarizing1(0.2))
    sampled = sample_losses(c, lossmodel=model)

    expected = Circuit()
    expected.push(QubitLoss(), 1)
    expected.push(Depolarizing1(0.2), 0)

    assert sampled == expected
    assert c.sample_losses(lossmodel=model) == expected
    assert c.sample_losses(model) == expected
    assert sample_losses(c, model) == expected


def test_sample_losses_drop_priority_wins_over_replace():
    c = Circuit()
    c.push(QubitLoss(), 1)
    c.push(GateSWAP(), 0, 1)

    model = LossModel(
        [
            ReplaceRule(GateSWAP(), GateX()),
            DropRule(GateSWAP()),
        ]
    )

    sampled = sample_losses(c, lossmodel=model)

    expected = Circuit()
    expected.push(QubitLoss(), 1)

    assert sampled == expected
    assert isinstance(model.rules[0], DropRule)


def test_sample_losses_reload_and_checks_are_kept():
    c = Circuit()
    c.push(QubitLoss(), 0)
    c.push(GateH(), 0)
    c.push(CheckLoss(), 0, 0)
    c.push(MeasureCheckLoss(), 0, 1, 2)
    c.push(QubitReload(), 0)
    c.push(GateX(), 0)

    sampled = sample_losses(c)

    expected = Circuit()
    expected.push(QubitLoss(), 0)
    expected.push(CheckLoss(), 0, 0)
    expected.push(MeasureCheckLoss(), 0, 1, 2)
    expected.push(QubitReload(), 0)
    expected.push(GateX(), 0)

    assert sampled == expected


def test_sample_losses_symbolic_losserr_requires_evaluation():
    p = se.Symbol("p")
    c = Circuit()
    c.push(LossErr(p), 0)

    with pytest.raises(ValueError, match="Use evaluate\\(\\)"):
        sample_losses(c)

    evaluated = c.evaluate({p: 0.5})
    sampled = sample_losses(evaluated, rng=DummyRng(0.4))

    expected = Circuit()
    expected.push(QubitLoss(), 0)

    assert sampled == expected


def test_sample_losses_custom_rule_receives_lost_map():
    c = Circuit()
    c.push(QubitLoss(), 1)
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

    sampled = sample_losses(c, lossmodel=model)

    expected = Circuit()
    expected.push(QubitLoss(), 1)
    expected.push(GateZ(), 0)

    assert sampled == expected


def test_decorate_rule_instruction_mapping_filters_lost_targets():
    c = Circuit()
    c.push(QubitLoss(), 1)
    c.push(GateCX(), 0, 1)

    model = LossModel(
        [
            DecorateRule(
                GateCX(),
                [
                    Instruction(GateX(), (0,)),
                    Instruction(GateZ(), (1,)),
                ],
                before=True,
            )
        ]
    )

    sampled = sample_losses(c, lossmodel=model)

    expected = Circuit()
    expected.push(QubitLoss(), 1)
    expected.push(GateX(), 0)

    assert sampled == expected
