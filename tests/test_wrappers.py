#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""The wrapper rule, as in MimiqCircuitsBase.jl: ``iswrapper()`` is true
exactly when ``get_operation()`` returns the wrapped operation. Code that
looks through wrappers relies on it, so a wrapper that breaks it either
recurses forever or reads an attribute it does not have."""

import pytest

import mimiqcircuits as mc

INNER = mc.GateRX(0.3)


@pytest.mark.parametrize(
    "wrapper",
    [
        mc.Control(2, INNER),
        mc.Inverse(INNER),
        mc.Power(INNER, 2),
        mc.Parallel(2, INNER),
        mc.Repeat(3, INNER),
        mc.IfStatement(INNER, mc.BitString("1")),
        mc.RescaledGate(INNER, 0.5),
    ],
    ids=lambda w: type(w).__name__,
)
def test_a_wrapper_returns_what_it_wraps(wrapper):
    assert wrapper.iswrapper()
    assert wrapper.get_operation() is INNER


def test_an_expectation_value_wraps_its_observable():
    observable = mc.PauliString("ZZ")
    ev = mc.ExpectationValue(observable)
    assert ev.iswrapper()
    assert ev.get_operation() is observable


# Named wrappers: aliases built on Power, Inverse and Control.
@pytest.mark.parametrize(
    "gate", [mc.GateT(), mc.GateS(), mc.GateSDG(), mc.GateSX(), mc.GateCX()],
    ids=lambda g: type(g).__name__,
)
def test_a_named_wrapper_does_not_return_itself(gate):
    assert gate.iswrapper()
    assert gate.get_operation() is not gate


@pytest.mark.parametrize(
    "op",
    [mc.RPauli(mc.PauliString("XY"), 0.3), mc.GateRNZ(3, 0.3), mc.GateH()],
    ids=lambda o: type(o).__name__,
)
def test_an_operation_wrapping_nothing_is_not_a_wrapper(op):
    assert not op.iswrapper()
