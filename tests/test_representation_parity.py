#
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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
# Representation choices shared with MimiqCircuitsBase.jl, which carries the
# same cases in test/representation_parity.jl.

from fractions import Fraction
import math

import numpy as np
import pytest
import symengine as se

from mimiqcircuits import *
from mimiqcircuits.proto import circuit_pb2
from mimiqcircuits.proto.circuitproto import fromproto_arg, toproto_arg


def test_getparams_of_depolarizing_is_the_probability():
    assert Depolarizing(2, 0.1).getparams() == [0.1]
    assert Depolarizing(2, 0.1).parnames == ("p",)


def test_getparams_of_repeat_forwards_to_the_wrapped_operation():
    assert Repeat(3, GateRX(0.2)).getparams() == [0.2]


def test_getparams_of_custom_gates_are_the_entries_in_column_major_order():
    g = GateCustom(np.array([[1, 0], [0, 1j]]) @ np.array([[0, 1], [1, 0]]))
    assert [complex(p) for p in g.getparams()] == [0, 1j, 1, 0]
    assert g.parnames == (1, 2, 3, 4)
    d = GateCustomDiagonal([1, 1j])
    assert [complex(p) for p in d.getparams()] == [1, 1j]
    assert d.parnames == (1, 2)


def test_custom_gate_with_symbolic_entry_is_symbolic():
    x = se.Symbol("x")
    assert GateCustomDiagonal([1, se.exp(se.I * x)]).is_symbolic()


@pytest.mark.parametrize(
    "op",
    [Projector0(), SigmaMinus(), DiagonalOp(1, 0.5), RescaledGate(GateX(), 0.5)],
)
def test_operators_that_are_not_gates_cannot_be_pushed(op):
    with pytest.raises(ValueError):
        Circuit().push(op, 0)


def test_float_exponents_that_are_exact_fractions_become_fractions():
    assert isinstance(Power(GateX(), 0.5), GateSX)
    assert Power(GateCX(), 0.5).exponent == Fraction(1, 2)
    assert isinstance(Power(GateCX(), 0.5).exponent, Fraction)
    assert Power(GateX(), 2.0).exponent == 2
    assert isinstance(Power(GateX(), 2.0).exponent, int)
    # Not a fraction with a small denominator: kept as it is.
    assert Power(GateX(), math.sqrt(2)).exponent == math.sqrt(2)
    assert Power(GateCX(), 1 / 3).exponent == Fraction(1, 3)
    assert Power(GateCX(), 0.001).exponent == Fraction(1, 1000)


def test_fraction_exponent_scales_symbolic_parameters_exactly():
    x = se.Symbol("x")
    assert GateRX(x).power(0.5) == GateRX(x / 2)
    assert GateP(se.pi).power(0.5) == GateP(se.pi / 2)


def test_inverse_of_gateu_does_not_produce_a_negative_zero():
    g = GateU(0.1, 0.0, 0.3).inverse()
    assert math.copysign(1.0, g.lmbda) == 1.0
    assert math.copysign(1.0, g.gamma) == 1.0


def _mtype(expr):
    return toproto_arg(expr).argfunction_value.mtype


def test_symbolic_quotients_are_serialized_as_div():
    x, y = se.symbols("x y")
    DIV = circuit_pb2.FunctionType.DIV
    for expr in [1 / x, 2 / x, x / y, 1 / (x + y), se.pi / x, x**-0.5, 1 / x**2]:
        assert _mtype(expr) == DIV
        assert se.expand(fromproto_arg(toproto_arg(expr)) - expr) == 0
    # A numeric denominator keeps the product form, as in Julia.
    assert _mtype(3 * se.pi / 4) == circuit_pb2.FunctionType.MUL
