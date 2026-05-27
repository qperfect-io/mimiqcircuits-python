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

import pytest
import mimiqcircuits as mc
from mimiqcircuits.proto.circuitproto import toproto_circuit, fromproto_circuit


def test_default_traits_reject_aliasing():
    # Default Operation does not allow bit/zvar aliasing.
    assert mc.ParityCheck.allow_bit_aliasing() is False
    assert mc.GateX.allow_zvar_aliasing() is False

    with pytest.raises(ValueError):
        mc.Instruction(mc.ParityCheck(), (), (1, 1, 2), ())


def test_qubit_uniqueness_never_relaxed():
    # No-cloning: qubit aliasing is always rejected.
    with pytest.raises(ValueError):
        mc.Instruction(mc.GateCX(), (1, 1), ())


@pytest.mark.parametrize("op_factory", [
    lambda: mc.And(3),
    lambda: mc.Or(3),
    lambda: mc.Xor(3),
])
def test_classical_logic_ops_allow_bit_aliasing(op_factory):
    op = op_factory()
    assert type(op).allow_bit_aliasing() is True

    c = mc.Circuit()
    c.push(op, 0, 0, 1)
    assert len(c) == 1
    assert c[0].bits == (0, 0, 1)


def test_ifstatement_allows_bit_aliasing():
    ifs = mc.IfStatement(mc.Not(), mc.BitString("1"))
    assert type(ifs).allow_bit_aliasing() is True

    c = mc.Circuit()
    c.push(ifs, 0, 0)
    assert len(c) == 1


@pytest.mark.parametrize("op_factory", [
    lambda: mc.Add(2),
    lambda: mc.Multiply(2),
])
def test_zvar_ops_allow_zvar_aliasing(op_factory):
    op = op_factory()
    assert type(op).allow_zvar_aliasing() is True

    c = mc.Circuit()
    c.push(op, 0, 0)
    assert len(c) == 1


def test_pow_allows_zvar_aliasing():
    op = mc.Pow(2)
    assert type(op).allow_zvar_aliasing() is True


def test_wrappers_do_not_opt_in():
    # Block, Repeat must keep outer-target uniqueness.
    assert mc.Block.allow_bit_aliasing() is False
    assert mc.Repeat.allow_bit_aliasing() is False


def test_aliased_circuit_protobuf_roundtrip():
    c = mc.Circuit()
    c.push(mc.And(3), 0, 0, 1)
    c.push(mc.Add(2), 0, 0)

    proto = toproto_circuit(c)
    restored = fromproto_circuit(proto)
    assert restored == c
