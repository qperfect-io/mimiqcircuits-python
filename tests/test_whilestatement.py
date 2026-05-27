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
from mimiqcircuits.decomposition import DecompositionError


def test_whilestatement_constructor():
    gate = mc.GateX()
    bs = mc.BitString("1")
    ws = mc.WhileStatement(gate, bs)

    assert ws.num_qubits == gate.num_qubits
    assert ws.num_bits == gate.num_bits + len(bs)
    assert ws.num_zvars == 0
    assert ws.op is gate
    assert ws.bitstring == bs
    assert ws.allow_bit_aliasing() is True
    assert ws.allow_zvar_aliasing() is True


def test_whilestatement_equality():
    a = mc.WhileStatement(mc.GateX(), mc.BitString("1"))
    b = mc.WhileStatement(mc.GateX(), mc.BitString("1"))
    c = mc.WhileStatement(mc.GateX(), mc.BitString("0"))
    d = mc.WhileStatement(mc.GateY(), mc.BitString("1"))

    assert a == b
    assert a != c
    assert a != d


def test_whilestatement_inverse_power_control_unsupported():
    ws = mc.WhileStatement(mc.GateX(), mc.BitString("1"))
    with pytest.raises(NotImplementedError):
        ws.inverse()
    with pytest.raises(NotImplementedError):
        ws.power(2)
    with pytest.raises(NotImplementedError):
        ws.control(1)


def test_whilestatement_push_with_aliased_bits():
    ws = mc.WhileStatement(mc.Not(), mc.BitString("1"))
    assert ws.num_bits == 2

    c = mc.Circuit()
    c.push(ws, 0, 0)
    assert len(c) == 1
    inst = c[0]
    assert inst.bits == (0, 0)
    assert inst.qubits == ()


def test_whilestatement_protobuf_roundtrip():
    ws = mc.WhileStatement(mc.Not(), mc.BitString("1"))
    c = mc.Circuit()
    c.push(ws, 0, 0)

    proto = toproto_circuit(c)
    restored = fromproto_circuit(proto)
    assert restored == c
    assert isinstance(restored[0].operation, mc.WhileStatement)
    assert restored[0].operation.bitstring == mc.BitString("1")


def test_whilestatement_decompose_preserves_loop():
    ws = mc.WhileStatement(mc.Not(), mc.BitString("1"))
    c = mc.Circuit()
    c.push(ws, 0, 0)

    d = c.decompose()
    assert len(d) == 1
    assert isinstance(d[0].operation, mc.WhileStatement)


def test_whilestatement_decompose_nonterminal_body_raises():
    ws = mc.WhileStatement(mc.GateH(), mc.BitString("1"))
    c = mc.Circuit()
    c.push(ws, 0, 0)

    with pytest.raises(
        DecompositionError,
        match="Operation While is not supported by CanonicalBasis and cannot be decomposed.",
    ):
        c.decompose()
