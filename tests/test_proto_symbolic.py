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

import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.proto import circuit_pb2
from mimiqcircuits.proto.circuitproto import fromproto_arg, toproto_arg

x = se.Symbol("x")
P = se.pi


def _argbytes(e):
    return toproto_arg(e).SerializeToString(deterministic=True).hex()


def _argfromhex(h):
    arg = circuit_pb2.Arg()
    arg.ParseFromString(bytes.fromhex(h))
    return fromproto_arg(arg)


@pytest.mark.parametrize(
    "e",
    [
        P / 4, -P / 4, 2 * P / 3, P / 2, 3 * P / 2, 7 * P / 8, P, P**2,
        x * P / 4, x / 2 + P / 3, se.E / 2, 0.5 * P, se.Rational(1, 4), 0.5, 3,
        x, -x,
    ],
)
def test_roundtrip(e, tmp_path):
    c = mc.Circuit()
    c.push(mc.GateRX(e), 0)
    f = tmp_path / "c.pb"
    c.saveproto(str(f))
    c2 = mc.Circuit.loadproto(str(f))
    assert c2 == c
    assert c2[0].operation.theta == e


def test_plain_numbers_stay_plain():
    assert toproto_arg(0.5).HasField("argvalue_value")
    assert toproto_arg(3).HasField("argvalue_value")


# The same vectors are pinned in the Julia test suite (test/proto/symbolic.jl):
# both languages must write these bytes and read them back to the expression.
VECTORS = [
    (P / 4, "1a0c08021202200012040a020804"),
    (-P / 4, "1a21080112191a170802120d0a0b08ffffffffffffffffff0112040a02080412022000"),
    (2 * P / 3, "1a18080112101a0e080212040a02080212040a02080312022000"),
    (7 * P / 8, "1a18080112101a0e080212040a02080712040a02080812022000"),
    (x * P / 4, "1a1f0801120512030a017812101a0e080212040a02080112040a02080412022000"),
    (
        x / 2 + P / 3,
        "1a23120e1a0c08021202200012040a02080312111a0f0802120512030a017812040a020802",
    ),
    (se.Rational(1, 4), "1a0e080212040a02080112040a020804"),
    (0.5, "0a0911000000000000e03f"),
    (3, "0a020803"),
]


@pytest.mark.parametrize("e,h", VECTORS)
def test_byte_vectors(e, h):
    assert _argbytes(e) == h
    assert _argfromhex(h) == e


def test_older_payloads():
    # `MUL(0.25, PI)`, written by earlier Python encoders.
    v = _argfromhex("1a130801120b0a0911000000000000d03f12022000")
    assert float(v) == pytest.approx(float(P / 4))
    # A plain double, written by earlier Julia encoders.
    assert _argfromhex("0a0911182d4454fb21e93f") == pytest.approx(float(P / 4))
