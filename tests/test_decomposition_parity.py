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
"""Decompositions pinned to the same result as MimiqCircuitsBase.jl.

MimiqCircuitsBase.jl carries the same cases in test/decomposition_parity.jl
(with 1-based qubits). A change to one of these decompositions has to be
made in both packages.
"""

import pytest
from symengine import pi

import mimiqcircuits as mc
from mimiqcircuits.decomposition import CanonicalRewrite

# One step of the canonical rewrite: (operation, [(operation, qubits), ...]).
STEP_CASES = [
    (
        mc.GateCS(),
        [
            (mc.GateT(), (0,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateTDG(), (1,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateT(), (1,)),
        ],
    ),
    (
        mc.GateCSDG(),
        [
            (mc.GateTDG(), (0,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateT(), (1,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateTDG(), (1,)),
        ],
    ),
    (mc.GateCSX(), [(mc.GateH(), (1,)), (mc.GateCS(), (0, 1)), (mc.GateH(), (1,))]),
    (
        mc.GateCSXDG(),
        [(mc.GateH(), (1,)), (mc.GateCSDG(), (0, 1)), (mc.GateH(), (1,))],
    ),
    (
        mc.GateISWAPDG(),
        [
            (mc.GateSDG(), (0,)),
            (mc.GateSDG(), (1,)),
            (mc.GateH(), (0,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateCX(), (1, 0)),
            (mc.GateH(), (1,)),
        ],
    ),
    (
        mc.GateCRX(0.3),
        [
            (mc.GateS(), (1,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateU(-0.15, 0, 0), (1,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateU(0.15, -pi / 2, 0), (1,)),
        ],
    ),
    (
        mc.MeasureXX(),
        [
            (mc.GateCX(), (0, 1)),
            (mc.GateH(), (0,)),
            (mc.Measure(), (0,)),
            (mc.GateH(), (0,)),
            (mc.GateCX(), (0, 1)),
        ],
    ),
    (
        mc.MeasureYY(),
        [
            (mc.GateS(), (0,)),
            (mc.GateS(), (1,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateH(), (0,)),
            (mc.Measure(), (0,)),
            (mc.GateH(), (0,)),
            (mc.GateCX(), (0, 1)),
            (mc.GateSDG(), (0,)),
            (mc.GateSDG(), (1,)),
        ],
    ),
    (
        mc.GateCCP(0.3),
        [
            (mc.GateCP(0.15), (1, 2)),
            (mc.GateCX(), (0, 1)),
            (mc.GateCP(-0.15), (1, 2)),
            (mc.GateCX(), (0, 1)),
            (mc.GateCP(0.15), (0, 2)),
        ],
    ),
    (mc.GateU2(0.3, 0.7), [(mc.GateU(pi / 2, 0.3, 0.7), (0,))]),
    (
        mc.QFT(3),
        [
            (mc.GateH(), (2,)),
            (mc.GateCP(pi / 2), (1, 2)),
            (mc.GateH(), (1,)),
            (mc.GateCP(pi / 4), (0, 2)),
            (mc.GateCP(pi / 2), (0, 1)),
            (mc.GateH(), (0,)),
        ],
    ),
    (
        mc.PhaseGradient(3),
        [
            (mc.GateP(pi / 4), (2,)),
            (mc.GateP(pi / 2), (1,)),
            (mc.GateP(pi), (0,)),
        ],
    ),
    (mc.Inverse(mc.GateU(0.3, 0.7, 1.1)), [(mc.GateU(-0.3, -1.1, -0.7), (0,))]),
    (mc.Power(mc.GateCCX(), 0.5), [(mc.Control(2, mc.GateSX()), (0, 1, 2))]),
    (
        mc.Control(1, mc.Parallel(2, mc.GateH())),
        [(mc.GateCH(), (0, 1)), (mc.GateCH(), (0, 2))],
    ),
]


def _step(op):
    nq, nb, nz = op.num_qubits, op.num_bits, op.num_zvars
    return CanonicalRewrite().decompose_step(
        op, list(range(nq)), list(range(nb)), list(range(nz))
    )


@pytest.mark.parametrize("op, expected", STEP_CASES, ids=lambda x: str(x))
def test_canonical_step_matches_julia(op, expected):
    got = [(inst.operation, tuple(inst.qubits)) for inst in _step(op)]
    assert len(got) == len(expected)
    for (gop, gq), (eop, eq) in zip(got, expected):
        assert gq == eq
        assert gop == eop, f"{gop} != {eop}"


# Full decompositions into a basis: (operation, basis, [(operation, qubits, bits), ...]).
def _basis_cases():
    from mimiqcircuits.decomposition import QASMBasis

    return [
        (
            mc.MeasureX(),
            QASMBasis(),
            [(mc.GateH(), (0,), ()), (mc.Measure(), (0,), (0,)), (mc.GateH(), (0,), ())],
        ),
        (
            mc.GateCP(0.3),
            QASMBasis(),
            [
                (mc.GateU(0, 0, 0.15), (0,), ()),
                (mc.GateCX(), (0, 1), ()),
                (mc.GateU(0, 0, -0.15), (1,), ()),
                (mc.GateCX(), (0, 1), ()),
                (mc.GateU(0, 0, 0.15), (1,), ()),
            ],
        ),
        (
            mc.ResetX(),
            QASMBasis(),
            [(mc.GateH(), (0,), ()), (mc.Reset(), (0,), ()), (mc.GateH(), (0,), ())],
        ),
    ]


@pytest.mark.parametrize("case", range(len(_basis_cases())))
def test_basis_decomposition_matches_julia(case):
    from mimiqcircuits.decomposition import decompose

    op, basis, expected = _basis_cases()[case]
    c = mc.Circuit().push(op, *range(op.num_qubits), *range(op.num_bits))
    got = [(i.operation, tuple(i.qubits), tuple(i.bits)) for i in decompose(c, basis)]
    assert got == expected


def test_qasm_basis_rejects_noise():
    from mimiqcircuits.decomposition import DecompositionError, QASMBasis, decompose

    # OpenQASM 2.0 cannot express a noise channel.
    with pytest.raises(DecompositionError):
        decompose(mc.Circuit().push(mc.PauliX(0.1), 0), QASMBasis())
