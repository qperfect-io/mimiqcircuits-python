#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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
"""Tests for the decomposition framework."""

import math

import numpy as np
import pytest
from symengine import pi

import mimiqcircuits as mc
from mimiqcircuits.matrices import reorder_qubits_matrix
from mimiqcircuits import (
    Circuit,
    GateH,
    GateCX,
    GateCCX,
    GateU,
    GateRX,
    GateRY,
    GateRZ,
    GateT,
    GateS,
    GateX,
    GateY,
    GateZ,
    Measure,
    Reset,
    Barrier,
)
from mimiqcircuits.decomposition import (
    RewriteRule,
    DecompositionBasis,
    DecompositionError,
    CanonicalBasis,
    CliffordTBasis,
    FlattenedBasis,
    QASMBasis,
    StimBasis,
    RuleBasis,
    CanonicalRewrite,
    FlattenContainers,
    GridSynthRewrite,
    SolovayKitaevRewrite,
    ToffoliToCliffordTRewrite,
    ZYZRewrite,
    SpecialAngleRewrite,
    ToZRotationRewrite,
    decompose,
    decompose_step,
    eachdecomposed,
    DecomposeIterator,
)


class TestRewriteRules:
    """Tests for RewriteRule implementations."""

    def test_canonical_rewrite_matches_decomposable_operations(self):
        """CanonicalRewrite matches every operation with a canonical decomposition."""
        rule = CanonicalRewrite()
        assert rule.matches(GateH())
        assert rule.matches(GateCCX())
        assert not rule.matches(GateCX())
        assert not rule.matches(GateU(pi / 2, 0, pi))
        assert not rule.matches(mc.Measure())
        assert not rule.matches(mc.Reset())
        assert not rule.matches(mc.Delay(0.5))
        assert not rule.matches(mc.IfStatement(GateU(0.1, 0.2, 0.3), mc.BitString("1")))

    def test_canonical_rewrite_does_not_depend_on_a_basis(self):
        """A channel that CanonicalBasis keeps terminal is still matched."""
        rule = CanonicalRewrite()
        assert CanonicalBasis().isterminal(mc.ResetX())
        assert rule.matches(mc.ResetX())
        assert rule.matches(mc.ResetY())

    def test_resetx_expanded_by_canonical_rewrite(self):
        """A basis that does not keep ResetX expands it with the canonical rewrite."""
        c = mc.Circuit().push(mc.ResetX(), 0)
        d = decompose(c, RuleBasis(CanonicalRewrite()))
        assert [type(i.operation) for i in d] == [mc.GateU, mc.Reset, mc.GateU]

    def test_resetx_under_clifford_t(self):
        """CliffordTBasis reaches its own terminals on ResetX instead of raising."""
        c = mc.Circuit().push(mc.ResetX(), 0)
        d = decompose(c, CliffordTBasis())
        assert all(CliffordTBasis().isterminal(i.operation) for i in d)

    def test_canonical_rewrite_does_not_match_while(self):
        """WhileStatement is non-terminal but not canonically rewriteable."""
        rule = CanonicalRewrite()
        assert not rule.matches(mc.WhileStatement(GateH(), mc.BitString("1")))

    def test_canonical_rewrite_decomposes_h(self):
        """CanonicalRewrite should decompose H to U gate."""
        rule = CanonicalRewrite()
        circ = rule.decompose_step(GateH(), [0], [], [])
        assert len(circ) == 1
        assert isinstance(circ[0].operation, mc.GateU)

    @pytest.mark.parametrize(
        "op",
        [
            mc.Loss(),
            mc.Reload(),
            mc.Loss(0.5),
            mc.Check(),
            mc.MeasureCheck(),
        ],
    )
    def test_canonical_rewrite_does_not_match_loss_operations(self, op):
        """CanonicalRewrite should treat loss operations as terminal."""
        rule = CanonicalRewrite()
        assert not rule.matches(op)

    def test_zyz_rewrite_matches_gateu(self):
        """ZYZRewrite should match GateU (except identity)."""
        rule = ZYZRewrite()
        assert rule.matches(GateU(pi / 2, pi / 4, pi / 3))
        assert not rule.matches(GateU(0, 0, 0))  # identity

    def test_zyz_rewrite_matches_gaterx(self):
        """ZYZRewrite should match GateRX."""
        rule = ZYZRewrite()
        assert rule.matches(GateRX(pi / 2))

    def test_zyz_rewrite_decomposes_to_rz_ry(self):
        """ZYZRewrite should decompose GateU to RZ/RY gates."""
        rule = ZYZRewrite()
        circ = rule.decompose_step(GateU(pi / 2, pi / 4, pi / 3), [0], [], [])
        # Should have RZ, RY, RZ sequence (some may be omitted if zero)
        for inst in circ:
            assert isinstance(inst.operation, (mc.GateRZ, mc.GateRY, mc.GateU))

    def test_special_angle_rewrite_matches_pi_multiples(self):
        """SpecialAngleRewrite should match rotations with k*pi/4 angles."""
        rule = SpecialAngleRewrite()
        assert rule.matches(GateRZ(pi / 4))  # k=1
        assert rule.matches(GateRZ(pi / 2))  # k=2
        assert rule.matches(GateRZ(pi))  # k=4
        assert not rule.matches(GateRZ(0.123))  # arbitrary

    def test_special_angle_rewrite_decomposes_to_t(self):
        """RZ(pi/4) should decompose to T gate."""
        rule = SpecialAngleRewrite()
        circ = rule.decompose_step(GateRZ(pi / 4), [0], [], [])
        assert len(circ) == 1
        assert str(circ[0].operation) == "T"

    def test_special_angle_rewrite_decomposes_to_s(self):
        """RZ(pi/2) should decompose to S gate."""
        rule = SpecialAngleRewrite()
        circ = rule.decompose_step(GateRZ(pi / 2), [0], [], [])
        assert len(circ) == 1
        assert str(circ[0].operation) == "S"

    def test_special_angle_rewrite_clifford_only(self):
        """SpecialAngleRewrite(only_cliffords=True) should only match k*pi/2."""
        rule = SpecialAngleRewrite(only_cliffords=True)
        assert rule.matches(GateRZ(pi / 2))  # k=2
        assert rule.matches(GateRZ(pi))  # k=4
        assert not rule.matches(GateRZ(pi / 4))  # k=1 (not Clifford)

    def test_to_z_rotation_rewrite_matches_rx_ry(self):
        """ToZRotationRewrite should match RX and RY."""
        rule = ToZRotationRewrite()
        assert rule.matches(GateRX(pi / 3))
        assert rule.matches(GateRY(pi / 3))
        assert not rule.matches(GateRZ(pi / 3))

    def test_to_z_rotation_rewrite_converts_to_rz(self):
        """ToZRotationRewrite should convert RX/RY to RZ + Cliffords."""
        rule = ToZRotationRewrite()
        circ = rule.decompose_step(GateRX(pi / 3), [0], [], [])
        # Should contain H, RZ, H
        has_rz = any(isinstance(inst.operation, mc.GateRZ) for inst in circ)
        assert has_rz

    def test_toffoli_to_clifford_t_rewrite(self):
        """Toffoli rewrite should produce the standard Clifford+T sequence."""
        rule = ToffoliToCliffordTRewrite()
        assert rule.matches(GateCCX())

        circ = rule.decompose_step(GateCCX(), [0, 1, 2], [], [])
        assert len(circ) == 15
        for inst in circ:
            assert isinstance(
                inst.operation, (mc.GateH, mc.GateCX, mc.GateT, mc.GateTDG)
            )

    def test_flatten_containers_rewrite_flattens_gatecall(self):
        """FlattenContainers should expand GateCall without gate-level decomposition."""
        inner = Circuit()
        inner.push(GateH(), 0)
        decl = mc.GateDecl("Inner", (), inner)
        call = mc.GateCall(decl, ())

        rule = FlattenContainers()
        assert rule.matches(call)

        circ = rule.decompose_step(call, [0], [], [])
        assert len(circ) == 1
        assert isinstance(circ[0].operation, mc.GateH)


class TestStimBasis:
    """Stim basis terminals."""

    def test_depolarizing_terminal(self):
        basis = StimBasis()
        assert basis.isterminal(mc.Depolarizing1(0.1))
        assert basis.isterminal(mc.Depolarizing2(0.1))
        assert not basis.isterminal(mc.Depolarizing(3, 0.1))


class TestDecompositionBases:
    """Tests for DecompositionBasis implementations."""

    def test_canonical_basis_terminal_gates(self):
        """CanonicalBasis should have GateU and GateCX as terminal."""
        basis = CanonicalBasis()
        assert basis.isterminal(GateU(pi / 2, 0, pi))
        assert basis.isterminal(GateCX())
        assert basis.isterminal(Measure())
        assert basis.isterminal(Reset())
        assert basis.isterminal(Barrier(2))

    @pytest.mark.parametrize(
        "op",
        [
            mc.Loss(),
            mc.Reload(),
            mc.Loss(0.5),
            mc.Check(),
            mc.MeasureCheck(),
        ],
    )
    def test_canonical_basis_terminal_loss_operations(self, op):
        """CanonicalBasis should keep loss operations as terminal primitives."""
        basis = CanonicalBasis()
        assert basis.isterminal(op)

    def test_canonical_basis_non_terminal_gates(self):
        """CanonicalBasis should have other gates as non-terminal."""
        basis = CanonicalBasis()
        assert not basis.isterminal(GateH())
        assert not basis.isterminal(GateCCX())
        assert not basis.isterminal(GateRX(pi / 2))

    def test_clifford_t_basis_terminal_gates(self):
        """CliffordTBasis should have Clifford+T gates as terminal."""
        basis = CliffordTBasis()
        assert basis.isterminal(GateH())
        assert basis.isterminal(GateS())
        assert basis.isterminal(GateT())
        assert basis.isterminal(GateX())
        assert basis.isterminal(GateY())
        assert basis.isterminal(GateZ())
        assert basis.isterminal(GateCX())

    def test_clifford_t_basis_non_terminal_gates(self):
        """CliffordTBasis should have parametric gates as non-terminal."""
        basis = CliffordTBasis()
        assert not basis.isterminal(GateU(pi / 2, 0, pi))
        assert not basis.isterminal(GateRX(pi / 3))
        assert not basis.isterminal(GateCCX())

    def test_rule_basis_wraps_rule(self):
        """RuleBasis should wrap a RewriteRule as a DecompositionBasis."""
        rule = ZYZRewrite()
        basis = RuleBasis(rule)
        # Terminal if rule doesn't match
        assert basis.isterminal(GateH())  # ZYZRewrite doesn't match H
        assert not basis.isterminal(GateU(pi / 2, pi / 4, pi / 3))

    def test_flattened_basis_container_terminals(self):
        """FlattenedBasis should mark only container ops as non-terminal."""
        basis = FlattenedBasis()
        assert basis.isterminal(GateH())

        inner = Circuit()
        inner.push(GateH(), 0)
        decl = mc.GateDecl("Inner", (), inner)
        call = mc.GateCall(decl, ())
        assert not basis.isterminal(call)


def _unitary(instructions, n):
    """Unitary of a list of (operation, qubits) pairs on n qubits."""
    U = np.eye(2**n, dtype=complex)
    for op, qs in instructions:
        M = np.array([[complex(x) for x in row] for row in op.matrix().tolist()])
        k = len(qs)
        psi = np.moveaxis(U.reshape([2] * n + [2**n]), list(qs), list(range(k)))
        psi = np.tensordot(
            M.reshape([2] * (2 * k)), psi, axes=(range(k, 2 * k), range(k))
        )
        U = np.moveaxis(psi, list(range(k)), list(qs)).reshape(2**n, 2**n)
    return U


def _same_up_to_phase(A, B):
    return np.isclose(abs(np.trace(A.conj().T @ B)), A.shape[0])


class TestStimBasis:
    """Stim gates added with the Stim 1.16 support."""

    basis = StimBasis()

    def test_new_terminals(self):
        for op in (
            mc.GateHXY(),
            mc.GateHYZ(),
            mc.MeasureXX(),
            mc.MeasureYY(),
            mc.MeasureZZ(),
            mc.MeasurePauli("XYZ"),
            mc.PauliNoise([0.9, 0.1], ["IXZ", "YYY"]),
        ):
            assert self.basis.isterminal(op)

    def test_pauli_rotations_terminal_only_at_half_pi(self):
        P = mc.PauliString
        for g in (
            mc.GateRXX(pi / 2),
            mc.GateRXX(-pi / 2),
            mc.GateRYY(pi / 2),
            mc.GateRZZ(3 * pi / 2),
            mc.RPauli(P("XIZ"), pi / 2),
            mc.RPauli(P("Y"), -pi / 2),
        ):
            assert self.basis.isterminal(g)
        for g in (
            mc.GateRXX(pi),
            mc.GateRYY(0.3),
            mc.GateRZZ(0),
            mc.RPauli(P("XZ"), pi),
            mc.RPauli(P("II"), pi / 2),
        ):
            assert not self.basis.isterminal(g)

    def test_feedback_terminals(self):
        B = mc.BitString
        for P in (GateX(), mc.GateY(), mc.GateZ()):
            assert self.basis.isterminal(mc.IfStatement(P, B("1")))
            assert not self.basis.isterminal(mc.IfStatement(P, B("0")))
            assert not self.basis.isterminal(mc.IfStatement(P, B("11")))
        assert not self.basis.isterminal(mc.IfStatement(GateH(), B("1")))

    def test_depolarizing_terminal(self):
        assert self.basis.isterminal(mc.Depolarizing1(0.1))
        assert self.basis.isterminal(mc.Depolarizing2(0.1))
        assert not self.basis.isterminal(mc.Depolarizing(3, 0.1))

    def test_conditional_pauli_on_zero(self):
        # P if c == 0 becomes P, then P if c == 1
        c = Circuit().push(mc.IfStatement(mc.GateY(), mc.BitString("0")), 1, 0)
        d = decompose(c, self.basis)
        assert [i.operation for i in d] == [
            mc.GateY(),
            mc.IfStatement(mc.GateY(), mc.BitString("1")),
        ]
        assert [i.qubits for i in d] == [(1,), (1,)]
        assert [i.bits for i in d] == [(), (0,)]

    def test_conditional_rotation_by_pi(self):
        # rotations by multiples of pi become the conditional Pauli they equal
        c = Circuit().push(mc.IfStatement(GateRX(pi), mc.BitString("1")), 0, 2)
        d = decompose(c, self.basis)
        assert [i.operation for i in d] == [
            mc.IfStatement(GateX(), mc.BitString("1"))
        ]
        assert [i.bits for i in d] == [(2,)]
        c = Circuit().push(mc.IfStatement(GateRZ(-pi), mc.BitString("0")), 0, 0)
        d = decompose(c, self.basis)
        assert [i.operation for i in d] == [
            mc.GateZ(),
            mc.IfStatement(mc.GateZ(), mc.BitString("1")),
        ]

    @pytest.mark.parametrize(
        "op,targets",
        [
            (mc.IfStatement(GateX(), mc.BitString("10")), (0, 0, 1)),
            (mc.IfStatement(GateH(), mc.BitString("1")), (0, 0)),
            (mc.IfStatement(GateRZ(pi / 2), mc.BitString("1")), (0, 0)),
            (mc.IfStatement(mc.GateCX(), mc.BitString("1")), (0, 1, 0)),
        ],
        ids=str,
    )
    def test_other_conditionals_refused(self, op, targets):
        with pytest.raises(DecompositionError):
            decompose(Circuit().push(op, *targets), self.basis)

    @pytest.mark.parametrize(
        "g",
        [mc.GateRXX(pi), mc.RPauli(mc.PauliString("XZ"), pi), mc.GateRZZ(0)],
        ids=str,
    )
    def test_pauli_rotations_away_from_half_pi_decompose(self, g):
        d = decompose(Circuit().push(g, 0, 1), self.basis)
        assert all(self.basis.isterminal(i.operation) for i in d)
        assert not any(
            isinstance(i.operation, (mc.GateRXX, mc.GateRZZ, mc.RPauli)) for i in d
        )

    @pytest.mark.parametrize("basis", [StimBasis(), CliffordTBasis()], ids=repr)
    def test_phase_only_gateu_dropped(self, basis):
        """A global phase is dropped, in this basis and in CliffordTBasis."""
        P = mc.PauliString
        for op in (GateU(0, 0, 0, 0.3), GateU(0, 0, 0, 0), mc.RPauli(P("I"), 0.3)):
            assert len(decompose(Circuit().push(op, 0), basis)) == 0
        assert len(decompose(Circuit().push(mc.RPauli(P("II"), pi / 3), 0, 1), basis)) == 0

    @pytest.mark.parametrize(
        "basis,nc,gamma,atol",
        [
            (StimBasis(), 1, pi / 2, 1e-10),
            (StimBasis(), 1, pi, 1e-10),
            (StimBasis(), 2, pi, 1e-10),
            (CliffordTBasis(), 1, pi / 2, 1e-10),
            (CliffordTBasis(), 2, pi / 2, 1e-10),
            # Gridsynth approximates the phase to 1e-10; dropping it would be off by about 0.3.
            (CliffordTBasis(), 1, 0.3, 1e-8),
            (CliffordTBasis(), 2, 0.3, 1e-8),
        ],
        ids=str,
    )
    def test_controlled_phase_only_gateu_keeps_phase(self, basis, nc, gamma, atol):
        """Under a Control the phase is relative and must be kept."""
        op = mc.Control(nc, GateU(0, 0, 0, gamma))
        n = nc + 1
        d = decompose(Circuit().push(op, *range(n)), basis)
        assert all(basis.isterminal(i.operation) for i in d)
        A = _unitary([(op, list(range(n)))], n)
        B = _unitary([(i.operation, list(i.qubits)) for i in d], n)
        phase = np.trace(B.conj().T @ A)
        phase /= abs(phase)
        assert np.linalg.norm(A - phase * B, 2) < atol

    def test_controlled_non_clifford_phase_refused(self):
        with pytest.raises(DecompositionError):
            decompose(Circuit().push(mc.Control(1, GateU(0, 0, 0, 0.3)), 0, 1), self.basis)


class TestCliffordTBasisFollowsStim:
    """CliffordTBasis keeps what StimBasis keeps, plus T, T-dagger, classical
    operations and conditionals of any of its terminal operations."""

    basis = CliffordTBasis()

    # One instance of every kind of operation.
    OPS = [
        mc.GateID(), GateX(), GateY(), GateZ(), GateH(), mc.GateHXY(), mc.GateHYZ(), mc.GateHXZ(),
        GateS(), mc.GateSDG(), mc.GateSX(), mc.GateSXDG(), mc.GateSY(), mc.GateSYDG(),
        mc.GateP(0.3), GateRX(0.3), GateRY(pi / 2), GateRZ(pi / 4), mc.GateR(0.3, 0.2),
        GateU(0.1, 0.2, 0.3), GateU(0, 0, 0, 0.3), mc.GateU1(0.3), mc.GateU2(0.1, 0.2),
        mc.GateU3(0.1, 0.2, 0.3), mc.GateRNZ(2, 0.3),
        GateCX(), mc.GateCY(), mc.GateCZ(), mc.GateCH(), mc.GateCS(), mc.GateCSDG(), mc.GateCSX(),
        mc.GateCSXDG(), mc.GateCP(0.3), mc.GateCRX(0.3), mc.GateCRY(0.3), mc.GateCRZ(0.3),
        mc.GateCU(0.1, 0.2, 0.3, 0.4), mc.GateSWAP(), mc.GateISWAP(), mc.GateISWAPDG(),
        mc.GateDCX(), mc.GateECR(), mc.GateRXX(pi / 2), mc.GateRXX(0.3), mc.GateRYY(-pi / 2),
        mc.GateRYY(0.3), mc.GateRZZ(3 * pi / 2), mc.GateRZZ(pi), mc.GateRZX(0.3),
        mc.GateXXplusYY(0.1, 0.2), mc.GateXXminusYY(0.1, 0.2),
        GateCCX(), mc.GateC3X(), mc.GateCCP(0.3), mc.GateCSWAP(),
        mc.RPauli(mc.PauliString("XYZ"), pi / 2), mc.RPauli(mc.PauliString("II"), pi / 2),
        mc.RPauli(mc.PauliString("XZ"), 0.3), mc.PauliString("XYZ"),
        mc.Control(1, GateH()), mc.Control(2, GateU(0, 0, 0, 0.3)), mc.Power(GateX(), 0.5),
        mc.Parallel(2, GateH()), mc.Repeat(2, GateX()), mc.QFT(2), mc.PhaseGradient(2),
        mc.Diffusion(2), mc.RescaledGate(GateX(), 0.5),
        Measure(), mc.MeasureX(), mc.MeasureY(), mc.MeasureZ(), mc.MeasureXX(), mc.MeasureYY(),
        mc.MeasureZZ(), mc.MeasurePauli("XZ"), Reset(), mc.ResetX(), mc.ResetY(), mc.ResetZ(),
        mc.MeasureReset(), mc.MeasureResetX(), mc.MeasureResetY(), mc.MeasureResetZ(), Barrier(1),
        mc.PauliX(0.1), mc.PauliY(0.1), mc.PauliZ(0.1), mc.Depolarizing1(0.1),
        mc.Depolarizing2(0.1), mc.Depolarizing(3, 0.1), mc.PauliNoise([0.9, 0.1], ["I", "X"]),
        mc.AmplitudeDamping(0.1), mc.GeneralizedAmplitudeDamping(0.1, 0.2),
        mc.PhaseAmplitudeDamping(0.1, 0.2, 0.3), mc.ThermalNoise(0.1, 0.2, 0.3, 0.4),
        mc.ProjectiveNoise("X"), mc.ReadoutErr(0.1, 0.2),
        mc.Detector(1), mc.ObservableInclude(1), mc.QubitCoordinates([0.0, 0.0]), mc.Tick(),
        mc.Amplitude(mc.BitString("0")), mc.ExpectationValue(GateZ()), mc.BondDim(),
        mc.SchmidtRank(), mc.VonNeumannEntropy(), mc.Add(2), mc.Multiply(2), mc.Pow(2),
        mc.Delay(0.1), mc.Loss(0.1), mc.Reload(), mc.Check(), mc.MeasureCheck(), mc.Lost(),
        mc.Reloaded(), mc.PrepareStateT(),
    ]

    @pytest.mark.parametrize("op", OPS, ids=str)
    def test_same_terminals_as_stim(self, op):
        assert self.basis.isterminal(op) == StimBasis().isterminal(op)

    def test_t_gates_and_classical(self):
        for op in (GateT(), mc.GateTDG(), mc.Inverse(GateT())):
            assert self.basis.isterminal(op)
            assert not StimBasis().isterminal(op)
        for op in (mc.Not(), mc.SetBit0(), mc.SetBit1(), mc.And(), mc.Or(), mc.Xor()):
            assert self.basis.isterminal(op)

    def test_conditionals(self):
        """A conditional is terminal when its body is, whatever the condition."""
        for body in (GateX(), GateZ(), GateH(), GateT(), GateCX(), Measure(), mc.ResetX()):
            for cond in ("1", "0", "10"):
                assert self.basis.isterminal(mc.IfStatement(body, mc.BitString(cond)))
        for body in (GateRX(0.3), GateCCX(), GateU(0.1, 0.2, 0.3)):
            assert not self.basis.isterminal(mc.IfStatement(body, mc.BitString("1")))

    def test_stim_operations_are_kept(self):
        c = Circuit().push(mc.GateHXY(), 0)
        c.push(mc.GateRZZ(pi / 2), 0, 1)
        c.push(mc.RPauli(mc.PauliString("XYZ"), -pi / 2), 0, 1, 2)
        c.push(mc.MeasurePauli("XZ"), 0, 2, 0)
        c.push(mc.MeasureResetY(), 1, 1)
        c.push(mc.ResetX(), 2)
        c.push(mc.ResetY(), 0)
        c.push(mc.Depolarizing2(0.1), 0, 1)
        c.push(mc.Detector(1), 0)
        assert [i.operation for i in decompose(c, self.basis)] == [i.operation for i in c]

    def test_other_gates_are_decomposed(self):
        c = Circuit().push(GateCCX(), 0, 1, 2)
        c.push(mc.GateRXX(pi / 4), 0, 1)
        c.push(mc.RPauli(mc.PauliString("XZ"), pi), 1, 2)
        c.push(mc.GateCP(pi / 2), 0, 2)
        c.push(mc.IfStatement(mc.GateCRZ(pi / 2), mc.BitString("0")), 0, 1, 0)
        d = decompose(c, self.basis)
        assert all(self.basis.isterminal(i.operation) for i in d)
        unitary = [(i.operation, list(i.qubits)) for i in c][:4]
        decomposed = [(i.operation, list(i.qubits)) for i in d if not isinstance(i.operation, mc.IfStatement)]
        assert _same_up_to_phase(_unitary(unitary, 3), _unitary(decomposed, 3))

    def test_unsupported_noise_raises(self):
        with pytest.raises(DecompositionError):
            decompose(Circuit().push(mc.AmplitudeDamping(0.1), 0), self.basis)


class TestConditionalDecomposition:
    """IfStatement handling in the decomposition bases."""

    ALL_BASES = (
        CanonicalBasis(),
        CliffordTBasis(),
        QASMBasis(),
        StimBasis(),
        FlattenedBasis(),
    )

    @pytest.mark.parametrize("basis", ALL_BASES, ids=repr)
    def test_feedback_is_terminal(self, basis):
        """Measure + conditional X and MeasureReset reach terminal operations."""
        feedback = Circuit().push(Measure(), 0, 0)
        feedback.push(mc.IfStatement(GateX(), mc.BitString("1")), 1, 0)
        reset = Circuit().push(mc.MeasureReset(), 0, 0)
        for c in (feedback, reset):
            decomposed = decompose(c, basis)
            assert all(basis.isterminal(i.operation) for i in decomposed)

    def test_stim_refuses_conditional_gates(self):
        c = Circuit().push(mc.IfStatement(GateH(), mc.BitString("1")), 0, 0)
        with pytest.raises(DecompositionError):
            decompose(c, StimBasis())

    @pytest.mark.parametrize(
        "gate",
        [GateCCX(), GateRX(pi / 4), mc.GateCSWAP(), GateU(pi / 2, pi / 4, -pi / 4)],
        ids=str,
    )
    def test_clifford_t_conditional_is_exact(self, gate):
        """A conditional gate decomposes into conditional Clifford+T gates with
        the same condition, and their product is the original gate."""
        basis = CliffordTBasis()
        n = gate.num_qubits
        qubits = list(range(1, n + 1))
        c = Circuit().push(mc.IfStatement(gate, mc.BitString("1")), *qubits, 0)
        decomposed = decompose(c, basis)
        assert all(isinstance(i.operation, mc.IfStatement) for i in decomposed)
        assert all(i.operation.bitstring == mc.BitString("1") for i in decomposed)
        assert all(i.bits == (0,) for i in decomposed)
        assert all(basis.isterminal(i.operation.op) for i in decomposed)
        body = [(i.operation.op, [q - 1 for q in i.qubits]) for i in decomposed]
        assert _same_up_to_phase(
            _unitary(body, n), _unitary([(gate, list(range(n)))], n)
        )

    def test_clifford_t_conditional_approximate_rotation(self):
        """A conditional arbitrary rotation becomes conditional Clifford+T gates."""
        basis = CliffordTBasis(sk_depth=0)
        c = Circuit().push(mc.IfStatement(GateRX(0.3), mc.BitString("0")), 0, 2)
        decomposed = decompose(c, basis)
        assert len(decomposed) > 0
        for i in decomposed:
            assert isinstance(i.operation, mc.IfStatement)
            assert i.operation.bitstring == mc.BitString("0")
            assert i.bits == (2,)
            assert basis.isterminal(i.operation.op)

    def test_nested_conditions_are_merged(self):
        """A conditional MeasureReset gives one IfStatement per instruction."""
        op = mc.IfStatement(mc.MeasureReset(), mc.BitString("0"))
        c = Circuit().push(op, 0, 0, 1)
        decomposed = decompose_step(c)
        assert [i.operation for i in decomposed] == [
            mc.IfStatement(Measure(), mc.BitString("0")),
            mc.IfStatement(GateX(), mc.BitString("10")),
        ]
        assert [i.bits for i in decomposed] == [(0, 1), (0, 1)]

    def test_flattened_conditional_container(self):
        """FlattenedBasis expands a conditional GateCall into conditional gates."""
        inner = Circuit().push(GateH(), 0).push(GateCX(), 0, 1)
        call = mc.GateCall(mc.GateDecl("Inner", (), inner), ())
        c = Circuit().push(mc.IfStatement(call, mc.BitString("1")), 0, 1, 0)
        decomposed = decompose(c, FlattenedBasis())
        assert [i.operation for i in decomposed] == [
            mc.IfStatement(GateH(), mc.BitString("1")),
            mc.IfStatement(GateCX(), mc.BitString("1")),
        ]

    def test_rule_basis_conditional(self):
        """RuleBasis applies its rule to the body of a conditional."""
        basis = RuleBasis(ZYZRewrite())
        op = mc.IfStatement(GateU(pi / 2, pi / 4, pi / 3), mc.BitString("1"))
        assert not basis.isterminal(op)
        decomposed = decompose(Circuit().push(op, 0, 0), basis)
        assert all(isinstance(i.operation, mc.IfStatement) for i in decomposed)
        assert all(basis.isterminal(i.operation) for i in decomposed)


class TestDecomposeFunction:
    """Tests for the decompose function."""

    def test_decompose_single_gate(self):
        """decompose should decompose a single gate."""
        c = Circuit()
        c.push(GateH(), 0)
        decomposed = decompose(c)
        assert len(decomposed) >= 1
        # All gates should be terminal in CanonicalBasis
        basis = CanonicalBasis()
        for inst in decomposed:
            assert basis.isterminal(inst.operation)

    def test_decompose_multi_qubit_gate(self):
        """decompose should decompose multi-qubit gates."""
        c = Circuit()
        c.push(GateCCX(), 0, 1, 2)
        decomposed = decompose(c)
        # All gates should be terminal
        basis = CanonicalBasis()
        for inst in decomposed:
            assert basis.isterminal(inst.operation)

    def test_decompose_preserves_terminal(self):
        """decompose should preserve terminal operations."""
        c = Circuit()
        c.push(GateU(pi / 2, 0, pi), 0)
        c.push(GateCX(), 0, 1)
        c.push(Measure(), 0, 0)
        decomposed = decompose(c)
        # Should be unchanged (all terminal)
        assert len(decomposed) == 3

    def test_decompose_preserves_loss_operations(self):
        """decompose should preserve terminal loss operations unchanged."""
        c = Circuit()
        c.push(mc.Loss(0.5), 0)
        c.push(mc.Loss(), 1)
        c.push(mc.Check(), 0, 0)
        c.push(mc.MeasureCheck(), 1, 1, 2)
        c.push(mc.Reload(), 1)

        decomposed = decompose(c)

        assert decomposed == c

    def test_decompose_with_clifford_t_basis(self):
        """decompose should work with CliffordTBasis."""
        c = Circuit()
        c.push(GateRZ(pi / 4), 0)  # Should become T
        c.push(GateRZ(pi / 2), 0)  # Should become S
        decomposed = decompose(c, CliffordTBasis())
        basis = CliffordTBasis()
        for inst in decomposed:
            assert basis.isterminal(inst.operation)

    def test_clifford_t_basis_toffoli_rewrite(self):
        """CliffordTBasis should decompose CCX directly to Clifford+T gates."""
        c = Circuit()
        c.push(GateCCX(), 0, 1, 2)
        decomposed = decompose(c, CliffordTBasis(sk_depth=0))
        basis = CliffordTBasis(sk_depth=0)
        for inst in decomposed:
            assert basis.isterminal(inst.operation)
            assert not isinstance(inst.operation, mc.GateU)

    def test_decompose_with_flattened_basis(self):
        """FlattenedBasis should recursively flatten nested GateCalls."""
        inner = Circuit()
        inner.push(GateH(), 0)
        decl_inner = mc.GateDecl("Inner", (), inner)

        middle = Circuit()
        middle.push(mc.GateCall(decl_inner, ()), 0)
        decl_middle = mc.GateDecl("Middle", (), middle)

        c = Circuit()
        c.push(mc.GateCall(decl_middle, ()), 0)

        decomposed = decompose(c, FlattenedBasis())
        assert len(decomposed) == 1
        assert isinstance(decomposed[0].operation, mc.GateH)

    def test_decompose_with_rewrite_rule(self):
        """decompose should accept a RewriteRule (wrapped in RuleBasis)."""
        c = Circuit()
        c.push(GateU(pi / 2, pi / 4, pi / 3), 0)
        decomposed = decompose(c, ZYZRewrite())
        # ZYZ decomposes to RZ/RY
        for inst in decomposed:
            assert isinstance(inst.operation, (mc.GateRZ, mc.GateRY, mc.GateU, mc.GateH))

    def test_decompose_with_canonical_rewrite_terminates(self):
        """decompose should terminate with CanonicalRewrite via RuleBasis."""
        c = Circuit()
        c.push(GateH(), 0)
        decomposed = decompose(c, CanonicalRewrite())
        assert len(decomposed) == 1
        assert isinstance(decomposed[0].operation, mc.GateU)

    def test_decompose_operation_directly(self):
        """decompose should work on a single Operation."""
        decomposed = decompose(GateH())
        assert len(decomposed) >= 1

    def test_decompose_measure_reset_x(self):
        """Non-terminal measurement variants should still decompose canonically."""
        c = Circuit()
        c.push(mc.MeasureResetX(), 0, 0)
        decomposed = decompose(c)
        assert len(decomposed) >= 1

    def test_measure_reset_decomposition(self):
        """MeasureReset expands to Measure and a conditional X on the same bit."""
        c = Circuit().push(mc.MeasureReset(), 2, 1)
        stepped = decompose_step(c)
        assert [i.operation for i in stepped] == [
            mc.Measure(),
            mc.IfStatement(GateX(), mc.BitString("1")),
        ]
        assert [i.qubits for i in stepped] == [(2,), (2,)]
        assert [i.bits for i in stepped] == [(1,), (1,)]

    def test_clifford_t_measurement_terminals(self):
        """CliffordTBasis keeps measurements and resets in every Pauli basis."""
        basis = CliffordTBasis()
        ops = [
            mc.Measure(), mc.MeasureX(), mc.MeasureY(), mc.MeasureZZ(),
            mc.MeasurePauli("XZ"), mc.Reset(), mc.ResetX(), mc.ResetY(),
            mc.MeasureReset(), mc.MeasureResetX(), mc.MeasureResetY(),
        ]
        for op in ops:
            assert basis.isterminal(op)
            c = Circuit().push(op, *range(op.num_qubits), *range(op.num_bits))
            assert [i.operation for i in decompose(c, basis=basis)] == [op]

    def test_circuit_decompose_method(self):
        """Circuit.decompose() should use the new framework."""
        c = Circuit()
        c.push(GateH(), 0)
        c.push(GateCCX(), 0, 1, 2)
        decomposed = c.decompose()
        basis = CanonicalBasis()
        for inst in decomposed:
            assert basis.isterminal(inst.operation)

    def test_circuit_decompose_with_basis(self):
        """Circuit.decompose(basis) should accept a basis argument."""
        c = Circuit()
        c.push(GateRZ(pi / 4), 0)
        decomposed = c.decompose(CliffordTBasis())
        assert any(str(inst.operation) == "T" for inst in decomposed)


class TestDecomposeStep:
    """Tests for the decompose_step function."""

    def test_decompose_step_applies_once(self):
        """decompose_step should apply the rule only once (non-recursive)."""
        c = Circuit()
        c.push(GateH(), 0)
        stepped = decompose_step(c)
        # H decomposes to U, which is terminal
        assert len(stepped) == 1

    def test_decompose_step_with_rule(self):
        """decompose_step should work with a specific rule."""
        c = Circuit()
        c.push(GateU(pi / 2, pi / 4, pi / 3), 0)
        stepped = decompose_step(c, ZYZRewrite())
        # Should have RZ/RY gates
        for inst in stepped:
            assert isinstance(
                inst.operation, (mc.GateRZ, mc.GateRY, mc.GateU, mc.GateH)
            )

    def test_decompose_step_preserves_non_matching(self):
        """decompose_step should preserve operations that don't match the rule."""
        c = Circuit()
        c.push(GateH(), 0)  # ZYZRewrite doesn't match H
        c.push(GateU(pi / 2, 0, pi), 0)  # ZYZRewrite matches U
        stepped = decompose_step(c, ZYZRewrite())
        # H should be preserved, U should be decomposed
        assert any(isinstance(inst.operation, mc.GateH) for inst in stepped)

    def test_circuit_decompose_step_uses_default_rule(self):
        """Circuit.decompose_step() should default to CanonicalRewrite."""
        c = Circuit()
        c.push(GateH(), 0)

        stepped = c.decompose_step()

        assert stepped == decompose_step(c)
        assert len(stepped) == 1
        assert isinstance(stepped[0].operation, mc.GateU)

    def test_circuit_decompose_step_accepts_rule(self):
        """Circuit.decompose_step(rule) should forward to the module helper."""
        c = Circuit()
        c.push(GateU(pi / 2, pi / 4, pi / 3), 0)

        stepped = c.decompose_step(ZYZRewrite())

        assert stepped == decompose_step(c, ZYZRewrite())
        assert all(
            isinstance(inst.operation, (mc.GateRZ, mc.GateRY, mc.GateU, mc.GateH))
            for inst in stepped
        )


class TestEachDecomposed:
    """Tests for the eachdecomposed iterator."""

    def test_eachdecomposed_is_iterator(self):
        """eachdecomposed should return an iterator."""
        c = Circuit()
        c.push(GateH(), 0)
        it = eachdecomposed(c)
        assert hasattr(it, "__iter__")
        assert hasattr(it, "__next__")

    def test_eachdecomposed_yields_instructions(self):
        """eachdecomposed should yield decomposed instructions."""
        c = Circuit()
        c.push(GateH(), 0)
        c.push(GateCX(), 0, 1)
        instructions = list(eachdecomposed(c))
        assert len(instructions) >= 2
        for inst in instructions:
            assert isinstance(inst, mc.Instruction)

    def test_eachdecomposed_with_basis(self):
        """eachdecomposed should work with a specific basis."""
        c = Circuit()
        c.push(GateRZ(pi / 4), 0)
        instructions = list(eachdecomposed(c, CliffordTBasis()))
        assert any(str(inst.operation) == "T" for inst in instructions)


class TestCustomRules:
    """Tests for creating custom rewrite rules."""

    def test_custom_rewrite_rule(self):
        """Custom RewriteRule should work with decompose."""

        class SwapToThreeCX(RewriteRule):
            """Decompose SWAP to 3 CX gates."""

            def matches(self, op):
                return isinstance(op, mc.GateSWAP)

            def decompose_step(self, op, qubits, bits, zvars):
                c = Circuit()
                q0, q1 = qubits[0], qubits[1]
                c.push(GateCX(), q0, q1)
                c.push(GateCX(), q1, q0)
                c.push(GateCX(), q0, q1)
                return c

        rule = SwapToThreeCX()
        c = Circuit()
        c.push(mc.GateSWAP(), 0, 1)
        decomposed = decompose(c, rule)
        # Should have 3 CX gates
        assert len(decomposed) == 3
        for inst in decomposed:
            assert str(inst.operation) == "CX"


class TestDecompositionError:
    """Tests for DecompositionError handling."""

    def test_decomposition_error_raised(self):
        """DecompositionError should be raised for unsupported operations."""

        class NoOpBasis(DecompositionBasis):
            """Basis that can't decompose anything."""

            def isterminal(self, op):
                return False

            def decompose(self, op, qubits, bits, zvars):
                raise DecompositionError(f"Cannot decompose {type(op).__name__}")

        basis = NoOpBasis()
        c = Circuit()
        c.push(GateH(), 0)
        with pytest.raises(DecompositionError):
            decompose(c, basis)


class TestIsTerminalByType:
    """``isterminal`` caches its answer per operation type, so operations of
    one type whose answer depends on what they wrap must not share it."""

    def test_repeated_calls_agree(self):
        from mimiqcircuits.decomposition import QASMBasis, StimBasis

        ops = [GateH(), GateCX(), GateCCX(), GateU(0.1, 0.2, 0.3), GateRX(0.3), Measure()]
        for basis in (CanonicalBasis(), CliffordTBasis(), QASMBasis(), StimBasis()):
            first = [basis.isterminal(op) for op in ops]
            assert [basis.isterminal(op) for op in ops] == first

    @pytest.mark.parametrize("basis_name", ["CanonicalBasis", "QASMBasis"])
    def test_ifstatement_follows_body(self, basis_name):
        import mimiqcircuits.decomposition as dec

        basis = getattr(dec, basis_name)()
        cond = mc.BitString("1")
        terminal = mc.IfStatement(GateCX(), cond)
        nonterminal = mc.IfStatement(mc.Control(3, GateH()), cond)
        assert basis.isterminal(terminal)
        assert not basis.isterminal(nonterminal)
        assert basis.isterminal(terminal)

    def test_qasm_gatecall_follows_body(self):
        from mimiqcircuits.decomposition import QASMBasis

        basis = QASMBasis()
        good = mc.GateDecl("good", (), Circuit().push(GateH(), 0))
        bad = mc.GateDecl("bad", (), Circuit().push(mc.Control(3, GateH()), 0, 1, 2, 3))
        assert basis.isterminal(mc.GateCall(good, ()))
        assert not basis.isterminal(mc.GateCall(bad, ()))
        assert basis.isterminal(mc.GateCall(good, ()))

    def test_flattened_inverse_follows_body(self):
        basis = FlattenedBasis()
        decl = mc.GateDecl("g", (), Circuit().push(GateH(), 0))
        assert basis.isterminal(mc.Inverse(GateRX(0.2)))
        assert not basis.isterminal(mc.Inverse(mc.GateCall(decl, ())))
        assert basis.isterminal(mc.Inverse(GateRX(0.2)))


def _circuit_unitary(circuit, nq):
    u = np.eye(2**nq, dtype=complex)
    for inst in circuit:
        m = reorder_qubits_matrix(inst.operation.matrix(), list(inst.qubits), nq)
        m = np.array(
            [[complex(m[i, j]) for j in range(m.cols)] for i in range(m.rows)],
            dtype=complex,
        )
        u = m @ u
    return u


def _equal_up_to_phase(a, b, atol):
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    return np.allclose(a, (a[k] / b[k]) * b, atol=atol)


class TestCliffordTBasisRotations:
    """Tests for the rotation synthesis rule of CliffordTBasis."""

    def test_default_is_gridsynth(self):
        assert isinstance(CliffordTBasis().rotations, GridSynthRewrite)

    def test_default_basis_phase_only_gateu(self):
        """A phase-only GateU (e.g. from GateHYZ) must not reach the rotations
        rule, which only accepts GateRZ."""
        basis = CliffordTBasis()
        for op, q, b in [(GateU(0, 0, 0, 0.3), [0], []), (mc.GateHYZ(), [0], []), (mc.MeasureY(), [0], [0])]:
            d = decompose(Circuit().push(op, *q, *b), basis=basis)
            assert all(basis.isterminal(i.operation) for i in d)
            assert not any(isinstance(i.operation, mc.GateID) for i in d)
        assert len(decompose(Circuit().push(GateU(0, 0, 0, 0.3), 0), basis=basis)) == 0

    @pytest.mark.parametrize("gamma", [0.3, math.pi / 3])
    @pytest.mark.parametrize("nc", [1, 2])
    def test_controlled_phase_only_gateu_keeps_phase(self, nc, gamma):
        """Under a Control the phase of GateU(0, 0, 0, gamma) is relative and
        must survive the decomposition."""
        op = mc.Control(nc, GateU(0, 0, 0, gamma))
        nq = nc + 1
        c = Circuit().push(op, *range(nq))
        d = decompose(c, basis=CliffordTBasis())
        assert all(CliffordTBasis().isterminal(i.operation) for i in d)
        assert _equal_up_to_phase(_circuit_unitary(c, nq), _circuit_unitary(d, nq), atol=1e-8)

    def test_equality(self):
        assert CliffordTBasis(sk_depth=3) == CliffordTBasis(sk_depth=3)
        assert hash(CliffordTBasis(sk_depth=3)) == hash(CliffordTBasis(sk_depth=3))
        assert CliffordTBasis(sk_depth=3) != CliffordTBasis(sk_depth=2)
        assert CliffordTBasis() == CliffordTBasis()
        assert CliffordTBasis() != CliffordTBasis(rotations=GridSynthRewrite(epsilon=1e-6))

    def test_sk_depth_selects_solovay_kitaev(self):
        basis = CliffordTBasis(sk_depth=2)
        assert isinstance(basis.rotations, SolovayKitaevRewrite)
        assert basis.rotations.depth == 2

    def test_both_raise(self):
        with pytest.raises(ValueError):
            CliffordTBasis(sk_depth=2, rotations=GridSynthRewrite())

    def test_rule_must_match_rz(self):
        with pytest.raises(ValueError):
            CliffordTBasis(rotations=ZYZRewrite())

    def test_custom_rotations(self):
        basis = CliffordTBasis(rotations=GridSynthRewrite(epsilon=1e-4))
        c = Circuit()
        c.push(GateRZ(0.3), 0)
        d = decompose(c, basis)
        assert all(basis.isterminal(i.operation) for i in d)
        assert d == decompose_step(GateRZ(0.3), GridSynthRewrite(epsilon=1e-4))

    def test_default_output_gate_set(self):
        c = Circuit()
        c.push(GateU(0.3, 0.4, 0.5), 0)
        c.push(GateRX(0.7), 1)
        basis = CliffordTBasis()
        d = decompose(c, basis)
        assert all(basis.isterminal(i.operation) for i in d)
        allowed = (GateH, mc.GateS, mc.GateSDG, GateZ, GateT, GateX)
        assert all(isinstance(i.operation, allowed) for i in d)
