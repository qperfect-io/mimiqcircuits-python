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

import pytest
from symengine import pi

import mimiqcircuits as mc
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
    RuleBasis,
    CanonicalRewrite,
    FlattenContainers,
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

    def test_canonical_rewrite_matches_non_terminal_only(self):
        """CanonicalRewrite should match only non-terminal canonical operations."""
        rule = CanonicalRewrite()
        assert rule.matches(GateH())
        assert rule.matches(GateCCX())
        assert not rule.matches(GateCX())
        assert not rule.matches(GateU(pi / 2, 0, pi))

    def test_canonical_rewrite_decomposes_h(self):
        """CanonicalRewrite should decompose H to U gate."""
        rule = CanonicalRewrite()
        circ = rule.decompose_step(GateH(), [0], [], [])
        assert len(circ) == 1
        assert isinstance(circ[0].operation, mc.GateU)

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
