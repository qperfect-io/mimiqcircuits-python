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
from mimiqcircuits import *
from mimiqcircuits.circuit_extras import remove_swaps
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.inverse import Inverse
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.ifstatement import IfStatement


# ---------------------------------------------------------------------------
# Non-recursive tests
# ---------------------------------------------------------------------------

class TestRemoveSwapsNonRecursive:

    def test_single_swap(self):
        c = Circuit()
        c.push(GateH(), 1)
        c.push(GateSWAP(), 1, 2)
        c.push(GateCX(), 2, 3)
        new_c, perm = remove_swaps(c)

        assert len(new_c.instructions) == 2
        assert perm == [0, 2, 1, 3]
        assert not any(isinstance(i.get_operation(), GateSWAP) for i in new_c)
        # H stays on qubit 1, CX(2,3) becomes CX(1,3)
        assert new_c.instructions[0].get_qubits() == (1,)
        assert isinstance(new_c.instructions[0].get_operation(), GateH)
        assert new_c.instructions[1].get_qubits() == (1, 3)

    def test_multiple_swaps_chain(self):
        c = Circuit()
        c.push(GateSWAP(), 1, 2)
        c.push(GateSWAP(), 2, 3)
        c.push(GateCX(), 1, 3)
        new_c, perm = remove_swaps(c)

        assert perm == [0, 2, 3, 1]
        assert not any(isinstance(i.get_operation(), GateSWAP) for i in new_c)
        assert len(new_c.instructions) == 2
        assert new_c.instructions[0].get_qubits() == (2, 1)
        assert isinstance(new_c.instructions[1].get_operation(), GateID)

    def test_cancelling_swaps(self):
        c = Circuit()
        c.push(GateSWAP(), 0, 1)
        c.push(GateSWAP(), 0, 1)
        c.push(GateH(), 0)
        new_c, perm = remove_swaps(c)

        assert perm == [0, 1]
        assert len(new_c.instructions) == 2
        assert isinstance(new_c.instructions[0].get_operation(), GateH)
        assert new_c.instructions[0].get_qubits() == (0,)
        assert isinstance(new_c.instructions[1].get_operation(), GateID)

    def test_no_swaps(self):
        c = Circuit()
        c.push(GateH(), 0)
        c.push(GateCX(), 0, 1)
        new_c, perm = remove_swaps(c)

        assert perm == [0, 1]
        assert len(new_c.instructions) == 2

    def test_empty_circuit(self):
        c = Circuit()
        new_c, perm = remove_swaps(c)
        assert len(new_c.instructions) == 0
        assert perm == []

    def test_swap_only_circuit(self):
        c = Circuit()
        c.push(GateSWAP(), 0, 1)
        c.push(GateSWAP(), 1, 2)
        new_c, perm = remove_swaps(c)

        assert len(new_c.instructions) == 3
        assert perm == [1, 2, 0]
        assert all(isinstance(i.get_operation(), GateID) for i in new_c)

    def test_swap_at_end(self):
        c = Circuit()
        c.push(GateH(), 0)
        c.push(GateSWAP(), 0, 1)
        new_c, perm = remove_swaps(c)

        assert perm == [1, 0]
        assert len(new_c.instructions) == 2
        assert isinstance(new_c.instructions[0].get_operation(), GateH)
        assert new_c.instructions[0].get_qubits() == (0,)
        assert isinstance(new_c.instructions[1].get_operation(), GateID)

    def test_preserves_bits_and_zvars(self):
        c = Circuit()
        c.push(GateSWAP(), 0, 1)
        c.push(Measure(), 1, 0)
        new_c, perm = remove_swaps(c)

        assert len(new_c.instructions) == 2
        assert new_c.instructions[0].get_qubits() == (0,)
        assert new_c.instructions[0].get_bits() == (0,)
        assert isinstance(new_c.instructions[1].get_operation(), GateID)

    def test_does_not_recurse_without_flag(self):
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateH(), 0)
        Inner = GateDecl("Inner", (), inner)

        c = Circuit()
        c.push(GateCall(Inner, ()), 0, 1)
        new_c, perm = remove_swaps(c)

        assert perm == [0, 1]
        op = new_c.instructions[0].get_operation()
        assert isinstance(op, GateCall)
        # Inner SWAPs should still be there
        assert any(
            isinstance(i.get_operation(), GateSWAP) for i in op._decl.circuit
        )


# ---------------------------------------------------------------------------
# Recursive GateCall tests
# ---------------------------------------------------------------------------

class TestRemoveSwapsRecursiveGateCall:

    def test_simple_gatecall(self):
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateCX(), 0, 1)
        Inner = GateDecl("Inner", (), inner)

        c = Circuit()
        c.push(GateCall(Inner, ()), 0, 1)
        c.push(GateH(), 0)
        new_c, perm = remove_swaps(c, recursive=True)

        assert perm == [1, 0]
        # Internal SWAPs removed
        gcall = new_c.instructions[0].get_operation()
        assert isinstance(gcall, GateCall)
        assert not any(
            isinstance(i.get_operation(), GateSWAP) for i in gcall._decl.circuit
        )
        # H remapped due to block permutation
        assert new_c.instructions[1].get_qubits() == (1,)

    def test_deeply_nested_gatedecl(self):
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateCX(), 1, 0)
        Inner = GateDecl("Inner", (), inner)

        mid = Circuit()
        mid.push(GateSWAP(), 0, 1)
        mid.push(GateCall(Inner, ()), 0, 1)
        Mid = GateDecl("Mid", (), mid)

        outer_c = Circuit()
        outer_c.push(GateSWAP(), 0, 1)
        outer_c.push(GateCall(Mid, ()), 0, 1)
        Outer = GateDecl("Outer", (), outer_c)

        c = Circuit()
        c.push(GateSWAP(), 0, 1)
        c.push(GateCall(Outer, ()), 0, 1)

        new_c, _ = remove_swaps(c, recursive=True)

        # No SWAPs at any nesting level
        assert not any(isinstance(i.get_operation(), GateSWAP) for i in new_c)

        Outer2 = new_c.instructions[0].get_operation()._decl
        assert not any(
            isinstance(i.get_operation(), GateSWAP) for i in Outer2.circuit
        )

        Mid2 = Outer2.circuit.instructions[0].get_operation()._decl
        assert not any(
            isinstance(i.get_operation(), GateSWAP) for i in Mid2.circuit
        )

        Inner2 = Mid2.circuit.instructions[0].get_operation()._decl
        assert not any(
            isinstance(i.get_operation(), GateSWAP) for i in Inner2.circuit
        )

    def test_gatedecl_arity_preservation(self):
        """GateDecl should keep its original qubit count even if SWAPs are removed."""
        inner = Circuit()
        inner.push(GateH(), 0)
        inner.push(GateSWAP(), 1, 2)
        Inner = GateDecl("Test", (), inner)

        c = Circuit()
        c.push(GateCall(Inner, ()), 0, 1, 2)
        new_c, perm = remove_swaps(c, recursive=True)

        gcall = new_c.instructions[0].get_operation()
        # Should still be a 3-qubit gate
        assert gcall._decl.num_qubits == 3

    def test_shared_gatedecl_caching(self):
        """Same GateDecl used twice should produce same processed decl."""
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateH(), 0)
        Inner = GateDecl("Shared", (), inner)

        c = Circuit()
        c.push(GateCall(Inner, ()), 0, 1)
        c.push(GateCall(Inner, ()), 2, 3)
        new_c, perm = remove_swaps(c, recursive=True)

        decl1 = new_c.instructions[0].get_operation()._decl
        decl2 = new_c.instructions[1].get_operation()._decl
        assert decl1 is decl2


# ---------------------------------------------------------------------------
# Recursive Block tests
# ---------------------------------------------------------------------------

class TestRemoveSwapsRecursiveBlock:

    def test_simple_block(self):
        block = Block([
            Instruction(GateSWAP(), (0, 1), (), ()),
            Instruction(GateH(), (0,), (), ()),
        ])
        c = Circuit()
        c.push(block, 2, 3)
        c.push(GateCX(), 2, 3)
        new_c, perm = remove_swaps(c, recursive=True)

        assert perm == [0, 1, 3, 2]
        assert not any(isinstance(i.get_operation(), GateSWAP) for i in new_c)

    def test_block_dimension_preservation(self):
        """Block with explicit dims larger than instruction qubits."""
        block = Block(3, 0, 0, [
            Instruction(GateSWAP(), (0, 1), (), ()),
            Instruction(GateH(), (0,), (), ()),
        ])
        c = Circuit()
        c.push(block, 0, 1, 2)
        c.push(GateCX(), 0, 2)
        new_c, perm = remove_swaps(c, recursive=True)

        new_block = new_c.instructions[0].get_operation()
        assert new_block._num_qubits == 3
        assert perm == [1, 0, 2]

    def test_block_with_bits_preserved(self):
        block = Block(2, 1, 0, [
            Instruction(GateSWAP(), (0, 1), (), ()),
            Instruction(Measure(), (0,), (0,), ()),
        ])
        c = Circuit()
        c.push(block, 0, 1, 0)
        new_c, perm = remove_swaps(c, recursive=True)

        new_block = new_c.instructions[0].get_operation()
        assert new_block._num_bits == 1


# ---------------------------------------------------------------------------
# Inverse/Control/IfStatement treated as opaque
# ---------------------------------------------------------------------------

class TestOpaqueWrappers:

    def _make_swap_decl(self):
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateCX(), 0, 1)
        return GateDecl("Inner", (), inner)

    def test_inverse_recursed_with_expanded_inverse(self):
        decl = self._make_swap_decl()
        c = Circuit()
        c.push(Inverse(GateCall(decl, ())), 0, 1)
        c.push(GateH(), 0)
        new_c, perm = remove_swaps(c, recursive=True)

        # The inverse has SWAP(0,1) internally, so qubit_map = [1,0]
        assert perm == [1, 0]
        # Result is an Inverse(GateCall(...)) (inverse expanded into new GateDecl)
        assert isinstance(new_c.instructions[0].get_operation(), Inverse)
        assert isinstance(new_c.instructions[0].get_operation().op, GateCall)
        # Internal SWAPs should be removed
        new_decl = new_c.instructions[0].get_operation().op._decl
        assert not any(
            isinstance(i.get_operation(), GateSWAP) for i in new_decl.circuit
        )
        # H should be remapped due to the permutation
        assert new_c.instructions[1].get_qubits() == (1,)

    def test_control_not_recursed(self):
        decl = self._make_swap_decl()
        c = Circuit()
        c.push(Control(1, GateCall(decl, ())), 0, 1, 2)
        new_c, perm = remove_swaps(c, recursive=True)

        assert perm == [0, 1, 2]
        assert isinstance(new_c.instructions[0].get_operation(), Control)

    def test_ifstatement_not_recursed(self):
        block = Block([
            Instruction(GateSWAP(), (0, 1), (), ()),
            Instruction(GateH(), (0,), (), ()),
        ])
        ifs = IfStatement(block, BitString("1"))
        c = Circuit()
        c.push(ifs, 0, 1, 0)
        new_c, perm = remove_swaps(c, recursive=True)

        assert perm == [0, 1]
        assert isinstance(new_c.instructions[0].get_operation(), IfStatement)

    def test_outer_swap_and_inverse_swap_cancel(self):
        """Outer SWAP + Inverse's internal SWAP should cancel."""
        decl = self._make_swap_decl()
        c = Circuit()
        c.push(GateSWAP(), 0, 1)
        c.push(Inverse(GateCall(decl, ())), 0, 1)
        new_c, perm = remove_swaps(c, recursive=True)

        # Outer SWAP [1,0] composed with Inverse's internal SWAP [1,0] = identity
        assert perm == [0, 1]


# ---------------------------------------------------------------------------
# Permutation composition tests
# ---------------------------------------------------------------------------

class TestPermutationComposition:

    def test_swap_before_recursive_gatecall(self):
        """Outer SWAP + inner SWAP on same pair should cancel."""
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateH(), 0)
        Inner = GateDecl("Inner", (), inner)

        c = Circuit()
        c.push(GateSWAP(), 0, 1)
        c.push(GateCall(Inner, ()), 0, 1)
        c.push(GateH(), 0)
        new_c, perm = remove_swaps(c, recursive=True)

        assert perm == [0, 1]

    def test_swap_after_recursive_gatecall(self):
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateH(), 0)
        Inner = GateDecl("Inner", (), inner)

        c = Circuit()
        c.push(GateCall(Inner, ()), 0, 1)
        c.push(GateSWAP(), 0, 1)
        c.push(GateH(), 0)
        new_c, perm = remove_swaps(c, recursive=True)

        # Block swap + outer swap cancel
        assert perm == [0, 1]

    def test_mixed_gatecalls_and_swaps(self):
        inner = Circuit()
        inner.push(GateSWAP(), 0, 1)
        inner.push(GateH(), 0)
        Inner = GateDecl("Inner", (), inner)

        c = Circuit()
        c.push(GateCall(Inner, ()), 0, 1)
        c.push(GateCall(Inner, ()), 1, 2)
        new_c, perm = remove_swaps(c, recursive=True)

        # First GateCall @ (0,1): block_map=[1,0] → perm[0]=1, perm[1]=0
        # Second GateCall @ (1,2): new_qubits=(perm[1],perm[2])=(0,2)
        #   block_map=[1,0] → perm[1]=old_perm[2]=2, perm[2]=old_perm[1]=0
        # Final: perm = [1, 2, 0]
        assert perm == [1, 2, 0]
