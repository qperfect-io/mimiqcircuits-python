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
"""Target validation of ``Instruction`` and of the circuit entry points."""

import copy
import pickle
import re
import weakref

import numpy as np
import pytest

import mimiqcircuits as mc


class TwoZVars(mc.Operation):
    """A user-defined operation with two z-variable targets."""

    _name = "TwoZVars"
    _num_qubits = 0
    _num_bits = 0
    _num_zvars = 2
    _qregsizes = ()
    _cregsizes = ()
    _zregsizes = (2,)

    def iswrapper(self):
        return False


class Tagged(mc.Instruction):
    """A user subclass of Instruction that adds attributes of its own."""


def raises(exc, msg):
    return pytest.raises(exc, match=re.escape(msg))


class TestInstruction:
    def test_targets_must_be_tuples(self):
        with raises(TypeError, "Target qubits should be given in a tuple of integers."):
            mc.Instruction(mc.GateX(), [0])
        with raises(TypeError, "Target bits should be given in a tuple of integers."):
            mc.Instruction(mc.Measure(), (0,), [0])
        with raises(TypeError, "Target z-variables should be given in a tuple"):
            mc.Instruction(mc.GateX(), (0,), (), [0])

    def test_operation_type(self):
        with raises(TypeError, "Operation must be a subclass of Operation."):
            mc.Instruction("X", (0,))
        with raises(TypeError, "Operation must be a subclass of Operation."):
            mc.Instruction(mc.GateX, (0,))

    def test_duplicated_targets(self):
        with raises(ValueError, "Duplicated qubit target in instruction"):
            mc.Instruction(mc.GateCX(), (1, 1))
        with raises(ValueError, "Duplicated classical bit target in instruction"):
            mc.Instruction(mc.Detector(3), (), (0, 1, 0))
        with raises(ValueError, "Duplicated z-variables target in instruction"):
            mc.Instruction(TwoZVars(), (), (), (3, 3))

    def test_aliasing_operations_accept_repeated_bits(self):
        inst = mc.Instruction(mc.And(), (), (1, 1, 2))
        assert inst.bits == (1, 1, 2)

    def test_negative_targets(self):
        with raises(ValueError, "Qubit target index cannot be negative"):
            mc.Instruction(mc.GateX(), (-1,))
        with raises(ValueError, "Bit target index cannot be negative"):
            mc.Instruction(mc.Measure(), (0,), (-2,))
        with raises(ValueError, "Z-variable target index cannot be negative"):
            mc.Instruction(TwoZVars(), (), (), (0, -1))

    def test_wrong_counts(self):
        with raises(
            ValueError,
            "Wrong number of target qubits for operation CX wanted  2, given 1",
        ):
            mc.Instruction(mc.GateCX(), (0,))
        with raises(
            ValueError, "Wrong number of target bits for operation M wanted  1, given 0"
        ):
            mc.Instruction(mc.Measure(), (0,))
        with raises(
            ValueError,
            "Wrong number of z-targets for operation TwoZVars wanted  2, given 1",
        ):
            mc.Instruction(TwoZVars(), (), (), (0,))

    def test_user_subclass(self):
        inst = mc.Instruction(TwoZVars(), (), (), (0, 1))
        assert inst.zvars == (0, 1)
        c = mc.Circuit().push(TwoZVars(), 4, 5)
        assert c[0] == inst.__class__(TwoZVars(), (), (), (4, 5))


class TestPush:
    def test_int_targets(self):
        c = mc.Circuit()
        with raises(ValueError, "Qubit target index cannot be negative"):
            c.push(mc.GateX(), -1)
        with raises(ValueError, "Duplicated qubit target in instruction"):
            c.push(mc.GateCX(), 2, 2)
        with raises(ValueError, "Bit target index cannot be negative"):
            c.push(mc.Measure(), 0, -1)
        with raises(
            ValueError, "Wrong number of target qubits/bits/zvars: got 1, expected 2+0+0"
        ):
            c.push(mc.GateCX(), 0)
        with raises(
            ValueError, "Wrong number of target qubits/bits/zvars: got 3, expected 1+1+0"
        ):
            c.push(mc.Measure(), 0, 1, 2)
        assert len(c) == 0

    def test_iterable_targets(self):
        c = mc.Circuit()
        with raises(ValueError, "Duplicated qubit target in instruction"):
            c.push(mc.GateCX(), [0, 1], [0, 2])
        with raises(ValueError, "Qubit target index cannot be negative"):
            c.push(mc.GateH(), range(-1, 2))
        with raises(TypeError, "Invalid target type"):
            c.push(mc.GateH(), 0.5)
        with raises(TypeError, "Invalid target type"):
            c.push(mc.GateCX(), 0.5, 1)

    def test_non_operation(self):
        c = mc.Circuit()
        with raises(TypeError, "Non Operation object passed to push."):
            c.push("H", 0)
        with raises(TypeError, "Non Operation object passed to push."):
            c.push(mc.GateH, 0)
        inst = mc.Instruction(mc.GateH(), (0,))
        with raises(ValueError, "No extra arguments allowed when pushing an instruction."):
            c.push(inst, 0)

    def test_int_like_targets(self):
        c = mc.Circuit()
        c.push(mc.GateCX(), np.int64(0), 1)
        c.push(mc.GateCX(), True, 0)
        c.push(mc.GateCX(), [0, 1], 2)
        assert [i.qubits for i in c] == [(0, 1), (1, 0), (0, 2), (1, 2)]
        assert type(c[0].qubits[0]) is int

    def test_block_push_validates(self):
        b = mc.Block(2, 0, 0)
        with raises(ValueError, "Duplicated qubit target in instruction"):
            b.push(mc.GateCX(), 1, 1)
        with raises(ValueError, "Qubit target index cannot be negative"):
            b.push(mc.GateH(), -1)

    def test_insert_validates(self):
        c = mc.Circuit().push(mc.GateH(), 0)
        with raises(ValueError, "Duplicated qubit target in instruction"):
            c.insert(0, mc.GateCX(), 1, 1)
        with raises(ValueError, "Qubit target index cannot be negative"):
            c.insert(0, mc.GateH(), -3)
        assert len(c) == 1


class TestInstructionObject:
    """Instruction keeps its object behaviour: pickling, copying, weak
    references and subclassing."""

    def make(self):
        return mc.Instruction(mc.GateRX(0.25), (3,))

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_roundtrip(self, protocol):
        inst = self.make()
        back = pickle.loads(pickle.dumps(inst, protocol=protocol))
        assert back == inst
        assert str(back) == str(inst)
        c = mc.Circuit().push(mc.GateH(), 0).push(mc.Measure(), 0, 0)
        assert pickle.loads(pickle.dumps(c, protocol=protocol)) == c

    def test_copy(self):
        inst = self.make()
        shallow = copy.copy(inst)
        deep = copy.deepcopy(inst)
        assert shallow == inst and deep == inst
        assert shallow.operation is inst.operation
        assert inst.copy() == inst and inst.deepcopy() == inst

    def test_read_only_targets(self):
        inst = self.make()
        with pytest.raises(AttributeError):
            inst.qubits = (0,)
        with pytest.raises(AttributeError):
            inst.operation = mc.GateH()

    def test_weakref(self):
        inst = self.make()
        assert weakref.ref(inst)() is inst

    def test_subclass_without_slots(self):
        inst = Tagged(mc.GateH(), (0,))
        inst.tag = "mine"
        back = pickle.loads(pickle.dumps(inst))
        assert back.tag == "mine" and back == inst
        assert copy.deepcopy(inst).tag == "mine"
        assert mc.Circuit().push(inst)[0] is inst
