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


from mimiqcircuits.instruction import Instruction, _OPERATION_TYPES
from mimiqcircuits.operations.operation import Operation
import mimiqcircuits as mc
from collections.abc import Iterable
from itertools import repeat
import numpy as np


# Types already accepted by `_check_pushable`, so the check costs one set
# lookup per push.
_PUSHABLE_TYPES = {Instruction}


def _check_pushable(operation):
    # An operator that is not a gate (a projector, `SigmaMinus`, a
    # `RescaledGate`, ...) is not a physical operation, so it cannot be an
    # instruction on its own. It belongs inside `ExpectationValue`, a Kraus
    # channel, and similar.
    if isinstance(operation, mc.AbstractOperator) and not isinstance(operation, mc.Gate):
        raise ValueError(
            f"Cannot add an operator {type(operation).__name__} that is not a gate to the circuit."
        )
    _PUSHABLE_TYPES.add(type(operation))


def push_instruction_container(container, operation, *args, check_fn=None):
    """
    Shared push logic for Circuit or Block.

    Args:
        container: the target, which appends built instructions through its
            ``_append_raw`` method (Circuit also invalidates its caches there).
        operation: Operation or Instruction to push.
        args: target qubits/bits/zvars.
        check_fn: optional validator function (e.g., container._check_instruction_block).
    """
    # Fast paths for the two common calls: an already-built instruction, and
    # an operation with plain int targets, which describe exactly one
    # instruction. Anything else (numpy integers, iterables, bools, bad types)
    # takes the general path below, which also reports the errors.
    optype = type(operation)
    if optype is Instruction and not args:
        if check_fn:
            check_fn(operation)
        container._append_raw(operation)
        return container

    if optype not in _PUSHABLE_TYPES:
        _check_pushable(operation)

    if optype in _OPERATION_TYPES or (
        optype is not Instruction and isinstance(operation, Operation)
    ):
        for arg in args:
            if type(arg) is not int:
                break
        else:
            nq = operation.num_qubits
            nc = operation.num_bits
            nz = operation.num_zvars
            if len(args) != nq + nc + nz:
                raise ValueError(
                    f"Wrong number of target qubits/bits/zvars: got {len(args)}, expected {nq}+{nc}+{nz}"
                )
            inst = Instruction(
                operation, args[:nq], args[nq : nq + nc], args[nq + nc :]
            )
            if check_fn:
                check_fn(inst)
            container._append_raw(inst)
            return container

    args = tuple(int(arg) if isinstance(arg, np.integer) else arg for arg in args)
    nq = nc = nz = 0
    L = len(args)

    if isinstance(operation, Instruction):
        if L != 0:
            raise ValueError("No extra arguments allowed when pushing an instruction.")
        if check_fn:
            check_fn(operation)
        container._append_raw(operation)
        return container

    if not isinstance(operation, Operation):
        raise TypeError("Non Operation object passed to push.")

    nq = operation.num_qubits
    nc = operation.num_bits
    nz = operation.num_zvars

    if L != nq + nc + nz:
        raise ValueError(
            f"Wrong number of target qubits/bits/zvars: got {L}, expected {nq}+{nc}+{nz}"
        )

    targets = []
    hasiterable = False

    for arg in args[:-1]:
        if isinstance(arg, int):
            targets.append(repeat(arg))
        elif isinstance(arg, Iterable):
            targets.append(arg)
            hasiterable = True
        else:
            raise TypeError(f"Invalid target type: {arg}")

    if args:
        larg = args[-1]
        if isinstance(larg, int):
            targets.append(repeat(larg) if hasiterable else [larg])
        elif isinstance(larg, Iterable):
            targets.append(larg)
        else:
            raise TypeError(f"Invalid target type: {larg}")

        for tg in zip(*targets):
            inst = Instruction(
                operation, tuple(tg[:nq]), tuple(tg[nq : nq + nc]), tuple(tg[nq + nc :])
            )
            if check_fn:
                check_fn(inst)
            container._append_raw(inst)
    else:
        inst = Instruction(operation)
        if check_fn:
            check_fn(inst)
        container._append_raw(inst)

    return container


__all__ = ["push_instruction_container"]
