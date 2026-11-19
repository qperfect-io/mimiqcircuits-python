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
"""Noise model definitions and application.

A :class:`NoiseModel` is a list of noise rules. :func:`apply_noise_model`
goes through a circuit and, for each instruction, applies the first rule that
matches it. Rules are tried by priority (lower value first).

Examples:
    >>> from mimiqcircuits import *
    >>> # depolarizing noise after every CX, readout error after every measurement
    >>> model = NoiseModel()
    >>> _ = model.add_operation_noise(GateCX(), Depolarizing2(0.01))
    >>> _ = model.add_readout_noise(ReadoutErr(0.01, 0.02))
    >>> c = Circuit()
    >>> _ = c.push(GateCX(), 0, 1)
    >>> _ = c.push(Measure(), 1, 0)
    >>> apply_noise_model(c, model)
    2-qubit, 1-bit circuit with 4 instructions:
    ├── CX @ q[0], q[1]
    ├── Depolarizing(0.01) @ q[0:1]
    ├── M @ q[1], c[0]
    └── RErr(0.01, 0.02) @ c[0]
    <BLANKLINE>
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Set, Union

from mimiqcircuits.blockdecl import BlockCall
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.gatedecl import GateCall
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.gates.delay import Delay
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.magic import PrepareStateT
from mimiqcircuits.operations.measure import AbstractMeasurement
from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.parallel import Parallel
from mimiqcircuits.operations.readouterr import ReadoutErr
from mimiqcircuits.operations.repeat import Repeat
from mimiqcircuits.operations.reset import Reset, ResetX, ResetY, ResetZ
from mimiqcircuits.operations.whilestatement import WhileStatement
from mimiqcircuits.circuitrules import AbstractCircuitRule
from mimiqcircuits.symbolics import (
    _extract_variables,
    _validate_rule_gate_params,
    applyparams,
)


# Priority order (lower value = higher priority)
PRIORITY_USER_OVERRIDE = 0
PRIORITY_EXACT_OPERATION = 40
PRIORITY_EXACT_READOUT = 50
PRIORITY_SET_OPERATION = 60
PRIORITY_SET_READOUT = 70
PRIORITY_GLOBAL_OPERATION = 80
PRIORITY_GLOBAL_READOUT = 90
PRIORITY_SET_IDLE = 190
PRIORITY_IDLE = 200


def _bind_before_method(instance, before_value: bool):
    """Convert dataclass bool field `before` to bound method `before()`."""
    object.__delattr__(instance, "before")
    object.__setattr__(
        instance,
        "before",
        (lambda self=instance, value=before_value: value).__get__(
            instance, type(instance)
        ),
    )


def _is_reset(operation: Operation) -> bool:
    return isinstance(
        operation, (Reset, ResetX, ResetY, ResetZ, PrepareStateT)
    )


def _supports_symbolic_operation_pattern(operation: Operation) -> bool:
    return isinstance(operation, (Gate, AbstractMeasurement)) or _is_reset(
        operation
    )


def _is_symbolic_operation_pattern(operation: Operation) -> bool:
    return _supports_symbolic_operation_pattern(operation) and operation.is_symbolic()


def _validate_rule_operation_target(operation: Operation):
    if not isinstance(
        operation,
        (
            Gate,
            AbstractMeasurement,
            Block,
            Repeat,
            IfStatement,
            WhileStatement,
            Reset,
            ResetX,
            ResetY,
            ResetZ,
            PrepareStateT,
        ),
    ):
        raise ValueError(
            "Rule target operation must be a gate, measurement, reset, "
            "Block, Repeat, IfStatement, or WhileStatement operation."
        )


def _resolve_noise_for_operation_match(
    op_pattern: Operation,
    op_instance: Operation,
    noise_pattern: Union[krauschannel, Gate],
):
    """Build noise op from a matched operation instance."""
    # Non-symbolic patterns (including wrappers) are static.
    if not _is_symbolic_operation_pattern(op_pattern):
        return noise_pattern

    variables = _extract_variables(op_pattern)
    if variables is None or all(var is None for var in variables):
        return noise_pattern

    return applyparams(op_instance, (variables, noise_pattern))


class AbstractNoiseRule(AbstractCircuitRule):
    """Base class for the rules of a :class:`NoiseModel`.

    A rule says which instructions it matches (``matches``), what to emit for
    a match (``apply_rule``), where to put it (``before``, ``replaces``) and
    how early it is tried (``priority``, lower first).
    """


@dataclass(frozen=True)
class GlobalReadoutNoise(AbstractNoiseRule):
    """Add a readout error after every measurement.

    The readout error acts on the classical bits the measurement writes to,
    so it is placed right after the measurement and does not replace it.

    Args:
        noise: The ``ReadoutErr`` to apply.

    Examples:
        >>> from mimiqcircuits import *
        >>> # every measurement gets a 1% / 2% readout error
        >>> GlobalReadoutNoise(ReadoutErr(0.01, 0.02))
        GlobalReadoutNoise(noise=RErr(0.01,0.02))
    """
    noise: ReadoutErr

    def priority(self):
        return PRIORITY_GLOBAL_READOUT

    def matches(self, inst: Instruction):
        return isinstance(inst.get_operation(), AbstractMeasurement)

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        return Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class ExactQubitReadoutNoise(AbstractNoiseRule):
    """Add a readout error after measurements on exactly the given qubits.

    A measurement matches only when its qubits are ``qubits``, in the same
    order. The readout error acts on the bits the measurement writes to.

    Args:
        qubits: Qubits the measurement must act on, in order.
        noise: The ``ReadoutErr`` to apply.

    Raises:
        ValueError: If ``qubits`` is empty or has repetitions.

    Examples:
        >>> from mimiqcircuits import *
        >>> # only a measurement of exactly (q0, q1), in this order, matches
        >>> ExactQubitReadoutNoise([0, 1], ReadoutErr(0.01, 0.02))
        ExactQubitReadoutNoise(qubits=(0, 1), noise=RErr(0.01,0.02))
    """
    qubits: Sequence[int]
    noise: ReadoutErr

    def __post_init__(self):
        qs = list(self.qubits)
        if not qs:
            raise ValueError("Qubit list must not be empty")
        if len(set(qs)) != len(qs):
            raise ValueError("Qubit list must not contain repetitions")
        object.__setattr__(self, "qubits", tuple(qs))

    def priority(self):
        return PRIORITY_EXACT_READOUT

    def matches(self, inst: Instruction):
        return isinstance(inst.get_operation(), AbstractMeasurement) and tuple(
            inst.get_qubits()
        ) == self.qubits

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        return Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class SetQubitReadoutNoise(AbstractNoiseRule):
    """Add a readout error after measurements whose qubits are all in a set.

    The readout error acts on the bits the measurement writes to.

    Args:
        qubits: Qubits where the readout error applies.
        noise: The ``ReadoutErr`` to apply.

    Raises:
        ValueError: If ``qubits`` is empty or has repetitions.

    Examples:
        >>> from mimiqcircuits import *
        >>> # measurements of qubits 0, 2 or 4 get the readout error
        >>> SetQubitReadoutNoise([0, 2, 4], ReadoutErr(0.01, 0.02))
        SetQubitReadoutNoise(qubits=[0, 2, 4], noise=RErr(0.01,0.02))
    """
    qubits: Iterable[int]
    noise: ReadoutErr
    _qubit_set: frozenset = field(init=False, repr=False)

    def __post_init__(self):
        qs = list(self.qubits)
        if not qs:
            raise ValueError("Qubit list must not be empty")
        if len(set(qs)) != len(qs):
            raise ValueError("Qubit list must not contain repetitions")
        object.__setattr__(self, "_qubit_set", frozenset(qs))

    def priority(self):
        return PRIORITY_SET_READOUT

    def matches(self, inst: Instruction):
        return isinstance(inst.get_operation(), AbstractMeasurement) and all(
            q in self._qubit_set for q in inst.get_qubits()
        )

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        return Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class OperationInstanceNoise(AbstractNoiseRule):
    """Add noise to every instance of an operation.

    An instruction matches when its operation has the same type as
    ``operation``. If ``operation`` has symbolic parameters, such as
    ``GateRX(a)``, any parameter value matches and the symbols in ``noise``
    are replaced by the values of the matched instance. Otherwise the
    operation must be equal to ``operation``.

    Args:
        operation: Operation to match: a gate, measurement, reset, ``Block``,
            ``Repeat``, ``IfStatement`` or ``WhileStatement``. Symbolic
            parameters must be plain symbols.
        noise: Kraus channel or gate to add, on the same number of qubits as
            ``operation``.
        before: Add the noise before the operation instead of after it.
        replace: Replace the operation with the noise.

    Raises:
        ValueError: If ``operation`` is of an unsupported kind, if ``noise``
            acts on a different number of qubits, or if both ``before`` and
            ``replace`` are set.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols, pi
        >>> # the noise strength grows with the rotation angle
        >>> a = symbols("a")
        >>> rule = OperationInstanceNoise(GateRX(a), Depolarizing1(a / pi))
        >>> c = Circuit().push(GateRX(pi / 2), 0)
        >>> NoiseModel([rule]).apply_noise_model(c)
        1-qubit circuit with 2 instructions:
        ├── RX((1/2)*pi) @ q[0]
        └── Depolarizing(0.5) @ q[0]
        <BLANKLINE>
    """
    operation: Operation
    noise: Union[krauschannel, Gate]
    before: bool = False
    replace: bool = False

    def __post_init__(self):
        _validate_rule_operation_target(self.operation)

        if _supports_symbolic_operation_pattern(self.operation):
            _validate_rule_gate_params(self.operation)

        if self.operation.num_qubits != self.noise.num_qubits:
            raise ValueError(
                "Noise operation must act on the same number of qubits as "
                "the operation instance"
            )

        if self.before and self.replace:
            raise ValueError("Cannot set both before=True and replace=True")

        _bind_before_method(self, self.before)

    def priority(self):
        return PRIORITY_GLOBAL_OPERATION

    def replaces(self):
        return self.replace

    def matches(self, inst: Instruction):
        op_inst = inst.get_operation()
        op_rule = self.operation

        if type(op_inst) is not type(op_rule):
            return False

        if not _is_symbolic_operation_pattern(op_rule):
            return op_inst == op_rule

        return True

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()
        noise_op = _resolve_noise_for_operation_match(self.operation, op_inst, self.noise)
        return Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class ExactOperationInstanceQubitNoise(AbstractNoiseRule):
    """Add noise to an operation acting on exactly the given qubits.

    Matches like ``OperationInstanceNoise``, and also requires the
    instruction's qubits to be ``qubits``, in the same order.

    Args:
        operation: Operation to match (see ``OperationInstanceNoise``).
        qubits: Qubits the operation must act on, in order.
        noise: Kraus channel or gate to add, on the same number of qubits as
            ``operation``.
        before: Add the noise before the operation instead of after it.
        replace: Replace the operation with the noise.

    Raises:
        ValueError: If ``operation`` is of an unsupported kind, if ``qubits``
            has repetitions or a length other than the number of qubits of
            ``operation``, if ``noise`` acts on a different number of qubits,
            or if both ``before`` and ``replace`` are set.

    Examples:
        >>> from mimiqcircuits import *
        >>> # only the CX with control 0 and target 1 gets noise
        >>> rule = ExactOperationInstanceQubitNoise(GateCX(), [0, 1], Depolarizing2(0.01))
        >>> c = Circuit().push(GateCX(), 0, 1).push(GateCX(), 1, 0)
        >>> NoiseModel([rule]).apply_noise_model(c)
        2-qubit circuit with 3 instructions:
        ├── CX @ q[0], q[1]
        ├── Depolarizing(0.01) @ q[0:1]
        └── CX @ q[1], q[0]
        <BLANKLINE>
    """
    operation: Operation
    qubits: Sequence[int]
    noise: Union[krauschannel, Gate]
    before: bool = False
    replace: bool = False

    def __post_init__(self):
        _validate_rule_operation_target(self.operation)

        if _supports_symbolic_operation_pattern(self.operation):
            _validate_rule_gate_params(self.operation)

        qs = list(self.qubits)
        if len(set(qs)) != len(qs):
            raise ValueError("Qubit list must not contain repetitions")
        if len(qs) != self.operation.num_qubits:
            raise ValueError(
                "Qubit list length must match the number of qubits the operation acts on"
            )
        if self.operation.num_qubits != self.noise.num_qubits:
            raise ValueError(
                "Noise operation must act on the same number of qubits as "
                "the operation instance"
            )
        if self.before and self.replace:
            raise ValueError("Cannot set both before=True and replace=True")

        object.__setattr__(self, "qubits", tuple(qs))
        _bind_before_method(self, self.before)

    def priority(self):
        return PRIORITY_EXACT_OPERATION

    def replaces(self):
        return self.replace

    def matches(self, inst: Instruction):
        op_inst = inst.get_operation()
        op_rule = self.operation

        if type(op_inst) is not type(op_rule):
            return False

        if tuple(inst.get_qubits()) != self.qubits:
            return False

        if not _is_symbolic_operation_pattern(op_rule):
            return op_inst == op_rule

        return True

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()
        noise_op = _resolve_noise_for_operation_match(self.operation, op_inst, self.noise)
        return Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class SetOperationInstanceQubitNoise(AbstractNoiseRule):
    """Add noise to an operation whose qubits are all in a set.

    Matches like ``OperationInstanceNoise``, and also requires every qubit
    of the instruction to be in ``qubits``.

    Args:
        operation: Operation to match (see ``OperationInstanceNoise``).
        qubits: Qubits where the noise applies.
        noise: Kraus channel or gate to add, on the same number of qubits as
            ``operation``.
        before: Add the noise before the operation instead of after it.
        replace: Replace the operation with the noise.

    Raises:
        ValueError: If ``operation`` is of an unsupported kind, if ``qubits``
            has repetitions or fewer qubits than ``operation``, if ``noise``
            acts on a different number of qubits, or if both ``before`` and
            ``replace`` are set.

    Examples:
        >>> from mimiqcircuits import *
        >>> # H on qubit 0 or 1 gets noise, H on qubit 2 does not
        >>> rule = SetOperationInstanceQubitNoise(GateH(), [0, 1], AmplitudeDamping(0.001))
        >>> c = Circuit().push(GateH(), 0).push(GateH(), 2)
        >>> NoiseModel([rule]).apply_noise_model(c)
        3-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── AmplitudeDamping(0.001) @ q[0]
        └── H @ q[2]
        <BLANKLINE>
    """
    operation: Operation
    qubits: Iterable[int]
    noise: Union[krauschannel, Gate]
    before: bool = False
    replace: bool = False
    _qubit_set: frozenset = field(init=False, repr=False)

    def __post_init__(self):
        _validate_rule_operation_target(self.operation)

        if _supports_symbolic_operation_pattern(self.operation):
            _validate_rule_gate_params(self.operation)

        qs = list(self.qubits)
        if len(qs) < self.operation.num_qubits:
            raise ValueError(
                "Qubit set must contain at least as many qubits as the operation acts on"
            )
        if len(qs) != len(set(qs)):
            raise ValueError("Qubit set must not contain repetitions")
        if self.operation.num_qubits != self.noise.num_qubits:
            raise ValueError(
                "Noise operation must act on the same number of qubits as "
                "the operation instance"
            )
        if self.before and self.replace:
            raise ValueError("Cannot set both before=True and replace=True")

        object.__setattr__(self, "_qubit_set", frozenset(qs))
        _bind_before_method(self, self.before)

    def priority(self):
        return PRIORITY_SET_OPERATION

    def replaces(self):
        return self.replace

    def matches(self, inst: Instruction):
        op_inst = inst.get_operation()
        op_rule = self.operation

        if type(op_inst) is not type(op_rule):
            return False

        if not all(q in self._qubit_set for q in inst.get_qubits()):
            return False

        if not _is_symbolic_operation_pattern(op_rule):
            return op_inst == op_rule

        return True

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()
        noise_op = _resolve_noise_for_operation_match(self.operation, op_inst, self.noise)
        return Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class IdleNoise(AbstractNoiseRule):
    """Replace every ``Delay`` with a noise operation.

    Idle time is marked explicitly with ``Delay(t)`` instructions; qubits
    that are simply not acted upon get no noise. Each matching ``Delay`` is
    replaced by the noise on the same qubits.

    Args:
        relation: Either a constant noise operation, or a tuple
            ``(t, noise(t))`` where ``t`` is a symbol. In the second form ``t``
            is replaced by the duration of each ``Delay``.

    Raises:
        ValueError: If ``relation`` is neither an operation nor a
            ``(symbol, operation)`` tuple.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols
        >>> # a Delay(t) becomes amplitude damping with gamma = t / 1000
        >>> t = symbols("t")
        >>> rule = IdleNoise((t, AmplitudeDamping(t / 1000)))
        >>> c = Circuit().push(GateH(), 0).push(Delay(10), 1)
        >>> NoiseModel([rule]).apply_noise_model(c)
        2-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[1]
        <BLANKLINE>
    """
    relation: Union[tuple, Operation]

    def __post_init__(self):
        if isinstance(self.relation, Operation):
            return

        if not isinstance(self.relation, tuple) or len(self.relation) != 2:
            raise ValueError("IdleNoise relation must be (var, target) or Operation")

        var, target = self.relation

        if not (hasattr(var, "is_symbol") and var.is_symbol):
            raise ValueError(
                f"Left side must be a simple symbolic variable, got: {var}"
            )
        if not isinstance(target, Operation):
            raise ValueError("Right side of idle relation must be an Operation")

    def priority(self):
        return PRIORITY_IDLE

    def replaces(self):
        return True

    def matches(self, inst: Instruction):
        return isinstance(inst.get_operation(), Delay)

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()

        if isinstance(self.relation, Operation):
            noise_op = self.relation
        else:
            var, target = self.relation
            noise_op = applyparams(op_inst, ((var,), target))

        return Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class SetIdleQubitNoise(AbstractNoiseRule):
    """Replace every ``Delay`` on qubits in a set with a noise operation.

    Like ``IdleNoise``, restricted to ``Delay`` instructions whose qubits are
    all in ``qubits``.

    Args:
        relation: Constant noise operation or ``(t, noise(t))`` tuple (see
            ``IdleNoise``).
        qubits: Qubits where the noise applies.

    Raises:
        ValueError: If ``qubits`` is empty, or ``relation`` is neither an
            operation nor a ``(symbol, operation)`` tuple.

    Examples:
        >>> from mimiqcircuits import *
        >>> # only delays on qubit 0 become noise
        >>> rule = SetIdleQubitNoise(AmplitudeDamping(1e-4), [0])
        >>> c = Circuit().push(Delay(10), 0).push(Delay(10), 1)
        >>> NoiseModel([rule]).apply_noise_model(c)
        2-qubit circuit with 2 instructions:
        ├── AmplitudeDamping(0.0001) @ q[0]
        └── Delay(10) @ q[1]
        <BLANKLINE>
    """
    relation: Union[tuple, Operation]
    qubits: Iterable[int]

    def __post_init__(self):
        qs = list(self.qubits)
        if not qs:
            raise ValueError("Qubit set must not be empty")

        object.__setattr__(self, "qubits", frozenset(qs))

        if isinstance(self.relation, Operation):
            return

        if not isinstance(self.relation, tuple) or len(self.relation) != 2:
            raise ValueError("IdleNoise relation must be (var, target) or Operation")

        var, target = self.relation
        if not (hasattr(var, "is_symbol") and var.is_symbol):
            raise ValueError("Left side must be symbolic variable")
        if not isinstance(target, Operation):
            raise ValueError("Right side must be Operation")

    def priority(self):
        return PRIORITY_SET_IDLE

    def replaces(self):
        return True

    def matches(self, inst: Instruction):
        op = inst.get_operation()
        return isinstance(op, Delay) and all(q in self.qubits for q in inst.get_qubits())

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()

        if isinstance(self.relation, Operation):
            noise_op = self.relation
        else:
            var, target = self.relation
            noise_op = applyparams(op_inst, ((var,), target))

        return Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class CustomNoiseRule(AbstractNoiseRule):
    """Add noise chosen by user-supplied functions.

    Args:
        matcher: Function ``inst -> bool`` that selects the instructions.
        generator: Function ``inst -> Instruction`` that builds the noise
            instruction for a matched instruction, with the circuit's qubit
            and bit indices.
        priority_val: Priority of the rule; lower runs first. Defaults to
            ``PRIORITY_USER_OVERRIDE``, ahead of every built-in rule.
        before: Add the noise before the instruction instead of after it.
        replace: Replace the instruction with the noise.

    A noise model containing a ``CustomNoiseRule`` cannot be saved with
    ``saveproto``: the matcher and generator are Python callables with no
    serialized form.

    Examples:
        >>> from mimiqcircuits import *
        >>> # two-qubit depolarizing noise after every two-qubit operation
        >>> rule = CustomNoiseRule(
        ...     lambda inst: inst.get_operation().num_qubits == 2,
        ...     lambda inst: Instruction(Depolarizing2(0.01), inst.get_qubits()),
        ... )
        >>> c = Circuit().push(GateCX(), 0, 1)
        >>> NoiseModel([rule]).apply_noise_model(c)
        2-qubit circuit with 2 instructions:
        ├── CX @ q[0], q[1]
        └── Depolarizing(0.01) @ q[0:1]
        <BLANKLINE>
    """

    matcher: Callable[[Instruction], bool]
    generator: Callable[[Instruction], Instruction]
    priority_val: int = PRIORITY_USER_OVERRIDE
    before: bool = False
    replace: bool = False

    def __post_init__(self):
        _bind_before_method(self, self.before)

    def priority(self):
        return self.priority_val

    def matches(self, inst: Instruction):
        return self.matcher(inst)

    def apply_rule(self, inst: Instruction):
        return self.generator(inst) if self.matcher(inst) else None

    def replaces(self):
        return self.replace


@dataclass
class NoiseModel:
    """Collection of noise rules applied to circuits.

    :func:`apply_noise_model` goes through a circuit and gives each
    instruction to the rules in order; the first rule that matches adds its
    noise (after the instruction, before it, or in its place) and the others
    are skipped. Instructions that no rule matches are kept as they are.

    Rules are kept sorted by ``priority()``, lower first, and rules of equal
    priority keep the order they were added in. The built-in rules make
    specific rules win over general ones:

    =========================================  ========
    Rule                                       Priority
    =========================================  ========
    :class:`CustomNoiseRule` (default)         0
    :class:`ExactOperationInstanceQubitNoise`  40
    :class:`ExactQubitReadoutNoise`            50
    :class:`SetOperationInstanceQubitNoise`    60
    :class:`SetQubitReadoutNoise`              70
    :class:`OperationInstanceNoise`            80
    :class:`GlobalReadoutNoise`                90
    :class:`SetIdleQubitNoise`                 190
    :class:`IdleNoise`                         200
    =========================================  ========

    The ``add_*`` methods build these rules for you and return the model, so
    calls can be chained.

    Args:
        rules: Initial rules. The list is sorted in place.
        name: Name shown by :meth:`describe`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols, pi
        >>> theta = symbols("theta")
        >>> model = NoiseModel(name="demo")
        >>> # RX(theta) is followed by depolarizing noise of strength theta/pi
        >>> _ = model.add_operation_noise(GateRX(theta), Depolarizing1(theta / pi))
        >>> # every CX is followed by two-qubit depolarizing noise
        >>> _ = model.add_operation_noise(GateCX(), Depolarizing2(0.01))
        >>> # every measured bit gets a readout error
        >>> _ = model.add_readout_noise(ReadoutErr(0.01, 0.02))
        >>> # every Delay becomes amplitude damping
        >>> _ = model.add_idle_noise(AmplitudeDamping(0.001))
        >>> c = Circuit()
        >>> _ = c.push(GateRX(0.4), 0)
        >>> _ = c.push(GateCX(), 0, 1)
        >>> _ = c.push(Delay(5), 1)
        >>> _ = c.push(Measure(), 0, 0)
        >>> model.apply_noise_model(c)
        2-qubit, 1-bit circuit with 7 instructions:
        ├── RX(0.4) @ q[0]
        ├── Depolarizing(0.4/pi) @ q[0]
        ├── CX @ q[0], q[1]
        ├── Depolarizing(0.01) @ q[0:1]
        ├── AmplitudeDamping(0.001) @ q[1]
        ├── M @ q[0], c[0]
        └── RErr(0.01, 0.02) @ c[0]
        <BLANKLINE>

        >>> # rules can also be given directly
        >>> model = NoiseModel(
        ...     [
        ...         ExactOperationInstanceQubitNoise(GateCX(), [0, 1], Depolarizing(2, 0.01)),
        ...         GlobalReadoutNoise(ReadoutErr(0.01, 0.02)),
        ...     ],
        ...     name="explicit",
        ... )
    """

    rules: List[AbstractNoiseRule] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        self.rules.sort(key=lambda r: r.priority())

    def add_rule(self, rule: AbstractNoiseRule):
        """Add a rule and return the model.

        The rule is placed after the existing rules of the same priority.

        Args:
            rule: Any :class:`AbstractNoiseRule`.

        Returns:
            NoiseModel: This model, changed in place.
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority())
        return self

    def add_readout_noise(self, noise: ReadoutErr, *, qubits=None, exact=False):
        """Add a rule that puts a readout error after measurements.

        Without ``qubits`` the rule is a :class:`GlobalReadoutNoise`. With
        ``qubits`` it is a :class:`SetQubitReadoutNoise`, or an
        :class:`ExactQubitReadoutNoise` when ``exact`` is ``True``.

        Args:
            noise: Readout error to add.
            qubits: Keyword-only. Restrict the rule to measurements on these
                qubits.
            exact: Keyword-only. Require the measured qubits to equal
                ``qubits``, in order, instead of being a subset of them.

        Returns:
            NoiseModel: This model, changed in place.

        Raises:
            ValueError: If ``qubits`` is empty or has repetitions.

        Examples:
            >>> from mimiqcircuits import *
            >>> model = NoiseModel()
            >>> # all measurements
            >>> model.add_readout_noise(ReadoutErr(0.01, 0.02))
            NoiseModel(rules=[GlobalReadoutNoise(noise=RErr(0.01,0.02))], name='')
            >>> # measurements on qubits 0 and 2 only; tried before the global rule
            >>> model.add_readout_noise(ReadoutErr(0.03, 0.04), qubits=[0, 2])
            NoiseModel(rules=[SetQubitReadoutNoise(qubits=(0, 2), noise=RErr(0.03,0.04)), GlobalReadoutNoise(noise=RErr(0.01,0.02))], name='')
            >>> # a measurement of exactly (q2, q0); tried first
            >>> model.add_readout_noise(ReadoutErr(0.05, 0.06), qubits=[2, 0], exact=True)
            NoiseModel(rules=[ExactQubitReadoutNoise(qubits=(2, 0), noise=RErr(0.05,0.06)), SetQubitReadoutNoise(qubits=(0, 2), noise=RErr(0.03,0.04)), GlobalReadoutNoise(noise=RErr(0.01,0.02))], name='')
        """
        if qubits is None:
            rule = GlobalReadoutNoise(noise)
        elif exact:
            rule = ExactQubitReadoutNoise(tuple(qubits), noise)
        else:
            rule = SetQubitReadoutNoise(tuple(qubits), noise)
        return self.add_rule(rule)

    def add_operation_noise(
        self, operation, noise, *, qubits=None, exact=False, before=False, replace=False
    ):
        """Add a rule that puts noise next to an operation.

        Without ``qubits`` the rule is an :class:`OperationInstanceNoise`.
        With ``qubits`` it is a :class:`SetOperationInstanceQubitNoise`, or an
        :class:`ExactOperationInstanceQubitNoise` when ``exact`` is ``True``.

        A concrete ``operation`` matches only equal operations. One with
        symbolic parameters matches any operation of the same type, and its
        symbols are replaced in ``noise`` by the values of the matched
        operation.

        Args:
            operation: Operation pattern: a gate, measurement, reset,
                :class:`Block`, :class:`Repeat`, :class:`IfStatement` or
                :class:`WhileStatement`.
            noise: Kraus channel or gate to add, on as many qubits as
                ``operation``.
            qubits: Keyword-only. Restrict the rule to instructions on these
                qubits.
            exact: Keyword-only. Require the instruction's qubits to equal
                ``qubits``, in order, instead of being a subset of them.
            before: Keyword-only. Add the noise before the operation.
            replace: Keyword-only. Replace the operation with the noise.

        Returns:
            NoiseModel: This model, changed in place.

        Raises:
            ValueError: If ``operation`` is not a supported kind, ``noise``
                acts on a different number of qubits, ``qubits`` is invalid,
                or both ``before`` and ``replace`` are set.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import symbols, pi
            >>> theta, alpha, beta = symbols("theta alpha beta")
            >>> model = NoiseModel()
            >>> # only RX(pi/2) matches
            >>> _ = model.add_operation_noise(GateRX(pi / 2), AmplitudeDamping(0.001))
            >>> # any RX matches; theta in the noise takes the gate's angle
            >>> _ = model.add_operation_noise(GateRX(theta), Depolarizing(1, theta / pi))
            >>> # several parameters at once
            >>> _ = model.add_operation_noise(
            ...     GateU(alpha, beta, 0),
            ...     Depolarizing(1, (alpha**2 + beta**2) / (2 * pi**2)),
            ... )
            >>> # restricted to RX on qubits 0, 1 or 2
            >>> _ = model.add_operation_noise(
            ...     GateRX(theta),
            ...     Depolarizing(1, theta / pi),
            ...     qubits=[0, 1, 2],
            ... )
            >>> # restricted to RX on exactly qubit 0
            >>> _ = model.add_operation_noise(
            ...     GateRX(theta),
            ...     Depolarizing(1, theta / pi),
            ...     qubits=[0],
            ...     exact=True,
            ... )
            >>> # a bit flip just before every measurement
            >>> _ = model.add_operation_noise(Measure(), PauliX(0.02), before=True)
            >>> _ = model.add_operation_noise(Reset(), Depolarizing(1, 0.01))
            >>> # H is removed and only the noise is left
            >>> _ = model.add_operation_noise(GateH(), AmplitudeDamping(0.001), replace=True)
        """
        _validate_rule_operation_target(operation)
        if qubits is None:
            rule = OperationInstanceNoise(
                operation, noise, before=before, replace=replace
            )
        elif exact:
            rule = ExactOperationInstanceQubitNoise(
                operation, tuple(qubits), noise, before=before, replace=replace
            )
        else:
            rule = SetOperationInstanceQubitNoise(
                operation, tuple(qubits), noise, before=before, replace=replace
            )
        return self.add_rule(rule)

    def add_idle_noise(self, noise, qubits=None):
        """Add a rule that replaces ``Delay`` instructions with noise.

        Without ``qubits`` the rule is an :class:`IdleNoise`, otherwise a
        :class:`SetIdleQubitNoise`. Only explicit :class:`Delay` instructions
        are affected; qubits that are simply not acted upon get no noise.

        Args:
            noise: A noise operation, or a tuple ``(t, noise(t))`` where ``t``
                is a symbol replaced by the duration of each ``Delay``.
            qubits: Restrict the rule to delays on these qubits.

        Returns:
            NoiseModel: This model, changed in place.

        Raises:
            ValueError: If ``noise`` is neither an operation nor a
                ``(symbol, operation)`` tuple, or ``qubits`` is empty.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import symbols
            >>> t = symbols("t")
            >>> model = NoiseModel()
            >>> # the same noise for every delay
            >>> model.add_idle_noise(AmplitudeDamping(1e-4))
            NoiseModel(rules=[IdleNoise(relation=AmplitudeDamping(0.0001))], name='')
            >>> # noise depending on the delay duration, on qubits 0 to 2
            >>> model.add_idle_noise((t, AmplitudeDamping(t / 1000)), qubits=[0, 1, 2])
            NoiseModel(rules=[SetIdleQubitNoise(relation=(t, AmplitudeDamping((1/1000)*t)), qubits=frozenset({0, 1, 2})), IdleNoise(relation=AmplitudeDamping(0.0001))], name='')
        """
        if qubits is None:
            rule = IdleNoise(noise)
        else:
            rule = SetIdleQubitNoise(noise, qubits)
        return self.add_rule(rule)

    def apply_noise_model(self, circuit: Circuit):
        """Return a noisy copy of ``circuit``; see :func:`apply_noise_model`."""
        return apply_noise_model(circuit, self)

    def describe(self):
        """Print a readable summary of the rules, in the order they are tried.

        Examples:
            >>> from mimiqcircuits import *
            >>> NoiseModel(name="demo").add_idle_noise(AmplitudeDamping(0.001)).describe()
            NoiseModel: demo
            ================================================================================
            Rule 1 (priority 200): IdleNoise
              -> Applies AmplitudeDamping(0.001) to all idle qubits
                (REPLACES the matched operation)
            --------------------------------------------------------------------------------
        """
        title = f"NoiseModel: {self.name}" if self.name else "NoiseModel"
        print(title)
        print("=" * 80)

        for i, rule in enumerate(self.rules, 1):
            print(f"Rule {i} (priority {rule.priority()}): {type(rule).__name__}")

            if isinstance(rule, GlobalReadoutNoise):
                print(f"  -> Applies {rule.noise} to all measurements")
            elif isinstance(rule, ExactQubitReadoutNoise):
                print(
                    f"  -> Applies {rule.noise} to measurements on qubits "
                    f"{list(rule.qubits)} (exact order)"
                )
            elif isinstance(rule, SetQubitReadoutNoise):
                print(
                    f"  -> Applies {rule.noise} to measurements on qubits in "
                    f"{sorted(rule._qubit_set)}"
                )
            elif isinstance(rule, OperationInstanceNoise):
                print(
                    f"  -> Applies {rule.noise} to operation instances matching "
                    f"{rule.operation}"
                )
            elif isinstance(rule, ExactOperationInstanceQubitNoise):
                print(
                    f"  -> Applies {rule.noise} to {rule.operation} on qubits "
                    f"{list(rule.qubits)} (exact order)"
                )
            elif isinstance(rule, SetOperationInstanceQubitNoise):
                print(
                    f"  -> Applies {rule.noise} to {rule.operation} on qubits in "
                    f"{sorted(rule._qubit_set)}"
                )
            elif isinstance(rule, IdleNoise):
                if isinstance(rule.relation, Operation):
                    print(f"  -> Applies {rule.relation} to all idle qubits")
                else:
                    var, target = rule.relation
                    print(
                        f"  -> Applies time-dependent idle noise {target} with "
                        f"variable {var} to all idle qubits"
                    )
            elif isinstance(rule, SetIdleQubitNoise):
                if isinstance(rule.relation, Operation):
                    print(f"  -> Applies {rule.relation} to idle qubits {sorted(rule.qubits)}")
                else:
                    var, target = rule.relation
                    print(
                        f"  -> Applies time-dependent idle noise {target} with "
                        f"variable {var} to idle qubits {sorted(rule.qubits)}"
                    )
            elif isinstance(rule, CustomNoiseRule):
                print("  -> Custom rule (user-defined matcher)")

            if rule.before():
                print("    (applied BEFORE the operation)")
            if rule.replaces():
                print("    (REPLACES the matched operation)")

            print("-" * 80)

    def saveproto(self, file):
        """Save the model to a protobuf file.

        Args:
            file: Path or writable binary file object.

        Raises:
            ValueError: If the model contains a :class:`CustomNoiseRule`,
                which cannot be serialized.
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load a model saved with :meth:`saveproto`.

        Args:
            file: Path or readable binary file object.

        Returns:
            NoiseModel: The loaded model.
        """
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, NoiseModel)


def _canonical_targets(op: Operation):
    return (
        tuple(range(op.num_qubits)),
        tuple(range(op.num_bits)),
        tuple(range(op.num_zvars)),
    )


def _collapse_local_instructions_to_operation(
    instructions: List[Instruction], op: Operation
):
    qcanon, bcanon, zcanon = _canonical_targets(op)

    if (
        len(instructions) == 1
        and instructions[0].get_qubits() == qcanon
        and instructions[0].get_bits() == bcanon
        and instructions[0].get_zvars() == zcanon
    ):
        return instructions[0].get_operation()

    return Block(op.num_qubits, op.num_bits, op.num_zvars, instructions)


def _apply_rules_to_instruction(inst: Instruction, model: NoiseModel):
    for rule in model.rules:
        noise_inst = rule.apply_rule(inst)
        if noise_inst is None:
            continue

        if rule.replaces():
            return [noise_inst], True
        if rule.before():
            return [noise_inst, inst], True
        return [inst, noise_inst], True

    return [inst], False


def _rewrite_nested_operation(
    op: Operation, model: NoiseModel, active_decls: Set[object]
):
    if isinstance(op, Block):
        noisy_instructions = _apply_noise_to_instructions(
            op.instructions, model, active_decls
        )
        if noisy_instructions == op.instructions:
            return op
        return Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)

    if isinstance(op, IfStatement):
        inner = op.get_operation()
        qcanon, bcanon, zcanon = _canonical_targets(inner)
        inner_inst = Instruction(inner, qcanon, bcanon, zcanon)

        noisy_inner = _apply_noise_to_instruction(inner_inst, model, active_decls)
        rewritten_inner = _collapse_local_instructions_to_operation(noisy_inner, inner)

        if rewritten_inner == inner:
            return op

        return IfStatement(rewritten_inner, op.get_bitstring())

    if isinstance(op, WhileStatement):
        inner = op.get_operation()
        qcanon, bcanon, zcanon = _canonical_targets(inner)
        inner_inst = Instruction(inner, qcanon, bcanon, zcanon)

        noisy_inner = _apply_noise_to_instruction(inner_inst, model, active_decls)
        rewritten_inner = _collapse_local_instructions_to_operation(noisy_inner, inner)

        if rewritten_inner == inner:
            return op

        return WhileStatement(rewritten_inner, op.get_bitstring())

    if isinstance(op, Parallel):
        inner = op.get_operation()
        nq = inner.num_qubits
        nb = inner.num_bits
        nz = inner.num_zvars

        repeated_instructions = []
        for rep in range(op.num_repeats):
            qtargets = tuple(nq * rep + i for i in range(nq))
            btargets = tuple(nb * rep + i for i in range(nb))
            ztargets = tuple(nz * rep + i for i in range(nz))
            repeated_instructions.append(
                Instruction(inner, qtargets, btargets, ztargets)
            )

        noisy_instructions = _apply_noise_to_instructions(
            repeated_instructions, model, active_decls
        )

        if noisy_instructions == repeated_instructions:
            return op

        return Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)

    if isinstance(op, Repeat):
        inner = op.get_operation()
        qcanon, bcanon, zcanon = _canonical_targets(inner)
        repeated_instructions = [
            Instruction(inner, qcanon, bcanon, zcanon) for _ in range(op.repeats)
        ]

        noisy_instructions = _apply_noise_to_instructions(
            repeated_instructions, model, active_decls
        )
        if noisy_instructions == repeated_instructions:
            return op

        return Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)

    if isinstance(op, (GateCall, BlockCall)):
        decl = op.decl

        # Prevent infinite recursion for self-referential declarations.
        if decl in active_decls:
            return op

        active_decls.add(decl)
        try:
            substitutions = dict(zip(decl.arguments, op.arguments))
            expanded_instructions = []
            for inst in (decl.circuit if isinstance(op, GateCall) else decl):
                expanded_op = inst.get_operation().evaluate(substitutions)
                expanded_instructions.append(
                    Instruction(
                        expanded_op, inst.get_qubits(), inst.get_bits(), inst.get_zvars()
                    )
                )

            noisy_instructions = _apply_noise_to_instructions(
                expanded_instructions, model, active_decls
            )

            if noisy_instructions == expanded_instructions:
                return op

            return Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)
        finally:
            active_decls.remove(decl)

    return op


def _apply_noise_to_instruction(
    inst: Instruction, model: NoiseModel, active_decls: Set[object]
):
    rewritten_instructions, matched = _apply_rules_to_instruction(inst, model)
    if matched:
        return rewritten_instructions

    op = inst.get_operation()
    rewritten_op = _rewrite_nested_operation(op, model, active_decls)

    if rewritten_op is op:
        return rewritten_instructions

    return [
        Instruction(
            rewritten_op, inst.get_qubits(), inst.get_bits(), inst.get_zvars()
        )
    ]


def _apply_noise_to_instructions(
    instructions: List[Instruction], model: NoiseModel, active_decls: Set[object]
):
    noisy_instructions = []
    for inst in instructions:
        noisy_instructions.extend(_apply_noise_to_instruction(inst, model, active_decls))
    return noisy_instructions


def apply_noise_model(circuit: Circuit, model: NoiseModel):
    """Return a copy of ``circuit`` with the noise of ``model`` added.

    Each instruction is given to the rules of ``model`` in priority order, and
    only the first rule that matches is applied: its noise goes after the
    instruction, before it, or in its place. ``circuit`` is not changed.

    When no rule matches a wrapper operation, the rules are applied to the
    operations inside it: :class:`Block`, :class:`IfStatement`,
    :class:`WhileStatement`, :class:`Parallel`, :class:`Repeat`,
    :class:`GateCall` and :class:`BlockCall`. A wrapper whose content gets
    noise is turned into a :class:`Block` (or keeps its condition, for
    ``IfStatement`` and ``WhileStatement``); one whose content gets none is
    kept as it is.

    Args:
        circuit: Circuit to make noisy.
        model: Noise model to apply.

    Returns:
        Circuit: A new circuit with the noise added.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols, pi
        >>> theta = symbols("theta")
        >>> c = Circuit()
        >>> _ = c.push(GateRX(0.4), 0)
        >>> _ = c.push(GateRX(0.8), 1)
        >>> _ = c.push(Measure(), 0, 0)
        >>> _ = c.push(Measure(), 1, 1)
        >>> # noise after each RX that depends on its angle, and readout errors
        >>> model = NoiseModel([
        ...     OperationInstanceNoise(GateRX(theta), Depolarizing(1, theta / pi)),
        ...     GlobalReadoutNoise(ReadoutErr(0.01, 0.02)),
        ... ])
        >>> apply_noise_model(c, model)
        2-qubit, 2-bit circuit with 8 instructions:
        ├── RX(0.4) @ q[0]
        ├── Depolarizing(0.4/pi) @ q[0]
        ├── RX(0.8) @ q[1]
        ├── Depolarizing(0.8/pi) @ q[1]
        ├── M @ q[0], c[0]
        ├── RErr(0.01, 0.02) @ c[0]
        ├── M @ q[1], c[1]
        └── RErr(0.01, 0.02) @ c[1]
        <BLANKLINE>

        Noise also reaches operations inside wrappers. Here every ``H`` gets
        amplitude damping; ``decompose_step`` opens the wrapper to show it.

        >>> model = NoiseModel([OperationInstanceNoise(GateH(), AmplitudeDamping(0.01))])
        >>> inner = Circuit().push(GateH(), 0)
        >>> # inside a Block
        >>> c_block = Circuit().push(Block(inner), 0)
        >>> apply_noise_model(c_block, model).decompose_step()
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        >>> # inside a gate declaration call
        >>> decl = GateDecl("local_h", (), inner)
        >>> c_call = Circuit().push(GateCall(decl, ()), 0)
        >>> apply_noise_model(c_call, model).decompose_step()
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        >>> # each copy of a Parallel gets its own noise
        >>> c_parallel = Circuit().push(Parallel(2, GateH()), 0, 1)
        >>> apply_noise_model(c_parallel, model).decompose_step()
        2-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── AmplitudeDamping(0.01) @ q[0]
        ├── H @ q[1]
        └── AmplitudeDamping(0.01) @ q[1]
        <BLANKLINE>
        >>> # so does each repetition of a Repeat
        >>> c_repeat = Circuit().push(Repeat(2, GateH()), 0)
        >>> apply_noise_model(c_repeat, model).decompose_step()
        1-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── AmplitudeDamping(0.01) @ q[0]
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        >>> # a conditional gate keeps its condition, and so does its noise
        >>> c_if = Circuit().push(IfStatement(GateH(), BitString("1")), 0, 0)
        >>> apply_noise_model(c_if, model).decompose_step()
        1-qubit, 1-bit circuit with 2 instructions:
        ├── IF(c==1) H @ q[0], condition[0]
        └── IF(c==1) AmplitudeDamping(0.01) @ q[0], condition[0]
        <BLANKLINE>
        >>> # nested gate calls are opened at every level
        >>> decl_inner = GateDecl("inner_h", (), inner)
        >>> middle = Circuit().push(GateCall(decl_inner, ()), 0)
        >>> decl_outer = GateDecl("outer_call", (), middle)
        >>> c_nested = Circuit().push(GateCall(decl_outer, ()), 0).push(GateCall(decl_outer, ()), 1)
        >>> apply_noise_model(c_nested, model).decompose_step().decompose_step()
        2-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── AmplitudeDamping(0.01) @ q[0]
        ├── H @ q[1]
        └── AmplitudeDamping(0.01) @ q[1]
        <BLANKLINE>
    """
    active_decls: Set[object] = set()
    noisy_instructions = _apply_noise_to_instructions(
        circuit.instructions, model, active_decls
    )
    return Circuit(noisy_instructions)


__all__ = [
    "AbstractNoiseRule",
    "GlobalReadoutNoise",
    "ExactQubitReadoutNoise",
    "SetQubitReadoutNoise",
    "OperationInstanceNoise",
    "ExactOperationInstanceQubitNoise",
    "SetOperationInstanceQubitNoise",
    "IdleNoise",
    "SetIdleQubitNoise",
    "CustomNoiseRule",
    "NoiseModel",
    "apply_noise_model",
]
