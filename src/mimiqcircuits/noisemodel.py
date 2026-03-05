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

This module implements rule-based noise injection for ``mimiqcircuits`` circuits.
Rules are evaluated by priority (lower value means higher priority), and the first
matching rule is applied to each instruction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Set, Union

import mimiqcircuits as mc
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


def _is_reset(operation: mc.Operation) -> bool:
    return isinstance(operation, (mc.Reset, mc.ResetX, mc.ResetY, mc.ResetZ))


def _supports_symbolic_operation_pattern(operation: mc.Operation) -> bool:
    return isinstance(operation, (mc.Gate, mc.AbstractMeasurement)) or _is_reset(
        operation
    )


def _is_symbolic_operation_pattern(operation: mc.Operation) -> bool:
    return _supports_symbolic_operation_pattern(operation) and operation.is_symbolic()


def _validate_rule_operation_target(operation: mc.Operation):
    if not isinstance(
        operation,
        (
            mc.Gate,
            mc.AbstractMeasurement,
            mc.Block,
            mc.Repeat,
            mc.IfStatement,
            mc.Reset,
            mc.ResetX,
            mc.ResetY,
            mc.ResetZ,
        ),
    ):
        raise ValueError(
            "Rule target operation must be a gate, measurement, reset, "
            "Block, Repeat, or IfStatement operation."
        )


def _resolve_noise_for_operation_match(
    op_pattern: mc.Operation,
    op_instance: mc.Operation,
    noise_pattern: Union[mc.krauschannel, mc.Gate],
):
    """Build noise op from a matched operation instance."""
    # Non-symbolic patterns (including wrappers) are static.
    if not _is_symbolic_operation_pattern(op_pattern):
        return noise_pattern

    variables = _extract_variables(op_pattern)
    if variables is None or all(var is None for var in variables):
        return noise_pattern

    return applyparams(op_instance, (variables, noise_pattern))


class AbstractNoiseRule:
    """Abstract base class for all noise rules."""

    def priority(self):
        """Lower = higher priority."""
        return 100

    def before(self):
        """If True, apply noise before the operation."""
        return False

    def replaces(self):
        """If True, noise instruction replaces the original instruction."""
        return False

    def matches(self, inst: mc.Instruction):
        raise NotImplementedError

    def apply_rule(self, inst: mc.Instruction):
        raise NotImplementedError


@dataclass(frozen=True)
class GlobalReadoutNoise(AbstractNoiseRule):
    noise: mc.ReadoutErr

    def priority(self):
        return PRIORITY_GLOBAL_READOUT

    def matches(self, inst: mc.Instruction):
        return isinstance(inst.get_operation(), mc.AbstractMeasurement)

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None
        return mc.Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class ExactQubitReadoutNoise(AbstractNoiseRule):
    qubits: Sequence[int]
    noise: mc.ReadoutErr

    def __post_init__(self):
        qs = list(self.qubits)
        if not qs:
            raise ValueError("Qubit list must not be empty")
        if len(set(qs)) != len(qs):
            raise ValueError("Qubit list must not contain repetitions")
        object.__setattr__(self, "qubits", tuple(qs))

    def priority(self):
        return PRIORITY_EXACT_READOUT

    def matches(self, inst: mc.Instruction):
        return isinstance(inst.get_operation(), mc.AbstractMeasurement) and tuple(
            inst.get_qubits()
        ) == self.qubits

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None
        return mc.Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class SetQubitReadoutNoise(AbstractNoiseRule):
    qubits: Iterable[int]
    noise: mc.ReadoutErr
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

    def matches(self, inst: mc.Instruction):
        return isinstance(inst.get_operation(), mc.AbstractMeasurement) and all(
            q in self._qubit_set for q in inst.get_qubits()
        )

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None
        return mc.Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class OperationInstanceNoise(AbstractNoiseRule):
    operation: mc.Operation
    noise: Union[mc.krauschannel, mc.Gate]
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

    def matches(self, inst: mc.Instruction):
        op_inst = inst.get_operation()
        op_rule = self.operation

        if type(op_inst) is not type(op_rule):
            return False

        if not _is_symbolic_operation_pattern(op_rule):
            return op_inst == op_rule

        return True

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()
        noise_op = _resolve_noise_for_operation_match(self.operation, op_inst, self.noise)
        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class ExactOperationInstanceQubitNoise(AbstractNoiseRule):
    operation: mc.Operation
    qubits: Sequence[int]
    noise: Union[mc.krauschannel, mc.Gate]
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

    def matches(self, inst: mc.Instruction):
        op_inst = inst.get_operation()
        op_rule = self.operation

        if type(op_inst) is not type(op_rule):
            return False

        if tuple(inst.get_qubits()) != self.qubits:
            return False

        if not _is_symbolic_operation_pattern(op_rule):
            return op_inst == op_rule

        return True

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()
        noise_op = _resolve_noise_for_operation_match(self.operation, op_inst, self.noise)
        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class SetOperationInstanceQubitNoise(AbstractNoiseRule):
    operation: mc.Operation
    qubits: Iterable[int]
    noise: Union[mc.krauschannel, mc.Gate]
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

    def matches(self, inst: mc.Instruction):
        op_inst = inst.get_operation()
        op_rule = self.operation

        if type(op_inst) is not type(op_rule):
            return False

        if not all(q in self._qubit_set for q in inst.get_qubits()):
            return False

        if not _is_symbolic_operation_pattern(op_rule):
            return op_inst == op_rule

        return True

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()
        noise_op = _resolve_noise_for_operation_match(self.operation, op_inst, self.noise)
        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class IdleNoise(AbstractNoiseRule):
    relation: Union[tuple, mc.Operation]

    def __post_init__(self):
        if isinstance(self.relation, mc.Operation):
            return

        if not isinstance(self.relation, tuple) or len(self.relation) != 2:
            raise ValueError("IdleNoise relation must be (var, target) or Operation")

        var, target = self.relation

        if not (hasattr(var, "is_symbol") and var.is_symbol):
            raise ValueError(
                f"Left side must be a simple symbolic variable, got: {var}"
            )
        if not isinstance(target, mc.Operation):
            raise ValueError("Right side of idle relation must be an Operation")

    def priority(self):
        return PRIORITY_IDLE

    def replaces(self):
        return True

    def matches(self, inst: mc.Instruction):
        return isinstance(inst.get_operation(), mc.Delay)

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()

        if isinstance(self.relation, mc.Operation):
            noise_op = self.relation
        else:
            var, target = self.relation
            noise_op = applyparams(op_inst, ((var,), target))

        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class SetIdleQubitNoise(AbstractNoiseRule):
    relation: Union[tuple, mc.Operation]
    qubits: Iterable[int]

    def __post_init__(self):
        qs = list(self.qubits)
        if not qs:
            raise ValueError("Qubit set must not be empty")

        object.__setattr__(self, "qubits", frozenset(qs))

        if isinstance(self.relation, mc.Operation):
            return

        if not isinstance(self.relation, tuple) or len(self.relation) != 2:
            raise ValueError("IdleNoise relation must be (var, target) or Operation")

        var, target = self.relation
        if not (hasattr(var, "is_symbol") and var.is_symbol):
            raise ValueError("Left side must be symbolic variable")
        if not isinstance(target, mc.Operation):
            raise ValueError("Right side must be Operation")

    def priority(self):
        return PRIORITY_SET_IDLE

    def replaces(self):
        return True

    def matches(self, inst: mc.Instruction):
        op = inst.get_operation()
        return isinstance(op, mc.Delay) and all(q in self.qubits for q in inst.get_qubits())

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None

        op_inst = inst.get_operation()

        if isinstance(self.relation, mc.Operation):
            noise_op = self.relation
        else:
            var, target = self.relation
            noise_op = applyparams(op_inst, ((var,), target))

        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class CustomNoiseRule(AbstractNoiseRule):
    matcher: Callable[[mc.Instruction], bool]
    generator: Callable[[mc.Instruction], mc.Instruction]
    priority_val: int = PRIORITY_USER_OVERRIDE
    before: bool = False
    replace: bool = False

    def __post_init__(self):
        _bind_before_method(self, self.before)

    def priority(self):
        return self.priority_val

    def matches(self, inst: mc.Instruction):
        return self.matcher(inst)

    def apply_rule(self, inst: mc.Instruction):
        return self.generator(inst) if self.matcher(inst) else None

    def replaces(self):
        return self.replace


@dataclass
class NoiseModel:
    """Collection of prioritized noise rules.

    Rules are always kept sorted by ``priority()`` so more specific rules win over
    generic fallbacks.

    Priority order (lower number = higher priority):
    - ``CustomNoiseRule`` (default ``PRIORITY_USER_OVERRIDE``)
    - ``ExactOperationInstanceQubitNoise`` (``PRIORITY_EXACT_OPERATION``)
    - ``ExactQubitReadoutNoise`` (``PRIORITY_EXACT_READOUT``)
    - ``SetOperationInstanceQubitNoise`` (``PRIORITY_SET_OPERATION``)
    - ``SetQubitReadoutNoise`` (``PRIORITY_SET_READOUT``)
    - ``OperationInstanceNoise`` (``PRIORITY_GLOBAL_OPERATION``)
    - ``GlobalReadoutNoise`` (``PRIORITY_GLOBAL_READOUT``)
    - ``SetIdleQubitNoise`` (``PRIORITY_SET_IDLE``)
    - ``IdleNoise`` (``PRIORITY_IDLE``)

    Args:
        rules: Initial list of rules.
        name: Optional model name.

    Example:
        >>> import mimiqcircuits as mc
        >>> from symengine import symbols, pi
        >>> theta = symbols("theta")
        >>> model = mc.NoiseModel(
        ...     [
        ...         mc.OperationInstanceNoise(mc.GateRX(theta), mc.Depolarizing(1, theta / pi)),
        ...         mc.ExactOperationInstanceQubitNoise(mc.GateCX(), [0, 1], mc.Depolarizing(2, 0.01)),
        ...         mc.GlobalReadoutNoise(mc.ReadoutErr(0.01, 0.02)),
        ...         mc.IdleNoise(mc.AmplitudeDamping(1e-4)),
        ...     ],
        ...     name="angle-dependent",
        ... )
    """

    rules: List[AbstractNoiseRule] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        self.rules.sort(key=lambda r: r.priority())

    def add_rule(self, rule: AbstractNoiseRule):
        """Add a rule and keep model rules sorted by priority."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority())
        return self

    def add_readout_noise(self, noise: mc.ReadoutErr, *, qubits=None, exact=False):
        """Add a readout-noise rule.

        Behavior:
        - ``qubits is None``: add ``GlobalReadoutNoise``.
        - ``qubits`` with ``exact=False``: add ``SetQubitReadoutNoise``.
        - ``qubits`` with ``exact=True``: add ``ExactQubitReadoutNoise``.

        Args:
            noise: Readout error channel.
            qubits: Optional qubit targets.
            exact: When ``True``, qubit order must match exactly.

        Returns:
            ``self`` (for chaining).

        Example:
            >>> import mimiqcircuits as mc
            >>> model = mc.NoiseModel()
            >>> model.add_readout_noise(mc.ReadoutErr(0.01, 0.02))
            NoiseModel(rules=[GlobalReadoutNoise(noise=RErr(0.01,0.02))], name='')
            >>> model.add_readout_noise(mc.ReadoutErr(0.03, 0.04), qubits=[0, 2])
            NoiseModel(rules=[SetQubitReadoutNoise(qubits=(0, 2), noise=RErr(0.03,0.04)), GlobalReadoutNoise(noise=RErr(0.01,0.02))], name='')
            >>> model.add_readout_noise(mc.ReadoutErr(0.05, 0.06), qubits=[2, 0], exact=True)
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
        """Add operation-instance noise.

        ``operation`` can be concrete (exact parameter match) or symbolic (match by
        operation type and substitute parameters into ``noise``).

        Args:
            operation: Operation pattern to match.
            noise: Noise operation/channel to inject.
            qubits: Optional qubit restriction.
            exact: If ``True``, qubits must match the exact ordered tuple.
            before: If ``True``, insert noise before the matched instruction.
            replace: If ``True``, replace the matched instruction with noise.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: If operation target is unsupported or arguments are invalid.

        Examples:
            >>> import mimiqcircuits as mc
            >>> from symengine import symbols, pi
            >>> theta, alpha, beta = symbols("theta alpha beta")
            >>> model = mc.NoiseModel()
            >>> _ = model.add_operation_noise(mc.GateRX(pi / 2), mc.AmplitudeDamping(0.001))
            >>> _ = model.add_operation_noise(mc.GateRX(theta), mc.Depolarizing(1, theta / pi))
            >>> _ = model.add_operation_noise(
            ...     mc.GateU(alpha, beta, 0),
            ...     mc.Depolarizing(1, (alpha**2 + beta**2) / (2 * pi**2)),
            ... )
            >>> _ = model.add_operation_noise(
            ...     mc.GateRX(theta),
            ...     mc.Depolarizing(1, theta / pi),
            ...     qubits=[0, 1, 2],
            ... )
            >>> _ = model.add_operation_noise(
            ...     mc.GateRX(theta),
            ...     mc.Depolarizing(1, theta / pi),
            ...     qubits=[0],
            ...     exact=True,
            ... )
            >>> _ = model.add_operation_noise(mc.Measure(), mc.PauliX(0.02), before=True)
            >>> _ = model.add_operation_noise(mc.Reset(), mc.Depolarizing(1, 0.01))
            >>> _ = model.add_operation_noise(mc.GateH(), mc.AmplitudeDamping(0.001), replace=True)
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
        """Add idle-noise rule(s) for ``Delay`` instructions.

        ``noise`` can be a constant operation or a relation tuple
        ``(time_symbol, target_operation_expr)``.

        Args:
            noise: Idle noise operation or symbolic relation.
            qubits: Optional qubit subset where idle noise is allowed.

        Returns:
            ``self`` (for chaining).

        Example:
            >>> import mimiqcircuits as mc
            >>> from symengine import symbols
            >>> t = symbols("t")
            >>> model = mc.NoiseModel()
            >>> model.add_idle_noise(mc.AmplitudeDamping(1e-4))
            NoiseModel(rules=[IdleNoise(relation=AmplitudeDamping(0.0001))], name='')
            >>> model.add_idle_noise((t, mc.AmplitudeDamping(t / 1000)), qubits=[0, 1, 2])
            NoiseModel(rules=[SetIdleQubitNoise(relation=(t, AmplitudeDamping((1/1000)*t)), qubits=frozenset({0, 1, 2})), IdleNoise(relation=AmplitudeDamping(0.0001))], name='')
        """
        if qubits is None:
            rule = IdleNoise(noise)
        else:
            rule = SetIdleQubitNoise(noise, qubits)
        return self.add_rule(rule)

    def apply_noise_model(self, circuit: mc.Circuit):
        """Apply this noise model and return a new noisy circuit.

        This is a convenience wrapper around module-level ``apply_noise_model``.
        """
        return apply_noise_model(circuit, self)

    def describe(self):
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
                if isinstance(rule.relation, mc.Operation):
                    print(f"  -> Applies {rule.relation} to all idle qubits")
                else:
                    var, target = rule.relation
                    print(
                        f"  -> Applies time-dependent idle noise {target} with "
                        f"variable {var} to all idle qubits"
                    )
            elif isinstance(rule, SetIdleQubitNoise):
                if isinstance(rule.relation, mc.Operation):
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
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, NoiseModel)


def _canonical_targets(op: mc.Operation):
    return (
        tuple(range(op.num_qubits)),
        tuple(range(op.num_bits)),
        tuple(range(op.num_zvars)),
    )


def _collapse_local_instructions_to_operation(
    instructions: List[mc.Instruction], op: mc.Operation
):
    qcanon, bcanon, zcanon = _canonical_targets(op)

    if (
        len(instructions) == 1
        and instructions[0].get_qubits() == qcanon
        and instructions[0].get_bits() == bcanon
        and instructions[0].get_zvars() == zcanon
    ):
        return instructions[0].get_operation()

    return mc.Block(op.num_qubits, op.num_bits, op.num_zvars, instructions)


def _apply_rules_to_instruction(inst: mc.Instruction, model: NoiseModel):
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
    op: mc.Operation, model: NoiseModel, active_decls: Set[object]
):
    if isinstance(op, mc.Block):
        noisy_instructions = _apply_noise_to_instructions(
            op.instructions, model, active_decls
        )
        if noisy_instructions == op.instructions:
            return op
        return mc.Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)

    if isinstance(op, mc.IfStatement):
        inner = op.get_operation()
        qcanon, bcanon, zcanon = _canonical_targets(inner)
        inner_inst = mc.Instruction(inner, qcanon, bcanon, zcanon)

        noisy_inner = _apply_noise_to_instruction(inner_inst, model, active_decls)
        rewritten_inner = _collapse_local_instructions_to_operation(noisy_inner, inner)

        if rewritten_inner == inner:
            return op

        return mc.IfStatement(rewritten_inner, op.get_bitstring())

    if isinstance(op, mc.Parallel):
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
                mc.Instruction(inner, qtargets, btargets, ztargets)
            )

        noisy_instructions = _apply_noise_to_instructions(
            repeated_instructions, model, active_decls
        )

        if noisy_instructions == repeated_instructions:
            return op

        return mc.Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)

    if isinstance(op, mc.Repeat):
        inner = op.get_operation()
        qcanon, bcanon, zcanon = _canonical_targets(inner)
        repeated_instructions = [
            mc.Instruction(inner, qcanon, bcanon, zcanon) for _ in range(op.repeats)
        ]

        noisy_instructions = _apply_noise_to_instructions(
            repeated_instructions, model, active_decls
        )
        if noisy_instructions == repeated_instructions:
            return op

        return mc.Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)

    if isinstance(op, mc.GateCall):
        decl = op.decl

        # Prevent infinite recursion for self-referential declarations.
        if decl in active_decls:
            return op

        active_decls.add(decl)
        try:
            substitutions = dict(zip(decl.arguments, op.arguments))
            expanded_instructions = []
            for inst in decl.circuit:
                expanded_op = inst.get_operation().evaluate(substitutions)
                expanded_instructions.append(
                    mc.Instruction(
                        expanded_op, inst.get_qubits(), inst.get_bits(), inst.get_zvars()
                    )
                )

            noisy_instructions = _apply_noise_to_instructions(
                expanded_instructions, model, active_decls
            )

            if noisy_instructions == expanded_instructions:
                return op

            return mc.Block(op.num_qubits, op.num_bits, op.num_zvars, noisy_instructions)
        finally:
            active_decls.remove(decl)

    return op


def _apply_noise_to_instruction(
    inst: mc.Instruction, model: NoiseModel, active_decls: Set[object]
):
    rewritten_instructions, matched = _apply_rules_to_instruction(inst, model)
    if matched:
        return rewritten_instructions

    op = inst.get_operation()
    rewritten_op = _rewrite_nested_operation(op, model, active_decls)

    if rewritten_op is op:
        return rewritten_instructions

    return [
        mc.Instruction(
            rewritten_op, inst.get_qubits(), inst.get_bits(), inst.get_zvars()
        )
    ]


def _apply_noise_to_instructions(
    instructions: List[mc.Instruction], model: NoiseModel, active_decls: Set[object]
):
    noisy_instructions = []
    for inst in instructions:
        noisy_instructions.extend(_apply_noise_to_instruction(inst, model, active_decls))
    return noisy_instructions


def apply_noise_model(circuit: mc.Circuit, model: NoiseModel):
    """Apply a noise model to a circuit and return a new circuit.

    Rules are evaluated in priority order. For each instruction, only the first
    matching rule is applied.

    Wrapper operations are traversed recursively:
    ``Block``, ``IfStatement``, ``Parallel``, ``Repeat``, and ``GateCall``.

    Args:
        circuit: Input circuit.
        model: Noise model to apply.

    Returns:
        A new circuit with injected noise.

    Examples:
        >>> import mimiqcircuits as mc
        >>> from symengine import symbols, pi
        >>> theta = symbols("theta")
        >>> c = mc.Circuit()
        >>> c.push(mc.GateRX(0.4), 0)
        1-qubit circuit with 1 instruction:
        └── RX(0.4) @ q[0]
        <BLANKLINE>
        >>> c.push(mc.GateRX(0.8), 1)
        2-qubit circuit with 2 instructions:
        ├── RX(0.4) @ q[0]
        └── RX(0.8) @ q[1]
        <BLANKLINE>
        >>> c.push(mc.Measure(), 0, 0)
        2-qubit, 1-bit circuit with 3 instructions:
        ├── RX(0.4) @ q[0]
        ├── RX(0.8) @ q[1]
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> c.push(mc.Measure(), 1, 1)
        2-qubit, 2-bit circuit with 4 instructions:
        ├── RX(0.4) @ q[0]
        ├── RX(0.8) @ q[1]
        ├── M @ q[0], c[0]
        └── M @ q[1], c[1]
        <BLANKLINE>
        >>> model = mc.NoiseModel([mc.OperationInstanceNoise(mc.GateRX(theta), mc.Depolarizing(1, theta / pi)),
        ...         mc.GlobalReadoutNoise(mc.ReadoutErr(0.01, 0.02)),])
        >>> noisy = mc.apply_noise_model(c, model)

    Recursive wrapper examples:
        >>> model = mc.NoiseModel([mc.OperationInstanceNoise(mc.GateH(), mc.AmplitudeDamping(0.01))])
        >>> inner = mc.Circuit().push(mc.GateH(), 0)
        >>> c_block = mc.Circuit().push(mc.Block(inner), 0)
        >>> noisy_block= mc.apply_noise_model(c_block, model)
        >>> noisy_block.decompose()
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        >>> decl = mc.GateDecl("local_h", (), inner)
        >>> c_call = mc.Circuit().push(mc.GateCall(decl, ()), 0)
        >>> noisy_call = mc.apply_noise_model(c_call, model)
        >>> noisy_call.decompose()
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        >>> c_parallel = mc.Circuit().push(mc.Parallel(2, mc.GateH()), 0, 1)
        >>> noisy_parallel = mc.apply_noise_model(c_parallel, model)
        >>> noisy_parallel.decompose()
        2-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── AmplitudeDamping(0.01) @ q[0]
        ├── H @ q[1]
        └── AmplitudeDamping(0.01) @ q[1]
        <BLANKLINE>
        >>> c_repeat = mc.Circuit().push(mc.Repeat(2, mc.GateH()), 0)
        >>> noisy_repeat = mc.apply_noise_model(c_repeat, model)
        >>> noisy_repeat.decompose()
        1-qubit circuit with 4 instructions:
        ├── H @ q[0]
        ├── AmplitudeDamping(0.01) @ q[0]
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        >>> c_if = mc.Circuit().push(mc.IfStatement(mc.GateH(), mc.BitString("1")), 0, 0)
        >>> noisy_if = mc.apply_noise_model(c_if, model)
        >>> noisy_if.decompose()
        1-qubit, 1-bit circuit with 2 instructions:
        ├── IF(c==1) H @ q[0], condition[0]
        └── IF(c==1) AmplitudeDamping(0.01) @ q[0], condition[0]
        <BLANKLINE>
        
        >>> decl_inner = mc.GateDecl("inner_h", (), inner)
        >>> middle = mc.Circuit().push(mc.GateCall(decl_inner, ()), 0)
        >>> decl_outer = mc.GateDecl("outer_call", (), middle)
        >>> c_nested = mc.Circuit().push(mc.GateCall(decl_outer, ()), 0)
        >>> noisy_nested = mc.apply_noise_model(c_nested, model)
        >>> noisy_nested.decompose().decompose()
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── AmplitudeDamping(0.01) @ q[0]
        <BLANKLINE>
        
    Recursive wrapper examples deeply nested:
        >>> c_nested2 = mc.Circuit().push(mc.GateCall(decl_outer, ()), 0).push(mc.GateCall(decl_outer, ()), 1)
        >>> c_nested2.decompose()
        2-qubit circuit with 2 instructions:
        ├── inner_h() @ q[0]
        └── inner_h() @ q[1]
        <BLANKLINE>
        >>> noisy_nested2 = mc.apply_noise_model(c_nested2, model)
        >>> noisy_nested2.decompose().decompose()
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
    return mc.Circuit(noisy_instructions)


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
