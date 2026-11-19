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
"""Loss model definitions and qubit-loss sampling.

This module provides the rule system used by :func:`lower_losses` (and hence
:func:`resolve_losses`) when a circuit instruction touches both lost and
surviving qubits. A :class:`LossModel` lets users decide whether such an
instruction should be dropped, replaced, decorated, or handled by custom logic.

Examples:
    >>> from mimiqcircuits import *
    >>> # qubit 1 is lost before a CX that also acts on qubit 0
    >>> circuit = Circuit()
    >>> _ = circuit.push(Loss(), 1)
    >>> _ = circuit.push(GateCX(), 0, 1)
    >>> # replace the broken CX by depolarizing noise on the surviving qubit
    >>> model = LossModel().add_replace(GateCX(), Depolarizing1(0.2))
    >>> resolve_losses(circuit, lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── Lost @ q[1]
    └── Depolarizing(0.2) @ q[0]
    <BLANKLINE>
"""

from __future__ import annotations

import inspect
import random
from typing import Iterable, List, Optional

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.classical.setbits import SetBit0, SetBit1
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.annotations import Lost, Reloaded
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.losschannel import Check, Loss, MeasureCheck, Reload
from mimiqcircuits.operations.magic import PrepareStateT
from mimiqcircuits.operations.measure import AbstractMeasurement, Measure
from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.repeat import Repeat
from mimiqcircuits.operations.reset import Reset, ResetX, ResetY, ResetZ
from mimiqcircuits.circuitrules import AbstractCircuitRule
from mimiqcircuits.symbolics import (
    UndefinedValue,
    _extract_variables,
    _validate_rule_gate_params,
    applyparams,
    unwrapvalue,
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
            Reset,
            ResetX,
            ResetY,
            ResetZ,
            PrepareStateT,
        ),
    ):
        raise ValueError(
            "Rule target operation must be a gate, measurement, reset, "
            "Block, Repeat, or IfStatement operation."
        )


def _matches_operation_pattern(op_inst: Operation, op_rule: Operation) -> bool:
    if type(op_inst) is not type(op_rule):
        return False

    if not _is_symbolic_operation_pattern(op_rule):
        return op_inst == op_rule

    return True


def _resolve_symbolic_replacement(
    op_inst: Operation, op_rule: Operation, replacement: Operation
):
    if not _is_symbolic_operation_pattern(op_rule):
        return replacement

    variables = _extract_variables(op_rule)
    if variables is None or all(var is None for var in variables):
        return replacement

    return applyparams(op_inst, (variables, replacement))


def _normalize_to_instructions(result) -> List[Instruction]:
    if result is None:
        return []

    if isinstance(result, Instruction):
        return [result]

    if hasattr(result, "instructions"):
        insts = list(result.instructions)
    elif isinstance(result, (list, tuple)):
        insts = list(result)
    else:
        raise TypeError(
            "Expected None, Instruction, Circuit, Block, or a sequence of Instructions."
        )

    if not all(isinstance(inst, Instruction) for inst in insts):
        raise TypeError("All generated items must be Instruction instances.")

    return insts


def _validate_replacement_operation(
    operation: Operation, replacement: Operation, label: str
):
    if replacement.num_bits != 0 or replacement.num_zvars != 0:
        raise ValueError(
            f"{label} operation must not target classical bits or z-variables."
        )

    nq_op = operation.num_qubits
    nq_repl = replacement.num_qubits
    if nq_repl != nq_op and nq_repl != 1:
        raise ValueError(
            f"{label} must have the same number of qubits as the operation ({nq_op}) "
            f"or exactly 1 qubit. Got {nq_repl}."
        )


def _validate_replacement_instructions(
    operation: Operation, instructions: List[Instruction], label: str
):
    nq = operation.num_qubits
    for inst in instructions:
        if inst.get_bits() or inst.get_zvars():
            raise ValueError(
                f"{label} instructions must not target classical bits or z-variables."
            )
        for q in inst.get_qubits():
            if q < 0 or q >= nq:
                raise ValueError(
                    f"{label} instructions must use canonical qubits in range(0, {nq})."
                )


def _build_replacement_instructions(replacement, op_pattern: Operation, inst):
    if isinstance(replacement, Operation):
        op_inst = inst.get_operation()
        qs = inst.get_qubits()
        resolved = _resolve_symbolic_replacement(op_inst, op_pattern, replacement)

        if resolved.num_qubits == len(qs):
            return [Instruction(resolved, tuple(qs))]

        return [Instruction(resolved, (q,)) for q in qs]

    qubit_map = {i: q for i, q in enumerate(inst.get_qubits())}
    return [
        Instruction(
            repl_inst.get_operation(),
            tuple(qubit_map[q] for q in repl_inst.get_qubits()),
            tuple(repl_inst.get_bits()),
            tuple(repl_inst.get_zvars()),
        )
        for repl_inst in replacement
    ]


def _call_custom_generator(generator, inst, lost, rng):
    signature = inspect.signature(generator)
    params = list(signature.parameters.values())

    positional = [
        param
        for param in params
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL for param in params
    )
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)

    if "rng" in signature.parameters or has_kwargs:
        return generator(inst, lost, rng=rng)
    if has_varargs or len(positional) >= 3:
        return generator(inst, lost, rng)
    if len(positional) >= 2:
        return generator(inst, lost)
    return generator(inst)


class DropRule(AbstractCircuitRule):
    """Drop matched instructions that touch lost qubits.

    A ``DropRule`` removes a partially affected operation without putting
    anything in its place. Dropping is also what happens when no rule matches;
    a ``DropRule`` is useful to stop a later rule from matching. Drop rules
    have priority 0, so a :class:`LossModel` tries them before any other rule.

    An operation pattern matches instructions whose operation has the same type
    and is equal to it. A gate, measurement or reset pattern with symbolic
    parameters (for example ``GateRX(theta)``) matches every operation of that
    type.

    Args:
        operation: Operation pattern to drop: a gate, measurement, reset,
            :class:`Block`, :class:`Repeat` or :class:`IfStatement`. If
            omitted, the rule matches every instruction.

    Raises:
        ValueError: If ``operation`` is not one of the supported kinds.

    Examples:
        >>> from mimiqcircuits import *
        >>> # drop every CX that touches a lost qubit
        >>> model = LossModel().add_drop(GateCX())
        >>> model
        LossModel (unnamed, 1 rules)
        └── DropRule(CX)

        >>> circuit = Circuit()
        >>> _ = circuit.push(Loss(), 1)
        >>> _ = circuit.push(GateCX(), 0, 1)
        >>> circuit.resolve_losses(lossmodel=model)
        2-qubit circuit with 1 instruction:
        └── Lost @ q[1]
        <BLANKLINE>
    """

    def __init__(self, operation: Optional[Operation] = None):
        if operation is not None:
            _validate_rule_operation_target(operation)
            if _supports_symbolic_operation_pattern(operation):
                _validate_rule_gate_params(operation)
        self.operation = operation

    def priority(self):
        return 0

    def matches(self, inst: Instruction):
        if self.operation is None:
            return True
        return _matches_operation_pattern(inst.get_operation(), self.operation)

    def replaces(self):
        return True

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        return []

    def __str__(self):
        if self.operation is None:
            return "DropRule(*)"
        return f"DropRule({self.operation})"

    __repr__ = __str__


class DecorateRule(AbstractCircuitRule):
    """Keep a matched instruction and add a decoration before or after it.

    When the loss model rewrites an instruction, every resulting instruction
    that still touches a lost qubit is removed, and that includes the matched
    instruction itself. What remains is the decoration on the surviving qubits,
    which models a side effect of an operation attempted with a qubit missing.

    The decoration is either an operation or a sequence of instructions (a
    list, :class:`Circuit` or :class:`Block`). An operation with as many
    qubits as ``operation`` is applied to the same targets; a one-qubit
    operation is applied to each target. Instructions use the local qubit
    indices ``0`` to ``N-1`` of the matched operation.

    Args:
        operation: Operation pattern to match (see :class:`DropRule`). The
            pair ``(operation, decoration)`` may also be passed as a single
            tuple.
        decoration: Operation or instruction sequence to add.
        before (bool): Keyword-only. If ``True``, add the decoration before the
            matched instruction, otherwise after it (default).

    Raises:
        TypeError: If ``decoration`` is missing and ``operation`` is not a
            pair, or the decoration is not an operation or instructions.
        ValueError: If ``operation`` is not a supported pattern, the
            decoration targets classical bits or z-variables, or its qubit
            count or indices do not fit ``operation``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # a CZ attempted with a lost partner depolarizes the survivor
        >>> model = LossModel().add_decorate(GateCZ(), Depolarizing1(0.01), before=True)
        >>> model
        LossModel (unnamed, 1 rules)
        └── DecorateRule(CZ, Depolarizing(0.01), before)

        >>> circuit = Circuit()
        >>> _ = circuit.push(Loss(), 1)
        >>> _ = circuit.push(GateCZ(), 0, 1)
        >>> circuit.resolve_losses(lossmodel=model)
        2-qubit circuit with 2 instructions:
        ├── Lost @ q[1]
        └── Depolarizing(0.01) @ q[0]
        <BLANKLINE>
    """

    def __init__(self, operation, decoration=None, *, before=False):
        if decoration is None:
            if not isinstance(operation, tuple) or len(operation) != 2:
                raise TypeError(
                    "DecorateRule expects (operation, decoration) or separate arguments."
                )
            operation, decoration = operation

        _validate_rule_operation_target(operation)
        if _supports_symbolic_operation_pattern(operation):
            _validate_rule_gate_params(operation)

        self.operation = operation
        self._before = bool(before)

        if isinstance(decoration, Operation):
            _validate_replacement_operation(operation, decoration, "Decoration")
            self.decoration = decoration
        else:
            decoration_insts = _normalize_to_instructions(decoration)
            _validate_replacement_instructions(
                operation, decoration_insts, "Decoration"
            )
            self.decoration = decoration_insts

    def before(self):
        return self._before

    def matches(self, inst: Instruction):
        return _matches_operation_pattern(inst.get_operation(), self.operation)

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        decoration_insts = _build_replacement_instructions(
            self.decoration, self.operation, inst
        )
        if self.before():
            return decoration_insts + [inst]
        return [inst] + decoration_insts

    def __str__(self):
        position = "before" if self.before() else "after"
        return f"DecorateRule({self.operation}, {self.decoration}, {position})"

    __repr__ = __str__


class ReplaceRule(AbstractCircuitRule):
    """Replace a matched instruction with new instructions.

    Use ``ReplaceRule`` when a partially affected operation should be removed
    and replaced by something acting on the surviving qubits. The replacement
    follows the same conventions as the decoration of :class:`DecorateRule`:
    an operation with as many qubits as ``operation`` is applied to the same
    targets, a one-qubit operation is applied to each target, and instructions
    use local qubit indices ``0`` to ``N-1``. Resulting instructions on lost
    qubits are removed.

    If ``operation`` has symbolic parameters, a symbolic replacement can reuse
    them: the values of the matched operation are substituted in.

    Args:
        operation: Operation pattern to match (see :class:`DropRule`). The
            pair ``(operation, replacement)`` may also be passed as a single
            tuple.
        replacement: Replacement operation or instruction sequence.

    Raises:
        TypeError: If ``replacement`` is missing and ``operation`` is not a
            pair, or the replacement is not an operation or instructions.
        ValueError: If ``operation`` is not a supported pattern, the
            replacement targets classical bits or z-variables, or its qubit
            count or indices do not fit ``operation``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # a CX with a lost qubit becomes depolarizing noise on the survivor
        >>> model = LossModel().add_replace(GateCX(), Depolarizing1(0.2))
        >>> model
        LossModel (unnamed, 1 rules)
        └── ReplaceRule(CX => Depolarizing(0.2))

        >>> circuit = Circuit()
        >>> _ = circuit.push(Loss(), 1)
        >>> _ = circuit.push(GateCX(), 0, 1)
        >>> circuit.resolve_losses(lossmodel=model)
        2-qubit circuit with 2 instructions:
        ├── Lost @ q[1]
        └── Depolarizing(0.2) @ q[0]
        <BLANKLINE>
    """

    def __init__(self, operation, replacement=None):
        if replacement is None:
            if not isinstance(operation, tuple) or len(operation) != 2:
                raise TypeError(
                    "ReplaceRule expects (operation, replacement) or separate arguments."
                )
            operation, replacement = operation

        _validate_rule_operation_target(operation)
        if _supports_symbolic_operation_pattern(operation):
            _validate_rule_gate_params(operation)

        self.operation = operation

        if isinstance(replacement, Operation):
            _validate_replacement_operation(operation, replacement, "Replacement")
            self.replacement = replacement
        else:
            replacement_insts = _normalize_to_instructions(replacement)
            _validate_replacement_instructions(
                operation, replacement_insts, "Replacement"
            )
            self.replacement = replacement_insts

    def matches(self, inst: Instruction):
        return _matches_operation_pattern(inst.get_operation(), self.operation)

    def replaces(self):
        return True

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        return _build_replacement_instructions(self.replacement, self.operation, inst)

    def __str__(self):
        return f"ReplaceRule({self.operation} => {self.replacement})"

    __repr__ = __str__


class CustomRule(AbstractCircuitRule):
    """User-defined loss rule.

    ``CustomRule`` is the escape hatch for loss policies that cannot be
    expressed with :class:`DropRule`, :class:`ReplaceRule`, or
    :class:`DecorateRule`. The matcher decides whether the rule applies. The
    generator returns ``None`` to drop the instruction, one
    :class:`Instruction`, or a sequence of instructions.

    The generator may accept ``(inst)``, ``(inst, lost)``, or
    ``(inst, lost, rng)``. It may also accept ``rng`` as a keyword argument.
    ``lost`` maps qubit indices to ``True`` when the qubit is lost. A qubit
    that was never lost is not in the map and a reloaded one maps to
    ``False``, so read it with ``lost.get(q, False)``. ``rng`` is the random number generator given
    to :func:`lower_losses`. Returned instructions use the circuit's qubit
    indices, and any that touch a lost qubit are removed.

    Args:
        matcher: Callable that receives an instruction and returns ``True`` if
            the rule should apply.
        generator: Callable that returns the instructions to emit instead of
            the matched one.

    Examples:
        Define a custom fallback for a partially lost ``CX``. If the control
        qubit survives, replace the failed ``CX`` by ``X`` on the control. If
        the control is lost, return ``None`` to drop the instruction.

        >>> from mimiqcircuits import *
        >>> def cx_control_fallback(inst, lost):
        ...     control = inst.get_qubits()[0]
        ...     if lost.get(control, False):
        ...         return None
        ...     return Instruction(GateX(), (control,))
        ...
        >>> model = LossModel([
        ...     CustomRule(
        ...         lambda inst: isinstance(inst.get_operation(), GateCX),
        ...         cx_control_fallback,
        ...     )
        ... ])
        >>> model
        LossModel (unnamed, 1 rules)
        └── CustomRule(<callable>)

        >>> circuit = Circuit()
        >>> _ = circuit.push(Loss(), 1)
        >>> _ = circuit.push(GateCX(), 0, 1)
        >>> circuit.resolve_losses(lossmodel=model)
        2-qubit circuit with 2 instructions:
        ├── Lost @ q[1]
        └── X @ q[0]
        <BLANKLINE>

        Another custom rule can generate one instruction for each surviving
        qubit. Here a partially lost ``CX`` becomes ``Z`` on every qubit that
        is still present:

        >>> model = LossModel([
        ...     CustomRule(
        ...         lambda inst: isinstance(inst.get_operation(), GateCX),
        ...         lambda inst, lost: [
        ...             Instruction(GateZ(), (q,))
        ...             for q in inst.get_qubits()
        ...             if not lost.get(q, False)
        ...         ],
        ...     )
        ... ])
        >>> circuit.resolve_losses(lossmodel=model)
        2-qubit circuit with 2 instructions:
        ├── Lost @ q[1]
        └── Z @ q[0]
        <BLANKLINE>
    """

    def __init__(self, matcher, generator):
        self.matcher = matcher
        self.generator = generator

    def matches(self, inst: Instruction):
        return self.matcher(inst)

    def replaces(self):
        return True

    def apply_rule(self, inst: Instruction):
        if not self.matches(inst):
            return None
        return _normalize_to_instructions(
            _call_custom_generator(self.generator, inst, {}, None)
        )

    def __str__(self):
        return "CustomRule(<callable>)"

    __repr__ = __str__


class LossModel:
    """Ordered collection of loss rules.

    A ``LossModel`` tells :func:`lower_losses` (and so :func:`resolve_losses`)
    what to do with an instruction that touches both lost and surviving qubits.
    Instructions that touch only lost qubits are always dropped, and those that
    touch none are always kept.

    For each such instruction the rules are tried in order and the first one
    that matches decides. Rules are ordered by priority, then by insertion
    order: :class:`DropRule` has priority 0 and all other rules 100, so drop
    rules come first. When no rule matches, the instruction is dropped, which
    is also what an empty model does.

    Args:
        rules: Rules to start with, each an :class:`AbstractCircuitRule`.
        name (str): Name shown when the model is printed.

    Raises:
        TypeError: If a rule is not an :class:`AbstractCircuitRule`.

    Examples:
        >>> from mimiqcircuits import *
        >>> model = LossModel(name="My Loss Model")
        >>> model
        LossModel (My Loss Model, 0 rules)

        >>> # the add_* methods return the model, so they can be chained
        >>> model.add_replace(GateCX(), Depolarizing1(0.2))
        LossModel (My Loss Model, 1 rules)
        └── ReplaceRule(CX => Depolarizing(0.2))

        >>> # pass the model when resolving the losses of a circuit
        >>> circuit = Circuit()
        >>> _ = circuit.push(Loss(), 1)
        >>> _ = circuit.push(GateCX(), 0, 1)
        >>> circuit.resolve_losses(lossmodel=model)
        2-qubit circuit with 2 instructions:
        ├── Lost @ q[1]
        └── Depolarizing(0.2) @ q[0]
        <BLANKLINE>
    """

    def __init__(self, rules: Optional[Iterable[AbstractCircuitRule]] = None, name=""):
        self.rules = list(rules or [])
        self.name = name

        for rule in self.rules:
            if not isinstance(rule, AbstractCircuitRule):
                raise TypeError("LossModel rules must inherit from AbstractCircuitRule.")

        self._sort_rules()

    def _sort_rules(self):
        self.rules.sort(key=lambda rule: rule.priority())

    def add_rule(self, rule: AbstractCircuitRule):
        """Add a rule and return the model.

        The rule is placed after the existing rules of the same priority.

        Args:
            rule: Any :class:`AbstractCircuitRule`.

        Returns:
            LossModel: This model, changed in place.

        Raises:
            TypeError: If ``rule`` is not an :class:`AbstractCircuitRule`.
        """
        if not isinstance(rule, AbstractCircuitRule):
            raise TypeError("LossModel rules must inherit from AbstractCircuitRule.")
        self.rules.append(rule)
        self._sort_rules()
        return self

    def add_drop(self, operation: Optional[Operation] = None):
        """Add a :class:`DropRule` and return the model.

        Args:
            operation: Operation pattern to drop. If omitted, every
                partially lost instruction is dropped.

        Returns:
            LossModel: This model, changed in place.
        """
        return self.add_rule(DropRule(operation))

    def add_replace(self, operation, replacement=None):
        """Add a :class:`ReplaceRule` and return the model.

        Args:
            operation: Operation pattern to match.
            replacement: Operation or instructions to emit instead.

        Returns:
            LossModel: This model, changed in place.
        """
        return self.add_rule(ReplaceRule(operation, replacement))

    def add_decorate(self, operation, decoration=None, *, before=False):
        """Add a :class:`DecorateRule` and return the model.

        Args:
            operation: Operation pattern to match.
            decoration: Operation or instructions to add.
            before (bool): Add the decoration before the instruction instead
                of after it.

        Returns:
            LossModel: This model, changed in place.
        """
        return self.add_rule(DecorateRule(operation, decoration, before=before))

    def lower_losses(self, circuit: Circuit, rng=None):
        """Run :func:`lower_losses` on ``circuit`` with this model."""
        return lower_losses(circuit, rng=rng, lossmodel=self)

    def resolve_losses(self, circuit: Circuit, rng=None):
        """Run :func:`resolve_losses` on ``circuit`` with this model."""
        return resolve_losses(circuit, rng=rng, lossmodel=self)

    def describe(self):
        """Print a readable summary of the rules, in the order they are tried.

        Examples:
            >>> from mimiqcircuits import *
            >>> LossModel(name="demo").add_drop(GateCX()).describe()
            LossModel: demo
            ==================================================
            Rule 1: DropRule
              -> Drop CX when touching lost qubits
        """
        title = f"LossModel: {self.name}" if self.name else "LossModel"
        print(title)
        print("=" * 50)

        if not self.rules:
            print("  (no rules - all gates touching lost qubits will be dropped)")
            return

        for index, rule in enumerate(self.rules, start=1):
            print(f"Rule {index}: {type(rule).__name__}")

            if isinstance(rule, DropRule):
                if rule.operation is None:
                    print("  -> Drop any gate touching lost qubits")
                else:
                    print(f"  -> Drop {rule.operation} when touching lost qubits")
            elif isinstance(rule, ReplaceRule):
                print(
                    f"  -> Replace {rule.operation} with {rule.replacement} on surviving qubits"
                )
            elif isinstance(rule, DecorateRule):
                position = "before" if rule.before() else "after"
                print(
                    f"  -> Keep {rule.operation}, add {rule.decoration} {position} on surviving qubits"
                )
            elif isinstance(rule, CustomRule):
                print("  -> Custom rule (callable)")

    def saveproto(self, file):
        """Save the model to a protobuf file.

        Args:
            file: Path or writable binary file object.

        Raises:
            TypeError: If the model has a :class:`CustomRule`, which cannot
                be saved.
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load a model saved with :meth:`saveproto`.

        Args:
            file: Path or readable binary file object.

        Returns:
            LossModel: The loaded model.
        """
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, LossModel)

    def __str__(self):
        name = self.name if self.name else "unnamed"
        return f"LossModel ({name}, {len(self.rules)} rules)"

    def __repr__(self):
        if not self.rules:
            return str(self)

        lines = [str(self)]
        for index, rule in enumerate(self.rules):
            prefix = "└── " if index == len(self.rules) - 1 else "├── "
            lines.append(f"{prefix}{rule}")
        return "\n".join(lines)


def lossmodel_rewrite(inst: Instruction, lost, model: LossModel, rng=None):
    """Rewrite one partially lost instruction according to a loss model.

    This is the per-instruction step of :func:`lower_losses`. The rules of
    ``model`` are tried in order and the first that matches decides; emitted
    instructions that still touch a lost qubit are removed. When no rule
    matches, the instruction is dropped.

    Args:
        inst: Instruction touching both lost and present qubits.
        lost: Map from qubit index to ``True`` when the qubit is lost; missing
            qubits and those mapped to ``False`` are present.
        model: Loss model whose rules are applied.
        rng (random.Random): Passed to the generator of a :class:`CustomRule`;
            other rules ignore it. Defaults to a new ``random.Random()``.

    Returns:
        list[Instruction]: Instructions to emit in place of ``inst``, empty
        when it is dropped.
    """
    if rng is None:
        rng = random.Random()
    for rule in model.rules:
        if isinstance(rule, CustomRule):
            if not rule.matches(inst):
                continue
            result = _normalize_to_instructions(
                _call_custom_generator(rule.generator, inst, dict(lost), rng)
            )
        else:
            result = rule.apply_rule(inst)

        if result is None:
            continue

        # discard instructions where any target qubit is lost; first match wins
        return [
            replacement
            for replacement in result
            if not any(lost.get(q, False) for q in replacement.get_qubits())
        ]
    # No rule matched -> drop (default)
    return []


def _apply_lossmodel_rules(out, inst: Instruction, model: LossModel, lost, rng):
    for replacement in lossmodel_rewrite(inst, lost, model, rng):
        out.push(replacement)


def _sample_loss_probability(op: "Loss"):
    try:
        value = unwrapvalue(op.p)
    except UndefinedValue as exc:
        raise ValueError(
            "Loss probability must be numeric for sampling. "
            "Use evaluate() to substitute symbolic parameters first."
        ) from exc

    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError("Loss probability must be real for sampling.")
        value = value.real

    # Range is not re-checked here: the `Loss` constructor already rejects
    # out-of-range numeric `p`, and `_loss_fires` treats `p >= 1` as certain
    # loss.
    return value


def _loss_fires(op: "Loss", rng) -> bool:
    """Whether a ``Loss(p)`` fires. A certain loss (p == 1) always fires."""
    p = _sample_loss_probability(op)
    return p >= 1.0 or rng.random() < p


def _coerce_rng_lossmodel(rng, lossmodel):
    if isinstance(rng, LossModel):
        if lossmodel is not None:
            raise TypeError(
                "LossModel provided twice: once positionally and once via lossmodel=."
            )
        lossmodel = rng
        rng = None
    if rng is None:
        rng = random.Random()
    elif not hasattr(rng, "random"):
        raise TypeError("rng must provide a random() method.")
    if lossmodel is None:
        lossmodel = LossModel()
    return rng, lossmodel


def sample_losses(circuit: Circuit, rng=None):
    """Resolve the random qubit-loss events in a circuit.

    Each :class:`Loss` ``(p)`` is drawn: with probability ``p`` it becomes a
    certain loss ``Loss(1.0)``, otherwise it is removed. Every other
    operation, including :class:`Reload`, :class:`Check` and
    :class:`MeasureCheck`, is kept unchanged. The result is deterministic but
    still contains loss operations; :func:`lower_losses` turns it into a
    runnable, loss-free circuit, and :func:`resolve_losses` does both steps.

    Args:
        circuit: Circuit to sample. It is not changed.
        rng: Random number generator with a ``random()`` method, such as
            ``random.Random(seed)``. Defaults to a new ``random.Random()``.

    Returns:
        Circuit: A new circuit with only certain losses left.

    Raises:
        TypeError: If ``rng`` has no ``random()`` method.
        ValueError: If a loss probability is still symbolic.

    Examples:
        >>> from mimiqcircuits import *
        >>> import random
        >>> c = Circuit()
        >>> _ = c.push(Loss(0.0), 0)  # never fires: removed
        >>> _ = c.push(Loss(1.0), 1)  # always fires: kept
        >>> sample_losses(c, rng=random.Random(1))
        2-qubit circuit with 1 instruction:
        └── Loss(1.0) @ q[1]
        <BLANKLINE>
    """
    if rng is None:
        rng = random.Random()
    elif not hasattr(rng, "random"):
        raise TypeError("rng must provide a random() method.")

    out = Circuit()
    for inst in circuit:
        op = inst.get_operation()
        if isinstance(op, Loss):
            if _loss_fires(op, rng):
                out.push(Loss(1.0), inst.get_qubits()[0])
        else:
            out.push(inst)
    return out


def lower_losses(circuit: Circuit, rng=None, lossmodel: Optional[LossModel] = None):
    """Lower a circuit with loss operations into one using only primitives.

    This is the deterministic half of loss resolution. It expects the random
    ``Loss(p)`` events to be resolved already (see :func:`sample_losses`) and
    treats any remaining ``Loss`` as a certain loss. Going through the circuit
    in order and tracking which qubits are lost, it rewrites:

    - :class:`Loss` into a :class:`Lost` marker (nothing if the qubit is
      already lost);
    - :class:`Reload` into :class:`Reset` and a :class:`Reloaded` marker;
    - :class:`Check` into :class:`SetBit1` (present) or :class:`SetBit0`
      (lost);
    - :class:`MeasureCheck` into :class:`Measure` and :class:`SetBit1` when
      present, two :class:`SetBit0` when lost;
    - a one-qubit measurement of a lost qubit into :class:`SetBit0`.

    Other instructions are kept if they touch no lost qubit, dropped if they
    touch only lost qubits, and passed to ``lossmodel`` otherwise.

    Args:
        circuit: Circuit to lower. It is not changed.
        rng: Random number generator passed to :class:`CustomRule`
            generators. Defaults to a new ``random.Random()``. A
            :class:`LossModel` passed in this position is used as
            ``lossmodel``.
        lossmodel: Rules for instructions touching both lost and present
            qubits. Defaults to an empty model, which drops them.

    Returns:
        Circuit: A new circuit with no loss operations.

    Raises:
        TypeError: If ``rng`` has no ``random()`` method, or a loss model is
            given both positionally and as ``lossmodel``.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(Loss(), 0)
        >>> _ = c.push(MeasureCheck(), 0, 0, 1)  # lost: both bits set to 0
        >>> _ = c.push(Reload(), 0)               # back to |0>, present again
        >>> _ = c.push(Check(), 0, 2)             # present: bit set to 1
        >>> lower_losses(c)
        1-qubit, 3-bit circuit with 6 instructions:
        ├── Lost @ q[0]
        ├── c[0] = 0
        ├── c[1] = 0
        ├── Reset @ q[0]
        ├── Reloaded @ q[0]
        └── c[2] = 1
        <BLANKLINE>
    """
    rng, lossmodel = _coerce_rng_lossmodel(rng, lossmodel)

    lost = {}
    out = Circuit()
    for inst in circuit:
        op = inst.get_operation()
        qubits = tuple(inst.get_qubits())

        if isinstance(op, Loss):
            q = qubits[0]
            if lost.get(q, False):
                continue
            lost[q] = True
            out.push(Lost(), q)
            continue

        if isinstance(op, Reload):
            q = qubits[0]
            out.push(Reset(), q)
            out.push(Reloaded(), q)
            lost[q] = False
            continue

        if isinstance(op, Check):
            q = qubits[0]
            b = inst.get_bits()[0]
            out.push(SetBit0() if lost.get(q, False) else SetBit1(), b)
            continue

        if isinstance(op, MeasureCheck):
            q = qubits[0]
            bits = tuple(inst.get_bits())
            if lost.get(q, False):
                out.push(SetBit0(), bits[0])
                out.push(SetBit0(), bits[1])
            else:
                out.push(Measure(), q, bits[0])
                out.push(SetBit1(), bits[1])
            continue

        # a single-qubit measurement on a lost qubit reads 0
        if (
            isinstance(op, AbstractMeasurement)
            and len(qubits) == 1
            and lost.get(qubits[0], False)
        ):
            out.push(SetBit0(), inst.get_bits()[0])
            continue

        if not any(lost.get(q, False) for q in qubits):
            out.push(inst)
            continue

        if all(lost.get(q, False) for q in qubits):
            continue

        _apply_lossmodel_rules(out, inst, lossmodel, lost, rng)

    return out


def resolve_losses(circuit: Circuit, rng=None, lossmodel: Optional[LossModel] = None):
    """Fully resolve loss into a runnable, loss-free circuit.

    It runs :func:`sample_losses` to draw the random ``Loss(p)`` events, then
    :func:`lower_losses` to rewrite the result into primitives. Each call
    draws a new loss pattern, so resolve once per shot to sample loss
    statistics.

    Args:
        circuit: Circuit to resolve. It is not changed.
        rng: Random number generator with a ``random()`` method, used for both
            steps. Defaults to a new ``random.Random()``. A
            :class:`LossModel` passed in this position is used as
            ``lossmodel``.
        lossmodel: Rules for instructions touching both lost and present
            qubits. Defaults to an empty model, which drops them.

    Returns:
        Circuit: A new circuit with no loss operations.

    Raises:
        TypeError: If ``rng`` has no ``random()`` method, or a loss model is
            given both positionally and as ``lossmodel``.
        ValueError: If a loss probability is still symbolic.

    Examples:
        >>> from mimiqcircuits import *
        >>> # qubit 1 is lost for sure, so the CX touching it is dropped
        >>> c = Circuit()
        >>> _ = c.push(Loss(), 1)
        >>> _ = c.push(GateCX(), 0, 1)
        >>> resolve_losses(c)
        2-qubit circuit with 1 instruction:
        └── Lost @ q[1]
        <BLANKLINE>
    """
    rng, lossmodel = _coerce_rng_lossmodel(rng, lossmodel)
    return lower_losses(sample_losses(circuit, rng=rng), rng=rng, lossmodel=lossmodel)


def _normalize_loss_indices(loss_indices) -> set:
    if isinstance(loss_indices, int):
        loss_indices = (loss_indices,)
    indices = set()
    for idx in loss_indices:
        idx = int(idx)
        if idx < 1:
            raise ValueError(f"Loss indices must be >= 1, got {idx}.")
        indices.add(idx)
    return indices


def sample_loss_scenario(
    circuit: Circuit,
    loss_indices,
    *,
    p: float = 1.0,
    rng=None,
    lossmodel: Optional[LossModel] = None,
):
    """Build a deterministic "what-if" loss scenario from a circuit.

    The selected :class:`mimiqcircuits.Loss` instructions, counted in circuit
    order (1-based), are forced to ``Loss(p)`` while every other ``Loss`` is
    forced to ``Loss(0.0)``. The result is then resolved with
    :func:`resolve_losses`, so the returned circuit shows the effect of losing
    exactly those sites.

    Args:
        circuit: Circuit containing ``Loss`` instructions. It is not changed.
        loss_indices: A 1-based index or iterable of indices selecting which
            ``Loss`` instructions to force.
        p: Keyword-only. Probability assigned to the selected sites
            (default 1.0, a certain loss).
        rng: Keyword-only. Random number generator, used when ``p < 1`` and
            by :class:`CustomRule` generators.
        lossmodel: Keyword-only. Rules for instructions touching both lost
            and present qubits.

    Returns:
        Circuit: A loss-free circuit for the chosen scenario.

    Raises:
        ValueError: If an index is below 1 or beyond the number of ``Loss``
            instructions, ``p`` is outside :math:`[0, 1]`, or the circuit has
            no ``Loss`` instruction.

    Examples:
        >>> from mimiqcircuits import *
        >>> # two possible loss sites; force the second one only
        >>> circuit = Circuit()
        >>> _ = circuit.push(Loss(0.2), 0)
        >>> _ = circuit.push(GateCX(), 0, 1)
        >>> _ = circuit.push(Loss(0.4), 1)
        >>> sample_loss_scenario(circuit, 2)
        2-qubit circuit with 2 instructions:
        ├── CX @ q[0], q[1]
        └── Lost @ q[1]
        <BLANKLINE>
    """
    forced_indices = _normalize_loss_indices(loss_indices)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probability p must be between 0 and 1, got {p}.")

    scenario = Circuit()
    current_index = 0
    for inst in circuit:
        op = inst.get_operation()
        if isinstance(op, Loss):
            current_index += 1
            forced_p = p if current_index in forced_indices else 0.0
            scenario.push(Loss(forced_p), inst.get_qubits()[0])
        else:
            scenario.push(inst)

    if current_index == 0:
        raise ValueError("Circuit does not contain any Loss instructions.")

    invalid = sorted(i for i in forced_indices if i > current_index)
    if invalid:
        raise ValueError(
            f"Loss index/indices {invalid} out of range for a circuit with "
            f"{current_index} Loss instruction(s)."
        )

    return resolve_losses(scenario, rng=rng, lossmodel=lossmodel)


__all__ = [
    "AbstractCircuitRule",
    "DropRule",
    "DecorateRule",
    "ReplaceRule",
    "CustomRule",
    "LossModel",
    "sample_losses",
    "lower_losses",
    "resolve_losses",
    "sample_loss_scenario",
]
