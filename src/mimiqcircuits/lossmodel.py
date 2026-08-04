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
    >>> circuit = Circuit()
    >>> _ = circuit.push(Loss(), 1)
    >>> _ = circuit.push(GateCX(), 0, 1)
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

import mimiqcircuits as mc
from mimiqcircuits.circuitrules import AbstractCircuitRule
from mimiqcircuits.symbolics import (
    UndefinedValue,
    _extract_variables,
    _validate_rule_gate_params,
    applyparams,
    unwrapvalue,
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


def _matches_operation_pattern(op_inst: mc.Operation, op_rule: mc.Operation) -> bool:
    if type(op_inst) is not type(op_rule):
        return False

    if not _is_symbolic_operation_pattern(op_rule):
        return op_inst == op_rule

    return True


def _resolve_symbolic_replacement(
    op_inst: mc.Operation, op_rule: mc.Operation, replacement: mc.Operation
):
    if not _is_symbolic_operation_pattern(op_rule):
        return replacement

    variables = _extract_variables(op_rule)
    if variables is None or all(var is None for var in variables):
        return replacement

    return applyparams(op_inst, (variables, replacement))


def _normalize_to_instructions(result) -> List[mc.Instruction]:
    if result is None:
        return []

    if isinstance(result, mc.Instruction):
        return [result]

    if hasattr(result, "instructions"):
        insts = list(result.instructions)
    elif isinstance(result, (list, tuple)):
        insts = list(result)
    else:
        raise TypeError(
            "Expected None, Instruction, Circuit, Block, or a sequence of Instructions."
        )

    if not all(isinstance(inst, mc.Instruction) for inst in insts):
        raise TypeError("All generated items must be Instruction instances.")

    return insts


def _validate_replacement_operation(
    operation: mc.Operation, replacement: mc.Operation, label: str
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
    operation: mc.Operation, instructions: List[mc.Instruction], label: str
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


def _build_replacement_instructions(replacement, op_pattern: mc.Operation, inst):
    if isinstance(replacement, mc.Operation):
        op_inst = inst.get_operation()
        qs = inst.get_qubits()
        resolved = _resolve_symbolic_replacement(op_inst, op_pattern, replacement)

        if resolved.num_qubits == len(qs):
            return [mc.Instruction(resolved, tuple(qs))]

        return [mc.Instruction(resolved, (q,)) for q in qs]

    qubit_map = {i: q for i, q in enumerate(inst.get_qubits())}
    return [
        mc.Instruction(
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

    A ``DropRule`` is useful when a partially affected operation should not be
    salvaged. If ``operation`` is omitted, the rule matches any operation that
    reaches the loss model.

    Args:
        operation (optional): Operation pattern to drop. If omitted, the rule
            is a catch-all rule.

    Examples:
        >>> from mimiqcircuits import *
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

    def __init__(self, operation: Optional[mc.Operation] = None):
        if operation is not None:
            _validate_rule_operation_target(operation)
            if _supports_symbolic_operation_pattern(operation):
                _validate_rule_gate_params(operation)
        self.operation = operation

    def priority(self):
        return 0

    def matches(self, inst: mc.Instruction):
        if self.operation is None:
            return True
        return _matches_operation_pattern(inst.get_operation(), self.operation)

    def replaces(self):
        return True

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None
        return []

    def __str__(self):
        if self.operation is None:
            return "DropRule(*)"
        return f"DropRule({self.operation})"

    __repr__ = __str__


class DecorateRule(AbstractCircuitRule):
    """Add a decoration before or after a matched instruction.

    During loss sampling, any generated instruction that still touches a lost
    qubit is filtered out. This makes ``DecorateRule`` useful for modeling a
    side effect on surviving qubits when an operation was attempted but one of
    its qubits was missing.

    Args:
        operation: Operation pattern to match.
        decoration: Operation or instruction sequence to add.
        before (bool): If ``True``, add the decoration before the matched
            instruction. Otherwise, add it after.

    Examples:
        >>> from mimiqcircuits import *
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

        if isinstance(decoration, mc.Operation):
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

    def matches(self, inst: mc.Instruction):
        return _matches_operation_pattern(inst.get_operation(), self.operation)

    def apply_rule(self, inst: mc.Instruction):
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
    and replaced by another operation on surviving qubits. A one-qubit
    replacement is broadcast to each target of the matched instruction, and
    copies on lost qubits are filtered out.

    Args:
        operation: Operation pattern to match.
        replacement: Replacement operation or instruction sequence.

    Examples:
        >>> from mimiqcircuits import *
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

        if isinstance(replacement, mc.Operation):
            _validate_replacement_operation(operation, replacement, "Replacement")
            self.replacement = replacement
        else:
            replacement_insts = _normalize_to_instructions(replacement)
            _validate_replacement_instructions(
                operation, replacement_insts, "Replacement"
            )
            self.replacement = replacement_insts

    def matches(self, inst: mc.Instruction):
        return _matches_operation_pattern(inst.get_operation(), self.operation)

    def replaces(self):
        return True

    def apply_rule(self, inst: mc.Instruction):
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

    Args:
        matcher: Callable that receives an instruction and returns ``True`` if
            the rule should apply.
        generator: Callable that generates replacement instructions.

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

    def matches(self, inst: mc.Instruction):
        return self.matcher(inst)

    def replaces(self):
        return True

    def apply_rule(self, inst: mc.Instruction):
        if not self.matches(inst):
            return None
        return _normalize_to_instructions(
            _call_custom_generator(self.generator, inst, {}, None)
        )

    def __str__(self):
        return "CustomRule(<callable>)"

    __repr__ = __str__


class LossModel:
    """Collection of prioritized loss rules.

    A ``LossModel`` tells :func:`sample_losses` what to do when an instruction
    touches both lost and surviving qubits. With no rules, the conservative
    behavior is used: instructions touching lost qubits are dropped.

    Args:
        rules (optional): Iterable of loss rules.
        name (str): Optional model name used in display output.

    Examples:
        >>> from mimiqcircuits import *
        >>> model = LossModel(name="My Loss Model")
        >>> model
        LossModel (My Loss Model, 0 rules)

        Rules can be added incrementally:

        >>> model.add_replace(GateCX(), Depolarizing1(0.2))
        LossModel (My Loss Model, 1 rules)
        └── ReplaceRule(CX => Depolarizing(0.2))

        The model can then be passed to ``sample_losses``:

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
        if not isinstance(rule, AbstractCircuitRule):
            raise TypeError("LossModel rules must inherit from AbstractCircuitRule.")
        self.rules.append(rule)
        self._sort_rules()
        return self

    def add_drop(self, operation: Optional[mc.Operation] = None):
        return self.add_rule(DropRule(operation))

    def add_replace(self, operation, replacement=None):
        return self.add_rule(ReplaceRule(operation, replacement))

    def add_decorate(self, operation, decoration=None, *, before=False):
        return self.add_rule(DecorateRule(operation, decoration, before=before))

    def lower_losses(self, circuit: mc.Circuit, rng=None):
        return lower_losses(circuit, rng=rng, lossmodel=self)

    def resolve_losses(self, circuit: mc.Circuit, rng=None):
        return resolve_losses(circuit, rng=rng, lossmodel=self)

    def describe(self):
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
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
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


def lossmodel_rewrite(inst: mc.Instruction, lost, model: LossModel, rng):
    """Decide how a single instruction touching both lost and present qubits is
    rewritten under a :class:`LossModel`, returning the instructions to emit in
    its place (an empty list when the instruction is dropped).

    This is the per-instruction decision core shared by the offline
    :func:`lower_losses` and the online runtime-loss driver: both feed it the
    same ``lost`` map (qubit -> lost?) and apply whatever it returns. Rules are
    tried in order, the first matching rule wins, and any emitted instruction
    still touching a lost qubit is filtered out. When no rule matches, the
    instruction is dropped.
    """
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


def _apply_lossmodel_rules(out, inst: mc.Instruction, model: LossModel, lost, rng):
    for replacement in lossmodel_rewrite(inst, lost, model, rng):
        out.push(replacement)


def _sample_loss_probability(op: "mc.Loss"):
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
    # loss, matching the Julia `_loss_fires`.
    return value


def _loss_fires(op: "mc.Loss", rng) -> bool:
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


def sample_losses(circuit: mc.Circuit, rng=None):
    """Resolve the random qubit-loss events in a circuit.

    Each :class:`mimiqcircuits.Loss` ``(p)`` is drawn: with probability ``p`` it
    becomes a certain loss ``Loss(1.0)``, otherwise it is removed. Every other
    operation, including the deterministic loss bookkeeping (``Reload``,
    ``Check``, ``MeasureCheck``), is kept unchanged. The result is deterministic
    but still expressed with loss operations; use :func:`lower_losses` or
    :func:`resolve_losses` to turn it into a runnable, loss-free circuit.

    Args:
        circuit: Circuit to sample.
        rng (optional): Random number generator providing a ``random()`` method.

    Returns:
        mc.Circuit: A circuit whose random loss events have been resolved.
    """
    if rng is None:
        rng = random.Random()
    elif not hasattr(rng, "random"):
        raise TypeError("rng must provide a random() method.")

    out = mc.Circuit()
    for inst in circuit:
        op = inst.get_operation()
        if isinstance(op, mc.Loss):
            if _loss_fires(op, rng):
                out.push(mc.Loss(1.0), inst.get_qubits()[0])
        else:
            out.push(inst)
    return out


def lower_losses(circuit: mc.Circuit, rng=None, lossmodel: Optional[LossModel] = None):
    """Lower a circuit with loss operations into one using only primitives.

    This is the deterministic half of loss resolution. It expects the random
    ``Loss(p)`` events to be resolved already (see :func:`sample_losses`) and
    treats any remaining ``Loss`` as a certain loss. It rewrites every loss
    operation into ``Reset`` / ``Measure`` / ``SetBit0`` / ``SetBit1`` plus
    passive ``Lost`` / ``Reloaded`` markers, and drops or replaces gates on lost
    qubits according to ``lossmodel``. The returned circuit contains no loss
    operations.

    Args:
        circuit: Circuit to lower.
        rng (optional): Random number generator for stochastic loss-model rules.
            A positional :class:`LossModel` is also accepted.
        lossmodel (optional): A :class:`LossModel` for gates touching lost qubits.

    Returns:
        mc.Circuit: A loss-free circuit that runs on any backend.
    """
    rng, lossmodel = _coerce_rng_lossmodel(rng, lossmodel)

    lost = {}
    out = mc.Circuit()
    for inst in circuit:
        op = inst.get_operation()
        qubits = tuple(inst.get_qubits())

        if isinstance(op, mc.Loss):
            q = qubits[0]
            if lost.get(q, False):
                continue
            lost[q] = True
            out.push(mc.Lost(), q)
            continue

        if isinstance(op, mc.Reload):
            q = qubits[0]
            out.push(mc.Reset(), q)
            out.push(mc.Reloaded(), q)
            lost[q] = False
            continue

        if isinstance(op, mc.Check):
            q = qubits[0]
            b = inst.get_bits()[0]
            out.push(mc.SetBit0() if lost.get(q, False) else mc.SetBit1(), b)
            continue

        if isinstance(op, mc.MeasureCheck):
            q = qubits[0]
            bits = tuple(inst.get_bits())
            if lost.get(q, False):
                out.push(mc.SetBit0(), bits[0])
                out.push(mc.SetBit0(), bits[1])
            else:
                out.push(mc.Measure(), q, bits[0])
                out.push(mc.SetBit1(), bits[1])
            continue

        # a single-qubit measurement on a lost qubit reads 0
        if (
            isinstance(op, mc.AbstractMeasurement)
            and len(qubits) == 1
            and lost.get(qubits[0], False)
        ):
            out.push(mc.SetBit0(), inst.get_bits()[0])
            continue

        if not any(lost.get(q, False) for q in qubits):
            out.push(inst)
            continue

        if all(lost.get(q, False) for q in qubits):
            continue

        _apply_lossmodel_rules(out, inst, lossmodel, lost, rng)

    return out


def resolve_losses(circuit: mc.Circuit, rng=None, lossmodel: Optional[LossModel] = None):
    """Fully resolve loss into a runnable, loss-free circuit.

    The user-facing entry point: it runs :func:`sample_losses` to draw the
    random ``Loss(p)`` events, then :func:`lower_losses` to rewrite the result
    into primitives. The returned circuit contains no loss operations.

    Args:
        circuit: Circuit to resolve.
        rng (optional): Random number generator. A positional :class:`LossModel`
            is also accepted.
        lossmodel (optional): A :class:`LossModel` for gates touching lost qubits.

    Returns:
        mc.Circuit: A loss-free circuit that runs on any backend.
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
    circuit: mc.Circuit,
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
        circuit: Circuit containing ``Loss`` instructions.
        loss_indices: A 1-based index or iterable of indices selecting which
            ``Loss`` instructions to force.
        p (optional): Probability assigned to the selected sites (default 1.0).
        rng (optional): Random number generator.
        lossmodel (optional): A :class:`LossModel` for gates touching lost qubits.

    Returns:
        mc.Circuit: A loss-free circuit for the chosen scenario.

    Examples:
        >>> from mimiqcircuits import *
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

    scenario = mc.Circuit()
    current_index = 0
    for inst in circuit:
        op = inst.get_operation()
        if isinstance(op, mc.Loss):
            current_index += 1
            forced_p = p if current_index in forced_indices else 0.0
            scenario.push(mc.Loss(forced_p), inst.get_qubits()[0])
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
