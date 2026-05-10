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
"""Loss model definitions and qubit-loss sampling."""

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
    """Drop matched instructions."""

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
    """Keep a matched instruction and add a decoration around it."""

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
    """Replace a matched instruction with new instructions."""

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
    """User-defined loss rule."""

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
    """Collection of prioritized loss rules."""

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

    def sample_losses(self, circuit: mc.Circuit, rng=None):
        return sample_losses(circuit, rng=rng, lossmodel=self)

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


def _apply_lossmodel_rules(out, inst: mc.Instruction, model: LossModel, lost, rng):
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

        filtered = [
            replacement
            for replacement in result
            if not any(lost.get(q, False) for q in replacement.get_qubits())
        ]
        for replacement in filtered:
            out.push(replacement)
        return


def _sample_loss_probability(op: mc.LossErr):
    try:
        value = unwrapvalue(op.p)
    except UndefinedValue as exc:
        raise ValueError(
            "LossErr probability must be numeric for sampling. "
            "Use evaluate() to substitute symbolic parameters first."
        ) from exc

    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError("LossErr probability must be real for sampling.")
        value = value.real

    if not (0 <= value <= 1):
        raise ValueError("LossErr probability must be between 0 and 1 for sampling.")

    return value


def _process_losserr(out, op: mc.LossErr, qubits, lost, rng):
    q = qubits[0]

    if lost.get(q, False):
        return

    if rng.random() < _sample_loss_probability(op):
        lost[q] = True
        out.push(mc.QubitLoss(), q)


def _process_qubitloss(out, op: mc.QubitLoss, qubits, lost):
    q = qubits[0]
    lost[q] = True
    out.push(mc.Instruction(op, tuple(qubits)))


def _process_qubitreload(out, op: mc.QubitReload, qubits, lost):
    q = qubits[0]

    if lost.get(q, False):
        lost[q] = False
        out.push(mc.Instruction(op, tuple(qubits)))


def sample_losses(circuit: mc.Circuit, rng=None, lossmodel: Optional[LossModel] = None):
    """
    Sample qubit-loss events in a circuit and apply a loss model.

    Args:
        circuit: Circuit to sample.
        rng (optional): Random number generator used to sample ``LossErr`` events.
            Any object providing a ``random()`` method is accepted.
        lossmodel (optional): A :class:`LossModel` describing how to rewrite gates
            touching lost qubits. A positional :class:`LossModel` is also accepted as
            the second argument.

    Returns:
        mc.Circuit: A new circuit where loss events are sampled and the
        corresponding loss rules are applied.

    Examples:
        Reusing one seeded RNG advances its state and yields a reproducible sequence:

        >>> import random
        >>> from mimiqcircuits import Circuit, GateH, LossErr, sample_losses
        >>> c = Circuit()
        >>> c.push(LossErr(0.5), 1)
        2-qubit circuit with 1 instruction:
        └── LossErr(0.5) @ q[1]
        <BLANKLINE>
        >>> c.push(GateH(), 1)
        2-qubit circuit with 2 instructions:
        ├── LossErr(0.5) @ q[1]
        └── H @ q[1]
        <BLANKLINE>
        >>> rng = random.Random(70)
        >>> sample_losses(c, rng=rng)
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
        >>> sample_losses(c, rng=rng)
        2-qubit circuit with 1 instruction:
        └── QubitLoss @ q[1]
        <BLANKLINE>

        Creating a fresh RNG with the same seed for each call repeats the same sample:

        >>> sample_losses(c, rng=random.Random(20))
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
        >>> sample_losses(c, rng=random.Random(20))
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
    """

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

    lost = {}
    out = mc.Circuit()

    for inst in circuit:
        op = inst.get_operation()
        qubits = tuple(inst.get_qubits())

        if isinstance(op, mc.LossErr):
            _process_losserr(out, op, qubits, lost, rng)
            continue

        if isinstance(op, mc.QubitLoss):
            _process_qubitloss(out, op, qubits, lost)
            continue

        if isinstance(op, mc.QubitReload):
            _process_qubitreload(out, op, qubits, lost)
            continue

        if isinstance(op, (mc.CheckLoss, mc.MeasureCheckLoss)):
            out.push(inst)
            continue

        if not any(lost.get(q, False) for q in qubits):
            out.push(inst)
            continue

        if all(lost.get(q, False) for q in qubits):
            continue

        _apply_lossmodel_rules(out, inst, lossmodel, lost, rng)

    return out


__all__ = [
    "AbstractCircuitRule",
    "DropRule",
    "DecorateRule",
    "ReplaceRule",
    "CustomRule",
    "LossModel",
    "sample_losses",
]
