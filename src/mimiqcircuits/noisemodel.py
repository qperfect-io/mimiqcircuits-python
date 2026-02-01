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
"""Noise model definitions and application."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set, Union
import mimiqcircuits as mc
from mimiqcircuits.symbolics import (
    _extract_variables,
    _validate_rule_gate_params,
    applyparams,
)


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
        return 90

    def matches(self, inst: mc.Instruction):
        return isinstance(inst.get_operation(), mc.AbstractMeasurement)

    def apply_rule(self, inst: mc.Instruction):
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
        return 70

    def matches(self, inst: mc.Instruction):
        op = inst.get_operation()
        return (
            isinstance(op, mc.AbstractMeasurement)
            and tuple(inst.get_qubits()) == self.qubits
        )

    def apply_rule(self, inst: mc.Instruction):
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
        return 80

    def matches(self, inst: mc.Instruction):
        op = inst.get_operation()
        return isinstance(op, mc.AbstractMeasurement) and all(
            q in self._qubit_set for q in inst.get_qubits()
        )

    def apply_rule(self, inst: mc.Instruction):
        return mc.Instruction(self.noise, tuple(), tuple(inst.get_bits()), tuple())


@dataclass(frozen=True)
class GateInstanceNoise(AbstractNoiseRule):
    gate: mc.Gate
    noise: Union[mc.krauschannel, mc.Gate]
    before: bool = False

    def __post_init__(self):
        _validate_rule_gate_params(self.gate)

        # Enforce arity equality
        if self.gate._num_qubits != self.noise._num_qubits:
            raise ValueError("Noise must have same arity as gate")

        # Turn before into a method
        val = self.before
        object.__delattr__(self, "before")
        object.__setattr__(
            self, "before", (lambda self=self, v=val: v).__get__(self, type(self))
        )

    def priority(self):
        return 80

    def matches(self, inst: mc.Instruction):
        op_inst = inst.get_operation()
        op_rule = self.gate
        if type(op_inst) is not type(op_rule):
            return False

        if not op_rule.is_symbolic():
            return op_inst == op_rule

        if op_inst.is_symbolic():
            return False

        return True

    def apply_rule(self, inst: mc.Instruction):
        op_inst = inst.get_operation()

        variables = _extract_variables(self.gate)

        if variables is None:
            noise_op = self.noise
        else:
            noise_op = applyparams(op_inst, (variables, self.noise))

        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class ExactGateInstanceQubitNoise(AbstractNoiseRule):
    gate: mc.Gate
    qubits: Sequence[int]
    noise: Union[mc.krauschannel, mc.Gate]
    before: bool = False

    def __post_init__(self):
        _validate_rule_gate_params(self.gate)

        qs = list(self.qubits)
        if len(set(qs)) != len(qs):
            raise ValueError("Duplicate qubits not allowed")
        if len(qs) != self.gate._num_qubits:
            raise ValueError("Qubit list length mismatch")
        if self.gate._num_qubits != self.noise._num_qubits:
            raise ValueError("Noise arity mismatch")
        object.__setattr__(self, "qubits", tuple(qs))

        val = self.before
        object.__delattr__(self, "before")
        object.__setattr__(
            self, "before", (lambda self=self, v=val: v).__get__(self, type(self))
        )

    def priority(self):
        return 40

    def matches(self, inst: mc.Instruction):
        op_inst = inst.get_operation()
        op_rule = self.gate

        # Check gate type
        if type(op_inst) is not type(op_rule):
            return False

        # Check exact qubit order
        if tuple(inst.get_qubits()) != self.qubits:
            return False

        # Concrete rule → require exact match
        if not op_rule.is_symbolic():
            return op_inst == op_rule

        # Symbolic rule → match only concrete gate instance
        if op_inst.is_symbolic():
            return False

        return True

    def apply_rule(self, inst: mc.Instruction):
        op_inst = inst.get_operation()

        # Extract symbolic variables from gate pattern
        variables = _extract_variables(self.gate)

        if variables is None:
            noise_op = self.noise
        else:
            # relation: variables => noise
            noise_op = applyparams(op_inst, (variables, self.noise))

        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class SetGateInstanceQubitNoise(AbstractNoiseRule):
    gate: mc.Gate
    qubits: Iterable[int]
    noise: Union[mc.krauschannel, mc.Gate]
    before: bool = False
    _qubit_set: frozenset = field(init=False, repr=False)

    def __post_init__(self):
        _validate_rule_gate_params(self.gate)

        qs = list(self.qubits)

        if len(qs) < self.gate._num_qubits:
            raise ValueError("Qubit set too small for gate arity")

        if len(qs) != len(set(qs)):
            raise ValueError("Duplicate qubits not allowed")

        if self.gate._num_qubits != self.noise._num_qubits:
            raise ValueError("Noise arity mismatch")
        object.__setattr__(self, "_qubit_set", frozenset(qs))

        val = self.before
        object.__delattr__(self, "before")
        object.__setattr__(
            self, "before", (lambda self=self, v=val: v).__get__(self, type(self))
        )

    def priority(self):
        return 50

    def matches(self, inst: mc.Instruction):
        op_inst = inst.get_operation()
        op_rule = self.gate

        if type(op_inst) is not type(op_rule):
            return False

        if not all(q in self._qubit_set for q in inst.get_qubits()):
            return False

        if not op_rule.is_symbolic():
            return op_inst == op_rule

        if op_inst.is_symbolic():
            return False

        return True

    def apply_rule(self, inst: mc.Instruction):
        op_inst = inst.get_operation()

        variables = _extract_variables(self.gate)

        if variables is None:
            noise_op = self.noise
        else:
            noise_op = applyparams(op_inst, (variables, self.noise))

        return mc.Instruction(noise_op, tuple(inst.get_qubits()))


@dataclass(frozen=True)
class IdleNoise(AbstractNoiseRule):
    relation: Union[tuple, mc.Operation]
    qubits: Optional[Iterable[int]] = None

    def __post_init__(self):
        # Convert qubits to frozenset if provided
        if self.qubits is not None:
            object.__setattr__(self, "qubits", frozenset(self.qubits))

        # Validate relation
        # Case 1: constant noise operation
        if isinstance(self.relation, mc.Operation):
            return

        # Case 2: relation must be (variable, target_operation)
        if not isinstance(self.relation, tuple) or len(self.relation) != 2:
            raise ValueError("IdleNoise relation must be (var, target) or Operation")

        var, target = self.relation

        # var must be a symbolic variable
        if not (hasattr(var, "is_symbol") and var.is_symbol):
            raise ValueError(
                f"Left side must be a simple symbolic variable, got: {var}"
            )

        if not isinstance(target, mc.Operation):
            raise ValueError("Right side of idle relation must be an Operation")

    def priority(self):
        # Idle noise has lowest priority
        return 200

    def replaces(self):
        return True

    def matches(self, inst: mc.Instruction):
        op = inst.get_operation()

        # Must be Delay operation
        if not isinstance(op, mc.Delay):
            return False

        # Global idle noise
        if self.qubits is None:
            return True

        # Restricted idle noise
        return all(q in self.qubits for q in inst.get_qubits())

    def apply_rule(self, inst: mc.Instruction):
        op_inst = inst.get_operation()

        # Case 1: constant noise
        if isinstance(self.relation, mc.Operation):
            noise_op = self.relation

        # Case 2: symbolic relation (var => target)
        else:
            var, target = self.relation
            # relation passed as (variables, noise_target)
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
        return 199

    def replaces(self):
        return True

    def matches(self, inst: mc.Instruction):
        op = inst.get_operation()
        if not isinstance(op, mc.Delay):
            return False
        return all(q in self.qubits for q in inst.get_qubits())

    def apply_rule(self, inst: mc.Instruction):
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
    priority_val: int = 100
    before: bool = False
    replace: bool = False

    def __post_init__(self):
        val = self.before
        object.__delattr__(self, "before")
        object.__setattr__(
            self, "before", (lambda self=self, v=val: v).__get__(self, type(self))
        )

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
    rules: List[AbstractNoiseRule] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        self.rules.sort(key=lambda r: r.priority())

    def add_rule(self, rule: AbstractNoiseRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority())
        return self

    def add_readout_noise(self, noise: mc.ReadoutErr, *, qubits=None, exact=False):
        if qubits is None:
            rule = GlobalReadoutNoise(noise)
        elif exact:
            rule = ExactQubitReadoutNoise(tuple(qubits), noise)
        else:
            rule = SetQubitReadoutNoise(tuple(qubits), noise)
        return self.add_rule(rule)

    def add_gate_noise(self, gate, noise, *, qubits=None, exact=False, before=False):
        if qubits is None:
            rule = GateInstanceNoise(gate, noise, before=before)
        elif exact:
            rule = ExactGateInstanceQubitNoise(
                gate, tuple(qubits), noise, before=before
            )
        else:
            rule = SetGateInstanceQubitNoise(gate, tuple(qubits), noise, before=before)
        return self.add_rule(rule)

    def add_idle_noise(self, noise, qubits=None):
        if qubits is None:
            rule = IdleNoise(noise)
        else:
            rule = SetIdleQubitNoise(noise, qubits)
        self.add_rule(rule)
        return rule

    def describe(self):
        title = f"NoiseModel: {self.name}" if self.name else "NoiseModel"
        print(title)
        print("=" * 80)

        for i, r in enumerate(self.rules, 1):
            print(f"Rule {i} (priority {r.priority()}): {type(r).__name__}")

            if isinstance(r, GlobalReadoutNoise):
                print(f"  → Applies {r.noise} to all measurements")

            elif isinstance(r, ExactQubitReadoutNoise):
                print(
                    f"  → Applies {r.noise} "
                    f"to measurements on qubits {list(r.qubits)} (exact order)"
                )

            elif isinstance(r, SetQubitReadoutNoise):
                print(
                    f"  → Applies {r.noise} "
                    f"to measurements on qubits in {sorted(r._qubit_set)}"
                )

            elif isinstance(r, GateInstanceNoise):
                print(f"  → Applies {r.noise} to gate pattern {r.gate}")

            elif isinstance(r, ExactGateInstanceQubitNoise):
                print(
                    f"  → Applies {r.noise} to gate pattern {r.gate} "
                    f"on qubits {list(r.qubits)} (exact order)"
                )

            elif isinstance(r, SetGateInstanceQubitNoise):
                print(
                    f"  → Applies {r.noise} to gate pattern {r.gate} "
                    f"on qubits in {sorted(r._qubit_set)}"
                )

            elif isinstance(r, IdleNoise):
                if isinstance(r.relation, mc.Operation):
                    print(f"  → Applies {r.relation} to all idle qubits")
                else:
                    var, target = r.relation
                    print(
                        f"  → Applies time-dependent idle noise {target} "
                        f"with variable {var} to all idle qubits"
                    )

            elif isinstance(r, SetIdleQubitNoise):
                if isinstance(r.relation, mc.Operation):
                    print(f"  → Applies {r.relation} to idle qubits {sorted(r.qubits)}")
                else:
                    var, target = r.relation
                    print(
                        f"  → Applies time-dependent idle noise {target} "
                        f"with variable {var} to idle qubits {sorted(r.qubits)}"
                    )

            elif isinstance(r, CustomNoiseRule):
                print("  → Custom rule (user-defined matcher)")

            if hasattr(r, "before") and callable(r.before) and r.before():
                print("    (applied BEFORE the operation)")

            if hasattr(r, "replaces") and callable(r.replaces) and r.replaces():
                print("    (REPLACES the original operation)")

            print("-" * 80)

    def saveproto(self, file):
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, NoiseModel)


def apply_noise_model(circuit: mc.Circuit, model: NoiseModel):
    """apply_noise_model(Circuit(), NoiseModel())

    Apply a noise model to a quantum circuit.

    This function takes a noiseless circuit and a NoiseModel containing one or
    more noise rules (readout, gate-type, idle, etc.), and returns a new circuit
    where noise operations have been inserted according to the model.

    The rules are applied in order of priority (lower value = higher priority).
    Each instruction is matched against the model’s rules, and if a rule matches,
    its corresponding noise operation is inserted before or after the original
    instruction, depending on the rule’s definition.

    Args:
        circuit (mc.Circuit):
            The noiseless input circuit to which the noise model will be applied.
        model (NoiseModel):
            The collection of noise rules defining how noise should be applied.

    Examples:

        >>> from mimiqcircuits import *

        **Adds readout error to all measurements (single or multi-qubit)**

        >>> nm_global = NoiseModel(name="Global Readout")
        >>> nm_global.add_rule(GlobalReadoutNoise(ReadoutErr(0.02, 0.03)))
        NoiseModel(rules=[GlobalReadoutNoise(noise=RErr(0.02,0.03))], name='Global Readout')

        >>> c = Circuit()
        >>> c.push(GateH(), 1)
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
        >>> c.push(GateCX(), 1, 2)
        3-qubit circuit with 2 instructions:
        ├── H @ q[1]
        └── CX @ q[1], q[2]
        <BLANKLINE>
        >>> c.push(MeasureZZ(), 1, 2, 1)
        3-qubit, 2-bit circuit with 3 instructions:
        ├── H @ q[1]
        ├── CX @ q[1], q[2]
        └── MZZ @ q[1:2], c[1]
        <BLANKLINE>
        >>> noisy = apply_noise_model(c, nm_global)
        >>> noisy
        3-qubit, 2-bit circuit with 4 instructions:
        ├── H @ q[1]
        ├── CX @ q[1], q[2]
        ├── MZZ @ q[1:2], c[1]
        └── RErr(0.02, 0.03) @ c[1]
        <BLANKLINE>

        **Adds noise to every instance of a gate type (e.g., all H gates) Global**

        >>> c = Circuit()
        >>> c.push(GateH(), 1)
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
        >>> c.push(GateCX(), 1, 2)
        3-qubit circuit with 2 instructions:
        ├── H @ q[1]
        └── CX @ q[1], q[2]
        <BLANKLINE>
        >>> c.push(MeasureZZ(), 1, 2, 1)
        3-qubit, 2-bit circuit with 3 instructions:
        ├── H @ q[1]
        ├── CX @ q[1], q[2]
        └── MZZ @ q[1:2], c[1]
        <BLANKLINE>

        >>> nm_gate = NoiseModel(name="Hadamard Gate Noise")
        >>> nm_gate.add_rule(GateInstanceNoise(GateH(), AmplitudeDamping(0.001)))
        NoiseModel(rules=[GateInstanceNoise(gate=H, noise=AmplitudeDamping(0.001), before=<bound method GateInstanceNoise.__post_init__.<locals>.<lambda> of ...>)], name='Hadamard Gate Noise')
        >>> noisy_c = apply_noise_model(c, nm_gate)
        >>> noisy_c
        3-qubit, 2-bit circuit with 4 instructions:
        ├── H @ q[1]
        ├── AmplitudeDamping(0.001) @ q[1]
        ├── CX @ q[1], q[2]
        └── MZZ @ q[1:2], c[1]
        <BLANKLINE>

        **Local: Restricts noise to gates that act only on a certain qubit set**

        >>> c = Circuit()
        >>> c.push(GateH(), 1)    # affected
        2-qubit circuit with 1 instruction:
        └── H @ q[1]
        <BLANKLINE>
        >>> c.push(GateH(), 4)    # not affected
        5-qubit circuit with 2 instructions:
        ├── H @ q[1]
        └── H @ q[4]
        <BLANKLINE>
        >>> c.push(GateCX(), 1, 2)
        5-qubit circuit with 3 instructions:
        ├── H @ q[1]
        ├── H @ q[4]
        └── CX @ q[1], q[2]
        <BLANKLINE>

        >>> nm_set_qubits = NoiseModel(name="Hadamard on {1,2,3}")
        >>> nm_set_qubits.add_rule(SetGateInstanceQubitNoise(GateH(), [1,2,3], AmplitudeDamping(0.002)))
        NoiseModel(rules=[SetGateInstanceQubitNoise(gate=H, qubits=[1, 2, 3], noise=AmplitudeDamping(0.002), before=<bound method SetGateInstanceQubitNoise.__post_init__.<locals>.<lambda> of ...>)], name='Hadamard on {1,2,3}')
        >>> nm_set_qubits.describe()
        NoiseModel: Hadamard on {1,2,3}
        ================================================================================
        Rule 1 (priority 50): SetGateInstanceQubitNoise
          → Applies AmplitudeDamping(0.002) to gate pattern H on qubits in [1, 2, 3]
        --------------------------------------------------------------------------------
        >>> noisy_c = apply_noise_model(c, nm_set_qubits)
        >>> noisy_c
        5-qubit circuit with 4 instructions:
        ├── H @ q[1]
        ├── AmplitudeDamping(0.002) @ q[1]
        ├── H @ q[4]
        └── CX @ q[1], q[2]
        <BLANKLINE>
        >>> c
        5-qubit circuit with 3 instructions:
        ├── H @ q[1]
        ├── H @ q[4]
        └── CX @ q[1], q[2]
        <BLANKLINE>
    """
    noisy = mc.Circuit()
    for inst in circuit:
        matched = False

        for rule in model.rules:
            if not rule.matches(inst):
                continue

            noise_inst = rule.apply_rule(inst)

            if rule.replaces():
                # noise replaces original instruction
                noisy.push(noise_inst)
            elif rule.before():
                # noise before op
                noisy.push(noise_inst)
                noisy.push(inst)
            else:
                # noise after op
                noisy.push(inst)
                noisy.push(noise_inst)

            matched = True
            break

        if not matched:
            # no rule matched → keep original instruction unchanged
            noisy.push(inst)

    return noisy


__all__ = [
    "AbstractNoiseRule",
    "ExactQubitReadoutNoise",
    "GlobalReadoutNoise",
    "SetQubitReadoutNoise",
    "NoiseModel",
    "apply_noise_model",
    "GateTypeNoise",
    "SetGateTypeQubitNoise",
    "ExactGateTypeQubitNoise",
    "IdleNoise",
    "GlobalReadoutNoise",
    "CustomNoiseRule",
    "ExactGateInstanceQubitNoise",
    "SetGateInstanceQubitNoise",
    "GateInstanceNoise",
]
