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

from mimiqcircuits.proto import noisemodel_pb2
import mimiqcircuits as mc

from mimiqcircuits.proto.circuitproto import (
    toproto_operation,
    fromproto_operation,
    fromproto_arg,
    toproto_arg,
)

# ==========================================================
#   Registry Infrastructure (same pattern as circuitproto)
# ==========================================================


class NoiseProtoRegistry:
    def __init__(self):
        self.toproto_registry = {}
        self.fromproto_registry = {}

    def register_toproto(self, rule_type):
        """Decorator to register a toproto converter for a rule type."""

        def decorator(func):
            self.toproto_registry[rule_type] = func
            return func

        return decorator

    def register_fromproto(self, field_name):
        """Decorator to register a fromproto converter for a message field."""

        def decorator(func):
            self.fromproto_registry[field_name] = func
            return func

        return decorator

    # added declcache in case
    def toproto(self, rule, declcache=None):
        """Convert any registered rule to its proto."""
        for cls, func in self.toproto_registry.items():
            if isinstance(rule, cls):
                return func(rule, declcache)
        raise TypeError(f"No registered toproto converter for rule type: {type(rule)}")

    def fromproto(self, msg, declcache=None):
        """Convert proto message to corresponding rule object."""
        field = msg.WhichOneof("kind")
        if field in self.fromproto_registry:
            return self.fromproto_registry[field](getattr(msg, field), declcache)
        raise ValueError(f"Unknown noise rule field: {field}")


# Global instance
noise_registry = NoiseProtoRegistry()

# ======================
#   Concrete Converters
# =====================

######## Readout Noise #############


@noise_registry.register_toproto(mc.GlobalReadoutNoise)
def toproto_global_readout(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    msg.global_readout.noise.CopyFrom(toproto_operation(rule.noise))
    return msg


@noise_registry.register_fromproto("global_readout")
def fromproto_global_readout(msg, declcache=None):
    return mc.GlobalReadoutNoise(fromproto_operation(msg.noise))


@noise_registry.register_toproto(mc.ExactQubitReadoutNoise)
def toproto_exact_qubit_readout(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    msg.exact_qubit_readout.noise.CopyFrom(toproto_operation(rule.noise))
    msg.exact_qubit_readout.qubits.extend(q + 1 for q in rule.qubits)
    return msg


@noise_registry.register_fromproto("exact_qubit_readout")
def fromproto_exact_qubit_readout(msg, declcache=None):
    return mc.ExactQubitReadoutNoise(
        tuple(q - 1 for q in msg.qubits),
        fromproto_operation(msg.noise),
    )


@noise_registry.register_toproto(mc.SetQubitReadoutNoise)
def toproto_set_qubit_readout(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    msg.set_qubit_readout.noise.CopyFrom(toproto_operation(rule.noise))
    msg.set_qubit_readout.qubits.extend(q + 1 for q in rule._qubit_set)
    return msg


@noise_registry.register_fromproto("set_qubit_readout")
def fromproto_set_qubit_readout(msg, declcache=None):
    return mc.SetQubitReadoutNoise(
        tuple(q - 1 for q in msg.qubits),
        fromproto_operation(msg.noise),
    )


######### Operation Instance Noise ############


@noise_registry.register_toproto(mc.OperationInstanceNoise)
def toproto_operation_instance_noise(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    msg.operation_instance_noise.operation.CopyFrom(toproto_operation(rule.operation))
    msg.operation_instance_noise.noise.CopyFrom(toproto_operation(rule.noise))
    msg.operation_instance_noise.before = rule.before()
    msg.operation_instance_noise.replace = rule.replaces()
    return msg


@noise_registry.register_fromproto("operation_instance_noise")
def fromproto_operation_instance_noise(msg, declcache=None):
    return mc.OperationInstanceNoise(
        fromproto_operation(msg.operation),
        fromproto_operation(msg.noise),
        before=msg.before,
        replace=msg.replace,
    )


@noise_registry.register_toproto(mc.ExactOperationInstanceQubitNoise)
def toproto_exact_operation_instance_noise(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    msg.exact_operation_instance_noise.operation.CopyFrom(
        toproto_operation(rule.operation)
    )
    msg.exact_operation_instance_noise.noise.CopyFrom(toproto_operation(rule.noise))
    msg.exact_operation_instance_noise.before = rule.before()
    msg.exact_operation_instance_noise.replace = rule.replaces()
    msg.exact_operation_instance_noise.qubits.extend(q + 1 for q in rule.qubits)
    return msg


@noise_registry.register_fromproto("exact_operation_instance_noise")
def fromproto_exact_operation_instance_noise(msg, declcache=None):
    return mc.ExactOperationInstanceQubitNoise(
        fromproto_operation(msg.operation),
        tuple(q - 1 for q in msg.qubits),
        fromproto_operation(msg.noise),
        before=msg.before,
        replace=msg.replace,
    )


@noise_registry.register_toproto(mc.SetOperationInstanceQubitNoise)
def toproto_set_operation_instance_noise(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    msg.set_operation_instance_noise.operation.CopyFrom(
        toproto_operation(rule.operation)
    )
    msg.set_operation_instance_noise.noise.CopyFrom(toproto_operation(rule.noise))
    msg.set_operation_instance_noise.before = rule.before()
    msg.set_operation_instance_noise.replace = rule.replaces()
    msg.set_operation_instance_noise.qubits.extend(q + 1 for q in rule._qubit_set)
    return msg


@noise_registry.register_fromproto("set_operation_instance_noise")
def fromproto_set_operation_instance_noise(msg, declcache=None):
    return mc.SetOperationInstanceQubitNoise(
        fromproto_operation(msg.operation),
        tuple(q - 1 for q in msg.qubits),
        fromproto_operation(msg.noise),
        before=msg.before,
        replace=msg.replace,
    )


######### Idle Noise ############


@noise_registry.register_toproto(mc.IdleNoise)
def toproto_idle_noise(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    rel = msg.idle_noise.relation

    # constant noise: IdleNoise(op)
    if isinstance(rule.relation, mc.Operation):
        rel.operation.CopyFrom(toproto_operation(rule.relation))
        return msg

    # symbolic IdleNoise(var => op)
    var, op = rule.relation
    rel.variables.append(toproto_arg(var))

    # Operation is still an operation
    rel.operation.CopyFrom(toproto_operation(op))
    return msg


@noise_registry.register_fromproto("idle_noise")
def fromproto_idle_noise(msg, declcache=None):
    vars_ = [fromproto_arg(v) for v in msg.relation.variables]
    op = fromproto_operation(msg.relation.operation)

    if len(vars_) == 0:
        # IdleNoise(op)
        relation = op
    else:
        # IdleNoise(var => op)
        relation = (vars_[0], op)

    return mc.IdleNoise(relation)


@noise_registry.register_toproto(mc.SetIdleQubitNoise)
def toproto_set_idle_noise(rule, declcache=None):
    msg = noisemodel_pb2.NoiseRule()
    rel = msg.set_idle_noise.relation

    # constant noise SetIdleQubitNoise(op, qubits)
    if isinstance(rule.relation, mc.Operation):
        rel.operation.CopyFrom(toproto_operation(rule.relation))
        msg.set_idle_noise.qubits.extend(q + 1 for q in rule.qubits)
        return msg

    # symbolic SetIdleQubitNoise(var => op, qubits)
    var, op = rule.relation

    # encode variable as Arg, NOT Operation
    rel.variables.append(toproto_arg(var))

    rel.operation.CopyFrom(toproto_operation(op))
    msg.set_idle_noise.qubits.extend(q + 1 for q in rule.qubits)
    return msg


@noise_registry.register_fromproto("set_idle_noise")
def fromproto_set_idle_noise(msg, declcache=None):
    vars_ = [fromproto_arg(v) for v in msg.relation.variables]

    # operation is an Operation
    op = fromproto_operation(msg.relation.operation)

    # Qubits are 1-based in proto, convert to 0-based
    qubits = [q - 1 for q in msg.qubits]

    if len(vars_) == 0:
        relation = op
    else:
        relation = (vars_[0], op)

    return mc.SetIdleQubitNoise(relation, qubits)


######## Custom Noise ########


# The matcher and generator are arbitrary Python callables, which have no
# portable representation. Storing their names would decode into a different
# rule, so both directions refuse instead.
@noise_registry.register_toproto(mc.CustomNoiseRule)
def toproto_custom_noise(rule, declcache=None):
    raise ValueError(
        "CustomNoiseRule holds Python callables and cannot be serialized. "
        "Remove custom rules from the noise model before saving it."
    )


@noise_registry.register_fromproto("custom_noise")
def fromproto_custom_noise(msg, declcache=None):
    raise ValueError(
        "CustomNoiseRule cannot be deserialized: its matcher and generator "
        "callables are not stored in the file."
    )


# ==========================================================
#   Top-Level Conversion Functions
# ==========================================================


def toproto_NoiseRule(rule, declcache=None):
    """Convert an AbstractNoiseRule to NoiseRule protobuf."""
    return noise_registry.toproto(rule, declcache)


def fromproto_NoiseRule(msg, declcache=None):
    """Convert a NoiseRule protobuf to an AbstractNoiseRule."""
    return noise_registry.fromproto(msg, declcache)


def toproto_noisemodel(model):
    """Convert a full NoiseModel into protobuf format."""
    msg = noisemodel_pb2.NoiseModel()
    msg.name = model.name
    for rule in model.rules:
        msg.rules.append(toproto_NoiseRule(rule))
    return msg


def fromproto_noisemodel(msg):
    """Convert a NoiseModel protobuf message back into a NoiseModel."""
    nm = mc.NoiseModel(name=msg.name)
    for rmsg in msg.rules:
        nm.add_rule(fromproto_NoiseRule(rmsg))
    return nm
