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

import mimiqcircuits as mc
from mimiqcircuits.proto import circuitrules_pb2
from mimiqcircuits.proto.circuitproto import toproto_operation, fromproto_operation


class CircuitRuleProtoRegistry:
    def __init__(self):
        self.toproto_registry = {}
        self.fromproto_registry = {}

    def register_toproto(self, rule_type):
        def decorator(func):
            self.toproto_registry[rule_type] = func
            return func

        return decorator

    def register_fromproto(self, field_name):
        def decorator(func):
            self.fromproto_registry[field_name] = func
            return func

        return decorator

    def toproto(self, rule, declcache=None):
        for cls, func in self.toproto_registry.items():
            if isinstance(rule, cls):
                return func(rule, declcache)
        raise TypeError(f"No registered toproto converter for rule type: {type(rule)}")

    def fromproto(self, msg, declcache=None):
        field = msg.WhichOneof("kind")
        if field in self.fromproto_registry:
            return self.fromproto_registry[field](getattr(msg, field), declcache)
        raise ValueError(f"Unknown circuit rule field: {field}")


circuitrule_registry = CircuitRuleProtoRegistry()


@circuitrule_registry.register_toproto(mc.DropRule)
def toproto_drop_rule(rule, declcache=None):
    msg = circuitrules_pb2.CircuitRule()
    msg.drop_rule.SetInParent()
    if rule.operation is not None:
        msg.drop_rule.operation.CopyFrom(toproto_operation(rule.operation, declcache))
    return msg


@circuitrule_registry.register_fromproto("drop_rule")
def fromproto_drop_rule(msg, declcache=None):
    if msg.HasField("operation"):
        return mc.DropRule(fromproto_operation(msg.operation, declcache))
    return mc.DropRule()


@circuitrule_registry.register_toproto(mc.DecorateRule)
def toproto_decorate_rule(rule, declcache=None):
    if not isinstance(rule.decoration, mc.Operation):
        raise ValueError(
            "DecorateRule with instruction-list decoration is not serializable. "
            "The Julia schema only supports Operation decoration."
        )

    msg = circuitrules_pb2.CircuitRule()
    msg.decorate_rule.operation.CopyFrom(toproto_operation(rule.operation, declcache))
    msg.decorate_rule.decoration.CopyFrom(
        toproto_operation(rule.decoration, declcache)
    )
    msg.decorate_rule.before = rule.before()
    return msg


@circuitrule_registry.register_fromproto("decorate_rule")
def fromproto_decorate_rule(msg, declcache=None):
    return mc.DecorateRule(
        fromproto_operation(msg.operation, declcache),
        fromproto_operation(msg.decoration, declcache),
        before=msg.before,
    )


@circuitrule_registry.register_toproto(mc.ReplaceRule)
def toproto_replace_rule(rule, declcache=None):
    if not isinstance(rule.replacement, mc.Operation):
        raise ValueError(
            "ReplaceRule with instruction-list replacement is not serializable. "
            "The Julia schema only supports Operation replacement."
        )

    msg = circuitrules_pb2.CircuitRule()
    msg.replace_rule.operation.CopyFrom(toproto_operation(rule.operation, declcache))
    msg.replace_rule.replacement.CopyFrom(
        toproto_operation(rule.replacement, declcache)
    )
    return msg


@circuitrule_registry.register_fromproto("replace_rule")
def fromproto_replace_rule(msg, declcache=None):
    return mc.ReplaceRule(
        fromproto_operation(msg.operation, declcache),
        fromproto_operation(msg.replacement, declcache),
    )


def toproto_CircuitRule(rule, declcache=None):
    return circuitrule_registry.toproto(rule, declcache)


def fromproto_CircuitRule(msg, declcache=None):
    return circuitrule_registry.fromproto(msg, declcache)


def toproto_lossmodel(model, declcache=None):
    msg = circuitrules_pb2.LossModel()
    msg.name = model.name
    for rule in model.rules:
        msg.rules.append(toproto_CircuitRule(rule, declcache))
    return msg


def fromproto_lossmodel(msg, declcache=None):
    model = mc.LossModel(name=msg.name)
    for rule_msg in msg.rules:
        model.add_rule(fromproto_CircuitRule(rule_msg, declcache))
    return model
