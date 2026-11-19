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
"""Shared circuit-rule abstractions used by noise and loss models."""


class AbstractCircuitRule:
    """Base class for the rules of a :class:`NoiseModel` or a :class:`LossModel`.

    A rule looks at one instruction at a time. :meth:`matches` says whether it
    applies, and :meth:`apply_rule` builds what it produces. The model keeps
    its rules sorted by :meth:`priority` and uses the first one that
    produces something for an instruction. :meth:`before` and
    :meth:`replaces` say where a noise rule's output goes relative to the
    matched instruction.

    Subclasses must implement :meth:`matches` and :meth:`apply_rule`. Two
    rules are equal when they have the same type and the same attributes.
    """

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__

    def __hash__(self):
        def make_hashable(value):
            if isinstance(value, list):
                return tuple(make_hashable(v) for v in value)
            if isinstance(value, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
            return value

        return hash(
            (
                type(self),
                tuple(sorted((k, make_hashable(v)) for k, v in self.__dict__.items())),
            )
        )

    def priority(self):
        """Return the priority of the rule. Lower values are tried first.

        The default is ``100``.
        """
        return 100

    def before(self):
        """Return ``True`` if the rule's output goes before the matched instruction.

        The default is ``False``: the output goes after it.
        """
        return False

    def replaces(self):
        """Return ``True`` if the rule's output replaces the matched instruction.

        The default is ``False``: the matched instruction is kept.
        """
        return False

    def matches(self, inst):
        """Return ``True`` if the rule applies to the instruction ``inst``.

        Raises:
            NotImplementedError: Unless a subclass implements it.
        """
        raise NotImplementedError

    def apply_rule(self, inst):
        """Return what the rule produces for ``inst``, or ``None`` if it does not apply.

        Raises:
            NotImplementedError: Unless a subclass implements it.
        """
        raise NotImplementedError


__all__ = ["AbstractCircuitRule"]
