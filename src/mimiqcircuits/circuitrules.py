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
    """Shared base class for circuit transformation rules."""

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
        return 100

    def before(self):
        return False

    def replaces(self):
        return False

    def matches(self, inst):
        raise NotImplementedError

    def apply_rule(self, inst):
        raise NotImplementedError


__all__ = ["AbstractCircuitRule"]
