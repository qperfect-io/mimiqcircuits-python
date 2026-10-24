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

from mimiqcircuits import Operation


class AbstractClassical(Operation):
    """AbstractClassical

    SuperType for classical operations
    """

    pass


def _format_logical(symbol, bits):
    """Format a commutative bitwise operation, ``c[t] = c[a] op c[b] ...``.

    When the target is also one of the inputs, the operation updates the bit in
    place and is written as ``c[t] op= ...`` with that input removed, which is
    valid because the operation is commutative. More than five inputs are
    written as ``c[t] = op @ c[a, b, ...]``.
    """
    target, inputs = bits[0], list(bits[1:])
    if len(inputs) > 5:
        return f"c[{target}] = {symbol} @ c[{', '.join(map(str, inputs))}]"
    if target in inputs:
        inputs.remove(target)
        return f"c[{target}] {symbol}= " + f" {symbol} ".join(f"c[{b}]" for b in inputs)
    return f"c[{target}] = " + f" {symbol} ".join(f"c[{b}]" for b in inputs)
