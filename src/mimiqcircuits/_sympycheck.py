#
# Copyright © 2026 QPerfect. All Rights Reserved.
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

"""Type checks for sympy objects that do not import sympy."""

import sys

import symengine as se


# sympy is slow to import and only a few operations need it, so these checks
# look it up in ``sys.modules`` instead of importing it: if the caller never
# imported sympy, no value it passes can be a sympy object.
def _is_basic(x):
    """Return True if ``x`` is a symengine or sympy expression."""
    if isinstance(x, se.Basic):
        return True
    sp = sys.modules.get("sympy")
    return sp is not None and isinstance(x, sp.Basic)


def _is_sympy_matrix(x):
    """Return True if ``x`` is a sympy matrix."""
    sp = sys.modules.get("sympy")
    return sp is not None and isinstance(x, sp.MatrixBase)
