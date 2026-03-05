#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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
"""Decomposition bases for quantum circuits.

Available bases:
    - :class:`CanonicalBasis`: Decomposes to GateU + GateCX (default)
    - :class:`CliffordTBasis`: Decomposes to Clifford+T gate set
    - :class:`FlattenedBasis`: Flattens container operations
    - :class:`QASMBasis`: Decomposes to OpenQASM 2.0 gate set
    - :class:`StimBasis`: Decomposes to Stim Clifford gate set
    - :class:`RuleBasis`: Wraps a RewriteRule as a DecompositionBasis
"""

from mimiqcircuits.decomposition.basis.canonical import CanonicalBasis
from mimiqcircuits.decomposition.basis.clifford_t import CliffordTBasis
from mimiqcircuits.decomposition.basis.flattened import FlattenedBasis
from mimiqcircuits.decomposition.basis.qasm import QASMBasis
from mimiqcircuits.decomposition.basis.rule_basis import RuleBasis
from mimiqcircuits.decomposition.basis.stim import StimBasis

__all__ = [
    "CanonicalBasis",
    "CliffordTBasis",
    "FlattenedBasis",
    "QASMBasis",
    "RuleBasis",
    "StimBasis",
]
