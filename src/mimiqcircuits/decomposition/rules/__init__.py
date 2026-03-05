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
"""Rewrite rules for quantum circuit decomposition.

Available rules:
    - :class:`CanonicalRewrite`: Uses built-in gate decomposition methods
    - :class:`FlattenContainers`: Flattens GateCall/Block containers
    - :class:`ZYZRewrite`: ZYZ Euler angle decomposition
    - :class:`SpecialAngleRewrite`: Decompose rotations with special angles to Clifford+T
    - :class:`ToZRotationRewrite`: Convert RX/RY to RZ + Clifford gates
    - :class:`ToffoliToCliffordTRewrite`: Optimal Toffoli -> Clifford+T rewrite
"""

from mimiqcircuits.decomposition.rules.canonical import CanonicalRewrite
from mimiqcircuits.decomposition.rules.flatten_containers import FlattenContainers
from mimiqcircuits.decomposition.rules.solovay_kitaev import SolovayKitaevRewrite
from mimiqcircuits.decomposition.rules.special_angle import SpecialAngleRewrite
from mimiqcircuits.decomposition.rules.to_z_rotation import ToZRotationRewrite
from mimiqcircuits.decomposition.rules.toffoli_to_clifford_t import (
    ToffoliToCliffordTRewrite,
)
from mimiqcircuits.decomposition.rules.zyz import ZYZRewrite

__all__ = [
    "CanonicalRewrite",
    "FlattenContainers",
    "ZYZRewrite",
    "SpecialAngleRewrite",
    "ToZRotationRewrite",
    "ToffoliToCliffordTRewrite",
    "SolovayKitaevRewrite",
]
