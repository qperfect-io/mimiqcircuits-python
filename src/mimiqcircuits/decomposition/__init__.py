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
"""Quantum circuit decomposition framework.

This module provides a flexible framework for decomposing quantum circuits
into different target gate sets.

Core Concepts
-------------

**RewriteRule**: Single-step transformation rules that define *how* to
transform operations, but not *when* to stop.

**DecompositionBasis**: Recursive decomposition targets that define both
*what is terminal* (operations that shouldn't be decomposed) and *how to
decompose* non-terminal operations.

Main Functions
--------------

- :func:`decompose`: Recursively decompose a circuit to a target basis
- :func:`decompose_step`: Apply a single decomposition step

Available Bases
---------------

- :class:`CanonicalBasis`: Decomposes to GateU + GateCX (default)
- :class:`CliffordTBasis`: Decomposes to Clifford+T gate set
- :class:`FlattenedBasis`: Flattens GateCall/Block containers
- :class:`QASMBasis`: Decomposes to OpenQASM 2.0 gate set
- :class:`StimBasis`: Decomposes to Stim Clifford gate set
- :class:`RuleBasis`: Wraps any RewriteRule as a DecompositionBasis

Available Rules
---------------

- :class:`CanonicalRewrite`: Uses built-in gate decomposition methods
- :class:`FlattenContainers`: Flattens GateCall/Block containers
- :class:`ZYZRewrite`: ZYZ Euler angle decomposition
- :class:`SpecialAngleRewrite`: Rotations with special angles -> Clifford+T
- :class:`ToZRotationRewrite`: Convert RX/RY to RZ + Clifford
- :class:`ToffoliToCliffordTRewrite`: Toffoli -> Clifford+T decomposition

Examples
--------

Basic decomposition to canonical basis (GateU + GateCX):

    >>> from mimiqcircuits import Circuit, GateH, GateCCX
    >>> from mimiqcircuits.decomposition import decompose
    >>> c = Circuit()
    >>> c.push(GateH(), 0)
    1-qubit circuit with 1 instruction:
    └── H @ q[0]
    <BLANKLINE>
    >>> c.push(GateCCX(), 0, 1, 2)
    3-qubit circuit with 2 instructions:
    ├── H @ q[0]
    └── C₂X @ q[0:1], q[2]
    <BLANKLINE>
    >>> decomposed = decompose(c)

Decompose to Clifford+T:

    >>> from mimiqcircuits.decomposition import decompose, CliffordTBasis
    >>> decomposed = decompose(c, CliffordTBasis())

Use a specific rewrite rule:

    >>> from mimiqcircuits.decomposition import decompose_step, ZYZRewrite
    >>> stepped = decompose_step(c, ZYZRewrite())

Creating Custom Rules
---------------------

You can create custom rewrite rules by subclassing :class:`RewriteRule`:

    >>> from mimiqcircuits.decomposition import RewriteRule
    >>> class MyRewrite(RewriteRule):
    ...     def matches(self, op):
    ...         return isinstance(op, SomeGate)
    ...
    ...     def decompose_step(self, op, qubits, bits, zvars):
    ...         from mimiqcircuits import Circuit
    ...         circ = Circuit()
    ...         # ... decomposition logic ...
    ...         return circ

And custom decomposition bases by subclassing :class:`DecompositionBasis`:

    >>> from mimiqcircuits.decomposition import DecompositionBasis
    >>> class MyBasis(DecompositionBasis):
    ...     def isterminal(self, op):
    ...         return isinstance(op, (TargetGate1, TargetGate2))
    ...
    ...     def decompose(self, op, qubits, bits, zvars):
    ...         # Use a rewrite rule or custom logic
    ...         return SomeRewrite().decompose_step(op, qubits, bits, zvars)
"""

# Abstract base classes
from mimiqcircuits.decomposition.abstract import (
    DecompositionBasis,
    DecompositionError,
    RewriteRule,
)

# Decomposition bases
from mimiqcircuits.decomposition.basis import (
    CanonicalBasis,
    CliffordTBasis,
    FlattenedBasis,
    QASMBasis,
    RuleBasis,
    StimBasis,
)

# Rewrite rules
from mimiqcircuits.decomposition.rules import (
    CanonicalRewrite,
    FlattenContainers,
    SolovayKitaevRewrite,
    SpecialAngleRewrite,
    ToZRotationRewrite,
    ToffoliToCliffordTRewrite,
    ZYZRewrite,
)

# Main decomposition functions
from mimiqcircuits.decomposition.decompose import (
    DecomposeIterator,
    decompose,
    decompose_step,
    eachdecomposed,
)

__all__ = [
    # Abstract base classes
    "RewriteRule",
    "DecompositionBasis",
    "DecompositionError",
    # Decomposition bases
    "CanonicalBasis",
    "CliffordTBasis",
    "FlattenedBasis",
    "QASMBasis",
    "RuleBasis",
    "StimBasis",
    # Rewrite rules
    "CanonicalRewrite",
    "FlattenContainers",
    "ZYZRewrite",
    "SpecialAngleRewrite",
    "ToZRotationRewrite",
    "ToffoliToCliffordTRewrite",
    "SolovayKitaevRewrite",
    # Main functions
    "decompose",
    "decompose_step",
    "eachdecomposed",
    "DecomposeIterator",
]
