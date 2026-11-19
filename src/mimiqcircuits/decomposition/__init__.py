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
"""Rewrite circuits into a target gate set.

The framework has two building blocks. A :class:`RewriteRule` says *how* to
rewrite one operation into simpler ones, one step at a time, and nothing
about when to stop. A :class:`DecompositionBasis` says *what is terminal*,
the operations to keep, and how to rewrite everything else; :func:`decompose`
applies it recursively until only terminal operations are left.

.. rubric:: Functions

- :func:`decompose`: recursively decompose a circuit, instruction or
  operation to a basis, and return a new circuit.
- :func:`eachdecomposed`: the same, as a lazy iterator over the resulting
  instructions.
- :func:`decompose_step`: apply a rewrite rule once, without recursion.
- :class:`DecomposeIterator`: the iterator behind :func:`decompose` and
  :func:`eachdecomposed`.

A function that takes a basis also accepts a rewrite rule, wrapped in a
:class:`RuleBasis`.

.. rubric:: Bases

- :class:`CanonicalBasis`: ``GateU`` and ``GateCX``, plus the non-unitary
  primitives. The default.
- :class:`CliffordTBasis`: Clifford+T, approximating arbitrary rotations.
- :class:`StimBasis`: the Clifford gates, measurements and noise of Stim.
- :class:`QASMBasis`: the OpenQASM 2.0 gate library (``qelib1.inc``).
- :class:`FlattenedBasis`: expands ``GateCall``, ``Block`` and ``BlockCall``
  containers and keeps everything else.
- :class:`RuleBasis`: applies one rewrite rule until it no longer matches.

.. rubric:: Rules

- :class:`CanonicalRewrite`: the built-in decomposition of each operation.
- :class:`FlattenContainers`: expands one level of ``GateCall``, ``Block``
  and ``BlockCall``.
- :class:`ZYZRewrite`: ``GateU`` and ``GateRX`` to Z and Y rotations.
- :class:`ToZRotationRewrite`: ``GateRX`` and ``GateRY`` to ``GateRZ`` and
  Clifford gates.
- :class:`SpecialAngleRewrite`: rotations by multiples of ``pi/4`` to exact
  Clifford+T gates.
- :class:`ToffoliToCliffordTRewrite`: ``GateCCX`` to Clifford+T with 7 T
  gates.
- :class:`GridSynthRewrite`: ``GateRZ`` approximated with Clifford+T by the
  Ross-Selinger algorithm.
- :class:`SolovayKitaevRewrite`: single-qubit rotations approximated with
  Clifford+T by the Solovay-Kitaev algorithm.

An operation that a basis cannot rewrite raises :class:`DecompositionError`.

Examples:
    >>> from mimiqcircuits import *
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

    >>> # the default basis leaves only GateU and GateCX
    >>> sorted({type(inst.operation).__name__ for inst in decompose(c)})
    ['GateCX', 'GateU']

    >>> # the Clifford+T basis rewrites the Toffoli with H, T, T† and CX
    >>> sorted({type(inst.operation).__name__ for inst in decompose(c, CliffordTBasis())})
    ['GateCX', 'GateH', 'GateT', 'GateTDG']

    >>> # a rule applied once rewrites the Toffoli into 15 gates and keeps H
    >>> stepped = decompose_step(c, ToffoliToCliffordTRewrite())
    >>> len(stepped), stepped[0].operation
    (16, H)

    To add a gate set of your own, subclass :class:`RewriteRule` or
    :class:`DecompositionBasis`; their documentation shows how.
"""

# Abstract base classes
from mimiqcircuits.decomposition.abstract import (
    DecompositionBasis,
    DecompositionContext,
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
    GridSynthRewrite,
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
    "DecompositionContext",
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
    "GridSynthRewrite",
    # Main functions
    "decompose",
    "decompose_step",
    "eachdecomposed",
    "DecomposeIterator",
]
