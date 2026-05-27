#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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
"""Resolution-time taxonomy for operations as classified by a backend.

Mirrors the Julia surface in `AbstractQCSs.jl/src/stochastic_kind.jl`.
The classification is **backend-dependent**: a ``MixedUnitary`` is
``TrajectorySampleable`` for an MPS-style simulator that resolves it
offline but could be ``RuntimeOnly`` for a simulator that resolves it
state-dependently. The dispatch hook is therefore a method on
:class:`Backend` (``backend.stochastic_kind(op)``), not a property of
the op alone. Free-function helpers
(``is_deterministic(backend, x)``, ``first_stochastic(backend, c)``,
…) wrap the method and accept either an :class:`Operation` or an
:class:`Instruction`.

This file deliberately avoids importing from
:mod:`mimiqcircuits.backends.backend` to keep the dependency cycle one
way: ``backend.py`` imports from this module, not the other way
around.
"""
from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.losschannel import LossErr
from mimiqcircuits.operations.measure import AbstractMeasurement

if TYPE_CHECKING:  # pragma: no cover
    from mimiqcircuits.backends.backend import Backend
    from mimiqcircuits.circuit import Circuit


class StochasticKind(enum.IntEnum):
    """Resolution-time taxonomy for operations.

    Ordered so worst-case combines (``max``) compose across composite
    ops: ``Deterministic < TrajectorySampleable < RuntimeOnly``.

    - ``Deterministic`` — no randomness; ``compile()`` can lower the
      op into the backend's compiled form using only the
      source-circuit information.
    - ``TrajectorySampleable`` — non-deterministic but resolvable in
      ``prepare_trajectory()`` using RNG draws alone. Classical sample
      space and probabilities are state-independent *as far as this
      backend is concerned*.
    - ``RuntimeOnly`` — must be resolved during ``evolve()`` because
      branch probabilities or control-flow truth values depend on the
      live quantum or classical state.

    **Backend-dependent.** See :meth:`Backend.stochastic_kind`.
    """

    Deterministic = 0
    TrajectorySampleable = 1
    RuntimeOnly = 2


def _resolve_op(x):
    """Accept either an :class:`Operation` or an :class:`Instruction`."""
    if isinstance(x, Instruction):
        return x.operation
    return x


def default_stochastic_kind(op) -> StochasticKind:
    """Backend-agnostic default classification used by
    :meth:`Backend.stochastic_kind`. Concrete backends override the
    method on :class:`Backend` (not this function) when they handle a
    specific op type differently.

    Defaults:

    - ``IfStatement`` / similar wrappers: recurse on the inner op.
    - ``krauschannel``: ``TrajectorySampleable`` if ``ismixedunitary``,
      else ``RuntimeOnly``.
    - ``LossErr``: ``RuntimeOnly`` (until ``sample_losses`` moves into
      ``prepare_trajectory()``; F-S2 follow-up).
    - ``AbstractMeasurement``: ``RuntimeOnly``.
    - everything else: ``Deterministic``.
    """
    op = _resolve_op(op)
    if isinstance(op, IfStatement):
        return default_stochastic_kind(op.get_operation())
    if isinstance(op, krauschannel):
        return (
            StochasticKind.TrajectorySampleable
            if op.ismixedunitary()
            else StochasticKind.RuntimeOnly
        )
    if isinstance(op, LossErr):
        return StochasticKind.RuntimeOnly
    if isinstance(op, AbstractMeasurement):
        return StochasticKind.RuntimeOnly
    return StochasticKind.Deterministic


# ── Free-function helpers (Julia-parity call syntax) ──────────────────────


def stochastic_kind(backend: "Backend", x) -> StochasticKind:
    """Free-function form: ``stochastic_kind(backend, x)``. Delegates
    to ``backend.stochastic_kind(x)`` so subclasses' overrides bite."""
    return backend.stochastic_kind(x)


def is_deterministic(backend: "Backend", x) -> bool:
    return backend.stochastic_kind(x) == StochasticKind.Deterministic


def is_trajectory_sampleable(backend: "Backend", x) -> bool:
    return backend.stochastic_kind(x) == StochasticKind.TrajectorySampleable


def is_runtime_only(backend: "Backend", x) -> bool:
    return backend.stochastic_kind(x) == StochasticKind.RuntimeOnly


def is_stochastic(backend: "Backend", x) -> bool:
    return backend.stochastic_kind(x) != StochasticKind.Deterministic


def first_stochastic(backend: "Backend", circuit: "Circuit"):
    """Index of the first non-deterministic instruction in ``circuit``
    as classified by ``backend``, or ``None`` if purely deterministic."""
    for i, inst in enumerate(circuit):
        if is_stochastic(backend, inst):
            return i
    return None


def last_stochastic(backend: "Backend", circuit: "Circuit"):
    """Index of the last non-deterministic instruction in ``circuit``,
    or ``None`` if purely deterministic."""
    last = None
    for i, inst in enumerate(circuit):
        if is_stochastic(backend, inst):
            last = i
    return last


def first_runtime_only(backend: "Backend", circuit: "Circuit"):
    """Index of the first ``RuntimeOnly`` instruction in ``circuit``,
    or ``None`` if none exists. Distinct from
    :func:`first_stochastic`: ``TrajectorySampleable`` ops do not
    require barrier protection in MPS-style backends."""
    for i, inst in enumerate(circuit):
        if is_runtime_only(backend, inst):
            return i
    return None


def last_runtime_only(backend: "Backend", circuit: "Circuit"):
    last = None
    for i, inst in enumerate(circuit):
        if is_runtime_only(backend, inst):
            last = i
    return last


__all__ = [
    "StochasticKind",
    "default_stochastic_kind",
    "stochastic_kind",
    "is_deterministic",
    "is_trajectory_sampleable",
    "is_runtime_only",
    "is_stochastic",
    "first_stochastic",
    "last_stochastic",
    "first_runtime_only",
    "last_runtime_only",
]
