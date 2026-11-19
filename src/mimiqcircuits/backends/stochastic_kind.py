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
from mimiqcircuits.operations.losschannel import Loss
from mimiqcircuits.operations.measure import AbstractMeasurement

if TYPE_CHECKING:  # pragma: no cover
    from mimiqcircuits.backends.backend import Backend
    from mimiqcircuits.circuit import Circuit


class StochasticKind(enum.IntEnum):
    """When a backend resolves the randomness of an operation.

    The members are ordered, ``Deterministic < TrajectorySampleable <
    RuntimeOnly``, so ``max`` gives the kind of a group of operations.

    - ``Deterministic``: no randomness; :meth:`LocalBackend.compile` can
      lower the operation from the circuit alone.
    - ``TrajectorySampleable``: random, but resolvable in
      :meth:`LocalBackend.prepare_trajectory` from random draws alone,
      because the outcomes and their probabilities do not depend on the
      state, for this backend.
    - ``RuntimeOnly``: must be resolved during :meth:`LocalBackend.evolve`,
      because the probabilities or a control-flow condition depend on the
      live quantum or classical state.

    The kind of an operation depends on the backend; see
    :meth:`Backend.stochastic_kind`.
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
    """Classify an operation without reference to a backend.

    This is the default of :meth:`Backend.stochastic_kind`. Backends that
    classify an operation differently override that method, not this
    function.

    - ``IfStatement``: the kind of the operation it wraps.
    - Kraus channels: ``TrajectorySampleable`` if mixed-unitary, otherwise
      ``RuntimeOnly``.
    - ``Loss`` and every measurement: ``RuntimeOnly``.
    - Everything else: ``Deterministic``.

    Args:
        op: An operation, or an instruction whose operation is classified.

    Returns:
        A :class:`StochasticKind`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import default_stochastic_kind
        >>> default_stochastic_kind(GateH()).name
        'Deterministic'

        A depolarizing channel applies a random Pauli with fixed
        probabilities, so its branch can be drawn ahead of time.

        >>> default_stochastic_kind(Depolarizing(1, 0.1)).name
        'TrajectorySampleable'

        The branch probabilities of amplitude damping depend on the state.

        >>> default_stochastic_kind(AmplitudeDamping(0.1)).name
        'RuntimeOnly'
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
    if isinstance(op, Loss):
        return StochasticKind.RuntimeOnly
    if isinstance(op, AbstractMeasurement):
        return StochasticKind.RuntimeOnly
    return StochasticKind.Deterministic


# ── Free-function helpers ──────────────────────


def stochastic_kind(backend: "Backend", x) -> StochasticKind:
    """Return ``backend.stochastic_kind(x)``.

    Args:
        backend: The backend that classifies ``x``.
        x: An operation or an instruction.

    Returns:
        A :class:`StochasticKind`.
    """
    return backend.stochastic_kind(x)


def is_deterministic(backend: "Backend", x) -> bool:
    """Return whether ``backend`` classifies ``x`` as ``Deterministic``."""
    return backend.stochastic_kind(x) == StochasticKind.Deterministic


def is_trajectory_sampleable(backend: "Backend", x) -> bool:
    """Return whether ``backend`` classifies ``x`` as
    ``TrajectorySampleable``."""
    return backend.stochastic_kind(x) == StochasticKind.TrajectorySampleable


def is_runtime_only(backend: "Backend", x) -> bool:
    """Return whether ``backend`` classifies ``x`` as ``RuntimeOnly``."""
    return backend.stochastic_kind(x) == StochasticKind.RuntimeOnly


def is_stochastic(backend: "Backend", x) -> bool:
    """Return whether ``backend`` classifies ``x`` as anything but
    ``Deterministic``."""
    return backend.stochastic_kind(x) != StochasticKind.Deterministic


def first_stochastic(backend: "Backend", circuit: "Circuit"):
    """Return the index of the first instruction of ``circuit`` that
    ``backend`` does not classify as ``Deterministic``, or ``None``."""
    for i, inst in enumerate(circuit):
        if is_stochastic(backend, inst):
            return i
    return None


def last_stochastic(backend: "Backend", circuit: "Circuit"):
    """Return the index of the last instruction of ``circuit`` that
    ``backend`` does not classify as ``Deterministic``, or ``None``."""
    last = None
    for i, inst in enumerate(circuit):
        if is_stochastic(backend, inst):
            last = i
    return last


def first_runtime_only(backend: "Backend", circuit: "Circuit"):
    """Return the index of the first instruction of ``circuit`` that
    ``backend`` classifies as ``RuntimeOnly``, or ``None``.

    Unlike :func:`first_stochastic`, it skips ``TrajectorySampleable``
    instructions, which a backend can resolve before evolving.
    """
    for i, inst in enumerate(circuit):
        if is_runtime_only(backend, inst):
            return i
    return None


def last_runtime_only(backend: "Backend", circuit: "Circuit"):
    """Return the index of the last instruction of ``circuit`` that
    ``backend`` classifies as ``RuntimeOnly``, or ``None``."""
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
