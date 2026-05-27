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

"""Optimization-pass framework.

A pass transforms a :class:`Circuit` before evolution; the
:class:`PassPipeline` runs them in order and tracks qubit
permutations so amplitude and expectation queries can be
unscrambled back to the user's original qubit space at the end.

Parameters travel over the wire (to remote backends) through a
JSON-safe ADT, :class:`PassParam`, instead of a loose ``dict``: the
tagged variants preserve type information that a plain dict would
silently lose.

Custom passes subclass :class:`AbstractPass` and implement
:meth:`~AbstractPass.spec` (a declarative :class:`PassSpec`) and
:meth:`~AbstractPass.apply` (the actual rewrite). See
:doc:`/manual/implementing_backends` for a worked example.
"""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# ──────────────────────────────────────────────────────────────────────────
# PassParam ADT
# ──────────────────────────────────────────────────────────────────────────


class PassParam:
    """Base class for the JSON-safe sum used to serialise pass parameters.

    The tagged variants (:class:`PStr`, :class:`PInt`, :class:`PFloat`,
    :class:`PBool`, :class:`PSym`, :class:`PList`, :class:`PDict`) keep
    type information a plain dict would lose on the wire — e.g.
    ``"greedy"`` (a :class:`PStr`) must not silently degrade and come
    back as a :class:`PSym`. Equality is tag-sensitive:
    ``PSym("x") != PStr("x")``.
    """


@dataclass(frozen=True)
class PStr(PassParam):
    value: str


@dataclass(frozen=True)
class PInt(PassParam):
    value: int


@dataclass(frozen=True)
class PFloat(PassParam):
    value: float


@dataclass(frozen=True)
class PBool(PassParam):
    value: bool


@dataclass(frozen=True)
class PSym(PassParam):
    """Pass parameter that is semantically a :class:`Symbol` (Julia) or
    :class:`enum.Enum`-like in Python. Distinct from :class:`PStr` at the
    tag level.
    """

    value: str


@dataclass(frozen=True)
class PList(PassParam):
    items: tuple[PassParam, ...]


@dataclass(frozen=True)
class PDict(PassParam):
    # Tuple of (key, value) pairs rather than a dict so the wrapper
    # itself remains hashable.
    items: tuple[tuple[str, PassParam], ...]


def to_pass_param(x) -> PassParam:
    """Coerce a common Python value into a :class:`PassParam`.

    The bool-before-int check is load-bearing: ``bool`` is a subclass
    of ``int`` in Python, so ``isinstance(True, int)`` is ``True`` and
    reversing the order would tag every boolean as :class:`PInt`.
    """
    if isinstance(x, PassParam):
        return x
    if isinstance(x, bool):
        return PBool(x)
    if isinstance(x, int):
        return PInt(int(x))
    if isinstance(x, float):
        return PFloat(float(x))
    if isinstance(x, str):
        return PStr(str(x))
    if isinstance(x, (list, tuple)):
        return PList(tuple(to_pass_param(v) for v in x))
    if isinstance(x, dict):
        return PDict(tuple((str(k), to_pass_param(v)) for k, v in x.items()))
    raise TypeError(f"cannot coerce {x!r} to PassParam")


# ──────────────────────────────────────────────────────────────────────────
# PassSpec / PassResult / PassContext
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PassSpec:
    """Declarative summary of a pass.

    Used for equality / memoisation (two passes with the same spec
    have the same hash and compare equal), for remote dispatch (the
    spec is the only thing shipped over the wire), and for
    observability. Equality is structural.

    ``requires`` names other passes that must run *before* this one;
    ``conflicts`` names passes that cannot coexist with this one in
    the same pipeline. Both are advisory.
    """

    name: str
    parameters: tuple[tuple[str, PassParam], ...] = ()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @staticmethod
    def from_dict(name: str, parameters: Optional[dict] = None,
                  requires: Iterable[str] = (), conflicts: Iterable[str] = ()
                  ) -> "PassSpec":
        params = tuple((str(k), to_pass_param(v))
                       for k, v in (parameters or {}).items())
        return PassSpec(name, params, tuple(requires), tuple(conflicts))


@dataclass
class PassResult:
    """Side effects returned from :meth:`AbstractPass.apply`.

    ``qubit_permutation`` is ``None`` when the pass leaves qubit
    indices unchanged. Any pass that *does* relabel qubits must
    return the relabel here so the pipeline can compose permutations
    and un-shuffle downstream outputs.

    ``metadata`` is free-form pass-specific information (timings,
    gate counts, …) surfaced for observability.
    """

    qubit_permutation: Optional[list[int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassContext:
    """Inputs a pass needs but should not fetch from global state.

    ``backend`` is the Backend the pipeline is compiling for, or
    ``None`` for backend-agnostic runs. ``rng`` is the dedicated
    pass-internal RNG. ``bitstrings`` are user-supplied amplitude
    targets in the original qubit space; the pipeline rewrites them
    into post-pass space as permutations compose. ``features`` are
    detected on the input circuit and filter passes via
    :py:meth:`AbstractPass.preserves`.
    """

    backend: Optional[Any] = None
    rng: random.Random = field(default_factory=random.Random)
    bitstrings: list = field(default_factory=list)
    features: set[str] = field(default_factory=set)


# ──────────────────────────────────────────────────────────────────────────
# AbstractPass
# ──────────────────────────────────────────────────────────────────────────


class AbstractPass(abc.ABC):
    """Base class for circuit-transformation passes.

    Subclasses implement:

    - :meth:`spec` — return a :class:`PassSpec` describing the pass
      (used for equality, serialisation, and remote dispatch).
    - :meth:`apply` — return ``(new_circuit, PassResult)``.
    - :meth:`preserves` — override when the pass *breaks* a circuit
      feature; the default returns ``True`` for every feature. The
      pipeline filters passes by feature when the input circuit has
      a feature that the pass must preserve.

    Recognised feature tokens: ``"feed_forward"``,
    ``"midcircuit_measure"``, ``"parametric"``, ``"loss"``,
    ``"exact_equivalence"``, ``"strict_qubit_count"``.
    """

    @abc.abstractmethod
    def spec(self) -> PassSpec: ...

    @abc.abstractmethod
    def apply(self, ctx: PassContext, circuit) -> tuple[Any, PassResult]: ...

    def preserves(self, feature: str) -> bool:
        return True


# ──────────────────────────────────────────────────────────────────────────
# PassPipeline
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PassPipeline:
    """An ordered, iterable sequence of passes.

    Run via :func:`apply_passes`, which composes each pass's
    ``qubit_permutation`` into a single permutation. Callers use the
    inverse of that permutation to map samples and
    ``Amplitude`` / ``ExpectationValue`` results back to the user's
    original qubit space.
    """

    passes: list[AbstractPass] = field(default_factory=list)

    def __iter__(self):
        return iter(self.passes)

    def __len__(self):
        return len(self.passes)

    def __getitem__(self, i):
        return self.passes[i]


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class UnacceptedPassError(Exception):
    """Raised when a pipeline contains a pass the backend rejects.

    The backend's :meth:`Backend.accepts_pass` is the gate; the
    pipeline fails loudly rather than silently dropping a pass the
    user explicitly requested.
    """

    def __init__(self, backend_name: str, pass_name: str):
        self.backend_name = backend_name
        self.pass_name = pass_name
        super().__init__(
            f"UnacceptedPassError: backend {backend_name} does not accept "
            f"pass :{pass_name}"
        )


class RemotePassOrderError(Exception):
    """Raised when an ordered pipeline is submitted to a remote
    backend that does not advertise the ``"pass_order_honored"``
    capability while ``strict_pass_order=True``.

    Pass ``strict_pass_order=False`` to opt into an unordered
    submission (recognised passes are translated into the server's
    flat option set; unrecognised passes are dropped with a warning).
    """

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        super().__init__(
            f"RemotePassOrderError: backend {backend_name} does not declare "
            "'pass_order_honored' — set strict_pass_order=False to submit "
            "as an unordered pass set instead"
        )


# ──────────────────────────────────────────────────────────────────────────
# apply_passes
# ──────────────────────────────────────────────────────────────────────────


def _compose_perm(prev: list[int], new: list[int]) -> list[int]:
    """Compose two 1-based permutations left-to-right.

    ``(new ∘ prev)[i] = new[prev[i]]``: each new permutation acts on
    qubits already shuffled by every preceding pass, so applying
    inverse-composed at the end unscrambles back to user space.
    """
    if len(prev) != len(new):
        raise ValueError(
            f"permutation length mismatch: {len(prev)} vs {len(new)}"
        )
    return [new[p - 1] for p in prev]


def invert_perm(perm: list[int]) -> list[int]:
    """Inverse of a 1-based permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm, start=1):
        inv[p - 1] = i
    return inv


def apply_passes(pipeline: PassPipeline, ctx: PassContext, circuit
                 ) -> tuple[Any, Optional[list[int]], list[PassResult]]:
    """Run ``pipeline`` against ``circuit``.

    Returns ``(transformed_circuit, composed_permutation,
    per_pass_results)``. ``composed_permutation`` is ``None`` when
    every pass left qubit indices unchanged.

    Raises :class:`UnacceptedPassError` if the backend rejects any
    pass. When the backend reports ``delegates_pass(p) is True`` the
    pass is *not* run here: the pipeline records a marker result and
    the backend handles the pass natively inside its ``compile`` or
    ``evolve``.
    """
    backend = ctx.backend
    results: list[PassResult] = []
    composed_perm: Optional[list[int]] = None
    current = circuit

    for p in pipeline:
        if backend is not None:
            if not backend.accepts_pass(p):
                raise UnacceptedPassError(backend.name, p.spec().name)
            if backend.delegates_pass(p):
                results.append(PassResult(metadata={"delegated": True}))
                continue
        current, r = p.apply(ctx, current)
        results.append(r)
        if r.qubit_permutation is not None:
            composed_perm = (
                list(r.qubit_permutation)
                if composed_perm is None
                else _compose_perm(composed_perm, r.qubit_permutation)
            )

    return current, composed_perm, results
