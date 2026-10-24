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
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np


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
    """Pass parameter that names one of a fixed set of choices, like an
    :class:`enum.Enum` member. Distinct from :class:`PStr` at the
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

    def __add__(self, other):
        if isinstance(other, PassPipeline):
            return PassPipeline(self.passes + other.passes)
        return PassPipeline(self.passes + list(other))


# ──────────────────────────────────────────────────────────────────────────
# Preparation flags
# ──────────────────────────────────────────────────────────────────────────


#: Which pass each preparation flag speaks about, in the order those passes
#: run. A backend's own pass of a kind must report the same ``spec`` name
#: (Exaqt's kernel-cost reorder reports ``reorder_qubits``), so a flag finds
#: it in ``default_passes`` whichever backend declared it.
_FLAG_PASS_NAMES = {
    "remove_swaps": "remove_swaps",
    "simplify": "simplify",
    "canonicaldecompose": "canonical_decompose",
    "fuse": "fuse_gates",
    "reorderqubits": "reorder_qubits",
}


def check_prep_flags(
    *,
    fuse=None,
    fuse_threshold=None,
    canonicaldecompose=None,
    reorderqubits=None,
    remove_swaps=None,
    simplify=None,
    reorderqubits_seed=None,
):
    """Validate the circuit-preparation flags and return them normalised, as a
    dict keyed by flag name.

    ``fuse``, ``canonicaldecompose``, ``remove_swaps``, ``simplify`` and
    ``reorderqubits`` must be ``None`` or a bool (NumPy bools become ``bool``); ``reorderqubits``
    may also be a method name. ``fuse_threshold`` and ``reorderqubits_seed``
    must be ``None`` or an integer other than a bool, and become ``int``.
    Anything else raises :class:`TypeError`: a value like ``fuse=0`` would
    otherwise read as "set" and install the pass.
    """
    out = {}
    for name, value in (
        ("fuse", fuse),
        ("canonicaldecompose", canonicaldecompose),
        ("reorderqubits", reorderqubits),
        ("remove_swaps", remove_swaps),
        ("simplify", simplify),
    ):
        if isinstance(value, np.bool_):
            value = bool(value)
        if not (value is None or isinstance(value, bool)
                or (name == "reorderqubits" and isinstance(value, str))):
            allowed = "None, a bool or a str" if name == "reorderqubits" else "None or a bool"
            raise TypeError(f"{name} must be {allowed}, not {value!r}")
        out[name] = value
    for name, value in (
        ("fuse_threshold", fuse_threshold),
        ("reorderqubits_seed", reorderqubits_seed),
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be None or an integer, not {value!r}")
            value = int(value)
        out[name] = value
    return out


def _flag_pass(backend, flag, value, fuse_threshold, reorderqubits_seed):
    """The pass a flag asks for, when set to anything but ``False``.

    The backend builds it, so a flag never swaps a backend's own pass for the
    generic one.
    """
    if flag == "remove_swaps":
        return backend.remove_swaps_pass()
    if flag == "simplify":
        return backend.simplify_pass()
    if flag == "canonicaldecompose":
        return backend.canonical_decompose_pass()
    if flag == "fuse":
        return backend.fuse_pass(threshold=fuse_threshold)
    method = None if value is True else str(value)
    return backend.reorder_pass(method, seed=reorderqubits_seed)


def _flag_rank(name):
    """Where a pass of this kind runs; passes no flag names rank first."""
    order = list(_FLAG_PASS_NAMES.values())
    return order.index(name) + 1 if name in order else 0


def _insert_position(names, name):
    rank = _flag_rank(name)
    for i, n in enumerate(names):
        if _flag_rank(n) > rank:
            return i
    return len(names)


def resolve_pipeline(
    backend,
    passes,
    *,
    fuse=None,
    fuse_threshold=None,
    canonicaldecompose=None,
    reorderqubits=None,
    remove_swaps=None,
    simplify=None,
    reorderqubits_seed=None,
) -> "PassPipeline":
    """Circuit preparation, resolved: the backend's
    :meth:`~mimiqcircuits.backends.Backend.default_passes`, adjusted by the
    flags, then extended with the caller's own ``passes``.

    Each flag speaks about one kind of pass. Set to ``False`` it drops that
    pass from the backend's declaration; set to anything else it replaces it
    with the pass the backend builds for that flag
    (:meth:`~mimiqcircuits.backends.Backend.fuse_pass`,
    :meth:`~mimiqcircuits.backends.Backend.reorder_pass`,
    :meth:`~mimiqcircuits.backends.Backend.canonical_decompose_pass`,
    :meth:`~mimiqcircuits.backends.Backend.remove_swaps_pass`,
    :meth:`~mimiqcircuits.backends.Backend.simplify_pass`), or adds that
    one if the backend declared none. Everything the flags say nothing about is left exactly as the backend
    wants it.

    ``passes`` are appended at the end, as given. Invalid flag values raise
    :class:`TypeError`; see :func:`check_prep_flags`.
    """
    flags = check_prep_flags(
        fuse=fuse,
        fuse_threshold=fuse_threshold,
        canonicaldecompose=canonicaldecompose,
        reorderqubits=reorderqubits,
        remove_swaps=remove_swaps,
        simplify=simplify,
        reorderqubits_seed=reorderqubits_seed,
    )
    fuse = flags["fuse"]
    fuse_threshold = flags["fuse_threshold"]
    canonicaldecompose = flags["canonicaldecompose"]
    reorderqubits = flags["reorderqubits"]
    remove_swaps = flags["remove_swaps"]
    simplify = flags["simplify"]
    reorderqubits_seed = flags["reorderqubits_seed"]
    if fuse is False and fuse_threshold is not None:
        raise ValueError(
            f"fuse_threshold={fuse_threshold!r} given with fuse=False: "
            "nothing to threshold"
        )

    # A threshold on its own is a request to fuse.
    if fuse is None and fuse_threshold is not None:
        fuse = True

    values = {
        "remove_swaps": remove_swaps,
        "simplify": simplify,
        "canonicaldecompose": canonicaldecompose,
        "fuse": fuse,
        "reorderqubits": reorderqubits,
    }

    # Nothing to adjust: leave the backend's declaration alone rather than
    # taking it apart and putting it back together.
    if all(v is None for v in values.values()):
        if passes is None:
            return backend.default_passes()
        return backend.default_passes() + passes

    out = list(backend.default_passes())
    # spec() builds a whole PassSpec, parameters and all, so it is far too
    # expensive to call per flag per pass: take the names once and keep them in
    # step with `out`.
    names = [p.spec().name for p in out]

    for flag, name in _FLAG_PASS_NAMES.items():
        value = values[flag]
        if value is None:
            continue

        i = names.index(name) if name in names else None
        if value is False:
            if i is not None:
                del out[i]
                del names[i]
        else:
            p = _flag_pass(backend, flag, value, fuse_threshold, reorderqubits_seed)
            if i is None:
                j = _insert_position(names, name)
                out.insert(j, p)
                names.insert(j, name)
            else:
                out[i] = p

    if passes is not None:
        out.extend(passes)

    return PassPipeline(out)


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
