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

A pass transforms a :class:`Circuit` before evolution. A
:class:`PassPipeline` holds passes in order, and :func:`apply_passes` runs
them and composes the qubit permutations they report.

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
    """Base class for the typed values used as pass parameters.

    The variants (:class:`PStr`, :class:`PInt`, :class:`PFloat`,
    :class:`PBool`, :class:`PSym`, :class:`PList`, :class:`PDict`) keep type
    information that a plain dictionary would lose when sent to a remote
    backend: the string ``"greedy"`` (a :class:`PStr`) does not come back as
    a symbol (a :class:`PSym`). Equality depends on the variant:
    ``PSym("x") != PStr("x")``. Use :func:`to_pass_param` to convert plain
    Python values.
    """


@dataclass(frozen=True)
class PStr(PassParam):
    """String pass parameter.

    Attributes:
        value (str): The string.
    """

    value: str


@dataclass(frozen=True)
class PInt(PassParam):
    """Integer pass parameter.

    Attributes:
        value (int): The integer.
    """

    value: int


@dataclass(frozen=True)
class PFloat(PassParam):
    """Floating-point pass parameter.

    Attributes:
        value (float): The number.
    """

    value: float


@dataclass(frozen=True)
class PBool(PassParam):
    """Boolean pass parameter.

    Attributes:
        value (bool): The boolean.
    """

    value: bool


@dataclass(frozen=True)
class PSym(PassParam):
    """Pass parameter that names one of a fixed set of choices, like an
    :class:`enum.Enum` member.

    It never compares equal to a :class:`PStr` with the same text, and
    :func:`to_pass_param` never produces it: build it explicitly.

    Attributes:
        value (str): The name of the choice.
    """

    value: str


@dataclass(frozen=True)
class PList(PassParam):
    """List pass parameter.

    Attributes:
        items (tuple[PassParam, ...]): The elements.
    """

    items: tuple[PassParam, ...]


@dataclass(frozen=True)
class PDict(PassParam):
    """Dictionary pass parameter with string keys.

    Attributes:
        items (tuple[tuple[str, PassParam], ...]): The ``(key, value)``
            pairs, as a tuple so that the parameter stays hashable.
    """

    items: tuple[tuple[str, PassParam], ...]


def to_pass_param(x) -> PassParam:
    """Convert a plain Python value into a :class:`PassParam`.

    Args:
        x: A :class:`PassParam` (returned as it is), a ``bool``, ``int``,
            ``float`` or ``str``, or a list, tuple or dictionary of such
            values, converted recursively. Dictionary keys become strings.

    Returns:
        The matching :class:`PassParam`. Strings become :class:`PStr`,
        never :class:`PSym`.

    Raises:
        TypeError: For any other type.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import to_pass_param
        >>> to_pass_param(True)
        PBool(value=True)
        >>> to_pass_param({"method": "greedy", "steps": [1, 2]})
        PDict(items=(('method', PStr(value='greedy')), ('steps', PList(items=(PInt(value=1), PInt(value=2))))))
    """
    if isinstance(x, PassParam):
        return x
    # bool is a subclass of int, so it must be checked first.
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
    """Description of a pass: its name and parameters.

    Two specs with the same fields compare equal and hash the same. The
    name identifies the kind of pass: the preparation flags of
    :meth:`Backend.execute` find the pass they act on by name, and
    :class:`MimiqRemoteBackend` translates passes to server options by name.

    Attributes:
        name (str): Name of the kind of pass, such as ``"fuse_gates"``.
        parameters (tuple[tuple[str, PassParam], ...]): The parameters, as
            ``(name, value)`` pairs.
        requires (tuple[str, ...]): Names of passes that must run before
            this one. Informational; nothing enforces it.
        conflicts (tuple[str, ...]): Names of passes that cannot be in the
            same pipeline. Informational; nothing enforces it.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import PassSpec
        >>> PassSpec.from_dict("reorder_qubits", {"method": "sa"})
        PassSpec(name='reorder_qubits', parameters=(('method', PStr(value='sa')),), requires=(), conflicts=())
    """

    name: str
    parameters: tuple[tuple[str, PassParam], ...] = ()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @staticmethod
    def from_dict(name: str, parameters: Optional[dict] = None,
                  requires: Iterable[str] = (), conflicts: Iterable[str] = ()
                  ) -> "PassSpec":
        """Build a spec from plain Python values.

        Args:
            name: The name of the kind of pass.
            parameters: The parameters, converted with
                :func:`to_pass_param`.
            requires: Names of passes that must run before this one.
            conflicts: Names of passes that cannot be in the same pipeline.

        Returns:
            The new :class:`PassSpec`.
        """
        params = tuple((str(k), to_pass_param(v))
                       for k, v in (parameters or {}).items())
        return PassSpec(name, params, tuple(requires), tuple(conflicts))


@dataclass
class PassResult:
    """What a pass reports besides the new circuit, returned by
    :meth:`AbstractPass.apply`.

    Attributes:
        qubit_permutation (list[int] or None): The 1-based relabelling of
            qubits the pass applied, or ``None`` when indices are unchanged.
            :func:`apply_passes` composes these across the pipeline.
        metadata (dict[str, Any]): Free-form information about the run, such
            as timings or gate counts.
    """

    qubit_permutation: Optional[list[int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassContext:
    """Inputs a pass may need, given to :meth:`AbstractPass.apply`.

    Attributes:
        backend (Backend or None): The backend the circuit is prepared for,
            or ``None``. With a backend, :func:`apply_passes` also checks
            :meth:`Backend.accepts_pass` and :meth:`Backend.delegates_pass`.
        rng (random.Random): The generator for randomised passes;
            :meth:`LocalBackend.execute` passes :attr:`RNGs.pass_`.
        bitstrings (list): Basis states for passes that need them. The
            execute driver leaves it empty.
        features (set[str]): Circuit features for passes that need them.
            The execute driver leaves it empty.
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

    To write a pass, subclass it and implement :meth:`spec` and :meth:`apply`.
    Override :meth:`preserves` when the pass breaks a circuit feature.

    Examples:
        A pass that removes every ``Barrier``.

        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import (
        ...     AbstractPass, PassContext, PassResult, PassSpec)
        >>> class DropBarriersPass(AbstractPass):
        ...     def spec(self):
        ...         return PassSpec("drop_barriers")
        ...     def apply(self, ctx, circuit):
        ...         kept = [i for i in circuit if not isinstance(i.operation, Barrier)]
        ...         # No qubit is relabelled, so the result reports nothing.
        ...         return Circuit(kept), PassResult()
        >>> c = Circuit().push(GateH(), 0).push(Barrier(1), 0).push(GateX(), 0)
        >>> new, result = DropBarriersPass().apply(PassContext(), c)
        >>> new
        1-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── X @ q[0]
        <BLANKLINE>

        To run it in an execution, give it to a backend as
        ``backend.execute(c, passes=PassPipeline([DropBarriersPass()]))``.
    """

    @abc.abstractmethod
    def spec(self) -> PassSpec:
        """Return a :class:`PassSpec` describing the pass.

        Its name is how the preparation flags and
        :class:`MimiqRemoteBackend` identify the pass.
        """

    @abc.abstractmethod
    def apply(self, ctx: PassContext, circuit) -> tuple[Any, PassResult]:
        """Run the pass on ``circuit``.

        Args:
            ctx: The :class:`PassContext` of the run.
            circuit: The input circuit. Return a new circuit rather than
                changing it in place.

        Returns:
            A tuple ``(new_circuit, PassResult)``. A pass that relabels
            qubits reports it in :attr:`PassResult.qubit_permutation`.
        """

    def preserves(self, feature: str) -> bool:
        """Return whether the pass keeps ``feature`` of the input circuit intact.

        The default returns ``True`` for every feature; override it when the
        pass breaks one. Neither :func:`apply_passes` nor the execute driver
        reads it.

        Args:
            feature: One of ``"feed_forward"``, ``"midcircuit_measure"``,
                ``"parametric"``, ``"loss"``, ``"exact_equivalence"``,
                ``"strict_qubit_count"``.
        """
        return True


# ──────────────────────────────────────────────────────────────────────────
# PassPipeline
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PassPipeline:
    """An ordered sequence of passes.

    It supports ``len``, indexing and iteration, and ``+`` appends another
    pipeline or a list of passes, returning a new pipeline. Run it with
    :func:`apply_passes`, or give it to :meth:`Backend.execute` as
    ``passes``.

    Attributes:
        passes (list[AbstractPass]): The passes, in the order they run.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import (
        ...     CanonicalDecomposePass, PassContext, PassPipeline, apply_passes)
        >>> pipeline = PassPipeline([CanonicalDecomposePass()])
        >>> len(pipeline)
        1

        Run the pipeline on a circuit. The Hadamard becomes a ``U`` gate,
        and no qubit was relabelled.

        >>> c = Circuit().push(GateH(), 0)
        >>> new, perm, results = apply_passes(pipeline, PassContext(), c)
        >>> new
        1-qubit circuit with 1 instruction:
        └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
        <BLANKLINE>
        >>> perm is None
        True
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
    """Error for a pass the backend refuses in :meth:`Backend.accepts_pass`.

    :func:`apply_passes` raises it instead of skipping the pass.

    Args:
        backend_name: Name of the backend.
        pass_name: Name of the pass, from its :class:`PassSpec`.
    """

    def __init__(self, backend_name: str, pass_name: str):
        self.backend_name = backend_name
        self.pass_name = pass_name
        super().__init__(
            f"UnacceptedPassError: backend {backend_name} does not accept "
            f"pass :{pass_name}"
        )


class RemotePassOrderError(Exception):
    """Error for a non-empty ``passes`` given to a remote backend that does
    not declare ``"pass_order_honored"``, with ``strict_pass_order=True``.

    Pass ``strict_pass_order=False`` to submit the passes as an unordered
    set instead: :class:`MimiqRemoteBackend` translates the passes it knows
    into server options and drops the others with a warning.

    Args:
        backend_name: Name of the backend.
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
    """Return the inverse of a 1-based permutation.

    Args:
        perm: The permutation, as a list of the values ``1`` to ``len(perm)``.

    Returns:
        The inverse permutation, also 1-based.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import invert_perm
        >>> invert_perm([2, 3, 1])
        [3, 1, 2]
    """
    inv = [0] * len(perm)
    for i, p in enumerate(perm, start=1):
        inv[p - 1] = i
    return inv


def apply_passes(pipeline: PassPipeline, ctx: PassContext, circuit
                 ) -> tuple[Any, Optional[list[int]], list[PassResult]]:
    """Run the passes of ``pipeline`` on ``circuit``, in order.

    When ``ctx.backend`` is set, each pass is first checked with
    :meth:`Backend.accepts_pass`. A pass for which
    :meth:`Backend.delegates_pass` returns ``True`` is not run: its result
    is ``PassResult(metadata={"delegated": True})`` and the backend does the
    work itself.

    Args:
        pipeline: The passes to run.
        ctx: The :class:`PassContext` given to every pass.
        circuit: The input circuit. It is passed to the first pass as it is.

    Returns:
        A tuple ``(circuit, permutation, results)``: the transformed circuit,
        the composition of every reported ``qubit_permutation`` (``None``
        when no pass reported one), and one :class:`PassResult` per pass.

    Raises:
        UnacceptedPassError: If the backend refuses a pass.

    See Also:
        :class:`PassPipeline`, whose example runs a pipeline.
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
