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

"""Concrete :class:`RemoteBackend` for the MIMIQ cloud service.

:class:`MimiqRemoteBackend` wraps a
:class:`mimiqcircuits.remote.RemoteConnection` so cloud execution looks
like every other backend: it advertises a capability set, accepts the
same ``seed`` / ``rng`` / ``passes`` arguments as the local simulators,
and returns the same :class:`mimiqcircuits.QCSResults`. Submit-level
options like ``bonddim`` / ``mpscutoff`` are forwarded to
:meth:`MimiqRemoteBackend.submit`.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional

from mimiqcircuits.backends._rng_utils import normalize_seed
from mimiqcircuits.backends.backend import RemoteBackend
from mimiqcircuits.backends.capabilities import (
    AllToAll,
    Limits,
    Topology,
)
from mimiqcircuits.backends.fidelity import (
    ExactFidelity,
    Fidelity,
    TruncationLowerBound,
    UnknownFidelity,
)
from mimiqcircuits.backends.passes import (
    AbstractPass,
    PassPipeline,
    RemotePassOrderError,
)


# Synthesised client-side; the server has no capability endpoint yet.
# Capabilities omitted on purpose:
#   :pass_order_honored      — server takes a flat bag of booleans, so
#                              pipeline order is not honoured. Submitting
#                              with `strict_pass_order=True` raises
#                              `RemotePassOrderError`.
#   :expectation_state       — no top-level `expectation(state, op)` wire.
#   :expectation_paulistring — server dispatch unknown.
#   :loss, :streaming        — server dispatch unknown.
#   :while_statement,
#   :calibrated_noise,
#   :shared_prefix_batch,
#   :parametric_batch,
#   :final_measure_only      — pending a documented server contract.
_BASE_REMOTE_CAPABILITIES: frozenset[str] = frozenset({
    "amplitude",
    "sampling",
    "classical_bits",
    "zvars",
    "midcircuit_measure",
    "midcircuit_reset",
    "reset_after_measure",
    "feed_forward",
    "noise",
    "expectation_1q",
    "expectation_2q",
    "parametric",
    "batch",
})


# Translate `PassSpec.name` → legacy server knob. Keyed by the spec
# name string (not by class) so this module never has to import
# concrete-pass types.
_LEGACY_KNOB_REGISTRY: dict[str, Callable[[AbstractPass], dict]] = {
    "remove_swaps": lambda p: {"remove_swaps": True},
    "fuse_gates": lambda p: {"fuse": True},
    "canonical_decompose": lambda p: {"canonicaldecompose": True},
    "reorder_qubits": lambda p: {
        "reorderqubits": _extract_reorder_method(p),
    },
}


def _extract_reorder_method(p: AbstractPass):
    """Read the optional ``method`` PassParam from a reorder_qubits spec.

    Returns the raw value if present, otherwise ``True`` (server
    default). ``PassSpec.parameters`` is declared as a tuple of
    ``(key, value)`` pairs (see :mod:`passes`) rather than a dict, so
    we convert before lookup; using ``.get(...)`` directly on the
    tuple would silently succeed only for fixtures that violated the
    type contract.
    """
    params = p.spec().parameters
    if not params:
        return True
    method = dict(params).get("method")
    if method is None:
        return True
    # PassParam wraps a `value` attribute, but accept bare scalars too.
    return getattr(method, "value", method)


def _pipeline_to_legacy_knobs(
    pipeline: PassPipeline,
) -> tuple[dict, list[str]]:
    """Translate a PassPipeline into the server's legacy boolean knobs.

    Returns ``(knobs, unrecognised)``. The caller emits a single
    ``UserWarning`` per ``submit()`` for any unrecognised passes; one
    warning per pass would spam users that hit the same translation
    miss on every call.
    """
    knobs: dict = {}
    unknown: list[str] = []
    for p in pipeline:
        name = p.spec().name
        translator = _LEGACY_KNOB_REGISTRY.get(name)
        if translator is None:
            unknown.append(name)
        else:
            knobs.update(translator(p))
    return knobs, unknown


def _wrap_fidelity_by_simulator(sim_string: str, value) -> Fidelity:
    """Pick a :class:`Fidelity` subclass for a server-returned float.

    The server's protobuf ``QCSResults.fidelities`` is ``repeated
    double``; the client must restore the typed-Fidelity invariant
    from each result's ``simulator`` identity string (set by the
    server, e.g. ``"MIMIQ-MPS 0.18.3"``).

    Mapping:

    ``MIMIQ-StateVector*``
        → :class:`ExactFidelity`. The server's value (typically 1.0)
        is discarded because the state-vector algorithm is
        numerically exact under its own contract. Caveat: for
        non-trace-preserving Kraus channels (e.g. amplitude damping
        with explicit loss) the renormalising state-vec path does
        not track the surviving-trajectory probability, so
        ``ExactFidelity`` over-states reliability — same caveat the
        local ``QuantaniumQCS`` docstring carries.

    ``MIMIQ-MPS*``
        → :class:`TruncationLowerBound`. Caveat: when Kraus channels
        are present the server's accumulator currently conflates the
        truncation product with the per-trajectory branch weight in
        the same scalar; a future split accumulator will separate
        them.

    anything else / empty
        → :class:`UnknownFidelity` (conservative). Will be replaced
        once the server sends a typed Fidelity on the wire.

    Idempotent: already-typed :class:`Fidelity` values pass through
    unchanged so callers can re-invoke ``_MimiqJob.wait()`` safely.
    """
    if isinstance(value, Fidelity):
        return value
    s = sim_string or ""
    if s.startswith("MIMIQ-StateVector"):
        return ExactFidelity()
    if s.startswith("MIMIQ-MPS"):
        return TruncationLowerBound(float(value))
    return UnknownFidelity()


class _MimiqJob:
    """Handle returned by :meth:`MimiqRemoteBackend.submit`.

    Holds the server-issued ``request_id`` and a back-reference to
    the connection. :meth:`wait` polls until the job completes,
    downloads the results, and re-types each result's ``fidelities``
    into the appropriate :class:`Fidelity` subclass based on the
    simulator identity string returned by the server.
    """

    def __init__(self, connection, request_id, *, interval: float = 1.0):
        self._connection = connection
        self._request_id = request_id
        self._interval = interval

    @property
    def request_id(self) -> str:
        return self._request_id

    def wait(self, *, timeout: Optional[float] = None):
        """Block until the remote job completes and return the results.

        Args:
            timeout: seconds to wait. ``None`` blocks indefinitely
                (matching ``connection.get_results``). A positive
                value raises :class:`TimeoutError` instead of hanging
                on a stuck server.

        Returns:
            list[QCSResults]: one entry per submitted Circuit, with
            ``fidelities`` wrapped into typed :class:`Fidelity`
            subclasses.
        """
        deadline = (
            time.monotonic() + timeout if timeout is not None else None
        )
        # Re-implement the poll loop instead of delegating to
        # `get_results` so the deadline can be enforced cleanly.
        inner = self._connection.connection
        while not inner.isJobDone(self._request_id):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Remote job {self._request_id} did not complete "
                    f"within {timeout}s."
                )
            time.sleep(self._interval)

        # The job is done, so `get_results`' internal poll loop exits
        # on its first check.
        results = self._connection.get_results(
            self._request_id, interval=self._interval
        )
        for r in results:
            sim_id = getattr(r, "simulator", "") or ""
            r.fidelities = [
                _wrap_fidelity_by_simulator(sim_id, v)
                for v in (r.fidelities or [])
            ]
        return results


class MimiqRemoteBackend(RemoteBackend):
    """Cloud backend wrapping a
    :class:`mimiqcircuits.remote.RemoteConnection`.

    Construct with an authenticated connection (or any object that
    exposes callable ``submit(...)`` and ``get_results(...)``
    methods). The constructor pins the simulator ``algorithm`` (which
    in turn shapes :meth:`capabilities`) and the label / poll
    cadence; all other server knobs (``bonddim``, ``entdim``,
    ``mpscutoff``, ``mpsmethod``, ``mpotraversal``, ``timelimit``,
    ``noisemodel``) are passed at :meth:`submit` / :meth:`execute`
    time.

    Caveats:

    1. **Capabilities are synthesised client-side.** The server has
       no capability endpoint yet, so :meth:`capabilities` returns a
       conservative best-guess (``_BASE_REMOTE_CAPABILITIES`` plus
       algorithm-conditional MPS tokens).

    2. **``pass_order_honored`` is not declared.** The server takes a
       flat bag of booleans (``remove_swaps``, ``fuse``, …) with no
       notion of pipeline order. A non-empty ``passes=`` with the
       default ``strict_pass_order=True`` raises
       :class:`RemotePassOrderError`. Opt out with
       ``strict_pass_order=False``: recognised passes are translated
       into legacy knobs and unrecognised ones are dropped with a
       single :class:`UserWarning`.

    3. **``param_grid=`` raises** :class:`NotImplementedError`. Loop
       on the client side instead::

           for params in grid:
               backend.execute(c.evaluate(params), nsamples=n, seed=...)

    4. **``noisemodel=`` and ``ApplyNoiseModelPass`` are mutually
       exclusive.** Specifying both raises :class:`ValueError`.

    5. **``_MimiqJob.wait()`` blocks indefinitely by default.** Pass
       ``timeout=<seconds>`` for a bounded wait that raises
       :class:`TimeoutError`.

    6. **``seed=`` / ``rng=`` are mutually exclusive.** The wire seed
       is a deterministic xor-fold of the four-stream :class:`RNGs`
       bundle derived from your input, not literally the integer
       passed in.

    7. **The connection must be authenticated** (``conn.connect()``
       called) before the first :meth:`execute`. The constructor
       does not trigger interactive auth, so an unauthenticated
       connection produces a 401 at submit time.

    8. **Output shape mirrors input:** a single ``Circuit`` returns
       a single :class:`QCSResults`; a list returns a list of
       matching length.

    9. **``algorithm="statevector"`` + ``noisemodel=``** is a
       server-side runtime error — state-vector cannot apply Kraus —
       and there is currently no client-side guard.

    10. **``algorithm="statevector"`` makes MPS knobs meaningless**
        (``bonddim=``, ``entdim=``, ``mpscutoff=``, ``mpsmethod=``,
        ``mpotraversal=``, ``BondDim`` / ``SchmidtRank`` instructions).
        :meth:`capabilities` correctly omits ``:bond_dim`` /
        ``:schmidt_rank`` in that mode and :meth:`limits` returns
        ``max_bond_dim=None``, but :meth:`submit` still forwards the
        kwargs to the server, which is canonical.

    Example::

        import mimiqcircuits as mc
        conn = mc.MimiqConnection()
        conn.connect()
        backend = mc.backends.MimiqRemoteBackend(conn, algorithm="mps")
        results = backend.execute(circuit, nsamples=1000, seed=42, bonddim=64)
    """

    def __init__(
        self,
        connection,
        *,
        algorithm: str = "auto",
        label: Optional[str] = None,
        poll_interval: float = 1.0,
    ):
        # Duck-type guard: accept anything exposing the two methods
        # the backend actually calls. Keeps test fakes lightweight and
        # avoids importing the concrete connection type here.
        for attr in ("submit", "get_results"):
            if not callable(getattr(connection, attr, None)):
                raise TypeError(
                    f"connection must expose callable .{attr}(...); "
                    f"got {type(connection).__name__}. Wrap a raw "
                    f"mimiqlink connection in "
                    f"mimiqcircuits.remote.MimiqConnection first."
                )

        # Surface a configuration error at construction time rather
        # than on the first submit() round-trip.
        inner = getattr(connection, "connection", None)
        if inner is not None and getattr(inner, "url", None) is None:
            warnings.warn(
                "Underlying connection has no URL set; remote calls "
                "will fail at submit time.",
                stacklevel=2,
            )

        self._connection = connection
        self.algorithm = algorithm
        self.label = label
        self.poll_interval = poll_interval

    # ── identity ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Mimiq"

    @property
    def version(self) -> str:
        from mimiqcircuits.__version__ import __version__
        return __version__

    # ── advertisement ──────────────────────────────────────────────────────

    def capabilities(self) -> set[str]:
        """Synthesised client-side. Tensor-network annotations
        (``:bond_dim``, ``:schmidt_rank``) are declared only when the
        configured ``algorithm`` admits MPS execution.
        """
        caps = set(_BASE_REMOTE_CAPABILITIES)
        if self.algorithm in {"auto", "mps"}:
            caps |= {"bond_dim", "schmidt_rank"}
        return caps

    def limits(self) -> Limits:
        """Static client-side limits. ``max_bond_dim`` is dropped on
        state-vector deployments where bond dimension is meaningless.
        """
        from mimiqcircuits.remote import MAX_BONDDIM, MAX_SAMPLES

        max_bd = (
            MAX_BONDDIM if self.algorithm in {"auto", "mps"} else None
        )
        return Limits(max_samples=MAX_SAMPLES, max_bond_dim=max_bd)

    def topology(self) -> Topology:
        return AllToAll()

    # ── submit / execute ───────────────────────────────────────────────────

    def submit(
        self,
        circuits,
        nsamples: int = 1000,
        *,
        rngs=None,
        passes: Optional[PassPipeline] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
        # Submit-time tuning knobs. Forwarded to the server as-is; a
        # `None` value means "let the server pick its default".
        bonddim: Optional[int] = None,
        entdim: Optional[int] = None,
        mpscutoff: Optional[float] = None,
        mpsmethod: Optional[str] = None,
        mpotraversal: Optional[str] = None,
        timelimit: Optional[int] = None,
        noisemodel=None,
        **kwargs,
    ) -> _MimiqJob:
        """Submit ``circuits`` to the remote server.

        Returns a :class:`_MimiqJob`. Calling ``job.wait()`` blocks
        until results are available and post-processes
        :class:`QCSResults.fidelities` into typed :class:`Fidelity`
        instances.
        """
        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {sorted(kwargs)}"
            )

        if param_grid:
            raise NotImplementedError(
                "param_grid= is not yet supported on "
                "MimiqRemoteBackend. Loop in user code, substituting "
                "parameters per point — e.g.\n"
                "    for params in grid:\n"
                "        backend.execute(c.evaluate(params), nsamples=n)"
            )

        # Run the noisemodel/ApplyNoiseModelPass collision check
        # *before* the strict-pass-order check: a user who passed
        # both wants "you specified noise twice" first, not the
        # generic order-not-honoured diagnostic.
        if noisemodel is not None and passes is not None:
            for p in passes:
                if p.spec().name == "apply_noise_model":
                    raise ValueError(
                        "Specify noise via either the submit() "
                        "'noisemodel=' kwarg or via "
                        "ApplyNoiseModelPass in passes=, not both."
                    )

        # Re-run the strict-order check inside submit() so direct
        # `submit()` callers (bypassing `execute()`) cannot smuggle
        # an ordered pipeline past the guard.
        if (
            passes is not None
            and len(passes) > 0
            and strict_pass_order
            and "pass_order_honored" not in self.capabilities()
        ):
            raise RemotePassOrderError(self.name)

        # When `execute()` already coerced an int into an RNGs bundle
        # the helper is a no-op; direct `submit()` callers may still
        # pass an int, and `normalize_seed` handles both.
        seed = normalize_seed(None, rngs)

        # `strict_pass_order=False` opt-in: translate recognised
        # passes into legacy boolean knobs.
        pipeline_knobs: dict = {}
        if (
            passes is not None
            and len(passes) > 0
            and not strict_pass_order
        ):
            pipeline_knobs, unknown = _pipeline_to_legacy_knobs(passes)
            if unknown:
                warnings.warn(
                    f"MimiqRemoteBackend dropped unrecognised passes "
                    f"{unknown}; the server only consumes the legacy "
                    f"boolean knobs. Recognised passes were forwarded.",
                    stacklevel=2,
                )

        submit_kwargs = {
            "label": self.label or "pyapi_remote_backend",
            "algorithm": self.algorithm,
            "nsamples": nsamples,
            "timelimit": timelimit,
            "bonddim": bonddim,
            "entdim": entdim,
            "mpscutoff": mpscutoff,
            "mpsmethod": mpsmethod,
            "mpotraversal": mpotraversal,
            "noisemodel": noisemodel,
            "seed": seed,
        }
        submit_kwargs.update(pipeline_knobs)

        request_id = self._connection.submit(circuits, **submit_kwargs)
        return _MimiqJob(
            self._connection,
            request_id,
            interval=self.poll_interval,
        )

    def execute(
        self,
        circuit,
        *,
        nsamples: int = 1000,
        seed: Optional[int] = None,
        rng=None,
        passes: Optional[PassPipeline] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
        **kwargs,
    ):
        """Submit ``circuit`` to the cloud and wait for results.

        ``seed`` and ``rng`` are mutually exclusive sources of
        randomness. Submit-time tuning options (``bonddim``,
        ``entdim``, ``mpscutoff``, ``mpsmethod``, ``mpotraversal``,
        ``timelimit``, ``noisemodel``) pass through to :meth:`submit`.

        Output shape mirrors input: a single :class:`Circuit` returns a
        single :class:`QCSResults`; ``list[Circuit]`` returns
        ``list[QCSResults]``.

        To request specific amplitudes, push :class:`Amplitude`
        instructions into ``circuit`` and read ``results.zstates``.
        """
        # The noisemodel duplication error is more specific than the
        # order-not-honoured diagnostic, so check it first.
        noisemodel = kwargs.get("noisemodel")
        if noisemodel is not None and passes is not None:
            for p in passes:
                if p.spec().name == "apply_noise_model":
                    raise ValueError(
                        "Specify noise via either the submit() "
                        "'noisemodel=' kwarg or via "
                        "ApplyNoiseModelPass in passes=, not both."
                    )

        # Surface the strict-pass-order error before any network call.
        if (
            passes is not None
            and len(passes) > 0
            and strict_pass_order
            and "pass_order_honored" not in self.capabilities()
        ):
            raise RemotePassOrderError(self.name)

        rngs = self._resolve_rngs(seed, rng)
        job = self.submit(
            circuit,
            nsamples,
            rngs=rngs,
            passes=passes,
            callback=callback,
            param_grid=param_grid,
            strict_pass_order=strict_pass_order,
            **kwargs,
        )
        results = job.wait()  # list[QCSResults]
        # Preserve input shape: a single Circuit in must produce a
        # single QCSResults out, not `[results]`.
        if not isinstance(circuit, list) and len(results) == 1:
            return results[0]
        return results

    def __repr__(self) -> str:
        return (
            f"MimiqRemoteBackend({self._connection!r}, "
            f"algorithm={self.algorithm!r})"
        )
