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
        ``ExactFidelity`` over-states reliability.

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
            # Optimization jobs return OptimizationResults, which carry no
            # fidelities of their own.
            if not hasattr(r, "fidelities"):
                continue
            sim_id = getattr(r, "simulator", "") or ""
            r.fidelities = [
                _wrap_fidelity_by_simulator(sim_id, v)
                for v in (r.fidelities or [])
            ]
        return results


class MimiqRemoteBackend(RemoteBackend):
    """Backend that runs circuits on the MIMIQ cloud service.

    It wraps a :class:`~mimiqcircuits.MimiqConnection` (or another
    :class:`mimiqcircuits.remote.RemoteConnection`) so that cloud execution
    takes the same ``seed``, ``rng`` and ``passes`` arguments as a local
    backend and returns the same :class:`~mimiqcircuits.QCSResults`. The
    simulator algorithm is fixed at construction; the other server options
    (``bonddim``, ``entdim``, ``mpscutoff``, ``mpsmethod``,
    ``mpotraversal``, ``timelimit``, ``noisemodel``) are given to
    :meth:`execute` or :meth:`submit`.

    Args:
        connection: An authenticated connection, or any object with callable
            ``submit(...)`` and ``get_results(...)`` methods. The constructor
            does not log in, so call ``connection.connect()`` first.
        algorithm: ``"auto"``, ``"mps"`` or ``"statevector"``. It also sets
            :meth:`capabilities` and :meth:`limits`.
        label: Label of the jobs on the service. Defaults to
            ``"pyapi_remote_backend"``.
        poll_interval: Seconds between two checks of a running job.

    Raises:
        TypeError: If ``connection`` lacks ``submit`` or ``get_results``.

    Notes:
        1. The capabilities are decided on the client, since the service does
           not report them: a fixed set, plus ``"bond_dim"`` and
           ``"schmidt_rank"`` when ``algorithm`` is ``"auto"`` or ``"mps"``.
        2. ``"pass_order_honored"`` is not declared, because the service takes
           a flat set of options (``remove_swaps``, ``fuse``, …), not an
           ordered pipeline. A non-empty ``passes`` with the default
           ``strict_pass_order=True`` raises :class:`RemotePassOrderError`.
           With ``strict_pass_order=False`` the passes the service knows are
           translated into options, and the others are dropped with one
           :class:`UserWarning`.
        3. ``param_grid`` raises :class:`NotImplementedError`. Evaluate the
           circuit for each point and execute the results instead.
        4. ``noisemodel=`` together with a pass named
           ``"apply_noise_model"`` in ``passes`` raises :class:`ValueError`.
        5. ``seed`` and ``rng`` are mutually exclusive. The seed sent to the
           service is derived from them through an :class:`RNGs` bundle, so
           it is not the integer passed in.
        6. A single circuit returns a single :class:`~mimiqcircuits.QCSResults`;
           a list returns a list of the same length. A list of more than
           one circuit needs ``algorithm`` set to ``"mps"`` or
           ``"statevector"``; with ``"auto"`` it raises :class:`ValueError`.
        7. With ``algorithm="statevector"``, a ``noisemodel`` fails on the
           service, and the MPS options (``bonddim``, ``entdim``,
           ``mpscutoff``, ``mpsmethod``, ``mpotraversal``) have no effect.
           They are still sent; the backend does not reject them for this
           algorithm.
        8. :meth:`execute` waits for the job without a time limit. For a
           bounded wait, call :meth:`submit` and then
           ``job.wait(timeout=seconds)``, which raises :class:`TimeoutError`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import MimiqRemoteBackend

        Log in once, then wrap the connection in a backend that uses the MPS
        simulator.

        >>> conn = MimiqConnection()
        >>> conn.connect()  # doctest: +SKIP
        >>> backend = MimiqRemoteBackend(conn, algorithm="mps")

        Run a Bell circuit; ``bonddim`` is an option of the service.

        >>> c = Circuit().push(GateH(), 0).push(GateCX(), 0, 1)
        >>> c.push(Measure(), range(2), range(2))  # doctest: +ELLIPSIS
        2-qubit, 2-bit circuit with 4 instructions:
        ...
        >>> res = backend.execute(c, nsamples=1000, seed=42, bonddim=64)  # doctest: +SKIP
        >>> res.histogram()  # doctest: +SKIP
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
        """``"Mimiq"``."""
        return "Mimiq"

    @property
    def version(self) -> str:
        """The version of the ``mimiqcircuits`` package."""
        from mimiqcircuits.__version__ import __version__
        return __version__

    # ── advertisement ──────────────────────────────────────────────────────

    def capabilities(self) -> set[str]:
        """Return the capabilities, decided on the client.

        ``"bond_dim"`` and ``"schmidt_rank"`` are included only when
        ``algorithm`` is ``"auto"`` or ``"mps"``.
        """
        caps = set(_BASE_REMOTE_CAPABILITIES)
        if self.algorithm in {"auto", "mps"}:
            caps |= {"bond_dim", "schmidt_rank"}
        return caps

    def limits(self) -> Limits:
        """Return the limits known to the client.

        ``max_samples`` is the largest number of shots the client sends;
        ``max_bond_dim`` is set only when ``algorithm`` is ``"auto"`` or
        ``"mps"``.
        """
        from mimiqcircuits.remote import MAX_BONDDIM, MAX_SAMPLES

        max_bd = (
            MAX_BONDDIM if self.algorithm in {"auto", "mps"} else None
        )
        return Limits(max_samples=MAX_SAMPLES, max_bond_dim=max_bd)

    def topology(self) -> Topology:
        """Return :class:`AllToAll`."""
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
        # Circuit-preparation knobs run server-side. Alternatives to
        # `passes=`; the server consumes them as a flat option set.
        fuse: Optional[bool] = None,
        reorderqubits=None,
        canonicaldecompose: Optional[bool] = None,
        remove_swaps: Optional[bool] = None,
        reorderqubits_seed: Optional[int] = None,
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
        """Submit ``circuits`` to the service without waiting.

        :meth:`execute` calls it; call it directly to wait with a time
        limit.

        Args:
            circuits: A circuit or a list of circuits.
            nsamples: Number of shots.
            rngs: An :class:`RNGs` bundle or an integer seed, folded into the
                single seed sent to the service; ``None`` draws a random
                seed.
            passes: A :class:`PassPipeline`. See the class notes for how it
                is handled.
            callback: Accepted for compatibility with
                :meth:`RemoteBackend.submit`; not used.
            param_grid: Must be empty or ``None``.
            strict_pass_order: See :meth:`Backend.execute`.
            fuse: Server-side gate fusion; ``None`` lets the service choose.
            reorderqubits: Server-side qubit reordering, as a bool or a
                method name.
            canonicaldecompose: Server-side decomposition into the canonical
                gate set.
            remove_swaps: Server-side SWAP removal.
            reorderqubits_seed: Seed of the server-side reordering.
            bonddim: Bond dimension of the MPS simulator.
            entdim: Entanglement dimension of the MPS simulator.
            mpscutoff: Singular-value cutoff of the MPS simulator.
            mpsmethod: MPO application method of the MPS simulator.
            mpotraversal: MPO compression traversal of the MPS simulator.
            timelimit: Time limit of the job, in minutes.
            noisemodel: A :class:`~mimiqcircuits.NoiseModel` applied to each
                circuit before it is uploaded.
            **kwargs: Only ``simplify``, which must be falsy.

        Returns:
            A job handle. Its ``request_id`` is the service's job id, and its
            ``wait(timeout=None)`` blocks until the job is done and returns
            the list of :class:`~mimiqcircuits.QCSResults`, with their
            fidelities converted to :class:`Fidelity` types (exact for the
            state-vector simulator, a truncation lower bound for MPS,
            unknown otherwise). With a ``timeout`` in seconds it raises
            :class:`TimeoutError` instead of waiting longer.

        Raises:
            ValueError: If a preparation flag is given with a non-empty
                ``passes``, or ``noisemodel`` with a pass named
                ``"apply_noise_model"``.
            NotImplementedError: If ``param_grid`` is not empty, or
                ``simplify`` is set.
            RemotePassOrderError: See ``strict_pass_order``.
            TypeError: For an unknown keyword.
        """
        if kwargs.pop("simplify", None):
            raise NotImplementedError(
                "simplify= is not available on the remote service yet; call "
                "mimiqcircuits.simplify(circuit) before submitting instead"
            )
        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {sorted(kwargs)}"
            )

        # Deliberately stricter than local execution, which merges the two
        # (flags contribute their pass, `passes=` follows, then the backend's
        # defaults fill in the rest). Here the server owns pass selection: the
        # knobs go on the wire as options and a pipeline goes as a pipeline,
        # so a merge would have to happen server-side, where we cannot see it.
        # Rejecting the pair keeps their precedence from being ambiguous.
        prep_knobs = {
            k: v
            for k, v in (
                ("fuse", fuse),
                ("reorderqubits", reorderqubits),
                ("canonicaldecompose", canonicaldecompose),
                ("remove_swaps", remove_swaps),
                ("reorderqubits_seed", reorderqubits_seed),
            )
            if v is not None
        }
        if prep_knobs and passes is not None and len(passes) > 0:
            raise ValueError(
                "pass either `passes=` or the preparation flags "
                "(fuse=, reorderqubits=, …), not both. Local execution merges "
                "them; remote submission cannot, because the server chooses "
                "the passes"
            )

        if param_grid:
            raise NotImplementedError(
                "param_grid= is not yet supported on "
                "MimiqRemoteBackend. Loop in user code, substituting "
                "parameters per point — e.g.\n"
                "    for params in grid:\n"
                "        backend.execute(c.evaluate(params), nsamples=n)"
            )

        # Run the noisemodel/apply_noise_model collision check
        # *before* the strict-pass-order check: a user who passed
        # both wants "you specified noise twice" first, not the
        # generic order-not-honoured diagnostic.
        if noisemodel is not None and passes is not None:
            for p in passes:
                if p.spec().name == "apply_noise_model":
                    raise ValueError(
                        "Specify noise via either the submit() "
                        "'noisemodel=' kwarg or via "
                        "an \"apply_noise_model\" pass in passes=, not both."
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
        submit_kwargs.update(prep_knobs)

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
        """Submit ``circuit`` to the service and wait for the results.

        To request specific amplitudes, push ``Amplitude`` instructions into
        ``circuit`` and read ``results.zstates``.

        Args:
            circuit: A circuit or a list of circuits.
            nsamples: Number of shots.
            seed: Integer seed. Mutually exclusive with ``rng``; with
                neither, a fresh seed is drawn.
            rng: A :class:`random.Random` to draw the seed from.
            passes: See the class notes.
            callback: Not used.
            param_grid: Must be empty or ``None``.
            strict_pass_order: See :meth:`Backend.execute`.
            **kwargs: Preparation flags and service options, passed to
                :meth:`submit`: ``fuse``, ``reorderqubits``,
                ``canonicaldecompose``, ``remove_swaps``,
                ``reorderqubits_seed``, ``bonddim``, ``entdim``,
                ``mpscutoff``, ``mpsmethod``, ``mpotraversal``,
                ``timelimit``, ``noisemodel``.

        Returns:
            A :class:`~mimiqcircuits.QCSResults` for a single circuit, or a
            list of them for a list of circuits.

        Raises:
            TypeError: If both ``seed`` and ``rng`` are given, or for an
                unknown keyword.
            ValueError: See :meth:`submit`.
            NotImplementedError: See :meth:`submit`.
            RemotePassOrderError: See ``strict_pass_order``.
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
                        "an \"apply_noise_model\" pass in passes=, not both."
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

    def optimize(
        self,
        experiment,
        initparams=None,
        *,
        return_history: bool = False,
        seed: Optional[int] = None,
        rng=None,
        **kwargs,
    ):
        """Run optimization experiments on the service and wait for the
        results.

        Takes the same experiment arguments as :meth:`LocalBackend.optimize`:
        an :class:`~mimiqcircuits.OptimizationExperiment`, a list of them, or
        a circuit with its initial values and the experiment fields
        (``optimizer``, ``maxiters``, ``zregister``, ``label``) as keywords.
        The service runs the optimizer with the backend's ``algorithm``.

        Args:
            experiment: An :class:`~mimiqcircuits.OptimizationExperiment`, a
                list of them, or a symbolic circuit.
            initparams: Initial values when ``experiment`` is a circuit.
            return_history: Also return every evaluated run.
            seed: Integer seed. Mutually exclusive with ``rng``.
            rng: A :class:`random.Random` to draw the seed from.
            **kwargs: Experiment fields (with a circuit only), and options of
                the connection's ``optimize``: ``nsamples``, ``timelimit``,
                ``bonddim``, ``entdim``, ``mpscutoff``, ``fuse``,
                ``reorderqubits``, ``reorderqubits_seed``,
                ``canonicaldecompose``, ``remove_swaps``, ``mpsmethod``,
                ``mpotraversal``, ``streaming``.

        Returns:
            An :class:`~mimiqcircuits.OptimizationResults`, or a list of them
            for a list of experiments.

        Raises:
            TypeError: If the connection has no ``optimize`` method, or both
                ``seed`` and ``rng`` are given.
            ValueError: For more than one experiment with
                ``algorithm="auto"``; use ``"mps"`` or ``"statevector"``.

        Examples:
            >>> from mimiqcircuits import *
            >>> from mimiqcircuits.backends import MimiqRemoteBackend
            >>> from symengine import symbols
            >>> theta = symbols("theta")
            >>> c = Circuit().push(GateRY(theta), 0)
            >>> c.push(ExpectationValue(GateZ()), 0, 0)  # doctest: +ELLIPSIS
            1-qubit, 1-zvar circuit with 2 instructions:
            ...
            >>> conn = MimiqConnection()
            >>> conn.connect()  # doctest: +SKIP
            >>> backend = MimiqRemoteBackend(conn, algorithm="statevector")
            >>> res = backend.optimize(c, {theta: 0.1}, optimizer="COBYLA")  # doctest: +SKIP
            >>> res.get_best().cost  # doctest: +SKIP
        """
        from mimiqcircuits.backends.optimize import as_experiments

        if not callable(getattr(self._connection, "optimize", None)):
            raise TypeError(
                f"{type(self._connection).__name__} cannot run optimization "
                "jobs: it has no optimize(...) method"
            )
        experiments, single = as_experiments(experiment, initparams, kwargs)
        request_id = self._connection.optimize(
            experiments,
            label=self.label or "pyapi_remote_backend",
            algorithm=self.algorithm,
            history=return_history,
            seed=normalize_seed(None, self._resolve_rngs(seed, rng)),
            **kwargs,
        )
        results = _MimiqJob(
            self._connection, request_id, interval=self.poll_interval
        ).wait()
        return results[0] if single else results

    def __repr__(self) -> str:
        return (
            f"MimiqRemoteBackend({self._connection!r}, "
            f"algorithm={self.algorithm!r})"
        )
