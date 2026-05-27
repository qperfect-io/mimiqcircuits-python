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

"""Conformance harness for :class:`MimiqRemoteBackend`.

Covers the :class:`RemoteBackend` ABC contract against a mock
connection — no network, no live mimiqlink — mirroring the conformance
shape used by the local-simulator backends.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from mimiqcircuits import Circuit, GateCX, GateH, Measure
from mimiqcircuits.backends import (
    AbstractPass,
    AllToAll,
    CAPABILITY_VOCABULARY,
    ExactFidelity,
    Limits,
    MimiqRemoteBackend,
    normalize_seed,
    PassPipeline,
    PassResult,
    PassSpec,
    RemoteBackend,
    RemotePassOrderError,
    RNGs,
    Topology,
    TruncationLowerBound,
    UnknownFidelity,
)
from mimiqcircuits.qcsresults import QCSResults


# ──────────────────────────────────────────────────────────────────────────
# Test fixtures: mock pass + mock connection
# ──────────────────────────────────────────────────────────────────────────


class _NamedPass(AbstractPass):
    """Concrete-pass fake parametrised by ``spec().name`` and an
    optional ``parameters`` dict.

    Built via :py:meth:`PassSpec.from_dict` so the spec carries the
    declared ``tuple[tuple[str, PassParam], ...]`` shape. Stuffing a
    raw dict into ``PassSpec.parameters`` would mask real bugs in
    ``_extract_reorder_method`` and similar consumers.
    """

    def __init__(self, name: str, parameters: Optional[dict] = None):
        self._spec = PassSpec.from_dict(name, parameters)

    def spec(self) -> PassSpec:
        return self._spec

    def apply(self, ctx, c):
        return c, PassResult()


class _FakeMimiqlinkInner:
    """Mock of the inner mimiqlink.MimiqConnection.

    The wrapping ``RemoteConnection`` accesses ``.connection.isJobDone``
    and ``.connection.url`` directly.
    """

    def __init__(self, *, done_after: int = 0, url: str = "https://fake.invalid"):
        self.url = url
        self._done_after = done_after
        self._poll_count = 0

    def isJobDone(self, request_id: str) -> bool:
        self._poll_count += 1
        if self._done_after < 0:
            # Infinite-hang mode for timeout tests.
            return False
        return self._poll_count > self._done_after


class _FakeRemoteConnection:
    """Mock of mimiqcircuits.remote.RemoteConnection.

    Exposes the duck-type contract MimiqRemoteBackend needs:
      - ``submit(circuits, **kwargs) -> request_id: str``
      - ``get_results(request_id, interval=...) -> list[QCSResults]``
      - ``.connection`` attribute (mock of the inner mimiqlink layer)

    ``_counter`` is intentionally an instance attribute — keeping it
    class-level would leak request IDs between parallel pytest-xdist
    workers and rerun-on-failure scenarios.
    """

    def __init__(
        self,
        *,
        simulator: str = "MIMIQ-MPS 0.18.3",
        n_results: int = 1,
        fidelities_per_result: Optional[list[float]] = None,
        done_after: int = 0,
        url: str = "https://fake.invalid",
    ):
        self.connection = _FakeMimiqlinkInner(done_after=done_after, url=url)
        self._simulator = simulator
        self._n_results = n_results
        self._fids = fidelities_per_result or [0.97]

        # Recordings for assertions.
        self.last_submit_args: tuple = ()
        self.last_submit_kwargs: dict = {}
        self.submit_call_count: int = 0
        self.get_results_call_count: int = 0
        self._counter: int = 0

    def submit(self, circuits, **kwargs) -> str:
        self.submit_call_count += 1
        self.last_submit_args = (circuits,)
        self.last_submit_kwargs = dict(kwargs)
        self._counter += 1
        return f"req-{id(self):x}-{self._counter:04d}"

    def get_results(self, request_id: str, interval: float = 1.0):
        self.get_results_call_count += 1
        return [self._make_result() for _ in range(self._n_results)]

    def _make_result(self) -> QCSResults:
        r = QCSResults(
            simulator=self._simulator,
            version="0.0.0+fake",
            fidelities=list(self._fids),
        )
        return r


@pytest.fixture
def fake_conn():
    return _FakeRemoteConnection()


@pytest.fixture
def backend(fake_conn):
    return MimiqRemoteBackend(fake_conn, algorithm="mps", label="test")


def _bell_circuit() -> Circuit:
    c = Circuit()
    c.push(GateH(), 0)
    c.push(GateCX(), 0, 1)
    c.push(Measure(), 0, 0)
    c.push(Measure(), 1, 1)
    return c


# ──────────────────────────────────────────────────────────────────────────
# 1-4: identity, advertisement, topology, capability-shape
# ──────────────────────────────────────────────────────────────────────────


def test_is_remote_backend(backend):
    assert isinstance(backend, RemoteBackend)


def test_name_version_capabilities(backend):
    assert backend.name == "Mimiq"
    assert isinstance(backend.version, str) and backend.version
    caps = backend.capabilities()
    assert isinstance(caps, set) and len(caps) > 0
    assert caps <= CAPABILITY_VOCABULARY, (
        f"Unknown capability tokens: {caps - CAPABILITY_VOCABULARY}"
    )
    # Plan §3 honest non-declarations (positive assertions).
    assert "pass_order_honored" not in caps
    assert "expectation_state" not in caps
    assert "expectation_paulistring" not in caps
    assert "loss" not in caps
    assert "streaming" not in caps


def test_limits_topology(backend, fake_conn):
    # mps algorithm → max_bond_dim populated.
    lim = backend.limits()
    assert isinstance(lim, Limits)
    assert lim.max_bond_dim is not None and lim.max_bond_dim > 0
    # statevector algorithm → max_bond_dim dropped (the concept is
    # meaningless without a tensor-network representation).
    sv = MimiqRemoteBackend(fake_conn, algorithm="statevector")
    assert sv.limits().max_bond_dim is None
    # Topology default.
    assert isinstance(backend.topology(), Topology)
    assert isinstance(backend.topology(), AllToAll)


def test_capabilities_depend_on_algorithm(fake_conn):
    """``:bond_dim`` and ``:schmidt_rank`` must appear only when the
    configured algorithm admits MPS execution.
    """
    mps = MimiqRemoteBackend(fake_conn, algorithm="mps")
    auto = MimiqRemoteBackend(fake_conn, algorithm="auto")
    sv = MimiqRemoteBackend(fake_conn, algorithm="statevector")
    for b in (mps, auto):
        assert "bond_dim" in b.capabilities()
        assert "schmidt_rank" in b.capabilities()
    assert "bond_dim" not in sv.capabilities()
    assert "schmidt_rank" not in sv.capabilities()


# ──────────────────────────────────────────────────────────────────────────
# 5-7: submit / execute happy paths
# ──────────────────────────────────────────────────────────────────────────


def test_submit_returns_job_with_wait(backend):
    job = backend.submit(_bell_circuit(), nsamples=10)
    assert hasattr(job, "wait") and callable(job.wait)
    results = job.wait()
    assert isinstance(results, list)
    assert all(isinstance(r, QCSResults) for r in results)


def test_execute_roundtrip_single_circuit(backend, fake_conn):
    r = backend.execute(_bell_circuit(), nsamples=10)
    # Shape mirroring: single Circuit → single QCSResults (not list).
    assert isinstance(r, QCSResults)
    assert not isinstance(r, list)
    assert fake_conn.submit_call_count == 1


def test_execute_roundtrip_batch_circuit(fake_conn):
    fake_conn._n_results = 3
    backend = MimiqRemoteBackend(fake_conn, algorithm="mps")
    circuits = [_bell_circuit() for _ in range(3)]
    results = backend.execute(circuits, nsamples=10)
    assert isinstance(results, list)
    assert len(results) == len(circuits)
    assert all(isinstance(r, QCSResults) for r in results)


# ──────────────────────────────────────────────────────────────────────────
# 8: constructor + submit-time knobs flow through to the connection
# ──────────────────────────────────────────────────────────────────────────


def test_submit_forwards_constructor_knobs(fake_conn):
    """Every backend knob must reach the connection's submit call —
    a silent drop would let the server pick its default while the
    user thought they had configured the run.
    """
    backend = MimiqRemoteBackend(
        fake_conn,
        algorithm="mps",
        label="my-label",
    )
    backend.execute(
        _bell_circuit(),
        nsamples=42,
        seed=12345,
        bonddim=64,
        entdim=16,
        mpscutoff=1e-8,
        mpsmethod="vmpoa",
        mpotraversal="bfs",
        timelimit=15,
        noisemodel=None,
    )
    k = fake_conn.last_submit_kwargs
    assert k["algorithm"] == "mps"
    assert k["label"] == "my-label"
    assert k["nsamples"] == 42
    assert k["bonddim"] == 64
    assert k["entdim"] == 16
    assert k["mpscutoff"] == 1e-8
    assert k["mpsmethod"] == "vmpoa"
    assert k["mpotraversal"] == "bfs"
    assert k["timelimit"] == 15
    assert k["noisemodel"] is None
    # The wire seed is the xor-fold of `RNGs.from_seed(12345)`. Both
    # the derivation and the fold are pure, so a silent regression in
    # either trips this exact-equality check.
    expected_seed = normalize_seed(None, RNGs.from_seed(12345))
    assert k["seed"] == expected_seed


# ──────────────────────────────────────────────────────────────────────────
# 9-10: pass-pipeline strict / lax modes
# ──────────────────────────────────────────────────────────────────────────


def test_execute_passes_strict_order_raises(backend):
    """Default strict_pass_order=True with non-empty passes raises
    RemotePassOrderError (server lacks :pass_order_honored)."""
    pipe = PassPipeline(passes=[_NamedPass("remove_swaps")])
    with pytest.raises(RemotePassOrderError):
        backend.execute(_bell_circuit(), nsamples=10, passes=pipe)


def test_execute_passes_lax_warns_and_translates(backend, fake_conn):
    """Lax mode translates recognised passes into legacy knobs,
    drops unrecognised passes, and emits exactly one ``UserWarning``
    per ``submit()`` call. One warning per pass would spam users that
    hit the same translation miss on every call.
    """
    pipe = PassPipeline(passes=[
        _NamedPass("remove_swaps"),
        _NamedPass("fuse_gates"),
        _NamedPass("unknown_xyz"),
        _NamedPass("also_unknown"),
        _NamedPass("reorder_qubits"),
    ])
    with pytest.warns(UserWarning) as record:
        backend.execute(_bell_circuit(), nsamples=10, passes=pipe, strict_pass_order=False)
    # Exactly one warning for the whole submit() call.
    assert len(record) == 1
    msg = str(record[0].message)
    assert "unknown_xyz" in msg
    assert "also_unknown" in msg

    k = fake_conn.last_submit_kwargs
    # Recognised passes ARE forwarded.
    assert k["remove_swaps"] is True
    assert k["fuse"] is True
    assert k["reorderqubits"] is True  # no `method` param → True default
    # Unrecognised passes are NOT forwarded under any name.
    for forbidden in ("unknown_xyz", "also_unknown"):
        assert forbidden not in k
        assert forbidden not in str(k)


# ──────────────────────────────────────────────────────────────────────────
# 11: seed resolution
# ──────────────────────────────────────────────────────────────────────────


def test_seed_resolution_in_submit(fake_conn):
    """``seed=`` must reach the fake connection as a single int after
    `normalize_seed`. The xor-fold of the four streams is deterministic,
    so identically-seeded inputs produce identical wire seeds.
    """
    backend = MimiqRemoteBackend(fake_conn, algorithm="mps")

    backend.execute(_bell_circuit(), nsamples=10, seed=42)
    s_a = fake_conn.last_submit_kwargs["seed"]
    backend.execute(_bell_circuit(), nsamples=10, seed=42)
    s_b = fake_conn.last_submit_kwargs["seed"]
    assert isinstance(s_a, int)
    assert s_a == s_b, "seed=42 fold is non-deterministic"

    backend.execute(_bell_circuit(), nsamples=10, seed=7)
    s2 = fake_conn.last_submit_kwargs["seed"]
    backend.execute(_bell_circuit(), nsamples=10, seed=7)
    s3 = fake_conn.last_submit_kwargs["seed"]
    assert s2 == s3


# ──────────────────────────────────────────────────────────────────────────
# 12-13: error guards
# ──────────────────────────────────────────────────────────────────────────


def test_seed_and_rng_mutually_exclusive(backend):
    """`seed=` and `rng=` together must raise — passing both is
    ambiguous, and silently picking one would defeat reproducibility.
    """
    import random
    with pytest.raises(TypeError, match="mutually exclusive"):
        backend.execute(
            _bell_circuit(), nsamples=10, seed=42, rng=random.Random(0),
        )


def test_noisemodel_and_apply_noise_pass_conflict_raises(backend):
    """`noisemodel=` together with an `ApplyNoiseModelPass` is
    ambiguous (two different noise sources); the backend must reject
    rather than silently pick one.
    """
    pipe = PassPipeline(passes=[_NamedPass("apply_noise_model")])
    sentinel = object()  # treat as opaque noisemodel
    with pytest.raises(ValueError, match="noisemodel"):
        backend.execute(
            _bell_circuit(), nsamples=10,
            passes=pipe,
            strict_pass_order=False,
            noisemodel=sentinel,
        )


def test_noisemodel_collision_beats_strict_pass_order(backend):
    """With both `noisemodel=` and `ApplyNoiseModelPass` and the
    default `strict_pass_order=True`, the duplication ValueError must
    fire first — the user's mistake is double-noise, not ordering.
    """
    pipe = PassPipeline(passes=[_NamedPass("apply_noise_model")])
    sentinel = object()
    with pytest.raises(ValueError, match="noisemodel"):
        backend.execute(
            _bell_circuit(), nsamples=10,
            passes=pipe,
            noisemodel=sentinel,
        )


# ──────────────────────────────────────────────────────────────────────────
# 14: param_grid raises BEFORE touching the network
# ──────────────────────────────────────────────────────────────────────────


def test_param_grid_raises_before_network(backend, fake_conn):
    """`param_grid=` must raise NotImplementedError before any
    network round-trip — silently submitting an unsupported feature
    would charge the user for a job that cannot succeed.
    """

    class _NoCallConn:
        connection = _FakeMimiqlinkInner()

        def submit(self, *a, **kw):
            raise AssertionError("submit() should not be reached")

        def get_results(self, *a, **kw):
            raise AssertionError("get_results() should not be reached")

    b = MimiqRemoteBackend(_NoCallConn(), algorithm="mps")
    with pytest.raises(NotImplementedError, match="param_grid"):
        b.execute(_bell_circuit(), nsamples=10, param_grid=[{}, {}])


# ──────────────────────────────────────────────────────────────────────────
# 15: idempotent wait()
# ──────────────────────────────────────────────────────────────────────────


def test_job_wait_idempotent(backend, fake_conn):
    """`wait()` must be safe to call twice on the same job: the
    server retains results until `deleteFiles`, and the second wait
    must re-poll and re-download. Value-equality is asserted (not
    just type) so a regression that drops the underlying float
    between calls trips the test.
    """
    job = backend.submit(_bell_circuit(), 10)
    r1 = job.wait()
    r2 = job.wait()
    assert isinstance(r1, list) and isinstance(r2, list)
    assert len(r1) == len(r2) == 1
    # Same simulator string → same wrap type.
    f1 = r1[0].fidelities[0]
    f2 = r2[0].fidelities[0]
    assert type(f1) is type(f2)  # noqa: E721
    # Same wrapped value (TruncationLowerBound carries `.value`).
    assert getattr(f1, "value", None) == getattr(f2, "value", None)
    # And the fake recorded two get_results invocations.
    assert fake_conn.get_results_call_count == 2


# ──────────────────────────────────────────────────────────────────────────
# Typed-Fidelity wrap on remote results
# ──────────────────────────────────────────────────────────────────────────


def test_fidelity_typed_on_remote_results():
    """The server returns raw floats; `wait()` must re-type them into
    Fidelity subclasses based on the simulator identity string so
    callers see the same ADT they would from a local backend.
    """
    # MPS simulator → TruncationLowerBound.
    mps_conn = _FakeRemoteConnection(
        simulator="MIMIQ-MPS 0.18.3", fidelities_per_result=[0.97]
    )
    b = MimiqRemoteBackend(mps_conn, algorithm="mps")
    r = b.execute(_bell_circuit(), nsamples=10)
    assert len(r.fidelities) == 1
    assert isinstance(r.fidelities[0], TruncationLowerBound)
    assert r.fidelities[0].value == 0.97

    # StateVector simulator → ExactFidelity (value discarded).
    sv_conn = _FakeRemoteConnection(
        simulator="MIMIQ-StateVector 1.2.0", fidelities_per_result=[1.0]
    )
    b2 = MimiqRemoteBackend(sv_conn, algorithm="statevector")
    r2 = b2.execute(_bell_circuit(), nsamples=10)
    assert isinstance(r2.fidelities[0], ExactFidelity)

    # Unknown / empty simulator string → UnknownFidelity.
    unknown_conn = _FakeRemoteConnection(
        simulator="", fidelities_per_result=[0.5]
    )
    b3 = MimiqRemoteBackend(unknown_conn, algorithm="auto")
    r3 = b3.execute(_bell_circuit(), nsamples=10)
    assert isinstance(r3.fidelities[0], UnknownFidelity)


# ──────────────────────────────────────────────────────────────────────────
# 17: wait(timeout=...)
# ──────────────────────────────────────────────────────────────────────────


def test_wait_timeout(fake_conn):
    """`wait(timeout=...)` must raise `TimeoutError` rather than hang
    when the server never reports completion.
    """
    # done_after=-1 → infinite hang.
    hung_conn = _FakeRemoteConnection(done_after=-1)
    backend = MimiqRemoteBackend(
        hung_conn, algorithm="mps", poll_interval=0.01
    )
    job = backend.submit(_bell_circuit(), 10)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="did not complete"):
        job.wait(timeout=0.1)
    elapsed = time.monotonic() - t0
    # Bound: at least the timeout, but not so much that we wedged.
    assert 0.05 < elapsed < 1.0


# ──────────────────────────────────────────────────────────────────────────
# Extra guard: duck-type rejection at __init__
# ──────────────────────────────────────────────────────────────────────────


def test_constructor_rejects_non_connection():
    """Any object missing `submit()` / `get_results()` must be
    rejected at construction time with a hint, not at the first
    network call.
    """

    class _NotAConnection:
        pass

    with pytest.raises(TypeError, match=r"submit"):
        MimiqRemoteBackend(_NotAConnection())


def test_reorder_qubits_method_param_propagates(fake_conn):
    """A `reorder_qubits` pass with a `method` parameter must reach
    the server as `reorderqubits="sa"`, not the default `True`.

    Exercises the PStr-unwrap branch of `_extract_reorder_method`.
    The fixture must build the spec through `PassSpec.from_dict` so
    that `parameters` carries the declared tuple-of-pairs shape;
    stuffing a raw dict directly into `PassSpec.parameters` would
    mask real bugs here.
    """
    backend = MimiqRemoteBackend(fake_conn, algorithm="mps")
    pipe = PassPipeline(passes=[
        _NamedPass("reorder_qubits", parameters={"method": "sa"}),
    ])
    backend.execute(
        _bell_circuit(), nsamples=10, passes=pipe, strict_pass_order=False,
    )
    k = fake_conn.last_submit_kwargs
    assert k["reorderqubits"] == "sa", (
        f"expected reorderqubits='sa', got {k.get('reorderqubits')!r}"
    )


def test_fidelity_wrap_is_idempotent():
    """`_wrap_fidelity_by_simulator` must short-circuit when its
    input is already a :class:`Fidelity` subclass.

    Today the production `RemoteConnection.get_results` re-allocates
    QCSResults from the wire on each call, so the contract is
    trivially satisfied. A future server-side result cache could
    silently break this assumption if `wait()` were not idempotent.
    """
    from mimiqcircuits.backends.remote import _wrap_fidelity_by_simulator

    # First wrap: float → TruncationLowerBound.
    a = _wrap_fidelity_by_simulator("MIMIQ-MPS 0.18", 0.95)
    assert isinstance(a, TruncationLowerBound)
    assert a.value == 0.95

    # Second wrap on the already-wrapped value: returned as-is.
    b = _wrap_fidelity_by_simulator("MIMIQ-MPS 0.18", a)
    assert b is a, "wrap should be idempotent on a Fidelity input"

    # Same for ExactFidelity.
    e = _wrap_fidelity_by_simulator("MIMIQ-StateVector 1.0", 1.0)
    assert isinstance(e, ExactFidelity)
    e2 = _wrap_fidelity_by_simulator("MIMIQ-StateVector 1.0", e)
    assert e2 is e


# ──────────────────────────────────────────────────────────────────────────
# Anti-capability harness
# ──────────────────────────────────────────────────────────────────────────


def test_can_handle_rejects_anticap_remote_mps(backend):
    """For every probe in CAP_PROBES that the MPS-algorithm
    MimiqRemoteBackend does not declare, the backend must reject
    via can_handle or raise during execute. Silent acceptance is a
    positive-lie capability.

    `:noise` is declared by every algorithm, so the noise probe is
    expected to flow without rejection (handled by skip_caps in the
    helper if it ever needs special treatment).
    """
    from mimiqcircuits.backends._anticap_helper import assert_anticap_rejected

    assert_anticap_rejected(backend)


def test_can_handle_rejects_anticap_remote_statevector(fake_conn):
    """Same probe family against the statevector algorithm — caps
    differ from MPS (no :bond_dim / :schmidt_rank), so the
    undeclared set is different and the test exercises a different
    rejection profile."""
    from mimiqcircuits.backends._anticap_helper import assert_anticap_rejected

    sv_backend = MimiqRemoteBackend(fake_conn, algorithm="statevector")
    assert_anticap_rejected(sv_backend)
