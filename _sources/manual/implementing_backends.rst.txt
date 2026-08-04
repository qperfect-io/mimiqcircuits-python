Implementing a Custom Backend
=============================

The :mod:`mimiqcircuits.backends` module defines a small, explicit
contract that every simulator must satisfy. Writing a new backend
means declaring what your simulator can do, providing a handful of
primitives, and (optionally) overriding the high-level
:meth:`~mimiqcircuits.backends.Backend.execute` loop for performance.

This guide walks through the contract end-to-end and shows complete
working examples for both in-process simulators
(:class:`~mimiqcircuits.backends.LocalBackend`) and remote services
(:class:`~mimiqcircuits.backends.RemoteBackend`).

.. contents::
    :local:
    :depth: 2

When to subclass which base
---------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Base class
      - Use when
    * - :class:`~mimiqcircuits.backends.LocalBackend`
      - The simulator runs in the same Python process and exposes a
        state object you can mutate instruction-by-instruction.
    * - :class:`~mimiqcircuits.backends.RemoteBackend`
      - The simulator runs elsewhere (cloud service, queued
        executor) and you submit a request and poll for results.
    * - :class:`~mimiqcircuits.backends.Backend`
      - You need full control of :meth:`execute` and neither of
        the higher-level helpers fits — rare.

Every backend must implement the small set of methods on
:class:`~mimiqcircuits.backends.Backend`:

.. list-table::
    :header-rows: 1

    * - Member
      - Returns
      - Default
    * - ``name`` (property)
      - ``str`` identifying the simulator
      - abstract
    * - ``version`` (property)
      - ``str`` simulator version
      - abstract
    * - :meth:`capabilities`
      - ``set[str]`` of feature tokens (see below)
      - abstract
    * - :meth:`limits`
      - :class:`~mimiqcircuits.backends.Limits`
      - all-``None``
    * - :meth:`topology`
      - :class:`~mimiqcircuits.backends.Topology`
      - :class:`~mimiqcircuits.backends.AllToAll`
    * - :meth:`can_handle`
      - :class:`~mimiqcircuits.backends.AdmissionResult` —
        one of :class:`~mimiqcircuits.backends.Admissible`,
        :class:`~mimiqcircuits.backends.Marginal` (admissible but
        near a resource limit; carries a user-facing warning), or
        :class:`~mimiqcircuits.backends.Inadmissible` (carries a
        rejection reason).
      - always :class:`~mimiqcircuits.backends.Admissible`
    * - :meth:`execute`
      - :class:`~mimiqcircuits.QCSResults`
      - inherited from base subclass

The full contract — including capability tokens, the Fidelity ADT,
and pass-pipeline plumbing — is described in
:ref:`backend-checklist` at the end of this page.

Declaring capabilities honestly
-------------------------------

A backend's :meth:`capabilities` method returns a set of *capability
tokens* drawn from
:data:`~mimiqcircuits.backends.CAPABILITY_VOCABULARY`. Each token is
a positive claim: "this backend can execute circuits that exercise
this feature".

**Do not advertise a capability your simulator cannot actually
deliver.** The conformance suite verifies that every declared
capability has a working code path, and every undeclared capability
is rejected — silent degradation lets bugs slip through.

A representative subset of the vocabulary:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Token
      - Meaning
    * - ``"amplitude"``
      - The backend can compute amplitudes ``⟨bs|ψ⟩``.
    * - ``"sampling"``
      - The backend can sample bitstrings from the final state.
    * - ``"midcircuit_measure"``
      - Measurements may appear mid-circuit, not only at the end.
    * - ``"midcircuit_reset"``
      - ``Reset`` may appear mid-circuit.
    * - ``"feed_forward"``
      - ``IfStatement`` (conditioning on classical bits) is supported.
    * - ``"noise"``
      - Trace-preserving Kraus channels are supported.
    * - ``"loss"``
      - Non-trace-preserving channels (e.g. amplitude damping with
        explicit loss) are supported.
    * - ``"expectation_1q"`` / ``"expectation_2q"``
      - ``ExpectationValue`` is supported for 1- and 2-qubit
        operators.
    * - ``"expectation_paulistring"``
      - ``ExpectationValue`` is supported for Pauli strings on
        more than 2 qubits.
    * - ``"expectation_state"``
      - The backend implements
        :meth:`~mimiqcircuits.backends.Backend.expectation` for
        computing ``⟨ψ|op|ψ⟩`` directly on a built state, outside
        of an evolving circuit.
    * - ``"bond_dim"`` / ``"schmidt_rank"``
      - Tensor-network annotations are supported (MPS-class
        backends).
    * - ``"streaming"``
      - The circuit is compressed lazily; peak memory is bounded
        even for very long circuits.
    * - ``"parametric"``
      - Compile accepts circuits with free symbolic parameters
        (``bind`` resolves them).

A simulator that renormalises after every Kraus step (state-vector
sims do this) should *not* declare ``"loss"``: the surviving
trajectory probability is silently discarded and the user would
get a fidelity of ``1.0`` for a circuit that actually lost
amplitude. Declaring only ``"noise"`` is the honest choice.

The Fidelity ADT
----------------

:meth:`~mimiqcircuits.backends.LocalBackend.evolve` must return a
typed :class:`~mimiqcircuits.backends.Fidelity`, not a plain float.
The variants record *what the number means*:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Variant
      - When to return it
    * - :class:`~mimiqcircuits.backends.ExactFidelity`
      - Your simulator is exact under its own algorithm (state-vector
        without lossy noise).
    * - :class:`~mimiqcircuits.backends.UnknownFidelity`
      - You genuinely do not track fidelity. Better than inventing
        a placeholder.
    * - :class:`~mimiqcircuits.backends.TruncationLowerBound`
      - Single scalar lower bound on
        ``|⟨ψ_exact|ψ_sim⟩|²``. The standard MPS choice.
    * - :class:`~mimiqcircuits.backends.LowerBoundPerStep`
      - Per-step contributions. The product is a lower bound only if
        successive truncation errors are independent; collapse to
        :class:`~mimiqcircuits.backends.TruncationLowerBound`
        otherwise.
    * - :class:`~mimiqcircuits.backends.EstimatedFidelity`
      - Sample-based estimate with a standard error (randomised
        benchmarking, direct fidelity estimation, …).

Common trap — the ``1.0`` collision:

.. warning::

    A truncation-lower-bound that happens to land at exactly
    ``1.0`` (small circuit, fits in the bond budget) must still be
    wrapped as
    :class:`~mimiqcircuits.backends.TruncationLowerBound`, not
    :class:`~mimiqcircuits.backends.ExactFidelity`. The two carry
    different semantics; do not route through
    ``_to_fidelity(1.0)``, which collapses to
    :class:`~mimiqcircuits.backends.ExactFidelity`.

A worked example: writing a LocalBackend
----------------------------------------

The simplest possible custom backend wraps a Python-side simulator
function. The example below shows every required piece.

.. code-block:: python

    from typing import Optional
    import random

    from mimiqcircuits import Circuit, QCSResults, BitString
    from mimiqcircuits.backends import (
        AllToAll,
        Capability,
        CompileMetadata,
        CompiledCircuit,
        DefaultCompiledCircuit,
        ExactFidelity,
        Fidelity,
        Limits,
        LocalBackend,
        State,
        Topology,
    )


    class _ToyState(State):
        """Minimal `State` for a 1-shot, sample-only simulator."""

        def __init__(self, nq: int, nb: int, nz: int):
            self._nq = nq
            self.c = [0] * nb
            self.z = [complex(0)] * nz

        @property
        def num_qubits(self) -> int:
            return self._nq

        @property
        def num_bits(self) -> int:
            return len(self.c)

        @property
        def num_zvars(self) -> int:
            return len(self.z)

        def amplitude(self, bs) -> complex:
            # Trivial uniform distribution → all amplitudes equal.
            return complex(1.0 / (2 ** self._nq) ** 0.5)

        def sample(self, nsamples: int,
                   rng: Optional[random.Random] = None, *,
                   seed: Optional[int] = None) -> list:
            if rng is not None and seed is not None:
                raise TypeError("pass either rng= or seed=, not both")
            if rng is None:
                rng = random.Random(seed)  # `seed=None` → fresh entropy.
            return [
                BitString([rng.randint(0, 1) for _ in range(self._nq)])
                for _ in range(nsamples)
            ]

        @property
        def classical_bits(self):
            return self.c

        @property
        def complex_values(self):
            return self.z


    _TOY_CAPS: frozenset[str] = frozenset({"sampling", "classical_bits"})


    class ToySimulator(LocalBackend):
        """Sample-only simulator: returns uniformly random bitstrings.

        Useful as a control in tests and as a template for real backends.
        """

        @property
        def name(self) -> str:
            return "Toy"

        @property
        def version(self) -> str:
            return "0.1.0"

        def capabilities(self) -> set[Capability]:
            return set(_TOY_CAPS)

        def limits(self) -> Limits:
            return Limits()

        def topology(self) -> Topology:
            return AllToAll()

        def build_state(self, nq: int, nb: int = 0, nz: int = 0,
                        **kwargs) -> _ToyState:
            return _ToyState(nq, nb, nz)

        def compile(self, circuit: Circuit) -> CompiledCircuit:
            meta = CompileMetadata(
                active_qubits=list(range(circuit.num_qubits()))
            )
            return DefaultCompiledCircuit(_source=circuit, _metadata=meta)

        def evolve(self, state, compiled, *,
                   rng=None, callback=None, stopped=None
                   ) -> tuple[State, Fidelity]:
            # Toy simulator: no real evolution.
            return state, ExactFidelity()

The four blocks above — identity, advertisement, state construction,
and the ``compile``/``evolve`` pair — are everything
:class:`~mimiqcircuits.backends.LocalBackend` needs to drive its
default :meth:`execute` loop:

1. Run the user-supplied pass pipeline over the circuit.
2. Call :meth:`compile` once.
3. Call :meth:`prepare_trajectory` per trajectory (default
   identity).
4. Allocate a fresh state via :meth:`build_state`.
5. Call :meth:`evolve` to mutate the state.
6. Sample ``nsamples`` bitstrings via the state's ``sample``.
7. Wrap everything in a :class:`~mimiqcircuits.QCSResults`.

Override :meth:`execute` directly only when you need richer outputs
(per-shot amplitudes, expectation values, multi-circuit batching),
or when the per-shot cost is so low that the default loop's
overhead matters.

Compile must be pure
~~~~~~~~~~~~~~~~~~~~

:meth:`~mimiqcircuits.backends.LocalBackend.compile` must not
consume any RNG and must not sample anything. Round-tripping the
same ``(backend, circuit)`` must produce equivalent compiled
artifacts. This is what lets generic drivers re-use the compiled
output across many trajectories or many parameter points.

If your simulator needs to sample noise (mixed-unitary,
trace-preserving Kraus) or build a stochastic lossy suffix, put
that work in :meth:`prepare_trajectory`, which is called once per
trajectory with a dedicated RNG.

Optional: a parametric fast path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Declare the ``"parametric"`` capability if you can compile circuits
that still carry free symbolic parameters. The default
:meth:`~mimiqcircuits.backends.LocalBackend.bind` substitutes
parameters and re-runs :meth:`compile`; override when your backend
can re-bind a pre-compiled artifact (slot maps, pre-baked gate
templates) without paying the full compile cost again.

A worked example: writing a RemoteBackend
-----------------------------------------

For a service that exposes a submit / poll API, subclass
:class:`~mimiqcircuits.backends.RemoteBackend`:

.. code-block:: python

    from mimiqcircuits.backends import (
        AllToAll,
        Limits,
        RemoteBackend,
        Topology,
    )


    _REMOTE_CAPS = frozenset({
        "amplitude", "sampling", "classical_bits",
        "midcircuit_measure", "feed_forward", "noise",
    })


    class _JobHandle:
        """Minimal job handle.

        Must expose ``wait(timeout=None)`` returning a
        :class:`~mimiqcircuits.QCSResults` (or a list thereof). The
        ``timeout`` kwarg is part of the production contract — raise
        :class:`TimeoutError` rather than hang on a stuck server.
        """

        def __init__(self, conn, request_id):
            self._conn = conn
            self._request_id = request_id

        def wait(self, *, timeout=None):
            return self._conn.get_results(
                self._request_id, timeout=timeout
            )


    class MyCloudBackend(RemoteBackend):
        """Cloud backend wrapping a generic submit/poll connection."""

        def __init__(self, connection, *, algorithm="auto"):
            self._connection = connection
            self.algorithm = algorithm

        @property
        def name(self) -> str:
            return "MyCloud"

        @property
        def version(self) -> str:
            return "0.1.0"

        def capabilities(self) -> set[str]:
            return set(_REMOTE_CAPS)

        def limits(self) -> Limits:
            return Limits(max_samples=1_000_000)

        def topology(self) -> Topology:
            return AllToAll()

        def submit(self, circuits, nsamples=1000, **kwargs):
            request_id = self._connection.submit(
                circuits,
                algorithm=self.algorithm,
                nsamples=nsamples,
                **kwargs,
            )
            return _JobHandle(self._connection, request_id)

The inherited :meth:`execute` calls :meth:`submit` and then
``job.wait()``. Override :meth:`execute` only when you need
extra steps before or after the round-trip (re-typing fidelities,
shape-mirroring single circuit vs. list inputs, etc.).

Three things to watch for in a remote wrapper:

1. **Capability honesty.** Synthesise the capability set from
   server features you can actually deliver through the wire
   format. If the server takes a flat bag of booleans instead of an
   ordered pipeline, you cannot honestly advertise
   ``"pass_order_honored"`` — and the framework will raise
   :class:`~mimiqcircuits.backends.RemotePassOrderError` for users
   who try.
2. **Typed fidelities on the wire.** Servers usually return raw
   floats. Re-wrap each result's ``fidelities`` into the
   appropriate :class:`~mimiqcircuits.backends.Fidelity` subclass
   based on the simulator identity string, so downstream code sees
   the same ADT it would from a local backend.
3. **``seed=`` / ``rng=``.** Accept both, mutually exclusive — the
   inherited :meth:`Backend._resolve_rngs` handles the validation.

Working with passes
-------------------

Backends interact with the pass pipeline through three optional
methods:

- :meth:`~mimiqcircuits.backends.Backend.accepts_pass` — return
  ``False`` to reject a pass the backend cannot run. The pipeline
  will raise
  :class:`~mimiqcircuits.backends.UnacceptedPassError` rather than
  silently dropping the pass.
- :meth:`~mimiqcircuits.backends.Backend.delegates_pass` — return
  ``True`` if your backend implements the pass natively inside
  :meth:`compile` or :meth:`evolve`. The pipeline records a marker
  result but does not run the pass.
- :meth:`~mimiqcircuits.backends.Backend.default_passes` — return
  the pipeline the backend wants applied when the caller passes
  ``passes=None``.

Custom passes subclass
:class:`~mimiqcircuits.backends.AbstractPass`. A minimal example:

.. code-block:: python

    from mimiqcircuits import Circuit
    from mimiqcircuits.backends import (
        AbstractPass,
        PassContext,
        PassResult,
        PassSpec,
    )


    class StripBarriersPass(AbstractPass):
        """Drop every Barrier from the circuit."""

        def spec(self) -> PassSpec:
            return PassSpec("strip_barriers")

        def apply(self, ctx: PassContext, circuit
                  ) -> tuple[Circuit, PassResult]:
            from mimiqcircuits import Barrier
            new_c = Circuit()
            for inst in circuit.instructions:
                if not isinstance(inst.operation, Barrier):
                    new_c.push(inst.operation, *inst.qubits,
                               *inst.bits, *inst.zvars)
            return new_c, PassResult()

A pass that *renames* qubits must return the relabel as
``PassResult.qubit_permutation`` so the pipeline can compose
permutations and unscramble downstream samples and observables.

.. _backend-checklist:

Conformance checklist
---------------------

Before shipping a new backend, walk through the following list.
Each item corresponds to an assertion the conformance suite makes;
failing one means the backend is "lying" in some way the framework
can detect.

**Identity and advertisement**

#. :attr:`name` returns a non-empty string.
#. :attr:`version` returns a parseable version string.
#. :meth:`capabilities` returns a :class:`set` of strings; every
   entry is in
   :data:`~mimiqcircuits.backends.CAPABILITY_VOCABULARY` (extras
   are allowed but trigger a warning).
#. :meth:`limits` returns a :class:`~mimiqcircuits.backends.Limits`
   instance. By convention each numeric field is either ``None``
   (no advertised bound) or strictly positive.
#. :meth:`topology` returns one of
   :class:`~mimiqcircuits.backends.AllToAll`,
   :class:`~mimiqcircuits.backends.CouplingMap`,
   :class:`~mimiqcircuits.backends.LinearChain`.

**Admission**

#. :meth:`can_handle` returns
   :class:`~mimiqcircuits.backends.Admissible` (or
   :class:`~mimiqcircuits.backends.Marginal`, which
   :func:`~mimiqcircuits.backends.is_admissible` also treats as
   accepted) for a small circuit the backend can actually run.
#. :meth:`can_handle` returns
   :class:`~mimiqcircuits.backends.Inadmissible` (or :meth:`execute`
   raises) for every capability the backend does *not* advertise.

**LocalBackend primitives**

#. :meth:`build_state` returns a
   :class:`~mimiqcircuits.backends.State`-derived object with the
   requested register sizes.
#. :meth:`compile` is pure — no RNG, no sampling. Two consecutive
   calls on the same input produce equivalent artifacts.
#. :meth:`evolve` returns a tuple ``(state, fidelity)`` where
   ``fidelity`` is a :class:`~mimiqcircuits.backends.Fidelity`
   subclass.
#. A truncation-lower-bound of exactly ``1.0`` is wrapped as
   :class:`~mimiqcircuits.backends.TruncationLowerBound`, not
   :class:`~mimiqcircuits.backends.ExactFidelity`.
#. If the backend declares ``"parametric"``,
   :meth:`compile` accepts a symbolic circuit and :meth:`bind`
   resolves the parameters to a fully bound
   :class:`~mimiqcircuits.backends.CompiledCircuit`.

**RemoteBackend primitives**

#. :meth:`submit` returns a job handle with a working ``wait()``.
#. Results' fidelities are re-typed into
   :class:`~mimiqcircuits.backends.Fidelity` subclasses based on
   the simulator identity string.
#. If the backend does not advertise ``"pass_order_honored"``, a
   non-empty pipeline with default ``strict_pass_order=True`` must
   raise :class:`~mimiqcircuits.backends.RemotePassOrderError`.

**Cross-cutting**

#. The backend does not advertise any capability it cannot deliver.
#. Two ``execute`` calls with the same ``seed=`` produce identical
   results (within the algorithm's own numerical tolerance).
#. Passing both ``seed=`` and ``rng=`` simultaneously raises
   :class:`TypeError` — the two are mutually exclusive.
#. A non-empty ``param_grid=`` iterates the circuit through Python
   substitution and runs ``execute`` per parameter point with a
   distinct derived seed (see
   :func:`~mimiqcircuits.backends.derive_grid_seeds`).

See also
--------

- :doc:`/apidocs/mimiqcircuits` for the full API reference of the
  :mod:`mimiqcircuits.backends` module.
- :doc:`simulation` for using the backends as a consumer rather
  than implementing one.
- :doc:`compilation` for the built-in pass set.
