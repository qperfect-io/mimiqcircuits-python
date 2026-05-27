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

"""Abstract base classes for simulator backends.

This module defines the contract every simulator must satisfy:

- :class:`Backend` — the common interface, advertising identity,
  capabilities, limits, topology, and an :meth:`~Backend.execute`
  entry point.
- :class:`LocalBackend` — for in-process simulators. Adds
  :meth:`~LocalBackend.build_state`, :meth:`~LocalBackend.compile`,
  :meth:`~LocalBackend.evolve`, and a default :meth:`~Backend.execute`
  loop built on top of them.
- :class:`RemoteBackend` — for cloud or submit/poll execution. Adds
  :meth:`~RemoteBackend.submit` and provides a default
  :meth:`~Backend.execute` that waits on the returned job.

See :doc:`/manual/implementing_backends` for a step-by-step guide.
"""

from __future__ import annotations

import abc
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from mimiqcircuits.backends.capabilities import (
    AdmissionResult,
    Admissible,
    Capability,
    Inadmissible,
    Limits,
    Topology,
    AllToAll,
)
from mimiqcircuits.backends.compiled import (
    CompiledCircuit,
    CompiledParametricCircuit,
    DefaultCompiledCircuit,
    CompileMetadata,
)
from mimiqcircuits.backends.fidelity import (
    Fidelity, _to_fidelity, as_lower_bound,
)
from mimiqcircuits.backends.passes import (
    AbstractPass,
    PassContext,
    PassPipeline,
    apply_passes,
    RemotePassOrderError,
)


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers for the default `can_handle`
# ──────────────────────────────────────────────────────────────────────────


def _circuit_count(circuit, attr_name: str) -> int:
    """Read a register size from a circuit.

    Accepts both method-style (``circuit.num_qubits()``) and
    attribute-style accessors so remote or duck-typed circuit
    handles work too. Returns ``0`` if the attribute is missing.
    """
    accessor = getattr(circuit, attr_name, None)
    if accessor is None:
        return 0
    value = accessor() if callable(accessor) else accessor
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _circuit_has_loss(circuit) -> bool:
    """Return ``True`` if ``circuit`` contains any loss-bearing
    operation (``LossErr`` / ``QubitLoss``)."""
    instructions = getattr(circuit, "instructions", None)
    if instructions is None:
        return False
    try:
        from mimiqcircuits.operations.losschannel import LossErr, QubitLoss
    except ImportError:  # pragma: no cover - defensive
        return False
    for inst in instructions:
        op = getattr(inst, "operation", None)
        if isinstance(op, (LossErr, QubitLoss)):
            return True
    return False


def _circuit_has_kraus(circuit) -> bool:
    """Return ``True`` if ``circuit`` contains any Kraus channel,
    including channels wrapped inside an ``IfStatement`` or
    ``WhileStatement`` body (matches Julia's `_op_has_kraus` recursion)."""
    instructions = getattr(circuit, "instructions", None)
    if instructions is None:
        return False
    try:
        from mimiqcircuits.operations.krauschannel import krauschannel
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.whilestatement import WhileStatement
    except ImportError:  # pragma: no cover - defensive
        return False

    def _op_has_kraus(op) -> bool:
        if isinstance(op, krauschannel):
            return True
        if isinstance(op, (IfStatement, WhileStatement)):
            return _op_has_kraus(op.get_operation())
        return False

    for inst in instructions:
        if _op_has_kraus(getattr(inst, "operation", None)):
            return True
    return False


def _circuit_is_symbolic(circuit) -> bool:
    """Return ``True`` if ``circuit`` carries any unbound symbolic
    parameter."""
    is_symbolic = getattr(circuit, "is_symbolic", None)
    if is_symbolic is None:
        return False
    try:
        return bool(is_symbolic())
    except Exception:  # pragma: no cover - defensive
        return False


# ──────────────────────────────────────────────────────────────────────────
# RNGs — tagged bundle of seed-streams (matches Julia)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class RNGs:
    """Four independent RNG streams, one per simulation stage.

    Keeping the streams separate means seeding one stage (for example
    sampling) does not affect the determinism of another (for example
    noise selection), even when the same execution touches both.

    Attributes
    ----------
    shot : random.Random
        Sampling shots from the final state.
    noise : random.Random
        Kraus and mixed-unitary noise sampling.
    trajectory : random.Random
        Monte Carlo trajectory selection.
    pass_ : random.Random
        Randomised compilation passes (e.g. simulated-annealing
        reorderers).

    Construct from a single int seed via :py:meth:`from_seed`.
    """

    shot: random.Random
    noise: random.Random
    trajectory: random.Random
    pass_: random.Random

    @staticmethod
    def from_seed(master: int) -> "RNGs":
        """Derive four streams from a single ``master`` seed.

        XOR-tags the seed bits per stream — cheap, deterministic, and
        stable across Python versions, which matters because the
        downstream backends must reproduce results bit-for-bit from
        the same input.
        """
        return RNGs(
            shot=random.Random(master ^ 0x1),
            noise=random.Random(master ^ 0x2),
            trajectory=random.Random(master ^ 0x4),
            pass_=random.Random(master ^ 0x8),
        )

    @staticmethod
    def from_generator(gen: random.Random) -> "RNGs":
        """Draw a master seed from ``gen`` and forward to
        :py:meth:`from_seed`."""
        master = gen.getrandbits(63)
        return RNGs.from_seed(master)


# ──────────────────────────────────────────────────────────────────────────
# State (abstract)
# ──────────────────────────────────────────────────────────────────────────


class State(abc.ABC):
    """Composite simulation state held by a backend.

    A state bundles three registers:

    - the quantum register (whatever representation the backend uses;
      state vector, MPS, tensor network, …);
    - the classical-bit register written by measurements and
      ``IfStatement`` outcomes;
    - the complex-valued register written by non-destructive
      observations (``Amplitude``, ``ExpectationValue``, ``BondDim``,
      …).

    Required surface for every subclass:

    - :attr:`num_qubits`, :attr:`num_bits`, :attr:`num_zvars` —
      register sizes.
    - :meth:`amplitude`, :meth:`sample` — observation primitives.
    - :attr:`classical_bits`, :attr:`complex_values` — register
      accessors.

    Optional surface:

    - :meth:`expectation` — backend overrides when it can compute
      ``⟨ψ|O|ψ⟩`` non-destructively (gated by the
      ``"expectation_state"`` capability).
    - :meth:`reset` — backend overrides when it can rebuild itself
      in place to the zero state.
    """

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits in the quantum register."""

    @property
    @abc.abstractmethod
    def num_bits(self) -> int:
        """Number of classical bits in the classical register."""

    @property
    @abc.abstractmethod
    def num_zvars(self) -> int:
        """Number of complex slots in the z-register."""

    @abc.abstractmethod
    def amplitude(self, bs) -> complex: ...

    @abc.abstractmethod
    def sample(
        self,
        nsamples: int,
        rng: Optional[random.Random] = None,
        *,
        seed: Optional[int] = None,
    ) -> list:
        """Sample ``nsamples`` measurement outcomes from the state.

        Exactly one source of randomness is consumed:

        - ``rng`` (positional or keyword): a :class:`random.Random`
          whose ``getrandbits(63)`` produces the seed forwarded to
          the simulator's PRNG.
        - ``seed`` (keyword): an int that seeds the simulator's
          default PRNG directly.
        - neither: the simulator draws fresh cryptographic entropy.

        Passing both ``rng`` and ``seed`` must raise
        :class:`TypeError`. The two arguments are mutually
        exclusive.
        """
        ...

    def expectation(self, op, *qubits: int) -> complex:
        """Compute ``⟨ψ|op|ψ⟩`` on this state.

        Most backends implement expectation on the Backend rather
        than the State (so they can route through the simulator's
        compile / evolve machinery). State-level expectation is
        provided as a convenience hook for backends whose quantum
        register supports it directly.
        """
        raise NotImplementedError(
            "expectation(op, *qubits) not implemented for this state "
            "type — call backend.expectation(state, op, *qubits) instead"
        )

    @property
    @abc.abstractmethod
    def classical_bits(self): ...

    @property
    @abc.abstractmethod
    def complex_values(self): ...

    def reset(self) -> None:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────
# Backend / LocalBackend / RemoteBackend
# ──────────────────────────────────────────────────────────────────────────


class Backend(abc.ABC):
    """Abstract base class for any simulator backend.

    A concrete backend must:

    - identify itself via :attr:`name` and :attr:`version`;
    - advertise its feature set via :meth:`capabilities`,
      :meth:`limits`, :meth:`topology`;
    - implement :meth:`execute` (or subclass :class:`LocalBackend`
      / :class:`RemoteBackend` and inherit a sensible default).

    Pick the right base for your simulator:

    - :class:`LocalBackend` — the simulator runs in-process and you
      can hand it a :class:`State` it mutates with each instruction.
    - :class:`RemoteBackend` — the simulator runs elsewhere (cloud
      service, queued executor) and your wrapper submits jobs and
      polls for results.

    See :doc:`/manual/implementing_backends`.
    """

    # ── identity ───────────────────────────────────────────────────────────
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def version(self) -> str: ...

    # ── advertisement ──────────────────────────────────────────────────────
    @abc.abstractmethod
    def capabilities(self) -> set[Capability]: ...

    def limits(self) -> Limits:
        return Limits()

    def topology(self) -> Topology:
        return AllToAll()

    def can_handle(self, circuit) -> AdmissionResult:
        """Default admission check against :meth:`limits` and the
        backend's advertised :meth:`capabilities`.

        Backends with richer admission criteria (bond-dimension
        estimates, hardware connectivity, gate-set whitelists) should
        override.

        Rejects (returns :class:`Inadmissible`) when:

        - the circuit exceeds one of ``max_qubits`` /
          ``max_classical_bits`` / ``max_zvars`` declared by
          :meth:`limits`;
        - the circuit contains a loss-bearing operation
          (``LossErr``, ``QubitLoss``) but the backend does not
          advertise the ``"loss"`` capability.

        Returns :class:`Admissible` otherwise.
        """
        lim = self.limits()
        caps = self.capabilities()

        nq = _circuit_count(circuit, "num_qubits")
        if lim.max_qubits is not None and nq > lim.max_qubits:
            return Inadmissible(
                reason=(
                    f"circuit needs {nq} qubits; backend "
                    f"{self.name} supports at most {lim.max_qubits}"
                )
            )

        nb = _circuit_count(circuit, "num_bits")
        if lim.max_classical_bits is not None and nb > lim.max_classical_bits:
            return Inadmissible(
                reason=(
                    f"circuit needs {nb} classical bits; backend "
                    f"{self.name} supports at most "
                    f"{lim.max_classical_bits}"
                )
            )

        nz = _circuit_count(circuit, "num_zvars")
        if lim.max_zvars is not None and nz > lim.max_zvars:
            return Inadmissible(
                reason=(
                    f"circuit needs {nz} z-variables; backend "
                    f"{self.name} supports at most {lim.max_zvars}"
                )
            )

        if _circuit_has_loss(circuit) and "loss" not in caps:
            return Inadmissible(
                reason=(
                    "circuit contains a loss-bearing operation "
                    f"but backend {self.name} does not declare "
                    "the 'loss' capability"
                )
            )

        if _circuit_has_kraus(circuit) and "noise" not in caps:
            return Inadmissible(
                reason=(
                    "circuit contains a noise channel "
                    f"but backend {self.name} does not declare "
                    "the 'noise' capability"
                )
            )

        # Without this gate a circuit with free symbolic parameters slips
        # past admission and fails at evolve time with an opaque error.
        if _circuit_is_symbolic(circuit) and "parametric" not in caps:
            return Inadmissible(
                reason=(
                    "circuit has free symbolic parameters "
                    f"but backend {self.name} does not declare "
                    "the 'parametric' capability"
                )
            )

        return Admissible()

    # ── resolution-time taxonomy ───────────────────────────────────────────
    def stochastic_kind(self, op_or_instruction) -> "StochasticKind":
        """Classify how this backend resolves ``op_or_instruction`` —
        :class:`StochasticKind.Deterministic`,
        :class:`StochasticKind.TrajectorySampleable`, or
        :class:`StochasticKind.RuntimeOnly`.

        The default delegates to
        :func:`default_stochastic_kind` (mix-unitary Kraus is TS;
        non-mix-unitary Kraus, mid-circuit ``Measure``, and ``LossErr``
        are RT; everything else Deterministic). Backends that handle a
        specific op type differently override this method.

        Backend-dependent: a ``MixedUnitary`` is ``TrajectorySampleable``
        for an MPS-style sampler but could be ``RuntimeOnly`` for a
        backend that resolves it state-dependently. The classification
        lives on the backend, not on the op.
        """
        from mimiqcircuits.backends.stochastic_kind import default_stochastic_kind
        return default_stochastic_kind(op_or_instruction)

    # ── pass plumbing ──────────────────────────────────────────────────────
    def default_passes(self) -> PassPipeline:
        return PassPipeline()

    def accepts_pass(self, p: AbstractPass) -> bool:
        return True

    def delegates_pass(self, p: AbstractPass) -> bool:
        return False

    # ── execution ──────────────────────────────────────────────────────────
    @abc.abstractmethod
    def execute(
        self,
        circuit,
        *,
        nsamples: int = 1000,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        passes: Optional[PassPipeline] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
    ):
        """Run ``circuit`` on this backend and return
        :class:`~mimiqcircuits.QCSResults`.

        Pass ``circuit`` as a single :class:`Circuit` or a list of
        circuits — a list returns a list of results in the same shape.

        ``seed`` and ``rng`` are mutually exclusive sources of
        randomness; pass at most one. With neither, the backend draws
        fresh entropy. To request specific amplitudes, push
        :class:`Amplitude` instructions into ``circuit`` and read the
        resulting ``results.zstates``.
        """
        ...

    def expectation(self, state, op, *qubits: int) -> complex:
        """Compute ``⟨ψ|op|ψ⟩`` on ``state`` non-destructively.

        ``qubits`` is the list of qubit indices ``op`` acts on
        (0-based). Backends override when they advertise the
        ``"expectation_state"`` capability; the default raises
        :class:`NotImplementedError` so undeclared backends fail
        loudly rather than silently degrading.
        """
        raise NotImplementedError(
            "expectation(state, op, *qubits) not declared by this "
            "backend (see :expectation_state capability)"
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _resolve_rngs(
        self,
        seed: Optional[int],
        rng: Optional[random.Random],
    ) -> RNGs:
        """Materialise an :class:`RNGs` bundle from the user's choice of
        ``seed`` / ``rng``. Raises :class:`TypeError` if both are given.

        With neither, the backend draws fresh entropy.
        """
        if seed is not None and rng is not None:
            raise TypeError(
                "`seed` and `rng` are mutually exclusive; pass at most one"
            )
        if rng is not None:
            return RNGs.from_generator(rng)
        if seed is not None:
            return RNGs.from_seed(int(seed))
        return RNGs.from_seed(random.SystemRandom().getrandbits(63))


class LocalBackend(Backend):
    """Base class for simulators that run in the local process.

    Subclasses implement a handful of primitives and inherit a full
    :meth:`execute` driver — the Python analog of the Julia
    ``AbstractQCSs.execute`` driver. Once the hooks are wired up,
    routing (sampling-vs-trajectory), the final-block projection
    circuit, loss-sampling pre-pass, and amplitude lookups all work
    without any per-backend duplication.

    Required hooks:

    - :meth:`build_state` — allocate a fresh zero state.
    - :meth:`compile` — turn a :class:`Circuit` into a
      :class:`CompiledCircuit` (a backend-specific lowered form).
    - :meth:`evolve` — apply the compiled circuit to a state and
      return the mutated state plus a typed :class:`Fidelity`.

    Optional hooks (sensible defaults provided):

    - :meth:`prepare_trajectory` — refresh the compiled artifact
      once per Monte Carlo trajectory (default: identity).
    - :meth:`recompile_per_trajectory` — predicate; default returns
      ``True`` iff the circuit contains a mixed-unitary Kraus
      channel.
    - :meth:`bind` — substitute parameters into a parametric
      compile artifact (default: re-compile after substitution).
    """

    @abc.abstractmethod
    def build_state(self, nq: int, nb: int = 0, nz: int = 0, **kwargs) -> State: ...

    @abc.abstractmethod
    def compile(self, circuit) -> CompiledCircuit: ...

    def prepare_trajectory(self, compiled: CompiledCircuit, rng) -> CompiledCircuit:
        """Refresh ``compiled`` for one Monte Carlo trajectory.

        Override when the compiled artifact contains stochastic
        elements (sampled mixed-unitary channels, sampled Kraus
        branches, …) that must be redrawn per trajectory. The
        default leaves ``compiled`` unchanged, which is correct for
        fully deterministic compilation.
        """
        return compiled

    @abc.abstractmethod
    def evolve(self, state: State, compiled: CompiledCircuit, *,
               rng=None, callback=None, stopped=None
               ) -> tuple[State, Fidelity]: ...

    def recompile_per_trajectory(self, circuit) -> bool:
        """Return ``True`` iff :meth:`compile` should be re-run for
        every trajectory. The default fires on any mixed-unitary
        :class:`krauschannel`: those backends sample a branch at
        compile time, so the compiled artifact has to be regenerated
        per trajectory to expose a fresh sample. Backends that own
        their per-trajectory sampling internally (inside
        :meth:`prepare_trajectory` or :meth:`evolve`) should
        override to return ``False``.

        Mirrors the Julia ``AbstractQCSs.recompile_per_trajectory``.
        """
        from mimiqcircuits.backends.measure_analysis import any_mixed_unitary
        return any_mixed_unitary(circuit)

    def bind(self, compiled: CompiledParametricCircuit, params: dict
             ) -> CompiledCircuit:
        """Substitute ``params`` into a parametric compiled circuit.

        The default implementation substitutes the symbols in the
        source circuit and re-runs :meth:`compile`. This is correct
        but pays the full compile cost at every parameter point.
        Override when your backend can re-bind a pre-compiled
        artifact in-place (slot maps, pre-baked gate templates, …).
        """
        # Avoid a circular import at module load time.
        from mimiqcircuits import substitute
        bound = substitute(compiled.source, params)
        return self.compile(bound)

    # ── execute driver ────────────────────────────────────────────────────
    #
    # Mirrors `AbstractQCSs.execute` in Julia. Backend-agnostic: runs the
    # pass pipeline, detects loss sampling, splits off the classical
    # projection, and routes to either `_execute_sampling` or
    # `_execute_trajectories`. Concrete backends inherit this unchanged.

    def execute(
        self,
        circuit,
        *,
        nsamples: int = 1000,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        passes: Optional[PassPipeline] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
        stopped=None,
        num_qubits: Optional[int] = None,
    ):
        from mimiqcircuits.qcsresults import QCSResults
        from mimiqcircuits.backends.measure_analysis import (
            extract_projection,
            needs_trajectories,
            needs_loss_sampling,
        )

        rngs = self._resolve_rngs(seed, rng)
        passes = passes if passes is not None else self.default_passes()

        if isinstance(circuit, list):
            return [
                self._execute_resolved(
                    c, nsamples=nsamples, rngs=rngs, passes=passes,
                    callback=callback, param_grid=param_grid,
                    strict_pass_order=strict_pass_order,
                    stopped=stopped, num_qubits=num_qubits,
                )
                for c in circuit
            ]

        return self._execute_resolved(
            circuit, nsamples=nsamples, rngs=rngs, passes=passes,
            callback=callback, param_grid=param_grid,
            strict_pass_order=strict_pass_order,
            stopped=stopped, num_qubits=num_qubits,
        )

    def _execute_resolved(
        self,
        circuit,
        *,
        nsamples: int,
        rngs: RNGs,
        passes: PassPipeline,
        callback,
        param_grid: Optional[list[dict]],
        strict_pass_order: bool,
        stopped,
        num_qubits: Optional[int],
    ):
        """Single-circuit body of :meth:`execute`. Skips the seed/rng
        resolution and list dispatch so :meth:`execute` stays a thin
        front-door."""
        from mimiqcircuits.qcsresults import QCSResults
        from mimiqcircuits.backends.measure_analysis import (
            extract_projection,
            needs_trajectories,
            needs_loss_sampling,
        )

        # Parametric grid: substitute each parameter dict into the
        # source circuit and execute the resulting concrete circuit.
        # Per-point seeds are derived from the master seed so each
        # grid point has independent randomness.
        if param_grid:
            from mimiqcircuits.backends._rng_utils import (
                normalize_seed, derive_grid_seeds,
            )
            master = normalize_seed(None, rngs)
            grid_seeds = derive_grid_seeds(master, len(param_grid))
            return [
                self._execute_resolved(
                    circuit.evaluate(params),
                    nsamples=nsamples,
                    rngs=(RNGs.from_seed(s) if s is not None else rngs),
                    passes=passes,
                    callback=callback,
                    param_grid=None,
                    strict_pass_order=strict_pass_order,
                    stopped=stopped, num_qubits=num_qubits,
                )
                for params, s in zip(param_grid, grid_seeds)
            ]

        t_total = time.time()

        # Reorder passes fold their permutation into every qubit
        # reference (including `Amplitude.bs`), so downstream analysis
        # needs no out-of-band permutation propagation. Per-pass
        # permutations remain on `PassResult.qubit_permutation`.
        ctx = PassContext(backend=self, rng=rngs.pass_)
        processed_circuit, _composed_perm, _ = apply_passes(
            passes, ctx, circuit,
        )

        results = QCSResults(simulator=self.name, version=self.version)

        # Loss sampling: pre-sample a deterministic variant per
        # trajectory so the simulator only ever sees trace-preserving
        # channels.
        if needs_loss_sampling(processed_circuit):
            self._execute_with_loss_sampling(
                processed_circuit, nsamples, rngs,
                callback, stopped, num_qubits, results,
            )
            results.timings["total"] = time.time() - t_total
            return results

        quantum_circuit, projection = extract_projection(processed_circuit)

        # Extend the projection when `num_qubits` widens the simulator
        # above the circuit's qubit count and there is no classical
        # register: each extra qubit gets a `Measure(q, q)` so the
        # output cstate covers the wider register.
        if (num_qubits is not None and processed_circuit.num_bits() == 0
                and projection.num_qubits() < num_qubits):
            import mimiqcircuits as mc
            for user_q in range(projection.num_qubits(), num_qubits):
                projection.push(mc.Measure(), user_q, user_q)

        if needs_trajectories(quantum_circuit):
            self._execute_trajectories(
                processed_circuit, nsamples, rngs,
                callback, stopped, num_qubits, results,
            )
        else:
            self._execute_sampling(
                quantum_circuit, projection, processed_circuit, nsamples,
                rngs, callback, stopped, num_qubits, results,
            )

        results.timings["total"] = time.time() - t_total
        return results

    # ── routing helpers (override for surgical changes only) ──────────────

    @staticmethod
    def _count_two_qubit_gates(circuit) -> int:
        """Number of ≥2-qubit gates in ``circuit`` — used to derive
        the per-trajectory average gate error from the circuit
        fidelity."""
        import mimiqcircuits as mc
        n = 0
        for inst in circuit.instructions:
            op = inst.operation
            if isinstance(op, mc.Gate) and op.num_qubits >= 2:
                n += 1
        return n

    @staticmethod
    def _avg_gate_error(fidelity: float, num_2q: int) -> float:
        """Estimate average multi-qubit gate error from a circuit
        fidelity: ``|expm1(log(fid) / num_2q)|``. Returns 0 when
        ``num_2q == 0`` or ``fid == 1.0`` exactly."""
        if num_2q <= 0 or fidelity == 1.0:
            return 0.0
        try:
            f = float(fidelity)
        except (TypeError, ValueError):
            return 0.0
        return abs(-math.expm1(math.log(max(f, 1e-300)) / num_2q))

    def _execute_sampling(
        self, quantum_circuit, projection, processed_circuit,
        nsamples, rngs, callback, stopped, num_qubits, results,
    ):
        """Pure unitary tail: one evolve, then sample-and-project.

        ``projection`` evaluates per-shot against the raw quantum
        sample to produce the user's `cstate`. Amplitudes and
        expectation values flow through in-circuit ``Amplitude`` /
        ``ExpectationValue`` ops into ``results.zstates``.
        """
        from mimiqcircuits.backends.measure_analysis import evaluate_projection

        # `quantum_circuit.num_qubits()` may be smaller than the
        # source when some qubits were fully absorbed by the
        # projection. The simulator state needs to span every qubit
        # the projection references plus every qubit the user asked
        # for via `num_qubits=`.
        nq = max(
            processed_circuit.num_qubits(),
            projection.num_qubits(),
            num_qubits or 0,
        )
        nb_proj = projection.num_bits()
        nb_state = max(nb_proj, processed_circuit.num_bits())
        nz = processed_circuit.num_zvars()

        num_2q = self._count_two_qubit_gates(processed_circuit)

        t_compile = time.time()
        compiled = self.compile(quantum_circuit)
        compiled = self.prepare_trajectory(compiled, rngs.trajectory)
        results.timings["compile"] = time.time() - t_compile

        state = self.build_state(nq, nb_state, nz)
        t_apply = time.time()
        state, fid = self.evolve(
            state, compiled,
            rng=rngs.noise, callback=callback, stopped=stopped,
        )
        results.timings["apply"] = time.time() - t_apply

        scalar = as_lower_bound(_to_fidelity(fid))
        results.fidelities.append(scalar)
        results.avggateerrors.append(self._avg_gate_error(scalar, num_2q))

        t_sample = time.time()
        samples = state.sample(nsamples, rngs.shot)
        for s in samples:
            results.cstates.append(evaluate_projection(projection, s))
        results.timings["sample"] = time.time() - t_sample

        if nz > 0:
            results.zstates.append(state.complex_values)

    def _execute_trajectories(
        self, processed_circuit, nsamples, rngs,
        callback, stopped, num_qubits, results,
    ):
        """Per-shot evolution: a fresh state per trajectory."""
        nq = max(processed_circuit.num_qubits(), num_qubits or 0)
        nb = processed_circuit.num_bits()
        nz = processed_circuit.num_zvars()

        num_2q = self._count_two_qubit_gates(processed_circuit)

        # Compile-once if the backend says the artifact is stable
        # across trajectories; otherwise recompile per shot.
        recompile = self.recompile_per_trajectory(processed_circuit)
        compiled = None if recompile else self.compile(processed_circuit)

        t_apply_total = 0.0
        for _ in range(nsamples):
            if recompile:
                compiled = self.compile(processed_circuit)
            prepared = self.prepare_trajectory(compiled, rngs.trajectory)
            state = self.build_state(nq, nb, nz)
            t0 = time.time()
            state, fid = self.evolve(
                state, prepared,
                rng=rngs.noise, callback=callback, stopped=stopped,
            )
            t_apply_total += time.time() - t0
            scalar = as_lower_bound(_to_fidelity(fid))
            results.fidelities.append(scalar)
            results.avggateerrors.append(self._avg_gate_error(scalar, num_2q))
            if nb > 0:
                results.cstates.append(state.classical_bits)
            if nz > 0:
                results.zstates.append(state.complex_values)
        results.timings["apply"] = t_apply_total

    def _execute_with_loss_sampling(
        self, processed_circuit, nsamples, rngs,
        callback, stopped, num_qubits, results,
    ):
        """Method-1 loss sampling: a fresh loss pattern per shot, then
        evolve the resulting deterministic circuit variant.
        """
        from mimiqcircuits import sample_losses

        nq = max(processed_circuit.num_qubits(), num_qubits or 0)
        nb = processed_circuit.num_bits()
        nz = processed_circuit.num_zvars()
        num_2q = self._count_two_qubit_gates(processed_circuit)

        t_apply_total = 0.0
        for _ in range(nsamples):
            sampled = sample_losses(processed_circuit, rng=rngs.trajectory)
            compiled = self.compile(sampled)
            prepared = self.prepare_trajectory(compiled, rngs.trajectory)
            state = self.build_state(nq, nb, nz)
            t0 = time.time()
            state, fid = self.evolve(
                state, prepared,
                rng=rngs.noise, callback=callback, stopped=stopped,
            )
            t_apply_total += time.time() - t0
            scalar = as_lower_bound(_to_fidelity(fid))
            results.fidelities.append(scalar)
            results.avggateerrors.append(self._avg_gate_error(scalar, num_2q))
            if nb > 0:
                results.cstates.append(state.classical_bits)
            if nz > 0:
                results.zstates.append(state.complex_values)
        results.timings["apply"] = t_apply_total


class RemoteBackend(Backend):
    """Base class for simulators that run on a remote service.

    Subclasses implement :meth:`submit`, which dispatches the request
    and returns a job handle. The inherited :meth:`execute` calls
    :meth:`submit`, then blocks on the returned ``job.wait()``.
    """

    @abc.abstractmethod
    def submit(self, circuits, nsamples: int, **kwargs):
        """Send the request and return a job handle.

        The job handle must expose a ``wait()`` method that blocks
        until results are available and returns
        :class:`mimiqcircuits.QCSResults` (or a list when
        ``circuits`` was a list).
        """

    def execute(
        self,
        circuit,
        *,
        nsamples: int = 1000,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        passes: Optional[PassPipeline] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
    ):
        if (
            passes is not None
            and len(passes) > 0
            and strict_pass_order
            and "pass_order_honored" not in self.capabilities()
        ):
            raise RemotePassOrderError(self.name)

        rngs = self._resolve_rngs(seed, rng)
        job = self.submit(
            circuit, nsamples,
            rngs=rngs, passes=passes,
            callback=callback,
            param_grid=param_grid,
            strict_pass_order=strict_pass_order,
        )
        return job.wait()
