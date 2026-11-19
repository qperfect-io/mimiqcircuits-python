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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from mimiqcircuits.sweep import SweepExperiment
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
)
from mimiqcircuits.backends.fidelity import (
    Fidelity, _to_fidelity, as_lower_bound,
)
from mimiqcircuits.backends.passes import (
    AbstractPass,
    PassContext,
    PassPipeline,
    apply_passes,
    resolve_pipeline,
    RemotePassOrderError,
)
from mimiqcircuits.backends.progress import (
    NoProgress,
    to_progress,
    execution_progress_callback,
)

if TYPE_CHECKING:
    from mimiqcircuits.backends.stochastic_kind import StochasticKind


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


def _circuit_has_kraus(circuit) -> bool:
    """Return ``True`` if ``circuit`` contains any Kraus channel,
    including channels wrapped inside an ``IfStatement`` or
    ``WhileStatement`` body."""
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


def _circuit_has_runtime_loss(circuit) -> bool:
    """Return ``True`` if ``circuit`` contains a ``LossyOperator``-bearing
    Kraus channel, the only loss whose outcome depends on the quantum state
    and so cannot be resolved before the backend runs. Such a circuit needs
    the ``loss`` capability; all other loss is resolved upstream."""
    instructions = getattr(circuit, "instructions", None)
    if instructions is None:
        return False
    try:
        from mimiqcircuits.operations.krauschannel import krauschannel
        from mimiqcircuits.operations.operators.lossyoperator import LossyOperator
        from mimiqcircuits.operations.ifstatement import IfStatement
        from mimiqcircuits.operations.whilestatement import WhileStatement
    except ImportError:  # pragma: no cover - defensive
        return False

    def _op_has_runtime_loss(op) -> bool:
        if isinstance(op, krauschannel):
            # A mixed-unitary channel never has a `LossyOperator`: its lossy
            # branches fire with fixed probabilities, independent of the state,
            # and are resolved before simulation. Checked first because building
            # the operators of, say, `Depolarizing` costs Kronecker products.
            if op.ismixedunitary():
                return False
            try:
                return any(isinstance(k, LossyOperator) for k in op.krausoperators())
            except Exception:  # pragma: no cover - defensive
                return False
        if isinstance(op, (IfStatement, WhileStatement)):
            return _op_has_runtime_loss(op.get_operation())
        return False

    for inst in instructions:
        if _op_has_runtime_loss(getattr(inst, "operation", None)):
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
# RNGs — tagged bundle of seed-streams
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class RNGs:
    """Four independent random streams, one per simulation stage.

    Keeping the streams separate means that the draws of one stage (for
    example sampling) do not shift the draws of another (for example noise
    selection), even when the same execution uses both.
    :meth:`LocalBackend.execute` builds one bundle per call from ``seed`` or
    ``rng`` and hands each stream to the stage it belongs to. Build one with
    :meth:`from_seed` or :meth:`from_generator`.

    Attributes:
        shot (random.Random): Sampling shots from the final state.
        noise (random.Random): Kraus and mixed-unitary noise sampling
            during :meth:`LocalBackend.evolve`.
        trajectory (random.Random): Per-trajectory work, such as
            :meth:`LocalBackend.prepare_trajectory` and loss sampling.
        pass_ (random.Random): Randomised compilation passes, read through
            :attr:`PassContext.rng`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import RNGs

        The same master seed always gives the same four streams.

        >>> a = RNGs.from_seed(42)
        >>> b = RNGs.from_seed(42)
        >>> a.shot.random() == b.shot.random()
        True

        Each stream is seeded differently, so they do not repeat each other.

        >>> c = RNGs.from_seed(42)
        >>> c.shot.random() == c.noise.random()
        False
    """

    shot: random.Random
    noise: random.Random
    trajectory: random.Random
    pass_: random.Random

    @staticmethod
    def from_seed(master: int) -> "RNGs":
        """Derive the four streams from one integer seed.

        The same seed always gives the same streams. The stream seeds are
        scrambled from ``master``, so two different master seeds, however
        close, have no stream in common.

        Args:
            master: The master seed.

        Returns:
            A new :class:`RNGs` bundle.
        """
        from mimiqcircuits.backends._rng_utils import _stream_seeds

        shot, noise, trajectory, pass_ = _stream_seeds(master)
        return RNGs(
            shot=random.Random(shot),
            noise=random.Random(noise),
            trajectory=random.Random(trajectory),
            pass_=random.Random(pass_),
        )

    @staticmethod
    def from_generator(gen: random.Random) -> "RNGs":
        """Draw a master seed from ``gen`` and pass it to :meth:`from_seed`.

        This advances ``gen``, so two calls with the same generator give two
        different bundles.

        Args:
            gen: The generator to draw a 63-bit master seed from.

        Returns:
            A new :class:`RNGs` bundle.
        """
        master = gen.getrandbits(63)
        return RNGs.from_seed(master)


# ──────────────────────────────────────────────────────────────────────────
# State (abstract)
# ──────────────────────────────────────────────────────────────────────────


class State(abc.ABC):
    """Simulation state that a :class:`LocalBackend` builds and evolves.

    A state bundles three registers: the quantum register, in whatever
    representation the backend uses (state vector, MPS, tensor network, …);
    the classical-bit register, written by measurements and classical
    operations; and the z-register of complex values, written by
    ``Amplitude``, ``ExpectationValue`` and similar operations.

    Subclasses implement :attr:`num_qubits`, :attr:`num_bits`,
    :attr:`num_zvars`, :meth:`amplitude`, :meth:`sample`,
    :attr:`classical_bits` and :attr:`complex_values`; they may override
    :meth:`expectation`, :meth:`reset` and :meth:`get_loss_state`, whose
    defaults raise :class:`NotImplementedError`.

    See Also:
        :class:`LocalBackend`, whose example defines a complete state.
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
        """Number of complex values in the z-register."""

    @abc.abstractmethod
    def amplitude(self, bs) -> complex:
        """Return the amplitude ``⟨bs|ψ⟩`` of a computational basis state.

        Args:
            bs: The basis state, as a :class:`~mimiqcircuits.BitString` with
                one bit per qubit.

        Returns:
            The complex amplitude.
        """

    @abc.abstractmethod
    def sample(
        self,
        nsamples: int,
        rng: Optional[random.Random] = None,
    ) -> list:
        """Draw ``nsamples`` measurement outcomes of all qubits from the state.

        The state is not changed. ``rng`` is a generator whose draws seed
        the simulator's own generator, which advances it: sampling twice
        with the same generator gives two independent sets of outcomes.
        Without ``rng`` the simulator draws fresh entropy.

        Args:
            nsamples: Number of outcomes to draw.
            rng: The generator to draw from. :meth:`LocalBackend.execute`
                passes :attr:`RNGs.shot` here.

        Returns:
            A list of ``nsamples`` :class:`~mimiqcircuits.BitString`, one bit
            per qubit.
        """
        ...

    def expectation(self, op, *qubits: int) -> complex:
        """Return ``⟨ψ|op|ψ⟩`` on this state, without changing it.

        Most backends compute expectation values on the backend instead, with
        :meth:`Backend.expectation`. States whose quantum register can do it
        directly override this; the default raises
        :class:`NotImplementedError`.

        Args:
            op: The operator to measure.
            *qubits: The qubits ``op`` acts on, 0-based.

        Returns:
            The complex expectation value.

        Raises:
            NotImplementedError: If the state does not support it.
        """
        raise NotImplementedError(
            "expectation(op, *qubits) not implemented for this state "
            "type — call backend.expectation(state, op, *qubits) instead"
        )

    @property
    @abc.abstractmethod
    def classical_bits(self):
        """The classical register, as a :class:`~mimiqcircuits.BitString`.

        The execute driver stores it as one entry of
        :attr:`QCSResults.cstates <mimiqcircuits.QCSResults.cstates>` per
        trajectory.
        """

    @property
    @abc.abstractmethod
    def complex_values(self):
        """The z-register, as a list of complex numbers.

        The execute driver stores it as one entry of
        :attr:`QCSResults.zstates <mimiqcircuits.QCSResults.zstates>` per
        evolution.
        """

    def reset(self) -> None:
        """Return the state to all qubits in ``|0⟩``, with all registers
        cleared, in place.

        States that can do this override it; the default raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError

    def get_loss_state(self) -> "LossState":
        """Return the per-qubit loss register (see :class:`LossState`).

        States of backends that use the runtime-loss driver
        (:meth:`LocalBackend.uses_loss_driver`) hold one, clear it in
        :meth:`reset`, and override this to return it. The default raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "get_loss_state() not implemented for this state type"
        )


class LossState:
    """Per-qubit register that records which qubits are lost.

    It sits next to the quantum, classical and complex registers of a
    :class:`State`. Entry ``q`` is ``True`` when qubit ``q`` has left the
    system. The register holds no loss policy: the runtime-loss driver
    (:meth:`LocalBackend.evolve_with_loss`) is the only code that marks
    qubits; a state that holds one only creates it, clears it in
    :meth:`State.reset`, and returns it from :meth:`State.get_loss_state`.

    Args:
        n_or_flags: The number of qubits, all present, or an iterable of
            booleans giving the initial flag of each qubit.

    Attributes:
        lost (list[bool]): The flag of each qubit.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import LossState

        Start with three qubits, all present, then lose qubit 1.

        >>> ls = LossState(3)
        >>> ls.mark_lost(1)
        >>> ls.is_lost(1), ls.any_lost([0, 1]), ls.all_lost([0, 1])
        (True, True, False)
        >>> ls.reset()
        >>> ls.lost
        [False, False, False]
    """

    def __init__(self, n_or_flags):
        if isinstance(n_or_flags, int):
            self.lost = [False] * n_or_flags
        else:
            self.lost = list(n_or_flags)

    def is_lost(self, q) -> bool:
        """Return whether qubit ``q`` is lost."""
        return self.lost[q]

    def mark_lost(self, q) -> None:
        """Mark qubit ``q`` as lost."""
        self.lost[q] = True

    def mark_present(self, q) -> None:
        """Mark qubit ``q`` as present."""
        self.lost[q] = False

    def any_lost(self, qubits) -> bool:
        """Return whether at least one of ``qubits`` is lost."""
        return any(self.lost[q] for q in qubits)

    def all_lost(self, qubits) -> bool:
        """Return whether every one of ``qubits`` is lost (``True`` when
        ``qubits`` is empty)."""
        return all(self.lost[q] for q in qubits)

    def reset(self) -> None:
        """Mark every qubit as present."""
        self.lost = [False] * len(self.lost)

    def __len__(self) -> int:
        return len(self.lost)

    def as_map(self) -> dict:
        """Return the lost qubits as a ``{qubit: True}`` dictionary.

        Present qubits get no entry. This is the form the loss-model rewrite
        reads.
        """
        return {q: True for q, v in enumerate(self.lost) if v}


# ──────────────────────────────────────────────────────────────────────────
# Backend / LocalBackend / RemoteBackend
# ──────────────────────────────────────────────────────────────────────────


class Backend(abc.ABC):
    """Abstract base class for every simulator backend.

    Subclasses implement :attr:`name`, :attr:`version`, :meth:`capabilities`
    and :meth:`execute`; they may override :meth:`limits`, :meth:`topology`,
    :meth:`can_handle`, :meth:`stochastic_kind`, :meth:`default_passes`, the
    pass hooks (:meth:`fuse_pass`, :meth:`reorder_pass`,
    :meth:`canonical_decompose_pass`, :meth:`remove_swaps_pass`,
    :meth:`simplify_pass`), :meth:`accepts_pass`, :meth:`delegates_pass` and
    :meth:`expectation`.

    Most backends subclass one of the two bases below, which implement
    :meth:`execute`:

    - :class:`LocalBackend`, when the simulator runs in this process and
      evolves a :class:`State`;
    - :class:`RemoteBackend`, when the simulator runs elsewhere (a cloud
      service, a job queue) and the backend submits jobs and waits for the
      results.

    See :doc:`/manual/implementing_backends` for a step-by-step guide.
    """

    # ── identity ───────────────────────────────────────────────────────────
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the simulator, such as ``"Mimiq"``.

        It is stored in :attr:`QCSResults.simulator
        <mimiqcircuits.QCSResults.simulator>` and used in error messages.
        """

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Version string of the simulator, stored in
        :attr:`QCSResults.version <mimiqcircuits.QCSResults.version>`."""

    # ── advertisement ──────────────────────────────────────────────────────
    @abc.abstractmethod
    def capabilities(self) -> set[Capability]:
        """Return the set of features this backend can run.

        Each entry is a capability token, normally one of
        :data:`CAPABILITY_VOCABULARY`. Declare only what the backend really
        delivers: :meth:`can_handle` and the conformance tests rely on it.

        Returns:
            A set of capability strings.
        """

    def limits(self) -> Limits:
        """Return the numeric limits of this backend.

        The default returns ``Limits()``, with every limit unbounded.
        """
        return Limits()

    def topology(self) -> Topology:
        """Return the qubit connectivity of this backend.

        The default returns :class:`AllToAll`.
        """
        return AllToAll()

    def can_handle(self, circuit) -> AdmissionResult:
        """Check whether this backend can run ``circuit``.

        The default compares ``circuit`` with :meth:`limits` and
        :meth:`capabilities`. It returns :class:`Inadmissible` when the
        circuit:

        - has more qubits, classical bits or z-variables than ``max_qubits``,
          ``max_classical_bits`` or ``max_zvars`` allow;
        - contains a Kraus channel with a ``LossyOperator`` (loss decided at
          run time) and the backend does not declare ``"loss"``;
        - contains any Kraus channel, also inside an ``IfStatement`` or
          ``WhileStatement``, and the backend does not declare ``"noise"``;
        - has free symbolic parameters and the backend does not declare
          ``"parametric"``.

        Otherwise it returns :class:`Admissible`. It never returns
        :class:`Marginal`. Backends with other criteria (bond-dimension
        estimates, connectivity, gate sets) override it.

        Args:
            circuit: The circuit to check.

        Returns:
            An :class:`AdmissionResult`. Use :func:`is_admissible` to turn
            it into a bool.
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

        if _circuit_has_runtime_loss(circuit) and "loss" not in caps:
            return Inadmissible(
                reason=(
                    "circuit contains runtime (LossyOperator) loss "
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
        """Classify when this backend resolves the randomness of an operation.

        The answer depends on the backend as well as on the operation: a
        mixed-unitary channel is ``TrajectorySampleable`` for a backend that
        draws its branch before evolving, but ``RuntimeOnly`` for one that
        draws it from the live state. The default returns
        :func:`default_stochastic_kind`; backends that handle an operation
        differently override this.

        Args:
            op_or_instruction: An operation, or an instruction whose operation
                is classified.

        Returns:
            A :class:`StochasticKind`.
        """
        from mimiqcircuits.backends.stochastic_kind import default_stochastic_kind
        return default_stochastic_kind(op_or_instruction)

    # ── pass plumbing ──────────────────────────────────────────────────────
    def default_passes(self) -> PassPipeline:
        """Return the pipeline this backend wants applied before evolution.

        This is the backend's whole part in circuit preparation: it declares
        the passes it expects, and :meth:`execute` adjusts that declaration to
        match the flags the caller set. The backend never sees the flags;
        when a flag asks for a pass, the backend builds it through
        :meth:`fuse_pass`, :meth:`reorder_pass`,
        :meth:`canonical_decompose_pass`, :meth:`remove_swaps_pass` or
        :meth:`simplify_pass`. A flag finds the pass it acts on by its
        ``spec().name``, so a pass that no flag names is never removed by one.

        Returns:
            A :class:`PassPipeline`. The default is empty.
        """
        return PassPipeline()

    # One hook per preparation flag. Each returns the pass the flag installs;
    # a backend with its own pass of that kind overrides the hook, so the flag
    # retunes that pass instead of swapping in the generic one.

    def fuse_pass(self, *, threshold: Optional[int] = None) -> AbstractPass:
        """Return the pass that ``execute(fuse=True)`` or ``fuse_threshold=n``
        installs.

        Backends with their own fusion pass override this, so the flag retunes
        that pass instead of replacing it.

        Args:
            threshold: The ``fuse_threshold`` the caller gave, or ``None``.

        Returns:
            The default is :class:`~mimiqcircuits.fusion.FusePass` with
            ``qubit_threshold=threshold`` (``0`` for ``None``).
        """
        from mimiqcircuits.fusion import FusePass

        return FusePass(qubit_threshold=threshold or 0)

    def reorder_pass(
        self, method: Optional[str] = None, *, seed: Optional[int] = None
    ) -> AbstractPass:
        """Return the pass that ``execute(reorderqubits=...)`` installs.

        Backends that can reorder qubits override this; there is no generic
        local reorder pass, so the default raises :class:`NotImplementedError`.

        Args:
            method: The method name given as ``reorderqubits``, or ``None``
                for ``reorderqubits=True``, which leaves the choice to the
                backend.
            seed: The ``reorderqubits_seed`` the caller gave, or ``None``.

        Returns:
            The reorder pass.

        Raises:
            NotImplementedError: In the default implementation.
        """
        raise NotImplementedError(
            "reorderqubits not available for local execution on "
            f"{type(self).__name__}: no local pass implements it. Submit to a "
            "remote MIMIQ backend, which runs it server-side, or pass a "
            "backend-specific pass through `passes=`."
        )

    def canonical_decompose_pass(self) -> AbstractPass:
        """Return the pass that ``execute(canonicaldecompose=True)`` installs.

        The default is :class:`CanonicalDecomposePass`.
        """
        from mimiqcircuits.backends.concrete_passes import CanonicalDecomposePass

        return CanonicalDecomposePass()

    def remove_swaps_pass(self) -> AbstractPass:
        """Return the pass that ``execute(remove_swaps=True)`` installs.

        Backends that can remove SWAP gates override this; there is no
        generic local remove-swaps pass, so the default raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "remove_swaps not available for local execution on "
            f"{type(self).__name__}: no local pass implements it. Submit to a "
            "remote MIMIQ backend, which runs it server-side, or pass a "
            "backend-specific pass through `passes=`."
        )

    def simplify_pass(self) -> AbstractPass:
        """Return the pass that ``execute(simplify=True)`` installs.

        The default is :class:`~mimiqcircuits.SimplifyPass` with
        :data:`~mimiqcircuits.DEFAULT_SIMPLIFY_PASSES`.
        """
        from mimiqcircuits.simplify import SimplifyPass

        return SimplifyPass()

    def accepts_pass(self, p: AbstractPass) -> bool:
        """Return whether this backend can run pass ``p``.

        :func:`apply_passes` raises :class:`UnacceptedPassError` for a pass
        the backend refuses, instead of skipping it. The default accepts
        every pass.
        """
        return True

    def delegates_pass(self, p: AbstractPass) -> bool:
        """Return whether this backend runs pass ``p`` itself.

        For a delegated pass, :func:`apply_passes` does not call
        :meth:`AbstractPass.apply`; it records a :class:`PassResult` with
        ``metadata={"delegated": True}`` and leaves the work to the backend's
        :meth:`LocalBackend.compile` or :meth:`LocalBackend.evolve`. The
        default delegates nothing.
        """
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
        fuse: Optional[bool] = None,
        canonicaldecompose: Optional[bool] = None,
        reorderqubits: Optional[Union[bool, str]] = None,
        remove_swaps: Optional[bool] = None,
        simplify: Optional[bool] = None,
        reorderqubits_seed: Optional[int] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
    ):
        """Run ``circuit`` on this backend and return its results.

        To request specific amplitudes or expectation values, push
        ``Amplitude`` or ``ExpectationValue`` instructions into ``circuit``
        and read them from ``results.zstates``.

        The preparation flags (``fuse``, ``canonicaldecompose``,
        ``reorderqubits``, ``remove_swaps``, ``simplify``) adjust the
        backend's :meth:`default_passes` instead of replacing them. Each
        speaks about one kind of pass: ``None`` leaves that pass as the
        backend declared it, ``False`` drops it, and any other value replaces
        it with the pass the backend builds for that flag, or adds that pass
        when the backend declared none of that kind. So ``fuse=False`` means
        "the backend's pipeline, without fusion". A pass that no flag names
        stays even when every flag is ``False``.

        The flags have a fixed order among themselves: remove swaps,
        simplify, canonical decompose, fuse, reorder. A pass added by a flag
        goes just before the first declared pass of a kind that runs later,
        or at the end if there is none. ``passes`` are appended after all of
        that, as given; nothing is deduplicated.

        ``simplify=True`` rewrites the circuit with :meth:`simplify_pass`
        while keeping its unitary exactly: inverse pairs cancel, rotations
        merge, and one- and two-qubit blocks are resynthesised with fewer
        gates.

        Args:
            circuit: A :class:`~mimiqcircuits.Circuit` or a list of circuits.
                Local backends also take a
                :class:`~mimiqcircuits.SweepExperiment`, which runs the
                circuit once per point of its table.
            nsamples: Number of shots.
            seed: Integer seed for all randomness of the run. Mutually
                exclusive with ``rng``; with neither, the backend draws fresh
                entropy.
            rng: A :class:`random.Random` to draw the seed from. It is
                advanced, so two runs with the same generator differ.
            passes: Extra passes to run after the backend's adjusted
                pipeline.
            fuse: Install (``True``) or drop (``False``) gate fusion.
            canonicaldecompose: Install or drop the decomposition into the
                canonical gate set.
            reorderqubits: Install or drop qubit reordering; a string names
                the reordering method.
            remove_swaps: Install or drop SWAP removal.
            simplify: Install or drop circuit simplification. Off unless set.
            reorderqubits_seed: Seed for the reordering pass. It configures
                reordering; it does not request it.
            callback: Called by the backend during evolution, once per step.
            param_grid: A list of dictionaries, each mapping the circuit's
                symbols (or their names) to values, such as
                ``[{theta: 0.1}, {theta: 0.2}]``. Each dictionary is
                substituted into ``circuit`` with
                :meth:`~mimiqcircuits.Circuit.evaluate` and runs as its own
                execution, with a seed derived from ``seed`` or ``rng``, so
                points are independent and a repeated sweep with the same
                ``seed`` gives the same results. Every dictionary must bind
                all free symbols. An empty list runs ``circuit`` once.
            strict_pass_order: On a remote backend that does not declare
                ``"pass_order_honored"``, refuse a non-empty ``passes``
                instead of submitting it as an unordered set.

        Returns:
            A :class:`~mimiqcircuits.QCSResults`. A list of circuits gives a
            list of results in the same order; ``param_grid`` or a
            :class:`~mimiqcircuits.SweepExperiment` gives one result per
            point (a list of such lists for a list of circuits).

        Raises:
            TypeError: If both ``seed`` and ``rng`` are given, or a flag has
                the wrong type (for example ``fuse=0``).
            NotImplementedError: If ``param_grid`` is given to a remote
                backend that does not support it (such as
                :class:`MimiqRemoteBackend`), or a flag asks for a pass the
                backend cannot build.
            RemotePassOrderError: See ``strict_pass_order``.
        """
        ...

    def expectation(self, state, op, *qubits: int) -> complex:
        """Return ``⟨ψ|op|ψ⟩`` on ``state``, without changing it.

        Backends that declare the ``"expectation_state"`` capability override
        this; the default raises :class:`NotImplementedError`.

        Args:
            state: A :class:`State` built by this backend.
            op: The operator to measure.
            *qubits: The qubits ``op`` acts on, 0-based.

        Returns:
            The complex expectation value.

        Raises:
            NotImplementedError: In the default implementation.
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


def _loss_segment_role(op, qs, ls):
    """Role of an instruction in the runtime-loss walk, given the loss register:
    ``"event"`` is a loss boundary the driver resolves, ``"buffer"`` joins the
    all-present run, ``"drop"`` is an all-lost op skipped without breaking the
    run.
    """
    from mimiqcircuits.operations.losschannel import Check, Loss, MeasureCheck, Reload
    from mimiqcircuits.operations.measure import Measure
    from mimiqcircuits.operations.krauschannel import krauschannel

    if isinstance(op, (Loss, Reload, Check, MeasureCheck)) or \
            isinstance(op, krauschannel):
        return "event"
    if isinstance(op, Measure) and ls.is_lost(qs[0]):
        return "event"
    if not ls.any_lost(qs):
        return "buffer"
    if ls.all_lost(qs):
        return "drop"
    return "event"


def _next_loss_segment(insts, start, ls):
    """Collect the maximal all-present run from ``start`` under loss register
    ``ls``. Returns ``(run, event_index_or_None, next_index)``; the run stops
    just before the first loss event (or the circuit end).
    """
    seg = []
    n = len(insts)
    j = start
    while j < n:
        inst = insts[j]
        role = _loss_segment_role(inst.get_operation(), tuple(inst.get_qubits()), ls)
        if role == "buffer":
            seg.append(inst)
        elif role == "event":
            return seg, j, j + 1
        # "drop": skip the all-lost op, keep the run open
        j += 1
    return seg, None, n


class LocalBackend(Backend):
    """Base class for simulators that run in this process.

    Subclasses implement :meth:`build_state`, :meth:`compile` and
    :meth:`evolve`, plus the :class:`Backend` identity members; they may
    override :meth:`prepare_trajectory`, :meth:`recompile_per_trajectory`,
    :meth:`bind` and :meth:`compile_progress`, and opt into the runtime-loss
    driver with :meth:`uses_loss_driver`, :meth:`apply_instruction`,
    :meth:`sample_kraus` and :meth:`apply_segment`.

    In return they inherit :meth:`execute`, which runs the pass pipeline,
    splits the trailing measurements off into a classical projection,
    and then either evolves the circuit once and samples the final state, or
    evolves one fresh state per shot when the circuit has mid-circuit
    measurements, noise, or loss.

    Examples:
        A backend for circuits of ``X`` and ``CX`` gates only. Its state is a
        single computational basis state, so it is exact and fully
        deterministic.

        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import (
        ...     DefaultCompiledCircuit, ExactFidelity, LocalBackend, State)

        The state holds the three registers, and samples by returning its
        basis state ``nsamples`` times.

        >>> class BasisState(State):
        ...     def __init__(self, nq, nb, nz):
        ...         self.qubits = [0] * nq
        ...         self.bits = [0] * nb
        ...         self.zvars = [0j] * nz
        ...     @property
        ...     def num_qubits(self):
        ...         return len(self.qubits)
        ...     @property
        ...     def num_bits(self):
        ...         return len(self.bits)
        ...     @property
        ...     def num_zvars(self):
        ...         return len(self.zvars)
        ...     @property
        ...     def classical_bits(self):
        ...         return BitString(self.bits)
        ...     @property
        ...     def complex_values(self):
        ...         return list(self.zvars)
        ...     def amplitude(self, bs):
        ...         return 1.0 + 0j if list(bs) == self.qubits else 0j
        ...     def sample(self, nsamples, rng=None):
        ...         return [BitString(self.qubits) for _ in range(nsamples)]

        The backend declares what it supports, builds states, and evolves
        them. ``compile`` has nothing to lower here, so it wraps the circuit
        as it is.

        >>> class BasisBackend(LocalBackend):
        ...     @property
        ...     def name(self):
        ...         return "Basis"
        ...     @property
        ...     def version(self):
        ...         return "0.1"
        ...     def capabilities(self):
        ...         return {"sampling", "classical_bits", "midcircuit_measure"}
        ...     def build_state(self, nq, nb=0, nz=0, **kwargs):
        ...         return BasisState(nq, nb, nz)
        ...     def compile(self, circuit):
        ...         return DefaultCompiledCircuit(circuit)
        ...     def evolve(self, state, compiled, *, rng=None, callback=None,
        ...                stopped=None):
        ...         for inst in compiled.source:
        ...             op, qs = inst.operation, inst.qubits
        ...             if isinstance(op, GateX):
        ...                 state.qubits[qs[0]] ^= 1
        ...             elif isinstance(op, GateCX):
        ...                 state.qubits[qs[1]] ^= state.qubits[qs[0]]
        ...             elif isinstance(op, Measure):
        ...                 state.bits[inst.bits[0]] = state.qubits[qs[0]]
        ...             else:
        ...                 raise ValueError(f"unsupported operation {op}")
        ...         return state, ExactFidelity()

        ``execute`` does the rest. The final measurements never reach
        ``evolve``: the driver samples the state and maps the samples to
        the classical register itself.

        >>> backend = BasisBackend()
        >>> c = Circuit()
        >>> c.push(GateX(), 0)
        1-qubit circuit with 1 instruction:
        └── X @ q[0]
        <BLANKLINE>
        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
        >>> c.push(Measure(), range(2), range(2))
        2-qubit, 2-bit circuit with 4 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── M @ q[0], c[0]
        └── M @ q[1], c[1]
        <BLANKLINE>
        >>> res = backend.execute(c, nsamples=3, seed=1)
        >>> res.cstates
        [bs"11", bs"11", bs"11"]
        >>> res.fidelities
        [1.0]

        The default :meth:`~Backend.can_handle` rejects what the backend does
        not declare, such as noise.

        >>> noisy = Circuit().push(Depolarizing(1, 0.1), 0)
        >>> backend.can_handle(noisy)
        Inadmissible(reason="circuit contains a noise channel but backend Basis does not declare the 'noise' capability")
    """

    @abc.abstractmethod
    def build_state(self, nq: int, nb: int = 0, nz: int = 0, **kwargs) -> State:
        """Return a new state with every qubit in ``|0⟩`` and every register
        cleared.

        Args:
            nq: Number of qubits.
            nb: Number of classical bits.
            nz: Number of z-variables.
            **kwargs: Backend-specific options. The execute driver passes
                none.

        Returns:
            A :class:`State` of this backend.
        """

    @abc.abstractmethod
    def compile(self, circuit) -> CompiledCircuit:
        """Lower a circuit into the form :meth:`evolve` runs.

        Compilation should be pure: it gets no random generator, and a result
        that samples nothing can be reused across trajectories and parameter
        points. Per-trajectory randomness belongs in
        :meth:`prepare_trajectory`. A backend that does draw random choices
        while compiling relies on :meth:`recompile_per_trajectory` to be
        compiled again for every trajectory.

        Args:
            circuit: The prepared :class:`~mimiqcircuits.Circuit`.

        Returns:
            A :class:`CompiledCircuit`. A backend that runs circuits as they
            are returns ``DefaultCompiledCircuit(circuit)``.
        """

    def prepare_trajectory(self, compiled: CompiledCircuit, rng) -> CompiledCircuit:
        """Refresh ``compiled`` for one Monte Carlo trajectory.

        Backends whose compiled form holds random choices (sampled
        mixed-unitary branches, …) override this to redraw them. The default
        returns ``compiled`` unchanged.

        Args:
            compiled: The result of :meth:`compile`.
            rng: The trajectory generator, :attr:`RNGs.trajectory`.

        Returns:
            The compiled circuit for this trajectory.
        """
        return compiled

    @abc.abstractmethod
    def evolve(self, state: State, compiled: CompiledCircuit, *,
               rng=None, callback=None, stopped=None
               ) -> tuple[State, Fidelity]:
        """Apply a compiled circuit to ``state``.

        The state can be changed in place and returned. Measurements and
        classical operations write the classical register, and
        ``Amplitude``, ``ExpectationValue`` and similar operations write the
        z-register.

        Args:
            state: A state from :meth:`build_state`.
            compiled: The compiled circuit, after :meth:`prepare_trajectory`.
            rng: The generator that randomness resolved during evolution
                (measurements, resets, noise) is drawn from;
                :meth:`execute` passes :attr:`RNGs.noise`. It is advanced,
                so successive calls sharing one generator draw independent
                numbers. Without it the backend draws fresh entropy.
            callback: ``None``, or a callable to report progress. When
                progress reporting is on, the driver passes a callable that
                expects ``callback(i, total)`` once per step, with ``i`` the
                1-based step and ``total`` the number of steps or ``None``.
            stopped: The ``stopped`` argument of :meth:`execute`, passed
                through unchanged.

        Returns:
            A tuple ``(state, fidelity)``: the evolved state and a
            :class:`Fidelity`. A plain float is also accepted: ``1.0`` is
            read as :class:`ExactFidelity`, any other value as
            :class:`TruncationLowerBound`, and ``None`` as
            :class:`UnknownFidelity`.
        """

    def recompile_per_trajectory(self, circuit) -> bool:
        """Return whether :meth:`compile` runs again for every trajectory.

        The default returns ``True`` when ``circuit`` has a mixed-unitary
        Kraus channel, for backends that draw its branch while compiling.
        Backends that draw it in :meth:`prepare_trajectory` or
        :meth:`evolve` override this to return ``False`` and compile once.

        Args:
            circuit: The prepared circuit.

        Returns:
            Whether to recompile per trajectory.
        """
        from mimiqcircuits.backends.measure_analysis import any_mixed_unitary
        return any_mixed_unitary(circuit)

    def bind(self, compiled: CompiledParametricCircuit, params: dict
             ) -> CompiledCircuit:
        """Substitute ``params`` into a compiled parametric circuit.

        The default evaluates the source circuit with ``params`` and calls
        :meth:`compile` again, which pays the full compile cost at every
        point. Backends that can rebind a compiled form in place override
        this.

        Args:
            compiled: A :class:`CompiledParametricCircuit` whose ``source`` is
                the symbolic circuit.
            params: Values for the free symbols, as for
                :meth:`~mimiqcircuits.Circuit.evaluate`.

        Returns:
            A fully bound :class:`CompiledCircuit`.
        """
        return self.compile(compiled.source.evaluate(params))

    def optimize(self, experiment, initparams=None, *, return_history=False, **kwargs):
        """Minimize a cost read from a parametric circuit by running it on
        this backend.

        At every step the optimizer proposes values for the circuit's
        symbols, the circuit is evaluated with them and run with
        :meth:`execute`, and the cost is the real part of the mean of
        z-register ``zregister`` over the samples. Push the quantity to
        minimize into that register, for example with
        :meth:`~mimiqcircuits.Circuit.push_expval`.

        Call it with an :class:`~mimiqcircuits.OptimizationExperiment`, a
        list of them, or a circuit and a dictionary of initial values for
        all its symbols plus the experiment's fields as keywords:

        - ``optimizer``: ``"BFGS"``, ``"LBFGS"``, ``"CG"``, ``"NELDERMEAD"``,
          ``"NEWTON"``, ``"COBYLA"`` (default) or ``"CMAES"``. All but
          ``"CMAES"`` come from :func:`scipy.optimize.minimize`; ``"CMAES"``
          needs the ``cmaes`` extra (``pip install 'mimiqcircuits[cmaes]'``).
        - ``maxiters``: iteration limit, or ``None`` for the optimizer's own.
        - ``zregister``: index of the z-register holding the cost (default 0).
        - ``label``: a name for the experiment.

        With an experiment these fields come from it, and passing them again
        is an error. Every other keyword goes to :meth:`execute` on each
        evaluation: ``nsamples``, ``seed``, ``fuse``, ``passes`` and so on.
        With a fixed ``seed`` every evaluation draws the same random numbers,
        so a noisy cost does not jump between otherwise equal points.

        Args:
            experiment: An :class:`~mimiqcircuits.OptimizationExperiment`, a
                list of them, or a symbolic circuit.
            initparams: Initial values when ``experiment`` is a circuit;
                must be ``None`` otherwise.
            return_history: Also keep every evaluated run.
            **kwargs: Experiment fields (with a circuit only) and
                :meth:`execute` options.

        Returns:
            An :class:`~mimiqcircuits.OptimizationResults` with the
            lowest-cost :class:`~mimiqcircuits.OptimizationRun` and, with
            ``return_history=True``, every evaluated run in order. A list of
            experiments gives a list of results in the same order.

        Raises:
            ValueError: For an unknown optimizer, a ``zregister`` the circuit
                does not have, or a circuit and initial values that do not
                form a valid experiment.
            TypeError: If ``initparams`` or an experiment field is given
                together with an experiment.

        Examples:
            A one-qubit backend for circuits of ``RY`` gates followed by
            ``⟨Z⟩`` measurements. It tracks the rotation angle and writes
            ``cos(theta)`` to the z-register.

            >>> from mimiqcircuits import *
            >>> from mimiqcircuits.backends import (
            ...     DefaultCompiledCircuit, LocalBackend, State)
            >>> import math
            >>> from symengine import symbols
            >>> class RYState(State):
            ...     num_qubits = property(lambda self: 1)
            ...     num_bits = property(lambda self: 0)
            ...     num_zvars = property(lambda self: len(self.z))
            ...     classical_bits = property(lambda self: BitString(0))
            ...     complex_values = property(lambda self: list(self.z))
            ...     def __init__(self, nz):
            ...         self.theta, self.z = 0.0, [0j] * nz
            ...     def amplitude(self, bs):
            ...         raise NotImplementedError
            ...     def sample(self, nsamples, rng=None):
            ...         return [BitString(1)] * nsamples
            >>> class RYBackend(LocalBackend):
            ...     name = "RY"
            ...     version = "0.1"
            ...     def capabilities(self):
            ...         return {"zvars", "expectation_1q"}
            ...     def build_state(self, nq, nb=0, nz=0, **kwargs):
            ...         return RYState(nz)
            ...     def compile(self, circuit):
            ...         return DefaultCompiledCircuit(circuit)
            ...     def evolve(self, state, compiled, *, rng=None,
            ...                callback=None, stopped=None):
            ...         for inst in compiled.source:
            ...             if isinstance(inst.operation, GateRY):
            ...                 state.theta += float(inst.operation.theta)
            ...             else:
            ...                 state.z[inst.zvars[0]] = math.cos(state.theta)
            ...         return state, 1.0

            The cost is ``⟨Z⟩`` on qubit 0, written to z-register 0. It is
            lowest, ``-1``, at ``theta = pi``.

            >>> theta = symbols("theta")
            >>> c = Circuit().push(GateRY(theta), 0)
            >>> c.push(ExpectationValue(GateZ()), 0, 0)
            1-qubit, 1-zvar circuit with 2 instructions:
            ├── RY(theta) @ q[0]
            └── ⟨Z⟩ @ q[0], z[0]
            <BLANKLINE>

            Start from ``theta = 0.1``. ``nsamples=1`` is enough because the
            cost is exact.

            >>> res = RYBackend().optimize(c, {theta: 0.1},
            ...                            optimizer="NELDERMEAD", nsamples=1)
            >>> best = res.get_best()
            >>> round(best.cost, 4), round(best.parameters[theta], 2)
            (-1.0, 3.14)
        """
        from mimiqcircuits.backends.optimize import optimize

        return optimize(self, experiment, initparams,
                        return_history=return_history, **kwargs)

    def compile_progress(self, circuit, progress):
        """Compile ``circuit``, optionally reporting progress.

        The default ignores ``progress`` and calls :meth:`compile`. Backends
        whose compilation has countable steps (such as fusing gates into
        tensors) override this to open a stage on ``progress`` and step it.

        Args:
            circuit: The prepared circuit.
            progress: A :class:`Progress` sink.

        Returns:
            The result of :meth:`compile`.
        """
        return self.compile(circuit)

    # ── runtime-loss driver ───────────────────────────────────────────────
    #
    # Runtime `LossyOperator` loss cannot be resolved before the backend runs
    # (which qubit leaks depends on which Kraus branch fires), so it is
    # resolved live. A backend opts in by returning `True` from
    # `uses_loss_driver` and implementing the two loss-agnostic primitives
    # below; all loss bookkeeping lives in the driver.

    def uses_loss_driver(self) -> bool:
        """Return whether :meth:`execute` resolves runtime loss through
        :meth:`evolve_with_loss`.

        Runtime loss comes from Kraus channels with a ``LossyOperator``
        branch: which qubit is lost depends on the branch that fires. A
        backend opts in by returning ``True``, implementing
        :meth:`apply_instruction` and :meth:`sample_kraus`, and giving its
        state a :class:`LossState` (see :meth:`State.get_loss_state`). The
        default returns ``False``.
        """
        return False

    def apply_instruction(self, state, inst, *, rng=None):
        """Apply one instruction to ``state`` for the runtime-loss driver.

        Backends that use the runtime-loss driver override this; the default
        raises :class:`NotImplementedError`. The instruction never involves
        a lost qubit, so the method needs no loss logic.

        Args:
            state: The state to change in place.
            inst: The instruction to apply.
            rng: The noise generator.

        Returns:
            The fidelity factor of this step, ``1.0`` when it is exact.
        """
        raise NotImplementedError(
            "apply_instruction(state, inst) not implemented for this backend"
        )

    def sample_kraus(self, state, channel, targets, *, rng=None):
        """Draw and apply one branch of a Kraus channel, for the runtime-loss
        driver.

        Backends that use the runtime-loss driver override this; the default
        raises :class:`NotImplementedError`.

        Args:
            state: The state to change in place.
            channel: The Kraus channel.
            targets: The qubits it acts on.
            rng: The noise generator.

        Returns:
            A tuple ``(fidelity, fired)``: the fidelity factor and the Kraus
            operator that fired, as the channel holds it, so that a
            ``LossyOperator`` keeps its type and the driver can mark its
            qubits as lost.
        """
        raise NotImplementedError(
            "sample_kraus(state, channel, targets) not implemented for this backend"
        )

    def apply_segment(self, state, insts, *, rng=None):
        """Apply a run of instructions on present qubits, for the runtime-loss
        driver.

        The driver passes the longest run between two loss events, and
        every instruction in it acts only on present qubits, so a backend
        that compresses or fuses gates (MPS, MPO) can override this to
        process the whole run at once. The default calls
        :meth:`apply_instruction` on each instruction in turn.

        Args:
            state: The state to change in place.
            insts: The instructions to apply.
            rng: The noise generator.

        Returns:
            The product of the fidelity factors.
        """
        fid = 1.0
        for inst in insts:
            fid *= self.apply_instruction(state, inst, rng=rng)
        return fid

    def evolve_with_loss(self, state, circuit, lossmodel, *,
                         rng=None, callback=None, stopped=None):
        """Evolve ``circuit`` on ``state`` while resolving qubit loss as it
        happens.

        The lost qubits are tracked in the state's :class:`LossState` and
        interpreted as :func:`~mimiqcircuits.lossmodel.lower_losses` does.
        The circuit is walked as alternating runs and loss events: each run of
        instructions on present qubits goes to :meth:`apply_segment`, then the
        loss event after it (``Loss``, ``Reload``, ``Check``,
        ``MeasureCheck``, a Kraus channel, or an operation on a lost qubit)
        is resolved against the live state. :meth:`execute` calls this once
        per shot when :meth:`uses_loss_driver` returns ``True`` and the
        circuit has a loss that can only be resolved while it runs.

        Args:
            state: A state from :meth:`build_state` whose
                :meth:`~State.get_loss_state` works.
            circuit: The prepared circuit.
            lossmodel: The :class:`~mimiqcircuits.LossModel` that decides how
                an operation on partly lost qubits is rewritten.
            rng: The noise generator.
            callback: Called as ``callback(i, total)`` once per instruction
                of ``circuit``, with ``i`` from 1 to ``total``, its length.
                An instruction dropped because its qubits are lost still
                counts, so the total doesn't depend on which qubits are lost.
            stopped: Accepted for symmetry with :meth:`evolve`; not used.

        Returns:
            A tuple ``(state, fidelity)``, where ``fidelity`` is the product
            of the per-step fidelity factors.
        """
        ls = state.get_loss_state()
        fid = 1.0
        insts = list(circuit)
        n = len(insts)
        i = 0
        while i < n:
            start = i
            seg, evidx, i = _next_loss_segment(insts, start, ls)
            if seg:
                fid *= self.apply_segment(state, seg, rng=rng)
            if callback is not None:
                # One tick per source instruction in the run (buffered or
                # dropped), keeping the per-instruction callback contract.
                for k in range(start, n if evidx is None else evidx):
                    callback(k + 1, n)
            if evidx is None:
                break
            ev = insts[evidx]
            fid *= self._resolve_loss_event(state, ev, ls, lossmodel, rng=rng)
            if callback is not None:
                callback(evidx + 1, n)
        return state, fid

    def _resolve_loss_event(self, state, inst, ls, lossmodel, *, rng=None):
        """Resolve a single loss event against the live state and register,
        returning its fidelity factor. The only place loss marks are written.
        """
        from mimiqcircuits.classical.setbits import SetBit0, SetBit1
        from mimiqcircuits.instruction import Instruction
        from mimiqcircuits.operations.losschannel import (
            Check,
            Loss,
            MeasureCheck,
            Reload,
        )
        from mimiqcircuits.operations.measure import Measure
        from mimiqcircuits.operations.reset import Reset
        from mimiqcircuits.lossmodel import lossmodel_rewrite, _loss_fires
        from mimiqcircuits.operations.krauschannel import krauschannel
        from mimiqcircuits.operations.operators.lossyoperator import LossyOperator

        op = inst.get_operation()
        qs = tuple(inst.get_qubits())

        def _apply(o, qubits=(), bits=()):
            return self.apply_instruction(
                state, Instruction(o, tuple(qubits), tuple(bits), ()), rng=rng
            )

        if isinstance(op, Loss):
            q = qs[0]
            if not ls.is_lost(q) and _loss_fires(op, rng):
                ls.mark_lost(q)
            return 1.0

        if isinstance(op, Reload):
            # Mark present before resetting: a backend may guard `Reset` against
            # lost qubits, and a reload must always reset.
            q = qs[0]
            ls.mark_present(q)
            return _apply(Reset(), (q,))

        if isinstance(op, Check):
            b = inst.get_bits()[0]
            _apply(SetBit0() if ls.is_lost(qs[0]) else SetBit1(), (), (b,))
            return 1.0

        if isinstance(op, MeasureCheck):
            bits = tuple(inst.get_bits())
            if ls.is_lost(qs[0]):
                _apply(SetBit0(), (), (bits[0],))
                _apply(SetBit0(), (), (bits[1],))
                return 1.0
            f = _apply(Measure(), (qs[0],), (bits[0],))
            _apply(SetBit1(), (), (bits[1],))
            return f

        if isinstance(op, krauschannel):
            f, fired = self.sample_kraus(state, op, qs, rng=rng)
            if isinstance(fired, LossyOperator):
                for t in fired.lossytargets(*qs):
                    ls.mark_lost(t)
            return f

        if isinstance(op, Measure):
            # a measurement on a lost qubit reads 0
            _apply(SetBit0(), (), (inst.get_bits()[0],))
            return 1.0

        # partially lost -> defer to the shared LossModel decision core
        fid = 1.0
        for r in lossmodel_rewrite(inst, ls.as_map(), lossmodel, rng):
            fid *= self.apply_instruction(state, r, rng=rng)
        return fid

    # ── execute driver ────────────────────────────────────────────────────
    #
    # Backend-agnostic: runs the
    # pass pipeline, detects loss sampling, splits off the classical
    # projection, and routes to either `_execute_sampling` or
    # `_execute_trajectories`. Concrete backends inherit this unchanged.

    def _resolve_pipeline(self, passes, **flags):
        """Resolve the circuit-preparation pipeline for local execution.

        The backend's :meth:`default_passes`, adjusted by the flags, then
        extended with ``passes``. See
        :func:`~mimiqcircuits.backends.passes.resolve_pipeline`.
        """
        return resolve_pipeline(self, passes, **flags)

    # mimiq-qiskit's local estimator calls the old name; keep it until it moves.
    _resolve_prep_passes = _resolve_pipeline

    def execute(
        self,
        circuit,
        *,
        nsamples: int = 1000,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        passes: Optional[PassPipeline] = None,
        fuse: Optional[bool] = None,
        fuse_threshold: Optional[int] = None,
        canonicaldecompose: Optional[bool] = None,
        reorderqubits: Optional[Union[bool, str]] = None,
        remove_swaps: Optional[bool] = None,
        simplify: Optional[bool] = None,
        reorderqubits_seed: Optional[int] = None,
        callback=None,
        param_grid: Optional[list[dict]] = None,
        strict_pass_order: bool = True,
        stopped=None,
        num_qubits: Optional[int] = None,
        progress=False,
    ):
        """Run ``circuit`` on this backend and return its results.

        Takes every argument of :meth:`Backend.execute`, plus the ones
        below. ``circuit`` may also be a
        :class:`~mimiqcircuits.SweepExperiment`; passing one together with
        ``param_grid`` raises :class:`TypeError`, and one with an empty table
        returns ``[]``.

        Args:
            fuse_threshold: Fuse gates only from this many qubits up. On its
                own it also requests fusion; with ``fuse=False`` it raises
                :class:`ValueError`.
            stopped: Passed through unchanged to :meth:`evolve` and
                :meth:`evolve_with_loss`.
            num_qubits: Simulate at least this many qubits. When the circuit
                has no classical bits and runs as a single evolution, the
                samples cover the extra qubits too.
            progress: ``True`` to show progress bars with
                :class:`TqdmProgress`, ``False`` for none, or a
                :class:`Progress` sink.

        Returns:
            See :meth:`Backend.execute`.

        See Also:
            :meth:`Backend.execute` for the shared arguments.
        """
        # A sweep runs through the `param_grid` loop, so it gets the same
        # per-point seeds as the equivalent `param_grid`.
        if isinstance(circuit, SweepExperiment):
            if param_grid:
                raise TypeError(
                    "pass either a SweepExperiment or param_grid=, not both"
                )
            circuit, param_grid = circuit.circuit, list(circuit.table)
            if not param_grid:
                return []

        rngs = self._resolve_rngs(seed, rng)
        passes = self._resolve_pipeline(
            passes,
            fuse=fuse,
            fuse_threshold=fuse_threshold,
            canonicaldecompose=canonicaldecompose,
            reorderqubits=reorderqubits,
            remove_swaps=remove_swaps,
            simplify=simplify,
            reorderqubits_seed=reorderqubits_seed,
        )
        prog = to_progress(progress)

        if isinstance(circuit, list):
            return [
                self._execute_resolved(
                    c, nsamples=nsamples, rngs=rngs, passes=passes,
                    callback=callback, param_grid=param_grid,
                    strict_pass_order=strict_pass_order,
                    stopped=stopped, num_qubits=num_qubits, progress=prog,
                )
                for c in circuit
            ]

        return self._execute_resolved(
            circuit, nsamples=nsamples, rngs=rngs, passes=passes,
            callback=callback, param_grid=param_grid,
            strict_pass_order=strict_pass_order,
            stopped=stopped, num_qubits=num_qubits, progress=prog,
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
        progress,
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
                    stopped=stopped, num_qubits=num_qubits, progress=progress,
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

        runtime_loss = (
            _circuit_has_runtime_loss(processed_circuit) and self.uses_loss_driver()
        )

        # Runtime loss: resolve the state-dependent `LossyOperator` branch live
        # via the shared driver (backends that opt in via `uses_loss_driver`).
        if runtime_loss:
            self._execute_runtime_loss(
                processed_circuit, nsamples, rngs,
                callback, stopped, num_qubits, results, progress,
            )
            results.timings["total"] = time.time() - t_total
            return results

        # Loss sampling: pre-sample a deterministic variant per
        # trajectory so the simulator only ever sees trace-preserving
        # channels. Gate on `runtime_loss` (not bare `_circuit_has_runtime_loss`)
        # so a circuit that mixes pre-resolvable loss with a runtime Kraus on a
        # `:loss` backend that does not use the driver still gets its
        # pre-resolvable loss lowered here.
        if needs_loss_sampling(processed_circuit) and not runtime_loss:
            self._execute_with_loss_sampling(
                processed_circuit, nsamples, rngs,
                callback, stopped, num_qubits, results, progress,
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
            from mimiqcircuits.operations.measure import Measure
            for user_q in range(projection.num_qubits(), num_qubits):
                projection.push(Measure(), user_q, user_q)

        if needs_trajectories(quantum_circuit):
            self._execute_trajectories(
                processed_circuit, nsamples, rngs,
                callback, stopped, num_qubits, results, progress,
            )
        else:
            self._execute_sampling(
                quantum_circuit, projection, processed_circuit, nsamples,
                rngs, callback, stopped, num_qubits, results, progress,
            )

        results.timings["total"] = time.time() - t_total
        return results

    # ── routing helpers (override for surgical changes only) ──────────────

    @staticmethod
    def _count_two_qubit_gates(circuit) -> int:
        """Number of ≥2-qubit gates in ``circuit`` — used to derive
        the per-trajectory average gate error from the circuit
        fidelity."""
        from mimiqcircuits.operations.gates.gate import Gate
        n = 0
        for inst in circuit.instructions:
            op = inst.operation
            if isinstance(op, Gate) and op.num_qubits >= 2:
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
        progress=None,
    ):
        """Pure unitary tail: one evolve, then sample-and-project.

        ``projection`` evaluates per-shot against the raw quantum
        sample to produce the user's `cstate`. Amplitudes and
        expectation values flow through in-circuit ``Amplitude`` /
        ``ExpectationValue`` ops into ``results.zstates``.
        """
        from mimiqcircuits.backends.compiled_projection import CompiledProjection

        progress = progress if progress is not None else NoProgress()

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
        compiled = self.compile_progress(quantum_circuit, progress)
        compiled = self.prepare_trajectory(compiled, rngs.trajectory)
        results.timings["compile"] = time.time() - t_compile

        # Single evolution: drive the execution bar off the backend's
        # per-step callback. The trajectory paths leave the callback
        # unwrapped so the trajectory bar stays the only live bar.
        box = [None]
        stepcb = (
            callback if isinstance(progress, NoProgress)
            else execution_progress_callback(progress, callback, box)
        )

        state = self.build_state(nq, nb_state, nz)
        t_apply = time.time()
        state, fid = self.evolve(
            state, compiled,
            rng=rngs.noise, callback=stepcb, stopped=stopped,
        )
        results.timings["apply"] = time.time() - t_apply
        if box[0] is not None:
            box[0].finish()

        scalar = as_lower_bound(_to_fidelity(fid))
        results.fidelities.append(scalar)
        results.avggateerrors.append(self._avg_gate_error(scalar, num_2q))

        t_sample = time.time()
        samples = state.sample(nsamples, rngs.shot)
        results.cstates.extend(
            CompiledProjection.from_circuit(projection).evaluate_batch(samples)
        )
        results.timings["sample"] = time.time() - t_sample

        if nz > 0:
            results.zstates.append(state.complex_values)

    def _execute_trajectories(
        self, processed_circuit, nsamples, rngs,
        callback, stopped, num_qubits, results, progress=None,
    ):
        """Per-shot evolution: a fresh state per trajectory."""
        progress = progress if progress is not None else NoProgress()
        nq = max(processed_circuit.num_qubits(), num_qubits or 0)
        nb = processed_circuit.num_bits()
        nz = processed_circuit.num_zvars()

        num_2q = self._count_two_qubit_gates(processed_circuit)

        # Compile-once if the backend says the artifact is stable
        # across trajectories; otherwise recompile per shot. The
        # one-shot compile shows a compression bar; per-shot recompiles
        # would redraw it `nsamples` times, so they are left unmuted.
        recompile = self.recompile_per_trajectory(processed_circuit)
        compiled = (
            None if recompile
            else self.compile_progress(processed_circuit, progress)
        )

        # The trajectory loop owns the only live bar; the per-shot
        # execution detail is left unwrapped to keep it the single bar.
        bar = progress.stage("trajectories", total=nsamples)
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
            bar.step()
        bar.finish()
        results.timings["apply"] = t_apply_total

    def _execute_with_loss_sampling(
        self, processed_circuit, nsamples, rngs,
        callback, stopped, num_qubits, results, progress=None,
    ):
        """Method-1 loss resolution: a fresh loss pattern per shot, resolved
        into a primitive (loss-free) circuit variant, then evolved.
        """
        from mimiqcircuits.lossmodel import resolve_losses

        progress = progress if progress is not None else NoProgress()
        nq = max(processed_circuit.num_qubits(), num_qubits or 0)
        nb = processed_circuit.num_bits()
        nz = processed_circuit.num_zvars()
        num_2q = self._count_two_qubit_gates(processed_circuit)

        bar = progress.stage("trajectories", total=nsamples)
        t_apply_total = 0.0
        for _ in range(nsamples):
            sampled = resolve_losses(processed_circuit, rng=rngs.trajectory)
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
            bar.step()
        bar.finish()
        results.timings["apply"] = t_apply_total

    def _execute_runtime_loss(
        self, processed_circuit, nsamples, rngs,
        callback, stopped, num_qubits, results, progress=None,
    ):
        """Per-shot evolution through the shared runtime-loss driver: a fresh
        state and loss register per trajectory, loss resolved live."""
        from mimiqcircuits.lossmodel import LossModel

        progress = progress if progress is not None else NoProgress()
        nq = max(processed_circuit.num_qubits(), num_qubits or 0)
        nb = processed_circuit.num_bits()
        nz = processed_circuit.num_zvars()
        num_2q = self._count_two_qubit_gates(processed_circuit)
        lossmodel = LossModel()

        bar = progress.stage("trajectories", total=nsamples)
        t_apply_total = 0.0
        for _ in range(nsamples):
            state = self.build_state(nq, nb, nz)
            t0 = time.time()
            state, fid = self.evolve_with_loss(
                state, processed_circuit, lossmodel,
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
            bar.step()
        bar.finish()
        results.timings["apply"] = t_apply_total


class RemoteBackend(Backend):
    """Base class for simulators that run on a remote service.

    Subclasses implement :meth:`submit`, plus the :class:`Backend` identity
    members; they may override :meth:`execute`. The inherited
    :meth:`execute` checks the pass order, calls :meth:`submit`, and blocks
    on the returned job's ``wait()``.

    See Also:
        :class:`MimiqRemoteBackend`, the backend for the MIMIQ cloud service.
    """

    @abc.abstractmethod
    def submit(self, circuits, nsamples: int, **kwargs):
        """Send ``circuits`` to the service and return a job handle.

        :meth:`execute` calls it with the keywords ``rngs`` (an
        :class:`RNGs` bundle), ``passes``, ``callback``, ``param_grid`` and
        ``strict_pass_order``.

        Args:
            circuits: A circuit or a list of circuits.
            nsamples: Number of shots.
            **kwargs: The keywords above, and any backend-specific options.

        Returns:
            A job handle with a ``wait()`` method that blocks until the
            results are available and returns them: a
            :class:`~mimiqcircuits.QCSResults`, or a list for a list of
            circuits.
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
        """Submit ``circuit`` with :meth:`submit` and wait for the results.

        Takes the arguments of :meth:`Backend.execute` except the
        preparation flags, which a remote backend may accept in its own
        override.

        Returns:
            Whatever the job's ``wait()`` returns.

        Raises:
            RemotePassOrderError: If ``passes`` is non-empty,
                ``strict_pass_order`` is ``True`` and the backend does not
                declare ``"pass_order_honored"``.
            TypeError: If both ``seed`` and ``rng`` are given.
        """
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
