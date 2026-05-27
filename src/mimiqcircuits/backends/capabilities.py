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

"""Capability vocabulary, numeric limits, topology, and admission types.

A backend's :meth:`Backend.capabilities` returns a set of
:data:`Capability` tokens drawn from :data:`CAPABILITY_VOCABULARY`.
A backend's :meth:`Backend.limits` returns a :class:`Limits`
instance, and :meth:`Backend.topology` returns one of the
:class:`Topology` subclasses (:class:`AllToAll`,
:class:`CouplingMap`, :class:`LinearChain`).

:meth:`Backend.can_handle` returns an :class:`AdmissionResult`
(:class:`Admissible`, :class:`Marginal`, or :class:`Inadmissible`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────────
# Capability vocabulary
# ──────────────────────────────────────────────────────────────────────────

Capability = str

#: Canonical capability tokens a backend may advertise. Informational:
#: backends may declare tokens outside this set; conformance tests warn
#: but never fail on extras.
CAPABILITY_VOCABULARY: frozenset[Capability] = frozenset(
    [
        # core ops
        "amplitude",
        "sampling",
        "classical_bits",
        "zvars",
        # measurement
        "midcircuit_measure",
        "midcircuit_reset",
        "final_measure_only",
        "reset_after_measure",
        # classical control
        "feed_forward",
        "while_statement",
        # noise
        "noise",
        "calibrated_noise",
        "loss",
        # observables
        "expectation_1q",
        "expectation_2q",
        "expectation_paulistring",
        "expectation_state",
        # tensor-network
        "bond_dim",
        "schmidt_rank",
        "streaming",
        # parameters
        "parametric",
        "parametric_batch",
        # batching
        "batch",
        "shared_prefix_batch",
        # pass plumbing
        "pass_order_honored",
    ]
)


# ──────────────────────────────────────────────────────────────────────────
# Limits
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Limits:
    """Numeric limits a simulator advertises.

    `None` for any field means "unbounded" (or "not applicable" for fields
    like `max_bond_dim` on a state-vector simulator).
    """

    max_qubits: Optional[int] = None
    max_bond_dim: Optional[int] = None
    max_classical_bits: Optional[int] = None
    max_zvars: Optional[int] = None
    max_samples: Optional[int] = None
    max_circuit_depth: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────
# Topology
# ──────────────────────────────────────────────────────────────────────────


class Topology:
    """Base class for device / connectivity topologies."""


@dataclass(frozen=True)
class AllToAll(Topology):
    """All-to-all connectivity (no constraint). Default for simulators."""


@dataclass(frozen=True)
class CouplingMap(Topology):
    """Coupling map: each edge ``(i, j)`` means qubits ``i`` and ``j`` can
    interact directly. 1-based qubit indices to match the rest of MIMIQ.
    """

    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LinearChain(Topology):
    """Linear chain of ``n`` qubits with nearest-neighbor coupling."""

    n: int


# ──────────────────────────────────────────────────────────────────────────
# AdmissionResult
# ──────────────────────────────────────────────────────────────────────────


class AdmissionResult:
    """Base class for ``can_handle(backend, circuit)`` results."""


@dataclass(frozen=True)
class Admissible(AdmissionResult):
    """The backend can execute the circuit."""


@dataclass(frozen=True)
class Marginal(AdmissionResult):
    """The backend can execute the circuit but it is near a limit
    (memory, bond dimension, runtime). ``warning`` is a user-facing
    message; downstream code still treats this as admissible.
    """

    warning: str


@dataclass(frozen=True)
class Inadmissible(AdmissionResult):
    """The backend cannot execute the circuit."""

    reason: str


def is_admissible(result: AdmissionResult) -> bool:
    """``True`` for :class:`Admissible` and :class:`Marginal`."""
    return isinstance(result, (Admissible, Marginal))


# ──────────────────────────────────────────────────────────────────────────
# UnsupportedCapabilityError
# ──────────────────────────────────────────────────────────────────────────


class UnsupportedCapabilityError(Exception):
    """Raised when a backend is asked to run a feature it does not
    advertise in :meth:`Backend.capabilities`.

    Backends use this to fail loudly rather than silently degrade.
    Tests rely on it to verify a backend's capability set is honest
    (no "positive lies").
    """

    def __init__(self, backend_name: str, capability: Capability, detail: str = ""):
        self.backend_name = backend_name
        self.capability = capability
        self.detail = detail
        super().__init__(
            f"UnsupportedCapabilityError: backend {backend_name} does not "
            f"declare :{capability}"
            + (f" ({detail})" if detail else "")
        )


# ──────────────────────────────────────────────────────────────────────────
# CAP_PROBES — anti-capability conformance harness registry
# ──────────────────────────────────────────────────────────────────────────


#: Registry of probe-circuit factories keyed by capability. Populated
#: lazily by callers (typically the conformance test module) so that
#: importing this module does not drag in the full circuit-construction
#: surface.
CAP_PROBES: dict[Capability, Callable] = {}


def register_cap_probe(capability: Capability, factory: Callable):
    """Register a probe-circuit constructor for ``capability``.

    ``factory()`` must return a Circuit whose execution requires the
    named capability. Conformance tests use this registry to assert
    that backends reject what they do not advertise — silently
    accepting an undeclared feature would let "positive lies" slip
    through.
    """
    CAP_PROBES[capability] = factory
    return factory
