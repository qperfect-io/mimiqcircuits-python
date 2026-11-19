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

A backend's :meth:`Backend.capabilities` returns a set of capability
tokens drawn from :data:`CAPABILITY_VOCABULARY`.
A backend's :meth:`Backend.limits` returns a :class:`Limits`
instance, and :meth:`Backend.topology` returns one of the
:class:`Topology` subclasses (:class:`AllToAll`,
:class:`CouplingMap`, :class:`LinearChain`).

:meth:`Backend.can_handle` returns an :class:`AdmissionResult`
(:class:`Admissible`, :class:`Marginal`, or :class:`Inadmissible`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────────
# Capability vocabulary
# ──────────────────────────────────────────────────────────────────────────

#: Type of a capability token: a string such as ``"noise"``.
Capability = str

#: Capability tokens a backend may declare in :meth:`Backend.capabilities`.
#: Informational: backends may declare tokens outside this set; conformance
#: tests warn but never fail on extras.
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
    """Numeric limits a backend declares in :meth:`Backend.limits`.

    ``None`` for a field means unbounded, or not applicable (such as
    ``max_bond_dim`` on a state-vector simulator). The default
    :meth:`Backend.can_handle` checks ``max_qubits``,
    ``max_classical_bits`` and ``max_zvars``; the other fields are
    informational.

    Attributes:
        max_qubits (int or None): Largest number of qubits.
        max_bond_dim (int or None): Largest bond dimension.
        max_classical_bits (int or None): Largest number of classical bits.
        max_zvars (int or None): Largest number of z-variables.
        max_samples (int or None): Largest number of shots per execution.
        max_circuit_depth (int or None): Largest circuit depth.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import Limits
        >>> lim = Limits(max_qubits=30)
        >>> lim.max_qubits, lim.max_samples
        (30, None)
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
    """Base class for the qubit connectivity a backend declares in
    :meth:`Backend.topology`."""


@dataclass(frozen=True)
class AllToAll(Topology):
    """Every pair of qubits can interact. The default for simulators."""


@dataclass(frozen=True)
class CouplingMap(Topology):
    """Connectivity given as a list of edges.

    Each edge ``(i, j)`` means qubits ``i`` and ``j`` can interact directly.
    Indices are 1-based.

    Attributes:
        edges (tuple[tuple[int, int], ...]): The coupled pairs.
    """

    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LinearChain(Topology):
    """Linear chain of ``n`` qubits, each coupled to its neighbours.

    Attributes:
        n (int): Number of qubits.
    """

    n: int


# ──────────────────────────────────────────────────────────────────────────
# AdmissionResult
# ──────────────────────────────────────────────────────────────────────────


class AdmissionResult:
    """Base class for the results of :meth:`Backend.can_handle`."""


@dataclass(frozen=True)
class Admissible(AdmissionResult):
    """The backend can execute the circuit."""


@dataclass(frozen=True)
class Marginal(AdmissionResult):
    """The backend can execute the circuit, but it is near a limit
    (memory, bond dimension, runtime).

    :func:`is_admissible` treats it as admissible.

    Attributes:
        warning (str): Message for the user.
    """

    warning: str


@dataclass(frozen=True)
class Inadmissible(AdmissionResult):
    """The backend cannot execute the circuit.

    Attributes:
        reason (str): Why the circuit is rejected.
    """

    reason: str


def is_admissible(result: AdmissionResult) -> bool:
    """Return whether an admission result accepts the circuit.

    Args:
        result: The result of :meth:`Backend.can_handle`.

    Returns:
        ``True`` for :class:`Admissible` and :class:`Marginal`, ``False``
        otherwise.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import (
        ...     Admissible, Inadmissible, Marginal, is_admissible)
        >>> is_admissible(Admissible())
        True
        >>> is_admissible(Marginal(warning="close to the memory limit"))
        True
        >>> is_admissible(Inadmissible(reason="too many qubits"))
        False
    """
    return isinstance(result, (Admissible, Marginal))


# ──────────────────────────────────────────────────────────────────────────
# UnsupportedCapabilityError
# ──────────────────────────────────────────────────────────────────────────


class UnsupportedCapabilityError(Exception):
    """Error for a feature the backend does not declare in
    :meth:`Backend.capabilities`.

    Backends raise it when asked to run such a feature, instead of running
    something else in its place.

    Args:
        backend_name: Name of the backend.
        capability: The capability token the circuit needs.
        detail: Optional extra text for the message.
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


#: Registry of probe-circuit factories keyed by capability, used by the
#: conformance tests to check that a backend rejects what it does not
#: declare. Importing :mod:`mimiqcircuits.backends` registers the default
#: probes; :func:`register_cap_probe` adds more.
CAP_PROBES: dict[Capability, Callable] = {}


def register_cap_probe(capability: Capability, factory: Callable):
    """Register a probe-circuit factory for ``capability`` in
    :data:`CAP_PROBES`.

    Conformance tests run each probe on every backend that does not declare
    its capability, and check that the backend rejects it. A factory
    registered for a capability that already has one replaces it.

    Args:
        capability: The capability token.
        factory: A function of no arguments that returns a circuit whose
            execution needs ``capability``.

    Returns:
        ``factory``, unchanged.
    """
    CAP_PROBES[capability] = factory
    return factory
