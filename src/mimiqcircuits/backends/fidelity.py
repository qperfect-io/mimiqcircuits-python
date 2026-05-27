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

"""Typed fidelity values returned by :meth:`Backend.evolve`.

Different simulators report fidelity with different semantics: a
state-vector backend can claim exactness, an MPS backend can only
return a truncation lower bound, and a randomised-benchmarking
backend can return a sample-based estimate with a standard error.

A plain ``float`` return would silently lose those distinctions.
The tagged variants below preserve them:

- :class:`ExactFidelity`
- :class:`UnknownFidelity`
- :class:`TruncationLowerBound`
- :class:`LowerBoundPerStep`
- :class:`EstimatedFidelity`

Reduce to plain numbers via :func:`as_lower_bound` (conservative),
:func:`as_expected` (central), or :func:`as_interval` (``(lo, hi)``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class Fidelity:
    """Base class for typed fidelity values. See the module docstring
    for the list of variants and the reducer functions."""


@dataclass(frozen=True)
class ExactFidelity(Fidelity):
    """The simulator believes the state is exact.

    BLAS and SIMD round-off are not tracked; this is a structural
    claim about the algorithm, not a bit-perfect guarantee.
    """


@dataclass(frozen=True)
class UnknownFidelity(Fidelity):
    """The backend does not track fidelity.

    Reducers return ``nan``. Prefer this variant over inventing a
    placeholder value when you genuinely do not know.
    """


@dataclass(frozen=True)
class TruncationLowerBound(Fidelity):
    """A single scalar lower bound on the whole-circuit fidelity.

    Typical for MPS / MPO simulators, where ``value`` is the product
    of singular-value tails discarded during compression.
    """

    value: float


@dataclass(frozen=True)
class LowerBoundPerStep(Fidelity):
    """Per-step lower-bound contributions.

    ``as_lower_bound`` returns ``prod(clip(values, 0, 1))``, which is
    a lower bound on the circuit fidelity *only when successive
    truncation errors are uncorrelated*. If your backend cannot make
    that assumption, collapse to a single
    :class:`TruncationLowerBound` at construction time.
    """

    values: tuple[float, ...]


@dataclass(frozen=True)
class EstimatedFidelity(Fidelity):
    """Sample-based fidelity estimate with a standard error.

    Use for randomised benchmarking, direct fidelity estimation, or
    cross-entropy benchmarking. ``as_lower_bound`` returns
    ``max(0, mean − 3·stderr)``; ``as_interval`` returns the ±1σ band.
    """

    mean: float
    stderr: float


# ──────────────────────────────────────────────────────────────────────────
# Reducers
# ──────────────────────────────────────────────────────────────────────────


def as_lower_bound(f: Fidelity) -> float:
    """Reduce ``f`` to a conservative scalar lower bound.

    Returns ``nan`` when ``f`` is :class:`UnknownFidelity` so that
    statistics over mixed-type fidelities do not silently treat
    "unknown" as "good".
    """
    if isinstance(f, ExactFidelity):
        return 1.0
    if isinstance(f, UnknownFidelity):
        return math.nan
    if isinstance(f, TruncationLowerBound):
        return f.value
    if isinstance(f, LowerBoundPerStep):
        prod = 1.0
        for v in f.values:
            prod *= max(0.0, min(1.0, v))
        return prod
    if isinstance(f, EstimatedFidelity):
        return max(0.0, f.mean - 3 * f.stderr)
    raise TypeError(f"Unknown Fidelity subtype: {type(f).__name__}")


def as_expected(f: Fidelity) -> float:
    """Reduce ``f`` to a central / expected scalar.

    For :class:`TruncationLowerBound` the lower bound *is* the
    estimate; for :class:`EstimatedFidelity` it is the sample mean.
    Returns ``nan`` for :class:`UnknownFidelity`.
    """
    if isinstance(f, ExactFidelity):
        return 1.0
    if isinstance(f, UnknownFidelity):
        return math.nan
    if isinstance(f, TruncationLowerBound):
        return f.value
    if isinstance(f, LowerBoundPerStep):
        prod = 1.0
        for v in f.values:
            prod *= max(0.0, min(1.0, v))
        return prod
    if isinstance(f, EstimatedFidelity):
        return f.mean
    raise TypeError(f"Unknown Fidelity subtype: {type(f).__name__}")


def as_interval(f: Fidelity) -> tuple[float, float]:
    """``(lo, hi)`` band. ±1σ for :class:`EstimatedFidelity`; otherwise
    ``(as_lower_bound, as_expected)``.
    """
    if isinstance(f, EstimatedFidelity):
        return (f.mean - f.stderr, f.mean + f.stderr)
    return (as_lower_bound(f), as_expected(f))


# ──────────────────────────────────────────────────────────────────────────
# Coercion helper (private)
# ──────────────────────────────────────────────────────────────────────────


def _to_fidelity(x) -> Fidelity:
    """Coerce a stand-in scalar into a typed :class:`Fidelity`.

    - ``None``  → :class:`UnknownFidelity`
    - ``1.0``   → :class:`ExactFidelity`
    - other float-likes → :class:`TruncationLowerBound`
    - existing :class:`Fidelity` instances pass through unchanged.

    Transitional adapter for backends still returning plain floats.

    WARNING: a value of exactly ``1.0`` re-types as
    :class:`ExactFidelity`. MPS-style truncation lower bounds that
    happen to land at 1.0 should be wrapped explicitly with
    :class:`TruncationLowerBound` to keep their semantics.
    """
    if x is None:
        return UnknownFidelity()
    if isinstance(x, Fidelity):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise TypeError(f"cannot coerce {x!r} to Fidelity")
    return ExactFidelity() if v == 1.0 else TruncationLowerBound(v)
