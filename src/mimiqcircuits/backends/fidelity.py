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

"""Typed fidelity values returned by :meth:`LocalBackend.evolve`.

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
    """Base class for typed fidelity values.

    The subclasses say what a fidelity number means: :class:`ExactFidelity`,
    :class:`UnknownFidelity`, :class:`TruncationLowerBound`,
    :class:`LowerBoundPerStep` and :class:`EstimatedFidelity`.
    :func:`as_lower_bound`, :func:`as_expected` and :func:`as_interval` reduce
    any of them to plain numbers.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import *

        An MPS simulator that discarded 1% of the norm reports a lower
        bound, not an exact value.

        >>> f = TruncationLowerBound(0.99)
        >>> as_lower_bound(f), as_expected(f)
        (0.99, 0.99)

        An estimate with a standard error is reduced conservatively by
        :func:`as_lower_bound` (mean minus three standard errors).

        >>> e = EstimatedFidelity(mean=0.9, stderr=0.01)
        >>> round(as_lower_bound(e), 2), as_expected(e)
        (0.87, 0.9)
        >>> as_lower_bound(UnknownFidelity())
        nan
    """


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
    """A single lower bound on the fidelity of the whole circuit.

    Typical for MPS and MPO simulators, where the bound comes from the
    singular values discarded during compression. Return it even when the
    bound is exactly ``1.0``: a plain ``1.0`` from :meth:`LocalBackend.evolve`
    is read as :class:`ExactFidelity`.

    Attributes:
        value (float): The lower bound.
    """

    value: float


@dataclass(frozen=True)
class LowerBoundPerStep(Fidelity):
    """Per-step lower-bound contributions.

    :func:`as_lower_bound` returns the product of the values, each clipped
    to ``[0, 1]``. That product bounds the circuit fidelity only when
    successive truncation errors are uncorrelated; a backend that cannot
    assume this returns a single :class:`TruncationLowerBound` instead.

    Attributes:
        values (tuple[float, ...]): One contribution per step.
    """

    values: tuple[float, ...]


@dataclass(frozen=True)
class EstimatedFidelity(Fidelity):
    """Sample-based fidelity estimate with a standard error.

    Use it for randomised benchmarking, direct fidelity estimation, or
    cross-entropy benchmarking. :func:`as_lower_bound` returns
    ``max(0, mean - 3 * stderr)`` and :func:`as_interval` returns
    ``(mean - stderr, mean + stderr)``.

    Attributes:
        mean (float): The estimate.
        stderr (float): Its standard error.
    """

    mean: float
    stderr: float


# ──────────────────────────────────────────────────────────────────────────
# Reducers
# ──────────────────────────────────────────────────────────────────────────


def as_lower_bound(f: Fidelity) -> float:
    """Reduce ``f`` to a conservative lower bound.

    Args:
        f: The fidelity.

    Returns:
        ``1.0`` for :class:`ExactFidelity`; ``nan`` for
        :class:`UnknownFidelity`, so that statistics over many results do
        not count an unknown fidelity as a good one; the value for
        :class:`TruncationLowerBound`; the product of the clipped values for
        :class:`LowerBoundPerStep`; ``max(0, mean - 3 * stderr)`` for
        :class:`EstimatedFidelity`.

    Raises:
        TypeError: If ``f`` is not one of these types.
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
    """Reduce ``f`` to a central estimate.

    Args:
        f: The fidelity.

    Returns:
        The same as :func:`as_lower_bound`, except for
        :class:`EstimatedFidelity`, which gives its mean.

    Raises:
        TypeError: If ``f`` is not a known :class:`Fidelity` type.
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
    """Reduce ``f`` to a ``(low, high)`` interval.

    Args:
        f: The fidelity.

    Returns:
        ``(mean - stderr, mean + stderr)`` for :class:`EstimatedFidelity`,
        otherwise ``(as_lower_bound(f), as_expected(f))``.

    Raises:
        TypeError: If ``f`` is not a known :class:`Fidelity` type.
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
