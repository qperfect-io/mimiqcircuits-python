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

"""Seed / RNG helpers shared by every backend wrapper.

Two helpers live here:

- :func:`normalize_seed` reconciles a caller's ``seed=`` and
  ``rngs=`` arguments into a single ``int`` (or ``None``) suitable
  for forwarding to a downstream simulator that accepts only one
  seed value.
- :func:`derive_grid_seeds` deterministically produces ``n``
  distinct seeds from one base seed, used by parameter-grid loops
  so each grid point gets an independent RNG stream.

The XOR-fold of an :class:`RNGs` bundle into one int is a
compromise: the underlying simulator PRNGs (Rust ``Rng``, Julia
``MersenneTwister``) take a single seed, so the four logical
streams collapse to one PRNG until per-stream seeding is plumbed
end-to-end.
"""

from __future__ import annotations

from typing import Optional, Union

from mimiqcircuits.backends.backend import RNGs


_MASK64 = (1 << 64) - 1
_MASK63 = (1 << 63) - 1


def normalize_seed(
    seed: Optional[int], rngs: Union[RNGs, int, None]
) -> Optional[int]:
    """Reconcile the legacy ``seed=`` and the new ``rngs=`` kwargs.

    Either may be provided, never both. ``rngs`` may be an ``int``
    (passed through), an :class:`RNGs` bundle (xor-folded to one
    seed), or ``None`` (returns the ``seed`` argument).
    """
    if seed is not None and rngs is not None:
        raise TypeError("pass either seed= or rngs=, not both")
    if rngs is None:
        return seed
    if isinstance(rngs, int):
        return rngs
    if isinstance(rngs, RNGs):
        return (
            rngs.shot.getrandbits(63)
            ^ rngs.noise.getrandbits(63)
            ^ rngs.trajectory.getrandbits(63)
            ^ rngs.pass_.getrandbits(63)
        )
    raise TypeError(f"rngs must be RNGs, int, or None; got {type(rngs).__name__}")


def _splitmix64(x: int) -> int:
    """SplitMix64 finaliser. Consecutive ``x`` values produce
    decorrelated 63-bit outputs, which a plain LCG mix cannot
    guarantee — `(base*c1 + i*c2)` collapses to a constant at
    ``base_seed == 0`` and gives every grid point the same stream.
    """
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK63


def derive_grid_seeds(
    base_seed: Optional[int], n: int
) -> list[Optional[int]]:
    """Deterministically derive ``n`` distinct seeds from ``base_seed``.

    The ``param_grid`` loop calls this so each parameter point gets a
    different RNG stream; reusing ``base_seed`` directly would tie
    measurement outcomes across grid points. A ``None`` ``base_seed``
    propagates as ``[None] * n`` (caller wants nondeterminism).
    """
    if base_seed is None:
        return [None] * n
    return [_splitmix64(base_seed + i) for i in range(n)]
