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

- :func:`normalize_seed` turns a caller's ``seed`` or ``rngs`` into a
  single ``int`` (or ``None``) for a simulator that takes one seed.
- :func:`derive_grid_seeds` produces ``n`` distinct seeds from one base
  seed, so each point of a parameter grid gets its own random stream.

Folding an :class:`RNGs` bundle into one integer merges its four streams
into one, because those simulators take a single seed.
"""

from __future__ import annotations

from typing import Optional, Union

from mimiqcircuits.backends.backend import RNGs


_MASK64 = (1 << 64) - 1
_MASK63 = (1 << 63) - 1


def normalize_seed(
    seed: Optional[int], rngs: Union[RNGs, int, None]
) -> Optional[int]:
    """Turn ``seed`` or ``rngs`` into a single integer seed.

    Args:
        seed: An integer seed, or ``None``.
        rngs: An integer, returned as it is; an :class:`RNGs` bundle, folded
            into one seed by XOR-ing one 63-bit draw from each of its four
            streams (which advances them); or ``None``, in which case
            ``seed`` is returned.

    Returns:
        The seed, or ``None`` when both arguments are ``None``.

    Raises:
        TypeError: If both are given, or ``rngs`` has another type.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import RNGs, normalize_seed
        >>> normalize_seed(7, None)
        7
        >>> normalize_seed(None, RNGs.from_seed(7)) == normalize_seed(None, RNGs.from_seed(7))
        True
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


# Seeds derived from one master are `_splitmix64(scrambled + k)`. Scrambling
# the master first keeps nearby masters from sharing seeds, which adding `k`
# to the master itself would not. `k` is 1 to 4 for the four `RNGs` streams
# and starts at 5 for grid points, so the two never meet.
_NUM_STREAMS = 4


def _stream_seeds(master: int) -> tuple[int, int, int, int]:
    """The seeds of the four :class:`RNGs` streams under ``master``, in
    field order."""
    scrambled = _splitmix64(master)
    return tuple(
        _splitmix64(scrambled + k) for k in range(1, _NUM_STREAMS + 1)
    )


def derive_grid_seeds(
    base_seed: Optional[int], n: int
) -> list[Optional[int]]:
    """Derive ``n`` distinct seeds from ``base_seed``.

    The ``param_grid`` loop of :meth:`LocalBackend.execute` uses it so that
    each point gets its own random stream; reusing ``base_seed`` for every
    point would correlate their outcomes. The result depends only on
    ``base_seed`` and the position, and two different base seeds, however
    close, have no seed in common.

    Args:
        base_seed: The base seed, or ``None``.
        n: Number of seeds.

    Returns:
        A list of ``n`` 63-bit seeds, or ``[None] * n`` when ``base_seed``
        is ``None``.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import derive_grid_seeds
        >>> seeds = derive_grid_seeds(42, 3)
        >>> len(set(seeds)), seeds == derive_grid_seeds(42, 3)
        (3, True)
        >>> derive_grid_seeds(None, 2)
        [None, None]
    """
    if base_seed is None:
        return [None] * n
    scrambled = _splitmix64(base_seed)
    return [
        _splitmix64(scrambled + _NUM_STREAMS + 1 + i) for i in range(n)
    ]
