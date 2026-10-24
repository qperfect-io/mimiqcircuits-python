#
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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
"""Ross–Selinger gridsynth: Clifford+T approximation of Z-rotations.

The synthesis itself is implemented in Rust in the ``mimiqcircuits._core``
extension.

References:
    - N. J. Ross and P. Selinger, "Optimal ancilla-free Clifford+T
      approximation of z-rotations", Quantum Inf. Comput. 16, 901 (2016),
      arXiv:1403.2975.
"""

from __future__ import annotations

import math
import numbers
import operator
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from mimiqcircuits._core import gridsynth as _gridsynth
from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Operation

_MASK64 = (1 << 64) - 1

# Smaller tolerances still work but take seconds per rotation, and a float
# angle is only known to about 1e-16 anyway.
_GRIDSYNTH_MIN_EPSILON = 1e-100

# Circuits often repeat the same angle (QFT, Trotter steps), and synthesis
# costs milliseconds or more per angle, so results are memoized. The cache
# is cleared rather than evicted when full; entries are cheap to recompute.
_GRIDSYNTH_CACHE: dict[tuple[float, float, int], str] = {}
_GRIDSYNTH_CACHE_LOCK = threading.Lock()
_GRIDSYNTH_CACHE_MAX = 10_000


def _gridsynth_word(epsilon: float, seed: int, theta: float) -> str:
    key = (theta, epsilon, seed)
    with _GRIDSYNTH_CACHE_LOCK:
        cached = _GRIDSYNTH_CACHE.get(key)
    if cached is not None:
        return cached
    word = _gridsynth(theta, epsilon, seed)
    with _GRIDSYNTH_CACHE_LOCK:
        if len(_GRIDSYNTH_CACHE) >= _GRIDSYNTH_CACHE_MAX:
            _GRIDSYNTH_CACHE.clear()
        _GRIDSYNTH_CACHE[key] = word
    return word


@dataclass(frozen=True)
class GridSynthRewrite(RewriteRule):
    """Rewrite rule that approximates ``GateRZ`` with a Clifford+T sequence,
    using the Ross–Selinger gridsynth algorithm.

    For ``GateRZ(λ)`` the rule emits a sequence of ``GateH``, ``GateS``,
    ``GateSDG``, ``GateZ``, ``GateT`` and ``GateX`` whose unitary ``V``
    satisfies ``min_φ ‖RZ(λ) − e^{iφ}V‖ ≤ epsilon`` in operator norm. The
    approximation holds up to a global phase, which is dropped. The T-count
    is about ``3·log₂(1/epsilon)`` and, except for rare failures of the
    integer factoring step, optimal among ancilla-free Clifford+T circuits
    with this error. Angles that are multiples of ``π/4`` give their exact
    short decomposition.

    The factoring step is randomized. For a fixed ``seed``, the same angle and
    tolerance always give the same sequence. Results are cached per angle,
    since circuits often repeat the same rotation.

    The synthesis runs in the compiled ``mimiqcircuits._core`` extension
    without holding the GIL, so several threads can synthesize in parallel.

    Args:
        epsilon: Operator-norm tolerance, ``1e-100 <= epsilon < 1`` (default
            ``1e-10``). A rotation takes a few milliseconds at ``1e-10`` and
            up to about a second at ``1e-100``.
        seed: Seed of the randomized factoring step, an integer in
            ``[0, 2**64)`` (default ``0``).

    Raises:
        ValueError: If ``epsilon`` or ``seed`` is out of range.
        TypeError: If ``epsilon`` is not a real number or ``seed`` is not an
            integer.

    Supported operations:
        - ``GateRZ(λ)`` with a numeric angle ``λ``.

    Symbolic angles are not matched. ``GateRX``, ``GateRY`` and ``GateU``
    must first be reduced to Z-rotations, for example with
    :class:`ZYZRewrite` and :class:`ToZRotationRewrite`.
    :class:`CliffordTBasis` does this and uses this rule by default.

    Example:
        >>> from mimiqcircuits import GateRZ, GateT
        >>> from mimiqcircuits.decomposition import GridSynthRewrite
        >>> def tcount(c):
        ...     return sum(isinstance(i.operation, GateT) for i in c)
        >>> # Approximate one rotation to the default tolerance 1e-10
        >>> circ = GridSynthRewrite().decompose_step(GateRZ(0.123), [0], [], [])
        >>> # A coarser tolerance gives fewer T gates
        >>> coarse = GridSynthRewrite(epsilon=1e-4).decompose_step(GateRZ(0.123), [0], [], [])
        >>> tcount(coarse) < tcount(circ)
        True

    References:
        N. J. Ross and P. Selinger, "Optimal ancilla-free Clifford+T
        approximation of z-rotations", Quantum Inf. Comput. 16, 901 (2016),
        arXiv:1403.2975.
    """

    epsilon: float = 1e-10
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.epsilon, numbers.Real):
            raise TypeError(f"epsilon must be a real number, got {self.epsilon!r}")
        epsilon = float(self.epsilon)
        if not _GRIDSYNTH_MIN_EPSILON <= epsilon < 1:
            raise ValueError(
                f"epsilon must be in [{_GRIDSYNTH_MIN_EPSILON}, 1), got {self.epsilon}"
            )
        seed = operator.index(self.seed)
        if not 0 <= seed <= _MASK64:
            raise ValueError(f"seed must be in [0, 2**64), got {self.seed}")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "seed", seed)

    def matches(self, op: Operation) -> bool:
        """Check if this rule can decompose the operation.

        Args:
            op: The operation to check.

        Returns:
            True if the operation is a ``GateRZ`` with a numeric angle.
        """
        import mimiqcircuits as mc

        return isinstance(op, mc.GateRZ) and not op.is_symbolic()

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose the Z-rotation into Clifford+T gates.

        Args:
            op: The ``GateRZ`` to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A Circuit containing the Clifford+T approximation.

        Raises:
            DecompositionError: If the operation is not a numeric ``GateRZ``.
            ValueError: If the angle is not finite.
        """
        import mimiqcircuits as mc

        if not self.matches(op):
            raise DecompositionError(
                f"GridSynthRewrite cannot decompose {type(op).__name__}"
            )
        theta = float(op.lmbda)
        if not math.isfinite(theta):
            raise ValueError(f"GridSynthRewrite: angle {theta} is not finite")

        word = _gridsynth_word(self.epsilon, self.seed, theta)

        circ = mc.Circuit()
        q = qubits[0]
        # The word is in matrix-product order, so the last letter acts first.
        i = len(word) - 1
        while i >= 0:
            c = word[i]
            if c == "S":
                n = 0
                while i >= 0 and word[i] == "S":
                    n += 1
                    i -= 1
                n &= 3
                if n == 1:
                    circ.push(mc.GateS(), q)
                elif n == 2:
                    circ.push(mc.GateZ(), q)
                elif n == 3:
                    circ.push(mc.GateSDG(), q)
                continue
            if c == "H":
                circ.push(mc.GateH(), q)
            elif c == "T":
                circ.push(mc.GateT(), q)
            elif c == "X":
                circ.push(mc.GateX(), q)
            i -= 1
        return circ


__all__ = ["GridSynthRewrite"]
