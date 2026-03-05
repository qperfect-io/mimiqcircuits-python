#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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
"""Solovay-Kitaev algorithm for approximating arbitrary unitaries with Clifford+T.

The Solovay-Kitaev algorithm recursively refines an initial approximation using
group commutators, achieving ε-approximation with O(log^c(1/ε)) gates where c ≈ 3.97.

References:
    - Dawson & Nielsen, "The Solovay-Kitaev algorithm" (2005)
    - Kitaev, Shen, Vyalyi, "Classical and Quantum Computation" (2002)
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.spatial import KDTree

from mimiqcircuits.decomposition.abstract import DecompositionError, RewriteRule

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, Gate, Operation

# =============================================================================
# Constants
# =============================================================================

SK_NET_MAX_DEPTH = 10
SK_NET_MAX_POINTS = 50_000
SK_NET_MIN_DIST = 0.02

# =============================================================================
# SU(2) Utilities
# =============================================================================


def to_su2(U: np.ndarray) -> np.ndarray:
    """Project a 2x2 unitary matrix to SU(2) by removing the global phase.

    Args:
        U: A 2x2 unitary matrix.

    Returns:
        The SU(2) projection of U (determinant = 1).
    """
    det = np.linalg.det(U)
    phase = cmath.sqrt(det)
    return U / phase


def matrix_to_point(U: np.ndarray) -> np.ndarray:
    """Map an SU(2) matrix to a point in R^4 (the 3-sphere S³).

    For SU(2), a matrix [[a, -b*], [b, a*]] is determined by its first column [a, b].
    We map this to [Re(a), Im(a), Re(b), Im(b)] ∈ S³.

    Args:
        U: An SU(2) matrix.

    Returns:
        A 4D point on the unit sphere.
    """
    a, b = U[0, 0], U[1, 0]
    return np.array([a.real, a.imag, b.real, b.imag], dtype=np.float64)


def operator_norm_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Compute the Frobenius norm distance ||U - V||_F.

    For SU(2) matrices, this is an upper bound on the operator norm
    and is tight up to a constant factor.

    Args:
        U: First SU(2) matrix.
        V: Second SU(2) matrix.

    Returns:
        The Frobenius norm of the difference.
    """
    return np.linalg.norm(U - V, "fro")


def extract_rotation_angle(U: np.ndarray) -> float:
    """Extract the rotation angle θ from an SU(2) matrix U = exp(iθ n·σ/2).

    Args:
        U: An SU(2) matrix.

    Returns:
        The rotation angle θ ∈ [0, 2π].
    """
    # For U = cos(θ/2)I + i·sin(θ/2)(n·σ), we have tr(U) = 2cos(θ/2)
    cos_half = np.trace(U).real / 2
    cos_half = np.clip(cos_half, -1.0, 1.0)
    return 2 * math.acos(cos_half)


def extract_rotation_axis(U: np.ndarray) -> np.ndarray:
    """Extract the rotation axis n from an SU(2) matrix U = exp(iθ n·σ/2).

    Args:
        U: An SU(2) matrix.

    Returns:
        A unit vector in R³. If U ≈ I, returns [0, 0, 1].
    """
    # U = cos(θ/2)I + i⋅sin(θ/2)(n⋅σ)
    # n_k = Im(Tr(U σ_k)) / (2 sin(θ/2))
    nx = (U[0, 1] + U[1, 0]).imag / 2  # sin(θ/2)·nx
    ny = (U[0, 1] - U[1, 0]).real / 2  # sin(θ/2)·ny
    nz = (U[0, 0] - U[1, 1]).imag / 2  # sin(θ/2)·nz

    n = np.array([nx, ny, nz], dtype=np.float64)
    norm_n = np.linalg.norm(n)

    if norm_n < 1e-10:
        # U ≈ I, axis is undefined; return default
        return np.array([0.0, 0.0, 1.0])

    return n / norm_n


def axis_angle_to_su2(axis: np.ndarray, theta: float) -> np.ndarray:
    """Construct an SU(2) matrix from rotation axis and angle.

    U = cos(θ/2)·I + i·sin(θ/2)·(n·σ)

    Args:
        axis: Unit vector for rotation axis.
        theta: Rotation angle.

    Returns:
        The SU(2) rotation matrix.
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    nx, ny, nz = axis

    return np.array(
        [
            [c + 1j * s * nz, s * (1j * nx + ny)],
            [s * (1j * nx - ny), c - 1j * s * nz],
        ],
        dtype=np.complex128,
    )


def find_orthogonal_axes(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find two orthonormal vectors v, w perpendicular to n.

    Args:
        n: A unit vector.

    Returns:
        Two unit vectors (v, w) such that v×w = n (right-handed).
    """
    # Choose initial vector not parallel to n
    if abs(n[0]) < 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    else:
        seed = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt
    v = seed - np.dot(seed, n) * n
    v = v / np.linalg.norm(v)

    w = np.cross(v, n)
    w = w / np.linalg.norm(w)

    return v, w


# =============================================================================
# Gate Sequence Operations
# =============================================================================


def get_basic_gates() -> tuple:
    """Get the basic Clifford+T gates for the epsilon-net."""
    import mimiqcircuits as mc

    return (
        mc.GateH(),
        mc.GateT(),
        mc.GateTDG(),
        mc.GateS(),
        mc.GateSDG(),
        mc.GateX(),
        mc.GateY(),
        mc.GateZ(),
    )


def gate_to_matrix(gate: Gate) -> np.ndarray:
    """Convert a gate to a numpy matrix."""
    return np.array(gate.matrix().tolist(), dtype=np.complex128)


def sequence_to_matrix(gates: list[Gate]) -> np.ndarray:
    """Compute the unitary matrix for a gate sequence."""
    M = np.eye(2, dtype=np.complex128)
    for g in gates:
        M = gate_to_matrix(g) @ M
    return M


def invert_sequence(gates: list[Gate]) -> list[Gate]:
    """Return the inverse of a gate sequence: (g₁ g₂ ... gₙ)† = gₙ† ... g₂† g₁†"""
    return [g.inverse() for g in reversed(gates)]


# =============================================================================
# Sequence Simplification
# =============================================================================


def _gates_cancel(g: Gate, h: Gate) -> bool:
    """Check if two gates cancel (g·h = I)."""
    g_str = str(g)
    h_str = str(h)

    # Inverse pairs
    pairs = [
        ("T", "T†"),
        ("T†", "T"),
        ("S", "S†"),
        ("S†", "S"),
    ]
    for s1, s2 in pairs:
        if g_str == s1 and h_str == s2:
            return True

    # Self-inverse gates
    self_inverse = ("H", "X", "Y", "Z")
    if g_str in self_inverse and g_str == h_str:
        return True

    return False


def _try_combine_gates(g: Gate, h: Gate) -> Gate | None:
    """Try to combine two gates into one."""
    import mimiqcircuits as mc

    g_str = str(g)
    h_str = str(h)

    # T * T = S
    if g_str == "T" and h_str == "T":
        return mc.GateS()
    if g_str == "T†" and h_str == "T†":
        return mc.GateSDG()
    # S * S = Z
    if g_str == "S" and h_str == "S":
        return mc.GateZ()
    if g_str == "S†" and h_str == "S†":
        return mc.GateZ()

    return None


def simplify_sequence(gates: list[Gate]) -> list[Gate]:
    """Simplify a gate sequence by canceling adjacent gates."""
    if not gates:
        return gates

    result: list[Gate] = []
    for g in gates:
        if not result:
            result.append(g)
        else:
            last_g = result[-1]
            if _gates_cancel(last_g, g):
                result.pop()
            elif (combined := _try_combine_gates(last_g, g)) is not None:
                result.pop()
                result.append(combined)
            else:
                result.append(g)

    # Multiple passes until no more simplifications
    prev_len = len(gates)
    while len(result) < prev_len:
        prev_len = len(result)
        result = _simplify_pass(result)

    return result


def _simplify_pass(gates: list[Gate]) -> list[Gate]:
    """Single simplification pass."""
    if not gates:
        return gates

    result: list[Gate] = []
    for g in gates:
        if not result:
            result.append(g)
        else:
            last_g = result[-1]
            if _gates_cancel(last_g, g):
                result.pop()
            elif (combined := _try_combine_gates(last_g, g)) is not None:
                result.pop()
                result.append(combined)
            else:
                result.append(g)
    return result


# =============================================================================
# Epsilon-Net Generation
# =============================================================================


def generate_epsilon_net(
    basis_gates: tuple[Gate, ...],
    max_depth: int = SK_NET_MAX_DEPTH,
    max_points: int = SK_NET_MAX_POINTS,
    min_dist: float = SK_NET_MIN_DIST,
) -> tuple[list[np.ndarray], list[list[Gate]]]:
    """Generate an ε-net over SU(2) using Breadth-First Search.

    Uses a KDTree for efficient distance checking as the net grows.

    Args:
        basis_gates: The basis gates to use.
        max_depth: Maximum search depth.
        max_points: Maximum number of points in the net.
        min_dist: Minimum distance between points.

    Returns:
        (points, sequences): Lists of points and corresponding gate sequences.
    """
    I2 = np.eye(2, dtype=np.complex128)

    points: list[np.ndarray] = [matrix_to_point(I2)]
    sequences: list[list[Gate]] = [[]]

    # Build initial KDTree (will be rebuilt periodically)
    tree = KDTree(np.array(points))
    rebuild_interval = 100  # Rebuild tree every N points

    current_level: list[tuple[np.ndarray, list[Gate]]] = [(I2, [])]

    for _ in range(max_depth):
        next_level: list[tuple[np.ndarray, list[Gate]]] = []

        for U, gates in current_level:
            for g in basis_gates:
                U_new = gate_to_matrix(g) @ U
                U_new_su2 = to_su2(U_new)
                p_new = matrix_to_point(U_new_su2)

                # Use KDTree for efficient distance check
                dist, _ = tree.query(p_new.reshape(1, -1))
                if dist[0] >= min_dist:
                    new_gates = gates + [g]
                    next_level.append((U_new_su2, new_gates))
                    points.append(p_new)
                    sequences.append(new_gates)

                    # Rebuild tree periodically
                    if len(points) % rebuild_interval == 0:
                        tree = KDTree(np.array(points))

                    if len(points) >= max_points:
                        return points, sequences

        current_level = next_level
        if not current_level:
            break

    return points, sequences


# =============================================================================
# Cache for Epsilon-Net
# =============================================================================


@dataclass
class SKCacheEntry:
    """Cache entry for Solovay-Kitaev lookup tables."""

    net_tree: KDTree
    net_sequences: list[list[Gate]]
    net_matrices: list[np.ndarray]


# Global cache with lock for thread safety
_SK_CACHE: dict[int, SKCacheEntry] = {}
_SK_CACHE_LOCK = Lock()


def get_sk_cache(
    basis_gates: tuple[Gate, ...],
    max_depth: int = SK_NET_MAX_DEPTH,
    max_points: int = SK_NET_MAX_POINTS,
    min_dist: float = SK_NET_MIN_DIST,
) -> SKCacheEntry:
    """Get or initialize the Solovay-Kitaev lookup tables.

    Args:
        basis_gates: The basis gates to use.
        max_depth: Maximum depth for epsilon-net generation.
        max_points: Maximum points in epsilon-net.
        min_dist: Minimum distance between points.

    Returns:
        The cache entry with KDTree and gate sequences.
    """
    # Create hash key using string representations of gates
    gate_names = tuple(str(g) for g in basis_gates)
    h = hash((gate_names, max_depth, max_points, min_dist))

    with _SK_CACHE_LOCK:
        if h in _SK_CACHE:
            return _SK_CACHE[h]

        points, sequences = generate_epsilon_net(
            basis_gates, max_depth, max_points, min_dist
        )
        points_mat = np.array(points)
        net_tree = KDTree(points_mat)

        net_matrices = [sequence_to_matrix(seq) if seq else np.eye(2, dtype=np.complex128)
                        for seq in sequences]

        entry = SKCacheEntry(net_tree, sequences, net_matrices)
        _SK_CACHE[h] = entry
        return entry


# =============================================================================
# Group Commutator Decomposition
# =============================================================================


def gc_decompose(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose an SU(2) matrix Δ (close to identity) into V and W.

    The group commutator [V, W] = V W V† W† approximates Δ.

    This is the balanced group commutator construction from Dawson-Nielsen.
    For Δ representing a rotation by angle θ, V and W will be rotations
    by angle ≈ √θ around carefully chosen axes.

    Args:
        delta: An SU(2) matrix close to identity.

    Returns:
        (V, W): Two SU(2) matrices whose commutator approximates delta.
    """
    theta = extract_rotation_angle(delta)

    # Handle near-identity case
    if theta < 1e-12:
        I2 = np.eye(2, dtype=np.complex128)
        return I2, I2

    n = extract_rotation_axis(delta)

    # Balanced group commutator construction (Dawson-Nielsen)
    # φ satisfies: 4 sin²(φ/2) cos²(φ/2) = sin²(θ/2)
    sin_half_theta = math.sin(theta / 2)

    # Clamp argument to valid range for asin
    arg = math.sqrt(np.clip(sin_half_theta / 2, 0.0, 1.0))
    phi = 2 * math.asin(arg)

    # Choose v and w perpendicular to each other, with v×w = n
    v, w = find_orthogonal_axes(n)

    V = axis_angle_to_su2(v, phi)
    W = axis_angle_to_su2(w, phi)

    return V, W


# =============================================================================
# Solovay-Kitaev Algorithm
# =============================================================================


def find_nearest_in_net(U: np.ndarray, cache: SKCacheEntry) -> list[Gate]:
    """Find the gate sequence in the ε-net that best approximates U.

    Handles the SU(2) double cover (U and -U represent the same rotation).

    Args:
        U: An SU(2) matrix to approximate.
        cache: The epsilon-net cache.

    Returns:
        The gate sequence that best approximates U.
    """
    pt = matrix_to_point(U)

    # Find nearest neighbor
    _, idx = cache.net_tree.query(pt)

    # Also check antipodal point (SU(2) double cover: U ≡ -U)
    pt_neg = -pt
    _, idx_neg = cache.net_tree.query(pt_neg)

    # Compare actual operator distances
    seq = cache.net_sequences[idx]
    seq_neg = cache.net_sequences[idx_neg]

    mat = cache.net_matrices[idx]
    mat_neg = cache.net_matrices[idx_neg]

    dist = operator_norm_distance(U, mat)
    dist_neg = operator_norm_distance(U, mat_neg)

    return list(seq) if dist <= dist_neg else list(seq_neg)


def _matrix_hash(U: np.ndarray, tolerance: float = 1e-8) -> int:
    """Create a hash for an SU(2) matrix based on discretized elements."""
    scale = round(1 / tolerance)
    h = 0
    for i in range(2):
        for j in range(2):
            re = round(U[i, j].real * scale)
            im = round(U[i, j].imag * scale)
            h = hash((re, im, h))
    return h


def sk_approximate(
    U: np.ndarray,
    depth: int,
    basis_gates: tuple[Gate, ...],
    cache: SKCacheEntry,
    memo: dict[tuple[int, int], list[Gate]] | None = None,
    tolerance: float = 1e-8,
) -> list[Gate]:
    """Approximate an SU(2) matrix using the Solovay-Kitaev algorithm.

    Args:
        U: The SU(2) matrix to approximate.
        depth: Recursion depth (higher = better precision, more gates).
        basis_gates: The basis gates to use.
        cache: The epsilon-net cache.
        memo: Memoization dictionary.
        tolerance: Tolerance for matrix hashing.

    Returns:
        A sequence of gates approximating U.
    """
    if memo is None:
        memo = {}

    # Create memo key
    key = _matrix_hash(U, tolerance)
    memo_key = (key, depth)

    if memo_key in memo:
        return list(memo[memo_key])

    if depth == 0:
        result = find_nearest_in_net(U, cache)
        memo[memo_key] = result
        return list(result)

    # Recursive case
    U_prev_gates = sk_approximate(U, depth - 1, basis_gates, cache, memo, tolerance)
    U_prev = to_su2(sequence_to_matrix(U_prev_gates) if U_prev_gates else np.eye(2, dtype=np.complex128))

    delta = to_su2(U @ U_prev.conj().T)

    # Handle SU(2) double cover
    if np.trace(delta).real < 0:
        delta = -delta

    V_mat, W_mat = gc_decompose(delta)

    V_gates = sk_approximate(V_mat, depth - 1, basis_gates, cache, memo, tolerance)
    W_gates = sk_approximate(W_mat, depth - 1, basis_gates, cache, memo, tolerance)

    V_inv_gates = invert_sequence(V_gates)
    W_inv_gates = invert_sequence(W_gates)

    # Commutator: [V, W] = V W V† W†
    # Full approximation: U_prev * [V, W] ≈ U_prev * Δ = U
    result = U_prev_gates + W_inv_gates + V_inv_gates + W_gates + V_gates
    memo[memo_key] = result
    return list(result)


# =============================================================================
# SolovayKitaevRewrite Rule
# =============================================================================


@dataclass(frozen=True)
class SolovayKitaevRewrite(RewriteRule):
    """Rewrite rule using the Solovay-Kitaev algorithm.

    Approximates arbitrary single-qubit unitaries using a sequence of
    Clifford+T gates. The algorithm recursively refines an initial approximation
    using group commutators, achieving ε-approximation with O(log^c(1/ε)) gates.

    Args:
        depth: Recursion depth (default: 3). Higher = better precision, more gates.
        simplify: Whether to simplify the resulting sequence (default: True).
        net_max_depth: Maximum depth for epsilon-net generation (default: 15).
        net_max_points: Maximum points in epsilon-net (default: 100,000).
        net_min_dist: Minimum distance between epsilon-net points (default: 0.01).

    Supported operations:
        - GateRZ(λ): Z-rotation by angle λ
        - GateRY(θ): Y-rotation by angle θ
        - GateRX(θ): X-rotation by angle θ
        - GateU(θ, φ, λ, γ): General single-qubit unitary

    Symbolic parameters are NOT supported.

    Example:
        >>> from mimiqcircuits import GateRZ
        >>> from mimiqcircuits.decomposition import SolovayKitaevRewrite
        >>> rule = SolovayKitaevRewrite(depth=0)
        >>> circ = rule.decompose_step(GateRZ(0.123), [0], [], [])

    Note:
        The first call may be slow due to epsilon-net generation.
        Subsequent calls reuse the cached net.
    """

    depth: int = 3
    simplify: bool = True
    net_max_depth: int = SK_NET_MAX_DEPTH
    net_max_points: int = SK_NET_MAX_POINTS
    net_min_dist: float = SK_NET_MIN_DIST

    def __post_init__(self):
        if self.depth < 0:
            raise ValueError("depth must be non-negative")

    def matches(self, op: Operation) -> bool:
        """Check if this rule can decompose the operation.

        Matches single-qubit rotations with concrete (non-symbolic) angles.

        Args:
            op: The operation to check.

        Returns:
            True if this rule can decompose the operation.
        """
        import mimiqcircuits as mc

        if isinstance(op, mc.GateRZ):
            return not op.is_symbolic()
        if isinstance(op, mc.GateRY):
            return not op.is_symbolic()
        if isinstance(op, mc.GateRX):
            return not op.is_symbolic()
        if isinstance(op, mc.GateU):
            return not op.is_symbolic()
        return False

    def decompose_step(
        self,
        op: Operation,
        qubits: Sequence[int],
        bits: Sequence[int],
        zvars: Sequence[int],
    ) -> Circuit:
        """Decompose the operation using Solovay-Kitaev.

        Args:
            op: The operation to decompose.
            qubits: Qubit indices for the operation.
            bits: Classical bit indices (unused).
            zvars: Z-variable indices (unused).

        Returns:
            A Circuit containing the Clifford+T approximation.

        Raises:
            DecompositionError: If the operation cannot be decomposed.
        """
        import mimiqcircuits as mc

        if not self.matches(op):
            raise DecompositionError(
                f"SolovayKitaevRewrite cannot decompose {type(op).__name__}"
            )

        # Get the unitary matrix
        U = np.array(op.matrix().tolist(), dtype=np.complex128)

        # Project to SU(2)
        U_su2 = to_su2(U)

        # Get basis gates and cache
        basis_gates = get_basic_gates()
        cache = get_sk_cache(
            basis_gates,
            self.net_max_depth,
            self.net_max_points,
            self.net_min_dist,
        )

        # Run Solovay-Kitaev
        memo: dict[tuple[int, int], list[mc.Gate]] = {}
        gates = sk_approximate(U_su2, self.depth, basis_gates, cache, memo)

        # Optionally simplify
        if self.simplify:
            gates = simplify_sequence(gates)

        # Build circuit
        circ = mc.Circuit()
        q = qubits[0]
        for gate in gates:
            circ.push(gate, q)

        return circ


__all__ = ["SolovayKitaevRewrite"]
