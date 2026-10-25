"""Native extension of mimiqcircuits (compiled from the Rust sources in rust/)."""

__version__: str

def gridsynth(theta: float, epsilon: float, seed: int = 0) -> str:
    """Clifford+T approximation of Rz(theta) up to a global phase.

    Returns a word over ``H``, ``S``, ``T``, ``X`` in matrix-product order
    whose operator-norm distance to Rz(theta), minimized over the global
    phase, is at most ``epsilon``. Raises ValueError if ``theta`` is not
    finite or ``epsilon`` is not in (0, 1), and OverflowError if ``seed`` is
    not in [0, 2**64). Runs without holding the GIL.
    """
    ...
