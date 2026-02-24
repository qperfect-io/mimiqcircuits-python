Simulation Parameters
=====================

This page describes the advanced parameters available for fine-tuning the circuit simulations on MIMIQ, particularly for the MPS backend.

General Parameters
------------------

fuse
~~~~

**Type:** ``bool``
**Default:** ``False`` (backend dependent)

Controls whether to fuse consecutive gates acting on the same qubits into a single gate before execution. This can reduce the total number of operations and potentially improve performance, but might increase the complexity of the individual operations.

canonical_decompose
~~~~~~~~~~~~~~~~~~~

**Type:** ``bool``
**Default:** ``False`` (backend dependent)

Controls whether to decompose the circuit into a canonical set of gates (typically ``GateU`` and ``GateCX``). This is often a preliminary step for other optimizations and can help ensure compatibility with specific simulator backends.

remove_swaps
~~~~~~~~~~~~

**Type:** ``bool``
**Default:** ``False`` (backend dependent)

Controls whether to automatically remove SWAP gates by permuting the logical-to-physical qubit mapping. When enabled, SWAP gates are absorbed into the qubit ordering rather than being explicitly simulated. This can significantly reduce circuit depth and gate count.

reorderqubits
~~~~~~~~~~~~~

**Type:** ``bool`` or ``str``
**Default:** ``False`` (backend dependent)

Controls whether to automatically reorder qubits to minimize long-range interactions in the circuit. This is particularly useful for MPS simulations, where nearest-neighbor gates are significantly more efficient than long-range ones. A good qubit ordering can dramatically reduce bond dimension requirements and improve simulation fidelity.

When set to ``True``, the default method (``"sa_warm_start"``) is used. You can also pass a string to select a specific reordering algorithm:

**Deterministic methods** (produce the same result every time):

- ``"greedy"`` -- Greedy weight-first insertion. Builds the ordering incrementally by placing qubits to minimize interaction cost at each step.
- ``"spectral"`` -- Spectral ordering via the Fiedler vector of the interaction graph Laplacian. Produces a globally smooth ordering.
- ``"rcm"`` -- Reverse Cuthill-McKee ordering. A bandwidth-reduction algorithm that minimizes the spread of interactions.

**Stochastic methods** (results depend on the RNG seed; use ``reorderqubits_seed`` for reproducibility):

- ``"sa_warm_start"`` / ``"auto"`` -- Deterministic warm-start (greedy) refined with simulated annealing. This is the **default** when ``reorderqubits=True``.
- ``"sa_only"`` -- Pure simulated annealing starting from the identity ordering.
- ``"memetic"`` -- Evolutionary algorithm with local search. Restricted to circuits with **50 qubits or fewer**.
- ``"multilevel"`` -- Multi-level coarsening and refinement algorithm.
- ``"grasp"`` -- GRASP (Greedy Randomized Adaptive Search Procedure) with path relinking. Restricted to circuits with **50 qubits or fewer**.
- ``"hybrid"`` -- Multi-level + GRASP hybrid for best quality at higher computational cost. Restricted to circuits with **50 qubits or fewer**.

.. note::
    The methods ``"grasp"``, ``"hybrid"``, and ``"memetic"`` are computationally intensive and are restricted to circuits with at most 50 qubits on the remote backend.

reorderqubits_seed
~~~~~~~~~~~~~~~~~~

**Type:** ``int`` or ``None``
**Default:** ``None``

Independent seed for the qubit reordering random number generator. When set, uses a separate RNG seeded with this value for the reordering algorithm, instead of deriving the RNG from the main simulation ``seed``.

This is useful when you want to:

- Reproduce a specific qubit ordering while varying the simulation seed.
- Vary the qubit ordering while keeping the simulation seed fixed.

Only affects stochastic reordering methods (``"sa_warm_start"``, ``"sa_only"``, ``"memetic"``, ``"multilevel"``, ``"grasp"``, ``"hybrid"``). Deterministic methods (``"greedy"``, ``"spectral"``, ``"rcm"``) ignore this parameter.

MPS Specific Parameters
-----------------------

mpsmethod
~~~~~~~~~

**Type:** ``str``
**Options:** ``"dmpo"``, ``"vmpoa"``, ``"vmpob"``
**Default:** ``None`` (backend dependent)

Specifies the method used for Matrix Product Operator (MPO) application to the Matrix Product State (MPS).

- ``"dmpo"``: Direct MPO application. Applies the gate MPO to the MPS and then compresses the result.
- ``"vmpoa"`` / ``"vmpob"``: Variational MPO application. Uses a variational search to find the best MPS approximation of the result.

traversal
~~~~~~~~~

**Type:** ``str``
**Options:** ``"sequential"``, ``"bfs"``
**Default:** ``"sequential"``

Specifies the strategy used to traverse the circuit while compressing it into MPOs.

- ``"sequential"``: Processes the tensors in a sequential order (e.g., from left to right). This is the standard approach for MPS.
- ``"bfs"``: Uses a Breadth-First Search approach. This can be beneficial for certain circuit structures.

bonddim
~~~~~~~

**Type:** ``int``
**Default:** ``256``

The maximum bond dimension of the MPS. Controls the amount of entanglement that can be captured. Higher values allow for higher accuracy but require more memory and computational time.

entdim
~~~~~~

**Type:** ``int``
**Default:** ``16``

The maximum entangling dimension for MPO intermediates. Controls the complexity of the gates/MPOs before they are applied to the state.

streaming
~~~~~~~~~

**Type:** ``bool``
**Default:** ``False``

Controls whether to use the streaming MPS simulator. The streaming simulator processes the circuit without storing the full MPS state in memory, instead computing and streaming tensors on the fly. This enables simulation of larger circuits that would otherwise exceed available memory, at the cost of not being able to compute amplitudes.
