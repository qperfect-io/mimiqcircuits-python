Simulation Parameters
=====================

This page describes the advanced parameters available for fine-tuning the circuit simulations on MIMIQ, particularly for the MPS backend.

General Parameters
------------------

remove_swaps
~~~~~~~~~~~~

**Type:** `bool`
**Default:** `False` (backend dependent)

Controls whether to automatically remove SWAP gates by permuting the logical-to-physical qubit mapping. This can significantly reduce the depth of the circuit and the number of gates to be executed.

canonical_decompose
~~~~~~~~~~~~~~~~~~~

**Type:** `bool`
**Default:** `False` (backend dependent)

Controls whether to decompose the circuit into a canonical set of gates, typically GateU and GateCX. This is often a preliminary step for other optimizations.

fuse
~~~~

**Type:** `bool`
**Default:** `False` (backend dependent)

Controls whether to fuse consecutive gates acting on the same qubits into a single gate before execution. This can reduce the total number of operations and potentially improve performance, but might increase the complexity of the individual operations.

reorderqubits
~~~~~~~~~~~~~

**Type:** `bool`
**Default:** `False` (backend dependent)

Controls whether to automatically reorder qubits to minimize entanglement and bond dimension requirements. This is particularly useful for MPS simulations where qubit ordering significantly impacts performance.

MPS Specific Parameters
-----------------------

mpsmethod
~~~~~~~~~

**Type:** `str`
**Options:** `"dmpo"`, `"vmpoa"`, `"vmpob"`
**Default:** `None` (backend dependent)

Specifies the method used for Matrix Product Operator (MPO) application to the Matrix Product State (MPS).

- `"dmpo"`: Direct MPO application. Applies the gate MPO to the MPS and then compresses the result.
- `"vmpoa"` / `"vmpob"`: Variational MPO application. Uses a variational search to find the best MPS approximation of the result.

traversal
~~~~~~~~~

**Type:** `str`
**Options:** `"sequential"`, `"bfs"`
**Default:** `"sequential"`

Specifies the strategy used to traverse the circuit while compressing it into MPOs.

- `"sequential"`: Processes the tensors in a sequential order (e.g., from left to right). This is the standard approach for MPS.
- `"bfs"`: Uses a Breadth-First Search approach. This can be beneficial for certain circuit structures.

bonddim
~~~~~~~

**Type:** `int`
**Default:** `256`

The maximum bond dimension of the MPS. Controls the amount of entanglement that can be captured. Higher values allow for higher accuracy but require more memory and computational time.

entdim
~~~~~~

**Type:** `int`
**Default:** `16`

The maximum entangling dimension for MPO intermediates. Controls the complexity of the gates/MPOs before they are applied to the state.
