Circuit Compilation & Optimization
==================================

MIMIQ provides tools to optimize and transform quantum circuits.

Cleaning Circuits
-----------------

Remove Unused
~~~~~~~~~~~~~

The :func:`~mimiqcircuits.remove_unused` function cleans up a circuit by removing unused qubits, classical bits, and Z-variables.

.. code-block:: python

    new_circuit, qubit_map, bit_map, zvar_map = remove_unused(circuit)

It returns:

1. The new optimized :class:`~mimiqcircuits.Circuit`.
2. A mapping (dict) of old qubit indices to new ones.
3. A mapping (dict) of old bit indices to new ones.
4. A mapping (dict) of old Z-variable indices to new ones.

**Example:**

.. doctest::

    >>> from mimiqcircuits import *
    >>> c = Circuit()
    >>> c.push(GateH(), 0)
    >>> c.push(GateCX(), 0, 2)  # Qubit 1 is unused
    >>> new_c, qmap, bmap, zmap = remove_unused(c)
    >>> new_c
    2-qubit circuit with 2 instructions:
    ├── H @ q[0]
    └── CX @ q[0], q[1]

Optimization Passes
-------------------

Remove SWAPs
~~~~~~~~~~~~

The :func:`~mimiqcircuits.remove_swaps` function eliminates ``SWAP`` gates by tracking qubit permutations and remapping subsequent operations.

.. code-block:: python

    new_circuit, final_permutation = remove_swaps(circuit, recursive=False)

**Arguments:**

* ``circuit``: The input circuit.
* ``recursive`` (default ``False``): If ``True``, recursively removes swaps from nested blocks.

It returns:

1. A new :class:`~mimiqcircuits.Circuit` without SWAP gates.
2. A list representing the final physical permutation of qubits.

**Example:**

.. doctest::

    >>> c = Circuit()
    >>> c.push(GateH(), 1)
    >>> c.push(GateSWAP(), 1, 2)
    >>> c.push(GateCX(), 2, 3)
    >>> new_c, perm = remove_swaps(c)
    >>> new_c
    └── CX @ q[1], q[3]

Reference
---------

.. autofunction:: mimiqcircuits.remove_unused
    :noindex:
.. autofunction:: mimiqcircuits.remove_swaps
    :noindex:
