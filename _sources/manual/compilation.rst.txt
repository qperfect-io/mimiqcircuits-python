Circuit compilation & optimization
==================================

MIMIQ provides tools to optimize and transform quantum circuits.

Cleaning circuits
-----------------

Remove unused
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

Optimization passes
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

    >>> from mimiqcircuits import *
    >>> c = Circuit()
    >>> c.push(GateH(), 1)
    2-qubit circuit with 1 instruction:
    └── H @ q[1]
    <BLANKLINE>
    >>> c.push(GateSWAP(), 1, 2)
    3-qubit circuit with 2 instructions:
    ├── H @ q[1]
    └── SWAP @ q[1:2]
    <BLANKLINE>
    >>> c.push(GateCX(), 2, 3)
    4-qubit circuit with 3 instructions:
    ├── H @ q[1]
    ├── SWAP @ q[1:2]
    └── CX @ q[2], q[3]
    <BLANKLINE>
    >>> new_c, perm = remove_swaps(c)
    >>> new_c
    4-qubit circuit with 2 instructions:
    ├── H @ q[1]
    └── CX @ q[1], q[3]
    <BLANKLINE>
    >>> perm
    [0, 2, 1, 3]

Pauli-based computation
-----------------------

In Pauli-based computation a circuit is expressed with two kinds of operations only: :math:`\pi/8` rotations about multi-qubit Pauli operators, and measurements of multi-qubit Pauli operators. This is the form used by lattice-surgery architectures for surface codes, where every logical operation is a Pauli product measurement.

The :func:`~mimiqcircuits.pauli_based_computation` function (also available as :meth:`~mimiqcircuits.Circuit.pauli_based_computation`) performs this rewrite:

.. code-block:: python

    new_circuit = pauli_based_computation(circuit, basis=None, magic_states=False)

**Arguments:**

* ``circuit``: The input circuit.
* ``basis`` (default :class:`~mimiqcircuits.CliffordTBasis`): The basis used to decompose the circuit into Clifford+T gates first. Measurements, resets and conditional Pauli operators are kept whole, whatever the basis.
* ``magic_states`` (default ``False``): If ``True``, each rotation is further replaced by the injection of a magic state.

The circuit is decomposed into Clifford+T, then every Clifford gate is moved to the end of the circuit. On the way it changes each T gate into an :class:`~mimiqcircuits.RPauli` rotation of angle :math:`\pm\pi/4` and each measurement of a Pauli product (:class:`~mimiqcircuits.Measure`, :class:`~mimiqcircuits.MeasureX`, :class:`~mimiqcircuits.MeasureXX`, :class:`~mimiqcircuits.MeasurePauli` and the like) into a :class:`~mimiqcircuits.MeasurePauli`. The Cliffords that end up after the last measurement do not change any outcome, so they are dropped. The new circuit gives the same measurement statistics as the original one, but not the same final state.

**Example:**

Prepare a Bell pair, apply a T gate to qubit 1 and measure both qubits:

.. doctest::

    >>> from mimiqcircuits import *
    >>> c = Circuit()
    >>> _ = c.push(GateH(), 0)
    >>> _ = c.push(GateCX(), 0, 1)
    >>> _ = c.push(GateT(), 1)
    >>> _ = c.push(Measure(), range(2), range(2))
    >>> pauli_based_computation(c)
    2-qubit, 2-bit circuit with 3 instructions:
    ├── R("XZ", (1/4)*pi) @ q[0:1]
    ├── MP(X) @ q[0], c[0]
    └── MP(XZ) @ q[0:1], c[1]
    <BLANKLINE>

The Hadamard and CX gates before the T gate map :math:`Z_1` to :math:`X_0 Z_1`, so the T gate becomes a :math:`\pi/8` rotation about :math:`X_0 Z_1` (``RPauli`` with angle :math:`\pi/4`). The same gates turn the measurements of :math:`Z_0` and :math:`Z_1` into measurements of :math:`X_0` and :math:`X_0 Z_1`, written to the original bits.

Conditional operations and resets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`~mimiqcircuits.IfStatement` whose body contains only Pauli operators, directly or inside a :class:`~mimiqcircuits.PauliString`, :class:`~mimiqcircuits.Block`, :class:`~mimiqcircuits.Parallel`, :class:`~mimiqcircuits.Repeat` or :class:`~mimiqcircuits.GateCall`, applies one Pauli product. It becomes one conditional ``PauliString``, with the Clifford gates before it moved past it like past a measurement. A conditional T or T† gate becomes a conditional ``RPauli`` rotation.

A reset (:class:`~mimiqcircuits.Reset`, :class:`~mimiqcircuits.ResetX`, :class:`~mimiqcircuits.ResetY`) measures its qubit into a new bit, placed after the bits of the circuit, then applies a conditional Pauli that brings the qubit back to the reset state. The measure-and-reset operations do the same with the bit they are given.

.. doctest::

    >>> c2 = Circuit()
    >>> _ = c2.push(GateH(), 0)
    >>> _ = c2.push(Measure(), 0, 0)
    >>> _ = c2.push(GateH(), 1)
    >>> _ = c2.push(IfStatement(GateZ(), BitString("1")), 1, 0)
    >>> _ = c2.push(Reset(), 0)
    >>> _ = c2.push(Measure(), range(2), range(1, 3))
    >>> pauli_based_computation(c2)
    2-qubit, 4-bit circuit with 6 instructions:
    ├── MP(X) @ q[0], c[0]
    ├── IF(c==1) X @ q[1], condition[0]
    ├── MP(X) @ q[0], c[3]
    ├── IF(c==1) Z @ q[0], condition[3]
    ├── MP(X) @ q[0], c[1]
    └── MP(X) @ q[1], c[2]
    <BLANKLINE>

The Hadamard gate on qubit 1 turns the conditional :math:`Z_1` into a conditional :math:`X_1`. The reset of qubit 0 measures :math:`X_0` into the new bit 3 and applies :math:`Z_0` when the outcome is 1.

Other conditional operations are refused with a ``ValueError``. After a conditional Clifford gate, such as a conditional Hadamard, the Clifford gates to move would depend on the condition bits, so the later rotations and measurements could not be computed before the circuit runs.

Magic-state injection
~~~~~~~~~~~~~~~~~~~~~

A :math:`\pi/8` rotation can itself be carried out with Pauli product measurements by consuming a magic state :math:`T\ket{+}`, prepared with :class:`~mimiqcircuits.PrepareStateT`. With ``magic_states=True`` every rotation is replaced in this way, using one extra ancilla qubit, two extra bits per rotation and two bits per data qubit for the Pauli frame:

.. doctest::

    >>> pauli_based_computation(c, magic_states=True)
    3-qubit, 8-bit circuit with 8 instructions:
    ├── PrepareStateT @ q[2]
    ├── MP(XZZ) @ q[0:2], c[2]
    ├── IF(c==0) MP(X) @ q[2], c[3], condition[2]
    ├── IF(c==1) MP(Y) @ q[2], c[3], condition[2]
    ├── c[4] ^= c[3]
    ├── c[7] ^= c[3]
    ├── MP(X) @ q[0], c[0]
    └── MP(XZ) @ q[0:1], c[1]
    <BLANKLINE>

Reading the result line by line:

1. :class:`~mimiqcircuits.PrepareStateT` puts the ancilla (qubit 2) in the magic state.
2. ``MP(XZZ)`` measures the rotation axis :math:`X_0 Z_1` together with :math:`Z` on the ancilla, into bit 2.
3. Depending on bit 2, the ancilla is measured in X or in Y, into bit 3. The two :class:`~mimiqcircuits.IfStatement` lines implement this choice.
4. The two :class:`~mimiqcircuits.Xor` lines record the byproduct described below in the Pauli frame.
5. The final measurements are the same as before.

When bit 3 is ``1`` the rotation comes with an extra Pauli :math:`X_0 Z_1` on the data qubits. It is not corrected with gates. Instead, the product of all such Paulis is kept in classical bits, the Pauli frame: bit 4 holds its X part on qubit 0 and bit 7 its Z part on qubit 1. Later outcomes that the frame would flip are corrected in place with 3-bit ``Xor`` operations. Here both final measurements commute with :math:`X_0 Z_1`, so no correction is needed.

With magic states a conditional Pauli is not applied either: it multiplies the frame when its condition holds. A conditional T or T† gate is refused, since a magic state cannot be injected conditionally.

Reference
---------

.. autofunction:: mimiqcircuits.remove_unused
    :noindex:
.. autofunction:: mimiqcircuits.remove_swaps
    :noindex:
.. autofunction:: mimiqcircuits.pauli_based_computation
    :noindex:
