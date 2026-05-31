Z-register Operations
=====================

Operations on the Z-register allow users to manipulate complex-valued variables inside a quantum circuit.
This section covers all the information needed to perform operations on the complex-valued variables stored in the Z-register.

What is the Z-register?
-----------------------

In **MIMIQ**, complex-valued variables are stored in a special register called the *Z-register*. 
Practically speaking, the Z-register is just a vector of complex numbers (`Complex` values) 
that can be manipulated with circuit operations, similarly to the quantum register (qubits) or the classical register (bits).

When visualizing a circuit, Z-register indices are represented with the letter ``z``.

What can you do with the Z-register?
------------------------------------

Operations involving the Z-register can be broadly divided into two types:

- Operations that **store** information into the Z-register.
- Operations that **manipulate** information already stored in the Z-register.

Storing information in the Z-register
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations compute values from the quantum state and store the result into the Z-register:

- :class:`~mimiqcircuits.ExpectationValue`: Computes the expectation value of an observable.
- :class:`~mimiqcircuits.Amplitude`: Stores the complex amplitude of a bitstring.

.. doctest::
    :hide:

    >>> from mimiqcircuits import *

.. doctest::

    >>> c = Circuit()
    >>> c.push(GateH(), 1)
    2-qubit circuit with 1 instructions:
    └── H @ q[1]
    <BLANKLINE>
    >>> c.push(GateCX(), 1, 2)
    3-qubit circuit with 2 instructions:
    ├── H @ q[1]
    └── CX @ q[1], q[2]
    <BLANKLINE>
    >>> c.push(ExpectationValue(PauliString("XX")), 1, 2, 1)  # qubits 1,2 -> z[1]
    3-qubit, 2-zvar circuit with 3 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    └── ⟨XX⟩ @ q[1,2], z[1]
    <BLANKLINE>
    >>> c.push(ExpectationValue(PauliString("ZZ")), 1, 2, 2)  # qubits 1,2 -> z[2]
    3-qubit, 3-zvar circuit with 4 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── ⟨XX⟩ @ q[1,2], z[1]
    └── ⟨ZZ⟩ @ q[1,2], z[2]
    <BLANKLINE>
    >>> c.push(Amplitude(BitString(2)), 3)                      # bitstring -> z[3]
    3-qubit, 4-zvar circuit with 5 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── ⟨XX⟩ @ q[1,2], z[1]
    ├── ⟨ZZ⟩ @ q[1,2], z[2]
    └── Amplitude(bs"00") @ z[3]
    <BLANKLINE>

Manipulating the Z-register
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following operations perform arithmetic on Z-register variables:

- :class:`~mimiqcircuits.Add`: Adds Z-register variables.
- :class:`~mimiqcircuits.Multiply`: Multiplies Z-register variables.
- :class:`~mimiqcircuits.Pow`: Raises a Z-register variable to a power.

Both :class:`~mimiqcircuits.Add` and :class:`~mimiqcircuits.Multiply` are
**assignments**: the *first* z-target is the destination (its previous value is
overwritten), and the remaining targets are the inputs.

.. doctest::

    >>> c.push(Add(3), 4, 1, 2)         # z[4] = z[1] + z[2]
    3-qubit, 5-zvar circuit with 6 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── ⟨XX⟩ @ q[1,2], z[1]
    ├── ⟨ZZ⟩ @ q[1,2], z[2]
    ├── Amplitude(bs"00") @ z[3]
    └── z[4] = 0.0 + z[1] + z[2]
    <BLANKLINE>
    >>> c.push(Multiply(3), 5, 1, 2)    # z[5] = z[1] * z[2]
    3-qubit, 6-zvar circuit with 7 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── ⟨XX⟩ @ q[1,2], z[1]
    ├── ⟨ZZ⟩ @ q[1,2], z[2]
    ├── Amplitude(bs"00") @ z[3]
    ├── z[4] = 0.0 + z[1] + z[2]
    └── z[5] = 1.0 * z[1] * z[2]
    <BLANKLINE>
    >>> c.push(Multiply(2, 0.2), 3, 3)  # z[3] = 0.2 * z[3] (in-place via aliasing)
    3-qubit, 6-zvar circuit with 8 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── ⟨XX⟩ @ q[1,2], z[1]
    ├── ⟨ZZ⟩ @ q[1,2], z[2]
    ├── Amplitude(bs"00") @ z[3]
    ├── z[4] = 0.0 + z[1] + z[2]
    ├── z[5] = 1.0 * z[1] * z[2]
    └── z[3] = 0.2 * z[3]
    <BLANKLINE>
    >>> c.push(Pow(-1), 3)              # z[3] = z[3] ** -1
    3-qubit, 6-zvar circuit with 9 instructions:
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── ⟨XX⟩ @ q[1,2], z[1]
    ├── ⟨ZZ⟩ @ q[1,2], z[2]
    ├── Amplitude(bs"00") @ z[3]
    ├── z[4] = 0.0 + z[1] + z[2]
    ├── z[5] = 1.0 * z[1] * z[2]
    ├── z[3] = 0.2 * z[3]
    └── z[3] = z[3]^(-1)
    <BLANKLINE>

These operations can take an arbitrary number of input Z-register indices.
For example, ``Add(4)`` is a 4-zvar instruction that adds three input
Z-register values together and stores the result in the destination.

Additionally, a constant value can be added or multiplied in. For instance, ``Add(3, 0.5)`` evaluates ``z[a] = 0.5 + z[b] + z[c]``, where ``z[a]`` is the destination and ``z[b]``, ``z[c]`` are the inputs.

.. note::

    Because :class:`~mimiqcircuits.Add`, :class:`~mimiqcircuits.Multiply` and :class:`~mimiqcircuits.Pow` opt into *z-variable aliasing*, the destination z-variable may also appear among the input z-variables. This is how you recover the previous in-place behaviour: ``c.push(Add(3), 0, 0, 1)`` evaluates ``z[0] = z[0] + z[1]`` (i.e. ``z[0] += z[1]``), and ``c.push(Multiply(2, 0.2), 3, 3)`` evaluates ``z[3] = 0.2 * z[3]`` (i.e. ``z[3] *= 0.2``). See :ref:`Target aliasing` for the general rules.

Example: Ising Hamiltonian Expectation Value (Energy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more complex example is the computation of the expectation value of a 1D Ising Hamiltonian with transverse field:

.. math::

    H = - J \sum_{j=1}^{N-1} \sigma^z_j \sigma^z_{j+1} - h \sum_{j=1}^N \sigma^x_j

where \( \sigma^z_j \) and \( \sigma^x_j \) are Pauli matrices on the \( j \)-th qubit, 
\( J \) is the coupling constant, and \( h \) is the transverse magnetic field.

The circuit to compute the expectation value is:

Example: Ising Hamiltonian Expectation Value (Energy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more complex example is the computation of the expectation value of a 1D Ising Hamiltonian with transverse field:

.. math::

    H = - J \sum_{j=0}^{N-2} \sigma^z_j \sigma^z_{j+1} - h \sum_{j=0}^{N-1} \sigma^x_j

where \( \sigma^z_j \) and \( \sigma^x_j \) are Pauli matrices on the \( j \)-th qubit, 
\( J \) is the coupling constant, and \( h \) is the transverse magnetic field.

The circuit to compute the expectation value is:

.. doctest:: python

    >>> from mimiqcircuits import *

    >>> N = 4        # number of qubits
    >>> J = 1.0      # coupling strength
    >>> h = 0.5      # transverse field

    >>> c = Circuit()

    >>> for j in range(N - 1):
    ...     newz = c.num_zvars()
    ...     _ = c.push(ExpectationValue(PauliString("ZZ")), j, j + 1, newz)
    ...     # in-place scale: z[newz] = -J * z[newz]
    ...     _ = c.push(Multiply(2, -J), newz, newz)

    >>> c
    4-qubit, 3-zvar circuit with 6 instructions:
    ├── ⟨ZZ⟩ @ q[0,1], z[0]
    ├── z[0] = -1.0 * z[0]
    ├── ⟨ZZ⟩ @ q[1,2], z[1]
    ├── z[1] = -1.0 * z[1]
    ├── ⟨ZZ⟩ @ q[2,3], z[2]
    └── z[2] = -1.0 * z[2]
    <BLANKLINE>

    >>> for j in range(N):
    ...     newz = c.num_zvars()
    ...     _ = c.push(ExpectationValue(GateX()), j, newz)
    ...     # in-place scale: z[newz] = -h * z[newz]
    ...     _ = c.push(Multiply(2, -h), newz, newz)

    >>> c
    4-qubit, 7-zvar circuit with 14 instructions:
    ├── ⟨ZZ⟩ @ q[0,1], z[0]
    ├── z[0] = -1.0 * z[0]
    ├── ⟨ZZ⟩ @ q[1,2], z[1]
    ├── z[1] = -1.0 * z[1]
    ├── ⟨ZZ⟩ @ q[2,3], z[2]
    ├── z[2] = -1.0 * z[2]
    ├── ⟨X⟩ @ q[0], z[3]
    ├── z[3] = -0.5 * z[3]
    ├── ⟨X⟩ @ q[1], z[4]
    ├── z[4] = -0.5 * z[4]
    ├── ⟨X⟩ @ q[2], z[5]
    ├── z[5] = -0.5 * z[5]
    ├── ⟨X⟩ @ q[3], z[6]
    └── z[6] = -0.5 * z[6]
    <BLANKLINE>

    >>> total_terms = c.num_zvars()
    >>> # store the total energy in a fresh z-variable z[total_terms]
    >>> _ = c.push(Add(total_terms + 1), total_terms, *range(total_terms))
    >>> c
    4-qubit, 8-zvar circuit with 15 instructions:
    ├── ⟨ZZ⟩ @ q[0,1], z[0]
    ├── z[0] = -1.0 * z[0]
    ├── ⟨ZZ⟩ @ q[1,2], z[1]
    ├── z[1] = -1.0 * z[1]
    ├── ⟨ZZ⟩ @ q[2,3], z[2]
    ├── z[2] = -1.0 * z[2]
    ├── ⟨X⟩ @ q[0], z[3]
    ├── z[3] = -0.5 * z[3]
    ├── ⟨X⟩ @ q[1], z[4]
    ├── z[4] = -0.5 * z[4]
    ├── ⟨X⟩ @ q[2], z[5]
    ├── z[5] = -0.5 * z[5]
    ├── ⟨X⟩ @ q[3], z[6]
    ├── z[6] = -0.5 * z[6]
    └── z[7] = 0.0 + z[0] + z[1] + z[2] + z[3] + z[4] + z[5] + z[6]
    <BLANKLINE>

Reference
---------

.. autoclass:: mimiqcircuits.Add
    :noindex:
.. autoclass:: mimiqcircuits.Multiply
    :noindex:
.. autoclass:: mimiqcircuits.Pow
    :noindex:
