Hamiltonians and Time Evolution
===============================

MIMIQ provides tools to define quantum Hamiltonians and simulate their time evolution using 
Trotterization methods. This page explains how to construct Hamiltonians, compute their 
expectation values, and apply Lie-Trotter, Suzuki-Trotter, and Yoshida decompositions.

This workflow allows you to:

    Build realistic Hamiltonians from Pauli terms

    Simulate time evolution efficiently using Trotter expansions

    Measure physical observables like energy

Contents
========

.. contents::
   :local:
   :depth: 2
   :backlinks: entry


Hamiltonian
-----------------------
.. _hamiltonian:

In quantum computing, Hamiltonians play a central role in algorithms such as the Variational Quantum Eigensolver (VQE),
Quantum Phase Estimation, and the Quantum Approximate Optimization Algorithm (QAOA). They are used to encode the energy landscape of
a physical system, generate dynamics, or define cost functions for optimization.

A typical Hamiltonian is expressed as a sum of weighted Pauli strings:

.. math::

    H = \sum_j c_j \cdot P_j

Each term consists of a real coefficient :math:`c_j` and a Pauli string :math:`P_j`, such as ``X``, ``ZZ``, or ``XYZ``,
which acts on a specific subset of qubits.

In quantum circuit frameworks, these Hamiltonians are often represented programmatically by associating each
Pauli string with a set of target qubit indices and a coefficient, allowing users to construct and simulate physical models
or optimization problems.

    
Simulating the Ising Model
--------------------------
.. _ising-model-simulation:

A fundamental use case for Hamiltonians in quantum computing is to estimate physical quantities like the **ground state energy** of a system. 
This is at the heart of quantum algorithms such as the Variational Quantum Eigensolver (VQE), quantum simulation of materials, and quantum optimization.

Here, we demonstrate how to use MIMIQ to:

- Build the Hamiltonian for the **1D transverse-field Ising model**
- Apply **Trotterized time evolution** (first and second order)
- **Measure the energy** (expectation value of the Hamiltonian)

This provides a concrete template for simulating many-body quantum systems.

Ising Model
---------------------
.. _isinh-model:

The **1D transverse-field Ising model** is defined by the Hamiltonian:

.. math::

    H = -J \sum_{j=0}^{N-1} Z_j Z_{j+1} - h \sum_{j=0}^{N} X_j

This models a chain of spins with:

- nearest-neighbor interaction (`ZZ` terms),
- and a transverse magnetic field (`X` terms).

It’s widely used in condensed matter physics and quantum algorithm benchmarks.

Building the Hamiltonian 
----------------------------------
.. _building-the-hamiltonian:

To construct this model in MIMIQ you can use :class:`~mimiqcircuits.Hamiltonian` class to easily build your hamiltonian:

.. doctest::
    :hide:

    >>> from mimiqcircuits import *


.. doctest::

    >>> N = 4  # number of qubits (spins)
    >>> J = 1.0  # interaction strength
    >>> h = 0.5  # field strength

    >>> hamiltonian = Hamiltonian()

    >>> for j in range(N - 1):
    ...     _ = hamiltonian.push(-J, PauliString("ZZ"), j, j + 1)
   

    >>> for j in range(N):
    ...     _ = hamiltonian.push(-h, PauliString("X"), j)

    >>> print(hamiltonian)
    4-qubit Hamiltonian with 7 terms:
    ├── -1.0 * ZZ @ q[0,1]
    ├── -1.0 * ZZ @ q[1,2]
    ├── -1.0 * ZZ @ q[2,3]
    ├── -0.5 * X @ q[0]
    ├── -0.5 * X @ q[1]
    ├── -0.5 * X @ q[2]
    └── -0.5 * X @ q[3]


Simulating Time Evolution
--------------------------------
.. __simulating-time-evolution:

Suppose we want to apply :math:`e^{-iHt}` to a quantum state. This is useful for preparing ground states via imaginary-time evolution (approximate cooling),
or evolving an initial state in real time.

Because `H` has non-commuting terms, we use a **Trotter approximation**.

`First-order Trotterization (Lie)` (:meth:`~mimiqcircuits.Circuit.push_lietrotter`)

.. doctest::

    >>> c = Circuit()
    >>> c.push_lietrotter(hamiltonian, tuple(range(N)), t = 0.1, steps = 1)
    4-qubit circuit with 1 instructions:
    └── trotter(0.1) @ q[0,1,2,3]
    <BLANKLINE>
   
    >>> c.decompose()
    4-qubit circuit with 7 instructions:
    ├── RZZ(-0.2) @ q[0,1]
    ├── RZZ(-0.2) @ q[1,2]
    ├── RZZ(-0.2) @ q[2,3]
    ├── RX(-0.1) @ q[0]
    ├── RX(-0.1) @ q[1]
    ├── RX(-0.1) @ q[2]
    └── RX(-0.1) @ q[3]
    <BLANKLINE>
    

`Second-order Trotterization (Suzuki)` (:meth:`~mimiqcircuits.Circuit.push_suzukitrotter`)

.. doctest::

    >>> c = Circuit()
    >>> c.push_suzukitrotter(hamiltonian, tuple(range(N)), t = 0.1, steps = 1, order = 2)
    4-qubit circuit with 1 instructions:
    └── suzukitrotter_2(0.1) @ q[0,1,2,3]
    <BLANKLINE>
   
    >>> c.decompose()
    4-qubit circuit with 14 instructions:
    ├── RZZ(-0.1) @ q[0,1]
    ├── RZZ(-0.1) @ q[1,2]
    ├── RZZ(-0.1) @ q[2,3]
    ├── RX(-0.05) @ q[0]
    ├── RX(-0.05) @ q[1]
    ├── RX(-0.05) @ q[2]
    ├── RX(-0.05) @ q[3]
    ├── RX(-0.05) @ q[3]
    ├── RX(-0.05) @ q[2]
    ├── RX(-0.05) @ q[1]
    ├── RX(-0.05) @ q[0]
    ├── RZZ(-0.1) @ q[2,3]
    ├── RZZ(-0.1) @ q[1,2]
    └── RZZ(-0.1) @ q[0,1]
    <BLANKLINE>

Measuring the Energy
-----------------------
.. _measuring-energy:

Once the circuit has prepared the desired quantum state such as by applying time evolution with Trotterization we can measure the energy 
by evaluating the expectation value (:meth:`~mimiqcircuits.Circuit.push_expval`) of the Hamiltonian.

.. doctest::

    >>> c.push_expval(hamiltonian, *range(N))
    4-qubit, 7-zvar circuit with 16 instructions:
    ├── suzukitrotter_2(0.1) @ q[0,1,2,3]
    ├── ⟨ZZ⟩ @ q[0,1], z[0]
    ├── z[0] *= -1.0
    ├── ⟨ZZ⟩ @ q[1,2], z[1]
    ├── z[1] *= -1.0
    ├── ⟨ZZ⟩ @ q[2,3], z[2]
    ├── z[2] *= -1.0
    ├── ⟨X⟩ @ q[0], z[3]
    ├── z[3] *= -0.5
    ├── ⟨X⟩ @ q[1], z[4]
    ├── z[4] *= -0.5
    ├── ⟨X⟩ @ q[2], z[5]
    ├── z[5] *= -0.5
    ├── ⟨X⟩ @ q[3], z[6]
    ├── z[6] *= -0.5
    └── z[0] += 0.0 + z[1] + z[2] + z[3] + z[4] + z[5] + z[6]
    <BLANKLINE>

The result is stored in the Z-register — the first index contains the total expectation value (energy).
You can access it from the simulation result.
