Symbolic Operations in MIMIQ
============================

This section provides detailed information on how to use symbolic operations in MIMIQ, including defining symbols, creating symbolic operations, substituting values, and running circuits with symbolic parameters.

contents
========
.. contents::
   :local:
   :depth: 2
   :backlinks: entry


When Symbolic Operations Can Be Useful
---------------------------------------
.. _when-symbolic-operations-can-be-useful:

Symbolic operations are valuable in several quantum computing scenarios:

- **Parameter Optimization**: In algorithms like the **Variational Quantum Eigensolver (VQE)**, parameters need to be optimized iteratively. Using symbolic variables allows you to define a circuit once and update it with new parameter values during the optimization process. However, before executing the circuit, you must substitute the symbolic parameters with concrete values. This approach simplifies managing parameterized circuits.

- **Circuit Analysis**: Symbolic operations are useful for analyzing the structure of a quantum circuit. By keeping parameters symbolic, you can explore how different components of the circuit affect the output, such as measurement probabilities or expectation values, without needing to reconstruct the circuit.

.. **Parameter Optimization in VQE**:

.. 1. Preparing a parameterized quantum state.
.. 2. Measuring the expectation value of the Hamiltonian.
.. 3. Updating the parameters to minimize this expectation value.

.. - **Step 1**: Use symbolic variables to define the parameterized circuit.
.. - **Step 2**: Substitute the symbolic variables with specific values during each optimization iteration and execute the circuit.

.. **Exploring Parameter Sensitivity in Circuit Analysis**:

.. - **Step 1**: Use symbolic variables to define the parameterized circuit.
.. - **Step 2**: Substitute different values for the symbolic parameters and analyze the resulting circuit outputs.

.. .. warning::

..     Symbolic parameters are useful tools for circuit design and analysis, but before executing a circuit on a simulator, you must substitute all symbolic parameters with numerical values.



Defining Symbols
----------------
.. _defining-symbols:

MIMIQ leverages symbolic libraries like `symengine` to define symbolic variables, acting as placeholders for parameters in quantum operations.
To define symbols in MIMIQ, you will need to create symbolic variables and then use them as parameters for parametric operations. Here's how to get started:

**Example**:

.. doctest:: symbolic
    :hide:

    >>> from mimiqcircuits import *  # hide
   
    
.. doctest:: symbolic

    >>> from symengine import *
    >>> theta, phi = symbols('theta phi')

Defining Symbolic Operations
----------------------------
.. _defining-symbolic-operations:

Once you have defined symbols, you can use them in your quantum circuit operations. This allows you to create parameterized gates that depend on symbolic variables.

**Example**:


.. doctest:: symbolic

    >>> c = Circuit()
    >>> c.push(GateRX(theta), 1)
    2-qubit circuit with 1 instructions:
    └── RX(theta) @ q[1]
    <BLANKLINE>
    >>> c.push(GateRY(phi), 2)
    3-qubit circuit with 2 instructions:
    ├── RX(theta) @ q[1]
    └── RY(phi) @ q[2]
    <BLANKLINE>
    >>> c
    3-qubit circuit with 2 instructions:
    ├── RX(theta) @ q[1]
    └── RY(phi) @ q[2]
    <BLANKLINE>

In this example, `theta` and `phi` are symbolic variables that can be used in operations.
  
Substituting Symbols with Values
--------------------------------
.. _substituting-symbols-with-values:

Before executing a circuit that includes symbolic parameters, you need to replace these symbols with specific numerical values. This is done using a dictionary to map each symbolic variable to its corresponding value. Here is an example of how to do it. 

**Example**:

.. doctest:: symbolic

    >>> evaluated_circuit = c.evaluate({"theta": pi/2, "phi": pi/4})
    >>> evaluated_circuit
    3-qubit circuit with 2 instructions:
    ├── RX((1/2)*pi) @ q[1]
    └── RY((1/4)*pi) @ q[2]
    <BLANKLINE>

In this example, :meth:`~mimiqcircuits.AbstractOperator.evaluate` is used to create a new circuit where `theta` is replaced by `π/2` and `phi` by `π/4`.
