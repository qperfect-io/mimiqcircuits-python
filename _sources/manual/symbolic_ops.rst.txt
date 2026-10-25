Symbolic Operations in MIMIQ
============================

This section provides detailed information on how to use symbolic operations in MIMIQ, including defining symbols, creating symbolic operations, substituting values, and running circuits with symbolic parameters.

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

Running Parameter Sweeps
------------------------
.. _running-parameter-sweeps:

To run the same symbolic circuit for many parameter values, describe the sweep with a :class:`~mimiqcircuits.ParameterTable` and pair it with the circuit in a :class:`~mimiqcircuits.SweepExperiment`. In a sweep every point binds the same symbols, so a table stores them once, with one row of values per point. Executing the experiment on a local backend returns one :class:`~mimiqcircuits.QCSResults` per point, in the order of the table.

**Example**:

.. code-block:: python

    from symengine import symbols
    import numpy as np
    from mimiqcircuits import *

    theta, phi = symbols("theta phi")

    # A single symbolic circuit with two parameters, followed by a
    # measurement.
    c = Circuit()
    c.push(GateRX(theta), 0)
    c.push(GateRZ(phi), 0)
    c.push(Measure(), 0, 0)

    # Every combination of 5 values of theta and 2 values of phi: 10 points.
    # theta varies slowest, phi fastest.
    table = gridtable({theta: np.linspace(0, np.pi, 5), phi: [0.0, np.pi / 2]})

    # The experiment checks that the table gives values for exactly the
    # symbols of the circuit, so a missing or misspelled symbol fails here.
    ex = SweepExperiment(c, table)

    # `sim` is any local backend.
    results = sim.execute(ex, nsamples=1000, seed=42)

    # One QCSResults per point. The probability of measuring 1 depends only
    # on theta: 0 at theta = 0, 1 at theta = pi.
    for point, res in zip(table, results):
        ones = sum(bs[0] for bs in res.cstates)
        print(point[theta], point[phi], ones / len(res.cstates))

Each point runs as a separate execution of the circuit evaluated at that point (``ex[i]``), with all the other arguments of ``execute`` unchanged. Every point gets its own seed, derived from ``seed`` (or ``rng``), so the points are statistically independent, and running the same sweep again with the same ``seed`` gives the same results.

A table can be built in several ways:

- ``ParameterTable([theta, phi], values)`` takes the symbols and an ``(npoints, nsymbols)`` array.
- ``ParameterTable.from_columns({theta: xs, phi: ys})`` takes one column per symbol, all of the same length: point ``i`` takes the ``i``-th value of each.
- ``gridtable({theta: xs, phi: ys})`` takes every combination of the values.
- ``ParameterTable.from_dicts(points)`` takes a list of dictionaries, one per point.

Tables also compose: ``a * b`` is the Cartesian product, ``a.hcat(b)`` pairs the points of two tables of the same length, and ``a.vcat(b)`` appends the points of ``b`` after those of ``a``.

``param_grid`` is a shorthand for the same thing: ``sim.execute(c, param_grid=points)``, with a list of dictionaries, runs exactly like ``sim.execute(SweepExperiment(c, ParameterTable.from_dicts(points)))`` and gives the same results for the same ``seed``. Unlike the experiment, it does not check the dictionaries in advance: a point that leaves a symbol unbound raises an error when it runs. An empty ``param_grid`` runs the circuit once, as if it was not given.

Sweeps run on local backends only. Remote backends raise :class:`NotImplementedError` for ``param_grid``; to run a sweep remotely, submit the list of evaluated circuits, ``list(ex)``.

Reference
---------

.. autofunction:: mimiqcircuits.AbstractOperator.evaluate
    :noindex:
