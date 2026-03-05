Quick Start
===========

In this tutorial, we will walk you through the fundamental procedures for simulating a quantum circuit using MIMIQ. Throughout the tutorial, we will provide links to detailed documentation and examples that can provide a deeper understanding of each topic.

Install and load MIMIQ
----------------------

To install MIMIQ use the following command:

.. code-block::

    pip install "mimiqcircuits @ git+https://github.com/qperfect-io/mimiqcircuits-python.git"


Check the :doc:`installation <manual/installation>` page for more details.

In order to use MIMIQ, we simply need to import the `mimiqcircuits` Python module within your workspace like this:

.. doctest:: quick_start

    from mimiqcircuits import *

.. doctest:: quick_start
    :hide:

    >>> from mimiqcircuits import *
    >>> import os

    >>> conn = MimiqConnection(url="https://mimiq.qperfect.io")
    >>> conn.connect(os.getenv("MIMIQUSER"), os.getenv("MIMIQPASS"))
    MimiqConnection:
    ├── url: https://mimiq.qperfect.io
    ├── Max time limit per request: 360 minutes
    ├── Default time limit is equal to max time limit: 360 minutes
    └── status: open

Connect to remote service
-------------------------

To execute circuits you have to connect to MIMIQ's remote service, which can be achieved with the following instructions

.. doctest:: quick_start

    conn = MimiqConnection()
    conn.connect()

For more details see :doc:`cloud execution page<manual/remote_execution>` or see the documentation of :meth:`~mimiqcircuits.MimiqConnection.connect`. If executed without supplemental arguments, :meth:`~mimiqcircuits.MimiqConnection.connect` will start a local webpage and will try to open it with your default browser. As an alternative, :code:`connect("john.smith@example.com", "jonspassword")` allows to insert directly the username and password of the user.

.. note::

    In order to complete this step you need an active subscription to MIMIQ. To obtain one, please `contact us <https://qperfect.io>`_ or, if your organization already has a subscription, contact the organization account holder.

Example: Simulate a GHZ circuit
-------------------------------

Construct basic circuit
^^^^^^^^^^^^^^^^^^^^^^^

A circuit is basically a sequence of quantum operations (gates, measurements, etc...) that act on a set of qubits and potentially store information in a classical or "z" register (see :doc:`circuit page<manual/circuits>`). The classical register is a boolean vector to store the results of measurements, and the z register is a complex vector to store the result of mathematical calculations like expectation values.

The MIMIQ interface is similar to other software, but there are some important differences:

- There are no hardcoded quantum registers. Qubits are simply indicated by an integer index starting at 0 (Python convention). The same for classical and z registers.
- A :class:`~mimiqcircuits.Circuit` object doesn't have a fixed number of qubits. The number of qubits in a circuit is taken from looking at the qubits the gates are applied to. It is the maximum integer index used in a circuit. The same for the number of classical bits.

To construct a GHZ circuit, we start by defining an empty :meth:`~mimiqcircuits.Circuit`

.. doctest:: quick_start

    >>> circuit = Circuit()


The most important tool to build circuits is the :meth:`~mimiqcircuits.Circuit.push` function. It is used like this: :code:`circuit.push(quantum_operation, targets...)`. It accepts a circuit, a single quantum operation, and a series of targets, one for every qubit or bit the operation supports.

We apply a :class:`~mimiqcircuits.GateH` on the first qubit as

.. doctest:: quick_start

    >>> circuit.push(GateH(), 0)
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>


The text representation `H @ q[0]` informs us that there is an instruction which applies the Hadamard gate to the qubit with index `0`.
Note that qubits start by default in the state `0`.

Multiple gates can be added at once through the same :meth:`~mimiqcircuits.Circuit.push` syntax using iterables, see :doc:`circuit <manual/circuits>` and :doc:`unitary gates<manual/unitary_gates>` page for more information.
To prepare a 5-qubit GHZ state, we add 5 CX gates or control-`X` gates between the qubit 0 and all the qubits from 1 to 4.

.. doctest:: quick_start

    >>> circuit.push(GateCX(), 0, range(1, 5))
    5-qubit circuit with 5 instructions:
    ├── H @ q[0]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    ├── CX @ q[0], q[3]
    └── CX @ q[0], q[4]
    <BLANKLINE>


Measure, add noise, and extract information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can extract information about the state of the system (without affecting the state) at any point in the circuit, see :doc:`statistical operations<manual/statistical_ops>` page.
For example, we can compute the expectation value of :math:`| 11 \rangle\langle 11 |` of qubits 0 and 3, and store it in the first z register as:

.. doctest:: quick_start

    >>> circuit.push(ExpectationValue(Projector11()), 0, 4, 0)
    5-qubit, 1-zvar circuit with 6 instructions:
    ├── H @ q[0]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    ├── CX @ q[0], q[3]
    ├── CX @ q[0], q[4]
    └── ⟨P₁₁(1)⟩ @ q[0,4], z[0]
    <BLANKLINE>


We can measure the qubits and add other :doc:`non-unitary operations <manual/non_unitary_ops>` at any point in the circuit, for example:

.. doctest:: quick_start

    >>> circuit.push(Measure(), range(0, 5), range(0, 5))
    5-qubit, 5-bit, 1-zvar circuit with 11 instructions:
    ├── H @ q[0]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    ├── CX @ q[0], q[3]
    ├── CX @ q[0], q[4]
    ├── ⟨P₁₁(1)⟩ @ q[0,4], z[0]
    ├── M @ q[0], c[0]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    └── M @ q[4], c[4]
    <BLANKLINE>


Here, we measure qubits 0 to 4 and store the result in classical register 0 to 4.
In general, the ordering of targets is always like :code:`circuit.push(op, quantum_targets..., classical_targets..., z_targets...)`.

.. warning::

    Classical and z registers can be overwritten. If you do :code:`circuit.push(Measure(), 0, 0)` followed by :code:`circuit.push(Measure(), 1, 0)`, the second measurement will overwrite the first one since it will be stored on the same classical register 0. To avoid this in a circuit with many measurements you can, for example, keep track of the index of the last used register.

To simulate imperfect quantum computers we can add noise to the circuit. Noise operations can be added just like any other operations using `push`. However, noise can also be added after the circuit has been built to all gates of a certain type using :meth:`~mimiqcircuits.Circuit.add_noise`. For example:

.. doctest:: quick_start

    >>> circuit.add_noise(GateH(), AmplitudeDamping(0.01))
    5-qubit, 5-bit, 1-zvar circuit with 12 instructions:
    ├── H @ q[0]
    ├── AmplitudeDamping(0.01) @ q[0]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    ├── CX @ q[0], q[3]
    ├── CX @ q[0], q[4]
    ├── ⟨P₁₁(1)⟩ @ q[0,4], z[0]
    ├── M @ q[0], c[0]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    └── M @ q[4], c[4]
    <BLANKLINE>
    >>> circuit.add_noise(GateCX(), Depolarizing2(0.1), parallel=True)
    5-qubit, 5-bit, 1-zvar circuit with 16 instructions:
    ├── H @ q[0]
    ├── AmplitudeDamping(0.01) @ q[0]
    ├── CX @ q[0], q[1]
    ├── Depolarizing(0.1) @ q[0,1]
    ├── CX @ q[0], q[2]
    ├── Depolarizing(0.1) @ q[0,2]
    ├── CX @ q[0], q[3]
    ├── Depolarizing(0.1) @ q[0,3]
    ├── CX @ q[0], q[4]
    ├── Depolarizing(0.1) @ q[0,4]
    ├── ⟨P₁₁(1)⟩ @ q[0,4], z[0]
    ├── M @ q[0], c[0]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    └── M @ q[4], c[4]
    <BLANKLINE>
    >>> circuit.add_noise(Measure(), PauliX(0.05), before=True, parallel=True)
    5-qubit, 5-bit, 1-zvar circuit with 21 instructions:
    ├── H @ q[0]
    ├── AmplitudeDamping(0.01) @ q[0]
    ├── CX @ q[0], q[1]
    ├── Depolarizing(0.1) @ q[0,1]
    ├── CX @ q[0], q[2]
    ├── Depolarizing(0.1) @ q[0,2]
    ├── CX @ q[0], q[3]
    ├── Depolarizing(0.1) @ q[0,3]
    ├── CX @ q[0], q[4]
    ├── Depolarizing(0.1) @ q[0,4]
    ├── ⟨P₁₁(1)⟩ @ q[0,4], z[0]
    ├── PauliX(0.05) @ q[0]
    ├── PauliX(0.05) @ q[0]
    ├── PauliX(0.05) @ q[0]
    ├── PauliX(0.05) @ q[0]
    ├── PauliX(0.05) @ q[0]
    ├── M @ q[0], c[0]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ⋮   ⋮
    └── M @ q[4], c[4]
    <BLANKLINE>


See :doc:`symbolic operations<manual/symbolic_ops>` and :doc:`special operations<manual/special_ops>` pages for other supported operations.

The number of qubits, classical bits, and complex z-values of a circuit can be obtained from:

.. doctest:: quick_start

    >>> circuit.num_qubits(), circuit.num_bits(), circuit.num_zvars()
    (5, 5, 1)


A circuit behaves in many ways like a vector (of instructions, i.e. operations + targets). You can get the length as :code:`len(circuit)`, access elements as `circuit[2]`, insert elements with :meth:`~mimiqcircuits.Circuit.insert`, append other circuits with :meth:`~mimiqcircuits.Circuit.append` etc. You can also visualize circuits with :meth:`~mimiqcircuits.Circuit.draw`. See :doc:`circuit page <manual/circuits>` for more information.

Execute circuit
^^^^^^^^^^^^^^^

Executing a circuit on MIMIQ requires three steps:

1. opening a connection to the MIMIQ Remote Services (which we did at the beginning of the tutorial),
2. send a circuit for execution,
3. retrieve the results of the execution.

After a connection has been established, an execution can be sent to the remote services using :meth:`~mimiqcircuits.MimiqConnection.schedule`.

.. doctest:: quick_start

    >>> job = conn.schedule(circuit)


This will execute a simulation of the given circuit with default parameters. The default choice of algorithm is `"auto"`.  Generally, there are three available options:

* `"auto"` for the automatically selecting the best algorithm according to circuit size and complexity,
* `"statevector"` for a highly optimized state vector engine, and
* `"mps"` for the large-scale Matrix Product States (MPS) method.

Check out the documentation of the :meth:`~mimiqcircuits.MimiqConnection.schedule` function for details.

Once the execution has finished, the results can be retrieved via the :meth:`~mimiqcircuits.MimiqConnection.getresults` function, which returns a :class:`~mimiqcircuitsQCSResults` structure.

.. doctest:: quick_start

    >>> res = conn.get_result(job)


To make a histogram out of the retrieved samples, it suffices to execute

.. doctest:: quick_start

      >>> res.histogram()
      {frozenbitarray('00000'): 289, frozenbitarray('11111'): 307, frozenbitarray('00001'): 16, frozenbitarray('00111'): 6, frozenbitarray('10000'): 90, frozenbitarray('10011'): 16, frozenbitarray('01100'): 15, frozenbitarray('00011'): 3, frozenbitarray('11110'): 23, frozenbitarray('01111'): 84, frozenbitarray('10111'): 29, frozenbitarray('01110'): 15, frozenbitarray('01000'): 19, frozenbitarray('11100'): 6, frozenbitarray('10001'): 20, frozenbitarray('11101'): 11, frozenbitarray('10010'): 3, frozenbitarray('00100'): 9, frozenbitarray('01001'): 2, frozenbitarray('00010'): 12, frozenbitarray('11011'): 8, frozenbitarray('10100'): 3, frozenbitarray('01101'): 2, frozenbitarray('11010'): 1, frozenbitarray('01011'): 3, frozenbitarray('11001'): 1, frozenbitarray('10101'): 3, frozenbitarray('00101'): 1, frozenbitarray('11000'): 3}


To plot the results use the following:

.. doctest:: quick_start

    >>> from mimiqcircuits.visualization import plothistogram
    >>> plothistogram(res)
    <Figure size 960x720 with 1 Axes>


Check the :doc:`cloud execution<manual/remote_execution>` page for more details on job handling.



OpenQASM and Stim
^^^^^^^^^^^^^^^^^

OpenQASM and Stim files, defining quantum algorithms can be executed on MIMIQ in the same way native circuits can, simply use :meth:`~mimiqcircuits.MimiqConnection.schedule` and provide the path of the file to upload.
See the :doc:`import-export<manual/import_export>` page for more details on how include files are handled.

