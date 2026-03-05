Quantanium
==========

On this page you can find all the information to install and use the local statevector simulator Quantanium


Install quantanium
----------------------------------------------------

the local statevector quantanium is an optional dependency of mimiqcircuits.
It can be installed with pip:

.. code-block:: shell

    pip3 install mimiqcircuits[quantanium]


Basic usage
----------------------------------------------------

Quantanium can be used to execute a circuit.
It must be instantiated before calling the method :code:`execute`.
Execute will always start from the initial state \|0...0\>.

.. doctest:: python

    >>> from quantanium import *
    >>> from mimiqcircuits import *
    >>> circuit = Circuit()
    >>> circuit.push(GateH(), 0) 
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>

    >>> sim = Quantanium()
    >>> sim.execute(circuit)
    QCSResults:
    ├── simulator: Quantanium StateVector 1.0
    ├── timings:
    │    ├── sample time: 0.000122642s
    │    └── apply time: 4.3511e-05s
    ├── fidelity estimate: 1
    ├── average multi-qubit gate error estimate: 0
    ├── most sampled:
    │    ├── bs"1" => 527
    │    └── bs"0" => 473
    ├── 1 executions
    ├── 0 amplitudes
    └── 1000 samples

:code:`execute` returns a QCSResult object.

Execute options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`execute` method of Quantanium supports a subset of the options available on MIMIQ remote.
Here is an example of all the  options available:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateH(), 0) 
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>

    >>> sim = Quantanium()
    >>> sim.execute(circuit, nsamples=15, seed=123)
    QCSResults:
    ├── simulator: Quantanium StateVector 1.0
    ├── timings:
    │    ├── sample time: 2.863e-06s
    │    └── apply time: 1.8508e-05s
    ├── fidelity estimate: 1
    ├── average multi-qubit gate error estimate: 0
    ├── most sampled:
    │    ├── bs"1" => 10
    │    └── bs"0" => 5
    ├── 1 executions
    ├── 0 amplitudes
    └── 15 samples


Get Statevector
--------------------------------

After running a simulation with execute or evolve you can extract the statevector using the :code:`get_statevector` method.
Users can not modify the state directly.

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateH(), 0) 
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>

    >>> sim = Quantanium()
    >>> sim.execute(circuit, seed=123)
    QCSResults:
    ├── simulator: Quantanium StateVector 1.0
    ├── timings:
    │    ├── sample time: 7.9968e-05s
    │    └── apply time: 1.6692e-05s
    ├── fidelity estimate: 1
    ├── average multi-qubit gate error estimate: 0
    ├── most sampled:
    │    ├── bs"0" => 509
    │    └── bs"1" => 491
    ├── 1 executions
    ├── 0 amplitudes
    └── 1000 samples
    >>> sim.get_statevector()
    [(0.7071067811865475+0j), (0.7071067811865475+0j)]


Evolve Statevector
-------------------------------

The state can also be evolved without restarting from the initial state everytime by using the method :code:`evolve`:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateH(), 0) 
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>

    >>> sim = Quantanium()
    >>> sim.evolve(circuit)
    >>> sim.evolve(circuit) # apply the circuit again
    >>> sim.get_statevector()
    [(0.9999999999999998+0j), (2.2371143170757382e-17+0j)]


Parse QASM
-------------------------------
The local statevector simulator comes with its own OpenQASM 2 parser
it can be used with :code:`execute`:


.. doctest:: python

    >>> qasm = """
    ... // Implementation of Deutsch algorithm with two qubits for f(x)=x
    ... // taken from https://github.com/pnnl/QASMBench/blob/master/small/deutsch_n2/deutsch_n2.qasm
    ... OPENQASM 2.0;
    ... include "qelib1.inc";
    ... qreg q[2];
    ... creg c[2];
    ... x q[1];
    ... h q[0];
    ... h q[1];
    ... cx q[0],q[1];
    ... h q[0];
    ... measure q[0] -> c[0];
    ... measure q[1] -> c[1];
    ... """

    # Write the OPENQASM as a file
    >>> with open("/tmp/deutsch_n2.qasm", "w") as file:
    ...     file.write(qasm)
    308

    >>> sim = Quantanium()
    >>> sim.execute("/tmp/deutsch_n2.qasm")
    QCSResults:
    ├── simulator: Quantanium StateVector 1.0
    ├── timings:
    │    ├── sample time: 9.8686e-05s
    │    └── apply time: 3.7855e-05s
    ├── fidelity estimate: 1
    ├── average multi-qubit gate error estimate: 0
    ├── most sampled:
    │    ├── bs"11" => 527
    │    └── bs"10" => 473
    ├── 1 executions
    ├── 0 amplitudes
    ├── 0 amplitudes
    └── 1000 samples

Reference
---------

.. autoclass:: mimiqcircuits.quantanium.Quantanium
    :noindex:


