Local Statevector Simulation
============================

To simulate circuits on your own machine, use **Exaqt**, QPerfect's
statevector simulator. It is free, ships as the separate package
``mimiq-exaqt`` on PyPI, and runs MIMIQ circuits directly: no account or
connection is needed.

Exaqt simulates the full statevector, so memory grows as :math:`2^n` with the
number of qubits :math:`n`. It is the right tool for debugging and for
circuits up to a few tens of qubits. For larger circuits, use the
:doc:`MIMIQ cloud <remote_execution>`.


Install Exaqt
-------------

.. code-block:: shell

    pip3 install mimiq-exaqt

Wheels are published for Linux (x86_64), macOS (arm64) and Windows (x86_64),
for Python 3.10 and newer. The package installs as ``mimiq-exaqt`` and is
imported as ``exaqt``.


Run a circuit
-------------

Build a circuit with mimiqcircuits as usual, create the simulator, and call
``execute``. Every execution starts from :math:`|0\dots0\rangle`, and the
result is a :class:`~mimiqcircuits.QCSResults`, the same type the MIMIQ cloud
returns.

.. code-block:: python

    from mimiqcircuits import *
    from exaqt import ExaqtQCS

    # A Bell pair, measured into two classical bits.
    circuit = Circuit()
    circuit.push(GateH(), 0)
    circuit.push(GateCX(), 0, 1)
    circuit.push(Measure(), range(2), range(2))

    sim = ExaqtQCS()

    # nsamples sets the number of shots; seed makes the run reproducible.
    results = sim.execute(circuit, nsamples=1000, seed=123)
    print(results.histogram())   # about half bs"00", half bs"11"

A seed can also be fixed once, for every execution of the simulator:
``ExaqtQCS(seed=42)``.


Amplitudes and expectation values
---------------------------------

Exaqt does not hand out its state. To read amplitudes or expectation values,
put :class:`~mimiqcircuits.Amplitude` and
:class:`~mimiqcircuits.ExpectationValue` operations in the circuit, as you
would for the cloud. Each one writes its value to a Z-register variable, and
the values come back in ``results.zstates``.

.. code-block:: python

    circuit = Circuit()
    circuit.push(GateH(), 0)
    circuit.push(GateCX(), 0, 1)

    # Amplitude of |00>, stored in z[0].
    circuit.push(Amplitude(BitString("00")), 0)

    # <ZZ> on qubits 0 and 1, stored in z[1].
    circuit.push(ExpectationValue(PauliString("ZZ")), 0, 1, 1)

    results = ExaqtQCS().execute(circuit)
    print(results.zstates[0])    # about [0.7071, 1.0]

Exaqt's own documentation is bundled with the package. Open it with
``exaqt docs``.
