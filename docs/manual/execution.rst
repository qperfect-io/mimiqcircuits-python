Circuit Execution on MIMIQ
==========================

Understanding sampling and executions
-------------------------------------

Executions on MIMIQ are carried out in different ways depending on the
type of the circuit submitted. There are three fundamental cases based
on the presence of non-unitary operations such as ``Measure``,
``IfStatement``, or ``Reset``.

No non-unitary operations
~~~~~~~~~~~~~~~~~~~~~~~~~

In this case the circuit is executed only once and the final state is
sampled as many times as specified by the number of samples
(``nsamples``) parameter of the execution. The sampled value of all the
qubits is returned (in the obvious ordering).

No mid-circuit measurements and no non-unitary operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case the circuit is executed only once again, and the final
state is sampled as many times as specified by ``nsamples``, but only
the sampled value of all the classical bits used in the circuit is
returned (usually the targets of the measurements at the end of the
circuit).

Mid-circuit measurements or non-unitary operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case the circuit is executed ``nsamples`` times, and the final
state is sampled only once per run. The sampled value of all the
classical bits used in the circuit is returned.

Fidelity and Error estimates
----------------------------

Since we allow for the execution of circuits on MIMIQ with non exact
methods, we return always a **fidelity estimate** for each execution.

Fidelity in this case is defined as the squared modulus of the overlap
between the final state obtained by the execution and the ideal one. It
is a number between ``0`` and ``1``, where ``1`` means that the final
state is exactly the one we wanted to obtain.

The fidelity will always be ``1.0`` for exact methods, but it can be
less than that for non exact methods.

In the case of **MPS** methods, the number returned is an estimate of
the actual fidelity of the state. More specifically, it is a **lower
bound** for the fidelity, meaning that the actual fidelity will always
be larger or equal to the number reported.

The **average multiqubit error estimate** is the higher bound of the
error done in average when applying a two qubit gate.

Timings
-------

+-----------------------------------+-----------------------------------+
| Name                              | Description                       |
+===================================+===================================+
| apply                             | Time used to apply all the        |
|                                   | unitaries of the circuit to the   |
|                                   | initial state. It also includes   |
|                                   | the time to allocate the initial  |
|                                   | state.                            |
+-----------------------------------+-----------------------------------+
| compression                       | Time used to compress the circuit |
|                                   | or convert it into a format that  |
|                                   | is more efficient for the         |
|                                   | execution on MIMIQ.               |
+-----------------------------------+-----------------------------------+
| sample                            | Time used to sample the final     |
|                                   | state of the circuit.             |
+-----------------------------------+-----------------------------------+
| amplitudes                        | Time used to retrieve the         |
|                                   | amplitudes of the final state of  |
|                                   | the circuit.                      |
+-----------------------------------+-----------------------------------+
| total                             | Time elapsed from the start of    |
|                                   | the execution to the end          |
+-----------------------------------+-----------------------------------+

When reading these timings consider that sampling only account for the
time needed to sample the final state. If a circuit needs to be executed
multiple time to obtain such samples, the “sample” time will be zero.
The contrary is true for the “apply” time, which is the sum of the apply
time for all the executions of the circuit.
