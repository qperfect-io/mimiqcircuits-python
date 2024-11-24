Simulating Circuits
===================

This page provides information on how MIMIQ simulates quantum circuits.

Contents
========
.. contents::
   :local:
   :depth: 2
   :backlinks: entry

Simulator Backends
------------------
.. _simulator-backends:

To execute quantum circuits with MIMIQ you can use different simulator backends, i.e. different numerical methods to apply operations to the state of the qubits. 
Here we give a short introduction into these methods.

State Vector
~~~~~~~~~~~~
.. _state-vector:

The pure quantum state of a system of `N` qubits can be represented exactly by `2N` 
complex numbers. The state vector method consists in explicitly storing the full state of the system and 
evolving it exactly as described by the given quantum algorithm. The method can be considered **exact** up to 
machine precision errors (due to the finite representation of complex numbers on a computer). 
Since every added qubit doubles the size of the state, this method is practically impossible to 
be used on systems > 50 qubits. On a laptop with 8GB of free RAM, only 28-29 qubits can be simulated.

On our current remote service, we can simulate circuits 
of up to 32 qubits with this method. Premium plans and on-premises solutions are designed to 
increase this limit. If you are interested, contact us at contact@qperfect.io.

Our state vector simulator is highly optimized for simulation on a single CPU, 
delivering a significant speedup with respect to publicly available alternatives.

**Performance Tips:**

The efficiency of the state vector simulator can depend on the specific way that the circuit is implemented. 
Specifically, it depends on how gates are defined by the user. The most important thing is to 
avoid using [:class:`~mimiqcircuits.GateCustom`] whenever possible, and instead use specific gate 
implementations. For example, use :class:`~mimiqcircuits.GateX` instead of `GateCustom(Matrix([[0,1], [1,0]]))`, 
and use `CU(...)` instead of `GateCustom(matrix_of_GateCU)`.

Matrix-Product States (MPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _matrix-product-states:

Matrix-Product States (MPS) algorithms were originally developed in the field of many-body quantum physics 
(see the `Wikipedia article <https://en.wikipedia.org/wiki/Matrix_product_state>`_ and references therein), and they offer an 
alternative approach to scaling up the size of simulations. Instead of tracking the entire quantum state, 
MPS efficiently captures its entanglement structure. This method excels in scenarios where entanglement 
is localized, providing significant computational advantages and reducing memory requirements compared to 
state vector simulations. 

Key benefits of MPS include the ability to simulate larger quantum systems and to represent states with 
substantial entanglement efficiently. However, challenges arise in circuits with extensive long-range 
entanglement, where the method may lose efficiency.

Our MPS simulator is optimized for both execution speed and fidelity. For circuits with bounded 
entanglement, MPS can simulate systems of hundreds of qubits exactly. For circuits with high entanglement, 
it provides approximate solutions along with a lower bound estimate of the actual fidelity (see 
:ref:`Fidelity and Error Estimates <fidelity-and-error-estimates>` section).


**Performance Tips:**
The efficiency of the MPS simulation can depend on implementation details.
To optimize performance, you can vary the following parameters that specify the level of approximation and 
compression (see also :doc:`remote executon </manual/remote_execution>`):

- **Bond dimension** (bonddim): The bond dimension specifies the maximal dimension of the matrices that appear in the MPS representation. Choosing a larger *bonddim* means we can simulate more entanglement, which leads to larger fidelities, at the cost of more memory and typically longer run times. However, when *bonddim* is chosen large enough that the simulation can be run exactly, then it generally runs much faster than lower bond dimensions. The default is 256.

- **Entanglement dimension** (entdim): The entanglement dimension is related to the way gates are applied to the state. In some cases, a large *entdim* can lead to better compression and thus shorter runtimes, at the potential cost of more memory. In others, a lower *entdim* is more favourable. The default is 16.

Moreover, the performance of MPS, especially the bond dimension required, also depends on the specific way that circuits are implemented. Here are some general tips:

- **Qubit ordering:** The most crucial choice that affects MPS performance is the ordering of qubits in the circuit. If we have qubits 1 to N, it matters which qubit of the algorithm has which index. Ideally, the indices should be chosen such that qubits that are strongly entangled during the circuit are close to each other, i.e. small ``|i-j|`` where `i` and `j` are indices. When a good ordering is chosen, this will translate into lower bond dimensions.

- **Gate ordering:** In nature, the order of transversal gates, or gates that commute with each other, does not play a role. However, for MPS it can change the performance. The reason is that in a simulation we typicall apply gates, even transversal ones, sequentially. During the application of the gates, the entanglement of the intermediate state can depend on the order in which the gates are applied. Thus, experimenting with different gate orderings can lead to better performance, typically because of lower bond dimensions.

You can access the bond dimensions of the state during execution through [:class:`~mimiqcircuits.BondDim`], see also page on [:ref:`entanglement<BitString>`]. This can be helpful to understand the effect of different optimizations.

Fidelity and Error Estimates
----------------------------
.. _fidelity-and-error-estimates:

Since we allow for the execution of circuits on MIMIQ with non exact methods (MPS), 
we return always a **fidelity estimate** for each execution.

Fidelity in this case is defined as the squared modulus of the overlap between the final state obtained by the execution and the ideal one. 
It is a number between `0` and `1`, where `1` means that the final state is exactly the one we wanted to obtain.
The fidelity will always be `1.0` for exact methods (State Vector), but it can be less than that for non exact methods.

In the case of **MPS** methods, the number returned is an estimate of the actual fidelity of the state. More specifically, it is a **lower bound** for the 
fidelity, meaning that the actual fidelity will always be larger or equal to the number reported.
For example, if the fidelity is 0.8 it means that the state computed by MPS has at least an 80% overlap with the real solution.

Understanding Sampling
----------------------
.. _understanding-sampling:

When running a circuit with MIMIQ we compute and return measurement samples, among other quantities (see :doc:`Cloaud Execution </manual/remote_execution>` section). Which measurement samples are returned depends on the type of circuit 
executed. There are three fundamental cases based on the presence of 
:doc:`non-unitary operations </manual/non_unitary_ops>` such as measurements (:class:`~mimiqcircuits.Measure`...), resets (:class:`~mimiqcircuits.Reset`...), if statements (:class:`~mimiqcircuits.IfStatement`), or :doc:`noise </manual/noise>`.

**No Non-Unitary Operations**

In this case the circuit is executed only once and the final state is sampled as many times as specified by the  number of samples (`nsamples`) parameter of the execution. The sampled value of all the qubits is returned (in the obvious ordering).

**No Mid-Circuit Measurements or Non-Unitary Operations**

In this case the circuit is executed only once again, and the final state is sampled as many times as specified by `nsamples`, but only 
the sampled value of all the classical bits used in the circuit is returned (usually the targets of the measurements at the end of the circuit).

**Mid-Circuit Measurements or Non-Unitary Operations**

In this case the circuit is executed `nsamples` times, and the final state is sampled only once per run. The sampled value of all the classical bits used in the circuit is returned.
