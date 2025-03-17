Noisy simulations on MIMIQ
==========================

This page provides detailed information on simulating noise in quantum circuits using MIMIQ.

Contents
========
.. contents::
   :local:
   :depth: 2
   :backlinks: entry

Summary of noise functionality
------------------------------
.. _summary-of-noise-functionality:

Custom noise channels:

* :class:`~mimiqcircuits.Kraus`
* :class:`~mimiqcircuits.MixedUnitary`

Specialized noise channels:

* :class:`~mimiqcircuits.Depolarizing`
* :class:`~mimiqcircuits.Depolarizing1`
* :class:`~mimiqcircuits.Depolarizing2`
* :class:`~mimiqcircuits.PauliNoise`
* :class:`~mimiqcircuits.PauliX`
* :class:`~mimiqcircuits.PauliY`
* :class:`~mimiqcircuits.PauliZ`
* :class:`~mimiqcircuits.AmplitudeDamping`
* :class:`~mimiqcircuits.GeneralizedAmplitudeDamping`
* :class:`~mimiqcircuits.PhaseAmplitudeDamping`
* :class:`~mimiqcircuits.ThermalNoise`
* :class:`~mimiqcircuits.ProjectiveNoise`

Note that the :class:`~mimiqcircuits.Reset` type operations can also be thought of as noisy operations.
Coherent noise can be added by using any of the supported gates (e.g., :class:`~mimiqcircuits.Gate`).

Noise channels come with the following methods:

* :func:`~mimiqcircuits.krauschannel.krausmatrices` and :func:`~mimiqcircuits.krauschannel.krausoperators`
* :func:`~mimiqcircuits.krauschannel.unitarymatrices` and :func:`~mimiqcircuits.PauliNoise.unitarygates` (only for mixed-unitary)
* :func:`~mimiqcircuits.krauschannel.probabilities` (only for mixed-unitary)
* :func:`~mimiqcircuits.krauschannel.ismixedunitary`

To add noise channels to a circuit, you can use:

* :func:`~mimiqcircuits.Circuit.push` (like gates)
* :func:`~mimiqcircuits.Circuit.add_noise` (to add noise to every instance of a gate)

To generate one sample of a circuit with mixed unitaries, use:

* :func:`~mimiqcircuits.Circuit.sample_mixedunitaries`

See below for further information. You can also run help(Circuit().sample_mixedunitaries).

Mathematical background
-----------------------
.. _mathematical-background:

Kraus operators
~~~~~~~~~~~~~~~

Noise in a quantum circuit refers to any kind of unwanted interaction of the qubits with the 
environment (or with itself). Mathematically, this puts us in the framework of open systems 
and the state of the qubits now needs to be described in terms of a density matrix :math:`\rho`, 
which fulfills :math:`\rho = \rho^\dagger` and :math:`\mathrm{Tr} (\rho) = 1`.

A quantum operation such as noise can then be described using the Kraus operator representation as:

.. math::

    \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger.

We consider only completely positive and trace-preserving (CPTP) operations.
In this case, the Kraus operators :math:`E_k` can be any matrix as long as they fulfill the completeness relation:

.. math::

    \sum_k E_k^\dagger E_k = I.

Note that unitary gates :math:`U` just correspond to a single Kraus operator, :math:`E_1 = U`.

When all Kraus operators are proportional to a unitary matrix, :math:`E_k = \alpha_k U_k`, this is called a mixed-unitary quantum operation and can be written as:

.. math::

    \mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger,

where :math:`p_k = |\alpha_k|^2`.

Such operations are easier to implement as we'll see below.

**Remarks**:

- Unitary gates :math:`U` correspond to a single Kraus operator, :math:`E_1 = U`.
- The number of Kraus operators depends on the noise considered.
- For a given quantum operation :math:`\mathcal{E}`, the Kraus operator representation is not unique. One can change the basis of Kraus operators using a unitary matrix :math:`U` as :math:`\tilde{E}_i = \sum_j U_{ij} E_j`.

We define a noise channel (or Kraus channel) 
as a quantum operation :math:`\mathcal{E}` described by a set of Kraus operators as given above. 
A common way of modeling noisy quantum computers is by considering each operation :math:`O` 
in the circuit as a noisy quantum operation :math:`\mathcal{E}_O`.

Evolution with noise
~~~~~~~~~~~~~~~~~~~~

There are two common ways to evolve the state of the system when acting with Kraus channels:

1. **Density matrix**: If we use a density matrix to describe the qubits, then a Kraus channel can simply be applied by directly performing the matrix multiplications as:

   .. math::

       \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger.

   The advantage of this approach is that the density matrix contains the full information of the system and we only need to run the circuit once. The disadvantage is that :math:`\rho` requires more memory to be stored (:math:`2^{2N}` as opposed to :math:`2^N` for a state vector), so we can simulate fewer qubits.

2. **Quantum trajectories**: This method consists in simulating the evolution of the state vector :math:`|\psi_\alpha \rangle` for a set of iterations :math:`\alpha = 1, \ldots, n`. In each iteration, a noise channel is applied by randomly selecting one of the Kraus operators according to some probabilities (see below) and applying that Kraus operator to the state vector. The advantage of this approach is that we need less memory since we work with a state vector. The disadvantage is that we need to run the circuit many times to collect samples (one sample per run).

Currently, MIMIQ only implements the quantum trajectories method.

The basis for quantum trajectories is that a Kraus channel can be rewritten as:

.. math::

    \mathcal{E}(\rho) = \sum_k p_k \tilde{E}_k \rho \tilde{E}_k^\dagger,

where :math:`p_k = \mathrm{Tr}(E_k \rho E_k^\dagger)` and :math:`\tilde{E}_k = E_k / \sqrt{p_k}`.

The parameters :math:`p_k` can be interpreted as probabilities since they fulfill :math:`0 \leq p_k \leq 1` and :math:`\sum_k p_k = 1`. In this way, the Kraus channel can be viewed as a linear combination of operations with different Kraus operators weighted by the probabilities :math:`p_k`.

Note that the probabilities :math:`p_k` generally depend on the state, so they need to be computed at runtime. The exception is mixed-unitary channels, for which the probabilities are fixed (state-independent).


Building noise channels
~~~~~~~~~~~~~~~~~~~~~~~
.. _building-noise-channels:

You can create noise channels using one of the many functions available. Most noise channels take one or more parameters, and custom channels require passing the Kraus matrices and/or probabilities. Here are some examples of how to build noise channels:

.. doctest::
    :hide: 

    >>> from mimiqcircuits import *

.. doctest::

    >>> p = 0.1    # probability
    >>> PauliX(p)
    PauliX(0.1)

.. doctest:: 

    >>> p, gamma = 0.1, 0.2    # parameters
    >>> GeneralizedAmplitudeDamping(p, gamma)
    GeneralizedAmplitudeDamping((0.1, 0.2))

.. doctest:: python

    >>> ps = [0.8, 0.1, 0.1]    # probabilities
    >>> paulis = ["II", "XX", "YY"]    # Pauli strings
    >>> PauliNoise(ps, paulis)
    PauliNoise((0.8, pauli"II"), (0.1, pauli"XX"), (0.1, pauli"YY"))

.. doctest:: python

    >>> from symengine import *
    >>> ps = [0.9, 0.1]    # probabilities
    >>> unitaries = [Matrix([[1, 0], [0, 1]]), Matrix([[1, 0], [0, -1]])]    # unitary matrices
    >>> MixedUnitary(ps, unitaries)
    MixedUnitary((0.9, "Custom([1 0; 0 1])"), (0.1, "Custom([1 0; 0 -1])"))

.. doctest:: python

    >>> kmatrices = [Matrix([[1, 0], [0, (0.9)**0.5]]), Matrix([[0, (0.1)**0.5], [0, 0]])]    # Kraus matrices
    >>> Kraus(kmatrices)
    Kraus(Operator([[1, 0], [0, 0.948683298050514]]), Operator([[0, 0.316227766016838], [0, 0]]))

Check the documentation for each noise channel to understand the conditions that each parameter needs to fulfill for the noise channel to be valid.

In MIMIQ, the most important distinction between noise channels is whether they are mixed-unitary or general Kraus channels. You can use :func:`~mimiqcircuits.krauschannel.ismixedunitary` to check if a channel is mixed unitary:

.. doctest:: python

    >>> PauliX(0.1).ismixedunitary()
    True

    >>> AmplitudeDamping(0.1).ismixedunitary()
    False

You can retrieve the Kraus matrices/operators used to define a given channel using :func:`~mimiqcircuits.krauschannel.krausmatrices` or :func:`~mimiqcircuits.krauschannel.krausoperators`. For example:

.. doctest:: python

    >>> ProjectiveNoise("Z").krausmatrices()
    [[1.0, 0]
    [0, 0]
    , [0, 0]
    [0, 1.0]
    ]

For mixed unitary channels, you can obtain the list of probabilities and the list of unitary gates/matrices separately using :func:`~mimiqcircuits.krauschannel.probabilities`, :func:`~mimiqcircuits.krauschannel.unitarymatrices`, or :func:`~mimiqcircuits.krauschannel.unitarygates`, respectively:

.. doctest:: python

    >>> PauliZ(0.1).unitarymatrices()
    [[1.0, 0]
    [0, 1.0]
    , [1.0, 0]
    [0, -1.0]
    ]
    >>> Depolarizing1(0.1).unitarygates()
    [I, X, Y, Z]
    >>> PauliNoise([0.1, 0.9], ["II", "ZZ"]).probabilities()
    [0.1, 0.9]

In MIMIQ, noise channels can be added at any point in a circuit to make any operation noisy. For noisy gates, one would normally add a noise channel after an ideal gate. To model measurement, preparation, and reset errors, noise channels can be added before and/or after the corresponding operation. More information can be found in the next section.

How to add noise
----------------
.. _how-to-add-noise:

Adding noise one by one
~~~~~~~~~~~~~~~~~~~~~~~

The simplest and most flexible way to add noise to a circuit is by using the :meth:`~mimiqcircuits.Circuit.push` method, similar to how gates are added. Here's an example of how to create a noisy 5-qubit GHZ circuit:

.. doctest:: python

    >>> c = Circuit()
    >>> c.push(PauliX(0.1), [1, 2, 3, 4, 5])    # preparation/reset error since all qubits start in 0
    6-qubit circuit with 5 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    └── PauliX(0.1) @ q[5]
    <BLANKLINE>

    >>> c.push(GateH(), 1)
    6-qubit circuit with 6 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    └── H @ q[1]
    <BLANKLINE>
    >>> c.push(AmplitudeDamping(0.1), 1)    # 1-qubit noise for GateH
    6-qubit circuit with 7 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    └── AmplitudeDamping(0.1) @ q[1]
    <BLANKLINE>

    >>> c.push(GateCX(), 1, [2, 3, 4, 5])
    6-qubit circuit with 11 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    └── CX @ q[1], q[5]
    <BLANKLINE>
    >>> c.push(Depolarizing2(0.1), 1, [2, 3, 4, 5])    # 2-qubit noise for GateCX
    6-qubit circuit with 15 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    ├── CX @ q[1], q[5]
    ├── Depolarizing(0.1) @ q[1,2]
    ├── Depolarizing(0.1) @ q[1,3]
    ├── Depolarizing(0.1) @ q[1,4]
    └── Depolarizing(0.1) @ q[1,5]
    <BLANKLINE>

    >>> c.push(PauliX(0.1), [1, 2, 3, 4, 5])    # measurement error. Note it's added before the measurement
    6-qubit circuit with 20 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    ├── CX @ q[1], q[5]
    ├── Depolarizing(0.1) @ q[1,2]
    ├── Depolarizing(0.1) @ q[1,3]
    ├── Depolarizing(0.1) @ q[1,4]
    ├── Depolarizing(0.1) @ q[1,5]
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    └── PauliX(0.1) @ q[5]
    <BLANKLINE>
    >>> c=c.push(Measure(), [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    >>> print(c)
    6-qubit circuit with 25 instructions:
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    ├── CX @ q[1], q[5]
    ├── Depolarizing(0.1) @ q[1,2]
    ├── Depolarizing(0.1) @ q[1,3]
    ├── Depolarizing(0.1) @ q[1,4]
    ├── Depolarizing(0.1) @ q[1,5]
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    ├── M @ q[4], c[4]
    └── M @ q[5], c[5]

Note how bit-flip error (:class:`~mimiqcircuits.PauliX`) is added at the beginning for state preparation/reset errors and right before measuring for measurement errors.

Adding noise to all gates of the same type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually, when adding noise to a circuit, the same type of noise is added to each instance of a given gate. Instead of adding noise channels one by one, you can use :meth:`~mimiqcircuits.Circuit.add_noise`. It takes several parameters:

.. doctest:: python

    Circuit().add_noise(gate, kraus, before=False, parallel=False)

This function adds the noise channel specified by `kraus` to every instance of the gate `g` in the circuit `circuit`. The optional parameter `before` determines whether to add the noise before or after the operation, and `parallel` determines whether to add the noise in parallel after/before a block of transversal gates.

Here's the same noisy GHZ circuit, using :meth:`~mimiqcircuits.Circuit.add_noise`:

.. doctest:: python
    >>> from mimiqcircuits import *
    >>> c = Circuit()
    >>> c.push(Reset(), [1, 2, 3, 4, 5])
    6-qubit circuit with 5 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    └── Reset @ q[5]
    <BLANKLINE>
    >>> c.push(GateH(), 1)
    6-qubit circuit with 6 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    ├── Reset @ q[5]
    └── H @ q[1]
    <BLANKLINE>
    >>> c.push(GateCX(), 1, [2, 3, 4, 5])
    6-qubit circuit with 10 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    ├── Reset @ q[5]
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    └── CX @ q[1], q[5]
    <BLANKLINE>
    >>> c.push(Measure(), [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    6-qubit circuit with 15 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    ├── Reset @ q[5]
    ├── H @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    ├── CX @ q[1], q[5]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    ├── M @ q[4], c[4]
    └── M @ q[5], c[5]
    <BLANKLINE>

    >>> cnoise = c.add_noise(Reset(), PauliX(0.1), parallel=True)
    >>> cnoise.add_noise(GateH(), AmplitudeDamping(0.1))
    6-qubit circuit with 21 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    ├── Reset @ q[5]
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[1], q[4]
    ├── CX @ q[1], q[5]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    ⋮   ⋮
    └── M @ q[5], c[5]
    <BLANKLINE>
    >>> cnoise.add_noise(GateCX(), Depolarizing2(0.1), parallel=True)
    6-qubit circuit with 25 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    ├── Reset @ q[5]
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── Depolarizing(0.1) @ q[1,2]
    ├── CX @ q[1], q[3]
    ├── Depolarizing(0.1) @ q[1,3]
    ├── CX @ q[1], q[4]
    ├── Depolarizing(0.1) @ q[1,4]
    ├── CX @ q[1], q[5]
    ⋮   ⋮
    └── M @ q[5], c[5]
    <BLANKLINE>
    >>> cnoise.add_noise(Measure(), PauliX(0.1), before=True, parallel=True)
    6-qubit circuit with 30 instructions:
    ├── Reset @ q[1]
    ├── Reset @ q[2]
    ├── Reset @ q[3]
    ├── Reset @ q[4]
    ├── Reset @ q[5]
    ├── PauliX(0.1) @ q[1]
    ├── PauliX(0.1) @ q[2]
    ├── PauliX(0.1) @ q[3]
    ├── PauliX(0.1) @ q[4]
    ├── PauliX(0.1) @ q[5]
    ├── H @ q[1]
    ├── AmplitudeDamping(0.1) @ q[1]
    ├── CX @ q[1], q[2]
    ├── Depolarizing(0.1) @ q[1,2]
    ├── CX @ q[1], q[3]
    ├── Depolarizing(0.1) @ q[1,3]
    ├── CX @ q[1], q[4]
    ├── Depolarizing(0.1) @ q[1,4]
    ├── CX @ q[1], q[5]
    ⋮   ⋮
    └── M @ q[5], c[5]
    <BLANKLINE>

Running a noisy circuit
-----------------------
.. _running-a-noisy-circuit:

Circuits with noise can be run using the :func:`~mimiqcircuits.MimiqConnection.execute` function, 
just as with circuits without noise. Currently, noisy simulations are run using the quantum trajectories 
method. In this case, when running a circuit with noise for `n` samples, the circuit will internally be 
run once for each sample, with a different set of random Kraus operators selected based on the 
corresponding probabilities.

When the noise channel is a mixed-unitary channel, the unitary operators to be applied can be selected 
before applying the operations on the state. Use the :func:`~mimiqcircuits.Circuit.sample_mixedunitaries` 
function to generate samples of a circuit with mixed-unitary noise:

.. doctest:: python

    >>> from mimiqcircuits import *
    >>> import random

    >>> rng = random.Random(42)

    >>> c = Circuit()
    >>> c.push(Depolarizing1(0.5), [1, 2, 3, 4, 5])
    6-qubit circuit with 5 instructions:
    ├── Depolarizing(0.5) @ q[1]
    ├── Depolarizing(0.5) @ q[2]
    ├── Depolarizing(0.5) @ q[3]
    ├── Depolarizing(0.5) @ q[4]
    └── Depolarizing(0.5) @ q[5]
    <BLANKLINE>

    # Produce a circuit with either I, X, Y, or Z in place of each depolarizing channel
    >>> c.sample_mixedunitaries(rng=rng, ids=True)
    6-qubit circuit with 5 instructions:
    ├── X @ q[1]
    ├── I @ q[2]
    ├── I @ q[3]
    ├── I @ q[4]
    └── Y @ q[5]
    <BLANKLINE>

This function is called internally when executing a circuit, but it can also be used separately.
