Unitary Gates
===============================

Unitary gates are fundamental components of quantum circuits. Here we explain how to work with unitary gates in MIMIQ.



Contents
========
.. contents::
   :local:
   :depth: 2
   :backlinks: entry



.. doctest:: quick_start
    :hide:

    >>> from mimiqcircuits import *
    >>> import numpy as np
    >>> import os
    >>> import math

    >>> conn = MimiqConnection(url="https://mimiqfast.qperfect.io/api")
    >>> conn.connect(os.getenv("MIMIQUSER"), os.getenv("MIMIQPASS"))
    Connection:
    ├── url: https://mimiqfast.qperfect.io/api
    ├── Computing time: 599/10000 minutes
    ├── Executions: 605/10000
    ├── Max time limit per request: 180 minutes
    └── status: open
    <BLANKLINE>

Mathematical background
--------------------------------------------------------------------

State vector and probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In quantum mechanics, every transformation applied to a quantum state must be unitary (in a closed system). To understand why, we can expand the quantum state as  

.. math::
    \begin{aligned}
    \ket{\psi} = \sum_{i=1}^{k} c_{i} \ket{\psi_{i}}
    \end{aligned}

where :math:`\ket{\psi_i}` are orthonormal basis states.
For this state, the following condition must hold true:  

.. math::

    \begin{aligned}
    \sum_{i=1}^{k} |c_i|² = 1 
    \end{aligned}

Since :math:`|c_i|^2` corresponds to the probability of measuring state :math:`\ket{\psi_i}`, this condition simply says that the probabilities must add up to one. Unitary gates preserve this normalization condition, see below.


Unitary transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative way to compute the probability is through the inner product. Given two states in Hilbert space, :math:`\ket{\alpha}` and :math:`\ket{\psi}`, the squared inner product :math:`|\braket{\alpha|\psi}|^2` reflects the probability of measuring the system in state :math:`\ket{\alpha}`. 
Thus, the normalization condition can be written as :math:`|\braket{\psi|\psi}|^2 = 1`. In other words, the length of the state vector in complex space must be one.

When evolving the state :math:`\ket{\psi}` using an operator U, the normalization condition becomes (omitting the square):  

.. math::

    \begin{aligned}
    \bra{\psi} U^\dagger U \ket{\psi} = 1
    \end{aligned}

To fulfill this, the operator U must satisfy the condition:  

.. math::

    \begin{aligned}
    U^\dagger U = I
    \end{aligned}

An operator that fulfills this requirement is called a unitary operator and its matrix representation is unitary too.


Unitary gates in MIMIQ.
--------------------------------------------------------------------

MIMIQ offers a large number of gates to build quantum circuits. For an overview, type the following line in your Python session:

.. code::

    help(GATES)


Similarly, to get more information about a specific gate, you can type the following command in your Python session using the gate of your choice:

.. code::

    help(GateID)


There are different categories of gates depending on the number of targets, parameters etc. We discuss how to implement them in the following.

Single-qubit gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List of single-qubit gates: :class:`~mimiqcircuits.GateID`, :class:`~mimiqcircuits.GateX`, :class:`~mimiqcircuits.GateY`, :class:`~mimiqcircuits.GateZ`, :class:`~mimiqcircuits.GateH`, :class:`~mimiqcircuits.GateS`, :class:`~mimiqcircuits.GateSDG`, :class:`~mimiqcircuits.GateT`, :class:`~mimiqcircuits.GateTDG`, :class:`~mimiqcircuits.GateSX`, :class:`~mimiqcircuits.GateSXDG`. :class:`~mimiqcircuits.GateSY`, :class:`~mimiqcircuits.GateSYDG`.

| For single-qubit gates you don't need to give any argument to the gate constructor (ex: `GateX()`).
| You only need to give the index of the target qubit when adding it to your circuit with the :meth:`~mimiqcircuits.Circuit.push` function.

.. doctest:: unitary

    >>> circuit = Circuit()
    >>> circuit.push(GateX(), 0)
    1-qubit circuit with 1 instructions:
    └── X @ q[0]
    <BLANKLINE>

Single-qubit parametric gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List of single-qubit parametric gates:  :class:`~mimiqcircuits.GateU`, :class:`~mimiqcircuits.GateP`, :class:`~mimiqcircuits.GateRX`, :class:`~mimiqcircuits.GateRY`, :class:`~mimiqcircuits.GateRZ`, :class:`~mimiqcircuits.GateR`, :class:`~mimiqcircuits.GateU1`, :class:`~mimiqcircuits.GateU2`, :class:`~mimiqcircuits.GateU3`, :meth:`~mimiqcircuits.Delay`.

| For single-qubit parametric gates you need to give the expected number of parameters to the gate constructor (ex: ```GateU(0.5, 0.5, 0.5)``` or ```GateU1(0.5)```), if you are unsure of the expected number of parameters use the :code:`help()` function in your Python interactive session and give it the oject you are interested in (ex: :code:`help(GateU)`).
| As for any single qubit gates you can add it to your circuit by using the :meth:`~mimiqcircuits.Circuit.push` function and give the index of the target qubit.

.. doctest:: unitary


    >>> circuit = Circuit()
    >>> circuit.push(GateRX(math.pi/2), 0)
    1-qubit circuit with 1 instructions:
    └── RX(1.5707963267948966) @ q[0]
    <BLANKLINE>


Two qubit gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List of two qubits gates: :class:`~mimiqcircuits.GateCX`, :class:`~mimiqcircuits.GateCY`, :class:`~mimiqcircuits.GateCZ`, :class:`~mimiqcircuits.GateCH`, :class:`~mimiqcircuits.GateSWAP`, :class:`~mimiqcircuits.GateISWAP`, :class:`~mimiqcircuits.GateCS`, :class:`~mimiqcircuits.GateCSDG`, :class:`~mimiqcircuits.GateCSX`, :class:`~mimiqcircuits.GateCSXDG`, :class:`~mimiqcircuits.GateECR`, :class:`~mimiqcircuits.GateDCX`.

| Two-qubit gates can be instantiated without any arguments just like single-qubit gates (ex: `GateCX()`).
| You will need to give the index of both qubits to the :meth:`~mimiqcircuits.Circuit.push` function to add it to the circuit.
| To understand the ordering of the targets check the documentation of each particular gate. For controlled gates we use the convention that the first register corresponds to the control and the second to the target.

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> circuit.push(GateCH(), 0, 1)
    2-qubit circuit with 1 instructions:
    └── CH @ q[0], q[1]
    <BLANKLINE>


Two-qubit parametric gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
List of two qubits parametric gates : :class:`~mimiqcircuits.GateCP`, :class:`~mimiqcircuits.GateCU`, :class:`~mimiqcircuits.GateCRX`, :class:`~mimiqcircuits.GateCRY`, :class:`~mimiqcircuits.GateCRZ`, :class:`~mimiqcircuits.GateRXX`, :class:`~mimiqcircuits.GateRYY`, :class:`~mimiqcircuits.GateRZZ`,
:class:`~mimiqcircuits.GateRZX`, :class:`~mimiqcircuits.GateXXplusYY`, :class:`~mimiqcircuits.GateXXminusYY`.

| Two-qubit parametric gates are instantiated exactly like single-qubit parametric gates. You will need to give the expected number of parameters of the gate to its constructor (ex: :code:`GateCU(math.pi, math.pi, math.pi)`).
| You can then add it to the circuit just like a two-qubit gate by giving the index of the target qubits to the :meth:`~mimiqcircuits.Circuit.push` function. Again, check each gate's documentation to understand the qubit ordering; for controlled gates the first qubit corresponds to the control qubit, the second to the target.

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> circuit.push(GateRXX(math.pi/2), 0, 1)
    2-qubit circuit with 1 instructions:
    └── RXX(1.5707963267948966) @ q[0,1]
    <BLANKLINE>

Multi-qubit gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List of multi-qubit gates: :class:`~mimiqcircuits.GateCCX`, :class:`~mimiqcircuits.GateC3X`, :class:`~mimiqcircuits.GateCCP`, :class:`~mimiqcircuits.GateCSWAP`.

For the multi-qubit controlled gates you will need to give the index of each qubit to the :meth:`~mimiqcircuits.Circuit.push` function. As usual, first the control qubits, then the targets; check the specific documentation of each gate.

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> circuit.push(GateC3X(), 0, 1, 2, 3)
    4-qubit circuit with 1 instructions:
    └── C₃X @ q[0,1,2], q[3]
    <BLANKLINE>


Generalized gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some common gate combinations are available as generalized gates: :class:`~mimiqcircuits.PauliString`, :class:`~mimiqcircuits.QFT`, :class:`~mimiqcircuits.PhaseGradient`, :class:`~mimiqcircuits.Diffusion`, :class:`~mimiqcircuits.PolynomialOracle`.

Generalized gates can be applied to a variable number of qubits.
It is highly recommended to check their docstrings to understand their usage :code:`help(QFT)`.

Here is an example of use:

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> circuit.push(PhaseGradient(10), *range(0, 10))
    10-qubit circuit with 1 instructions:
    └── PhaseGradient @ q[0,1,2,3,4,5,6,7,8,9]
    <BLANKLINE>

These gates target a variable number of gates, so you have to specify in the constructor how many target qubits will be used, and give to the :meth:`~mimiqcircuits.Circuit.push` function one index per target qubit.

More about generalized gates on :doc:`special operations <special_ops>`.

Custom Gates
--------------------------------------------------------------------

If you need to use a specific unitary gate that is not provided by MIMIQ, you can use :class:`~mimiqcircuits.GateCustom`` to create your own unitary gate.

.. note::

    Only **one** qubit or **two** qubits gates can be created using MIMIQ's :class:`~mimiqcircuits.GateCustom`.

.. note::

    Avoid using :class:`~mimiqcircuits.GateCustom` if you can define the same gate using a pre-defined gate from MIMIQ's library, as it could impact negatively peformance.

To create a custom unitary gate you first have to define the matrix of your gate in Python:

.. doctest:: unitary

    # define the matrix for a 2 qubits gate
    >>> custom_matrix = np.array([[np.exp(1j*math.pi/3), 0, 0, 0], [0, np.exp(1j*math.pi/5), 0, 0 ], [0, 0, np.exp(1j*math.pi/7), 0], [0, 0, 0, np.exp(1j*math.pi/11)]])


Then you can create your unitary gate and use it like any other gate using :meth:`~mimiqcircuits.Circuit.push`

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> custom_gate = GateCustom(custom_matrix)
    >>> circuit.push(custom_gate, 0, 1)
    2-qubit circuit with 1 instructions:
    └── Custom([0.5 + 0.866025403784439*I, 0.0 + 0.0*I, 0.0 + 0.0*I, 0.0 + 0.0*I]...[0.0 + 0.0*I, 0.0 + 0.0*I, 0.0 + 0.0*I, 0.959492973614497 + 0.28173255684143*I]) @ q[0,1]
    <BLANKLINE>

Composition: Control, Power, Inverse, Parallel
--------------------------------------------------------------------

Gates in MIMIQ can be combined to create more complex gates using :class:`~mimiqcircuits.Control`, :class:`~mimiqcircuits.Circuit.Power`, :class:`~mimiqcircuits.Circuit.Inverse`, :class:`~mimiqcircuits.Circuit.Parallel`.

Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A controlled version of every gate can be built using the :func:`~mimiqcircuits.control` function.  
For example, `CX` can be built with the following instruction:

.. doctest:: unitary

    >>> CX = control(1, GateX())

The first argument indicates the number of control qubits and is completely up to the user.
For example a CCCCCX can be built with the following instruction:

.. doctest:: unitary

    >>> CCCCCX = control(5, GateX())

.. admonition:: Details

    A wrapper for :class:`~mimiqcircuits.GateCX` is already provided by MIMIQ. Whenever possible, it is recommended to use the gates already provided by the framework instead of creating your own composite gate to prevent performances loss.

Be careful when adding the new control gate to your circuit. When using the :meth:`~mimiqcircuits.Circuit.push` function, the first expected indices should be the control qubits specified in :class:`~mimiqcircuits.Control` and the last indices the target qubits of the gate, for instance:

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> circuit.push(CCCCCX, 0, 1, 2, 3, 4, 5)
    6-qubit circuit with 1 instructions:
    └── C₅X @ q[0,1,2,3,4], q[5]
    <BLANKLINE>


Power
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To raise the power of a gate you can use the :func:`~mimiqcircuits.power` function.
For example, :math:`\sqrt{\mathrm{GateS}} = \mathrm{GateT}`, therefore, the following instruction can be used to generate the GateS:

.. doctest:: unitary

    >>> power(GateS(), 1/2)
    T

The power method will attempt to realize simplifications whenever it can, for example asking for the square of :class:`~mimiqcircuits.GateX` will directly give you :class:`~mimiqcircuits.GateID`.

Inverse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the inverse of an operator you can use the :func:`~mimiqcircuits.inverse` method.
Remember that the inverse of a unitary matrix is the same as the adjoint (conjugate transpose), so this is a simple way to get the adjoint of a gate.
For example here is how to get the inverse of a :class:`~mimiqcircuits.GateH`

.. doctest:: unitary

    >>> inv_H = inverse(GateH())


Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a composite gate applying a specific gate to multiple qubits at once you can use the :func:`~mimiqcircuits.parallel` method.

.. doctest:: unitary

    >>> circuit = Circuit() 
    >>> X_gate_4 = parallel(4, GateX())
    >>> circuit.push(X_gate_4, 0, 1, 2, 3)
    4-qubit circuit with 1 instructions:
    └── ⨷ ⁴ X @ q[0], q[1], q[2], q[3]
    <BLANKLINE>
    >>> circuit.draw()                                                                         
            ┌─┐                                                                     
     q[0]: ╶┤X├────────────────────────────────────────────────────────────────────╴
            ┌─┐                                                                     
     q[1]: ╶┤X├────────────────────────────────────────────────────────────────────╴
            ┌─┐                                                                     
     q[2]: ╶┤X├────────────────────────────────────────────────────────────────────╴
            ┌─┐                                                                     
     q[3]: ╶┤X├────────────────────────────────────────────────────────────────────╴
            └─┘                                                                     
                                                                                
                                                                                
                                                                                
                                                                                

To check the number of repetition of your custom parallel gate you can use the :meth:`~mimiqcircuits.Circuit.num_repeats` method:

.. doctest:: unitary

    >>> X_gate_4.num_repeats
    4


Be careful when using a multi-qubit gate with :func:`~mimiqcircuits.parallel` as the index of the targeted qubits in :meth:`~mimiqcircuits.Circuit.push` can become confusing.
for example see below the parallel applicatoin of a `CX` gate:

.. doctest:: unitary

    >>> circuit = Circuit()
    >>> double_CX = Parallel(2, GateCX())
    >>> circuit.push(double_CX, 0, 1, 2, 3)
    4-qubit circuit with 1 instructions:
    └── ⨷ ² CX @ q[0], q[1], q[2], q[3]
    <BLANKLINE>
    >>> circuit.draw()                                                                         
            ┌────┐                                                                  
     q[0]: ╶┤0   ├─────────────────────────────────────────────────────────────────╴
            │  CX│                                                                  
     q[1]: ╶┤1   ├─────────────────────────────────────────────────────────────────╴
            ┌────┐                                                                  
     q[2]: ╶┤0   ├─────────────────────────────────────────────────────────────────╴
            │  CX│                                                                  
     q[3]: ╶┤1   ├─────────────────────────────────────────────────────────────────╴
            └────┘                                                                  
                                                                                
                                                                                
                                                                                
                                                                                
                                                                  
                                                                       
Here the index 0 and 1 correspond to the control and target of the first `CX` gate and 2 and 3 correspond to the second `CX` gate.



Extract information of unitary gates
--------------------------------------------------------------------

MIMIQ priovides a few methods to extract information about the unitary gates.

Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the matrix of a unitary gate you can use the :meth:`~mimiqcircuits.Gate.matrix`:

.. doctest:: unitary

    >>> GateCX().matrix() 
    [1.0, 0, 0, 0]
    [0, 1.0, 0, 0]
    [0, 0, 0, 1.0]
    [0, 0, 1.0, 0]
    <BLANKLINE>



Number of targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another way to know how many qubits, bits or z-variables are targeted by one unitary gate you can use :meth:`~mimiqcircuits.Circuit.num_qubits`, :meth:`~mimiqcircuits.Circuit.num_bits` and :meth:`~mimiqcircuits.Circuit.num_zvars`, respectively.

.. doctest:: unitary

    >>> GateCX().num_qubits, GateCX().num_bits, GateCX().num_zvars
    (2, 0, 0)


.. doctest:: unitary

    >>> Measure().num_qubits, Measure().num_bits, Measure().num_zvars
    (1, 1, 0)


.. doctest:: unitary

    >>> Amplitude("01").num_qubits,Amplitude("01").num_bits, Amplitude("01").num_zvars
    (0, 0, 1)


See :doc:`non-unitary operations <non_unitary_ops>` and :doc:`statistical operations <statistical_ops>` pages for more information on :class:`~mimiqcircuits.Measure` and :class:`~mimiqcircuits.Amplitude`.

