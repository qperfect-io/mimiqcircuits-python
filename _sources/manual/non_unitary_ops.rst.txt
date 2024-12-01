Non-unitary Operations
========================

Contrary to :doc:`unitary gates <unitary_gates>`, non-unitary operations based on measurements make the quantum state collapse. Find in the following sections all the non-unitary operations supported by MIMIQ.

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

.. note:: 

    As a rule of thumb all non-unitary operations can be added to the circuit using the function :meth:`~mimiqcircuits.Circuit.push` by giving the index of the targets in the following order:raw:`:` quantum register index -> classical register index. 

.. note:: 

    Noise can also be interpreted as a non-unitary operations but will not be treated here, check the :doc:`noise <noise>` documentation page to learn more about it.

.. note:: 

    Once a non-unitary operation is added to your circuit the speed of execution might be reduced. This is because in this case the circuit needs to be re-run for every sample since the final quantum state might be different each time. This is always true except for :class:`~mimiqcircuits.Measure` operations placed at the very end of a circuit.
    To learn more about this head to the :ref:`simulation <understanding-sampling>` page.

.. note:: 

    Some features of unitary gates are not available for non-unitary operations, for instance, :func:`~mimiqcircuits.matrix`, :func:`~mimiqcircuits.inverse`, :func:`~mimiqcircuits.power`, :func:`~mimiqcircuits.control`, :func:`~mimiqcircuits.parallel`.

.. _Measure:

Measure
------------------------------------------------------------------

Mathematical definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measurements are defined by a basis of projection operators :math:`P_k`, one for each different possible outcome :math:`k`. The probability :math:`p_k` of measuring outcome :math:`k` is given by the expectation value of :math:`P_k`, that is

.. math:: 

    p_k = \bra{\psi} P_k \ket{\psi}.

If the outcome :math:`k` is observed, the system is left in the state

.. math:: 

    \frac{P_k\ket{\psi}}{\sqrt{p_k}}.

It is common to measure in the Z basis (:math:`P_0=\ket{0}\bra{0}` and :math:`P_1=\ket{1}\bra{1}`), but measurements in other bases are possible too.

How to use measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available measurement operations: :class:`~mimiqcircuits.Measure`, :class:`~mimiqcircuits.MeasureX`, :class:`~mimiqcircuits.MeasureY`, :class:`~mimiqcircuits.MeasureZ`, :class:`~mimiqcircuits.MeasureXX`, :class:`~mimiqcircuits.MeasureYY`, :class:`~mimiqcircuits.MeasureZZ`.

With MIMIQ you can measure the qubits at any point in the circuit (not only at the end of the circuit) using one of the measurement operations (:class:`~mimiqcircuits.Measure`...). You can add it to the circuit like gates using :meth:`~mimiqcircuits.Circuit.push`, but you will need to precise both the index for the quantum register (qubit to measure) and classical register (where to store the result):

.. doctest:: non_unitary 

    >>> circuit = Circuit() 
    >>> circuit.push(Measure(), 0, 0)
    1-qubit circuit with 1 instructions:
    └── M @ q[0], c[0]
    <BLANKLINE>


This will add a :func:`~mimiqcircuits.Measure` on the first qubit of the quantum register to the `circuit` and write the result on the first bit of the classical register. Recall that the targets are always ordered as quantum register -> classical register -> z register. To learn more about registers head to the :ref:`registers` section.  

You can also use iterators to Measure multiple qubits at once, as for gates:

.. doctest:: non_unitary 

    >>> circuit.push(Measure(), range(0, 10), range(0,10))
    10-qubit circuit with 11 instructions:
    ├── M @ q[0], c[0]
    ├── M @ q[0], c[0]
    ├── M @ q[1], c[1]
    ├── M @ q[2], c[2]
    ├── M @ q[3], c[3]
    ├── M @ q[4], c[4]
    ├── M @ q[5], c[5]
    ├── M @ q[6], c[6]
    ├── M @ q[7], c[7]
    ├── M @ q[8], c[8]
    └── M @ q[9], c[9]
    <BLANKLINE>


.. note:: 

    In the absence of any non-unitary operations in the circuit, MIMIQ will sample (and, therefore, measure) all the qubits at the end of the circuit by default, see :doc:`simulation <simulation>` page.

.. _Reset: 

Reset
------------------------------------------------------------------

Available reset operations: :class:`~mimiqcircuits.Reset`, :class:`~mimiqcircuits.ResetX`, :class:`~mimiqcircuits.ResetY`, :class:`~mimiqcircuits.ResetZ`.

A reset operation consists in measuring the qubits in some basis and then applying an operation conditioned on the measurement outcome to leave the qubits in some pre-defined state. For example, :func:`~mimiqcircuits.Reset` leaves all qubits in :math:`\ket{0}` (by measuring in :math:`Z` and flipping the state if the outcome is `1`).

Here is an example of how to add a reset operation to a circuit:

.. doctest:: non_unitary 

    >>> circuit = Circuit()  
    >>> circuit.push(Reset(), 0) 
    1-qubit circuit with 1 instructions:
    └── Reset @ q[0]
    <BLANKLINE>


Importantly, even though a reset operation technically measures the qubits, the information is not stored in the classical register, so we only need to specify the qubit register. If you want to store the result, see the :ref:`Measure Reset` section.

Note that a reset operation can be technically seen as noise and is described by the same mathematical machinery, see :doc:`noise <noise>` page. For this reason, some of the functionality provided by MIMIQ for noise is also available for resets. Here is one example:

.. doctest:: non_unitary 

    >>> Reset().krausoperators()
    [P₀(1.0), SigmaMinus(1.0)]

.. _Measure Reset:

Measure-Reset
------------------------------------------------------------------

Available measure-reset operations: :class:`~mimiqcircuits.MeasureReset`, :class:`~mimiqcircuits.MeasureResetX`, :class:`~mimiqcircuits.MeasureResetY`, :class:`~mimiqcircuits.MeasureResetZ`.

A measure-reset operation is the same as a reset operation except that we store the result of the measurement, see :ref:`Measure` and :ref:`Reset` sections. Because of that, we need to specify both quantum and classical registers when adding it to a circuit:

.. doctest:: non_unitary 

    >>> circuit = Circuit()  
    >>> circuit.push(MeasureReset(), 0, 0)
    1-qubit circuit with 1 instructions:
    └── MR @ q[0], c[0]
    <BLANKLINE>



Conditional logic
------------------------------------------------------------------

If statements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An *if* statement consists in applying an operation conditional on the value of some classical register. In that sense, it resembles a classical *if* statement.

In MIMIQ you can implement it using :class:`~mimiqcircuits.IfStatement`, which requires two arguments: an operation to apply and a :class:`~mimiqcircuits.BitString` as the condition (see :ref:`Bitstrings <Bitstring>` page for more information):

.. doctest:: non_unitary 

    >>> IfStatement(GateX(), BitString("111"))
    IF (c==111) X


.. note:: 

    At the moment, MIMIQ only allows to pass unitary gates as arguments to an if statement (which makes if statements unitary for now).

To add an :class:`~mimiqcircuits.IfStatement` to a circuit use the :meth:`~mimiqcircuits.Circuit.push` function. The first (quantum) indices will determine the qubits to apply the gate to, whereas the last (classical) indices will be used to compare against the condition given. For example:

.. doctest:: non_unitary 

    >>> circuit  = Circuit()

    >>> # Apply a GateX on qubit 1 if the qubits 2 and 4 are in the state 1 and qubit 3 in the state 0. 
    >>> circuit.push(IfStatement(GateX(), BitString("101")), 0, 1, 2, 3)
    1-qubit circuit with 1 instructions:
    └── IF (c==101) X @ q[0], c[1,2,3]
    <BLANKLINE>


Here, an `X` gate will be applied to qubit 1, if classical registers 2 and 4 are `1`, and classical register 3 is `0`. Of course, if the gate targets more than 1 qubit, then all qubit indices will be specified before the classical registers, as usual (see :doc:`circuit <circuits>` page).

.. _Operators:

Operators
------------------------------------------------------------------

Mathematical definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operators refer to any linear operation on a state. An operator does not need to be unitary, as is the case of a gate. This means that any :math:`2^N \times 2^N` matrix can in principle represent an operator on :math:`N` qubits.

.. note:: 

    Do not confuse *operator* with *operation*. In MIMIQ, the word operation is used as the supertype for all transformations of a quantum state (gates, measurements, statistical operations...), whereas an operator is a sort of generalized gate, a linear tranformation.


Operators available in MIMIQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom operators: :class:`~mimiqcircuits.Operator`

Special operators: :class:`~mimiqcircuits.DiagonalOp`, :class:`~mimiqcircuits.SigmaPlus`, :class:`~mimiqcircuits.SigmaMinus`, :class:`~mimiqcircuits.Projector0`, :class:`~mimiqcircuits.ProjectorZ0`, :class:`~mimiqcircuits.Projector1`, :class:`~mimiqcircuits.ProjectorZ1`, :class:`~mimiqcircuits.ProjectorX0`, :class:`~mimiqcircuits.ProjectorX1`, :class:`~mimiqcircuits.ProjectorY0`, :class:`~mimiqcircuits.ProjectorY1`, :class:`~mimiqcircuits.Projector00`, :class:`~mimiqcircuits.Projector01`, :class:`~mimiqcircuits.Projector10`, :class:`~mimiqcircuits.Projector11`

Methods available: :meth:`~mimiqcircuits.Gate;matrix`.


How to use operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doctest:: operators
    :hide:

    >>> from mimiqcircuits import *
    >>> import math

Operators cannot be applied to a state directly (it cannot be added to a circuit using :meth:`~mimiqcircuits.Circuit.push`), because that would correspond to an unphysical transformation. However, they can be used within other operations such as :class:`~mimiqcircuits.ExpectationValue` or to create custom noise models with :class:`~mimiqcircuits.Kraus`, see :doc:`noise <noise>` and :doc:`statistical operations <statistical_ops>` pages.

Operators can be used to compute expectation values as follows (see also :class:`~mimiqcircuits.ExpectationValue`):

.. doctest:: operators 

    >>> op = SigmaPlus()
    >>> ev = ExpectationValue(op)


.. doctest:: operators 

    >>> circuit = Circuit()
    >>> circuit.push(ev, 0, 0)
    1-qubit circuit with 1 instructions:
    └── ⟨SigmaPlus(1)⟩ @ q[0], z[0]
    <BLANKLINE>


Similarly, operators can also be used to define non-mixed unitary Kraus channels (see also :class:`~mimiqcircuits.Kraus`).
For example, we can define the amplitude damping channel as follows:

.. doctest:: operators 

    >>> gamma = 0.1
    >>> k1 = DiagonalOp(1, math.sqrt(1-gamma))    # Kraus operator 1
    >>> k2 = SigmaMinus(math.sqrt(gamma))    # Kraus operator 2
    >>> kraus = Kraus([k1,k2])


This is equivalent to

.. doctest:: operators 

    >>> gamma = 0.1
    >>> ampdamp = AmplitudeDamping(gamma)
    >>> ampdamp.krausoperators()
    [D(1, 0.9486832980505138), SigmaMinus(0.31622776601683794)]


.. note:: 

    Whenever possible, using specialized operators, such as `DiagonalOp` and `SigmaMinus`, as opposed to custom operators, such as `Operator`, is generally better for performance.


