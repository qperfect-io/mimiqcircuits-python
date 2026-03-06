Circuits
========

On this page you can find all the information needed to build a circuit using MIMIQ. Every useful function will be presented below, accompanied by an explanation of their purpose and examples of use.

.. doctest:: quick_start
    :hide:

    >>> from mimiqcircuits import *
    >>> import numpy as np
    >>> import os
    >>> import math

    >>> conn = MimiqConnection(url="https://mimiq.qperfect.io")
    >>> conn.connect(os.getenv("MIMIQUSER"), os.getenv("MIMIQPASS"))
    MimiqConnection:
    ├── url: https://mimiq.qperfect.io
    ├── Max time limit per request: 360 minutes
    ├── Default time limit is equal to max time limit: 360 minutes
    └── status: open


What is a circuit and what are instructions 
----------------------------------------------------

A quantum circuit, similar to a classical circuit, represents a sequence of quantum gates applied to qubits, which are the carriers of quantum information. Quantum circuits are essential for designing quantum algorithms. The complexity of a quantum circuit is typically measured by two key metrics: width and depth. Width refers to the number of qubits in the circuit, while depth indicates the maximum number of sequential gates applied to any single qubit.

Here is a representation of a simple GHZ circuit on 4 qubits:

.. doctest:: python

    from mimiqcircuits import *
    >>> ghz = Circuit()
    >>> ghz.push(GateH(), 0)    
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>
    >>> ghz.push(GateCX(), 0, range(1, 4))
    4-qubit circuit with 4 instructions:
    ├── H @ q[0]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    └── CX @ q[0], q[3]
    <BLANKLINE>
    >>>
    >>> ghz.draw()
                    ┌─┐                                                                     
             q[0]: ╶┤H├─●──●──●────────────────────────────────────────────────────────────╴
                    └─┘┌┴┐ │  │                                                             
             q[1]: ╶───┤X├─┼──┼────────────────────────────────────────────────────────────╴
                       └─┘┌┴┐ │                                                             
             q[2]: ╶──────┤X├─┼────────────────────────────────────────────────────────────╴
                          └─┘┌┴┐                                                            
             q[3]: ╶─────────┤X├───────────────────────────────────────────────────────────╴
                             └─┘                                                            
                                                                                
                                                                            

In this representation, each qubit is depicted by a horizontal line labeled q[x], where x is the qubit’s index. The circuit is read from left to right, with each 'block' or symbol along a line representing an operation applied to that specific qubit.

Circuits & Instructions in MIMIQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIMIQ implements a circuit using the :class:`~mimiqcircuits.Circuit`  structure, in essence this structure is a wrapper for a vector of :class:`~mimiqcircuits.Instruction`  to be applied on the qubits in the order of the vector. Since it is a vector a circuit can be manipulated as such, for example you can use for loops to iterate over the different instructions of the circuit, do vector comprehension or access all common vector attributes such as the length.

An :class:`~mimiqcircuits.Instruction`  is composed of the quantum operation to be applied to the qubits, and the targets on which to apply it. There are many types of quantum operations, as discussed in the :doc:`unitary gates <unitary_gates>`, :doc:`non-unitary operations <non_unitary_ops>` and other pages of the manual. The targets can be qubits, as well as boolean or complex number vectors where classical information can be stored.
You will generally not need to interact with the :class:`~mimiqcircuits.Instruction`  class directly (for exceptions, see :doc:`special operations <special_ops>`), but it is useful to understand how MIMIQ works.

See the following sections to learn how to add operations to your circuit.

.. _registers:

Registers: quantum/classical/Z-register
----------------------------------------------------

Before explaining how to build a circuit it is important to make a distinction between the different target registers your operations will be applied to. 

The circuits in MIMIQ are composed of three registers that can be used by the instructions:
* The Quantum Register: Used to store the **qubits** state. Most of the operators in MIMIQ will interact with the quatum register. When printing or drawing a circuit (with the function :meth:`~mimiqcircuits.Circuit.draw` ) the quantum registers will be denoted as `q[x]` with x being the index of the qubit in the quantum register. 
* The classical register: Used to store the **bits** state. Some gates will need to interact with classical bits (ex: :class:`~mimiqcircuits.Measure` ) and the state of the classical bits is stored in the classical register, which is a vector of booleans. When printing or drawing a circuit the classical register will be denoted by the letter `c`.
* The Z-register: Used to store the result of some specific operations when the expected result is a **complex number** (ex: :class:`~mimiqcircuits.ExpectationValue` ). The Z-register is basically a vector of complex numbers. When printing or drawing a circuit the Z-Register will be denoted by the letter `z`.

For the three registers operators can be applied on an arbitrary index starting from 0 (as does Python in general contrary to Julia). When possible you should always use the minimal index available as going for an arbitrary high index ``N`` will imply that ``N`` qubits will be simulated and might result in a loss of performance and will also make the circuit drawing more complex to understand. 

Here is a circuit interacting with all registers:

.. doctest:: python

    from mimiqcircuits import *
    >>> # create empty circuit
    >>> circuit = Circuit()

    >>> # add X to the first qubit of the Quantum register
    >>> circuit.push(GateX(), 0)
    1-qubit circuit with 1 instructions:
    └── X @ q[0]
    <BLANKLINE>

    >>> # compute Expectation value of qubit 1 and store complex number on the first Z-Register
    >>> ev = ExpectationValue(GateZ())
    >>> circuit.push(ev, 0, 0)
    1-qubit, 1-zvar circuit with 2 instructions:
    ├── X @ q[0]
    └── ⟨Z⟩ @ q[0], z[0]
    <BLANKLINE>

    >>> # Measure the qubit state and store bit into the first classical register
    >>> circuit.push(Measure(), 0, 0)
    1-qubit, 1-bit, 1-zvar circuit with 3 instructions:
    ├── X @ q[0]
    ├── ⟨Z⟩ @ q[0], z[0]
    └── M @ q[0], c[0]
    <BLANKLINE>

    >>> # draw the circuit
    >>> circuit.draw()
                    ┌─┐┌─────────┐┌──────┐                                                  
             q[0]: ╶┤X├┤   ⟨Z⟩   ├┤  M   ├─────────────────────────────────────────────────╴
                    └─┘└────╥────┘└───╥──┘                                                  
                            ║         ║                                                     
                            ║         ║                                                     
             c:    ═════════╬═════════╩═════════════════════════════════════════════════════
                            ║         0                                                     
             z:    ═════════╩═══════════════════════════════════════════════════════════════
                            0                                                               

As you can see in the code above the indexing of the different registers always starts by the quantum register. If your operator interacts with the three registers the index will have to be provided in the following order: 
#. Index of the qantum register.
#. Index of the classical register.
#. Index of the z-register.


Be careful when writing information to the z-register or to the classical register as the information can be easily overwritten if the same index is used multiple times. For example if you measure two different qubits and store both in the same classical bit the results of the sampling will only report the last measurement.

To retrieve information on the number of element of each register you can use the :meth:`~mimiqcircuits.Circuit.num_qubits` , :meth:`~mimiqcircuits.Circuit.num_bits`  and :meth:`~mimiqcircuits.Circuit.numz_vars` .

.. doctest:: python

    >>> circuit.num_qubits(), circuit.num_bits(), circuit.num_zvars()
    (1, 1, 1)

 

In the following sections you will learn in details how to build a circuit in MIMIQ.

Creating a circuit
----------------------------------------------------

The first step in executing quantum algorithm on MIMIQ always consists in implementing the corresonding quantum circuit, a sequence of quantum operations (quantum gates, measurements, resets, etc...) that acts on a set of qubits. In MIMIQ we always start by defining an empty circuit

.. doctest:: python

    >>> circuit = Circuit()

 
There is no need to give any arguments. Not even the number of qubits, classical or Z-registers is necessary as it will be directly inferred from the operations added to the circuit.


Adding Gates
----------------------------------------------------

Once a circuit is instantiated operations can be added to it.
To see the list of gates available head to :class:`~mimiqcircuits.OPERATIONS` , :class:`~mimiqcircuits.GATES` , :class:`~mimiqcircuits.NOISECHANNELS`  and :class:`~mimiqcircuits.GENERALIZED`  or enter the following command in your Python session:

.. doctest::

    help(Gates)

To know more about the types of operations you can use in a circuit head to the :doc:`unitary gates <unitary_gates>`, :doc:`non-unitary operations <non_unitary_ops>`, :doc:`noise <noise>`, :doc:`symbolic operations <symbolic_ops>` and :doc:`special operations <special_ops>` pages.


`push`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add gates to circuits in Python we will mainly be using the :meth:`~mimiqcircuits.Circuit.push` method. The arguments needed by :meth:`~mimiqcircuits.Circuit.push` can vary, but in general it expects the following: 
#. The circuit to add the operation to.
#. The operator to be added. 
#. As many targets as needed by the operator (qubits/bits/zvars).


For instance you can add the gate `X` by simply running the following command:

.. doctest:: python

    >>> circuit.push(GateX(), 0)
    1-qubit circuit with 1 instructions:
    └── X @ q[0]
    <BLANKLINE>

 
The text representation ```H @ q[0]``` informs us that there is an instruction which applies the Hadamard gate to the qubit of index 1.


Some gates require multiple target qubits such as the CX gate.
Here is how to add such a gate to the circuit:

.. doctest:: python

    >>> circuit = Circuit() 
    >>> circuit.push(GateCX(), 0, 1)
    2-qubit circuit with 1 instructions:
    └── CX @ q[0], q[1]
    <BLANKLINE>

 
This will add the gate :class:`~mimiqcircuits.GateCX`  using the qubit number :code:`1` as the control qubit and number :code:`2` as the target qubit in the :code:`circuit`.

`push` specifics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~mimiqcircuits.Circuit.push` is very versatile, it can be used to add multiple operators to multiple targets at once using iterators.

To add one type of gate to multiple qubits use:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateX(), range(0, 11))
    11-qubit circuit with 11 instructions:
    ├── X @ q[0]
    ├── X @ q[1]
    ├── X @ q[2]
    ├── X @ q[3]
    ├── X @ q[4]
    ├── X @ q[5]
    ├── X @ q[6]
    ├── X @ q[7]
    ├── X @ q[8]
    ├── X @ q[9]
    └── X @ q[10]
    <BLANKLINE>

 
This will add one `X` gate on each qubit from number 1 to 10.

This also works on 2-qubit gates:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateID(), 0) # For documentation purpose, ignore this line
    1-qubit circuit with 1 instructions:
    └── ID @ q[0]
    <BLANKLINE>

    >>> # Adds 3 CX gates using respectively 1, 2 & 3 as the control qubits and 4 as the target qubit for all 
    >>> circuit.push(GateCX(), range(0, 3), 3)
    4-qubit circuit with 4 instructions:
    ├── ID @ q[0]
    ├── CX @ q[0], q[3]
    ├── CX @ q[1], q[3]
    └── CX @ q[2], q[3]
    <BLANKLINE>

    >>> # Adds 3 CX gates using respectively 2, 3 & 4 qubits as the target and 1 as the control qubit for all
    >>> circuit.push(GateCX(), 0, range(1, 4))
    4-qubit circuit with 7 instructions:
    ├── ID @ q[0]
    ├── CX @ q[0], q[3]
    ├── CX @ q[1], q[3]
    ├── CX @ q[2], q[3]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    └── CX @ q[0], q[3]
    <BLANKLINE>

    >>> # adds 3 CX gates using respectively the couples (1, 4), (2, 5), (3, 6) as the control and target qubits
    >>> circuit.push(GateCX(), range(0, 3), range(3, 6))
    6-qubit circuit with 10 instructions:
    ├── ID @ q[0]
    ├── CX @ q[0], q[3]
    ├── CX @ q[1], q[3]
    ├── CX @ q[2], q[3]
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    ├── CX @ q[0], q[3]
    ├── CX @ q[0], q[3]
    ├── CX @ q[1], q[4]
    └── CX @ q[2], q[5]
    <BLANKLINE>

    >>> circuit.draw()
                    ┌──┐                                                                    
             q[0]: ╶┤ID├─●────────●──●──●──●───────────────────────────────────────────────╴
                    └──┘ │       ┌┴┐ │  │  │                                                
             q[1]: ╶─────┼──●────┤X├─┼──┼──┼──●────────────────────────────────────────────╴
                         │  │    └─┘┌┴┐ │  │  │                                             
             q[2]: ╶─────┼──┼──●────┤X├─┼──┼──┼──●─────────────────────────────────────────╴
                        ┌┴┐┌┴┐┌┴┐   └─┘┌┴┐┌┴┐ │  │                                          
             q[3]: ╶────┤X├┤X├┤X├──────┤X├┤X├─┼──┼─────────────────────────────────────────╴
                        └─┘└─┘└─┘      └─┘└─┘┌┴┐ │                                          
             q[4]: ╶─────────────────────────┤X├─┼─────────────────────────────────────────╴
                                             └─┘┌┴┐                                         
             q[5]: ╶────────────────────────────┤X├────────────────────────────────────────╴
                                                └─┘                                         
                                                                                

Be careful when using vectors for both control and target, if one of the two vectors in longer than the other only the `N` first element of the vector will be accounted for with :code:`N = min(length.(vector1, vector2))`.
See the output of the code below to see the implication in practice:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateID(), 0) # For documentation purpose, ignore this line
    1-qubit circuit with 1 instructions:
    └── ID @ q[0]
    <BLANKLINE>

    >>> # Adds only 3 CX gates
    >>> circuit.push(GateCX(), range(0, 3), range(3, 18))
    6-qubit circuit with 4 instructions:
    ├── ID @ q[0]
    ├── CX @ q[0], q[3]
    ├── CX @ q[1], q[4]
    └── CX @ q[2], q[5]
    <BLANKLINE>

    >>> circuit.draw()
                    ┌──┐                                                                    
             q[0]: ╶┤ID├─●─────────────────────────────────────────────────────────────────╴
                    └──┘ │                                                                  
             q[1]: ╶─────┼──●──────────────────────────────────────────────────────────────╴
                         │  │                                                               
             q[2]: ╶─────┼──┼──●───────────────────────────────────────────────────────────╴
                        ┌┴┐ │  │                                                            
             q[3]: ╶────┤X├─┼──┼───────────────────────────────────────────────────────────╴
                        └─┘┌┴┐ │                                                            
             q[4]: ╶───────┤X├─┼───────────────────────────────────────────────────────────╴
                           └─┘┌┴┐                                                           
             q[5]: ╶──────────┤X├──────────────────────────────────────────────────────────╴
                              └─┘                                                           

You can also use tuples or vectors in the exact same fashion:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateID(), 0) # For documentation purpose, ignore this line
    1-qubit circuit with 1 instructions:
    └── ID @ q[0]
    <BLANKLINE>
    >>> circuit.push(GateCX(), (0, 1), (2, 3))
    4-qubit circuit with 3 instructions:
    ├── ID @ q[0]
    ├── CX @ q[0], q[2]
    └── CX @ q[1], q[3]
    <BLANKLINE>
    >>> circuit.push(GateCX(), [0, 2], [1, 3])
    4-qubit circuit with 5 instructions:
    ├── ID @ q[0]
    ├── CX @ q[0], q[2]
    ├── CX @ q[1], q[3]
    ├── CX @ q[0], q[1]
    └── CX @ q[2], q[3]
    <BLANKLINE>

    >>> circuit.draw()
                    ┌──┐                                                                    
             q[0]: ╶┤ID├─●─────●───────────────────────────────────────────────────────────╴
                    └──┘ │    ┌┴┐                                                           
             q[1]: ╶─────┼──●─┤X├──────────────────────────────────────────────────────────╴
                        ┌┴┐ │ └─┘                                                           
             q[2]: ╶────┤X├─┼─────●────────────────────────────────────────────────────────╴
                        └─┘┌┴┐   ┌┴┐                                                        
             q[3]: ╶───────┤X├───┤X├───────────────────────────────────────────────────────╴
                           └─┘   └─┘                                                        
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                                                                     

Insert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also insert an operation at a given index in the circuit using the :meth:`~mimiqcircuits.Circuit.insert` function:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateX(), 1)
    2-qubit circuit with 1 instructions:
    └── X @ q[1]
    <BLANKLINE>
    >>> circuit.push(GateZ(), 1)
    2-qubit circuit with 2 instructions:
    ├── X @ q[1]
    └── Z @ q[1]
    <BLANKLINE>

    >>> # Insert the gate at a specific index
    >>> circuit.insert(2, GateY(), 1)
    2-qubit circuit with 3 instructions:
    ├── X @ q[1]
    ├── Z @ q[1]
    └── Y @ q[1]
    <BLANKLINE>
    >>> circuit
    2-qubit circuit with 3 instructions:
    ├── X @ q[1]
    ├── Z @ q[1]
    └── Y @ q[1]
    <BLANKLINE>

 
This will insert :class:`~mimiqcircuits.GateY`  applied on qubit ```1``` at the second position in the circuit.

Append
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To append one circuit to another you can use the :meth:`~mimiqcircuits.Circuit.append` function:

.. doctest:: python

    >>> # Build a first circuit
    >>> circuit1 = Circuit()
    >>> circuit1.push(GateX(), range(1, 4))
    4-qubit circuit with 3 instructions:
    ├── X @ q[1]
    ├── X @ q[2]
    └── X @ q[3]
    <BLANKLINE>

    >>> # Build a second circuit
    >>> circuit2 = Circuit()
    >>> circuit2.push(GateY(), range(1, 4))
    4-qubit circuit with 3 instructions:
    ├── Y @ q[1]
    ├── Y @ q[2]
    └── Y @ q[3]
    <BLANKLINE>

    >>> # Append the second circuit to the first one
    >>> circuit1.append(circuit2)
    >>> circuit1
    4-qubit circuit with 6 instructions:
    ├── X @ q[1]
    ├── X @ q[2]
    ├── X @ q[3]
    ├── Y @ q[1]
    ├── Y @ q[2]
    └── Y @ q[3]
    <BLANKLINE>

 
This will modify `circuit1` by appending all the operations from `circuit2`.

This function is particularly useful for building circuits by combining smaller circuit blocks.

Visualizing circuits
----------------------------------------------------

To visualize a circuit use the :meth:`draw`  method.
ere is a representation of a sim
.. doctest:: python

    >>> circuit = Circuit() 
    >>> circuit.push(GateX(), range(0, 5)) 
    5-qubit circuit with 5 instructions:
    ├── X @ q[0]
    ├── X @ q[1]
    ├── X @ q[2]
    ├── X @ q[3]
    └── X @ q[4]
    <BLANKLINE>
    >>> circuit.draw()
                    ┌─┐                                                                     
             q[0]: ╶┤X├────────────────────────────────────────────────────────────────────╴
                    └─┘┌─┐                                                                  
             q[1]: ╶───┤X├─────────────────────────────────────────────────────────────────╴
                       └─┘┌─┐                                                               
             q[2]: ╶──────┤X├──────────────────────────────────────────────────────────────╴
                          └─┘┌─┐                                                            
             q[3]: ╶─────────┤X├───────────────────────────────────────────────────────────╴
                             └─┘┌─┐                                                         
             q[4]: ╶────────────┤X├────────────────────────────────────────────────────────╴
                                └─┘                                                         
                                                                                
Information such as the :meth:`~mimiqcircuits.Circuit.depth`  and the width (:meth:`~mimiqcircuits.Circuit.num_qubits` ) can be extracted from the circuit:

.. doctest:: python

    >>> circuit.depth(), circuit.num_qubits()
    (1, 5)

 

Decompose
----------------------------------------------------

Most gates can be decomposed into a combination of `U` and `CX` gates, the :meth:`~mimiqcircuits.Circuit.decompose`  function extracts such decomposition from a given circuit:

.. doctest:: python

    >>> circuit = Circuit()
    >>> circuit.push(GateX(), 0)
    1-qubit circuit with 1 instructions:
    └── X @ q[0]
    <BLANKLINE>

    >>> # decompose the circuit
    >>> circuit.decompose()
    1-qubit circuit with 1 instructions:
    └── U(pi, 0, pi, 0.0) @ q[0]
    <BLANKLINE>

Reference
---------

.. autoclass:: mimiqcircuits.Circuit
    :noindex:
.. autoclass:: mimiqcircuits.Instruction
    :noindex:
.. autofunction:: mimiqcircuits.Circuit.push
    :noindex:
.. autofunction:: mimiqcircuits.Circuit.insert
    :noindex:
.. autofunction:: mimiqcircuits.Circuit.append
    :noindex:
.. autofunction:: mimiqcircuits.Circuit.draw
    :noindex:
.. autofunction:: mimiqcircuits.Circuit.decompose
    :noindex:
