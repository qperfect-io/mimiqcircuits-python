Special Operations
==================

MIMIQ offers further possibilities to create circuits, such as new gate declarations, or wrappers for common combinations of gates.

Contents
========

.. contents::
   :local:
   :depth: 2
   :backlinks: entry


Gate Declaration & Gate Calls
---------------------------------
.. _gate-declaration--gate-calls:

Using MIMIQ you can define your own gates with a given name, arguments and instructions.
For examples if you wish to apply an `H` gate followed by an `RX` gate with a specific argument for 
the rotation you can use :class:`~mimiqcircuits.gatedecl.GateDecl` as follows:

.. doctest::
    :hide:

    >>> from mimiqcircuits import *

.. doctest::

    >>> from symengine import *
    >>> rot = symbols('x')
    >>> @gatedecl("ansatz")
    ... def ansatz(rot):
    ...     c= Circuit()
    ...     c.push(GateX(),0)
    ...     c.push(GateP(rot),1)
    ...     return c

    >>> ansatz(rot)
    ansatz(x)

Here, `ansatz` is simply the name that will be shown when printing or drawing 
the circuit, `(rot)` defines the gate parameters.

As you can see in the code above, to generate your own gate declaration you will need to 
instantiate :class:`~mimiqcircuits.Circuit`. A circuit is created by sequentially applying operations using :meth:`~mimiqcircuits.Circuit.push`, 
where each operation is followed by its targets. The target order follows the standard convention quantum register -> classical register -> Z-register order.
passed as an argument.

.. doctest::

    >>> ansatz(rot)
    ansatz(x)


You can check the instructions inside `ansatz` by using `decompose` method:

.. doctest::

    >>> ansatz(pi/2).decompose()
    2-qubit circuit with 2 instructions:
    ├── X @ q[0]
    └── P((1/2)*pi) @ q[1]
    <BLANKLINE>


After declaration you can add it to your circuit using :meth:`~mimiqcircuits.Circuit.push`.

.. doctest::

    >>> c = Circuit()
    >>> c.push(ansatz(pi/2), 0 , 1)  
    2-qubit circuit with 1 instructions:
    └── ansatz((1/2)*pi) @ q[0,1]
    <BLANKLINE>

.. Note::

    A gate declared with :class:`~mimiqcircuits.GateDecl` must be unitary.

.. Note::

    The :func:`~mimiqcircuits.gatedecl` decorator transforms a function into one that produces 
    :class:`~mimiqcircuits.GateCall` objects based on the logic defined in a 
    :class:`~mimiqcircuits.GateDecl`. When you call `ansatz(pi)`, it creates an instance of 
    :class:`~mimiqcircuits.GateCall`, representing a specific instantiation of the unitary gates 
    with the provided parameters.



Creating a gate declaration allows you to add easily the same sequence of gates in a very versatile way and manipulate your new gate 
like you would with any other gate. This means that you can combine it with other gates via :class:`~mimiqcircuits.Control`, add noise to 
the whole block in one call, use it as an operator for :class:`~mimiqcircuits.ExpectationValue`, use it within an :class:`~mimiqcircuits.IfStatement` etc. 
See :doc:`non-unitary operations </manual/non_unitary_ops>`, and :doc:`noise </manual/noise>` pages.

For example, here is how to add noise to the previous gate declaration:

.. doctest::

    >>> c = Circuit()
    >>> my_gate = ansatz(pi)
    >>> c.push(my_gate, 0, 1)
    2-qubit circuit with 1 instructions:
    └── ansatz(pi) @ q[0,1]
    <BLANKLINE>
    >>> c.add_noise(my_gate, Depolarizing2(0.1))
    2-qubit circuit with 2 instructions:
    ├── ansatz(pi) @ q[0,1]
    └── Depolarizing(0.1) @ q[0,1]
    <BLANKLINE>

    >>> c.draw()
            ┌────────────┐  ┌───────────────────┐                                   
     q[0]: ╶┤0           ├──┤0                  ├──────────────────────────────────╴
            │  ansatz(pi)│  │  Depolarizing(0.1)│                                   
     q[1]: ╶┤1           ├──┤1                  ├──────────────────────────────────╴
            └────────────┘  └───────────────────┘                                   
                                                                                
                                                                                                                                                                                                                                                                                                                
Gate declarations can be combined with other quantum operations like :class:`~mimiqcircuits.Control`, noise, or even conditional logic.
Use it within an :class:`~mimiqcircuits.IfStatement`:

.. doctest::

    >>> IfStatement(my_gate, BitString("111"))
    IF (c==111) ansatz(pi)

Note that this type of combined operation does not work if we pass a circuit as an argument, instead of a declared gate 
(more precisely, a :class:`~mimiqcircuits.GateCall`, see note above).


Blocks of Instructions
----------------------
.. _blocks-of-instructions:

Blocks in MIMIQ allow you to encapsulate a collection of quantum operations as a reusable unit. 
They're particularly useful when you want to group instructions together that implement a specific algorithm or subroutine.

Unlike gate declarations that create new gates, blocks simply group existing instructions without restriction on their nature or type.

A :class:`~mimiqcircuits.Block`` can be created in several ways:

.. doctest::

    # From an existing circuit
    >>> circ = Circuit()
    >>> circ.push(GateH(), 0)
    1-qubit circuit with 1 instructions:
    └── H @ q[0]
    <BLANKLINE>

    >>> circ.push(GateCX(), 0, 1)
    2-qubit circuit with 2 instructions:
    ├── H @ q[0]
    └── CX @ q[0], q[1]
    <BLANKLINE>

    >>> block = Block(circ)
    >>> block
    2-qubit block ... with 2 instructions:
    ├── H @ q[0]
    └── CX @ q[0], q[1]

    # From a list of instructions
    >>> inst = [Instruction(GateX(), (0,)), Instruction(GateY(), (1,))]
    >>> block2 = Block(inst)
    >>> block2
    2-qubit block ... with 2 instructions:
    ├── X @ q[0]
    └── Y @ q[1]

    # Empty block with specified dimensions
    >>> block3 = Block(2, 1, 0)  # 2 qubits, 1 classical bit, 0 z-variables
    >>> block3
    empty circuit

Example: Error Correction Code Block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _Error-Correction-block:

.. doctest::

    >>> def error_detection_block():
    ...     c = Circuit()
    ...     c.push(GateCX(), 0, 1)
    ...     c.push(GateCX(), 0, 2)
    ...     c.push(MeasureZ(), 1, 0)  # Measure q[1] → c[0]
    ...     c.push(MeasureZ(), 2, 1)  # Measure q[2] → c[1]
    ...     c.push(IfStatement(GateX(), BitString("01")), 0, 0, 1)  # Error correction
    ...     c.push(IfStatement(GateX(), BitString("10")), 0, 0, 1)  # Error correction
    ...     return Block(c)

    >>> error_detection = error_detection_block()
    >>> error_detection
    3-qubit, 2-bit block ... with 6 instructions:
    ├── CX @ q[0], q[1]
    ├── CX @ q[0], q[2]
    ├── M @ q[1], c[0]
    ├── M @ q[2], c[1]
    ├── IF (c==01) X @ q[0], c[0,1]
    └── IF (c==10) X @ q[0], c[0,1]
    
    >>> main_circuit = Circuit()
    >>> main_circuit.push(error_detection, 0, 1, 2, 0, 1)
    3-qubit, 2-bit circuit with 1 instructions:
    └── block ... @ q[0,1,2], c[0,1]
    <BLANKLINE>
   
    >>> main_circuit.push(GateH(), 1)
    3-qubit, 2-bit circuit with 2 instructions:
    ├── block ... @ q[0,1,2], c[0,1]
    └── H @ q[1]
    <BLANKLINE>
   
   
    >>> main_circuit.push(error_detection, 0, 1, 2, 0, 1)
    3-qubit, 2-bit circuit with 3 instructions:
    ├── block ... @ q[0,1,2], c[0,1]
    ├── H @ q[1]
    └── block ... @ q[0,1,2], c[0,1]
    <BLANKLINE>
    
   
Working with Blocks
^^^^^^^^^^^^^^^^^^^
.. _working-with-blocks:

.. doctest::

    >>> b = Block(2, 1, 0)
    >>> b
    empty circuit
    >>> b.push(GateH(), 0)
    >>> b.push(GateX(), 1)
    >>> b.push(GateCX(), 0, 1)

Blocks can be iterated over, indexed, and have a length just like circuits:

.. doctest::

    >>> len(b)
    3
   
    >>> b[0]
    H @ q[0]

Trying to add operations that use more resources than the block dimensions will result in an error:

.. doctest::

    >>> try:
    ...     b.push(GateZ(), 3)
    ... except Exception as e:
    ...     print(type(e).__name__)
    ValueError
 

When to Use Blocks vs Gate Declarations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _when-Use-blocks-vs-gatedeclaration:

Use :class:`~mimiqcircuits.Block` when you want to:

- Group a sequence of operations for organization
- Reuse a routine with control/measurement logic
- Include operations that are not unitary

Use :class:`~mimiqcircuits.GateDecl` when you want to:

- Define a new named unitary gate
- Create parameterized gates
- Use symbolic arguments and controlled gate behavior


Repeated Operations
-------------------
.. _reseated-operations:

The :class:`~mimiqcircuits.Repeat` operation allows you to apply the same quantum instruction multiple times. 
operation allows you to apply the same quantum operation multiple times. This can be particularly useful for algorithms that 
require iterative application of the same operation, such as quantum walks or amplitude amplification.

Creating Repeated Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _creating-reapeated-operations:

There are two main ways to create repeated operations:

.. code-block:: python

    # Method 1: Using the Repeat constructor directly
    Repeat(n, operation)  # Repeats the given operation `n` times

    # Method 2: Using the `repeat` helper function (recommended)
    repeat(n, operation)  # Repeats the operation with optional simplification logic

    # Alternative: Using the `.repeat()` method on an operation instance
    operation.repeat(n)   # Shorthand for repeating an operation `n` times


.. doctest::
    :hide:

    >>> from symengine import Symbol

.. doctest::

    >>> Repeat(3, GateX())
    ∏³ X
   
    >>> GateRX(Symbol("theta")).repeat(5)
    ∏⁵ RX(theta)
    
    >>> repeat(5, GateRX(Symbol("theta"))).evaluate({"theta": 1})
    ∏⁵ RX(1)

    >>> c = Circuit()
    >>> c.push(repeat(3, GateH()), 0)
    1-qubit circuit with 1 instructions:
    └── ∏³ H @ q[0]
    <BLANKLINE>
   
    >>> c.push(repeat(5, GateRX(Symbol("θ"))), 1)
    2-qubit circuit with 2 instructions:
    ├── ∏³ H @ q[0]
    └── ∏⁵ RX(θ) @ q[1]
    <BLANKLINE>
   
    >>> c.decompose()
    2-qubit circuit with 8 instructions:
    ├── H @ q[0]
    ├── H @ q[0]
    ├── H @ q[0]
    ├── RX(θ) @ q[1]
    ├── RX(θ) @ q[1]
    ├── RX(θ) @ q[1]
    ├── RX(θ) @ q[1]
    └── RX(θ) @ q[1]
    <BLANKLINE>
 

Composite Gates
---------------------------------
.. _composite-gates:

MIMIQ provides several composite gates to facilitate circuit building. These gates simplify constructing complex operations.

Pauli String
---------------------------------
.. _pauli-string:

A :class:`~mimiqcircuits.PauliString` is an `N`-qubit tensor product of Pauli operators of the form:

.. math::

   P_1 \otimes P_2 \otimes P_3 \otimes \ldots \otimes P_N,

where each :math:`P_i \in \{ I, X, Y, Z \}` is a single-qubit Pauli operator, including the identity.

To create an operator using :class:`~mimiqcircuits.PauliString` we simply pass as argument the Pauli string written as a `String`:

.. doctest:: python

    >>> c = Circuit()
    >>> c.push(PauliString("IXYZ"), 1, 2, 3, 4)
    5-qubit circuit with 1 instructions:
    └── IXYZ @ q[1,2,3,4]
    <BLANKLINE>

You can specify any number of Pauli operators.

Quantum Fourier Transform
---------------------------------
.. _quantum-fourier-transform:

The :class:`~mimiqcircuits.QFT` gate implements the The `Quantum Fourier Transform <https://en.wikipedia.org/wiki/Quantum_Fourier_transform>`__ which is a 
circuit used to realize a linear transformation on qubits and is a building block of many larger circuits such as `Shor's Algorithm <https://en.wikipedia.org/wiki/Shor%27s_algorithm>`__ or 
the `Quantum Phase Estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__.

The QFT maps an arbitrary quantum state :math:`\ket{x} = \sum_{j=0}^{N-1} x_{j} \ket{j}`  
to a quantum state :math:`\sum_{k=0}^{N-1} y_{k} \ket{k}` according to the formula:


.. math::

    y_{k} = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_{j}w_{N}^{-jk}

where :math:`w_N = e^{2\pi i / N}`.

In MIMIQ, the :class:`~mimiqcircuits.QFT` gate allows you to quickly implement a QFT in your circuit on an arbitrary ``N`` number of qubits.  
You can instantiate the QFT gate by providing the number of qubits you want to use, `QFT(N)`, and add it like any other gate in the circuit.

.. doctest:: 

    >>> c = Circuit()
    >>> c.push(QFT(5), 1, 2, 3, 4, 5)
    6-qubit circuit with 1 instructions:
    └── QFT @ q[1,2,3,4,5]
    <BLANKLINE>

This adds a 5-qubit QFT to the circuit.

Phase Gradient
---------------------------------
.. _phase-gradient:

The :class:`~mimiqcircuits.PhaseGradient` applies a phase shift to a quantum register of ``N`` qubits, where each computational basis state :math:`\ket{k}` 
experiences a phase proportional to its integer value ``k``:

.. math::

    \operatorname{PhaseGradient} =
    \sum_{k=0}^{N-1} \mathrm{e}^{i \frac{2 \pi}{N} k} \ket{k}\bra{k}

To use it, you can simply provide the number of qubit targets and add it to the circuit as shown in the following examples:

.. doctest:: 

    >>> c = Circuit()
    >>> c.push(PhaseGradient(5), 1, 2, 3, 4, 5)
    6-qubit circuit with 1 instructions:
    └── PhaseGradient @ q[1,2,3,4,5]
    <BLANKLINE>

This will add a 5 qubits :class:`~mimiqcircuits.PhaseGradient` to the first 5 qubits of the quantum register.


Polynomial Oracle
---------------------------------
.. _polynomial-oracle:

.. warning::

    The :class:`~mimiqcircuits.PolynomialOracle` works only with the state vector simulator and not with MPS, because of 
    ancillas qubit use.

The :class:`~mimiqcircuits.PolynomialOracle` is a quantum oracle for a polynomial function of two 
registers. It applies a :math:`\pi` phase shift to any basis state that satisfies 
:math:`a \cdot xy + b \cdot x + c \cdot y + d = 0`, 
where :math:`\ket{x}` and :math:`\ket{y}` are the states of the two registers.

Here is how to use the :class:`~mimiqcircuits.PolynomialOracle`:


.. doctest:: 

    >>> c = Circuit()
    >>> c.push(PolynomialOracle(5, 5, 1, 2, 3, 4), *range(10))
    10-qubit circuit with 1 instructions:
    └── PolynomialOracle(1, 2, 3, 4) @ q[0,1,2,3,4], q[5,6,7,8,9]
    <BLANKLINE>

Diffusion
---------------------------------
.. _diffusion:

The :class:`~mimiqcircuits.Diffusion` operator corresponds to `Grover's diffusion operator <https://en.wikipedia.org/wiki/Grover%27s_algorithm>`__.  
It implements the unitary transformation:

.. math::

    H^{\otimes n} (1 - 2\ket{0^n} \bra{0^n}) H^{\otimes n}

Here is how to use :class:`~mimiqcircuits.Diffusion`:

.. doctest:: 

    >>> c = Circuit()
    >>> c.push(Diffusion(10), *range(10))
    10-qubit circuit with 1 instructions:
    └── Diffusion @ q[0,1,2,3,4,5,6,7,8,9]
    <BLANKLINE>

You need to specify both the number of targets and their corresponding indices.


More about composite gates
---------------------------------
.. _more-about-composite-gates:

All composite gates can be decomposed using :meth:`~mimiqcircuits.Circuit.decompose` to 
extract their implementation, except for :class:`~mimiqcircuits.PolynomialOracle`.


.. doctest:: python

    >>> QFT(5).decompose()
    5-qubit circuit with 15 instructions:
    ├── H @ q[4]
    ├── CP(0.5*pi) @ q[3], q[4]
    ├── H @ q[3]
    ├── CP(0.25*pi) @ q[2], q[4]
    ├── CP(0.5*pi) @ q[2], q[3]
    ├── H @ q[2]
    ├── CP(0.125*pi) @ q[1], q[4]
    ├── CP(0.25*pi) @ q[1], q[3]
    ├── CP(0.5*pi) @ q[1], q[2]
    ├── H @ q[1]
    ├── CP(0.0625*pi) @ q[0], q[4]
    ├── CP(0.125*pi) @ q[0], q[3]
    ├── CP(0.25*pi) @ q[0], q[2]
    ├── CP(0.5*pi) @ q[0], q[1]
    └── H @ q[0]
    <BLANKLINE>

Barrier
---------------------------------
.. _barrier:

The :class:`~mimiqcircuits.Barrier` is a non-op operation that does not affect the quantum state but 
prevents compression or optimization across execution.  
As of now, :class:`~mimiqcircuits.Barrier` is only useful when combined with the MPS backend.

To add barriers to the circuit, you can use the :class:`~mimiqcircuits.Barrier` operation:

Example usage:

.. doctest::
    :hide:

    >>> from mimiqcircuits import *
    >>> c = Circuit()

.. doctest::

    # Add a Gate
    >>> c.push(GateX(), 0)
    1-qubit circuit with 1 instructions:
    └── X @ q[0]
    <BLANKLINE>

    # Apply the Barrier on qubit 0.
    >>> c.push(Barrier(1), 0)
    1-qubit circuit with 2 instructions:
    ├── X @ q[0]
    └── Barrier @ q[0]
    <BLANKLINE>

    # Add a Gate between barriers
    >>> c.push(GateX(), 0)
    1-qubit circuit with 3 instructions:
    ├── X @ q[0]
    ├── Barrier @ q[0]
    └── X @ q[0]
    <BLANKLINE>

    # Apply individual barriers on multiple qubits
    >>> c.push(Barrier(1), range(3))
    3-qubit circuit with 6 instructions:
    ├── X @ q[0]
    ├── Barrier @ q[0]
    ├── X @ q[0]
    ├── Barrier @ q[0]
    ├── Barrier @ q[1]
    └── Barrier @ q[2]
    <BLANKLINE>

    # Add gates on multiple qubits
    >>> c.push(GateX(), range(3))
    3-qubit circuit with 9 instructions:
    ├── X @ q[0]
    ├── Barrier @ q[0]
    ├── X @ q[0]
    ├── Barrier @ q[0]
    ├── Barrier @ q[1]
    ├── Barrier @ q[2]
    ├── X @ q[0]
    ├── X @ q[1]
    └── X @ q[2]
    <BLANKLINE>

    # Apply one general Barrier on multiple qubits (effectively the same as above)
    >>> c.push(Barrier(3), *range(3))
    3-qubit circuit with 10 instructions:
    ├── X @ q[0]
    ├── Barrier @ q[0]
    ├── X @ q[0]
    ├── Barrier @ q[0]
    ├── Barrier @ q[1]
    ├── Barrier @ q[2]
    ├── X @ q[0]
    ├── X @ q[1]
    ├── X @ q[2]
    └── Barrier @ q[0,1,2]
    <BLANKLINE>

    >>> c.draw()
            ┌─┐ ┌─┐   ┌─┐                                                           
     q[0]: ╶┤X├░┤X├░──┤X├──────░───────────────────────────────────────────────────╴
            └─┘░└─┘░  └─┘┌─┐   ░                                                    
     q[1]: ╶────────░────┤X├───░───────────────────────────────────────────────────╴
                    ░    └─┘┌─┐░                                                    
     q[2]: ╶─────────░──────┤X├░───────────────────────────────────────────────────╴
                     ░      └─┘░                                                    
                                                                                
