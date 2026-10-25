Circuit Equivalence Testing
===========================

This section outlines methodologies for verifying the equivalence of two quantum circuits, :math:`U` and :math:`V`.

Definition
----------

Two quantum circuits :math:`U` and :math:`V` are defined as equivalent if they perform the same unitary operation on the Hilbert space :math:`\mathcal{H}`. That is, for every state :math:`|\psi\rangle \in \mathcal{H}`:

.. math::

    U|\psi\rangle = V|\psi\rangle

This condition holds if and only if :math:`V^\dagger U = I`, where :math:`I` is the identity operator. Verification requires confirming this equality across a basis spanning :math:`\mathcal{H}`.

Verification Methodologies
--------------------------

Exact verification via full simulation typically entails exponential scaling with the number of qubits :math:`n`. Alternative approaches include statistical sampling and specific algorithmic techniques.

Monte-Carlo Sampling
~~~~~~~~~~~~~~~~~~~~

Monte-Carlo methods estimate equivalence by satisfying the condition for a randomly sampled subset of state vectors. While this method does not provide a formal proof of equivalence (as it does not cover the entire Hilbert space), it is computationally efficient and effective for identifying non-equivalence. If :math:`U|\psi_i\rangle \neq V|\psi_i\rangle` for any sampled state :math:`|\psi_i\rangle`, the circuits are not equivalent.

Choi-State Analysis (Implemented in MIMIQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Choi–Jamiołkowski isomorphism** allows representing a quantum channel (or unitary) as a quantum state. This method enables exact verification using :math:`2n` qubits for an :math:`n`-qubit unitary.

**MIMIQ Circuits implements this method** via the :class:`~mimiqcircuits.CircuitTesterExperiment` class.

The procedure involves:

1.  Preparing a maximally entangled state on :math:`2n` qubits.
2.  Applying :math:`U` to the first :math:`n` qubits to create a Choi state.
3.  Applying :math:`V^\dagger` (the inverse of :math:`V`) to the first :math:`n` qubits (or equivalently, applying the inverse preparation sequence).

If the circuits are equivalent, the final state will return to the initial maximally entangled state with probability 1. Verification checks if the measurement probability :math:`P(|0\rangle^{\otimes 2n}) = 1`.

Formalism Conversion
~~~~~~~~~~~~~~~~~~~~

Circuits can also be verified by converting them into alternative representations that facilitate analytical comparison.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Formalism
     - Description
     - Equivalence Check Method
   * - **Decision Diagrams (DD)**
     - Describes circuits in a compact form by exploiting matrix sparseness and repetitions.
     - Check if both diagrams are identical or if their product results in the identity diagram.
   * - **ZX-Calculus**
     - Focuses on topological equivalence and flexible fusion/unfusion of operators.
     - Construct a :math:`V^\dagger U` diagram and use optimization tools to simplify it to the identity diagram.

Choi-State Preparation Algorithm
--------------------------------

The construction of the Choi state :math:`(U \otimes I)|\Phi^+\rangle` follows these steps:

1.  **Initialization:** Initialize :math:`2n` qubits in the state :math:`|0\rangle^{\otimes 2n}`.
2.  **Superposition:** Apply Hadamard gates to the first :math:`n` qubits: :math:`H^{\otimes n}`.
3.  **Entanglement:** Apply CNOT gates with control qubit :math:`i` and target qubit :math:`i+n` for each :math:`i \in \{0, \dots, n-1\}`. This creates the maximally entangled state :math:`|\Phi^+\rangle^{\otimes n}`.
4.  **Operator Application:** Apply the unitary :math:`U` to the first :math:`n` qubits. The resulting state corresponds to the Choi state of :math:`U`.

Usage Guide
-----------

MIMIQ provides a built-in workflow for equivalence checking using the :class:`~mimiqcircuits.CircuitTesterExperiment`.

**1. Define the circuits**

Define the two circuits you wish to compare. They can be identical or structurally different, as long as they implement the same unitary.

.. code-block:: python

    from mimiqcircuits import *

    # Number of qubits
    n_qubits = 4

    # Circuit 1: GHZ state using "Fountain" (or Star) strategy
    # CNOTs from qubit 0 to all others
    c1 = Circuit()
    c1.push(GateH(), 0)
    for i in range(1, n_qubits):
        c1.push(GateCX(), 0, i)

    # Circuit 2: GHZ state using "Ladder" (or Chain) strategy
    # CNOTs from i to i+1
    c2 = Circuit()
    c2.push(GateH(), 0)
    for i in range(n_qubits - 1):
        c2.push(GateCX(), i, i+1)

**2. Create the Experiment**

Instantiate a :class:`~mimiqcircuits.CircuitTesterExperiment` with the two circuits.

You can choose between two verification methods by setting the ``method`` parameter:

*   ``"samples"`` (default): Measures the final state in the computational basis. Equivalence is verified if all samples are :math:`|0\dots0\rangle`. This is efficient but probabilistic (unless all states are sampled).
*   ``"amplitudes"``: Directly computes the amplitude of the :math:`|0\dots0\rangle` state. This method returns the exact probability :math:`|A|^2`.

.. code-block:: python

    from mimiqcircuits import CircuitTesterExperiment

    # Default method="samples"
    exp = CircuitTesterExperiment(c1, c2)

    # Alternative: use method="amplitudes"
    exp_amp = CircuitTesterExperiment(c1, c2, method="amplitudes")

**3. Run the Equivalence Check**

Use the ``check_equivalence`` method of your connection object to run the test. This method handles circuit construction, execution, and result interpretation.

.. code-block:: python

    conn = MimiqConnection()
    conn.connect()

    # Returns the probability of the all-zero state
    # 1.0 indicates equivalence, < 1.0 indicates non-equivalence
    score = conn.check_equivalence(exp)

    if score >= 1.0:
        print("Circuits are equivalent!")
    else:
        print(f"Circuits are NOT equivalent (Score: {score})")

For partial checks on specific sub-registers, you can specify the number of input qubits ``nqinput`` when creating the experiment, though the default (all qubits of ``c1``) is usually sufficient.
