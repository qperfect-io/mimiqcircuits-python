Variational Optimization
====================================

MIMIQ includes a lightweight workflow to optimize symbolic circuits against
a scalar cost accumulated in the Z-register (e.g., the expectation value of a Hamiltonian).
This enables a broad class of Variational Quantum Algorithms (VQAs), including
the Variational Quantum Eigensolver (VQE) and related variational tasks.

Concepts
-----------------------
.. _optimization-concepts:


**Variational Quantum Algorithms (VQAs)** are hybrid quantum–classical methods that
optimize a parameterized quantum circuit against a classical cost function. Different
choices of cost define different algorithms (e.g., **QAOA**, variational classifiers), and
**VQE** is one such example.

**Variational Quantum Eigensolver (VQE)** approximates the ground-state energy of a
Hamiltonian :math:`H` and is a specific instance of the broader VQA family.

The method relies on:

- A **parameterized ansatz** :math:`|\psi(\vec{\theta})\rangle` prepared by a quantum circuit
- A **cost function** given by the energy expectation value

.. math::

   E(\vec{\theta}) \;=\;
   \langle \psi(\vec{\theta}) \,|\, H \,|\, \psi(\vec{\theta}) \rangle

The parameters :math:`\vec{\theta}` are updated by a **classical optimizer** to minimize
:math:`E(\vec{\theta})`. At convergence, the variational principle guarantees

.. math::

   E(\vec{\theta}^*) \;\geq\; E_0

where :math:`E_0` is the true ground-state energy of :math:`H`.


Quick Start: VQE in MIMIQ
-------------------------

.. _optimization-ising-example:

We use the **1D Ising Hamiltonian** and define a simple RX/RY ansatz.  
Then we define the cost as the **energy** :math:`\langle H \rangle` and optimize it.


**Build the Ising Hamiltonian**

.. doctest::

    >>> from mimiqcircuits import *
    >>> from symengine import *
    >>> N = 4; J = 1.0; h = 0.5
    >>> H = Hamiltonian()
    >>> for j in range(N - 1):
    ...     _ = H.push(-J, PauliString("ZZ"), j, j + 1)
    >>> for j in range(N):
    ...     _ = H.push(-h, PauliString("X"), j)
    >>> H
    4-qubit Hamiltonian with 7 terms:
    ├── -1.0 * ZZ @ q[0,1]
    ├── -1.0 * ZZ @ q[1,2]
    ├── -1.0 * ZZ @ q[2,3]
    ├── -0.5 * X @ q[0]
    ├── -0.5 * X @ q[1]
    ├── -0.5 * X @ q[2]
    └── -0.5 * X @ q[3]

**Define a symbolic ansatz** :math:`|\psi(\theta)\rangle`

.. doctest::

    >>> x0, x1, x2, x3, y0, y1, y2, y3 = symbols("x0 x1 x2 x3 y0 y1 y2 y3")
    >>> c = Circuit()
    >>> for q, (x, y) in enumerate([(x0,y0),(x1,y1),(x2,y2),(x3,y3)]):
    ...     _ = c.push(GateRX(x), q)
    ...     _ = c.push(GateRY(y), q)

**Append the cost (accumulate ⟨H⟩ into z[0])**

.. doctest::

    >>> c.push_expval(H, *range(N))
    4-qubit, 7-zvar circuit with 23 instructions:
    ├── RX(x0) @ q[0]
    ├── RY(y0) @ q[0]
    ├── RX(x1) @ q[1]
    ├── RY(y1) @ q[1]
    ├── RX(x2) @ q[2]
    ├── RY(y2) @ q[2]
    ├── RX(x3) @ q[3]
    ├── RY(y3) @ q[3]
    ├── ⟨ZZ⟩ @ q[0,1], z[0]
    ├── z[0] *= -1.0
    ├── ⟨ZZ⟩ @ q[1,2], z[1]
    ├── z[1] *= -1.0
    ├── ⟨ZZ⟩ @ q[2,3], z[2]
    ├── z[2] *= -1.0
    ├── ⟨X⟩ @ q[0], z[3]
    ├── z[3] *= -0.5
    ├── ⟨X⟩ @ q[1], z[4]
    ├── z[4] *= -0.5
    ├── ⟨X⟩ @ q[2], z[5]
    ⋮   ⋮
    └── z[0] += 0.0 + z[1] + z[2] + z[3] + z[4] + z[5] + z[6]
    <BLANKLINE>

**Create the OptimizationExperiment**

An :class:`~mimiqcircuits.OptimizationExperiment` specifies:

- the **symbolic quantum circuit**,
- the **initial parameter values**,
- the **classical optimizer** to use,
- and additional **experiment settings** (such as label, maximum iterations, and
  the Z-register index where the cost is accumulated).

.. doctest::

    >>> init = {x0: 0.0, y0: 0.0, x1: 0.0, y1: 0.0, x2: 0.0, y2: 0.0, x3: 0.0, y3: 0.0}
    >>> exp = OptimizationExperiment(circuit=c, initparams=init,
    ...                              optimizer="COBYLA", maxiters=50, zregister=0)
    >>> exp
    OptimizationExperiment:
    ├── optimizer: COBYLA
    ├── label: optexp_python
    ├── maxiters: 50
    ├── zregister: 0
    └── initparams:
        ├── x0 => 0.0
        ├── y0 => 0.0
        ├── x1 => 0.0
        ├── y1 => 0.0
        ├── x2 => 0.0
        ├── y2 => 0.0
        ├── x3 => 0.0
        └── y3 => 0.0
   
   

Running Optimization on Cloud
-----------------------------

.. _optimization-cloud:

You can submit a single :class:`~mimiqcircuits.OptimizationExperiment` or a list of them
to the MIMIQ Cloud. The return type depends on the ``history`` flag:

- If ``history=True`` you will receive an :class:`~mimiqcircuits.OptimizationResults`,
  which contains the best run and the full history of runs.
- If ``history=False`` (default), you will receive only the best
  :class:`~mimiqcircuits.OptimizationRun`.

An :class:`~mimiqcircuits.OptimizationRun` represents a **single evaluation of the cost
function** during optimization. It contains:

- the **final cost value** for the given parameters,
- the **parameter values** used in that evaluation,
- and the **raw execution results** (:class:`~mimiqcircuits.QCSResults`) from the quantum
  simulation or hardware.

An :class:`~mimiqcircuits.OptimizationResults` collects all runs into a history and
tracks the **best run** found during the optimization.




The snippet below **assumes** you already constructed ``exp`` as in the Ising
example above.

**Connect to the cloud**

.. code-block:: python

    conn = MimiqConnection()
    conn.connect()

**Submit the optimization job (choose backend/algorithm)**

.. code-block:: python

    job = conn.optimize(exp, algorithm="mps", history=True, label="ising_vqe")

**Retrieve results (blocks until the job finishes)**

.. code-block:: python

    optres = conn.get_result(job)   # use get_results if submitting a batch

**Inspect best run and history**

.. code-block:: python

    best = optres.get_best()
    print(best)

**Optional. Access raw results objects for each evaluation**

.. code-block:: python

    history_results = optres.get_resultsofhistory()
    print(len(history_results))

