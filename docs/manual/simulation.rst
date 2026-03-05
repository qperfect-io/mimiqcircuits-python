Simulation & Execution
======================

Once you have built your circuit, you need to execute it. MimiqCircuits offers two primary paths: running small simulations locally on your machine or submitting large-scale jobs to the QPerfect cloud.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`home` Local Simulation
        :link: local_statevector
        :link-type: doc

        Perform exact statevector simulations directly on your local machine using the :class:`~mimiqcircuits.quantanium.Quantanium` backend. Ideal for debugging and small circuits.

    .. grid-item-card:: :octicon:`cloud` Remote Execution
        :link: remote_execution
        :link-type: doc

        Connect to the QPerfect cloud via :class:`~mimiqcircuits.MimiqConnection` to access high-performance backends like MIMIQ-CIRC (Tensor Network types) and MIMIQ-STIM (Clifford).

.. _understanding-sampling:

Understanding Sampling
----------------------

When measuring a quantum circuit, the result is probabilistic. To obtain an estimate of the output distribution, the circuit must be executed multiple times (sampled).

For unitary circuits (gates, no intermediate measurements), the final statevector can be computed once and then sampled efficiently. However, if the circuit contains **non-unitary operations** (like intermediate measurements or reset operations) that affect the subsequent path of execution, the simulation state might change in each shot. This typically requires re-simulating the circuit for every sample, which can be computationally more expensive than sampling from a final statevector.

.. toctree::
    :maxdepth: 2
    :hidden:

    local_statevector
    remote_execution
    simulation_parameters
