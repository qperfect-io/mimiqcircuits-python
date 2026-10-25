Manual
======

Welcome to the **MimiqCircuits** Python manual. This comprehensive guide covers everything from installation to advanced circuit compilation and remote execution.

.. grid:: 1 2 2 3
    :gutter: 2

    .. grid-item-card:: :octicon:`rocket` Getting Started
        :link: getting_started
        :link-type: doc

        Installation guide and a high-level overview of the library's capabilities.

    .. grid-item-card:: :octicon:`cpu` Core Concepts
        :link: core_concepts
        :link-type: doc

        Learn about the fundamental building blocks: :class:`~mimiqcircuits.Circuit` and :class:`~mimiqcircuits.Hamiltonian`.

    .. grid-item-card:: :octicon:`typography` Operations
        :link: operations
        :link-type: doc

        Explore the vast library of quantum gates, classical operations, noise models, and measurements.

    .. grid-item-card:: :octicon:`pulse` Simulation
        :link: simulation
        :link-type: doc

        Run exact local simulations or submit large-scale jobs to the QPerfect cloud.

    .. grid-item-card:: :octicon:`tools` Advanced Topics
        :link: advanced
        :link-type: doc

        QASM/Stim import/export, circuit compilation, and special features.

.. toctree::
    :maxdepth: 2
    :hidden:
   
    getting_started
    core_concepts
    operations
    simulation
    advanced
