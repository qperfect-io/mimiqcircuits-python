Core Concepts
=============

At the heart of the MimiqCircuits library lie two fundamental objects: the **Circuit** and the **Hamiltonian**. Understanding these is key to constructing and manipulating quantum programs.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`cpu` Circuits
        :link: circuits
        :link-type: doc

        The :class:`~mimiqcircuits.Circuit` class is the main vessel for your quantum operations. Learn how to create, manipulate, and inspect them.

    .. grid-item-card:: :octicon:`graph` Hamiltonian
        :link: hamiltonian
        :link-type: doc

        The :class:`~mimiqcircuits.Hamiltonian` class allows you to define physical models and observables, essential for simulation and algorithms like VQE.

.. toctree::
    :maxdepth: 2
    :hidden:

    circuits
    hamiltonian
