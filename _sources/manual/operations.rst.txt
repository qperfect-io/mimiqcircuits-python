Operations
==========

MimiqCircuits provides a rich set of operations to build your quantum circuits. From standard unitary gates to advanced noise channels and classical control flow, explore the available tools below.

.. grid:: 1 2 2 3
    :gutter: 3

    .. grid-item-card:: :octicon:`circle` Unitary Gates
        :link: unitary_gates
        :link-type: doc

        Standard single and multi-qubit gates (X, H, CX, etc.).

    .. grid-item-card:: :octicon:`code` Classical Operations
        :link: classical_ops
        :link-type: doc

        Manipulate classical bits logic (AND, XOR) and registers.

    .. grid-item-card:: :octicon:`stop` Non-Unitary Ops
        :link: non_unitary_ops
        :link-type: doc

        Measurements, resets, and conditional logic.

    .. grid-item-card:: :octicon:`pulse` Noise Models
        :link: noise
        :link-type: doc

        Define custom noise models and apply them to your circuits.

    .. grid-item-card:: :octicon:`pencil` Symbolic Ops
        :link: symbolic_ops
        :link-type: doc

        Work with parameterized circuits using symbolic variables.

    .. grid-item-card:: :octicon:`graph` Statistical Ops
        :link: statistical_ops
        :link-type: doc

        Operations for extracting statistical properties.

    .. grid-item-card:: :octicon:`star` Special Ops
        :link: special_ops
        :link-type: doc

        Specialized operations and structures.

    .. grid-item-card:: :octicon:`plus` Z-Operations
        :link: zops
        :link-type: doc

        Operations on the Z-register.

.. toctree::
    :maxdepth: 2
    :hidden:

    unitary_gates
    classical_ops
    non_unitary_ops
    noise
    symbolic_ops
    statistical_ops
    special_ops
    zops

Reference
---------

.. autoclass:: mimiqcircuits.Operation
    :noindex:
.. autoclass:: mimiqcircuits.Gate
    :noindex:
