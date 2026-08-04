Classical Operations
====================

.. currentmodule:: mimiqcircuits

MimiqCircuits allows you to perform classical logic and arithmetic operations directly on the classical register bits (`c`). This is essential for workflows involving feed-forward, error correction, and conditional logic.

Core Concepts
-------------

Classical operations manipulate the bits stored in the circuit's classical register. Unlike quantum gates, these operations are irreversible (except for `Not`) and act on classical data.

Most logic operations in MIMIQ follow a specific pattern:
1.  They take a variable number of input bits.
2.  They write the result to a single **destination bit**.

When adding these operations to a circuit, the **first** index you provide is always the destination bit.

Setting and Flipping Bits
-------------------------

The simplest operations involve setting a bit to a specific value or flipping its current state.

**SetBit0 and SetBit1**

These operations force a classical bit to `0` or `1`, regardless of its previous state.

.. code-block:: python

    from mimiqcircuits import *

    c = Circuit()
    
    # Set bit 0 to 0
    c.push(SetBit0(), 0)

    # Set bit 1 to 1
    c.push(SetBit1(), 1)


**Not (Invert)**

The :class:`~mimiqcircuits.Not` operation flips the value of a bit (`0` becomes `1`, `1` becomes `0`).

.. code-block:: python

    # Flip bit 2
    c.push(Not(), 2)


Boolean Logic
-------------

MIMIQ supports standard boolean logic gates. These operations can take multiple input bits and store the result in a target bit.

**AND Operation**

The :class:`~mimiqcircuits.And` operation computes the logical AND of two or more inputs.

.. code-block:: python

    # c[0] = c[1] AND c[2]
    # First index (0) is the target.
    c.push(And(), 0, 1, 2)

    # Multiple inputs: c[0] = c[1] AND c[2] AND c[3] AND c[4]
    # We must specify the total number of bits involved (1 target + 4 inputs = 5)
    c.push(And(5), 0, 1, 2, 3, 4)


**OR Operation**

The :class:`~mimiqcircuits.Or` operation computes the logical OR of two or more inputs.

.. code-block:: python

    # c[5] = c[6] OR c[7]
    c.push(Or(), 5, 6, 7)


**XOR Operation**

The :class:`~mimiqcircuits.Xor` operation computes the exclusive OR (XOR) of the inputs.

.. code-block:: python

    # c[0] = c[1] XOR c[2]
    c.push(Xor(), 0, 1, 2)


Arithmetic
----------

**Parity Check**

The :class:`~mimiqcircuits.ParityCheck` operation computes the sum of the input bits modulo 2. While functionally similar to XOR, it is often used conceptually for syndrome measurements in error correction.

.. code-block:: python

    # c[0] = (c[1] + c[2]) mod 2
    c.push(ParityCheck(), 0, 1, 2)

    # Multi-bit parity check
    c.push(ParityCheck(5), 0, 1, 2, 3, 4)

Reference
---------

.. autoclass:: mimiqcircuits.SetBit0
    :noindex:
.. autoclass:: mimiqcircuits.SetBit1
    :noindex:
.. autoclass:: mimiqcircuits.Not
    :noindex:
.. autoclass:: mimiqcircuits.And
    :noindex:
.. autoclass:: mimiqcircuits.Or
    :noindex:
.. autoclass:: mimiqcircuits.Xor
    :noindex:
.. autoclass:: mimiqcircuits.ParityCheck
    :noindex:
