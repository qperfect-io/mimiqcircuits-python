Special Topics
==============

This page provides detailed information on specialized functionalities in MIMIQ.

Contents
========
.. contents::
   :local:
   :depth: 2
   :backlinks: entry

BitString
---------
.. _bitstring:

The :class:`~mimiqcircuits.BitString` class represents the state of bits and can be used 
to represent classical registers with specified values for each bit (0 or 1). At its core, 
it is simply a vector of booleans. :class:`~mimiqcircuits.BitString`  allows direct bit manipulation, bitwise operations, and conversion to other data formats like integers. It’s designed for flexibility in binary manipulation tasks within quantum computations.

Using BitString in MIMIQ Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In MIMIQ, several operations use BitString as a direct input for conditional logic 
or specific quantum operations, such as :class:`~mimiqcircuits.IfStatement` 
and :class:`~mimiqcircuits.Amplitude`. See :doc:`non-unitary operations </manual/non_unitary_ops>` 
and :doc:`statistical operations </manual/statistical_ops>` pages. Here are some examples:

.. doctest:: BitString
    :hide:

    >>> from mimiqcircuits import *

.. doctest:: BitString

    >>> if_statement = IfStatement(GateX(), BitString("01011"))
    >>> if_statement
    IF (c==01011) X

    # Amplitude Operation
    >>> Amplitude(BitString("001"))
    Amplitude(bs"001")

Constructors
~~~~~~~~~~~~
:class:`~mimiqcircuits.BitString` objects can be constructed in different ways.

- **From a String:** Use `BitString("binary_string")` to initialize a BitString by parsing a string in binary format.

  .. doctest:: BitString

      >>> BitString("1101")
      bs"1101"


- **From bit locations:** Use `BitString(numbits, bit_indices)` to initialize a BitString of `numbits`, setting specific bits to 1.

  .. doctest:: BitString

      # Initializing with Specific Bits
      >>> BitString(8, [2, 4, 6])
      bs"00101010"

- **From a function:** Use `BitString(f, numbits)` to initialize a BitString with `numbits`, where each bit is set based on the function `f`.

  .. doctest:: BitString

      # Initialize an 8-bit BitString where bits are set based on even indices
      >>> BitString.fromfunction(8, lambda i: i % 2 == 0)
      bs"10101010"

Accessing and Modifying Bits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each bit in a `BitString` can be accessed or modified individually in the same way as vectors, 
making it easy to retrieve or set specific bit values.

.. doctest:: BitString

    # Accessing a Bit
    >>> bs = BitString(4, [1, 3])
    >>> bs
    bs"0101"
    >>> bs[2]
    0

    >>> # Modifying a Bit
    >>> bs[2] = True
    >>> bs
    bs"0111"

A useful function is :func:`~mimiqcircuits.nonzeros`, which returns the indices of non-zero bits in a BitString.

.. doctest:: BitString

    >>> bs = BitString(6, [1, 3, 5])
    >>> bs.nonzeros()
    [1, 3, 5]

Conversion and Manipulation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`~mimiqcircuits.BitString` class includes functionality for 
conversion to integer representations, indexing, and other methods for 
retrieving and manipulating bit values:

- **BitString to Integer:** Use :meth:`~mimiqcircuits.BitString.tointeger` to convert a `BitString` into its integer representation, 

By default it uses a big-endian order.

  .. doctest:: BitString

      >>> bs = BitString("101010")
      >>> bs
      bs"101010"

      # Convert BitString to Integer (big-endian by default)
      >>> bs.tointeger()
      21

      # Convert BitString to Integer (little-endian)
      >>> bs.tointeger(endianess='little')
      42

Alternatively, you can use the function :meth:`~mimiqcircuits.BitString.toindex`, which converts a `BitString` to an index for 
purposes like vector indexing, checking bounds, and compatibility with 64-bit indexing constraints. It's 
essentially the same as :meth:`~mimiqcircuits.BitString.tointeger`.

  .. doctest:: BitString

      >>> bs.toindex()
      21

- **BitString to String:** Use :func:`~mimiqcircuits.to01`.

  .. doctest:: BitString

      # Convert BitString to String of "0"s and "1"s (big-endian (default))
      >>> bs.to01()
      '101010'

      # Convert BitString to String of "0"s and "1"s (little-endian)
      >>> bs.to01(endianess='little')
      '010101'

   

Bitwise Operators
~~~~~~~~~~~~~~~~~
:class:`~mimiqcircuits.BitString` supports bitwise operations such as NOT, AND, OR, XOR, 
as well as bitwise shifts:

- **NOT (~):**

  .. doctest:: BitString

      >>> bs = BitString("1011")
      >>> ~bs
      bs"0100"

- **AND (&), OR (|):**

  .. doctest:: BitString

      >>> bs1 = BitString("1100")
      >>> bs2 = BitString("0110")
      
      # Bitwise AND
      >>> bs1 & bs2
      bs"0100"
      
      # Bitwise OR
      >>> bs1 | bs2
      bs"1110"

- **XOR (^):**

  .. doctest:: BitString

      # Bitwise XOR
      >>> bs1 ^ bs2
      bs"1010"

- **Left Shift (<<) and Right Shift (>>):**

  .. doctest:: BitString

      # Left Shift
      >>> bs << 1
      bs"0110"

      # Right Shift
      >>> bs >> 1 
      bs"0101"

Concatenation and Repetition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BitString supports concatenation and repetition, allowing you to 
combine or extend bitstrings efficiently:

- **Concatenation (+)** Use `+` perator to combines two `BitString` objects by appending the bits of `rhs` to `lhs`.

  .. doctest:: BitString

      >>> bs1 = BitString("1010")
      >>> bs2 = BitString("0101")
      >>> bs1 + bs2
      bs"10100101"

- **Repetition (*)** Use `*` perator to repeat a `BitString` to a specified number of times, creating a new `BitString` with the pattern repeated.

  .. doctest:: BitString

      >>> bs = BitString("1010")
      >>> bs * 2
      bs"10101010"
