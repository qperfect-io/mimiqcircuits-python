OpenQASM in MIMIQ-CIRC
======================

The remote MIMIQ-CIRC services can can readily process and execute OpenQASM
files, thanks to fast and feature-complete Julia and C++ parsers and
interpreters.

Here is a simple comprehensive example of executing a QASM file on MIMIQ.

.. code:: python

    from mimiqcircuits import *
    import matplotlib.pyplot as plt

    conn = MimiqConnection().connect()

    qasm = """
    // Implementation of Deutsch algorithm with two qubits for f(x)=x
    // taken from https://github.com/pnnl/QASMBench/blob/master/small/deutsch_n2/deutsch_n2.qasm
    OPENQASM 2.0;

    include "qelib1.inc";

    qreg q[2];
    creg c[2];

    x q[1];
    h q[0];
    h q[1];
    cx q[0],q[1];
    h q[0];

    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """

    # Write the OPENQASM as a file
    with open("/tmp/deutsch_n2.qasm", "w") as file:
        file.write(qasm)

    # actual execution of the QASM file
    job = conn.execute("/tmp/deutsch_n2.qasm"; algorithm="statevector")
    res = conn.get_results(job)

    plt.plot(res)


For more informations, read the documentation of :meth:`~mimiqcircuits.MimiqConnection.executeqasm`.

Behaviour of include files
--------------------------

A common file used by many QASM files is the `qelib1.inc` file.
This file is not defined as being part of OpenQASM 2.0, but its usage is so widespread that it might be considered as de-facto part of the specifications.

.. note::

    We remind the careful reader that OpenQASM 2.0 specifications only define 6
    operations: `U`,`CX`, `measure`, `reset`, `barrier` and `if`.

If we would parse every file together with `qelib1.inc`, we would have at the
end just a list of simple `U` and `CX` gates, leaving behind any speed
improvement that we would gain by using more complex gates as a block. For this
reason, if you do not provide us explicitly the include files, we would not
parse the common `qelib1.inc` but a simplified version of it, where almost all
gate definitions are replaced by `opaque` definitions. These opaque definitions
will be converted to the corresponding MIMIQ-CIRC gates listed in
:meth:`~mimiqcircuits.GATES`.

Another alternative is to use the `mimiqlib.inc` directly in your file. For now
is almost a copy of the modified `qelib1.inc` but in the future it will be
extended to contain more gates and operations, diverging from `qelib1.inc`.

Relations between OpenQASM registers and MIMIQ indices
------------------------------------------------------

During the parsing of the QASM file, we will assign a unique index to each qubit
and classical bit. This index will be used to identify the qubit or bit in the
MIMIQ-CIRC service.

The indices are assigned in the following way:

- The first qubit is assigned index `1` (Julia), the second `2`, and so on.
- All registers retain the same ordering as in the QASM file.
- Qubits and classical bits behave similarly but have they have each other its
  own sequence from indices, starting from `1`.

A simple example will clarify this behaviour:

.. code::

    OPENQASM 2.0;
    qreg q[2];
    creg m[10];
    qreg a[10];
    creg g[2];

Will be parsed as:

========= ================= ===============
QASM name MIMIQ Qubit index MIMIQ Bit index
========= ================= ===============
``q[0]``  ``0``
``q[1]``  ``1``
``a[0]``  ``2``
``a[1]``  ``3``
…         …                 …
``a[9]``  ``11``
``m[0]``                    ``0``
``m[1]``                    ``1``
…         …                 …
``m[9]``                    ``0``
``g[0]``                    ``10``
``g[1]``                    ``11``
========= ================= ===============
