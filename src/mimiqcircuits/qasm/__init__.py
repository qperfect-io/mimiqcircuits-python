"""Import and export of OpenQASM 2.0 circuits.

:func:`load` and :func:`loads` read OpenQASM 2.0 from a file or a string into
a :class:`~mimiqcircuits.Circuit`. :func:`dump` and :func:`dumps` write a
circuit as OpenQASM 2.0. Errors raised while reading or writing derive from
:class:`QASMError`, except for an operation with no OpenQASM 2.0 equivalent,
which makes the writers raise
:class:`~mimiqcircuits.decomposition.DecompositionError`.

Examples:
    >>> from mimiqcircuits import *
    >>> from mimiqcircuits import qasm
    >>> c = qasm.loads('''
    ... OPENQASM 2.0;
    ... include "qelib1.inc";
    ... qreg q[1];
    ... creg c[1];
    ... h q[0];
    ... measure q[0] -> c[0];
    ... ''')
    >>> c
    1-qubit, 1-bit circuit with 2 instructions:
    ├── H @ q[0]
    └── M @ q[0], c[0]
    <BLANKLINE>
"""

from mimiqcircuits.qasm.exceptions import (
    QASMError,
    ParseError,
    QASMSerializationError,
    QASMArgumentError,
    UndefinedGateError,
    DuplicateRegisterError,
)
from mimiqcircuits.qasm.interpreter import load, loads
from mimiqcircuits.qasm.serializer import dump, dumps

__all__ = [
    "QASMError",
    "ParseError",
    "QASMSerializationError",
    "QASMArgumentError",
    "UndefinedGateError",
    "DuplicateRegisterError",
    "load",
    "loads",
    "dump",
    "dumps",
]
