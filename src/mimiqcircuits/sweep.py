#
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Parameter sweeps: :class:`ParameterTable`, :func:`gridtable` and
:class:`SweepExperiment`."""

from functools import reduce

import numpy as np
import symengine as se


def _as_symbol(s):
    if isinstance(s, str):
        return se.Symbol(s)
    s = se.sympify(s)
    if not isinstance(s, se.Symbol):
        raise ValueError(f"`{s}` is not a plain symbolic variable")
    return s


class ParameterTable:
    r"""Values for the symbolic parameters of a circuit, one row per point of
    a sweep.

    In a sweep every point binds the same symbols, so a table stores them once,
    in a fixed order, together with a matrix of values: row ``i`` holds the
    values of point ``i``, and column ``j`` the values of ``symbols[j]``.

    ``ParameterTable(symbols, values)`` takes the symbols and an
    ``(npoints, nsymbols)`` array. :meth:`from_columns` builds a table from one
    column per symbol, :meth:`from_dicts` from a list of dictionaries (the
    shape of ``param_grid``), and :func:`gridtable` from every combination of
    the given values.

    Every symbol must be a plain variable, such as one created with
    ``symengine.symbols``, or its name as a string, and appear only once.
    Values are stored as ``float``.

    Tables compose: ``a * b`` is the Cartesian product, ``a.hcat(b)`` pairs the
    points of two tables of the same length, and ``a.vcat(b)`` appends the
    points of ``b`` after those of ``a``.

    ``len(t)`` is the number of points, ``t[i]`` is point ``i`` as a
    dictionary ready for :meth:`~mimiqcircuits.Circuit.evaluate`, and
    ``t[i:j]`` or ``t[[i, j]]`` is the table of the selected points;
    :meth:`listvars` gives the symbols and :meth:`to_numpy` the values.
    Iterating over a table gives its points.

    Examples:
        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> theta, phi = symbols("theta phi")
        >>> t = ParameterTable.from_columns({theta: [0.0, 0.5, 1.0], phi: [0.0, 0.25, 0.5]})
        >>> t
        ParameterTable with 3 points over (theta, phi):
        ├── theta = 0.0, phi = 0.0
        ├── theta = 0.5, phi = 0.25
        └── theta = 1.0, phi = 0.5
        >>> t[1][phi]
        0.25
        >>> ParameterTable.from_columns({theta: [0.0, 1.0]}) * ParameterTable.from_columns({phi: [0.0, 0.5]})
        ParameterTable with 4 points over (theta, phi):
        ├── theta = 0.0, phi = 0.0
        ├── theta = 0.0, phi = 0.5
        ├── theta = 1.0, phi = 0.0
        └── theta = 1.0, phi = 0.5
    """

    __slots__ = ("_symbols", "_values")

    def __init__(self, symbols, values):
        syms = tuple(_as_symbol(s) for s in symbols)
        vals = np.array(values, dtype=float)
        if vals.ndim != 2 or vals.shape[1] != len(syms):
            raise ValueError(
                f"the table has {len(syms)} symbols, so values must have shape "
                f"(npoints, {len(syms)}), not {vals.shape}"
            )
        if len(set(syms)) != len(syms):
            raise ValueError("every symbol must appear only once in a table")
        vals.flags.writeable = False
        self._symbols = syms
        self._values = vals

    @classmethod
    def from_columns(cls, columns):
        """Build a table from one column of values per symbol.

        ``columns`` maps each symbol to its values, such as
        ``{theta: xs, phi: ys}``. All the columns must have the same length,
        and point ``i`` takes the ``i``-th value of each. For every combination
        of the values instead, use :func:`gridtable`.
        """
        columns = list(dict(columns).items())
        if not columns:
            raise ValueError("a table needs at least one column")
        n = len(columns[0][1])
        if any(len(v) != n for _, v in columns):
            raise ValueError(
                "all columns must have the same length; use `gridtable` for "
                "every combination of values"
            )
        values = np.column_stack([np.asarray(v, dtype=float) for _, v in columns])
        return cls([s for s, _ in columns], values)

    @classmethod
    def from_dicts(cls, points):
        """Build a table from a list of dictionaries, one per point.

        Every dictionary must bind the same symbols. Symbols are ordered by
        name, since dictionaries give no order of their own.
        """
        converted = []
        for i, p in enumerate(points):
            q = {_as_symbol(k): v for k, v in p.items()}
            # `{theta: 1, "theta": 2}` names one symbol twice.
            if len(q) != len(p):
                raise ValueError(f"point {i} gives a symbol twice")
            converted.append(q)
        points = converted
        if not points:
            raise ValueError("a table needs at least one point")
        syms = sorted(points[0], key=str)
        for i, p in enumerate(points):
            if set(p) != set(syms):
                raise ValueError(f"point {i} binds different symbols than point 0")
        return cls(syms, [[float(p[s]) for s in syms] for p in points])

    def __len__(self):
        return self._values.shape[0]

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return dict(zip(self._symbols, self._values[i].tolist()))
        return ParameterTable(self._symbols, self._values[i])

    def __eq__(self, other):
        if not isinstance(other, ParameterTable):
            return NotImplemented
        return self._symbols == other._symbols and np.array_equal(
            self._values, other._values
        )

    def __hash__(self):
        return hash((self._symbols, tuple(self._values.ravel().tolist())))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def listvars(self):
        """The symbols of the table, in column order."""
        return list(self._symbols)

    def to_numpy(self):
        """A copy of the values, as an ``(npoints, nsymbols)`` array."""
        return self._values.copy()

    def _check_disjoint(self, other):
        shared = [s for s in self._symbols if s in other._symbols]
        if shared:
            raise ValueError(
                f"both tables give values for {', '.join(map(str, shared))}"
            )

    def __mul__(self, other):
        """The Cartesian product, with the points of ``self`` varying slowest."""
        if not isinstance(other, ParameterTable):
            return NotImplemented
        self._check_disjoint(other)
        na, nb = len(self), len(other)
        values = np.hstack(
            [np.repeat(self._values, nb, axis=0), np.tile(other._values, (na, 1))]
        )
        return ParameterTable(self._symbols + other._symbols, values)

    def hcat(self, *others):
        """Pair the points of tables of the same length, side by side."""
        out = self
        for other in others:
            out._check_disjoint(other)
            if len(out) != len(other):
                raise ValueError(
                    f"cannot pair the points of tables with {len(out)} and "
                    f"{len(other)} points"
                )
            out = ParameterTable(
                out._symbols + other._symbols, np.hstack([out._values, other._values])
            )
        return out

    def vcat(self, *others):
        """Append the points of tables over the same symbols, in any order."""
        out = self
        for other in others:
            if set(out._symbols) != set(other._symbols):
                raise ValueError(
                    "only tables over the same symbols can be concatenated"
                )
            # Lay `other` out like `out`.
            perm = [other._symbols.index(s) for s in out._symbols]
            out = ParameterTable(
                out._symbols, np.vstack([out._values, other._values[:, perm]])
            )
        return out

    def _format_point(self, i):
        return ", ".join(
            f"{s} = {v}" for s, v in zip(self._symbols, self._values[i].tolist())
        )

    def _short(self):
        return f"ParameterTable({len(self)} points over ({', '.join(map(str, self._symbols))}))"

    def __repr__(self):
        n = len(self)
        head = (
            f"ParameterTable with {n} point{'' if n == 1 else 's'} over "
            f"({', '.join(map(str, self._symbols))}):"
        )
        # Long tables show their first and last points only.
        shown = list(range(n)) if n <= 10 else list(range(5)) + list(range(n - 5, n))
        lines = [head]
        for k, i in enumerate(shown):
            if n > 10 and k == 5:
                lines.append("├── ⋮")
            prefix = "└── " if k == len(shown) - 1 else "├── "
            lines.append(prefix + self._format_point(i))
        return "\n".join(lines)

    __str__ = __repr__


def gridtable(columns):
    """Build the :class:`ParameterTable` of every combination of the given
    values, the Cartesian product of the columns.

    ``columns`` maps each symbol to its values. The first symbol varies
    slowest and the last one fastest, so
    ``gridtable({theta: [0, 1], phi: [0, 2, 4]})`` has 6 points: ``theta = 0``
    with each ``phi``, then ``theta = 1`` with each ``phi``. It is the same as
    multiplying the one-column tables with ``*``.

    Examples:
        >>> from symengine import symbols
        >>> from mimiqcircuits import *
        >>> theta, phi = symbols("theta phi")
        >>> gridtable({theta: [0.0, 1.0], phi: [0.0, 2.0, 4.0]})
        ParameterTable with 6 points over (theta, phi):
        ├── theta = 0.0, phi = 0.0
        ├── theta = 0.0, phi = 2.0
        ├── theta = 0.0, phi = 4.0
        ├── theta = 1.0, phi = 0.0
        ├── theta = 1.0, phi = 2.0
        └── theta = 1.0, phi = 4.0
    """
    columns = dict(columns)
    if not columns:
        raise ValueError("a table needs at least one column")
    return reduce(
        lambda a, b: a * b,
        (ParameterTable.from_columns({s: v}) for s, v in columns.items()),
    )


class SweepExperiment:
    r"""A parameter sweep: a symbolic ``circuit`` and the
    :class:`ParameterTable` of values to run it with, one execution per point.

    The table must give values for exactly the free symbols of the circuit, no
    more and no fewer, so a missing or misspelled symbol is caught when the
    experiment is built rather than when the sweep runs.

    ``len(ex)`` is the number of points, ``ex[i]`` is the circuit evaluated at
    point ``i``, ``ex[i:j]`` or ``ex[[i, j]]`` is the experiment over the
    selected points, and
    iterating over the experiment gives the evaluated circuits
    in order. Executing it on a local backend returns one
    :class:`~mimiqcircuits.QCSResults` per point.

    Examples:
        >>> from symengine import symbols, pi
        >>> from mimiqcircuits import *
        >>> theta = symbols("theta")
        >>> c = Circuit().push(GateRX(theta), 0)
        >>> ex = SweepExperiment(c, ParameterTable.from_columns({theta: [0.0, 0.5, 1.0]}))
        >>> ex
        SweepExperiment:
        ├── circuit: 1-qubit circuit with 1 instruction
        └── table: ParameterTable(3 points over (theta))
        >>> ex[1]
        1-qubit circuit with 1 instruction:
        └── RX(0.5) @ q[0]
        <BLANKLINE>
    """

    __slots__ = ("_circuit", "_table")

    def __init__(self, circuit, table):
        if not isinstance(table, ParameterTable):
            raise TypeError(
                f"table must be a ParameterTable, not {type(table).__name__}"
            )
        if not circuit.is_symbolic():
            raise ValueError("the circuit of a sweep must be symbolic")
        cvars = {str(s) for s in circuit.listvars()}
        tvars = {str(s) for s in table.listvars()}
        missing = sorted(cvars - tvars)
        if missing:
            raise ValueError(f"the table gives no values for {', '.join(missing)}")
        extra = sorted(tvars - cvars)
        if extra:
            raise ValueError(
                f"the table gives values for {', '.join(extra)}, which the "
                "circuit does not use"
            )
        self._circuit = circuit
        self._table = table

    @property
    def circuit(self):
        """The symbolic circuit."""
        return self._circuit

    @property
    def table(self):
        """The :class:`ParameterTable` of values."""
        return self._table

    def __len__(self):
        return len(self._table)

    def __getitem__(self, i):
        point = self._table[i]
        if isinstance(point, ParameterTable):
            return SweepExperiment(self._circuit, point)
        return self._circuit.evaluate(point)

    # The circuit is mutable, so an experiment is left unhashable.
    def __eq__(self, other):
        if not isinstance(other, SweepExperiment):
            return NotImplemented
        return self._circuit == other._circuit and self._table == other._table

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def listvars(self):
        """The symbols of the sweep, in the table's column order."""
        return self._table.listvars()

    def num_qubits(self):
        return self._circuit.num_qubits()

    def num_bits(self):
        return self._circuit.num_bits()

    def num_zvars(self):
        return self._circuit.num_zvars()

    def __repr__(self):
        header = str(self._circuit).split("\n", 1)[0].rstrip(":")
        return "\n".join(
            [
                "SweepExperiment:",
                f"├── circuit: {header}",
                f"└── table: {self._table._short()}",
            ]
        )

    __str__ = __repr__


__all__ = ["ParameterTable", "gridtable", "SweepExperiment"]
