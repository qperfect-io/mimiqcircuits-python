#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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
"""Directed-acyclic-graph view of a :class:`Circuit`.

A circuit is stored as a flat list of instructions, but its true ordering
constraints form a partial order: two instructions must keep their relative
order only when they share a wire (a qubit, a classical bit, or a
z-variable). :class:`CircuitDAG` captures that partial order as a dependency
graph, which lets callers iterate a circuit in orders other than the literal
insertion order — for example grouping mutually independent gates into layers.

The vertices are instruction positions and an edge ``u -> v`` records that
``v`` depends on the most recent earlier instruction ``u`` acting on a shared
wire.
"""

from __future__ import annotations


class CircuitDAG:
    """Dependency graph of a circuit's instructions.

    Vertices are the integer positions ``0 .. n-1`` of the instructions in the
    circuit. An edge ``u -> v`` means instruction ``v`` depends on instruction
    ``u``: they touch a common qubit, bit, or z-variable and ``u`` is the most
    recent earlier instruction on that wire. Only the immediate predecessor on
    each wire is linked, so the graph is the transitive reduction along wires
    rather than a dense reachability graph.

    The graph is normally obtained from :meth:`Circuit.dag`, which builds it
    on first use and caches it. ``CircuitDAG(n)`` creates a graph with ``n``
    vertices and no edges.

    Args:
        n (int): The number of vertices.

    Raises:
        TypeError: If ``n`` is not an integer, for example a circuit.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateH(), 1)
        >>> _ = c.push(GateCX(), 0, 1)
        >>> _ = c.push(GateX(), 0)
        >>> dag = c.dag()
        >>> dag
        CircuitDAG(vertices=4, edges=3)
        >>> # the CX (instruction 2) waits for both H gates
        >>> dag.in_neighbors(2)
        [0, 1]
        >>> # the X on qubit 0 only depends on the CX
        >>> dag.in_neighbors(3)
        [2]
    """

    def __init__(self, n):
        if not isinstance(n, int):
            raise TypeError(
                f"CircuitDAG(n) takes a vertex count (int), got "
                f"{type(n).__name__}; to build the DAG of a circuit use "
                "circuit.dag()."
            )
        self._n = n
        # Adjacency lists are kept sorted and duplicate-free so that traversals
        # are deterministic and in-degree counts match the number of distinct
        # predecessors (Kahn's algorithm relies on the latter).
        self._out = [[] for _ in range(n)]
        self._in = [[] for _ in range(n)]

    def __repr__(self):
        return f"CircuitDAG(vertices={self._n}, edges={self.num_edges()})"

    def num_vertices(self):
        """Return the number of vertices, that is the number of instructions."""
        return self._n

    def num_edges(self):
        """Return the number of dependency edges."""
        return sum(len(o) for o in self._out)

    def vertices(self):
        """Return the vertex indices, as ``range(num_vertices())``."""
        return range(self._n)

    def edges(self):
        """Return the list of edges ``(u, v)``, with ``u`` before ``v`` in the circuit."""
        return [(u, v) for u in range(self._n) for v in self._out[u]]

    def out_neighbors(self, v):
        """Return the vertices that depend directly on ``v`` (its successors), sorted."""
        return self._out[v]

    def in_neighbors(self, v):
        """Return the vertices that ``v`` depends on directly (its predecessors), sorted."""
        return self._in[v]

    def in_degree(self, v):
        """Return the number of direct predecessors of ``v``."""
        return len(self._in[v])

    def out_degree(self, v):
        """Return the number of direct successors of ``v``."""
        return len(self._out[v])

    def has_vertex(self, v):
        """Return ``True`` if ``v`` is a vertex index of the graph."""
        return 0 <= v < self._n

    def has_edge(self, u, v):
        """Return ``True`` if the graph has the edge ``u -> v``."""
        return v in self._out[u]

    def _add_edge(self, u, v):
        # During a build all edges added for a given target ``v`` are emitted
        # consecutively and ``v`` only ever grows, so a single tail check keeps
        # each adjacency list sorted and free of the duplicate that arises when
        # two wires share the same immediate predecessor.
        if self._out[u] and self._out[u][-1] == v:
            return
        self._out[u].append(v)
        self._in[v].append(u)


def _dag_qubits(inst, nq):
    """Qubits ``inst`` depends on for ordering purposes.

    Most operations touch only the qubits they are applied to. The exceptions
    are the global state observables, whose value depends on the whole circuit
    history rather than the wires they declare:

    - ``Amplitude`` reads ``⟨bs|ψ⟩`` over the entire register and declares no
      qubit at all;
    - ``BondDim``, ``SchmidtRank``, and ``VonNeumannEntropy`` declare only the
      bond they probe, but the entanglement across that cut is set by gates on
      either side.

    For dependency purposes each is treated as acting on every qubit, so it
    follows all earlier gates and precedes all later ones — a full-register
    synchronisation point.
    """
    from mimiqcircuits.operations.amplitude import Amplitude
    from mimiqcircuits.operations.entanglement import (
        BondDim,
        SchmidtRank,
        VonNeumannEntropy,
    )

    if isinstance(
        inst.operation, (Amplitude, BondDim, SchmidtRank, VonNeumannEntropy)
    ):
        return range(nq)
    return inst.qubits


def build_dag(circuit):
    """Build the :class:`CircuitDAG` of ``circuit``.

    Each instruction is linked to the last earlier instruction on each of its
    qubits, bits and z-variables. The result holds exactly the orderings that
    must be kept for the circuit to stay equivalent. :class:`Amplitude`,
    :class:`BondDim`, :class:`SchmidtRank` and :class:`VonNeumannEntropy`
    depend on the whole state, so they are linked as if they acted on every
    qubit.

    :meth:`Circuit.dag` calls this and caches the result; use that instead.

    Returns:
        CircuitDAG: The dependency graph.
    """
    n = len(circuit)
    nq = circuit.num_qubits()
    dag = CircuitDAG(n)

    last_q = {}
    last_b = {}
    last_z = {}

    for i, inst in enumerate(circuit):
        for q in _dag_qubits(inst, nq):
            prev = last_q.get(q)
            if prev is not None:
                dag._add_edge(prev, i)
            last_q[q] = i
        for b in inst.bits:
            prev = last_b.get(b)
            if prev is not None:
                dag._add_edge(prev, i)
            last_b[b] = i
        for z in inst.zvars:
            prev = last_z.get(z)
            if prev is not None:
                dag._add_edge(prev, i)
            last_z[z] = i

    # Predecessors may be discovered out of order (a later wire can point back
    # to an earlier instruction); sort so in-neighbor lists are deterministic.
    for preds in dag._in:
        preds.sort()

    return dag


def _require_dag(dag):
    if not isinstance(dag, CircuitDAG):
        raise TypeError(
            f"expected a CircuitDAG, got {type(dag).__name__}; "
            "build one with circuit.dag()."
        )


def topological_sort_by_bfs(dag):
    """Return the vertices of ``dag`` in breadth-first topological order.

    The order is built layer by layer with Kahn's algorithm: first every
    vertex with no predecessor, then every vertex whose predecessors are all
    in the first layer, and so on. The first layer comes in increasing index
    order, and each later vertex is added when its last predecessor is
    visited, so the result is deterministic.

    Args:
        dag (CircuitDAG): The graph, usually from :meth:`Circuit.dag`.

    Returns:
        list of int: Every vertex once, each after all its predecessors.

    Raises:
        TypeError: If ``dag`` is not a :class:`CircuitDAG`.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateX(), 0)
        >>> _ = c.push(GateH(), 1)
        >>> # the two H gates form the first layer, the X the second
        >>> topological_sort_by_bfs(c.dag())
        [0, 2, 1]
    """
    _require_dag(dag)
    n = dag.num_vertices()
    in_degree = [dag.in_degree(v) for v in range(n)]

    queue = [v for v in range(n) if in_degree[v] == 0]

    order = []
    idx = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        order.append(u)
        for v in dag.out_neighbors(u):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return order


def topological_sort_by_dfs(dag):
    """Return the vertices of ``dag`` in depth-first topological order.

    The order is the reverse post-order of a depth-first search. It follows
    one chain of dependencies as far as possible before moving on, so an
    instruction and the ones that depend on it tend to stay close together.

    Args:
        dag (CircuitDAG): The graph, usually from :meth:`Circuit.dag`.

    Returns:
        list of int: Every vertex once, each after all its predecessors.

    Raises:
        TypeError: If ``dag`` is not a :class:`CircuitDAG`.
        ValueError: If the graph has a cycle. A graph built from a circuit
            never has one.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateX(), 0)
        >>> _ = c.push(GateH(), 1)
        >>> topological_sort_by_dfs(c.dag())
        [2, 0, 1]
    """
    _require_dag(dag)
    n = dag.num_vertices()
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []

    for source in range(n):
        if color[source] != WHITE:
            continue
        stack = [source]
        color[source] = GRAY
        while stack:
            u = stack[-1]
            nxt = -1
            for v in dag.out_neighbors(u):
                if color[v] == GRAY:
                    raise ValueError("The circuit DAG contains a cycle.")
                if color[v] == WHITE:
                    nxt = v
                    break
            if nxt != -1:
                color[nxt] = GRAY
                stack.append(nxt)
            else:
                color[u] = BLACK
                order.append(u)
                stack.pop()

    order.reverse()
    return order


def traverse_by_bfs(circuit):
    """Yield the instructions of ``circuit`` in breadth-first topological order.

    Instructions come out layer by layer, as in
    :func:`topological_sort_by_bfs`: all instructions that can run at the
    same time are yielded one after the other. Pushing them in this order
    gives an equivalent circuit.

    Args:
        circuit (Circuit): The circuit to traverse.

    Yields:
        Instruction: The instructions of the circuit, each once.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateX(), 0)
        >>> _ = c.push(GateH(), 1)
        >>> list(traverse_by_bfs(c))
        [H @ q[0], H @ q[1], X @ q[0]]
    """
    dag = circuit.dag()
    instructions = circuit.instructions
    for i in topological_sort_by_bfs(dag):
        yield instructions[i]


def traverse_by_dfs(circuit):
    """Yield the instructions of ``circuit`` in depth-first topological order.

    The order is that of :func:`topological_sort_by_dfs`. Pushing the
    instructions in this order gives an equivalent circuit.

    Args:
        circuit (Circuit): The circuit to traverse.

    Yields:
        Instruction: The instructions of the circuit, each once.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateX(), 0)
        >>> _ = c.push(GateH(), 1)
        >>> list(traverse_by_dfs(c))
        [H @ q[1], H @ q[0], X @ q[0]]
    """
    dag = circuit.dag()
    instructions = circuit.instructions
    for i in topological_sort_by_dfs(dag):
        yield instructions[i]


def to_networkx(circuit):
    """Return the circuit's dependency graph as a ``networkx.DiGraph``.

    Node ``i`` is instruction ``i`` of the circuit and carries it in its
    ``instruction`` attribute; edges are those of :meth:`Circuit.dag`. The
    graph can be passed to any networkx algorithm or drawing function.

    Args:
        circuit (Circuit): The circuit.

    Returns:
        networkx.DiGraph: The dependency graph.

    Raises:
        ImportError: If ``networkx`` is not installed. It is an optional
            dependency, installed with ``pip install mimiqcircuits[graph]``.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit().push(GateH(), 0).push(GateCX(), 0, 1)
        >>> g = to_networkx(c)  # doctest: +SKIP
        >>> g.nodes[1]["instruction"]  # doctest: +SKIP
        CX @ q[0], q[1]
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "to_networkx requires the optional 'networkx' dependency; "
            "install it with: pip install mimiqcircuits[graph]"
        ) from exc

    dag = circuit.dag()
    instructions = circuit.instructions

    g = nx.DiGraph()
    for v in dag.vertices():
        g.add_node(v, instruction=instructions[v])
    for u, v in dag.edges():
        g.add_edge(u, v)
    return g


__all__ = [
    "CircuitDAG",
    "build_dag",
    "topological_sort_by_bfs",
    "topological_sort_by_dfs",
    "traverse_by_bfs",
    "traverse_by_dfs",
    "to_networkx",
]
