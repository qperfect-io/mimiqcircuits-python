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

This mirrors the graph integration of the Julia ``MimiqCircuitsBase``: the
vertices are instruction positions and an edge ``u -> v`` records that ``v``
depends on the most recent earlier instruction ``u`` acting on a shared wire.
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
    lazily and caches it; construct one directly only when working with the
    graph in isolation.
    """

    def __init__(self, n):
        self._n = n
        # Adjacency lists are kept sorted and duplicate-free so that traversals
        # are deterministic and in-degree counts match the number of distinct
        # predecessors (Kahn's algorithm relies on the latter).
        self._out = [[] for _ in range(n)]
        self._in = [[] for _ in range(n)]

    def num_vertices(self):
        """Number of vertices, i.e. instructions in the circuit."""
        return self._n

    def num_edges(self):
        """Number of dependency edges."""
        return sum(len(o) for o in self._out)

    def vertices(self):
        """Range over all vertex indices."""
        return range(self._n)

    def edges(self):
        """List of ``(u, v)`` dependency edges with ``u`` before ``v``."""
        return [(u, v) for u in range(self._n) for v in self._out[u]]

    def out_neighbors(self, v):
        """Instructions that depend directly on ``v`` (its successors)."""
        return self._out[v]

    def in_neighbors(self, v):
        """Instructions that ``v`` depends on directly (its predecessors)."""
        return self._in[v]

    def in_degree(self, v):
        """Number of direct predecessors of ``v``."""
        return len(self._in[v])

    def out_degree(self, v):
        """Number of direct successors of ``v``."""
        return len(self._out[v])

    def has_vertex(self, v):
        return 0 <= v < self._n

    def has_edge(self, u, v):
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

    Walks the instructions in order, remembering the last instruction seen on
    each qubit, bit, and z-variable, and links every instruction to those last
    writers. The result encodes exactly the orderings that must be preserved
    for the circuit to remain equivalent. The global state observables depend
    on every qubit (see :func:`_dag_qubits`).
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


def topological_sort_by_bfs(dag):
    """Topological order via Kahn's algorithm (breadth-first, layer by layer).

    Independent instructions — those at the same dependency depth — come out
    grouped together, which is the order the MPS simulator uses to fuse gates.
    Vertices with no remaining predecessor are visited in ascending index
    order, making the result deterministic.
    """
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
    """Topological order via depth-first search (reverse post-order).

    Follows one dependency chain as deep as possible before backtracking, so a
    gate and its downstream cone tend to stay close together in the output.
    Raises :class:`ValueError` if the graph contains a cycle, which for a
    well-formed circuit should never happen.
    """
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

    Equivalent gates that can run in parallel are emitted as consecutive
    layers. Use this when the relative order of independent gates does not
    matter but layering does (e.g. tensor-network compression).
    """
    dag = circuit.dag()
    instructions = circuit.instructions
    for i in topological_sort_by_bfs(dag):
        yield instructions[i]


def traverse_by_dfs(circuit):
    """Yield the instructions of ``circuit`` in depth-first topological order."""
    dag = circuit.dag()
    instructions = circuit.instructions
    for i in topological_sort_by_dfs(dag):
        yield instructions[i]


def to_networkx(circuit):
    """Return the circuit's dependency graph as a ``networkx.DiGraph``.

    Each node carries its :class:`Instruction` under the ``instruction``
    attribute, so the returned graph plugs straight into networkx algorithms
    and drawing. Requires the optional ``networkx`` dependency, installable
    with ``pip install mimiqcircuits[graph]``.
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
