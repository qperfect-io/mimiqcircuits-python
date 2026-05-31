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

import importlib.util

import pytest

import mimiqcircuits as mc
from mimiqcircuits.dag import (
    CircuitDAG,
    build_dag,
    topological_sort_by_bfs,
    topological_sort_by_dfs,
)

HAS_NETWORKX = importlib.util.find_spec("networkx") is not None


def assert_topological(order, dag):
    """Every dependency edge must place its source before its target."""
    assert sorted(order) == list(range(dag.num_vertices()))
    position = {v: i for i, v in enumerate(order)}
    for u, v in dag.edges():
        assert position[u] < position[v]


def bell_then_local():
    """H(0), H(1), CX(0,1), X(0), Z(1).

    The two H gates are independent, both feed the CX, and the final X/Z
    depend on the CX — a small diamond useful for checking edges and order.
    """
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 1)
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateX(), 0)
    c.push(mc.GateZ(), 1)
    return c


def test_empty_circuit():
    c = mc.Circuit()
    dag = c.dag()
    assert dag.num_vertices() == 0
    assert dag.num_edges() == 0
    assert list(c.traverse_by_bfs()) == []
    assert list(c.traverse_by_dfs()) == []


def test_single_instruction():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    dag = c.dag()
    assert dag.num_vertices() == 1
    assert dag.num_edges() == 0
    assert [str(i.operation) for i in c.traverse_by_bfs()] == ["H"]


def test_linear_chain_same_qubit():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateX(), 0)
    c.push(mc.GateZ(), 0)
    dag = c.dag()
    # Each gate depends only on the immediately preceding one on qubit 0.
    assert dag.edges() == [(0, 1), (1, 2)]
    assert topological_sort_by_bfs(dag) == [0, 1, 2]


def test_independent_gates_have_no_edges():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateH(), 1)
    c.push(mc.GateH(), 2)
    dag = c.dag()
    assert dag.num_edges() == 0
    assert topological_sort_by_bfs(dag) == [0, 1, 2]


def test_qubit_dependency_edges():
    dag = bell_then_local().dag()
    assert dag.num_vertices() == 5
    assert dag.edges() == [(0, 2), (1, 2), (2, 3), (2, 4)]
    assert dag.in_degree(2) == 2
    assert sorted(dag.in_neighbors(2)) == [0, 1]
    assert sorted(dag.out_neighbors(2)) == [3, 4]


def test_bit_dependency_edges():
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Measure(), 1, 0)  # different qubit, same classical bit
    dag = c.dag()
    assert dag.edges() == [(0, 1)]


def test_zvar_dependency_edges():
    c = mc.Circuit()
    c.push(mc.ExpectationValue(mc.GateX()), 0, 0)
    c.push(mc.ExpectationValue(mc.GateX()), 1, 0)  # different qubit, same zvar
    dag = c.dag()
    assert dag.edges() == [(0, 1)]


def test_shared_predecessor_is_single_edge():
    # The second CX shares both qubits with the first, but the dependency
    # between them is a single edge, not one per shared wire.
    c = mc.Circuit()
    c.push(mc.GateCX(), 0, 1)
    c.push(mc.GateCX(), 0, 1)
    dag = c.dag()
    assert dag.edges() == [(0, 1)]
    assert dag.in_degree(1) == 1


def test_bfs_groups_by_depth():
    # X(0) -> X(0) is a chain; X(1) is independent. BFS visits both
    # depth-0 gates (indices 0 and 2) before the depth-1 gate (index 1).
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    c.push(mc.GateX(), 0)
    c.push(mc.GateX(), 1)
    dag = c.dag()
    assert topological_sort_by_bfs(dag) == [0, 2, 1]


def test_bfs_and_dfs_are_topological():
    c = bell_then_local()
    dag = c.dag()
    assert_topological(topological_sort_by_bfs(dag), dag)
    assert_topological(topological_sort_by_dfs(dag), dag)


def test_traverse_yields_instructions_in_order():
    c = bell_then_local()
    order = topological_sort_by_bfs(c.dag())
    expected = [c.instructions[i] for i in order]
    assert list(c.traverse_by_bfs()) == expected


def test_cache_rebuilt_after_mutation():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    assert c.dag().num_vertices() == 1

    c.push(mc.GateCX(), 0, 1)
    assert c.dag().num_vertices() == 2
    assert c.dag().edges() == [(0, 1)]

    c.insert(0, mc.GateX(), 0)
    assert c.dag().num_vertices() == 3

    c.remove(0)
    assert c.dag().num_vertices() == 2

    c.append(mc.Circuit([mc.Instruction(mc.GateH(), (1,))]))
    assert c.dag().num_vertices() == 3


def test_resources_cached_and_recomputed():
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    assert c.num_qubits() == 1
    c.push(mc.GateH(), 3)
    assert c.num_qubits() == 4  # recomputed after the cache was invalidated


def test_instructions_is_read_only():
    c = bell_then_local()
    with pytest.raises(AttributeError):
        c.instructions = []


def test_circuitdag_api():
    dag = bell_then_local().dag()
    assert list(dag.vertices()) == [0, 1, 2, 3, 4]
    assert dag.has_vertex(4)
    assert not dag.has_vertex(5)
    assert dag.has_edge(0, 2)
    assert not dag.has_edge(0, 3)
    assert dag.out_degree(2) == 2
    assert dag.in_degree(0) == 0


def test_dfs_detects_cycle():
    dag = CircuitDAG(2)
    dag._add_edge(0, 1)
    dag._add_edge(1, 0)
    with pytest.raises(ValueError):
        topological_sort_by_dfs(dag)


def test_build_dag_matches_circuit_dag():
    c = bell_then_local()
    assert build_dag(c).edges() == c.dag().edges()


@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
def test_to_networkx_roundtrip():
    import networkx as nx

    c = bell_then_local()
    g = c.to_networkx()
    dag = c.dag()
    assert g.number_of_nodes() == dag.num_vertices()
    assert g.number_of_edges() == dag.num_edges()
    assert set(g.edges()) == set(dag.edges())
    assert nx.is_directed_acyclic_graph(g)
    # Each node carries its instruction for downstream analysis/drawing.
    assert g.nodes[2]["instruction"] is c.instructions[2]


def test_to_networkx_without_dependency_raises():
    if HAS_NETWORKX:
        pytest.skip("networkx is installed; cannot test the missing-dependency path")
    c = bell_then_local()
    with pytest.raises(ImportError, match="networkx"):
        c.to_networkx()
