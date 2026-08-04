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
"""Clustering gate-fusion pass.

Replaces maximal runs of adjacent unitary gates acting on at most
``max_support`` qubits with a single :class:`GateCustom` block whose matrix is
the ordered product of the run. Non-unitary or opaque operations
(measurements, resets, noise channels, ``Barrier``, control flow, and gates
with symbolic parameters) are emitted unchanged and act as fusion boundaries:
no gate fuses across one on a shared wire.
"""

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.dag import _dag_qubits
from mimiqcircuits.matrices import reorder_qubits_matrix
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.custom import GateCustom
from mimiqcircuits.backends.passes import AbstractPass, PassSpec, PassResult


def _is_fusible(inst, n):
    """Whether ``inst`` is a plain unitary gate with a concrete numeric matrix.

    Requiring a :class:`Gate` already excludes barriers, measurements, resets,
    noise channels and control flow (none subclass it), so we never rely on
    ``isunitary`` alone — a ``Barrier`` reports unitary but is not a gate.
    Gates with symbolic parameters have no numeric matrix, so
    ``unwrappedmatrix`` raises and they fall through as boundaries.
    """
    op = inst.operation
    if not isinstance(op, Gate):
        return False
    if len(inst.qubits) > n:
        return False
    if not op.isunitary():
        return False
    try:
        op.unwrappedmatrix()
    except Exception:
        return False  # symbolic or no concrete matrix -> treat as a boundary
    return True


def _synthesize(circuit, members, support):
    """Dense matrix of a cluster on the sorted ``support``.

    Each member is embedded at its local position within ``support`` and the
    embeddings are multiplied in circuit order (the later gate on the left).
    Members are numeric by construction — ``_is_fusible`` rejects symbolic
    gates — so the product runs in NumPy rather than through SymEngine.
    """
    u = None
    for i in sorted(members):
        inst = circuit[i]
        localq = [support.index(q) for q in inst.qubits]
        e = reorder_qubits_matrix(inst.operation.unwrappedmatrix(), localq, len(support))
        u = e if u is None else e @ u
    return u


def fuse_circuit(circuit, max_support=2):
    """Fuse runs of adjacent gates in ``circuit`` into ``GateCustom`` blocks.

    Returns a new circuit implementing the same unitary. See the module
    docstring for the boundary rules. A run of a single gate is left as its
    original instruction (never rewrapped as a one-qubit ``GateCustom``), and
    qubit indices are never relabeled.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateH(), 0)      # a Hadamard ...
        >>> _ = c.push(mc.GateCX(), 0, 1)  # ... feeding a CX on {0, 1}
        >>> fused = mc.fuse_circuit(c)      # both act on {0, 1}: one 2-qubit block
        >>> len(fused)
        1
        >>> isinstance(fused[0].operation, mc.GateCustom)
        True
    """
    n = len(circuit)
    nq = circuit.num_qubits()

    owner = {}  # qubit -> cluster id (a boundary owns its wires too)
    clusters = []  # {"members": [...], "support": set(), "kind": "FUSE" | "PASS"}
    cluster_of = [None] * n

    def new_cluster(i, qs, kind):
        cid = len(clusters)
        clusters.append({"members": [i], "support": set(qs), "kind": kind})
        return cid

    for i, inst in enumerate(circuit):
        qs = list(inst.qubits)
        if not _is_fusible(inst, max_support):
            cid = new_cluster(i, qs, "PASS")
            # The boundary owns every wire it depends on, so a later gate on one
            # of those wires cannot fuse back into a cluster sitting before it.
            # A few global observables synchronise the whole register, hence
            # `_dag_qubits` rather than `inst.qubits`.
            for q in _dag_qubits(inst, nq):
                owner[q] = cid
            cluster_of[i] = cid
            continue

        live = {owner[q] for q in qs if q in owner}
        if len(live) == 1:
            g = next(iter(live))
            # Join only a fusible cluster that already owns the immediate
            # predecessor on each shared wire. A boundary-owned wire is "PASS"
            # and so blocks the join; fresh wires carry no owner.
            if (
                clusters[g]["kind"] == "FUSE"
                and len(clusters[g]["support"] | set(qs)) <= max_support
            ):
                clusters[g]["members"].append(i)
                clusters[g]["support"].update(qs)
                for q in qs:
                    owner[q] = g
                cluster_of[i] = g
                continue

        cid = new_cluster(i, qs, "FUSE")
        for q in qs:
            owner[q] = cid
        cluster_of[i] = cid

    # Contract the instruction DAG by cluster id and topologically sort it: any
    # topological order is a valid, equivalent circuit (independent clusters
    # commute). Single-owner greedy keeps every cluster convex, so this is a DAG.
    dag = circuit.dag()
    succ = {c: set() for c in range(len(clusters))}
    indeg = [0] * len(clusters)
    for u in range(dag.num_vertices()):
        cu = cluster_of[u]
        for v in dag.out_neighbors(u):
            cv = cluster_of[v]
            if cu != cv and cv not in succ[cu]:
                succ[cu].add(cv)
                indeg[cv] += 1

    queue = sorted(c for c in range(len(clusters)) if indeg[c] == 0)
    order = []
    while queue:
        c = queue.pop(0)
        order.append(c)
        for d in sorted(succ[c]):
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(d)

    out = Circuit()
    for cid in order:
        cluster = clusters[cid]
        if cluster["kind"] == "PASS" or len(cluster["members"]) == 1:
            for i in cluster["members"]:
                out.push(circuit[i])  # verbatim: keeps qubits, bits and zvars
        else:
            support = sorted(cluster["support"])
            out.push(GateCustom(_synthesize(circuit, cluster["members"], support)), *support)
    return out


class FusePass(AbstractPass):
    """Pass that fuses adjacent gates into ``GateCustom`` blocks.

    Wraps :func:`fuse_circuit`; ``max_support`` caps the block width (default
    ``2``). Runs through :class:`PassPipeline` like any other pass and does not
    relabel qubits, so :attr:`PassResult.qubit_permutation` is ``None``.

    Fusion runs only when the circuit has at least ``qubit_threshold`` qubits;
    smaller circuits pass through unchanged, since fusing does not pay off until
    the statevector is large enough that each gate application dominates.
    ``qubit_threshold = 0`` (the default) always fuses.
    """

    def __init__(self, max_support=2, qubit_threshold=0):
        self.max_support = int(max_support)
        self.qubit_threshold = int(qubit_threshold)

    def spec(self):
        return PassSpec.from_dict(
            "fuse_gates",
            {"max_support": self.max_support, "qubit_threshold": self.qubit_threshold},
        )

    def apply(self, ctx, circuit):
        if circuit.num_qubits() < self.qubit_threshold:
            return circuit, PassResult()
        fused = fuse_circuit(circuit, self.max_support)
        result = PassResult(
            qubit_permutation=None,
            metadata={"pass": "fuse_gates", "before": len(circuit), "after": len(fused)},
        )
        return fused, result
