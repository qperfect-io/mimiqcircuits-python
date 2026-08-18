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

Replaces runs of unitary gates spanning at most ``max_support`` qubits with a
single :class:`GateCustom` block whose matrix is the ordered product of the
run. Non-unitary or opaque operations (measurements, resets, noise channels,
``Barrier``, control flow, and gates with symbolic parameters) are emitted
unchanged and act as fusion boundaries: no gate fuses across one on a shared
wire.

Clusters grow by absorbing the clusters that own a gate's wires. Merging is
what makes ``max_support`` above two useful: on an entangling circuit every
wire is owned after the first layer, so a gate that could only ever extend a
single cluster would start a fresh one every time.

Merging two clusters is only sound when nothing outside them sits in between,
otherwise the contracted DAG gains a cycle and the circuit cannot be reordered.
The ``open`` flag is what keeps that safe: see ``seal`` in `fuse_circuit`.
"""

import numpy as np

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.dag import _dag_qubits
from mimiqcircuits.matrices import _apply_local
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

    Each member acts at its local position within ``support``, applied in
    circuit order (the later gate on the left). Members are numeric by
    construction — ``_is_fusible`` rejects symbolic gates — so this runs in
    NumPy rather than through SymEngine.
    """
    nq = len(support)
    u = np.eye(2**nq, dtype=complex)
    for i in sorted(members):
        inst = circuit[i]
        localq = [support.index(q) for q in inst.qubits]
        u = _apply_local(u, inst.operation.unwrappedmatrix(), localq, nq)
    return u


def fuse_circuit(circuit, max_support=2):
    """Fuse runs of adjacent gates in ``circuit`` into ``GateCustom`` blocks.

    Returns a new circuit implementing the same unitary. See the module
    docstring for the boundary rules. A run of a single gate is left as its
    original instruction (never rewrapped as a one-qubit ``GateCustom``), and
    qubit indices are never relabeled.

    Raising ``max_support`` lets a block cover more wires and so emit fewer,
    wider blocks. Which gates end up together is decided greedily, so the
    result is not guaranteed to be the smallest possible circuit.

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
    # {"members": [...], "support": set(), "kind": "FUSE" | "PASS", "open": bool}
    clusters = []
    cluster_of = [None] * n

    def new_cluster(i, qs, kind):
        cid = len(clusters)
        clusters.append(
            {"members": [i], "support": set(qs), "kind": kind, "open": kind == "FUSE"}
        )
        return cid

    # A cluster stays *open* while it owns every wire it spans, which makes it a
    # sink in the contracted DAG: nothing downstream depends on it yet. Losing a
    # wire to a later instruction gives it a successor, and from then on merging
    # it could close a cycle (A -> X -> B with X left outside), so it is sealed
    # for good.
    def seal(qs):
        for q in qs:
            g = owner.get(q)
            if g is not None:
                clusters[g]["open"] = False

    for i, inst in enumerate(circuit):
        qs = list(inst.qubits)
        if not _is_fusible(inst, max_support):
            # The boundary owns every wire it depends on, so a later gate on one
            # of those wires cannot fuse back into a cluster sitting before it.
            # A few global observables synchronise the whole register, hence
            # `_dag_qubits` rather than `inst.qubits`.
            dq = _dag_qubits(inst, nq)
            seal(dq)
            cid = new_cluster(i, qs, "PASS")
            for q in dq:
                owner[q] = cid
            cluster_of[i] = cid
            continue

        # Candidates are the open fusible clusters owning this gate's wires.
        # Merging several of them at once is what lets a cluster grow past two
        # qubits: every one absorbed is an operation removed from the output, so
        # take them cheapest-first to fit as many as `max_support` allows.
        cand = []
        for q in qs:
            g = owner.get(q)
            if g is None or g in cand:
                continue
            if clusters[g]["kind"] == "FUSE" and clusters[g]["open"]:
                cand.append(g)
        cand.sort(key=lambda g: len(clusters[g]["support"] - set(qs)))

        support = set(qs)
        chosen = []
        for g in cand:
            u = support | clusters[g]["support"]
            if len(u) <= max_support:
                support = u
                chosen.append(g)

        if not chosen:
            seal(qs)
            cid = new_cluster(i, qs, "FUSE")
            for q in qs:
                owner[q] = cid
            cluster_of[i] = cid
            continue

        # Fold the chosen clusters into the earliest of them. Any other cluster
        # holding one of this gate's wires loses it here, so it is sealed.
        s = min(chosen)
        for q in qs:
            g = owner.get(q)
            if g is not None and g not in chosen:
                clusters[g]["open"] = False
        for g in chosen:
            if g == s:
                continue
            clusters[s]["members"].extend(clusters[g]["members"])
            for j in clusters[g]["members"]:
                cluster_of[j] = s
            clusters[s]["support"] |= clusters[g]["support"]
            clusters[g]["members"] = []  # folded away, emits nothing
            clusters[g]["support"] = set()
        clusters[s]["members"].append(i)
        clusters[s]["support"].update(qs)
        cluster_of[i] = s
        for q in support:
            owner[q] = s

    # Contract the instruction DAG by cluster id and topologically sort it: any
    # topological order is a valid, equivalent circuit (independent clusters
    # commute). Merging only sinks keeps every cluster convex, so this is a DAG.
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
        if not cluster["members"]:
            continue
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
