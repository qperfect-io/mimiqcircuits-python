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

Runs that are diagonal in the computational basis get their own, usually wider,
budget ``max_diagonal_support`` and collapse into a :class:`GateCustomDiagonal`:
composing two diagonals is an elementwise product of ``2**k`` entries, so both
the block and the work to build it stay linear in the state size rather than
quadratic. A cluster keeps that budget only while every member is diagonal;
absorbing one dense gate drops it back to ``max_support``.

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
from mimiqcircuits.operations.gates.customdiagonal import GateCustomDiagonal
from mimiqcircuits.backends.passes import AbstractPass, PassSpec, PassResult


# Off-diagonal weight below this is dropped when a gate is classified as
# diagonal: a fused block accumulates rounding noise where the exact gate has
# zeros, and refusing to see those gates as diagonal would make fusion depend on
# the last bits of the mantissa.
_DIAGONAL_ATOL = 1e-12


def _is_diagonal_matrix(u):
    return bool(np.max(np.abs(u - np.diag(np.diag(u)))) <= _DIAGONAL_ATOL)


def _fusiondata(inst, max_support, max_diagonal_support):
    """What ``inst`` contributes to a cluster.

    Its numeric diagonal (a 1-D array) if it is diagonal in the computational
    basis, its numeric matrix (2-D) if it is not, and ``None`` if it cannot be
    fused at all — a boundary.

    Requiring a :class:`Gate` already excludes barriers, measurements, resets,
    noise channels and control flow (none subclass it), so we never rely on
    ``isunitary`` alone — a ``Barrier`` reports unitary but is not a gate.
    Gates with symbolic parameters have no numeric matrix, so ``unwrappedmatrix``
    raises and they fall through as boundaries.
    """
    op = inst.operation
    if not isinstance(op, Gate):
        return None
    if not op.isunitary():
        return None
    nq = len(inst.qubits)
    if nq > max(max_support, max_diagonal_support):
        return None

    # Densifying a wide GateCustomDiagonal just to notice it is diagonal would
    # defeat the point of storing it as a diagonal in the first place.
    if isinstance(op, GateCustomDiagonal):
        if nq > max_diagonal_support:
            return None
        try:
            return op.unwrappeddiagonal()
        except Exception:
            return None

    try:
        u = op.unwrappedmatrix()
    except Exception:
        return None  # symbolic or no concrete matrix -> treat as a boundary

    if nq <= max_diagonal_support and _is_diagonal_matrix(u):
        return np.diag(u).copy()
    return u if nq <= max_support else None


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


def _synthesize_diagonal(circuit, members, support, diagonals):
    """Diagonal of a cluster, on the sorted ``support``.

    Diagonals compose elementwise, so the cluster's diagonal is the product of
    its members', each read at the entry its own targets select. Order is
    irrelevant here — diagonal gates commute — but members are walked in circuit
    order anyway. ``support`` is sorted and the first qubit of a target list is
    the most significant bit, so ``support[p]`` contributes bit ``k - 1 - p`` of
    the cluster index.
    """
    k = len(support)
    x = np.arange(1 << k)
    d = np.ones(1 << k, dtype=complex)
    for i in sorted(members):
        j = np.zeros(1 << k, dtype=np.int64)
        for q in circuit[i].qubits:
            j = (j << 1) | ((x >> (k - 1 - support.index(q))) & 1)
        d *= diagonals[i][j]
    return d


def fuse_circuit(circuit, max_support=2, max_diagonal_support=None):
    """Fuse runs of adjacent gates in ``circuit`` into ``GateCustom`` blocks.

    Returns a new circuit implementing the same unitary. See the module
    docstring for the boundary rules. A run of a single gate is left as its
    original instruction (never rewrapped as a one-qubit ``GateCustom``), and
    qubit indices are never relabeled.

    Raising ``max_support`` lets a block cover more wires and so emit fewer,
    wider blocks. Which gates end up together is decided greedily, so the
    result is not guaranteed to be the smallest possible circuit.

    Runs where every gate is diagonal in the computational basis are budgeted
    separately, by ``max_diagonal_support`` (default: ``max_support``), and
    collapse into a ``GateCustomDiagonal`` instead: such a block is ``2**k``
    numbers rather than ``4**k``, and it is applied without mixing amplitudes,
    so it is usually worth allowing wider than a dense one. A cluster keeps the
    diagonal budget only while all of its gates are diagonal — the first dense
    gate joining it brings it back under ``max_support``.

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
        >>> d = mc.Circuit()
        >>> _ = d.push(mc.GateP(0.1), 0)
        >>> _ = d.push(mc.GateCZ(), 0, 1)   # both diagonal, so the block is too
        >>> isinstance(mc.fuse_circuit(d)[0].operation, mc.GateCustomDiagonal)
        True
    """
    if max_diagonal_support is None:
        max_diagonal_support = max_support

    n = len(circuit)
    nq = circuit.num_qubits()

    owner = {}  # qubit -> cluster id (a boundary owns its wires too)
    # {"members": [...], "support": set(), "kind": "FUSE" | "PASS",
    #  "open": bool, "diag": bool}
    clusters = []
    cluster_of = [None] * n
    diagonals = {}  # instruction index -> its diagonal

    def new_cluster(i, qs, kind, diag):
        cid = len(clusters)
        clusters.append(
            {
                "members": [i],
                "support": set(qs),
                "kind": kind,
                "open": kind == "FUSE",
                "diag": diag,
            }
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
        data = _fusiondata(inst, max_support, max_diagonal_support)
        if data is None:
            # The boundary owns every wire it depends on, so a later gate on one
            # of those wires cannot fuse back into a cluster sitting before it.
            # A few global observables synchronise the whole register, hence
            # `_dag_qubits` rather than `inst.qubits`.
            dq = _dag_qubits(inst, nq)
            seal(dq)
            cid = new_cluster(i, qs, "PASS", False)
            for q in dq:
                owner[q] = cid
            cluster_of[i] = cid
            continue

        gdiag = data.ndim == 1
        if gdiag:
            diagonals[i] = data

        # Candidates are the open fusible clusters owning this gate's wires.
        # Merging several of them at once is what lets a cluster grow past two
        # qubits: every one absorbed is an operation removed from the output, so
        # take them cheapest-first to fit as many as the budget allows. A
        # diagonal gate takes diagonal clusters first: absorbing a dense one
        # forfeits `max_diagonal_support` for the whole run.
        cand = []
        for q in qs:
            g = owner.get(q)
            if g is None or g in cand:
                continue
            if clusters[g]["kind"] == "FUSE" and clusters[g]["open"]:
                cand.append(g)
        cand.sort(
            key=lambda g: (
                gdiag and not clusters[g]["diag"],
                len(clusters[g]["support"] - set(qs)),
            )
        )

        support = set(qs)
        alldiag = gdiag
        chosen = []
        for g in cand:
            u = support | clusters[g]["support"]
            dg = alldiag and clusters[g]["diag"]
            if len(u) <= (max_diagonal_support if dg else max_support):
                support = u
                alldiag = dg
                chosen.append(g)

        if not chosen:
            seal(qs)
            cid = new_cluster(i, qs, "FUSE", gdiag)
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
        clusters[s]["diag"] = alldiag
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
        elif cluster["diag"]:
            support = sorted(cluster["support"])
            d = _synthesize_diagonal(circuit, cluster["members"], support, diagonals)
            out.push(GateCustomDiagonal(d), *support)
        else:
            support = sorted(cluster["support"])
            out.push(GateCustom(_synthesize(circuit, cluster["members"], support)), *support)
    return out


class FusePass(AbstractPass):
    """Pass that fuses adjacent gates into ``GateCustom`` blocks.

    Wraps :func:`fuse_circuit`; ``max_support`` caps the block width (default
    ``2``). Runs through :class:`PassPipeline` like any other pass and does not
    relabel qubits, so :attr:`PassResult.qubit_permutation` is ``None``.

    Runs made only of gates diagonal in the computational basis are budgeted by
    ``max_diagonal_support`` instead and fuse into a ``GateCustomDiagonal``,
    which stores ``2**k`` entries rather than ``4**k``. It defaults to
    ``max_support``; raise it for backends that apply a diagonal without mixing
    amplitudes.

    Fusion runs only when the circuit has at least ``qubit_threshold`` qubits;
    smaller circuits pass through unchanged, since fusing does not pay off until
    the statevector is large enough that each gate application dominates.
    ``qubit_threshold = 0`` (the default) always fuses.
    """

    def __init__(self, max_support=2, max_diagonal_support=None, qubit_threshold=0):
        self.max_support = int(max_support)
        self.max_diagonal_support = (
            self.max_support
            if max_diagonal_support is None
            else int(max_diagonal_support)
        )
        self.qubit_threshold = int(qubit_threshold)

    def spec(self):
        return PassSpec.from_dict(
            "fuse_gates",
            {
                "max_support": self.max_support,
                "max_diagonal_support": self.max_diagonal_support,
                "qubit_threshold": self.qubit_threshold,
            },
        )

    def apply(self, ctx, circuit):
        if circuit.num_qubits() < self.qubit_threshold:
            return circuit, PassResult()
        fused = fuse_circuit(circuit, self.max_support, self.max_diagonal_support)
        result = PassResult(
            qubit_permutation=None,
            metadata={"pass": "fuse_gates", "before": len(circuit), "after": len(fused)},
        )
        return fused, result
