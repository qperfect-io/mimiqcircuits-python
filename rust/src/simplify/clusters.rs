//
// Copyright © 2026 QPerfect. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//! Grouping gates into blocks on few qubits, the same algorithm as
//! `fuse_circuit`, followed step by step.

use super::core::{Ctx, Inst};
use std::collections::{BTreeSet, HashMap};

/// What an instruction contributes to a block.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Data {
    Boundary,
    Dense,
    Diagonal,
}

pub struct Clusters {
    /// Cluster ids in a valid circuit order.
    pub order: Vec<usize>,
    /// Instruction indices of each cluster, empty if folded into another.
    pub members: Vec<Vec<usize>>,
    pub support: Vec<BTreeSet<usize>>,
    /// Whether the cluster is a block of gates rather than a boundary.
    pub fuse: Vec<bool>,
}

pub fn clusters(
    ctx: &Ctx,
    c: &[Inst],
    max_support: usize,
    max_diagonal_support: usize,
    data_of: &dyn Fn(&Ctx, &Inst) -> Data,
) -> Clusters {
    let n = c.len();
    let nq = super::core::num_qubits(c);

    let mut owner: HashMap<usize, usize> = HashMap::new();
    let mut members: Vec<Vec<usize>> = vec![];
    let mut support: Vec<BTreeSet<usize>> = vec![];
    let mut fuse: Vec<bool> = vec![];
    let mut isopen: Vec<bool> = vec![];
    let mut isdiagonal: Vec<bool> = vec![];
    let mut clusterof = vec![0usize; n];

    // A cluster stays open while it owns every wire it spans: nothing after it
    // depends on it yet, so merging it cannot close a cycle.
    let seal = |owner: &HashMap<usize, usize>, isopen: &mut Vec<bool>, qs: &[usize]| {
        for q in qs {
            if let Some(&g) = owner.get(q) {
                isopen[g] = false;
            }
        }
    };

    for i in 0..n {
        let inst = &c[i];
        let data = data_of(ctx, inst);
        if data == Data::Boundary {
            let dq = ctx.dag_qubits(inst, nq);
            seal(&owner, &mut isopen, &dq);
            members.push(vec![i]);
            support.push(inst.qubits.iter().copied().collect());
            fuse.push(false);
            isopen.push(false);
            isdiagonal.push(false);
            let cid = members.len() - 1;
            for q in dq {
                owner.insert(q, cid);
            }
            clusterof[i] = cid;
            continue;
        }

        let qs = &inst.qubits;
        let gdiag = data == Data::Diagonal;

        let mut cand: Vec<usize> = vec![];
        for q in qs {
            if let Some(&g) = owner.get(q) {
                if fuse[g] && isopen[g] && !cand.contains(&g) {
                    cand.push(g);
                }
            }
        }
        if cand.len() > 1 {
            let qset: BTreeSet<usize> = qs.iter().copied().collect();
            cand.sort_by_key(|&g| (gdiag && !isdiagonal[g], support[g].difference(&qset).count()));
        }

        let mut s: BTreeSet<usize> = qs.iter().copied().collect();
        let mut alldiag = gdiag;
        let mut chosen: Vec<usize> = vec![];
        for &g in &cand {
            let u: BTreeSet<usize> = s.union(&support[g]).copied().collect();
            let dg = alldiag && isdiagonal[g];
            if u.len() <= if dg { max_diagonal_support } else { max_support } {
                s = u;
                alldiag = dg;
                chosen.push(g);
            }
        }

        if chosen.is_empty() {
            seal(&owner, &mut isopen, qs);
            members.push(vec![i]);
            support.push(qs.iter().copied().collect());
            fuse.push(true);
            isopen.push(true);
            isdiagonal.push(gdiag);
            let cid = members.len() - 1;
            for &q in qs {
                owner.insert(q, cid);
            }
            clusterof[i] = cid;
            continue;
        }

        // Fold the chosen clusters into the earliest. Any other cluster holding
        // one of this gate's wires loses it here, so it is sealed.
        let first = *chosen.iter().min().unwrap();
        for q in qs {
            if let Some(&g) = owner.get(q) {
                if !chosen.contains(&g) {
                    isopen[g] = false;
                }
            }
        }
        for &g in &chosen {
            if g == first {
                continue;
            }
            let moved = std::mem::take(&mut members[g]);
            for &j in &moved {
                clusterof[j] = first;
            }
            members[first].extend(moved);
            let sup = std::mem::take(&mut support[g]);
            support[first].extend(sup);
        }
        members[first].push(i);
        support[first].extend(qs.iter().copied());
        isdiagonal[first] = alldiag;
        clusterof[i] = first;
        for &q in &s {
            owner.insert(q, first);
        }
    }

    // Contract the instruction DAG by cluster and sort it with Kahn's
    // algorithm: ready clusters first-in first-out, successors by id.
    let nc = members.len();
    let mut succ: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nc];
    let mut last_q: HashMap<usize, usize> = HashMap::new();
    let mut last_b: HashMap<usize, usize> = HashMap::new();
    let mut last_z: HashMap<usize, usize> = HashMap::new();
    let link = |prev: Option<usize>, i: usize, succ: &mut Vec<BTreeSet<usize>>| {
        if let Some(p) = prev {
            let (cu, cv) = (clusterof[p], clusterof[i]);
            if cu != cv {
                succ[cu].insert(cv);
            }
        }
    };
    for (i, inst) in c.iter().enumerate() {
        for q in ctx.dag_qubits(inst, nq) {
            link(last_q.insert(q, i), i, &mut succ);
        }
        for &b in &inst.bits {
            link(last_b.insert(b, i), i, &mut succ);
        }
        for &z in &inst.zvars {
            link(last_z.insert(z, i), i, &mut succ);
        }
    }
    let mut indeg = vec![0usize; nc];
    for s in &succ {
        for &d in s {
            indeg[d] += 1;
        }
    }
    let mut order: Vec<usize> = (0..nc).filter(|&g| indeg[g] == 0).collect();
    let mut k = 0;
    while k < order.len() {
        for &d in &succ[order[k]] {
            indeg[d] -= 1;
            if indeg[d] == 0 {
                order.push(d);
            }
        }
        k += 1;
    }

    Clusters {
        order,
        members,
        support,
        fuse,
    }
}
