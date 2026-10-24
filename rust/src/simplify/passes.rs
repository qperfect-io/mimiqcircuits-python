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
//! The simplification passes. Each returns the new circuit and whether it
//! differs from the input.

use super::clusters::{Data, clusters};
use super::core::{Action, Ctx, Inst, LOOKBACK, num_qubits};
use super::kak::kak_synthesis;
use super::linalg::{Mat, embed, fold, zyz};
use super::linear::{affine_map, linear_synthesis};
use super::table::Code;
use pyo3::PyResult;
use std::collections::HashMap;

/// The passes, by the codes the Python side uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    RemoveIdentities = 0,
    CancelInverses = 1,
    MergeRotations = 2,
    Resynthesize1q = 3,
    Resynthesize2q = 4,
    ResynthesizeLinear = 5,
}

impl Pass {
    pub fn from_code(code: u8) -> Option<Pass> {
        Some(match code {
            0 => Pass::RemoveIdentities,
            1 => Pass::CancelInverses,
            2 => Pass::MergeRotations,
            3 => Pass::Resynthesize1q,
            4 => Pass::Resynthesize2q,
            5 => Pass::ResynthesizeLinear,
            _ => return None,
        })
    }
}

pub fn run(ctx: &mut Ctx, pass: Pass, c: Vec<Inst>) -> PyResult<(Vec<Inst>, bool)> {
    Ok(match pass {
        Pass::RemoveIdentities => remove_identities(ctx, c),
        Pass::CancelInverses => peephole(ctx, c, false)?,
        Pass::MergeRotations => peephole(ctx, c, true)?,
        Pass::Resynthesize1q => resynthesize_1q(ctx, c),
        Pass::Resynthesize2q => resynthesize_2q(ctx, c),
        Pass::ResynthesizeLinear => resynthesize_linear(ctx, c),
    })
}

fn remove_identities(ctx: &mut Ctx, c: Vec<Inst>) -> (Vec<Inst>, bool) {
    let n = c.len();
    let out: Vec<Inst> = c
        .into_iter()
        .filter(|i| !ctx.mat(i.entry).is_some_and(Mat::is_identity))
        .collect();
    let changed = out.len() != n;
    (out, changed)
}

/// The engine of `cancel_inverses` and `merge_rotations`: each gate walks back
/// over earlier instructions on its wires, past those it commutes with, until
/// it finds one on the same qubits it cancels or merges with.
fn peephole(ctx: &mut Ctx, c: Vec<Inst>, merge: bool) -> PyResult<(Vec<Inst>, bool)> {
    let nq = num_qubits(&c);
    let mut out: Vec<Option<Inst>> = Vec::with_capacity(c.len());
    let mut onwire: Vec<Vec<usize>> = vec![vec![]; nq];
    let mut syncpoint = 0usize; // 1-based, 0 for none
    let mut ptrs: Vec<usize> = vec![];
    let mut changed = false;

    for inst in c {
        let mut done = false;
        if ctx.plain(inst.entry) {
            ptrs.clear();
            for &q in &inst.qubits {
                ptrs.push(onwire[q].len());
            }
            let mut step = 0;
            loop {
                // 1-based index of the newest unvisited instruction on the wires
                let mut j = 0;
                for (k, &q) in inst.qubits.iter().enumerate() {
                    if ptrs[k] > 0 {
                        j = j.max(onwire[q][ptrs[k] - 1]);
                    }
                }
                for (k, &q) in inst.qubits.iter().enumerate() {
                    if ptrs[k] > 0 && onwire[q][ptrs[k] - 1] == j {
                        ptrs[k] -= 1;
                    }
                }
                step += 1;
                if j <= syncpoint || step > LOOKBACK {
                    break;
                }
                let Some(prev) = out[j - 1].clone() else { continue };
                if ctx.plain(prev.entry) && same_qubits(&prev.qubits, &inst.qubits) {
                    let action = if merge {
                        ctx.merge_pair(&prev, &inst)?
                    } else {
                        ctx.cancel_pair(&prev, &inst)?
                    };
                    match action {
                        Action::Cancel => {
                            out[j - 1] = None;
                            done = true;
                            break;
                        }
                        Action::Replace(e) => {
                            out[j - 1] = Some(Inst::new(e, prev.qubits));
                            done = true;
                            break;
                        }
                        Action::Keep => {}
                    }
                }
                if !ctx.commutes(&prev, &inst) {
                    break;
                }
            }
        }
        if done {
            changed = true;
            continue;
        }
        let dq = ctx.dag_qubits(&inst, nq);
        out.push(Some(inst));
        let idx = out.len();
        if ctx.entries[out[idx - 1].as_ref().unwrap().entry].facts.global {
            syncpoint = idx;
        }
        for q in dq {
            onwire[q].push(idx);
        }
    }
    Ok((out.into_iter().flatten().collect(), changed))
}

fn same_qubits(a: &[usize], b: &[usize]) -> bool {
    a.len() == b.len() && a.iter().all(|q| b.contains(q))
}

fn resynthesize_1q(ctx: &mut Ctx, c: Vec<Inst>) -> (Vec<Inst>, bool) {
    let nq = num_qubits(&c);
    let mut out: Vec<Option<Inst>> = Vec::with_capacity(c.len());
    let mut runs: HashMap<usize, Vec<(usize, Mat)>> = HashMap::new();
    let mut changed = false;

    // The run's gates are adjacent on their wire, so the replacement can take
    // the place of the first one.
    let mut flush = |ctx: &mut Ctx,
                     out: &mut Vec<Option<Inst>>,
                     runs: &mut HashMap<usize, Vec<(usize, Mat)>>,
                     q: usize| {
        let Some(run) = runs.remove(&q) else { return };
        if run.len() < 2 {
            return;
        }
        let mut u = run.last().unwrap().1.clone();
        for (_, m) in run.iter().rev().skip(1) {
            u = u.mul(m);
        }
        for (j, _) in &run {
            out[*j] = None;
        }
        changed = true;
        if !u.is_identity() {
            let (t, p, l, g) = zyz(&u);
            let e = ctx.add_code(Code::U, vec![t, p, l, g]);
            out[run[0].0] = Some(Inst::new(e, vec![q]));
        }
    };

    for inst in c {
        let m = if inst.qubits.len() == 1 {
            ctx.mat(inst.entry).cloned()
        } else {
            None
        };
        match m {
            None => {
                for q in ctx.dag_qubits(&inst, nq) {
                    flush(ctx, &mut out, &mut runs, q);
                }
                out.push(Some(inst));
            }
            Some(m) => {
                let q = inst.qubits[0];
                out.push(Some(inst));
                runs.entry(q).or_default().push((out.len() - 1, m));
            }
        }
    }
    let qs: Vec<usize> = runs.keys().copied().collect();
    for q in qs {
        flush(ctx, &mut out, &mut runs, q);
    }
    (out.into_iter().flatten().collect(), changed)
}

fn is_two_qubit(i: &Inst) -> bool {
    i.qubits.len() == 2
}

/// `_fusiondata` with two-qubit budgets: what a block may absorb.
fn dense_data(ctx: &Ctx, inst: &Inst) -> Data {
    let e = &ctx.entries[inst.entry];
    if !e.facts.fusable || e.facts.nq > 2 {
        return Data::Boundary;
    }
    match e.facts.matrix.as_ref() {
        None => Data::Boundary,
        Some(m) if m.is_diagonal() => Data::Diagonal,
        Some(_) => Data::Dense,
    }
}

fn local(qubits: &[usize], support: &[usize]) -> Vec<usize> {
    qubits
        .iter()
        .map(|q| support.iter().position(|s| s == q).unwrap())
        .collect()
}

fn resynthesize_2q(ctx: &mut Ctx, c: Vec<Inst>) -> (Vec<Inst>, bool) {
    let cl = clusters(ctx, &c, 2, 2, &dense_data);
    let mut out = Vec::with_capacity(c.len());
    let mut changed = false;
    for &cid in &cl.order {
        let mut members = cl.members[cid].clone();
        members.sort_unstable();
        if members.is_empty() {
            continue;
        }
        let s: Vec<usize> = cl.support[cid].iter().copied().collect();
        let n2q = members.iter().filter(|&&i| is_two_qubit(&c[i])).count();
        if cl.fuse[cid] && s.len() == 2 && n2q >= 2 {
            let ms: Vec<Mat> = members
                .iter()
                .map(|&i| {
                    embed(
                        ctx.entries[c[i].entry].facts.matrix.as_ref().unwrap(),
                        &local(&c[i].qubits, &s),
                        2,
                    )
                })
                .collect();
            if let Some((ncx, gates)) = kak_synthesis(&fold(&ms, 4)) {
                if ncx < n2q {
                    for (code, params, qs) in gates {
                        let e = ctx.add_code(code, params);
                        out.push(Inst::new(e, qs.iter().map(|&q| s[q]).collect()));
                    }
                    changed = true;
                    continue;
                }
            }
        }
        for &i in &members {
            out.push(c[i].clone());
        }
    }
    (out, changed)
}

fn linear_data(ctx: &Ctx, inst: &Inst) -> Data {
    if ctx.entries[inst.entry].facts.linear != 0 {
        Data::Dense
    } else {
        Data::Boundary
    }
}

fn resynthesize_linear(ctx: &mut Ctx, c: Vec<Inst>) -> (Vec<Inst>, bool) {
    let nq = num_qubits(&c);
    let cl = clusters(ctx, &c, nq, nq, &linear_data);
    let mut out = Vec::with_capacity(c.len());
    let mut changed = false;
    for &cid in &cl.order {
        let mut members = cl.members[cid].clone();
        members.sort_unstable();
        if members.is_empty() {
            continue;
        }
        if cl.fuse[cid] && members.len() > 1 {
            let s: Vec<usize> = cl.support[cid].iter().copied().collect();
            let gates: Vec<(u8, Vec<usize>)> = members
                .iter()
                .map(|&i| (ctx.entries[c[i].entry].facts.linear, local(&c[i].qubits, &s)))
                .collect();
            let (a, b) = affine_map(&gates, s.len());
            if let Some((cx, xs)) = linear_synthesis(&a, &b, s.len()) {
                let old = (
                    members.iter().filter(|&&i| is_two_qubit(&c[i])).count(),
                    members.len(),
                );
                if (cx.len(), cx.len() + xs.len()) < old {
                    let ecx = ctx.add_code(Code::CX, vec![]);
                    let ex = ctx.add_code(Code::X, vec![]);
                    for (ctrl, t) in cx {
                        out.push(Inst::new(ecx, vec![s[ctrl], s[t]]));
                    }
                    for q in xs {
                        out.push(Inst::new(ex, vec![s[q]]));
                    }
                    changed = true;
                    continue;
                }
            }
        }
        for &i in &members {
            out.push(c[i].clone());
        }
    }
    (out, changed)
}

fn cost(c: &[Inst]) -> (usize, usize) {
    (c.iter().filter(|i| is_two_qubit(i)).count(), c.len())
}

/// `simplify`: rounds of `passes` until a round no longer lowers the number of
/// two-qubit gates or, failing that, of instructions. A pass that already ran
/// on the current circuit is skipped until another pass changes it.
pub fn simplify(ctx: &mut Ctx, mut c: Vec<Inst>, passes: &[Pass], maxiter: usize) -> PyResult<Vec<Inst>> {
    let mut best = cost(&c);
    let mut version = 0usize;
    let mut seen: Vec<Option<usize>> = vec![None; passes.len()];
    for _ in 0..maxiter {
        let mut next = c.clone();
        for (k, &pass) in passes.iter().enumerate() {
            if seen[k] == Some(version) {
                continue;
            }
            let (d, changed) = run(ctx, pass, next)?;
            if changed {
                version += 1;
            }
            seen[k] = Some(version);
            next = d;
        }
        let nextcost = cost(&next);
        if nextcost >= best {
            break;
        }
        c = next;
        best = nextcost;
    }
    Ok(c)
}
