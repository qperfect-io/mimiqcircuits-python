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
//! Instructions, the operation table, and the relations between two gates.

use super::linalg::{ATOL, Mat, embed};
use super::table::{Code, Entry, Facts, Origin, code_facts};
use pyo3::PyResult;
use std::collections::HashMap;

/// An instruction: an entry of the operation table on some targets. `orig` is
/// the index of the input instruction it is, unchanged, if any.
#[derive(Clone, Debug)]
pub struct Inst {
    pub entry: usize,
    pub qubits: Vec<usize>,
    pub bits: Vec<usize>,
    pub zvars: Vec<usize>,
    pub orig: Option<usize>,
}

impl Inst {
    pub fn new(entry: usize, qubits: Vec<usize>) -> Inst {
        Inst {
            entry,
            qubits,
            bits: vec![],
            zvars: vec![],
            orig: None,
        }
    }
}

/// How Python can build an operation.
#[derive(Clone, Debug)]
pub enum Desc {
    Py(usize),
    Code(Code, Vec<f64>),
    Sum(Box<Desc>, f64),
}

/// The few things only Python can do: arithmetic on symbolic parameters.
pub trait Host {
    /// The merge of two operations of the same merge class, registered on the
    /// Python side: its index there and its facts.
    fn merge(&mut self, a: Desc, b: Desc) -> PyResult<(usize, Facts)>;
    /// Whether `b` is the inverse of `a`, compared structurally.
    fn inverse(&mut self, a: Desc, b: Desc) -> PyResult<bool>;
}

const RELATION_COMMUTES: u8 = 1;
const RELATION_INVERSE: u8 = 2;

/// How far back along the wires a gate looks for a partner.
pub const LOOKBACK: usize = 64;

pub struct Ctx<'h> {
    pub entries: Vec<Entry>,
    relations: HashMap<(u8, usize, usize, u64), bool>,
    codes: HashMap<(Code, Vec<u64>), usize>,
    sums: HashMap<(i64, u64), usize>,
    /// Merge classes of the additive codes P, RZ, RX, RY.
    pub classes: [i64; 4],
    pub host: &'h mut dyn Host,
}

/// What combining two gates gives.
pub enum Action {
    Keep,
    Cancel,
    Replace(usize),
}

impl<'h> Ctx<'h> {
    pub fn new(entries: Vec<Entry>, classes: [i64; 4], host: &'h mut dyn Host) -> Ctx<'h> {
        Ctx {
            entries,
            relations: HashMap::new(),
            codes: HashMap::new(),
            sums: HashMap::new(),
            classes,
            host,
        }
    }

    pub fn add_code(&mut self, code: Code, params: Vec<f64>) -> usize {
        let key = (code, params.iter().map(|x| x.to_bits()).collect());
        if let Some(&id) = self.codes.get(&key) {
            return id;
        }
        let facts = code_facts(code, &params, &self.classes);
        self.entries.push(Entry::new(facts, Origin::Code(code, params)));
        let id = self.entries.len() - 1;
        self.codes.insert(key, id);
        id
    }

    fn add_sum(&mut self, base: usize, value: f64, m: Option<Mat>) -> usize {
        let key = (self.entries[base].facts.mergeclass, value.to_bits());
        if let Some(&id) = self.sums.get(&key) {
            return id;
        }
        let mut facts = self.entries[base].facts.clone();
        facts.matrix = m;
        facts.first = value;
        self.entries.push(Entry::new(facts, Origin::Sum(base, value)));
        let id = self.entries.len() - 1;
        self.sums.insert(key, id);
        id
    }

    pub fn desc(&self, id: usize) -> Desc {
        match &self.entries[id].origin {
            Origin::Py(k) => Desc::Py(*k),
            Origin::Code(code, params) => Desc::Code(*code, params.clone()),
            Origin::Sum(base, v) => Desc::Sum(Box::new(self.desc(*base)), *v),
        }
    }

    pub fn plain(&self, id: usize) -> bool {
        self.entries[id].facts.plain
    }

    pub fn mat(&self, id: usize) -> Option<&Mat> {
        self.entries[id].mat()
    }

    /// Qubits an instruction depends on for ordering: all of them for the
    /// global observables.
    pub fn dag_qubits(&self, inst: &Inst, nq: usize) -> Vec<usize> {
        if self.entries[inst.entry].facts.global {
            (0..nq).collect()
        } else {
            inst.qubits.clone()
        }
    }

    fn joint_matrices(&self, a: &Inst, b: &Inst) -> (Mat, Mat) {
        let mut s: Vec<usize> = a.qubits.iter().chain(&b.qubits).copied().collect();
        s.sort_unstable();
        s.dedup();
        let locs = |inst: &Inst| -> Vec<usize> {
            inst.qubits
                .iter()
                .map(|q| s.iter().position(|x| x == q).unwrap())
                .collect()
        };
        (
            embed(self.mat(a.entry).unwrap(), &locs(a), s.len()),
            embed(self.mat(b.entry).unwrap(), &locs(b), s.len()),
        )
    }

    /// Commutation by structure, at any width, or `None` when structure says
    /// nothing: the rules for multi-controlled X gates.
    fn structural_commute(&self, a: &Inst, b: &Inst) -> Option<bool> {
        let (mut a, mut b) = (a, b);
        let (mut ta, mut tb) = (
            self.entries[a.entry].facts.xtarget,
            self.entries[b.entry].facts.xtarget,
        );
        if ta == 0 && tb == 0 {
            return None;
        }
        if ta == 0 {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut ta, &mut tb);
        }
        let x = a.qubits[ta - 1];
        if tb != 0 {
            let y = b.qubits[tb - 1];
            return Some(x == y || (!b.qubits.contains(&x) && !a.qubits.contains(&y)));
        }
        let eb = &self.entries[b.entry];
        if eb.diag {
            return Some(!b.qubits.contains(&x));
        }
        if eb.xaxis() {
            let q = b.qubits[0];
            return Some(q == x || !a.qubits.contains(&q));
        }
        None
    }

    pub fn commutes(&mut self, a: &Inst, b: &Inst) -> bool {
        if !(self.plain(a.entry) && self.plain(b.entry)) {
            return false;
        }
        if self.entries[a.entry].diag && self.entries[b.entry].diag {
            return true;
        }
        if let Some(r) = self.structural_commute(a, b) {
            return r;
        }
        if self.mat(a.entry).is_none() || self.mat(b.entry).is_none() {
            return false;
        }
        let layout = layout_code(&a.qubits, &b.qubits);
        if layout == 0 {
            return false;
        }
        let key = (RELATION_COMMUTES, a.entry, b.entry, layout);
        if let Some(&r) = self.relations.get(&key) {
            return r;
        }
        let (ma, mb) = self.joint_matrices(a, b);
        let r = ma.mul(&mb).dist(&mb.mul(&ma)) <= ATOL;
        self.relations.insert(key, r);
        r
    }

    /// `_cancelpair`: `b` follows `a` on the same set of qubits.
    pub fn cancel_pair(&mut self, a: &Inst, b: &Inst) -> PyResult<Action> {
        let (ta, tb) = (
            self.entries[a.entry].facts.xtarget,
            self.entries[b.entry].facts.xtarget,
        );
        if ta != 0 && tb != 0 {
            return Ok(if a.qubits[ta - 1] == b.qubits[tb - 1] {
                Action::Cancel
            } else {
                Action::Keep
            });
        }
        if self.mat(a.entry).is_none() || self.mat(b.entry).is_none() {
            if a.qubits != b.qubits {
                return Ok(Action::Keep);
            }
            let (da, db) = (self.desc(a.entry), self.desc(b.entry));
            return Ok(if self.host.inverse(da, db)? {
                Action::Cancel
            } else {
                Action::Keep
            });
        }
        let layout = layout_code(&a.qubits, &b.qubits);
        if layout == 0 {
            return Ok(Action::Keep);
        }
        let key = (RELATION_INVERSE, a.entry, b.entry, layout);
        let r = match self.relations.get(&key) {
            Some(&r) => r,
            None => {
                let (ma, mb) = self.joint_matrices(a, b);
                let r = mb.mul(&ma).is_identity();
                self.relations.insert(key, r);
                r
            }
        };
        Ok(if r { Action::Cancel } else { Action::Keep })
    }

    /// `_mergepair`: rotations of one class add their parameters, other gates of
    /// one rotation family their angles.
    pub fn merge_pair(&mut self, a: &Inst, b: &Inst) -> PyResult<Action> {
        if a.qubits != b.qubits {
            return Ok(Action::Keep);
        }
        let (ea, eb) = (&self.entries[a.entry], &self.entries[b.entry]);
        let class = ea.facts.mergeclass;
        if class >= 0 && class == eb.facts.mergeclass {
            if ea.facts.first.is_nan() || eb.facts.first.is_nan() {
                let (da, db) = (self.desc(a.entry), self.desc(b.entry));
                let (k, facts) = self.host.merge(da, db)?;
                let entry = Entry::new(facts, Origin::Py(k));
                let identity = entry.mat().is_some_and(Mat::is_identity);
                if identity {
                    return Ok(Action::Cancel);
                }
                self.entries.push(entry);
                return Ok(Action::Replace(self.entries.len() - 1));
            }
            // G(x) G(y) = G(x + y) for these gates, so the product is the
            // merged gate's matrix. Gates too wide for one still merge.
            let m = match (eb.mat(), ea.mat()) {
                (Some(mb), Some(ma)) => Some(mb.mul(ma)),
                _ => None,
            };
            if m.as_ref().is_some_and(Mat::is_identity) {
                return Ok(Action::Cancel);
            }
            let value = ea.facts.first + eb.facts.first;
            return Ok(Action::Replace(self.add_sum(a.entry, value, m)));
        }
        let fam = ea.family;
        if fam == 0 || fam != eb.family {
            return Ok(Action::Keep);
        }
        let lambda = ea.angle + eb.angle;
        Ok(match super::table::family_gate(fam, lambda) {
            None => Action::Cancel,
            Some((code, params)) => Action::Replace(self.add_code(code, params)),
        })
    }
}

/// Positions of the qubits of `qa` and `qb` within their sorted union, packed
/// in one integer, or 0 when the union is wider than four qubits.
pub fn layout_code(qa: &[usize], qb: &[usize]) -> u64 {
    let n = qa.len() + qb.iter().filter(|q| !qa.contains(q)).count();
    if n > 4 {
        return 0;
    }
    let rank = |q: usize| -> u64 {
        1 + qa.iter().filter(|&&x| x < q).count() as u64
            + qb.iter().filter(|&&x| x < q && !qa.contains(&x)).count() as u64
    };
    let mut code: u64 = 1;
    for &q in qa {
        code = 8 * code + rank(q);
    }
    code *= 8;
    for &q in qb {
        code = 8 * code + rank(q);
    }
    code
}

/// Number of qubits a circuit touches: one past the highest index.
pub fn num_qubits(c: &[Inst]) -> usize {
    c.iter()
        .flat_map(|i| i.qubits.iter())
        .map(|q| q + 1)
        .max()
        .unwrap_or(0)
}

/// Facts of an operation that only the structure of `facts` describes.
pub fn facts_entry(facts: Facts, k: usize) -> Entry {
    Entry::new(facts, Origin::Py(k))
}
