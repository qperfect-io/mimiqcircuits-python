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
//! What the passes know about each operation.
//!
//! The passes never look at gate types: Python describes every distinct
//! operation once by a few facts (its matrix, whether it is a multi-controlled
//! X, …), and the gates Rust creates itself come from a short fixed list.

use super::linalg::{ATOL, Mat, zyz};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Facts about an operation, as computed by the Python side.
#[derive(Clone, Debug)]
pub struct Facts {
    /// A unitary gate other than `Delay`: what the passes may move or rewrite.
    pub plain: bool,
    /// A unitary gate, `Delay` included: what `fuse`-style blocks may absorb.
    pub fusable: bool,
    pub nq: usize,
    /// Numeric matrix of a fusable gate on at most six qubits.
    pub matrix: Option<Mat>,
    /// Diagonal by structure; only read when there is no matrix.
    pub diag: bool,
    /// 1-based position of the target of a multi-controlled X, 0 otherwise.
    pub xtarget: usize,
    /// A symbolic X rotation.
    pub xaxis: bool,
    /// Gates of the same class merge by adding their first parameter; -1 for none.
    pub mergeclass: i64,
    /// That first parameter, NaN when symbolic.
    pub first: f64,
    /// 1 for X, 2 for CX, 3 for SWAP, 0 otherwise.
    pub linear: u8,
    /// Depends on the whole register, like `Amplitude`.
    pub global: bool,
}

/// Gates Rust creates, by code. The Python side maps each code to its class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum Code {
    U = 0,
    P = 1,
    RZ = 2,
    RX = 3,
    RY = 4,
    H = 5,
    CX = 6,
    X = 7,
    T = 8,
    S = 9,
    Z = 10,
    SDG = 11,
    TDG = 12,
    SX = 13,
    SXDG = 14,
    Y = 15,
    SY = 16,
    SYDG = 17,
}

/// Where an operation comes from, so that Python can build it.
#[derive(Clone, Debug)]
pub enum Origin {
    /// An operation Python already has, by its index in Python's list.
    Py(usize),
    /// A gate Rust created.
    Code(Code, Vec<f64>),
    /// The operation of an entry with its first parameter replaced.
    Sum(usize, f64),
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub facts: Facts,
    pub origin: Origin,
    pub diag: bool,
    pub family: usize,
    pub angle: f64,
}

impl Entry {
    pub fn new(facts: Facts, origin: Origin) -> Entry {
        let m = if facts.plain { facts.matrix.as_ref() } else { None };
        let (family, angle) = m.map_or((0, 0.0), rotation_family);
        let diag = m.map_or(facts.diag, Mat::is_diagonal);
        Entry {
            facts,
            origin,
            diag,
            family,
            angle,
        }
    }

    /// The matrix the passes reason with: none for anything but a plain gate.
    pub fn mat(&self) -> Option<&Mat> {
        if self.facts.plain {
            self.facts.matrix.as_ref()
        } else {
            None
        }
    }

    pub fn xaxis(&self) -> bool {
        self.facts.xaxis || self.family == 2 || self.family == 5
    }
}

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn cis(x: f64) -> C64 {
    C64::from_polar(1.0, x)
}

pub fn u_matrix(t: f64, p: f64, l: f64, g: f64) -> Mat {
    let (co, si) = ((t / 2.0).cos(), (t / 2.0).sin());
    Mat::from_rows(
        2,
        &[
            cis(g) * co,
            -cis(g + l) * si,
            cis(g + p) * si,
            cis(g + p + l) * co,
        ],
    )
}

/// Matrix of a created gate, matching its definition in MimiqCircuits.
pub fn code_matrix(code: Code, params: &[f64]) -> Mat {
    let h = 1.0 / 2f64.sqrt();
    let o = c(0.0, 0.0);
    let one = c(1.0, 0.0);
    match code {
        Code::U => u_matrix(params[0], params[1], params[2], params[3]),
        Code::P => Mat::from_rows(2, &[one, o, o, cis(params[0])]),
        Code::RZ => Mat::from_rows(2, &[cis(-params[0] / 2.0), o, o, cis(params[0] / 2.0)]),
        Code::RX => {
            let (co, si) = ((params[0] / 2.0).cos(), (params[0] / 2.0).sin());
            Mat::from_rows(2, &[c(co, 0.0), c(0.0, -si), c(0.0, -si), c(co, 0.0)])
        }
        Code::RY => {
            let (co, si) = ((params[0] / 2.0).cos(), (params[0] / 2.0).sin());
            Mat::from_rows(2, &[c(co, 0.0), c(-si, 0.0), c(si, 0.0), c(co, 0.0)])
        }
        Code::H => Mat::from_rows(2, &[c(h, 0.0), c(h, 0.0), c(h, 0.0), c(-h, 0.0)]),
        Code::CX => Mat::from_rows(4, &[one, o, o, o, o, one, o, o, o, o, o, one, o, o, one, o]),
        Code::X => Mat::from_rows(2, &[o, one, one, o]),
        Code::Y => Mat::from_rows(2, &[o, c(0.0, -1.0), c(0.0, 1.0), o]),
        Code::Z => Mat::from_rows(2, &[one, o, o, c(-1.0, 0.0)]),
        Code::S => Mat::from_rows(2, &[one, o, o, c(0.0, 1.0)]),
        Code::SDG => Mat::from_rows(2, &[one, o, o, c(0.0, -1.0)]),
        Code::T => Mat::from_rows(2, &[one, o, o, cis(PI / 4.0)]),
        Code::TDG => Mat::from_rows(2, &[one, o, o, cis(-PI / 4.0)]),
        Code::SX => Mat::from_rows(2, &[c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)]),
        Code::SXDG => Mat::from_rows(2, &[c(0.5, -0.5), c(0.5, 0.5), c(0.5, 0.5), c(0.5, -0.5)]),
        Code::SY => Mat::from_rows(2, &[c(0.5, 0.5), c(-0.5, -0.5), c(0.5, 0.5), c(0.5, 0.5)]),
        Code::SYDG => Mat::from_rows(2, &[c(0.5, -0.5), c(0.5, -0.5), c(-0.5, 0.5), c(0.5, -0.5)]),
    }
}

/// Facts of a created gate. `classes` holds the merge classes Python gave the
/// additive codes (P, RZ, RX, RY).
pub fn code_facts(code: Code, params: &[f64], classes: &[i64; 4]) -> Facts {
    let m = code_matrix(code, params);
    let nq = if code == Code::CX { 2 } else { 1 };
    let (mergeclass, first) = match code {
        Code::P => (classes[0], params[0]),
        Code::RZ => (classes[1], params[0]),
        Code::RX => (classes[2], params[0]),
        Code::RY => (classes[3], params[0]),
        _ => (-1, f64::NAN),
    };
    Facts {
        plain: true,
        fusable: true,
        nq,
        matrix: Some(m),
        diag: false,
        xtarget: match code {
            Code::X => 1,
            Code::CX => 2,
            _ => 0,
        },
        xaxis: false,
        mergeclass,
        first,
        linear: match code {
            Code::X => 1,
            Code::CX => 2,
            _ => 0,
        },
        global: false,
    }
}

fn family_basis(k: usize) -> Mat {
    let h = 1.0 / 2f64.sqrt();
    match k {
        1 => Mat::identity(2),
        2 => Mat::from_rows(2, &[c(h, 0.0), c(h, 0.0), c(h, 0.0), c(-h, 0.0)]),
        _ => Mat::from_rows(2, &[c(h, 0.0), c(h, 0.0), c(0.0, h), c(0.0, -h)]),
    }
}

/// (family, λ) of a 2×2 unitary, or (0, 0.0).
///
/// Families 1 to 3 are B P(λ) B† and 4 to 6 are B RZ(λ) B†, with B the
/// identity, H, or the basis change that takes Z to Y.
pub fn rotation_family(m: &Mat) -> (usize, f64) {
    if m.n != 2 {
        return (0, 0.0);
    }
    for k in 1..=3 {
        let b = family_basis(k);
        let d = b.adjoint().mul(m).mul(&b);
        if !(d.get(0, 1).norm() <= ATOL && d.get(1, 0).norm() <= ATOL) {
            continue;
        }
        if (d.get(0, 0) - c(1.0, 0.0)).norm() <= ATOL {
            return (k, d.get(1, 1).arg());
        }
        if (d.get(0, 0) - d.get(1, 1).conj()).norm() <= ATOL {
            return (k + 3, 2.0 * d.get(1, 1).arg());
        }
    }
    (0, 0.0)
}

/// The gate of `family` with angle `λ`, or `None` for the identity.
pub fn family_gate(family: usize, lambda: f64) -> Option<(Code, Vec<f64>)> {
    let period = if family <= 3 { 8 } else { 16 };
    let q = lambda / (PI / 4.0);
    if (q - q.round_ties_even()).abs() <= 1e-9 {
        let k = (q.round_ties_even() as i64).rem_euclid(period);
        if k == 0 {
            return None;
        }
        let named = match (family, k) {
            (1, 1) => Some(Code::T),
            (1, 2) => Some(Code::S),
            (1, 4) => Some(Code::Z),
            (1, 6) => Some(Code::SDG),
            (1, 7) => Some(Code::TDG),
            (2, 2) => Some(Code::SX),
            (2, 4) => Some(Code::X),
            (2, 6) => Some(Code::SXDG),
            (3, 2) => Some(Code::SY),
            (3, 4) => Some(Code::Y),
            (3, 6) => Some(Code::SYDG),
            _ => None,
        };
        if let Some(code) = named {
            return Some((code, vec![]));
        }
    }
    match family {
        1 => Some((Code::P, vec![lambda])),
        4 => Some((Code::RZ, vec![lambda])),
        5 => Some((Code::RX, vec![lambda])),
        6 => Some((Code::RY, vec![lambda])),
        _ => {
            let b = family_basis(family);
            let p = Mat::from_rows(2, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), cis(lambda)]);
            let (t, ph, l, g) = zyz(&b.mul(&p).mul(&b.adjoint()));
            Some((Code::U, vec![t, ph, l, g]))
        }
    }
}
