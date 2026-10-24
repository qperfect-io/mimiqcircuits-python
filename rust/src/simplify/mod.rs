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
//! Circuit simplification passes, the implementation behind
//! `mimiqcircuits.simplify`.
//!
//! The circuit crosses from Python once, as arrays: each instruction is an index
//! into a list of distinct operations plus its targets, and each operation comes
//! with a few facts (see `table::Facts`). The passes run here, and the result
//! goes back as "keep input instruction i" or "this operation on these qubits",
//! with the operations Rust created described by a code and parameters.

mod clusters;
mod core;
mod kak;
mod linalg;
mod linear;
mod passes;
mod table;

use self::core::{Ctx, Desc, Host, Inst, facts_entry};
use self::linalg::Mat;
use self::passes::Pass;
use self::table::Facts;
use num_complex::Complex64 as C64;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use std::panic::{AssertUnwindSafe, catch_unwind};

type FactsTuple<'py> = (
    bool,
    bool,
    usize,
    Option<Bound<'py, PyBytes>>,
    bool,
    usize,
    bool,
    i64,
    f64,
    u8,
    bool,
);

fn facts_from_py(t: FactsTuple) -> PyResult<Facts> {
    let (plain, fusable, nq, matrix, diag, xtarget, xaxis, mergeclass, first, linear, global) = t;
    let matrix = match matrix {
        None => None,
        Some(b) => {
            let bytes = b.as_bytes();
            let dim = 1usize << nq;
            if bytes.len() != dim * dim * 16 {
                return Err(PyValueError::new_err("matrix has the wrong size"));
            }
            let a = bytes
                .chunks_exact(16)
                .map(|ch| {
                    let re = f64::from_ne_bytes(ch[..8].try_into().unwrap());
                    let im = f64::from_ne_bytes(ch[8..].try_into().unwrap());
                    C64::new(re, im)
                })
                .collect();
            Some(Mat { n: dim, a })
        }
    };
    Ok(Facts {
        plain,
        fusable,
        nq,
        matrix,
        diag,
        xtarget,
        xaxis,
        mergeclass,
        first,
        linear,
        global,
    })
}

fn desc_to_py<'py>(py: Python<'py>, d: &Desc) -> PyResult<Bound<'py, PyTuple>> {
    match d {
        Desc::Py(k) => Ok((0u8, *k).into_pyobject(py)?),
        Desc::Code(code, params) => Ok((1u8, *code as u8, params.clone()).into_pyobject(py)?),
        Desc::Sum(base, v) => Ok((2u8, desc_to_py(py, base)?, *v).into_pyobject(py)?),
    }
}

/// An output instruction: (input index, -1, ()) or (-1, operation, qubits).
type OutInst = (i64, i64, Vec<usize>);

struct PyHost<'py> {
    host: Bound<'py, PyAny>,
}

impl Host for PyHost<'_> {
    fn merge(&mut self, a: Desc, b: Desc) -> PyResult<(usize, Facts)> {
        let py = self.host.py();
        let r = self
            .host
            .call_method1("merge", (desc_to_py(py, &a)?, desc_to_py(py, &b)?))?;
        let (k, t): (usize, FactsTuple) = r.extract()?;
        Ok((k, facts_from_py(t)?))
    }

    fn inverse(&mut self, a: Desc, b: Desc) -> PyResult<bool> {
        let py = self.host.py();
        self.host
            .call_method1("inverse", (desc_to_py(py, &a)?, desc_to_py(py, &b)?))?
            .extract()
    }
}

/// Run simplification passes on a circuit described by arrays.
///
/// - `facts`: one tuple per distinct operation, see `mimiqcircuits.simplify`.
/// - `ops`: per instruction, the index of its operation in `facts`.
/// - `qubits`, `bits`, `zvars`: the targets of all instructions, concatenated,
///   with `nqubits`, `nbits`, `nzvars` giving how many each instruction has.
/// - `classes`: merge classes of the gates P, RZ, RX and RY.
/// - `passes`: pass codes, run once in order when `maxiter` is 0, or in rounds
///   as `simplify` does otherwise.
/// - `host`: object with `merge(a, b)` and `inverse(a, b)`, called for gates
///   with symbolic parameters.
///
/// Returns the descriptions of the operations created here, in order after the
/// input ones, and the output instructions as `(i, -1, ())` for input
/// instruction `i` kept as is, or `(-1, op, qubits)` otherwise.
#[pyfunction(name = "simplify_circuit")]
#[pyo3(signature = (facts, ops, qubits, nqubits, bits, nbits, zvars, nzvars, classes, passes, maxiter, host))]
#[allow(clippy::too_many_arguments)]
pub fn py_simplify_circuit<'py>(
    py: Python<'py>,
    facts: Vec<FactsTuple<'py>>,
    ops: Vec<usize>,
    qubits: Vec<usize>,
    nqubits: Vec<usize>,
    bits: Vec<usize>,
    nbits: Vec<usize>,
    zvars: Vec<usize>,
    nzvars: Vec<usize>,
    classes: [i64; 4],
    passes: Vec<u8>,
    maxiter: usize,
    host: Bound<'py, PyAny>,
) -> PyResult<(Vec<Bound<'py, PyTuple>>, Vec<OutInst>)> {
    let n0 = facts.len();
    let mut entries = Vec::with_capacity(n0);
    for (k, t) in facts.into_iter().enumerate() {
        entries.push(facts_entry(facts_from_py(t)?, k));
    }
    let passes: Vec<Pass> = passes
        .iter()
        .map(|&p| Pass::from_code(p).ok_or_else(|| PyValueError::new_err(format!("unknown pass {p}"))))
        .collect::<PyResult<_>>()?;

    let n = ops.len();
    if [nqubits.len(), nbits.len(), nzvars.len()].iter().any(|&l| l != n) {
        return Err(PyValueError::new_err(
            "target counts do not match the instructions",
        ));
    }
    let (mut qi, mut bi, mut zi) = (0, 0, 0);
    let mut circuit = Vec::with_capacity(n);
    for i in 0..n {
        if ops[i] >= n0 {
            return Err(PyValueError::new_err("operation index out of range"));
        }
        let take = |v: &[usize], at: &mut usize, k: usize| -> PyResult<Vec<usize>> {
            let s = v
                .get(*at..*at + k)
                .ok_or_else(|| PyValueError::new_err("too few targets"))?;
            *at += k;
            Ok(s.to_vec())
        };
        circuit.push(Inst {
            entry: ops[i],
            qubits: take(&qubits, &mut qi, nqubits[i])?,
            bits: take(&bits, &mut bi, nbits[i])?,
            zvars: take(&zvars, &mut zi, nzvars[i])?,
            orig: Some(i),
        });
    }

    let mut pyhost = PyHost { host };
    let mut ctx = Ctx::new(entries, classes, &mut pyhost);
    // A failing internal invariant would be a bug; report it as an ordinary
    // exception rather than a PanicException, which escapes `except Exception`.
    let result = catch_unwind(AssertUnwindSafe(|| -> PyResult<Vec<Inst>> {
        if maxiter == 0 {
            let mut c = circuit;
            for &p in &passes {
                c = passes::run(&mut ctx, p, c)?.0;
            }
            Ok(c)
        } else {
            passes::simplify(&mut ctx, circuit, &passes, maxiter)
        }
    }))
    .map_err(|_| PyRuntimeError::new_err("simplify failed: internal error"))??;

    let descs = (n0..ctx.entries.len())
        .map(|id| desc_to_py(py, &ctx.desc(id)))
        .collect::<PyResult<Vec<_>>>()?;
    let out = result
        .into_iter()
        .map(|inst| match inst.orig {
            Some(i) => (i as i64, -1, vec![]),
            None => (-1, inst.entry as i64, inst.qubits),
        })
        .collect();
    Ok((descs, out))
}

/// Matrix of a gate Rust creates, row-major, for tests that compare it with
/// the Python definition.
#[pyfunction(name = "simplify_code_matrix")]
pub fn py_simplify_code_matrix(code: u8, params: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
    use self::table::Code::*;
    let all = [
        U, P, RZ, RX, RY, H, CX, X, T, S, Z, SDG, TDG, SX, SXDG, Y, SY, SYDG,
    ];
    let code = *all
        .get(code as usize)
        .ok_or_else(|| PyValueError::new_err("unknown code"))?;
    Ok(table::code_matrix(code, &params)
        .a
        .iter()
        .map(|z| (z.re, z.im))
        .collect())
}
