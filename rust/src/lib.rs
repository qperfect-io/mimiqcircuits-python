//
// Copyright © 2023-2025 QPerfect. All Rights Reserved.
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
//! Native extension of `mimiqcircuits`, imported as `mimiqcircuits._core`.

mod gridsynth;
mod simplify;

use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Clifford+T approximation of Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2})
/// up to a global phase, with operator-norm error at most `epsilon`.
///
/// Returns a word over `H`, `S`, `T`, `X` in matrix-product order (the last
/// letter is the first gate applied). The output depends only on `theta`,
/// `epsilon` and `seed`, which seeds the randomized factoring step.
///
/// Raises ValueError if `theta` is not finite or `epsilon` is not in (0, 1).
/// Runs without holding the GIL.
#[pyfunction(name = "gridsynth")]
#[pyo3(signature = (theta, epsilon, seed = 0))]
fn py_gridsynth(py: Python<'_>, theta: f64, epsilon: f64, seed: u64) -> PyResult<String> {
    if !theta.is_finite() {
        return Err(PyValueError::new_err(format!(
            "angle {theta} is not finite"
        )));
    }
    if !(epsilon > 0.0 && epsilon < 1.0) {
        return Err(PyValueError::new_err(format!(
            "epsilon must be in (0, 1), got {epsilon}"
        )));
    }
    // An internal invariant failing would be a bug; report it as an ordinary
    // exception rather than a PanicException, which derives from
    // BaseException and escapes `except Exception`.
    py.detach(|| {
        catch_unwind(AssertUnwindSafe(|| {
            gridsynth::gridsynth(theta, epsilon, seed, 10, 10)
        }))
    })
    .map_err(|_| PyRuntimeError::new_err("gridsynth failed: internal error"))
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_gridsynth, m)?)?;
    m.add_function(wrap_pyfunction!(simplify::py_simplify_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(simplify::py_simplify_code_matrix, m)?)?;
    Ok(())
}
