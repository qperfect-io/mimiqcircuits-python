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
// The gridsynth implementation (this module and its submodules other than
// `real`) is a port of pygridsynth 1.2.0
// (https://github.com/quantum-programming/pygridsynth), distributed under the
// MIT License:
//
//     Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka
//
//     Permission is hereby granted, free of charge, to any person obtaining a
//     copy of this software and associated documentation files (the
//     "Software"), to deal in the Software without restriction, including
//     without limitation the rights to use, copy, modify, merge, publish,
//     distribute, sublicense, and/or sell copies of the Software, and to
//     permit persons to whom the Software is furnished to do so, subject to
//     the following conditions:
//
//     The above copyright notice and this permission notice shall be included
//     in all copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//     TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//! Ross–Selinger gridsynth: Clifford+T approximation of Z-rotations.
//!
//! Notation follows the paper: ω = e^{iπ/4}, λ = 1 + √2, and `conj2` is the
//! √2-conjugation (√2 ↦ -√2, written x• in the paper). Gate sequences are
//! words over `H`, `S`, `T`, `X` in matrix-product order (leftmost gate is the
//! leftmost matrix factor, i.e. the last one applied).
//!
//! The order in which grid points are enumerated, the order of every
//! floating-point operation and the seeded random generator used for factoring
//! are all fixed, so the same angle, tolerance and seed always give the same
//! gate sequence.
//!
//! Reference: N. J. Ross and P. Selinger, "Optimal ancilla-free Clifford+T
//! approximation of z-rotations", Quantum Inf. Comput. 16, 901 (2016),
//! arXiv:1403.2975.

mod diophantine;
mod exact;
mod gridop;
mod gridproblem;
mod numeric;
mod real;
mod region;
mod rings;
mod rng;
mod upright;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use diophantine::*;
use exact::*;
use numeric::*;
use real::Real;
use region::*;
use rings::*;

// =============================================================================
// Approximate synthesis: the real-valued geometry at one precision
// =============================================================================

/// λⁿ with the real values of λⁿ and (λⁿ)•.
type LambdaPower = (ZRootTwo, Real, Real);

struct Synthesizer {
    prec: u64,
    sqrt2: Real,
    lam: Real,
    loglam: Real,
    // The grid solvers rescale by the same few powers of λ over and over.
    lambda_powers: RefCell<HashMap<i64, Rc<LambdaPower>>>,
}

impl Synthesizer {
    fn new(prec: u64) -> Self {
        let sqrt2 = Real::from_i64(2, prec).sqrt();
        let lam = &Real::from_i64(1, prec) + &sqrt2;
        let loglam = lam.ln();
        Synthesizer {
            prec,
            sqrt2,
            lam,
            loglam,
            lambda_powers: RefCell::new(HashMap::new()),
        }
    }

    fn lambda_power(&self, n: i64) -> Rc<LambdaPower> {
        if let Some(p) = self.lambda_powers.borrow().get(&n) {
            return Rc::clone(p);
        }
        let z = lambda().pow(n);
        let p = Rc::new((z.clone(), self.zr_real(&z), self.zr_real(&z.conj2())));
        self.lambda_powers.borrow_mut().insert(n, Rc::clone(&p));
        p
    }

    fn real_int(&self, n: &Int) -> Real {
        Real::int(n, self.prec)
    }

    // --- conversions to reals ------------------------------------------------

    fn pow_sqrt2(&self, k: i64) -> Real {
        let p = self.real_int(&pow2((k >> 1) as u64));
        if k & 1 == 1 { &p * &self.sqrt2 } else { p }
    }

    fn zr_real(&self, x: &ZRootTwo) -> Real {
        self.real_int(&x.a) + &self.sqrt2 * &self.real_int(&x.b)
    }

    fn dr_real(&self, x: &DRootTwo) -> Real {
        self.zr_real(&x.alpha) / self.pow_sqrt2(x.k)
    }

    fn zreal(&self, x: &ZOmega) -> Real {
        self.real_int(&x.d) + (&self.sqrt2 * &self.real_int(&(&x.c - &x.a))) / 2
    }

    fn zimag(&self, x: &ZOmega) -> Real {
        self.real_int(&x.b) + (&self.sqrt2 * &self.real_int(&(&x.c + &x.a))) / 2
    }

    fn real(&self, x: &DOmega) -> Real {
        self.zreal(&x.u) / self.pow_sqrt2(x.k)
    }

    fn imag(&self, x: &DOmega) -> Real {
        self.zimag(&x.u) / self.pow_sqrt2(x.k)
    }

    // --- top level -----------------------------------------------------------

    fn solve_fixed_k(
        &self,
        sets: &UprightPair,
        k: i64,
        has_phase: bool,
        seed: u64,
        dloop: u32,
        floop: u32,
    ) -> Option<DOmegaUnitary> {
        let mut result = None;
        self.solve_tdgp(sets, k, &mut |z| {
            if (&z * &z.conj()).residue() == 0 {
                return false;
            }
            let z = if has_phase {
                &z * &DOmega::new(ZOmega::of(0, -1, 1, 0), 1)
            } else {
                z
            };
            let xi = &DRootTwo::of_int(1) - &(&z.conj() * &z).to_droottwo();
            let Some(w) = diophantine_dyadic(&xi, seed, dloop, floop) else {
                return false;
            };
            let z = z.reduce_denomexp();
            let w = w.reduce_denomexp();
            let (z, w) = align_do(&z, &w);
            let k1 = (&z + &w).reduce_denomexp().k;
            let k2 = (&z + &w.mul_by_omega()).reduce_denomexp().k;
            let k3 = (&z + &w.mul_by_omega_inv()).reduce_denomexp().k;
            result = Some(if has_phase {
                if k1 <= k2 && k1 <= k3 {
                    DOmegaUnitary::new(z, w, -1)
                } else {
                    DOmegaUnitary::new(z, w.mul_by_omega_inv(), -1)
                }
            } else if k1 <= k2 {
                DOmegaUnitary::new(z, w, 0)
            } else {
                DOmegaUnitary::new(z, w.mul_by_omega(), 0)
            });
            true
        });
        result
    }

    fn gridsynth(
        &self,
        theta: &Real,
        epsilon: &Real,
        seed: u64,
        dloop: u32,
        floop: u32,
    ) -> Vec<u8> {
        let er0 = self.epsilon_region(theta, epsilon, ZRootTwo::of(1, 0));
        let ud0 = self.unit_disk(ZRootTwo::of(1, 0));
        // Solutions of the form e^{iπ/8}·U are searched for in the regions
        // scaled by (2 ± √2); they are equivalent up to global phase.
        let er1 = self.epsilon_region(theta, epsilon, ZRootTwo::of(2, 1));
        let ud1 = self.unit_disk(ZRootTwo::of(2, -1));
        let op_g = self.to_upright_ellipse_pair(&er0.ellipse, &ud0.ellipse);
        let sets0 = self.to_upright_set_pair(er0, ud0, &op_g);
        let sets1 = self.to_upright_set_pair(er1, ud1, &op_g);

        // Same (k, has_phase) visiting order as the reference implementation.
        let mut k = 0;
        let mut has_phase = false;
        loop {
            let sets = if has_phase { &sets1 } else { &sets0 };
            if let Some(u) = self.solve_fixed_k(sets, k, has_phase, seed, dloop, floop) {
                return decompose_domega_unitary(u);
            }
            if k >= 2 {
                has_phase = !has_phase;
                if has_phase {
                    k += 1;
                }
            } else if k == 0 && !has_phase {
                k = 1;
            } else if k == 1 && !has_phase {
                k = 0;
                has_phase = true;
            } else if k == 0 && has_phase {
                k = 1;
            } else {
                k = 2;
                has_phase = false;
            }
        }
    }
}

/// Working precision in bits needed to synthesize with error `epsilon` (the
/// digit count used by newsynth: 15 + 2.5⌈log₁₀(1/ε)⌉ decimal digits).
pub fn precision_bits(epsilon: f64) -> u64 {
    let digits = 15.0 + 2.5 * (-epsilon.log10()).ceil();
    (digits * std::f64::consts::LOG2_10).ceil() as u64
}

/// Clifford+T approximation of Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}) up to a
/// global phase, with operator-norm error at most `epsilon`.
///
/// `seed` seeds the randomized factoring step; `dloop` bounds the retries of
/// unsolved factors per candidate and `floop` the Pollard rho rounds per
/// factoring attempt. Returns a word over `H`, `S`, `T`, `X` in matrix-product
/// order. `theta` must be finite and `0 < epsilon < 1`.
pub fn gridsynth(theta: f64, epsilon: f64, seed: u64, dloop: u32, floop: u32) -> String {
    let prec = precision_bits(epsilon);
    let synth = Synthesizer::new(prec);
    let word = synth.gridsynth(
        &Real::from_f64(theta, prec),
        &Real::from_f64(epsilon, prec),
        seed,
        dloop,
        floop,
    );
    String::from_utf8(word).unwrap_or_default()
}
