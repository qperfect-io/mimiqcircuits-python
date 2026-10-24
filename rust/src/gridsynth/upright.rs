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
// This file is part of a port of pygridsynth 1.2.0
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
//! Reduction of an ellipse pair to upright form by repeated application of
//! the Step Lemma (Ross–Selinger Appendix A).

use num_traits::{One, ToPrimitive, Zero};

use super::Synthesizer;
use super::gridop::*;
use super::numeric::*;
use super::real::Real;
use super::region::*;
use super::rings::*;

impl Synthesizer {
    fn shift_ellipse_pair(&self, p: &EllipsePair, n: i64) -> EllipsePair {
        let ln = self.lambda_power(n).1.clone();
        let lin = self.lambda_power(-n).1.clone();
        let a = Ellipse {
            a: &p.a.a * &lin,
            b: p.a.b.clone(),
            d: &p.a.d * &ln,
            px: p.a.px.clone(),
            py: p.a.py.clone(),
        };
        let b = Ellipse {
            a: &p.b.a * &ln,
            b: if n & 1 == 1 { -&p.b.b } else { p.b.b.clone() },
            d: &p.b.d * &lin,
            px: p.b.px.clone(),
            py: p.b.py.clone(),
        };
        EllipsePair { a, b }
    }

    /// round(ln(bias) / ln(λ) / div), as the reference computes it at full
    /// precision. A double-precision estimate settles it unless it lies near
    /// a tie, where the full-precision value decides.
    fn round_log_bias(&self, bias: &Real, div: i64) -> i64 {
        let est = bias.log2_f64() / (1.0 + std::f64::consts::SQRT_2).log2() / div as f64;
        if (est - est.floor() - 0.5).abs() > 1e-6 {
            return est.round() as i64;
        }
        ((bias.ln() / &self.loglam) / div)
            .nint()
            .to_i64()
            .unwrap_or(0)
    }

    /// One step of the Step Lemma. Returns the new pair and operators, and
    /// whether the pair is already upright enough.
    // Written as the reference writes the comparisons; `Real` has no
    // `RangeInclusive<f64>::contains`.
    #[allow(clippy::manual_range_contains)]
    fn step_lemma(
        &self,
        p: EllipsePair,
        gl: GridOp,
        gr: GridOp,
    ) -> (EllipsePair, GridOp, GridOp, bool) {
        let apply = |op: GridOp| (self.act_pair(&op, &p), gl.clone(), op.compose(&gr), false);
        let bias_a = p.a.bias();
        let bias_b = p.b.bias();
        let bias_p = p.bias();
        let within = |x: &Real| *x >= 0.24410 && *x <= 4.0968;
        if p.b.b < 0 {
            apply(op_z())
        } else if &bias_a * &bias_b < 1 {
            apply(op_x())
        } else if bias_p > 33.971 || bias_p < 0.029437 {
            let n = self.round_log_bias(&bias_p, 8);
            apply(op_s().pow(n))
        } else if p.skew() <= 15 {
            (p, gl, gr, true)
        } else if bias_p > 5.8285 || bias_p < 0.17157 {
            let n = self.round_log_bias(&bias_p, 4);
            let p = self.shift_ellipse_pair(&p, n);
            if n >= 0 {
                (
                    p,
                    gl.compose(&op_sigma_l().pow(n)),
                    op_sigma_r().pow(n).compose(&gr),
                    false,
                )
            } else {
                (
                    p,
                    gl.compose(&op_sigma_l_inv().pow(-n)),
                    op_sigma_r_inv().pow(-n).compose(&gr),
                    false,
                )
            }
        } else if within(&bias_a) && within(&bias_b) {
            apply(op_r())
        } else if p.a.b >= 0 && bias_a <= 1.6969 {
            apply(op_k())
        } else if p.a.b >= 0 && bias_b <= 1.6969 {
            apply(op_k_conj2())
        } else {
            let m = if bias_b < bias_a { &bias_b } else { &bias_a };
            if p.a.b >= 0 {
                let n = floorsqrt(&(m / 4).floor()).max(Int::one());
                apply(GridOp {
                    u0: ZOmega::of(0, 0, 0, 1),
                    u1: ZOmega::new(Int::zero(), Int::one(), Int::zero(), n << 1u32),
                })
            } else {
                let n = floorsqrt(&(m / 2).floor()).max(Int::one());
                apply(GridOp {
                    u0: ZOmega::of(0, 0, 0, 1),
                    u1: ZOmega::new(n.clone(), Int::one(), -n, Int::zero()),
                })
            }
        }
    }

    pub(super) fn to_upright_ellipse_pair(&self, a: &Ellipse, b: &Ellipse) -> GridOp {
        let mut p = EllipsePair {
            a: self.normalize(a),
            b: self.normalize(b),
        };
        let mut gl = op_i();
        let mut gr = op_i();
        loop {
            let (np, ngl, ngr, done) = self.step_lemma(p, gl, gr);
            p = np;
            gl = ngl;
            gr = ngr;
            if done {
                break;
            }
        }
        gl.compose(&gr)
    }

    pub(super) fn to_upright_set_pair(
        &self,
        set_a: Region,
        set_b: Region,
        op_g: &GridOp,
    ) -> UprightPair {
        let p = self.act_pair(
            op_g,
            &EllipsePair {
                a: set_a.ellipse.clone(),
                b: set_b.ellipse.clone(),
            },
        );
        UprightPair {
            bbox_a: self.bbox(&p.a),
            bbox_b: self.bbox(&p.b),
            set_a,
            set_b,
            op_g: op_g.clone(),
        }
    }
}
