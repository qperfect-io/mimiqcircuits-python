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
//! Grid operators: 2×2 matrices [u0 u1] acting on Z[ω] viewed as a lattice,
//! and the constant operators used by the Step Lemma.

use num_traits::{One, Signed, Zero};

use super::Synthesizer;
use super::numeric::*;
use super::real::Real;
use super::rings::*;

#[derive(Clone, Debug)]
pub(super) struct GridOp {
    pub(super) u0: ZOmega,
    pub(super) u1: ZOmega,
}

impl GridOp {
    fn of(u0: [i64; 4], u1: [i64; 4]) -> Self {
        GridOp {
            u0: ZOmega::of(u0[0], u0[1], u0[2], u0[3]),
            u1: ZOmega::of(u1[0], u1[1], u1[2], u1[3]),
        }
    }

    fn apply(&self, x: &ZOmega) -> ZOmega {
        let (a0, b0, c0, d0) = (&self.u0.a, &self.u0.b, &self.u0.c, &self.u0.d);
        let (a1, b1, c1, d1) = (&self.u1.a, &self.u1.b, &self.u1.c, &self.u1.d);
        let h = |v: Int| v >> 1u32;
        let nd = d0 * &x.d + d1 * &x.b + h(c1 - a1 + c0 - a0) * &x.c + h(c1 - a1 - c0 + a0) * &x.a;
        let nc = c0 * &x.d + c1 * &x.b + h(b1 + d1 + b0 + d0) * &x.c + h(b1 + d1 - b0 - d0) * &x.a;
        let nb = b0 * &x.d + b1 * &x.b + h(c1 + a1 + c0 + a0) * &x.c + h(c1 + a1 - c0 - a0) * &x.a;
        let na = a0 * &x.d + a1 * &x.b + h(b1 - d1 + b0 - d0) * &x.c + h(b1 - d1 - b0 + d0) * &x.a;
        ZOmega::new(na, nb, nc, nd)
    }

    pub(super) fn compose(&self, x: &GridOp) -> GridOp {
        GridOp {
            u0: self.apply(&x.u0),
            u1: self.apply(&x.u1),
        }
    }

    pub(super) fn apply_d(&self, x: &DOmega) -> DOmega {
        DOmega::new(self.apply(&x.u), x.k)
    }

    pub(super) fn conj2(&self) -> GridOp {
        GridOp {
            u0: self.u0.conj2(),
            u1: self.u1.conj2(),
        }
    }

    pub(super) fn inv(&self) -> GridOp {
        let (a0, b0, c0, d0) = (&self.u0.a, &self.u0.b, &self.u0.c, &self.u0.d);
        let (a1, b1, c1, d1) = (&self.u1.a, &self.u1.b, &self.u1.c, &self.u1.d);
        let v = &self.u0.conj() * &self.u1;
        assert!(
            (&v.a + &v.c).is_zero() && v.b.abs().is_one(),
            "grid operator is not special"
        );
        let h = |v: Int| v >> 1u32;
        let n0 = ZOmega::new(h(-c1 - a1 - c0 - a0), -b0, h(c1 + a1 - c0 - a0), b1.clone());
        let n1 = ZOmega::new(h(c1 - a1 + c0 - a0), d0.clone(), h(-c1 + a1 + c0 - a0), -d1);
        if v.b.is_negative() {
            GridOp { u0: -&n0, u1: -&n1 }
        } else {
            GridOp { u0: n0, u1: n1 }
        }
    }

    pub(super) fn pow(&self, n: i64) -> GridOp {
        if n < 0 {
            return self.inv().pow(-n);
        }
        let mut r = op_i();
        let mut t = self.clone();
        let mut n = n;
        while n > 0 {
            if n & 1 == 1 {
                r = r.compose(&t);
            }
            t = t.compose(&t);
            n >>= 1;
        }
        r
    }
}

pub(super) fn op_i() -> GridOp {
    GridOp::of([0, 0, 0, 1], [0, 1, 0, 0])
}
pub(super) fn op_z() -> GridOp {
    GridOp::of([0, 0, 0, 1], [0, -1, 0, 0])
}
pub(super) fn op_x() -> GridOp {
    GridOp::of([0, 1, 0, 0], [0, 0, 0, 1])
}
pub(super) fn op_s() -> GridOp {
    GridOp::of([-1, 0, 1, 1], [1, -1, 1, 0])
}
pub(super) fn op_r() -> GridOp {
    GridOp::of([0, 0, 1, 0], [1, 0, 0, 0])
}
pub(super) fn op_k() -> GridOp {
    GridOp::of([-1, -1, 0, 0], [0, -1, 1, 0])
}
pub(super) fn op_k_conj2() -> GridOp {
    GridOp::of([1, -1, 0, 0], [0, -1, -1, 0])
}
pub(super) fn op_sigma_l() -> GridOp {
    GridOp::of([-1, 0, 1, 1], [0, 1, 0, 0])
}
pub(super) fn op_sigma_r() -> GridOp {
    GridOp::of([0, 0, 0, 1], [1, -1, 1, 0])
}
pub(super) fn op_sigma_l_inv() -> GridOp {
    GridOp::of([-1, 0, 1, -1], [0, 1, 0, 0])
}
pub(super) fn op_sigma_r_inv() -> GridOp {
    GridOp::of([0, 0, 0, 1], [1, 1, 1, 0])
}

impl Synthesizer {
    pub(super) fn to_matrix(&self, g: &GridOp) -> (Real, Real, Real, Real) {
        (
            self.zreal(&g.u0),
            self.zreal(&g.u1),
            self.zimag(&g.u0),
            self.zimag(&g.u1),
        )
    }
}
