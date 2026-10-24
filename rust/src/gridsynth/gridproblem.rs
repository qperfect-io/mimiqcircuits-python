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
//! One- and two-dimensional grid problem solvers. Solvers enumerate solutions
//! in the reference order and hand each one to `f`; `f` returns true to stop
//! the enumeration, and the solver then returns true as well.

use super::Synthesizer;
use super::numeric::*;
use super::real::Real;
use super::region::*;
use super::rings::*;

impl Synthesizer {
    /// One-dimensional grid problem: all β ∈ Z[√2] with β ∈ I, β• ∈ J.
    fn odgp_internal(
        &self,
        i: &Interval,
        j: &Interval,
        f: &mut dyn FnMut(ZRootTwo) -> bool,
    ) -> bool {
        if i.width() < 0 || j.width() < 0 {
            return false;
        }
        let (mut i, mut j, swap) = if i.width() > 0 && j.width() <= 0 {
            (j.clone(), i.clone(), true)
        } else {
            (i.clone(), j.clone(), false)
        };
        // Rescale by powers of λ until J has width in [1, λ): the solutions of
        // the scaled problem map back by multiplication with λ⁻ⁿ.
        let mut mult = ZRootTwo::of(1, 0);
        loop {
            let jw = j.width();
            let n = if jw <= 0 {
                0
            } else {
                self.floorlog(&jw, &self.lam)
            };
            if n == 0 {
                break;
            }
            let p = self.lambda_power(n);
            i = i.mul(&p.1);
            j = j.mul(&p.2);
            mult = &mult * &self.lambda_power(-n).0;
        }
        let s2 = &self.sqrt2;
        let alo = ((&i.l + &j.l) / 2).ceil();
        let ahi = ((&i.r + &j.r) / 2).floor();
        let mut a = alo;
        while a <= ahi {
            let ar = self.real_int(&a);
            let blo = ((s2 * &(&ar - &j.r)) / 2).ceil();
            let bhi = ((s2 * &(&ar - &j.l)) / 2).floor();
            let mut b = blo;
            while b <= bhi {
                let beta = &ZRootTwo::new(a.clone(), b.clone()) * &mult;
                if f(if swap { beta.conj2() } else { beta }) {
                    return true;
                }
                b += 1;
            }
            a += 1;
        }
        false
    }

    fn solve_odgp(&self, i: &Interval, j: &Interval, f: &mut dyn FnMut(ZRootTwo) -> bool) -> bool {
        if i.width() < 0 || j.width() < 0 {
            return false;
        }
        let alpha = ZRootTwo::new(
            ((&i.l + &j.l) / 2).floor(),
            ((&self.sqrt2 * &(&i.l - &j.l)) / 4).floor(),
        );
        let ii = i.shift_sub(&self.zr_real(&alpha));
        let jj = j.shift_sub(&self.zr_real(&alpha.conj2()));
        self.odgp_internal(&ii, &jj, &mut |beta| {
            let beta = &beta + &alpha;
            if i.within(&self.zr_real(&beta)) && j.within(&self.zr_real(&beta.conj2())) {
                f(beta)
            } else {
                false
            }
        })
    }

    fn solve_odgp_with_parity(
        &self,
        i: &Interval,
        j: &Interval,
        beta: &ZRootTwo,
        f: &mut dyn FnMut(ZRootTwo) -> bool,
    ) -> bool {
        let p = i64::from(beta.parity());
        let s2 = &self.sqrt2;
        let two = Real::from_i64(2, self.prec);
        let pr = Real::from_i64(p, self.prec);
        let ii = i.shift_sub(&pr).mul(s2).div(&two);
        let jj = j.shift_sub(&pr).mul(&(-s2)).div(&two);
        self.solve_odgp(&ii, &jj, &mut |alpha| {
            f(&(&alpha * &ZRootTwo::of(0, 1)) + &ZRootTwo::of(p, 0))
        })
    }

    fn solve_scaled_odgp(
        &self,
        i: &Interval,
        j: &Interval,
        k: i64,
        f: &mut dyn FnMut(DRootTwo) -> bool,
    ) -> bool {
        let scale = self.pow_sqrt2(k);
        let js = if k & 1 == 1 {
            j.neg().mul(&scale)
        } else {
            j.mul(&scale)
        };
        self.solve_odgp(&i.mul(&scale), &js, &mut |alpha| f(DRootTwo::new(alpha, k)))
    }

    fn solve_scaled_odgp_with_parity(
        &self,
        i: &Interval,
        j: &Interval,
        k: i64,
        beta: &DRootTwo,
        f: &mut dyn FnMut(DRootTwo) -> bool,
    ) -> bool {
        if k == 0 {
            let b = beta.renew(0).alpha;
            return self.solve_odgp_with_parity(i, j, &b, &mut |alpha| f(DRootTwo::new(alpha, 0)));
        }
        let p = beta.renew(k).parity();
        let offset = if p {
            power_of_inv_sqrt2(k)
        } else {
            DRootTwo::of_int(0)
        };
        let ii = i.shift_sub(&self.dr_real(&offset));
        let jj = j.shift_sub(&self.dr_real(&offset.conj2()));
        self.solve_scaled_odgp(&ii, &jj, k - 1, &mut |alpha| f(&alpha + &offset))
    }

    /// Two-dimensional grid problem: all z ∈ D[ω] with denominator exponent k
    /// such that z ∈ setA and z• ∈ setB, using the upright bounding boxes
    /// computed by `to_upright_set_pair`.
    pub(super) fn solve_tdgp(
        &self,
        sets: &UprightPair,
        k: i64,
        f: &mut dyn FnMut(DOmega) -> bool,
    ) -> bool {
        let op_g_inv = sets.op_g.inv();
        let mut alpha0 = None;
        self.solve_scaled_odgp(&sets.bbox_a.x, &sets.bbox_b.x, k + 1, &mut |a| {
            alpha0 = Some(a);
            true
        });
        let Some(alpha0) = alpha0 else {
            return false;
        };

        let dx = power_of_inv_sqrt2(k);
        let v = op_g_inv.apply_d(&from_droottwo_vector(&dx, &DRootTwo::of_int(0), k));
        let vc = v.conj2();
        let iy_a = sets.bbox_a.y.fatten(&(&sets.bbox_a.y.width() * 1e-4));
        let iy_b = sets.bbox_b.y.fatten(&(&sets.bbox_b.y.width() * 1e-4));
        let scale_k = self.real_int(&pow2(k as u64));
        let ten = Real::from_i64(10, self.prec);
        let (set_a, set_b) = (&sets.set_a, &sets.set_b);

        self.solve_scaled_odgp(&iy_a, &iy_b, k + 1, &mut |beta| {
            let z0 = op_g_inv.apply_d(&from_droottwo_vector(&alpha0, &beta, k + 1));
            let Some(ta) = self.intersect(set_a, &z0, &v) else {
                return false;
            };
            let Some(tb) = self.intersect(set_b, &z0.conj2(), &vc) else {
                return false;
            };
            let par = (&beta - &alpha0).renewing(k);
            let int_a = Interval::new(ta.0, ta.1);
            let int_b = Interval::new(tb.0, tb.1);
            let max_ten = |x: Real| if x > ten { x } else { ten.clone() };
            let dt_a = &ten / &max_ten(&scale_k * &int_b.width());
            let dt_b = &ten / &max_ten(&scale_k * &int_a.width());
            self.solve_scaled_odgp_with_parity(
                &int_a.fatten(&dt_a),
                &int_b.fatten(&dt_b),
                1,
                &par,
                &mut |alpha| {
                    let z = op_g_inv.apply_d(&from_droottwo_vector(
                        &(&(&alpha * &dx) + &alpha0),
                        &beta,
                        k,
                    ));
                    if self.inside(set_a, &z) && self.inside(set_b, &z.conj2()) {
                        f(z)
                    } else {
                        false
                    }
                },
            )
        })
    }
}
