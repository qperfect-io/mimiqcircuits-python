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
//! Regions: intervals, upright rectangles, ellipses and ellipse pairs, and
//! the two convex sets of the problem, the ε-region and the unit disk.

use super::Synthesizer;
use super::gridop::*;
use super::numeric::*;
use super::real::Real;
use super::rings::*;

#[derive(Clone, Debug)]
pub(super) struct Interval {
    pub(super) l: Real,
    pub(super) r: Real,
}

impl Interval {
    pub(super) fn new(l: Real, r: Real) -> Self {
        Interval { l, r }
    }

    pub(super) fn shift_sub(&self, x: &Real) -> Self {
        Interval::new(&self.l - x, &self.r - x)
    }

    pub(super) fn neg(&self) -> Self {
        Interval::new(-&self.r, -&self.l)
    }

    pub(super) fn mul(&self, x: &Real) -> Self {
        if *x >= 0 {
            Interval::new(&self.l * x, &self.r * x)
        } else {
            Interval::new(&self.r * x, &self.l * x)
        }
    }

    pub(super) fn div(&self, x: &Real) -> Self {
        if *x > 0 {
            Interval::new(&self.l / x, &self.r / x)
        } else {
            Interval::new(&self.r / x, &self.l / x)
        }
    }

    pub(super) fn width(&self) -> Real {
        &self.r - &self.l
    }

    pub(super) fn fatten(&self, e: &Real) -> Self {
        Interval::new(&self.l - e, &self.r + e)
    }

    pub(super) fn within(&self, x: &Real) -> bool {
        self.l <= *x && *x <= self.r
    }
}

#[derive(Clone, Debug)]
pub(super) struct Rectangle {
    pub(super) x: Interval,
    pub(super) y: Interval,
}

/// The ellipse {v : (v - p)ᵀ D (v - p) ≤ 1} with D = [a b; b d].
#[derive(Clone, Debug)]
pub(super) struct Ellipse {
    pub(super) a: Real,
    pub(super) b: Real,
    pub(super) d: Real,
    pub(super) px: Real,
    pub(super) py: Real,
}

impl Ellipse {
    fn skew(&self) -> Real {
        self.b.sq()
    }

    pub(super) fn bias(&self) -> Real {
        &self.d / &self.a
    }
}

#[derive(Clone, Debug)]
pub(super) struct EllipsePair {
    pub(super) a: Ellipse,
    pub(super) b: Ellipse,
}

impl EllipsePair {
    pub(super) fn skew(&self) -> Real {
        self.a.skew() + self.b.skew()
    }

    pub(super) fn bias(&self) -> Real {
        self.b.bias() / self.a.bias()
    }
}

/// A convex region of candidate top-left entries u: the disk |u|² ≤ scale,
/// optionally cut down to the ε-cap {u : zx·Re u + zy·Im u ≥ d} around
/// e^{-iθ/2} (Ross–Selinger §7).
#[derive(Clone, Debug)]
pub(super) struct Region {
    scale: ZRootTwo,
    cap: Option<(Real, Real, Real)>,
    pub(super) ellipse: Ellipse,
}

/// A region pair with the grid operator making it upright and the upright
/// bounding boxes of both regions.
pub(super) struct UprightPair {
    pub(super) set_a: Region,
    pub(super) set_b: Region,
    pub(super) op_g: GridOp,
    pub(super) bbox_a: Rectangle,
    pub(super) bbox_b: Rectangle,
}

impl Synthesizer {
    fn sqrt_det(&self, e: &Ellipse) -> Real {
        (&(&e.d * &e.a) - &e.b.sq()).sqrt()
    }

    pub(super) fn bbox(&self, e: &Ellipse) -> Rectangle {
        let s = self.sqrt_det(e);
        let w = e.d.sqrt() / &s;
        let h = e.a.sqrt() / &s;
        Rectangle {
            x: Interval::new(&e.px - &w, &e.px + &w),
            y: Interval::new(&e.py - &h, &e.py + &h),
        }
    }

    pub(super) fn normalize(&self, e: &Ellipse) -> Ellipse {
        let s = self.sqrt_det(e);
        let t = s.sqrt();
        Ellipse {
            a: &e.a / &s,
            b: &e.b / &s,
            d: &e.d / &s,
            px: &e.px * &t,
            py: &e.py * &t,
        }
    }

    fn act_ellipse(&self, g: &GridOp, e: &Ellipse) -> Ellipse {
        let (m00, m01, m10, m11) = self.to_matrix(&g.inv());
        let a = &(&(&(&e.a * &m00) * &m00) + &(&(&(2 * &e.b) * &m00) * &m10))
            + &(&(&e.d * &m10) * &m10);
        let b = &(&(&(&e.a * &m00) * &m01) + &(&e.b * &(&(&m00 * &m11) + &(&m01 * &m10))))
            + &(&(&e.d * &m10) * &m11);
        let d = &(&(&(&e.a * &m01) * &m01) + &(&(&(2 * &e.b) * &m11) * &m01))
            + &(&(&e.d * &m11) * &m11);
        let (n00, n01, n10, n11) = self.to_matrix(g);
        Ellipse {
            a,
            b,
            d,
            px: &(&n00 * &e.px) + &(&n01 * &e.py),
            py: &(&n10 * &e.px) + &(&n11 * &e.py),
        }
    }

    pub(super) fn act_pair(&self, g: &GridOp, p: &EllipsePair) -> EllipsePair {
        EllipsePair {
            a: self.act_ellipse(g, &p.a),
            b: self.act_ellipse(&g.conj2(), &p.b),
        }
    }

    pub(super) fn epsilon_region(&self, theta: &Real, epsilon: &Real, scale: ZRootTwo) -> Region {
        let s = self.zr_real(&scale);
        let d = (1 - &(epsilon.sq() / 4)).sqrt() * s.sqrt();
        let (zy, zx) = (&(-theta) / 2).sin_cos();
        // D = R(z) · diag(64/ε⁴, 4/ε²)/scale · R(z)ᵀ
        let inv_eps = 1 / epsilon;
        let s1 = (64 * &inv_eps.powi(4)) / &s;
        let s2 = (4 * &inv_eps.sq()) / &s;
        let ea = &(&(&s1 * &zx) * &zx) + &(&(&s2 * &zy) * &zy);
        let eb = &(&zx * &zy) * &(&s1 - &s2);
        let ed = &(&(&s1 * &zy) * &zy) + &(&(&s2 * &zx) * &zx);
        let ellipse = Ellipse {
            a: ea,
            b: eb,
            d: ed,
            px: &d * &zx,
            py: &d * &zy,
        };
        Region {
            scale,
            cap: Some((d, zx, zy)),
            ellipse,
        }
    }

    pub(super) fn unit_disk(&self, scale: ZRootTwo) -> Region {
        let sinv = 1 / &self.zr_real(&scale);
        let zero = Real::zero(self.prec);
        let ellipse = Ellipse {
            a: sinv.clone(),
            b: zero.clone(),
            d: sinv,
            px: zero.clone(),
            py: zero,
        };
        Region {
            scale,
            cap: None,
            ellipse,
        }
    }

    pub(super) fn inside(&self, s: &Region, u: &DOmega) -> bool {
        if !(&u.conj() * u)
            .to_droottwo()
            .le(&DRootTwo::new(s.scale.clone(), 0))
        {
            return false;
        }
        match &s.cap {
            None => true,
            Some((d, zx, zy)) => &(zx * &self.real(u)) + &(zy * &self.imag(u)) >= *d,
        }
    }

    pub(super) fn intersect(&self, s: &Region, u0: &DOmega, v: &DOmega) -> Option<(Real, Real)> {
        let a = &v.conj() * v;
        let b = &v.conj().scale(&int(2)) * u0;
        let c = &(&u0.conj() * u0) - &DOmega::from_droottwo(&DRootTwo::new(s.scale.clone(), 0));
        let t = self.solve_quadratic(self.real(&a), self.real(&b), self.real(&c));
        let Some((d, zx, zy)) = &s.cap else {
            return t;
        };
        let vz = &(zx * &self.real(v)) + &(zy * &self.imag(v));
        let rhs = &(d - &(zx * &self.real(u0))) - &(zy * &self.imag(u0));
        let (t0, t1) = t?;
        if vz > 0 {
            let t2 = &rhs / &vz;
            if t0 > t2 {
                Some((t0, t1))
            } else {
                Some((t2, t1))
            }
        } else if vz < 0 {
            let t2 = &rhs / &vz;
            if t1 < t2 {
                Some((t0, t1))
            } else {
                Some((t0, t2))
            }
        } else if rhs <= 0 {
            Some((t0, t1))
        } else {
            None
        }
    }
}
