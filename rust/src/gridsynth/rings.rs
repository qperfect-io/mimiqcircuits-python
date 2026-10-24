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
//! The rings Z[√2], D[√2], Z[ω] and D[ω], with the √2-conjugation `conj2`,
//! norms, Euclidean division and gcd.

use std::ops::{Add, Mul, Neg, Sub};

use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use super::numeric::*;

macro_rules! binops {
    ($T:ty, $Tr:ident, $m:ident, $f:path) => {
        impl $Tr<&$T> for &$T {
            type Output = $T;
            fn $m(self, y: &$T) -> $T {
                $f(self, y)
            }
        }
        impl $Tr<$T> for $T {
            type Output = $T;
            fn $m(self, y: $T) -> $T {
                $f(&self, &y)
            }
        }
        impl $Tr<&$T> for $T {
            type Output = $T;
            fn $m(self, y: &$T) -> $T {
                $f(&self, y)
            }
        }
        impl $Tr<$T> for &$T {
            type Output = $T;
            fn $m(self, y: $T) -> $T {
                $f(self, &y)
            }
        }
    };
}

// =============================================================================
// Z[√2]: a + b√2
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ZRootTwo {
    pub(super) a: Int,
    pub(super) b: Int,
}

impl ZRootTwo {
    pub(super) fn new(a: Int, b: Int) -> Self {
        ZRootTwo { a, b }
    }

    pub(super) fn of(a: i64, b: i64) -> Self {
        ZRootTwo::new(int(a), int(b))
    }

    pub(super) fn from_int(a: Int) -> Self {
        ZRootTwo::new(a, Int::zero())
    }

    fn add_(x: &Self, y: &Self) -> Self {
        ZRootTwo::new(&x.a + &y.a, &x.b + &y.b)
    }

    fn sub_(x: &Self, y: &Self) -> Self {
        ZRootTwo::new(&x.a - &y.a, &x.b - &y.b)
    }

    fn mul_(x: &Self, y: &Self) -> Self {
        ZRootTwo::new(
            &x.a * &y.a + ((&x.b * &y.b) << 1u32),
            &x.a * &y.b + &x.b * &y.a,
        )
    }

    fn scale(&self, n: &Int) -> Self {
        ZRootTwo::new(&self.a * n, &self.b * n)
    }

    pub(super) fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    /// Exact ordering of the real values a + b√2.
    pub(super) fn lt(&self, y: &Self) -> bool {
        let da = &self.a - &y.a;
        let db = &self.b - &y.b;
        let da2 = &da * &da;
        let db2 = (&db * &db) << 1u32;
        if db.is_positive() {
            da.is_negative() && da2 > db2
        } else if db.is_negative() {
            da.is_negative() || da2 < db2
        } else {
            da.is_negative()
        }
    }

    pub(super) fn conj2(&self) -> Self {
        ZRootTwo::new(self.a.clone(), -&self.b)
    }

    pub(super) fn znorm(&self) -> Int {
        &self.a * &self.a - ((&self.b * &self.b) << 1u32)
    }

    pub(super) fn parity(&self) -> bool {
        self.a.is_odd()
    }

    fn inv(&self) -> Self {
        let n = self.znorm();
        if n.is_one() {
            self.conj2()
        } else if n == int(-1) {
            -self.conj2()
        } else {
            panic!("ZRootTwo is not a unit")
        }
    }

    pub(super) fn pow(&self, n: i64) -> Self {
        if n < 0 {
            return self.inv().pow(-n);
        }
        let mut r = ZRootTwo::of(1, 0);
        let mut t = self.clone();
        let mut n = n;
        while n > 0 {
            if n & 1 == 1 {
                r = &r * &t;
            }
            t = &t * &t;
            n >>= 1;
        }
        r
    }
}

binops!(ZRootTwo, Add, add, ZRootTwo::add_);
binops!(ZRootTwo, Sub, sub, ZRootTwo::sub_);
binops!(ZRootTwo, Mul, mul, ZRootTwo::mul_);

impl Neg for ZRootTwo {
    type Output = ZRootTwo;
    fn neg(self) -> ZRootTwo {
        ZRootTwo::new(-self.a, -self.b)
    }
}

fn zr_divrem(x: &ZRootTwo, y: &ZRootTwo) -> (ZRootTwo, ZRootTwo) {
    let p = x * &y.conj2();
    let k = y.znorm();
    let q = ZRootTwo::new(rounddiv(&p.a, &k), rounddiv(&p.b, &k));
    let r = x - &(y * &q);
    (q, r)
}

pub(super) fn zr_div(x: &ZRootTwo, y: &ZRootTwo) -> ZRootTwo {
    zr_divrem(x, y).0
}

pub(super) fn zr_rem(x: &ZRootTwo, y: &ZRootTwo) -> ZRootTwo {
    zr_divrem(x, y).1
}

pub(super) fn zr_similar(x: &ZRootTwo, y: &ZRootTwo) -> bool {
    zr_rem(x, y).is_zero() && zr_rem(y, x).is_zero()
}

pub(super) fn zr_gcd(a: &ZRootTwo, b: &ZRootTwo) -> ZRootTwo {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let r = zr_rem(&a, &b);
        a = b;
        b = r;
    }
    a
}

pub(super) fn zr_sqrt(x: &ZRootTwo) -> Option<ZRootTwo> {
    let n = x.znorm();
    if n.is_negative() || x.a.is_negative() {
        return None;
    }
    let r = floorsqrt(&n);
    let a1 = floorsqrt(&(&x.a + &r).div_floor(&int(2)));
    let b1 = floorsqrt(&(&x.a - &r).div_floor(&int(4)));
    let a2 = floorsqrt(&(&x.a - &r).div_floor(&int(2)));
    let b2 = floorsqrt(&(&x.a + &r).div_floor(&int(4)));
    let (w1, w2) = if sign(&x.a) * sign(&x.b) >= 0 {
        (ZRootTwo::new(a1, b1), ZRootTwo::new(a2, b2))
    } else {
        (ZRootTwo::new(a1, -b1), ZRootTwo::new(a2, -b2))
    };
    if *x == &w1 * &w1 {
        return Some(w1);
    }
    if *x == &w2 * &w2 {
        return Some(w2);
    }
    None
}

pub(super) fn lambda() -> ZRootTwo {
    ZRootTwo::of(1, 1)
}

// =============================================================================
// D[√2]: α / √2^k with α ∈ Z[√2]
// =============================================================================

#[derive(Clone, Debug)]
pub(super) struct DRootTwo {
    pub(super) alpha: ZRootTwo,
    pub(super) k: i64,
}

impl DRootTwo {
    pub(super) fn new(alpha: ZRootTwo, k: i64) -> Self {
        DRootTwo { alpha, k }
    }

    pub(super) fn of_int(n: i64) -> Self {
        DRootTwo::new(ZRootTwo::of(n, 0), 0)
    }

    fn mul_by_sqrt2_power(&self, d: i64) -> Self {
        let (a, b) = (&self.alpha.a, &self.alpha.b);
        if d >= 0 {
            let mut alpha = self.alpha.scale(&pow2((d >> 1) as u64));
            if d & 1 == 1 {
                alpha = &alpha * &ZRootTwo::of(0, 1);
            }
            return DRootTwo::new(alpha, self.k);
        }
        let h = ((-d) >> 1) as u64;
        if (-d) & 1 == 0 {
            assert!(
                divisible_pow2(a, h) && divisible_pow2(b, h),
                "not divisible"
            );
            return DRootTwo::new(ZRootTwo::new(a >> h, b >> h), self.k);
        }
        assert!(
            divisible_pow2(a, h + 1) && divisible_pow2(b, h),
            "not divisible"
        );
        DRootTwo::new(ZRootTwo::new(b >> h, a >> (h + 1)), self.k)
    }

    pub(super) fn renew(&self, k: i64) -> Self {
        DRootTwo::new(self.mul_by_sqrt2_power(k - self.k).alpha, k)
    }

    pub(super) fn parity(&self) -> bool {
        self.alpha.parity()
    }

    pub(super) fn conj2(&self) -> Self {
        let c = self.alpha.conj2();
        DRootTwo::new(if self.k & 1 == 1 { -c } else { c }, self.k)
    }

    pub(super) fn renewing(&self, d: i64) -> Self {
        assert!(d <= self.k, "exponent too large");
        DRootTwo::new(self.alpha.clone(), self.k - d)
    }

    fn lt(&self, y: &Self) -> bool {
        let (x, y) = align_dr(self, y);
        x.alpha.lt(&y.alpha)
    }

    fn eq(&self, y: &Self) -> bool {
        let (x, y) = align_dr(self, y);
        x.alpha == y.alpha
    }

    pub(super) fn le(&self, y: &Self) -> bool {
        self.lt(y) || self.eq(y)
    }

    fn add_(x: &Self, y: &Self) -> Self {
        let (x, y) = align_dr(x, y);
        DRootTwo::new(&x.alpha + &y.alpha, x.k)
    }

    fn sub_(x: &Self, y: &Self) -> Self {
        let (x, y) = align_dr(x, y);
        DRootTwo::new(&x.alpha - &y.alpha, x.k)
    }

    fn mul_(x: &Self, y: &Self) -> Self {
        DRootTwo::new(&x.alpha * &y.alpha, x.k + y.k)
    }
}

binops!(DRootTwo, Add, add, DRootTwo::add_);
binops!(DRootTwo, Sub, sub, DRootTwo::sub_);
binops!(DRootTwo, Mul, mul, DRootTwo::mul_);

fn align_dr(x: &DRootTwo, y: &DRootTwo) -> (DRootTwo, DRootTwo) {
    if x.k < y.k {
        (x.renew(y.k), y.clone())
    } else if x.k > y.k {
        (x.clone(), y.renew(x.k))
    } else {
        (x.clone(), y.clone())
    }
}

pub(super) fn power_of_inv_sqrt2(k: i64) -> DRootTwo {
    DRootTwo::new(ZRootTwo::of(1, 0), k)
}

// =============================================================================
// Z[ω]: aω³ + bω² + cω + d
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ZOmega {
    pub(super) a: Int,
    pub(super) b: Int,
    pub(super) c: Int,
    pub(super) d: Int,
}

impl ZOmega {
    pub(super) fn new(a: Int, b: Int, c: Int, d: Int) -> Self {
        ZOmega { a, b, c, d }
    }

    pub(super) fn of(a: i64, b: i64, c: i64, d: i64) -> Self {
        ZOmega::new(int(a), int(b), int(c), int(d))
    }

    pub(super) fn from_int(n: Int) -> Self {
        ZOmega::new(Int::zero(), Int::zero(), Int::zero(), n)
    }

    pub(super) fn from_zroottwo(x: &ZRootTwo) -> Self {
        ZOmega::new(-&x.b, Int::zero(), x.b.clone(), x.a.clone())
    }

    fn add_(x: &Self, y: &Self) -> Self {
        ZOmega::new(&x.a + &y.a, &x.b + &y.b, &x.c + &y.c, &x.d + &y.d)
    }

    fn sub_(x: &Self, y: &Self) -> Self {
        ZOmega::new(&x.a - &y.a, &x.b - &y.b, &x.c - &y.c, &x.d - &y.d)
    }

    fn mul_(x: &Self, y: &Self) -> Self {
        // ω⁴ = -1
        let d = &x.d * &y.d - &x.c * &y.a - &x.b * &y.b - &x.a * &y.c;
        let c = &x.d * &y.c + &x.c * &y.d - &x.b * &y.a - &x.a * &y.b;
        let b = &x.d * &y.b + &x.c * &y.c + &x.b * &y.d - &x.a * &y.a;
        let a = &x.d * &y.a + &x.c * &y.b + &x.b * &y.c + &x.a * &y.d;
        ZOmega::new(a, b, c, d)
    }

    fn scale(&self, n: &Int) -> Self {
        ZOmega::new(&self.a * n, &self.b * n, &self.c * n, &self.d * n)
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero() && self.c.is_zero() && self.d.is_zero()
    }

    pub(super) fn eq_int(&self, n: &Int) -> bool {
        self.a.is_zero() && self.b.is_zero() && self.c.is_zero() && self.d == *n
    }

    pub(super) fn pow(&self, n: i64) -> Self {
        assert!(n >= 0, "negative power {n}");
        let mut r = ZOmega::of(0, 0, 0, 1);
        let mut t = self.clone();
        let mut n = n;
        while n > 0 {
            if n & 1 == 1 {
                r = &r * &t;
            }
            t = &t * &t;
            n >>= 1;
        }
        r
    }

    pub(super) fn conj(&self) -> Self {
        ZOmega::new(-&self.c, -&self.b, -&self.a, self.d.clone())
    }

    pub(super) fn conj2(&self) -> Self {
        ZOmega::new(-&self.a, self.b.clone(), -&self.c, self.d.clone())
    }

    fn znorm(&self) -> Int {
        let (a, b, c, d) = (&self.a, &self.b, &self.c, &self.d);
        let s = a * a + b * b + c * c + d * d;
        let t = a * b + b * c + c * d - d * a;
        &s * &s - ((&t * &t) << 1u32)
    }

    pub(super) fn residue(&self) -> u8 {
        (u8::from(self.a.is_odd()) << 3)
            | (u8::from(self.b.is_odd()) << 2)
            | (u8::from(self.c.is_odd()) << 1)
            | u8::from(self.d.is_odd())
    }

    fn mul_by_omega(&self) -> Self {
        ZOmega::new(self.b.clone(), self.c.clone(), self.d.clone(), -&self.a)
    }

    fn mul_by_omega_inv(&self) -> Self {
        ZOmega::new(-&self.d, self.a.clone(), self.b.clone(), self.c.clone())
    }

    pub(super) fn mul_by_omega_power(&self, n: i64) -> Self {
        let mut x = self.clone();
        for _ in 0..(n & 7) {
            x = x.mul_by_omega();
        }
        x
    }

    pub(super) fn to_zroottwo(&self) -> ZRootTwo {
        assert!(
            self.b.is_zero() && self.a == -&self.c,
            "{self:?} is not in Z[√2]"
        );
        ZRootTwo::new(self.d.clone(), self.c.clone())
    }
}

binops!(ZOmega, Add, add, ZOmega::add_);
binops!(ZOmega, Sub, sub, ZOmega::sub_);
binops!(ZOmega, Mul, mul, ZOmega::mul_);

impl Neg for &ZOmega {
    type Output = ZOmega;
    fn neg(self) -> ZOmega {
        ZOmega::new(-&self.a, -&self.b, -&self.c, -&self.d)
    }
}

fn zo_divrem(x: &ZOmega, y: &ZOmega) -> (ZOmega, ZOmega) {
    let yc = y.conj();
    let p = &(&(x * &yc) * &yc.conj2()) * &y.conj2();
    let k = y.znorm();
    let q = ZOmega::new(
        rounddiv(&p.a, &k),
        rounddiv(&p.b, &k),
        rounddiv(&p.c, &k),
        rounddiv(&p.d, &k),
    );
    let r = x - &(y * &q);
    (q, r)
}

pub(super) fn zo_gcd(a: &ZOmega, b: &ZOmega) -> ZOmega {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let r = zo_divrem(&a, &b).1;
        a = b;
        b = r;
    }
    a
}

pub(super) fn zo_similar(x: &ZOmega, y: &ZOmega) -> bool {
    zo_divrem(x, y).1.is_zero() && zo_divrem(y, x).1.is_zero()
}

// =============================================================================
// D[ω]: u / √2^k with u ∈ Z[ω]
// =============================================================================

#[derive(Clone, Debug)]
pub(super) struct DOmega {
    pub(super) u: ZOmega,
    pub(super) k: i64,
}

impl DOmega {
    pub(super) fn new(u: ZOmega, k: i64) -> Self {
        DOmega { u, k }
    }

    pub(super) fn of_int(n: i64) -> Self {
        DOmega::new(ZOmega::from_int(int(n)), 0)
    }

    pub(super) fn from_droottwo(x: &DRootTwo) -> Self {
        DOmega::new(ZOmega::from_zroottwo(&x.alpha), x.k)
    }

    pub(super) fn to_droottwo(&self) -> DRootTwo {
        DRootTwo::new(self.u.to_zroottwo(), self.k)
    }

    pub(super) fn mul_by_inv_sqrt2(&self) -> Self {
        let u = &self.u;
        let bd = &u.b + &u.d;
        let ca = &u.c + &u.a;
        assert!(bd.is_even() && ca.is_even(), "not divisible");
        DOmega::new(
            ZOmega::new(
                (&u.b - &u.d) >> 1u32,
                ca >> 1u32,
                bd >> 1u32,
                (&u.c - &u.a) >> 1u32,
            ),
            self.k,
        )
    }

    fn mul_by_sqrt2_power(&self, d: i64) -> Self {
        let u = &self.u;
        if d >= 0 {
            let mut v = u.scale(&pow2((d >> 1) as u64));
            if d & 1 == 1 {
                v = &v * &ZOmega::of(-1, 0, 1, 0);
            }
            return DOmega::new(v, self.k);
        }
        if d == -1 {
            return self.mul_by_inv_sqrt2();
        }
        let h = ((-d) >> 1) as u64;
        if (-d) & 1 == 0 {
            assert!(
                divisible_pow2(&u.a, h)
                    && divisible_pow2(&u.b, h)
                    && divisible_pow2(&u.c, h)
                    && divisible_pow2(&u.d, h),
                "not divisible"
            );
            return DOmega::new(
                ZOmega::new(&u.a >> h, &u.b >> h, &u.c >> h, &u.d >> h),
                self.k,
            );
        }
        let (p, q, r, s) = (&u.b - &u.d, &u.c + &u.a, &u.b + &u.d, &u.c - &u.a);
        let h1 = h + 1;
        assert!(
            divisible_pow2(&p, h1)
                && divisible_pow2(&q, h1)
                && divisible_pow2(&r, h1)
                && divisible_pow2(&s, h1),
            "not divisible"
        );
        DOmega::new(ZOmega::new(p >> h1, q >> h1, r >> h1, s >> h1), self.k)
    }

    pub(super) fn renew(&self, k: i64) -> Self {
        DOmega::new(self.mul_by_sqrt2_power(k - self.k).u, k)
    }

    pub(super) fn reduce_denomexp(&self) -> Self {
        let u = &self.u;
        let k = self.k;
        let tz = |n: &Int| -> i64 { n.trailing_zeros().map_or(k, |t| t as i64) };
        let r = tz(&u.a).min(tz(&u.b)).min(tz(&u.c)).min(tz(&u.d));
        let mut newk = k - 2 * r;
        let m = (r + 1).max(0) as u64;
        if divisible_pow2(&(&u.c + &u.a), m) && divisible_pow2(&(&u.b + &u.d), m) {
            newk -= 1;
        }
        self.renew(newk.max(0))
    }

    fn add_(x: &Self, y: &Self) -> Self {
        let (x, y) = align_do(x, y);
        DOmega::new(&x.u + &y.u, x.k)
    }

    fn sub_(x: &Self, y: &Self) -> Self {
        let (x, y) = align_do(x, y);
        DOmega::new(&x.u - &y.u, x.k)
    }

    fn mul_(x: &Self, y: &Self) -> Self {
        DOmega::new(&x.u * &y.u, x.k + y.k)
    }

    fn mul_z(&self, y: &ZOmega) -> Self {
        DOmega::new(&self.u * y, self.k)
    }

    pub(super) fn scale(&self, n: &Int) -> Self {
        DOmega::new(self.u.scale(n), self.k)
    }

    pub(super) fn eq(&self, y: &Self) -> bool {
        let (x, y) = align_do(self, y);
        x.u == y.u
    }

    pub(super) fn mul_by_omega(&self) -> Self {
        DOmega::new(self.u.mul_by_omega(), self.k)
    }

    pub(super) fn mul_by_omega_inv(&self) -> Self {
        DOmega::new(self.u.mul_by_omega_inv(), self.k)
    }

    pub(super) fn mul_by_omega_power(&self, n: i64) -> Self {
        DOmega::new(self.u.mul_by_omega_power(n), self.k)
    }

    pub(super) fn residue(&self) -> u8 {
        self.u.residue()
    }

    pub(super) fn conj(&self) -> Self {
        DOmega::new(self.u.conj(), self.k)
    }

    pub(super) fn conj2(&self) -> Self {
        let c = self.u.conj2();
        DOmega::new(if self.k & 1 == 1 { -&c } else { c }, self.k)
    }
}

binops!(DOmega, Add, add, DOmega::add_);
binops!(DOmega, Sub, sub, DOmega::sub_);
binops!(DOmega, Mul, mul, DOmega::mul_);

pub(super) fn align_do(x: &DOmega, y: &DOmega) -> (DOmega, DOmega) {
    if x.k < y.k {
        (x.renew(y.k), y.clone())
    } else if x.k > y.k {
        (x.clone(), y.renew(x.k))
    } else {
        (x.clone(), y.clone())
    }
}

pub(super) fn from_droottwo_vector(x: &DRootTwo, y: &DRootTwo, k: i64) -> DOmega {
    (DOmega::from_droottwo(x) + DOmega::from_droottwo(y).mul_z(&ZOmega::of(0, 1, 0, 0))).renew(k)
}
