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
//! Binary floating point with a fixed number of mantissa bits.
//!
//! Gridsynth enumerates grid points by comparing and flooring real numbers, so
//! its output depends on the last bit of every intermediate result. This type
//! follows the contract of MPFR, whose operations are correctly rounded to
//! nearest, ties to even: `+ - * /` and `sqrt` are exact and then rounded
//! once, and `sin`, `cos` are evaluated with enough guard bits to round
//! correctly (Ziv's strategy). `ln` only feeds a rounding to the nearest integer, so it is
//! computed with guard bits but without the correctness loop.
//!
//! A value is `(-1)^neg · man · 2^exp`. Results carry at most `prec` mantissa
//! bits; integers converted with [`Real::int`] are kept exact, so mixing them
//! into an operation rounds once, as MPFR and mpmath do.

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};

#[derive(Clone, Debug)]
pub struct Real {
    neg: bool,
    man: BigUint,
    exp: i64,
    prec: u64,
}

impl Real {
    /// Rounds `(-1)^neg · man · 2^exp` to `prec` bits, to nearest, ties to even.
    fn make(neg: bool, man: BigUint, exp: i64, prec: u64) -> Real {
        if man.is_zero() {
            return Real::zero(prec);
        }
        let bits = man.bits();
        if bits <= prec {
            return Real {
                neg,
                man,
                exp,
                prec,
            };
        }
        let shift = bits - prec;
        let mut q = &man >> shift;
        let half = man.bit(shift - 1);
        let sticky = man.trailing_zeros().is_some_and(|tz| tz < shift - 1);
        let mut exp = exp + shift as i64;
        if half && (sticky || q.bit(0)) {
            q += 1u32;
            if q.bits() > prec {
                q >>= 1;
                exp += 1;
            }
        }
        Real {
            neg,
            man: q,
            exp,
            prec,
        }
    }

    pub fn zero(prec: u64) -> Real {
        Real {
            neg: false,
            man: BigUint::zero(),
            exp: 0,
            prec,
        }
    }

    /// The integer `n`, exactly (not rounded to `prec`).
    pub fn int(n: &BigInt, prec: u64) -> Real {
        let (sign, man) = n.clone().into_parts();
        Real {
            neg: sign == Sign::Minus,
            man,
            exp: 0,
            prec,
        }
    }

    pub fn from_i64(n: i64, prec: u64) -> Real {
        Real::int(&BigInt::from(n), prec)
    }

    /// The value of a finite `f64`, exactly.
    pub fn from_f64(x: f64, prec: u64) -> Real {
        assert!(x.is_finite(), "non-finite value {x}");
        let bits = x.to_bits();
        let neg = bits >> 63 == 1;
        let raw_exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        let (man, exp) = if raw_exp == 0 {
            (frac, -1074)
        } else {
            (frac | (1u64 << 52), raw_exp - 1075)
        };
        if man == 0 {
            return Real::zero(prec);
        }
        Real {
            neg,
            man: BigUint::from(man),
            exp,
            prec,
        }
    }

    /// The integer `n` at the precision of `self`.
    pub fn lit(&self, n: i64) -> Real {
        Real::from_i64(n, self.prec)
    }

    pub fn is_zero(&self) -> bool {
        self.man.is_zero()
    }

    pub fn is_neg(&self) -> bool {
        self.neg && !self.man.is_zero()
    }

    fn top(&self) -> i64 {
        self.exp + self.man.bits() as i64
    }

    fn signed_man(&self) -> BigInt {
        BigInt::from_biguint(
            if self.neg { Sign::Minus } else { Sign::Plus },
            self.man.clone(),
        )
    }

    fn add_impl(&self, y: &Real, negate_y: bool) -> Real {
        let prec = self.prec.max(y.prec);
        let yneg = y.neg ^ negate_y;
        if y.is_zero() {
            return Real::make(self.neg, self.man.clone(), self.exp, prec);
        }
        if self.is_zero() {
            return Real::make(yneg, y.man.clone(), y.exp, prec);
        }
        // x is the operand of larger magnitude, or at least not smaller top bit.
        let (xneg, xm, xe, yneg, ym, ye) = if self.top() >= y.top() {
            (self.neg, &self.man, self.exp, yneg, &y.man, y.exp)
        } else {
            (yneg, &y.man, y.exp, self.neg, &self.man, self.exp)
        };
        // Widen x to prec + 3 bits. When y lies entirely below the last of
        // them, only its sign and non-zeroness can affect the rounding, so it
        // is replaced by a small stand-in instead of shifting x by a huge gap.
        let ext = (prec + 3).saturating_sub(xm.bits());
        let xl = xe - ext as i64;
        let ytop = ye + ym.bits() as i64;
        let (xm2, ym2, e) = if ytop < xl {
            (xm << (ext + 2), BigUint::one(), xl - 2)
        } else {
            let e = xe.min(ye);
            (xm << (xe - e) as u64, ym << (ye - e) as u64, e)
        };
        if xneg == yneg {
            Real::make(xneg, xm2 + ym2, e, prec)
        } else {
            match xm2.cmp(&ym2) {
                Ordering::Greater => Real::make(xneg, xm2 - ym2, e, prec),
                Ordering::Less => Real::make(yneg, ym2 - xm2, e, prec),
                Ordering::Equal => Real::zero(prec),
            }
        }
    }

    fn mul_impl(&self, y: &Real) -> Real {
        let prec = self.prec.max(y.prec);
        Real::make(self.neg ^ y.neg, &self.man * &y.man, self.exp + y.exp, prec)
    }

    fn div_impl(&self, y: &Real) -> Real {
        let prec = self.prec.max(y.prec);
        assert!(!y.is_zero(), "division by zero");
        if self.is_zero() {
            return Real::zero(prec);
        }
        // Dividing by a power of two (the /2, /4 and √2^k of the grid
        // solvers) is exact before rounding, like the general path.
        if y.man.count_ones() == 1 {
            let tz = y.man.bits() as i64 - 1;
            return Real::make(
                self.neg ^ y.neg,
                self.man.clone(),
                self.exp - y.exp - tz,
                prec,
            );
        }
        // Quotient with at least prec + 2 bits, and a sticky bit for the rest.
        let shift = (prec as i64 + 2 + y.man.bits() as i64 - self.man.bits() as i64).max(0) as u64;
        let (q, r) = (&self.man << shift).div_rem(&y.man);
        let q = (q << 1u32) | BigUint::from(u32::from(!r.is_zero()));
        Real::make(
            self.neg ^ y.neg,
            q,
            self.exp - shift as i64 - y.exp - 1,
            prec,
        )
    }

    pub fn sq(&self) -> Real {
        self.mul_impl(self)
    }

    /// `self^n`, computed exactly and rounded once.
    pub fn powi(&self, n: u32) -> Real {
        let neg = self.neg && n % 2 == 1;
        Real::make(neg, self.man.pow(n), self.exp * i64::from(n), self.prec)
    }

    pub fn sqrt(&self) -> Real {
        assert!(!self.is_neg(), "sqrt of negative number");
        if self.is_zero() {
            return self.clone();
        }
        let prec = self.prec;
        let mut shift = (2 * (prec as i64 + 2) - self.man.bits() as i64).max(0);
        if (self.exp - shift).is_odd() {
            shift += 1;
        }
        let m = &self.man << shift as u64;
        let r = m.sqrt();
        let exact = &r * &r == m;
        let r = (r << 1u32) | BigUint::from(u32::from(!exact));
        Real::make(false, r, (self.exp - shift) / 2 - 1, prec)
    }

    pub fn floor(&self) -> BigInt {
        if self.exp >= 0 {
            return self.signed_man() << self.exp as u64;
        }
        // Arithmetic shift of a BigInt rounds towards negative infinity.
        self.signed_man() >> (-self.exp) as u64
    }

    pub fn ceil(&self) -> BigInt {
        -(-self).floor()
    }

    /// Nearest integer, ties to even.
    pub fn nint(&self) -> BigInt {
        if self.exp >= 0 {
            return self.floor();
        }
        let s = (-self.exp) as u64;
        let m = self.signed_man();
        let q = &m >> s;
        let r = &m - (&q << s);
        let half = BigInt::one() << (s - 1);
        match r.cmp(&half) {
            Ordering::Greater => q + 1,
            Ordering::Equal if q.is_odd() => q + 1,
            _ => q,
        }
    }

    /// The value as a fixed-point integer, floor(self · 2^w).
    fn to_fixed(&self, w: u64) -> BigInt {
        let m = self.signed_man();
        let e = self.exp + w as i64;
        if e >= 0 {
            m << e as u64
        } else {
            m >> (-e) as u64
        }
    }

    /// `(sin(x), cos(x))`, each correctly rounded to `prec` bits.
    pub fn sin_cos(&self) -> (Real, Real) {
        let prec = self.prec;
        if self.is_zero() {
            return (Real::zero(prec), Real::from_i64(1, prec));
        }
        let mut guard: u64 = 64;
        loop {
            let w = prec + guard;
            // Enough extra bits that the error of k·π/2 in the reduction stays
            // below 2^-w even for large arguments.
            let wp = w + self.top().max(0) as u64 + 16;
            let x = self.to_fixed(wp);
            let half_pi = fixed_pi(wp) >> 1u32;
            let k = rounddiv(&x, &half_pi);
            let r = &x - &k * &half_pi;
            let (s, c) = fixed_sin_cos(&r, wp);
            let (s, c) = match k.mod_floor(&BigInt::from(4)).try_into().unwrap_or(0u8) {
                0 => (s, c),
                1 => (c, -s),
                2 => (-s, -c),
                _ => (-c, s),
            };
            // The fixed-point results are within 2^(wp - w) units of the true
            // values; accept them when both ends of that interval round alike.
            let err = BigInt::one() << (wp - w);
            let sr = round_fixed_checked(&s, &err, wp, prec);
            let cr = round_fixed_checked(&c, &err, wp, prec);
            match (sr, cr) {
                (Some(sr), Some(cr)) => return (sr, cr),
                _ => {
                    // A tiny result needs as many extra bits as it has leading
                    // zeros below 1.
                    let lead = |v: &BigInt| (wp as i64 - v.bits() as i64).max(0) as u64;
                    guard = guard.max(lead(&s).max(lead(&c))) + 64;
                }
            }
        }
    }

    /// log2 of the value, to about double precision, for any magnitude.
    pub fn log2_f64(&self) -> f64 {
        let drop = self.man.bits().saturating_sub(64);
        let top = (&self.man >> drop).to_u64().unwrap_or(u64::MAX) as f64;
        top.log2() + (self.exp + drop as i64) as f64
    }

    /// Natural logarithm, to about `prec` bits. Only used where the result is
    /// rounded to an integer, so it is not guaranteed to be correctly rounded.
    pub fn ln(&self) -> Real {
        assert!(
            !self.is_neg() && !self.is_zero(),
            "log of non-positive number"
        );
        let prec = self.prec;
        // self = f · 2^e with f in [1, 2).
        let e = self.top() - 1;
        let w = prec + 64 + (64 - e.unsigned_abs().leading_zeros()) as u64;
        let one = BigInt::one() << w;
        let f = BigInt::from(self.man.clone()) << (w - (self.man.bits() - 1));
        // ln f = 2 atanh((f - 1) / (f + 1)), with (f - 1) / (f + 1) in [0, 1/3).
        let z = ((&f - &one) << w) / (&f + &one);
        let lnf = fixed_atanh(&z, w) << 1u32;
        let ln2 = fixed_atanh(&((&one) / BigInt::from(3)), w) << 1u32;
        let v = lnf + ln2 * BigInt::from(e);
        let (sign, man) = v.into_parts();
        Real::make(sign == Sign::Minus, man, -(w as i64), prec)
    }
}

fn rounddiv(x: &BigInt, y: &BigInt) -> BigInt {
    (x + (y >> 1u32)).div_floor(y)
}

/// Rounds the fixed-point value `v · 2^-w` to `prec` bits if every value in
/// `[v - err, v + err]` rounds to the same result.
fn round_fixed_checked(v: &BigInt, err: &BigInt, w: u64, prec: u64) -> Option<Real> {
    let round = |x: BigInt| {
        let (sign, man) = x.into_parts();
        Real::make(sign == Sign::Minus, man, -(w as i64), prec)
    };
    let lo = round(v - err);
    let hi = round(v + err);
    (lo.cmp(&hi) == Ordering::Equal && !lo.is_zero()).then_some(lo)
}

/// π · 2^w, truncated, from Machin's formula.
fn fixed_pi(w: u64) -> BigInt {
    let g = 16;
    let one = BigInt::one() << (w + g);
    let a = fixed_atan_inv(5, &one);
    let b = fixed_atan_inv(239, &one);
    ((a << 4u32) - (b << 2u32)) >> g
}

/// atan(1/n) scaled by `one`.
fn fixed_atan_inv(n: u32, one: &BigInt) -> BigInt {
    let n = BigInt::from(n);
    let n2 = &n * &n;
    let mut power = one / &n;
    let mut sum = power.clone();
    let mut k = 1u32;
    while !power.is_zero() {
        power = &power / &n2;
        let term = &power / BigInt::from(2 * k + 1);
        if k.is_odd() {
            sum -= term;
        } else {
            sum += term;
        }
        k += 1;
    }
    sum
}

/// atanh(z) for a fixed-point `z` (scale 2^w) with |z| < 1/2.
fn fixed_atanh(z: &BigInt, w: u64) -> BigInt {
    let z2 = (z * z) >> w;
    let mut power = z.clone();
    let mut sum = z.clone();
    let mut k = 1u32;
    while !power.is_zero() {
        power = (&power * &z2) >> w;
        sum += &power / BigInt::from(2 * k + 1);
        k += 1;
    }
    sum
}

/// Taylor series of sin and cos for a fixed-point `r` (scale 2^w), |r| ≤ π/4.
fn fixed_sin_cos(r: &BigInt, w: u64) -> (BigInt, BigInt) {
    let one = BigInt::one() << w;
    let r2 = (r * r) >> w;
    let mut s = r.clone();
    let mut term = r.clone();
    let mut n = 1u64;
    while !term.is_zero() {
        term = -((&term * &r2) >> w) / BigInt::from((2 * n) * (2 * n + 1));
        s += &term;
        n += 1;
    }
    let mut c = one.clone();
    let mut term = one;
    let mut n = 1u64;
    while !term.is_zero() {
        term = -((&term * &r2) >> w) / BigInt::from((2 * n - 1) * (2 * n));
        c += &term;
        n += 1;
    }
    (s, c)
}

impl PartialEq for Real {
    fn eq(&self, other: &Real) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Real {}

impl PartialOrd for Real {
    fn partial_cmp(&self, other: &Real) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Real {
    fn cmp(&self, y: &Real) -> Ordering {
        let sx = if self.is_zero() {
            0
        } else if self.neg {
            -1
        } else {
            1
        };
        let sy = if y.is_zero() {
            0
        } else if y.neg {
            -1
        } else {
            1
        };
        if sx != sy || sx == 0 {
            return sx.cmp(&sy);
        }
        let mag = match self.top().cmp(&y.top()) {
            Ordering::Equal => {
                let e = self.exp.min(y.exp);
                let a = &self.man << (self.exp - e) as u64;
                let b = &y.man << (y.exp - e) as u64;
                a.cmp(&b)
            }
            o => o,
        };
        if sx < 0 { mag.reverse() } else { mag }
    }
}

impl PartialEq<f64> for Real {
    fn eq(&self, other: &f64) -> bool {
        *self == Real::from_f64(*other, self.prec)
    }
}

impl PartialOrd<f64> for Real {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        Some(self.cmp(&Real::from_f64(*other, self.prec)))
    }
}

impl PartialEq<i64> for Real {
    fn eq(&self, other: &i64) -> bool {
        *self == self.lit(*other)
    }
}

impl PartialOrd<i64> for Real {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        if *other == 0 {
            return Some(if self.is_zero() {
                Ordering::Equal
            } else if self.neg {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        Some(self.cmp(&self.lit(*other)))
    }
}

impl Neg for &Real {
    type Output = Real;
    fn neg(self) -> Real {
        Real {
            neg: !self.neg && !self.man.is_zero(),
            man: self.man.clone(),
            exp: self.exp,
            prec: self.prec,
        }
    }
}

impl Neg for Real {
    type Output = Real;
    fn neg(self) -> Real {
        -&self
    }
}

macro_rules! real_binop {
    ($Tr:ident, $m:ident, |$x:ident, $y:ident| $body:expr) => {
        impl $Tr<&Real> for &Real {
            type Output = Real;
            fn $m(self, $y: &Real) -> Real {
                let $x = self;
                $body
            }
        }
        impl $Tr<Real> for Real {
            type Output = Real;
            fn $m(self, y: Real) -> Real {
                (&self).$m(&y)
            }
        }
        impl $Tr<&Real> for Real {
            type Output = Real;
            fn $m(self, y: &Real) -> Real {
                (&self).$m(y)
            }
        }
        impl $Tr<Real> for &Real {
            type Output = Real;
            fn $m(self, y: Real) -> Real {
                self.$m(&y)
            }
        }
        // Integers mix in exactly, as in MPFR and mpmath.
        impl $Tr<i64> for &Real {
            type Output = Real;
            fn $m(self, y: i64) -> Real {
                self.$m(&self.lit(y))
            }
        }
        impl $Tr<i64> for Real {
            type Output = Real;
            fn $m(self, y: i64) -> Real {
                (&self).$m(&self.lit(y))
            }
        }
        impl $Tr<&Real> for i64 {
            type Output = Real;
            fn $m(self, y: &Real) -> Real {
                y.lit(self).$m(y)
            }
        }
        impl $Tr<Real> for i64 {
            type Output = Real;
            fn $m(self, y: Real) -> Real {
                y.lit(self).$m(&y)
            }
        }
        // Float literals mix in exactly as well.
        impl $Tr<f64> for &Real {
            type Output = Real;
            fn $m(self, y: f64) -> Real {
                self.$m(&Real::from_f64(y, self.prec))
            }
        }
    };
}

real_binop!(Add, add, |x, y| x.add_impl(y, false));
real_binop!(Sub, sub, |x, y| x.add_impl(y, true));
real_binop!(Mul, mul, |x, y| x.mul_impl(y));
real_binop!(Div, div, |x, y| x.div_impl(y));

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f64(x: &Real) -> f64 {
        // Exact for values that fit in 53 bits, which the tests use.
        let m: f64 = x.man.to_string().parse().unwrap_or(f64::NAN);
        let v = m * 2f64.powi(x.exp as i32);
        if x.neg { -v } else { v }
    }

    #[test]
    fn matches_f64_at_53_bits() {
        // At 53 bits the operations must agree with IEEE double arithmetic.
        let vals = [0.1, 1.0 / 3.0, 2.5, -7.25, 1e-10, 12345.678, -0.0625, 3.0];
        for &a in &vals {
            for &b in &vals {
                let (x, y) = (Real::from_f64(a, 53), Real::from_f64(b, 53));
                assert_eq!(to_f64(&(&x + &y)), a + b);
                assert_eq!(to_f64(&(&x - &y)), a - b);
                assert_eq!(to_f64(&(&x * &y)), a * b);
                assert_eq!(to_f64(&(&x / &y)), a / b);
            }
            if a >= 0.0 {
                assert_eq!(to_f64(&Real::from_f64(a, 53).sqrt()), a.sqrt());
            }
        }
    }

    #[test]
    fn rounding_helpers() {
        let r = |x: f64| Real::from_f64(x, 64);
        assert_eq!(r(2.5).nint(), BigInt::from(2));
        assert_eq!(r(3.5).nint(), BigInt::from(4));
        assert_eq!(r(-2.5).nint(), BigInt::from(-2));
        assert_eq!(r(-2.6).nint(), BigInt::from(-3));
        assert_eq!(r(-2.5).floor(), BigInt::from(-3));
        assert_eq!(r(-2.5).ceil(), BigInt::from(-2));
        assert_eq!(r(7.0).floor(), BigInt::from(7));
    }

    #[test]
    fn transcendental() {
        for &x in &[0.5, -1.25, 3.0, 1000.0, 1e-9] {
            let (s, c) = Real::from_f64(x, 53).sin_cos();
            assert!((to_f64(&s) - f64::sin(x)).abs() <= f64::EPSILON * f64::sin(x).abs());
            assert!((to_f64(&c) - f64::cos(x)).abs() <= f64::EPSILON * f64::cos(x).abs());
        }
        for &x in &[0.5, 2.0, 1e10, 1e-10, 2.414] {
            let l = Real::from_f64(x, 53).ln();
            assert!((to_f64(&l) - f64::ln(x)).abs() <= 2.0 * f64::EPSILON * f64::ln(x).abs());
        }
    }
}
