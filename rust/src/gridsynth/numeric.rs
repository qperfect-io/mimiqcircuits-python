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
//! Integer helpers, with Python semantics: `//`, `%` and `>>` round towards
//! -∞, and the real-valued numeric helpers of [`Synthesizer`].

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed};

use super::Synthesizer;
use super::real::Real;

pub(super) type Int = BigInt;

pub(super) fn int(n: i64) -> Int {
    Int::from(n)
}

pub(super) fn floorsqrt(x: &Int) -> Int {
    assert!(!x.is_negative(), "floorsqrt of negative number {x}");
    x.sqrt()
}

pub(super) fn rounddiv(x: &Int, y: &Int) -> Int {
    if y.is_positive() {
        (x + (y >> 1u32)).div_floor(y)
    } else {
        (x - ((-y) >> 1u32)).div_floor(y)
    }
}

pub(super) fn sign(x: &Int) -> i32 {
    if x.is_positive() {
        1
    } else if x.is_negative() {
        -1
    } else {
        0
    }
}

/// Whether 2^m divides x.
pub(super) fn divisible_pow2(x: &Int, m: u64) -> bool {
    x.trailing_zeros().is_none_or(|tz| tz >= m)
}

pub(super) fn pow2(n: u64) -> Int {
    Int::one() << n
}

impl Synthesizer {
    /// Kept identical to the reference implementation (including its behaviour
    /// for x < 1) so that grid points are enumerated in the same order.
    pub(super) fn floorlog(&self, x: &Real, y: &Real) -> i64 {
        assert!(*x > 0, "floorlog of non-positive number");
        let mut tmp = y.clone();
        let mut m = 0;
        while *x >= tmp || (x * &tmp) < 1 {
            tmp = tmp.sq();
            m += 1;
        }
        let mut pows = vec![y.clone()];
        for _ in 1..m {
            let last = pows[pows.len() - 1].sq();
            pows.push(last);
        }
        let (mut n, mut r) = if *x >= 1 {
            (0i64, x.clone())
        } else {
            (-1i64, x * &tmp)
        };
        for p in pows.iter().rev() {
            n <<= 1;
            if r > *p {
                r = &r / p;
                n += 1;
            }
        }
        n
    }

    pub(super) fn solve_quadratic(&self, a: Real, b: Real, c: Real) -> Option<(Real, Real)> {
        let (a, b, c) = if a < 0 { (-a, -b, -c) } else { (a, b, c) };
        let disc = b.sq() - (&(4 * &a) * &c);
        if disc < 0 {
            return None;
        }
        let sq = disc.sqrt();
        let s1 = &(-&b) - &sq;
        let s2 = &(-&b) + &sq;
        if b >= 0 {
            let a2 = 2 * &a;
            Some((&s1 / &a2, &s2 / &a2))
        } else if c.is_zero() {
            Some((Real::zero(self.prec), &(-&b) / &a))
        } else {
            let c2 = 2 * &c;
            Some((&c2 / &s2, &c2 / &s1))
        }
    }
}
