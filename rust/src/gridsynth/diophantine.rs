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
//! Solver for the Diophantine equation t†t = ξ (Ross–Selinger §6): factoring,
//! primality testing, modular square roots and the effort bounds that limit
//! them. Pure integer arithmetic.

use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use super::numeric::*;
use super::rings::*;
use super::rng::*;

/// Bounds the effort spent on factoring (Pollard rho rounds) and on retrying
/// unsolved factors. When a bound is hit the candidate is abandoned and the
/// caller moves on to the next one.
pub(super) struct Effort {
    rng: SplitMix64,
    dloop: u32,
    floop: u32,
    dcount: u32,
    fcount: u32,
}

impl Effort {
    fn diophantine_continue(&mut self) -> bool {
        if self.dcount < self.dloop {
            self.dcount += 1;
            true
        } else {
            false
        }
    }

    fn factoring_continue(&mut self) -> bool {
        if self.fcount < self.floop {
            self.fcount += 1;
            true
        } else {
            false
        }
    }
}

pub(super) enum Dec {
    Sol(ZOmega),
    NoSolution,
    Unsolved,
}

fn modpow(b: &Int, e: &Int, m: &Int) -> Int {
    b.mod_floor(m).modpow(e, m)
}

/// Brent's variant of Pollard rho. Returns a nontrivial factor or None.
fn find_factor(n: &Int, e: &mut Effort) -> Option<Int> {
    const M: u32 = 128;
    let two = int(2);
    if n.is_even() && *n > two {
        return Some(two);
    }
    let a = e.rng.randint(&Int::one(), n);
    let mut y = a.clone();
    let mut r: u64 = 1;
    let mut k: u64 = 0;
    let digits = n.abs().to_str_radix(10).len();
    let bound = (10f64.powf(digits as f64 / 4.0) * 1.1774 + 10.0).floor();
    e.fcount = 0;
    loop {
        let x = &y + n;
        while k < r {
            let mut q = Int::one();
            let y0 = y.clone();
            for _ in 0..M {
                y = (&y * &y + &a).mod_floor(n);
                q = (&q * (&x - &y)).mod_floor(n);
                k += 1;
                if k == r {
                    break;
                }
            }
            let mut g = q.gcd(n);
            if !g.is_one() {
                if g == *n {
                    y = y0;
                    for _ in 0..M {
                        y = (&y * &y + &a).mod_floor(n);
                        g = (&x - &y).gcd(n);
                        if !g.is_one() {
                            break;
                        }
                    }
                }
                return if g == *n { None } else { Some(g) };
            }
            if k as f64 >= bound || !e.factoring_continue() {
                return None;
            }
        }
        r <<= 1;
    }
}

fn sqrt_negative_one(p: &Int, e: &mut Effort) -> Option<Int> {
    let pm1: Int = p - 1u32;
    for _ in 0..100 {
        let b = e.rng.randint(&Int::one(), &pm1);
        let h = modpow(&b, &(&pm1 >> 2u32), p);
        let r = (&h * &h).mod_floor(p);
        if r == pm1 {
            return Some(h);
        }
        if !r.is_one() {
            return None;
        }
    }
    None
}

/// First component of (x + √base)^n in F_p[√base].
fn fp2_pow_first(x: &Int, base: &Int, p: &Int, n: &Int) -> Int {
    let mut ra = Int::one();
    let mut rb = Int::zero();
    let mut ta = x.mod_floor(p);
    let mut tb = Int::one();
    let mut n = n.clone();
    while n.is_positive() {
        if n.is_odd() {
            let na = (&ra * &ta + (&rb * &tb).mod_floor(p) * base).mod_floor(p);
            let nb = (&ra * &tb + &rb * &ta).mod_floor(p);
            ra = na;
            rb = nb;
        }
        let na = (&ta * &ta + (&tb * &tb).mod_floor(p) * base).mod_floor(p);
        let nb = ((&ta * &tb) << 1u32).mod_floor(p);
        ta = na;
        tb = nb;
        n >>= 1u32;
    }
    ra
}

/// Square root of x modulo the prime p (Cipolla), or None.
fn root_mod(x: &Int, p: &Int, e: &mut Effort) -> Option<Int> {
    let x = x.mod_floor(p);
    let two = int(2);
    if *p == two {
        return Some(x);
    }
    if x.is_zero() {
        return Some(x);
    }
    if p.is_even() && *p > two {
        return None;
    }
    let pm1: Int = p - 1u32;
    let half = pm1.div_floor(&two);
    if !modpow(&x, &half, p).is_one() {
        return None;
    }
    for _ in 0..100 {
        let b = e.rng.randint(&Int::one(), &pm1);
        if !modpow(&b, &pm1, p).is_one() {
            return None;
        }
        let base = (&b * &b + p - &x).mod_floor(p);
        if !modpow(&base, &half, p).is_one() {
            return Some(fp2_pow_first(&b, &base, p, &(p + 1u32).div_floor(&two)));
        }
    }
    None
}

/// Probabilistic primality test, kept identical to the reference
/// implementation (including its early acceptance) so that the random stream
/// is consumed in the same way.
fn is_prime(n: &Int, e: &mut Effort) -> bool {
    let n = n.abs();
    if n.is_zero() || n.is_one() {
        return false;
    }
    if n.is_even() {
        return n == int(2);
    }
    let nm1: Int = &n - 1u32;
    let r = nm1.trailing_zeros().unwrap_or(0);
    let d = &nm1 >> r;
    for _ in 0..4 {
        let mut a = modpow(&e.rng.randint(&Int::one(), &nm1), &d, &n);
        if a.is_one() {
            return true;
        }
        for _ in 0..r {
            if a == nm1 {
                return true;
            }
            a = (&a * &a).mod_floor(&n);
        }
    }
    false
}

/// Refines a partial factorisation into pairwise coprime factors.
fn decompose_relatively_int_prime(partial: Vec<(Int, i64)>) -> (Int, Vec<(Int, i64)>) {
    let mut u = Int::one();
    let mut stack: Vec<(Int, i64)> = partial.into_iter().rev().collect();
    let mut facs: Vec<(Int, i64)> = Vec::new();
    while let Some((b, kb)) = stack.pop() {
        let mut i = 0;
        loop {
            if i >= facs.len() {
                if b.is_one() || b == int(-1) {
                    if b == int(-1) && kb & 1 == 1 {
                        u = -u;
                    }
                } else {
                    facs.push((b, kb));
                }
                break;
            }
            let (a, ka) = facs[i].clone();
            if a == b || a == -&b {
                if a == -&b && kb & 1 == 1 {
                    u = -u;
                }
                facs[i] = (a, ka + kb);
                break;
            }
            let g = a.gcd(&b);
            if g.is_one() {
                i += 1;
                continue;
            }
            let (ua, facsa) =
                decompose_relatively_int_prime(vec![(a.div_floor(&g), ka), (g.clone(), ka + kb)]);
            u *= ua;
            let mut it = facsa.into_iter();
            if let Some(first) = it.next() {
                facs[i] = first;
            }
            facs.extend(it);
            stack.push((b.div_floor(&g), kb));
            break;
        }
    }
    (u, facs)
}

fn decompose_relatively_zomega_prime(
    partial: Vec<(ZRootTwo, i64)>,
) -> (ZRootTwo, Vec<(ZRootTwo, i64)>) {
    let mut u = ZRootTwo::of(1, 0);
    let one = ZRootTwo::of(1, 0);
    let mut stack: Vec<(ZRootTwo, i64)> = partial.into_iter().rev().collect();
    let mut facs: Vec<(ZRootTwo, i64)> = Vec::new();
    while let Some((b, kb)) = stack.pop() {
        let mut i = 0;
        loop {
            if i >= facs.len() {
                if zr_similar(&b, &one) {
                    u = &u * &b.pow(kb);
                } else {
                    facs.push((b, kb));
                }
                break;
            }
            let (a, ka) = facs[i].clone();
            if zr_similar(&a, &b) {
                u = &u * &zr_div(&b, &a).pow(kb);
                facs[i] = (a, ka + kb);
                break;
            }
            let g = zr_gcd(&a, &b);
            if zr_similar(&g, &one) {
                i += 1;
                continue;
            }
            let (ua, facsa) =
                decompose_relatively_zomega_prime(vec![(zr_div(&a, &g), ka), (g.clone(), ka + kb)]);
            u = &u * &ua;
            let mut it = facsa.into_iter();
            if let Some(first) = it.next() {
                facs[i] = first;
            }
            facs.extend(it);
            stack.push((zr_div(&b, &g), kb));
            break;
        }
    }
    (u, facs)
}

/// Finds t ∈ Z[ω] with t†t ~ η for a prime (or prime-power candidate) η,
/// where p = |N(η)|. `divisor` is the element t must divide.
fn adj_decompose_prime(
    p: &Int,
    divisor: &ZOmega,
    check: &dyn Fn(&ZOmega) -> bool,
    e: &mut Effort,
) -> Dec {
    let p = p.abs();
    if p.is_zero() || p.is_one() {
        return Dec::Sol(ZOmega::from_int(p));
    }
    if p == int(2) {
        return Dec::Sol(ZOmega::of(-1, 0, 1, 0));
    }
    let p8 = (&p & int(7)).to_u8().unwrap_or(0);
    let root2_check = |e: &mut Effort| match root_mod(&int(2), &p, e) {
        None => Dec::Unsolved,
        Some(_) => Dec::NoSolution,
    };
    if is_prime(&p, e) {
        if p8 & 3 == 1 {
            let Some(h) = sqrt_negative_one(&p, e) else {
                return Dec::Unsolved;
            };
            let t = zo_gcd(&(ZOmega::from_int(h) + ZOmega::of(0, 1, 0, 0)), divisor);
            return if check(&t) {
                Dec::Sol(t)
            } else {
                Dec::Unsolved
            };
        } else if p8 == 3 {
            let Some(h) = root_mod(&int(-2), &p, e) else {
                return Dec::Unsolved;
            };
            let t = zo_gcd(&(ZOmega::from_int(h) + ZOmega::of(1, 0, 1, 0)), divisor);
            return if check(&t) {
                Dec::Sol(t)
            } else {
                Dec::Unsolved
            };
        } else if p8 == 7 {
            return root2_check(e);
        }
        return Dec::Unsolved;
    }
    if p8 == 7 {
        return root2_check(e);
    }
    Dec::Unsolved
}

fn adj_decompose_int_prime(p: &Int, e: &mut Effort) -> Dec {
    let pn = -p;
    let check = |t: &ZOmega| {
        let n = &t.conj() * t;
        n.eq_int(p) || n.eq_int(&pn)
    };
    adj_decompose_prime(p, &ZOmega::from_int(p.abs()), &check, e)
}

fn adj_decompose_int_prime_power(p: &Int, k: i64, e: &mut Effort) -> Dec {
    if k % 2 == 0 {
        return Dec::Sol(ZOmega::from_int(num_traits::pow(
            p.clone(),
            (k / 2) as usize,
        )));
    }
    match adj_decompose_int_prime(p, e) {
        Dec::Sol(t) => Dec::Sol(t.pow(k)),
        other => other,
    }
}

fn adj_decompose_int(n: &Int, e: &mut Effort) -> Dec {
    let mut facs = vec![(n.abs(), 1i64)];
    let mut t = ZOmega::of(0, 0, 0, 1);
    while let Some((p, k)) = facs.pop() {
        match adj_decompose_int_prime_power(&p, k, e) {
            Dec::NoSolution => return Dec::NoSolution,
            Dec::Unsolved => match find_factor(&p, e) {
                None => {
                    facs.push((p, k));
                    if !e.diophantine_continue() {
                        return Dec::NoSolution;
                    }
                }
                Some(fac) => {
                    facs.push((p.div_floor(&fac), k));
                    facs.push((fac, k));
                    facs = decompose_relatively_int_prime(facs).1;
                }
            },
            Dec::Sol(tp) => t = &t * &tp,
        }
    }
    Dec::Sol(t)
}

/// ξ ~ ξ•
fn adj_decompose_selfassociate(xi: &ZRootTwo, e: &mut Effort) -> Dec {
    if xi.is_zero() {
        return Dec::Sol(ZOmega::of(0, 0, 0, 0));
    }
    let n = xi.a.gcd(&xi.b);
    let r = zr_div(xi, &ZRootTwo::from_int(n.clone()));
    let Dec::Sol(t1) = adj_decompose_int(&n, e) else {
        return Dec::NoSolution;
    };
    if zr_rem(&r, &ZRootTwo::of(0, 1)).is_zero() {
        return Dec::Sol(&t1 * &ZOmega::of(0, 0, 1, 1));
    }
    Dec::Sol(t1)
}

fn adj_decompose_zomega_prime(eta: &ZRootTwo, e: &mut Effort) -> Dec {
    let eta_w = ZOmega::from_zroottwo(eta);
    let check = |t: &ZOmega| zo_similar(&(&t.conj() * t), &eta_w);
    adj_decompose_prime(&eta.znorm(), &eta_w, &check, e)
}

fn adj_decompose_zomega_prime_power(eta: &ZRootTwo, k: i64, e: &mut Effort) -> Dec {
    if k % 2 == 0 {
        return Dec::Sol(ZOmega::from_zroottwo(&eta.pow(k / 2)));
    }
    match adj_decompose_zomega_prime(eta, e) {
        Dec::Sol(t) => Dec::Sol(t.pow(k)),
        other => other,
    }
}

/// gcd(ξ, ξ•) = 1
fn adj_decompose_selfcoprime(xi: &ZRootTwo, e: &mut Effort) -> Dec {
    let mut facs = vec![(xi.clone(), 1i64)];
    let mut t = ZOmega::of(0, 0, 0, 1);
    while let Some((eta, k)) = facs.pop() {
        match adj_decompose_zomega_prime_power(&eta, k, e) {
            Dec::NoSolution => return Dec::NoSolution,
            Dec::Unsolved => match find_factor(&eta.znorm().abs(), e) {
                None => {
                    facs.push((eta, k));
                    if !e.diophantine_continue() {
                        return Dec::NoSolution;
                    }
                }
                Some(facn) => {
                    let fac = zr_gcd(xi, &ZRootTwo::from_int(facn));
                    facs.push((zr_div(&eta, &fac), k));
                    facs.push((fac, k));
                    facs = decompose_relatively_zomega_prime(facs).1;
                }
            },
            Dec::Sol(teta) => t = &t * &teta,
        }
    }
    Dec::Sol(t)
}

fn adj_decompose(xi: &ZRootTwo, e: &mut Effort) -> Dec {
    if xi.is_zero() {
        return Dec::Sol(ZOmega::of(0, 0, 0, 0));
    }
    let d = zr_gcd(xi, &xi.conj2());
    let eta = zr_div(xi, &d);
    let Dec::Sol(t1) = adj_decompose_selfassociate(&d, e) else {
        return Dec::NoSolution;
    };
    let Dec::Sol(t2) = adj_decompose_selfcoprime(&eta, e) else {
        return Dec::NoSolution;
    };
    Dec::Sol(&t1 * &t2)
}

fn diophantine(xi: &ZRootTwo, e: &mut Effort) -> Option<ZOmega> {
    e.dcount = 0;
    if xi.is_zero() {
        return Some(ZOmega::of(0, 0, 0, 0));
    }
    let zero = ZRootTwo::of(0, 0);
    if xi.lt(&zero) || xi.conj2().lt(&zero) {
        return None;
    }
    let Dec::Sol(t) = adj_decompose(xi, e) else {
        return None;
    };
    let u = zr_div(xi, &(&t.conj() * &t).to_zroottwo());
    let v = zr_sqrt(&u)?;
    Some(&ZOmega::from_zroottwo(&v) * &t)
}

/// Solves t†t = ξ for ξ ∈ D[√2].
pub(super) fn diophantine_dyadic(
    xi: &DRootTwo,
    seed: u64,
    dloop: u32,
    floop: u32,
) -> Option<DOmega> {
    let mut e = Effort {
        rng: SplitMix64 { s: seed },
        dloop,
        floop,
        dcount: 0,
        fcount: 0,
    };
    let kodd = xi.k & 1;
    let arg = if kodd == 1 {
        &xi.alpha * &lambda()
    } else {
        xi.alpha.clone()
    };
    let mut t = diophantine(&arg, &mut e)?;
    if kodd == 1 {
        t = &t * &ZOmega::of(0, -1, 1, 0);
    }
    Some(DOmega::new(t, (xi.k >> 1) + kodd))
}
