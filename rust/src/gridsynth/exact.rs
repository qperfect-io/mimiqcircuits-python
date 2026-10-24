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
//! Exact synthesis of U = [z  -w†ωⁿ; w  z†ωⁿ] over D[ω] (Kliuchnikov–Maslov–
//! Mosca) and Matsumoto–Amano normal form.

use super::rings::*;

#[derive(Clone, Debug)]
pub(super) struct DOmegaUnitary {
    z: DOmega,
    w: DOmega,
    n: i64,
}

impl DOmegaUnitary {
    pub(super) fn new(z: DOmega, w: DOmega, n: i64) -> Self {
        let (z, w) = align_do(&z, &w);
        DOmegaUnitary { z, w, n: n & 7 }
    }

    fn with_k(z: &DOmega, w: &DOmega, n: i64, k: i64) -> Self {
        DOmegaUnitary {
            z: z.renew(k),
            w: w.renew(k),
            n: n & 7,
        }
    }

    fn denomexp(&self) -> i64 {
        self.w.k
    }

    fn renew(&self, k: i64) -> Self {
        DOmegaUnitary::with_k(&self.z, &self.w, self.n, k)
    }

    fn eq(&self, v: &Self) -> bool {
        self.z.eq(&v.z) && self.w.eq(&v.w) && self.n == v.n
    }
}

fn mul_t_inv(u: &DOmegaUnitary) -> DOmegaUnitary {
    DOmegaUnitary::new(u.z.clone(), u.w.mul_by_omega_inv(), u.n - 1)
}

fn mul_t_power(u: &DOmegaUnitary, m: i64) -> DOmegaUnitary {
    let m = m & 7;
    DOmegaUnitary::new(u.z.clone(), u.w.mul_by_omega_power(m), u.n + m)
}

fn mul_s_power(u: &DOmegaUnitary, m: i64) -> DOmegaUnitary {
    let m = m & 3;
    DOmegaUnitary::new(u.z.clone(), u.w.mul_by_omega_power(2 * m), u.n + 2 * m)
}

fn mul_h(u: &DOmegaUnitary) -> DOmegaUnitary {
    DOmegaUnitary::new(
        (&u.z + &u.w).mul_by_inv_sqrt2(),
        (&u.z - &u.w).mul_by_inv_sqrt2(),
        u.n + 4,
    )
}

fn mul_ht_power(u: &DOmegaUnitary, m: i64) -> DOmegaUnitary {
    mul_h(&mul_t_power(u, m))
}

fn mul_x(u: &DOmegaUnitary) -> DOmegaUnitary {
    DOmegaUnitary::new(u.w.clone(), u.z.clone(), u.n + 4)
}

fn mul_w_power(u: &DOmegaUnitary, m: i64) -> DOmegaUnitary {
    let m = m & 7;
    DOmegaUnitary::new(
        u.z.mul_by_omega_power(m),
        u.w.mul_by_omega_power(m),
        u.n + 2 * m,
    )
}

const BIT_SHIFT: [i64; 16] = [0, 0, 1, 0, 2, 0, 1, 3, 3, 3, 0, 2, 2, 1, 0, 0];
const BIT_COUNT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
const T_POWER_AND_H: [&str; 4] = ["H", "TH", "SH", "TSH"];

/// One step of exact synthesis: peel a (T^m H) prefix off U, lowering the
/// denominator exponent.
fn reduce_denomexp_step(u: &DOmegaUnitary) -> (&'static str, DOmegaUnitary) {
    let rz = u.z.residue() as usize;
    let rw = u.w.residue() as usize;
    let rsq = (&u.z.u * &u.z.u.conj()).residue();
    let mut m = BIT_SHIFT[rw] - BIT_SHIFT[rz];
    if m < 0 {
        m += 4;
    }
    let k = u.denomexp();
    match rsq {
        0b0000 => (T_POWER_AND_H[0], mul_ht_power(u, 0).renew(k - 1)),
        0b1010 => (T_POWER_AND_H[m as usize], mul_ht_power(u, -m).renew(k - 1)),
        0b0001 => {
            let mut v = mul_ht_power(u, -m);
            if BIT_COUNT[rz] == BIT_COUNT[rw] {
                v = v.renew(k - 1);
            }
            (T_POWER_AND_H[m as usize], v)
        }
        _ => panic!("invalid residue in exact synthesis"),
    }
}

/// Clifford group element E^a X^b S^c ω^d (Matsumoto–Amano).
#[derive(Clone, Copy, Debug)]
pub(super) struct Clifford {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

impl Clifford {
    fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Clifford {
            a: a % 3,
            b: b & 1,
            c: c & 3,
            d: d & 7,
        }
    }

    fn mul(self, y: Clifford) -> Clifford {
        let [a1, b1, c1, d1] = CONJ3_TABLE[((y.a << 3) | (self.b << 2) | self.c) as usize];
        let [c2, d2] = CONJ2_TABLE[((c1 << 1) | y.b) as usize];
        Clifford::new(self.a + a1, b1 + y.b, c2 + y.c, d2 + d1 + self.d + y.d)
    }

    fn inv(self) -> Clifford {
        let [a1, b1, c1, d1] = CINV_TABLE[((self.a << 3) | (self.b << 2) | self.c) as usize];
        Clifford::new(a1, b1, c1, d1 + 8 - self.d)
    }
}

const CONJ2_TABLE: [[u8; 2]; 8] = [
    [0, 0],
    [0, 0],
    [1, 0],
    [3, 2],
    [2, 0],
    [2, 4],
    [3, 0],
    [1, 6],
];
#[rustfmt::skip]
const CONJ3_TABLE: [[u8; 4]; 24] = [
    [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0],
    [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0],
    [1, 0, 0, 0], [2, 0, 3, 6], [1, 1, 2, 2], [2, 1, 3, 6],
    [1, 0, 2, 0], [2, 1, 1, 0], [1, 1, 0, 6], [2, 0, 1, 4],
    [2, 0, 0, 0], [1, 1, 3, 4], [2, 1, 0, 0], [1, 0, 1, 2],
    [2, 1, 2, 2], [1, 1, 1, 0], [2, 0, 2, 6], [1, 0, 3, 2],
];
#[rustfmt::skip]
const CINV_TABLE: [[u8; 4]; 24] = [
    [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 2, 0], [0, 0, 1, 0],
    [0, 1, 0, 0], [0, 1, 1, 6], [0, 1, 2, 4], [0, 1, 3, 2],
    [2, 0, 0, 0], [1, 0, 1, 2], [2, 1, 0, 0], [1, 1, 3, 4],
    [2, 1, 1, 2], [1, 1, 1, 6], [2, 0, 2, 2], [1, 0, 3, 4],
    [1, 0, 0, 0], [2, 1, 3, 6], [1, 1, 2, 2], [2, 0, 3, 6],
    [1, 0, 2, 0], [2, 1, 1, 6], [1, 1, 0, 2], [2, 0, 1, 6],
];
// Axis: 0 = I, 1 = H, 2 = SH
const TCONJ_TABLE: [[u8; 3]; 6] = [
    [0, 0, 0],
    [0, 1, 7],
    [1, 3, 3],
    [1, 2, 0],
    [2, 0, 5],
    [2, 1, 4],
];

const CLIFFORD_I: Clifford = Clifford {
    a: 0,
    b: 0,
    c: 0,
    d: 0,
};
const CLIFFORD_X: Clifford = Clifford {
    a: 0,
    b: 1,
    c: 0,
    d: 0,
};
const CLIFFORD_H: Clifford = Clifford {
    a: 1,
    b: 0,
    c: 1,
    d: 5,
};
const CLIFFORD_S: Clifford = Clifford {
    a: 0,
    b: 0,
    c: 1,
    d: 0,
};
const CLIFFORD_W: Clifford = Clifford {
    a: 0,
    b: 0,
    c: 0,
    d: 1,
};

fn clifford_from_char(g: u8) -> Clifford {
    match g {
        b'H' => CLIFFORD_H,
        b'S' => CLIFFORD_S,
        b'X' => CLIFFORD_X,
        b'W' => CLIFFORD_W,
        _ => panic!("unexpected gate {}", g as char),
    }
}

fn decompose_coset(x: Clifford) -> (u8, Clifford) {
    match x.a {
        0 => (0, x),
        1 => (1, CLIFFORD_H.inv().mul(x)),
        _ => (2, CLIFFORD_S.mul(CLIFFORD_H).inv().mul(x)),
    }
}

fn decompose_tconj(x: Clifford) -> (u8, Clifford) {
    let [axis, c1, d1] = TCONJ_TABLE[((x.a << 1) | x.b) as usize];
    (axis, Clifford::new(0, x.b, c1 + x.c, d1 + x.d))
}

fn clifford_chars(x: Clifford, out: &mut Vec<u8>) {
    // Global phases (W = ω) are dropped: the caller only needs U up to phase.
    let (axis, c) = decompose_coset(x);
    if axis == 1 {
        out.push(b'H');
    } else if axis == 2 {
        out.extend_from_slice(b"SH");
    }
    out.extend(std::iter::repeat_n(b'X', c.b as usize));
    out.extend(std::iter::repeat_n(b'S', c.c as usize));
}

fn normal_form(gates: &[u8]) -> Vec<u8> {
    let clifford_hs = CLIFFORD_H.mul(CLIFFORD_S);
    let clifford_shs = CLIFFORD_S.mul(CLIFFORD_H).mul(CLIFFORD_S);
    // Syllables: 1 = T, 2 = HT, 3 = SHT
    let mut syl: Vec<u8> = Vec::new();
    let mut c = CLIFFORD_I;
    for &g in gates {
        if g == b'T' {
            let (axis, newc) = decompose_tconj(c);
            if axis == 0 {
                match syl.last() {
                    None => syl.push(1),
                    Some(1) => {
                        syl.pop();
                        c = CLIFFORD_S.mul(newc);
                    }
                    Some(2) => {
                        syl.pop();
                        c = clifford_hs.mul(newc);
                    }
                    Some(_) => {
                        syl.pop();
                        c = clifford_shs.mul(newc);
                    }
                }
            } else {
                syl.push(if axis == 1 { 2 } else { 3 });
                c = newc;
            }
        } else {
            c = c.mul(clifford_from_char(g));
        }
    }
    let mut out = Vec::new();
    for s in syl {
        if s == 2 {
            out.push(b'H');
        } else if s == 3 {
            out.extend_from_slice(b"SH");
        }
        out.push(b'T');
    }
    clifford_chars(c, &mut out);
    out
}

pub(super) fn decompose_domega_unitary(u: DOmegaUnitary) -> Vec<u8> {
    let mut u = u;
    let mut gates: Vec<u8> = Vec::new();
    while u.denomexp() > 0 {
        let (g, v) = reduce_denomexp_step(&u);
        gates.extend_from_slice(g.as_bytes());
        u = v;
    }
    if u.n & 1 == 1 {
        gates.push(b'T');
        u = mul_t_inv(&u);
    }
    if u.z.eq(&DOmega::of_int(0)) {
        gates.push(b'X');
        u = mul_x(&u);
    }
    let one = ZOmega::of(0, 0, 0, 1);
    for m in 0..8 {
        if u.z.eq(&DOmega::new(one.mul_by_omega_power(m), 0)) {
            u = mul_w_power(&u, -m);
            break;
        }
    }
    let ms = u.n >> 1;
    gates.extend(std::iter::repeat_n(b'S', ms as usize));
    u = mul_s_power(&u, -ms);
    assert!(
        u.eq(&DOmegaUnitary::new(DOmega::of_int(1), DOmega::of_int(0), 0)),
        "exact synthesis failed"
    );
    normal_form(&gates)
}
