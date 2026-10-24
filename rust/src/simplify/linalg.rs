//
// Copyright © 2026 QPerfect. All Rights Reserved.
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
//! Small dense complex matrices and the decompositions built on them.

use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Entries below this are zero when deciding identity, commutation, cancellation
/// and diagonality.
pub const ATOL: f64 = 1e-12;

/// A square complex matrix, stored row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub n: usize,
    pub a: Vec<C64>,
}

impl Mat {
    pub fn zeros(n: usize) -> Mat {
        Mat {
            n,
            a: vec![C64::new(0.0, 0.0); n * n],
        }
    }

    pub fn identity(n: usize) -> Mat {
        let mut m = Mat::zeros(n);
        for i in 0..n {
            m.a[i * n + i] = C64::new(1.0, 0.0);
        }
        m
    }

    /// A matrix from its entries, row by row.
    pub fn from_rows(n: usize, rows: &[C64]) -> Mat {
        assert_eq!(rows.len(), n * n);
        Mat { n, a: rows.to_vec() }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> C64 {
        self.a[r * self.n + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: C64) {
        self.a[r * self.n + c] = v;
    }

    pub fn mul(&self, o: &Mat) -> Mat {
        let n = self.n;
        let mut out = Mat::zeros(n);
        for r in 0..n {
            for c in 0..n {
                let mut s = C64::new(0.0, 0.0);
                for k in 0..n {
                    s += self.a[r * n + k] * o.a[k * n + c];
                }
                out.a[r * n + c] = s;
            }
        }
        out
    }

    pub fn adjoint(&self) -> Mat {
        let n = self.n;
        let mut out = Mat::zeros(n);
        for r in 0..n {
            for c in 0..n {
                out.a[c * n + r] = self.a[r * n + c].conj();
            }
        }
        out
    }

    pub fn transpose(&self) -> Mat {
        let n = self.n;
        let mut out = Mat::zeros(n);
        for r in 0..n {
            for c in 0..n {
                out.a[c * n + r] = self.a[r * n + c];
            }
        }
        out
    }

    pub fn scale(&self, s: C64) -> Mat {
        Mat {
            n: self.n,
            a: self.a.iter().map(|x| x * s).collect(),
        }
    }

    /// Frobenius norm of `self - o`.
    pub fn dist(&self, o: &Mat) -> f64 {
        self.a
            .iter()
            .zip(&o.a)
            .map(|(x, y)| (x - y).norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Whether the Frobenius distance to the identity is at most `ATOL`.
    pub fn is_identity(&self) -> bool {
        self.dist(&Mat::identity(self.n)) <= ATOL
    }

    pub fn is_diagonal(&self) -> bool {
        let n = self.n;
        (0..n).all(|r| (0..n).all(|c| r == c || self.a[r * n + c].norm() <= ATOL))
    }

    /// Index (row, column) of the entry of largest magnitude, the first one in
    /// column-major order on ties.
    pub fn argmax_abs(&self) -> (usize, usize) {
        let n = self.n;
        let (mut best, mut at) = (-1.0, (0, 0));
        for c in 0..n {
            for r in 0..n {
                let v = self.a[r * n + c].norm();
                if v > best {
                    best = v;
                    at = (r, c);
                }
            }
        }
        at
    }
}

/// `M` acting on the qubits at positions `locs` (0-based, first is most
/// significant) of an `n`-qubit register, as a matrix on all `n` qubits.
pub fn embed(m: &Mat, locs: &[usize], n: usize) -> Mat {
    let dim = 1usize << n;
    let sub = |x: usize| {
        locs.iter()
            .fold(0usize, |acc, &p| (acc << 1) | ((x >> (n - 1 - p)) & 1))
    };
    let rest: Vec<usize> = (0..n).filter(|p| !locs.contains(p)).collect();
    let mut out = Mat::zeros(dim);
    for r in 0..dim {
        for c in 0..dim {
            if rest
                .iter()
                .all(|&p| (r >> (n - 1 - p)) & 1 == (c >> (n - 1 - p)) & 1)
            {
                out.a[r * dim + c] = m.get(sub(r), sub(c));
            }
        }
    }
    out
}

/// Product of matrices given in circuit order, later ones on the left:
/// `((I·Mₖ)·Mₖ₋₁)…·M₁`, the same fold as `_foldmatrices`.
pub fn fold(ms: &[Mat], dim: usize) -> Mat {
    let mut acc = Mat::identity(dim);
    for m in ms.iter().rev() {
        acc = acc.mul(m);
    }
    acc
}

/// Angles (θ, ϕ, λ, γ) with `U == GateU(θ, ϕ, λ, γ)` for a 2×2 unitary.
pub fn zyz(u: &Mat) -> (f64, f64, f64, f64) {
    let (u00, u01, u10, u11) = (u.get(0, 0), u.get(0, 1), u.get(1, 0), u.get(1, 1));
    let c = u00.norm();
    let s = u10.norm();
    let theta = 2.0 * s.atan2(c);
    if s == 0.0 {
        let gamma = u00.arg();
        return (0.0, 0.0, u11.arg() - gamma, gamma);
    }
    if c <= 1e-8 * c.max(s) {
        let gamma = u10.arg();
        return (PI, 0.0, u01.arg() - gamma - PI, gamma);
    }
    let gamma = u00.arg();
    (theta, u10.arg() - gamma, u11.arg() - u10.arg(), gamma)
}

/// Eigenvectors of a real symmetric matrix (row-major, `n × n`) as the columns
/// of an orthogonal matrix, by cyclic Jacobi rotations.
pub fn jacobi_eigvecs(s: &[f64], n: usize) -> Vec<f64> {
    let mut a = s.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..50 {
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += a[i * n + j] * a[i * n + j];
                }
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n - 1 {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    v
}

/// Determinant by LU decomposition with partial pivoting.
pub fn det(m: &Mat) -> C64 {
    let n = m.n;
    let mut a = m.a.clone();
    let mut d = C64::new(1.0, 0.0);
    for k in 0..n {
        let mut p = k;
        for r in k + 1..n {
            if a[r * n + k].norm() > a[p * n + k].norm() {
                p = r;
            }
        }
        if a[p * n + k].norm() == 0.0 {
            return C64::new(0.0, 0.0);
        }
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
            d = -d;
        }
        let piv = a[k * n + k];
        d *= piv;
        for r in k + 1..n {
            let f = a[r * n + k] / piv;
            for c in k..n {
                let v = a[k * n + c];
                a[r * n + c] -= f * v;
            }
        }
    }
    d
}

/// Solution of the real system `a x = b`, by Gaussian elimination with partial
/// pivoting.
pub fn solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for k in 0..n {
        let mut p = k;
        for r in k + 1..n {
            if m[r * n + k].abs() > m[p * n + k].abs() {
                p = r;
            }
        }
        if p != k {
            for c in 0..n {
                m.swap(k * n + c, p * n + c);
            }
            x.swap(k, p);
        }
        for r in k + 1..n {
            let f = m[r * n + k] / m[k * n + k];
            for c in k..n {
                m[r * n + c] -= f * m[k * n + c];
            }
            x[r] -= f * x[k];
        }
    }
    for k in (0..n).rev() {
        let mut s = x[k];
        for c in k + 1..n {
            s -= m[k * n + c] * x[c];
        }
        x[k] = s / m[k * n + k];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zyz_reconstructs() {
        let h = 1.0 / 2f64.sqrt();
        let m = Mat::from_rows(
            2,
            &[
                C64::new(h, 0.0),
                C64::new(h, 0.0),
                C64::new(h, 0.0),
                C64::new(-h, 0.0),
            ],
        );
        let (t, p, l, g) = zyz(&m);
        let e = |x: f64| C64::from_polar(1.0, x);
        let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
        let u = Mat::from_rows(2, &[e(g) * c, -e(g + l) * s, e(g + p) * s, e(g + p + l) * c]);
        assert!(u.dist(&m) < 1e-12);
    }

    #[test]
    fn jacobi_diagonalizes() {
        let s = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 1.0];
        let v = jacobi_eigvecs(&s, 3);
        // vᵀ s v is diagonal
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let mut x = 0.0;
                for k in 0..3 {
                    for l in 0..3 {
                        x += v[k * 3 + i] * s[k * 3 + l] * v[l * 3 + j];
                    }
                }
                assert!(x.abs() < 1e-12);
            }
        }
    }
}
