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
//! Two-qubit KAK decomposition and minimal-CX synthesis.
//!
//! Every U ∈ U(4) is e^{iφ} (A1 ⊗ A2) exp(i(a XX + b YY + c ZZ)) (B1 ⊗ B2), and
//! (a, b, c) alone tell how many CX gates U needs. In the magic basis the local
//! factors are real orthogonal matrices around a diagonal core.

use super::linalg::{Mat, det, embed, fold, jacobi_eigvecs, solve, zyz};
use super::table::Code;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn magic() -> Mat {
    let h = 1.0 / 2f64.sqrt();
    let (o, l, i) = (c(0.0, 0.0), c(h, 0.0), c(0.0, h));
    Mat::from_rows(4, &[l, o, o, i, o, i, l, o, o, i, -l, o, l, o, o, -i])
}

fn pauli(k: usize) -> Mat {
    let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
    match k {
        0 => Mat::from_rows(2, &[o, l, l, o]),
        1 => Mat::from_rows(2, &[o, c(0.0, -1.0), c(0.0, 1.0), o]),
        _ => Mat::from_rows(2, &[l, o, o, -l]),
    }
}

fn kron(a: &Mat, b: &Mat) -> Mat {
    let mut out = Mat::zeros(4);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out.set(2 * i + j, 2 * k + l, a.get(i, k) * b.get(j, l));
                }
            }
        }
    }
    out
}

/// Row j: (1, ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩) of magic basis vector j.
fn signs() -> Vec<f64> {
    let m = magic();
    let mut s = vec![1.0; 16];
    for j in 0..4 {
        for k in 0..3 {
            let pp = kron(&pauli(k), &pauli(k));
            let mut x = c(0.0, 0.0);
            for r in 0..4 {
                for q in 0..4 {
                    x += m.get(r, j).conj() * pp.get(r, q) * m.get(q, j);
                }
            }
            s[j * 4 + k + 1] = x.re;
        }
    }
    s
}

fn simultaneous_real_diagonalization(m: &Mat) -> Option<Vec<f64>> {
    for k in 0..20 {
        let t = 0.3 + 1.1 * k as f64;
        let s: Vec<f64> = m.a.iter().map(|z| t.cos() * z.re + t.sin() * z.im).collect();
        let p = jacobi_eigvecs(&s, 4);
        let pm = real_to_mat(&p);
        let d = pm.transpose().mul(m).mul(&pm);
        let mut off = 0.0;
        for r in 0..4 {
            for q in 0..4 {
                if r != q {
                    off += d.get(r, q).norm_sqr();
                }
            }
        }
        if off.sqrt() < 1e-10 {
            return Some(p);
        }
    }
    None
}

fn real_to_mat(p: &[f64]) -> Mat {
    Mat {
        n: 4,
        a: p.iter().map(|&x| c(x, 0.0)).collect(),
    }
}

fn kron_factor(k: &Mat) -> (Mat, Mat) {
    let (r, col) = k.argmax_abs();
    let (i0, j0) = (r / 2, r % 2);
    let (k0, l0) = (col / 2, col % 2);
    let mut a = Mat::zeros(2);
    let mut b = Mat::zeros(2);
    for i in 0..2 {
        for kk in 0..2 {
            a.set(i, kk, k.get(2 * i + j0, 2 * kk + l0));
        }
    }
    let pivot = k.get(r, col);
    for j in 0..2 {
        for l in 0..2 {
            b.set(j, l, k.get(2 * i0 + j, 2 * k0 + l) / pivot);
        }
    }
    let alpha = (a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0)).sqrt();
    (a.scale(1.0 / alpha), b.scale(alpha))
}

/// U = e^{iφ} (A1 ⊗ A2) exp(i(a XX + b YY + c ZZ)) (B1 ⊗ B2), as
/// (φ, (A1, A2), [a, b, c], (B1, B2)).
pub type Kak = (f64, (Mat, Mat), [f64; 3], (Mat, Mat));

/// The KAK decomposition of a 4×4 unitary, or `None` if it did not converge.
pub fn kak_decomposition(u: &Mat) -> Option<Kak> {
    let phi = det(u).arg() / 4.0;
    let mg = magic();
    let up = mg.adjoint().mul(&u.scale(C64::from_polar(1.0, -phi))).mul(&mg);
    let m2 = up.transpose().mul(&up);
    let mut p = simultaneous_real_diagonalization(&m2)?;
    let pm0 = real_to_mat(&p);
    if det(&pm0).re < 0.0 {
        for r in 0..4 {
            p[r * 4] = -p[r * 4];
        }
    }
    let pm = real_to_mat(&p);
    let dd = pm.transpose().mul(&up.transpose()).mul(&up).mul(&pm);
    let mut d: Vec<f64> = (0..4).map(|i| dd.get(i, i).arg() / 2.0).collect();
    if d.iter().sum::<f64>().cos() < 0.0 {
        d[0] += PI;
    }
    let mut dinv = Mat::zeros(4);
    for (i, &di) in d.iter().enumerate() {
        dinv.set(i, i, C64::from_polar(1.0, -di));
    }
    let k1 = up.mul(&pm).mul(&dinv);
    let k1 = Mat {
        n: 4,
        a: k1.a.iter().map(|z| c(z.re, 0.0)).collect(),
    };
    let (a1, a2) = kron_factor(&mg.mul(&k1).mul(&mg.adjoint()));
    let (b1, b2) = kron_factor(&mg.mul(&pm.transpose()).mul(&mg.adjoint()));
    let x = solve(&signs(), &d, 4);
    Some((phi + x[0], (a1, a2), [x[1], x[2], x[3]], (b1, b2)))
}

/// A gate of a synthesized block, on local qubits 0 and 1.
pub type Gate = (Code, Vec<f64>, Vec<usize>);

fn core(k: usize, co: [f64; 3]) -> Vec<Gate> {
    let [a, b, cc] = co;
    match k {
        0 => vec![],
        1 => vec![
            (Code::H, vec![], vec![1]),
            (Code::CX, vec![], vec![0, 1]),
            (Code::H, vec![], vec![1]),
            (Code::RZ, vec![-PI / 2.0], vec![0]),
            (Code::RZ, vec![-PI / 2.0], vec![1]),
        ],
        2 => vec![
            (Code::CX, vec![], vec![0, 1]),
            (Code::RX, vec![-2.0 * a], vec![0]),
            (Code::RZ, vec![-2.0 * cc], vec![1]),
            (Code::CX, vec![], vec![0, 1]),
        ],
        _ => vec![
            (Code::RZ, vec![PI / 2.0], vec![1]),
            (Code::CX, vec![], vec![1, 0]),
            (Code::RZ, vec![-PI / 2.0 - 2.0 * cc], vec![0]),
            (Code::RY, vec![-PI / 2.0 - 2.0 * a], vec![1]),
            (Code::CX, vec![], vec![0, 1]),
            (Code::RY, vec![PI / 2.0 + 2.0 * b], vec![1]),
            (Code::CX, vec![], vec![1, 0]),
            (Code::RZ, vec![-PI / 2.0], vec![0]),
        ],
    }
}

fn swap_clifford(pair: (usize, usize)) -> Mat {
    let h = 1.0 / 2f64.sqrt();
    match pair {
        (1, 2) => Mat::from_rows(2, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)]),
        (1, 3) => Mat::from_rows(2, &[c(h, 0.0), c(h, 0.0), c(h, 0.0), c(-h, 0.0)]),
        _ => Mat::from_rows(2, &[c(h, 0.0), c(0.0, -h), c(0.0, -h), c(h, 0.0)]),
    }
}

fn gate_matrix(g: &Gate) -> Mat {
    embed(&super::table::code_matrix(g.0, &g.1), &g.2, 2)
}

/// Gates on local qubits 0 and 1 implementing `u` with the fewest CX, and that
/// number, or `None` if the decomposition did not converge.
pub fn kak_synthesis(u: &Mat) -> Option<(usize, Vec<Gate>)> {
    let atol = 1e-9;
    let (_, (mut a1, mut a2), mut co, (mut b1, mut b2)) = kak_decomposition(u)?;

    for (j, x) in co.iter_mut().enumerate() {
        let k = (*x / (PI / 2.0)).round_ties_even();
        *x -= k * PI / 2.0;
        if (k as i64).rem_euclid(2) == 1 {
            let p = pauli(j);
            b1 = p.mul(&b1);
            b2 = p.mul(&b2);
        }
    }

    let iszero = |x: f64| x.abs() < atol;
    let nzero = co.iter().filter(|&&x| iszero(x)).count();
    let nquarter = co.iter().filter(|&&x| (x.abs() - PI / 4.0).abs() < atol).count();
    let k = if nzero == 3 {
        0
    } else if nzero == 2 && nquarter == 1 {
        1
    } else if nzero >= 1 {
        2
    } else {
        3
    };

    let (target, wanted) = match k {
        1 => (3, co.iter().position(|&x| !iszero(x)).unwrap() + 1),
        2 => (2, co.iter().position(|&x| iszero(x)).unwrap() + 1),
        _ => (0, 0),
    };
    if target != 0 && wanted != target {
        let w = swap_clifford((wanted.min(target), wanted.max(target)));
        co.swap(wanted - 1, target - 1);
        a1 = a1.mul(&w.adjoint());
        a2 = a2.mul(&w.adjoint());
        b1 = w.mul(&b1);
        b2 = w.mul(&b2);
    }
    if k == 1 && co[2] < 0.0 {
        co[2] += PI / 2.0;
        let z = pauli(2);
        b1 = z.mul(&b1);
        b2 = z.mul(&b2);
    }

    let ug = |m: &Mat, q: usize| -> Gate {
        let (t, p, l, g) = zyz(m);
        (Code::U, vec![t, p, l, g], vec![q])
    };
    let mut gates = vec![ug(&b1, 0), ug(&b2, 1)];
    gates.extend(core(k, co));
    gates.push(ug(&a1, 0));
    gates.push(ug(&a2, 1));

    // Every step holds only up to a global phase: measure it and put it in the
    // last single-qubit gate.
    let v = fold(&gates.iter().map(gate_matrix).collect::<Vec<_>>(), 4);
    let (r, col) = v.argmax_abs();
    let chi = (u.get(r, col) / v.get(r, col)).arg();
    if u.dist(&v.scale(C64::from_polar(1.0, chi))) >= 1e-8 {
        return None;
    }
    let last = gates.last_mut().unwrap();
    last.1[3] += chi;
    Some((k, gates))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_u4(seed: u64) -> Mat {
        // A product of fixed rotations is enough to exercise the generic case.
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 * PI
        };
        let u1 = |t: f64, p: f64, l: f64| super::super::table::u_matrix(t, p, l, 0.0);
        let cx = super::super::table::code_matrix(Code::CX, &[]);
        let mut m = Mat::identity(4);
        for _ in 0..4 {
            m = kron(&u1(next(), next(), next()), &u1(next(), next(), next()))
                .mul(&cx)
                .mul(&m);
        }
        m
    }

    #[test]
    fn synthesis_is_exact() {
        for seed in 0..50 {
            let u = random_u4(seed);
            let (k, gates) = kak_synthesis(&u).expect("converges");
            assert!(k <= 3);
            let v = fold(&gates.iter().map(gate_matrix).collect::<Vec<_>>(), 4);
            assert!(v.dist(&u) < 1e-9, "seed {seed}");
        }
    }

    #[test]
    fn cx_counts() {
        let cx = super::super::table::code_matrix(Code::CX, &[]);
        assert_eq!(kak_synthesis(&cx).unwrap().0, 1);
        assert_eq!(kak_synthesis(&Mat::identity(4)).unwrap().0, 0);
        let o = c(0.0, 0.0);
        let l = c(1.0, 0.0);
        let swap = Mat::from_rows(4, &[l, o, o, o, o, o, l, o, o, l, o, o, o, o, o, l]);
        assert_eq!(kak_synthesis(&swap).unwrap().0, 3);
    }
}
