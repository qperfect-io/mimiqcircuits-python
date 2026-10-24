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
//! Synthesis of linear reversible circuits.
//!
//! A circuit of X, CX and SWAP maps the basis state x to A x ⊕ b over GF(2).
//! Row i of A lists the input bits whose parity becomes output bit i, and a CX
//! with control c and target t adds row c to row t.

use std::collections::HashMap;

/// Row additions (c, t), row t ⊻= row c, that bring `a` (n × n, row-major) to
/// upper triangular form with a unit diagonal, in sections of `m` columns
/// (Patel, Markov and Hayes, Quantum Inf. Comput. 8, 282 (2008)). `None` if
/// `a` is singular.
fn pmh_lower(a: &mut [bool], n: usize, m: usize) -> Option<Vec<(usize, usize)>> {
    let mut ops = vec![];
    let addrow = |a: &mut [bool], ops: &mut Vec<(usize, usize)>, c: usize, t: usize| {
        for j in 0..n {
            a[t * n + j] ^= a[c * n + j];
        }
        ops.push((c, t));
    };
    let mut s = 0;
    while s < n {
        let section = s..(s + m).min(n);
        let mut seen: HashMap<Vec<bool>, usize> = HashMap::new();
        for r in s..n {
            let pattern: Vec<bool> = section.clone().map(|j| a[r * n + j]).collect();
            if !pattern.iter().any(|&x| x) {
                continue;
            }
            let first = *seen.entry(pattern).or_insert(r);
            if first != r {
                addrow(a, &mut ops, first, r);
            }
        }
        for col in section {
            if !a[col * n + col] {
                let r = (col + 1..n).find(|&r| a[r * n + col])?;
                addrow(a, &mut ops, r, col);
            }
            for r in col + 1..n {
                if a[r * n + col] {
                    addrow(a, &mut ops, col, r);
                }
            }
        }
        s += m;
    }
    Some(ops)
}

/// CX gates (control, target) implementing x ↦ A x, for one section size.
fn pmh_synthesis(a: &[bool], n: usize, m: usize) -> Option<Vec<(usize, usize)>> {
    let mut u = a.to_vec();
    let lower = pmh_lower(&mut u, n, m)?;
    let mut ut = vec![false; n * n];
    for r in 0..n {
        for c in 0..n {
            ut[c * n + r] = u[r * n + c];
        }
    }
    let upper = pmh_lower(&mut ut, n, m)?;
    let mut out: Vec<(usize, usize)> = upper.iter().map(|&(c, t)| (t, c)).collect();
    out.extend(lower.iter().rev());
    Some(out)
}

/// CX gates as (control, target), then the qubits that get an X.
pub type Synthesis = (Vec<(usize, usize)>, Vec<usize>);

/// CX gates for A, trying every section size up to log2(n) and keeping the
/// fewest, then the X gates for b.
pub fn linear_synthesis(a: &[bool], b: &[bool], n: usize) -> Option<Synthesis> {
    let mut best: Option<Vec<(usize, usize)>> = None;
    let top = ((n as f64).log2().ceil() as usize).max(1);
    for m in 1..=top {
        let ops = pmh_synthesis(a, n, m)?;
        if best.as_ref().is_none_or(|b| ops.len() < b.len()) {
            best = Some(ops);
        }
    }
    let xs = (0..n).filter(|&q| b[q]).collect();
    Some((best.unwrap(), xs))
}

/// (A, b) of a sequence of linear gates (1 X, 2 CX, 3 SWAP) on local qubits.
pub fn affine_map(gates: &[(u8, Vec<usize>)], n: usize) -> (Vec<bool>, Vec<bool>) {
    let mut a = vec![false; n * n];
    for i in 0..n {
        a[i * n + i] = true;
    }
    let mut b = vec![false; n];
    for (kind, qs) in gates {
        match kind {
            1 => b[qs[0]] ^= true,
            2 => {
                let (c, t) = (qs[0], qs[1]);
                for j in 0..n {
                    a[t * n + j] ^= a[c * n + j];
                }
                b[t] ^= b[c];
            }
            _ => {
                let (p, q) = (qs[0], qs[1]);
                for j in 0..n {
                    a.swap(p * n + j, q * n + j);
                }
                b.swap(p, q);
            }
        }
    }
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_reproduces_the_map() {
        let mut s: u64 = 7;
        let mut next = |k: usize| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as usize) % k
        };
        for n in [2usize, 3, 5, 9, 17] {
            for _ in 0..10 {
                let mut gates = vec![];
                for _ in 0..3 * n * n {
                    let c = next(n);
                    let t = (c + 1 + next(n - 1)) % n;
                    gates.push((if next(5) == 0 { 1 } else { 2 }, vec![c, t]));
                }
                let (a, b) = affine_map(&gates, n);
                let (cx, xs) = linear_synthesis(&a, &b, n).unwrap();
                let mut again: Vec<(u8, Vec<usize>)> = cx.iter().map(|&(c, t)| (2u8, vec![c, t])).collect();
                again.extend(xs.iter().map(|&q| (1u8, vec![q])));
                assert_eq!(affine_map(&again, n), (a, b));
            }
        }
    }
}
