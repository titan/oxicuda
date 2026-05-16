//! Crossover operators: one-point, two-point, uniform, and Simulated Binary Crossover.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// One-point crossover: split at a random locus; child1 = p1\[..cut\] + p2\[cut..\].
pub fn one_point_crossover(
    p1: &[f64],
    p2: &[f64],
    rng: &mut LcgRng,
) -> EvolResult<(Vec<f64>, Vec<f64>)> {
    let n = p1.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if p2.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: p2.len(),
        });
    }
    let cut = rng.next_usize(n + 1); // [0, n]
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    c1.extend_from_slice(&p1[..cut]);
    c1.extend_from_slice(&p2[cut..]);
    c2.extend_from_slice(&p2[..cut]);
    c2.extend_from_slice(&p1[cut..]);
    Ok((c1, c2))
}

/// Two-point crossover: swap the segment `[cut1, cut2)` between parents.
pub fn two_point_crossover(
    p1: &[f64],
    p2: &[f64],
    rng: &mut LcgRng,
) -> EvolResult<(Vec<f64>, Vec<f64>)> {
    let n = p1.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if p2.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: p2.len(),
        });
    }
    let mut c1 = p1.to_vec();
    let mut c2 = p2.to_vec();
    let mut cut1 = rng.next_usize(n + 1);
    let mut cut2 = rng.next_usize(n + 1);
    if cut1 > cut2 {
        std::mem::swap(&mut cut1, &mut cut2);
    }
    if cut1 < cut2 {
        c1[cut1..cut2].copy_from_slice(&p2[cut1..cut2]);
        c2[cut1..cut2].copy_from_slice(&p1[cut1..cut2]);
    }
    Ok((c1, c2))
}

/// Uniform crossover: for each gene, swap with probability `p_swap`.
pub fn uniform_crossover(
    p1: &[f64],
    p2: &[f64],
    p_swap: f64,
    rng: &mut LcgRng,
) -> EvolResult<(Vec<f64>, Vec<f64>)> {
    let n = p1.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if p2.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: p2.len(),
        });
    }
    if !(0.0..=1.0).contains(&p_swap) {
        return Err(EvolError::InvalidParameter(format!(
            "p_swap {p_swap} is outside [0,1]"
        )));
    }
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        if rng.next_f64() < p_swap {
            c1.push(p2[i]);
            c2.push(p1[i]);
        } else {
            c1.push(p1[i]);
            c2.push(p2[i]);
        }
    }
    Ok((c1, c2))
}

/// Simulated Binary Crossover (SBX): generates offspring that mimic the distribution
/// produced by one-point crossover in the binary encoding domain.
///
/// Distribution index `eta` controls spread: larger eta → offspring closer to parents.
/// Typical value: `eta = 20.0`.
pub fn sbx_crossover(
    p1: &[f64],
    p2: &[f64],
    eta: f64,
    bounds: (f64, f64),
    rng: &mut LcgRng,
) -> EvolResult<(Vec<f64>, Vec<f64>)> {
    let n = p1.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if p2.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: p2.len(),
        });
    }
    if eta <= 0.0 {
        return Err(EvolError::InvalidParameter(format!(
            "SBX eta ({eta}) must be positive"
        )));
    }
    let (lb, ub) = bounds;
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        if rng.next_f64() <= 0.5 {
            // This gene is crossed over
            let x1 = p1[i].max(lb).min(ub);
            let x2 = p2[i].max(lb).min(ub);
            let (lo, hi) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
            let diff = (hi - lo).max(1e-14);
            let u = rng.next_f64();
            // beta_q from Deb & Agrawal (1995)
            let beta_q = if u <= 0.5 {
                let beta = 1.0 + 2.0 * (lo - lb) / diff;
                let alpha = 2.0 - beta.powf(-(eta + 1.0));
                let u2 = u * 2.0;
                if u2 <= 1.0 / alpha {
                    (u2 * alpha).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 - u2 * alpha)).powf(1.0 / (eta + 1.0))
                }
            } else {
                let beta = 1.0 + 2.0 * (ub - hi) / diff;
                let alpha = 2.0 - beta.powf(-(eta + 1.0));
                let u2 = 2.0 * (1.0 - u);
                if u2 <= 1.0 / alpha {
                    (u2 * alpha).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 - u2 * alpha)).powf(1.0 / (eta + 1.0))
                }
            };
            let o1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1));
            let o2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1));
            c1.push(o1.max(lb).min(ub));
            c2.push(o2.max(lb).min(ub));
        } else {
            c1.push(p1[i]);
            c2.push(p2[i]);
        }
    }
    Ok((c1, c2))
}
