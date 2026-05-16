#![allow(clippy::needless_range_loop)]
//! Diagnostic metrics for transport plans.
//!
//! All functions in this module are non-allocating except for the few that
//! must materialise a temporary distribution (e.g. the mid-point distribution
//! for Jensen-Shannon divergence). They are written so that the user can call
//! them on Sinkhorn / EMD / GW outputs without further conversion.

use crate::error::{OtError, OtResult};

/// Threshold below which a probability is treated as numerically zero.
const TINY: f32 = 1e-12;

/// Returns `(row_violation_inf, col_violation_inf)` where
///
/// ```text
/// row_violation = max_i | Σ_j P_ij − a_i |
/// col_violation = max_j | Σ_i P_ij − b_j |
/// ```
///
/// `plan` is shape `[m × n]` row-major, `a` length `m`, `b` length `n`.
pub fn marginal_violation(
    plan: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) -> OtResult<(f32, f32)> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if plan.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != m || b.len() != n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    for &p in plan {
        if !p.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite plan entry".to_string(),
            });
        }
    }
    let mut row_violation = 0.0_f32;
    for i in 0..m {
        let row_off = i * n;
        let mut row_sum = 0.0_f32;
        for j in 0..n {
            row_sum += plan[row_off + j];
        }
        let r = (row_sum - a[i]).abs();
        if r > row_violation {
            row_violation = r;
        }
    }
    let mut col_violation = 0.0_f32;
    for j in 0..n {
        let mut col_sum = 0.0_f32;
        for i in 0..m {
            col_sum += plan[i * n + j];
        }
        let r = (col_sum - b[j]).abs();
        if r > col_violation {
            col_violation = r;
        }
    }
    Ok((row_violation, col_violation))
}

/// Compute `KL(p ‖ q) = Σ_i p_i · log(p_i / q_i)`. Terms with `p_i ≈ 0`
/// contribute zero (the standard convention `0 · log(0/q) = 0`). Terms with
/// `q_i ≈ 0` and `p_i > 0` produce `+∞`, returned as `f32::INFINITY` and
/// mathematically valid for absolute-continuity diagnostics.
pub fn kl_divergence(p: &[f32], q: &[f32]) -> OtResult<f32> {
    if p.is_empty() || q.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if p.len() != q.len() {
        return Err(OtError::IncompatibleLength {
            a: p.len(),
            b: q.len(),
        });
    }
    for &x in p {
        if !x.is_finite() || x < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    for &x in q {
        if !x.is_finite() || x < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    let mut acc = 0.0_f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi <= TINY {
            continue;
        }
        if qi <= TINY {
            return Ok(f32::INFINITY);
        }
        acc += pi * (pi / qi).ln();
    }
    Ok(acc)
}

/// Compute `Σ_{ij} P_ij · C_ij` for transport cost.
///
/// `plan` and `cost` must have the same length (both flattened in the same
/// row-major layout).
pub fn transport_cost(plan: &[f32], cost: &[f32]) -> OtResult<f32> {
    if plan.is_empty() || cost.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if plan.len() != cost.len() {
        return Err(OtError::IncompatibleLength {
            a: plan.len(),
            b: cost.len(),
        });
    }
    let mut acc = 0.0_f32;
    for (&p, &c) in plan.iter().zip(cost.iter()) {
        if !p.is_finite() || !c.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite plan or cost entry".to_string(),
            });
        }
        if p < 0.0 {
            return Err(OtError::NegativeWeight);
        }
        acc += p * c;
    }
    Ok(acc)
}

/// Compute the Shannon entropy `−Σ P_ij · log P_ij`. Skips zero entries; uses
/// natural log so the result is in nats.
pub fn entropy(plan: &[f32]) -> OtResult<f32> {
    if plan.is_empty() {
        return Err(OtError::EmptyInput);
    }
    let mut acc = 0.0_f32;
    for &p in plan {
        if !p.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite plan entry".to_string(),
            });
        }
        if p < 0.0 {
            return Err(OtError::NegativeWeight);
        }
        if p > TINY {
            acc -= p * p.ln();
        }
    }
    Ok(acc)
}

/// Jensen-Shannon divergence
/// `JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m)` where `m = (p + q) / 2`.
///
/// Bounded by `log 2` (≈ 0.6931) for distributions and is symmetric.
pub fn js_divergence(p: &[f32], q: &[f32]) -> OtResult<f32> {
    if p.is_empty() || q.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if p.len() != q.len() {
        return Err(OtError::IncompatibleLength {
            a: p.len(),
            b: q.len(),
        });
    }
    for &x in p {
        if !x.is_finite() || x < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    for &x in q {
        if !x.is_finite() || x < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    let mut mid = vec![0.0_f32; p.len()];
    for (mi, (&pi, &qi)) in mid.iter_mut().zip(p.iter().zip(q.iter())) {
        *mi = 0.5 * (pi + qi);
    }
    let kl_pm = kl_divergence(p, &mid)?;
    let kl_qm = kl_divergence(q, &mid)?;
    Ok(0.5 * (kl_pm + kl_qm))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn marginal_violation_perfect_plan() {
        // Plan with row sums == a, col sums == b.
        let m = 2;
        let n = 2;
        let plan = vec![0.25_f32, 0.25, 0.25, 0.25];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let (row_v, col_v) = marginal_violation(&plan, &a, &b, m, n).expect("ok");
        assert!(approx(row_v, 0.0, 1e-6));
        assert!(approx(col_v, 0.0, 1e-6));
    }

    #[test]
    fn marginal_violation_off_marginals() {
        let m = 2;
        let n = 2;
        let plan = vec![0.30_f32, 0.20, 0.10, 0.40];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let (row_v, col_v) = marginal_violation(&plan, &a, &b, m, n).expect("ok");
        assert!(approx(row_v, 0.0, 1e-6));
        assert!(approx(col_v, 0.1, 1e-6));
    }

    #[test]
    fn marginal_violation_rejects_shape() {
        let plan = vec![0.0_f32; 5];
        let a = vec![0.5_f32; 2];
        let b = vec![0.5_f32; 2];
        let res = marginal_violation(&plan, &a, &b, 2, 2);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn marginal_violation_rejects_empty() {
        let res = marginal_violation(&[], &[], &[], 0, 0);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn kl_zero_for_identical_distributions() {
        let p = vec![0.25_f32; 4];
        let q = vec![0.25_f32; 4];
        let kl = kl_divergence(&p, &q).expect("ok");
        assert!(approx(kl, 0.0, 1e-5));
    }

    #[test]
    fn kl_known_value() {
        // KL((0.5, 0.5) ‖ (0.25, 0.75)) = 0.5·log(2) + 0.5·log(2/3)
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.25_f32, 0.75];
        let expected = 0.5_f32 * 2.0_f32.ln() + 0.5_f32 * (2.0_f32 / 3.0_f32).ln();
        let kl = kl_divergence(&p, &q).expect("ok");
        assert!(approx(kl, expected, 1e-5));
    }

    #[test]
    fn kl_skips_zero_p_terms() {
        let p = vec![0.0_f32, 1.0];
        let q = vec![0.5_f32, 0.5];
        let kl = kl_divergence(&p, &q).expect("ok");
        assert!(approx(kl, 2.0_f32.ln(), 1e-5));
    }

    #[test]
    fn kl_zero_q_with_positive_p_is_infinity() {
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.0_f32, 1.0];
        let kl = kl_divergence(&p, &q).expect("ok");
        assert!(kl.is_infinite() && kl > 0.0);
    }

    #[test]
    fn kl_rejects_length_mismatch() {
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.5_f32; 3];
        let res = kl_divergence(&p, &q);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn transport_cost_simple() {
        let plan = vec![0.25_f32, 0.25, 0.25, 0.25];
        let cost = vec![1.0_f32, 2.0, 3.0, 4.0];
        let tc = transport_cost(&plan, &cost).expect("ok");
        assert!(approx(tc, 2.5, 1e-6));
    }

    #[test]
    fn transport_cost_zero_for_zero_plan() {
        let plan = vec![0.0_f32; 4];
        let cost = vec![1.0_f32; 4];
        let tc = transport_cost(&plan, &cost).expect("ok");
        assert!(approx(tc, 0.0, 1e-6));
    }

    #[test]
    fn transport_cost_rejects_length_mismatch() {
        let plan = vec![0.5_f32; 3];
        let cost = vec![1.0_f32; 4];
        let res = transport_cost(&plan, &cost);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn entropy_uniform_max() {
        // Uniform 1/n distribution has entropy log(n).
        let n = 4;
        let plan = vec![1.0_f32 / n as f32; n];
        let h = entropy(&plan).expect("ok");
        assert!(approx(h, (n as f32).ln(), 1e-5));
    }

    #[test]
    fn entropy_zero_for_dirac() {
        let plan = vec![0.0_f32, 0.0, 1.0, 0.0];
        let h = entropy(&plan).expect("ok");
        assert!(approx(h, 0.0, 1e-5));
    }

    #[test]
    fn entropy_rejects_negative_entry() {
        let plan = vec![-0.5_f32, 1.5];
        let res = entropy(&plan);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn js_zero_for_identical_distributions() {
        let p = vec![0.25_f32; 4];
        let q = vec![0.25_f32; 4];
        let js = js_divergence(&p, &q).expect("ok");
        assert!(approx(js, 0.0, 1e-5));
    }

    #[test]
    fn js_bounded_by_log2() {
        // Worst case: disjoint support yields JS = log 2.
        let p = vec![1.0_f32, 0.0];
        let q = vec![0.0_f32, 1.0];
        let js = js_divergence(&p, &q).expect("ok");
        assert!(approx(js, 2.0_f32.ln(), 1e-4));
    }

    #[test]
    fn js_symmetric() {
        let p = vec![0.7_f32, 0.3];
        let q = vec![0.4_f32, 0.6];
        let js_pq = js_divergence(&p, &q).expect("ok");
        let js_qp = js_divergence(&q, &p).expect("ok");
        assert!(approx(js_pq, js_qp, 1e-5));
    }

    #[test]
    fn js_rejects_length_mismatch() {
        let p = vec![0.5_f32, 0.5];
        let q = vec![1.0_f32; 3];
        let res = js_divergence(&p, &q);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }
}
