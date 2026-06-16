//! Partial and point-biserial correlation.
//!
//! - **Partial correlation** `r_{xy·Z}` measures the linear association between
//!   `x` and `y` after linearly removing the influence of one or more control
//!   variables `Z`. It is computed by regressing `x` on `Z` and `y` on `Z`
//!   (via the normal equations with an intercept) and correlating the two
//!   residual vectors. The first-order case (a single control `z`) reduces to
//!   the familiar formula
//!   `r_{xy·z} = (r_xy − r_xz·r_yz) / √((1 − r_xz²)(1 − r_yz²))`.
//!   Significance uses a t-test with `n − 2 − q` degrees of freedom, where `q`
//!   is the number of controls.
//!
//! - **Point-biserial correlation** `r_pb` is the Pearson correlation between a
//!   continuous variable and a *dichotomous* (0/1) variable. It is algebraically
//!   equivalent to a two-sample comparison of means and is tested with the same
//!   `t = r·√((n−2)/(1−r²))` statistic on `n − 2` degrees of freedom.
//!
//! ## References
//! - Baba, K., Shibata, R. & Sibuya, M. (2004). "Partial correlation and
//!   conditional correlation as measures of conditional independence."
//!   Aust. N. Z. J. Stat. 46(4).

use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};

/// Result of a partial correlation.
#[derive(Debug, Clone, Copy)]
pub struct PartialCorrResult {
    /// Partial correlation coefficient in `[−1, 1]`.
    pub r: f64,
    /// t-statistic for the null `r = 0`.
    pub t_statistic: f64,
    /// Degrees of freedom (`n − 2 − q`).
    pub df: f64,
    /// Two-sided p-value.
    pub p_value_two_sided: f64,
}

/// Result of a point-biserial correlation.
#[derive(Debug, Clone, Copy)]
pub struct PointBiserialResult {
    /// Point-biserial correlation coefficient in `[−1, 1]`.
    pub r: f64,
    /// t-statistic for the null `r = 0`.
    pub t_statistic: f64,
    /// Degrees of freedom (`n − 2`).
    pub df: f64,
    /// Two-sided p-value.
    pub p_value_two_sided: f64,
}

// ─── Linear helpers ───────────────────────────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Pearson correlation of two equal-length finite vectors (no inference).
fn pearson_raw(a: &[f64], b: &[f64]) -> StatsResult<f64> {
    let ma = mean(a);
    let mb = mean(b);
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        sab += (ai - ma) * (bi - mb);
        saa += (ai - ma).powi(2);
        sbb += (bi - mb).powi(2);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "partial: zero variance in a residual / variable".into(),
        ));
    }
    Ok((sab / (saa * sbb).sqrt()).clamp(-1.0, 1.0))
}

/// Solve the (q+1)×(q+1) normal equations `(Gᵀ G) β = Gᵀ y` for an OLS fit of
/// `y` on a design with an intercept column followed by the `q` controls, and
/// return the residuals `y − ŷ`. Uses Gaussian elimination with partial
/// pivoting (no external linear algebra).
fn regress_out(y: &[f64], controls: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let n = y.len();
    let q = controls.len();
    let p = q + 1; // intercept + controls

    // Build design matrix rows: [1, z_1, ..., z_q].
    // Normal-equation matrix A = GᵀG (p×p) and vector rhs = Gᵀy (p).
    let mut a = vec![0.0_f64; p * p];
    let mut rhs = vec![0.0_f64; p];
    for i in 0..n {
        let mut row = Vec::with_capacity(p);
        row.push(1.0);
        for c in controls {
            row.push(c[i]);
        }
        for r in 0..p {
            rhs[r] += row[r] * y[i];
            for s in 0..p {
                a[r * p + s] += row[r] * row[s];
            }
        }
    }

    // Gaussian elimination with partial pivoting on [A | rhs].
    let mut beta = rhs.clone();
    for col in 0..p {
        // Pivot selection.
        let mut pivot = col;
        let mut best = a[col * p + col].abs();
        for r in (col + 1)..p {
            let v = a[r * p + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return Err(StatsError::SingularMatrix(
                "partial correlation regression".into(),
            ));
        }
        if pivot != col {
            for s in 0..p {
                a.swap(col * p + s, pivot * p + s);
            }
            beta.swap(col, pivot);
        }
        // Eliminate below.
        let diag = a[col * p + col];
        for r in (col + 1)..p {
            let factor = a[r * p + col] / diag;
            if factor != 0.0 {
                for s in col..p {
                    a[r * p + s] -= factor * a[col * p + s];
                }
                beta[r] -= factor * beta[col];
            }
        }
    }
    // Back substitution.
    for col in (0..p).rev() {
        let mut acc = beta[col];
        for s in (col + 1)..p {
            acc -= a[col * p + s] * beta[s];
        }
        beta[col] = acc / a[col * p + col];
    }

    // Residuals y - ŷ.
    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        let mut pred = beta[0];
        for (c_idx, c) in controls.iter().enumerate() {
            pred += beta[c_idx + 1] * c[i];
        }
        resid[i] = y[i] - pred;
    }
    Ok(resid)
}

// ─── Partial correlation ──────────────────────────────────────────────────────

/// Partial correlation of `x` and `y` controlling for the variables in
/// `controls` (each the same length as `x` and `y`).
///
/// With an empty `controls` slice this reduces to an ordinary Pearson
/// correlation. Requires `n > q + 2` observations.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if any input length differs.
/// - [`StatsError::InsufficientSampleSize`] if `n ≤ q + 2`.
/// - [`StatsError::NonFiniteValue`] on non-finite data.
/// - [`StatsError::SingularMatrix`] / [`StatsError::NumericalInstability`] for a
///   rank-deficient control set or zero-variance residuals.
pub fn partial_correlation(
    x: &[f64],
    y: &[f64],
    controls: &[Vec<f64>],
) -> StatsResult<PartialCorrResult> {
    let n = x.len();
    if y.len() != n {
        return Err(StatsError::DimensionMismatch { a: n, b: y.len() });
    }
    for c in controls {
        if c.len() != n {
            return Err(StatsError::DimensionMismatch { a: n, b: c.len() });
        }
    }
    let q = controls.len();
    if n <= q + 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: q + 3,
        });
    }
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    for c in controls {
        for (i, &ci) in c.iter().enumerate() {
            if !ci.is_finite() {
                return Err(StatsError::NonFiniteValue(i));
            }
        }
    }

    let r = if q == 0 {
        pearson_raw(x, y)?
    } else {
        let rx = regress_out(x, controls)?;
        let ry = regress_out(y, controls)?;
        pearson_raw(&rx, &ry)?
    };

    let df = (n as f64) - 2.0 - q as f64;
    let t = r * (df / (1.0 - r * r).max(1e-300)).sqrt();
    let dist = StudentT::new(df)?;
    let cdf_t = dist.cdf(t)?;
    let p = (2.0 * cdf_t.min(1.0 - cdf_t)).clamp(0.0, 1.0);
    Ok(PartialCorrResult {
        r,
        t_statistic: t,
        df,
        p_value_two_sided: p,
    })
}

// ─── Point-biserial correlation ───────────────────────────────────────────────

/// Point-biserial correlation between a continuous variable `x` and a binary
/// grouping `binary` (each entry `0` or any non-zero value treated as `1`).
///
/// Requires at least one observation in each of the two groups and `n > 2`.
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if `x.len() != binary.len()`.
/// - [`StatsError::InsufficientSampleSize`] if `n ≤ 2`.
/// - [`StatsError::InvalidParameter`] if either group is empty.
/// - [`StatsError::NonFiniteValue`] on non-finite `x`.
/// - [`StatsError::NumericalInstability`] if `x` has zero variance.
pub fn point_biserial(x: &[f64], binary: &[u8]) -> StatsResult<PointBiserialResult> {
    let n = x.len();
    if binary.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: n,
            b: binary.len(),
        });
    }
    if n <= 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 3 });
    }
    let mut sum1 = 0.0;
    let mut sum0 = 0.0;
    let mut n1 = 0usize;
    let mut n0 = 0usize;
    for (i, (&xi, &b)) in x.iter().zip(binary.iter()).enumerate() {
        if !xi.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
        if b != 0 {
            sum1 += xi;
            n1 += 1;
        } else {
            sum0 += xi;
            n0 += 1;
        }
    }
    if n1 == 0 || n0 == 0 {
        return Err(StatsError::InvalidParameter {
            name: "binary".into(),
            reason: "both groups must be non-empty".into(),
        });
    }

    let nf = n as f64;
    let mean1 = sum1 / n1 as f64;
    let mean0 = sum0 / n0 as f64;
    // Population standard deviation of all x (the point-biserial convention).
    let mx = x.iter().sum::<f64>() / nf;
    let var: f64 = x.iter().map(|&v| (v - mx).powi(2)).sum::<f64>() / nf;
    let sd = var.sqrt();
    if sd <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "point_biserial: zero variance in x".into(),
        ));
    }

    let p1 = n1 as f64 / nf;
    let p0 = n0 as f64 / nf;
    let r = ((mean1 - mean0) / sd * (p1 * p0).sqrt()).clamp(-1.0, 1.0);

    let df = nf - 2.0;
    let t = r * (df / (1.0 - r * r).max(1e-300)).sqrt();
    let dist = StudentT::new(df)?;
    let cdf_t = dist.cdf(t)?;
    let p = (2.0 * cdf_t.min(1.0 - cdf_t)).clamp(0.0, 1.0);
    Ok(PointBiserialResult {
        r,
        t_statistic: t,
        df,
        p_value_two_sided: p,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_zero_controls_equals_pearson() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let r = partial_correlation(&x, &y, &[]).expect("ok");
        // Compare against the direct Pearson on the same data.
        let direct = pearson_raw(&x, &y).expect("ok");
        assert!((r.r - direct).abs() < 1e-9, "{} vs {}", r.r, direct);
    }

    #[test]
    fn partial_removes_common_cause() {
        // x = z + ε_x, y = z + ε_y; x and y correlate marginally but the
        // partial correlation given z is small.
        let z: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let x: Vec<f64> = z
            .iter()
            .map(|&v| v + ((v as i64 % 3) as f64 - 1.0))
            .collect();
        let y: Vec<f64> = z
            .iter()
            .map(|&v| v + ((v as i64 % 5) as f64 - 2.0))
            .collect();
        let marginal = pearson_raw(&x, &y).expect("ok");
        let partial = partial_correlation(&x, &y, std::slice::from_ref(&z)).expect("ok");
        assert!(marginal > 0.9, "marginal {marginal}");
        assert!(
            partial.r.abs() < marginal,
            "partial {} >= marginal",
            partial.r
        );
    }

    #[test]
    fn partial_first_order_formula_matches() {
        // First-order partial correlation should match the closed-form formula.
        let x = vec![1.0, 2.0, 4.0, 3.0, 7.0, 5.0, 9.0, 8.0];
        let y = vec![2.0, 1.0, 5.0, 4.0, 6.0, 8.0, 7.0, 10.0];
        let z = vec![1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0];
        let r_xy = pearson_raw(&x, &y).expect("ok");
        let r_xz = pearson_raw(&x, &z).expect("ok");
        let r_yz = pearson_raw(&y, &z).expect("ok");
        let closed = (r_xy - r_xz * r_yz) / ((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)).sqrt();
        let computed = partial_correlation(&x, &y, &[z]).expect("ok");
        assert!(
            (computed.r - closed).abs() < 1e-9,
            "{} vs {}",
            computed.r,
            closed
        );
    }

    #[test]
    fn partial_two_controls_finite() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).cos()).collect();
        let z1: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let z2: Vec<f64> = (0..30).map(|i| (i as f64).sqrt()).collect();
        let r = partial_correlation(&x, &y, &[z1, z2]).expect("ok");
        assert!(r.r.is_finite() && (-1.0..=1.0).contains(&r.r));
        assert_eq!(r.df, 30.0 - 2.0 - 2.0);
    }

    #[test]
    fn partial_p_value_in_range() {
        let x: Vec<f64> = (0..25).map(|i| i as f64 + 0.3).collect();
        let y: Vec<f64> = (0..25).map(|i| 2.0 * i as f64).collect();
        let z: Vec<f64> = (0..25).map(|i| (i * i) as f64).collect();
        let r = partial_correlation(&x, &y, &[z]).expect("ok");
        assert!((0.0..=1.0).contains(&r.p_value_two_sided));
    }

    #[test]
    fn partial_dimension_mismatch_error() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(matches!(
            partial_correlation(&x, &y, &[]).unwrap_err(),
            StatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn partial_insufficient_sample_error() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let z = vec![1.0, 2.0, 3.0];
        // n=3, q=1 → need n>q+2=3 → 3 is not > 3 → error.
        assert!(matches!(
            partial_correlation(&x, &y, &[z]).unwrap_err(),
            StatsError::InsufficientSampleSize { .. }
        ));
    }

    #[test]
    fn partial_non_finite_error() {
        let x = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(matches!(
            partial_correlation(&x, &y, &[]).unwrap_err(),
            StatsError::NonFiniteValue(_)
        ));
    }

    #[test]
    fn point_biserial_perfect_separation() {
        // Group 1 strictly above group 0 → strong positive r_pb.
        let x = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let b = vec![0u8, 0, 0, 1, 1, 1];
        let r = point_biserial(&x, &b).expect("ok");
        assert!(r.r > 0.8, "r={}", r.r);
        assert!(r.p_value_two_sided < 0.05, "p={}", r.p_value_two_sided);
    }

    #[test]
    fn point_biserial_sign_flips_with_groups() {
        let x = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let b_pos = vec![0u8, 0, 0, 1, 1, 1];
        let b_neg = vec![1u8, 1, 1, 0, 0, 0];
        let rp = point_biserial(&x, &b_pos).expect("ok");
        let rn = point_biserial(&x, &b_neg).expect("ok");
        assert!((rp.r + rn.r).abs() < 1e-9, "should be opposite signs");
    }

    #[test]
    fn point_biserial_no_separation_small_r() {
        // Interleaved groups → near-zero correlation.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0u8, 1, 0, 1, 0, 1];
        let r = point_biserial(&x, &b).expect("ok");
        assert!(r.r.abs() < 0.5, "r={}", r.r);
    }

    #[test]
    fn point_biserial_nonzero_treated_as_one() {
        let x = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let b = vec![0u8, 0, 0, 2, 5, 9]; // non-zero → group 1
        let r = point_biserial(&x, &b).expect("ok");
        assert!(r.r > 0.8, "r={}", r.r);
    }

    #[test]
    fn point_biserial_one_group_empty_error() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1u8, 1, 1, 1]; // no zeros
        assert!(matches!(
            point_biserial(&x, &b).unwrap_err(),
            StatsError::InvalidParameter { .. }
        ));
    }

    #[test]
    fn point_biserial_dimension_mismatch_error() {
        let x = vec![1.0, 2.0, 3.0];
        let b = vec![0u8, 1];
        assert!(matches!(
            point_biserial(&x, &b).unwrap_err(),
            StatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn point_biserial_zero_variance_error() {
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let b = vec![0u8, 0, 1, 1];
        assert!(matches!(
            point_biserial(&x, &b).unwrap_err(),
            StatsError::NumericalInstability(_)
        ));
    }

    #[test]
    fn point_biserial_r_in_range() {
        let x: Vec<f64> = (0..20).map(|i| (i as f64).sin() * 3.0).collect();
        let b: Vec<u8> = (0..20).map(|i| (i % 2) as u8).collect();
        let r = point_biserial(&x, &b).expect("ok");
        assert!((-1.0..=1.0).contains(&r.r));
        assert!((0.0..=1.0).contains(&r.p_value_two_sided));
    }

    #[test]
    fn partial_deterministic() {
        let x = vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let y = vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
        let z = vec![1.0, 1.0, 2.0, 3.0, 5.0, 8.0];
        let a = partial_correlation(&x, &y, std::slice::from_ref(&z)).expect("ok");
        let b = partial_correlation(&x, &y, &[z]).expect("ok");
        assert_eq!(a.r, b.r);
    }
}
