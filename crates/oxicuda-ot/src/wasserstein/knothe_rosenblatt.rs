//! Knothe-Rosenblatt (KR) triangular transport map.
//!
//! The Knothe-Rosenblatt rearrangement is a triangular transport coupling
//! constructed via conditional CDFs. In 1D it reduces to the classical quantile
//! coupling: `T(x) = Q_Y(F_X(x))`, where `F_X` is the empirical CDF of the
//! source and `Q_Y` is the quantile function of the target.
//!
//! Key properties:
//! - The KR map is the limit of regularised OT as ε → 0 under special orderings.
//! - In 1D it gives the closed-form W_1 / W_2 optimal coupling.
//! - The multivariate extension used here is the marginal (dimension-wise)
//!   approximation: one 1D KR per coordinate (independent-marginals baseline).

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the KR coupling estimator.
#[derive(Debug, Clone)]
pub struct KrConfig {
    /// Number of quantile evaluation points for CDF/quantile interpolation.
    /// Larger values give finer interpolation but are more expensive.
    pub n_quantile_points: usize,
}

impl Default for KrConfig {
    fn default() -> Self {
        Self {
            n_quantile_points: 100,
        }
    }
}

/// A fitted 1D Knothe-Rosenblatt coupling.
///
/// Stores the empirical CDF and quantile function of both source and target
/// so that new points can be mapped via the KR transform.
#[derive(Debug, Clone)]
pub struct KrFit {
    /// Sorted source support points (ascending), length `n`.
    pub source_sorted: Vec<f64>,
    /// Empirical CDF values at each source point: `F_X(x_i) = (i+1)/n`.
    pub source_cdf: Vec<f64>,
    /// Sorted target support points (ascending), length `m`.
    pub target_sorted: Vec<f64>,
    /// Empirical CDF values at each target point: `F_Y(y_j) = (j+1)/m`.
    pub target_cdf: Vec<f64>,
    /// Quantile grid probabilities in [0, 1], length `n_quantile_points`.
    pub source_quantiles: Vec<f64>,
    /// Quantile values for the target at the grid probabilities.
    pub target_quantiles: Vec<f64>,
    /// Number of source samples.
    pub n: usize,
    /// Number of target samples.
    pub m: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sort a slice of f64 values in ascending order (NaN-safe, NaN goes last).
fn sort_f64(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Evaluate the empirical CDF `F_X(t)` for sorted samples using linear
/// interpolation between sample points.
///
/// - For `t < sorted[0]`: returns 0.0.
/// - For `t >= sorted[n-1]`: returns 1.0.
/// - Otherwise: linearly interpolates between the bracket CDF values.
fn empirical_cdf_eval(sorted: &[f64], t: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if t < sorted[0] {
        return 0.0;
    }
    if t >= sorted[n - 1] {
        return 1.0;
    }
    // Binary search for the position of t.
    let mut lo = 0_usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if sorted[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // CDF values at bracket: F(sorted[lo]) = (lo+1)/n, F(sorted[hi]) = (hi+1)/n
    let f_lo = (lo + 1) as f64 / n as f64;
    let f_hi = (hi + 1) as f64 / n as f64;
    let x_lo = sorted[lo];
    let x_hi = sorted[hi];
    if (x_hi - x_lo).abs() < f64::EPSILON {
        return f_hi;
    }
    f_lo + (f_hi - f_lo) * (t - x_lo) / (x_hi - x_lo)
}

/// Evaluate the empirical quantile function `Q_Y(p)` for sorted target samples.
///
/// Uses linear interpolation between the bracket CDF values:
/// `Q_Y(p) ≈ lerp(sorted[lo], sorted[hi], (p - F(lo)) / (F(hi) - F(lo)))`.
///
/// - For `p ≤ 0`: returns `sorted[0]`.
/// - For `p ≥ 1`: returns `sorted[m-1]`.
fn empirical_quantile_eval(sorted: &[f64], p: f64) -> f64 {
    let m = sorted.len();
    if m == 0 {
        return 0.0;
    }
    if p <= 0.0 {
        return sorted[0];
    }
    if p >= 1.0 {
        return sorted[m - 1];
    }
    // Find bracket index: we want i such that F_Y(sorted[i]) ≥ p.
    // F_Y(sorted[i]) = (i+1)/m  =>  i = ceil(p*m) - 1
    let raw_idx = p * m as f64;
    let idx = (raw_idx.ceil() as usize).saturating_sub(1).min(m - 1);

    if idx == 0 {
        return sorted[0];
    }

    let f_lo = idx as f64 / m as f64; // = (idx)/m
    let f_hi = (idx + 1) as f64 / m as f64; // = (idx+1)/m
    let x_lo = sorted[idx - 1];
    let x_hi = sorted[idx];

    if (f_hi - f_lo).abs() < f64::EPSILON {
        return x_hi;
    }
    let t = (p - f_lo) / (f_hi - f_lo);
    x_lo + t * (x_hi - x_lo)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit the 1D Knothe-Rosenblatt coupling from `source` to `target` samples.
///
/// Both slices may have different lengths. The fit stores sorted copies,
/// empirical CDFs, and a fine quantile grid for the target.
///
/// # Errors
///
/// Returns [`OtError::EmptyInput`] if either slice is empty.
pub fn kr_fit_1d(source: &[f64], target: &[f64]) -> OtResult<KrFit> {
    kr_fit_1d_with_config(source, target, &KrConfig::default())
}

/// Fit the 1D KR coupling with explicit config (quantile grid size).
pub fn kr_fit_1d_with_config(source: &[f64], target: &[f64], cfg: &KrConfig) -> OtResult<KrFit> {
    if source.is_empty() || target.is_empty() {
        return Err(OtError::EmptyInput);
    }
    let n = source.len();
    let m = target.len();
    let nq = cfg.n_quantile_points.max(2);

    // Sort copies
    let mut source_sorted = source.to_vec();
    let mut target_sorted = target.to_vec();
    sort_f64(&mut source_sorted);
    sort_f64(&mut target_sorted);

    // Empirical CDF values at the sorted sample points
    let source_cdf: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
    let target_cdf: Vec<f64> = (1..=m).map(|j| j as f64 / m as f64).collect();

    // Quantile grid: uniformly spaced probabilities in (0, 1)
    let source_quantiles: Vec<f64> = (0..nq).map(|k| (k as f64 + 0.5) / nq as f64).collect();

    // Evaluate target quantile function at those probability levels
    let target_quantiles: Vec<f64> = source_quantiles
        .iter()
        .map(|&p| empirical_quantile_eval(&target_sorted, p))
        .collect();

    Ok(KrFit {
        source_sorted,
        source_cdf,
        target_sorted,
        target_cdf,
        source_quantiles,
        target_quantiles,
        n,
        m,
    })
}

/// Apply the fitted KR map `T(x) = Q_Y(F_X(x))` to a slice of new points.
///
/// The evaluation uses linear interpolation of `F_X` followed by linear
/// interpolation of `Q_Y`.
///
/// # Errors
///
/// Returns [`OtError::EmptyInput`] if `x` is empty.
pub fn kr_transform_1d(fit: &KrFit, x: &[f64]) -> OtResult<Vec<f64>> {
    if x.is_empty() {
        return Err(OtError::EmptyInput);
    }
    let mapped: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let p = empirical_cdf_eval(&fit.source_sorted, xi);
            empirical_quantile_eval(&fit.target_sorted, p)
        })
        .collect();
    Ok(mapped)
}

/// Compute the 1D Wasserstein-1 (Earth Mover's Distance) via the KR map.
///
/// `W_1(X, Y) = (1/n) Σ_i |T(x_i) − x_i|`
///
/// For equal-weight empirical distributions this equals
/// `∫ |F_X(t) − F_Y(t)| dt` via the dual representation.
///
/// # Errors
///
/// Returns [`OtError::EmptyInput`] if either slice is empty.
pub fn kr_wasserstein_1d(source: &[f64], target: &[f64]) -> OtResult<f64> {
    let fit = kr_fit_1d(source, target)?;
    let t_of_x = kr_transform_1d(&fit, source)?;
    let n = source.len() as f64;
    let w1 = source
        .iter()
        .zip(t_of_x.iter())
        .map(|(&xi, &txi)| (txi - xi).abs())
        .sum::<f64>()
        / n;
    Ok(w1)
}

/// Compute the 1D transport cost under the KR map (squared-L2 version).
///
/// Returns `(1/n) Σ_i ||x_i − T(x_i)||^2`.
pub fn kr_transport_cost_1d(fit: &KrFit, source: &[f64]) -> f64 {
    let n = source.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let t_of_x = kr_transform_1d(fit, source).unwrap_or_default();
    source
        .iter()
        .zip(t_of_x.iter())
        .map(|(&xi, &txi)| {
            let d = txi - xi;
            d * d
        })
        .sum::<f64>()
        / n
}

// ─────────────────────────────────────────────────────────────────────────────
// Multivariate (independent-marginals) API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit independent 1D KR couplings for each of the `d` dimensions.
///
/// `source` is `n × d` row-major; `target` is `m × d` row-major.
/// Returns a `Vec` of `d` independent [`KrFit`] objects, one per dimension.
///
/// # Errors
///
/// Returns errors if shapes are invalid or any dimension is empty.
pub fn kr_fit_nd(
    source: &[f64],
    target: &[f64],
    n: usize,
    m: usize,
    d: usize,
) -> OtResult<Vec<KrFit>> {
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if source.len() != n * d {
        return Err(OtError::IncompatibleLength {
            a: source.len(),
            b: n * d,
        });
    }
    if target.len() != m * d {
        return Err(OtError::IncompatibleLength {
            a: target.len(),
            b: m * d,
        });
    }

    let cfg = KrConfig::default();
    let mut fits = Vec::with_capacity(d);
    for dim in 0..d {
        let src_col: Vec<f64> = (0..n).map(|i| source[i * d + dim]).collect();
        let tgt_col: Vec<f64> = (0..m).map(|j| target[j * d + dim]).collect();
        let fit = kr_fit_1d_with_config(&src_col, &tgt_col, &cfg)?;
        fits.push(fit);
    }
    Ok(fits)
}

/// Apply `d` independent 1D KR maps to multi-dimensional input samples.
///
/// `x` is `n × d` row-major. Returns a `Vec<f64>` of the same shape with
/// each dimension independently transformed.
///
/// # Errors
///
/// Returns [`OtError::IncompatibleLength`] if `x.len() != n * d`.
pub fn kr_transform_nd(fits: &[KrFit], x: &[f64], n: usize, d: usize) -> OtResult<Vec<f64>> {
    if fits.len() != d {
        return Err(OtError::BadDim { got: fits.len() });
    }
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    if x.len() != n * d {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: n * d,
        });
    }

    let mut out = vec![0.0_f64; n * d];
    for dim in 0..d {
        let col: Vec<f64> = (0..n).map(|i| x[i * d + dim]).collect();
        let mapped = kr_transform_1d(&fits[dim], &col)?;
        for (i, &val) in mapped.iter().enumerate() {
            out[i * d + dim] = val;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mapping_same_distribution() {
        // T should map source to itself when source == target
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fit = kr_fit_1d(&source, &source).expect("ok");
        let mapped = kr_transform_1d(&fit, &source).expect("ok");
        for (xi, ti) in source.iter().zip(mapped.iter()) {
            assert!((xi - ti).abs() < 0.5, "xi={xi} ti={ti}");
        }
    }

    #[test]
    fn wasserstein_zero_same_distribution() {
        let source = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let w1 = kr_wasserstein_1d(&source, &source).expect("ok");
        assert!(w1 < 1e-10, "W1 should be 0 for same dist, got {w1}");
    }

    #[test]
    fn wasserstein_1d_translation() {
        // X ~ {0,1,2,3,4}, Y ~ {5,6,7,8,9} => W1 ≈ 5
        let source: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let target: Vec<f64> = (5..10).map(|i| i as f64).collect();
        let w1 = kr_wasserstein_1d(&source, &target).expect("ok");
        assert!((w1 - 5.0).abs() < 0.5, "W1 ≈ 5, got {w1}");
    }

    #[test]
    fn kr_map_pushes_uniform_to_uniform() {
        // Source: uniform on {0,1,...,9}, target: uniform on {10,...,19}
        // The KR map should translate each source point by ~10.
        let source: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let target: Vec<f64> = (10..20).map(|i| i as f64).collect();
        let fit = kr_fit_1d(&source, &target).expect("ok");
        let mapped = kr_transform_1d(&fit, &source).expect("ok");
        for &ti in &mapped {
            assert!(
                (9.0..=20.0).contains(&ti),
                "mapped value {ti} out of target range"
            );
        }
    }

    #[test]
    fn transport_cost_1d_non_negative() {
        let source = vec![0.0, 1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let fit = kr_fit_1d(&source, &target).expect("ok");
        let cost = kr_transport_cost_1d(&fit, &source);
        assert!(cost >= 0.0, "cost should be non-negative, got {cost}");
    }

    #[test]
    fn transport_cost_zero_same_distribution() {
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fit = kr_fit_1d(&source, &source).expect("ok");
        let cost = kr_transport_cost_1d(&fit, &source);
        assert!(cost < 0.5, "cost ≈ 0 for same dist, got {cost}");
    }

    #[test]
    fn kr_fit_nd_independent_marginals() {
        // 2D: source = grid 0..4, target = grid 10..14
        let n = 5;
        let m = 5;
        let d = 2;
        let source: Vec<f64> = (0..n)
            .flat_map(|i| vec![i as f64, i as f64 * 2.0])
            .collect();
        let target: Vec<f64> = (0..m)
            .flat_map(|i| vec![(i + 10) as f64, (i + 10) as f64 * 2.0])
            .collect();
        let fits = kr_fit_nd(&source, &target, n, m, d).expect("ok");
        assert_eq!(fits.len(), d);
    }

    #[test]
    fn kr_transform_nd_output_shape() {
        let n = 4;
        let m = 4;
        let d = 3;
        let source: Vec<f64> = (0..n * d).map(|i| i as f64).collect();
        let target: Vec<f64> = (0..m * d).map(|i| (i + 10) as f64).collect();
        let fits = kr_fit_nd(&source, &target, n, m, d).expect("ok");
        let out = kr_transform_nd(&fits, &source, n, d).expect("ok");
        assert_eq!(out.len(), n * d);
    }

    #[test]
    fn empty_input_returns_error() {
        let res = kr_fit_1d(&[], &[1.0, 2.0]);
        assert!(matches!(res, Err(OtError::EmptyInput)));
        let res2 = kr_fit_1d(&[1.0, 2.0], &[]);
        assert!(matches!(res2, Err(OtError::EmptyInput)));
    }

    #[test]
    fn kr_fit_nd_bad_dim_returns_error() {
        let res = kr_fit_nd(&[1.0, 2.0], &[3.0, 4.0], 2, 2, 0);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn kr_wasserstein_is_finite_and_nonneg() {
        let source: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5];
        let target: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let w1 = kr_wasserstein_1d(&source, &target).expect("ok");
        assert!(w1.is_finite() && w1 >= 0.0, "W1={w1}");
    }

    #[test]
    fn empirical_cdf_monotone() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut prev = 0.0_f64;
        for &x in &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.5] {
            let f = empirical_cdf_eval(&sorted, x);
            assert!(f >= prev - 1e-10, "CDF not monotone at x={x}: {f} < {prev}");
            assert!((0.0..=1.0).contains(&f), "CDF out of [0,1] at x={x}");
            prev = f;
        }
    }
}
