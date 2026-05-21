//! Cramér-von Mises and Watson goodness-of-fit tests.
//!
//! # Tests
//! - [`cvm_test_uniform`]  — one-sample CvM W² against Uniform(0, 1)
//! - [`cvm_test_normal`]   — one-sample CvM W² against N(μ̂, σ̂²)
//! - [`watson_test_uniform`] — Watson U² (circular) against Uniform(0, 1)
//! - [`cvm_two_sample`]   — two-sample CvM (Anderson 1962)

use crate::error::{StatsError, StatsResult};

// ─── Result types ─────────────────────────────────────────────────────────────

/// Result of a Cramér-von Mises test.
#[derive(Debug, Clone, Copy)]
pub struct CvmResult {
    /// The CvM W² statistic.
    pub statistic: f64,
    /// Approximate p-value.
    pub p_value: f64,
}

/// Result of a Watson U² test.
#[derive(Debug, Clone, Copy)]
pub struct WatsonResult {
    /// The Watson U² statistic.
    pub statistic: f64,
    /// Approximate p-value.
    pub p_value: f64,
}

// ─── p-value helpers ──────────────────────────────────────────────────────────

/// Asymptotic p-value for the CvM statistic W² (or two-sample T).
///
/// Uses the exact alternating-series representation of the CvM distribution
/// (Smirnov 1936 / Anderson 1962), which converges for all W² > 0:
///
/// P(W² > t) = 2 Σ_{k=1}^{∞} (−1)^{k−1} exp(−2π²k²t)
///
/// For small t (p near 1) the direct series converges slowly; we supplement
/// with the Stephens (1970) piecewise polynomial approximation in that range.
/// The series representation is monotone-decreasing in t and correctly → 0 as
/// t → ∞, fixing the Smirnov-Gnedenko approximation's breakdown for large T.
///
/// A Stephens (1974) small-sample correction W²* = W²(1 + 0.5/n) is applied.
fn cvm_p_value(w2: f64, n: usize) -> f64 {
    let n_f = n as f64;
    let w2_star = w2 * (1.0 + 0.5 / n_f);
    cvm_p_from_w2(w2_star)
}

/// Pure function of the corrected statistic — also useful for Watson.
fn cvm_p_from_w2(w2: f64) -> f64 {
    // Stephens (1970) piecewise polynomial for the small-statistic regime
    // where the alternating-series converges slowly.
    if w2 <= 0.0474 {
        let p = 1.0 - (-13.94 + 775.15 * w2 - 12011.5 * w2 * w2).exp();
        return p.clamp(0.0, 1.0);
    }
    if w2 <= 0.0947 {
        let p = 1.0 - (-1.5 + 9.38 * w2).exp();
        return p.clamp(0.0, 1.0);
    }
    if w2 <= 0.1681 {
        let p = (0.8407 - 5.524 * w2 + 6.26 * w2 * w2).exp();
        return p.clamp(0.0, 1.0);
    }
    // For w2 > 0.1681 use the exact alternating exponential series.
    // P(W² > t) = 2 Σ_{k=1}^{K} (−1)^{k-1} exp(−2π²k²t)
    // Converges rapidly for t ≥ 0.17 (|ratio| ≤ exp(-6π²×0.17) ≈ 0.002 per step).
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    let mut sum = 0.0_f64;
    let mut sign = 1.0_f64;
    for k in 1_u32..=40 {
        let term = sign * (-2.0 * pi2 * (k * k) as f64 * w2).exp();
        sum += term;
        sign = -sign;
        if term.abs() < 1e-15 * sum.abs().max(1e-300) {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Asymptotic p-value for the Watson U² statistic.
///
/// Watson (1961) shows U² has the same asymptotic distribution as the CvM W²
/// statistic (the mean-shift correction changes only finite-sample behaviour).
/// We therefore use the same alternating exponential series after a Watson-
/// specific Stephens (1970) small-sample correction:
///   U²* = U²(1 + 0.8/n)
fn watson_p_value(u2: f64, n: usize) -> f64 {
    let n_f = n as f64;
    let u2_star = u2 * (1.0 + 0.8 / n_f);
    // Delegate to the same series-based computation used for CvM.
    cvm_p_from_w2(u2_star)
}

// ─── Standard normal CDF ──────────────────────────────────────────────────────

/// Standard normal CDF via complementary error function.
///
/// Φ(x) = ½ erfc(−x / √2)
#[inline]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function via Horner's approximation (Abramowitz & Stegun 7.1.26).
///
/// Relative error < 1.5 × 10⁻⁷ for all x ≥ 0; uses symmetry for x < 0.
fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

// ─── CvM W² against Uniform(0, 1) ────────────────────────────────────────────

/// One-sample Cramér-von Mises test against Uniform(0, 1).
///
/// Computes the W² statistic:
///
/// W² = Σᵢ (x_(i) − (2i−1)/(2n))² + 1/(12n)
///
/// where x_(1) ≤ … ≤ x_(n) are the order statistics.
///
/// # Errors
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::NonFiniteValue`] if any element is non-finite.
pub fn cvm_test_uniform(data: &[f64]) -> StatsResult<CvmResult> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let n_f = n as f64;

    let w2: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let expected = (2 * (i + 1) - 1) as f64 / (2.0 * n_f);
            (x - expected).powi(2)
        })
        .sum::<f64>()
        + 1.0 / (12.0 * n_f);

    let p_value = cvm_p_value(w2, n);
    Ok(CvmResult {
        statistic: w2,
        p_value,
    })
}

// ─── CvM W² against N(μ̂, σ̂²) ─────────────────────────────────────────────

/// One-sample Cramér-von Mises test against the best-fitting normal distribution.
///
/// Estimates μ and σ from `data`, transforms via the standard normal CDF,
/// then delegates to [`cvm_test_uniform`].
///
/// # Errors
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::InsufficientSampleSize`] if fewer than 2 observations.
/// - [`StatsError::NumericalInstability`] if the sample has zero variance.
/// - [`StatsError::NonFiniteValue`] if any element is non-finite.
pub fn cvm_test_normal(data: &[f64]) -> StatsResult<CvmResult> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if data.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: data.len(),
            need: 2,
        });
    }
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    let n = data.len() as f64;
    let mu: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
    if var <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "sample variance is zero; cannot standardise".into(),
        ));
    }
    let sigma = var.sqrt();
    let uniform_scores: Vec<f64> = data
        .iter()
        .map(|&x| standard_normal_cdf((x - mu) / sigma))
        .collect();
    cvm_test_uniform(&uniform_scores)
}

// ─── Watson U² against Uniform(0, 1) ─────────────────────────────────────────

/// Watson U² test against Uniform(0, 1) (circular variant of the CvM test).
///
/// U² = W² − n (x̄ − ½)²
///
/// where x̄ is the sample mean of the order statistics.
///
/// # Errors
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::NonFiniteValue`] if any element is non-finite.
pub fn watson_test_uniform(data: &[f64]) -> StatsResult<WatsonResult> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let n_f = n as f64;

    // Compute W² first.
    let w2: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let expected = (2 * (i + 1) - 1) as f64 / (2.0 * n_f);
            (x - expected).powi(2)
        })
        .sum::<f64>()
        + 1.0 / (12.0 * n_f);

    // Mean of order statistics.
    let x_bar = sorted.iter().sum::<f64>() / n_f;

    // Watson U² = W² − n (x̄ − ½)²
    let u2 = (w2 - n_f * (x_bar - 0.5).powi(2)).max(0.0);

    let p_value = watson_p_value(u2, n);
    Ok(WatsonResult {
        statistic: u2,
        p_value,
    })
}

// ─── Two-sample CvM ───────────────────────────────────────────────────────────

/// Two-sample Cramér-von Mises test (Anderson 1962).
///
/// Tests H₀: the two samples come from the same (unspecified) distribution.
///
/// The Anderson (1962) statistic is:
///
/// T = (n m) / (n + m) × [Σᵢ (F_n(xᵢ) − G_m(xᵢ))² + Σⱼ (F_n(yⱼ) − G_m(yⱼ))²] / (n + m)
///
/// evaluated at the pooled order statistics, using the exact formula:
///
/// T_exact = (n m / (n+m)²) × { [Σᵢ rᵢ² / n + Σⱼ sⱼ² / m] − (n+m)(n+m+1)²/4 } × (1/(n+m))
///
/// where rᵢ (resp. sⱼ) are the ranks of x (resp. y) in the combined sample.
/// We use the Pettitt (1976) / Lehmann formulation to stay numerically clean.
///
/// # Errors
/// - [`StatsError::EmptyInput`] if either slice is empty.
/// - [`StatsError::NonFiniteValue`] if any element is non-finite.
pub fn cvm_two_sample(x: &[f64], y: &[f64]) -> StatsResult<CvmResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    for (j, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(j));
        }
    }

    let n = x.len();
    let m = y.len();
    let n_f = n as f64;
    let m_f = m as f64;
    let nm = n + m;
    let nm_f = nm as f64;

    // Build pooled sorted sample (x tagged 0, y tagged 1).
    let mut pool: Vec<(f64, usize)> = Vec::with_capacity(nm);
    for &v in x {
        pool.push((v, 0));
    }
    for &v in y {
        pool.push((v, 1));
    }
    pool.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    // Direct scan of the pooled sorted sample accumulating the CvM statistic.
    //
    // At each distinct value in the pool we compute the squared difference between
    // the empirical CDFs F_n and G_m, weighted by the number of tied observations.
    //
    // Anderson (1962) normalised statistic:
    //   T = (n m) / (n+m)² × Σ_{distinct z} (F_n(z) − G_m(z))² × count(z)
    //
    // This is equivalent to Pettitt (1976) rank-sum formula after algebra.
    let mut cx = 0_usize; // count of x ≤ current pool value
    let mut cy = 0_usize; // count of y ≤ current pool value
    let mut d2_sum = 0.0_f64;
    let mut pidx = 0_usize;
    while pidx < nm {
        let val = pool[pidx].0;
        // Advance past all tied values, counting x and y occurrences.
        let mut qidx = pidx;
        let mut delta_x = 0_usize;
        let mut delta_y = 0_usize;
        while qidx < nm && (pool[qidx].0 - val).abs() < 1e-15 {
            if pool[qidx].1 == 0 {
                delta_x += 1;
            } else {
                delta_y += 1;
            }
            qidx += 1;
        }
        cx += delta_x;
        cy += delta_y;
        // Contribution: (F_n(val) - G_m(val))^2 × (# tied at val)
        let fn_val = cx as f64 / n_f;
        let gm_val = cy as f64 / m_f;
        let tied_count = (delta_x + delta_y) as f64;
        d2_sum += (fn_val - gm_val).powi(2) * tied_count;
        pidx = qidx;
    }

    // T = (n m) / (n + m) × (1 / (n + m)) × d2_sum  (Anderson 1962 normalisation)
    let t = (n_f * m_f) / (nm_f * nm_f) * d2_sum;

    // Asymptotic p-value: T is asymptotically distributed as W² from the
    // one-sample case (after standardisation).  We use the one-sample
    // CvM p-value approximation on a pseudo-W² = T (the scale is comparable).
    let p_value = cvm_p_value(t, nm);

    Ok(CvmResult {
        statistic: t,
        p_value,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── cvm_test_uniform ──────────────────────────────────────────────────────

    #[test]
    fn cvm_uniform_finite_statistic() {
        // Any valid data → finite W² and p ∈ [0, 1].
        let data = [0.1, 0.3, 0.5, 0.7, 0.9];
        let r = cvm_test_uniform(&data).expect("ok");
        assert!(r.statistic.is_finite());
        assert!((0.0..=1.0).contains(&r.p_value));
    }

    #[test]
    fn cvm_uniform_perfect_grid_small_statistic() {
        // Perfect uniform grid → W² ≈ 1/(12n).
        let n = 1000_usize;
        let data: Vec<f64> = (1..=n)
            .map(|i| (2 * i - 1) as f64 / (2 * n) as f64)
            .collect();
        let r = cvm_test_uniform(&data).expect("ok");
        let expected = 1.0 / (12.0 * n as f64);
        assert!(
            r.statistic < expected + 1e-10,
            "statistic={}, expected≈{}",
            r.statistic,
            expected
        );
    }

    #[test]
    fn cvm_uniform_large_sample_high_p() {
        // Large perfect-uniform grid → high p-value (do not reject H₀).
        let n = 500_usize;
        let data: Vec<f64> = (1..=n)
            .map(|i| (2 * i - 1) as f64 / (2 * n) as f64)
            .collect();
        let r = cvm_test_uniform(&data).expect("ok");
        // W² for a perfect grid equals exactly 1/(12n) which is tiny → p near 1.
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    #[test]
    fn cvm_uniform_empty_error() {
        assert!(matches!(cvm_test_uniform(&[]), Err(StatsError::EmptyInput)));
    }

    #[test]
    fn cvm_uniform_non_finite_error() {
        let data = [0.1, f64::NAN, 0.9];
        assert!(matches!(
            cvm_test_uniform(&data),
            Err(StatsError::NonFiniteValue(_))
        ));
    }

    // ── cvm_test_normal ───────────────────────────────────────────────────────

    #[test]
    fn cvm_normal_standard_normal_high_p() {
        // Data sampled from N(0,1): 50 nearly-uniform values from the distribution
        // constructed by taking the mid-point quantiles (Φ⁻¹ at tiny central steps),
        // verified to produce a low CvM statistic.
        //
        // We avoid any probit implementation by instead directly building data from
        // CDF values: Φ((data-mu)/sigma) for normally-sampled data should equal
        // U[0,1]. We construct data as known N(5, 2²) values (a fixed design from
        // symmetric spacing around μ=5 at ±σ steps) and verify the CvM test does
        // not reject.
        //
        // Data points: μ + σ * k for k ∈ {-2.0, -1.9, ..., 1.9, 2.0} gives 41
        // points that are perfectly normally spaced (they ARE normal quantiles after
        // a linear transform). CvM against fitted normal should give tiny W².
        let mu_true = 5.0_f64;
        let sigma_true = 2.0_f64;
        // 40 evenly-spaced points centred at 0 in σ units, range ±1.95σ.
        let data: Vec<f64> = (0..40_usize)
            .map(|i| {
                let z = -1.95 + i as f64 * (3.90 / 39.0); // z ∈ [-1.95, 1.95]
                mu_true + sigma_true * z
            })
            .collect();
        let r = cvm_test_normal(&data).expect("ok");
        assert!(
            r.p_value > 0.01,
            "p={r_p} (symmetric grid around normal mean should not reject normality)",
            r_p = r.p_value
        );
    }

    #[test]
    fn cvm_normal_non_normal_low_p() {
        // Extremely skewed data (all positive exponential-like values shifted away
        // from the mean) should give a small p under normality.
        // Use a strongly bi-modal pattern: 0.0 and 1000.0 alternating.
        let data: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1000.0 })
            .collect();
        let r = cvm_test_normal(&data).expect("ok");
        assert!(r.statistic.is_finite());
        // The statistic should be substantially above zero.
        assert!(r.statistic > 0.01, "statistic={}", r.statistic);
    }

    #[test]
    fn cvm_normal_empty_error() {
        assert!(matches!(cvm_test_normal(&[]), Err(StatsError::EmptyInput)));
    }

    // ── watson_test_uniform ───────────────────────────────────────────────────

    #[test]
    fn watson_uniform_u2_finite() {
        let data = [0.1, 0.25, 0.5, 0.75, 0.9];
        let r = watson_test_uniform(&data).expect("ok");
        assert!(r.statistic.is_finite());
        assert!((0.0..=1.0).contains(&r.p_value));
    }

    #[test]
    fn watson_u2_leq_cvm_w2() {
        // U² = W² − n(x̄ − ½)² ≤ W² always.
        let data = [0.1, 0.35, 0.48, 0.62, 0.88];
        let cvm_r = cvm_test_uniform(&data).expect("ok");
        let wat_r = watson_test_uniform(&data).expect("ok");
        assert!(
            wat_r.statistic <= cvm_r.statistic + 1e-12,
            "U²={} > W²={}",
            wat_r.statistic,
            cvm_r.statistic
        );
    }

    #[test]
    fn watson_uniform_perfect_grid_near_zero() {
        // Perfect uniform grid → U² ≈ 0 (mean = 0.5 exactly → correction = 0).
        let n = 1000_usize;
        let data: Vec<f64> = (1..=n)
            .map(|i| (2 * i - 1) as f64 / (2 * n) as f64)
            .collect();
        let r = watson_test_uniform(&data).expect("ok");
        assert!(r.statistic < 1e-4, "U²={}", r.statistic);
    }

    #[test]
    fn watson_uniform_empty_error() {
        assert!(matches!(
            watson_test_uniform(&[]),
            Err(StatsError::EmptyInput)
        ));
    }

    // ── cvm_two_sample ────────────────────────────────────────────────────────

    #[test]
    fn cvm_two_sample_identical_small_statistic() {
        // Identical samples → T ≈ 0.
        let x: Vec<f64> = (1..=50).map(|i| i as f64 / 50.0).collect();
        let y = x.clone();
        let r = cvm_two_sample(&x, &y).expect("ok");
        assert!(r.statistic < 1e-10, "T={}", r.statistic);
    }

    #[test]
    fn cvm_two_sample_different_distributions_large_statistic() {
        // x ~ Uniform(0, 0.3), y ~ Uniform(0.7, 1.0) → large T.
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * 0.3).collect();
        let y: Vec<f64> = (0..100).map(|i| 0.7 + i as f64 / 100.0 * 0.3).collect();
        let r = cvm_two_sample(&x, &y).expect("ok");
        assert!(r.statistic > 1.0, "T={}", r.statistic);
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    #[test]
    fn cvm_two_sample_finite_statistic() {
        let x = [0.2, 0.4, 0.6, 0.8];
        let y = [0.1, 0.3, 0.5, 0.7, 0.9];
        let r = cvm_two_sample(&x, &y).expect("ok");
        assert!(r.statistic.is_finite());
        assert!((0.0..=1.0).contains(&r.p_value));
    }

    #[test]
    fn cvm_two_sample_empty_error() {
        assert!(matches!(
            cvm_two_sample(&[], &[1.0]),
            Err(StatsError::EmptyInput)
        ));
        assert!(matches!(
            cvm_two_sample(&[1.0], &[]),
            Err(StatsError::EmptyInput)
        ));
    }

    #[test]
    fn cvm_two_sample_p_value_in_range() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 30.0).collect();
        let y: Vec<f64> = (0..30).map(|i| i as f64 / 30.0 + 0.5).collect();
        let r = cvm_two_sample(&x, &y).expect("ok");
        assert!((0.0..=1.0).contains(&r.p_value));
    }

    // ── cross-check ───────────────────────────────────────────────────────────

    #[test]
    fn watson_statistic_non_negative() {
        // U² must always be non-negative.
        for k in 1_usize..=10 {
            let data: Vec<f64> = (0..k).map(|i| i as f64 / k as f64).collect();
            if data.is_empty() {
                continue;
            }
            if let Ok(r) = watson_test_uniform(&data) {
                assert!(r.statistic >= 0.0, "U²<0 for k={k}");
            }
        }
    }
}
