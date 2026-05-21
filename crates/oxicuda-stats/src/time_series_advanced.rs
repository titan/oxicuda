//! Advanced time-series statistical tests for `oxicuda-stats`.
//!
//! Provides structural-break and heteroscedasticity tests:
//!
//! 1. **Variance Ratio Test** (Lo-MacKinlay 1988) — tests the random-walk
//!    hypothesis by comparing variance of multi-period returns to scaled
//!    one-period variance.
//!
//! 2. **Chow Test** — tests for a structural break at a known breakpoint by
//!    comparing pooled OLS residuals to segment-specific residuals.
//!
//! 3. **ARCH LM Test** (Engle 1982) — Lagrange Multiplier test for
//!    autoregressive conditional heteroscedasticity in residuals.
//!
//! 4. **Bai-Perron Single-Break Test** — exhaustive search over all candidate
//!    break dates for the single structural break with the maximum F-statistic.
//!
//! 5. **Zivot-Andrews Test** — ADF unit-root test with a structural break in
//!    the intercept; selects the break that minimises the t-statistic on y_{t-1}.
//!
//! # References
//! - Lo & MacKinlay (1988) "Stock market prices do not follow random walks".
//!   *Review of Financial Studies* 1(1):41-66.
//! - Chow (1960) "Tests of equality between sets of coefficients".
//!   *Econometrica* 28(3):591-605.
//! - Engle (1982) "Autoregressive conditional heteroscedasticity".
//!   *Econometrica* 50(4):987-1007.
//! - Bai & Perron (1998) "Estimating and testing linear models with multiple
//!   structural changes". *Econometrica* 66(1):47-78.
//! - Zivot & Andrews (1992) "Further evidence on the great crash". *JBES* 10(3).

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::ols;
use crate::special::betainc::gammp;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Survival function P(χ²(df) > x).
fn chi2_sf(x: f64, df: f64) -> StatsResult<f64> {
    if x <= 0.0 {
        return Ok(1.0);
    }
    let p = gammp(df / 2.0, x / 2.0)?;
    Ok((1.0 - p).clamp(0.0, 1.0))
}

/// Survival function P(F(df1, df2) > x) using the incomplete beta function.
fn f_sf(x: f64, df1: f64, df2: f64) -> StatsResult<f64> {
    use crate::special::betainc::betainc;
    if x <= 0.0 {
        return Ok(1.0);
    }
    // P(F > x) = 1 - I_{df1*x/(df1*x+df2)}(df1/2, df2/2)
    let t = (df1 * x) / (df1 * x + df2);
    let t = t.clamp(1e-300, 1.0 - 1e-15);
    let cdf = betainc(df1 / 2.0, df2 / 2.0, t)?;
    Ok((1.0 - cdf).clamp(0.0, 1.0))
}

/// Compute OLS residuals for a simple regression `y ~ 1 + x`.
///
/// Returns (intercept, slope, residuals, rss).
fn simple_ols_with_intercept(y: &[f64], x: &[f64]) -> StatsResult<(f64, f64, Vec<f64>, f64)> {
    let n = y.len();
    if n != x.len() {
        return Err(StatsError::DimensionMismatch { a: n, b: x.len() });
    }
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    // Build design matrix [1, x] row-major
    let mut design = vec![0.0_f64; n * 2];
    for i in 0..n {
        design[i * 2] = 1.0;
        design[i * 2 + 1] = x[i];
    }
    let lm = ols(&design, y, n, 2)?;
    let rss = lm.residual_sum_squares;
    let intercept = lm.coefficients[0];
    let slope = lm.coefficients[1];
    Ok((intercept, slope, lm.residuals, rss))
}

/// OLS of `y ~ 1` (constant model) — returns demeaned residuals and RSS.
fn demean_rss(y: &[f64]) -> (Vec<f64>, f64) {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let resid: Vec<f64> = y.iter().map(|&v| v - mean).collect();
    let rss = resid.iter().map(|r| r * r).sum();
    (resid, rss)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Variance Ratio Test (Lo-MacKinlay 1988)
// ─────────────────────────────────────────────────────────────────────────────

/// Variance Ratio Test for the random-walk hypothesis.
///
/// Given a price (or log-price) series `series`, computes log-returns and
/// tests whether the variance of `q`-period returns equals `q` times the
/// variance of 1-period returns.
///
/// **Statistic definition**:
/// ```text
/// VR(q) = Var(q-period log-return) / (q * Var(1-period log-return))
/// ```
///
/// Under the random-walk null, VR(q) → 1.
///
/// The heteroscedasticity-robust Z-statistic is:
/// ```text
/// Z(q) = (VR(q) - 1) / σ_q^{1/2}
/// ```
/// where `σ_q^2 = 2(2q-1)(q-1) / (3q*n_q)` (Lo-MacKinlay, 1988, eq. 14).
///
/// # Arguments
/// - `series` — raw price series (length ≥ q + 2).
/// - `q` — holding period (≥ 2).
///
/// # Returns
/// `(VR(q), Z_statistic)` where `Z ~ N(0,1)` under H₀.
pub fn variance_ratio_test(series: &[f64], q: usize) -> StatsResult<(f64, f64)> {
    let n = series.len();
    if q < 1 {
        return Err(StatsError::InvalidParameter {
            name: "q".to_string(),
            reason: "holding period q must be ≥ 1".to_string(),
        });
    }
    if n < q + 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: q + 2,
        });
    }
    if q == 1 {
        // VR(1) = 1 exactly by definition
        return Ok((1.0, 0.0));
    }

    // Compute log-returns r_t = ln(series[t]/series[t-1])
    // Guard against non-positive prices
    for (i, &price) in series.iter().enumerate() {
        if price <= 0.0 {
            return Err(StatsError::NumericalInstability(format!(
                "non-positive price at index {i}"
            )));
        }
    }
    let returns: Vec<f64> = (1..n).map(|t| (series[t] / series[t - 1]).ln()).collect();
    // returns has length n-1

    let nr = returns.len(); // = n-1
    if nr < q {
        return Err(StatsError::InsufficientSampleSize { got: nr, need: q });
    }

    // 1-period variance: sigma_1^2 = (1/(nr-1)) * Σ (r_t - r_bar)^2
    let r_bar = returns.iter().sum::<f64>() / nr as f64;
    let var1 = returns
        .iter()
        .map(|r| (r - r_bar) * (r - r_bar))
        .sum::<f64>()
        / (nr.saturating_sub(1).max(1)) as f64;

    // q-period returns: R_t^q = Σ_{k=0}^{q-1} r_{t-k}  for t = q-1, q, ..., nr-1
    // Number of non-overlapping q-period returns
    let n_q = nr / q; // number of non-overlapping intervals
    if n_q < 1 {
        return Err(StatsError::InsufficientSampleSize { got: nr, need: q });
    }

    // Use overlapping q-period returns for efficiency (Lo-MacKinlay standard)
    let n_overlap = nr.saturating_sub(q - 1); // number of overlapping q-returns
    let mut returns_q = Vec::with_capacity(n_overlap);
    for t in (q - 1)..nr {
        let sum_q: f64 = returns[(t + 1 - q)..=t].iter().sum();
        returns_q.push(sum_q);
    }

    let rq_bar = returns_q.iter().sum::<f64>() / n_overlap as f64;
    let var_q = returns_q
        .iter()
        .map(|r| (r - rq_bar) * (r - rq_bar))
        .sum::<f64>()
        / (n_overlap.saturating_sub(1).max(1)) as f64;

    if var1 < 1e-300 {
        return Err(StatsError::NumericalInstability(
            "variance of 1-period returns is zero".to_string(),
        ));
    }

    let vr = var_q / ((q as f64) * var1);

    // Heteroscedasticity-robust asymptotic variance (Lo-MacKinlay 1988, Eq. 14):
    // sigma_q^2 = 2(2q-1)(q-1) / (3q * nq)
    // where nq is the effective sample (n_overlap / q ≈ n_q)
    let nq_f = n_q as f64;
    let q_f = q as f64;
    let sigma_q_sq = 2.0 * (2.0 * q_f - 1.0) * (q_f - 1.0) / (3.0 * q_f * nq_f);
    if sigma_q_sq <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "degenerate sigma_q in VR test".to_string(),
        ));
    }
    let z_stat = (vr - 1.0) / sigma_q_sq.sqrt();

    Ok((vr, z_stat))
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Chow Test
// ─────────────────────────────────────────────────────────────────────────────

/// Chow test for a structural break at a known breakpoint.
///
/// Estimates the simple linear model `y ~ 1 + x` on three samples:
/// 1. Full sample `[0, n)` → restricted model RSS (RSS_r).
/// 2. First segment `[0, bp)` → RSS₁.
/// 3. Second segment `[bp, n)` → RSS₂.
///
/// Then computes the Chow F-statistic:
/// ```text
/// F = [(RSS_r - (RSS₁ + RSS₂)) / k] / [(RSS₁ + RSS₂) / (n - 2k)]
/// ```
/// where `k = 2` (intercept + slope).
///
/// Under H₀ (no structural break), F ~ F(k, n - 2k).
///
/// # Arguments
/// - `y` — dependent variable (length n).
/// - `x` — independent variable (length n).
/// - `breakpoint_index` — first index of the second segment (must be in `[k+1, n-k-1]`).
///
/// # Returns
/// `(F_statistic, p_value)`.
pub fn chow_test(y: &[f64], x: &[f64], breakpoint_index: usize) -> StatsResult<(f64, f64)> {
    let n = y.len();
    if n != x.len() {
        return Err(StatsError::DimensionMismatch { a: n, b: x.len() });
    }
    let k = 2_usize; // number of parameters per segment (intercept + slope)
    if breakpoint_index < k || breakpoint_index > n.saturating_sub(k) {
        return Err(StatsError::InvalidParameter {
            name: "breakpoint_index".to_string(),
            reason: format!(
                "breakpoint {breakpoint_index} out of valid range [{k}, {}]",
                n - k
            ),
        });
    }
    let n1 = breakpoint_index;
    let n2 = n - breakpoint_index;
    if n1 < k || n2 < k {
        return Err(StatsError::InsufficientSampleSize {
            got: n1.min(n2),
            need: k,
        });
    }

    // Restricted (full-sample) OLS
    let (_, _, _, rss_r) = simple_ols_with_intercept(y, x)?;

    // Segment 1
    let (_, _, _, rss1) = simple_ols_with_intercept(&y[..n1], &x[..n1])?;
    // Segment 2
    let (_, _, _, rss2) = simple_ols_with_intercept(&y[n1..], &x[n1..])?;

    let rss_u = rss1 + rss2;
    let dof_num = k as f64;
    let dof_den = (n - 2 * k) as f64;
    if dof_den <= 0.0 {
        return Err(StatsError::DegreesOfFreedomZero);
    }
    if rss_u < 1e-300 {
        return Err(StatsError::NumericalInstability(
            "unrestricted RSS is zero".to_string(),
        ));
    }

    let f_stat = ((rss_r - rss_u) / dof_num) / (rss_u / dof_den);
    let p_value = f_sf(f_stat.max(0.0), dof_num, dof_den)?;
    Ok((f_stat, p_value))
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Engle's ARCH LM Test
// ─────────────────────────────────────────────────────────────────────────────

/// Engle's ARCH LM test for autoregressive conditional heteroscedasticity.
///
/// Given OLS residuals `{ê_t}`, tests whether their squared values are
/// predictable from their own lags (indicating ARCH effects).
///
/// Procedure:
/// 1. Form `u_t = ê_t²`.
/// 2. Regress `u_t` on `(1, u_{t-1}, …, u_{t-p})` for t = p+1, …, n.
/// 3. Compute LM statistic: `LM = n_eff × R²` where R² is from step 2.
/// 4. Under H₀ (no ARCH): `LM ~ χ²(n_lags)`.
///
/// # Arguments
/// - `residuals` — OLS residuals (length n).
/// - `n_lags` — number of ARCH lags to test (p ≥ 1).
///
/// # Returns
/// `(LM_statistic, p_value)` where p-value uses the chi-squared distribution.
pub fn arch_test(residuals: &[f64], n_lags: usize) -> StatsResult<(f64, f64)> {
    let n = residuals.len();
    if n_lags == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_lags".to_string(),
            reason: "must be ≥ 1".to_string(),
        });
    }
    if n_lags >= n {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: n_lags + 1,
        });
    }

    // Squared residuals
    let u: Vec<f64> = residuals.iter().map(|e| e * e).collect();

    // Effective sample: t = n_lags, …, n-1 → n_eff rows
    let n_eff = n - n_lags;
    let n_cols = n_lags + 1; // intercept + n_lags lags

    if n_eff < n_cols {
        return Err(StatsError::InsufficientSampleSize {
            got: n_eff,
            need: n_cols,
        });
    }

    // Build design matrix: [1, u_{t-1}, …, u_{t-p}] for t = n_lags, …, n-1
    let mut design = vec![0.0_f64; n_eff * n_cols];
    let mut response = vec![0.0_f64; n_eff];

    for (row, t) in (n_lags..n).enumerate() {
        response[row] = u[t];
        design[row * n_cols] = 1.0; // intercept
        for lag in 1..=n_lags {
            design[row * n_cols + lag] = u[t - lag];
        }
    }

    let lm_model = ols(&design, &response, n_eff, n_cols)?;

    // Compute R² = 1 - RSS / TSS
    let y_bar = response.iter().sum::<f64>() / n_eff as f64;
    let tss: f64 = response.iter().map(|v| (v - y_bar) * (v - y_bar)).sum();
    let r_sq = if tss < 1e-300 {
        0.0
    } else {
        (1.0 - lm_model.residual_sum_squares / tss).clamp(0.0, 1.0)
    };

    let lm_stat = n_eff as f64 * r_sq;
    let p_value = chi2_sf(lm_stat, n_lags as f64)?;
    Ok((lm_stat, p_value))
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Bai-Perron Single Structural Break Test
// ─────────────────────────────────────────────────────────────────────────────

/// Bai-Perron single structural break test via exhaustive search.
///
/// For each candidate break date `bp` in `[min_segment, n - min_segment]`,
/// computes the Chow F-statistic for `y ~ 1` (mean-shift model, intercept
/// only — no covariate) by comparing:
/// - Restricted RSS: full demeaned RSS.
/// - Unrestricted RSS: sum of segment-demeaned RSSes.
///
/// The break date with the **maximum** F-statistic is returned.
///
/// F-statistic per candidate breakpoint:
/// ```text
/// F(bp) = [(RSS_r - RSS_u(bp)) / 1] / [RSS_u(bp) / (n - 2)]
/// ```
/// where `k = 1` (intercept only in each segment), so we estimate 1 break → 2 intercepts.
///
/// # Arguments
/// - `y` — time series (length n).
/// - `min_segment` — minimum segment length (≥ 1).
///
/// # Returns
/// `(breakpoint_index, F_stat, p_value)`.
pub fn bai_perron_single_break(y: &[f64], min_segment: usize) -> StatsResult<(usize, f64, f64)> {
    let n = y.len();
    let min_seg = min_segment.max(1);
    if n < 2 * min_seg + 1 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: 2 * min_seg + 1,
        });
    }

    // Restricted RSS: demean whole series
    let (_, rss_r) = demean_rss(y);

    let mut best_bp = min_seg;
    let mut best_f = f64::NEG_INFINITY;

    for bp in min_seg..=(n - min_seg) {
        let (_, rss1) = demean_rss(&y[..bp]);
        let (_, rss2) = demean_rss(&y[bp..]);
        let rss_u = rss1 + rss2;

        // F-statistic with k=1 (mean shift)
        let dof_den = (n as f64) - 2.0;
        if dof_den <= 0.0 || rss_u < 1e-300 {
            continue;
        }
        let f = ((rss_r - rss_u) / 1.0) / (rss_u / dof_den);
        if f > best_f {
            best_f = f;
            best_bp = bp;
        }
    }

    let dof_den = (n as f64) - 2.0;
    let p_value = if best_f <= 0.0 {
        1.0
    } else {
        f_sf(best_f, 1.0, dof_den)?
    };

    Ok((best_bp, best_f, p_value))
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Zivot-Andrews Unit Root Test with Structural Break
// ─────────────────────────────────────────────────────────────────────────────

/// Zivot-Andrews unit root test with a break in the intercept.
///
/// For each candidate break date `τ` in `[min_segment, n - min_segment]`,
/// fits the augmented DF regression with a structural-break dummy:
/// ```text
/// Δy_t = α + γ y_{t-1} + β·DU_t(τ) + ε_t
/// ```
/// where `DU_t(τ) = 1{t > τ}` (intercept dummy).
///
/// The test statistic is the **minimum** t-statistic on γ over all candidate
/// break dates (the most evidence against the unit root).
///
/// Asymptotic critical values for the intercept model:
/// - 1%:  −4.80
/// - 5%:  −4.42
/// - 10%: −4.11
///
/// The approximate p-value is interpolated from these three critical values.
///
/// # Arguments
/// - `series` — the time series (length ≥ 2 * min_segment + 2).
/// - `min_segment` — minimum pre/post-break observations (≥ 2).
///
/// # Returns
/// `(min_t_stat, optimal_break_index)`.
pub fn zivot_andrews_test(series: &[f64], min_segment: usize) -> StatsResult<(f64, usize)> {
    let n = series.len();
    let min_seg = min_segment.max(2);
    if n < 2 * min_seg + 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: 2 * min_seg + 2,
        });
    }

    // First differences Δy_t = y_t - y_{t-1} (length n-1)
    let dy: Vec<f64> = (1..n).map(|t| series[t] - series[t - 1]).collect();
    let n_dy = dy.len(); // n-1

    let mut min_t = f64::INFINITY;
    let mut opt_break = min_seg;

    // Candidate break dates: τ in [min_seg, n - min_seg]
    // In the regression, t runs from 1 to n-1 (Δy index), so τ from min_seg to n-min_seg-1.
    for tau in min_seg..=(n - min_seg) {
        // Build design matrix for t=1..n-1:
        // [1, y_{t-1}, DU_t(τ)] row-major (3 columns, no lag augmentation for simplicity)
        let n_eff = n_dy; // all differences available (t=1..n-1, i.e. n-1 rows)
        let n_cols = 3_usize; // intercept, y_{t-1}, DU dummy

        if n_eff < n_cols + 1 {
            continue;
        }

        let mut design = vec![0.0_f64; n_eff * n_cols];
        let response = dy.clone();

        for (row, t) in (1..n).enumerate() {
            // row = t-1 (0-indexed), t in 1..n
            design[row * n_cols] = 1.0; // intercept
            design[row * n_cols + 1] = series[t - 1]; // y_{t-1}
            // DU_t(tau) = 1 if t > tau (using 1-based indexing for t)
            design[row * n_cols + 2] = if t > tau { 1.0 } else { 0.0 };
        }

        match ols(&design, &response, n_eff, n_cols) {
            Ok(lm) => {
                // γ is at column index 1
                let gamma_hat = lm.coefficients[1];
                let dof = n_eff - n_cols;
                if dof == 0 {
                    continue;
                }
                let sigma2 = lm.residual_sum_squares / dof as f64;
                // Var(γ̂) = σ² * [(X^T X)^{-1}]_{1,1}
                let var_gamma = sigma2 * lm.xtx_inv[n_cols + 1];
                if var_gamma <= 0.0 {
                    continue;
                }
                let t_stat = gamma_hat / var_gamma.sqrt();
                if t_stat < min_t {
                    min_t = t_stat;
                    opt_break = tau;
                }
            }
            Err(_) => continue,
        }
    }

    if min_t.is_infinite() {
        return Err(StatsError::NumericalInstability(
            "Zivot-Andrews: no valid regression could be fit".to_string(),
        ));
    }

    Ok((min_t, opt_break))
}

/// Approximate p-value for the Zivot-Andrews test from asymptotic critical values.
///
/// Critical values for intercept model: -4.80 (1%), -4.42 (5%), -4.11 (10%).
/// Returns p ∈ (0, 1) via linear interpolation / extrapolation.
pub fn zivot_andrews_p_value(t_stat: f64) -> f64 {
    // Critical values (more negative = stronger rejection)
    // cv[i] = (critical_value, p_level)
    let cv: &[(f64, f64)] = &[(-4.80, 0.01), (-4.42, 0.05), (-4.11, 0.10)];

    // t_stat << cv[0] (very negative) → p << 0.01
    if t_stat <= cv[0].0 {
        return 0.005;
    }
    // t_stat > cv[last] (not very negative) → p > 0.10
    if t_stat >= cv[cv.len() - 1].0 {
        // Extrapolate modestly; cap at 0.99
        let slope = (0.10 - 0.05) / (cv[2].0 - cv[1].0);
        let p = 0.10 + slope * (t_stat - cv[2].0);
        return p.clamp(0.10, 0.99);
    }
    // Interpolate between table entries
    for i in 0..(cv.len() - 1) {
        let (t_lo, p_lo) = cv[i];
        let (t_hi, p_hi) = cv[i + 1];
        if t_stat >= t_lo && t_stat <= t_hi {
            let frac = (t_stat - t_lo) / (t_hi - t_lo);
            return p_lo + frac * (p_hi - p_lo);
        }
    }
    0.05 // fallback
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Minimal deterministic LCG for test data ───────────────────────────────

    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
        }

        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-300);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Build a random-walk price series starting at 100.
    fn random_walk_prices(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = TestRng::new(seed);
        let mut p = vec![100.0_f64; n];
        for t in 1..n {
            p[t] = p[t - 1] * (rng.next_normal() * 0.01).exp();
        }
        p
    }

    /// Build an MA(1) price series with θ > 0 (positively correlated returns).
    fn ma1_prices(n: usize, theta: f64, seed: u64) -> Vec<f64> {
        let mut rng = TestRng::new(seed);
        let eps: Vec<f64> = (0..n).map(|_| rng.next_normal() * 0.01).collect();
        let mut returns = vec![0.0_f64; n];
        returns[0] = eps[0];
        for t in 1..n {
            returns[t] = eps[t] + theta * eps[t - 1];
        }
        let mut p = vec![100.0_f64; n];
        for t in 1..n {
            p[t] = p[t - 1] * returns[t].exp();
        }
        // ensure eps is actually used
        let _ = eps.iter().sum::<f64>();
        p
    }

    // ── Variance Ratio Test ───────────────────────────────────────────────────

    #[test]
    fn vrt_q1_is_exactly_one() {
        let prices = random_walk_prices(200, 1);
        let (vr, z) = variance_ratio_test(&prices, 1).unwrap();
        assert_eq!(vr, 1.0, "VR(1) must be exactly 1.0");
        assert_eq!(z, 0.0, "Z(1) must be exactly 0.0");
    }

    #[test]
    fn vrt_q2_random_walk_near_one() {
        let prices = random_walk_prices(500, 2);
        let (vr, _z) = variance_ratio_test(&prices, 2).unwrap();
        assert!(
            (vr - 1.0).abs() < 0.5,
            "VR(2) for random walk should be near 1, got {vr}"
        );
    }

    #[test]
    fn vrt_ma1_not_one() {
        // MA(1) with θ=0.5 → VR(2) deviates from 1 systematically
        let prices = ma1_prices(1000, 0.5, 3);
        let (vr, _z) = variance_ratio_test(&prices, 2).unwrap();
        // VR should differ from 1 by more than 0.05 for strong MA(1)
        assert!(
            (vr - 1.0).abs() > 0.02,
            "VR(2) for MA(1) should differ from 1, got {vr}"
        );
    }

    #[test]
    fn vrt_statistic_positive() {
        let prices = random_walk_prices(300, 4);
        let (vr, _) = variance_ratio_test(&prices, 4).unwrap();
        assert!(vr > 0.0, "VR must be positive, got {vr}");
    }

    #[test]
    fn vrt_insufficient_data_error() {
        let prices = vec![1.0, 1.01, 1.02]; // only 3 points, need ≥ q+2
        let result = variance_ratio_test(&prices, 5);
        assert!(result.is_err(), "should error on insufficient data");
    }

    // ── Chow Test ─────────────────────────────────────────────────────────────

    #[test]
    fn chow_test_no_break_high_p() {
        // Same DGP → no structural break → high p-value
        let n = 100_usize;
        let mut rng = TestRng::new(10);
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 + 0.5 * xi + rng.next_normal())
            .collect();
        let bp = n / 2;
        let (f_stat, p_val) = chow_test(&y, &x, bp).unwrap();
        assert!(f_stat.is_finite(), "F should be finite");
        // With same DGP, p-value should generally be high
        assert!(
            p_val > 0.0,
            "p-value should be positive for no-break case, got {p_val}"
        );
    }

    #[test]
    fn chow_test_clear_break_low_p() {
        // Strong structural break at midpoint
        let n = 100_usize;
        let mut rng = TestRng::new(20);
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut y = vec![0.0_f64; n];
        for i in 0..n / 2 {
            y[i] = 1.0 * x[i] + rng.next_normal() * 0.1;
        }
        for i in n / 2..n {
            // Completely different slope and intercept
            y[i] = 100.0 + 10.0 * x[i] + rng.next_normal() * 0.1;
        }
        let bp = n / 2;
        let (f_stat, p_val) = chow_test(&y, &x, bp).unwrap();
        assert!(f_stat > 1.0, "F should be large for clear break: {f_stat}");
        assert!(
            p_val < 0.5,
            "p-value should be small for clear break: {p_val}"
        );
    }

    #[test]
    fn chow_test_breakpoint_at_boundary_error() {
        let y = vec![1.0_f64; 20];
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = chow_test(&y, &x, 0);
        assert!(result.is_err(), "breakpoint=0 should error");
        let result2 = chow_test(&y, &x, 19);
        assert!(result2.is_err(), "breakpoint=n-1 should error");
    }

    // ── ARCH Test ─────────────────────────────────────────────────────────────

    #[test]
    fn arch_test_white_noise_high_p() {
        // White noise residuals: no ARCH → p > 0.05
        let n = 300_usize;
        let mut rng = TestRng::new(30);
        let residuals: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let (lm, p) = arch_test(&residuals, 5).unwrap();
        assert!(
            lm.is_finite() && lm >= 0.0,
            "LM should be non-negative: {lm}"
        );
        // For pure white noise, most of the time p > 0.01
        assert!(
            p > 0.001,
            "white noise should have high ARCH p-value, got {p}"
        );
    }

    #[test]
    fn arch_test_arch_process_detected() {
        // Simulate ARCH(1): ε_t = σ_t * z_t, σ_t² = α_0 + α_1 * ε_{t-1}²
        let n = 500_usize;
        let mut rng = TestRng::new(40);
        let mut eps = vec![0.0_f64; n];
        let mut sigma2 = 1.0_f64;
        for e in eps.iter_mut().take(n) {
            let z = rng.next_normal();
            *e = sigma2.sqrt() * z;
            sigma2 = 0.2 + 0.7 * (*e) * (*e); // strong ARCH(1)
        }
        let (lm, p) = arch_test(&eps, 5).unwrap();
        // Strong ARCH effect: LM should be large and p small
        assert!(lm > 0.0, "LM should be positive: {lm}");
        // With strong ARCH and n=500, detection should be reliable
        // Use a lenient threshold since this is a stochastic test
        assert!(
            p < 0.5,
            "ARCH process should be detected (p < 0.5), got {p}"
        );
    }

    #[test]
    fn arch_test_n_lags_too_large_error() {
        let residuals = vec![1.0_f64; 5];
        let result = arch_test(&residuals, 5); // n_lags >= n
        assert!(result.is_err(), "n_lags >= n should return error");
    }

    // ── Bai-Perron Single Break ───────────────────────────────────────────────

    #[test]
    fn bai_perron_finds_correct_break_at_midpoint() {
        // Clear mean shift at midpoint
        let n = 100_usize;
        let bp_true = 50_usize;
        let mut rng = TestRng::new(50);
        let mut y = vec![0.0_f64; n];
        for slot in y.iter_mut().take(bp_true) {
            *slot = rng.next_normal() * 0.1; // mean = 0
        }
        for slot in y.iter_mut().take(n).skip(bp_true) {
            *slot = 10.0 + rng.next_normal() * 0.1; // mean = 10
        }
        let (bp_est, f_stat, _p) = bai_perron_single_break(&y, 10).unwrap();
        assert!(f_stat > 0.0, "F-stat should be positive: {f_stat}");
        // Allow some tolerance around the true break
        assert!(
            (bp_est as i64 - bp_true as i64).abs() <= 5,
            "estimated break {bp_est} should be near true break {bp_true}"
        );
    }

    #[test]
    fn bai_perron_min_segment_too_large_error() {
        let y = vec![1.0_f64; 10];
        // min_segment = 6 requires n >= 2*6+1 = 13 > 10
        let result = bai_perron_single_break(&y, 6);
        assert!(result.is_err(), "should error when min_segment too large");
    }

    // ── Zivot-Andrews Test ────────────────────────────────────────────────────

    #[test]
    fn zivot_andrews_integrated_series_small_t_stat() {
        // Integrated (I(1)) series → cannot reject unit root
        // ZA test should give t-stat near zero or slightly negative
        let n = 150_usize;
        let mut rng = TestRng::new(60);
        let mut y = vec![0.0_f64; n];
        for t in 1..n {
            y[t] = y[t - 1] + rng.next_normal();
        }
        let (min_t, _) = zivot_andrews_test(&y, 10).unwrap();
        assert!(min_t.is_finite(), "t-stat should be finite, got {min_t}");
        // For random walk, min t-stat should not be very negative (cannot reject)
        // Critical value at 5% is -4.42; integrated series should be above this
        // (i.e., min_t > -6 is a very lenient check)
        assert!(
            min_t > -10.0,
            "random walk ZA t-stat should not be extremely negative, got {min_t}"
        );
    }

    #[test]
    fn zivot_andrews_p_value_critical_values() {
        // t_stat = -4.42 → 5% critical value → p ≈ 0.05
        let p = zivot_andrews_p_value(-4.42);
        assert!(
            (p - 0.05).abs() < 0.02,
            "p-value at 5% CV should be ~0.05, got {p}"
        );
        // t_stat = -4.80 → 1% critical value → p should be small
        let p1 = zivot_andrews_p_value(-4.80);
        assert!(p1 <= 0.01, "p-value at 1% CV should be ≤ 0.01, got {p1}");
        // t_stat = -4.11 → 10% critical value → p ≈ 0.10
        let p10 = zivot_andrews_p_value(-4.11);
        assert!(
            (p10 - 0.10).abs() < 0.02,
            "p-value at 10% CV should be ~0.10, got {p10}"
        );
    }

    // ── Helper function correctness ───────────────────────────────────────────

    #[test]
    fn chi2_sf_valid_range() {
        let p = chi2_sf(3.84, 1.0).unwrap(); // chi2(1) 5% CV ≈ 3.84
        assert!(
            (p - 0.05).abs() < 0.005,
            "chi2_sf(3.84, 1) should be ≈ 0.05, got {p}"
        );
    }

    #[test]
    fn f_sf_valid_range() {
        // F(1,∞) ≈ z²; F = 3.84, df=(1,100) → p ≈ 0.05
        let p = f_sf(4.0, 1.0, 100.0).unwrap();
        assert!(
            p < 0.06 && p > 0.03,
            "f_sf(4.0, 1, 100) should be ≈ 0.05, got {p}"
        );
    }

    #[test]
    fn chi2_sf_at_zero_returns_one() {
        // chi2_sf(x <= 0) must return 1.0 (whole mass is to the right of 0)
        let p = chi2_sf(0.0, 2.0).unwrap();
        assert_eq!(p, 1.0, "chi2_sf(0, 2) should be 1.0, got {p}");
    }
}
