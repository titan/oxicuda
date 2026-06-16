//! Vector Autoregression `VAR(p)` — multivariate least-squares estimation.
//!
//! # Model
//!
//! For a `k`-dimensional series `Y_t ∈ ℝ^k`,
//!
//! ```text
//! Y_t = c + A_1 Y_{t−1} + A_2 Y_{t−2} + … + A_p Y_{t−p} + ε_t,   ε_t ~ (0, Σ).
//! ```
//!
//! # Estimation (Lütkepohl 2005, §3.2)
//!
//! Stack each equation. With `T` usable observations (indices `p … N−1`), form the
//! regressor matrix `Z ∈ ℝ^{T×m}` whose `t`-th row is
//!
//! ```text
//! z_t = [1, Y_{t−1}ᵀ, Y_{t−2}ᵀ, …, Y_{t−p}ᵀ],   m = 1 + k·p,
//! ```
//!
//! and the response matrix `Yʳ ∈ ℝ^{T×k}` with row `Y_tᵀ`.  The coefficient block
//! `B = [c | A_1 | … | A_p]ᵀ ∈ ℝ^{m×k}` is the least-squares solution
//!
//! ```text
//! B = (ZᵀZ)⁻¹ ZᵀYʳ.
//! ```
//!
//! The residual covariance uses the small-sample (degrees-of-freedom corrected)
//! divisor
//!
//! ```text
//! Σ = Eᵀ E / (T − k·p − 1),   E = Yʳ − Z B.
//! ```
//!
//! # Stability
//!
//! The process is stable iff every eigenvalue of the `kp × kp` companion matrix
//!
//! ```text
//! F = | A_1  A_2  …  A_{p−1}  A_p |
//!     |  I    0   …     0      0  |
//!     |  0    I   …     0      0  |
//!     |  ⋮              ⋱      ⋮  |
//!     |  0    0   …     I      0  |
//! ```
//!
//! lies strictly inside the unit circle (spectral radius `< 1`).  The radius is
//! obtained from the dominant eigenvalue via power iteration on `F` (real part for
//! the modulus through a deflation-free complex-modulus estimate using `‖F^n v‖`).
//!
//! # Granger causality
//!
//! Variable `j` Granger-causes variable `i` if the lag coefficients of `j` in the
//! equation for `i` are not jointly zero.  We use the Wald statistic
//!
//! ```text
//! W = β̂ᵀ [R (ZᵀZ)⁻¹ Rᵀ ⊗ Σ̂_ii]⁻¹ β̂  ~  χ²(p)   under H₀,
//! ```
//!
//! specialised here to the single restricted equation (the `p` lag terms of `j`).
//!
//! # References
//! - Lütkepohl, H. (2005) *New Introduction to Multiple Time Series Analysis*.
//!   Springer. (Chapters 2-3.)

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

// ─────────────────────────────────────────────────────────────────────────────
// Result structs
// ─────────────────────────────────────────────────────────────────────────────

/// A fitted `VAR(p)` model.
#[derive(Debug, Clone)]
pub struct VarFit {
    /// Number of series (dimension `k`).
    pub k: usize,
    /// Lag order `p`.
    pub p: usize,
    /// Number of usable observations `T = N − p`.
    pub n_obs: usize,
    /// Intercept vector `c` (length `k`).
    pub intercept: Vec<f64>,
    /// Coefficient matrices `A_1, …, A_p`, each stored row-major `k × k`.
    pub coefficients: Vec<Vec<f64>>,
    /// Residual covariance `Σ` (row-major `k × k`, df-corrected).
    pub sigma: Vec<f64>,
    /// Residuals `E`, row-major `T × k` (row `t` is `ε̂_t`).
    pub residuals: Vec<f64>,
    /// `(ZᵀZ)⁻¹`, row-major `m × m` with `m = 1 + k·p` (cached for inference).
    pub xtx_inv: Vec<f64>,
    /// Full coefficient block `B`, row-major `m × k` (`[c; A_1; …; A_p]`).
    pub beta: Vec<f64>,
}

impl VarFit {
    /// Width of a regressor row, `m = 1 + k·p`.
    #[must_use]
    pub fn n_regressors(&self) -> usize {
        1 + self.k * self.p
    }

    /// Read coefficient matrix `A_lag` (`lag` is 1-based) at row `i`, column `j`,
    /// i.e. the effect of `Y_{j, t−lag}` on `Y_{i, t}`.
    #[must_use]
    pub fn coefficient(&self, lag: usize, i: usize, j: usize) -> f64 {
        self.coefficients[lag - 1][i * self.k + j]
    }
}

/// Result of a Granger-causality Wald test (`X → Y`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrangerResult {
    /// Wald statistic `W ~ χ²(p)` under H₀ ("no causality").
    pub statistic: f64,
    /// Restriction degrees of freedom (`= p`).
    pub df: usize,
    /// Two-sided p-value `P(χ²(p) > W)`.
    pub p_value: f64,
    /// `true` when H₀ is rejected at the supplied significance level.
    pub causes: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitting
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a `VAR(p)` by multivariate least squares.
///
/// `data` is row-major `n_obs × k` (row `t` is the observation `Y_tᵀ`).
///
/// # Errors
/// Returns an error for empty/degenerate input, when `p == 0`, when there are too
/// few observations to identify the coefficients (`N ≤ p + n_regressors`), or when
/// `ZᵀZ` is singular.
pub fn var_fit(data: &[f64], n_rows: usize, k: usize, p: usize) -> StatsResult<VarFit> {
    if k == 0 {
        return Err(StatsError::InvalidParameter {
            name: "k".to_string(),
            reason: "number of series must be ≥ 1".to_string(),
        });
    }
    if p == 0 {
        return Err(StatsError::InvalidParameter {
            name: "p".to_string(),
            reason: "lag order must be ≥ 1".to_string(),
        });
    }
    if data.len() != n_rows * k {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_rows, k],
            got: vec![data.len()],
        });
    }
    for (i, v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let m = 1 + k * p; // regressors per equation (incl. intercept)
    let t_eff = n_rows.saturating_sub(p); // usable observations
    if t_eff <= m {
        return Err(StatsError::InsufficientSampleSize {
            got: n_rows,
            need: p + m + 1,
        });
    }

    // Build Z (T×m) and response Yʳ (T×k).
    let mut z = vec![0.0; t_eff * m];
    let mut yr = vec![0.0; t_eff * k];
    for (row, t) in (p..n_rows).enumerate() {
        // Response Y_t.
        for c in 0..k {
            yr[row * k + c] = data[t * k + c];
        }
        // Intercept.
        z[row * m] = 1.0;
        // Lags Y_{t−1}, …, Y_{t−p}.
        let mut col = 1usize;
        for lag in 1..=p {
            let src = (t - lag) * k;
            for c in 0..k {
                z[row * m + col] = data[src + c];
                col += 1;
            }
        }
    }

    // ZᵀZ (m×m) and ZᵀYʳ (m×k).
    let mut ztz = vec![0.0; m * m];
    for a in 0..m {
        for b in a..m {
            let mut acc = 0.0;
            for r in 0..t_eff {
                acc += z[r * m + a] * z[r * m + b];
            }
            ztz[a * m + b] = acc;
            ztz[b * m + a] = acc;
        }
    }
    let mut zty = vec![0.0; m * k];
    for a in 0..m {
        for c in 0..k {
            let mut acc = 0.0;
            for r in 0..t_eff {
                acc += z[r * m + a] * yr[r * k + c];
            }
            zty[a * k + c] = acc;
        }
    }

    // B = (ZᵀZ)⁻¹ ZᵀYʳ  (m×k).
    let ztz_inv = matrix_inverse_lu(&ztz, m)?;
    let mut beta = vec![0.0; m * k];
    for a in 0..m {
        for c in 0..k {
            let mut acc = 0.0;
            for b in 0..m {
                acc += ztz_inv[a * m + b] * zty[b * k + c];
            }
            beta[a * k + c] = acc;
        }
    }

    // Residuals E = Yʳ − Z B  (T×k).
    let mut residuals = vec![0.0; t_eff * k];
    for r in 0..t_eff {
        for c in 0..k {
            let mut fitted = 0.0;
            for a in 0..m {
                fitted += z[r * m + a] * beta[a * k + c];
            }
            residuals[r * k + c] = yr[r * k + c] - fitted;
        }
    }

    // Σ = Eᵀ E / (T − kp − 1).
    let dof = t_eff as f64 - (k * p) as f64 - 1.0;
    let denom = if dof > 0.0 { dof } else { 1.0 };
    let mut sigma = vec![0.0; k * k];
    for a in 0..k {
        for b in a..k {
            let mut acc = 0.0;
            for r in 0..t_eff {
                acc += residuals[r * k + a] * residuals[r * k + b];
            }
            let val = acc / denom;
            sigma[a * k + b] = val;
            sigma[b * k + a] = val;
        }
    }

    // Unpack intercept and A_lag from B (B row 0 = intercept, rows for each lag).
    let mut intercept = vec![0.0; k];
    intercept.copy_from_slice(&beta[..k]); // row 0 of B
    let mut coefficients = Vec::with_capacity(p);
    for lag in 1..=p {
        let mut a_mat = vec![0.0; k * k];
        // Columns in B for this lag start at 1 + (lag−1)*k.
        let base = 1 + (lag - 1) * k;
        for i in 0..k {
            for j in 0..k {
                // Row (base + j) of B holds the loading of Y_{j,t−lag}; column i
                // is the equation for Y_{i,t}.  A_lag[i, j] = B[base + j, i].
                a_mat[i * k + j] = beta[(base + j) * k + i];
            }
        }
        coefficients.push(a_mat);
    }

    Ok(VarFit {
        k,
        p,
        n_obs: t_eff,
        intercept,
        coefficients,
        sigma,
        residuals,
        xtx_inv: ztz_inv,
        beta,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Forecasting
// ─────────────────────────────────────────────────────────────────────────────

/// Iterate the fitted recursion `h` steps ahead.
///
/// `history` is row-major `n_rows × k`; only its final `p` rows are used as the
/// initial condition.  The return is row-major `h × k` (row `s` is the forecast for
/// `Y_{N+s}`, `s = 0 … h−1`).
///
/// # Errors
/// Returns an error when `history` is shape-inconsistent or holds fewer than `p`
/// rows, or `h == 0`.
pub fn var_forecast(
    fit: &VarFit,
    history: &[f64],
    n_rows: usize,
    h: usize,
) -> StatsResult<Vec<f64>> {
    let k = fit.k;
    let p = fit.p;
    if history.len() != n_rows * k {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_rows, k],
            got: vec![history.len()],
        });
    }
    if n_rows < p {
        return Err(StatsError::InsufficientSampleSize {
            got: n_rows,
            need: p,
        });
    }
    if h == 0 {
        return Err(StatsError::InvalidParameter {
            name: "h".to_string(),
            reason: "forecast horizon must be ≥ 1".to_string(),
        });
    }

    // Rolling buffer of the last p observations, most-recent first.
    // lag_buf[lag-1] holds Y_{current − lag}.
    let mut lag_buf: Vec<Vec<f64>> = Vec::with_capacity(p);
    for lag in 1..=p {
        let src = (n_rows - lag) * k;
        lag_buf.push(history[src..src + k].to_vec());
    }

    let mut out = vec![0.0; h * k];
    for s in 0..h {
        let mut next = fit.intercept.clone();
        for lag in 1..=p {
            let a = &fit.coefficients[lag - 1];
            let y_lag = &lag_buf[lag - 1];
            for i in 0..k {
                let mut acc = 0.0;
                for j in 0..k {
                    acc += a[i * k + j] * y_lag[j];
                }
                next[i] += acc;
            }
        }
        for i in 0..k {
            out[s * k + i] = next[i];
        }
        // Shift the buffer: new observation becomes lag 1.
        lag_buf.pop();
        lag_buf.insert(0, next);
    }
    Ok(out)
}

/// Unconditional (long-run) mean `μ = (I − Σ A_i)⁻¹ c`.
///
/// For a stable process the forecasts converge to this value.
///
/// # Errors
/// Returns an error when `I − ΣA_i` is singular (non-invertible — the process is
/// not stable in the mean).
pub fn var_unconditional_mean(fit: &VarFit) -> StatsResult<Vec<f64>> {
    let k = fit.k;
    // M = I − Σ_i A_i.
    let mut m = vec![0.0; k * k];
    for i in 0..k {
        m[i * k + i] = 1.0;
    }
    for a in &fit.coefficients {
        for idx in 0..k * k {
            m[idx] -= a[idx];
        }
    }
    let m_inv = matrix_inverse_lu(&m, k)?;
    let mut mu = vec![0.0; k];
    for i in 0..k {
        let mut acc = 0.0;
        for j in 0..k {
            acc += m_inv[i * k + j] * fit.intercept[j];
        }
        mu[i] = acc;
    }
    Ok(mu)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stability via the companion matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Build the `kp × kp` companion matrix `F` of the VAR.
fn companion_matrix(fit: &VarFit) -> Vec<f64> {
    let k = fit.k;
    let p = fit.p;
    let kp = k * p;
    let mut f = vec![0.0; kp * kp];
    // Top block-row: [A_1 | A_2 | … | A_p].
    for (lag_idx, a) in fit.coefficients.iter().enumerate() {
        let col_base = lag_idx * k;
        for i in 0..k {
            for j in 0..k {
                f[i * kp + (col_base + j)] = a[i * k + j];
            }
        }
    }
    // Sub-diagonal identity blocks shifting the state.
    for block in 1..p {
        let row_base = block * k;
        let col_base = (block - 1) * k;
        for d in 0..k {
            f[(row_base + d) * kp + (col_base + d)] = 1.0;
        }
    }
    f
}

/// Spectral radius (largest eigenvalue modulus) of the companion matrix.
///
/// Estimated by the Gelfand limit `‖Fⁿ v‖^{1/n}`, which converges to the dominant
/// modulus even for complex-conjugate eigenpairs (where simple power iteration on a
/// real vector cannot lock onto a single eigenvector).
///
/// `< 1` ⇒ the VAR is stable (and stationary); `≥ 1` ⇒ explosive / unit-root.
#[must_use]
pub fn var_spectral_radius(fit: &VarFit) -> f64 {
    let f = companion_matrix(fit);
    let kp = fit.k * fit.p;
    if kp == 0 {
        return 0.0;
    }
    // Start from a uniform unit vector; renormalise each step to avoid overflow,
    // accumulating the log-growth so the n-th-root limit is numerically stable.
    let mut v = vec![1.0 / (kp as f64).sqrt(); kp];
    let mut log_growth = 0.0_f64;
    let iters = 200usize;
    for _ in 0..iters {
        let mut w = vec![0.0; kp];
        for i in 0..kp {
            let mut acc = 0.0;
            for j in 0..kp {
                acc += f[i * kp + j] * v[j];
            }
            w[i] = acc;
        }
        let norm = (w.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm < 1e-300 {
            // Nilpotent action ⇒ all eigenvalues zero.
            return 0.0;
        }
        log_growth += norm.ln();
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }
    // ‖Fⁿ v‖^{1/n} = exp(mean log step) → dominant eigenvalue modulus (Gelfand).
    (log_growth / iters as f64).exp().max(0.0)
}

/// Convenience predicate: `true` when the VAR is stable (`spectral radius < 1`).
#[must_use]
pub fn var_is_stable(fit: &VarFit) -> bool {
    var_spectral_radius(fit) < 1.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Granger causality
// ─────────────────────────────────────────────────────────────────────────────

/// Chi-squared survival `P(χ²(df) > x) = 1 − P(df/2, x/2)`.
fn chi2_sf(x: f64, df: f64) -> StatsResult<f64> {
    if x <= 0.0 {
        return Ok(1.0);
    }
    let p = crate::special::betainc::gammp(df / 2.0, x / 2.0)?;
    Ok((1.0 - p).clamp(0.0, 1.0))
}

/// Wald test that variable `cause` Granger-causes variable `effect`.
///
/// Tests `H₀`: every lag coefficient of `Y_cause` in the equation for `Y_effect`
/// is zero.  The Wald statistic is
///
/// ```text
/// W = β̂ᵀ [σ̂_effect² · R (ZᵀZ)⁻¹ Rᵀ]⁻¹ β̂  ~  χ²(p),
/// ```
///
/// where `β̂` gathers the `p` restricted coefficients, `R` selects their rows from
/// `(ZᵀZ)⁻¹`, and `σ̂_effect²` is the residual variance of the effect equation.
///
/// # Errors
/// Returns an error when `cause`/`effect` are out of range or the restricted
/// covariance block is singular.
pub fn granger_causality(
    fit: &VarFit,
    cause: usize,
    effect: usize,
    alpha: f64,
) -> StatsResult<GrangerResult> {
    let k = fit.k;
    let p = fit.p;
    if cause >= k {
        return Err(StatsError::IndexOutOfBounds {
            index: cause,
            len: k,
        });
    }
    if effect >= k {
        return Err(StatsError::IndexOutOfBounds {
            index: effect,
            len: k,
        });
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".to_string(),
            reason: format!("significance level must be in (0, 1); got {alpha}"),
        });
    }

    let m = fit.n_regressors();

    // Rows of B (= rows of (ZᵀZ)⁻¹) holding the `cause` lags: for lag ℓ the
    // regressor index is 1 + (ℓ−1)*k + cause.
    let restricted_rows: Vec<usize> = (1..=p).map(|lag| 1 + (lag - 1) * k + cause).collect();

    // β̂ — coefficients of those lags in the `effect` equation (column `effect` of B).
    let beta_r: Vec<f64> = restricted_rows
        .iter()
        .map(|&r| fit.beta[r * k + effect])
        .collect();

    // σ̂_effect² is the effect equation's residual variance.
    let sigma_ee = fit.sigma[effect * k + effect];
    if sigma_ee <= 0.0 || !sigma_ee.is_finite() {
        return Err(StatsError::NumericalInstability(
            "non-positive residual variance for the effect equation".to_string(),
        ));
    }

    // Restricted covariance V = σ̂² · R (ZᵀZ)⁻¹ Rᵀ  (p × p).
    let mut v = vec![0.0; p * p];
    for (a, &ra) in restricted_rows.iter().enumerate() {
        for (b, &rb) in restricted_rows.iter().enumerate() {
            v[a * p + b] = sigma_ee * fit.xtx_inv[ra * m + rb];
        }
    }

    let v_inv = matrix_inverse_lu(&v, p)?;

    // W = β̂ᵀ V⁻¹ β̂.
    let mut w = 0.0;
    for a in 0..p {
        let mut acc = 0.0;
        for b in 0..p {
            acc += v_inv[a * p + b] * beta_r[b];
        }
        w += beta_r[a] * acc;
    }
    let w = w.max(0.0);

    let p_value = chi2_sf(w, p as f64)?;
    Ok(GrangerResult {
        statistic: w,
        df: p,
        p_value,
        causes: p_value < alpha,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mini deterministic LCG (matches the convention used elsewhere in this crate).
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

    /// Simulate a VAR(1): Y_t = c + A Y_{t−1} + ε_t (ε scaled by `noise`).
    fn simulate_var1(a: &[f64], c: &[f64], k: usize, n: usize, noise: f64, seed: u64) -> Vec<f64> {
        let mut rng = TestRng::new(seed);
        let mut data = vec![0.0; n * k];
        // Start at the unconditional mean if invertible, else zero.
        for t in 1..n {
            for i in 0..k {
                let mut acc = c[i];
                for j in 0..k {
                    acc += a[i * k + j] * data[(t - 1) * k + j];
                }
                data[t * k + i] = acc + noise * rng.next_normal();
            }
        }
        data
    }

    // ── (a) Recover a known stable A_1 (k = 2) ────────────────────────────────

    #[test]
    fn recovers_known_a1() {
        let k = 2;
        let a = [0.5, 0.1, -0.2, 0.4];
        let c = [0.0, 0.0];
        let data = simulate_var1(&a, &c, k, 4000, 0.3, 1234);
        let fit = var_fit(&data, 4000, k, 1).expect("fit");
        for i in 0..k {
            for j in 0..k {
                let est = fit.coefficient(1, i, j);
                let truth = a[i * k + j];
                assert!(
                    (est - truth).abs() < 0.06,
                    "A1[{i},{j}] = {est}, expected {truth}"
                );
            }
        }
    }

    // ── (b) Forecasts converge to the unconditional mean ──────────────────────

    #[test]
    fn forecasts_converge_to_unconditional_mean() {
        let k = 2;
        let a = [0.5, 0.1, -0.2, 0.4];
        let c = [1.0, 2.0];
        let data = simulate_var1(&a, &c, k, 3000, 0.2, 99);
        let fit = var_fit(&data, 3000, k, 1).expect("fit");
        let mu = var_unconditional_mean(&fit).expect("mu");
        let fc = var_forecast(&fit, &data, 3000, 60).expect("forecast");
        // Final forecast row should be close to the unconditional mean.
        let last = &fc[59 * k..60 * k];
        for i in 0..k {
            assert!(
                (last[i] - mu[i]).abs() < 0.1,
                "forecast[{i}]={} mean={}",
                last[i],
                mu[i]
            );
        }
        // And it should have actually moved toward the mean from the first step.
        let first = &fc[0..k];
        let d_first: f64 = (0..k).map(|i| (first[i] - mu[i]).abs()).sum();
        let d_last: f64 = (0..k).map(|i| (last[i] - mu[i]).abs()).sum();
        assert!(d_last <= d_first + 1e-9, "forecasts did not converge");
    }

    // ── (c) Stability: radius < 1 for stable, ≥ 1 for explosive ───────────────

    #[test]
    fn stable_fixture_has_radius_below_one() {
        let k = 2;
        let a = [0.5, 0.1, -0.2, 0.4];
        let c = [0.0, 0.0];
        let data = simulate_var1(&a, &c, k, 2000, 0.3, 7);
        let fit = var_fit(&data, 2000, k, 1).expect("fit");
        let r = var_spectral_radius(&fit);
        assert!(r < 1.0, "stable VAR radius = {r}");
        assert!(var_is_stable(&fit));
    }

    #[test]
    fn explosive_a_is_flagged() {
        // Construct a fit directly with an explosive A (eigenvalue 1.2).
        let k = 1;
        let fit = VarFit {
            k,
            p: 1,
            n_obs: 10,
            intercept: vec![0.0],
            coefficients: vec![vec![1.2]],
            sigma: vec![1.0],
            residuals: vec![0.0; 10],
            xtx_inv: vec![1.0, 0.0, 0.0, 1.0],
            beta: vec![0.0, 1.2],
        };
        let r = var_spectral_radius(&fit);
        assert!(r >= 1.0, "explosive radius = {r}");
        assert!(!var_is_stable(&fit));
    }

    #[test]
    fn known_companion_radius_matches_eigenvalue() {
        // VAR(2), k=1: companion = [[a1, a2],[1,0]]; eigenvalues solve λ²−a1λ−a2=0.
        let a1 = 0.5;
        let a2 = 0.2;
        let fit = VarFit {
            k: 1,
            p: 2,
            n_obs: 10,
            intercept: vec![0.0],
            coefficients: vec![vec![a1], vec![a2]],
            sigma: vec![1.0],
            residuals: vec![0.0; 10],
            xtx_inv: vec![0.0; 9],
            beta: vec![0.0, a1, a2],
        };
        let lambda = 0.5 * (a1 + (a1 * a1 + 4.0 * a2).sqrt()); // dominant real root
        let r = var_spectral_radius(&fit);
        assert!(
            (r - lambda).abs() < 1e-3,
            "radius {r} vs eigenvalue {lambda}"
        );
    }

    // ── (d) Σ symmetric and PSD ───────────────────────────────────────────────

    #[test]
    fn sigma_symmetric_and_psd() {
        let k = 2;
        let a = [0.5, 0.1, -0.2, 0.4];
        let c = [0.0, 0.0];
        let data = simulate_var1(&a, &c, k, 1500, 0.5, 55);
        let fit = var_fit(&data, 1500, k, 1).expect("fit");
        // Symmetry.
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (fit.sigma[i * k + j] - fit.sigma[j * k + i]).abs() < 1e-12,
                    "Σ not symmetric at ({i},{j})"
                );
            }
        }
        // PSD via Sylvester for 2×2: diagonals ≥ 0 and determinant ≥ 0.
        assert!(fit.sigma[0] >= 0.0 && fit.sigma[3] >= 0.0);
        let det = fit.sigma[0] * fit.sigma[3] - fit.sigma[1] * fit.sigma[2];
        assert!(det >= -1e-12, "Σ determinant {det} negative");
        // xᵀΣx ≥ 0 for a probe vector.
        let probe = [0.7, -0.3];
        let mut q = 0.0;
        for i in 0..k {
            for j in 0..k {
                q += probe[i] * fit.sigma[i * k + j] * probe[j];
            }
        }
        assert!(q >= -1e-12, "quadratic form {q} negative");
    }

    // ── (e) Coefficient dimensions correct for (k, p) ─────────────────────────

    #[test]
    fn coefficient_dimensions_correct() {
        let k = 3;
        let p = 2;
        let n = 400;
        let mut rng = TestRng::new(2);
        let data: Vec<f64> = (0..n * k).map(|_| rng.next_normal()).collect();
        let fit = var_fit(&data, n, k, p).expect("fit");
        assert_eq!(fit.coefficients.len(), p);
        for a in &fit.coefficients {
            assert_eq!(a.len(), k * k);
        }
        assert_eq!(fit.intercept.len(), k);
        assert_eq!(fit.sigma.len(), k * k);
        assert_eq!(fit.n_regressors(), 1 + k * p);
        assert_eq!(fit.beta.len(), (1 + k * p) * k);
    }

    // ── (f) Granger causality: detects X→Y, ignores independence ──────────────

    #[test]
    fn granger_detects_directional_dependence() {
        // Build Y depending on lagged X, X independent of Y.
        // x_t = 0.5 x_{t−1} + u_t ;  y_t = 0.8 x_{t−1} + v_t.
        let k = 2; // column 0 = x, column 1 = y
        let n = 3000;
        let mut rng = TestRng::new(424);
        let mut data = vec![0.0; n * k];
        for t in 1..n {
            let x_prev = data[(t - 1) * k];
            data[t * k] = 0.5 * x_prev + rng.next_normal();
            data[t * k + 1] = 0.8 * x_prev + rng.next_normal();
        }
        let fit = var_fit(&data, n, k, 1).expect("fit");
        // x (col 0) Granger-causes y (col 1).
        let xy = granger_causality(&fit, 0, 1, 0.05).expect("granger");
        assert!(xy.causes, "expected X→Y; p={}", xy.p_value);
        // y (col 1) does NOT Granger-cause x (col 0).
        let yx = granger_causality(&fit, 1, 0, 0.05).expect("granger");
        assert!(!yx.causes, "did not expect Y→X; p={}", yx.p_value);
    }

    #[test]
    fn granger_independent_series_not_flagged() {
        let k = 2;
        let n = 3000;
        let mut rng = TestRng::new(909);
        let mut data = vec![0.0; n * k];
        for t in 1..n {
            data[t * k] = 0.4 * data[(t - 1) * k] + rng.next_normal();
            data[t * k + 1] = 0.4 * data[(t - 1) * k + 1] + rng.next_normal();
        }
        let fit = var_fit(&data, n, k, 1).expect("fit");
        let xy = granger_causality(&fit, 0, 1, 0.05).expect("granger");
        let yx = granger_causality(&fit, 1, 0, 0.05).expect("granger");
        assert!(!xy.causes, "false X→Y; p={}", xy.p_value);
        assert!(!yx.causes, "false Y→X; p={}", yx.p_value);
        assert!((0.0..=1.0).contains(&xy.p_value));
    }

    // ── (g) Univariate VAR(1) reduces to AR(1) ────────────────────────────────

    #[test]
    fn univariate_var1_is_ar1() {
        // y_t = 0.7 y_{t−1} + ε_t.
        let phi = 0.7;
        let n = 4000;
        let mut rng = TestRng::new(31);
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = phi * data[t - 1] + rng.next_normal();
        }
        let fit = var_fit(&data, n, 1, 1).expect("fit");
        let est = fit.coefficient(1, 0, 0);
        assert!((est - phi).abs() < 0.05, "AR(1) coeff {est} vs {phi}");
        // Stable & radius ≈ |phi|.
        let r = var_spectral_radius(&fit);
        assert!((r - est.abs()).abs() < 1e-6, "radius {r} vs |phi| {est}");
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn rejects_zero_lag_and_bad_shapes() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(var_fit(&data, 2, 2, 0).is_err()); // p=0
        assert!(var_fit(&data, 2, 0, 1).is_err()); // k=0
        assert!(var_fit(&data, 3, 2, 1).is_err()); // shape mismatch (3*2≠4)
    }

    #[test]
    fn rejects_too_few_observations() {
        // k=2, p=1 ⇒ m=3 ⇒ need T_eff > 3 ⇒ N > 4. N=4 must fail.
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        assert!(var_fit(&data, 4, 2, 1).is_err());
    }

    #[test]
    fn forecast_horizon_zero_errors() {
        let data: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let fit = var_fit(&data, 40, 1, 1).expect("fit");
        assert!(var_forecast(&fit, &data, 40, 0).is_err());
    }
}
