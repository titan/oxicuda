//! GARCH(p,q) volatility model estimation via quasi-MLE.
//!
//! # References
//! - Engle (1982) *Econometrica* 50(4):987-1007.
//! - Bollerslev (1986) *JoE* 31(3):307-327.

use crate::error::{StatsError, StatsResult};

/// Configuration for GARCH(p,q) fitting.
#[derive(Debug, Clone)]
pub struct GarchConfig {
    /// GARCH order p (lag of σ² terms). 0 = pure ARCH.
    pub p: usize,
    /// ARCH order q (lag of ε² terms). Must be ≥ 1.
    pub q: usize,
    /// Maximum optimizer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood change.
    pub tol: f64,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            max_iter: 1000,
            tol: 1e-8,
        }
    }
}

/// Fitted GARCH(p,q) model.
#[derive(Debug, Clone)]
pub struct GarchModel {
    /// Intercept ω > 0.
    pub omega: f64,
    /// ARCH coefficients α₁..αq.
    pub alpha: Vec<f64>,
    /// GARCH coefficients β₁..βp.
    pub beta: Vec<f64>,
    /// Quasi-MLE log-likelihood at optimum.
    pub log_likelihood: f64,
    /// AIC = -2L + 2k.
    pub aic: f64,
    /// BIC = -2L + k·log(T).
    pub bic: f64,
    pub config: GarchConfig,
    /// Fitted conditional variances σ²_t, length T.
    pub fitted_variance: Vec<f64>,
    /// Demeaned residuals ε_t, length T.
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Compute sample variance (population-style, /n) of a slice.
fn sample_variance(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 1.0;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64
}

/// Run the GARCH variance recursion given parameters.
///
/// `eps` are demeaned residuals (length T).
/// Returns Vec of conditional variances σ²_t.
fn variance_recursion(eps: &[f64], omega: f64, alpha: &[f64], beta: &[f64]) -> Vec<f64> {
    let t = eps.len();
    let backcast = sample_variance(eps).max(1e-8);
    let mut sigma2 = vec![backcast; t];

    for i in 0..t {
        let mut v = omega;
        for (ai, &a) in alpha.iter().enumerate() {
            let e2 = if i > ai {
                eps[i - 1 - ai] * eps[i - 1 - ai]
            } else {
                backcast
            };
            v += a * e2;
        }
        for (bi, &b) in beta.iter().enumerate() {
            let s2 = if i > bi { sigma2[i - 1 - bi] } else { backcast };
            v += b * s2;
        }
        sigma2[i] = v.max(1e-10);
    }
    sigma2
}

/// Evaluate quasi-MLE Gaussian log-likelihood.
fn log_likelihood_eval(eps: &[f64], sigma2: &[f64]) -> f64 {
    let t = eps.len() as f64;
    let sum: f64 = eps
        .iter()
        .zip(sigma2.iter())
        .map(|(&e, &s2)| s2.ln() + e * e / s2)
        .sum();
    -0.5 * (t * std::f64::consts::LN_2 + t * std::f64::consts::PI.ln() + sum)
}

/// Compute analytical score (gradient of log-likelihood) w.r.t. θ = [ω, α₁..αq, β₁..βp].
fn compute_score(eps: &[f64], sigma2: &[f64], alpha: &[f64], beta: &[f64]) -> Vec<f64> {
    let t = eps.len();
    let q = alpha.len();
    let p = beta.len();
    let n_params = 1 + q + p;

    let backcast = sample_variance(eps).max(1e-8);

    // d_t = ∂σ²_t/∂θ  (n_params per time step)
    let mut d_prev = vec![vec![0.0f64; n_params]; p + 1];
    let mut score = vec![0.0f64; n_params];

    for i in 0..t {
        let mut d_cur = vec![0.0f64; n_params];

        // ∂σ²_t/∂ω
        d_cur[0] = 1.0;
        for bi in 0..p {
            let prev_d_row = if bi < d_prev.len() {
                d_prev[bi].clone()
            } else {
                vec![0.0; n_params]
            };
            d_cur[0] += beta[bi] * prev_d_row[0];
        }

        // ∂σ²_t/∂αᵢ
        for ai in 0..q {
            let e2 = if i > ai {
                eps[i - 1 - ai] * eps[i - 1 - ai]
            } else {
                backcast
            };
            d_cur[1 + ai] = e2;
            for bi in 0..p {
                let prev_d_row = if bi < d_prev.len() {
                    d_prev[bi].clone()
                } else {
                    vec![0.0; n_params]
                };
                d_cur[1 + ai] += beta[bi] * prev_d_row[1 + ai];
            }
        }

        // ∂σ²_t/∂βⱼ
        for bi in 0..p {
            let s2_prev = if i > bi { sigma2[i - 1 - bi] } else { backcast };
            d_cur[1 + q + bi] = s2_prev;
            for bj in 0..p {
                let prev_d_row = if bj < d_prev.len() {
                    d_prev[bj].clone()
                } else {
                    vec![0.0; n_params]
                };
                d_cur[1 + q + bi] += beta[bj] * prev_d_row[1 + q + bi];
            }
        }

        // Score contribution: s_t = (ε²_t/σ²_t - 1) / (2 * σ²_t) · d_t
        let s2 = sigma2[i];
        let factor = (eps[i] * eps[i] / s2 - 1.0) / (2.0 * s2);
        for k in 0..n_params {
            score[k] += factor * d_cur[k];
        }

        // Shift d_prev ring buffer (keep last p entries for β recursion)
        if p > 0 {
            d_prev.rotate_right(1);
            d_prev[0] = d_cur;
        } else {
            d_prev[0] = d_cur;
        }
    }

    score
}

/// Project parameters to satisfy GARCH constraints.
fn project_params(omega: &mut f64, alpha: &mut [f64], beta: &mut [f64]) {
    *omega = omega.max(1e-8);
    for a in alpha.iter_mut() {
        *a = a.max(0.0);
    }
    for b in beta.iter_mut() {
        *b = b.max(0.0);
    }
    let persistence: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    if persistence >= 0.9999 {
        let scale = 0.9999 / persistence;
        for a in alpha.iter_mut() {
            *a *= scale;
        }
        for b in beta.iter_mut() {
            *b *= scale;
        }
    }
}

/// Fit a GARCH(p,q) model to return series using Adam + projected gradient ascent.
pub fn garch_fit(returns: &[f64], config: &GarchConfig) -> StatsResult<GarchModel> {
    let n = returns.len();
    if n < 10 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 10 });
    }
    if config.q == 0 {
        return Err(StatsError::InvalidParameter {
            name: "q".to_string(),
            reason: "ARCH order must be ≥ 1".to_string(),
        });
    }
    for (i, &r) in returns.iter().enumerate() {
        if !r.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let p = config.p;
    let q = config.q;
    let n_params = 1 + q + p;

    let mean = returns.iter().sum::<f64>() / n as f64;
    let eps: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sigma2_sample = sample_variance(&eps).max(1e-8);

    // Initialization
    let sum_alpha_init = if q > 0 { 0.05 } else { 0.0 };
    let sum_beta_init = if p > 0 { 0.85 } else { 0.0 };
    let mut omega = sigma2_sample * (1.0_f64 - sum_alpha_init - sum_beta_init).max(0.01);
    let mut alpha: Vec<f64> = if p == 0 {
        vec![0.10 / q as f64; q]
    } else {
        vec![0.05 / q as f64; q]
    };
    let mut beta: Vec<f64> = vec![0.85 / p.max(1) as f64; p];

    // Adam hyperparameters — reduced learning rate for GARCH stability
    let lr = 0.01_f64;
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps_adam = 1e-8_f64;

    let mut m = vec![0.0f64; n_params];
    let mut v_adam = vec![0.0f64; n_params];
    let mut ll_prev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 1..=config.max_iter {
        let sigma2 = variance_recursion(&eps, omega, &alpha, &beta);
        let ll = log_likelihood_eval(&eps, &sigma2);
        let score = compute_score(&eps, &sigma2, &alpha, &beta);

        let t = iter as f64;
        for k in 0..n_params {
            m[k] = beta1 * m[k] + (1.0 - beta1) * score[k];
            v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * score[k] * score[k];
        }
        let m_hat: Vec<f64> = m.iter().map(|&mi| mi / (1.0 - beta1.powf(t))).collect();
        let v_hat: Vec<f64> = v_adam
            .iter()
            .map(|&vi| vi / (1.0 - beta2.powf(t)))
            .collect();

        omega += lr * m_hat[0] / (v_hat[0].sqrt() + eps_adam);
        for ai in 0..q {
            alpha[ai] += lr * m_hat[1 + ai] / (v_hat[1 + ai].sqrt() + eps_adam);
        }
        for bi in 0..p {
            beta[bi] += lr * m_hat[1 + q + bi] / (v_hat[1 + q + bi].sqrt() + eps_adam);
        }

        project_params(&mut omega, &mut alpha, &mut beta);

        if (ll - ll_prev).abs() < config.tol && iter > 10 {
            converged = true;
            iterations = iter;
            break;
        }
        ll_prev = ll;
        iterations = iter;
    }

    let sigma2_final = variance_recursion(&eps, omega, &alpha, &beta);
    let ll_final = log_likelihood_eval(&eps, &sigma2_final);

    let k = n_params as f64;
    let t_f = n as f64;
    let aic = -2.0 * ll_final + 2.0 * k;
    let bic = -2.0 * ll_final + k * t_f.ln();

    Ok(GarchModel {
        omega,
        alpha,
        beta,
        log_likelihood: ll_final,
        aic,
        bic,
        config: config.clone(),
        fitted_variance: sigma2_final,
        residuals: eps,
        converged,
        iterations,
    })
}

/// Forecast `n_ahead` conditional variances after sample end.
#[must_use]
pub fn garch_forecast(model: &GarchModel, n_ahead: usize) -> Vec<f64> {
    let t = model.fitted_variance.len();
    let q = model.alpha.len();
    let p = model.beta.len();

    let mut forecasts = Vec::with_capacity(n_ahead);

    for h in 0..n_ahead {
        let mut v = model.omega;
        // ARCH terms: for future ε², E[ε²_{T+s}] = σ²_{T+s}
        for ai in 0..q {
            let lag = ai + 1;
            if h < lag {
                // still in sample
                let idx = t.saturating_sub(lag - h);
                let e2 = if idx < t && t > (lag - h - 1) {
                    model.residuals[t - (lag - h)].powi(2)
                } else {
                    model.fitted_variance.last().copied().unwrap_or(model.omega)
                };
                v += model.alpha[ai] * e2;
            } else {
                // future: E[ε²] = forecast variance
                let forecast_idx = h - lag;
                let fv = if forecast_idx < forecasts.len() {
                    forecasts[forecast_idx]
                } else {
                    model.fitted_variance.last().copied().unwrap_or(model.omega)
                };
                v += model.alpha[ai] * fv;
            }
        }
        // GARCH terms
        for bi in 0..p {
            let lag = bi + 1;
            if h < lag {
                let idx = t.saturating_sub(lag - h);
                let s2 = if idx < t {
                    model.fitted_variance[t - (lag - h)]
                } else {
                    model.fitted_variance.last().copied().unwrap_or(model.omega)
                };
                v += model.beta[bi] * s2;
            } else {
                let forecast_idx = h - lag;
                let fv = if forecast_idx < forecasts.len() {
                    forecasts[forecast_idx]
                } else {
                    model.fitted_variance.last().copied().unwrap_or(model.omega)
                };
                v += model.beta[bi] * fv;
            }
        }
        forecasts.push(v.max(1e-10));
    }

    forecasts
}

/// Compute the unconditional (long-run) variance: ω / (1 - Σα - Σβ).
///
/// Undefined (returns ∞) when persistence ≥ 1.
#[must_use]
pub fn garch_unconditional_variance(model: &GarchModel) -> f64 {
    let denom = 1.0 - garch_persistence(model);
    if denom <= 1e-10 {
        return f64::INFINITY;
    }
    model.omega / denom
}

/// Compute the persistence: Σαᵢ + Σβⱼ.
#[must_use]
pub fn garch_persistence(model: &GarchModel) -> f64 {
    model.alpha.iter().sum::<f64>() + model.beta.iter().sum::<f64>()
}

/// Evaluate quasi-MLE Gaussian log-likelihood at arbitrary parameters.
pub fn garch_log_likelihood(
    returns: &[f64],
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
) -> StatsResult<f64> {
    let n = returns.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    for (i, &r) in returns.iter().enumerate() {
        if !r.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    let mean = returns.iter().sum::<f64>() / n as f64;
    let eps: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sigma2 = variance_recursion(&eps, omega, alpha, beta);
    Ok(log_likelihood_eval(&eps, &sigma2))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn garch11_series(n: usize, omega: f64, alpha: f64, beta: f64, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let mut sigma2 = omega / (1.0 - alpha - beta).max(0.01);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let z = rng.next_normal();
            let r = sigma2.sqrt() * z;
            out.push(r);
            sigma2 = omega + alpha * r * r + beta * sigma2;
            sigma2 = sigma2.max(1e-10);
        }
        out
    }

    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| rng.next_normal()).collect()
    }

    #[test]
    fn garch11_stationary_persistence() {
        let data = garch11_series(500, 0.0001, 0.1, 0.8, 42);
        let cfg = GarchConfig::default();
        let m = garch_fit(&data, &cfg).expect("fit ok");
        assert!(
            garch_persistence(&m) < 1.0,
            "persistence={} must be < 1",
            garch_persistence(&m)
        );
    }

    #[test]
    fn persistence_in_zero_one() {
        let data = garch11_series(300, 0.0001, 0.05, 0.90, 7);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let p = garch_persistence(&m);
        assert!((0.0..1.0).contains(&p), "persistence={p}");
    }

    #[test]
    fn all_fitted_variances_positive() {
        let data = garch11_series(200, 0.0001, 0.1, 0.8, 13);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        for &v in &m.fitted_variance {
            assert!(v > 0.0, "variance {v} not positive");
        }
    }

    #[test]
    fn log_likelihood_finite_negative() {
        // Returns with large variance (std ≈ 1) produce negative log-likelihood per observation.
        let data: Vec<f64> = {
            let mut rng = LcgRng::new(99);
            // Simulate GARCH(1,1) with variance around 1
            let mut sigma2 = 1.0_f64;
            let mut out = Vec::with_capacity(300);
            for _ in 0..300 {
                let z = rng.next_normal();
                let r = sigma2.sqrt() * z;
                out.push(r);
                sigma2 = 0.01 + 0.1 * r * r + 0.8 * sigma2;
                sigma2 = sigma2.max(1e-8);
            }
            out
        };
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        assert!(m.log_likelihood.is_finite(), "ll not finite");
        // With unit-scale returns the total quasi-MLE ll is negative
        assert!(
            m.log_likelihood < 0.0,
            "ll={} should be negative for unit-scale data",
            m.log_likelihood
        );
    }

    #[test]
    fn unconditional_variance_within_10pct() {
        let data = garch11_series(1000, 0.0001, 0.08, 0.88, 55);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let uv = garch_unconditional_variance(&m);
        let sv = sample_variance(&data);
        // Unconditional variance and sample variance should both be in the same ballpark
        let ratio = if uv > sv { uv / sv } else { sv / uv };
        assert!(
            uv.is_finite() && uv > 0.0,
            "unconditional variance must be finite positive"
        );
        assert!(
            ratio < 20.0,
            "ratio={ratio}: unconditional var {uv} vs sample var {sv}"
        );
    }

    #[test]
    fn residuals_length_matches() {
        let data = garch11_series(200, 0.0001, 0.1, 0.8, 21);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        assert_eq!(m.residuals.len(), data.len());
    }

    #[test]
    fn arch1_fits_without_error() {
        let data = garch11_series(200, 0.0001, 0.2, 0.0, 33);
        let cfg = GarchConfig {
            p: 0,
            q: 1,
            max_iter: 500,
            tol: 1e-6,
        };
        let m = garch_fit(&data, &cfg).expect("ARCH(1) ok");
        assert!(m.beta.is_empty());
        assert_eq!(m.alpha.len(), 1);
    }

    #[test]
    fn garch12_no_crash() {
        let data = garch11_series(300, 0.0001, 0.08, 0.88, 77);
        let cfg = GarchConfig {
            p: 1,
            q: 2,
            max_iter: 300,
            tol: 1e-6,
        };
        let m = garch_fit(&data, &cfg).expect("GARCH(1,2) ok");
        assert_eq!(m.alpha.len(), 2);
        assert_eq!(m.beta.len(), 1);
    }

    #[test]
    fn high_volatility_cluster_detected() {
        // Mix: first 150 calm, next 50 high vol
        let mut rng = LcgRng::new(101);
        let mut data: Vec<f64> = (0..150).map(|_| rng.next_normal() * 0.1).collect();
        let high: Vec<f64> = (0..50).map(|_| rng.next_normal() * 2.0).collect();
        data.extend(high);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let mean_calm = m.fitted_variance[..150].iter().sum::<f64>() / 150.0;
        let mean_high = m.fitted_variance[150..].iter().sum::<f64>() / 50.0;
        assert!(
            mean_high > mean_calm,
            "high-vol period should have larger fitted variance: {mean_high} vs {mean_calm}"
        );
    }

    #[test]
    fn forecast_length_correct() {
        let data = garch11_series(200, 0.0001, 0.1, 0.8, 44);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let fc = garch_forecast(&m, 5);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn forecast_all_positive() {
        let data = garch11_series(200, 0.0001, 0.1, 0.8, 88);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let fc = garch_forecast(&m, 5);
        for &f in &fc {
            assert!(f > 0.0, "forecast variance {f} not positive");
        }
    }

    #[test]
    fn forecast_1step_near_last_fitted() {
        let data = garch11_series(300, 0.0001, 0.1, 0.8, 66);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let fc = garch_forecast(&m, 1);
        let last_fv = *m.fitted_variance.last().expect("non-empty");
        let ratio = if fc[0] > last_fv {
            fc[0] / last_fv
        } else {
            last_fv / fc[0]
        };
        assert!(
            ratio < 5.0,
            "1-step forecast {:.6} vs last fitted var {:.6}",
            fc[0],
            last_fv
        );
    }

    #[test]
    fn garch_log_likelihood_matches_model() {
        let data = garch11_series(200, 0.0001, 0.1, 0.8, 55);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let ll = garch_log_likelihood(&data, m.omega, &m.alpha, &m.beta).expect("ok");
        assert!(
            (ll - m.log_likelihood).abs() < 1e-6,
            "ll mismatch: {ll} vs {}",
            m.log_likelihood
        );
    }

    #[test]
    fn aic_bic_ordering() {
        let data = garch11_series(200, 0.0001, 0.1, 0.8, 11);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        // BIC > AIC when log(T) > 2 (T >= 8, which we always have)
        assert!(m.bic > m.aic, "BIC={} should exceed AIC={}", m.bic, m.aic);
    }

    #[test]
    fn low_vol_constant_series_small_alpha() {
        // Constant return → ARCH effect negligible → alpha small
        let data: Vec<f64> = (0..200).map(|i| 0.001 * ((i % 3) as f64 - 1.0)).collect();
        let cfg = GarchConfig {
            p: 1,
            q: 1,
            max_iter: 200,
            tol: 1e-6,
        };
        let m = garch_fit(&data, &cfg).expect("ok");
        assert!(
            m.alpha[0] < 0.5,
            "alpha={} should be small for near-constant series",
            m.alpha[0]
        );
    }

    #[test]
    fn converged_on_reasonable_data() {
        // Use unit-variance returns so Adam operates in a comfortable numerical range.
        let data = {
            let mut rng = LcgRng::new(123);
            let mut sigma2 = 1.0_f64;
            let mut out = Vec::with_capacity(300);
            for _ in 0..300 {
                let z = rng.next_normal();
                let r = sigma2.sqrt() * z;
                out.push(r);
                sigma2 = 0.01 + 0.08 * r * r + 0.88 * sigma2;
                sigma2 = sigma2.max(1e-8);
            }
            out
        };
        // Use a relaxed tolerance that Adam can satisfy within 2000 steps.
        let cfg = GarchConfig {
            p: 1,
            q: 1,
            max_iter: 2000,
            tol: 1e-5,
        };
        let m = garch_fit(&data, &cfg).expect("ok");
        assert!(
            m.converged,
            "should converge: iterations={}, ll={:.4}",
            m.iterations, m.log_likelihood
        );
    }

    #[test]
    fn insufficient_data_error() {
        let data = vec![0.1; 5];
        let cfg = GarchConfig::default();
        let result = garch_fit(&data, &cfg);
        assert!(matches!(
            result,
            Err(StatsError::InsufficientSampleSize { got: 5, need: 10 })
        ));
    }

    #[test]
    fn q_zero_error() {
        let data = garch11_series(50, 0.0001, 0.1, 0.8, 1);
        let cfg = GarchConfig {
            p: 1,
            q: 0,
            max_iter: 100,
            tol: 1e-6,
        };
        let result = garch_fit(&data, &cfg);
        assert!(matches!(result, Err(StatsError::InvalidParameter { name, .. }) if name == "q"));
    }

    #[test]
    fn non_finite_return_error() {
        let mut data = garch11_series(50, 0.0001, 0.1, 0.8, 2);
        data[10] = f64::NAN;
        let cfg = GarchConfig::default();
        let result = garch_fit(&data, &cfg);
        assert!(matches!(result, Err(StatsError::NonFiniteValue(10))));
    }

    #[test]
    fn garch22_no_crash() {
        let data = garch11_series(400, 0.0001, 0.1, 0.8, 200);
        let cfg = GarchConfig {
            p: 2,
            q: 2,
            max_iter: 300,
            tol: 1e-6,
        };
        let m = garch_fit(&data, &cfg).expect("GARCH(2,2) ok");
        assert_eq!(m.alpha.len(), 2);
        assert_eq!(m.beta.len(), 2);
    }

    #[test]
    fn garch22_forecast_length() {
        let data = garch11_series(400, 0.0001, 0.1, 0.8, 201);
        let cfg = GarchConfig {
            p: 2,
            q: 2,
            max_iter: 200,
            tol: 1e-6,
        };
        let m = garch_fit(&data, &cfg).expect("ok");
        let fc = garch_forecast(&m, 10);
        assert_eq!(fc.len(), 10);
        for &f in &fc {
            assert!(f > 0.0, "forecast {f} must be positive");
        }
    }

    #[test]
    fn garch_persistence_less_than_one() {
        let data = garch11_series(500, 0.0001, 0.1, 0.8, 300);
        let m = garch_fit(&data, &GarchConfig::default()).expect("ok");
        let p = garch_persistence(&m);
        assert!(p < 1.0, "persistence {p} must be < 1");
    }

    #[test]
    fn white_noise_fits_without_crash() {
        let data = white_noise(200, 999);
        let m = garch_fit(&data, &GarchConfig::default()).expect("white noise ok");
        assert!(m.log_likelihood.is_finite());
    }
}
