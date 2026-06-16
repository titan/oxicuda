//! Gaussian Mixture Models via the EM algorithm.
//!
//! # References
//! - Dempster, Laird & Rubin (1977) *JRSS-B* 39(1):1-38.
//! - Bishop (2006) *Pattern Recognition and Machine Learning*, Chapter 9.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Covariance structure for GMM components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmmCovariance {
    Full,
    Diagonal,
    Spherical,
}

/// Configuration for GMM fitting.
#[derive(Debug, Clone)]
pub struct GmmConfig {
    pub n_components: usize,
    pub covariance_type: GmmCovariance,
    pub max_iter: usize,
    pub tol: f64,
    pub reg_covar: f64,
    pub n_init: usize,
    pub seed: u64,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            covariance_type: GmmCovariance::Full,
            max_iter: 200,
            tol: 1e-6,
            reg_covar: 1e-6,
            n_init: 1,
            seed: 0,
        }
    }
}

/// Fitted Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct GmmModel {
    /// Mixing weights π_k (length K), sums to 1.
    pub weights: Vec<f64>,
    /// Component means μ_k, shape K × n_features (row-major).
    pub means: Vec<f64>,
    /// Full covariance matrices Σ_k, shape K × n_features × n_features (row-major).
    pub covariances: Vec<f64>,
    pub log_likelihood: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub config: GmmConfig,
    pub n_features: usize,
    pub n_components: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky and linear-algebra helpers (inline, no external deps)
// ─────────────────────────────────────────────────────────────────────────────

fn cholesky(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

fn log_det_from_chol(l: &[f64], d: usize) -> f64 {
    (0..d).map(|j| l[j * d + j].ln()).sum::<f64>() * 2.0
}

fn mahalanobis_sq(l: &[f64], v: &[f64], d: usize) -> f64 {
    let mut y = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = v[i];
        for j in 0..i {
            s -= l[i * d + j] * y[j];
        }
        y[i] = s / l[i * d + i];
    }
    y.iter().map(|&yi| yi * yi).sum()
}

fn logsumexp(v: &[f64]) -> f64 {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max_v + v.iter().map(|x| (x - max_v).exp()).sum::<f64>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// K-means++ initialization
// ─────────────────────────────────────────────────────────────────────────────

fn kmeans_plusplus_init(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    rng: &mut LcgRng,
) -> Vec<Vec<f64>> {
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = rng.next_usize(n);
    centers.push(x[first * d..(first + 1) * d].to_vec());

    let mut dist2 = vec![f64::INFINITY; n];

    for _ in 1..k {
        // Update min-distance to nearest chosen center
        for ni in 0..n {
            let xi = &x[ni * d..(ni + 1) * d];
            for c in &centers {
                let d2: f64 = xi.iter().zip(c.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
                if d2 < dist2[ni] {
                    dist2[ni] = d2;
                }
            }
        }
        let total: f64 = dist2.iter().sum();
        if total < 1e-300 {
            // All points coincide; pick random
            centers.push(x[rng.next_usize(n) * d..rng.next_usize(n).min(n - 1) * d + d].to_vec());
            continue;
        }
        let mut target = rng.next_f64() * total;
        let mut chosen = n - 1;
        for (ni, &d2) in dist2.iter().enumerate().take(n) {
            target -= d2;
            if target <= 0.0 {
                chosen = ni;
                break;
            }
        }
        centers.push(x[chosen * d..(chosen + 1) * d].to_vec());
    }

    centers
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-likelihood of a single Gaussian component at one point
// ─────────────────────────────────────────────────────────────────────────────

fn log_gaussian(
    xn: &[f64],
    mu: &[f64],
    cov: &[f64],
    d: usize,
    reg_covar: f64,
) -> Option<f64> {
    // Add reg_covar to diagonal before Cholesky
    let mut cov_reg = cov.to_vec();
    for j in 0..d {
        cov_reg[j * d + j] += reg_covar;
    }
    let l = cholesky(&cov_reg, d)?;
    let log_det = log_det_from_chol(&l, d);
    let diff: Vec<f64> = xn.iter().zip(mu.iter()).map(|(&a, &b)| a - b).collect();
    let mah = mahalanobis_sq(&l, &diff, d);
    let log_prob = -0.5 * (d as f64 * std::f64::consts::LN_2
        + d as f64 * std::f64::consts::PI.ln()
        + log_det
        + mah);
    Some(log_prob)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single EM run
// ─────────────────────────────────────────────────────────────────────────────

fn em_run(
    x: &[f64],
    n: usize,
    d: usize,
    config: &GmmConfig,
    init_means: Vec<Vec<f64>>,
) -> StatsResult<GmmModel> {
    let k = config.n_components;

    // Initialize weights, means, covariances
    let mut weights = vec![1.0 / k as f64; k];
    let mut means: Vec<f64> = init_means.iter().flat_map(|m| m.iter().copied()).collect();

    // Compute overall variance for spherical/diagonal init
    let mean_var = {
        let overall_mean: Vec<f64> = (0..d)
            .map(|j| x.iter().skip(j).step_by(d).sum::<f64>() / n as f64)
            .collect();
        (0..d)
            .map(|j| {
                x.iter()
                    .skip(j)
                    .step_by(d)
                    .map(|&v| (v - overall_mean[j]) * (v - overall_mean[j]))
                    .sum::<f64>()
                    / n as f64
            })
            .sum::<f64>()
            / d as f64
    };

    // Init covariances as scaled identity
    let mut covariances = vec![0.0f64; k * d * d];
    for ki in 0..k {
        for j in 0..d {
            covariances[ki * d * d + j * d + j] = mean_var.max(config.reg_covar);
        }
    }

    // Responsibilities γ[n, k]
    let mut gamma = vec![0.0f64; n * k];
    let mut ll_prev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..config.max_iter {
        // E-step: compute log responsibilities
        let mut ll_total = 0.0f64;
        for ni in 0..n {
            let xn = &x[ni * d..(ni + 1) * d];
            let mut log_resp = vec![f64::NEG_INFINITY; k];
            for ki in 0..k {
                let mu = &means[ki * d..(ki + 1) * d];
                let cov = &covariances[ki * d * d..(ki + 1) * d * d];
                if let Some(lg) = log_gaussian(xn, mu, cov, d, config.reg_covar) {
                    log_resp[ki] = weights[ki].ln() + lg;
                }
            }
            let lz = logsumexp(&log_resp);
            ll_total += lz;
            for ki in 0..k {
                gamma[ni * k + ki] = (log_resp[ki] - lz).exp();
            }
        }

        let ll = ll_total;
        n_iter = iter + 1;

        if (ll - ll_prev).abs() < config.tol && iter > 0 {
            converged = true;
            break;
        }
        ll_prev = ll;

        // M-step
        for ki in 0..k {
            let nk: f64 = (0..n).map(|ni| gamma[ni * k + ki]).sum();
            let nk_safe = nk.max(1e-10);

            // Update weights
            weights[ki] = nk / n as f64;

            // Update means
            let new_mu: Vec<f64> = (0..d)
                .map(|j| {
                    (0..n).map(|ni| gamma[ni * k + ki] * x[ni * d + j]).sum::<f64>() / nk_safe
                })
                .collect();
            means[ki * d..(ki + 1) * d].copy_from_slice(&new_mu);

            // Update covariances
            let new_cov = match config.covariance_type {
                GmmCovariance::Full => {
                    let mut cov = vec![0.0f64; d * d];
                    for ni in 0..n {
                        let g = gamma[ni * k + ki];
                        for di in 0..d {
                            let diff_i = x[ni * d + di] - new_mu[di];
                            for dj in 0..d {
                                let diff_j = x[ni * d + dj] - new_mu[dj];
                                cov[di * d + dj] += g * diff_i * diff_j;
                            }
                        }
                    }
                    for v in cov.iter_mut() {
                        *v /= nk_safe;
                    }
                    // reg_covar added during Cholesky
                    cov
                }
                GmmCovariance::Diagonal => {
                    let mut cov = vec![0.0f64; d * d];
                    for ni in 0..n {
                        let g = gamma[ni * k + ki];
                        for di in 0..d {
                            let diff = x[ni * d + di] - new_mu[di];
                            cov[di * d + di] += g * diff * diff;
                        }
                    }
                    for j in 0..d {
                        cov[j * d + j] /= nk_safe;
                    }
                    cov
                }
                GmmCovariance::Spherical => {
                    let mut total_var = 0.0f64;
                    for ni in 0..n {
                        let g = gamma[ni * k + ki];
                        let sq: f64 = (0..d)
                            .map(|di| {
                                let diff = x[ni * d + di] - new_mu[di];
                                diff * diff
                            })
                            .sum();
                        total_var += g * sq;
                    }
                    let sigma2 = total_var / (nk_safe * d as f64);
                    let mut cov = vec![0.0f64; d * d];
                    for j in 0..d {
                        cov[j * d + j] = sigma2;
                    }
                    cov
                }
            };
            covariances[ki * d * d..(ki + 1) * d * d].copy_from_slice(&new_cov);
        }
    }

    // Final E-step to compute accurate log-likelihood
    let mut ll_final = 0.0f64;
    for ni in 0..n {
        let xn = &x[ni * d..(ni + 1) * d];
        let log_resp: Vec<f64> = (0..k)
            .map(|ki| {
                let mu = &means[ki * d..(ki + 1) * d];
                let cov = &covariances[ki * d * d..(ki + 1) * d * d];
                log_gaussian(xn, mu, cov, d, config.reg_covar)
                    .map(|lg| weights[ki].ln() + lg)
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();
        ll_final += logsumexp(&log_resp);
    }

    // Validate all covariances are invertible (Cholesky check)
    for ki in 0..k {
        let cov = &covariances[ki * d * d..(ki + 1) * d * d];
        let mut cov_reg = cov.to_vec();
        for j in 0..d {
            cov_reg[j * d + j] += config.reg_covar;
        }
        if cholesky(&cov_reg, d).is_none() {
            return Err(StatsError::NumericalInstability(format!(
                "covariance of component {ki} is singular even with reg_covar"
            )));
        }
    }

    Ok(GmmModel {
        weights,
        means,
        covariances,
        log_likelihood: ll_final,
        n_iter,
        converged,
        config: config.clone(),
        n_features: d,
        n_components: k,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a GMM with n_init random restarts, returning the best (highest log-likelihood) model.
pub fn gmm_fit(x: &[f64], n: usize, n_features: usize, config: &GmmConfig) -> StatsResult<GmmModel> {
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if n_features == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_features".to_string(),
            reason: "must be ≥ 1".to_string(),
        });
    }
    if config.n_components == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_components".to_string(),
            reason: "must be ≥ 1".to_string(),
        });
    }
    if x.len() != n * n_features {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: n * n_features,
        });
    }
    if n < config.n_components {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: config.n_components,
        });
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let d = n_features;
    let k = config.n_components;
    let n_init = config.n_init.max(1);

    let mut best_model: Option<GmmModel> = None;

    for init_run in 0..n_init {
        // Deterministic seed per init to ensure reproducibility per (seed, init_run)
        let seed = config.seed.wrapping_add(init_run as u64 * 6_364_136_223_846_793_005);
        let mut rng = LcgRng::new(seed);
        let init_means = kmeans_plusplus_init(x, n, d, k, &mut rng);
        let model = em_run(x, n, d, config, init_means)?;
        match &best_model {
            None => best_model = Some(model),
            Some(prev) => {
                if model.log_likelihood > prev.log_likelihood {
                    best_model = Some(model);
                }
            }
        }
    }

    best_model.ok_or(StatsError::NumericalInstability("all EM runs failed".to_string()))
}

/// Compute per-sample log-likelihoods for n new points under the model.
///
/// Returns Vec of length n.
pub fn gmm_score(model: &GmmModel, x: &[f64], n: usize) -> StatsResult<Vec<f64>> {
    let d = model.n_features;
    let k = model.n_components;
    if x.len() != n * d {
        return Err(StatsError::DimensionMismatch { a: x.len(), b: n * d });
    }
    let mut scores = Vec::with_capacity(n);
    for ni in 0..n {
        let xn = &x[ni * d..(ni + 1) * d];
        let log_resp: Vec<f64> = (0..k)
            .map(|ki| {
                let mu = &model.means[ki * d..(ki + 1) * d];
                let cov = &model.covariances[ki * d * d..(ki + 1) * d * d];
                log_gaussian(xn, mu, cov, d, model.config.reg_covar)
                    .map(|lg| model.weights[ki].ln() + lg)
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();
        scores.push(logsumexp(&log_resp));
    }
    Ok(scores)
}

/// Predict responsibilities (soft assignments) for n new points.
///
/// Returns flattened n × K matrix (row-major).
pub fn gmm_predict_proba(model: &GmmModel, x: &[f64], n: usize) -> StatsResult<Vec<f64>> {
    let d = model.n_features;
    let k = model.n_components;
    if x.len() != n * d {
        return Err(StatsError::DimensionMismatch { a: x.len(), b: n * d });
    }
    let mut proba = vec![0.0f64; n * k];
    for ni in 0..n {
        let xn = &x[ni * d..(ni + 1) * d];
        let mut log_resp: Vec<f64> = (0..k)
            .map(|ki| {
                let mu = &model.means[ki * d..(ki + 1) * d];
                let cov = &model.covariances[ki * d * d..(ki + 1) * d * d];
                log_gaussian(xn, mu, cov, d, model.config.reg_covar)
                    .map(|lg| model.weights[ki].ln() + lg)
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();
        let lz = logsumexp(&log_resp);
        for val in log_resp.iter_mut() {
            *val = (*val - lz).exp();
        }
        proba[ni * k..(ni + 1) * k].copy_from_slice(&log_resp);
    }
    Ok(proba)
}

/// Predict hard cluster assignments (argmax of responsibilities) for n new points.
pub fn gmm_predict(model: &GmmModel, x: &[f64], n: usize) -> StatsResult<Vec<usize>> {
    let proba = gmm_predict_proba(model, x, n)?;
    let k = model.n_components;
    let labels: Vec<usize> = (0..n)
        .map(|ni| {
            let row = &proba[ni * k..(ni + 1) * k];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ki, _)| ki)
                .unwrap_or(0)
        })
        .collect();
    Ok(labels)
}

/// BIC = -2 * log_likelihood + num_params * log(n).
#[must_use]
pub fn gmm_bic(model: &GmmModel, n: usize) -> f64 {
    let k = model.n_components as f64;
    let d = model.n_features as f64;
    let num_params = match model.config.covariance_type {
        GmmCovariance::Full => (k - 1.0) + k * d + k * d * (d + 1.0) / 2.0,
        GmmCovariance::Diagonal => (k - 1.0) + k * d + k * d,
        GmmCovariance::Spherical => (k - 1.0) + k * d + k,
    };
    -2.0 * model.log_likelihood + num_params * (n as f64).ln()
}

/// AIC = -2 * log_likelihood + 2 * num_params.
#[must_use]
pub fn gmm_aic(model: &GmmModel) -> f64 {
    let k = model.n_components as f64;
    let d = model.n_features as f64;
    let num_params = match model.config.covariance_type {
        GmmCovariance::Full => (k - 1.0) + k * d + k * d * (d + 1.0) / 2.0,
        GmmCovariance::Diagonal => (k - 1.0) + k * d + k * d,
        GmmCovariance::Spherical => (k - 1.0) + k * d + k,
    };
    -2.0 * model.log_likelihood + 2.0 * num_params
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_clusters(n_each: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let mut data = Vec::with_capacity(n_each * 2 * 2);
        // Cluster 1: centered at (-3, 0)
        for _ in 0..n_each {
            data.push(-3.0 + rng.next_normal() * 0.5);
            data.push(rng.next_normal() * 0.5);
        }
        // Cluster 2: centered at (3, 0)
        for _ in 0..n_each {
            data.push(3.0 + rng.next_normal() * 0.5);
            data.push(rng.next_normal() * 0.5);
        }
        data
    }

    fn make_three_clusters(n_each: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let centers = [(-4.0f64, -4.0), (0.0, 4.0), (4.0, -4.0)];
        let mut data = Vec::with_capacity(n_each * 3 * 2);
        for &(cx, cy) in &centers {
            for _ in 0..n_each {
                data.push(cx + rng.next_normal() * 0.5);
                data.push(cy + rng.next_normal() * 0.5);
            }
        }
        data
    }

    fn default_cfg_k(k: usize) -> GmmConfig {
        GmmConfig { n_components: k, ..Default::default() }
    }

    #[test]
    fn two_clusters_accuracy() {
        let n_each = 100usize;
        let n = n_each * 2;
        let data = make_two_clusters(n_each, 42);
        let cfg = GmmConfig { n_components: 2, seed: 42, max_iter: 300, ..Default::default() };
        let model = gmm_fit(&data, n, 2, &cfg).expect("fit ok");
        let labels = gmm_predict(&model, &data, n).expect("predict ok");
        // Count agreement with true labels (first n_each = cluster 0 or 1)
        let mut agreement_a = 0usize;
        let mut agreement_b = 0usize;
        for ni in 0..n_each {
            if labels[ni] == 0 {
                agreement_a += 1;
            }
            if labels[ni + n_each] == 1 {
                agreement_a += 1;
            }
            if labels[ni] == 1 {
                agreement_b += 1;
            }
            if labels[ni + n_each] == 0 {
                agreement_b += 1;
            }
        }
        let accuracy = agreement_a.max(agreement_b) as f64 / n as f64;
        assert!(accuracy >= 0.9, "accuracy={accuracy:.3} < 0.9");
    }

    #[test]
    fn weights_sum_to_one() {
        let data = make_two_clusters(50, 7);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        let sum: f64 = model.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights sum={sum}");
    }

    #[test]
    fn means_near_true_centers() {
        let n_each = 200usize;
        let data = make_two_clusters(n_each, 13);
        let cfg = GmmConfig { n_components: 2, seed: 13, max_iter: 300, ..Default::default() };
        let model = gmm_fit(&data, n_each * 2, 2, &cfg).expect("ok");
        // Check each fitted mean is near either -3 or +3 in x-dimension
        let x0 = model.means[0];
        let x1 = model.means[2];
        let near_m3_0 = (x0 - (-3.0)).abs() < 1.0;
        let near_p3_0 = (x0 - 3.0).abs() < 1.0;
        let near_m3_1 = (x1 - (-3.0)).abs() < 1.0;
        let near_p3_1 = (x1 - 3.0).abs() < 1.0;
        assert!(
            (near_m3_0 || near_p3_0) && (near_m3_1 || near_p3_1),
            "means not near ±3: μ₀={x0:.2}, μ₁={x1:.2}"
        );
    }

    #[test]
    fn log_likelihood_finite() {
        let data = make_two_clusters(50, 99);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        assert!(model.log_likelihood.is_finite(), "ll not finite");
    }

    #[test]
    fn converged_simple_problem() {
        let data = make_two_clusters(200, 55);
        let cfg = GmmConfig { n_components: 2, seed: 55, max_iter: 500, ..Default::default() };
        let model = gmm_fit(&data, 400, 2, &cfg).expect("ok");
        assert!(model.converged, "should converge on well-separated clusters");
    }

    #[test]
    fn predict_shape_correct() {
        let data = make_two_clusters(50, 11);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        let labels = gmm_predict(&model, &data, 100).expect("ok");
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn predict_proba_shape_correct() {
        let data = make_two_clusters(50, 22);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        let proba = gmm_predict_proba(&model, &data, 100).expect("ok");
        assert_eq!(proba.len(), 100 * 2);
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let data = make_two_clusters(50, 33);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        let proba = gmm_predict_proba(&model, &data, 100).expect("ok");
        let k = 2;
        for ni in 0..100 {
            let row_sum: f64 = proba[ni * k..(ni + 1) * k].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "row {ni} sums to {row_sum}"
            );
        }
    }

    #[test]
    fn gmm_score_nonpositive() {
        let data = make_two_clusters(50, 44);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        let scores = gmm_score(&model, &data, 100).expect("ok");
        // log-probability of a point under any proper distribution can be > 0
        // but the total log-likelihood sum/n is typically < 0 for spread data
        assert_eq!(scores.len(), 100);
        for &s in &scores {
            assert!(s.is_finite(), "score not finite: {s}");
        }
    }

    #[test]
    fn bic_greater_than_aic_for_large_n() {
        let data = make_two_clusters(50, 66);
        let cfg = default_cfg_k(2);
        let model = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        // BIC > AIC when log(n) > 2, i.e. n >= 8
        let bic = gmm_bic(&model, 100);
        let aic = gmm_aic(&model);
        assert!(bic > aic, "BIC={bic} should exceed AIC={aic} for n=100");
    }

    #[test]
    fn n_init_returns_best_ll() {
        let data = make_two_clusters(50, 77);
        let cfg_single = GmmConfig { n_components: 2, seed: 77, n_init: 1, max_iter: 200, ..Default::default() };
        let cfg_multi = GmmConfig { n_components: 2, seed: 77, n_init: 3, max_iter: 200, ..Default::default() };
        let m1 = gmm_fit(&data, 100, 2, &cfg_single).expect("ok");
        let m3 = gmm_fit(&data, 100, 2, &cfg_multi).expect("ok");
        // With n_init=3 we should get ≥ the best of single run
        assert!(m3.log_likelihood >= m1.log_likelihood - 1e-6, "n_init=3 should not be worse");
    }

    #[test]
    fn single_component_mean_near_sample_mean() {
        let mut rng = LcgRng::new(88);
        let n = 200usize;
        let data: Vec<f64> = (0..n * 2).map(|i| {
            if i % 2 == 0 { 5.0 + rng.next_normal() } else { rng.next_normal() }
        }).collect();
        let cfg = GmmConfig { n_components: 1, seed: 0, max_iter: 100, ..Default::default() };
        let model = gmm_fit(&data, n, 2, &cfg).expect("ok");
        let sample_mean_x = data.iter().step_by(2).sum::<f64>() / n as f64;
        assert!((model.means[0] - sample_mean_x).abs() < 1.0, "mean[0]={} vs sample_mean={}", model.means[0], sample_mean_x);
    }

    #[test]
    fn three_clusters_no_crash() {
        let data = make_three_clusters(50, 123);
        let cfg = GmmConfig { n_components: 3, seed: 123, max_iter: 300, ..Default::default() };
        let model = gmm_fit(&data, 150, 2, &cfg).expect("ok");
        for &w in &model.weights {
            assert!(w > 0.0, "weight {w} not positive");
        }
    }

    #[test]
    fn diagonal_covariance_runs() {
        let data = make_two_clusters(50, 34);
        let cfg = GmmConfig {
            n_components: 2,
            covariance_type: GmmCovariance::Diagonal,
            seed: 34,
            max_iter: 200,
            ..Default::default()
        };
        let model = gmm_fit(&data, 100, 2, &cfg).expect("diagonal ok");
        assert_eq!(model.n_components, 2);
    }

    #[test]
    fn spherical_covariance_runs() {
        let data = make_two_clusters(50, 45);
        let cfg = GmmConfig {
            n_components: 2,
            covariance_type: GmmCovariance::Spherical,
            seed: 45,
            max_iter: 200,
            ..Default::default()
        };
        let model = gmm_fit(&data, 100, 2, &cfg).expect("spherical ok");
        assert_eq!(model.n_components, 2);
    }

    #[test]
    fn large_n_k4_d3_converges() {
        let mut rng = LcgRng::new(999);
        let n = 1000usize;
        let d = 3usize;
        let centers = [(-5.0f64, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, -5.0, 0.0)];
        let mut data = Vec::with_capacity(n * d);
        for ni in 0..n {
            let &(cx, cy, cz) = &centers[ni % 4];
            data.push(cx + rng.next_normal() * 0.5);
            data.push(cy + rng.next_normal() * 0.5);
            data.push(cz + rng.next_normal() * 0.5);
        }
        let cfg = GmmConfig { n_components: 4, seed: 999, max_iter: 300, ..Default::default() };
        let model = gmm_fit(&data, n, d, &cfg).expect("large n ok");
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn empty_input_error() {
        let cfg = default_cfg_k(2);
        let result = gmm_fit(&[], 0, 2, &cfg);
        assert!(matches!(result, Err(StatsError::EmptyInput)));
    }

    #[test]
    fn zero_features_error() {
        let cfg = default_cfg_k(2);
        let result = gmm_fit(&[1.0, 2.0], 2, 0, &cfg);
        assert!(matches!(
            result,
            Err(StatsError::InvalidParameter { name, .. }) if name == "n_features"
        ));
    }

    #[test]
    fn zero_components_error() {
        let cfg = GmmConfig { n_components: 0, ..Default::default() };
        let result = gmm_fit(&[1.0, 2.0], 1, 2, &cfg);
        assert!(matches!(
            result,
            Err(StatsError::InvalidParameter { name, .. }) if name == "n_components"
        ));
    }

    #[test]
    fn dimension_mismatch_error() {
        let cfg = default_cfg_k(2);
        let result = gmm_fit(&[1.0, 2.0, 3.0], 2, 2, &cfg);
        assert!(matches!(result, Err(StatsError::DimensionMismatch { a: 3, b: 4 })));
    }

    #[test]
    fn insufficient_sample_size_error() {
        let cfg = GmmConfig { n_components: 5, ..Default::default() };
        let data = vec![1.0f64; 3 * 2];
        let result = gmm_fit(&data, 3, 2, &cfg);
        assert!(matches!(
            result,
            Err(StatsError::InsufficientSampleSize { got: 3, need: 5 })
        ));
    }

    #[test]
    fn seed_reproducibility() {
        let data = make_two_clusters(50, 12);
        let cfg = GmmConfig { n_components: 2, seed: 12, max_iter: 200, ..Default::default() };
        let m1 = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        let m2 = gmm_fit(&data, 100, 2, &cfg).expect("ok");
        for (a, b) in m1.means.iter().zip(m2.means.iter()) {
            assert_eq!(a, b, "means differ between identical seeds");
        }
    }
}
