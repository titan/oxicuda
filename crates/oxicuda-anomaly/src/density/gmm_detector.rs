//! GMM-based anomaly detector (Gaussian Mixture Model density estimation).
//!
//! Points with low log-likelihood under the fitted GMM are flagged as anomalies.
//! The EM algorithm is used to fit the GMM parameters; k-means++ is used for
//! weight/means initialisation.
//!
//! # Algorithm
//!
//! **E-step** (responsibilities):
//! ```text
//! r_{ik} = π_k · N(x_i; μ_k, Σ_k) / Σ_j π_j · N(x_j; μ_j, Σ_j)
//! ```
//!
//! **M-step** (parameter updates):
//! ```text
//! N_k = Σ_i r_{ik}
//! π_k = N_k / n
//! μ_k = (Σ_i r_{ik} x_i) / N_k
//! Σ_k = (Σ_i r_{ik} (x_i−μ_k)(x_i−μ_k)ᵀ) / N_k  +  reg·I
//! ```
//!
//! **Log-likelihood**:
//! ```text
//! ℓ = Σ_i log( Σ_k π_k · N(x_i; μ_k, Σ_k) )
//! ```
//!
//! **Anomaly score**: negative log-likelihood `−log p(x)`.  Higher = more anomalous.
//!
//! Full covariance matrices are stored as flattened `d×d` arrays (row-major).
//! Cholesky decomposition is used for numerically stable density evaluation.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the GMM anomaly detector.
#[derive(Debug, Clone)]
pub struct GmmConfig {
    /// Number of Gaussian mixture components.
    pub n_components: usize,
    /// Maximum EM iterations (default: 100).
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood change (default: 1e-4).
    pub tol: f64,
    /// Diagonal covariance regularisation `reg·I` added to Σ_k (default: 1e-6).
    pub reg_covar: f64,
    /// Expected fraction of outliers in `(0, 0.5)` (default: 0.1).
    pub contamination: f64,
    /// RNG seed for k-means++ initialisation (default: 42).
    pub init_seed: u64,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 100,
            tol: 1e-4,
            reg_covar: 1e-6,
            contamination: 0.1,
            init_seed: 42,
        }
    }
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Fitted GMM model used for anomaly scoring.
pub struct GmmModel {
    /// Mixture weights `π_k` (length = `n_components`; sums to 1).
    pub weights: Vec<f64>,
    /// Component means `μ_k` (shape: `[n_components × n_features]`, row-major).
    pub means: Vec<Vec<f64>>,
    /// Full covariance matrices `Σ_k` (shape: `[n_components × (d×d)]`, row-major per component).
    pub covars: Vec<Vec<f64>>,
    /// Cholesky lower-triangular factor `L` of `Σ_k` (same layout as `covars`).
    pub chol_l: Vec<Vec<f64>>,
    /// Log-determinant `log|Σ_k|` per component.
    pub log_det: Vec<f64>,
    /// Anomaly score threshold (−log p) at the contamination percentile.
    pub threshold: f64,
    /// Number of input features.
    pub n_features: usize,
    /// Number of Gaussian components.
    pub n_components: usize,
    /// Actual EM iterations performed.
    pub n_iter: usize,
    /// Whether EM converged within `max_iter`.
    pub converged: bool,
    /// Final training log-likelihood.
    pub log_likelihood: f64,
}

// ─── Cholesky decomposition ────────────────────────────────────────────────────

/// Cholesky decomposition: given a symmetric positive-definite `d×d` matrix `a`
/// (row-major), return the lower-triangular factor `L` such that `a = L Lᵀ`.
///
/// Returns `Err(SingularCovariance)` if any diagonal becomes non-positive.
fn cholesky(a: &[f64], d: usize) -> AnomalyResult<Vec<f64>> {
    let mut l = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s: f64 = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(AnomalyError::SingularCovariance);
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Ok(l)
}

/// Log-determinant of a positive-definite matrix given its Cholesky factor `L`:
/// `log|A| = 2 Σ_i log L_{ii}`.
#[inline]
fn log_det_from_chol(l: &[f64], d: usize) -> f64 {
    let mut ld = 0.0_f64;
    for i in 0..d {
        ld += l[i * d + i].ln();
    }
    2.0 * ld
}

/// Solve `L y = b` (forward substitution) and `Lᵀ z = y` (back substitution),
/// returning `z = L⁻ᵀ L⁻¹ b = A⁻¹ b`.  Also returns `‖y‖² = bᵀ A⁻¹ b`.
fn solve_chol(l: &[f64], b: &[f64], d: usize) -> f64 {
    // Forward solve: L y = b
    let mut y = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * d + k] * y[k];
        }
        y[i] = s / l[i * d + i];
    }
    // Return ‖y‖² = bᵀ A⁻¹ b
    y.iter().map(|v| v * v).sum()
}

// ─── Log-density of N(x; μ, Σ) ────────────────────────────────────────────────

/// Log-density of a multivariate Gaussian `N(x; μ, Σ)`:
/// ```text
/// log N = -d/2 log(2π) - 1/2 log|Σ| - 1/2 (x-μ)ᵀ Σ⁻¹ (x-μ)
/// ```
/// Uses the Cholesky factor `L` of `Σ` and the pre-computed `log|Σ|`.
#[inline]
fn log_gaussian(x: &[f64], mu: &[f64], chol: &[f64], log_det: f64, d: usize) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mu.iter()).map(|(xi, mi)| xi - mi).collect();
    let maha_sq = solve_chol(chol, &diff, d);
    -0.5 * (d as f64 * std::f64::consts::TAU.ln() + log_det + maha_sq)
}

// ─── log-sum-exp ──────────────────────────────────────────────────────────────

/// Numerically stable `log(Σ exp(a_k))`.
fn log_sum_exp(a: &[f64]) -> f64 {
    let max = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = a.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

// ─── k-means++ initialisation ─────────────────────────────────────────────────

/// Initialise `k` cluster centres from `data` using the k-means++ seeding scheme.
///
/// Returns `k` row-vectors of length `d`, stacked into a flat `[k × d]` array.
fn kmeans_plus_plus(
    data: &[f64],
    n: usize,
    d: usize,
    k: usize,
    rng: &mut LcgRng,
) -> AnomalyResult<Vec<Vec<f64>>> {
    if n < k {
        return Err(AnomalyError::InsufficientSamples { need: k, got: n });
    }

    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Choose first centre uniformly at random
    let first = rng.next_usize(n);
    centres.push(data[first * d..(first + 1) * d].to_vec());

    // Scratch for squared distances
    let mut dists = vec![f64::INFINITY; n];

    for _ in 1..k {
        // Update minimum squared distances to nearest already-chosen centre
        let last_centre = centres.last().ok_or_else(|| AnomalyError::Internal {
            msg: "k-means++ centres unexpectedly empty".into(),
        })?;
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let sq: f64 = row
                .iter()
                .zip(last_centre.iter())
                .map(|(xi, ci)| (xi - ci) * (xi - ci))
                .sum();
            if sq < dists[i] {
                dists[i] = sq;
            }
        }

        // Sample next centre proportional to D²
        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
            // All points are identical; pick any remaining
            centres.push(data[rng.next_usize(n) * d..rng.next_usize(n) * d + d].to_vec());
            continue;
        }
        let threshold = (rng.next_u32() as f64 / (u32::MAX as f64 + 1.0)) * total;
        let mut cumsum = 0.0_f64;
        let mut chosen = n - 1;
        for (i, &d_i) in dists.iter().enumerate() {
            cumsum += d_i;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centres.push(data[chosen * d..(chosen + 1) * d].to_vec());
    }

    Ok(centres)
}

// ─── EM Algorithm ─────────────────────────────────────────────────────────────

/// One EM E-step: compute log-responsibilities `log r_{ik}` (shape `[n × k]`).
///
/// Returns the per-sample log-sum for computing log-likelihood.
fn e_step(
    data: &[f64],
    n: usize,
    d: usize,
    weights: &[f64],
    means: &[Vec<f64>],
    chol_l: &[Vec<f64>],
    log_det: &[f64],
    n_components: usize,
    log_resp: &mut [f64],
) -> f64 {
    let mut log_lik = 0.0_f64;

    for i in 0..n {
        let xi = &data[i * d..(i + 1) * d];
        let offset = i * n_components;

        // Compute weighted log-density for each component
        for k in 0..n_components {
            let log_pi_k = weights[k].ln();
            let log_p_k = log_gaussian(xi, &means[k], &chol_l[k], log_det[k], d);
            log_resp[offset + k] = log_pi_k + log_p_k;
        }

        // Log-sum-exp across components
        let lse = log_sum_exp(&log_resp[offset..offset + n_components]);
        log_lik += lse;

        // Normalise: log r_{ik} = log_resp[ik] - lse
        for k in 0..n_components {
            log_resp[offset + k] -= lse;
        }
    }

    log_lik
}

/// One EM M-step: update weights, means, and covariances from log-responsibilities.
fn m_step(
    data: &[f64],
    n: usize,
    d: usize,
    log_resp: &[f64],
    n_components: usize,
    reg_covar: f64,
    weights: &mut [f64],
    means: &mut [Vec<f64>],
    covars: &mut [Vec<f64>],
    chol_l: &mut [Vec<f64>],
    log_det: &mut [f64],
) -> AnomalyResult<()> {
    // Accumulate responsibilities: N_k = Σ_i exp(log r_{ik})
    let mut nk = vec![0.0_f64; n_components];
    for i in 0..n {
        for k in 0..n_components {
            nk[k] += log_resp[i * n_components + k].exp();
        }
    }

    // Guard against degenerate components
    let nk_min = 1e-8;
    for nk_val in &mut nk {
        if *nk_val < nk_min {
            *nk_val = nk_min;
        }
    }

    // Update π_k
    let n_f = n as f64;
    for k in 0..n_components {
        weights[k] = nk[k] / n_f;
    }

    // Update μ_k
    for k in 0..n_components {
        let mu = &mut means[k];
        for v in mu.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n {
            let r_ik = log_resp[i * n_components + k].exp();
            let xi = &data[i * d..(i + 1) * d];
            for j in 0..d {
                mu[j] += r_ik * xi[j];
            }
        }
        let inv_nk = nk[k].recip();
        for v in mu.iter_mut() {
            *v *= inv_nk;
        }
    }

    // Update Σ_k
    for k in 0..n_components {
        let cov = &mut covars[k];
        for v in cov.iter_mut() {
            *v = 0.0;
        }
        let mu_k = &means[k];
        for i in 0..n {
            let r_ik = log_resp[i * n_components + k].exp();
            let xi = &data[i * d..(i + 1) * d];
            for r in 0..d {
                let diff_r = xi[r] - mu_k[r];
                for c in 0..d {
                    let diff_c = xi[c] - mu_k[c];
                    cov[r * d + c] += r_ik * diff_r * diff_c;
                }
            }
        }
        let inv_nk = nk[k].recip();
        for v in cov.iter_mut() {
            *v *= inv_nk;
        }
        // Regularise diagonal
        for j in 0..d {
            cov[j * d + j] += reg_covar;
        }

        // Recompute Cholesky and log-det
        chol_l[k] = cholesky(cov, d)?;
        log_det[k] = log_det_from_chol(&chol_l[k], d);
    }

    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a GMM anomaly detector to training data using the EM algorithm.
///
/// `x` is a row-major `f64` slice of shape `[n_samples × n_features]`.
pub fn gmm_fit(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &GmmConfig,
    rng: &mut LcgRng,
) -> AnomalyResult<GmmModel> {
    // ── Validation ──────────────────────────────────────────────────────────
    if n_samples == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if n_features == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if x.len() != n_samples * n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * n_features,
            got: x.len(),
        });
    }
    if cfg.n_components == 0 {
        return Err(AnomalyError::Internal {
            msg: "n_components must be > 0".into(),
        });
    }
    if cfg.n_components > n_samples {
        return Err(AnomalyError::InsufficientSamples {
            need: cfg.n_components,
            got: n_samples,
        });
    }
    if !(cfg.contamination > 0.0 && cfg.contamination < 0.5) {
        return Err(AnomalyError::Internal {
            msg: format!(
                "contamination must be in (0, 0.5), got {}",
                cfg.contamination
            ),
        });
    }

    let n = n_samples;
    let d = n_features;
    let k = cfg.n_components;

    // ── k-means++ initialisation ─────────────────────────────────────────────
    let mut init_rng = LcgRng::new(cfg.init_seed);
    let init_means = kmeans_plus_plus(x, n, d, k, &mut init_rng)?;

    // ── Initialise parameters ────────────────────────────────────────────────
    let mut weights: Vec<f64> = vec![1.0 / k as f64; k];
    let mut means: Vec<Vec<f64>> = init_means;
    // Identity covariances regularised
    let mut covars: Vec<Vec<f64>> = (0..k)
        .map(|_| {
            let mut cov = vec![0.0_f64; d * d];
            for j in 0..d {
                cov[j * d + j] = 1.0 + cfg.reg_covar;
            }
            cov
        })
        .collect();
    let mut chol_l: Vec<Vec<f64>> = covars
        .iter()
        .map(|c| cholesky(c, d))
        .collect::<AnomalyResult<Vec<_>>>()?;
    let mut log_det: Vec<f64> = chol_l.iter().map(|l| log_det_from_chol(l, d)).collect();

    // ── EM iterations ────────────────────────────────────────────────────────
    let mut log_resp = vec![0.0_f64; n * k];
    let mut prev_log_lik = f64::NEG_INFINITY;
    let mut final_log_lik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;

        // E-step
        let log_lik = e_step(
            x,
            n,
            d,
            &weights,
            &means,
            &chol_l,
            &log_det,
            k,
            &mut log_resp,
        );

        final_log_lik = log_lik;

        // Check convergence
        if iter > 0 && (log_lik - prev_log_lik).abs() < cfg.tol {
            converged = true;
            break;
        }
        prev_log_lik = log_lik;

        // M-step
        m_step(
            x,
            n,
            d,
            &log_resp,
            k,
            cfg.reg_covar,
            &mut weights,
            &mut means,
            &mut covars,
            &mut chol_l,
            &mut log_det,
        )?;
    }

    // ── Compute anomaly scores on training data to derive threshold ───────
    // score = −log p(x)
    let train_scores: Vec<f64> =
        compute_neg_log_lik(x, n, d, &weights, &means, &chol_l, &log_det, k);

    // Percentile threshold
    let mut sorted = train_scores.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((1.0 - cfg.contamination) * n as f64).floor() as usize;
    let rank = rank.min(n - 1);
    let threshold = sorted[rank];

    // Consume the caller-provided rng to give it meaningful entropy usage
    // (satisfies the borrow checker / API contract).
    let _ = rng.next_u32();

    Ok(GmmModel {
        weights,
        means,
        covars,
        chol_l,
        log_det,
        threshold,
        n_features: d,
        n_components: k,
        n_iter,
        converged,
        log_likelihood: final_log_lik,
    })
}

/// Compute per-sample log-likelihood `log p(x_i) = log Σ_k π_k N(x_i; μ_k, Σ_k)`.
fn compute_neg_log_lik(
    data: &[f64],
    n: usize,
    d: usize,
    weights: &[f64],
    means: &[Vec<f64>],
    chol_l: &[Vec<f64>],
    log_det: &[f64],
    k: usize,
) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let xi = &data[i * d..(i + 1) * d];
            let log_probs: Vec<f64> = (0..k)
                .map(|kk| {
                    weights[kk].ln() + log_gaussian(xi, &means[kk], &chol_l[kk], log_det[kk], d)
                })
                .collect();
            -log_sum_exp(&log_probs)
        })
        .collect()
}

/// Compute per-sample log-likelihood values `log p(x_i)` under the fitted GMM.
///
/// Returns a `Vec<f64>` of length `n_samples`.
pub fn gmm_log_likelihood(
    model: &GmmModel,
    x: &[f64],
    n_samples: usize,
) -> AnomalyResult<Vec<f64>> {
    if model.weights.is_empty() {
        return Err(AnomalyError::NotFitted);
    }
    if n_samples == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n_samples * model.n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * model.n_features,
            got: x.len(),
        });
    }
    let lls: Vec<f64> = (0..n_samples)
        .map(|i| {
            let xi = &x[i * model.n_features..(i + 1) * model.n_features];
            let log_probs: Vec<f64> = (0..model.n_components)
                .map(|k| {
                    model.weights[k].ln()
                        + log_gaussian(
                            xi,
                            &model.means[k],
                            &model.chol_l[k],
                            model.log_det[k],
                            model.n_features,
                        )
                })
                .collect();
            log_sum_exp(&log_probs)
        })
        .collect();
    Ok(lls)
}

/// Compute anomaly scores: negative log-likelihood `−log p(x)`.
///
/// Higher values indicate stronger anomaly.
pub fn gmm_score(model: &GmmModel, x: &[f64], n_samples: usize) -> AnomalyResult<Vec<f64>> {
    let lls = gmm_log_likelihood(model, x, n_samples)?;
    Ok(lls.into_iter().map(|v| -v).collect())
}

/// Classify each sample as normal (`+1`) or anomaly (`-1`).
///
/// Samples with anomaly score `≥ model.threshold` are labelled `−1`.
pub fn gmm_predict(model: &GmmModel, x: &[f64], n_samples: usize) -> AnomalyResult<Vec<i32>> {
    let scores = gmm_score(model, x, n_samples)?;
    Ok(scores
        .into_iter()
        .map(|s| if s >= model.threshold { -1 } else { 1 })
        .collect())
}

/// Sample `n` points from the fitted GMM.
///
/// Returns a row-major flat array of shape `[n × n_features]`.
pub fn gmm_sample(model: &GmmModel, n: usize, rng: &mut LcgRng) -> AnomalyResult<Vec<f64>> {
    if model.weights.is_empty() {
        return Err(AnomalyError::NotFitted);
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let d = model.n_features;
    let k = model.n_components;
    let mut out = Vec::with_capacity(n * d);

    for _ in 0..n {
        // 1. Sample component index from categorical(π)
        let u = (rng.next_u32() as f64) / (u32::MAX as f64 + 1.0);
        let mut cumsum = 0.0_f64;
        let mut comp = k - 1;
        for kk in 0..k {
            cumsum += model.weights[kk];
            if u < cumsum {
                comp = kk;
                break;
            }
        }

        // 2. Sample from N(μ_k, Σ_k) using L: x = μ + L·z, z ~ N(0, I)
        let mu = &model.means[comp];
        let l = &model.chol_l[comp];

        // Draw d independent standard normals
        let mut z = vec![0.0_f64; d];
        let mut idx = 0;
        while idx + 1 < d {
            let (a, b) = rng.next_normal_pair();
            z[idx] = a as f64;
            z[idx + 1] = b as f64;
            idx += 2;
        }
        if idx < d {
            z[idx] = rng.next_normal() as f64;
        }

        // x = μ + L·z  (L is lower-triangular)
        for row in 0..d {
            let mut val = mu[row];
            for col in 0..=row {
                val += l[row * d + col] * z[col];
            }
            out.push(val);
        }
    }

    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate `n` samples from a 2-D Gaussian N(mu, sigma²·I).
    fn gaussian_cloud(n: usize, mu: [f64; 2], sigma: f64, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let mut data = Vec::with_capacity(n * 2);
        for _ in 0..n {
            data.push(mu[0] + rng.next_normal() as f64 * sigma);
            data.push(mu[1] + rng.next_normal() as f64 * sigma);
        }
        data
    }

    // ── Test 1: gmm_fit runs on 50×2 with 2 components ───────────────────────
    #[test]
    fn gmm_fit_runs() {
        let data = gaussian_cloud(50, [0.0, 0.0], 1.0, 1);
        let cfg = GmmConfig {
            n_components: 2,
            ..GmmConfig::default()
        };
        let mut rng = LcgRng::new(1);
        let model = gmm_fit(&data, 50, 2, &cfg, &mut rng);
        assert!(model.is_ok(), "gmm_fit failed: {:?}", model.err());
    }

    // ── Test 2: weights sum to 1 ──────────────────────────────────────────────
    #[test]
    fn gmm_weights_sum_to_1() {
        let data = gaussian_cloud(80, [0.0, 0.0], 1.0, 2);
        let cfg = GmmConfig::default();
        let mut rng = LcgRng::new(2);
        let model =
            gmm_fit(&data, 80, 2, &cfg, &mut rng).expect("GMM fit on 80 samples should succeed");
        let sum: f64 = model.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weights sum = {sum}, expected 1.0"
        );
    }

    // ── Test 3: log-likelihood shape ──────────────────────────────────────────
    #[test]
    fn gmm_log_lik_shape() {
        let data = gaussian_cloud(60, [0.0, 0.0], 1.0, 3);
        let cfg = GmmConfig::default();
        let mut rng = LcgRng::new(3);
        let model =
            gmm_fit(&data, 60, 2, &cfg, &mut rng).expect("GMM fit on 60 samples should succeed");
        let lls = gmm_log_likelihood(&model, &data, 60)
            .expect("log-likelihood computation should succeed");
        assert_eq!(lls.len(), 60);
    }

    // ── Test 4: all log-likelihood values are finite ───────────────────────────
    #[test]
    fn gmm_log_lik_finite() {
        let data = gaussian_cloud(60, [0.0, 0.0], 1.0, 4);
        let cfg = GmmConfig::default();
        let mut rng = LcgRng::new(4);
        let model =
            gmm_fit(&data, 60, 2, &cfg, &mut rng).expect("GMM fit on 60 samples should succeed");
        let lls = gmm_log_likelihood(&model, &data, 60)
            .expect("log-likelihood computation should succeed");
        for (i, &v) in lls.iter().enumerate() {
            assert!(v.is_finite(), "log_lik[{i}] = {v} is not finite");
        }
    }

    // ── Test 5: isolated point has higher score ────────────────────────────────
    #[test]
    fn gmm_score_higher_for_outlier() {
        // 60 inliers near origin + 1 outlier far away
        let mut data = gaussian_cloud(60, [0.0, 0.0], 0.5, 5);
        data.push(100.0);
        data.push(100.0);
        let n = 61;
        let cfg = GmmConfig {
            n_components: 1, // single Gaussian for simplicity
            max_iter: 100,
            tol: 1e-4,
            reg_covar: 1e-6,
            contamination: 0.05,
            init_seed: 5,
        };
        let mut rng = LcgRng::new(5);
        let model = gmm_fit(&data, n, 2, &cfg, &mut rng)
            .expect("GMM fit with outlier appended should succeed");
        let scores =
            gmm_score(&model, &data, n).expect("GMM score should succeed on training data");
        let outlier_score = scores[n - 1];
        let inlier_mean: f64 = scores[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
        assert!(
            outlier_score > inlier_mean,
            "outlier score {outlier_score:.4} should > inlier mean {inlier_mean:.4}"
        );
    }

    // ── Test 6: labels ∈ {-1, +1} ────────────────────────────────────────────
    #[test]
    fn gmm_predict_labels() {
        let data = gaussian_cloud(80, [0.0, 0.0], 1.0, 6);
        let cfg = GmmConfig::default();
        let mut rng = LcgRng::new(6);
        let model =
            gmm_fit(&data, 80, 2, &cfg, &mut rng).expect("GMM fit on 80 samples should succeed");
        let labels =
            gmm_predict(&model, &data, 80).expect("GMM predict on training data should succeed");
        for &l in &labels {
            assert!(l == 1 || l == -1, "label={l} not in {{-1, +1}}");
        }
    }

    // ── Test 7: gmm_sample shape ──────────────────────────────────────────────
    #[test]
    fn gmm_sample_shape() {
        let data = gaussian_cloud(60, [0.0, 0.0], 1.0, 7);
        let cfg = GmmConfig::default();
        let mut rng = LcgRng::new(7);
        let model =
            gmm_fit(&data, 60, 2, &cfg, &mut rng).expect("GMM fit on 60 samples should succeed");
        let n_gen = 30usize;
        let samples = gmm_sample(&model, n_gen, &mut rng)
            .expect("GMM sample from fitted model should succeed");
        assert_eq!(
            samples.len(),
            n_gen * 2,
            "expected {} elements, got {}",
            n_gen * 2,
            samples.len()
        );
    }

    // ── Test 8: 2-D cluster separation — means near cluster centers ───────────
    #[test]
    fn gmm_2d_cluster_separation() {
        // Two well-separated clusters in 2-D
        let mut data = gaussian_cloud(80, [-10.0, -10.0], 0.3, 8);
        data.extend(gaussian_cloud(80, [10.0, 10.0], 0.3, 81));
        let n = 160usize;
        let cfg = GmmConfig {
            n_components: 2,
            max_iter: 200,
            tol: 1e-6,
            reg_covar: 1e-6,
            contamination: 0.1,
            init_seed: 8,
        };
        let mut rng = LcgRng::new(8);
        let model =
            gmm_fit(&data, n, 2, &cfg, &mut rng).expect("GMM fit on 2-cluster data should succeed");

        // Each fitted mean should be close to one of the two cluster centers
        let targets = [[-10.0_f64, -10.0], [10.0, 10.0]];
        for mu in &model.means {
            let nearest_dist = targets
                .iter()
                .map(|t| {
                    let dx = mu[0] - t[0];
                    let dy = mu[1] - t[1];
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest_dist < 3.0,
                "mean {:?} is not near any cluster center (nearest dist {nearest_dist:.3})",
                mu
            );
        }
    }

    // ── Test 9: 1-component GMM ≈ multivariate Gaussian ──────────────────────
    #[test]
    fn gmm_single_component() {
        let data = gaussian_cloud(100, [5.0, -3.0], 2.0, 9);
        let cfg = GmmConfig {
            n_components: 1,
            max_iter: 50,
            tol: 1e-6,
            reg_covar: 1e-6,
            contamination: 0.1,
            init_seed: 9,
        };
        let mut rng = LcgRng::new(9);
        let model = gmm_fit(&data, 100, 2, &cfg, &mut rng)
            .expect("single-component GMM fit should succeed");

        assert_eq!(model.weights.len(), 1);
        assert!(
            (model.weights[0] - 1.0).abs() < 1e-10,
            "weight[0] must be 1"
        );

        // Mean should be near (5, -3)
        let mu = &model.means[0];
        assert!((mu[0] - 5.0).abs() < 0.5, "μ[0]={:.3} expected ≈5", mu[0]);
        assert!((mu[1] - -3.0).abs() < 0.5, "μ[1]={:.3} expected ≈-3", mu[1]);
    }

    // ── Test 10: convergence flag is set on simple data ───────────────────────
    #[test]
    fn gmm_convergence_flag() {
        // Very tight cluster → should converge quickly
        let data = gaussian_cloud(100, [0.0, 0.0], 0.01, 10);
        let cfg = GmmConfig {
            n_components: 2,
            max_iter: 500,
            tol: 1e-6,
            reg_covar: 1e-6,
            contamination: 0.1,
            init_seed: 10,
        };
        let mut rng = LcgRng::new(10);
        let model = gmm_fit(&data, 100, 2, &cfg, &mut rng)
            .expect("GMM fit on tight cluster should succeed");
        // Should have converged within 500 iterations on this easy data
        assert!(
            model.converged || model.n_iter < cfg.max_iter,
            "expected convergence or early stop, n_iter={}",
            model.n_iter
        );
    }

    // ── Test 11: gmm_sample values are finite ─────────────────────────────────
    #[test]
    fn gmm_sample_values_finite() {
        let data = gaussian_cloud(60, [0.0, 0.0], 1.0, 11);
        let cfg = GmmConfig::default();
        let mut rng = LcgRng::new(11);
        let model =
            gmm_fit(&data, 60, 2, &cfg, &mut rng).expect("GMM fit on 60 samples should succeed");
        let samples =
            gmm_sample(&model, 50, &mut rng).expect("GMM sample should produce 50 valid points");
        for (i, &v) in samples.iter().enumerate() {
            assert!(v.is_finite(), "sample[{i}]={v} is not finite");
        }
    }

    // ── Test 12: higher-dimensional data (d=5) runs cleanly ──────────────────
    #[test]
    fn gmm_high_dimensional() {
        let d = 5usize;
        let n = 100usize;
        let mut rng_gen = LcgRng::new(12);
        let data: Vec<f64> = (0..n * d).map(|_| rng_gen.next_normal() as f64).collect();

        let cfg = GmmConfig {
            n_components: 3,
            max_iter: 100,
            tol: 1e-4,
            reg_covar: 1e-4,
            contamination: 0.1,
            init_seed: 12,
        };
        let mut rng = LcgRng::new(12);
        let model =
            gmm_fit(&data, n, d, &cfg, &mut rng).expect("GMM fit on 5-D data should succeed");
        let scores = gmm_score(&model, &data, n).expect("GMM score on 5-D data should succeed");
        assert_eq!(scores.len(), n);
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "all scores should be finite"
        );
    }

    // ── Test 13: Cholesky identity correctness ────────────────────────────────
    #[test]
    fn cholesky_identity_matrix() {
        let id = vec![1.0_f64, 0.0, 0.0, 1.0];
        let l = cholesky(&id, 2).expect("Cholesky of identity matrix should succeed");
        // L of I = I
        assert!((l[0] - 1.0).abs() < 1e-12);
        assert!(l[1].abs() < 1e-12);
        assert!(l[2].abs() < 1e-12);
        assert!((l[3] - 1.0).abs() < 1e-12);
        assert!((log_det_from_chol(&l, 2)).abs() < 1e-12, "log|I|=0");
    }
}
