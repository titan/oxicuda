//! Soft-Boundary Deep SVDD (Ruff et al. 2018, Section 3.2).
//!
//! Extends the hard-boundary DeepSVDD by introducing a learnable radius `R` and
//! per-sample slack variables `ξ_i ≥ 0`, allowing a fraction `ν ∈ (0,1]` of
//! training points to lie outside the hypersphere.
//!
//! **Objective** (per epoch):
//! ```text
//! min_{W, R, c}  R²  +  (1/(ν·n)) Σ_i  max(0, ||φ(x_i; W) − c||² − R²)
//! ```
//!
//! Key differences from hard DeepSVDD:
//! * `R` is not fixed — updated each epoch as the `(1−ν)` quantile of squared
//!   distances (outlier quantile clamping).
//! * Gradients w.r.t. `W` come **only** from active constraints
//!   (`||φ(x_i)−c||² > R²`), scaled by `2/(ν·n)`.
//! * Center `c` is initialised once (warm-up forward pass average) and then
//!   kept fixed — identical to hard DeepSVDD.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Xavier initialisation ────────────────────────────────────────────────────

fn xavier_init_f64(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── MLP forward (no bias in final layer) ────────────────────────────────────

/// Per-layer activation caches: pre-activations and post-activations.
type LayerCache = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Lightweight 3-layer MLP encoder with ReLU hidden activations.
///
/// Layer dimensions: `input_dim → hidden1 → hidden2 → latent_dim`.
/// The final (latent) layer has **no bias** to prevent hypersphere collapse.
struct SoftSvddMlp {
    /// Weight matrices stored row-major: `w[l]` has shape `[out, in]`.
    w: [Vec<f64>; 3],
    /// Bias vectors; `b[2]` is always zero (never updated).
    b: [Vec<f64>; 3],
    /// Layer output sizes: `[hidden1, hidden2, latent_dim]`.
    out_dims: [usize; 3],
    /// Input dimension.
    in_dim: usize,
}

impl SoftSvddMlp {
    fn new(
        in_dim: usize,
        h1: usize,
        h2: usize,
        latent: usize,
        rng: &mut LcgRng,
    ) -> AnomalyResult<Self> {
        for (name, d) in [
            ("in_dim", in_dim),
            ("h1", h1),
            ("h2", h2),
            ("latent", latent),
        ] {
            if d == 0 {
                return Err(AnomalyError::InvalidLayerDims {
                    msg: format!("{name} must be > 0"),
                });
            }
        }
        let w = [
            xavier_init_f64(in_dim, h1, rng),
            xavier_init_f64(h1, h2, rng),
            xavier_init_f64(h2, latent, rng),
        ];
        let b = [
            vec![0.0_f64; h1],
            vec![0.0_f64; h2],
            vec![0.0_f64; latent], // no bias — permanently zero
        ];
        Ok(Self {
            w,
            b,
            out_dims: [h1, h2, latent],
            in_dim,
        })
    }

    /// Forward pass for a single sample.  Returns `(pre_activations, activations)` per layer
    /// where `pre_activations[l]` is the pre-activation (z) and `activations[l]` is
    /// the post-activation (a): ReLU for layers 0,1; linear for layer 2.
    fn forward_with_cache(&self, x: &[f64]) -> AnomalyResult<LayerCache> {
        if x.len() != self.in_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let n_layers = 3;
        let mut pre = Vec::with_capacity(n_layers);
        let mut post = Vec::with_capacity(n_layers);
        let mut act: Vec<f64> = x.to_vec();

        for l in 0..n_layers {
            let fan_in = if l == 0 {
                self.in_dim
            } else {
                self.out_dims[l - 1]
            };
            let fan_out = self.out_dims[l];
            // Compute pre-activations z[o] = bias + Σ_i w[o,i] * act[i]
            let z: Vec<f64> = (0..fan_out)
                .map(|o| {
                    let row_start = o * fan_in;
                    self.b[l][o]
                        + self.w[l][row_start..row_start + fan_in]
                            .iter()
                            .zip(act.iter())
                            .map(|(wi, ai)| wi * ai)
                            .sum::<f64>()
                })
                .collect();
            let a: Vec<f64> = if l < 2 {
                z.iter().map(|&v| v.max(0.0)).collect()
            } else {
                z.clone()
            };
            pre.push(z);
            post.push(a.clone());
            act = a;
        }
        Ok((pre, post))
    }

    /// Forward pass returning only the output representation.
    fn forward(&self, x: &[f64]) -> AnomalyResult<Vec<f64>> {
        let (_, post) = self.forward_with_cache(x)?;
        Ok(post[2].clone())
    }

    /// Backward pass for active-constraint samples (squared distance > R²).
    ///
    /// Gradient of `(1/(ν·n)) ||φ(x) − c||²` w.r.t. `W[l]`.
    ///
    /// Chain rule (output layer is linear):
    /// `δ_out = 2 * (φ(x) − c)` scaled by `1/(ν·n)`; back-propagated through
    /// ReLU layers using the cached pre-activations.
    fn backward_update(
        &mut self,
        x: &[f64],
        center: &[f64],
        scale: f64, // = 1/(ν·n)
        lr: f64,
    ) -> AnomalyResult<()> {
        let (pre, post) = self.forward_with_cache(x)?;
        let latent = &post[2];
        let n_lat = self.out_dims[2];

        // δ for layer 2 (linear, no ReLU): ∂L/∂z = 2 * scale * (φ - c)
        let mut delta: Vec<f64> = (0..n_lat)
            .map(|j| 2.0 * scale * (latent[j] - center[j]))
            .collect();

        // Back-propagate through layers 2, 1, 0
        for l in (0..3).rev() {
            let fan_in = if l == 0 {
                self.in_dim
            } else {
                self.out_dims[l - 1]
            };
            let fan_out = self.out_dims[l];
            let input_act: &[f64] = if l == 0 { x } else { &post[l - 1] };

            // Weight gradient and update using enumerate to satisfy clippy
            for (o, &d_o) in delta.iter().enumerate().take(fan_out) {
                let row_start = o * fan_in;
                for (k, &ai) in input_act.iter().enumerate().take(fan_in) {
                    self.w[l][row_start + k] -= lr * d_o * ai;
                }
                // Bias update for layers 0 and 1 only (layer 2 bias is always 0)
                if l < 2 {
                    self.b[l][o] -= lr * d_o;
                }
            }

            if l == 0 {
                break;
            }

            // Propagate delta to previous layer through ReLU
            let prev_out = self.out_dims[l - 1];
            let mut prev_delta = vec![0.0_f64; prev_out];
            for (i, pd) in prev_delta.iter_mut().enumerate().take(prev_out) {
                let acc: f64 = delta
                    .iter()
                    .enumerate()
                    .take(fan_out)
                    .map(|(o, &d_o)| self.w[l][o * fan_in + i] * d_o)
                    .sum();
                // ReLU derivative: 1 if pre-activation > 0
                *pd = if pre[l - 1][i] > 0.0 { acc } else { 0.0 };
            }
            delta = prev_delta;
        }
        Ok(())
    }
}

// ─── Public API types ─────────────────────────────────────────────────────────

/// Configuration for Soft-Boundary Deep SVDD.
#[derive(Debug, Clone)]
pub struct SoftSvddConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// First hidden layer width.
    pub hidden1: usize,
    /// Second hidden layer width.
    pub hidden2: usize,
    /// Latent (output) representation dimension.
    pub latent_dim: usize,
    /// Outlier fraction `ν ∈ (0, 1]`.  Smaller ν → tighter sphere.
    pub nu: f64,
    /// SGD learning rate.
    pub lr: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
}

impl SoftSvddConfig {
    /// Validate parameters.
    pub fn validate(&self) -> AnomalyResult<()> {
        for (name, d) in [
            ("input_dim", self.input_dim),
            ("hidden1", self.hidden1),
            ("hidden2", self.hidden2),
            ("latent_dim", self.latent_dim),
        ] {
            if d == 0 {
                return Err(AnomalyError::InvalidLayerDims {
                    msg: format!("{name} must be > 0"),
                });
            }
        }
        if !(self.nu > 0.0 && self.nu <= 1.0) {
            return Err(AnomalyError::InvalidNu { nu: self.nu as f32 });
        }
        if self.n_epochs == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_epochs must be > 0".into(),
            });
        }
        Ok(())
    }
}

/// Fitted Soft-Boundary DeepSVDD model.
///
/// Stores the three weight matrices, two hidden biases (no final bias),
/// the fixed hypersphere center `c`, and the learned radius `R`.
#[derive(Debug, Clone)]
pub struct SoftSvddFit {
    /// Layer-0 weight matrix `[hidden1 × input_dim]`.
    pub w1: Vec<f64>,
    /// Layer-0 bias `[hidden1]`.
    pub b1: Vec<f64>,
    /// Layer-1 weight matrix `[hidden2 × hidden1]`.
    pub w2: Vec<f64>,
    /// Layer-1 bias `[hidden2]`.
    pub b2: Vec<f64>,
    /// Layer-2 weight matrix `[latent_dim × hidden2]` (no bias).
    pub w3: Vec<f64>,
    /// Fixed hypersphere center.
    pub center: Vec<f64>,
    /// Learned soft radius (updated each epoch).
    pub radius: f64,
    /// Config dims for inference.
    pub input_dim: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub latent_dim: usize,
}

impl SoftSvddFit {
    /// Reconstruct an `SoftSvddMlp` from the stored weights for inference.
    fn to_mlp(&self, rng: &mut LcgRng) -> AnomalyResult<SoftSvddMlp> {
        let mut mlp = SoftSvddMlp::new(
            self.input_dim,
            self.hidden1,
            self.hidden2,
            self.latent_dim,
            rng,
        )?;
        mlp.w[0].clone_from(&self.w1);
        mlp.b[0].clone_from(&self.b1);
        mlp.w[1].clone_from(&self.w2);
        mlp.b[1].clone_from(&self.b2);
        mlp.w[2].clone_from(&self.w3);
        Ok(mlp)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the `(1−ν)` quantile of `squared_dists` (in-place sort copy).
/// Returns the element at `floor((1−ν) * n)` (0-indexed, clamped).
fn radius_quantile(squared_dists: &[f64], nu: f64) -> f64 {
    if squared_dists.is_empty() {
        return 0.0;
    }
    let mut sorted = squared_dists.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let idx = ((1.0 - nu) * n as f64).floor() as usize;
    let idx = idx.min(n - 1);
    sorted[idx].max(0.0)
}

/// Squared Euclidean distance between two equal-length slices.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum()
}

// ─── Training ─────────────────────────────────────────────────────────────────

/// Fit a Soft-Boundary Deep SVDD model on training data.
///
/// # Arguments
/// * `x` — row-major flat matrix `[n × input_dim]`.
/// * `n` — number of training samples.
/// * `cfg` — hyperparameter configuration.
/// * `seed` — RNG seed for reproducibility.
pub fn soft_svdd_fit(
    x: &[f64],
    n: usize,
    cfg: &SoftSvddConfig,
    seed: u64,
) -> AnomalyResult<SoftSvddFit> {
    cfg.validate()?;
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    let d = cfg.input_dim;
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let mut rng = LcgRng::new(seed);
    let mut mlp = SoftSvddMlp::new(d, cfg.hidden1, cfg.hidden2, cfg.latent_dim, &mut rng)?;

    // ── Warm-up: compute center c as mean of encoder outputs ──────────────────
    let mut center = vec![0.0_f64; cfg.latent_dim];
    for i in 0..n {
        let sample = &x[i * d..(i + 1) * d];
        let rep = mlp.forward(sample)?;
        for (cj, rj) in center.iter_mut().zip(rep.iter()) {
            *cj += rj;
        }
    }
    let inv_n = 1.0 / n as f64;
    for cj in &mut center {
        *cj *= inv_n;
        if cj.abs() < 0.01 {
            *cj = 0.01;
        }
    }

    // ── Initial radius estimate ────────────────────────────────────────────────
    let mut sq_dists: Vec<f64> = (0..n)
        .map(|i| {
            let rep = mlp
                .forward(&x[i * d..(i + 1) * d])
                .unwrap_or_else(|_| center.clone());
            sq_dist(&rep, &center)
        })
        .collect();
    let mut radius_sq = radius_quantile(&sq_dists, cfg.nu);

    // ── Training loop ─────────────────────────────────────────────────────────
    let scale = 1.0 / (cfg.nu * n as f64);
    for _epoch in 0..cfg.n_epochs {
        // Recompute squared distances
        for i in 0..n {
            let rep = mlp.forward(&x[i * d..(i + 1) * d])?;
            sq_dists[i] = sq_dist(&rep, &center);
        }

        // Update radius: (1−ν) quantile of squared distances
        radius_sq = radius_quantile(&sq_dists, cfg.nu);

        // Gradient step for active constraints only
        for i in 0..n {
            if sq_dists[i] > radius_sq {
                mlp.backward_update(&x[i * d..(i + 1) * d], &center, scale, cfg.lr)?;
            }
        }
    }

    // Extract weights
    Ok(SoftSvddFit {
        w1: mlp.w[0].clone(),
        b1: mlp.b[0].clone(),
        w2: mlp.w[1].clone(),
        b2: mlp.b[1].clone(),
        w3: mlp.w[2].clone(),
        center,
        radius: radius_sq.sqrt(),
        input_dim: cfg.input_dim,
        hidden1: cfg.hidden1,
        hidden2: cfg.hidden2,
        latent_dim: cfg.latent_dim,
    })
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

/// Compute anomaly scores for `n` test samples using a fitted Soft-SVDD model.
///
/// Score = `||φ(x) − c||² − R²`:
/// * Negative → inside sphere → inlier.
/// * Positive → outside sphere → anomaly.
pub fn soft_svdd_score(fit: &SoftSvddFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    let d = fit.input_dim;
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }
    let mut dummy_rng = LcgRng::new(0);
    let mlp = fit.to_mlp(&mut dummy_rng)?;
    let r_sq = fit.radius * fit.radius;

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let rep = mlp.forward(&x[i * d..(i + 1) * d])?;
        let dist_sq = sq_dist(&rep, &fit.center);
        scores.push(dist_sq - r_sq);
    }
    Ok(scores)
}

/// Predict binary anomaly labels (`true` = anomaly) for `n` test samples.
///
/// A sample is classified as anomalous if `score > 0` (outside the soft sphere).
pub fn soft_svdd_predict(fit: &SoftSvddFit, x: &[f64], n: usize) -> AnomalyResult<Vec<bool>> {
    let scores = soft_svdd_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s > 0.0).collect())
}

/// Return the learned soft-boundary radius `R`.
pub fn soft_svdd_radius(fit: &SoftSvddFit) -> f64 {
    fit.radius
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(input_dim: usize) -> SoftSvddConfig {
        SoftSvddConfig {
            input_dim,
            hidden1: 8,
            hidden2: 6,
            latent_dim: 4,
            nu: 0.1,
            lr: 1e-3,
            n_epochs: 5,
        }
    }

    #[test]
    fn soft_svdd_fit_returns_finite_radius() {
        let cfg = make_config(4);
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
        let fit = soft_svdd_fit(&x, 5, &cfg, 42).unwrap();
        assert!(
            fit.radius.is_finite() && fit.radius >= 0.0,
            "radius={}",
            fit.radius
        );
    }

    #[test]
    fn soft_svdd_scores_are_finite() {
        let cfg = make_config(4);
        let x_train: Vec<f64> = (0..40).map(|i| i as f64 * 0.02).collect();
        let fit = soft_svdd_fit(&x_train, 10, &cfg, 7).unwrap();
        let x_test: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 100.0, 100.0, 100.0, 100.0];
        let scores = soft_svdd_score(&fit, &x_test, 2).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.is_finite()), "scores={scores:?}");
    }

    #[test]
    fn soft_svdd_outlier_has_higher_score() {
        let cfg = make_config(4);
        let x_train: Vec<f64> = (0..40).map(|i| (i as f64) * 0.01).collect();
        let fit = soft_svdd_fit(&x_train, 10, &cfg, 13).unwrap();

        let inlier = vec![0.05, 0.05, 0.05, 0.05];
        let outlier = vec![999.0, 999.0, 999.0, 999.0];

        let s_in = soft_svdd_score(&fit, &inlier, 1).unwrap()[0];
        let s_out = soft_svdd_score(&fit, &outlier, 1).unwrap()[0];
        assert!(s_out > s_in, "s_out={s_out} s_in={s_in}");
    }

    #[test]
    fn soft_svdd_predict_extreme_outlier() {
        let cfg = SoftSvddConfig {
            input_dim: 2,
            hidden1: 6,
            hidden2: 4,
            latent_dim: 2,
            nu: 0.05,
            lr: 5e-3,
            n_epochs: 20,
        };
        let x_train: Vec<f64> = (0..20)
            .flat_map(|i| vec![i as f64 * 0.05, i as f64 * 0.05])
            .collect();
        let fit = soft_svdd_fit(&x_train, 20, &cfg, 99).unwrap();
        let x_test = vec![0.0, 0.0, 1000.0, 1000.0];
        let preds = soft_svdd_predict(&fit, &x_test, 2).unwrap();
        // The extreme outlier should be detected
        assert!(preds[1], "extreme outlier should be anomaly");
    }

    #[test]
    fn soft_svdd_radius_fn() {
        let cfg = make_config(3);
        let x: Vec<f64> = (0..15).map(|i| i as f64 * 0.1).collect();
        let fit = soft_svdd_fit(&x, 5, &cfg, 17).unwrap();
        assert_eq!(soft_svdd_radius(&fit), fit.radius);
    }

    #[test]
    fn soft_svdd_config_invalid_nu() {
        let mut cfg = make_config(4);
        cfg.nu = 0.0;
        let x: Vec<f64> = vec![0.0; 40];
        assert!(soft_svdd_fit(&x, 10, &cfg, 1).is_err());
    }

    #[test]
    fn soft_svdd_config_invalid_dim() {
        let mut cfg = make_config(4);
        cfg.hidden1 = 0;
        let x: Vec<f64> = vec![0.0; 40];
        assert!(soft_svdd_fit(&x, 10, &cfg, 1).is_err());
    }
}
