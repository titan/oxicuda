//! Trainable DeepSVDD with explicit backward pass.
//!
//! Implements gradient-based training of DeepSVDD (Ruff et al. 2018) with a
//! full manual backpropagation through a 3-layer MLP.
//!
//! ## Architecture
//!
//! `MLP: x → ReLU(W1 x + b1) → ReLU(W2 h1 + b2) → W3 h2`
//!
//! The final layer has **no bias** and **no activation** (required to prevent
//! hypersphere collapse in DeepSVDD).
//!
//! ## Training
//!
//! Minimises `L = (1/n) Σ_i ‖z_i − c‖²` where `z_i = MLP(x_i)`.
//!
//! After `warm_up_epochs` the centre `c` is fixed as `mean(z_i)`.
//!
//! ## Collapse prevention
//!
//! If `‖c‖₂ < ε`, `c` is shifted to `ε · 1` component-wise.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

const COLLAPSE_EPS: f64 = 1e-3;

// ─── Xavier initialisation (f64) ─────────────────────────────────────────────

fn xavier_init_f64(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── TrainableSvddConfig ─────────────────────────────────────────────────────

/// Configuration for trainable DeepSVDD.
#[derive(Debug, Clone)]
pub struct TrainableSvddConfig {
    /// Input feature dimensionality.
    pub input_dim: usize,
    /// Width of the first hidden layer.
    pub hidden1: usize,
    /// Width of the second hidden layer.
    pub hidden2: usize,
    /// Latent (output) dimensionality — the hypersphere lives here.
    pub latent_dim: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Total training epochs.
    pub n_epochs: usize,
    /// Epochs to run before fixing the hypersphere centre `c`.
    pub warm_up_epochs: usize,
}

impl Default for TrainableSvddConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            hidden1: 32,
            hidden2: 16,
            latent_dim: 8,
            lr: 1e-3,
            n_epochs: 50,
            warm_up_epochs: 10,
        }
    }
}

// ─── TrainableSvddFit ─────────────────────────────────────────────────────────

/// Trained DeepSVDD model.
///
/// Network layout:
/// - `W1 ∈ ℝ^{hidden1 × input_dim}`, `b1 ∈ ℝ^{hidden1}`
/// - `W2 ∈ ℝ^{hidden2 × hidden1}`, `b2 ∈ ℝ^{hidden2}`
/// - `W3 ∈ ℝ^{latent_dim × hidden2}` — **no bias**, **no activation**
pub struct TrainableSvddFit {
    pub w1: Vec<f64>,
    pub w2: Vec<f64>,
    pub w3: Vec<f64>,
    pub b1: Vec<f64>,
    pub b2: Vec<f64>,
    /// Hypersphere centre in latent space.
    pub center: Vec<f64>,
    /// Per-epoch mean SVDD loss (length = n_epochs).
    pub loss_history: Vec<f64>,
    // Dimensions stored for scoring
    input_dim: usize,
    hidden1: usize,
    hidden2: usize,
    latent_dim: usize,
}

impl TrainableSvddFit {
    /// Return a reference to the per-epoch loss history.
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }
}

// ─── Forward pass helpers ─────────────────────────────────────────────────────

/// Dense layer: `out = W x + b`, shape `[fan_out]`.
#[inline]
fn dense_f64(x: &[f64], w: &[f64], b: &[f64], fan_in: usize, fan_out: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; fan_out];
    for o in 0..fan_out {
        let mut acc = b[o];
        for i in 0..fan_in {
            acc += w[o * fan_in + i] * x[i];
        }
        out[o] = acc;
    }
    out
}

/// Dense layer without bias: `out = W x`, shape `[fan_out]`.
#[inline]
fn dense_no_bias(x: &[f64], w: &[f64], fan_in: usize, fan_out: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; fan_out];
    for o in 0..fan_out {
        let mut acc = 0.0_f64;
        for i in 0..fan_in {
            acc += w[o * fan_in + i] * x[i];
        }
        out[o] = acc;
    }
    out
}

/// ReLU activation in-place.
#[inline]
fn relu_inplace(v: &mut [f64]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

/// Returns 1.0 where `v > 0`, else 0.0 (ReLU derivative indicator).
#[inline]
fn relu_mask(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect()
}

/// Full forward pass: returns `(h1_pre, h1, h2_pre, h2, z)`.
#[allow(clippy::type_complexity)]
fn forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    w3: &[f64],
    input_dim: usize,
    hidden1: usize,
    hidden2: usize,
    latent_dim: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Layer 1: h1 = ReLU(W1 x + b1)
    let mut h1_pre = dense_f64(x, w1, b1, input_dim, hidden1);
    let h1_pre_copy = h1_pre.clone();
    relu_inplace(&mut h1_pre);
    let h1 = h1_pre;

    // Layer 2: h2 = ReLU(W2 h1 + b2)
    let mut h2_pre = dense_f64(&h1, w2, b2, hidden1, hidden2);
    let h2_pre_copy = h2_pre.clone();
    relu_inplace(&mut h2_pre);
    let h2 = h2_pre;

    // Layer 3: z = W3 h2 (no bias, no activation)
    let z = dense_no_bias(&h2, w3, hidden2, latent_dim);

    (h1_pre_copy, h1, h2_pre_copy, h2, z)
}

// ─── Training ─────────────────────────────────────────────────────────────────

/// Fit a trainable DeepSVDD model.
///
/// `x` is flat `[n × input_dim]` row-major training data (normal samples only).
pub fn trainable_svdd_fit(
    x: &[f64],
    n: usize,
    cfg: &TrainableSvddConfig,
    seed: u64,
) -> AnomalyResult<TrainableSvddFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    let d = cfg.input_dim;
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if cfg.hidden1 == 0 || cfg.hidden2 == 0 || cfg.latent_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "hidden1, hidden2, and latent_dim must all be > 0".into(),
        });
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let h1 = cfg.hidden1;
    let h2 = cfg.hidden2;
    let ld = cfg.latent_dim;
    let mut rng = LcgRng::new(seed);

    // Initialise weights (Xavier)
    let mut w1 = xavier_init_f64(d, h1, &mut rng);
    let mut b1 = vec![0.0_f64; h1];
    let mut w2 = xavier_init_f64(h1, h2, &mut rng);
    let mut b2 = vec![0.0_f64; h2];
    let mut w3 = xavier_init_f64(h2, ld, &mut rng);

    // Initialise centre as zeros (will be set after warm-up)
    let mut center = vec![0.0_f64; ld];
    let mut center_fixed = false;

    let mut loss_history = Vec::with_capacity(cfg.n_epochs);

    for epoch in 0..cfg.n_epochs {
        // ── Set / re-set centre after warm-up ──────────────────────────────
        if epoch == cfg.warm_up_epochs && !center_fixed {
            // Compute centre as mean of latent representations
            let mut c = vec![0.0_f64; ld];
            for i in 0..n {
                let xi = &x[i * d..(i + 1) * d];
                let (_, _, _, _, zi) = forward(xi, &w1, &b1, &w2, &b2, &w3, d, h1, h2, ld);
                for (cj, zj) in c.iter_mut().zip(zi.iter()) {
                    *cj += zj;
                }
            }
            let inv_n = 1.0 / n as f64;
            for cj in c.iter_mut() {
                *cj *= inv_n;
            }
            // Collapse prevention: if ||c|| is too small, perturb
            let norm_c: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_c < COLLAPSE_EPS {
                for cj in c.iter_mut() {
                    *cj += COLLAPSE_EPS;
                }
            }
            center = c;
            center_fixed = true;
        }

        // ── Gradient accumulators ──────────────────────────────────────────
        let mut dw1 = vec![0.0_f64; h1 * d];
        let mut db1 = vec![0.0_f64; h1];
        let mut dw2 = vec![0.0_f64; h2 * h1];
        let mut db2 = vec![0.0_f64; h2];
        let mut dw3 = vec![0.0_f64; ld * h2];

        let mut epoch_loss = 0.0_f64;

        // ── One pass over all training samples ────────────────────────────
        for i in 0..n {
            let xi = &x[i * d..(i + 1) * d];
            let (h1_pre, h1_act, h2_pre, h2_act, zi) =
                forward(xi, &w1, &b1, &w2, &b2, &w3, d, h1, h2, ld);

            // Per-sample loss contribution: ‖z_i − c‖²
            let loss_i: f64 = zi
                .iter()
                .zip(center.iter())
                .map(|(zj, cj)| (zj - cj).powi(2))
                .sum();
            epoch_loss += loss_i;

            // dL/dz_i = 2*(z_i - c) / n
            let inv_n = 1.0 / n as f64;
            let dz: Vec<f64> = zi
                .iter()
                .zip(center.iter())
                .map(|(zj, cj)| 2.0 * (zj - cj) * inv_n)
                .collect();

            // Backprop layer 3 (no bias): dL/dW3 += h2^T * dz, dL/dh2 = W3^T * dz
            let mut dh2 = vec![0.0_f64; h2];
            for o in 0..ld {
                for i2 in 0..h2 {
                    dw3[o * h2 + i2] += dz[o] * h2_act[i2];
                }
            }
            for i2 in 0..h2 {
                let mut acc = 0.0_f64;
                for o in 0..ld {
                    acc += w3[o * h2 + i2] * dz[o];
                }
                dh2[i2] = acc;
            }

            // ReLU gradient through layer 2
            let relu2 = relu_mask(&h2_pre);
            let dh2_pre: Vec<f64> = dh2.iter().zip(relu2.iter()).map(|(g, m)| g * m).collect();

            // Backprop layer 2: dL/dW2, dL/db2, dL/dh1
            let mut dh1 = vec![0.0_f64; h1];
            for o in 0..h2 {
                for i1 in 0..h1 {
                    dw2[o * h1 + i1] += dh2_pre[o] * h1_act[i1];
                }
                db2[o] += dh2_pre[o];
            }
            for i1 in 0..h1 {
                let mut acc = 0.0_f64;
                for o in 0..h2 {
                    acc += w2[o * h1 + i1] * dh2_pre[o];
                }
                dh1[i1] = acc;
            }

            // ReLU gradient through layer 1
            let relu1 = relu_mask(&h1_pre);
            let dh1_pre: Vec<f64> = dh1.iter().zip(relu1.iter()).map(|(g, m)| g * m).collect();

            // Backprop layer 1: dL/dW1, dL/db1
            for o in 0..h1 {
                for i0 in 0..d {
                    dw1[o * d + i0] += dh1_pre[o] * xi[i0];
                }
                db1[o] += dh1_pre[o];
            }
        }

        // ── SGD update ────────────────────────────────────────────────────
        let lr = cfg.lr;
        for v in w1.iter_mut().zip(dw1.iter()) {
            *v.0 -= lr * v.1;
        }
        for v in b1.iter_mut().zip(db1.iter()) {
            *v.0 -= lr * v.1;
        }
        for v in w2.iter_mut().zip(dw2.iter()) {
            *v.0 -= lr * v.1;
        }
        for v in b2.iter_mut().zip(db2.iter()) {
            *v.0 -= lr * v.1;
        }
        for v in w3.iter_mut().zip(dw3.iter()) {
            *v.0 -= lr * v.1;
        }

        loss_history.push(epoch_loss / n as f64);
    }

    // If warm_up_epochs >= n_epochs, centre was never fixed — set it now
    if !center_fixed {
        let mut c = vec![0.0_f64; ld];
        for i in 0..n {
            let xi = &x[i * d..(i + 1) * d];
            let (_, _, _, _, zi) = forward(xi, &w1, &b1, &w2, &b2, &w3, d, h1, h2, ld);
            for (cj, zj) in c.iter_mut().zip(zi.iter()) {
                *cj += zj;
            }
        }
        let inv_n = 1.0 / n as f64;
        for cj in c.iter_mut() {
            *cj *= inv_n;
        }
        let norm_c: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_c < COLLAPSE_EPS {
            for cj in c.iter_mut() {
                *cj += COLLAPSE_EPS;
            }
        }
        center = c;
    }

    Ok(TrainableSvddFit {
        w1,
        w2,
        w3,
        b1,
        b2,
        center,
        loss_history,
        input_dim: d,
        hidden1: h1,
        hidden2: h2,
        latent_dim: ld,
    })
}

// ─── Score ─────────────────────────────────────────────────────────────────────

/// Compute SVDD anomaly scores `‖z_i − c‖²` for `n` test samples.
pub fn trainable_svdd_score(
    fit: &TrainableSvddFit,
    x: &[f64],
    n: usize,
) -> AnomalyResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let d = fit.input_dim;
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let h1 = fit.hidden1;
    let h2 = fit.hidden2;
    let ld = fit.latent_dim;
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        let (_, _, _, _, zi) = forward(
            xi, &fit.w1, &fit.b1, &fit.w2, &fit.b2, &fit.w3, d, h1, h2, ld,
        );
        let score: f64 = zi
            .iter()
            .zip(fit.center.iter())
            .map(|(zj, cj)| (zj - cj).powi(2))
            .sum();
        scores.push(score);
    }

    Ok(scores)
}

/// Predict anomaly labels (true = anomaly) with a distance threshold.
pub fn trainable_svdd_predict(
    fit: &TrainableSvddFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = trainable_svdd_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

/// Return the per-epoch loss history stored in the fit.
pub fn trainable_svdd_loss_history(fit: &TrainableSvddFit) -> &[f64] {
    fit.loss_history()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cfg(d: usize) -> TrainableSvddConfig {
        TrainableSvddConfig {
            input_dim: d,
            hidden1: 8,
            hidden2: 4,
            latent_dim: 4,
            lr: 1e-3,
            n_epochs: 5,
            warm_up_epochs: 2,
        }
    }

    #[test]
    fn test_fit_returns_correct_dims() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 1).expect("DeepSVDD fit on 20 samples should succeed");
        assert_eq!(fit.w1.len(), cfg.hidden1 * d);
        assert_eq!(fit.w2.len(), cfg.hidden2 * cfg.hidden1);
        assert_eq!(fit.w3.len(), cfg.latent_dim * cfg.hidden2);
        assert_eq!(fit.b1.len(), cfg.hidden1);
        assert_eq!(fit.b2.len(), cfg.hidden2);
        assert_eq!(fit.center.len(), cfg.latent_dim);
    }

    #[test]
    fn test_loss_history_length() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 2).expect("DeepSVDD fit on 20 samples should succeed");
        assert_eq!(
            fit.loss_history.len(),
            cfg.n_epochs,
            "loss_history should have one entry per epoch"
        );
    }

    #[test]
    fn test_loss_history_finite() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 3).expect("DeepSVDD fit on 20 samples should succeed");
        for &l in fit.loss_history() {
            assert!(l.is_finite(), "loss not finite: {l}");
        }
    }

    #[test]
    fn test_score_output_length() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 4).expect("DeepSVDD fit on 20 samples should succeed");
        let test: Vec<f64> = (0..7 * d).map(|i| i as f64 * 0.1).collect();
        let scores =
            trainable_svdd_score(&fit, &test, 7).expect("scoring 7 test samples should succeed");
        assert_eq!(scores.len(), 7);
    }

    #[test]
    fn test_score_finite() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 5).expect("DeepSVDD fit on 20 samples should succeed");
        let test: Vec<f64> = vec![0.1_f64; d];
        let scores = trainable_svdd_score(&fit, &test, 1)
            .expect("scoring a single test sample should succeed");
        assert!(scores[0].is_finite(), "score not finite: {}", scores[0]);
    }

    #[test]
    fn test_score_non_negative() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 6).expect("DeepSVDD fit on 20 samples should succeed");
        let test: Vec<f64> = (0..5 * d).map(|i| i as f64 * 0.05).collect();
        let scores =
            trainable_svdd_score(&fit, &test, 5).expect("scoring 5 test samples should succeed");
        for &s in &scores {
            assert!(s >= 0.0, "negative score: {s}");
        }
    }

    #[test]
    fn test_predict_length() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 7).expect("DeepSVDD fit on 20 samples should succeed");
        let test: Vec<f64> = (0..6 * d).map(|i| i as f64 * 0.1).collect();
        let preds = trainable_svdd_predict(&fit, &test, 6, 1.0)
            .expect("predict on 6 test samples should succeed");
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_center_not_collapsed() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 8).expect("DeepSVDD fit on 20 samples should succeed");
        let norm: f64 = fit.center.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            norm >= COLLAPSE_EPS,
            "centre collapsed to near-zero: norm={norm}"
        );
    }

    #[test]
    fn test_outlier_higher_score() {
        // Dense normal data near 0; far outlier should score higher
        let d = 4_usize;
        let cfg = TrainableSvddConfig {
            input_dim: d,
            hidden1: 8,
            hidden2: 4,
            latent_dim: 4,
            lr: 5e-4,
            n_epochs: 20,
            warm_up_epochs: 5,
        };
        let x: Vec<f64> = (0..50 * d).map(|_| 0.1_f64).collect();
        let fit = trainable_svdd_fit(&x, 50, &cfg, 77)
            .expect("DeepSVDD fit on 50 constant samples should succeed");

        let normal: Vec<f64> = vec![0.1; d];
        let outlier: Vec<f64> = vec![100.0; d];

        let s_normal = trainable_svdd_score(&fit, &normal, 1)
            .expect("scoring a normal point should succeed")[0];
        let s_outlier = trainable_svdd_score(&fit, &outlier, 1)
            .expect("scoring an outlier point should succeed")[0];

        assert!(
            s_outlier > s_normal,
            "outlier score {s_outlier} should > normal score {s_normal}"
        );
    }

    #[test]
    fn test_loss_history_fn() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit =
            trainable_svdd_fit(&x, 20, &cfg, 9).expect("DeepSVDD fit on 20 samples should succeed");
        let hist = trainable_svdd_loss_history(&fit);
        assert_eq!(hist.len(), cfg.n_epochs);
    }

    #[test]
    fn test_error_on_empty_input() {
        let cfg = simple_cfg(4);
        let res = trainable_svdd_fit(&[], 0, &cfg, 10);
        assert!(res.is_err(), "expected EmptyInput error");
    }

    #[test]
    fn test_error_on_dimension_mismatch() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit = trainable_svdd_fit(&x, 20, &cfg, 11)
            .expect("DeepSVDD fit on 20 samples should succeed");
        let bad_test = vec![0.0_f64; 3]; // wrong length for d=4, n=1
        let res = trainable_svdd_score(&fit, &bad_test, 1);
        assert!(res.is_err(), "expected DimensionMismatch error");
    }

    #[test]
    fn test_warm_up_skip_when_warm_up_ge_n_epochs() {
        // warm_up_epochs >= n_epochs: centre fixed at end of training
        let d = 4_usize;
        let cfg = TrainableSvddConfig {
            n_epochs: 3,
            warm_up_epochs: 10, // larger than n_epochs
            ..simple_cfg(d)
        };
        let x: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.01).collect();
        let fit = trainable_svdd_fit(&x, 20, &cfg, 12)
            .expect("DeepSVDD fit with warm_up_epochs > n_epochs should succeed");
        // Centre should still be non-trivial
        let norm: f64 = fit.center.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm.is_finite(), "centre norm is not finite: {norm}");
    }
}
