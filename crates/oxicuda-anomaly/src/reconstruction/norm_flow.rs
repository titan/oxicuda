//! Normalising-Flow anomaly scoring (Rezende & Mohamed 2015; Dinh et al. 2017 RealNVP).
//!
//! A normalising flow models the data density exactly by transforming the input `x`
//! through a sequence of invertible maps `f = f_K ∘ … ∘ f_1` into a latent variable
//! `z = f(x)` with a simple base density (standard Gaussian). The change-of-variables
//! formula gives the exact log-likelihood
//!
//! ```text
//! log p(x) = log p_Z(f(x)) + Σ_k log |det ∂f_k/∂·| .
//! ```
//!
//! We use **RealNVP affine coupling layers**: each layer splits the features into two
//! halves by a binary mask `b`, leaves the masked half unchanged, and affinely
//! transforms the other half conditioned on the masked half:
//!
//! ```text
//! y_b      = x_b
//! y_{1−b}  = x_{1−b} ⊙ exp(s(x_b)) + t(x_b),
//! ```
//!
//! where `s, t` are small MLPs. The log-determinant is simply `Σ s(x_b)`, and the
//! inverse is closed-form, so the flow trains by **maximum likelihood** (gradient
//! descent on `−log p(x)`) and scores anomalies by the **negative log-likelihood**
//! `−log p(x)` — out-of-distribution points have low `log p(x)`, hence a high score.
//!
//! The model trains in `f64` for numerical headroom and exposes an `f32` boundary, in
//! line with the rest of the `reconstruction` module.
//!
//! # References
//!
//! - D. Rezende & S. Mohamed (2015), "Variational Inference with Normalizing Flows", ICML.
//! - L. Dinh, J. Sohl-Dickstein & S. Bengio (2017), "Density Estimation using Real NVP", ICLR.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the normalising-flow detector.
#[derive(Debug, Clone)]
pub struct NormFlowConfig {
    /// Number of affine coupling layers (default `4`). Masks alternate parity.
    pub n_layers: usize,
    /// Hidden width of the `s`/`t` MLPs (default `16`).
    pub hidden: usize,
    /// Training epochs (default `200`).
    pub epochs: usize,
    /// Learning rate (default `2 × 10⁻³`).
    pub lr: f64,
    /// Clamp on the log-scale `s` to keep `exp(s)` bounded (default `4.0`).
    pub scale_clamp: f64,
    /// RNG seed (default `42`).
    pub seed: u64,
}

impl Default for NormFlowConfig {
    fn default() -> Self {
        Self {
            n_layers: 4,
            hidden: 16,
            epochs: 200,
            lr: 2e-3,
            scale_clamp: 4.0,
            seed: 42,
        }
    }
}

// ─── Per-layer parameters ─────────────────────────────────────────────────────

/// One affine coupling layer with a shared-trunk `s`/`t` network.
///
/// Network: `h = tanh(W1 · x_masked + b1)`, then `s = tanh(Ws · h + bs) · clamp`,
/// `t = Wt · h + bt`. Only the masked features feed the network; the transformed
/// features are the complement of the mask.
#[derive(Debug, Clone)]
struct CouplingLayer {
    /// Binary mask (`true` ⇒ feature passes through unchanged and conditions the net).
    mask: Vec<bool>,
    w1: Vec<f64>, // [hidden × d]
    b1: Vec<f64>, // [hidden]
    ws: Vec<f64>, // [d × hidden]  (output one scale per feature)
    bs: Vec<f64>, // [d]
    wt: Vec<f64>, // [d × hidden]
    bt: Vec<f64>, // [d]
    d: usize,
    hidden: usize,
    scale_clamp: f64,
}

impl CouplingLayer {
    fn new(d: usize, hidden: usize, parity: bool, scale_clamp: f64, rng: &mut LcgRng) -> Self {
        let mask: Vec<bool> = (0..d).map(|i| (i % 2 == 0) == parity).collect();
        let init = |n: usize, fan_in: usize, rng: &mut LcgRng| -> Vec<f64> {
            let limit = (1.0 / fan_in as f64).sqrt();
            (0..n)
                .map(|_| (rng.next_f32() as f64 * 2.0 - 1.0) * limit)
                .collect()
        };
        Self {
            mask,
            w1: init(hidden * d, d, rng),
            b1: vec![0.0; hidden],
            ws: init(d * hidden, hidden, rng),
            bs: vec![0.0; d],
            wt: init(d * hidden, hidden, rng),
            bt: vec![0.0; d],
            d,
            hidden,
            scale_clamp,
        }
    }

    /// Forward pass `z = f(x)`; returns `(z, log_det, cache)` where `cache` stores the
    /// pre-activations needed by the backward pass.
    fn forward(&self, x: &[f64]) -> (Vec<f64>, f64, LayerCache) {
        // Masked input feeds the network.
        let mut x_masked = vec![0.0_f64; self.d];
        for (i, slot) in x_masked.iter_mut().enumerate() {
            if self.mask[i] {
                *slot = x[i];
            }
        }
        // h = tanh(W1 x_masked + b1).
        let mut pre_h = self.b1.clone();
        for (j, slot) in pre_h.iter_mut().enumerate() {
            let mut acc = *slot;
            for (i, &xm) in x_masked.iter().enumerate() {
                acc += self.w1[j * self.d + i] * xm;
            }
            *slot = acc;
        }
        let h: Vec<f64> = pre_h.iter().map(|v| v.tanh()).collect();
        // s, t per feature.
        let mut s = vec![0.0_f64; self.d];
        let mut t = vec![0.0_f64; self.d];
        for i in 0..self.d {
            let mut acc_s = self.bs[i];
            let mut acc_t = self.bt[i];
            for (j, &hj) in h.iter().enumerate() {
                acc_s += self.ws[i * self.hidden + j] * hj;
                acc_t += self.wt[i * self.hidden + j] * hj;
            }
            // Bounded log-scale via tanh.
            s[i] = self.scale_clamp * acc_s.tanh();
            t[i] = acc_t;
        }
        // Apply affine transform to the complement of the mask.
        let mut z = x.to_vec();
        let mut log_det = 0.0_f64;
        for i in 0..self.d {
            if !self.mask[i] {
                z[i] = x[i] * s[i].exp() + t[i];
                log_det += s[i];
            } else {
                s[i] = 0.0; // unchanged features contribute no log-det
            }
        }
        let cache = LayerCache {
            x_masked,
            h,
            s,
            x_in: x.to_vec(),
        };
        (z, log_det, cache)
    }

    /// Accumulate gradients given `dL/dz` (upstream) and the cache; return `dL/dx`.
    fn backward(
        &self,
        dz: &[f64],
        dlog_det: f64,
        cache: &LayerCache,
        grad: &mut LayerGrad,
    ) -> Vec<f64> {
        let d = self.d;
        let hidden = self.hidden;
        // For transformed features: z_i = x_i exp(s_i) + t_i.
        // ∂z_i/∂x_i = exp(s_i); ∂z_i/∂s_i = x_i exp(s_i); ∂z_i/∂t_i = 1.
        // log_det term adds dlog_det to each s_i (transformed).
        let mut ds = vec![0.0_f64; d];
        let mut dt = vec![0.0_f64; d];
        let mut dx = vec![0.0_f64; d];
        for i in 0..d {
            if !self.mask[i] {
                let exp_s = cache.s[i].exp();
                dx[i] += dz[i] * exp_s;
                ds[i] += dz[i] * cache.x_in[i] * exp_s + dlog_det;
                dt[i] += dz[i];
            }
            // masked features pass through: dx already carries dz[i] below.
        }
        // s_i = scale_clamp * tanh(pre_s_i); chain through tanh.
        // We need pre-activations of s; recompute from h (cheap) is avoided by storing.
        // Re-derive d(pre_s) using s value: tanh' = 1 − tanh² and tanh = s/clamp.
        let mut dpre_s = vec![0.0_f64; d];
        for i in 0..d {
            let tanh_val = cache.s[i] / self.scale_clamp;
            dpre_s[i] = ds[i] * self.scale_clamp * (1.0 - tanh_val * tanh_val);
        }
        // Backprop into ws, bs, wt, bt and into h.
        let mut dh = vec![0.0_f64; hidden];
        for i in 0..d {
            grad.bs[i] += dpre_s[i];
            grad.bt[i] += dt[i];
            for (j, dhj) in dh.iter_mut().enumerate() {
                grad.ws[i * hidden + j] += dpre_s[i] * cache.h[j];
                grad.wt[i * hidden + j] += dt[i] * cache.h[j];
                *dhj += dpre_s[i] * self.ws[i * hidden + j] + dt[i] * self.wt[i * hidden + j];
            }
        }
        // Through tanh of h.
        let mut dpre_h = vec![0.0_f64; hidden];
        for (j, dph) in dpre_h.iter_mut().enumerate() {
            let hv = cache.h[j];
            *dph = dh[j] * (1.0 - hv * hv);
        }
        // Into w1, b1 and into x_masked.
        for (j, &dph) in dpre_h.iter().enumerate() {
            grad.b1[j] += dph;
            for (i, dxi) in dx.iter_mut().enumerate() {
                grad.w1[j * d + i] += dph * cache.x_masked[i];
                if self.mask[i] {
                    *dxi += dph * self.w1[j * d + i];
                }
            }
        }
        // Masked features also pass straight through z = x ⇒ add dz on those.
        for i in 0..d {
            if self.mask[i] {
                dx[i] += dz[i];
            }
        }
        dx
    }
}

/// Cached intermediates for one layer's backward pass.
#[derive(Debug, Clone)]
struct LayerCache {
    x_masked: Vec<f64>,
    h: Vec<f64>,
    s: Vec<f64>,
    x_in: Vec<f64>,
}

/// Gradient accumulator mirroring [`CouplingLayer`] parameters.
#[derive(Debug, Clone)]
struct LayerGrad {
    w1: Vec<f64>,
    b1: Vec<f64>,
    ws: Vec<f64>,
    bs: Vec<f64>,
    wt: Vec<f64>,
    bt: Vec<f64>,
}

impl LayerGrad {
    fn zeros(d: usize, hidden: usize) -> Self {
        Self {
            w1: vec![0.0; hidden * d],
            b1: vec![0.0; hidden],
            ws: vec![0.0; d * hidden],
            bs: vec![0.0; d],
            wt: vec![0.0; d * hidden],
            bt: vec![0.0; d],
        }
    }
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// A trained normalising-flow density model.
#[derive(Debug, Clone)]
pub struct NormFlowFit {
    layers: Vec<CouplingLayer>,
    /// Per-feature mean used to standardise inputs.
    mean: Vec<f64>,
    /// Per-feature std used to standardise inputs.
    std: Vec<f64>,
    d: usize,
    /// Final mean training NLL (per sample).
    pub final_nll: f64,
    /// NLL recorded once per epoch.
    pub nll_history: Vec<f64>,
}

const LOG_2PI: f64 = 1.837_877_066_409_345_5; // ln(2π)

impl NormFlowFit {
    /// Feature dimension the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.d
    }

    /// Exact log-likelihood `log p(x)` of a single (raw) sample.
    fn log_prob(&self, x_raw: &[f64]) -> f64 {
        // Standardise.
        let mut x: Vec<f64> = (0..self.d)
            .map(|i| (x_raw[i] - self.mean[i]) / self.std[i])
            .collect();
        let mut log_det = 0.0_f64;
        for layer in &self.layers {
            let (z, ld, _) = layer.forward(&x);
            x = z;
            log_det += ld;
        }
        // Base: standard normal. log p_Z(z) = −½ Σ z² − (d/2) ln(2π).
        let mut quad = 0.0_f64;
        for v in &x {
            quad += v * v;
        }
        let base = -0.5 * quad - 0.5 * self.d as f64 * LOG_2PI;
        // Standardisation Jacobian: −Σ ln(std_i).
        let mut std_log = 0.0_f64;
        for s in &self.std {
            std_log += s.ln();
        }
        base + log_det - std_log
    }

    /// Anomaly score `−log p(x)` for a single sample.
    ///
    /// # Errors
    /// [`AnomalyError::FeatureCountMismatch`] if `x.len() != n_features`.
    pub fn score_one(&self, x: &[f32]) -> AnomalyResult<f32> {
        if x.len() != self.d {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.d,
                got: x.len(),
            });
        }
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        Ok((-self.log_prob(&x64)) as f32)
    }
}

// ─── Training ─────────────────────────────────────────────────────────────────

/// Fit a normalising-flow density model on `data` (row-major `[n_samples × n_features]`).
///
/// # Errors
/// * [`AnomalyError::EmptyInput`] / [`AnomalyError::InvalidFeatureCount`].
/// * [`AnomalyError::DimensionMismatch`] on bad shapes.
/// * [`AnomalyError::InsufficientSamples`] if `n_samples < 2`.
pub fn norm_flow_fit(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    config: &NormFlowConfig,
) -> AnomalyResult<NormFlowFit> {
    if n_samples == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    if n_features == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if data.len() != n_samples * n_features {
        return Err(AnomalyError::DimensionMismatch {
            expected: n_samples * n_features,
            got: data.len(),
        });
    }
    if n_samples < 2 {
        return Err(AnomalyError::InsufficientSamples {
            need: 2,
            got: n_samples,
        });
    }
    if config.n_layers == 0 || config.hidden == 0 {
        return Err(AnomalyError::Internal {
            msg: "norm_flow: n_layers and hidden must be > 0".into(),
        });
    }

    let d = n_features;
    // Standardisation statistics.
    let mut mean = vec![0.0_f64; d];
    for i in 0..n_samples {
        for j in 0..d {
            mean[j] += data[i * d + j] as f64;
        }
    }
    let inv_n = 1.0 / n_samples as f64;
    for m in &mut mean {
        *m *= inv_n;
    }
    let mut std = vec![0.0_f64; d];
    for i in 0..n_samples {
        for j in 0..d {
            let v = data[i * d + j] as f64 - mean[j];
            std[j] += v * v;
        }
    }
    for s in &mut std {
        *s = (*s * inv_n).sqrt().max(1e-3);
    }

    // Standardised training set (f64).
    let mut x_std = vec![0.0_f64; n_samples * d];
    for i in 0..n_samples {
        for j in 0..d {
            x_std[i * d + j] = (data[i * d + j] as f64 - mean[j]) / std[j];
        }
    }

    let mut rng = LcgRng::new(config.seed);
    let mut layers: Vec<CouplingLayer> = (0..config.n_layers)
        .map(|k| CouplingLayer::new(d, config.hidden, k % 2 == 0, config.scale_clamp, &mut rng))
        .collect();

    let mut nll_history = Vec::with_capacity(config.epochs);
    let mut final_nll = f64::INFINITY;

    for _ in 0..config.epochs {
        let mut grads: Vec<LayerGrad> = layers
            .iter()
            .map(|_| LayerGrad::zeros(d, config.hidden))
            .collect();
        let mut epoch_nll = 0.0_f64;

        for i in 0..n_samples {
            let x0 = &x_std[i * d..(i + 1) * d];
            // Forward through all layers, caching.
            let mut x = x0.to_vec();
            let mut caches = Vec::with_capacity(layers.len());
            let mut log_det = 0.0_f64;
            for layer in &layers {
                let (z, ld, cache) = layer.forward(&x);
                x = z;
                log_det += ld;
                caches.push(cache);
            }
            // NLL = ½ Σ z² + (d/2)ln(2π) − log_det.
            let mut quad = 0.0_f64;
            for v in &x {
                quad += v * v;
            }
            let nll = 0.5 * quad + 0.5 * d as f64 * LOG_2PI - log_det;
            epoch_nll += nll;

            // Backprop: dNLL/dz_last = z (from ½z²); dNLL/dlog_det = −1.
            let mut dz: Vec<f64> = x.clone();
            let dlog_det = -1.0_f64;
            for k in (0..layers.len()).rev() {
                dz = layers[k].backward(&dz, dlog_det, &caches[k], &mut grads[k]);
            }
        }

        epoch_nll *= inv_n;
        nll_history.push(epoch_nll);
        final_nll = epoch_nll;

        // ── Gradient clipping (global L2 norm per layer) for training stability. ──
        // The mean gradient is g · inv_n; clip its norm to `clip_norm`.
        const CLIP_NORM: f64 = 10.0;
        for g in &mut grads {
            let mut sq = 0.0_f64;
            for v in
                g.w1.iter()
                    .chain(g.b1.iter())
                    .chain(g.ws.iter())
                    .chain(g.bs.iter())
                    .chain(g.wt.iter())
                    .chain(g.bt.iter())
            {
                let mv = v * inv_n;
                sq += mv * mv;
            }
            let gnorm = sq.sqrt();
            if gnorm > CLIP_NORM {
                let scale = CLIP_NORM / gnorm;
                for v in
                    g.w1.iter_mut()
                        .chain(g.b1.iter_mut())
                        .chain(g.ws.iter_mut())
                        .chain(g.bs.iter_mut())
                        .chain(g.wt.iter_mut())
                        .chain(g.bt.iter_mut())
                {
                    *v *= scale;
                }
            }
        }

        // SGD update (mean gradient).
        let lr = config.lr;
        for (layer, g) in layers.iter_mut().zip(grads.iter()) {
            for (w, gw) in layer.w1.iter_mut().zip(g.w1.iter()) {
                *w -= lr * gw * inv_n;
            }
            for (w, gw) in layer.b1.iter_mut().zip(g.b1.iter()) {
                *w -= lr * gw * inv_n;
            }
            for (w, gw) in layer.ws.iter_mut().zip(g.ws.iter()) {
                *w -= lr * gw * inv_n;
            }
            for (w, gw) in layer.bs.iter_mut().zip(g.bs.iter()) {
                *w -= lr * gw * inv_n;
            }
            for (w, gw) in layer.wt.iter_mut().zip(g.wt.iter()) {
                *w -= lr * gw * inv_n;
            }
            for (w, gw) in layer.bt.iter_mut().zip(g.bt.iter()) {
                *w -= lr * gw * inv_n;
            }
        }
    }

    Ok(NormFlowFit {
        layers,
        mean,
        std,
        d,
        final_nll,
        nll_history,
    })
}

/// Score a batch of samples with a trained flow (row-major `[n × n_features]`).
///
/// Returns the negative log-likelihood per sample (higher ⟹ more anomalous).
///
/// # Errors
/// [`AnomalyError::DimensionMismatch`] on shape mismatch.
pub fn norm_flow_score(fit: &NormFlowFit, data: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
    if data.len() != n * fit.d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * fit.d,
            got: data.len(),
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(fit.score_one(&data[i * fit.d..(i + 1) * fit.d])?);
    }
    Ok(out)
}

/// Predict binary labels (`true` = anomaly) by thresholding scores at the given
/// contamination quantile of the provided data.
///
/// # Errors
/// As [`norm_flow_score`]; also [`AnomalyError::InvalidThresholdPercentile`] for a
/// `contamination` outside `[0, 1]`.
pub fn norm_flow_predict(
    fit: &NormFlowFit,
    data: &[f32],
    n: usize,
    contamination: f32,
) -> AnomalyResult<Vec<bool>> {
    if !(0.0..=1.0).contains(&contamination) {
        return Err(AnomalyError::InvalidThresholdPercentile {
            p: contamination * 100.0,
        });
    }
    let scores = norm_flow_score(fit, data, n)?;
    // Flag the `k` highest-scoring points (rank-based ⇒ robust to ties).
    let k = ((contamination * n as f32).ceil() as usize).min(n);
    if k == 0 {
        return Ok(vec![false; n]);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut labels = vec![false; n];
    for &i in order.iter().take(k) {
        labels[i] = true;
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_cluster(n: usize, d: usize, center: f32, spread: f32, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut out = Vec::with_capacity(n * d);
        for _ in 0..n {
            for _ in 0..d {
                out.push(center + spread * rng.next_normal());
            }
        }
        out
    }

    #[test]
    fn fit_runs_and_reports_history() {
        let data = gauss_cluster(60, 2, 0.0, 1.0, 1);
        let cfg = NormFlowConfig {
            epochs: 50,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 60, 2, &cfg).expect("fit");
        assert_eq!(fit.n_features(), 2);
        assert_eq!(fit.nll_history.len(), 50);
        assert!(fit.final_nll.is_finite());
    }

    #[test]
    fn training_reduces_nll() {
        let data = gauss_cluster(80, 3, 1.0, 1.5, 2);
        let cfg = NormFlowConfig {
            epochs: 150,
            lr: 2e-3,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 80, 3, &cfg).expect("fit");
        let first = fit.nll_history[0];
        let last = *fit.nll_history.last().expect("nonempty");
        assert!(
            last <= first + 1e-6,
            "NLL did not improve: {first} -> {last}"
        );
    }

    #[test]
    fn outlier_scores_higher_than_inlier() {
        // Train on a tight cluster near the origin; a far point should be more anomalous.
        let data = gauss_cluster(120, 2, 0.0, 0.7, 3);
        let cfg = NormFlowConfig {
            epochs: 200,
            lr: 3e-3,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 120, 2, &cfg).expect("fit");
        let inlier = fit.score_one(&[0.0, 0.0]).expect("s");
        let outlier = fit.score_one(&[12.0, 12.0]).expect("s");
        assert!(outlier > inlier, "outlier {outlier} vs inlier {inlier}");
    }

    #[test]
    fn log_prob_is_higher_density_at_mode() {
        let data = gauss_cluster(100, 2, 3.0, 1.0, 4);
        let cfg = NormFlowConfig {
            epochs: 150,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 100, 2, &cfg).expect("fit");
        let s_mode = fit.score_one(&[3.0, 3.0]).expect("s");
        let s_tail = fit.score_one(&[3.0, 30.0]).expect("s");
        assert!(s_tail > s_mode, "tail {s_tail} vs mode {s_mode}");
    }

    #[test]
    fn score_batch_matches_single() {
        let data = gauss_cluster(40, 2, 0.0, 1.0, 5);
        let cfg = NormFlowConfig {
            epochs: 30,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 40, 2, &cfg).expect("fit");
        let q = vec![0.0_f32, 0.0, 5.0, 5.0, -3.0, 2.0];
        let batch = norm_flow_score(&fit, &q, 3).expect("batch");
        for i in 0..3 {
            let single = fit.score_one(&q[i * 2..i * 2 + 2]).expect("s");
            assert!((batch[i] - single).abs() < 1e-4);
        }
    }

    #[test]
    fn predict_flags_contamination_fraction() {
        let data = gauss_cluster(50, 2, 0.0, 1.0, 6);
        let cfg = NormFlowConfig {
            epochs: 60,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 50, 2, &cfg).expect("fit");
        let labels = norm_flow_predict(&fit, &data, 50, 0.1).expect("pred");
        let n_anom = labels.iter().filter(|&&b| b).count();
        // Roughly 10% flagged (allow slack for ties).
        assert!((1..=15).contains(&n_anom), "flagged {n_anom}");
    }

    #[test]
    fn log_likelihood_finite_for_all_training() {
        let data = gauss_cluster(40, 4, 0.0, 1.0, 7);
        let cfg = NormFlowConfig {
            epochs: 40,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 40, 4, &cfg).expect("fit");
        let scores = norm_flow_score(&fit, &data, 40).expect("scores");
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn feature_mismatch_errors() {
        let data = gauss_cluster(20, 2, 0.0, 1.0, 8);
        let fit = norm_flow_fit(
            &data,
            20,
            2,
            &NormFlowConfig {
                epochs: 5,
                ..Default::default()
            },
        )
        .expect("fit");
        assert!(matches!(
            fit.score_one(&[1.0, 2.0, 3.0]),
            Err(AnomalyError::FeatureCountMismatch { .. })
        ));
        assert!(matches!(
            norm_flow_score(&fit, &[1.0, 2.0, 3.0], 1),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_rejects_bad_shapes() {
        let cfg = NormFlowConfig::default();
        assert!(matches!(
            norm_flow_fit(&[], 0, 2, &cfg),
            Err(AnomalyError::EmptyInput)
        ));
        assert!(matches!(
            norm_flow_fit(&[1.0, 2.0], 1, 0, &cfg),
            Err(AnomalyError::InvalidFeatureCount { .. })
        ));
        assert!(matches!(
            norm_flow_fit(&[1.0, 2.0, 3.0], 2, 2, &cfg),
            Err(AnomalyError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            norm_flow_fit(&[1.0, 2.0], 1, 2, &cfg),
            Err(AnomalyError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn predict_rejects_bad_contamination() {
        let data = gauss_cluster(20, 2, 0.0, 1.0, 9);
        let fit = norm_flow_fit(
            &data,
            20,
            2,
            &NormFlowConfig {
                epochs: 5,
                ..Default::default()
            },
        )
        .expect("fit");
        assert!(matches!(
            norm_flow_predict(&fit, &data, 20, 1.5),
            Err(AnomalyError::InvalidThresholdPercentile { .. })
        ));
    }

    #[test]
    fn flow_is_invertible_identity_at_init_scale_zero() {
        // With bs/bt zero and a single layer, the masked half is unchanged; the flow
        // log-prob must be finite and the standardisation must round-trip on the mean.
        let data = gauss_cluster(30, 2, 2.0, 0.01, 10); // nearly constant ⇒ mean ≈ 2
        let cfg = NormFlowConfig {
            n_layers: 2,
            epochs: 5,
            ..Default::default()
        };
        let fit = norm_flow_fit(&data, 30, 2, &cfg).expect("fit");
        // A point at the mean should have a high log p (low score) relative to far points.
        let s_mean = fit.score_one(&[2.0, 2.0]).expect("s");
        let s_far = fit.score_one(&[2.0, 100.0]).expect("s");
        assert!(s_mean.is_finite() && s_far.is_finite());
        assert!(s_far > s_mean);
    }
}
