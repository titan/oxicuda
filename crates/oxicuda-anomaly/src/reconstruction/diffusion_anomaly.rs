//! Diffusion-Based Anomaly Detection for Tabular Data.
//!
//! Score-based / denoising diffusion model (DDPM-style) trained on normal data.
//! Anomaly scores = reconstruction error under partial diffusion at inference.
//!
//! Reference: inspired by Gong et al. 2023 "DDPM-based Anomaly Detection".
//!
//! # Algorithm
//!
//! 1. Learn a linear beta noise schedule: `β_t ∈ [β_start, β_end]`.
//! 2. Compute `α_t = 1 − β_t` and `ᾱ_t = ∏_{s=1}^{t} α_s`.
//! 3. Train a score-network (MLP) to predict noise:
//!    - Sample `t ~ Uniform{1,..,T}`, `ε ~ N(0,I)`.
//!    - Noisy sample: `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`.
//!    - Input to MLP: `[x_t; t/T]` (dim+1 inputs).
//!    - Loss: `‖ε − ε̂‖²`.
//! 4. Anomaly score = one-step denoising error at `t = T/4`:
//!    - `x_noisy = √ᾱ_{T/4} · x + √(1−ᾱ_{T/4}) · ε`
//!    - `x̂ = (x_noisy − √(1−ᾱ_{T/4}) · net([x_noisy; (T/4)/T])) / √ᾱ_{T/4}`
//!    - score = `‖x − x̂‖²`

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── SiLU activation ─────────────────────────────────────────────────────────

/// SiLU(x) = x · σ(x) where σ(x) = 1 / (1 + e^{-x}).
#[inline]
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// SiLU derivative: σ(x) + x · σ(x) · (1 − σ(x)).
#[inline]
fn silu_grad(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

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

// ─── DiffusionAnomalyConfig ──────────────────────────────────────────────────

/// Configuration for the tabular diffusion anomaly detector.
#[derive(Debug, Clone)]
pub struct DiffusionAnomalyConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width in the score network.
    pub hidden_dim: usize,
    /// Number of diffusion steps T.
    pub n_steps: usize,
    /// Training epochs.
    pub n_epochs: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Starting noise level β₁ (linear schedule start).
    pub beta_start: f64,
    /// Ending noise level β_T (linear schedule end).
    pub beta_end: f64,
}

impl Default for DiffusionAnomalyConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            hidden_dim: 64,
            n_steps: 100,
            n_epochs: 50,
            lr: 1e-3,
            beta_start: 1e-4,
            beta_end: 0.02,
        }
    }
}

// ─── DiffusionAnomalyFit ────────────────────────────────────────────────────

/// Fitted diffusion anomaly model.
///
/// Score network architecture: `[input_dim + 1] → hidden_dim → hidden_dim → input_dim`
/// with SiLU activations on hidden layers and linear output.
///
/// Layer 1: `W1 ∈ ℝ^{hidden_dim × (input_dim+1)}`, `b1 ∈ ℝ^{hidden_dim}`
/// Layer 2: `W2 ∈ ℝ^{hidden_dim × hidden_dim}`, `b2 ∈ ℝ^{hidden_dim}`
/// Layer 3: `W3 ∈ ℝ^{input_dim × hidden_dim}`, `b3 ∈ ℝ^{input_dim}`
pub struct DiffusionAnomalyFit {
    pub score_net_w1: Vec<f64>,
    pub score_net_b1: Vec<f64>,
    pub score_net_w2: Vec<f64>,
    pub score_net_b2: Vec<f64>,
    pub score_net_w3: Vec<f64>,
    pub score_net_b3: Vec<f64>,
    /// `α_t = 1 − β_t` for t = 0..n_steps (1-indexed schedule stored 0-indexed).
    pub alphas: Vec<f64>,
    /// `ᾱ_t = ∏_{s=0}^{t} α_s` cumulative product.
    pub alpha_bars: Vec<f64>,
    pub input_dim: usize,
}

// ─── Score-network forward pass ──────────────────────────────────────────────

/// Forward pass returning all intermediate values and final output.
///
/// Returns `(h1_pre, h1, h2_pre, h2, eps_hat)`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn score_net_forward(
    x_with_t: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    w3: &[f64],
    b3: &[f64],
    input_aug: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Layer 1
    let mut h1_pre = vec![0.0_f64; hidden_dim];
    for o in 0..hidden_dim {
        let mut acc = b1[o];
        for i in 0..input_aug {
            acc += w1[o * input_aug + i] * x_with_t[i];
        }
        h1_pre[o] = acc;
    }
    let h1: Vec<f64> = h1_pre.iter().map(|&v| silu(v)).collect();

    // Layer 2
    let mut h2_pre = vec![0.0_f64; hidden_dim];
    for o in 0..hidden_dim {
        let mut acc = b2[o];
        for i in 0..hidden_dim {
            acc += w2[o * hidden_dim + i] * h1[i];
        }
        h2_pre[o] = acc;
    }
    let h2: Vec<f64> = h2_pre.iter().map(|&v| silu(v)).collect();

    // Layer 3 (linear output)
    let mut eps_hat = vec![0.0_f64; output_dim];
    for o in 0..output_dim {
        let mut acc = b3[o];
        for i in 0..hidden_dim {
            acc += w3[o * hidden_dim + i] * h2[i];
        }
        eps_hat[o] = acc;
    }

    (h1_pre, h1, h2_pre, h2, eps_hat)
}

// ─── Noise schedule ──────────────────────────────────────────────────────────

fn build_noise_schedule(n_steps: usize, beta_start: f64, beta_end: f64) -> (Vec<f64>, Vec<f64>) {
    let mut alphas = Vec::with_capacity(n_steps);
    let mut alpha_bars = Vec::with_capacity(n_steps);

    let mut cumprod = 1.0_f64;
    let denom = (n_steps.saturating_sub(1)).max(1) as f64;
    for step in 0..n_steps {
        let t_frac = step as f64 / denom;
        let beta = beta_start + t_frac * (beta_end - beta_start);
        let alpha = 1.0 - beta;
        cumprod *= alpha;
        alphas.push(alpha);
        alpha_bars.push(cumprod);
    }

    (alphas, alpha_bars)
}

// ─── Training ────────────────────────────────────────────────────────────────

/// Fit a diffusion anomaly model on normal training data.
///
/// `x` is a flat `[n × input_dim]` row-major matrix of normal samples.
pub fn diffusion_anomaly_fit(
    x: &[f64],
    n: usize,
    cfg: &DiffusionAnomalyConfig,
    seed: u64,
) -> AnomalyResult<DiffusionAnomalyFit> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    let d = cfg.input_dim;
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }
    if cfg.n_steps == 0 {
        return Err(AnomalyError::Internal {
            msg: "n_steps must be > 0".into(),
        });
    }
    if cfg.hidden_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "hidden_dim must be > 0".into(),
        });
    }

    let h = cfg.hidden_dim;
    let input_aug = d + 1; // [x_noisy; t/T]
    let mut rng = LcgRng::new(seed);

    // Initialise score-network weights (Xavier)
    let mut w1 = xavier_init_f64(input_aug, h, &mut rng);
    let mut b1 = vec![0.0_f64; h];
    let mut w2 = xavier_init_f64(h, h, &mut rng);
    let mut b2 = vec![0.0_f64; h];
    let mut w3 = xavier_init_f64(h, d, &mut rng);
    let mut b3 = vec![0.0_f64; d];

    // Build noise schedule
    let (alphas, alpha_bars) = build_noise_schedule(cfg.n_steps, cfg.beta_start, cfg.beta_end);

    // Training loop
    for _epoch in 0..cfg.n_epochs {
        // Iterate over training samples in shuffled order (Fisher-Yates)
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }

        for &idx in &order {
            let x0 = &x[idx * d..(idx + 1) * d];

            // Sample random timestep t in [0, n_steps)
            let t = rng.next_usize(cfg.n_steps);
            let ab = alpha_bars[t];
            let sqrt_ab = ab.sqrt();
            let sqrt_one_minus_ab = (1.0 - ab).max(0.0).sqrt();
            let t_norm = (t + 1) as f64 / cfg.n_steps as f64;

            // Sample noise ε ~ N(0, I)
            let eps: Vec<f64> = (0..d).map(|_| rng.next_normal() as f64).collect();

            // Noisy sample: x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
            let x_noisy: Vec<f64> = x0
                .iter()
                .zip(eps.iter())
                .map(|(x0j, ej)| sqrt_ab * x0j + sqrt_one_minus_ab * ej)
                .collect();

            // Augment input with timestep embedding
            let mut x_aug = Vec::with_capacity(input_aug);
            x_aug.extend_from_slice(&x_noisy);
            x_aug.push(t_norm);

            // Forward pass
            let (h1_pre, h1, h2_pre, h2, eps_hat) =
                score_net_forward(&x_aug, &w1, &b1, &w2, &b2, &w3, &b3, input_aug, h, d);

            // dL/d(ε̂)_j = −2(ε_j − ε̂_j)  (gradient of ||ε - ε̂||²)
            let mut d_eps_hat = vec![0.0_f64; d];
            for j in 0..d {
                d_eps_hat[j] = -2.0 * (eps[j] - eps_hat[j]);
            }

            // Backprop layer 3 (linear): dL/dW3, dL/db3, dL/dh2
            let mut dw3 = vec![0.0_f64; d * h];
            let db3_grad = d_eps_hat.clone();
            let mut d_h2 = vec![0.0_f64; h];

            for o in 0..d {
                for i in 0..h {
                    dw3[o * h + i] = d_eps_hat[o] * h2[i];
                }
            }
            for i in 0..h {
                let mut acc = 0.0_f64;
                for o in 0..d {
                    acc += w3[o * h + i] * d_eps_hat[o];
                }
                d_h2[i] = acc;
            }

            // Backprop SiLU at layer 2: d_h2_pre = d_h2 · SiLU'(h2_pre)
            let mut d_h2_pre = vec![0.0_f64; h];
            for i in 0..h {
                d_h2_pre[i] = d_h2[i] * silu_grad(h2_pre[i]);
            }

            // Backprop layer 2: dL/dW2, dL/db2, dL/dh1
            let mut dw2 = vec![0.0_f64; h * h];
            let db2_grad = d_h2_pre.clone();
            let mut d_h1 = vec![0.0_f64; h];

            for o in 0..h {
                for i in 0..h {
                    dw2[o * h + i] = d_h2_pre[o] * h1[i];
                }
            }
            for i in 0..h {
                let mut acc = 0.0_f64;
                for o in 0..h {
                    acc += w2[o * h + i] * d_h2_pre[o];
                }
                d_h1[i] = acc;
            }

            // Backprop SiLU at layer 1
            let mut d_h1_pre = vec![0.0_f64; h];
            for i in 0..h {
                d_h1_pre[i] = d_h1[i] * silu_grad(h1_pre[i]);
            }

            // Backprop layer 1: dL/dW1, dL/db1
            let mut dw1 = vec![0.0_f64; h * input_aug];
            let db1_grad: Vec<f64> = d_h1_pre.clone();

            for o in 0..h {
                for i in 0..input_aug {
                    dw1[o * input_aug + i] = d_h1_pre[o] * x_aug[i];
                }
            }

            // SGD update
            let lr = cfg.lr;
            for i in 0..w1.len() {
                w1[i] -= lr * dw1[i];
            }
            for i in 0..b1.len() {
                b1[i] -= lr * db1_grad[i];
            }
            for i in 0..w2.len() {
                w2[i] -= lr * dw2[i];
            }
            for i in 0..b2.len() {
                b2[i] -= lr * db2_grad[i];
            }
            for i in 0..w3.len() {
                w3[i] -= lr * dw3[i];
            }
            for i in 0..b3.len() {
                b3[i] -= lr * db3_grad[i];
            }
        }
    }

    Ok(DiffusionAnomalyFit {
        score_net_w1: w1,
        score_net_b1: b1,
        score_net_w2: w2,
        score_net_b2: b2,
        score_net_w3: w3,
        score_net_b3: b3,
        alphas,
        alpha_bars,
        input_dim: d,
    })
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

/// Compute diffusion anomaly scores for `n` test samples.
///
/// Score = one-step denoising error at `t = T/4`:
/// `‖x − x̂‖²` where `x̂ = (x_noisy − √(1−ᾱ_{T/4}) · ε̂) / √ᾱ_{T/4}`
pub fn diffusion_anomaly_score(
    fit: &DiffusionAnomalyFit,
    x: &[f64],
    n: usize,
    rng: &mut LcgRng,
) -> AnomalyResult<Vec<f64>> {
    let d = fit.input_dim;
    if n == 0 {
        return Ok(Vec::new());
    }
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let n_steps = fit.alpha_bars.len();
    if n_steps == 0 {
        return Err(AnomalyError::Internal {
            msg: "alpha_bars is empty".into(),
        });
    }

    let h = fit.score_net_b1.len();
    let input_aug = d + 1;

    // Use t = T/4 (quarter diffusion)
    let t_idx = (n_steps / 4).min(n_steps - 1);
    let ab = fit.alpha_bars[t_idx];
    let sqrt_ab = ab.sqrt();
    let sqrt_one_minus_ab = (1.0 - ab).max(0.0).sqrt();
    let t_norm = (t_idx + 1) as f64 / n_steps as f64;

    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let x0 = &x[i * d..(i + 1) * d];

        // Sample noise ε ~ N(0, I)
        let eps: Vec<f64> = (0..d).map(|_| rng.next_normal() as f64).collect();

        // Noisy sample
        let x_noisy: Vec<f64> = x0
            .iter()
            .zip(eps.iter())
            .map(|(x0j, ej)| sqrt_ab * x0j + sqrt_one_minus_ab * ej)
            .collect();

        // Augment with timestep
        let mut x_aug = Vec::with_capacity(input_aug);
        x_aug.extend_from_slice(&x_noisy);
        x_aug.push(t_norm);

        // Forward pass to get predicted noise ε̂
        let (_, _, _, _, eps_hat) = score_net_forward(
            &x_aug,
            &fit.score_net_w1,
            &fit.score_net_b1,
            &fit.score_net_w2,
            &fit.score_net_b2,
            &fit.score_net_w3,
            &fit.score_net_b3,
            input_aug,
            h,
            d,
        );

        // Denoised estimate: x̂ = (x_noisy − √(1−ᾱ_t) · ε̂) / √ᾱ_t
        let mut x_denoised = vec![0.0_f64; d];
        if sqrt_ab > 1e-12 {
            for j in 0..d {
                x_denoised[j] = (x_noisy[j] - sqrt_one_minus_ab * eps_hat[j]) / sqrt_ab;
            }
        }
        // Degenerate case: fully noised → treat as zero reconstruction error
        // (denoised stays as x0 so score = 0, conservative for outliers)

        // Score = ‖x_0 − x̂‖²
        let score: f64 = x0
            .iter()
            .zip(x_denoised.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        scores.push(score);
    }

    Ok(scores)
}

/// Predict anomaly labels (true = anomaly) using a reconstruction-error threshold.
pub fn diffusion_anomaly_predict(
    fit: &DiffusionAnomalyFit,
    x: &[f64],
    n: usize,
    threshold: f64,
    rng: &mut LcgRng,
) -> AnomalyResult<Vec<bool>> {
    let scores = diffusion_anomaly_score(fit, x, n, rng)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cfg(d: usize) -> DiffusionAnomalyConfig {
        DiffusionAnomalyConfig {
            input_dim: d,
            hidden_dim: 16,
            n_steps: 20,
            n_epochs: 2,
            lr: 1e-3,
            beta_start: 1e-4,
            beta_end: 0.02,
        }
    }

    #[test]
    fn test_noise_schedule_length() {
        let (alphas, alpha_bars) = build_noise_schedule(50, 1e-4, 0.02);
        assert_eq!(alphas.len(), 50);
        assert_eq!(alpha_bars.len(), 50);
    }

    #[test]
    fn test_noise_schedule_monotone() {
        let (_, alpha_bars) = build_noise_schedule(50, 1e-4, 0.02);
        for i in 1..alpha_bars.len() {
            assert!(
                alpha_bars[i] <= alpha_bars[i - 1] + 1e-12,
                "alpha_bars[{}]={} > alpha_bars[{}]={}",
                i,
                alpha_bars[i],
                i - 1,
                alpha_bars[i - 1]
            );
        }
    }

    #[test]
    fn test_silu_at_zero() {
        let v = silu(0.0);
        assert!(v.abs() < 1e-12, "silu(0)={v}");
    }

    #[test]
    fn test_silu_positive_for_positive() {
        let v = silu(1.0);
        assert!(v > 0.0, "silu(1.0)={v}");
    }

    #[test]
    fn test_fit_returns_correct_input_dim() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..10 * d).map(|i| i as f64 * 0.1).collect();
        let fit =
            diffusion_anomaly_fit(&x, 10, &cfg, 42).expect("fit on valid data should succeed");
        assert_eq!(fit.input_dim, d);
    }

    #[test]
    fn test_fit_score_network_dims() {
        let d = 6_usize;
        let h = 16_usize;
        let cfg = DiffusionAnomalyConfig {
            input_dim: d,
            hidden_dim: h,
            ..simple_cfg(d)
        };
        let x: Vec<f64> = (0..8 * d).map(|i| i as f64 * 0.05).collect();
        let fit = diffusion_anomaly_fit(&x, 8, &cfg, 7).expect("fit on valid data should succeed");
        assert_eq!(fit.score_net_w1.len(), h * (d + 1));
        assert_eq!(fit.score_net_b1.len(), h);
        assert_eq!(fit.score_net_w2.len(), h * h);
        assert_eq!(fit.score_net_b2.len(), h);
        assert_eq!(fit.score_net_w3.len(), d * h);
        assert_eq!(fit.score_net_b3.len(), d);
    }

    #[test]
    fn test_fit_alpha_bars_all_in_01() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..10 * d).map(|i| i as f64 * 0.1).collect();
        let fit =
            diffusion_anomaly_fit(&x, 10, &cfg, 99).expect("fit on valid data should succeed");
        for &ab in &fit.alpha_bars {
            assert!((0.0..=1.0).contains(&ab), "alpha_bar out of [0,1]: {ab}");
        }
    }

    #[test]
    fn test_score_output_length() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let train: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.05).collect();
        let fit = diffusion_anomaly_fit(&train, 20, &cfg, 1).expect("fit should succeed");
        let test: Vec<f64> = (0..5 * d).map(|i| i as f64 * 0.1).collect();
        let mut rng = LcgRng::new(11);
        let scores = diffusion_anomaly_score(&fit, &test, 5, &mut rng)
            .expect("score should succeed after fit");
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_scores_finite() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let train: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.05).collect();
        let fit = diffusion_anomaly_fit(&train, 20, &cfg, 2).expect("fit should succeed");
        let test: Vec<f64> = (0..5 * d).map(|i| i as f64 * 0.1).collect();
        let mut rng = LcgRng::new(22);
        let scores = diffusion_anomaly_score(&fit, &test, 5, &mut rng)
            .expect("score should succeed after fit");
        for &s in &scores {
            assert!(s.is_finite(), "score not finite: {s}");
        }
    }

    #[test]
    fn test_scores_non_negative() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let train: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.05).collect();
        let fit = diffusion_anomaly_fit(&train, 20, &cfg, 3).expect("fit should succeed");
        let test: Vec<f64> = (0..5 * d).map(|i| i as f64 * 0.1).collect();
        let mut rng = LcgRng::new(33);
        let scores = diffusion_anomaly_score(&fit, &test, 5, &mut rng)
            .expect("score should succeed after fit");
        for &s in &scores {
            assert!(s >= 0.0, "score negative: {s}");
        }
    }

    #[test]
    fn test_predict_returns_correct_length() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let train: Vec<f64> = (0..20 * d).map(|i| i as f64 * 0.05).collect();
        let fit = diffusion_anomaly_fit(&train, 20, &cfg, 4).expect("fit should succeed");
        let test: Vec<f64> = (0..7 * d).map(|i| i as f64 * 0.1).collect();
        let mut rng = LcgRng::new(44);
        let preds = diffusion_anomaly_predict(&fit, &test, 7, 1.0, &mut rng)
            .expect("predict should succeed after fit");
        assert_eq!(preds.len(), 7);
    }

    #[test]
    fn test_error_on_wrong_length() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = vec![0.0; 10 * d];
        let fit = diffusion_anomaly_fit(&x, 10, &cfg, 5).expect("fit on valid data should succeed");
        let bad_x = vec![0.0_f64; 3]; // wrong length for n=1, d=4
        let mut rng = LcgRng::new(55);
        let res = diffusion_anomaly_score(&fit, &bad_x, 1, &mut rng);
        assert!(res.is_err(), "expected error on wrong-length input");
    }

    #[test]
    fn test_error_on_empty_input() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x = vec![];
        let res = diffusion_anomaly_fit(&x, 0, &cfg, 6);
        assert!(res.is_err(), "expected EmptyInput error");
    }

    #[test]
    fn test_alphas_schedule_count() {
        let d = 4_usize;
        let cfg = simple_cfg(d);
        let x: Vec<f64> = (0..10 * d).map(|i| i as f64 * 0.1).collect();
        let fit = diffusion_anomaly_fit(&x, 10, &cfg, 8).expect("fit should succeed");
        assert_eq!(fit.alphas.len(), cfg.n_steps);
        assert_eq!(fit.alpha_bars.len(), cfg.n_steps);
    }

    #[test]
    fn test_diffusion_higher_score_for_ood() {
        // Normal data: clustered near 0.1
        // OOD data: far from training distribution (100.0)
        let d = 4_usize;
        let cfg = DiffusionAnomalyConfig {
            input_dim: d,
            hidden_dim: 16,
            n_steps: 40,
            n_epochs: 10,
            lr: 5e-4,
            beta_start: 1e-4,
            beta_end: 0.02,
        };
        let train: Vec<f64> = (0..50 * d).map(|_| 0.1).collect();
        let fit = diffusion_anomaly_fit(&train, 50, &cfg, 77)
            .expect("fit on normal training data should succeed");

        let normal: Vec<f64> = vec![0.1; d];
        let outlier: Vec<f64> = vec![100.0; d];

        // Average scores over multiple noise draws for stability
        let n_trials = 5_usize;
        let mut normal_sum = 0.0_f64;
        let mut outlier_sum = 0.0_f64;
        let mut rng = LcgRng::new(88);
        for _ in 0..n_trials {
            let sn = diffusion_anomaly_score(&fit, &normal, 1, &mut rng)
                .expect("normal score should succeed in OOD trial");
            let so = diffusion_anomaly_score(&fit, &outlier, 1, &mut rng)
                .expect("outlier score should succeed in OOD trial");
            normal_sum += sn[0];
            outlier_sum += so[0];
        }
        assert!(
            outlier_sum > normal_sum,
            "outlier avg {:.4} should exceed normal avg {:.4}",
            outlier_sum / n_trials as f64,
            normal_sum / n_trials as f64
        );
    }
}
