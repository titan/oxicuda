//! Self-supervised anomaly detection via rotation-prediction pretext task.
//!
//! Adapted from Gidaris et al. 2018 "Unsupervised Representation Learning by
//! Predicting Image Rotations" for tabular anomaly detection.
//!
//! # Pretext task
//!
//! A shared encoder is trained to recognise which of `n_rotations` feature
//! transformations was applied to a sample.  Normal data exhibits consistent,
//! learnable transformation structure; anomalous data does not, leading to
//! high prediction uncertainty.
//!
//! # Tabular "rotations"
//!
//! | Label | Transformation |
//! |-------|----------------|
//! | 0     | Identity (original `x`) |
//! | 1     | Negate all features (`-x`) |
//! | 2     | Swap first / second half of features (requires `n_rotations = 4`) |
//! | 3     | Negate **and** swap halves (requires `n_rotations = 4`) |
//!
//! # Anomaly score
//!
//! The score for sample `x` is the entropy of the rotation-prediction
//! probability vector over the original (identity) view:
//! ```text
//! H(x) = −Σ_r p_r · log(p_r + ε)
//! ```
//! Higher entropy → more uncertain → more anomalous.
//!
//! An alternative "confidence gap" score is also exposed:
//! `1 − max_r(p_r)`.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Constants ───────────────────────────────────────────────────────────────

const EPS: f64 = 1e-12;

// ─── Xavier initialisation ───────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32() as f64;
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── Dense layer forward pass ─────────────────────────────────────────────────

fn dense(x: &[f64], w: &[f64], b: &[f64], fan_in: usize, fan_out: usize) -> Vec<f64> {
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

fn relu(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(EPS);
    for e in exps.iter_mut() {
        *e /= sum;
    }
    exps
}

// ─── Gradient helpers ─────────────────────────────────────────────────────────

fn relu_backward(out: &[f64], grad_out: &[f64]) -> Vec<f64> {
    out.iter()
        .zip(grad_out.iter())
        .map(|(&o, &g)| if o > 0.0 { g } else { 0.0 })
        .collect()
}

fn dense_backward(
    x_in: &[f64],
    w: &[f64],
    grad_out: &[f64],
    fan_in: usize,
    fan_out: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dw = vec![0.0_f64; fan_out * fan_in];
    for o in 0..fan_out {
        for i in 0..fan_in {
            dw[o * fan_in + i] = grad_out[o] * x_in[i];
        }
    }
    let db = grad_out.to_vec();
    let mut dx = vec![0.0_f64; fan_in];
    for o in 0..fan_out {
        for i in 0..fan_in {
            dx[i] += w[o * fan_in + i] * grad_out[o];
        }
    }
    (dw, db, dx)
}

fn sgd_update(params: &mut [f64], grad: &[f64], lr: f64) {
    for (p, &g) in params.iter_mut().zip(grad.iter()) {
        *p -= lr * g;
    }
}

// ─── Feature transformations ("rotations") ───────────────────────────────────

/// Apply transformation `r` to input `x`.
///
/// * 0 → identity
/// * 1 → negate
/// * 2 → swap first/second half
/// * 3 → negate + swap first/second half
fn apply_rotation(x: &[f64], r: usize) -> Vec<f64> {
    let d = x.len();
    let half = d / 2;
    match r {
        0 => x.to_vec(),
        1 => x.iter().map(|&v| -v).collect(),
        2 => {
            // Swap first half [0..half] and second half [half..d]
            let mut out = vec![0.0_f64; d];
            out[..half].copy_from_slice(&x[half..half + half]);
            out[half..half + half].copy_from_slice(&x[..half]);
            // If odd dimension, copy last element unchanged
            if !d.is_multiple_of(2) {
                out[d - 1] = x[d - 1];
            }
            out
        }
        3 => {
            // Negate + swap halves (iterator-based to avoid manual_memcpy lint)
            let second_half_negated = x[half..half + half].iter().map(|&v| -v);
            let first_half_negated = x[..half].iter().map(|&v| -v);
            let mut out: Vec<f64> = second_half_negated.chain(first_half_negated).collect();
            // If odd dimension, negate last element
            if !d.is_multiple_of(2) {
                out.push(-x[d - 1]);
            }
            out
        }
        _ => x.to_vec(),
    }
}

// ─── SelfSupervisedConfig ─────────────────────────────────────────────────────

/// Configuration for self-supervised anomaly detection.
#[derive(Debug, Clone)]
pub struct SelfSupervisedConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden layer width (encoder: `input_dim → hidden_dim → hidden_dim/2`).
    pub hidden_dim: usize,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Number of rotation classes: must be 2 or 4.
    pub n_rotations: usize,
}

impl Default for SelfSupervisedConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            n_epochs: 20,
            lr: 1e-3,
            n_rotations: 4,
        }
    }
}

// ─── SelfSupervisedFit ────────────────────────────────────────────────────────

/// Fitted self-supervised anomaly detection model.
///
/// Architecture:
/// ```text
/// x → [enc_w1, enc_b1] → ReLU → [enc_w2, enc_b2] → ReLU
///   → [head_w, head_b] → softmax → rotation probabilities
/// ```
///
/// Weights are stored flat row-major: `W[o * fan_in + i]`.
#[derive(Debug, Clone)]
pub struct SelfSupervisedFit {
    /// Encoder layer 1: shape `[hidden_dim, input_dim]`.
    pub enc_w1: Vec<f64>,
    /// Encoder layer 1 bias: shape `[hidden_dim]`.
    pub enc_b1: Vec<f64>,
    /// Encoder layer 2: shape `[hidden_dim/2, hidden_dim]`.
    pub enc_w2: Vec<f64>,
    /// Encoder layer 2 bias: shape `[hidden_dim/2]`.
    pub enc_b2: Vec<f64>,
    /// Classification head: shape `[n_rotations, hidden_dim/2]`.
    pub head_w: Vec<f64>,
    /// Classification head bias: shape `[n_rotations]`.
    pub head_b: Vec<f64>,
    /// Stored configuration.
    pub config: SelfSupervisedConfig,
}

// ─── Forward pass ────────────────────────────────────────────────────────────

/// Encode input → feature representation (shape `[hidden_dim/2]`).
fn encode_ss(fit: &SelfSupervisedFit, x: &[f64]) -> Vec<f64> {
    let cfg = &fit.config;
    let half = cfg.hidden_dim / 2;
    let h1 = relu(&dense(
        x,
        &fit.enc_w1,
        &fit.enc_b1,
        cfg.input_dim,
        cfg.hidden_dim,
    ));
    relu(&dense(&h1, &fit.enc_w2, &fit.enc_b2, cfg.hidden_dim, half))
}

/// Predict rotation probabilities for input `x` (shape `[n_rotations]`).
fn predict_rotation_probs(fit: &SelfSupervisedFit, x: &[f64]) -> Vec<f64> {
    let cfg = &fit.config;
    let half = cfg.hidden_dim / 2;
    let feat = encode_ss(fit, x);
    let logits = dense(&feat, &fit.head_w, &fit.head_b, half, cfg.n_rotations);
    softmax(&logits)
}

// ─── Cross-entropy loss and gradient ─────────────────────────────────────────

/// Cross-entropy loss for one sample and its true label `y`.
///
/// `loss = -log(probs[y] + eps)`
/// `d_loss/d_logit_r = probs[r] - (r == y)`  (softmax + CE combined gradient)
fn cross_entropy_grad(probs: &[f64], label: usize) -> Vec<f64> {
    probs
        .iter()
        .enumerate()
        .map(|(r, &p)| p - if r == label { 1.0 } else { 0.0 })
        .collect()
}

// ─── Training ────────────────────────────────────────────────────────────────

/// Train a self-supervised rotation-prediction anomaly detector.
///
/// `x` is a flat row-major matrix of shape `[n, input_dim]`.
pub fn self_supervised_fit(
    x: &[f64],
    n: usize,
    cfg: &SelfSupervisedConfig,
    seed: u64,
) -> AnomalyResult<SelfSupervisedFit> {
    // --- Validate ---
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "input_dim and hidden_dim must be > 0".into(),
        });
    }
    if cfg.hidden_dim < 2 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "hidden_dim must be >= 2 (to allow hidden_dim/2 >= 1)".into(),
        });
    }
    if cfg.n_rotations != 2 && cfg.n_rotations != 4 {
        return Err(AnomalyError::InvalidLayerDims {
            msg: "n_rotations must be 2 or 4".into(),
        });
    }
    if n == 0 {
        return Err(AnomalyError::InsufficientSamples { need: 1, got: 0 });
    }
    if x.len() != n * cfg.input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * cfg.input_dim,
            got: x.len(),
        });
    }

    let half = cfg.hidden_dim / 2;
    let mut rng = LcgRng::new(seed);

    // --- Initialise weights ---
    let enc_w1 = xavier_init(cfg.input_dim, cfg.hidden_dim, &mut rng);
    let enc_b1 = vec![0.0_f64; cfg.hidden_dim];
    let enc_w2 = xavier_init(cfg.hidden_dim, half, &mut rng);
    let enc_b2 = vec![0.0_f64; half];
    let head_w = xavier_init(half, cfg.n_rotations, &mut rng);
    let head_b = vec![0.0_f64; cfg.n_rotations];

    let mut fit = SelfSupervisedFit {
        enc_w1,
        enc_b1,
        enc_w2,
        enc_b2,
        head_w,
        head_b,
        config: cfg.clone(),
    };

    let lr = cfg.lr;
    let input_dim = cfg.input_dim;
    let hidden_dim = cfg.hidden_dim;
    let n_rot = cfg.n_rotations;

    // --- Training loop ---
    for _epoch in 0..cfg.n_epochs {
        for s in 0..n {
            let xi = &x[s * input_dim..(s + 1) * input_dim];

            for r in 0..n_rot {
                let x_rot = apply_rotation(xi, r);

                // ── Encoder forward ──────────────────────────────────────────
                let enc_h1_pre = dense(&x_rot, &fit.enc_w1, &fit.enc_b1, input_dim, hidden_dim);
                let enc_h1 = relu(&enc_h1_pre);
                let enc_h2_pre = dense(&enc_h1, &fit.enc_w2, &fit.enc_b2, hidden_dim, half);
                let enc_h2 = relu(&enc_h2_pre);

                // ── Head forward (logits → softmax) ──────────────────────────
                let logits = dense(&enc_h2, &fit.head_w, &fit.head_b, half, n_rot);
                let probs = softmax(&logits);

                // ── Cross-entropy gradient w.r.t. logits ─────────────────────
                // Combined softmax + CE gradient: d_loss/d_logit_r = p_r - 1{r==label}
                let grad_logits = cross_entropy_grad(&probs, r);

                // ── Head backward ────────────────────────────────────────────
                let (dhw, dhb, grad_enc_h2) =
                    dense_backward(&enc_h2, &fit.head_w, &grad_logits, half, n_rot);

                // ── Encoder layer 2 backward ──────────────────────────────────
                let grad_enc_h2_pre = relu_backward(&enc_h2, &grad_enc_h2);
                let (dew2, deb2, grad_enc_h1) =
                    dense_backward(&enc_h1, &fit.enc_w2, &grad_enc_h2_pre, hidden_dim, half);

                // ── Encoder layer 1 backward ──────────────────────────────────
                let grad_enc_h1_pre = relu_backward(&enc_h1, &grad_enc_h1);
                let (dew1, deb1, _grad_x_rot) =
                    dense_backward(&x_rot, &fit.enc_w1, &grad_enc_h1_pre, input_dim, hidden_dim);

                // ── Parameter updates ─────────────────────────────────────────
                sgd_update(&mut fit.head_w, &dhw, lr);
                sgd_update(&mut fit.head_b, &dhb, lr);
                sgd_update(&mut fit.enc_w2, &dew2, lr);
                sgd_update(&mut fit.enc_b2, &deb2, lr);
                sgd_update(&mut fit.enc_w1, &dew1, lr);
                sgd_update(&mut fit.enc_b1, &deb1, lr);
            }
        }
    }

    Ok(fit)
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

/// Compute anomaly scores for `n` samples (flat row-major `[n, input_dim]`).
///
/// Score = prediction entropy on the identity view of `x`:
/// `H = -Σ_r p_r · log(p_r + ε)`.  Higher = more anomalous.
pub fn self_supervised_score(
    fit: &SelfSupervisedFit,
    x: &[f64],
    n: usize,
) -> AnomalyResult<Vec<f64>> {
    let input_dim = fit.config.input_dim;
    if n == 0 {
        return Err(AnomalyError::InsufficientSamples { need: 1, got: 0 });
    }
    if x.len() != n * input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * input_dim,
            got: x.len(),
        });
    }

    let mut scores = Vec::with_capacity(n);
    for s in 0..n {
        let xi = &x[s * input_dim..(s + 1) * input_dim];
        let probs = predict_rotation_probs(fit, xi);
        let entropy = -probs.iter().map(|&p| p * (p + EPS).ln()).sum::<f64>();
        scores.push(entropy);
    }
    Ok(scores)
}

/// Predict whether each of `n` samples is an anomaly (score > threshold).
pub fn self_supervised_predict(
    fit: &SelfSupervisedFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = self_supervised_score(fit, x, n)?;
    Ok(scores.into_iter().map(|s| s > threshold).collect())
}

/// Compute confidence-gap scores: `1 − max_r(p_r)`.
///
/// Higher = less confident = more anomalous.
pub fn self_supervised_confidence_gap(
    fit: &SelfSupervisedFit,
    x: &[f64],
    n: usize,
) -> AnomalyResult<Vec<f64>> {
    let input_dim = fit.config.input_dim;
    if n == 0 {
        return Err(AnomalyError::InsufficientSamples { need: 1, got: 0 });
    }
    if x.len() != n * input_dim {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * input_dim,
            got: x.len(),
        });
    }
    let mut scores = Vec::with_capacity(n);
    for s in 0..n {
        let xi = &x[s * input_dim..(s + 1) * input_dim];
        let probs = predict_rotation_probs(fit, xi);
        let max_p = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        scores.push(1.0 - max_p);
    }
    Ok(scores)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg_4() -> SelfSupervisedConfig {
        SelfSupervisedConfig {
            input_dim: 8,
            hidden_dim: 16,
            n_epochs: 5,
            lr: 1e-3,
            n_rotations: 4,
        }
    }

    fn default_cfg_2() -> SelfSupervisedConfig {
        SelfSupervisedConfig {
            n_rotations: 2,
            ..default_cfg_4()
        }
    }

    fn make_data(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim)
            .map(|_| 0.5 + (rng.next_f32() as f64) * 0.1)
            .collect()
    }

    // ── Test 1: Scores are finite (n_rotations=4) ────────────────────────────
    #[test]
    fn ss_scores_finite_n4() {
        let cfg = default_cfg_4();
        let n = 15_usize;
        let x = make_data(n, cfg.input_dim, 1);
        let fit = self_supervised_fit(&x, n, &cfg, 42).unwrap();
        let scores = self_supervised_score(&fit, &x, n).unwrap();
        assert_eq!(scores.len(), n);
        assert!(
            scores.iter().all(|&s| s.is_finite()),
            "scores not all finite"
        );
    }

    // ── Test 2: Scores are finite (n_rotations=2) ────────────────────────────
    #[test]
    fn ss_scores_finite_n2() {
        let cfg = default_cfg_2();
        let n = 10_usize;
        let x = make_data(n, cfg.input_dim, 2);
        let fit = self_supervised_fit(&x, n, &cfg, 7).unwrap();
        let scores = self_supervised_score(&fit, &x, n).unwrap();
        assert_eq!(scores.len(), n);
        assert!(scores.iter().all(|&s| s.is_finite()));
    }

    // ── Test 3: Entropy score in [0, log(n_rotations)] ───────────────────────
    #[test]
    fn ss_scores_in_entropy_range() {
        let cfg = default_cfg_4();
        let n = 20_usize;
        let x = make_data(n, cfg.input_dim, 3);
        let fit = self_supervised_fit(&x, n, &cfg, 1).unwrap();
        let scores = self_supervised_score(&fit, &x, n).unwrap();

        let max_entropy = (cfg.n_rotations as f64).ln();
        for &s in &scores {
            assert!(s >= -1e-9, "entropy should be >= 0 (got {s})");
            assert!(
                s <= max_entropy + 1e-9,
                "entropy should be <= ln({}) = {max_entropy:.4} (got {s})",
                cfg.n_rotations
            );
        }
    }

    // ── Test 4: predict returns correct boolean vector length ─────────────────
    #[test]
    fn ss_predict_length_correct() {
        let cfg = default_cfg_4();
        let n = 12_usize;
        let x = make_data(n, cfg.input_dim, 4);
        let fit = self_supervised_fit(&x, n, &cfg, 2).unwrap();
        let preds = self_supervised_predict(&fit, &x, n, 0.5).unwrap();
        assert_eq!(preds.len(), n);
    }

    // ── Test 5: DimensionMismatch on wrong input size ─────────────────────────
    #[test]
    fn ss_score_dim_mismatch_error() {
        let cfg = default_cfg_4();
        let n = 10_usize;
        let x = make_data(n, cfg.input_dim, 5);
        let fit = self_supervised_fit(&x, n, &cfg, 3).unwrap();

        // Pass wrong number of elements (only 3 instead of 8)
        let result = self_supervised_score(&fit, &[0.1, 0.2, 0.3], 1);
        assert!(
            matches!(result, Err(AnomalyError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    // ── Test 6: n_rotations=2 trains and scores correctly ────────────────────
    #[test]
    fn ss_n_rotations_2_works() {
        let cfg = default_cfg_2();
        let n = 20_usize;
        let x = make_data(n, cfg.input_dim, 6);
        let fit = self_supervised_fit(&x, n, &cfg, 4).unwrap();
        assert_eq!(fit.head_w.len(), 2 * (cfg.hidden_dim / 2));
        assert_eq!(fit.head_b.len(), 2);
        let scores = self_supervised_score(&fit, &x, n).unwrap();
        assert!(scores.iter().all(|&s| s.is_finite() && s >= -1e-9));
    }

    // ── Test 7: n_rotations=4 trains and scores correctly ────────────────────
    #[test]
    fn ss_n_rotations_4_works() {
        let cfg = default_cfg_4();
        let n = 20_usize;
        let x = make_data(n, cfg.input_dim, 7);
        let fit = self_supervised_fit(&x, n, &cfg, 5).unwrap();
        assert_eq!(fit.head_w.len(), 4 * (cfg.hidden_dim / 2));
        assert_eq!(fit.head_b.len(), 4);
        let scores = self_supervised_score(&fit, &x, n).unwrap();
        assert!(scores.iter().all(|&s| s.is_finite() && s >= -1e-9));
    }

    // ── Test 8: Predict flags high-entropy samples as anomalies ──────────────
    #[test]
    fn ss_predict_flags_anomalies() {
        let cfg = default_cfg_4();
        let n = 15_usize;
        let x = make_data(n, cfg.input_dim, 8);
        let fit = self_supervised_fit(&x, n, &cfg, 6).unwrap();

        // With threshold=0 (all entropy > 0) everything should be flagged
        let preds = self_supervised_predict(&fit, &x, n, 0.0).unwrap();
        let flagged = preds.iter().filter(|&&p| p).count();
        assert!(flagged > 0, "At least 1 sample should have entropy > 0");
    }

    // ── Test 9: apply_rotation is consistent (rotation 0 = identity) ─────────
    #[test]
    fn ss_rotation_0_is_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0_f64];
        let rot = apply_rotation(&x, 0);
        assert_eq!(rot, x);
    }

    // ── Test 10: apply_rotation 1 negates features ───────────────────────────
    #[test]
    fn ss_rotation_1_negates() {
        let x = vec![1.0, -2.0, 3.0, 0.5_f64];
        let rot = apply_rotation(&x, 1);
        let expected: Vec<f64> = x.iter().map(|&v| -v).collect();
        assert_eq!(rot, expected);
    }

    // ── Test 11: apply_rotation 2 swaps halves ────────────────────────────────
    #[test]
    fn ss_rotation_2_swaps_halves() {
        let x = vec![1.0, 2.0, 3.0, 4.0_f64];
        let rot = apply_rotation(&x, 2);
        assert_eq!(rot, vec![3.0, 4.0, 1.0, 2.0]);
    }

    // ── Test 12: Confidence-gap scores in [0, 1] ─────────────────────────────
    #[test]
    fn ss_confidence_gap_in_range() {
        let cfg = default_cfg_4();
        let n = 10_usize;
        let x = make_data(n, cfg.input_dim, 12);
        let fit = self_supervised_fit(&x, n, &cfg, 9).unwrap();
        let gaps = self_supervised_confidence_gap(&fit, &x, n).unwrap();
        for &g in &gaps {
            assert!(
                (-1e-9..=1.0 + 1e-9).contains(&g),
                "confidence gap {g} not in [0, 1]"
            );
        }
    }

    // ── Test 13: Error on zero samples in fit ─────────────────────────────────
    #[test]
    fn ss_fit_rejects_zero_samples() {
        let cfg = default_cfg_4();
        let result = self_supervised_fit(&[], 0, &cfg, 0);
        assert!(
            matches!(result, Err(AnomalyError::InsufficientSamples { .. })),
            "Expected InsufficientSamples"
        );
    }

    // ── Test 14: Rotation probs sum to 1 ─────────────────────────────────────
    #[test]
    fn ss_rotation_probs_sum_to_one() {
        let cfg = default_cfg_4();
        let n = 5_usize;
        let x = make_data(n, cfg.input_dim, 14);
        let fit = self_supervised_fit(&x, n, &cfg, 10).unwrap();

        for s in 0..n {
            let xi = &x[s * cfg.input_dim..(s + 1) * cfg.input_dim];
            let probs = predict_rotation_probs(&fit, xi);
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "probs sum = {sum:.6}, expected 1.0"
            );
            assert!(probs.iter().all(|&p| p >= 0.0), "prob < 0");
        }
    }
}
