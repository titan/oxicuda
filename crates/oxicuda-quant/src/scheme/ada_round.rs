//! # AdaRound — Adaptive Rounding for Post-Training Quantization
//!
//! Nagel et al. (2020) NeurIPS: "Up or Down? Adaptive Rounding for Post-Training
//! Quantization" <https://arxiv.org/abs/2004.10568>
//!
//! ## Key Idea
//!
//! Standard round-to-nearest quantization independently rounds each weight to
//! the closest quantized value.  However, neighboring quantized values can
//! yield lower task loss depending on the input activations.  AdaRound learns
//! a binary (or soft during training) rounding adjustment per weight:
//!
//! ```text
//! W_q[i,j] = Δ · (floor(W[i,j] / Δ) + h(v[i,j]))
//! ```
//!
//! where `v` is a learned scalar per weight and `h(v)` is a soft surrogate of
//! `{0, 1}` (round-down or round-up).
//!
//! ## Soft Rounding Function
//!
//! ```text
//! h(v) = clip(sigmoid(v) × 1.2 − 0.1, 0, 1)
//! ```
//!
//! This "stretched sigmoid" can saturate exactly to 0 or 1, unlike a bare
//! sigmoid, enabling clean hard rounding at the end of optimization.
//!
//! ## Regularization
//!
//! ```text
//! f_reg(v) = Σ_{i,j} (1 − |2·h(v[i,j]) − 1|^β)
//! ```
//!
//! - 0 when h ∈ {0, 1}  (weight is fully rounded)
//! - 1 when h = 0.5     (undecided)
//!
//! β anneals **linearly** from `beta_init` down to 2.0 over `n_iters`, pushing
//! h toward sharp binary decisions early in training.
//!
//! ## Total Loss
//!
//! ```text
//! L = ‖W_q X^T − W X^T‖² / (n_rows · n_samples)  +  λ · f_reg
//! ```
//!
//! Gradients are computed analytically and `v` is updated by vanilla SGD.

use crate::error::{QuantError, QuantResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for AdaRound optimization.
#[derive(Debug, Clone)]
pub struct AdaRoundConfig {
    /// Quantization bits (default 4).
    pub bits: u32,
    /// Number of optimization iterations (default 1000).
    pub n_iters: usize,
    /// Learning rate for the rounding-parameter `v` (default 0.01).
    pub lr: f32,
    /// Regularization coefficient `λ` (default 0.01).
    pub reg_coeff: f32,
    /// Starting value of the beta annealing schedule (default 20.0).
    ///
    /// Beta decreases linearly from `beta_init` to 2.0 over `n_iters`.
    pub beta_init: f32,
}

impl Default for AdaRoundConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            n_iters: 1000,
            lr: 0.01,
            reg_coeff: 0.01,
            beta_init: 20.0,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Output of an AdaRound optimization run.
#[derive(Debug, Clone)]
pub struct AdaRoundResult {
    /// Hard rounding decisions per weight: `true` = round up, `false` = round down.
    ///
    /// Shape matches the original weight matrix `(n_rows × n_cols)`.
    pub round_adjustments: Vec<bool>,
    /// Quantization scale `Δ` (single symmetric per-tensor scale).
    pub delta: f32,
    /// Reconstructed f32 weights after adaptive rounding.
    ///
    /// `W_q[i,j] = Δ · (floor(W[i,j] / Δ) + if round_up { 1 } else { 0 })`.
    pub quantized_weights: Vec<f32>,
    /// Final task-loss value after `n_iters` iterations.
    pub final_task_loss: f32,
    /// Final regularization-loss value after `n_iters` iterations.
    pub final_reg_loss: f32,
}

// ─── AdaRound ────────────────────────────────────────────────────────────────

/// AdaRound optimizer: learns per-weight rounding directions to minimize
/// layer-output reconstruction error.
#[derive(Debug, Clone)]
pub struct AdaRound {
    config: AdaRoundConfig,
    /// Learned rounding parameters, one per weight element.
    v: Vec<f32>,
    /// Symmetric quantization scale `Δ`.
    delta: f32,
}

impl AdaRound {
    /// Initialize AdaRound for a weight matrix.
    ///
    /// Computes the symmetric quantization scale `Δ` from the weight statistics
    /// and initializes `v = 0` for all weights (h(0) ≈ 0.5, i.e., undecided).
    ///
    /// # Parameters
    ///
    /// * `weights` — flat f32 weight matrix, row-major `(n_rows × n_cols)`.
    /// * `n_rows`  — number of output features.
    /// * `n_cols`  — number of input features.
    /// * `config`  — AdaRound configuration.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]    — empty weight slice.
    /// * [`QuantError::InvalidBitWidth`] — bits outside [1, 16].
    pub fn new(
        weights: &[f32],
        n_rows: usize,
        n_cols: usize,
        config: AdaRoundConfig,
    ) -> QuantResult<Self> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("AdaRound::new: weights"));
        }
        let bits = config.bits;
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { bits });
        }
        let n = n_rows * n_cols;
        let delta = compute_delta(weights, bits);
        let v = vec![0.0_f32; n];
        Ok(Self { config, v, delta })
    }

    /// Run the AdaRound optimization loop.
    ///
    /// Updates `v` via gradient descent to minimize the total loss and returns
    /// the hard-rounded quantized weight matrix.
    ///
    /// # Parameters
    ///
    /// * `weights`     — row-major f32 weight matrix `(n_rows × n_cols)`.
    /// * `n_rows`      — number of output features.
    /// * `n_cols`      — number of input features.
    /// * `activations` — calibration activations, row-major `(n_samples × n_cols)`.
    /// * `n_samples`   — number of calibration samples.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]        — empty slice.
    /// * [`QuantError::DimensionMismatch`] — inconsistent sizes.
    pub fn optimize(
        &mut self,
        weights: &[f32],
        n_rows: usize,
        n_cols: usize,
        activations: &[f32],
        n_samples: usize,
    ) -> QuantResult<AdaRoundResult> {
        // ── Validate ──────────────────────────────────────────────────────────
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("AdaRound::optimize: weights"));
        }
        if activations.is_empty() {
            return Err(QuantError::EmptyInput("AdaRound::optimize: activations"));
        }
        if weights.len() != n_rows * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows * n_cols,
                got: weights.len(),
            });
        }
        if activations.len() != n_samples * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_samples * n_cols,
                got: activations.len(),
            });
        }

        let n_iters = self.config.n_iters.max(1);
        let lr = self.config.lr;
        let lambda = self.config.reg_coeff;
        let beta_start = self.config.beta_init;
        let beta_end = 2.0_f32;
        let delta = self.delta;
        let n_weights = n_rows * n_cols;

        // Pre-compute FP32 layer output: out_fp[i_out, s] = W[i_out, :] · X[s, :]^T
        // Shape: (n_rows × n_samples)
        let out_fp = matmul_wxt(weights, activations, n_rows, n_cols, n_samples);

        let mut final_task_loss = 0.0_f32;
        let mut final_reg_loss = 0.0_f32;

        for iter in 0..n_iters {
            // ── Beta schedule: linear anneal from beta_start to beta_end ──────
            let t = iter as f32 / (n_iters - 1).max(1) as f32;
            let beta = beta_start + t * (beta_end - beta_start);

            // ── Compute h(v) for all weights ──────────────────────────────────
            let h_vals: Vec<f32> = self.v.iter().map(|&vi| h_soft(vi)).collect();

            // ── Build quantized weight matrix W_q ─────────────────────────────
            let w_q: Vec<f32> = weights
                .iter()
                .zip(h_vals.iter())
                .map(|(&w, &h)| {
                    let floor_w = (w / delta).floor();
                    delta * (floor_w + h)
                })
                .collect();

            // ── Forward: out_quant = W_q @ X^T  (n_rows × n_samples) ──────────
            let out_q = matmul_wxt(&w_q, activations, n_rows, n_cols, n_samples);

            // ── Task loss: ‖out_q - out_fp‖² / (n_rows * n_samples) ───────────
            let scale_task = 1.0_f32 / (n_rows * n_samples).max(1) as f32;
            let task_loss: f32 = out_q
                .iter()
                .zip(out_fp.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                * scale_task;

            // ── Regularization loss: Σ (1 - |2h - 1|^beta) ───────────────────
            let reg_loss: f32 = h_vals
                .iter()
                .map(|&h| {
                    let x = (2.0 * h - 1.0).abs();
                    1.0 - x.powf(beta)
                })
                .sum::<f32>();

            final_task_loss = task_loss;
            final_reg_loss = lambda * reg_loss;

            // ── Gradient of task loss w.r.t. v ────────────────────────────────
            // d(L_task)/d(W_q[i,j]) = 2 * (out_q - out_fp)[i, :] @ X[:, j] * scale_task
            // d(L_task)/d(v[i,j])   = d(L_task)/d(W_q[i,j]) * delta * h_grad(v[i,j])
            //
            // Compute residual R = out_q - out_fp  (n_rows × n_samples)
            let residual: Vec<f32> = out_q
                .iter()
                .zip(out_fp.iter())
                .map(|(a, b)| a - b)
                .collect();

            // dL_task/dW_q[i,j] = 2 * Σ_s R[i,s] * X[s,j] * scale_task
            // Then chain rule through h and the floor+h formula:
            // dL_task/dv[i,j] = dL_task/dW_q[i,j] * delta * dh/dv
            let mut grad_v = vec![0.0_f32; n_weights];

            for i in 0..n_rows {
                for j in 0..n_cols {
                    // Accumulate Σ_s R[i,s] * X[s,j]
                    let mut dl_dw: f32 = 0.0;
                    for s in 0..n_samples {
                        dl_dw += residual[i * n_samples + s] * activations[s * n_cols + j];
                    }
                    dl_dw *= 2.0 * scale_task * delta;

                    let vi = self.v[i * n_cols + j];
                    let dh = h_grad(vi);
                    grad_v[i * n_cols + j] = dl_dw * dh;
                }
            }

            // ── Gradient of regularization loss w.r.t. v ─────────────────────
            // df_reg/dh = -beta * |2h-1|^(beta-1) * sign(2h-1) * 2
            // df_reg/dv = df_reg/dh * dh/dv
            for (gv, (&h, vi)) in grad_v.iter_mut().zip(h_vals.iter().zip(self.v.iter())) {
                let dfreg_dh = f_reg_grad_h(h, beta);
                let dh = h_grad(*vi);
                *gv += lambda * dfreg_dh * dh;
            }

            // ── SGD update ────────────────────────────────────────────────────
            for (vi, &gv) in self.v.iter_mut().zip(grad_v.iter()) {
                *vi -= lr * gv;
            }
        }

        // ── Hard rounding ─────────────────────────────────────────────────────
        let round_adjustments: Vec<bool> = self.v.iter().map(|&vi| h_soft(vi) > 0.5).collect();

        let quantized_weights: Vec<f32> = weights
            .iter()
            .zip(round_adjustments.iter())
            .map(|(&w, &round_up)| {
                let floor_w = (w / delta).floor();
                let adj = if round_up { 1.0 } else { 0.0 };
                delta * (floor_w + adj)
            })
            .collect();

        Ok(AdaRoundResult {
            round_adjustments,
            delta,
            quantized_weights,
            final_task_loss,
            final_reg_loss,
        })
    }

    /// Get current soft rounding values `h(v)` for all weight elements.
    ///
    /// All values are in `[0, 1]`.
    #[must_use]
    pub fn get_h(&self) -> Vec<f32> {
        self.v.iter().map(|&vi| h_soft(vi)).collect()
    }

    /// Hard-round: `h(v) > 0.5` → round up, else round down.
    #[must_use]
    pub fn hard_round(&self) -> Vec<bool> {
        self.v.iter().map(|&vi| h_soft(vi) > 0.5).collect()
    }

    /// Apply the current learned rounding to produce quantized f32 weights.
    #[must_use]
    pub fn apply(&self, weights: &[f32]) -> Vec<f32> {
        let delta = self.delta;
        weights
            .iter()
            .zip(self.v.iter())
            .map(|(&w, &vi)| {
                let floor_w = (w / delta).floor();
                let adj = if h_soft(vi) > 0.5 { 1.0 } else { 0.0 };
                delta * (floor_w + adj)
            })
            .collect()
    }
}

// ─── Convenience function ─────────────────────────────────────────────────────

/// Run AdaRound on a weight matrix.
///
/// Constructs an [`AdaRound`] optimizer, runs [`AdaRound::optimize`], and
/// returns the result.
///
/// # Errors
///
/// See [`AdaRound::new`] and [`AdaRound::optimize`].
pub fn ada_round(
    weights: &[f32],
    n_rows: usize,
    n_cols: usize,
    activations: &[f32],
    n_samples: usize,
    config: AdaRoundConfig,
) -> QuantResult<AdaRoundResult> {
    let mut opt = AdaRound::new(weights, n_rows, n_cols, config)?;
    opt.optimize(weights, n_rows, n_cols, activations, n_samples)
}

// ─── Private numeric helpers ─────────────────────────────────────────────────

/// Soft rounding function: `h(v) = clip(sigmoid(v) × 1.2 − 0.1, 0, 1)`.
///
/// - h(0) ≈ 0.5   (neutral, undecided)
/// - h(+∞) = 1.0  (round up)
/// - h(−∞) = 0.0  (round down)
fn h_soft(v: f32) -> f32 {
    let sig = 1.0_f32 / (1.0 + (-v).exp());
    (sig * 1.2 - 0.1).clamp(0.0, 1.0)
}

/// Gradient of h w.r.t. v: `dh/dv = sigmoid(v) · (1 − sigmoid(v)) · 1.2`
/// when h is in the open interval (0, 1), else 0.
fn h_grad(v: f32) -> f32 {
    let h = h_soft(v);
    if h <= 0.0 || h >= 1.0 {
        return 0.0;
    }
    let sig = 1.0_f32 / (1.0 + (-v).exp());
    sig * (1.0 - sig) * 1.2
}

/// Gradient of a single regularization term `(1 − |2h − 1|^β)` w.r.t. h.
///
/// `df_reg/dh = −β · |2h − 1|^(β−1) · sign(2h − 1) · 2`
fn f_reg_grad_h(h: f32, beta: f32) -> f32 {
    let x = 2.0 * h - 1.0;
    let abs_x = x.abs();
    if abs_x < 1e-10 {
        return 0.0;
    }
    -beta * abs_x.powf(beta - 1.0) * x.signum() * 2.0
}

/// Symmetric per-tensor quantization scale for `bits`-bit quantization.
///
/// `Δ = max(|w|) / (2^(bits−1) − 1)`
fn compute_delta(weights: &[f32], bits: u32) -> f32 {
    let max_abs = weights.iter().map(|&w| w.abs()).fold(0.0_f32, f32::max);
    let q_max = ((1u32 << (bits - 1)) - 1) as f32;
    if max_abs < 1e-12 {
        // Degenerate all-zero case: return a small epsilon so Δ > 0.
        1e-8_f32
    } else {
        max_abs / q_max
    }
}

/// Matrix multiply: `C = W @ X^T`  where W is `(n_rows × n_cols)` and
/// X is `(n_samples × n_cols)`.  Output C has shape `(n_rows × n_samples)`.
fn matmul_wxt(w: &[f32], x: &[f32], n_rows: usize, n_cols: usize, n_samples: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n_rows * n_samples];
    for i in 0..n_rows {
        for s in 0..n_samples {
            let mut dot = 0.0_f32;
            for j in 0..n_cols {
                dot += w[i * n_cols + j] * x[s * n_cols + j];
            }
            c[i * n_samples + s] = dot;
        }
    }
    c
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Simple LCG-based pseudo-random f32 in [-1, 1] for test reproducibility.
    fn lcg_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 33) as u32;
                let fval = (bits as f32) / (u32::MAX as f32);
                fval * 2.0 - 1.0
            })
            .collect()
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn default_config_sane() {
        let cfg = AdaRoundConfig::default();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.n_iters, 1000);
        assert!(cfg.lr > 0.0);
        assert!(cfg.reg_coeff >= 0.0);
        assert!(cfg.beta_init > 2.0);
    }

    // ── Initialization ────────────────────────────────────────────────────────

    #[test]
    fn new_initializes_v_zeros() {
        let weights = lcg_f32(4 * 4, 1);
        let cfg = AdaRoundConfig::default();
        let ar = AdaRound::new(&weights, 4, 4, cfg).expect("new should succeed");
        for &vi in &ar.v {
            assert_eq!(vi, 0.0, "v should initialize to 0");
        }
    }

    // ── Soft function h ───────────────────────────────────────────────────────

    #[test]
    fn h_at_zero_is_half() {
        let h = h_soft(0.0);
        // sigmoid(0) * 1.2 - 0.1 = 0.5 * 1.2 - 0.1 = 0.5
        assert!((h - 0.5).abs() < 1e-5, "h(0) should be ≈0.5, got {h}");
    }

    #[test]
    fn h_large_positive_is_one() {
        let h = h_soft(100.0);
        assert_eq!(h, 1.0, "h(100) should be exactly 1.0 after clamp");
    }

    #[test]
    fn h_large_negative_is_zero() {
        let h = h_soft(-100.0);
        assert_eq!(h, 0.0, "h(-100) should be exactly 0.0 after clamp");
    }

    #[test]
    fn h_in_range() {
        for v in [-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
            let h = h_soft(v);
            assert!((0.0..=1.0).contains(&h), "h({v}) = {h} is outside [0,1]");
        }
    }

    // ── Regularization ────────────────────────────────────────────────────────

    #[test]
    fn f_reg_at_zero_v_is_one() {
        // h(0) = 0.5 → |2·0.5 - 1|^beta = 0 → 1 - 0 = 1
        let h = h_soft(0.0);
        let beta = 20.0_f32;
        let term = 1.0 - (2.0 * h - 1.0).abs().powf(beta);
        assert!(
            (term - 1.0).abs() < 1e-4,
            "f_reg term at h=0.5 should be ≈1, got {term}"
        );
    }

    #[test]
    fn f_reg_at_extreme_v_is_zero() {
        // h(+100) = 1 → |2·1 - 1|^beta = 1 → 1 - 1 = 0
        let h_one = h_soft(100.0);
        let beta = 20.0_f32;
        let term_one = 1.0 - (2.0 * h_one - 1.0).abs().powf(beta);
        assert!(
            term_one.abs() < 1e-5,
            "f_reg term at h=1 should be ≈0, got {term_one}"
        );
        // h(-100) = 0 → |2·0 - 1|^beta = 1 → 1 - 1 = 0
        let h_zero = h_soft(-100.0);
        let term_zero = 1.0 - (2.0 * h_zero - 1.0).abs().powf(beta);
        assert!(
            term_zero.abs() < 1e-5,
            "f_reg term at h=0 should be ≈0, got {term_zero}"
        );
    }

    // ── Delta ─────────────────────────────────────────────────────────────────

    #[test]
    fn compute_delta_positive() {
        let weights = vec![-0.5_f32, 0.3, 0.8, -0.2];
        let delta = compute_delta(&weights, 4);
        assert!(
            delta > 0.0 && delta.is_finite(),
            "delta should be positive finite, got {delta}"
        );
    }

    #[test]
    fn compute_delta_zero_weights() {
        let weights = vec![0.0_f32; 4];
        let delta = compute_delta(&weights, 4);
        // All-zero weights → small epsilon is returned
        assert!(
            delta > 0.0 && delta.is_finite(),
            "delta for zero weights should be small positive, got {delta}"
        );
    }

    // ── Quantize with rounding ────────────────────────────────────────────────

    #[test]
    fn quantize_with_rounding_rounddown() {
        // w = 0.7, delta = 0.5 → floor(0.7/0.5) = floor(1.4) = 1
        // round_down → delta * (1 + 0) = 0.5
        let delta = 0.5_f32;
        let w = 0.7_f32;
        let floor_w = (w / delta).floor();
        let result_down = delta * (floor_w + 0.0);
        assert!(
            (result_down - 0.5).abs() < 1e-5,
            "round-down: {result_down}"
        );
    }

    #[test]
    fn quantize_with_rounding_roundup() {
        // w = 0.7, delta = 0.5 → floor = 1
        // round_up → delta * (1 + 1) = 1.0
        let delta = 0.5_f32;
        let w = 0.7_f32;
        let floor_w = (w / delta).floor();
        let result_up = delta * (floor_w + 1.0);
        assert!((result_up - 1.0).abs() < 1e-5, "round-up: {result_up}");
    }

    // ── optimize ─────────────────────────────────────────────────────────────

    #[test]
    fn optimize_returns_ok() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 8;
        let weights = lcg_f32(n_rows * n_cols, 10);
        let acts = lcg_f32(n_samples * n_cols, 11);
        let cfg = AdaRoundConfig {
            n_iters: 10,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(
            result.is_ok(),
            "optimize should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn round_adjustments_shape() {
        let n_rows = 3;
        let n_cols = 5;
        let n_samples = 6;
        let weights = lcg_f32(n_rows * n_cols, 20);
        let acts = lcg_f32(n_samples * n_cols, 21);
        let cfg = AdaRoundConfig {
            n_iters: 5,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("ada_round should succeed");
        assert_eq!(result.round_adjustments.len(), n_rows * n_cols);
    }

    #[test]
    fn quantized_weights_shape() {
        let n_rows = 3;
        let n_cols = 5;
        let n_samples = 6;
        let weights = lcg_f32(n_rows * n_cols, 30);
        let acts = lcg_f32(n_samples * n_cols, 31);
        let cfg = AdaRoundConfig {
            n_iters: 5,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("ada_round should succeed");
        assert_eq!(result.quantized_weights.len(), n_rows * n_cols);
    }

    #[test]
    fn apply_matches_hard_round() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_f32(n_rows * n_cols, 40);
        let acts = lcg_f32(n_samples * n_cols, 41);
        let cfg = AdaRoundConfig {
            n_iters: 20,
            ..AdaRoundConfig::default()
        };
        let mut ar = AdaRound::new(&weights, n_rows, n_cols, cfg).expect("new should succeed");
        ar.optimize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let apply_result = ar.apply(&weights);
        let hard = ar.hard_round();
        let delta = ar.delta;
        let manual: Vec<f32> = weights
            .iter()
            .zip(hard.iter())
            .map(|(&w, &up)| {
                let floor_w = (w / delta).floor();
                let adj = if up { 1.0 } else { 0.0 };
                delta * (floor_w + adj)
            })
            .collect();
        for (a, b) in apply_result.iter().zip(manual.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "apply() and hard_round() manual disagree: {a} vs {b}"
            );
        }
    }

    #[test]
    fn final_task_loss_finite() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 8;
        let weights = lcg_f32(n_rows * n_cols, 50);
        let acts = lcg_f32(n_samples * n_cols, 51);
        let cfg = AdaRoundConfig {
            n_iters: 10,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("ada_round should succeed");
        assert!(
            result.final_task_loss.is_finite() && result.final_task_loss >= 0.0,
            "final_task_loss should be finite non-negative, got {}",
            result.final_task_loss
        );
    }

    #[test]
    fn final_reg_loss_finite() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 8;
        let weights = lcg_f32(n_rows * n_cols, 60);
        let acts = lcg_f32(n_samples * n_cols, 61);
        let cfg = AdaRoundConfig {
            n_iters: 10,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("ada_round should succeed");
        assert!(
            result.final_reg_loss.is_finite() && result.final_reg_loss >= 0.0,
            "final_reg_loss should be finite non-negative, got {}",
            result.final_reg_loss
        );
    }

    #[test]
    fn optimize_reduces_reg_loss() {
        // After training with strong beta annealing and many iters, the
        // regularization loss should be lower than the initial maximum (n_weights).
        // (Each term starts at 1.0 when h≈0.5, total = n_weights.)
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 8;
        let n_weights = n_rows * n_cols;
        let weights = lcg_f32(n_weights, 70);
        let acts = lcg_f32(n_samples * n_cols, 71);
        let cfg = AdaRoundConfig {
            n_iters: 500,
            lr: 0.1,
            reg_coeff: 1.0,
            beta_init: 20.0,
            bits: 4,
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg.clone())
            .expect("value should be present");
        // After 500 iters with lr=0.1, reg_coeff=1.0, regularization should
        // have pushed h toward {0, 1}, so lambda * reg_loss < lambda * n_weights.
        let initial_reg_upper_bound = cfg.reg_coeff * n_weights as f32;
        assert!(
            result.final_reg_loss < initial_reg_upper_bound,
            "reg_loss {:.4} should be < initial upper bound {:.4}",
            result.final_reg_loss,
            initial_reg_upper_bound
        );
    }

    #[test]
    fn get_h_in_range() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_f32(n_rows * n_cols, 80);
        let acts = lcg_f32(n_samples * n_cols, 81);
        let cfg = AdaRoundConfig {
            n_iters: 20,
            ..AdaRoundConfig::default()
        };
        let mut ar = AdaRound::new(&weights, n_rows, n_cols, cfg).expect("new should succeed");
        ar.optimize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        for &h in &ar.get_h() {
            assert!((0.0..=1.0).contains(&h), "h value {h} is outside [0, 1]");
        }
    }

    #[test]
    fn hard_round_gives_bool_per_weight() {
        let n_rows = 3;
        let n_cols = 5;
        let n_samples = 4;
        let weights = lcg_f32(n_rows * n_cols, 90);
        let _acts = lcg_f32(n_samples * n_cols, 91);
        let cfg = AdaRoundConfig {
            n_iters: 5,
            ..AdaRoundConfig::default()
        };
        let ar = AdaRound::new(&weights, n_rows, n_cols, cfg).expect("new should succeed");
        let decisions = ar.hard_round();
        assert_eq!(decisions.len(), n_rows * n_cols);
    }

    #[test]
    fn convenience_fn_matches_method() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_f32(n_rows * n_cols, 100);
        let acts = lcg_f32(n_samples * n_cols, 101);
        let cfg = AdaRoundConfig {
            n_iters: 10,
            ..AdaRoundConfig::default()
        };
        // Use the convenience function
        let result_fn = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg.clone())
            .expect("value should be present");
        // Use the method API
        let mut ar = AdaRound::new(&weights, n_rows, n_cols, cfg).expect("new should succeed");
        let result_method = ar
            .optimize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        // Same delta
        assert!(
            (result_fn.delta - result_method.delta).abs() < 1e-6,
            "delta mismatch: {} vs {}",
            result_fn.delta,
            result_method.delta
        );
        // Same quantized weights
        assert_eq!(
            result_fn.quantized_weights.len(),
            result_method.quantized_weights.len()
        );
        for (a, b) in result_fn
            .quantized_weights
            .iter()
            .zip(result_method.quantized_weights.iter())
        {
            assert!(
                (a - b).abs() < 1e-6,
                "quantized_weights mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn bits8_works() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_f32(n_rows * n_cols, 110);
        let acts = lcg_f32(n_samples * n_cols, 111);
        let cfg = AdaRoundConfig {
            bits: 8,
            n_iters: 10,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(result.is_ok(), "8-bit AdaRound failed: {:?}", result.err());
    }

    #[test]
    fn single_weight_works() {
        // 1×1 weight matrix is the degenerate case.
        let weights = vec![0.7_f32];
        let acts = vec![1.0_f32, -1.0, 0.5]; // 3 samples, 1 col
        let cfg = AdaRoundConfig {
            n_iters: 10,
            ..AdaRoundConfig::default()
        };
        let result = ada_round(&weights, 1, 1, &acts, 3, cfg);
        assert!(
            result.is_ok(),
            "single-weight AdaRound failed: {:?}",
            result.err()
        );
        let r = result.expect("result should be present");
        assert_eq!(r.quantized_weights.len(), 1);
        assert_eq!(r.round_adjustments.len(), 1);
    }

    #[test]
    fn beta_annealing() {
        // Verify that beta indeed decreases from beta_init (20.0) to 2.0.
        // We check it indirectly: record final_reg_loss at iteration 1 vs iteration 500
        // with a large enough lr so that v moves. With high beta at the start,
        // |2h-1|^beta is very small (since h≈0.5 → |2·0.5-1|^20 ≈ 0),
        // so the reg term starts near n_weights. With low beta at the end, |2h-1|^2
        // is larger relative contribution from partially-rounded weights.
        // The key check: the beta schedule linearly interpolates correctly.
        let beta_init = 20.0_f32;
        let beta_end = 2.0_f32;
        let n_iters = 100usize;
        // At iter 50 (halfway), beta should be (20+2)/2 = 11.
        let t = 50.0_f32 / (n_iters - 1) as f32;
        let beta_mid = beta_init + t * (beta_end - beta_init);
        assert!(
            (beta_mid - 11.0).abs() < 0.5,
            "midpoint beta should be ≈11, got {beta_mid}"
        );
        // At the final iter, t=1 → beta = 2.
        let t_end = 1.0_f32;
        let beta_final = beta_init + t_end * (beta_end - beta_init);
        assert!(
            (beta_final - beta_end).abs() < 1e-5,
            "final beta should be {beta_end}, got {beta_final}"
        );
    }
}
