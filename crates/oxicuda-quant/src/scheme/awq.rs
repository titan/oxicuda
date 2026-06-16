//! # AWQ — Activation-Aware Weight Quantization
//!
//! Lin et al. (2023) MLSys: "AWQ: Activation-aware Weight Quantization for LLM
//! Compression and Acceleration" <https://arxiv.org/abs/2306.00978>
//!
//! ## Key Idea
//!
//! Standard weight quantization clips small weights uniformly, losing precision
//! on *salient* (large-activation) channels. AWQ scales salient weight columns
//! before quantization to protect them.
//!
//! For a linear layer `y = W x` where W is `(n_rows × n_cols)`:
//! - Scale each input channel `i` by `s[i]`:
//!   `y = (W · diag(1/s)) · (diag(s) · x)`
//! - Quantize `W_scaled = W · diag(s)` (multiply each column i of W by `s[i]`).
//! - At inference: dequantize by dividing the group scale by `s[i]`.
//!
//! ## Scale Search (Grid Search over α)
//!
//! ```text
//! activation_scale[i] = mean(|X[:,i]|)        — per input channel importance
//! s = activation_scale ^ α                    — candidate scale
//! W_scaled = W .* s                           — broadcast over columns
//! W_quant  = minmax_quantize(W_scaled)        — per-group symmetric quantization
//! W_recon  = W_quant ./ s
//! err      = mean((W_recon - W)^2 * activation_scale^2)
//! ```
//!
//! Best α (min err) from grid `{0.00, 0.01, …, 1.00}` is retained.

use crate::error::{QuantError, QuantResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// AWQ quantizer configuration.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Number of quantization bits (2, 3, 4, or 8). Default: 4.
    pub bits: u32,
    /// Group size for per-group symmetric quantization. Default: 128.
    pub group_size: usize,
    /// Number of alpha values in search grid. Default: 101 (0.0..=1.0 by 0.01).
    pub n_alpha_steps: usize,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            n_alpha_steps: 101,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// Output of AWQ quantization, containing all data needed for deployment
/// and dequantization.
#[derive(Debug, Clone)]
pub struct AwqOutput {
    /// Dequantized weight codes in f32, row-major `(n_rows × n_cols)`.
    ///
    /// These are the reconstructed floating-point values of the quantized
    /// weight matrix after AWQ channel scaling has been applied.
    pub quantized_weights: Vec<f32>,
    /// Per-group scale factors `(n_groups,)` used for group-wise dequantization.
    ///
    /// `n_groups = n_rows * ceil(n_cols / group_size)`.
    pub scales: Vec<f32>,
    /// Per-input-channel activation-aware scales `s[i] = activation_scale[i]^alpha`.
    ///
    /// Length: `n_cols`.
    pub channel_scales: Vec<f32>,
    /// Best alpha found during grid search.
    pub best_alpha: f32,
    /// Weighted quantization error at best alpha.
    pub best_error: f32,
}

// ─── AwqQuantizer ────────────────────────────────────────────────────────────

/// AWQ quantizer performing activation-aware weight scaling and grid-search
/// alpha selection.
#[derive(Debug, Clone)]
pub struct AwqQuantizer {
    config: AwqConfig,
}

impl AwqQuantizer {
    /// Create a new AWQ quantizer with the supplied configuration.
    #[must_use]
    pub fn new(config: AwqConfig) -> Self {
        Self { config }
    }

    /// Compute per-input-channel activation importance: `mean(|X[:,i]|)` for each `i`.
    ///
    /// # Parameters
    ///
    /// * `activations` — row-major `(n_samples × n_cols)` activation matrix.
    /// * `n_samples`   — number of calibration samples (rows).
    /// * `n_cols`      — number of input channels (columns).
    ///
    /// # Returns
    ///
    /// A vector of length `n_cols` with the mean absolute activation per channel.
    #[must_use]
    pub fn compute_channel_scales(
        activations: &[f32],
        n_samples: usize,
        n_cols: usize,
    ) -> Vec<f32> {
        if n_samples == 0 || n_cols == 0 || activations.is_empty() {
            return vec![0.0_f32; n_cols];
        }
        let mut scales = vec![0.0_f32; n_cols];
        for s in 0..n_samples {
            let row = &activations[s * n_cols..(s * n_cols + n_cols).min(activations.len())];
            for (scale, &act) in scales.iter_mut().zip(row.iter()) {
                *scale += act.abs();
            }
        }
        let inv_n = 1.0_f32 / (n_samples as f32);
        for v in &mut scales {
            *v *= inv_n;
        }
        scales
    }

    /// Quantize weights using activation-aware scaling.
    ///
    /// # Parameters
    ///
    /// * `weights`     — row-major `(n_rows × n_cols)` f32 weight matrix.
    /// * `n_rows`      — output-feature count (rows of W).
    /// * `n_cols`      — input-feature count (columns of W).
    /// * `activations` — row-major `(n_samples × n_cols)` calibration activations.
    /// * `n_samples`   — number of calibration samples.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]    — empty weight or activation slice.
    /// * [`QuantError::DimensionMismatch`] — inconsistent lengths.
    /// * [`QuantError::InvalidBitWidth`]   — bits outside [1, 16].
    pub fn quantize(
        &self,
        weights: &[f32],
        n_rows: usize,
        n_cols: usize,
        activations: &[f32],
        n_samples: usize,
    ) -> QuantResult<AwqOutput> {
        // ── Validate inputs ───────────────────────────────────────────────────
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("AwqQuantizer::quantize: weights"));
        }
        if activations.is_empty() {
            return Err(QuantError::EmptyInput(
                "AwqQuantizer::quantize: activations",
            ));
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
        let bits = self.config.bits;
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { bits });
        }
        let group_size = self.config.group_size.max(1);

        // ── Step 1: Compute per-channel activation importance ─────────────────
        let act_scale = Self::compute_channel_scales(activations, n_samples, n_cols);

        // ── Step 2: Grid-search alpha ─────────────────────────────────────────
        let n_steps = self.config.n_alpha_steps.max(1);
        let mut best_alpha = 0.0_f32;
        let mut best_error = f32::INFINITY;

        for step in 0..n_steps {
            let alpha = if n_steps == 1 {
                0.0_f32
            } else {
                step as f32 / (n_steps - 1) as f32
            };

            // s[i] = act_scale[i]^alpha  (small act_scale gets s≈1 when alpha=0)
            let s: Vec<f32> = act_scale
                .iter()
                .map(|&a| {
                    let base = a.max(1e-6_f32);
                    base.powf(alpha)
                })
                .collect();

            // Scale columns of W by s
            let w_scaled = scale_columns(weights, n_rows, n_cols, &s);

            // Per-group symmetric minmax quantize → reconstructed f32
            let w_recon =
                per_group_minmax_quant_symmetric(&w_scaled, n_rows, n_cols, bits, group_size);

            // Unscale: divide each column j by s[j]
            let w_reconstructed = unscale_columns(&w_recon, n_rows, n_cols, &s);

            // Weighted error: mean((W_recon - W)^2 * activation_scale^2)
            let err = weighted_reconstruction_error(
                &w_reconstructed,
                weights,
                n_rows,
                n_cols,
                &act_scale,
            );

            if err < best_error {
                best_error = err;
                best_alpha = alpha;
            }
        }

        // ── Step 3: Apply best alpha and produce final quantization ───────────
        let best_s: Vec<f32> = act_scale
            .iter()
            .map(|&a| {
                let base = a.max(1e-6_f32);
                base.powf(best_alpha)
            })
            .collect();

        let w_scaled_final = scale_columns(weights, n_rows, n_cols, &best_s);

        // Per-group quantize and collect group scales
        let (quantized_weights, group_scales) =
            per_group_minmax_quant_with_scales(&w_scaled_final, n_rows, n_cols, bits, group_size);

        Ok(AwqOutput {
            quantized_weights,
            scales: group_scales,
            channel_scales: best_s,
            best_alpha,
            best_error,
        })
    }

    /// Dequantize: recover f32 weights from the AWQ output.
    ///
    /// The `quantized_weights` in `AwqOutput` hold reconstructed f32 values in the
    /// scaled weight space. This method divides each column j by `channel_scales[j]`
    /// to recover the original weight space approximation.
    ///
    /// # Parameters
    ///
    /// * `output`  — AWQ quantization output.
    /// * `n_rows`  — number of output features.
    /// * `n_cols`  — number of input features.
    #[must_use]
    pub fn dequantize(&self, output: &AwqOutput, n_rows: usize, n_cols: usize) -> Vec<f32> {
        // quantized_weights = Q(W * s) in scaled space
        // dequant = quantized_weights / channel_scale[j]   (undo AWQ scaling)
        unscale_columns(
            &output.quantized_weights,
            n_rows,
            n_cols,
            &output.channel_scales,
        )
    }
}

// ─── Convenience function ─────────────────────────────────────────────────────

/// Perform AWQ activation-aware weight quantization.
///
/// Equivalent to constructing an [`AwqQuantizer`] with `config` and calling
/// [`AwqQuantizer::quantize`].
///
/// # Errors
///
/// See [`AwqQuantizer::quantize`].
pub fn awq_quantize(
    weights: &[f32],
    n_rows: usize,
    n_cols: usize,
    activations: &[f32],
    n_samples: usize,
    config: AwqConfig,
) -> QuantResult<AwqOutput> {
    AwqQuantizer::new(config).quantize(weights, n_rows, n_cols, activations, n_samples)
}

// ─── Private numeric helpers ─────────────────────────────────────────────────

/// Multiply each column j of W `(n_rows × n_cols)` by `scales[j]` (row-major).
fn scale_columns(w: &[f32], n_rows: usize, n_cols: usize, scales: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            let s = if j < scales.len() { scales[j] } else { 1.0 };
            out.push(w[i * n_cols + j] * s);
        }
    }
    out
}

/// Divide each column j of W `(n_rows × n_cols)` by `scales[j]`.
fn unscale_columns(w: &[f32], n_rows: usize, n_cols: usize, scales: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            let s = if j < scales.len() {
                scales[j].max(1e-12_f32)
            } else {
                1.0_f32
            };
            out.push(w[i * n_cols + j] / s);
        }
    }
    out
}

/// Symmetric per-group minmax quantization; returns reconstructed f32 values.
///
/// Groups are formed over consecutive columns within each row.  For a weight
/// matrix of shape `(n_rows × n_cols)`, each row is split into groups of
/// `group_size` columns and quantized independently.
fn per_group_minmax_quant_symmetric(
    w: &[f32],
    n_rows: usize,
    n_cols: usize,
    bits: u32,
    group_size: usize,
) -> Vec<f32> {
    let q_max = ((1u32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0_f32; n_rows * n_cols];
    let gs = group_size.min(n_cols).max(1);

    for i in 0..n_rows {
        let row_off = i * n_cols;
        let mut col = 0usize;
        while col < n_cols {
            let end = (col + gs).min(n_cols);
            let group = &w[row_off + col..row_off + end];

            let abs_max = group
                .iter()
                .map(|&v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-8_f32);
            let scale = abs_max / q_max;

            for (k, &v) in group.iter().enumerate() {
                let q = (v / scale).round().clamp(-q_max - 1.0, q_max);
                out[row_off + col + k] = q * scale;
            }
            col += gs;
        }
    }
    out
}

/// Same as [`per_group_minmax_quant_symmetric`] but also collects one group
/// scale per `(row, group)` pair.
///
/// Returns `(reconstructed_weights, group_scales)`.
fn per_group_minmax_quant_with_scales(
    w: &[f32],
    n_rows: usize,
    n_cols: usize,
    bits: u32,
    group_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let q_max = ((1u32 << (bits - 1)) - 1) as f32;
    let gs = group_size.min(n_cols).max(1);
    let n_groups_per_row = n_cols.div_ceil(gs);
    let total_groups = n_rows * n_groups_per_row;

    let mut recon = vec![0.0_f32; n_rows * n_cols];
    let mut group_scales = Vec::with_capacity(total_groups);

    for i in 0..n_rows {
        let row_off = i * n_cols;
        let mut col = 0usize;
        while col < n_cols {
            let end = (col + gs).min(n_cols);
            let group = &w[row_off + col..row_off + end];

            let abs_max = group
                .iter()
                .map(|&v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-8_f32);
            let scale = abs_max / q_max;
            group_scales.push(scale);

            for (k, &v) in group.iter().enumerate() {
                let q = (v / scale).round().clamp(-q_max - 1.0, q_max);
                recon[row_off + col + k] = q * scale;
            }
            col += gs;
        }
    }
    (recon, group_scales)
}

/// Weighted reconstruction error: `mean((W_recon - W)^2 * activation_scale[j]^2)`.
///
/// The weight per column j is `activation_scale[j]^2`, reflecting how much
/// error on important input channels hurts output quality.
fn weighted_reconstruction_error(
    w_recon: &[f32],
    w_orig: &[f32],
    n_rows: usize,
    n_cols: usize,
    act_scale: &[f32],
) -> f32 {
    let n = (n_rows * n_cols).max(1) as f32;
    let mut err = 0.0_f32;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            let diff = w_recon[idx] - w_orig[idx];
            let w_j = if j < act_scale.len() {
                act_scale[j]
            } else {
                1.0_f32
            };
            err += diff * diff * w_j * w_j;
        }
    }
    err / n
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Simple LCG-based pseudo-random f32 in [-1, 1] for test reproducibility.
    fn lcg_weights(n: usize, seed: u64) -> Vec<f32> {
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

    fn mse(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().max(1) as f32;
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / n
    }

    fn variance(v: &[f32]) -> f32 {
        let n = v.len() as f32;
        if n < 2.0 {
            return 0.0;
        }
        let mean = v.iter().sum::<f32>() / n;
        v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n
    }

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn default_config_sane() {
        let cfg = AwqConfig::default();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
        assert_eq!(cfg.n_alpha_steps, 101);
    }

    // ── Channel scale tests ───────────────────────────────────────────────────

    #[test]
    fn compute_channel_scales_shape() {
        let acts = vec![1.0_f32; 10 * 8];
        let scales = AwqQuantizer::compute_channel_scales(&acts, 10, 8);
        assert_eq!(scales.len(), 8);
    }

    #[test]
    fn compute_channel_scales_nonneg() {
        let acts = lcg_weights(5 * 4, 42);
        let scales = AwqQuantizer::compute_channel_scales(&acts, 5, 4);
        for &s in &scales {
            assert!(s >= 0.0, "channel scale must be non-negative, got {s}");
        }
    }

    #[test]
    fn compute_channel_scales_all_zeros() {
        let acts = vec![0.0_f32; 6 * 3];
        let scales = AwqQuantizer::compute_channel_scales(&acts, 6, 3);
        for &s in &scales {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn channel_scales_reflect_magnitude() {
        // Column 0 has activations of magnitude 1, column 1 has magnitude 10.
        let acts = vec![1.0_f32, 10.0, -1.0, -10.0, 1.0, 10.0];
        let scales = AwqQuantizer::compute_channel_scales(&acts, 3, 2);
        assert!(
            scales[1] > scales[0],
            "larger-activation channel should have larger scale"
        );
    }

    // ── Quantize / output tests ───────────────────────────────────────────────

    #[test]
    fn quantize_simple() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 10;
        let weights = lcg_weights(n_rows * n_cols, 1);
        let acts = lcg_weights(n_samples * n_cols, 2);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 11,
        };
        let result = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(result.is_ok(), "quantize_simple failed: {:?}", result.err());
    }

    #[test]
    fn quantize_output_sizes() {
        let n_rows = 3;
        let n_cols = 8;
        let n_samples = 5;
        let weights = lcg_weights(n_rows * n_cols, 10);
        let acts = lcg_weights(n_samples * n_cols, 11);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 11,
        };
        let out = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("awq_quantize should succeed");
        assert_eq!(out.quantized_weights.len(), n_rows * n_cols);
    }

    #[test]
    fn quantize_channel_scales_len() {
        let n_rows = 2;
        let n_cols = 6;
        let n_samples = 4;
        let weights = lcg_weights(n_rows * n_cols, 20);
        let acts = lcg_weights(n_samples * n_cols, 21);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 3,
            n_alpha_steps: 5,
        };
        let out = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("awq_quantize should succeed");
        assert_eq!(out.channel_scales.len(), n_cols);
    }

    #[test]
    fn best_alpha_in_range() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_weights(n_rows * n_cols, 30);
        let acts = lcg_weights(n_samples * n_cols, 31);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 21,
        };
        let out = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("awq_quantize should succeed");
        assert!(
            out.best_alpha >= 0.0 && out.best_alpha <= 1.0,
            "best_alpha out of [0,1]: {}",
            out.best_alpha
        );
    }

    #[test]
    fn best_error_finite() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_weights(n_rows * n_cols, 40);
        let acts = lcg_weights(n_samples * n_cols, 41);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 21,
        };
        let out = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("awq_quantize should succeed");
        assert!(
            out.best_error.is_finite() && out.best_error >= 0.0,
            "best_error should be finite non-negative, got {}",
            out.best_error
        );
    }

    // ── Dequantize tests ──────────────────────────────────────────────────────

    #[test]
    fn dequantize_shape() {
        let n_rows = 3;
        let n_cols = 6;
        let n_samples = 4;
        let weights = lcg_weights(n_rows * n_cols, 50);
        let acts = lcg_weights(n_samples * n_cols, 51);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 3,
            n_alpha_steps: 5,
        };
        let q = AwqQuantizer::new(cfg);
        let out = q
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let deq = q.dequantize(&out, n_rows, n_cols);
        assert_eq!(deq.len(), n_rows * n_cols);
    }

    #[test]
    fn dequantize_finite() {
        let n_rows = 3;
        let n_cols = 6;
        let n_samples = 4;
        let weights = lcg_weights(n_rows * n_cols, 60);
        let acts = lcg_weights(n_samples * n_cols, 61);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 3,
            n_alpha_steps: 5,
        };
        let q = AwqQuantizer::new(cfg);
        let out = q
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let deq = q.dequantize(&out, n_rows, n_cols);
        for &v in &deq {
            assert!(v.is_finite(), "dequantized value is not finite: {v}");
        }
    }

    // ── Alpha-specific tests ──────────────────────────────────────────────────

    #[test]
    fn alpha_zero_gives_scale_one() {
        // When alpha=0, s[i] = act_scale[i]^0 = 1.0 for all i (no scaling).
        let act_scale = [2.0_f32, 5.0, 0.3, 10.0];
        let s: Vec<f32> = act_scale
            .iter()
            .map(|&a| a.max(1e-6_f32).powf(0.0))
            .collect();
        for &si in &s {
            assert!((si - 1.0).abs() < 1e-5, "alpha=0 should give s=1, got {si}");
        }
    }

    #[test]
    fn alpha_one_gives_full_scale() {
        // When alpha=1, s[i] = act_scale[i]^1 = act_scale[i].
        let act_scale = [2.0_f32, 5.0, 0.3, 10.0];
        let s: Vec<f32> = act_scale
            .iter()
            .map(|&a| a.max(1e-6_f32).powf(1.0))
            .collect();
        for (&si, &ai) in s.iter().zip(act_scale.iter()) {
            assert!(
                (si - ai.max(1e-6_f32)).abs() < 1e-5,
                "alpha=1 should give s=act_scale, got {si} vs {ai}"
            );
        }
    }

    // ── Numerical quality tests ───────────────────────────────────────────────

    #[test]
    fn higher_bits_lower_error() {
        let n_rows = 8;
        let n_cols = 8;
        let n_samples = 16;
        let weights = lcg_weights(n_rows * n_cols, 70);
        let acts = lcg_weights(n_samples * n_cols, 71);

        let cfg4 = AwqConfig {
            bits: 4,
            group_size: 8,
            n_alpha_steps: 11,
        };
        let cfg8 = AwqConfig {
            bits: 8,
            group_size: 8,
            n_alpha_steps: 11,
        };

        let q4 = AwqQuantizer::new(cfg4);
        let q8 = AwqQuantizer::new(cfg8);

        let out4 = q4
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let out8 = q8
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");

        let mse4 = mse(&q4.dequantize(&out4, n_rows, n_cols), &weights);
        let mse8 = mse(&q8.dequantize(&out8, n_rows, n_cols), &weights);
        assert!(
            mse8 < mse4,
            "8-bit error {mse8:.6} should be < 4-bit error {mse4:.6}"
        );
    }

    #[test]
    fn reconstruction_close_high_bits() {
        let n_rows = 8;
        let n_cols = 8;
        let n_samples = 16;
        let weights = lcg_weights(n_rows * n_cols, 80);
        let acts = lcg_weights(n_samples * n_cols, 81);
        let cfg = AwqConfig {
            bits: 8,
            group_size: 8,
            n_alpha_steps: 11,
        };
        let q = AwqQuantizer::new(cfg);
        let out = q
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let deq = q.dequantize(&out, n_rows, n_cols);
        let orig_var = variance(&weights);
        let err_mse = mse(&deq, &weights);
        // Tolerate up to 10% of original variance as MSE
        let tol = orig_var * 0.10 + 1e-6;
        assert!(
            err_mse < tol,
            "8-bit reconstruction MSE {err_mse:.6} exceeds 10% of variance {orig_var:.6}"
        );
    }

    #[test]
    fn group_size_one_extreme() {
        // group_size=1 means per-element quantization
        let n_rows = 3;
        let n_cols = 4;
        let n_samples = 4;
        let weights = lcg_weights(n_rows * n_cols, 90);
        let acts = lcg_weights(n_samples * n_cols, 91);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 1,
            n_alpha_steps: 5,
        };
        let result = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(
            result.is_ok(),
            "group_size=1 should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn single_row_works() {
        let n_rows = 1;
        let n_cols = 8;
        let n_samples = 4;
        let weights = lcg_weights(n_rows * n_cols, 100);
        let acts = lcg_weights(n_samples * n_cols, 101);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 11,
        };
        let result = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(result.is_ok(), "single_row should work: {:?}", result.err());
    }

    #[test]
    fn large_activation_channel_gets_protected() {
        // Channel 3 has activations 100× larger than others.
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 8;
        let weights = lcg_weights(n_rows * n_cols, 110);
        let mut acts = lcg_weights(n_samples * n_cols, 111);
        // Amplify channel 3
        for s in 0..n_samples {
            acts[s * n_cols + 3] *= 100.0;
        }
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 21,
        };
        let out = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("awq_quantize should succeed");
        // Channel with 100× activation importance should have a larger scale
        let s3 = out.channel_scales[3];
        let s0 = out.channel_scales[0];
        // When best_alpha > 0, the salient channel should get a proportionally larger scale
        if out.best_alpha > 0.01 {
            assert!(
                s3 > s0,
                "salient channel 3 (s={s3:.4}) should dominate channel 0 (s={s0:.4})"
            );
        }
        // At minimum, output should be valid
        assert!(out.best_error.is_finite());
    }

    #[test]
    fn convenience_fn_matches_method() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_weights(n_rows * n_cols, 120);
        let acts = lcg_weights(n_samples * n_cols, 121);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 11,
        };
        let q = AwqQuantizer::new(cfg.clone());
        let out_method = q
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let out_fn = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg)
            .expect("awq_quantize should succeed");
        assert!(
            (out_method.best_alpha - out_fn.best_alpha).abs() < 1e-6,
            "best_alpha mismatch"
        );
        assert!(
            (out_method.best_error - out_fn.best_error).abs() < 1e-6,
            "best_error mismatch"
        );
    }

    #[test]
    fn quantize_bits2_works() {
        let n_rows = 2;
        let n_cols = 4;
        let n_samples = 4;
        let weights = lcg_weights(n_rows * n_cols, 130);
        let acts = lcg_weights(n_samples * n_cols, 131);
        let cfg = AwqConfig {
            bits: 2,
            group_size: 4,
            n_alpha_steps: 5,
        };
        let result = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(
            result.is_ok(),
            "2-bit quantization failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn quantize_bits8_works() {
        let n_rows = 4;
        let n_cols = 8;
        let n_samples = 8;
        let weights = lcg_weights(n_rows * n_cols, 140);
        let acts = lcg_weights(n_samples * n_cols, 141);
        let cfg = AwqConfig {
            bits: 8,
            group_size: 8,
            n_alpha_steps: 5,
        };
        let result = awq_quantize(&weights, n_rows, n_cols, &acts, n_samples, cfg);
        assert!(
            result.is_ok(),
            "8-bit quantization failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn repeated_quantize_deterministic() {
        let n_rows = 4;
        let n_cols = 4;
        let n_samples = 6;
        let weights = lcg_weights(n_rows * n_cols, 150);
        let acts = lcg_weights(n_samples * n_cols, 151);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 4,
            n_alpha_steps: 11,
        };
        let q = AwqQuantizer::new(cfg);
        let out1 = q
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        let out2 = q
            .quantize(&weights, n_rows, n_cols, &acts, n_samples)
            .expect("value should be present");
        assert_eq!(
            out1.best_alpha, out2.best_alpha,
            "non-deterministic best_alpha"
        );
        assert_eq!(
            out1.best_error, out2.best_error,
            "non-deterministic best_error"
        );
        assert_eq!(
            out1.quantized_weights, out2.quantized_weights,
            "non-deterministic quantized_weights"
        );
    }
}
