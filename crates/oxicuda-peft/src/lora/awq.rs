//! AWQ — Activation-aware Weight Quantization.
//!
//! Reference: Lin J, Tang J, Tang H, Yang S, Chen W-M, Wang W-C, Xiao G,
//! Dang X, Gan C, Han S (2024) "AWQ: Activation-aware Weight Quantization for
//! LLM Compression and Acceleration", MLSys 2024.
//! <https://arxiv.org/abs/2306.00978>
//!
//! AWQ observes that a small fraction of *salient* weight channels — those
//! driven by large-magnitude activations — dominate the quantization error of a
//! linear layer. Rather than perturb those channels, AWQ scales the offending
//! input channels by a per-channel factor `s_i = mean(|x_i|)^α` (with a unit
//! geometric mean) so the rescaled weight `W' = diag(s) · W` is *flatter* along
//! the input axis. The standard per-group affine quantizer then operates on the
//! rescaled weights, and the saved `s` is folded back during dequantization
//! (`W̃ = diag(1/s) · dequant_affine(W')`).
//!
//! The implementation grid-searches `α ∈ {0/N, 1/N, …, N/N}` to minimise an
//! activation-weighted MSE between the original and round-trip weight, mirroring
//! the structure of [`crate::lora::gptq`] (per-group min/max scale/zero, integer
//! code storage as `i32`, fully deterministic — no RNG required).

use crate::error::{PeftError, PeftResult};

/// Configuration for the AWQ quantizer.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Bits per quantized value. Must be one of `{3, 4, 8}`.
    pub bits: u8,
    /// Number of consecutive output-channel columns sharing one `(scale, zero)` pair.
    /// Must be `> 0`.
    pub group_size: usize,
    /// Number of grid steps `N` used to sweep `α ∈ {0/N, 1/N, …, N/N}`.
    /// Must be `≥ 1`.
    pub alpha_search_steps: u32,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            alpha_search_steps: 20,
        }
    }
}

/// Output of [`Awq::quantize_weight`].
///
/// `q` holds integer codes in row-major `(rows, cols)` order. `scale` and `zero`
/// are per-output-channel-group affine parameters (length
/// `ceil(cols / group_size)`). `awq_scale` is the per-input-channel scaling
/// vector `s` (length `rows`) used to flatten the weight matrix prior to
/// quantization. `alpha` records the grid value that minimised the
/// activation-weighted MSE.
#[derive(Debug, Clone)]
pub struct AwqQuantized {
    /// Integer codes in `[0, 2^bits − 1]`, row-major `rows × cols`.
    pub q: Vec<i32>,
    /// Per-group affine scale (after the AWQ scaling is applied).
    pub scale: Vec<f32>,
    /// Per-group affine zero (after the AWQ scaling is applied).
    pub zero: Vec<f32>,
    /// Chosen salience exponent `α` selected by the grid search.
    pub alpha: f32,
    /// Per-input-channel scale `s_i = (mean|x_i| + ε)^α` (normalised to geomean 1).
    pub awq_scale: Vec<f32>,
    /// Bits per quantized value.
    pub bits: u8,
    /// Group size used by the quantizer.
    pub group_size: usize,
    /// Original `(rows, cols)` shape of the weight matrix.
    pub original_shape: (usize, usize),
}

/// AWQ algorithm namespace.
pub struct Awq;

impl Awq {
    /// Quantize `w` (row-major `rows × cols`, where `rows` indexes the input
    /// channel and `cols` indexes the output channel) using the AWQ procedure.
    ///
    /// `act_abs_mean` must have length `rows` and supplies the mean absolute
    /// activation per input channel collected over a calibration batch.
    ///
    /// # Errors
    /// Returns [`PeftError::Internal`] for any configuration or dimension
    /// violation (bits not in `{3, 4, 8}`, zero group size, zero search steps,
    /// zero-sized matrix, or length mismatches).
    pub fn quantize_weight(
        w: &[f32],
        rows: usize,
        cols: usize,
        act_abs_mean: &[f32],
        cfg: &AwqConfig,
    ) -> PeftResult<AwqQuantized> {
        validate(w, rows, cols, act_abs_mean, cfg)?;

        let group_size = cfg.group_size.min(cols);
        let n_groups = cols.div_ceil(group_size);
        let q_max = (1_i32 << cfg.bits) - 1;
        let q_max_f = q_max as f64;
        let n_steps = cfg.alpha_search_steps.max(1);

        let mut best_alpha: f32 = 0.0;
        let mut best_loss: f64 = f64::INFINITY;
        let mut best_q: Vec<i32> = vec![0_i32; rows * cols];
        let mut best_scale: Vec<f32> = vec![0.0_f32; n_groups];
        let mut best_zero: Vec<f32> = vec![0.0_f32; n_groups];
        let mut best_awq: Vec<f32> = vec![1.0_f32; rows];

        let mut s = vec![1.0_f64; rows];
        let mut w_scaled = vec![0.0_f64; rows * cols];

        // Squared activation weights for MSE, capped to keep the loss finite when
        // an input channel has near-zero mean activation.
        let mut act_w2 = vec![0.0_f64; rows];
        for (slot, &v) in act_w2.iter_mut().zip(act_abs_mean.iter()) {
            let av = (v as f64).abs();
            *slot = av * av;
        }

        for step in 0..=n_steps {
            let alpha = (step as f32) / (n_steps as f32);

            compute_awq_scale(act_abs_mean, alpha, &mut s);

            // w_scaled[i, j] = w[i, j] · s[i]
            for (i, &si) in s.iter().enumerate().take(rows) {
                let row_off = i * cols;
                for j in 0..cols {
                    w_scaled[row_off + j] = (w[row_off + j] as f64) * si;
                }
            }

            // Per-group affine quantization along the output-channel axis.
            let (q, scale, zero) =
                quantize_group_affine(&w_scaled, rows, cols, group_size, n_groups, q_max_f, q_max);

            // Compute activation-weighted MSE between original and round-trip weight.
            let view = DequantView {
                q: &q,
                scale: &scale,
                zero: &zero,
                s: &s,
                act_w2: &act_w2,
            };
            let layout = GroupLayout {
                rows,
                cols,
                group_size,
                n_groups,
            };
            let loss = activation_weighted_mse(w, &view, layout);

            if loss < best_loss {
                best_loss = loss;
                best_alpha = alpha;
                best_q.copy_from_slice(&q);
                best_scale.copy_from_slice(&scale);
                best_zero.copy_from_slice(&zero);
                for (dst, &v) in best_awq.iter_mut().zip(s.iter()) {
                    *dst = v as f32;
                }
            }
        }

        Ok(AwqQuantized {
            q: best_q,
            scale: best_scale,
            zero: best_zero,
            alpha: best_alpha,
            awq_scale: best_awq,
            bits: cfg.bits,
            group_size,
            original_shape: (rows, cols),
        })
    }

    /// Dequantize an [`AwqQuantized`] back to a row-major `rows × cols` `Vec<f32>`.
    ///
    /// Reverses the affine code, then divides by `awq_scale[i]` to recover the
    /// weight in original (un-scaled) space.
    ///
    /// # Errors
    /// Returns [`PeftError::Internal`] when stored metadata is inconsistent
    /// (`group_size == 0`, code count mismatch, group meta length mismatch, or
    /// `awq_scale` length disagreeing with `rows`).
    pub fn dequantize_and_apply(q: &AwqQuantized) -> PeftResult<Vec<f32>> {
        let (rows, cols) = q.original_shape;
        if q.group_size == 0 {
            return Err(PeftError::Internal {
                msg: "AWQ stored group_size is zero".to_string(),
            });
        }
        let expected = rows * cols;
        if q.q.len() != expected {
            return Err(PeftError::Internal {
                msg: format!("AWQ q length {} != rows*cols {expected}", q.q.len()),
            });
        }
        if q.awq_scale.len() != rows {
            return Err(PeftError::Internal {
                msg: format!("AWQ awq_scale length {} != rows {rows}", q.awq_scale.len()),
            });
        }
        let n_groups = cols.div_ceil(q.group_size);
        if q.scale.len() != n_groups || q.zero.len() != n_groups {
            return Err(PeftError::Internal {
                msg: format!(
                    "AWQ group meta length mismatch (expected {n_groups}, got scale={}, zero={})",
                    q.scale.len(),
                    q.zero.len()
                ),
            });
        }
        let mut out = vec![0.0_f32; expected];
        for i in 0..rows {
            // Guard against a vanishing per-row scale so we never produce ±∞.
            let si = q.awq_scale[i];
            let inv = if si.abs() > f32::EPSILON {
                1.0_f32 / si
            } else {
                0.0_f32
            };
            let row_off = i * cols;
            for j in 0..cols {
                let g = (j / q.group_size).min(n_groups - 1);
                let dq_scaled = q.scale[g] * (q.q[row_off + j] as f32) + q.zero[g];
                out[row_off + j] = dq_scaled * inv;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Validate user-supplied inputs and configuration.
fn validate(
    w: &[f32],
    rows: usize,
    cols: usize,
    act_abs_mean: &[f32],
    cfg: &AwqConfig,
) -> PeftResult<()> {
    if rows == 0 || cols == 0 {
        return Err(PeftError::Internal {
            msg: format!("AWQ requires non-zero rows and cols (rows={rows}, cols={cols})"),
        });
    }
    if !matches!(cfg.bits, 3 | 4 | 8) {
        return Err(PeftError::Internal {
            msg: format!("AWQ bits must be one of {{3, 4, 8}}, got {}", cfg.bits),
        });
    }
    if cfg.group_size == 0 {
        return Err(PeftError::Internal {
            msg: "AWQ group_size must be > 0".to_string(),
        });
    }
    if cfg.alpha_search_steps == 0 {
        return Err(PeftError::Internal {
            msg: "AWQ alpha_search_steps must be >= 1".to_string(),
        });
    }
    let expected = rows * cols;
    if w.len() != expected {
        return Err(PeftError::Internal {
            msg: format!("AWQ weight length {} != rows*cols {expected}", w.len()),
        });
    }
    if act_abs_mean.len() != rows {
        return Err(PeftError::Internal {
            msg: format!(
                "AWQ act_abs_mean length {} != rows {rows}",
                act_abs_mean.len()
            ),
        });
    }
    for (i, &v) in act_abs_mean.iter().enumerate() {
        if !v.is_finite() {
            return Err(PeftError::Internal {
                msg: format!("AWQ act_abs_mean[{i}]={v} is not finite"),
            });
        }
    }
    for (idx, &v) in w.iter().enumerate() {
        if !v.is_finite() {
            return Err(PeftError::Internal {
                msg: format!("AWQ w[{idx}]={v} is not finite"),
            });
        }
    }
    Ok(())
}

/// Compute the per-input-channel scaling factor `s_i = (|x_i| + ε)^α` and then
/// normalise so the geometric mean of `s` is exactly 1. The normalisation step
/// keeps the dynamic range of the rescaled weight independent of `α`.
fn compute_awq_scale(act_abs_mean: &[f32], alpha: f32, s: &mut [f64]) {
    let rows = act_abs_mean.len();
    let alpha_f = alpha as f64;
    let mut log_sum = 0.0_f64;
    for (i, &v) in act_abs_mean.iter().enumerate() {
        let base = (v as f64).abs() + 1e-8;
        let pow = base.powf(alpha_f);
        s[i] = pow;
        log_sum += pow.max(f64::MIN_POSITIVE).ln();
    }
    let log_geomean = log_sum / (rows as f64);
    let geomean = log_geomean.exp().max(f64::MIN_POSITIVE);
    let inv = 1.0_f64 / geomean;
    for slot in s.iter_mut() {
        *slot *= inv;
    }
}

/// Apply per-group affine quantization to `w_scaled` (row-major `rows × cols`),
/// grouping along the output-channel (column) axis. Returns the integer code
/// tensor, per-group scale, and per-group zero (all length `n_groups`).
fn quantize_group_affine(
    w_scaled: &[f64],
    rows: usize,
    cols: usize,
    group_size: usize,
    n_groups: usize,
    q_max_f: f64,
    q_max: i32,
) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0_i32; rows * cols];
    let mut scale = vec![0.0_f32; n_groups];
    let mut zero = vec![0.0_f32; n_groups];
    for g in 0..n_groups {
        let j_start = g * group_size;
        let j_end = (j_start + group_size).min(cols);
        let (lo, hi) = group_min_max(w_scaled, rows, cols, j_start, j_end);
        let span = (hi - lo).max(f64::EPSILON);
        let s = span / q_max_f;
        let s_safe = s.max(f64::EPSILON);
        let inv_s = 1.0_f64 / s_safe;
        scale[g] = s as f32;
        zero[g] = lo as f32;
        for i in 0..rows {
            let row_off = i * cols;
            for j in j_start..j_end {
                let code = ((w_scaled[row_off + j] - lo) * inv_s).round();
                q[row_off + j] = clamp_to_code(code as i64, q_max);
            }
        }
    }
    (q, scale, zero)
}

/// Min / max of every entry in columns `[j_start, j_end)` of a row-major
/// `rows × cols` matrix (`f64` working precision).
fn group_min_max(w: &[f64], rows: usize, cols: usize, j_start: usize, j_end: usize) -> (f64, f64) {
    let mut lo = w[j_start];
    let mut hi = w[j_start];
    for i in 0..rows {
        for j in j_start..j_end {
            let v = w[i * cols + j];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    (lo, hi)
}

/// Saturating clamp of an integer code into `[0, q_max]`.
#[inline]
fn clamp_to_code(code: i64, q_max: i32) -> i32 {
    code.clamp(0_i64, q_max as i64) as i32
}

/// Round-trip view of a quantized weight, used by [`activation_weighted_mse`].
struct DequantView<'a> {
    q: &'a [i32],
    scale: &'a [f32],
    zero: &'a [f32],
    s: &'a [f64],
    act_w2: &'a [f64],
}

/// Layout of the quantization tensors shared across MSE inputs.
#[derive(Copy, Clone)]
struct GroupLayout {
    rows: usize,
    cols: usize,
    group_size: usize,
    n_groups: usize,
}

/// Activation-weighted MSE between the original `w` and the round-trip
/// dequantization `w̃[i, j] = (scale[g] · q[i, j] + zero[g]) / s[i]`.
fn activation_weighted_mse(w: &[f32], view: &DequantView<'_>, layout: GroupLayout) -> f64 {
    let DequantView {
        q,
        scale,
        zero,
        s,
        act_w2,
    } = *view;
    let GroupLayout {
        rows,
        cols,
        group_size,
        n_groups,
    } = layout;
    let mut acc = 0.0_f64;
    for i in 0..rows {
        let si = s[i];
        let inv = if si.abs() > f64::EPSILON {
            1.0_f64 / si
        } else {
            0.0_f64
        };
        let wi = act_w2[i].max(f64::MIN_POSITIVE);
        let row_off = i * cols;
        for j in 0..cols {
            let g = (j / group_size).min(n_groups - 1);
            let dq_scaled = (scale[g] as f64) * (q[row_off + j] as f64) + (zero[g] as f64);
            let dq = dq_scaled * inv;
            let diff = (w[row_off + j] as f64) - dq;
            acc += diff * diff * wi;
        }
    }
    acc
}
