//! AWQ — Activation-aware Weight Quantization.
//!
//! Reference: Lin J, Tang J, Tang H, Yang S, Dang X, Gan C, Han S (2023)
//! "AWQ: Activation-aware Weight Quantization for LLM Compression and
//! Acceleration", MLSys 2024. <https://arxiv.org/abs/2306.00978>
//!
//! # Idea
//!
//! A linear layer computes `Y = X · W`, where `X ∈ ℝ^{batch × n_in}` are the
//! input activations and `W ∈ ℝ^{n_in × n_out}` the weight matrix (row-major,
//! `W[c · n_out + o]` for input channel `c` and output channel `o`).
//!
//! AWQ observes that the dominant quantization error of a linear layer comes
//! from a *small fraction of salient input channels* — those that the
//! activations excite most strongly. Crucially, salience is determined by
//! **activation** magnitude, not weight magnitude. Rather than keep those
//! channels in higher precision (mixed precision is hardware-unfriendly), AWQ
//! folds a per-input-channel scaling into the layer:
//!
//! ```text
//! Y = X · W = (X · diag(s⁻¹)) · (diag(s) · W)
//! ```
//!
//! The product is mathematically unchanged in full precision, but scaling a
//! salient channel `c` *up* by `s_c > 1` before quantization gives it more
//! INT-grid resolution; the matching `s_c⁻¹` is absorbed into the preceding
//! op (LayerNorm / previous linear), so it costs nothing at inference time.
//! The scaled weight `diag(s) · W` is then quantized group-wise to INT-`bits`
//! with per-group affine `(scale, zero)` round-to-nearest.
//!
//! # Scale search
//!
//! The per-input-channel scale is parameterised as
//!
//! ```text
//! s_c = act_scale_c^α   (optionally  / weight_scale_c^β)
//! ```
//!
//! normalised so that its geometric mean is exactly 1 (this keeps the scaled
//! weight's overall dynamic range — and hence the affine step size — roughly
//! independent of `α`, so the search trades resolution *between* channels
//! rather than globally). The exponent `α ∈ [0, 1]` is swept over an
//! `n_grid`-point grid to minimise the **output** reconstruction error
//!
//! ```text
//! ‖X · W − X · diag(s⁻¹) · Q(diag(s) · W)‖²
//! ```
//!
//! where `Q` is the group-wise INT quantizer. With only per-channel activation
//! magnitudes available (the usual calibration summary), treating channels as
//! mutually uncorrelated reduces the output MSE to the activation-weighted
//! weight MSE `Σ_o Σ_c act_scale_c² · (W[c,o] − s_c⁻¹ Q(s_c W)[c,o])²`, which is
//! exactly what the grid search minimises here. [`awq_output_mse`] evaluates the
//! *exact* objective against a supplied calibration batch and is used by the
//! tests to confirm the core claim (AWQ ≤ naive RTN) on real `X · W`.
//!
//! All arithmetic runs in `f64`; integer codes are stored as `i32`. The routine
//! is fully deterministic — no RNG is required.

use crate::error::{InferError, InferResult};

// ─── Config ────────────────────────────────────────────────────────────────

/// Configuration for the AWQ quantizer.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Bits per quantized value. Must be one of `{2, 3, 4, 8}`.
    pub bits: u8,
    /// Number of consecutive input-channel rows sharing one `(scale, zero)`
    /// pair. Must be `> 0` and must evenly divide `n_in`.
    pub group_size: usize,
    /// Number of grid points used to sweep `α ∈ [0, 1]` (inclusive of both
    /// endpoints). Must be `≥ 2`.
    pub n_grid: usize,
    /// Whether to divide the salience scale by `weight_scale_c^β`, where
    /// `weight_scale_c = mean(|W[c, :]|)` is the per-input-channel weight
    /// magnitude. The original AWQ ablation found this optional refinement
    /// helpful on some layers.
    pub use_weight_scale: bool,
    /// Exponent applied to the per-channel weight magnitude when
    /// `use_weight_scale` is set. Ignored otherwise.
    pub beta: f64,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            n_grid: 20,
            use_weight_scale: false,
            beta: 0.0,
        }
    }
}

// ─── Result types ────────────────────────────────────────────────────────────

/// Per-group affine quantization parameters for one input-channel group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupParams {
    /// Affine scale `Δ` mapping a code to `Δ · code + zero`.
    pub scale: f64,
    /// Affine zero-point (continuous, in the *scaled* weight space).
    pub zero: f64,
}

/// Output of [`awq_quantize`].
#[derive(Debug, Clone)]
pub struct AwqResult {
    /// Integer codes in `[0, 2^bits − 1]`, row-major `n_in × n_out`.
    pub q: Vec<i32>,
    /// Per-group affine parameters; length `ceil(n_in / group_size) · n_out`,
    /// laid out as `params[group · n_out + o]` (one entry per group, per output
    /// channel — groups partition the input axis, but the affine grid is fit
    /// per output column within each group).
    pub group_params: Vec<GroupParams>,
    /// Chosen salience exponent `α` selected by the grid search.
    pub alpha: f64,
    /// Per-input-channel scale `s_c` (length `n_in`, geomean-normalised to 1).
    pub awq_scale: Vec<f64>,
    /// Activation-weighted reconstruction MSE of the chosen `α` (the proxy
    /// objective the grid search minimised).
    pub recon_error: f64,
    /// Bits per quantized value.
    pub bits: u8,
    /// Group size used by the quantizer (after clamping to `n_in`).
    pub group_size: usize,
    /// Original `(n_in, n_out)` shape of the weight matrix.
    pub shape: (usize, usize),
}

impl AwqResult {
    /// Number of input-channel groups.
    #[must_use]
    pub fn n_groups(&self) -> usize {
        let (n_in, _) = self.shape;
        n_in.div_ceil(self.group_size)
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Quantize a linear weight `W` (row-major `n_in × n_out`) with AWQ.
///
/// * `weight` — row-major `n_in × n_out` weights (`weight[c * n_out + o]`).
/// * `n_in` / `n_out` — input- and output-channel counts.
/// * `act_scale` — per-input-channel activation magnitude `mean(|X[:, c]|)`,
///   length `n_in`. Larger values mark salient channels to be protected.
/// * `cfg` — see [`AwqConfig`].
///
/// Returns the quantized codes, per-group affine parameters, the salience
/// exponent `α` selected by the grid search, the applied per-channel scales,
/// and the (proxy) reconstruction error.
///
/// # Errors
///
/// Returns [`InferError::InvalidConfig`] for a malformed configuration (bits not
/// in `{2, 3, 4, 8}`, `group_size == 0`, `group_size` not dividing `n_in`,
/// `n_grid < 2`, or `beta < 0` with `use_weight_scale`), [`InferError::EmptyBatch`]
/// for a zero-sized matrix, [`InferError::DimensionMismatch`] when `weight` or
/// `act_scale` lengths disagree with the shape, and [`InferError::Other`] when a
/// non-finite input value is detected.
pub fn awq_quantize(
    weight: &[f64],
    n_in: usize,
    n_out: usize,
    act_scale: &[f64],
    cfg: &AwqConfig,
) -> InferResult<AwqResult> {
    validate(weight, n_in, n_out, act_scale, cfg)?;

    let group_size = cfg.group_size.min(n_in);
    let n_groups = n_in.div_ceil(group_size);
    let q_max = (1_i32 << cfg.bits) - 1;
    let q_max_f = f64::from(q_max);

    // Per-input-channel weight magnitude (only needed for the β refinement, but
    // cheap to always compute).
    let weight_scale = per_channel_weight_scale(weight, n_in, n_out);

    // Activation energy weights for the proxy MSE.
    let act_w2: Vec<f64> = act_scale.iter().map(|&v| v.abs() * v.abs()).collect();

    // Grid-search state.
    let mut best = SearchBest::new(n_in, n_out, n_groups);
    let mut s = vec![1.0_f64; n_in];
    let mut w_scaled = vec![0.0_f64; n_in * n_out];
    let layout = Layout {
        n_in,
        n_out,
        group_size,
        n_groups,
    };

    // Evaluate one candidate scaling `s` (already filled): quantize the scaled
    // weight, score the proxy MSE, and update the best-so-far. Returns the loss.
    let evaluate = |alpha: f64, s: &[f64], w_scaled: &mut [f64], best: &mut SearchBest| -> f64 {
        // Scale the weight up along the input axis: w_scaled[c,o] = s_c · W[c,o].
        for (c, &sc) in s.iter().enumerate() {
            let row = c * n_out;
            for o in 0..n_out {
                w_scaled[row + o] = weight[row + o] * sc;
            }
        }
        let (q, params) =
            quantize_groupwise(w_scaled, n_in, n_out, group_size, n_groups, q_max_f, q_max);
        let loss = proxy_mse(weight, &q, &params, s, &act_w2, layout);
        best.consider(alpha, loss, &q, &params, s);
        loss
    };

    // Identity baseline (s ≡ 1, i.e. α = 0 with no β term). This is exactly the
    // naive RTN grouping, so seeding the search with it guarantees the returned
    // AWQ result is never worse than RTN under the proxy objective — even when
    // the `use_weight_scale` β refinement happens to mis-shape the scales.
    evaluate(0.0, &vec![1.0_f64; n_in], &mut w_scaled, &mut best);

    let n_grid = cfg.n_grid.max(2);
    for step in 0..n_grid {
        // α grid over the closed interval [0, 1].
        let alpha = (step as f64) / ((n_grid - 1) as f64);
        compute_awq_scale(act_scale, &weight_scale, alpha, cfg, &mut s);
        evaluate(alpha, &s, &mut w_scaled, &mut best);
    }

    Ok(best.into_result(cfg.bits, group_size, (n_in, n_out)))
}

/// Dequantize an [`AwqResult`] back to a row-major `n_in × n_out` weight,
/// undoing both the affine code and the per-input-channel AWQ scaling so the
/// result lives in the original (un-scaled) weight space.
///
/// # Errors
///
/// Returns [`InferError::Other`] when the stored metadata is internally
/// inconsistent (`group_size == 0`, code-count mismatch, `awq_scale` length
/// disagreement, or group-parameter length mismatch).
pub fn awq_dequantize(result: &AwqResult) -> InferResult<Vec<f64>> {
    let (n_in, n_out) = result.shape;
    if result.group_size == 0 {
        return Err(InferError::Other("AWQ stored group_size is zero".into()));
    }
    let expected = n_in * n_out;
    if result.q.len() != expected {
        return Err(InferError::Other(format!(
            "AWQ q length {} != n_in*n_out {expected}",
            result.q.len()
        )));
    }
    if result.awq_scale.len() != n_in {
        return Err(InferError::Other(format!(
            "AWQ awq_scale length {} != n_in {n_in}",
            result.awq_scale.len()
        )));
    }
    let n_groups = n_in.div_ceil(result.group_size);
    if result.group_params.len() != n_groups * n_out {
        return Err(InferError::Other(format!(
            "AWQ group_params length {} != n_groups*n_out {}",
            result.group_params.len(),
            n_groups * n_out
        )));
    }
    let mut out = vec![0.0_f64; expected];
    for c in 0..n_in {
        let sc = result.awq_scale[c];
        let inv = if sc.abs() > f64::EPSILON {
            1.0 / sc
        } else {
            0.0
        };
        let g = (c / result.group_size).min(n_groups - 1);
        let row = c * n_out;
        let pbase = g * n_out;
        for o in 0..n_out {
            let p = result.group_params[pbase + o];
            let dq_scaled = p.scale * f64::from(result.q[row + o]) + p.zero;
            out[row + o] = dq_scaled * inv;
        }
    }
    Ok(out)
}

/// Group-wise INT-`bits` quantization of a *plain* (already-scaled) weight,
/// returning the codes and per-group affine parameters. Exposed so callers can
/// reuse the exact same affine grid AWQ uses internally (e.g. to implement a
/// naive RTN baseline by passing `bits`, `group_size`, and unit scaling).
///
/// `weight` is row-major `n_in × n_out`; groups partition the input axis.
///
/// # Errors
///
/// Returns [`InferError::InvalidConfig`] / [`InferError::EmptyBatch`] /
/// [`InferError::DimensionMismatch`] under the same conditions as
/// [`awq_quantize`] (minus the activation checks).
pub fn group_quantize(
    weight: &[f64],
    n_in: usize,
    n_out: usize,
    bits: u8,
    group_size: usize,
) -> InferResult<(Vec<i32>, Vec<GroupParams>)> {
    validate_dims(weight, n_in, n_out, bits, group_size)?;
    let gs = group_size.min(n_in);
    let n_groups = n_in.div_ceil(gs);
    let q_max = (1_i32 << bits) - 1;
    Ok(quantize_groupwise(
        weight,
        n_in,
        n_out,
        gs,
        n_groups,
        f64::from(q_max),
        q_max,
    ))
}

/// Dequantize raw group-wise codes (no AWQ scaling) back to `n_in × n_out`.
///
/// # Errors
///
/// Returns [`InferError::Other`] for code-count / parameter-count mismatch or a
/// zero group size.
pub fn group_dequantize(
    q: &[i32],
    params: &[GroupParams],
    n_in: usize,
    n_out: usize,
    group_size: usize,
) -> InferResult<Vec<f64>> {
    if group_size == 0 {
        return Err(InferError::Other("group_size is zero".into()));
    }
    let expected = n_in * n_out;
    if q.len() != expected {
        return Err(InferError::Other(format!(
            "code length {} != n_in*n_out {expected}",
            q.len()
        )));
    }
    let n_groups = n_in.div_ceil(group_size);
    if params.len() != n_groups * n_out {
        return Err(InferError::Other(format!(
            "params length {} != n_groups*n_out {}",
            params.len(),
            n_groups * n_out
        )));
    }
    let mut out = vec![0.0_f64; expected];
    for c in 0..n_in {
        let g = (c / group_size).min(n_groups - 1);
        let row = c * n_out;
        let pbase = g * n_out;
        for o in 0..n_out {
            let p = params[pbase + o];
            out[row + o] = p.scale * f64::from(q[row + o]) + p.zero;
        }
    }
    Ok(out)
}

/// Exact AWQ output reconstruction error
/// `‖X·W − X·diag(s⁻¹)·Q(diag(s)·W)‖²` against a supplied calibration batch.
///
/// * `x` — row-major `batch × n_in` activations.
/// * `weight` — the *original* (un-scaled) row-major `n_in × n_out` weight.
/// * `result` — output of [`awq_quantize`] for that weight.
///
/// This evaluates the true objective (not the per-channel proxy) and is the
/// quantity the correctness tests compare against the RTN baseline.
///
/// # Errors
///
/// Returns [`InferError::DimensionMismatch`] when `x`, `weight`, or the stored
/// result shapes are inconsistent, and [`InferError::Other`] on inconsistent
/// stored metadata via [`awq_dequantize`].
pub fn awq_output_mse(
    x: &[f64],
    batch: usize,
    weight: &[f64],
    result: &AwqResult,
) -> InferResult<f64> {
    let (n_in, n_out) = result.shape;
    if x.len() != batch * n_in {
        return Err(InferError::DimensionMismatch {
            expected: batch * n_in,
            got: x.len(),
        });
    }
    if weight.len() != n_in * n_out {
        return Err(InferError::DimensionMismatch {
            expected: n_in * n_out,
            got: weight.len(),
        });
    }
    // Effective dequantized weight in the *original* space: W̃ = diag(s⁻¹)·Q(diag(s)·W).
    let w_eff = awq_dequantize(result)?;
    Ok(output_mse_dense(x, batch, weight, &w_eff, n_in, n_out))
}

/// Exact output reconstruction error `‖X·W − X·W̃‖²` for an arbitrary
/// dequantized weight `w_eff` (row-major `n_in × n_out`), against batch `x`.
///
/// Useful for an RTN baseline: dequantize the raw codes with
/// [`group_dequantize`] and pass the result here.
///
/// # Errors
///
/// Returns [`InferError::DimensionMismatch`] on any shape disagreement.
pub fn dense_output_mse(
    x: &[f64],
    batch: usize,
    weight: &[f64],
    w_eff: &[f64],
    n_in: usize,
    n_out: usize,
) -> InferResult<f64> {
    if x.len() != batch * n_in {
        return Err(InferError::DimensionMismatch {
            expected: batch * n_in,
            got: x.len(),
        });
    }
    if weight.len() != n_in * n_out || w_eff.len() != n_in * n_out {
        return Err(InferError::DimensionMismatch {
            expected: n_in * n_out,
            got: w_eff.len(),
        });
    }
    Ok(output_mse_dense(x, batch, weight, w_eff, n_in, n_out))
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Shared dimensions for the proxy-MSE accumulation.
#[derive(Copy, Clone)]
struct Layout {
    n_in: usize,
    n_out: usize,
    group_size: usize,
    n_groups: usize,
}

/// Best-so-far accumulator for the grid search.
struct SearchBest {
    alpha: f64,
    loss: f64,
    q: Vec<i32>,
    params: Vec<GroupParams>,
    s: Vec<f64>,
    found: bool,
}

impl SearchBest {
    fn new(n_in: usize, n_out: usize, n_groups: usize) -> Self {
        Self {
            alpha: 0.0,
            loss: f64::INFINITY,
            q: vec![0_i32; n_in * n_out],
            params: vec![
                GroupParams {
                    scale: 0.0,
                    zero: 0.0
                };
                n_groups * n_out
            ],
            s: vec![1.0_f64; n_in],
            found: false,
        }
    }

    fn consider(&mut self, alpha: f64, loss: f64, q: &[i32], params: &[GroupParams], s: &[f64]) {
        if !self.found || loss < self.loss {
            self.found = true;
            self.alpha = alpha;
            self.loss = loss;
            self.q.copy_from_slice(q);
            self.params.copy_from_slice(params);
            self.s.copy_from_slice(s);
        }
    }

    fn into_result(self, bits: u8, group_size: usize, shape: (usize, usize)) -> AwqResult {
        AwqResult {
            q: self.q,
            group_params: self.params,
            alpha: self.alpha,
            awq_scale: self.s,
            recon_error: self.loss,
            bits,
            group_size,
            shape,
        }
    }
}

/// Per-input-channel weight magnitude `mean_o(|W[c, o]|)`, length `n_in`.
fn per_channel_weight_scale(weight: &[f64], n_in: usize, n_out: usize) -> Vec<f64> {
    let mut ws = vec![0.0_f64; n_in];
    let inv_n = 1.0 / (n_out as f64);
    for (c, slot) in ws.iter_mut().enumerate() {
        let row = c * n_out;
        let mut acc = 0.0_f64;
        for o in 0..n_out {
            acc += weight[row + o].abs();
        }
        *slot = acc * inv_n;
    }
    ws
}

/// Compute the per-input-channel scale `s_c = act_scale_c^α (/ weight_scale_c^β)`
/// and normalise so the geometric mean of `s` is exactly 1.
fn compute_awq_scale(
    act_scale: &[f64],
    weight_scale: &[f64],
    alpha: f64,
    cfg: &AwqConfig,
    s: &mut [f64],
) {
    let n = act_scale.len();
    let mut log_sum = 0.0_f64;
    for c in 0..n {
        let act = act_scale[c].abs() + 1e-8;
        let mut val = act.powf(alpha);
        if cfg.use_weight_scale {
            let wmag = weight_scale[c].abs() + 1e-8;
            val /= wmag.powf(cfg.beta);
        }
        // Guard: never allow a zero / non-finite factor to poison the geomean.
        if !(val.is_finite()) || val <= 0.0 {
            val = f64::MIN_POSITIVE;
        }
        s[c] = val;
        log_sum += val.ln();
    }
    let log_geomean = log_sum / (n as f64);
    let geomean = log_geomean.exp().max(f64::MIN_POSITIVE);
    let inv = 1.0 / geomean;
    for slot in s.iter_mut() {
        *slot *= inv;
    }
}

/// Group-wise asymmetric affine quantization of the (already scaled) weight.
///
/// Groups partition the **input** axis in chunks of `group_size`; within each
/// group the affine grid is fit *per output column* from that column's min/max
/// over the group's rows. This matches the way group-wise weight quantizers fit
/// one `(scale, zero)` per (group × output-channel) cell.
fn quantize_groupwise(
    weight: &[f64],
    n_in: usize,
    n_out: usize,
    group_size: usize,
    n_groups: usize,
    q_max_f: f64,
    q_max: i32,
) -> (Vec<i32>, Vec<GroupParams>) {
    let mut q = vec![0_i32; n_in * n_out];
    let mut params = vec![
        GroupParams {
            scale: 0.0,
            zero: 0.0
        };
        n_groups * n_out
    ];
    for g in 0..n_groups {
        let c_start = g * group_size;
        let c_end = (c_start + group_size).min(n_in);
        let pbase = g * n_out;
        for o in 0..n_out {
            // Min / max of this output column over the group's input rows.
            let mut lo = weight[c_start * n_out + o];
            let mut hi = lo;
            for c in c_start..c_end {
                let v = weight[c * n_out + o];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let span = (hi - lo).max(f64::EPSILON);
            let scale = (span / q_max_f).max(f64::EPSILON);
            let inv_s = 1.0 / scale;
            params[pbase + o] = GroupParams { scale, zero: lo };
            for c in c_start..c_end {
                let idx = c * n_out + o;
                let code = ((weight[idx] - lo) * inv_s).round();
                q[idx] = clamp_to_code(code, q_max);
            }
        }
    }
    (q, params)
}

/// Activation-weighted weight MSE proxy for the output reconstruction error.
///
/// `Σ_o Σ_c act_w2[c] · (W[c,o] − s_c⁻¹·(scale·code + zero))²`, which equals the
/// true output MSE when activation channels are mutually uncorrelated.
fn proxy_mse(
    weight: &[f64],
    q: &[i32],
    params: &[GroupParams],
    s: &[f64],
    act_w2: &[f64],
    layout: Layout,
) -> f64 {
    let Layout {
        n_in,
        n_out,
        group_size,
        n_groups,
    } = layout;
    let mut acc = 0.0_f64;
    for c in 0..n_in {
        let sc = s[c];
        let inv = if sc.abs() > f64::EPSILON {
            1.0 / sc
        } else {
            0.0
        };
        let wgt = act_w2[c];
        let g = (c / group_size).min(n_groups - 1);
        let row = c * n_out;
        let pbase = g * n_out;
        for o in 0..n_out {
            let p = params[pbase + o];
            let dq = (p.scale * f64::from(q[row + o]) + p.zero) * inv;
            let diff = weight[row + o] - dq;
            acc += diff * diff * wgt;
        }
    }
    acc
}

/// Exact dense output MSE `Σ_b Σ_o (Σ_c X[b,c]·W[c,o] − Σ_c X[b,c]·W̃[c,o])²`,
/// computed from the residual `δ = W − W̃` to halve the matmul work.
fn output_mse_dense(
    x: &[f64],
    batch: usize,
    weight: &[f64],
    w_eff: &[f64],
    n_in: usize,
    n_out: usize,
) -> f64 {
    let mut acc = 0.0_f64;
    for b in 0..batch {
        let xrow = b * n_in;
        for o in 0..n_out {
            let mut diff = 0.0_f64;
            for c in 0..n_in {
                let delta = weight[c * n_out + o] - w_eff[c * n_out + o];
                diff += x[xrow + c] * delta;
            }
            acc += diff * diff;
        }
    }
    acc
}

/// Saturating clamp of a rounded code into `[0, q_max]`.
#[inline]
fn clamp_to_code(code: f64, q_max: i32) -> i32 {
    if !code.is_finite() {
        return 0;
    }
    let c = code as i64;
    c.clamp(0, i64::from(q_max)) as i32
}

/// Validate configuration + dimensions (without activation-vector checks).
fn validate_dims(
    weight: &[f64],
    n_in: usize,
    n_out: usize,
    bits: u8,
    group_size: usize,
) -> InferResult<()> {
    if n_in == 0 || n_out == 0 {
        return Err(InferError::EmptyBatch);
    }
    if !matches!(bits, 2 | 3 | 4 | 8) {
        return Err(InferError::InvalidConfig(
            "AWQ bits must be one of {2, 3, 4, 8}",
        ));
    }
    if group_size == 0 {
        return Err(InferError::InvalidConfig("AWQ group_size must be > 0"));
    }
    if n_in % group_size != 0 {
        return Err(InferError::InvalidConfig(
            "AWQ group_size must evenly divide n_in",
        ));
    }
    let expected = n_in * n_out;
    if weight.len() != expected {
        return Err(InferError::DimensionMismatch {
            expected,
            got: weight.len(),
        });
    }
    Ok(())
}

/// Full validation including the activation vector and finiteness checks.
fn validate(
    weight: &[f64],
    n_in: usize,
    n_out: usize,
    act_scale: &[f64],
    cfg: &AwqConfig,
) -> InferResult<()> {
    validate_dims(weight, n_in, n_out, cfg.bits, cfg.group_size)?;
    if cfg.n_grid < 2 {
        return Err(InferError::InvalidConfig("AWQ n_grid must be >= 2"));
    }
    if cfg.use_weight_scale && cfg.beta < 0.0 {
        return Err(InferError::InvalidConfig(
            "AWQ beta must be >= 0 when use_weight_scale is set",
        ));
    }
    if act_scale.len() != n_in {
        return Err(InferError::DimensionMismatch {
            expected: n_in,
            got: act_scale.len(),
        });
    }
    for (c, &v) in act_scale.iter().enumerate() {
        if !v.is_finite() {
            return Err(InferError::Other(format!(
                "AWQ act_scale[{c}]={v} is not finite"
            )));
        }
    }
    for (idx, &v) in weight.iter().enumerate() {
        if !v.is_finite() {
            return Err(InferError::Other(format!(
                "AWQ weight[{idx}]={v} is not finite"
            )));
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG so tests need no external RNG and stay reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
        }
        /// Uniform in `[-1, 1)`.
        fn next_signed(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (self.0 >> 11) as f64; // 53 bits
            let unit = bits / ((1_u64 << 53) as f64); // [0, 1)
            unit * 2.0 - 1.0
        }
    }

    /// Build a synthetic weight `n_in × n_out` plus an activation profile in
    /// which `n_salient` input channels carry far larger activations. The
    /// salient channels also get wider weight ranges so naive RTN is forced to
    /// spend coarse INT steps on them — exactly the case AWQ is designed for.
    fn build_case(
        n_in: usize,
        n_out: usize,
        n_salient: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let mut rng = Lcg::new(seed);
        let mut w = vec![0.0_f64; n_in * n_out];
        let mut act = vec![0.0_f64; n_in];
        for c in 0..n_in {
            let salient = c < n_salient;
            act[c] = if salient { 30.0 } else { 0.4 };
            let amp = if salient { 6.0 } else { 0.8 };
            for o in 0..n_out {
                w[c * n_out + o] = amp * rng.next_signed();
            }
        }
        (w, act, n_salient)
    }

    /// Calibration batch consistent with `act` (channel `c` excited at scale
    /// `act[c]`), used to evaluate the *exact* output MSE.
    fn build_batch(act: &[f64], batch: usize, seed: u64) -> Vec<f64> {
        let n_in = act.len();
        let mut rng = Lcg::new(seed);
        let mut x = vec![0.0_f64; batch * n_in];
        for b in 0..batch {
            for c in 0..n_in {
                x[b * n_in + c] = act[c] * rng.next_signed();
            }
        }
        x
    }

    /// RTN baseline: group-wise quantize the *raw* weight (no AWQ scaling) and
    /// return its exact output MSE on `x`.
    fn rtn_output_mse(
        w: &[f64],
        n_in: usize,
        n_out: usize,
        bits: u8,
        group_size: usize,
        x: &[f64],
        batch: usize,
    ) -> f64 {
        let (q, params) = group_quantize(w, n_in, n_out, bits, group_size).expect("rtn quantize");
        let w_eff = group_dequantize(&q, &params, n_in, n_out, group_size).expect("rtn dequant");
        dense_output_mse(x, batch, w, &w_eff, n_in, n_out).expect("rtn mse")
    }

    #[test]
    fn awq_output_mse_le_rtn_core_claim() {
        let (n_in, n_out, n_salient) = (32, 24, 4);
        let (w, act, _) = build_case(n_in, n_out, n_salient, 7);
        let x = build_batch(&act, 48, 1234);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 8,
            n_grid: 20,
            use_weight_scale: false,
            beta: 0.0,
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg).expect("awq");
        let awq_mse = awq_output_mse(&x, 48, &w, &res).expect("awq mse");
        let rtn_mse = rtn_output_mse(&w, n_in, n_out, cfg.bits, cfg.group_size, &x, 48);
        assert!(
            awq_mse <= rtn_mse + 1e-9,
            "core AWQ claim violated: awq_mse={awq_mse} > rtn_mse={rtn_mse}"
        );
        // For a genuinely salient case AWQ should *strictly* win and pick α > 0.
        assert!(
            awq_mse < rtn_mse,
            "expected strict improvement (awq={awq_mse}, rtn={rtn_mse})"
        );
        assert!(
            res.alpha > 0.0,
            "expected a non-trivial α, got {}",
            res.alpha
        );
    }

    #[test]
    fn salient_channels_better_protected_than_rtn() {
        let (n_in, n_out, n_salient) = (24, 16, 3);
        let (w, act, salient) = build_case(n_in, n_out, n_salient, 11);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 8,
            n_grid: 20,
            use_weight_scale: false,
            beta: 0.0,
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg).expect("awq");
        let awq_w = awq_dequantize(&res).expect("awq dequant");

        let (q, params) =
            group_quantize(&w, n_in, n_out, cfg.bits, cfg.group_size).expect("rtn quantize");
        let rtn_w =
            group_dequantize(&q, &params, n_in, n_out, cfg.group_size).expect("rtn dequant");

        // Relative reconstruction error over the salient input channels.
        let mut awq_err = 0.0_f64;
        let mut rtn_err = 0.0_f64;
        let mut energy = 0.0_f64;
        for c in 0..salient {
            for o in 0..n_out {
                let idx = c * n_out + o;
                awq_err += (w[idx] - awq_w[idx]).powi(2);
                rtn_err += (w[idx] - rtn_w[idx]).powi(2);
                energy += w[idx].powi(2);
            }
        }
        let awq_rel = (awq_err / energy).sqrt();
        let rtn_rel = (rtn_err / energy).sqrt();
        assert!(
            awq_rel < rtn_rel,
            "salient channels not better protected: awq_rel={awq_rel}, rtn_rel={rtn_rel}"
        );
    }

    #[test]
    fn grid_aligned_round_trip_near_exact() {
        // Build a weight whose every value already sits exactly on a per-group
        // INT grid; quant→dequant must reproduce it (within fp slack).
        let (n_in, n_out, group_size, bits) = (8, 4, 4, 4);
        let q_max = (1_i32 << bits) - 1;
        let mut w = vec![0.0_f64; n_in * n_out];
        // Per (group, column) choose lo and step so values land on the grid.
        for g in 0..(n_in / group_size) {
            for o in 0..n_out {
                let lo = -1.0 + 0.1 * (o as f64);
                let step = 0.05 + 0.01 * (g as f64);
                for k in 0..group_size {
                    let c = g * group_size + k;
                    let code = (k as i32) % (q_max + 1);
                    w[c * n_out + o] = lo + step * f64::from(code);
                }
            }
        }
        let act = vec![1.0_f64; n_in];
        let cfg = AwqConfig {
            bits,
            group_size,
            n_grid: 2, // α ∈ {0, 1}; with unit activations both give s ≡ 1.
            use_weight_scale: false,
            beta: 0.0,
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg).expect("awq");
        // Uniform activations ⇒ unit AWQ scale.
        for &s in &res.awq_scale {
            assert!((s - 1.0).abs() < 1e-9, "expected unit scale, got {s}");
        }
        // Codes must stay within INT range.
        for &code in &res.q {
            assert!(code >= 0 && code <= q_max, "code {code} out of INT range");
        }
        let dq = awq_dequantize(&res).expect("dequant");
        for (i, (&a, &b)) in w.iter().zip(dq.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "grid round-trip[{i}] {a} != {b}");
        }
    }

    #[test]
    fn group_params_consistent_and_in_range() {
        let (n_in, n_out, group_size, bits) = (16, 8, 4, 4);
        let (w, act, _) = build_case(n_in, n_out, 2, 99);
        let cfg = AwqConfig {
            bits,
            group_size,
            n_grid: 8,
            use_weight_scale: false,
            beta: 0.0,
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg).expect("awq");
        assert_eq!(res.n_groups(), n_in / group_size);
        assert_eq!(res.group_params.len(), res.n_groups() * n_out);
        assert_eq!(res.awq_scale.len(), n_in);
        assert_eq!(res.q.len(), n_in * n_out);
        for p in &res.group_params {
            assert!(
                p.scale > 0.0 && p.scale.is_finite(),
                "bad scale {}",
                p.scale
            );
            assert!(p.zero.is_finite(), "bad zero {}", p.zero);
        }
        let q_max = (1_i32 << bits) - 1;
        for &code in &res.q {
            assert!(code >= 0 && code <= q_max);
        }
        // recon_error must be finite and non-negative.
        assert!(res.recon_error.is_finite() && res.recon_error >= 0.0);
    }

    #[test]
    fn weight_scale_refinement_runs_and_is_valid() {
        let (n_in, n_out, n_salient) = (24, 16, 3);
        let (w, act, _) = build_case(n_in, n_out, n_salient, 5);
        let x = build_batch(&act, 40, 321);
        let cfg = AwqConfig {
            bits: 4,
            group_size: 8,
            n_grid: 20,
            use_weight_scale: true,
            beta: 0.5,
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg).expect("awq+ws");
        let awq_mse = awq_output_mse(&x, 40, &w, &res).expect("mse");
        let rtn_mse = rtn_output_mse(&w, n_in, n_out, cfg.bits, cfg.group_size, &x, 40);
        // The β refinement should still not be worse than RTN.
        assert!(
            awq_mse <= rtn_mse + 1e-9,
            "weight-scale AWQ worse than RTN: awq={awq_mse}, rtn={rtn_mse}"
        );
        // geomean of awq_scale ≈ 1.
        let log_sum: f64 = res.awq_scale.iter().map(|&s| s.ln()).sum();
        let geomean = (log_sum / (n_in as f64)).exp();
        assert!((geomean - 1.0).abs() < 1e-6, "geomean {geomean} not ≈ 1");
    }

    #[test]
    fn err_shape_mismatch() {
        let n_in = 8;
        let n_out = 4;
        let w = vec![0.0_f64; n_in * n_out + 3];
        let act = vec![1.0_f64; n_in];
        let cfg = AwqConfig::default();
        let cfg = AwqConfig {
            group_size: 4,
            ..cfg
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg);
        assert!(matches!(res, Err(InferError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_act_length_mismatch() {
        let n_in = 8;
        let n_out = 4;
        let w = vec![0.0_f64; n_in * n_out];
        let act = vec![1.0_f64; n_in + 1];
        let cfg = AwqConfig {
            group_size: 4,
            ..AwqConfig::default()
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg);
        assert!(matches!(res, Err(InferError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_bits_out_of_range() {
        let n_in = 8;
        let n_out = 4;
        let w = vec![0.0_f64; n_in * n_out];
        let act = vec![1.0_f64; n_in];
        for &bits in &[0_u8, 1, 5, 6, 7, 16] {
            let cfg = AwqConfig {
                bits,
                group_size: 4,
                ..AwqConfig::default()
            };
            let res = awq_quantize(&w, n_in, n_out, &act, &cfg);
            assert!(
                matches!(res, Err(InferError::InvalidConfig(_))),
                "expected InvalidConfig for bits={bits}"
            );
        }
    }

    #[test]
    fn err_group_size_not_dividing_n_in() {
        let n_in = 10; // not divisible by 4
        let n_out = 4;
        let w = vec![0.0_f64; n_in * n_out];
        let act = vec![1.0_f64; n_in];
        let cfg = AwqConfig {
            group_size: 4,
            ..AwqConfig::default()
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg);
        assert!(matches!(res, Err(InferError::InvalidConfig(_))));
    }

    #[test]
    fn err_empty_input() {
        let act: Vec<f64> = vec![];
        let cfg = AwqConfig {
            group_size: 1,
            ..AwqConfig::default()
        };
        let res = awq_quantize(&[], 0, 4, &act, &cfg);
        assert!(matches!(res, Err(InferError::EmptyBatch)));
        let res2 = awq_quantize(&[], 4, 0, &[1.0_f64; 4], &cfg);
        assert!(matches!(res2, Err(InferError::EmptyBatch)));
    }

    #[test]
    fn err_n_grid_too_small() {
        let n_in = 8;
        let n_out = 4;
        let w = vec![0.0_f64; n_in * n_out];
        let act = vec![1.0_f64; n_in];
        let cfg = AwqConfig {
            group_size: 4,
            n_grid: 1,
            ..AwqConfig::default()
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg);
        assert!(matches!(res, Err(InferError::InvalidConfig(_))));
    }

    #[test]
    fn err_non_finite_input() {
        let n_in = 8;
        let n_out = 4;
        let mut w = vec![0.5_f64; n_in * n_out];
        w[3] = f64::NAN;
        let act = vec![1.0_f64; n_in];
        let cfg = AwqConfig {
            group_size: 4,
            ..AwqConfig::default()
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg);
        assert!(matches!(res, Err(InferError::Other(_))));
    }

    #[test]
    fn dequantize_round_trip_via_eff_weight_matches_output_mse() {
        // Sanity: awq_output_mse equals dense_output_mse(W, dequant(result)).
        let (n_in, n_out) = (16, 8);
        let (w, act, _) = build_case(n_in, n_out, 2, 314);
        let x = build_batch(&act, 20, 271);
        let cfg = AwqConfig {
            group_size: 4,
            ..AwqConfig::default()
        };
        let res = awq_quantize(&w, n_in, n_out, &act, &cfg).expect("awq");
        let via_api = awq_output_mse(&x, 20, &w, &res).expect("api mse");
        let w_eff = awq_dequantize(&res).expect("dequant");
        let direct = dense_output_mse(&x, 20, &w, &w_eff, n_in, n_out).expect("direct mse");
        assert!((via_api - direct).abs() < 1e-9, "{via_api} != {direct}");
    }
}
