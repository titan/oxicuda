//! Mamba residual block with all sub-operations (RMSNorm, linear, SiLU,
//! causal depthwise conv1d) and the full Mamba forward pass.
//!
//! # Architecture (Mamba paper, Gu & Dao 2023)
//!
//! ```text
//! input: [L, D]
//!   → RMSNorm([L, D])
//!   → in_proj [D → 2*D_inner]: split into x [L, D_inner] and z [L, D_inner]
//!   → x: causal depthwise conv1d (kernel d_conv) → SiLU
//!   → x_proj [D_inner → dt_rank + 2*N]:
//!       delta_raw [L, d_inner], B_proj [L, N], C_proj [L, N]
//!   → dt = softplus(dt_proj(delta_raw))  [L, D_inner]
//!   → selective_scan(x, dt, A_log, B_proj, C_proj) → ssm_out [L, D_inner]
//!   → y = ssm_out * SiLU(z)
//!   → out_proj [D_inner → D]
//!   → output = input + out_proj(y)   (residual)
//! ```

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::mamba::selective_scan::{SelectiveScanConfig, selective_scan};

// ─── MambaBlockConfig ────────────────────────────────────────────────────────

/// Configuration for a single Mamba residual block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MambaBlockConfig {
    /// Model dimension `D`.
    pub d_model: usize,
    /// Inner (expanded) dimension `D_inner = expand * D`.
    pub d_inner: usize,
    /// State size `N` per SSM channel.
    pub d_state: usize,
    /// Depthwise conv kernel size.
    pub d_conv: usize,
    /// Expansion factor (typically 2).
    pub expand: usize,
}

impl MambaBlockConfig {
    /// Create a new config with defaults: d_inner = 2*d_model, d_state=16, d_conv=4, expand=2.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidModelDim`] — if `d_model == 0`
    pub fn new(d_model: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        Ok(Self {
            d_model,
            d_inner: 2 * d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        })
    }

    /// Override the state dimension.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidSsmOrder`] — if `d_state == 0`
    pub fn with_d_state(mut self, d_state: usize) -> MambaResult<Self> {
        if d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(d_state));
        }
        self.d_state = d_state;
        Ok(self)
    }

    /// Override the depthwise conv kernel size.
    ///
    /// # Errors
    ///
    /// - [`MambaError::Internal`] — if `d_conv == 0`
    pub fn with_d_conv(mut self, d_conv: usize) -> MambaResult<Self> {
        if d_conv == 0 {
            return Err(MambaError::Internal("d_conv must be > 0".into()));
        }
        self.d_conv = d_conv;
        Ok(self)
    }

    // ── Derived weight sizes ──────────────────────────────────────────────────

    /// `in_proj` shape: `[2*d_inner, d_model]` (row-major; rows = out features).
    #[inline]
    pub fn in_proj_size(&self) -> usize {
        2 * self.d_inner * self.d_model
    }

    /// `conv_weight` shape: `[d_inner, d_conv]` (depthwise, one kernel per channel).
    #[inline]
    pub fn conv_weight_size(&self) -> usize {
        self.d_inner * self.d_conv
    }

    /// `conv_bias` shape: `[d_inner]`.
    #[inline]
    pub fn conv_bias_size(&self) -> usize {
        self.d_inner
    }

    /// `x_proj` projects `[D_inner]` → `[D_inner + 2*N]` (delta_raw, B, C packed).
    /// Shape: `[(d_inner + 2*d_state), d_inner]`.
    #[inline]
    pub fn x_proj_size(&self) -> usize {
        (self.d_inner + 2 * self.d_state) * self.d_inner
    }

    /// `dt_proj` shape: `[d_inner, d_inner]` — projects dt_raw to full dt.
    #[inline]
    pub fn dt_proj_size(&self) -> usize {
        self.d_inner * self.d_inner
    }

    /// `a_log` shape: `[d_inner, d_state]`.
    #[inline]
    pub fn a_log_size(&self) -> usize {
        self.d_inner * self.d_state
    }

    /// `d_skip` (D) shape: `[d_inner]`.
    #[inline]
    pub fn d_skip_size(&self) -> usize {
        self.d_inner
    }

    /// `out_proj` shape: `[d_model, d_inner]`.
    #[inline]
    pub fn out_proj_size(&self) -> usize {
        self.d_model * self.d_inner
    }

    /// `norm_weight` shape: `[d_model]`.
    #[inline]
    pub fn norm_weight_size(&self) -> usize {
        self.d_model
    }
}

// ─── MambaBlockWeights ───────────────────────────────────────────────────────

/// All learnable weights for a single Mamba residual block.
#[derive(Debug, Clone)]
pub struct MambaBlockWeights {
    /// `in_proj`: `[2*d_inner, d_model]` — projects input to (x, z) pair.
    pub in_proj: Vec<f32>,
    /// `conv_weight`: `[d_inner, d_conv]` — depthwise causal conv kernels.
    pub conv_weight: Vec<f32>,
    /// `conv_bias`: `[d_inner]`.
    pub conv_bias: Vec<f32>,
    /// `x_proj`: `[(d_inner + 2*d_state), d_inner]` — projects x to (Δ, B, C).
    pub x_proj: Vec<f32>,
    /// `dt_proj`: `[d_inner, d_inner]` — maps dt_raw back to full dt.
    pub dt_proj: Vec<f32>,
    /// `a_log`: `[d_inner, d_state]` — `log(-A)`, initialized to `log(n+1)`.
    pub a_log: Vec<f32>,
    /// `d_skip`: `[d_inner]` — skip-connection weight D in `y += D * u`.
    pub d_skip: Vec<f32>,
    /// `out_proj`: `[d_model, d_inner]`.
    pub out_proj: Vec<f32>,
    /// `norm_weight`: `[d_model]` — RMSNorm scale.
    pub norm_weight: Vec<f32>,
}

impl MambaBlockWeights {
    /// Allocate all weight tensors and zero-initialize them.
    pub fn zeros(config: &MambaBlockConfig) -> Self {
        Self {
            in_proj: vec![0.0; config.in_proj_size()],
            conv_weight: vec![0.0; config.conv_weight_size()],
            conv_bias: vec![0.0; config.conv_bias_size()],
            x_proj: vec![0.0; config.x_proj_size()],
            dt_proj: vec![0.0; config.dt_proj_size()],
            a_log: vec![0.0; config.a_log_size()],
            d_skip: vec![0.0; config.d_skip_size()],
            out_proj: vec![0.0; config.out_proj_size()],
            norm_weight: vec![0.0; config.norm_weight_size()],
        }
    }

    /// Initialize with paper-recommended defaults:
    ///
    /// - `a_log[d, n] = log(n + 1)` (stable, increasing decay rates)
    /// - `norm_weight = 1.0` (identity RMSNorm)
    /// - `d_skip = 1.0` (identity skip)
    /// - All other weights zero.
    pub fn default_init(config: &MambaBlockConfig) -> Self {
        let mut w = Self::zeros(config);
        // a_log: log(n+1) for n in 0..d_state, repeated for each d
        for d in 0..config.d_inner {
            for n in 0..config.d_state {
                w.a_log[d * config.d_state + n] = ((n + 1) as f32).ln();
            }
        }
        // RMSNorm weight: identity
        for v in &mut w.norm_weight {
            *v = 1.0;
        }
        // D skip: 1.0
        for v in &mut w.d_skip {
            *v = 1.0;
        }
        w
    }

    /// Initialize with small random weights from a normal distribution.
    /// `a_log` is initialized with `log(n+1)`, `norm_weight=1.0`.
    pub fn random(config: &MambaBlockConfig, rng: &mut LcgRng) -> Self {
        let mut w = Self::zeros(config);
        rng.fill_normal(&mut w.in_proj);
        rng.fill_normal(&mut w.conv_weight);
        rng.fill_normal(&mut w.conv_bias);
        rng.fill_normal(&mut w.x_proj);
        rng.fill_normal(&mut w.dt_proj);
        rng.fill_normal(&mut w.d_skip);
        rng.fill_normal(&mut w.out_proj);
        // a_log: use paper default regardless (random a_log can cause instability)
        for d in 0..config.d_inner {
            for n in 0..config.d_state {
                w.a_log[d * config.d_state + n] = ((n + 1) as f32).ln();
            }
        }
        // RMSNorm weight: identity
        for v in &mut w.norm_weight {
            *v = 1.0;
        }
        w
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// RMSNorm: normalize `x` by its root-mean-square, then scale by `weight`.
///
/// `x: [L * D]` (row-major: [L, D]), `weight: [D]` → `out: [L * D]`
///
/// For each row `x[t, :]`, the RMS is computed over the D dimension,
/// and each element is divided by the RMS, then multiplied by `weight[d]`.
///
/// # Errors
///
/// - [`MambaError::DimensionMismatch`] if `x.len() != seq_len * d_model`
///   or `weight.len() != d_model`.
pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    d_model: usize,
    eps: f32,
) -> MambaResult<Vec<f32>> {
    let expected_x = seq_len * d_model;
    if x.len() != expected_x {
        return Err(MambaError::DimensionMismatch {
            expected: expected_x,
            got: x.len(),
        });
    }
    if weight.len() != d_model {
        return Err(MambaError::DimensionMismatch {
            expected: d_model,
            got: weight.len(),
        });
    }
    let mut out = vec![0.0_f32; expected_x];
    for t in 0..seq_len {
        let row = &x[t * d_model..(t + 1) * d_model];
        // Compute RMS over the D dimension
        let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / d_model as f32;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();
        for (d, &x_val) in row.iter().enumerate() {
            out[t * d_model + d] = x_val * rms_inv * weight[d];
        }
    }
    Ok(out)
}

/// Linear projection: `out[L, out_dim] = x[L, in_dim] @ W^T`, where `W: [out_dim, in_dim]`.
///
/// # Errors
///
/// - [`MambaError::DimensionMismatch`] if `x.len() != seq_len * in_dim`
///   or `w.len() != out_dim * in_dim`.
pub fn linear(
    x: &[f32],
    w: &[f32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> MambaResult<Vec<f32>> {
    let expected_x = seq_len * in_dim;
    if x.len() != expected_x {
        return Err(MambaError::DimensionMismatch {
            expected: expected_x,
            got: x.len(),
        });
    }
    let expected_w = out_dim * in_dim;
    if w.len() != expected_w {
        return Err(MambaError::DimensionMismatch {
            expected: expected_w,
            got: w.len(),
        });
    }
    let mut out = vec![0.0_f32; seq_len * out_dim];
    for t in 0..seq_len {
        let x_row = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let w_row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0_f32;
            for k in 0..in_dim {
                acc += x_row[k] * w_row[k];
            }
            out[t * out_dim + o] = acc;
        }
    }
    Ok(out)
}

/// SiLU activation: `x * sigmoid(x) = x / (1 + exp(-x))`.
///
/// Applied element-wise; always returns a new `Vec<f32>` of the same length.
#[inline]
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// Causal depthwise conv1d.
///
/// Applies a per-channel 1-D causal convolution over the time dimension.
/// Zero-padding is added to the left so the output length equals `seq_len`.
///
/// Layout:
/// - `x`: `[L, C]` (time-major, row-major)
/// - `w`: `[C, kernel]` — one kernel per channel
/// - `b`: `[C]` — per-channel bias
///
/// For each channel `c` and time step `t`:
/// ```text
/// out[t, c] = b[c] + Σ_{k=0}^{kernel-1}  w[c, k] * x[t - k, c]   (x[t<0, c] = 0)
/// ```
///
/// # Errors
///
/// - [`MambaError::DimensionMismatch`] if any slice has wrong length.
pub fn causal_depthwise_conv1d(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    seq_len: usize,
    channels: usize,
    kernel: usize,
) -> MambaResult<Vec<f32>> {
    let expected_x = seq_len * channels;
    if x.len() != expected_x {
        return Err(MambaError::DimensionMismatch {
            expected: expected_x,
            got: x.len(),
        });
    }
    let expected_w = channels * kernel;
    if w.len() != expected_w {
        return Err(MambaError::DimensionMismatch {
            expected: expected_w,
            got: w.len(),
        });
    }
    if b.len() != channels {
        return Err(MambaError::DimensionMismatch {
            expected: channels,
            got: b.len(),
        });
    }

    let mut out = vec![0.0_f32; seq_len * channels];
    for t in 0..seq_len {
        for c in 0..channels {
            let mut acc = b[c];
            for k in 0..kernel {
                // Causal: look back k steps; use zero-padding for t < k
                if t >= k {
                    acc += w[c * kernel + k] * x[(t - k) * channels + c];
                }
                // else: x[t - k] is zero-padded (implicit)
            }
            out[t * channels + c] = acc;
        }
    }
    Ok(out)
}

// ─── MambaBlock ──────────────────────────────────────────────────────────────

/// A single Mamba residual block.
///
/// Implements the full Mamba forward pass including RMSNorm, input projection,
/// depthwise causal conv, selective scan (S6), gating, and output projection,
/// all with a residual connection.
pub struct MambaBlock {
    config: MambaBlockConfig,
    weights: MambaBlockWeights,
}

impl MambaBlock {
    /// Construct a new Mamba block, validating weight shapes.
    ///
    /// # Errors
    ///
    /// - [`MambaError::WeightShapeMismatch`] if any weight tensor has wrong length.
    pub fn new(config: MambaBlockConfig, weights: MambaBlockWeights) -> MambaResult<Self> {
        // Validate all weight shapes
        let checks: &[(&'static str, usize, usize)] = &[
            ("in_proj", weights.in_proj.len(), config.in_proj_size()),
            (
                "conv_weight",
                weights.conv_weight.len(),
                config.conv_weight_size(),
            ),
            (
                "conv_bias",
                weights.conv_bias.len(),
                config.conv_bias_size(),
            ),
            ("x_proj", weights.x_proj.len(), config.x_proj_size()),
            ("dt_proj", weights.dt_proj.len(), config.dt_proj_size()),
            ("a_log", weights.a_log.len(), config.a_log_size()),
            ("d_skip", weights.d_skip.len(), config.d_skip_size()),
            ("out_proj", weights.out_proj.len(), config.out_proj_size()),
            (
                "norm_weight",
                weights.norm_weight.len(),
                config.norm_weight_size(),
            ),
        ];
        for &(name, got, expected) in checks {
            if got != expected {
                return Err(MambaError::WeightShapeMismatch {
                    name,
                    expected: vec![expected],
                    got: vec![got],
                });
            }
        }
        Ok(Self { config, weights })
    }

    /// Forward pass: `x [L * D]` → `y [L * D]` with residual connection.
    ///
    /// `x` must have exactly `seq_len * config.d_model` elements (row-major `[L, D]`).
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `x.len() != seq_len * d_model`.
    /// - Propagated errors from sub-operations.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> MambaResult<Vec<f32>> {
        let cfg = &self.config;
        let w = &self.weights;
        let d = cfg.d_model;
        let d_inner = cfg.d_inner;
        let n = cfg.d_state;

        let expected = seq_len * d;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // ── 1. RMSNorm input ──────────────────────────────────────────────────
        let x_normed = rms_norm(x, &w.norm_weight, seq_len, d, 1e-5)?;

        // ── 2. Input projection: [L, D] → [L, 2*D_inner] ─────────────────────
        // in_proj: [2*d_inner, d_model] → out: [L, 2*d_inner]
        let xz = linear(&x_normed, &w.in_proj, seq_len, d, 2 * d_inner)?;

        // ── 3. Split x_branch [L, D_inner] and z_gate [L, D_inner] ───────────
        let mut x_branch = vec![0.0_f32; seq_len * d_inner];
        let mut z_gate = vec![0.0_f32; seq_len * d_inner];
        for t in 0..seq_len {
            for i in 0..d_inner {
                x_branch[t * d_inner + i] = xz[t * (2 * d_inner) + i];
                z_gate[t * d_inner + i] = xz[t * (2 * d_inner) + d_inner + i];
            }
        }

        // ── 4. Depthwise causal conv1d on x_branch → SiLU ────────────────────
        let x_conv = causal_depthwise_conv1d(
            &x_branch,
            &w.conv_weight,
            &w.conv_bias,
            seq_len,
            d_inner,
            cfg.d_conv,
        )?;
        let x_act = silu(&x_conv); // [L * d_inner]

        // ── 5. x_proj: [L, D_inner] → [L, D_inner + 2*N] ────────────────────
        // Packed: delta_raw [L, D_inner], b_proj [L, N], c_proj [L, N]
        let xbc_proj_dim = d_inner + 2 * n;
        let xbc = linear(&x_act, &w.x_proj, seq_len, d_inner, xbc_proj_dim)?;

        // ── 6. Unpack delta_raw, B_proj, C_proj ──────────────────────────────
        let mut delta_raw = vec![0.0_f32; seq_len * d_inner];
        let mut b_proj_seq = vec![0.0_f32; seq_len * n];
        let mut c_proj_seq = vec![0.0_f32; seq_len * n];
        for t in 0..seq_len {
            let row_start = t * xbc_proj_dim;
            // delta_raw: first d_inner features
            for i in 0..d_inner {
                delta_raw[t * d_inner + i] = xbc[row_start + i];
            }
            // B_proj: next n features
            for i in 0..n {
                b_proj_seq[t * n + i] = xbc[row_start + d_inner + i];
            }
            // C_proj: last n features
            for i in 0..n {
                c_proj_seq[t * n + i] = xbc[row_start + d_inner + n + i];
            }
        }

        // ── 7. dt_proj: delta_raw [L, D_inner] → dt [L, D_inner] ─────────────
        let dt_full = linear(&delta_raw, &w.dt_proj, seq_len, d_inner, d_inner)?;
        // softplus applied inside selective_scan; we pass dt_full directly as
        // the "raw delta" input (selective_scan applies softplus internally)

        // ── 8. Selective scan (S6) ────────────────────────────────────────────
        // u:      [1, L, D_inner] → batch=1
        // delta:  [1, L, D_inner]
        // a_log:  [D_inner, N]
        // b_proj: [1, L, N]
        // c_proj: [1, L, N]
        let scan_cfg = SelectiveScanConfig::new(1, seq_len, d_inner, n)?;
        let ssm_out = selective_scan(
            &x_act,
            &dt_full,
            &w.a_log,
            &b_proj_seq,
            &c_proj_seq,
            &scan_cfg,
        )?; // [L * D_inner]

        // ── 9. D skip connection: y += D_skip * x_act ─────────────────────────
        let mut ssm_with_skip = ssm_out.clone();
        for t in 0..seq_len {
            for i in 0..d_inner {
                ssm_with_skip[t * d_inner + i] += w.d_skip[i] * x_act[t * d_inner + i];
            }
        }

        // ── 10. Gating: y = ssm_out * SiLU(z) ───────────────────────────────
        let z_silu = silu(&z_gate);
        let mut y_inner = vec![0.0_f32; seq_len * d_inner];
        for i in 0..ssm_with_skip.len() {
            y_inner[i] = ssm_with_skip[i] * z_silu[i];
        }

        // ── 11. out_proj: [L, D_inner] → [L, D] ─────────────────────────────
        let y_out = linear(&y_inner, &w.out_proj, seq_len, d_inner, d)?;

        // ── 12. Residual connection: output = x + out_proj(y) ─────────────────
        let mut output = vec![0.0_f32; seq_len * d];
        for i in 0..output.len() {
            output[i] = x[i] + y_out[i];
        }

        Ok(output)
    }

    /// Return a reference to the block configuration.
    #[inline]
    pub fn config(&self) -> &MambaBlockConfig {
        &self.config
    }

    /// Return a reference to the block weights.
    #[inline]
    pub fn weights(&self) -> &MambaBlockWeights {
        &self.weights
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    // ── rms_norm ──────────────────────────────────────────────────────────────

    /// All-ones input, all-ones weight → each output element ≈ 1.0
    /// (RMS of all-ones is 1, so normalized is still 1, times weight=1).
    #[test]
    fn rms_norm_unit_vector() {
        let x = vec![1.0_f32; 4];
        let w = vec![1.0_f32; 4];
        let out = rms_norm(&x, &w, 1, 4, 1e-5).expect("rms_norm");
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < EPS, "rms_norm[{i}]={v}, expected 1.0");
        }
    }

    #[test]
    fn rms_norm_zeros_weight() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![0.0_f32; 4];
        let out = rms_norm(&x, &w, 1, 4, 1e-5).expect("rms_norm");
        for &v in &out {
            assert!(v.abs() < 1e-7, "expected zero output for zero weight");
        }
    }

    #[test]
    fn rms_norm_finite() {
        let mut rng = crate::handle::LcgRng::new(123);
        let mut x = vec![0.0_f32; 16];
        let mut w = vec![0.0_f32; 4];
        rng.fill_normal(&mut x);
        rng.fill_normal(&mut w);
        let out = rms_norm(&x, &w, 4, 4, 1e-5).expect("rms_norm");
        for &v in &out {
            assert!(v.is_finite(), "rms_norm output not finite: {v}");
        }
    }

    // ── linear ────────────────────────────────────────────────────────────────

    /// W = identity (out_dim == in_dim, W[i,j] = 1 if i==j) → output = input.
    #[test]
    fn linear_identity_weight() {
        let d = 4_usize;
        let l = 3_usize;
        let x: Vec<f32> = (0..l * d).map(|i| i as f32).collect();
        // Identity matrix [d, d]
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        let out = linear(&x, &w, l, d, d).expect("linear");
        for (i, (&xi, &oi)) in x.iter().zip(out.iter()).enumerate() {
            assert!((xi - oi).abs() < 1e-6, "linear identity mismatch at {i}");
        }
    }

    #[test]
    fn linear_shape() {
        let l = 5_usize;
        let in_d = 4_usize;
        let out_d = 8_usize;
        let x = vec![1.0_f32; l * in_d];
        let w = vec![0.5_f32; out_d * in_d];
        let out = linear(&x, &w, l, in_d, out_d).expect("linear");
        assert_eq!(out.len(), l * out_d, "linear output shape mismatch");
    }

    // ── silu ──────────────────────────────────────────────────────────────────

    #[test]
    fn silu_positive_stays_positive() {
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = silu(&x);
        for (i, &v) in out.iter().enumerate() {
            assert!(v > 0.0, "silu({})={v} should be positive", x[i]);
        }
    }

    /// For large positive x, SiLU ≈ x (sigmoid → 1).
    #[test]
    fn silu_large_positive_approx_x() {
        let x = [10.0_f32, 20.0, 50.0];
        let out = silu(&x);
        for (&xi, &oi) in x.iter().zip(out.iter()) {
            // For x >= 10, sigmoid(x) > 0.999, so silu(x) ≈ x within 0.1%
            let rel_err = ((oi - xi) / xi).abs();
            assert!(rel_err < 0.01, "silu({xi})={oi} should be ≈ {xi}");
        }
    }

    // ── causal_depthwise_conv1d ───────────────────────────────────────────────

    /// Identity kernel (w=[1.0], b=[0.0]) → output equals input.
    #[test]
    fn causal_conv1d_identity_kernel() {
        let l = 6_usize;
        let c = 3_usize;
        let x: Vec<f32> = (0..l * c).map(|i| i as f32).collect();
        let w = vec![1.0_f32; c]; // kernel=1: one weight per channel
        let b = vec![0.0_f32; c];
        let out = causal_depthwise_conv1d(&x, &w, &b, l, c, 1).expect("conv1d");
        for (i, (&xi, &oi)) in x.iter().zip(out.iter()).enumerate() {
            assert!((xi - oi).abs() < 1e-6, "identity conv mismatch at {i}");
        }
    }

    #[test]
    fn causal_conv1d_zero_kernel() {
        let l = 4_usize;
        let c = 2_usize;
        let x = vec![1.0_f32; l * c];
        let w = vec![0.0_f32; c * 3]; // kernel=3, all zeros
        let b = vec![0.0_f32; c];
        let out = causal_depthwise_conv1d(&x, &w, &b, l, c, 3).expect("conv1d");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-7, "zero kernel conv: out[{i}]={v} should be 0");
        }
    }

    // ── MambaBlockConfig ──────────────────────────────────────────────────────

    #[test]
    fn block_config_defaults() {
        let cfg = MambaBlockConfig::new(16).expect("valid config");
        assert_eq!(cfg.d_inner, 2 * 16, "d_inner should be 2 * d_model");
        assert_eq!(cfg.d_state, 16, "default d_state = 16");
        assert_eq!(cfg.d_conv, 4, "default d_conv = 4");
        assert_eq!(cfg.expand, 2, "default expand = 2");
    }

    // ── MambaBlockWeights ─────────────────────────────────────────────────────

    #[test]
    fn block_weights_zeros_shape() {
        let cfg = MambaBlockConfig::new(8).expect("valid config");
        let w = MambaBlockWeights::zeros(&cfg);
        // in_proj: [2*d_inner, d_model] = [2*16, 8] = [32, 8] = 256
        assert_eq!(w.in_proj.len(), cfg.in_proj_size(), "in_proj size mismatch");
        assert_eq!(w.in_proj.len(), 2 * cfg.d_inner * cfg.d_model);
        assert!(
            w.in_proj.iter().all(|&v| v == 0.0),
            "zeros weight should be zero"
        );
    }

    // ── MambaBlock forward ────────────────────────────────────────────────────

    #[test]
    fn block_forward_shape() {
        let d = 8_usize;
        let l = 4_usize;
        let cfg = MambaBlockConfig::new(d)
            .expect("valid config")
            .with_d_state(4)
            .expect("valid d_state");
        let weights = MambaBlockWeights::default_init(&cfg);
        let block = MambaBlock::new(cfg, weights).expect("valid block");
        let x = vec![0.5_f32; l * d];
        let y = block.forward(&x, l).expect("forward");
        assert_eq!(y.len(), l * d, "forward output should have L*D elements");
    }

    #[test]
    fn block_forward_finite() {
        let d = 8_usize;
        let l = 6_usize;
        let cfg = MambaBlockConfig::new(d)
            .expect("valid config")
            .with_d_state(4)
            .expect("valid d_state");
        let mut rng = crate::handle::LcgRng::new(42);
        let weights = MambaBlockWeights::random(&cfg, &mut rng);
        let block = MambaBlock::new(cfg, weights).expect("valid block");
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let y = block.forward(&x, l).expect("forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} is not finite");
        }
    }

    /// With zero input and zero (most) weights, forward should still be finite.
    /// RMSNorm with zero input gives zero output (no NaN since eps guards RMS).
    /// The residual adds x (zero) + out_proj(y) = zero. Result: all zeros, finite.
    #[test]
    fn block_forward_zero_input() {
        let d = 4_usize;
        let l = 3_usize;
        let cfg = MambaBlockConfig::new(d)
            .expect("valid config")
            .with_d_state(4)
            .expect("valid d_state");
        // Use zeros weights except norm_weight=1 (default_init includes norm=1, skip=1)
        let weights = MambaBlockWeights::zeros(&cfg);
        // Patch: norm_weight must be set, otherwise rms_norm returns zero but that's ok
        let block = MambaBlock::new(cfg, weights).expect("valid block");
        let x = vec![0.0_f32; l * d];
        let y = block.forward(&x, l).expect("forward with zero input");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite with zero input");
        }
    }

    // ── Additional helper tests ───────────────────────────────────────────────

    /// Causal property: output at time t does not depend on future inputs.
    #[test]
    fn causal_conv1d_causality() {
        let l = 5_usize;
        let c = 2_usize;
        let kernel = 3_usize;
        let x: Vec<f32> = (0..l * c).map(|i| (i + 1) as f32).collect();
        let w: Vec<f32> = vec![0.25_f32; c * kernel];
        let b = vec![0.0_f32; c];
        let out = causal_depthwise_conv1d(&x, &w, &b, l, c, kernel).expect("conv1d");
        // At t=0, the kernel can only see x[0]: out[0,c] = w[c,0] * x[0,c]
        for ch in 0..c {
            let expected_t0 = 0.25 * x[ch];
            assert!(
                (out[ch] - expected_t0).abs() < 1e-5,
                "causal conv: at t=0 ch={ch}: got {}, expected {expected_t0}",
                out[ch]
            );
        }
    }

    /// RMSNorm: multi-row normalization consistency.
    #[test]
    fn rms_norm_multi_row() {
        let l = 3_usize;
        let d = 4_usize;
        let x: Vec<f32> = (1..=l * d).map(|i| i as f32).collect();
        let w = vec![1.0_f32; d];
        let out = rms_norm(&x, &w, l, d, 1e-5).expect("rms_norm");
        // Verify each row independently
        for t in 0..l {
            let row = &x[t * d..(t + 1) * d];
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
            let rms = (mean_sq + 1e-5).sqrt();
            for i in 0..d {
                let expected = row[i] / rms;
                assert!(
                    (out[t * d + i] - expected).abs() < 1e-5,
                    "rms_norm multi_row mismatch at t={t} i={i}"
                );
            }
        }
    }
}
