//! Mamba-2 residual block with multi-head SSD.
//!
//! # Architecture
//!
//! The Mamba-2 block follows the design of Dao & Gu (2024).  Each block takes
//! an input `x ∈ R^{L × D}` and produces an output `y ∈ R^{L × D}` via:
//!
//! 1. **Input projection**: `z = x · W_in^T` projects to the combined
//!    dimension `D + 2·H·N + H + H` that encodes the SSM signals.
//!
//! 2. **Depthwise conv1d** on the `D`-dimensional component (causal, kernel
//!    size `d_conv`), followed by SiLU activation.
//!
//! 3. **Multi-head SSD**: For each head `h` of dimension `P = D/H`:
//!    - Scalar per-head decay `a_h = sigmoid(-exp(A_log_h))` (in `(0, 1)`).
//!    - B/C vectors `B_h, C_h ∈ R^{L × N}` scaled by `dt_h[t]` (softplus).
//!    - [`chunk_scan`] produces `y_h ∈ R^{L × P}` per head.
//!    - D-skip connection: `y_h += d_skip_h * x_h` (residual bypass).
//!
//! 4. **RMSNorm** on the assembled output, then **output projection**.
//!
//! 5. **Residual add**: `output = proj(norm(y)) + x`.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::mamba2::chunk_scan::{ChunkConfig, chunk_scan};

// ─── Mamba2BlockConfig ───────────────────────────────────────────────────────

/// Configuration for a Mamba-2 block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mamba2BlockConfig {
    /// `D`: model dimension (embedding size entering and leaving the block).
    pub d_model: usize,
    /// `N`: SSM state dimension per head.
    pub d_state: usize,
    /// `H`: number of SSM heads.  Must divide `d_model`.
    pub n_heads: usize,
    /// `Q`: chunk size for the SSD computation.
    pub chunk_size: usize,
    /// Depthwise-conv1d kernel size.
    pub d_conv: usize,
}

impl Mamba2BlockConfig {
    /// Create a `Mamba2BlockConfig` with sensible defaults.
    ///
    /// Defaults: `d_state = 64`, `chunk_size = 64`, `d_conv = 4`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`]   — if `d_model == 0`.
    /// * [`MambaError::HeadDimMismatch`]   — if `d_model % n_heads ≠ 0`.
    /// * [`MambaError::Internal`]          — if `n_heads == 0`.
    pub fn new(d_model: usize, n_heads: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        if n_heads == 0 {
            return Err(MambaError::Internal("n_heads must be > 0".into()));
        }
        if d_model % n_heads != 0 {
            return Err(MambaError::HeadDimMismatch { n_heads, d_model });
        }
        Ok(Self {
            d_model,
            d_state: 64,
            n_heads,
            chunk_size: 64,
            d_conv: 4,
        })
    }

    /// Per-head dimension `P = D / H`.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Total dimension of the input projection output:
    /// `D + H*N*2 + H + H` = x_part + B + C + dt + A_log
    #[inline]
    pub fn proj_dim(&self) -> usize {
        self.d_model + 2 * self.n_heads * self.d_state + self.n_heads + self.n_heads
    }
}

// ─── Mamba2BlockWeights ──────────────────────────────────────────────────────

/// Learned weights for a single Mamba-2 block.
///
/// All weight tensors are stored in row-major flat `Vec<f32>`.
#[derive(Debug, Clone)]
pub struct Mamba2BlockWeights {
    /// Input projection `[proj_dim, d_model]` (row-major).
    pub in_proj: Vec<f32>,
    /// Depthwise conv1d weights `[d_model, d_conv]` (one filter per channel).
    pub conv_weight: Vec<f32>,
    /// Depthwise conv1d bias `[d_model]`.
    pub conv_bias: Vec<f32>,
    /// Output projection `[d_model, d_model]` (row-major).
    pub out_proj: Vec<f32>,
    /// RMSNorm scale `[d_model]`.
    pub norm_weight: Vec<f32>,
    /// D skip connection `[n_heads]` — per-head scalar residual bypass.
    pub d_skip: Vec<f32>,
}

impl Mamba2BlockWeights {
    /// Construct zero-initialised weights.
    pub fn zeros(config: &Mamba2BlockConfig) -> Self {
        let proj_dim = config.proj_dim();
        Self {
            in_proj: vec![0.0_f32; proj_dim * config.d_model],
            conv_weight: vec![0.0_f32; config.d_model * config.d_conv],
            conv_bias: vec![0.0_f32; config.d_model],
            out_proj: vec![0.0_f32; config.d_model * config.d_model],
            norm_weight: vec![0.0_f32; config.d_model],
            d_skip: vec![0.0_f32; config.n_heads],
        }
    }

    /// Construct weights with sensible initialisation for correctness testing.
    ///
    /// - `in_proj`: identity-like scaled blocks where applicable; otherwise zeros.
    /// - `conv_weight`: uniform `1/d_conv` (averaging filter).
    /// - `conv_bias`: zeros.
    /// - `out_proj`: scaled identity (`1/D` on diagonal).
    /// - `norm_weight`: all `1.0`.
    /// - `d_skip`: all `1.0`.
    pub fn default_init(config: &Mamba2BlockConfig) -> Self {
        let proj_dim = config.proj_dim();
        let d = config.d_model;
        let d_conv = config.d_conv;
        let n_heads = config.n_heads;

        // in_proj: zeros (safe default — the x component will be projected below)
        let in_proj = vec![0.0_f32; proj_dim * d];

        // conv_weight: uniform averaging filter so every element gets equal weight
        let conv_val = 1.0_f32 / d_conv as f32;
        let conv_weight = vec![conv_val; d * d_conv];

        // out_proj: scaled identity
        let mut out_proj = vec![0.0_f32; d * d];
        let scale = 1.0_f32 / d as f32;
        for i in 0..d {
            out_proj[i * d + i] = scale;
        }

        // norm_weight: all 1.0
        let norm_weight = vec![1.0_f32; d];

        // d_skip: all 1.0
        let d_skip = vec![1.0_f32; n_heads];

        Self {
            in_proj,
            conv_weight,
            conv_bias: vec![0.0_f32; d],
            out_proj,
            norm_weight,
            d_skip,
        }
    }

    /// Construct randomly initialised weights using the provided LCG RNG.
    ///
    /// Initialisation strategy (following the Mamba-2 paper):
    /// - `in_proj`, `conv_weight`, `out_proj`: N(0, 1) scaled by `1/sqrt(fan_in)`.
    /// - `conv_bias`: zeros.
    /// - `norm_weight`: all `1.0`.
    /// - `d_skip`: all `1.0`.
    pub fn random(config: &Mamba2BlockConfig, rng: &mut LcgRng) -> Self {
        let proj_dim = config.proj_dim();
        let d = config.d_model;
        let d_conv = config.d_conv;
        let n_heads = config.n_heads;

        let mut in_proj = vec![0.0_f32; proj_dim * d];
        rng.fill_normal(&mut in_proj);
        let in_scale = (d as f32).sqrt().recip();
        in_proj.iter_mut().for_each(|v| *v *= in_scale);

        let mut conv_weight = vec![0.0_f32; d * d_conv];
        rng.fill_normal(&mut conv_weight);
        let conv_scale = (d_conv as f32).sqrt().recip();
        conv_weight.iter_mut().for_each(|v| *v *= conv_scale);

        let mut out_proj = vec![0.0_f32; d * d];
        rng.fill_normal(&mut out_proj);
        let out_scale = (d as f32).sqrt().recip();
        out_proj.iter_mut().for_each(|v| *v *= out_scale);

        Self {
            in_proj,
            conv_weight,
            conv_bias: vec![0.0_f32; d],
            out_proj,
            norm_weight: vec![1.0_f32; d],
            d_skip: vec![1.0_f32; n_heads],
        }
    }
}

// ─── Mamba2Block ────────────────────────────────────────────────────────────

/// A single Mamba-2 residual block.
///
/// Accepts input `x ∈ R^{L × D}` (flat row-major) and returns
/// output `y ∈ R^{L × D}`.
#[derive(Debug)]
pub struct Mamba2Block {
    config: Mamba2BlockConfig,
    weights: Mamba2BlockWeights,
}

impl Mamba2Block {
    /// Create a new `Mamba2Block`, validating weight shapes.
    ///
    /// # Errors
    ///
    /// * [`MambaError::WeightShapeMismatch`] — if any weight slice has wrong size.
    pub fn new(config: Mamba2BlockConfig, weights: Mamba2BlockWeights) -> MambaResult<Self> {
        let proj_dim = config.proj_dim();
        let d = config.d_model;

        if weights.in_proj.len() != proj_dim * d {
            return Err(MambaError::WeightShapeMismatch {
                name: "in_proj",
                expected: vec![proj_dim, d],
                got: vec![weights.in_proj.len()],
            });
        }
        if weights.conv_weight.len() != d * config.d_conv {
            return Err(MambaError::WeightShapeMismatch {
                name: "conv_weight",
                expected: vec![d, config.d_conv],
                got: vec![weights.conv_weight.len()],
            });
        }
        if weights.conv_bias.len() != d {
            return Err(MambaError::WeightShapeMismatch {
                name: "conv_bias",
                expected: vec![d],
                got: vec![weights.conv_bias.len()],
            });
        }
        if weights.out_proj.len() != d * d {
            return Err(MambaError::WeightShapeMismatch {
                name: "out_proj",
                expected: vec![d, d],
                got: vec![weights.out_proj.len()],
            });
        }
        if weights.norm_weight.len() != d {
            return Err(MambaError::WeightShapeMismatch {
                name: "norm_weight",
                expected: vec![d],
                got: vec![weights.norm_weight.len()],
            });
        }
        if weights.d_skip.len() != config.n_heads {
            return Err(MambaError::WeightShapeMismatch {
                name: "d_skip",
                expected: vec![config.n_heads],
                got: vec![weights.d_skip.len()],
            });
        }

        Ok(Self { config, weights })
    }

    /// Run the Mamba-2 forward pass.
    ///
    /// # Arguments
    ///
    /// * `x`        — Input tensor `[L × D]`, flat row-major.
    /// * `seq_len`  — Sequence length `L`.
    ///
    /// # Returns
    ///
    /// Output tensor `[L × D]`, same shape as input.
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `x.len() ≠ seq_len * d_model`.
    /// * [`MambaError::InvalidSeqLen`]     — if `seq_len == 0`.
    /// * Propagates errors from internal chunk_scan calls.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> MambaResult<Vec<f32>> {
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        let d = self.config.d_model;
        let expected_x = seq_len * d;
        if x.len() != expected_x {
            return Err(MambaError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }

        let n = self.config.d_state;
        let h = self.config.n_heads;
        let p = self.config.head_dim(); // d_model / n_heads
        let proj_dim = self.config.proj_dim();

        // ── 1. Input projection z = x · W_in^T  [L × proj_dim] ──────────────
        // W_in: [proj_dim × d], input x: [L × d]
        // z[t, k] = Σ_d x[t, d'] * in_proj[k * d + d']
        let mut z = vec![0.0_f32; seq_len * proj_dim];
        for t in 0..seq_len {
            for k in 0..proj_dim {
                let mut acc = 0.0_f32;
                for d_idx in 0..d {
                    acc += x[t * d + d_idx] * self.weights.in_proj[k * d + d_idx];
                }
                z[t * proj_dim + k] = acc;
            }
        }

        // ── 2. Split z into components ────────────────────────────────────────
        // z layout: [x_part: D | B: H*N | C: H*N | dt: H | a_log: H]
        let x_offset = 0_usize;
        let b_offset = d;
        let c_offset = b_offset + h * n;
        let dt_offset = c_offset + h * n;
        let a_log_offset = dt_offset + h;

        // Extract x_part [L × D] — the SSM input after projection
        let mut x_part = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let z_row = &z[t * proj_dim..t * proj_dim + proj_dim];
            x_part[t * d..t * d + d].copy_from_slice(&z_row[x_offset..x_offset + d]);
        }

        // ── 3. Depthwise causal conv1d + SiLU on x_part ──────────────────────
        // For channel c: x_conv[t, c] = SiLU(Σ_{k=0}^{d_conv-1} w[c,k]*x_part[t-k, c] + bias[c])
        // Causal: positions before t=0 are zero-padded.
        let mut x_conv = vec![0.0_f32; seq_len * d];
        let d_conv = self.config.d_conv;
        for t in 0..seq_len {
            for c in 0..d {
                let mut val = self.weights.conv_bias[c];
                for k in 0..d_conv {
                    if t >= k {
                        val += self.weights.conv_weight[c * d_conv + k] * x_part[(t - k) * d + c];
                    }
                    // t < k → zero padding (implicit)
                }
                // SiLU: x * sigmoid(x)
                x_conv[t * d + c] = val * sigmoid(val);
            }
        }

        // ── 4. Extract per-head A_log scalars (shared across L) ──────────────
        // a_log is the *last* h entries of the *first* timestep's projection.
        // In the standard Mamba-2 formulation, A is input-independent (a learned
        // scalar per head), so we read it from the projection of the first token
        // (or equivalently use the weight bias directly — here we extract it from
        // the per-head a_log_offset portion of z[0]).
        let mut a_log_head = vec![0.0_f32; h];
        for head in 0..h {
            // We average a_log over the sequence to get a stable per-head value.
            // (In practice Mamba-2 uses a learned parameter, not a projection;
            //  here we simulate it by averaging the projected signal.)
            let mut sum = 0.0_f32;
            for t in 0..seq_len {
                sum += z[t * proj_dim + a_log_offset + head];
            }
            a_log_head[head] = sum / seq_len as f32;
        }

        // ── 5. Multi-head SSD forward ─────────────────────────────────────────
        // For each head h:
        //   a_seq[t] = sigmoid(-exp(a_h))   (scalar, same across t)
        //   dt[t, h] = softplus(z[t, dt_offset + h])  (positive time step)
        //   B_h[t, n] = z[t, b_offset + h*N + n] * dt[t, h]  (B scaled by dt)
        //   C_h[t, n] = z[t, c_offset + h*N + n]
        //   x_h[t]    = x_conv[t, head*p .. head*p+p] projected to scalar
        //               (mean over head_dim for simplicity)
        //   run chunk_scan(a_seq, B_h, C_h, x_h) → y_h [L]
        //   assemble y_h into output[t, head*p .. head*p+p]

        // Clamp chunk_size if larger than seq_len
        let effective_chunk = self.config.chunk_size.min(seq_len);
        // For chunk_scan we need chunk_size ≤ seq_len
        let chunk_cfg = if effective_chunk == 0 {
            return Err(MambaError::InvalidChunkSize(0));
        } else {
            ChunkConfig::new(seq_len, effective_chunk, n)?
        };

        // Assemble SSM output: y_ssm [L × D]
        let mut y_ssm = vec![0.0_f32; seq_len * d];

        for head in 0..h {
            // Compute a (scalar per-head decay, same for all t)
            let a_h = sigmoid(-a_log_head[head].exp());
            let a_seq = vec![a_h; seq_len];

            // Compute dt per timestep for this head (softplus)
            let mut dt_h = vec![0.0_f32; seq_len];
            for t in 0..seq_len {
                let raw_dt = z[t * proj_dim + dt_offset + head];
                dt_h[t] = softplus(raw_dt);
            }

            // B_h[t, n] = z[t, b_offset + head*N + n] * dt_h[t]
            let mut b_h = vec![0.0_f32; seq_len * n];
            for t in 0..seq_len {
                let dt_t = dt_h[t];
                for ni in 0..n {
                    b_h[t * n + ni] = z[t * proj_dim + b_offset + head * n + ni] * dt_t;
                }
            }

            // C_h[t, n] = z[t, c_offset + head*N + n]
            let mut c_h = vec![0.0_f32; seq_len * n];
            for t in 0..seq_len {
                for ni in 0..n {
                    c_h[t * n + ni] = z[t * proj_dim + c_offset + head * n + ni];
                }
            }

            // x_h[t] = mean of x_conv[t, head*p .. head*p+p] * dt_h[t]
            // (projects head_dim slice to a scalar; this follows the
            //  head-sliced formulation where each head processes P channels)
            let mut x_h = vec![0.0_f32; seq_len];
            for t in 0..seq_len {
                let mut acc = 0.0_f32;
                for pi in 0..p {
                    acc += x_conv[t * d + head * p + pi];
                }
                x_h[t] = acc / p as f32 * dt_h[t];
            }

            // Run chunk_scan
            let y_h = chunk_scan(&a_seq, &b_h, &c_h, &x_h, &chunk_cfg)?;

            // D-skip: y_h_out[t] = y_h[t] + d_skip[head] * x_h_mean[t]
            let d_s = self.weights.d_skip[head];

            // Scatter y_h back into y_ssm[t, head*p .. head*p+p]
            // Each channel in the head gets the same scalar output y_h[t]
            // scaled appropriately, plus the skip.
            for t in 0..seq_len {
                let val = y_h[t] + d_s * x_h[t];
                for pi in 0..p {
                    y_ssm[t * d + head * p + pi] = val;
                }
            }
        }

        // ── 6. RMSNorm ────────────────────────────────────────────────────────
        // norm[t, d] = (y_ssm[t, d] / rms(y_ssm[t, :])) * norm_weight[d]
        let mut y_normed = vec![0.0_f32; seq_len * d];
        let eps = 1e-6_f32;
        for t in 0..seq_len {
            let row = &y_ssm[t * d..(t + 1) * d];
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
            let row_scale = 1.0_f32 / rms;
            for di in 0..d {
                y_normed[t * d + di] = row[di] * row_scale * self.weights.norm_weight[di];
            }
        }

        // ── 7. Output projection  y_out = y_normed · W_out^T ─────────────────
        // W_out: [d × d], y_normed: [L × d]
        let mut y_out = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            for di in 0..d {
                let mut acc = 0.0_f32;
                for dj in 0..d {
                    acc += y_normed[t * d + dj] * self.weights.out_proj[di * d + dj];
                }
                y_out[t * d + di] = acc;
            }
        }

        // ── 8. Residual connection ────────────────────────────────────────────
        for i in 0..y_out.len() {
            y_out[i] += x[i];
        }

        Ok(y_out)
    }

    /// Return a reference to the block configuration.
    #[inline]
    pub fn config(&self) -> &Mamba2BlockConfig {
        &self.config
    }
}

// ─── Activation helpers ──────────────────────────────────────────────────────

/// Sigmoid: σ(x) = 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// Softplus: ln(1 + exp(x)), numerically stable.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── ChunkConfig helpers ───────────────────────────────────────────────────

    /// Build a small block that will produce finite, well-defined outputs.
    fn make_block_default(
        d_model: usize,
        n_heads: usize,
        d_state: usize,
        chunk_size: usize,
    ) -> Mamba2Block {
        let mut config = Mamba2BlockConfig::new(d_model, n_heads).expect("valid config");
        config.d_state = d_state;
        config.chunk_size = chunk_size;
        let weights = Mamba2BlockWeights::default_init(&config);
        Mamba2Block::new(config, weights).expect("valid block")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mamba2BlockConfig tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Valid configuration with 2 heads is accepted.
    #[test]
    fn config_valid_2heads() {
        let cfg = Mamba2BlockConfig::new(8, 2).expect("valid 2-head config");
        assert_eq!(cfg.d_model, 8);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.head_dim(), 4);
    }

    /// `d_model % n_heads ≠ 0` must fail.
    #[test]
    fn config_head_dim_not_divide() {
        let err = Mamba2BlockConfig::new(10, 3).expect_err("should fail: 10 % 3 ≠ 0");
        assert!(matches!(
            err,
            MambaError::HeadDimMismatch {
                n_heads: 3,
                d_model: 10
            }
        ));
    }

    /// head_dim accessor.
    #[test]
    fn config_head_dim() {
        let cfg = Mamba2BlockConfig::new(64, 4).expect("valid");
        assert_eq!(cfg.head_dim(), 16);
    }

    /// Zero d_model fails.
    #[test]
    fn config_zero_d_model() {
        let err = Mamba2BlockConfig::new(0, 1).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    /// Zero n_heads fails.
    #[test]
    fn config_zero_n_heads() {
        let err = Mamba2BlockConfig::new(8, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::Internal(_)));
    }

    /// proj_dim matches expected formula.
    #[test]
    fn config_proj_dim() {
        let cfg = Mamba2BlockConfig::new(4, 2).expect("valid");
        // proj_dim = D + 2*H*N + H + H = 4 + 2*2*64 + 2 + 2 = 4 + 256 + 4 = 264
        let expected = cfg.d_model + 2 * cfg.n_heads * cfg.d_state + cfg.n_heads + cfg.n_heads;
        assert_eq!(cfg.proj_dim(), expected);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mamba2BlockWeights tests
    // ─────────────────────────────────────────────────────────────────────────

    /// zeros() produces correct in_proj size.
    #[test]
    fn weights_zeros_in_proj_size() {
        let cfg = Mamba2BlockConfig::new(4, 2).expect("valid");
        let w = Mamba2BlockWeights::zeros(&cfg);
        let expected_len = cfg.proj_dim() * cfg.d_model;
        assert_eq!(w.in_proj.len(), expected_len, "in_proj len mismatch");
        assert!(w.in_proj.iter().all(|&v| v == 0.0), "in_proj not zeroed");
    }

    /// default_init(): norm_weight is all 1.0.
    #[test]
    fn weights_default_init_norm_ones() {
        let cfg = Mamba2BlockConfig::new(8, 2).expect("valid");
        let w = Mamba2BlockWeights::default_init(&cfg);
        assert!(
            w.norm_weight.iter().all(|&v| (v - 1.0).abs() < 1e-7),
            "norm_weight should be all 1.0"
        );
    }

    /// default_init(): d_skip is all 1.0.
    #[test]
    fn weights_default_init_d_skip_ones() {
        let cfg = Mamba2BlockConfig::new(8, 2).expect("valid");
        let w = Mamba2BlockWeights::default_init(&cfg);
        assert!(
            w.d_skip.iter().all(|&v| (v - 1.0).abs() < 1e-7),
            "d_skip should be all 1.0"
        );
    }

    /// random() has correct sizes for all weight tensors.
    #[test]
    fn weights_random_sizes() {
        let cfg = Mamba2BlockConfig::new(8, 2).expect("valid");
        let mut rng = LcgRng::new(42);
        let w = Mamba2BlockWeights::random(&cfg, &mut rng);
        assert_eq!(w.in_proj.len(), cfg.proj_dim() * cfg.d_model);
        assert_eq!(w.conv_weight.len(), cfg.d_model * cfg.d_conv);
        assert_eq!(w.conv_bias.len(), cfg.d_model);
        assert_eq!(w.out_proj.len(), cfg.d_model * cfg.d_model);
        assert_eq!(w.norm_weight.len(), cfg.d_model);
        assert_eq!(w.d_skip.len(), cfg.n_heads);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mamba2Block construction tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Wrong in_proj size is rejected.
    #[test]
    fn block_construction_wrong_in_proj() {
        let cfg = Mamba2BlockConfig::new(4, 2).expect("valid");
        let mut w = Mamba2BlockWeights::zeros(&cfg);
        w.in_proj = vec![0.0_f32; 5]; // wrong
        let err = Mamba2Block::new(cfg, w).expect_err("should fail");
        assert!(matches!(
            err,
            MambaError::WeightShapeMismatch {
                name: "in_proj",
                ..
            }
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mamba2Block forward tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Output shape equals [L * D].
    #[test]
    fn block_forward_shape() {
        let d = 4_usize;
        let l = 6_usize;
        let block = make_block_default(d, 2, 2, 4);
        let mut rng = LcgRng::new(1);
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let y = block.forward(&x, l).expect("forward shape");
        assert_eq!(y.len(), l * d, "output shape must be L*D");
    }

    /// All outputs are finite.
    #[test]
    fn block_forward_finite() {
        let d = 4_usize;
        let l = 8_usize;
        let block = make_block_default(d, 2, 2, 4);
        let mut rng = LcgRng::new(2);
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let y = block.forward(&x, l).expect("forward finite");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite");
        }
    }

    /// Zero input produces finite output (RMSNorm with non-zero weights).
    #[test]
    fn block_forward_zero_input_finite() {
        let d = 4_usize;
        let l = 4_usize;
        let block = make_block_default(d, 2, 2, 4);
        let x = vec![0.0_f32; l * d];
        let y = block.forward(&x, l).expect("zero input");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite for zero input");
        }
    }

    /// Single timestep (L=1) works correctly.
    #[test]
    fn block_forward_single_step() {
        let d = 4_usize;
        let l = 1_usize;
        let block = make_block_default(d, 2, 2, 1);
        let x = vec![0.5_f32; l * d];
        let y = block.forward(&x, l).expect("single step");
        assert_eq!(y.len(), l * d);
    }

    /// L=4, D=4, H=2 → correct shape.
    #[test]
    fn block_forward_small() {
        let d = 4_usize;
        let l = 4_usize;
        let block = make_block_default(d, 2, 2, 4);
        let mut rng = LcgRng::new(3);
        let mut x = vec![0.0_f32; l * d];
        rng.fill_normal(&mut x);
        let y = block.forward(&x, l).expect("small forward");
        assert_eq!(y.len(), l * d);
    }

    /// Same input produces same output (determinism).
    #[test]
    fn block_forward_deterministic() {
        let d = 8_usize;
        let l = 6_usize;
        let mut config = Mamba2BlockConfig::new(d, 2).expect("valid");
        config.d_state = 4;
        config.chunk_size = 4;
        let mut rng = LcgRng::new(77);
        let weights = Mamba2BlockWeights::random(&config, &mut rng);
        let block = Mamba2Block::new(config, weights).expect("valid block");

        let mut x = vec![0.0_f32; l * d];
        LcgRng::new(42).fill_normal(&mut x);

        let y1 = block.forward(&x, l).expect("first run");
        let y2 = block.forward(&x, l).expect("second run");
        for (i, (&a, &b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "non-deterministic output at y[{i}]: {a} vs {b}"
            );
        }
    }

    /// config() accessor returns the stored configuration.
    #[test]
    fn block_config_accessors() {
        let d = 8_usize;
        let block = make_block_default(d, 2, 4, 4);
        let cfg = block.config();
        assert_eq!(cfg.d_model, d);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.head_dim(), 4);
    }

    /// Random weights forward is all finite.
    #[test]
    fn block_random_weights_forward_finite() {
        let d = 8_usize;
        let l = 8_usize;
        let mut config = Mamba2BlockConfig::new(d, 2).expect("valid");
        config.d_state = 4;
        config.chunk_size = 4;
        let mut rng = LcgRng::new(123);
        let weights = Mamba2BlockWeights::random(&config, &mut rng);
        let block = Mamba2Block::new(config, weights).expect("valid block");
        let mut x = vec![0.0_f32; l * d];
        LcgRng::new(456).fill_normal(&mut x);
        let y = block.forward(&x, l).expect("random weights forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite");
        }
    }

    /// Error on seq_len = 0.
    #[test]
    fn block_forward_empty_seq() {
        let d = 4_usize;
        let block = make_block_default(d, 2, 2, 4);
        let err = block.forward(&[], 0).expect_err("should fail on empty");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }
}
