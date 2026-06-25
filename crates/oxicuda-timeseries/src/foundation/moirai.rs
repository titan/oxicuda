//! Moirai universal time-series forecaster (Woo et al., Salesforce, 2024).
//!
//! Reference: *"Unified Training of Universal Time Series Forecasting
//! Transformers"* (Woo, Liu, Kumar, Xiong, Savarese, Sahoo; ICML 2024).
//!
//! This is a compact, faithful CPU core of the Moirai architecture:
//!
//! * **Any-variate patching.** A multivariate `[T, C]` series is split along
//!   the time axis into non-overlapping patches of length `patch_size`
//!   (the last patch is zero-padded so the patches always *cover* the full
//!   input — `n_patches = ceil(T / patch_size)`). Every variate's patches are
//!   flattened into a *single* token sequence so the model handles an arbitrary
//!   number of variates. A learned **variate-id embedding** is added to each
//!   token so the encoder can tell the variates apart.
//! * **Masked encoder.** A stack of pre-norm Transformer blocks (multi-head
//!   self-attention + position-wise feed-forward) processes the token sequence.
//!   The forecast horizon is represented by additional *target* tokens that
//!   carry only a learned **mask embedding** (no future data). A block-structured
//!   attention mask lets the target tokens read from the observed context while
//!   the context tokens attend only among themselves, so the future placeholders
//!   never leak into the context representation.
//! * **Multi-patch-size support.** The patch size is configurable (the paper
//!   trains several patch sizes selected per data frequency; here a single
//!   configurable size is used per model instance — see [`MoiraiConfig`]).
//! * **Distributional output head.** Each target patch is decoded into a
//!   per-future-step `(mean, log-scale)` pair. We use a **Gaussian** output
//!   distribution (the paper uses a mixture / Student-t; Gaussian is a faithful
//!   compact substitute). The scale is `exp(log_scale)` (clamped for numerical
//!   safety) and is therefore strictly positive.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for a [`MoiraiForecaster`].
#[derive(Debug, Clone)]
pub struct MoiraiConfig {
    /// Length of each (non-overlapping) time patch.
    pub patch_size: usize,
    /// Token embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of Transformer encoder layers.
    pub n_layers: usize,
    /// Feed-forward hidden expansion factor (applied to `d_model`).
    pub ffn_expansion: usize,
    /// Size of the learned variate-id embedding table. Variate `v` uses row
    /// `v % max_variates`, so any number of variates is supported.
    pub max_variates: usize,
}

impl MoiraiConfig {
    /// Small configuration suitable for tests and CPU smoke runs.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            patch_size: 4,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            ffn_expansion: 2,
            max_variates: 8,
        }
    }

    /// Base configuration (`d=128, heads=8, layers=4`).
    #[must_use]
    pub fn base() -> Self {
        Self {
            patch_size: 16,
            d_model: 128,
            n_heads: 8,
            n_layers: 4,
            ffn_expansion: 4,
            max_variates: 128,
        }
    }
}

// ─── Layer weights ──────────────────────────────────────────────────────────

/// Learnable parameters for one pre-norm Transformer encoder layer.
#[derive(Debug, Clone)]
pub struct MoiraiLayer {
    /// Pre-attention LayerNorm scale `[D]`.
    pub norm1_g: Vec<f32>,
    /// Pre-attention LayerNorm bias `[D]`.
    pub norm1_b: Vec<f32>,
    /// Query projection `[D, D]`.
    pub q_w: Vec<f32>,
    /// Key projection `[D, D]`.
    pub k_w: Vec<f32>,
    /// Value projection `[D, D]`.
    pub v_w: Vec<f32>,
    /// Output projection `[D, D]`.
    pub out_w: Vec<f32>,
    /// Pre-FFN LayerNorm scale `[D]`.
    pub norm2_g: Vec<f32>,
    /// Pre-FFN LayerNorm bias `[D]`.
    pub norm2_b: Vec<f32>,
    /// FFN first weight `[D*expansion, D]`.
    pub ff_w1: Vec<f32>,
    /// FFN first bias `[D*expansion]`.
    pub ff_b1: Vec<f32>,
    /// FFN second weight `[D, D*expansion]`.
    pub ff_w2: Vec<f32>,
    /// FFN second bias `[D]`.
    pub ff_b2: Vec<f32>,
}

impl MoiraiLayer {
    fn new(d: usize, expansion: usize, rng: &mut LcgRng) -> Self {
        let d_ff = d * expansion;
        let mut init = |rows: usize, cols: usize| xavier(rows, cols, rng);
        Self {
            norm1_g: vec![1.0; d],
            norm1_b: vec![0.0; d],
            q_w: init(d, d),
            k_w: init(d, d),
            v_w: init(d, d),
            out_w: init(d, d),
            norm2_g: vec![1.0; d],
            norm2_b: vec![0.0; d],
            ff_w1: init(d_ff, d),
            ff_b1: vec![0.0; d_ff],
            ff_w2: init(d, d_ff),
            ff_b2: vec![0.0; d],
        }
    }
}

// ─── Output ─────────────────────────────────────────────────────────────────

/// Distributional forecast produced by [`MoiraiForecaster::forward`].
///
/// Both buffers use **horizon-major `[H, C]`** layout (`idx = h * n_variates + c`).
#[derive(Debug, Clone)]
pub struct MoiraiForecast {
    /// Predicted mean per `(horizon, variate)` — the point forecast `[H, C]`.
    pub point: Vec<f32>,
    /// Predicted Gaussian scale (std-dev) per `(horizon, variate)` `[H, C]`;
    /// every entry is strictly positive.
    pub scale: Vec<f32>,
    /// Forecast horizon `H`.
    pub horizon: usize,
    /// Number of variates `C`.
    pub n_variates: usize,
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Moirai universal forecaster.
#[derive(Debug, Clone)]
pub struct MoiraiForecaster {
    /// Patch-embedding projection `[D, patch_size]`.
    pub patch_w: Vec<f32>,
    /// Patch-embedding bias `[D]`.
    pub patch_b: Vec<f32>,
    /// Learned variate-id embedding table `[max_variates, D]`.
    pub variate_emb: Vec<f32>,
    /// Learned mask/target token embedding `[D]`.
    pub mask_token: Vec<f32>,
    /// Encoder layers.
    pub layers: Vec<MoiraiLayer>,
    /// Final LayerNorm scale `[D]`.
    pub final_g: Vec<f32>,
    /// Final LayerNorm bias `[D]`.
    pub final_b: Vec<f32>,
    /// Distributional head weight `[2*patch_size, D]` (`mean`‖`log_scale`).
    pub head_w: Vec<f32>,
    /// Distributional head bias `[2*patch_size]`.
    pub head_b: Vec<f32>,
    /// Model configuration.
    pub cfg: MoiraiConfig,
}

impl MoiraiForecaster {
    /// Build a Moirai forecaster, initialising all parameters from `rng`.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidPatchLen`] when `patch_size == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    /// - [`TsError::InvalidNumVariates`] when `max_variates == 0`.
    /// - [`TsError::ShapeMismatch`] when `n_layers == 0` or `ffn_expansion == 0`.
    pub fn new(cfg: MoiraiConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.patch_size == 0 {
            return Err(TsError::InvalidPatchLen(0));
        }
        if cfg.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if cfg.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if cfg.d_model % cfg.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: cfg.d_model,
                n_heads: cfg.n_heads,
            });
        }
        if cfg.max_variates == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if cfg.n_layers == 0 {
            return Err(TsError::ShapeMismatch {
                msg: "n_layers must be >= 1".into(),
            });
        }
        if cfg.ffn_expansion == 0 {
            return Err(TsError::ShapeMismatch {
                msg: "ffn_expansion must be >= 1".into(),
            });
        }

        let d = cfg.d_model;
        let p = cfg.patch_size;

        let patch_w = xavier(d, p, rng);
        let patch_b = vec![0.0; d];

        let mut variate_emb = vec![0.0; cfg.max_variates * d];
        rng.fill_normal(&mut variate_emb);
        for v in &mut variate_emb {
            *v *= 0.02;
        }

        let mut mask_token = vec![0.0; d];
        rng.fill_normal(&mut mask_token);
        for v in &mut mask_token {
            *v *= 0.02;
        }

        let layers = (0..cfg.n_layers)
            .map(|_| MoiraiLayer::new(d, cfg.ffn_expansion, rng))
            .collect();

        let head_w = xavier(2 * p, d, rng);
        let head_b = vec![0.0; 2 * p];

        Ok(Self {
            patch_w,
            patch_b,
            variate_emb,
            mask_token,
            layers,
            final_g: vec![1.0; d],
            final_b: vec![0.0; d],
            head_w,
            head_b,
            cfg,
        })
    }

    /// Number of context patches needed to *cover* a length-`t` series:
    /// `ceil(t / patch_size)`.
    #[must_use]
    pub fn num_context_patches(&self, t: usize) -> usize {
        t.div_ceil(self.cfg.patch_size)
    }

    /// Number of target patches needed to cover a length-`horizon` forecast.
    #[must_use]
    pub fn num_target_patches(&self, horizon: usize) -> usize {
        horizon.div_ceil(self.cfg.patch_size)
    }

    /// Run the forecaster.
    ///
    /// # Arguments
    ///
    /// * `series` — `[T, C]` row-major (time-major) multivariate series.
    /// * `n_variates` — number of variates `C`; `series.len()` must be a
    ///   multiple of it.
    /// * `horizon` — number of future steps to forecast.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidNumVariates`] when `n_variates == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::DimensionMismatch`] when `series.len()` is not a multiple of
    ///   `n_variates`.
    /// - [`TsError::EmptyInput`] when the series is empty.
    pub fn forward(
        &self,
        series: &[f32],
        n_variates: usize,
        horizon: usize,
    ) -> TsResult<MoiraiForecast> {
        let (tokens, n_ctx_total, n_tgt_total, t) =
            self.build_tokens(series, n_variates, horizon)?;
        let encoded = self.encode(&tokens, n_ctx_total, n_tgt_total);

        let d = self.cfg.d_model;
        let p = self.cfg.patch_size;
        let n_tgt = self.num_target_patches(horizon);
        let _ = t;

        let mut point = vec![0.0; horizon * n_variates];
        let mut scale = vec![0.0; horizon * n_variates];

        for ci in 0..n_variates {
            for tp in 0..n_tgt {
                let tok = n_ctx_total + ci * n_tgt + tp;
                let h = &encoded[tok * d..(tok + 1) * d];
                // head: [2P, D] · [D] -> [2P]
                let out = linear(h, &self.head_w, &self.head_b, d, 2 * p);
                for k in 0..p {
                    let step = tp * p + k;
                    if step >= horizon {
                        break;
                    }
                    let mean = out[k];
                    let log_scale = out[p + k].clamp(-15.0, 15.0);
                    let idx = step * n_variates + ci;
                    point[idx] = mean;
                    scale[idx] = log_scale.exp();
                }
            }
        }

        Ok(MoiraiForecast {
            point,
            scale,
            horizon,
            n_variates,
        })
    }

    /// First-layer, first-head self-attention weights over the full token
    /// sequence, row-major `[N, N]`. Each query row sums to 1 (rows for context
    /// queries place exactly `0` on the masked target keys).
    ///
    /// # Errors
    ///
    /// Mirrors [`Self::forward`].
    pub fn attention_weights(
        &self,
        series: &[f32],
        n_variates: usize,
        horizon: usize,
    ) -> TsResult<Vec<f32>> {
        let (tokens, n_ctx_total, n_tgt_total, _t) =
            self.build_tokens(series, n_variates, horizon)?;
        let n = n_ctx_total + n_tgt_total;
        let d = self.cfg.d_model;
        let layer = &self.layers[0];

        let mut normed = tokens.clone();
        layer_norm(&mut normed, d, &layer.norm1_g, &layer.norm1_b);
        let q = matmul_rows(&normed, &layer.q_w, n, d);
        let k = matmul_rows(&normed, &layer.k_w, n, d);

        let head_dim = d / self.cfg.n_heads;
        let scale = (head_dim as f32).sqrt().recip();
        let mut weights = vec![0.0; n * n];
        for qi in 0..n {
            for ki in 0..n {
                let val = if attn_allowed(qi, ki, n_ctx_total) {
                    let mut dot = 0.0;
                    for hd in 0..head_dim {
                        dot += q[qi * d + hd] * k[ki * d + hd];
                    }
                    dot * scale
                } else {
                    f32::NEG_INFINITY
                };
                weights[qi * n + ki] = val;
            }
        }
        for qi in 0..n {
            softmax_row(&mut weights[qi * n..(qi + 1) * n]);
        }
        Ok(weights)
    }

    /// Build the flattened token sequence for context + target patches.
    ///
    /// Returns `(tokens [N, D], n_ctx_total, n_tgt_total, t)` where the first
    /// `n_ctx_total` tokens are observed context and the remainder are masked
    /// target tokens. `t` is the inferred sequence length per variate.
    fn build_tokens(
        &self,
        series: &[f32],
        n_variates: usize,
        horizon: usize,
    ) -> TsResult<(Vec<f32>, usize, usize, usize)> {
        if n_variates == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if series.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "series must not be empty".into(),
            });
        }
        if series.len() % n_variates != 0 {
            return Err(TsError::DimensionMismatch {
                expected: series.len() - (series.len() % n_variates),
                got: series.len(),
            });
        }
        let t = series.len() / n_variates;

        let d = self.cfg.d_model;
        let p = self.cfg.patch_size;
        let n_ctx = self.num_context_patches(t);
        let n_tgt = self.num_target_patches(horizon);
        let n_ctx_total = n_variates * n_ctx;
        let n_tgt_total = n_variates * n_tgt;
        let n = n_ctx_total + n_tgt_total;
        let pos = sinusoidal_pos_enc(n_ctx + n_tgt, d);

        let mut tokens = vec![0.0; n * d];

        // Context tokens: real patch values + variate emb + position emb.
        let mut patch_buf = vec![0.0; p];
        for ci in 0..n_variates {
            let ve = &self.variate_emb[(ci % self.cfg.max_variates) * d..];
            for cp in 0..n_ctx {
                for (k, pb) in patch_buf.iter_mut().enumerate() {
                    let time = cp * p + k;
                    *pb = if time < t {
                        series[time * n_variates + ci]
                    } else {
                        0.0
                    };
                }
                let proj = linear(&patch_buf, &self.patch_w, &self.patch_b, p, d);
                let tok = ci * n_ctx + cp;
                let row = &mut tokens[tok * d..(tok + 1) * d];
                let pe = &pos[cp * d..(cp + 1) * d];
                for j in 0..d {
                    row[j] = proj[j] + ve[j] + pe[j];
                }
            }
        }

        // Target tokens: learned mask embedding + variate emb + position emb.
        for ci in 0..n_variates {
            let ve = &self.variate_emb[(ci % self.cfg.max_variates) * d..];
            for tp in 0..n_tgt {
                let tok = n_ctx_total + ci * n_tgt + tp;
                let row = &mut tokens[tok * d..(tok + 1) * d];
                let pe = &pos[(n_ctx + tp) * d..(n_ctx + tp + 1) * d];
                for j in 0..d {
                    row[j] = self.mask_token[j] + ve[j] + pe[j];
                }
            }
        }

        Ok((tokens, n_ctx_total, n_tgt_total, t))
    }

    /// Run the pre-norm Transformer stack with the block-structured mask.
    fn encode(&self, tokens: &[f32], n_ctx_total: usize, n_tgt_total: usize) -> Vec<f32> {
        let n = n_ctx_total + n_tgt_total;
        let d = self.cfg.d_model;
        let mut x = tokens.to_vec();
        for layer in &self.layers {
            let delta = mhsa(&x, n, n_ctx_total, d, layer, self.cfg.n_heads);
            for (xi, di) in x.iter_mut().zip(delta.iter()) {
                *xi += di;
            }
            let fdelta = ffn(&x, n, d, layer, self.cfg.ffn_expansion);
            for (xi, di) in x.iter_mut().zip(fdelta.iter()) {
                *xi += di;
            }
        }
        layer_norm(&mut x, d, &self.final_g, &self.final_b);
        x
    }
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Xavier-uniform-magnitude initialised `[rows, cols]` matrix.
fn xavier(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
    let mut v = vec![0.0; rows * cols];
    rng.fill_normal(&mut v);
    for x in &mut v {
        *x *= scale;
    }
    v
}

/// Single linear layer: `y[o] = b[o] + Σ_k w[o*in+k] x[k]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; out_dim];
    for (o, ov) in out.iter_mut().enumerate() {
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = b[o];
        for k in 0..in_dim {
            acc += w_row[k] * x[k];
        }
        *ov = acc;
    }
    out
}

/// Apply `[D, D]` projection row-wise to a `[N, D]` token matrix.
fn matmul_rows(x: &[f32], w: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * d];
    for i in 0..n {
        for o in 0..d {
            let mut acc = 0.0;
            let w_row = &w[o * d..(o + 1) * d];
            let x_row = &x[i * d..(i + 1) * d];
            for k in 0..d {
                acc += w_row[k] * x_row[k];
            }
            out[i * d + o] = acc;
        }
    }
    out
}

/// Whether query token `i` may attend to key token `j` given `n_ctx` context
/// tokens. Context queries attend only among the context block; target queries
/// attend to everything.
#[inline]
fn attn_allowed(i: usize, j: usize, n_ctx: usize) -> bool {
    if i < n_ctx { j < n_ctx } else { true }
}

/// In-place row-wise LayerNorm over the last dimension `d`, eps = 1e-5.
fn layer_norm(x: &mut [f32], d: usize, gamma: &[f32], beta: &[f32]) {
    if d == 0 {
        return;
    }
    let n = x.len() / d;
    for i in 0..n {
        let row = &mut x[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = (var + 1e-5).sqrt().recip();
        for (j, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

/// GELU (tanh approximation).
#[inline]
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

/// In-place numerically-stable softmax that treats `-inf` entries as zero
/// probability.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All entries masked: leave as-is (should not happen — diagonal allowed).
        return;
    }
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Sinusoidal positional encoding `[len, d]`.
fn sinusoidal_pos_enc(len: usize, d: usize) -> Vec<f32> {
    let mut pe = vec![0.0; len * d];
    for p in 0..len {
        for i in 0..d / 2 {
            let freq = 10000.0_f32.powf((2 * i) as f32 / d as f32);
            pe[p * d + 2 * i] = (p as f32 / freq).sin();
            pe[p * d + 2 * i + 1] = (p as f32 / freq).cos();
        }
        if d % 2 == 1 {
            let i = d / 2;
            let freq = 10000.0_f32.powf((2 * i) as f32 / d as f32);
            pe[p * d + 2 * i] = (p as f32 / freq).sin();
        }
    }
    pe
}

/// Masked multi-head self-attention (pre-norm). Returns the residual delta.
fn mhsa(x: &[f32], n: usize, n_ctx: usize, d: usize, lw: &MoiraiLayer, n_heads: usize) -> Vec<f32> {
    let head_dim = d / n_heads;
    let scale = (head_dim as f32).sqrt().recip();

    let mut normed = x.to_vec();
    layer_norm(&mut normed, d, &lw.norm1_g, &lw.norm1_b);

    let q = matmul_rows(&normed, &lw.q_w, n, d);
    let k = matmul_rows(&normed, &lw.k_w, n, d);
    let v = matmul_rows(&normed, &lw.v_w, n, d);

    let mut attn_out = vec![0.0; n * d];
    let mut scores = vec![0.0; n];

    for h in 0..n_heads {
        let h0 = h * head_dim;
        for qi in 0..n {
            for ki in 0..n {
                scores[ki] = if attn_allowed(qi, ki, n_ctx) {
                    let mut dot = 0.0;
                    for hd in 0..head_dim {
                        dot += q[qi * d + h0 + hd] * k[ki * d + h0 + hd];
                    }
                    dot * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
            softmax_row(&mut scores);
            for hd in 0..head_dim {
                let mut acc = 0.0;
                for ki in 0..n {
                    acc += scores[ki] * v[ki * d + h0 + hd];
                }
                attn_out[qi * d + h0 + hd] = acc;
            }
        }
    }

    matmul_rows(&attn_out, &lw.out_w, n, d)
}

/// Position-wise feed-forward block (pre-norm). Returns the residual delta.
fn ffn(x: &[f32], n: usize, d: usize, lw: &MoiraiLayer, expansion: usize) -> Vec<f32> {
    let d_ff = d * expansion;
    let mut normed = x.to_vec();
    layer_norm(&mut normed, d, &lw.norm2_g, &lw.norm2_b);

    let mut out = vec![0.0; n * d];
    let mut hidden = vec![0.0; d_ff];
    for i in 0..n {
        let row = &normed[i * d..(i + 1) * d];
        for (fi, hv) in hidden.iter_mut().enumerate() {
            let w_row = &lw.ff_w1[fi * d..(fi + 1) * d];
            let mut acc = lw.ff_b1[fi];
            for k in 0..d {
                acc += w_row[k] * row[k];
            }
            *hv = gelu(acc);
        }
        for o in 0..d {
            let mut acc = lw.ff_b2[o];
            for (fi, &hv) in hidden.iter().enumerate() {
                acc += hv * lw.ff_w2[o * d_ff + fi];
            }
            out[i * d + o] = acc;
        }
    }
    out
}

// ─── Foundation-model adapter (checkpoint export / import) ───────────────────

impl crate::foundation::adapter::FoundationAdapter for MoiraiForecaster {
    fn export_weights(&self) -> crate::foundation::adapter::WeightStore {
        let mut s = crate::foundation::adapter::WeightStore::new();
        s.insert("patch_w", self.patch_w.clone());
        s.insert("patch_b", self.patch_b.clone());
        s.insert("variate_emb", self.variate_emb.clone());
        s.insert("mask_token", self.mask_token.clone());
        s.insert("final_g", self.final_g.clone());
        s.insert("final_b", self.final_b.clone());
        s.insert("head_w", self.head_w.clone());
        s.insert("head_b", self.head_b.clone());
        for (li, layer) in self.layers.iter().enumerate() {
            s.insert(format!("layer{li}.norm1_g"), layer.norm1_g.clone());
            s.insert(format!("layer{li}.norm1_b"), layer.norm1_b.clone());
            s.insert(format!("layer{li}.q_w"), layer.q_w.clone());
            s.insert(format!("layer{li}.k_w"), layer.k_w.clone());
            s.insert(format!("layer{li}.v_w"), layer.v_w.clone());
            s.insert(format!("layer{li}.out_w"), layer.out_w.clone());
            s.insert(format!("layer{li}.norm2_g"), layer.norm2_g.clone());
            s.insert(format!("layer{li}.norm2_b"), layer.norm2_b.clone());
            s.insert(format!("layer{li}.ff_w1"), layer.ff_w1.clone());
            s.insert(format!("layer{li}.ff_b1"), layer.ff_b1.clone());
            s.insert(format!("layer{li}.ff_w2"), layer.ff_w2.clone());
            s.insert(format!("layer{li}.ff_b2"), layer.ff_b2.clone());
        }
        s
    }

    fn import_weights(&mut self, store: &crate::foundation::adapter::WeightStore) -> TsResult<()> {
        self.patch_w = store.require_len("patch_w", self.patch_w.len())?.to_vec();
        self.patch_b = store.require_len("patch_b", self.patch_b.len())?.to_vec();
        self.variate_emb = store
            .require_len("variate_emb", self.variate_emb.len())?
            .to_vec();
        self.mask_token = store
            .require_len("mask_token", self.mask_token.len())?
            .to_vec();
        self.final_g = store.require_len("final_g", self.final_g.len())?.to_vec();
        self.final_b = store.require_len("final_b", self.final_b.len())?.to_vec();
        self.head_w = store.require_len("head_w", self.head_w.len())?.to_vec();
        self.head_b = store.require_len("head_b", self.head_b.len())?.to_vec();
        for (li, layer) in self.layers.iter_mut().enumerate() {
            layer.norm1_g = store
                .require_len(&format!("layer{li}.norm1_g"), layer.norm1_g.len())?
                .to_vec();
            layer.norm1_b = store
                .require_len(&format!("layer{li}.norm1_b"), layer.norm1_b.len())?
                .to_vec();
            layer.q_w = store
                .require_len(&format!("layer{li}.q_w"), layer.q_w.len())?
                .to_vec();
            layer.k_w = store
                .require_len(&format!("layer{li}.k_w"), layer.k_w.len())?
                .to_vec();
            layer.v_w = store
                .require_len(&format!("layer{li}.v_w"), layer.v_w.len())?
                .to_vec();
            layer.out_w = store
                .require_len(&format!("layer{li}.out_w"), layer.out_w.len())?
                .to_vec();
            layer.norm2_g = store
                .require_len(&format!("layer{li}.norm2_g"), layer.norm2_g.len())?
                .to_vec();
            layer.norm2_b = store
                .require_len(&format!("layer{li}.norm2_b"), layer.norm2_b.len())?
                .to_vec();
            layer.ff_w1 = store
                .require_len(&format!("layer{li}.ff_w1"), layer.ff_w1.len())?
                .to_vec();
            layer.ff_b1 = store
                .require_len(&format!("layer{li}.ff_b1"), layer.ff_b1.len())?
                .to_vec();
            layer.ff_w2 = store
                .require_len(&format!("layer{li}.ff_w2"), layer.ff_w2.len())?
                .to_vec();
            layer.ff_b2 = store
                .require_len(&format!("layer{li}.ff_b2"), layer.ff_b2.len())?
                .to_vec();
        }
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2024)
    }

    fn ramp(t: usize, c: usize) -> Vec<f32> {
        (0..t * c)
            .map(|i| ((i as f32) * 0.13).sin() + (i as f32) * 0.01)
            .collect()
    }

    #[test]
    fn moirai_num_patches_is_ceil() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        // patch_size = 4
        assert_eq!(m.num_context_patches(12), 3);
        assert_eq!(m.num_context_patches(13), 4); // ceil(13/4)
        assert_eq!(m.num_context_patches(1), 1);
        // patches must cover the whole input
        assert!(m.num_context_patches(13) * 4 >= 13);
    }

    #[test]
    fn moirai_single_variate_shape() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let t = 12;
        let c = 1;
        let h = 6;
        let series = ramp(t, c);
        let out = m.forward(&series, c, h).expect("forward");
        assert_eq!(out.point.len(), h * c);
        assert_eq!(out.scale.len(), h * c);
        assert_eq!(out.horizon, h);
        assert_eq!(out.n_variates, c);
    }

    #[test]
    fn moirai_three_variate_shape() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let t = 16;
        let c = 3;
        let h = 7;
        let series = ramp(t, c);
        let out = m.forward(&series, c, h).expect("forward");
        assert_eq!(out.point.len(), h * c);
        assert_eq!(out.scale.len(), h * c);
    }

    #[test]
    fn moirai_attention_rows_sum_to_one() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let t = 12;
        let c = 2;
        let h = 6;
        let series = ramp(t, c);
        let w = m.attention_weights(&series, c, h).expect("attn");
        let n = (w.len() as f64).sqrt() as usize;
        assert_eq!(n * n, w.len());
        for qi in 0..n {
            let s: f32 = w[qi * n..(qi + 1) * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row {qi} sums to {s}");
        }
    }

    #[test]
    fn moirai_context_attention_does_not_touch_targets() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let t = 12;
        let c = 2;
        let h = 6;
        let series = ramp(t, c);
        let (_tok, n_ctx_total, _n_tgt_total, _t) = m.build_tokens(&series, c, h).expect("tok");
        let w = m.attention_weights(&series, c, h).expect("attn");
        let n = n_ctx_total + c * m.num_target_patches(h);
        // Every context query places exactly zero mass on every target key.
        for qi in 0..n_ctx_total {
            for ki in n_ctx_total..n {
                assert_eq!(
                    w[qi * n + ki],
                    0.0,
                    "context query {qi} attends target {ki}"
                );
            }
        }
    }

    #[test]
    fn moirai_masked_positions_do_not_leak() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let t = 12;
        let c = 2;
        let h = 6;
        let series = ramp(t, c);
        let (mut tokens, n_ctx_total, n_tgt_total, _t) =
            m.build_tokens(&series, c, h).expect("tok");
        let d = m.cfg.d_model;
        let enc1 = m.encode(&tokens, n_ctx_total, n_tgt_total);
        // Arbitrarily corrupt the target (masked) tokens.
        for v in &mut tokens[n_ctx_total * d..] {
            *v += 99.0;
        }
        let enc2 = m.encode(&tokens, n_ctx_total, n_tgt_total);
        // The context representation must be unchanged.
        for i in 0..n_ctx_total * d {
            assert!(
                (enc1[i] - enc2[i]).abs() < 1e-5,
                "context leaked from target at {i}"
            );
        }
    }

    #[test]
    fn moirai_scale_is_positive() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let series = ramp(20, 2);
        let out = m.forward(&series, 2, 8).expect("forward");
        assert!(out.scale.iter().all(|&s| s > 0.0), "scale not positive");
    }

    #[test]
    fn moirai_output_is_finite() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let mut series = vec![0.0; 24 * 3];
        rng.fill_normal(&mut series);
        let out = m.forward(&series, 3, 5).expect("forward");
        assert!(out.point.iter().all(|v| v.is_finite()));
        assert!(out.scale.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn moirai_varying_input_changes_output() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let a = ramp(16, 2);
        let b: Vec<f32> = a.iter().map(|v| v * 2.0 + 0.5).collect();
        let oa = m.forward(&a, 2, 6).expect("a");
        let ob = m.forward(&b, 2, 6).expect("b");
        let diff: f32 = oa
            .point
            .iter()
            .zip(ob.point.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 1e-4, "output did not respond to input change");
    }

    #[test]
    fn moirai_deterministic_under_seed() {
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(7);
        let m1 = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut r1).expect("build");
        let m2 = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut r2).expect("build");
        let series = ramp(16, 2);
        let o1 = m1.forward(&series, 2, 6).expect("o1");
        let o2 = m2.forward(&series, 2, 6).expect("o2");
        assert_eq!(o1.point, o2.point);
        assert_eq!(o1.scale, o2.scale);
    }

    #[test]
    fn moirai_horizon_not_multiple_of_patch() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        // patch_size=4, horizon=5 -> 2 target patches, output truncated to 5.
        let series = ramp(10, 1);
        let out = m.forward(&series, 1, 5).expect("forward");
        assert_eq!(out.point.len(), 5);
        assert_eq!(m.num_target_patches(5), 2);
    }

    #[test]
    fn moirai_err_zero_variates() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        assert!(matches!(
            m.forward(&[1.0, 2.0], 0, 4).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn moirai_err_zero_horizon() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        assert!(matches!(
            m.forward(&[1.0, 2.0], 1, 0).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn moirai_err_ragged_series() {
        let mut rng = make_rng();
        let m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        // length 7 is not divisible by 2 variates.
        let series = vec![0.0; 7];
        assert!(matches!(
            m.forward(&series, 2, 4).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn moirai_err_bad_head_dim() {
        let mut rng = make_rng();
        let cfg = MoiraiConfig {
            d_model: 15,
            n_heads: 2,
            ..MoiraiConfig::tiny()
        };
        assert!(matches!(
            MoiraiForecaster::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    #[test]
    fn moirai_checkpoint_roundtrip_reproduces_forecast() {
        use crate::foundation::adapter::FoundationAdapter;
        let mut rng = make_rng();
        let src = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");

        // Export → serialise → load into a freshly-initialised model.
        let buf = src.to_checkpoint();
        let mut dst = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build dst");
        dst.load_checkpoint(&buf).expect("load");

        // After loading the same weights, the two models must forecast identically.
        let c = 2;
        let horizon = 8;
        let t = src.cfg.patch_size * 3;
        let series: Vec<f32> = (0..t * c).map(|i| (i as f32 * 0.07).sin()).collect();
        let f_src = src.forward(&series, c, horizon).expect("f_src");
        let f_dst = dst.forward(&series, c, horizon).expect("f_dst");
        for (a, b) in f_src.point.iter().zip(f_dst.point.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "checkpoint forecast mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn moirai_import_rejects_wrong_shape() {
        use crate::foundation::adapter::FoundationAdapter;
        let mut rng = make_rng();
        let mut m = MoiraiForecaster::new(MoiraiConfig::tiny(), &mut rng).expect("build");
        let mut store = m.export_weights();
        // Corrupt one tensor's length.
        store.insert("head_b", vec![0.0_f32; 1]);
        assert!(matches!(
            m.import_weights(&store).unwrap_err(),
            TsError::WeightShapeMismatch { .. }
        ));
    }
}
