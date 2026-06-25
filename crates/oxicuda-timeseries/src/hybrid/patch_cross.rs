//! PatchTST × Crossformer hybrid forecaster.
//!
//! This model fuses two complementary ideas:
//!
//! * **PatchTST patching** (Nie et al. 2023) — each variate's length-`T` series
//!   is split into overlapping patches of length `patch_len` with `stride`, and
//!   each patch is linearly embedded into a `d_model` token. This produces an
//!   `[n_vars, n_patches, d_model]` token tensor with channel independence.
//!
//! * **Crossformer Two-Stage Attention** (Zhang & Yan 2023) — the patch tokens
//!   are then mixed by two attention stages:
//!     1. *Cross-time*: for each variate independently, multi-head self-attention
//!        over its `n_patches` patch tokens (captures temporal dependencies),
//!     2. *Cross-dimension*: a small set of `n_routers` learnable router tokens
//!        aggregate information across the `n_vars` variates at each patch
//!        position (routers attend over variates, then variates attend back over
//!        routers), reducing cross-dimension cost from `O(n_vars²)` to
//!        `O(n_vars · n_routers)`.
//!
//! Unlike vanilla PatchTST (which is purely channel-independent and never mixes
//! variates), this hybrid adds explicit cross-dimension mixing via the
//! Crossformer router bottleneck while retaining patch tokenisation.
//!
//! Pure-Rust CPU reference. Token tensor layout is `[n_vars, n_patches, d_model]`
//! (d_model innermost). Input / output use time-major `[T, C]`.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for a [`PatchCrossformer`] hybrid model.
#[derive(Debug, Clone)]
pub struct PatchCrossConfig {
    /// Number of variates (channels).
    pub c: usize,
    /// Input sequence length.
    pub t: usize,
    /// Forecast horizon (steps).
    pub horizon: usize,
    /// Patch length.
    pub patch_len: usize,
    /// Stride between consecutive patches.
    pub stride: usize,
    /// Token embedding dimension (must divide by `n_heads`).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of fused (cross-time + cross-dimension) encoder layers.
    pub n_layers: usize,
    /// Number of router tokens for cross-dimension attention.
    pub n_routers: usize,
    /// FFN hidden expansion factor.
    pub ffn_expansion: usize,
}

impl PatchCrossConfig {
    /// Small configuration: `d=32, heads=4, layers=2, routers=3`.
    #[must_use]
    pub fn tiny(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            patch_len: 16,
            stride: 8,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            n_routers: 3,
            ffn_expansion: 4,
        }
    }

    /// Standard configuration: `d=64, heads=8, layers=3, routers=8`.
    #[must_use]
    pub fn base(c: usize, t: usize, horizon: usize) -> Self {
        Self {
            c,
            t,
            horizon,
            patch_len: 16,
            stride: 8,
            d_model: 64,
            n_heads: 8,
            n_layers: 3,
            n_routers: 8,
            ffn_expansion: 4,
        }
    }
}

// ─── Attention block weights ────────────────────────────────────────────────

/// Q/K/V/O projections plus pre-norm parameters for one attention sub-layer.
#[derive(Debug, Clone)]
struct AttnBlock {
    norm_g: Vec<f32>,
    norm_b: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    d: usize,
}

impl AttnBlock {
    fn new(d: usize, rng: &mut LcgRng) -> Self {
        Self {
            norm_g: vec![1.0_f32; d],
            norm_b: vec![0.0_f32; d],
            w_q: init_mat(d, d, rng),
            w_k: init_mat(d, d, rng),
            w_v: init_mat(d, d, rng),
            w_o: init_mat(d, d, rng),
            d,
        }
    }

    /// General cross-attention: queries `[nq, d]` attend over keys/values
    /// `[nk, d]`. Pre-norms the *query* stream. Returns output `[nq, d]`.
    fn cross_attend(
        &self,
        q_in: &[f32],
        kv_in: &[f32],
        nq: usize,
        nk: usize,
        n_heads: usize,
    ) -> Vec<f32> {
        let d = self.d;
        let head_dim = d / n_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let mut qn = q_in.to_vec();
        layer_norm(&mut qn, &self.norm_g, &self.norm_b);

        let q = matmul(&qn, &self.w_q, nq, d);
        let k = matmul(kv_in, &self.w_k, nk, d);
        let v = matmul(kv_in, &self.w_v, nk, d);

        let mut attn = vec![0.0_f32; nq * d];
        let mut scores = vec![0.0_f32; nk];
        for h in 0..n_heads {
            let hs = h * head_dim;
            for qi in 0..nq {
                for (ki, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0_f32;
                    for hd in 0..head_dim {
                        dot += q[qi * d + hs + hd] * k[ki * d + hs + hd];
                    }
                    *sc = dot * scale;
                }
                softmax_row(&mut scores);
                for hd in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (ki, &sc) in scores.iter().enumerate() {
                        acc += sc * v[ki * d + hs + hd];
                    }
                    attn[qi * d + hs + hd] = acc;
                }
            }
        }
        matmul(&attn, &self.w_o, nq, d)
    }

    /// Self-attention convenience wrapper (queries == keys == values).
    fn self_attend(&self, x: &[f32], n: usize, n_heads: usize) -> Vec<f32> {
        self.cross_attend(x, x, n, n, n_heads)
    }
}

// ─── Feed-forward weights ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Ffn {
    norm_g: Vec<f32>,
    norm_b: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d: usize,
    d_ff: usize,
}

impl Ffn {
    fn new(d: usize, expansion: usize, rng: &mut LcgRng) -> Self {
        let d_ff = d * expansion.max(1);
        Self {
            norm_g: vec![1.0_f32; d],
            norm_b: vec![0.0_f32; d],
            w1: init_mat(d_ff, d, rng),
            b1: vec![0.0_f32; d_ff],
            w2: init_mat(d, d_ff, rng),
            b2: vec![0.0_f32; d],
            d,
            d_ff,
        }
    }

    fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut normed = x.to_vec();
        layer_norm(&mut normed, &self.norm_g, &self.norm_b);
        let mut hidden = vec![0.0_f32; n * self.d_ff];
        for i in 0..n {
            for fi in 0..self.d_ff {
                let mut acc = self.b1[fi];
                let row = &self.w1[fi * self.d..(fi + 1) * self.d];
                for k in 0..self.d {
                    acc += normed[i * self.d + k] * row[k];
                }
                hidden[i * self.d_ff + fi] = gelu(acc);
            }
        }
        let mut out = vec![0.0_f32; n * self.d];
        for i in 0..n {
            for di in 0..self.d {
                let mut acc = self.b2[di];
                let row = &self.w2[di * self.d_ff..(di + 1) * self.d_ff];
                for fi in 0..self.d_ff {
                    acc += hidden[i * self.d_ff + fi] * row[fi];
                }
                out[i * self.d + di] = acc;
            }
        }
        out
    }
}

// ─── One fused encoder layer ────────────────────────────────────────────────

/// One hybrid layer: cross-time self-attention (per variate) + router-based
/// cross-dimension attention (per patch position) + position-wise FFN.
#[derive(Debug, Clone)]
struct HybridLayer {
    time_attn: AttnBlock,
    time_ffn: Ffn,
    /// Routers attend over variates (router queries, variate keys/values).
    router_in: AttnBlock,
    /// Variates attend back over routers (variate queries, router keys/values).
    router_out: AttnBlock,
    dim_ffn: Ffn,
    /// Learnable router tokens `[n_routers, d_model]`.
    routers: Vec<f32>,
    n_routers: usize,
    d: usize,
}

impl HybridLayer {
    fn new(d: usize, n_routers: usize, expansion: usize, rng: &mut LcgRng) -> Self {
        let mut routers = vec![0.0_f32; n_routers * d];
        rng.fill_normal(&mut routers);
        let rscale = (1.0_f32 / d as f32).sqrt();
        for r in &mut routers {
            *r *= rscale;
        }
        Self {
            time_attn: AttnBlock::new(d, rng),
            time_ffn: Ffn::new(d, expansion, rng),
            router_in: AttnBlock::new(d, rng),
            router_out: AttnBlock::new(d, rng),
            dim_ffn: Ffn::new(d, expansion, rng),
            routers,
            n_routers,
            d,
        }
    }

    /// Run the fused layer over a `[n_vars, n_patches, d]` token tensor in place.
    fn forward(&self, tokens: &mut [f32], n_vars: usize, n_patches: usize, n_heads: usize) {
        let d = self.d;

        // ── Stage 1: cross-time self-attention, per variate independently ──
        for vi in 0..n_vars {
            let base = vi * n_patches * d;
            let seq = &tokens[base..base + n_patches * d];
            let a = self.time_attn.self_attend(seq, n_patches, n_heads);
            for i in 0..n_patches * d {
                tokens[base + i] += a[i];
            }
            // FFN over this variate's patch tokens.
            let seq2 = &tokens[base..base + n_patches * d];
            let f = self.time_ffn.forward(seq2, n_patches);
            for i in 0..n_patches * d {
                tokens[base + i] += f[i];
            }
        }

        // ── Stage 2: router cross-dimension attention, per patch position ──
        // For each patch index p, gather the variate tokens [n_vars, d], run the
        // two-step router bottleneck, and scatter the result back.
        let mut variate_buf = vec![0.0_f32; n_vars * d];
        for p in 0..n_patches {
            // Gather variate tokens at patch position p.
            for vi in 0..n_vars {
                let src = (vi * n_patches + p) * d;
                variate_buf[vi * d..(vi + 1) * d].copy_from_slice(&tokens[src..src + d]);
            }
            // Step A: routers (queries) attend over variates (keys/values).
            let agg = self.router_in.cross_attend(
                &self.routers,
                &variate_buf,
                self.n_routers,
                n_vars,
                n_heads,
            );
            // Step B: variates (queries) attend back over aggregated routers.
            let mixed =
                self.router_out
                    .cross_attend(&variate_buf, &agg, n_vars, self.n_routers, n_heads);
            // Residual add + scatter back.
            for vi in 0..n_vars {
                let dst = (vi * n_patches + p) * d;
                for di in 0..d {
                    tokens[dst + di] += mixed[vi * d + di];
                }
            }
        }

        // ── Position-wise FFN over all tokens ──
        let n = n_vars * n_patches;
        let f = self.dim_ffn.forward(tokens, n);
        for i in 0..tokens.len() {
            tokens[i] += f[i];
        }
    }
}

// ─── Hybrid model ───────────────────────────────────────────────────────────

/// PatchTST × Crossformer hybrid forecaster.
#[derive(Debug, Clone)]
pub struct PatchCrossformer {
    /// Patch embedding weight `[d_model, patch_len]`.
    patch_w: Vec<f32>,
    /// Patch embedding bias `[d_model]`.
    patch_b: Vec<f32>,
    /// Sinusoidal positional encoding `[n_patches, d_model]`.
    pos_enc: Vec<f32>,
    layers: Vec<HybridLayer>,
    /// Per-variate forecast head `[horizon, n_patches * d_model]`.
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    n_patches: usize,
    config: PatchCrossConfig,
}

impl PatchCrossformer {
    /// Number of patches for a sequence of length `t`.
    #[must_use]
    fn compute_n_patches(t: usize, patch_len: usize, stride: usize) -> usize {
        if t < patch_len || stride == 0 {
            return 0;
        }
        (t - patch_len) / stride + 1
    }

    /// Build a hybrid forecaster from config, initialising all weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidNumVariates`] when `c == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::InvalidPatchLen`] when `patch_len == 0`.
    /// - [`TsError::InvalidStride`] when `stride == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    /// - [`TsError::InvalidSequenceLength`] when `t < patch_len`.
    /// - [`TsError::InvalidTopK`] when `n_routers == 0`.
    pub fn new(config: PatchCrossConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if config.c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if config.horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if config.patch_len == 0 {
            return Err(TsError::InvalidPatchLen(0));
        }
        if config.stride == 0 {
            return Err(TsError::InvalidStride(0));
        }
        if config.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if config.d_model % config.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: config.d_model,
                n_heads: config.n_heads,
            });
        }
        if config.t < config.patch_len {
            return Err(TsError::InvalidSequenceLength(config.t));
        }
        if config.n_routers == 0 {
            return Err(TsError::InvalidTopK(0));
        }

        let d = config.d_model;
        let n_patches = Self::compute_n_patches(config.t, config.patch_len, config.stride);
        if n_patches == 0 {
            return Err(TsError::InvalidSequenceLength(config.t));
        }

        let patch_w = init_mat(d, config.patch_len, rng);
        let patch_b = vec![0.0_f32; d];
        let pos_enc = sinusoidal_pos_enc(n_patches, d);

        let layers = (0..config.n_layers)
            .map(|_| HybridLayer::new(d, config.n_routers, config.ffn_expansion, rng))
            .collect();

        let flat = n_patches * d;
        let head_w = init_mat(config.horizon, flat, rng);
        let head_b = vec![0.0_f32; config.horizon];

        Ok(Self {
            patch_w,
            patch_b,
            pos_enc,
            layers,
            head_w,
            head_b,
            n_patches,
            config,
        })
    }

    /// Number of patches this model uses.
    #[must_use]
    pub fn num_patches(&self) -> usize {
        self.n_patches
    }

    /// Forecast a `[T, C]` series → `[horizon, C]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * c`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let cfg = &self.config;
        let expected = cfg.t * cfg.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let d = cfg.d_model;
        let np = self.n_patches;

        // Build [n_vars, n_patches, d] token tensor via patch embedding + PE.
        let mut tokens = vec![0.0_f32; cfg.c * np * d];
        for vi in 0..cfg.c {
            // Extract the variate series.
            let series: Vec<f32> = (0..cfg.t).map(|ti| x[ti * cfg.c + vi]).collect();
            for p in 0..np {
                let start = p * cfg.stride;
                // Linear patch embed: d_model = patch_w · patch + bias.
                for di in 0..d {
                    let mut acc = self.patch_b[di];
                    let row = &self.patch_w[di * cfg.patch_len..(di + 1) * cfg.patch_len];
                    for (pl, &wv) in row.iter().enumerate() {
                        acc += series[start + pl] * wv;
                    }
                    let idx = (vi * np + p) * d + di;
                    tokens[idx] = acc + self.pos_enc[p * d + di];
                }
            }
        }

        // Run the fused encoder layers.
        for layer in &self.layers {
            layer.forward(&mut tokens, cfg.c, np, cfg.n_heads);
        }

        // Per-variate flatten → linear head → [horizon, C].
        let flat = np * d;
        let mut forecast = vec![0.0_f32; cfg.horizon * cfg.c];
        for vi in 0..cfg.c {
            let base = vi * np * d;
            let feat = &tokens[base..base + flat];
            for hi in 0..cfg.horizon {
                let w = &self.head_w[hi * flat..(hi + 1) * flat];
                let mut acc = self.head_b[hi];
                for (k, &wv) in w.iter().enumerate() {
                    acc += wv * feat[k];
                }
                forecast[hi * cfg.c + vi] = acc;
            }
        }
        Ok(forecast)
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &PatchCrossConfig {
        &self.config
    }
}

// ─── Shared math helpers ────────────────────────────────────────────────────

fn init_mat(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (6.0_f32 / (rows + cols).max(1) as f32).sqrt();
    let mut v = vec![0.0_f32; rows * cols];
    rng.fill_normal(&mut v);
    for x in &mut v {
        *x *= scale;
    }
    v
}

/// `y = x · W^T` for `x: [n, d]`, `W: [d, d]` row-major → `[n, d]`.
fn matmul(x: &[f32], w: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        for di in 0..d {
            let row = &w[di * d..(di + 1) * d];
            let mut acc = 0.0_f32;
            for k in 0..d {
                acc += x[i * d + k] * row[k];
            }
            out[i * d + di] = acc;
        }
    }
    out
}

fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let d = gamma.len();
    if d == 0 {
        return;
    }
    let n = x.len() / d;
    for i in 0..n {
        let row = &mut x[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = (var + 1e-5_f32).sqrt().recip();
        for (j, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[inline]
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    let inner = c * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn sinusoidal_pos_enc(np: usize, d: usize) -> Vec<f32> {
    let mut pe = vec![0.0_f32; np * d];
    for p in 0..np {
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn patchcross_tiny_output_shape() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig::tiny(3, 96, 24);
        let model = PatchCrossformer::new(cfg.clone(), &mut rng).expect("build");
        let x: Vec<f32> = (0..cfg.t * cfg.c)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
    }

    #[test]
    fn patchcross_base_output_shape() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig::base(4, 96, 24);
        let model = PatchCrossformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.2_f32; cfg.t * cfg.c];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.horizon * cfg.c);
    }

    #[test]
    fn patchcross_num_patches() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig::tiny(2, 96, 24);
        let model = PatchCrossformer::new(cfg, &mut rng).expect("build");
        // (96 - 16) / 8 + 1 = 11
        assert_eq!(model.num_patches(), 11);
    }

    #[test]
    fn patchcross_output_finite() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig::tiny(3, 96, 12);
        let model = PatchCrossformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.t * cfg.c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn patchcross_deterministic_under_seed() {
        let cfg = PatchCrossConfig::tiny(2, 64, 8);
        let mut rng_a = LcgRng::new(11);
        let mut rng_b = LcgRng::new(11);
        let a = PatchCrossformer::new(cfg.clone(), &mut rng_a).expect("a");
        let b = PatchCrossformer::new(cfg, &mut rng_b).expect("b");
        let x: Vec<f32> = (0..a.config().t * a.config().c)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let oa = a.forward(&x).expect("fa");
        let ob = b.forward(&x).expect("fb");
        for (p, q) in oa.iter().zip(ob.iter()) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn patchcross_cross_dimension_mixing_active() {
        // The defining property vs vanilla PatchTST: changing ONE variate's input
        // must influence the forecast of OTHER variates (cross-dimension mixing).
        let mut rng = make_rng();
        let cfg = PatchCrossConfig::tiny(3, 64, 8);
        let model = PatchCrossformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x1 = vec![0.1_f32; cfg.t * cfg.c];
        rng.fill_normal(&mut x1);
        let mut x2 = x1.clone();
        // Perturb only variate 0 across all time steps.
        for ti in 0..cfg.t {
            x2[ti * cfg.c] += 3.0;
        }
        let o1 = model.forward(&x1).expect("o1");
        let o2 = model.forward(&x2).expect("o2");
        // Measure change in the forecasts of variate 1 and variate 2 (NOT 0).
        let mut other_diff = 0.0_f32;
        for hi in 0..cfg.horizon {
            for ci in 1..cfg.c {
                other_diff += (o1[hi * cfg.c + ci] - o2[hi * cfg.c + ci]).abs();
            }
        }
        assert!(
            other_diff > 1e-4,
            "cross-dimension mixing inactive: other-variate forecast unchanged ({other_diff})"
        );
    }

    #[test]
    fn patchcross_err_zero_variates() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig {
            c: 0,
            ..PatchCrossConfig::tiny(1, 64, 8)
        };
        assert!(matches!(
            PatchCrossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    #[test]
    fn patchcross_err_head_dim() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig {
            d_model: 30,
            n_heads: 4,
            ..PatchCrossConfig::tiny(2, 64, 8)
        };
        assert!(matches!(
            PatchCrossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    #[test]
    fn patchcross_err_seq_too_short() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig {
            t: 10,
            patch_len: 16,
            ..PatchCrossConfig::tiny(2, 64, 8)
        };
        assert!(matches!(
            PatchCrossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(10)
        ));
    }

    #[test]
    fn patchcross_err_zero_routers() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig {
            n_routers: 0,
            ..PatchCrossConfig::tiny(2, 64, 8)
        };
        assert!(matches!(
            PatchCrossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidTopK(0)
        ));
    }

    #[test]
    fn patchcross_err_bad_input_len() {
        let mut rng = make_rng();
        let cfg = PatchCrossConfig::tiny(2, 64, 8);
        let model = PatchCrossformer::new(cfg, &mut rng).expect("build");
        let x = vec![0.0_f32; 50];
        assert!(matches!(
            model.forward(&x).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }
}
