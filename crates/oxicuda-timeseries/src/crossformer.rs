//! Crossformer: Transformer utilizing cross-dimension dependency (Zhang & Yan 2023 ICLR).
//!
//! Reference: "Crossformer: Transformer Utilizing Cross-Dimension Dependency
//! for Multivariate Time Series Forecasting", Zhang & Yan, ICLR 2023.
//!
//! Crossformer introduces two ideas for multivariate forecasting:
//!
//! 1. **Dimension-Segment-Wise (DSW) embedding** — each variate's length-`seq_len`
//!    series is split into `n_segs = seq_len / seg_len` contiguous segments and
//!    each segment is linearly embedded into a `d_model` token. This yields an
//!    `[n_vars, n_segs, d_model]` tensor of segment tokens.
//!
//! 2. **Two-Stage Attention (TSA)** — captures both cross-time and cross-dimension
//!    dependencies:
//!    * *Stage 1 (cross-time):* for each variate independently, multi-head
//!      self-attention over its `n_segs` time-segment tokens.
//!    * *Stage 2 (cross-dimension):* a small set of `n_routers` learnable router
//!      tokens first attends over the `n_vars` variate-tokens (routers as queries,
//!      variates as keys/values) to aggregate cross-dimension information, then the
//!      variates attend back over the aggregated routers (variates as queries,
//!      routers as keys/values). This router bottleneck reduces the cost of
//!      cross-dimension mixing from `O(n_vars²)` to `O(n_vars · n_routers)`.
//!
//! This is a pure-Rust CPU reference. All tensors are row-major; the segment-token
//! tensor uses `[n_vars, n_segs, d_model]` layout (d_model innermost).

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for a Crossformer encoder.
#[derive(Debug, Clone)]
pub struct CrossformerConfig {
    /// Input sequence length (time axis, per variate).
    pub seq_len: usize,
    /// Number of variates (channels).
    pub n_vars: usize,
    /// Segment length for DSW embedding (must divide `seq_len`).
    pub seg_len: usize,
    /// Token embedding dimension.
    pub d_model: usize,
    /// Number of attention heads (must divide `d_model`).
    pub n_heads: usize,
    /// Number of router tokens for cross-dimension attention.
    pub n_routers: usize,
}

impl CrossformerConfig {
    /// Small configuration: `d_model = 16`, `n_heads = 2`, `n_routers = 3`.
    #[must_use]
    pub fn tiny(seq_len: usize, n_vars: usize, seg_len: usize) -> Self {
        Self {
            seq_len,
            n_vars,
            seg_len,
            d_model: 16,
            n_heads: 2,
            n_routers: 3,
        }
    }
}

// ─── Linear weight helper ─────────────────────────────────────────────────────

/// A learnable affine map `y = x · W^T + b` with `W` row-major `[out, in]`.
#[derive(Debug, Clone)]
struct Linear {
    /// Weight matrix `[out_dim, in_dim]` row-major.
    weight: Vec<f32>,
    /// Bias `[out_dim]`.
    bias: Vec<f32>,
    /// Input dimension.
    in_dim: usize,
    /// Output dimension.
    out_dim: usize,
}

impl Linear {
    /// Glorot-style initialised linear layer.
    fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let mut weight = vec![0.0_f32; out_dim * in_dim];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        Self {
            weight,
            bias: vec![0.0_f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Apply to a single `in_dim` vector → `out_dim` vector.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.out_dim];
        for (oi, o) in out.iter_mut().enumerate() {
            let row = &self.weight[oi * self.in_dim..(oi + 1) * self.in_dim];
            let mut acc = self.bias[oi];
            for (&xv, &wv) in x.iter().zip(row.iter()) {
                acc += xv * wv;
            }
            *o = acc;
        }
        out
    }
}

// ─── Multi-head attention weights ─────────────────────────────────────────────

/// Query/Key/Value/Output projections for a multi-head attention block.
#[derive(Debug, Clone)]
struct AttnWeights {
    /// Query projection `[d_model, d_model]`.
    w_q: Vec<f32>,
    /// Key projection `[d_model, d_model]`.
    w_k: Vec<f32>,
    /// Value projection `[d_model, d_model]`.
    w_v: Vec<f32>,
    /// Output projection `[d_model, d_model]`.
    w_o: Vec<f32>,
    /// Model dimension.
    d_model: usize,
}

impl AttnWeights {
    fn new(d_model: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / d_model as f32).sqrt();
        let mut mat = || -> Vec<f32> {
            let mut v = vec![0.0_f32; d_model * d_model];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };
        Self {
            w_q: mat(),
            w_k: mat(),
            w_v: mat(),
            w_o: mat(),
            d_model,
        }
    }
}

// ─── Crossformer model ─────────────────────────────────────────────────────────

/// Crossformer encoder.
///
/// Produces `[n_vars, n_segs, d_model]` segment-token features from a
/// `[n_vars, seq_len]` multivariate input via DSW embedding followed by
/// Two-Stage Attention.
#[derive(Debug, Clone)]
pub struct Crossformer {
    /// DSW segment embedding: `seg_len → d_model`.
    dsw: Linear,
    /// Cross-time attention weights (stage 1).
    time_attn: AttnWeights,
    /// Router → variate attention weights (stage 2, first pass).
    router_attn: AttnWeights,
    /// Variate → router attention weights (stage 2, second pass).
    dim_attn: AttnWeights,
    /// Learnable router tokens `[n_routers, d_model]` (row-major).
    routers: Vec<f32>,
    /// Output projection `[d_model, d_model]` applied per token.
    out_proj: Linear,
    /// Model configuration.
    cfg: CrossformerConfig,
}

impl Crossformer {
    /// Build a Crossformer encoder, initialising all weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`.
    /// - [`TsError::InvalidNumVariates`] when `n_vars == 0`.
    /// - [`TsError::InvalidPatchLen`] when `seg_len == 0` or `seq_len % seg_len != 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    /// - [`TsError::InvalidPoolSize`] when `n_routers == 0`.
    pub fn new(cfg: CrossformerConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.n_vars == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }
        if cfg.seg_len == 0 {
            return Err(TsError::InvalidPatchLen(0));
        }
        if cfg.seq_len % cfg.seg_len != 0 {
            return Err(TsError::InvalidPatchLen(cfg.seg_len));
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
        if cfg.n_routers == 0 {
            return Err(TsError::InvalidPoolSize(0));
        }

        let dsw = Linear::new(cfg.seg_len, cfg.d_model, rng);
        let time_attn = AttnWeights::new(cfg.d_model, rng);
        let router_attn = AttnWeights::new(cfg.d_model, rng);
        let dim_attn = AttnWeights::new(cfg.d_model, rng);

        // Learnable router tokens, small Gaussian init.
        let router_scale = (1.0_f32 / cfg.d_model as f32).sqrt();
        let mut routers = vec![0.0_f32; cfg.n_routers * cfg.d_model];
        rng.fill_normal(&mut routers);
        for r in &mut routers {
            *r *= router_scale;
        }

        let out_proj = Linear::new(cfg.d_model, cfg.d_model, rng);

        Ok(Self {
            dsw,
            time_attn,
            router_attn,
            dim_attn,
            routers,
            out_proj,
            cfg,
        })
    }

    /// Number of time segments per variate: `seq_len / seg_len`.
    #[must_use]
    #[inline]
    pub fn n_segs(&self) -> usize {
        self.cfg.seq_len / self.cfg.seg_len
    }

    /// Access the model configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &CrossformerConfig {
        &self.cfg
    }

    /// Dimension-Segment-Wise embedding.
    ///
    /// Splits each variate's length-`seq_len` series into `n_segs` contiguous
    /// segments of `seg_len` and linearly embeds each segment to `d_model`.
    ///
    /// * Input  `x`   — `[n_vars, seq_len]` row-major.
    /// * Output       — `[n_vars, n_segs, d_model]` row-major.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != n_vars * seq_len`.
    pub fn dsw_embed(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let n_vars = self.cfg.n_vars;
        let seq_len = self.cfg.seq_len;
        let seg_len = self.cfg.seg_len;
        let d = self.cfg.d_model;
        let expected = n_vars * seq_len;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let n_segs = self.n_segs();
        let mut out = vec![0.0_f32; n_vars * n_segs * d];
        for vi in 0..n_vars {
            let var_base = vi * seq_len;
            for si in 0..n_segs {
                let seg = &x[var_base + si * seg_len..var_base + (si + 1) * seg_len];
                let embedded = self.dsw.apply(seg);
                let dst = (vi * n_segs + si) * d;
                out[dst..dst + d].copy_from_slice(&embedded);
            }
        }
        Ok(out)
    }

    /// Two-Stage Attention over `[n_vars, n_segs, d_model]` segment tokens.
    ///
    /// * Stage 1 (cross-time): per-variate MHSA over the `n_segs` axis.
    /// * Stage 2 (cross-dimension): router-based attention. For each segment
    ///   position, routers attend over variates, then variates attend over the
    ///   aggregated routers — `O(n_vars · n_routers)` cross-dimension mixing.
    ///
    /// The output shape equals the input shape `[n_vars, n_segs, d_model]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `embedded.len() != n_vars * n_segs * d_model`.
    pub fn two_stage_attention(&self, embedded: &[f32]) -> TsResult<Vec<f32>> {
        let n_vars = self.cfg.n_vars;
        let d = self.cfg.d_model;
        let n_heads = self.cfg.n_heads;
        let n_segs = self.n_segs();
        let expected = n_vars * n_segs * d;
        if embedded.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: embedded.len(),
            });
        }

        // ── Stage 1: cross-time MHSA, per variate over the n_segs tokens. ──
        let mut stage1 = embedded.to_vec();
        for vi in 0..n_vars {
            let base = vi * n_segs * d;
            let tokens = &embedded[base..base + n_segs * d];
            let attn =
                multi_head_attention(tokens, n_segs, tokens, n_segs, &self.time_attn, n_heads);
            // Residual connection (token-wise add).
            for i in 0..n_segs * d {
                stage1[base + i] += attn[i];
            }
        }

        // ── Stage 2: router-based cross-dimension attention, per segment. ──
        // For each segment position si, gather the n_vars variate-tokens, then:
        //   (a) routers (Q) attend over variates (K/V)   → aggregated routers,
        //   (b) variates (Q) attend over aggregated routers (K/V) → mixed vars.
        let n_routers = self.cfg.n_routers;
        let mut output = stage1.clone();
        for si in 0..n_segs {
            // Gather variate-tokens at this segment position: [n_vars, d].
            let mut var_tokens = vec![0.0_f32; n_vars * d];
            for vi in 0..n_vars {
                let src = (vi * n_segs + si) * d;
                var_tokens[vi * d..(vi + 1) * d].copy_from_slice(&stage1[src..src + d]);
            }

            // (a) routers attend over variates → aggregated routers [n_routers, d].
            let agg_routers = multi_head_attention(
                &self.routers,
                n_routers,
                &var_tokens,
                n_vars,
                &self.router_attn,
                n_heads,
            );

            // (b) variates attend over aggregated routers → mixed variates [n_vars, d].
            let mixed = multi_head_attention(
                &var_tokens,
                n_vars,
                &agg_routers,
                n_routers,
                &self.dim_attn,
                n_heads,
            );

            // Residual add back into the per-variate output at this segment.
            for vi in 0..n_vars {
                let dst = (vi * n_segs + si) * d;
                for di in 0..d {
                    output[dst + di] = stage1[dst + di] + mixed[vi * d + di];
                }
            }
        }

        Ok(output)
    }

    /// Full forward pass: `[n_vars, seq_len]` → `[n_vars, n_segs, d_model]`.
    ///
    /// Applies DSW embedding, Two-Stage Attention, then a per-token output
    /// projection.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != n_vars * seq_len`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let embedded = self.dsw_embed(x)?;
        let attended = self.two_stage_attention(&embedded)?;

        // Per-token output projection.
        let n_tokens = attended.len() / d;
        let mut out = vec![0.0_f32; attended.len()];
        for ti in 0..n_tokens {
            let token = &attended[ti * d..(ti + 1) * d];
            let projected = self.out_proj.apply(token);
            out[ti * d..(ti + 1) * d].copy_from_slice(&projected);
        }
        Ok(out)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Multi-head scaled dot-product attention.
///
/// * `q_in` — `[n_q, d_model]` query tokens.
/// * `k_in` — `[n_kv, d_model]` key/value tokens (keys and values share input).
/// * Returns `[n_q, d_model]` attended output, projected through `W_o`.
///
/// Q/K/V projections use `w_q/w_k/w_v` (each `[d_model, d_model]`); the per-head
/// scale is `1/√head_dim` with `head_dim = d_model / n_heads`.
fn multi_head_attention(
    q_in: &[f32],
    n_q: usize,
    k_in: &[f32],
    n_kv: usize,
    w: &AttnWeights,
    n_heads: usize,
) -> Vec<f32> {
    let d = w.d_model;
    let head_dim = d / n_heads;
    let scale = (head_dim as f32).sqrt().recip();

    let q = project(q_in, n_q, &w.w_q, d);
    let k = project(k_in, n_kv, &w.w_k, d);
    let v = project(k_in, n_kv, &w.w_v, d);

    let mut concat = vec![0.0_f32; n_q * d];

    for h in 0..n_heads {
        let h_off = h * head_dim;
        let mut scores = vec![0.0_f32; n_kv];
        for qi in 0..n_q {
            // Scores for this query over all keys.
            for ki in 0..n_kv {
                let mut dot = 0.0_f32;
                for hd in 0..head_dim {
                    dot += q[qi * d + h_off + hd] * k[ki * d + h_off + hd];
                }
                scores[ki] = dot * scale;
            }
            softmax_row(&mut scores);
            // Weighted sum of values.
            for hd in 0..head_dim {
                let mut acc = 0.0_f32;
                for ki in 0..n_kv {
                    acc += scores[ki] * v[ki * d + h_off + hd];
                }
                concat[qi * d + h_off + hd] = acc;
            }
        }
    }

    // Output projection.
    project(&concat, n_q, &w.w_o, d)
}

/// Linear projection `[n, d] · W^T → [n, d]` with `W` row-major `[d, d]`.
fn project(x: &[f32], n: usize, w: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * d];
    for ti in 0..n {
        for oi in 0..d {
            let row = oi * d;
            let mut acc = 0.0_f32;
            for ki in 0..d {
                acc += x[ti * d + ki] * w[row + ki];
            }
            out[ti * d + oi] = acc;
        }
    }
    out
}

/// Numerically stable in-place softmax over a row.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = if sum == 0.0 { 1.0 } else { sum.recip() };
    for v in row.iter_mut() {
        *v *= inv_sum;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2023)
    }

    fn tiny(seq_len: usize, n_vars: usize, seg_len: usize) -> CrossformerConfig {
        CrossformerConfig {
            seq_len,
            n_vars,
            seg_len,
            d_model: 8,
            n_heads: 2,
            n_routers: 3,
        }
    }

    // 1. n_segs == seq_len / seg_len.
    #[test]
    fn n_segs_correct() {
        let mut rng = make_rng();
        let model = Crossformer::new(tiny(24, 3, 4), &mut rng).expect("build");
        assert_eq!(model.n_segs(), 6);
    }

    // 2. dsw_embed output length == n_vars * n_segs * d_model.
    #[test]
    fn dsw_embed_shape() {
        let mut rng = make_rng();
        let cfg = tiny(24, 3, 4);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.3_f32; cfg.n_vars * cfg.seq_len];
        let emb = model.dsw_embed(&x).expect("embed");
        assert_eq!(emb.len(), cfg.n_vars * model.n_segs() * cfg.d_model);
    }

    // 3. two_stage_attention preserves shape.
    #[test]
    fn tsa_preserves_shape() {
        let mut rng = make_rng();
        let cfg = tiny(24, 3, 4);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let n = cfg.n_vars * model.n_segs() * cfg.d_model;
        let embedded = vec![0.1_f32; n];
        let out = model.two_stage_attention(&embedded).expect("tsa");
        assert_eq!(out.len(), n);
    }

    // 4. forward output length == n_vars * n_segs * d_model.
    #[test]
    fn forward_shape() {
        let mut rng = make_rng();
        let cfg = tiny(30, 4, 5);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.2_f32; cfg.n_vars * cfg.seq_len];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.n_vars * model.n_segs() * cfg.d_model);
    }

    // 5. Deterministic given the same seed.
    #[test]
    fn deterministic_given_seed() {
        let cfg = tiny(24, 3, 4);
        let mut rng_a = LcgRng::new(55);
        let mut rng_b = LcgRng::new(55);
        let model_a = Crossformer::new(cfg.clone(), &mut rng_a).expect("build");
        let model_b = Crossformer::new(cfg.clone(), &mut rng_b).expect("build");
        let x: Vec<f32> = (0..cfg.n_vars * cfg.seq_len)
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let out_a = model_a.forward(&x).expect("forward");
        let out_b = model_b.forward(&x).expect("forward");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-7, "non-deterministic: {a} vs {b}");
        }
    }

    // 6. Output is finite.
    #[test]
    fn forward_finite() {
        let mut rng = make_rng();
        let cfg = tiny(24, 4, 4);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.n_vars * cfg.seq_len];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    // 7. seq_len % seg_len != 0 → Err.
    #[test]
    fn err_seg_len_not_divisible() {
        let mut rng = make_rng();
        let cfg = CrossformerConfig {
            seq_len: 25,
            seg_len: 4,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidPatchLen(4)
        ));
    }

    // 8. d_model % n_heads != 0 → Err.
    #[test]
    fn err_d_model_not_divisible() {
        let mut rng = make_rng();
        let cfg = CrossformerConfig {
            d_model: 9,
            n_heads: 2,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    // 9. Single variate (n_vars = 1) works.
    #[test]
    fn single_variate() {
        let mut rng = make_rng();
        let cfg = tiny(20, 1, 5);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.5_f32; cfg.n_vars * cfg.seq_len];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), model.n_segs() * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 10. Single segment (n_segs = 1) works.
    #[test]
    fn single_segment() {
        let mut rng = make_rng();
        let cfg = tiny(8, 3, 8); // seq_len == seg_len → 1 segment.
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        assert_eq!(model.n_segs(), 1);
        let x = vec![0.4_f32; cfg.n_vars * cfg.seq_len];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.n_vars * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 11. Single router works.
    #[test]
    fn single_router() {
        let mut rng = make_rng();
        let cfg = CrossformerConfig {
            n_routers: 1,
            ..tiny(24, 4, 4)
        };
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.3_f32; cfg.n_vars * cfg.seq_len];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.n_vars * model.n_segs() * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 12. Changing one variate's input changes the cross-dimension-mixed output
    //     of OTHER variates (router mixing is wired through).
    #[test]
    fn router_mixing_affects_other_variates() {
        let mut rng = make_rng();
        let cfg = tiny(16, 3, 4);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");

        let mut x = vec![0.1_f32; cfg.n_vars * cfg.seq_len];
        // Vary entries deterministically so attention is not degenerate.
        for (i, xv) in x.iter_mut().enumerate() {
            *xv = (i as f32 * 0.05).sin();
        }
        let out_base = model.forward(&x).expect("forward");

        // Perturb only variate 0's input (variate 0 occupies [0, seq_len)).
        let mut x2 = x.clone();
        for xv in x2.iter_mut().take(cfg.seq_len) {
            *xv += 1.5;
        }
        let out_pert = model.forward(&x2).expect("forward");

        // Inspect variate 2's output tokens; they must change because the
        // router stage mixes information across all variates.
        let n_segs = model.n_segs();
        let d = cfg.d_model;
        let other = 2usize;
        let base = other * n_segs * d;
        let mut max_diff = 0.0_f32;
        for i in 0..n_segs * d {
            max_diff = max_diff.max((out_base[base + i] - out_pert[base + i]).abs());
        }
        assert!(
            max_diff > 1e-5,
            "perturbing variate 0 did not affect variate 2 (max_diff={max_diff})"
        );
    }

    // 13. err: seq_len == 0.
    #[test]
    fn err_seq_len_zero() {
        let mut rng = make_rng();
        let cfg = CrossformerConfig {
            seq_len: 0,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    // 14. err: n_vars == 0.
    #[test]
    fn err_n_vars_zero() {
        let mut rng = make_rng();
        let cfg = CrossformerConfig {
            n_vars: 0,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumVariates(0)
        ));
    }

    // 15. err: seg_len == 0, n_heads == 0, n_routers == 0.
    #[test]
    fn err_zero_params() {
        let mut rng = make_rng();
        let cfg_seg = CrossformerConfig {
            seg_len: 0,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg_seg, &mut rng).unwrap_err(),
            TsError::InvalidPatchLen(0)
        ));
        let cfg_heads = CrossformerConfig {
            n_heads: 0,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg_heads, &mut rng).unwrap_err(),
            TsError::InvalidNumHeads(0)
        ));
        let cfg_routers = CrossformerConfig {
            n_routers: 0,
            ..tiny(24, 3, 4)
        };
        assert!(matches!(
            Crossformer::new(cfg_routers, &mut rng).unwrap_err(),
            TsError::InvalidPoolSize(0)
        ));
    }

    // 16. err: x wrong length (dsw_embed and forward).
    #[test]
    fn err_wrong_input_length() {
        let mut rng = make_rng();
        let cfg = tiny(24, 3, 4);
        let model = Crossformer::new(cfg, &mut rng).expect("build");
        let bad = vec![0.0_f32; 17];
        assert!(matches!(
            model.dsw_embed(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            model.forward(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
        // two_stage_attention validates its own input length too.
        let bad_emb = vec![0.0_f32; 5];
        assert!(matches!(
            model.two_stage_attention(&bad_emb).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 17. n_routers larger than n_vars works.
    #[test]
    fn routers_more_than_variates() {
        let mut rng = make_rng();
        let cfg = CrossformerConfig {
            n_routers: 8,
            ..tiny(16, 2, 4)
        };
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.25_f32; cfg.n_vars * cfg.seq_len];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.n_vars * model.n_segs() * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 18. dsw_embed embeds each segment independently (distinct segments → distinct tokens).
    #[test]
    fn dsw_embed_segment_independence() {
        let mut rng = make_rng();
        let cfg = tiny(16, 1, 4);
        let model = Crossformer::new(cfg.clone(), &mut rng).expect("build");
        // Construct a signal where segment 0 and segment 1 differ.
        let mut x = vec![0.0_f32; cfg.seq_len];
        for (i, xv) in x.iter_mut().enumerate() {
            *xv = i as f32;
        }
        let emb = model.dsw_embed(&x).expect("embed");
        let d = cfg.d_model;
        let seg0 = &emb[0..d];
        let seg1 = &emb[d..2 * d];
        let diff: f32 = seg0
            .iter()
            .zip(seg1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-4, "distinct segments produced identical tokens");
    }
}
