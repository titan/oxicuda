//! Analytic backward pass (explicit gradients) for [`SaintLayer`].
//!
//! SAINT (Somepalli et al. 2021) alternates two attention mechanisms per layer:
//!
//! 1. **Row-wise** multi-head self-attention across the feature tokens of each
//!    sample (Pre-LN, residual).
//! 2. **Inter-sample** attention: for every feature position, multi-head
//!    self-attention across the `n_samples` rows (Pre-LN, residual).
//!
//! followed by a position-wise FFN with the SAINT-specific *post-LayerNorm on
//! the FFN output*: `tok ← tok + LN₂(FFN(LN₁(tok)))`.  The CLS-free head
//! mean-pools the feature tokens per sample and projects to logits.
//!
//! This module re-implements the forward with full caching and differentiates
//! every piece: the softmax-Jacobian multi-head attention (shared by the row and
//! inter-sample paths via `mhsa_backward`), all four LayerNorms, the GELU FFN,
//! and the pooling head.  Verified against central finite differences.

use super::saint::SaintLayer;
use crate::error::TabularResult;

// ─── Gradient container ────────────────────────────────────────────────────────

/// Accumulated gradients for every learnable parameter of a [`SaintLayer`].
#[derive(Debug, Clone)]
pub struct SaintGradients {
    /// Per-layer row-attention `W_q`.
    pub row_wq: Vec<Vec<f32>>,
    /// Per-layer row-attention `W_k`.
    pub row_wk: Vec<Vec<f32>>,
    /// Per-layer row-attention `W_v`.
    pub row_wv: Vec<Vec<f32>>,
    /// Per-layer row-attention `W_o`.
    pub row_wo: Vec<Vec<f32>>,
    /// Per-layer inter-sample `W_q`.
    pub inter_wq: Vec<Vec<f32>>,
    /// Per-layer inter-sample `W_k`.
    pub inter_wk: Vec<Vec<f32>>,
    /// Per-layer inter-sample `W_v`.
    pub inter_wv: Vec<Vec<f32>>,
    /// Per-layer inter-sample `W_o`.
    pub inter_wo: Vec<Vec<f32>>,
    /// Per-layer FFN `W1`.
    pub ffn_w1: Vec<Vec<f32>>,
    /// Per-layer FFN `b1`.
    pub ffn_b1: Vec<Vec<f32>>,
    /// Per-layer FFN `W2`.
    pub ffn_w2: Vec<Vec<f32>>,
    /// Per-layer FFN `b2`.
    pub ffn_b2: Vec<Vec<f32>>,
    /// LayerNorm scales (`4 * n_layers` entries).
    pub ln_gamma: Vec<Vec<f32>>,
    /// LayerNorm biases (`4 * n_layers` entries).
    pub ln_beta: Vec<Vec<f32>>,
    /// Head weight.
    pub head_w: Vec<f32>,
    /// Head bias.
    pub head_b: Vec<f32>,
}

// ─── helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn gelu(v: f32) -> f32 {
    v / (1.0 + (-1.702 * v).exp())
}

#[inline]
fn gelu_grad(v: f32) -> f32 {
    let s = 1.0 / (1.0 + (-1.702 * v).exp());
    s + v * 1.702 * s * (1.0 - s)
}

/// `C = A·B`, A `[m×k]`, B `[k×n]`, row-major.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

/// Backward for `Y = X·W` (X `[seq×ed]`, W `[ed×ed]`): `dW += Xᵀ·dY`,
/// `dX += dY·Wᵀ`.
fn linear_bwd(
    d_y: &[f32],
    x: &[f32],
    w: &[f32],
    seq: usize,
    ed: usize,
    d_w: &mut [f32],
    d_x: &mut [f32],
) {
    for s in 0..seq {
        for j in 0..ed {
            let gy = d_y[s * ed + j];
            if gy == 0.0 {
                continue;
            }
            for i in 0..ed {
                d_w[i * ed + j] += x[s * ed + i] * gy;
                d_x[s * ed + i] += w[i * ed + j] * gy;
            }
        }
    }
}

// ─── LayerNorm forward / backward ──────────────────────────────────────────────

struct LnCache {
    mean: Vec<f32>,
    inv_std: Vec<f32>,
}

fn ln_fwd(x: &[f32], seq: usize, ed: usize, g: &[f32], b: &[f32]) -> (Vec<f32>, LnCache) {
    let n = ed as f32;
    let mut out = vec![0.0_f32; seq * ed];
    let mut mean = vec![0.0_f32; seq];
    let mut inv_std = vec![0.0_f32; seq];
    for s in 0..seq {
        let row = &x[s * ed..(s + 1) * ed];
        let m = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        mean[s] = m;
        inv_std[s] = inv;
        for d in 0..ed {
            out[s * ed + d] = (row[d] - m) * inv * g[d] + b[d];
        }
    }
    (out, LnCache { mean, inv_std })
}

#[allow(clippy::too_many_arguments)]
fn ln_bwd(
    d_out: &[f32],
    x: &[f32],
    c: &LnCache,
    g: &[f32],
    seq: usize,
    ed: usize,
    d_g: &mut [f32],
    d_b: &mut [f32],
    d_in: &mut [f32],
) {
    let n = ed as f32;
    for s in 0..seq {
        let row = &x[s * ed..(s + 1) * ed];
        let m = c.mean[s];
        let inv = c.inv_std[s];
        let mut sum_dxhat = 0.0_f32;
        let mut sum_dxhat_xhat = 0.0_f32;
        let mut dxhat = vec![0.0_f32; ed];
        for d in 0..ed {
            let xhat = (row[d] - m) * inv;
            let go = d_out[s * ed + d];
            d_g[d] += go * xhat;
            d_b[d] += go;
            let dh = go * g[d];
            dxhat[d] = dh;
            sum_dxhat += dh;
            sum_dxhat_xhat += dh * xhat;
        }
        for d in 0..ed {
            let xhat = (row[d] - m) * inv;
            d_in[s * ed + d] += inv / n * (n * dxhat[d] - sum_dxhat - xhat * sum_dxhat_xhat);
        }
    }
}

// ─── Multi-head self-attention forward / backward (no residual / LN) ───────────

struct MhsaCache {
    x: Vec<f32>, // input                          [seq*ed]
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<Vec<f32>>, // per head                        [nh][seq*seq]
    concat: Vec<f32>,    // pre-Wo                          [seq*ed]
}

/// MHSA forward returning the output and a cache for backprop.
fn mhsa_fwd(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    seq: usize,
    ed: usize,
    nh: usize,
) -> (Vec<f32>, MhsaCache) {
    let head_dim = ed / nh;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = matmul(x, wq, seq, ed, ed);
    let k = matmul(x, wk, seq, ed, ed);
    let v = matmul(x, wv, seq, ed, ed);
    let mut attn = Vec::with_capacity(nh);
    let mut concat = vec![0.0_f32; seq * ed];
    for h in 0..nh {
        let off = h * head_dim;
        let mut a = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            let mut row = vec![0.0_f32; seq];
            for (j, rj) in row.iter_mut().enumerate() {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[i * ed + off + d] * k[j * ed + off + d];
                }
                *rj = dot * scale;
            }
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for rj in &mut row {
                *rj = (*rj - mx).exp();
                sum += *rj;
            }
            let denom = if sum < 1e-30 { 1e-30 } else { sum };
            for (j, &rj) in row.iter().enumerate() {
                a[i * seq + j] = rj / denom;
            }
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for j in 0..seq {
                    acc += a[i * seq + j] * v[j * ed + off + d];
                }
                concat[i * ed + off + d] = acc;
            }
        }
        attn.push(a);
    }
    let out = matmul(&concat, wo, seq, ed, ed);
    (
        out,
        MhsaCache {
            x: x.to_vec(),
            q,
            k,
            v,
            attn,
            concat,
        },
    )
}

/// MHSA backward.  `d_out` is the gradient w.r.t. the projected output.
/// Accumulates `d_wq/d_wk/d_wv/d_wo` and returns `d_x` (gradient w.r.t. input).
#[allow(clippy::too_many_arguments)]
fn mhsa_backward(
    d_out: &[f32],
    c: &MhsaCache,
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    seq: usize,
    ed: usize,
    nh: usize,
    d_wq: &mut [f32],
    d_wk: &mut [f32],
    d_wv: &mut [f32],
    d_wo: &mut [f32],
) -> Vec<f32> {
    let head_dim = ed / nh;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // through Wo: out = concat·Wo
    let mut d_concat = vec![0.0_f32; seq * ed];
    for s in 0..seq {
        for o in 0..ed {
            let go = d_out[s * ed + o];
            if go == 0.0 {
                continue;
            }
            for i in 0..ed {
                d_wo[i * ed + o] += c.concat[s * ed + i] * go;
                d_concat[s * ed + i] += wo[i * ed + o] * go;
            }
        }
    }

    let mut d_q = vec![0.0_f32; seq * ed];
    let mut d_k = vec![0.0_f32; seq * ed];
    let mut d_v = vec![0.0_f32; seq * ed];
    for h in 0..nh {
        let off = h * head_dim;
        let a = &c.attn[h];
        let mut d_attn = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let aij = a[i * seq + j];
                let mut da = 0.0_f32;
                for d in 0..head_dim {
                    let dc = d_concat[i * ed + off + d];
                    da += dc * c.v[j * ed + off + d];
                    d_v[j * ed + off + d] += aij * dc;
                }
                d_attn[i * seq + j] = da;
            }
        }
        for i in 0..seq {
            let mut dot = 0.0_f32;
            for l in 0..seq {
                dot += a[i * seq + l] * d_attn[i * seq + l];
            }
            for j in 0..seq {
                let d_score = a[i * seq + j] * (d_attn[i * seq + j] - dot) * scale;
                for d in 0..head_dim {
                    d_q[i * ed + off + d] += d_score * c.k[j * ed + off + d];
                    d_k[j * ed + off + d] += d_score * c.q[i * ed + off + d];
                }
            }
        }
    }

    let mut d_x = vec![0.0_f32; seq * ed];
    linear_bwd(&d_q, &c.x, wq, seq, ed, d_wq, &mut d_x);
    linear_bwd(&d_k, &c.x, wk, seq, ed, d_wk, &mut d_x);
    linear_bwd(&d_v, &c.x, wv, seq, ed, d_wv, &mut d_x);
    d_x
}

// ─── Per-layer cache ───────────────────────────────────────────────────────────

struct SaintLayerCache {
    h_in: Vec<f32>,             // layer input                       [N*nf*ed]
    row_ln: Vec<LnCache>,       // per-sample LN1 stats
    row_ln_out: Vec<f32>,       // per-sample LN1 output (flat)      [N*nf*ed]
    row_mhsa: Vec<MhsaCache>,   // per-sample row MHSA caches
    row_attn_out: Vec<f32>,     // residual after row attn           [N*nf*ed]
    inter_ln: LnCache,          // inter LN (all tokens)
    inter_ln_out: Vec<f32>,     // inter LN output                   [N*nf*ed]
    inter_mhsa: Vec<MhsaCache>, // per-feature inter MHSA caches
    after_inter: Vec<f32>,      // residual after inter attn         [N*nf*ed]
    ffn1_ln: LnCache,           // FFN pre-LN
    ffn1_ln_out: Vec<f32>,      //                                   [N*nf*ed]
    ffn_pre: Vec<f32>,          // W1·ln + b1                        [N*nf*fh]
    ffn_act: Vec<f32>,          // GELU                              [N*nf*fh]
    ffn_out: Vec<f32>,          // W2·act + b2                       [N*nf*ed]
    ffn2_ln: LnCache,           // post-LN on FFN output
}

impl SaintLayer {
    /// Forward caching all intermediates for backprop.
    fn forward_cached(
        &self,
        x: &[f32],
        n_samples: usize,
    ) -> TabularResult<(Vec<f32>, Vec<SaintLayerCache>, Vec<f32>)> {
        let cfg = self.config_ref();
        let ed = cfg.embed_dim;
        let nf = cfg.n_features;
        let fh = cfg.ffn_hidden;
        let nh = cfg.n_heads;

        let mut h = x.to_vec();
        let mut caches = Vec::with_capacity(cfg.n_layers);

        for layer in 0..cfg.n_layers {
            let ln_row = layer * 4;
            let ln_inter = layer * 4 + 1;
            let ln_ffn1 = layer * 4 + 2;
            let ln_ffn2 = layer * 4 + 3;
            let h_in = h.clone();

            // ── Row attention per sample ──────────────────────────────────────
            let mut row_ln = Vec::with_capacity(n_samples);
            let mut row_ln_out = vec![0.0_f32; n_samples * nf * ed];
            let mut row_mhsa = Vec::with_capacity(n_samples);
            let mut row_attn_out = vec![0.0_f32; n_samples * nf * ed];
            for s in 0..n_samples {
                let base = s * nf * ed;
                let sample = &h[base..base + nf * ed];
                let (ln_out, ln_c) = ln_fwd(
                    sample,
                    nf,
                    ed,
                    self.ln_gamma_ref(ln_row),
                    self.ln_beta_ref(ln_row),
                );
                let (attn, mc) = mhsa_fwd(
                    &ln_out,
                    self.row_wq_ref(layer),
                    self.row_wk_ref(layer),
                    self.row_wv_ref(layer),
                    self.row_wo_ref(layer),
                    nf,
                    ed,
                    nh,
                );
                for i in 0..nf * ed {
                    row_attn_out[base + i] = sample[i] + attn[i];
                }
                row_ln_out[base..base + nf * ed].copy_from_slice(&ln_out);
                row_ln.push(ln_c);
                row_mhsa.push(mc);
            }

            // ── Inter-sample attention ────────────────────────────────────────
            let (inter_ln_out, inter_ln_c) = ln_fwd(
                &row_attn_out,
                n_samples * nf,
                ed,
                self.ln_gamma_ref(ln_inter),
                self.ln_beta_ref(ln_inter),
            );
            // gather per feature → MHSA over samples → scatter
            let mut inter_attn = vec![0.0_f32; n_samples * nf * ed];
            let mut inter_mhsa = Vec::with_capacity(nf);
            for f in 0..nf {
                let mut seqf = vec![0.0_f32; n_samples * ed];
                for s in 0..n_samples {
                    let src = s * nf * ed + f * ed;
                    seqf[s * ed..(s + 1) * ed].copy_from_slice(&inter_ln_out[src..src + ed]);
                }
                let (out, mc) = mhsa_fwd(
                    &seqf,
                    self.inter_wq_ref(layer),
                    self.inter_wk_ref(layer),
                    self.inter_wv_ref(layer),
                    self.inter_wo_ref(layer),
                    n_samples,
                    ed,
                    1,
                );
                for s in 0..n_samples {
                    let dst = s * nf * ed + f * ed;
                    inter_attn[dst..dst + ed].copy_from_slice(&out[s * ed..(s + 1) * ed]);
                }
                inter_mhsa.push(mc);
            }
            let after_inter: Vec<f32> = row_attn_out
                .iter()
                .zip(inter_attn.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // ── FFN per token: tok + LN2(FFN(LN1(tok))) ──────────────────────
            let (ffn1_ln_out, ffn1_ln_c) = ln_fwd(
                &after_inter,
                n_samples * nf,
                ed,
                self.ln_gamma_ref(ln_ffn1),
                self.ln_beta_ref(ln_ffn1),
            );
            let ntok = n_samples * nf;
            let mut ffn_pre = vec![0.0_f32; ntok * fh];
            let mut ffn_act = vec![0.0_f32; ntok * fh];
            let mut ffn_out = vec![0.0_f32; ntok * ed];
            let w1 = self.ffn_w1_ref(layer);
            let b1 = self.ffn_b1_ref(layer);
            let w2 = self.ffn_w2_ref(layer);
            let b2 = self.ffn_b2_ref(layer);
            for t in 0..ntok {
                let xin = &ffn1_ln_out[t * ed..(t + 1) * ed];
                for o in 0..fh {
                    let mut acc = b1[o];
                    for (i, &xi) in xin.iter().enumerate() {
                        acc += w1[o * ed + i] * xi;
                    }
                    ffn_pre[t * fh + o] = acc;
                    ffn_act[t * fh + o] = gelu(acc);
                }
                for o in 0..ed {
                    let mut acc = b2[o];
                    for i in 0..fh {
                        acc += w2[o * fh + i] * ffn_act[t * fh + i];
                    }
                    ffn_out[t * ed + o] = acc;
                }
            }
            let (ffn2_ln_out, ffn2_ln_c) = ln_fwd(
                &ffn_out,
                ntok,
                ed,
                self.ln_gamma_ref(ln_ffn2),
                self.ln_beta_ref(ln_ffn2),
            );
            let mut new_h = vec![0.0_f32; ntok * ed];
            for i in 0..ntok * ed {
                new_h[i] = after_inter[i] + ffn2_ln_out[i];
            }

            caches.push(SaintLayerCache {
                h_in,
                row_ln,
                row_ln_out,
                row_mhsa,
                row_attn_out,
                inter_ln: inter_ln_c,
                inter_ln_out,
                inter_mhsa,
                after_inter,
                ffn1_ln: ffn1_ln_c,
                ffn1_ln_out,
                ffn_pre,
                ffn_act,
                ffn_out,
                ffn2_ln: ffn2_ln_c,
            });
            h = new_h;
        }

        // ── Head: mean-pool features per sample → linear ─────────────────────
        let mut logits = Vec::with_capacity(n_samples * cfg.n_classes);
        let head_w = self.head_w_ref();
        let head_b = self.head_b_ref();
        for s in 0..n_samples {
            let base = s * nf * ed;
            let mut pooled = vec![0.0_f32; ed];
            for f in 0..nf {
                for d in 0..ed {
                    pooled[d] += h[base + f * ed + d];
                }
            }
            for v in &mut pooled {
                *v /= nf as f32;
            }
            let mut sl = head_b.to_vec();
            for (o, slo) in sl.iter_mut().enumerate() {
                for (d, &pv) in pooled.iter().enumerate() {
                    *slo += head_w[o * ed + d] * pv;
                }
            }
            logits.extend_from_slice(&sl);
        }
        Ok((logits, caches, h))
    }

    /// Analytic backward pass.
    ///
    /// `grad_logits` is `dL/d logits` (`[n_samples * n_classes]`).  Returns the
    /// parameter gradients and the gradient w.r.t. the input tokens
    /// (`[n_samples * n_features * embed_dim]`).
    pub fn backward(
        &self,
        x: &[f32],
        n_samples: usize,
        grad_logits: &[f32],
    ) -> TabularResult<(SaintGradients, Vec<f32>)> {
        let cfg = self.config_ref();
        let ed = cfg.embed_dim;
        let nf = cfg.n_features;
        let fh = cfg.ffn_hidden;
        let nh = cfg.n_heads;
        let nc = cfg.n_classes;
        let ntok = n_samples * nf;

        let (_logits, caches, h_final) = self.forward_cached(x, n_samples)?;
        let mut g = SaintGradients::zeros(self);

        // ── Head backward ─────────────────────────────────────────────────────
        let head_w = self.head_w_ref();
        let mut d_h = vec![0.0_f32; ntok * ed];
        for s in 0..n_samples {
            let base = s * nf * ed;
            // recompute pooled
            let mut pooled = vec![0.0_f32; ed];
            for f in 0..nf {
                for d in 0..ed {
                    pooled[d] += h_final[base + f * ed + d];
                }
            }
            for v in &mut pooled {
                *v /= nf as f32;
            }
            let gl = &grad_logits[s * nc..(s + 1) * nc];
            let mut d_pooled = vec![0.0_f32; ed];
            for (o, &go) in gl.iter().enumerate() {
                g.head_b[o] += go;
                for d in 0..ed {
                    g.head_w[o * ed + d] += go * pooled[d];
                    d_pooled[d] += go * head_w[o * ed + d];
                }
            }
            // pooled = mean_f h[base+f] → each token gets d_pooled / nf
            for f in 0..nf {
                for d in 0..ed {
                    d_h[base + f * ed + d] += d_pooled[d] / nf as f32;
                }
            }
        }

        // ── Layers in reverse ─────────────────────────────────────────────────
        for layer in (0..cfg.n_layers).rev() {
            let c = &caches[layer];
            let ln_row = layer * 4;
            let ln_inter = layer * 4 + 1;
            let ln_ffn1 = layer * 4 + 2;
            let ln_ffn2 = layer * 4 + 3;

            // === FFN block: new_h = after_inter + LN2(FFN(LN1(after_inter))) ===
            // d_h is gradient w.r.t. new_h.
            let mut d_after_inter = d_h.clone(); // residual path

            // through LN2
            let mut d_ffn_out = vec![0.0_f32; ntok * ed];
            ln_bwd(
                &d_h,
                &c.ffn_out,
                &c.ffn2_ln,
                self.ln_gamma_ref(ln_ffn2),
                ntok,
                ed,
                &mut g.ln_gamma[ln_ffn2],
                &mut g.ln_beta[ln_ffn2],
                &mut d_ffn_out,
            );

            // through FFN (W2,b2 / GELU / W1,b1) per token → d_ffn1_ln
            let w1 = self.ffn_w1_ref(layer);
            let w2 = self.ffn_w2_ref(layer);
            let mut d_ffn1_ln = vec![0.0_f32; ntok * ed];
            for t in 0..ntok {
                let d_out = &d_ffn_out[t * ed..(t + 1) * ed];
                let mut d_act = vec![0.0_f32; fh];
                for o in 0..ed {
                    let go = d_out[o];
                    g.ffn_b2[layer][o] += go;
                    for i in 0..fh {
                        g.ffn_w2[layer][o * fh + i] += go * c.ffn_act[t * fh + i];
                        d_act[i] += go * w2[o * fh + i];
                    }
                }
                let mut d_pre = vec![0.0_f32; fh];
                for o in 0..fh {
                    d_pre[o] = d_act[o] * gelu_grad(c.ffn_pre[t * fh + o]);
                }
                let xin = &c.ffn1_ln_out[t * ed..(t + 1) * ed];
                for o in 0..fh {
                    let gp = d_pre[o];
                    g.ffn_b1[layer][o] += gp;
                    for (i, &xi) in xin.iter().enumerate() {
                        g.ffn_w1[layer][o * ed + i] += gp * xi;
                        d_ffn1_ln[t * ed + i] += gp * w1[o * ed + i];
                    }
                }
            }
            // through LN1 → adds to d_after_inter
            ln_bwd(
                &d_ffn1_ln,
                &c.after_inter,
                &c.ffn1_ln,
                self.ln_gamma_ref(ln_ffn1),
                ntok,
                ed,
                &mut g.ln_gamma[ln_ffn1],
                &mut g.ln_beta[ln_ffn1],
                &mut d_after_inter,
            );

            // === Inter block: after_inter = row_attn_out + inter_attn ===
            // d_after_inter is gradient w.r.t. after_inter.
            let mut d_row_attn_out = d_after_inter.clone(); // residual path
            let d_inter_attn = d_after_inter; // the inter_attn branch

            // through per-feature MHSA → d_inter_ln_out (gathered/scattered)
            let mut d_inter_ln_out = vec![0.0_f32; ntok * ed];
            for f in 0..nf {
                // gather d_out for this feature
                let mut d_out_f = vec![0.0_f32; n_samples * ed];
                for s in 0..n_samples {
                    let src = s * nf * ed + f * ed;
                    d_out_f[s * ed..(s + 1) * ed].copy_from_slice(&d_inter_attn[src..src + ed]);
                }
                let d_seqf = mhsa_backward(
                    &d_out_f,
                    &c.inter_mhsa[f],
                    self.inter_wq_ref(layer),
                    self.inter_wk_ref(layer),
                    self.inter_wv_ref(layer),
                    self.inter_wo_ref(layer),
                    n_samples,
                    ed,
                    1,
                    &mut g.inter_wq[layer],
                    &mut g.inter_wk[layer],
                    &mut g.inter_wv[layer],
                    &mut g.inter_wo[layer],
                );
                for s in 0..n_samples {
                    let dst = s * nf * ed + f * ed;
                    for d in 0..ed {
                        d_inter_ln_out[dst + d] += d_seqf[s * ed + d];
                    }
                }
            }
            // through inter LN → adds to d_row_attn_out
            ln_bwd(
                &d_inter_ln_out,
                &c.row_attn_out,
                &c.inter_ln,
                self.ln_gamma_ref(ln_inter),
                ntok,
                ed,
                &mut g.ln_gamma[ln_inter],
                &mut g.ln_beta[ln_inter],
                &mut d_row_attn_out,
            );

            // === Row block (per sample): row_attn_out = h_in + MHSA(LN(h_in)) ===
            let mut d_h_in = d_row_attn_out.clone(); // residual path
            for s in 0..n_samples {
                let base = s * nf * ed;
                let d_attn_s = &d_row_attn_out[base..base + nf * ed];
                let d_ln_out_s = mhsa_backward(
                    d_attn_s,
                    &c.row_mhsa[s],
                    self.row_wq_ref(layer),
                    self.row_wk_ref(layer),
                    self.row_wv_ref(layer),
                    self.row_wo_ref(layer),
                    nf,
                    ed,
                    nh,
                    &mut g.row_wq[layer],
                    &mut g.row_wk[layer],
                    &mut g.row_wv[layer],
                    &mut g.row_wo[layer],
                );
                // through per-sample LN → adds to d_h_in for this sample
                let sample_in = &c.h_in[base..base + nf * ed];
                let mut d_in_s = vec![0.0_f32; nf * ed];
                ln_bwd(
                    &d_ln_out_s,
                    sample_in,
                    &c.row_ln[s],
                    self.ln_gamma_ref(ln_row),
                    nf,
                    ed,
                    &mut g.ln_gamma[ln_row],
                    &mut g.ln_beta[ln_row],
                    &mut d_in_s,
                );
                for i in 0..nf * ed {
                    d_h_in[base + i] += d_in_s[i];
                }
            }

            d_h = d_h_in;
            // silence unused (row_ln_out / inter_ln_out kept for clarity/debug)
            let _ = &c.row_ln_out;
            let _ = &c.inter_ln_out;
        }

        Ok((g, d_h))
    }
}

impl SaintGradients {
    fn zeros(model: &SaintLayer) -> Self {
        let cfg = model.config_ref();
        let ed = cfg.embed_dim;
        let fh = cfg.ffn_hidden;
        let nl = cfg.n_layers;
        let mk = |n: usize| vec![0.0_f32; n];
        Self {
            row_wq: (0..nl).map(|_| mk(ed * ed)).collect(),
            row_wk: (0..nl).map(|_| mk(ed * ed)).collect(),
            row_wv: (0..nl).map(|_| mk(ed * ed)).collect(),
            row_wo: (0..nl).map(|_| mk(ed * ed)).collect(),
            inter_wq: (0..nl).map(|_| mk(ed * ed)).collect(),
            inter_wk: (0..nl).map(|_| mk(ed * ed)).collect(),
            inter_wv: (0..nl).map(|_| mk(ed * ed)).collect(),
            inter_wo: (0..nl).map(|_| mk(ed * ed)).collect(),
            ffn_w1: (0..nl).map(|_| mk(fh * ed)).collect(),
            ffn_b1: (0..nl).map(|_| mk(fh)).collect(),
            ffn_w2: (0..nl).map(|_| mk(ed * fh)).collect(),
            ffn_b2: (0..nl).map(|_| mk(ed)).collect(),
            ln_gamma: (0..nl * 4).map(|_| mk(ed)).collect(),
            ln_beta: (0..nl * 4).map(|_| mk(ed)).collect(),
            head_w: mk(cfg.n_classes * ed),
            head_b: mk(cfg.n_classes),
        }
    }
}

// ─── Parameter handle for finite-difference tests ──────────────────────────────

/// Addresses one scalar SAINT parameter (test-only).
#[cfg(test)]
pub(crate) enum SaintParam {
    RowWq(usize, usize),
    RowWk(usize, usize),
    RowWv(usize, usize),
    RowWo(usize, usize),
    InterWq(usize, usize),
    InterWk(usize, usize),
    InterWv(usize, usize),
    InterWo(usize, usize),
    FfnW1(usize, usize),
    FfnB1(usize, usize),
    FfnW2(usize, usize),
    FfnB2(usize, usize),
    LnGamma(usize, usize),
    LnBeta(usize, usize),
    HeadW(usize),
    HeadB(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::saint::SaintConfig;
    use crate::handle::LcgRng;

    fn tiny() -> (SaintLayer, Vec<f32>, usize) {
        let cfg = SaintConfig {
            n_features: 3,
            embed_dim: 4,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 6,
            n_classes: 3,
        };
        let mut rng = LcgRng::new(456);
        let layer = SaintLayer::new(cfg, &mut rng).expect("new");
        let n_samples = 2;
        let mut x = vec![0.0_f32; n_samples * 3 * 4];
        let mut r = LcgRng::new(7);
        r.fill_normal_scaled(&mut x, 0.5);
        (layer, x, n_samples)
    }

    fn loss(logits: &[f32], dir: &[f32]) -> f32 {
        logits.iter().zip(dir.iter()).map(|(&a, &b)| a * b).sum()
    }

    #[test]
    fn forward_cached_matches_forward() {
        let (layer, x, ns) = tiny();
        let l1 = layer.forward(&x, ns).expect("forward");
        let (l2, _c, _h) = layer.forward_cached(&x, ns).expect("cached");
        for (a, b) in l1.iter().zip(l2.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn grad_check_input() {
        let (layer, x, ns) = tiny();
        let mut dir = vec![0.0_f32; ns * 3];
        let mut r = LcgRng::new(3);
        r.fill_normal_scaled(&mut dir, 0.7);
        let (_g, dx) = layer.backward(&x, ns, &dir).expect("bwd");
        // eps = 5e-3 escapes f32 rounding noise (the loss magnitude is O(1–10),
        // so smaller steps lose precision on tiny-gradient components).
        let eps = 5e-3_f32;
        for j in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let lp = loss(&layer.forward(&xp, ns).expect("f+"), &dir);
            let lm = loss(&layer.forward(&xm, ns).expect("f-"), &dir);
            let fd = (lp - lm) / (2.0 * eps);
            let abserr = (fd - dx[j]).abs();
            let rel = abserr / fd.abs().max(dx[j].abs()).max(1e-3);
            // Pass if either the relative OR the absolute error is small;
            // near-zero-gradient components are dominated by f32 round-off.
            assert!(
                rel < 4e-2 || abserr < 1e-3,
                "dx[{j}] analytic={} fd={fd} abserr={abserr} rel={rel}",
                dx[j]
            );
        }
    }

    #[test]
    fn grad_check_parameters() {
        let (mut layer, x, ns) = tiny();
        let mut dir = vec![0.0_f32; ns * 3];
        let mut r = LcgRng::new(9);
        r.fill_normal_scaled(&mut dir, 0.6);
        let (g, _dx) = layer.backward(&x, ns, &dir).expect("bwd");

        let checks: Vec<(&str, SaintParam, f32)> = vec![
            ("row_wq0[5]", SaintParam::RowWq(0, 5), g.row_wq[0][5]),
            ("row_wk1[3]", SaintParam::RowWk(1, 3), g.row_wk[1][3]),
            ("row_wv0[10]", SaintParam::RowWv(0, 10), g.row_wv[0][10]),
            ("row_wo1[7]", SaintParam::RowWo(1, 7), g.row_wo[1][7]),
            ("inter_wq0[2]", SaintParam::InterWq(0, 2), g.inter_wq[0][2]),
            (
                "inter_wk1[11]",
                SaintParam::InterWk(1, 11),
                g.inter_wk[1][11],
            ),
            ("inter_wv0[6]", SaintParam::InterWv(0, 6), g.inter_wv[0][6]),
            ("inter_wo1[1]", SaintParam::InterWo(1, 1), g.inter_wo[1][1]),
            ("ffn_w1_0[8]", SaintParam::FfnW1(0, 8), g.ffn_w1[0][8]),
            ("ffn_b1_1[2]", SaintParam::FfnB1(1, 2), g.ffn_b1[1][2]),
            ("ffn_w2_0[5]", SaintParam::FfnW2(0, 5), g.ffn_w2[0][5]),
            ("ffn_b2_1[1]", SaintParam::FfnB2(1, 1), g.ffn_b2[1][1]),
            (
                "ln_gamma_row[2]",
                SaintParam::LnGamma(0, 2),
                g.ln_gamma[0][2],
            ),
            (
                "ln_beta_inter[1]",
                SaintParam::LnBeta(1, 1),
                g.ln_beta[1][1],
            ),
            (
                "ln_gamma_ffn2[3]",
                SaintParam::LnGamma(7, 3),
                g.ln_gamma[7][3],
            ),
            ("head_w[5]", SaintParam::HeadW(5), g.head_w[5]),
            ("head_b[1]", SaintParam::HeadB(1), g.head_b[1]),
        ];
        let eps = 5e-3_f32;
        for (label, p, analytic) in checks {
            let orig = layer.param_get(&p);
            layer.param_set(&p, orig + eps);
            let lp = loss(&layer.forward(&x, ns).expect("f+"), &dir);
            layer.param_set(&p, orig - eps);
            let lm = loss(&layer.forward(&x, ns).expect("f-"), &dir);
            layer.param_set(&p, orig);
            let fd = (lp - lm) / (2.0 * eps);
            let abserr = (fd - analytic).abs();
            let rel = abserr / fd.abs().max(analytic.abs()).max(1e-3);
            assert!(
                rel < 4e-2 || abserr < 1e-3,
                "param {label}: analytic={analytic} fd={fd} abserr={abserr} rel={rel}"
            );
        }
    }
}
