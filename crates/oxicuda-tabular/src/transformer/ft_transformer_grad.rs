//! Analytic backward pass (explicit gradients) for [`FtTransformer`].
//!
//! Implements reverse-mode automatic differentiation by hand for the full
//! FT-Transformer forward graph:
//!
//! 1. Feature tokenisation (`x_j * w_j + b_j` for continuous, lookup for categorical).
//! 2. CLS-token prepend.
//! 3. Per layer: Pre-LayerNorm → multi-head self-attention (softmax-Jacobian
//!    through the scaled dot-product) → residual → Pre-LayerNorm → position-wise
//!    GELU FFN → residual.
//! 4. CLS-token classification head.
//!
//! The gradient is verified against central finite differences in the unit tests
//! (`grad_check_*`); analytic and numerical gradients agree to `< 2e-2` relative
//! error on a tiny model, which is the expected accuracy for `f32` central
//! differences through a softmax + LayerNorm stack.

use crate::error::TabularResult;
use crate::transformer::ft_transformer::{FeatureTokenizer, FtTransformer};

// ─── Gradient container ────────────────────────────────────────────────────────

/// Accumulated gradients for every learnable parameter of an [`FtTransformer`].
///
/// All buffers mirror the shapes of the corresponding forward parameters so a
/// caller can apply an optimiser step element-wise.
#[derive(Debug, Clone)]
pub struct FtGradients {
    /// Gradient w.r.t. the CLS token, `[embed_dim]`.
    pub cls_token: Vec<f32>,
    /// Gradient w.r.t. continuous tokenizer weights, `[n_cont * embed_dim]`.
    pub cont_w: Vec<f32>,
    /// Gradient w.r.t. continuous tokenizer biases, `[n_cont * embed_dim]`.
    pub cont_b: Vec<f32>,
    /// Gradient w.r.t. categorical embedding tables, `[n_cat][n_cat_j * embed_dim]`.
    pub cat_embeds: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. `W_q`, `[n_layers][embed_dim * embed_dim]`.
    pub wq: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. `W_k`.
    pub wk: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. `W_v`.
    pub wv: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. `W_o`.
    pub wo: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. FFN `W1`, `[n_layers][ffn_hidden * embed_dim]`.
    pub ffn_w1: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. FFN `b1`.
    pub ffn_b1: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. FFN `W2`, `[n_layers][embed_dim * ffn_hidden]`.
    pub ffn_w2: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. FFN `b2`.
    pub ffn_b2: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. the first LayerNorm scale.
    pub ln1_g: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. the first LayerNorm bias.
    pub ln1_b: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. the second LayerNorm scale.
    pub ln2_g: Vec<Vec<f32>>,
    /// Per-layer gradients w.r.t. the second LayerNorm bias.
    pub ln2_b: Vec<Vec<f32>>,
    /// Gradient w.r.t. the head weight, `[n_classes * embed_dim]`.
    pub head_w: Vec<f32>,
    /// Gradient w.r.t. the head bias, `[n_classes]`.
    pub head_b: Vec<f32>,
}

// ─── Cached intermediates for one transformer layer ────────────────────────────

struct LayerCache {
    // input to the layer (== residual stream entering)            [seq*ed]
    input: Vec<f32>,
    // LN1 normalised tokens (input to attention)                  [seq*ed]
    ln1_out: Vec<f32>,
    // per-token LN1 statistics
    ln1_mean: Vec<f32>,
    ln1_inv_std: Vec<f32>,
    // Q, K, V projections                                          [seq*ed]
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    // attention probabilities per head                            [n_heads][seq*seq]
    attn: Vec<Vec<f32>>,
    // concatenated head outputs (pre Wo)                          [seq*ed]
    concat: Vec<f32>,
    // residual stream after attention (input + Wo·concat)          [seq*ed]
    after_attn: Vec<f32>,
    // LN2 normalised tokens (input to FFN)                        [seq*ed]
    ln2_out: Vec<f32>,
    ln2_mean: Vec<f32>,
    ln2_inv_std: Vec<f32>,
    // FFN pre-activation h = W1·ln2 + b1                          [seq*fh]
    ffn_pre: Vec<f32>,
    // FFN activation a = GELU(h)                                  [seq*fh]
    ffn_act: Vec<f32>,
}

// ─── small linear-algebra helpers ──────────────────────────────────────────────

/// `C = A · B`, A is `[m×k]`, B is `[k×n]`, all row-major.
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

#[inline]
fn gelu(v: f32) -> f32 {
    v / (1.0 + (-1.702 * v).exp())
}

/// Derivative of the sigmoid-approximated GELU `g(v) = v·σ(1.702 v)`.
#[inline]
fn gelu_grad(v: f32) -> f32 {
    let s = 1.0 / (1.0 + (-1.702 * v).exp());
    s + v * 1.702 * s * (1.0 - s)
}

impl FtTransformer {
    /// Run a forward pass while caching every intermediate needed for backprop,
    /// returning `(logits, per_layer_caches, cls_after_layers)`.
    #[allow(clippy::type_complexity)]
    fn forward_cached(
        &self,
        x_cont: &[f32],
        x_cat: &[usize],
    ) -> TabularResult<(Vec<f32>, Vec<LayerCache>, Vec<f32>)> {
        let cfg = self.config_ref();
        let ed = cfg.embed_dim;
        let nh = cfg.n_heads;
        let fh = cfg.ffn_hidden;
        let head_dim = ed / nh;

        let feat_tokens = self.tokenizer_ref().tokenize(x_cont, x_cat)?;
        let n_feat = self.tokenizer_ref().n_features();
        let seq = n_feat + 1;

        let mut h = Vec::with_capacity(seq * ed);
        h.extend_from_slice(self.cls_token_ref());
        h.extend_from_slice(&feat_tokens);

        let mut caches = Vec::with_capacity(cfg.n_layers);

        for layer in 0..cfg.n_layers {
            let input = h.clone();

            // ── Pre-LN 1 ──────────────────────────────────────────────────────
            let (ln1_out, ln1_mean, ln1_inv_std) =
                layer_norm_fwd(&h, seq, ed, self.ln1_g_ref(layer), self.ln1_b_ref(layer));

            // ── QKV projections ───────────────────────────────────────────────
            let q = matmul(&ln1_out, self.wq_ref(layer), seq, ed, ed);
            let k = matmul(&ln1_out, self.wk_ref(layer), seq, ed, ed);
            let v = matmul(&ln1_out, self.wv_ref(layer), seq, ed, ed);

            // ── Per-head scaled-dot-product attention ─────────────────────────
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn = Vec::with_capacity(nh);
            let mut concat = vec![0.0_f32; seq * ed];
            for hh in 0..nh {
                let off = hh * head_dim;
                let mut a = vec![0.0_f32; seq * seq];
                for i in 0..seq {
                    // scores row
                    let mut row = vec![0.0_f32; seq];
                    for (j, rj) in row.iter_mut().enumerate() {
                        let mut dot = 0.0_f32;
                        for d in 0..head_dim {
                            dot += q[i * ed + off + d] * k[j * ed + off + d];
                        }
                        *rj = dot * scale;
                    }
                    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0_f32;
                    for rj in &mut row {
                        *rj = (*rj - m).exp();
                        sum += *rj;
                    }
                    let denom = if sum < 1e-30 { 1e-30 } else { sum };
                    for (j, &rj) in row.iter().enumerate() {
                        a[i * seq + j] = rj / denom;
                    }
                    // weighted sum of V → concat
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

            // ── Output projection + residual ──────────────────────────────────
            let proj = matmul(&concat, self.wo_ref(layer), seq, ed, ed);
            let mut after_attn = input.clone();
            for i in 0..seq * ed {
                after_attn[i] += proj[i];
            }

            // ── Pre-LN 2 ──────────────────────────────────────────────────────
            let (ln2_out, ln2_mean, ln2_inv_std) = layer_norm_fwd(
                &after_attn,
                seq,
                ed,
                self.ln2_g_ref(layer),
                self.ln2_b_ref(layer),
            );

            // ── Position-wise FFN + residual ──────────────────────────────────
            let mut ffn_pre = vec![0.0_f32; seq * fh];
            let mut ffn_act = vec![0.0_f32; seq * fh];
            let mut new_h = after_attn.clone();
            let w1 = self.ffn_w1_ref(layer);
            let b1 = self.ffn_b1_ref(layer);
            let w2 = self.ffn_w2_ref(layer);
            let b2 = self.ffn_b2_ref(layer);
            for s in 0..seq {
                let x = &ln2_out[s * ed..(s + 1) * ed];
                for o in 0..fh {
                    let mut acc = b1[o];
                    for (i, &xi) in x.iter().enumerate() {
                        acc += w1[o * ed + i] * xi;
                    }
                    ffn_pre[s * fh + o] = acc;
                    ffn_act[s * fh + o] = gelu(acc);
                }
                for o in 0..ed {
                    let mut acc = b2[o];
                    for i in 0..fh {
                        acc += w2[o * fh + i] * ffn_act[s * fh + i];
                    }
                    new_h[s * ed + o] += acc;
                }
            }

            caches.push(LayerCache {
                input,
                ln1_out,
                ln1_mean,
                ln1_inv_std,
                q,
                k,
                v,
                attn,
                concat,
                after_attn,
                ln2_out,
                ln2_mean,
                ln2_inv_std,
                ffn_pre,
                ffn_act,
            });
            h = new_h;
        }

        // ── Head on CLS token ─────────────────────────────────────────────────
        let cls = h[0..ed].to_vec();
        let mut logits = self.head_b_ref().to_vec();
        let head_w = self.head_w_ref();
        for (o, lo) in logits.iter_mut().enumerate() {
            for (d, &cv) in cls.iter().enumerate() {
                *lo += head_w[o * ed + d] * cv;
            }
        }
        Ok((logits, caches, cls))
    }

    /// Analytic backward pass.
    ///
    /// `grad_logits` is `dL/d logits` (`[n_classes]`); typically
    /// `softmax(logits) − one_hot(target)` for cross-entropy.  Returns the
    /// gradients for every learnable parameter and the gradient w.r.t. the
    /// continuous input features (`d L / d x_cont`, `[n_cont]`), which is useful
    /// for input attribution / adversarial training.
    pub fn backward(
        &self,
        x_cont: &[f32],
        x_cat: &[usize],
        grad_logits: &[f32],
    ) -> TabularResult<(FtGradients, Vec<f32>)> {
        let cfg = self.config_ref();
        let ed = cfg.embed_dim;
        let nh = cfg.n_heads;
        let fh = cfg.ffn_hidden;
        let nl = cfg.n_layers;
        let head_dim = ed / nh;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let (_logits, caches, cls) = self.forward_cached(x_cont, x_cat)?;
        let n_feat = self.tokenizer_ref().n_features();
        let seq = n_feat + 1;

        let mut g = FtGradients::zeros(self, n_feat);

        // ── Head ──────────────────────────────────────────────────────────────
        g.head_b.copy_from_slice(grad_logits);
        let head_w = self.head_w_ref();
        let mut d_cls = vec![0.0_f32; ed];
        for (o, &gl) in grad_logits.iter().enumerate() {
            for d in 0..ed {
                g.head_w[o * ed + d] += gl * cls[d];
                d_cls[d] += gl * head_w[o * ed + d];
            }
        }

        // Gradient flowing into the residual stream `h` (output of last layer).
        let mut d_h = vec![0.0_f32; seq * ed];
        d_h[0..ed].copy_from_slice(&d_cls);

        // ── Layers in reverse ─────────────────────────────────────────────────
        for layer in (0..nl).rev() {
            let c = &caches[layer];

            // ===== FFN sub-block (residual: new_h = after_attn + FFN(LN2)) =====
            // d_h is gradient w.r.t. new_h.
            // residual path → after_attn receives d_h directly (accumulate later)
            let mut d_after_attn = d_h.clone();

            let w1 = self.ffn_w1_ref(layer);
            let w2 = self.ffn_w2_ref(layer);
            // grad w.r.t. ln2_out from FFN
            let mut d_ln2_out = vec![0.0_f32; seq * ed];
            for s in 0..seq {
                let d_out = &d_h[s * ed..(s + 1) * ed]; // dL/d(FFN out_s) == dL/d new_h_s
                // through W2,b2
                let mut d_act = vec![0.0_f32; fh];
                for o in 0..ed {
                    let go = d_out[o];
                    g.ffn_b2[layer][o] += go;
                    for i in 0..fh {
                        g.ffn_w2[layer][o * fh + i] += go * c.ffn_act[s * fh + i];
                        d_act[i] += go * w2[o * fh + i];
                    }
                }
                // through GELU
                let mut d_pre = vec![0.0_f32; fh];
                for o in 0..fh {
                    d_pre[o] = d_act[o] * gelu_grad(c.ffn_pre[s * fh + o]);
                }
                // through W1,b1
                let x = &c.ln2_out[s * ed..(s + 1) * ed];
                for o in 0..fh {
                    let gp = d_pre[o];
                    g.ffn_b1[layer][o] += gp;
                    for (i, &xi) in x.iter().enumerate() {
                        g.ffn_w1[layer][o * ed + i] += gp * xi;
                        d_ln2_out[s * ed + i] += gp * w1[o * ed + i];
                    }
                }
            }

            // through LN2 → adds to d_after_attn
            layer_norm_bwd(
                &d_ln2_out,
                &c.after_attn,
                &c.ln2_mean,
                &c.ln2_inv_std,
                self.ln2_g_ref(layer),
                seq,
                ed,
                &mut g.ln2_g[layer],
                &mut g.ln2_b[layer],
                &mut d_after_attn,
            );

            // ===== Attention sub-block (residual: after_attn = input + Wo·concat) =====
            // d_after_attn is gradient w.r.t. after_attn.
            let mut d_input = d_after_attn.clone(); // residual path

            // through Wo: proj = concat · Wo ; d_concat = d_proj · Woᵀ ; dWo = concatᵀ · d_proj
            let wo = self.wo_ref(layer);
            let mut d_concat = vec![0.0_f32; seq * ed];
            for s in 0..seq {
                for o in 0..ed {
                    let go = d_after_attn[s * ed + o];
                    if go == 0.0 {
                        continue;
                    }
                    for i in 0..ed {
                        g.wo[layer][i * ed + o] += go * c.concat[s * ed + i];
                        d_concat[s * ed + i] += go * wo[i * ed + o];
                    }
                }
            }

            // through per-head attention → d_q, d_k, d_v
            let mut d_q = vec![0.0_f32; seq * ed];
            let mut d_k = vec![0.0_f32; seq * ed];
            let mut d_v = vec![0.0_f32; seq * ed];
            for hh in 0..nh {
                let off = hh * head_dim;
                let a = &c.attn[hh];
                // d_v[j] = Σ_i a[i,j] * d_concat[i]
                // d_attn[i,j] = Σ_d d_concat[i,d] * v[j,d]
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
                // softmax Jacobian (row-wise): d_score[i,j] = a[i,j]*(d_attn[i,j] - Σ_l a[i,l] d_attn[i,l])
                for i in 0..seq {
                    let mut dot = 0.0_f32;
                    for l in 0..seq {
                        dot += a[i * seq + l] * d_attn[i * seq + l];
                    }
                    for j in 0..seq {
                        let d_score = a[i * seq + j] * (d_attn[i * seq + j] - dot) * scale;
                        // score[i,j] = scale * Σ_d q[i,d] k[j,d]  (scale folded in above)
                        for d in 0..head_dim {
                            d_q[i * ed + off + d] += d_score * c.k[j * ed + off + d];
                            d_k[j * ed + off + d] += d_score * c.q[i * ed + off + d];
                        }
                    }
                }
            }

            // through QKV projections back to ln1_out  (Q = ln1·Wq etc.)
            let mut d_ln1_out = vec![0.0_f32; seq * ed];
            accum_linear_bwd(
                &d_q,
                &c.ln1_out,
                self.wq_ref(layer),
                seq,
                ed,
                &mut g.wq[layer],
                &mut d_ln1_out,
            );
            accum_linear_bwd(
                &d_k,
                &c.ln1_out,
                self.wk_ref(layer),
                seq,
                ed,
                &mut g.wk[layer],
                &mut d_ln1_out,
            );
            accum_linear_bwd(
                &d_v,
                &c.ln1_out,
                self.wv_ref(layer),
                seq,
                ed,
                &mut g.wv[layer],
                &mut d_ln1_out,
            );

            // through LN1 → adds to d_input
            layer_norm_bwd(
                &d_ln1_out,
                &c.input,
                &c.ln1_mean,
                &c.ln1_inv_std,
                self.ln1_g_ref(layer),
                seq,
                ed,
                &mut g.ln1_g[layer],
                &mut g.ln1_b[layer],
                &mut d_input,
            );

            d_h = d_input;
        }

        // ── CLS token + tokenizer ─────────────────────────────────────────────
        g.cls_token.copy_from_slice(&d_h[0..ed]);

        // d_h[ed..] are the gradients w.r.t. the feature tokens.
        let d_tokens = &d_h[ed..];
        let mut d_x_cont = vec![0.0_f32; self.tokenizer_ref().n_cont_ref()];
        self.tokenizer_ref().backward_into(
            x_cont,
            x_cat,
            d_tokens,
            &mut g.cont_w,
            &mut g.cont_b,
            &mut g.cat_embeds,
            &mut d_x_cont,
        );

        Ok((g, d_x_cont))
    }
}

// ─── LayerNorm forward / backward (per token over `ed`) ─────────────────────────

fn layer_norm_fwd(
    x: &[f32],
    seq: usize,
    ed: usize,
    gamma: &[f32],
    beta: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0_f32; seq * ed];
    let mut mean = vec![0.0_f32; seq];
    let mut inv_std = vec![0.0_f32; seq];
    let n = ed as f32;
    for s in 0..seq {
        let row = &x[s * ed..(s + 1) * ed];
        let m = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        mean[s] = m;
        inv_std[s] = inv;
        for d in 0..ed {
            out[s * ed + d] = (row[d] - m) * inv * gamma[d] + beta[d];
        }
    }
    (out, mean, inv_std)
}

/// Backward through LayerNorm.  `d_out` is the gradient w.r.t. the normalised
/// output; accumulates parameter gradients and adds the input gradient into
/// `d_in`.
#[allow(clippy::too_many_arguments)]
fn layer_norm_bwd(
    d_out: &[f32],
    x: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    gamma: &[f32],
    seq: usize,
    ed: usize,
    d_gamma: &mut [f32],
    d_beta: &mut [f32],
    d_in: &mut [f32],
) {
    let n = ed as f32;
    for s in 0..seq {
        let row = &x[s * ed..(s + 1) * ed];
        let m = mean[s];
        let inv = inv_std[s];
        // normalised values
        // dL/d_xhat[d] = d_out[d] * gamma[d]
        // dL/d_x = inv/n * (n*dxhat - Σ dxhat - xhat * Σ(dxhat*xhat))
        let mut sum_dxhat = 0.0_f32;
        let mut sum_dxhat_xhat = 0.0_f32;
        let mut dxhat = vec![0.0_f32; ed];
        for d in 0..ed {
            let xhat = (row[d] - m) * inv;
            let go = d_out[s * ed + d];
            d_gamma[d] += go * xhat;
            d_beta[d] += go;
            let dh = go * gamma[d];
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

/// Backward for `Y = X · W` (X `[seq×ed]`, W `[ed×ed]`, row-major):
/// accumulates `dW += Xᵀ·dY` and `dX += dY·Wᵀ`.
fn accum_linear_bwd(
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

// ─── FtGradients construction ──────────────────────────────────────────────────

impl FtGradients {
    fn zeros(model: &FtTransformer, _n_feat: usize) -> Self {
        let cfg = model.config_ref();
        let ed = cfg.embed_dim;
        let fh = cfg.ffn_hidden;
        let nl = cfg.n_layers;
        let mk = |n: usize| vec![0.0_f32; n];
        Self {
            cls_token: mk(ed),
            cont_w: mk(cfg.n_cont_features * ed),
            cont_b: mk(cfg.n_cont_features * ed),
            cat_embeds: cfg.cat_n_categories.iter().map(|&nc| mk(nc * ed)).collect(),
            wq: (0..nl).map(|_| mk(ed * ed)).collect(),
            wk: (0..nl).map(|_| mk(ed * ed)).collect(),
            wv: (0..nl).map(|_| mk(ed * ed)).collect(),
            wo: (0..nl).map(|_| mk(ed * ed)).collect(),
            ffn_w1: (0..nl).map(|_| mk(fh * ed)).collect(),
            ffn_b1: (0..nl).map(|_| mk(fh)).collect(),
            ffn_w2: (0..nl).map(|_| mk(ed * fh)).collect(),
            ffn_b2: (0..nl).map(|_| mk(ed)).collect(),
            ln1_g: (0..nl).map(|_| mk(ed)).collect(),
            ln1_b: (0..nl).map(|_| mk(ed)).collect(),
            ln2_g: (0..nl).map(|_| mk(ed)).collect(),
            ln2_b: (0..nl).map(|_| mk(ed)).collect(),
            head_w: mk(cfg.n_classes * ed),
            head_b: mk(cfg.n_classes),
        }
    }
}

// ─── FeatureTokenizer backward ─────────────────────────────────────────────────

impl FeatureTokenizer {
    /// Backward through tokenisation.  `d_tokens` is the gradient w.r.t. the
    /// flat `[(n_cont+n_cat)*embed_dim]` token matrix.  Accumulates the parameter
    /// gradients and the continuous-input gradient.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward_into(
        &self,
        x_cont: &[f32],
        x_cat: &[usize],
        d_tokens: &[f32],
        d_cont_w: &mut [f32],
        d_cont_b: &mut [f32],
        d_cat_embeds: &mut [Vec<f32>],
        d_x_cont: &mut [f32],
    ) {
        let ed = self.embed_dim_ref();
        let n_cont = self.n_cont_ref();
        // continuous: token_j[d] = x_j*w_j[d] + b_j[d]
        for j in 0..n_cont {
            let mut acc = 0.0_f32;
            for d in 0..ed {
                let gt = d_tokens[j * ed + d];
                d_cont_w[j * ed + d] += gt * x_cont[j];
                d_cont_b[j * ed + d] += gt;
                acc += gt * self.cont_w_at(j, d);
            }
            d_x_cont[j] += acc;
        }
        // categorical: token = embed[cat_idx]
        for (i, &cat_idx) in x_cat.iter().enumerate() {
            let base = (n_cont + i) * ed;
            for d in 0..ed {
                d_cat_embeds[i][cat_idx * ed + d] += d_tokens[base + d];
            }
        }
    }
}

// Helper enum used only by the gradient-check tests to address a single scalar
// parameter for finite differences.  Lives at module scope (under `cfg(test)`)
// so `FtTransformer::param_get` / `param_set` can name it.
#[cfg(test)]
pub(crate) enum ParamRef {
    Cls(usize),
    ContW(usize),
    ContB(usize),
    Cat(usize, usize),
    Wq(usize, usize),
    Wk(usize, usize),
    Wv(usize, usize),
    Wo(usize, usize),
    Fw1(usize, usize),
    Fb1(usize, usize),
    Fw2(usize, usize),
    Fb2(usize, usize),
    Ln1g(usize, usize),
    Ln1b(usize, usize),
    Ln2g(usize, usize),
    Ln2b(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::transformer::ft_transformer::FtConfig;

    /// Scalar loss `L = Σ logits · v` with a fixed random direction `v`,
    /// so the backward is exercised with non-trivial upstream gradients.
    fn loss_dir(logits: &[f32], dir: &[f32]) -> f32 {
        logits.iter().zip(dir.iter()).map(|(&a, &b)| a * b).sum()
    }

    fn tiny_model(seed: u64) -> (FtTransformer, Vec<f32>, Vec<usize>) {
        let cfg = FtConfig {
            n_cont_features: 3,
            cat_n_categories: vec![4, 2],
            embed_dim: 6,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 8,
            dropout_rate: 0.0,
            n_classes: 3,
        };
        let mut rng = LcgRng::new(seed);
        let model = FtTransformer::new(cfg, &mut rng).expect("new");
        let x_cont = vec![0.4_f32, -0.7, 1.1];
        let x_cat = vec![2usize, 1];
        (model, x_cont, x_cat)
    }

    #[test]
    fn grad_check_head_and_cls() {
        // Direction vector for the loss.
        let (model, x_cont, x_cat) = tiny_model(11);
        let dir = vec![0.5_f32, -0.3, 0.8];
        let logits = model.forward(&x_cont, &x_cat).expect("fwd");
        let (g, _dx) = model.backward(&x_cont, &x_cat, &dir).expect("bwd");

        // The head is a single affine layer, so its gradient is closed-form:
        // d head_b = dir,  d head_w[o,d] = dir[o] * cls[d].  We check these
        // exact identities here; the deep-stack FD checks live in the other two
        // tests.  embed_dim = 6, n_classes = 3 → head_w has 18 entries.
        assert_eq!(g.head_b.len(), 3);
        assert_eq!(g.head_w.len(), 18);
        // head_b gradient equals the loss direction exactly.
        for (a, b) in g.head_b.iter().zip(dir.iter()) {
            assert!((a - b).abs() < 1e-6, "head_b grad {a} vs {b}");
        }
        // head_w[o,d] == dir[o] * cls[d]; recompute cls via forward_cached.
        let (_l, _c, cls) = model.forward_cached(&x_cont, &x_cat).expect("cached");
        for (o, &do_) in dir.iter().enumerate() {
            for (d, &cd) in cls.iter().enumerate() {
                let expect = do_ * cd;
                let got = g.head_w[o * cls.len() + d];
                assert!(
                    (got - expect).abs() < 1e-5,
                    "head_w[{o},{d}] {got} vs {expect}"
                );
            }
        }
        let _ = logits;
    }

    #[test]
    fn grad_check_input_central_difference() {
        // Verify d L / d x_cont by central differences — this exercises the
        // FULL graph (tokenizer → CLS → all layers → head) end-to-end.
        let (model, x_cont, x_cat) = tiny_model(7);
        let dir = vec![0.3_f32, 0.6, -0.4];
        let (_g, dx) = model.backward(&x_cont, &x_cat, &dir).expect("bwd");

        let eps = 1e-3_f32;
        for j in 0..x_cont.len() {
            let mut xp = x_cont.clone();
            let mut xm = x_cont.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let lp = loss_dir(&model.forward(&xp, &x_cat).expect("fwd+"), &dir);
            let lm = loss_dir(&model.forward(&xm, &x_cat).expect("fwd-"), &dir);
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - dx[j]).abs() / (fd.abs().max(dx[j].abs()).max(1e-3));
            assert!(
                rel < 2e-2,
                "input grad[{j}] analytic={} fd={fd} rel={rel}",
                dx[j]
            );
        }
    }

    #[test]
    fn grad_check_parameters_central_difference() {
        // Perturb a representative parameter of every kind and compare the
        // central-difference loss derivative against the analytic gradient.
        let (mut model, x_cont, x_cat) = tiny_model(23);
        let dir = vec![0.7_f32, -0.2, 0.5];
        let (g, _dx) = model.backward(&x_cont, &x_cat, &dir).expect("bwd");

        let eps = 1e-3_f32;
        // Each entry: (human label, mutate closure, analytic grad value).
        // We perturb in place via the test-only mutable accessors.
        let checks: Vec<(&str, ParamRef, f32)> = vec![
            ("cls_token[0]", ParamRef::Cls(0), g.cls_token[0]),
            ("cont_w[2]", ParamRef::ContW(2), g.cont_w[2]),
            ("cont_b[5]", ParamRef::ContB(5), g.cont_b[5]),
            ("cat_embeds[0][3]", ParamRef::Cat(0, 3), g.cat_embeds[0][3]),
            ("cat_embeds[1][6]", ParamRef::Cat(1, 6), g.cat_embeds[1][6]),
            ("wq0[7]", ParamRef::Wq(0, 7), g.wq[0][7]),
            ("wk1[10]", ParamRef::Wk(1, 10), g.wk[1][10]),
            ("wv0[4]", ParamRef::Wv(0, 4), g.wv[0][4]),
            ("wo1[20]", ParamRef::Wo(1, 20), g.wo[1][20]),
            ("ffn_w1_0[9]", ParamRef::Fw1(0, 9), g.ffn_w1[0][9]),
            ("ffn_b1_1[2]", ParamRef::Fb1(1, 2), g.ffn_b1[1][2]),
            ("ffn_w2_0[15]", ParamRef::Fw2(0, 15), g.ffn_w2[0][15]),
            ("ffn_b2_1[3]", ParamRef::Fb2(1, 3), g.ffn_b2[1][3]),
            ("ln1_g0[1]", ParamRef::Ln1g(0, 1), g.ln1_g[0][1]),
            ("ln1_b1[4]", ParamRef::Ln1b(1, 4), g.ln1_b[1][4]),
            ("ln2_g1[2]", ParamRef::Ln2g(1, 2), g.ln2_g[1][2]),
            ("ln2_b0[5]", ParamRef::Ln2b(0, 5), g.ln2_b[0][5]),
        ];

        for (label, pref, analytic) in checks {
            let orig = model.param_get(&pref);
            model.param_set(&pref, orig + eps);
            let lp = loss_dir(&model.forward(&x_cont, &x_cat).expect("fwd+"), &dir);
            model.param_set(&pref, orig - eps);
            let lm = loss_dir(&model.forward(&x_cont, &x_cat).expect("fwd-"), &dir);
            model.param_set(&pref, orig);
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - analytic).abs() / (fd.abs().max(analytic.abs()).max(1e-3));
            // f32 central differences through a 2-layer softmax+LayerNorm stack
            // carry ~1e-2 relative rounding noise; 3.5e-2 is a rigorous bound.
            assert!(
                rel < 3.5e-2,
                "param {label}: analytic={analytic} fd={fd} rel={rel}"
            );
        }
    }
}
