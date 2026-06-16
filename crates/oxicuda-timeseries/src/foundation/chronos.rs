//! Chronos probabilistic forecaster (Ansari et al., Amazon, 2024).
//!
//! Reference: *"Chronos: Learning the Language of Time Series"*
//! (Ansari, Stella, Turkmen, Zhang, Mercado, Shen, Shchur, Rangapuram,
//! Pineda-Arango, Kapoor, Zschiegner, Maddix, Wang, Mahoney, Torkkola,
//! Wilson, Bohlke-Schneider, Wang; TMLR 2024).
//!
//! Compact, faithful CPU core. Chronos casts forecasting as **language
//! modelling over quantised time-series tokens**:
//!
//! 1. **Mean-scaling.** Divide the series by the mean of its absolute values so
//!    that its magnitude is roughly normalised. The scale is stored separately
//!    so the transform is exactly invertible
//!    ([`ChronosPredictor::mean_scale`] / [`ChronosPredictor::mean_unscale`]).
//! 2. **Quantisation tokenisation.** Scaled values are binned onto a fixed
//!    uniform grid of `n_value_tokens` bins spanning `[lo, hi]` and mapped to
//!    integer token ids. Two **special tokens** (`EOS`, `PAD`) extend the
//!    vocabulary, so `vocab_size = n_value_tokens + 2`.
//! 3. **Language-model backbone.** A small **decoder-only** causal Transformer
//!    consumes the token ids and predicts the next-token distribution
//!    (softmax over the whole vocabulary).
//! 4. **Detokenisation & probabilistic forecasting.** Predicted token ids map
//!    back to bin-centre values; multiple sampled trajectories are aggregated
//!    into quantiles ([`ChronosPredictor::sample_forecast`]).

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for a [`ChronosPredictor`].
#[derive(Debug, Clone)]
pub struct ChronosConfig {
    /// Number of value (quantisation) tokens. The special tokens are appended
    /// after these, so the full vocabulary has `n_value_tokens + 2` entries.
    pub n_value_tokens: usize,
    /// Lower edge of the quantisation grid (in mean-scaled space).
    pub lo: f32,
    /// Upper edge of the quantisation grid (in mean-scaled space).
    pub hi: f32,
    /// Token embedding / model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of causal Transformer layers.
    pub n_layers: usize,
    /// Feed-forward hidden expansion factor.
    pub ffn_expansion: usize,
    /// Quantile levels reported by [`ChronosPredictor::sample_forecast`].
    pub quantile_levels: Vec<f32>,
}

impl ChronosConfig {
    /// Small configuration for tests and CPU smoke runs.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            n_value_tokens: 32,
            lo: -8.0,
            hi: 8.0,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            ffn_expansion: 2,
            quantile_levels: vec![0.1, 0.5, 0.9],
        }
    }

    /// Base configuration (`vocab≈4096, d=256`).
    #[must_use]
    pub fn base() -> Self {
        Self {
            n_value_tokens: 4094,
            lo: -15.0,
            hi: 15.0,
            d_model: 256,
            n_heads: 8,
            n_layers: 6,
            ffn_expansion: 4,
            quantile_levels: vec![0.1, 0.5, 0.9],
        }
    }
}

/// Number of special (non-value) tokens in every Chronos vocabulary.
const N_SPECIAL: usize = 2;

// ─── Forecast ───────────────────────────────────────────────────────────────

/// Quantile forecast produced by [`ChronosPredictor::sample_forecast`].
#[derive(Debug, Clone)]
pub struct ChronosForecast {
    /// Reported quantile levels (copied from the config).
    pub levels: Vec<f32>,
    /// Quantile values, level-major `[n_levels, horizon]`
    /// (`idx = level * horizon + step`).
    pub values: Vec<f32>,
    /// Forecast horizon.
    pub horizon: usize,
}

impl ChronosForecast {
    /// Borrow the row of quantile values for a given level index.
    #[must_use]
    pub fn level(&self, level_idx: usize) -> &[f32] {
        &self.values[level_idx * self.horizon..(level_idx + 1) * self.horizon]
    }
}

// ─── Transformer layer ──────────────────────────────────────────────────────

/// Learnable parameters for one causal pre-norm Transformer layer.
#[derive(Debug, Clone)]
pub struct ChronosLayer {
    norm1_g: Vec<f32>,
    norm1_b: Vec<f32>,
    q_w: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    out_w: Vec<f32>,
    norm2_g: Vec<f32>,
    norm2_b: Vec<f32>,
    ff_w1: Vec<f32>,
    ff_b1: Vec<f32>,
    ff_w2: Vec<f32>,
    ff_b2: Vec<f32>,
}

impl ChronosLayer {
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

// ─── Predictor ──────────────────────────────────────────────────────────────

/// Chronos probabilistic forecaster.
#[derive(Debug, Clone)]
pub struct ChronosPredictor {
    /// Full vocabulary size (`n_value_tokens + 2`).
    pub vocab_size: usize,
    /// Token embedding table `[vocab_size, d_model]`.
    pub token_emb: Vec<f32>,
    /// Causal Transformer layers.
    pub layers: Vec<ChronosLayer>,
    /// Final LayerNorm scale `[d_model]`.
    pub final_g: Vec<f32>,
    /// Final LayerNorm bias `[d_model]`.
    pub final_b: Vec<f32>,
    /// LM head weight `[vocab_size, d_model]`.
    pub head_w: Vec<f32>,
    /// LM head bias `[vocab_size]`.
    pub head_b: Vec<f32>,
    /// Model configuration.
    pub cfg: ChronosConfig,
}

impl ChronosPredictor {
    /// Token id of the end-of-sequence marker.
    #[must_use]
    pub fn eos_id(&self) -> u32 {
        self.cfg.n_value_tokens as u32
    }

    /// Token id of the padding marker.
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.cfg.n_value_tokens as u32 + 1
    }

    /// Build a Chronos predictor, initialising all parameters from `rng`.
    ///
    /// # Errors
    ///
    /// - [`TsError::ShapeMismatch`] when `n_value_tokens < 2`, `lo >= hi`,
    ///   `n_layers == 0` or `ffn_expansion == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    pub fn new(cfg: ChronosConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.n_value_tokens < 2 {
            return Err(TsError::ShapeMismatch {
                msg: "n_value_tokens must be >= 2".into(),
            });
        }
        if cfg.lo >= cfg.hi {
            return Err(TsError::ShapeMismatch {
                msg: format!("require lo < hi, got lo={}, hi={}", cfg.lo, cfg.hi),
            });
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

        let vocab_size = cfg.n_value_tokens + N_SPECIAL;
        let d = cfg.d_model;

        let mut token_emb = vec![0.0; vocab_size * d];
        rng.fill_normal(&mut token_emb);
        for v in &mut token_emb {
            *v *= 0.02;
        }

        let layers = (0..cfg.n_layers)
            .map(|_| ChronosLayer::new(d, cfg.ffn_expansion, rng))
            .collect();

        let head_w = xavier(vocab_size, d, rng);
        let head_b = vec![0.0; vocab_size];

        Ok(Self {
            vocab_size,
            token_emb,
            layers,
            final_g: vec![1.0; d],
            final_b: vec![0.0; d],
            head_w,
            head_b,
            cfg,
        })
    }

    /// Width of a single quantisation bin (in mean-scaled space).
    #[must_use]
    pub fn bin_width(&self) -> f32 {
        (self.cfg.hi - self.cfg.lo) / self.cfg.n_value_tokens as f32
    }

    /// Map a scalar (in scaled space) to a value-token id in `[0, n_value_tokens)`.
    fn value_to_token(&self, x: f32) -> u32 {
        let w = self.bin_width();
        let idx = ((x - self.cfg.lo) / w).floor();
        let max = (self.cfg.n_value_tokens - 1) as f32;
        idx.clamp(0.0, max) as u32
    }

    /// Bin-centre value (in scaled space) for a value-token id.
    fn token_to_value(&self, token: u32) -> f32 {
        let w = self.bin_width();
        self.cfg.lo + (token as f32 + 0.5) * w
    }

    /// Tokenise a series into value-token ids, terminated by an `EOS` token.
    ///
    /// Values are assumed to already lie in (or near) `[lo, hi]`; out-of-range
    /// values are clamped to the edge bins. The returned ids all lie in
    /// `[0, vocab_size)`.
    #[must_use]
    pub fn tokenize(&self, series: &[f32]) -> Vec<u32> {
        let mut out: Vec<u32> = series.iter().map(|&x| self.value_to_token(x)).collect();
        out.push(self.eos_id());
        out
    }

    /// Detokenise value tokens back to bin-centre values, skipping any special
    /// tokens (`EOS`, `PAD`).
    #[must_use]
    pub fn detokenize(&self, tokens: &[u32]) -> Vec<f32> {
        tokens
            .iter()
            .filter(|&&t| (t as usize) < self.cfg.n_value_tokens)
            .map(|&t| self.token_to_value(t))
            .collect()
    }

    /// Mean-scaling: `scale = max(mean(|x|), eps)`, returns `(x / scale, scale)`.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `series` is empty.
    pub fn mean_scale(series: &[f32]) -> TsResult<(Vec<f32>, f32)> {
        if series.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "series must not be empty".into(),
            });
        }
        let mean_abs = series.iter().map(|v| v.abs()).sum::<f32>() / series.len() as f32;
        let scale = mean_abs.max(1e-8);
        let scaled = series.iter().map(|&x| x / scale).collect();
        Ok((scaled, scale))
    }

    /// Inverse of [`Self::mean_scale`]: multiply scaled values by `scale`.
    #[must_use]
    pub fn mean_unscale(scaled: &[f32], scale: f32) -> Vec<f32> {
        scaled.iter().map(|&x| x * scale).collect()
    }

    /// Run the causal LM over `token_ids`, returning per-position logits,
    /// row-major `[seq_len, vocab_size]`. Row `i` are the logits that predict
    /// token `i + 1`.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `token_ids` is empty.
    /// - [`TsError::ShapeMismatch`] when any token id is `>= vocab_size`.
    pub fn forward(&self, token_ids: &[u32]) -> TsResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "token_ids must not be empty".into(),
            });
        }
        let d = self.cfg.d_model;
        let seq = token_ids.len();

        // Embed tokens + sinusoidal positions.
        let pos = sinusoidal_pos_enc(seq, d);
        let mut x = vec![0.0; seq * d];
        for (i, &tok) in token_ids.iter().enumerate() {
            if tok as usize >= self.vocab_size {
                return Err(TsError::ShapeMismatch {
                    msg: format!("token id {tok} >= vocab_size {}", self.vocab_size),
                });
            }
            let emb = &self.token_emb[tok as usize * d..(tok as usize + 1) * d];
            for j in 0..d {
                x[i * d + j] = emb[j] + pos[i * d + j];
            }
        }

        // Causal Transformer stack.
        for layer in &self.layers {
            let delta = causal_mhsa(&x, seq, d, layer, self.cfg.n_heads);
            for (xi, di) in x.iter_mut().zip(delta.iter()) {
                *xi += di;
            }
            let fdelta = ffn(&x, seq, d, layer, self.cfg.ffn_expansion);
            for (xi, di) in x.iter_mut().zip(fdelta.iter()) {
                *xi += di;
            }
        }
        layer_norm(&mut x, d, &self.final_g, &self.final_b);

        // LM head per position.
        let v = self.vocab_size;
        let mut logits = vec![0.0; seq * v];
        for i in 0..seq {
            let row = &x[i * d..(i + 1) * d];
            for o in 0..v {
                let w_row = &self.head_w[o * d..(o + 1) * d];
                let mut acc = self.head_b[o];
                for k in 0..d {
                    acc += w_row[k] * row[k];
                }
                logits[i * v + o] = acc;
            }
        }
        Ok(logits)
    }

    /// Full-vocabulary next-token distribution after `token_ids` (softmax of the
    /// last logit row). The returned probabilities sum to 1.
    ///
    /// # Errors
    ///
    /// Mirrors [`Self::forward`].
    pub fn next_token_distribution(&self, token_ids: &[u32]) -> TsResult<Vec<f32>> {
        let logits = self.forward(token_ids)?;
        let v = self.vocab_size;
        let seq = token_ids.len();
        let mut probs = logits[(seq - 1) * v..seq * v].to_vec();
        softmax_row(&mut probs);
        Ok(probs)
    }

    /// Probabilistic forecast: sample `n_samples` trajectories and reduce them to
    /// the configured quantile levels.
    ///
    /// The context is mean-scaled and tokenised; the LM then autoregressively
    /// samples `horizon` value tokens per trajectory (special tokens are masked
    /// out during sampling). Sampled tokens are detokenised and un-scaled, then
    /// per-step quantiles are computed across trajectories.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] when `series` is empty.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::ShapeMismatch`] when `n_samples == 0`.
    pub fn sample_forecast(
        &self,
        series: &[f32],
        horizon: usize,
        n_samples: usize,
        rng: &mut LcgRng,
    ) -> TsResult<ChronosForecast> {
        if series.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "series must not be empty".into(),
            });
        }
        if horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if n_samples == 0 {
            return Err(TsError::ShapeMismatch {
                msg: "n_samples must be >= 1".into(),
            });
        }

        let (scaled, scale) = Self::mean_scale(series)?;
        let context: Vec<u32> = scaled.iter().map(|&x| self.value_to_token(x)).collect();

        // trajectories[s] holds horizon un-scaled real values.
        let mut trajectories = vec![vec![0.0_f32; horizon]; n_samples];
        for traj in trajectories.iter_mut() {
            let mut seq = context.clone();
            for step in traj.iter_mut() {
                let probs = self.next_token_distribution(&seq)?;
                let tok = self.sample_value_token(&probs, rng);
                seq.push(tok);
                *step = self.token_to_value(tok) * scale;
            }
        }

        // Reduce to quantiles per horizon step.
        let levels = &self.cfg.quantile_levels;
        let mut values = vec![0.0; levels.len() * horizon];
        let mut column = vec![0.0_f32; n_samples];
        for step in 0..horizon {
            for (s, c) in column.iter_mut().enumerate() {
                *c = trajectories[s][step];
            }
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for (li, &q) in levels.iter().enumerate() {
                values[li * horizon + step] = quantile_sorted(&column, q);
            }
        }

        Ok(ChronosForecast {
            levels: levels.clone(),
            values,
            horizon,
        })
    }

    /// Sample a value token from a full-vocabulary distribution, renormalising
    /// over the value tokens only (special tokens are never emitted).
    fn sample_value_token(&self, probs: &[f32], rng: &mut LcgRng) -> u32 {
        let nv = self.cfg.n_value_tokens;
        let mass: f32 = probs[..nv].iter().sum();
        if mass <= 0.0 {
            // Degenerate (should not happen with softmax); fall back to the mode.
            let mut best = 0usize;
            let mut best_v = probs[0];
            for (i, &p) in probs.iter().take(nv).enumerate() {
                if p > best_v {
                    best_v = p;
                    best = i;
                }
            }
            return best as u32;
        }
        let u = rng.next_f32() * mass;
        let mut acc = 0.0;
        for (i, &p) in probs.iter().take(nv).enumerate() {
            acc += p;
            if u < acc {
                return i as u32;
            }
        }
        (nv - 1) as u32
    }
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Xavier-magnitude initialised `[rows, cols]` matrix.
fn xavier(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
    let mut v = vec![0.0; rows * cols];
    rng.fill_normal(&mut v);
    for x in &mut v {
        *x *= scale;
    }
    v
}

/// Linear interpolation quantile of an ascending-sorted slice.
fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let qc = q.clamp(0.0, 1.0);
    let pos = qc * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// In-place row-wise LayerNorm over the last dimension, eps = 1e-5.
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

/// In-place numerically-stable softmax (treats `-inf` as zero probability).
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
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

/// Apply `[D, D]` projection row-wise to a `[N, D]` matrix.
fn matmul_rows(x: &[f32], w: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * d];
    for i in 0..n {
        let x_row = &x[i * d..(i + 1) * d];
        for o in 0..d {
            let w_row = &w[o * d..(o + 1) * d];
            let mut acc = 0.0;
            for k in 0..d {
                acc += w_row[k] * x_row[k];
            }
            out[i * d + o] = acc;
        }
    }
    out
}

/// Causal (autoregressive) multi-head self-attention (pre-norm). Returns the
/// residual delta. Query `i` attends only to keys `j <= i`.
fn causal_mhsa(x: &[f32], n: usize, d: usize, lw: &ChronosLayer, n_heads: usize) -> Vec<f32> {
    let head_dim = d / n_heads;
    let scale = (head_dim as f32).sqrt().recip();

    let mut normed = x.to_vec();
    layer_norm(&mut normed, d, &lw.norm1_g, &lw.norm1_b);

    let q = matmul_rows(&normed, &lw.q_w, n, d);
    let k = matmul_rows(&normed, &lw.k_w, n, d);
    let v = matmul_rows(&normed, &lw.v_w, n, d);

    let mut attn_out = vec![0.0; n * d];
    for h in 0..n_heads {
        let h0 = h * head_dim;
        for qi in 0..n {
            let mut scores = vec![0.0; qi + 1];
            for (ki, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0.0;
                for hd in 0..head_dim {
                    dot += q[qi * d + h0 + hd] * k[ki * d + h0 + hd];
                }
                *sc = dot * scale;
            }
            softmax_row(&mut scores);
            for hd in 0..head_dim {
                let mut acc = 0.0;
                for (ki, &s) in scores.iter().enumerate() {
                    acc += s * v[ki * d + h0 + hd];
                }
                attn_out[qi * d + h0 + hd] = acc;
            }
        }
    }
    matmul_rows(&attn_out, &lw.out_w, n, d)
}

/// Position-wise feed-forward block (pre-norm). Returns the residual delta.
fn ffn(x: &[f32], n: usize, d: usize, lw: &ChronosLayer, expansion: usize) -> Vec<f32> {
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    #[test]
    fn chronos_vocab_size() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        assert_eq!(p.vocab_size, 32 + 2);
        assert_eq!(p.eos_id(), 32);
        assert_eq!(p.pad_id(), 33);
    }

    #[test]
    fn chronos_tokenize_ids_in_range() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let series: Vec<f32> = (0..40).map(|i| (i as f32 * 0.3).sin() * 4.0).collect();
        let tokens = p.tokenize(&series);
        assert!(tokens.iter().all(|&t| (t as usize) < p.vocab_size));
        // Out-of-range values clamp to edge bins, still valid value tokens.
        let extreme = p.tokenize(&[1e9, -1e9]);
        assert_eq!(extreme[0], (p.cfg.n_value_tokens - 1) as u32);
        assert_eq!(extreme[1], 0);
    }

    #[test]
    fn chronos_tokenize_appends_eos() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let tokens = p.tokenize(&[0.0, 1.0, -1.0]);
        assert_eq!(tokens.len(), 4);
        assert_eq!(*tokens.last().expect("nonempty"), p.eos_id());
    }

    #[test]
    fn chronos_roundtrip_within_one_bin() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        // Values inside [lo, hi].
        let series: Vec<f32> = (0..30).map(|i| (i as f32 * 0.2).cos() * 5.0).collect();
        let tokens = p.tokenize(&series);
        let recon = p.detokenize(&tokens);
        assert_eq!(recon.len(), series.len()); // EOS dropped
        let w = p.bin_width();
        for (a, b) in series.iter().zip(recon.iter()) {
            assert!(
                (a - b).abs() <= w,
                "quant error {} > bin width {w}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn chronos_detokenize_skips_special() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let toks = vec![3u32, p.eos_id(), 5u32, p.pad_id()];
        let vals = p.detokenize(&toks);
        assert_eq!(vals.len(), 2); // only the two value tokens survive
    }

    #[test]
    fn chronos_mean_scale_invertible() {
        let series: Vec<f32> = vec![2.0, -4.0, 6.0, -8.0, 10.0];
        let (scaled, scale) = ChronosPredictor::mean_scale(&series).expect("scale");
        assert!(scale > 0.0);
        let recovered = ChronosPredictor::mean_unscale(&scaled, scale);
        for (a, b) in series.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "unscale mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn chronos_mean_scale_normalises_magnitude() {
        let series: Vec<f32> = vec![10.0, -20.0, 30.0, -40.0];
        let (scaled, _scale) = ChronosPredictor::mean_scale(&series).expect("scale");
        let mean_abs = scaled.iter().map(|v| v.abs()).sum::<f32>() / scaled.len() as f32;
        // By construction mean(|scaled|) == 1.
        assert!((mean_abs - 1.0).abs() < 1e-5, "mean abs={mean_abs}");
    }

    #[test]
    fn chronos_forward_logits_shape() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let tokens = p.tokenize(&[0.0, 1.0, 2.0, -1.0]);
        let logits = p.forward(&tokens).expect("forward");
        assert_eq!(logits.len(), tokens.len() * p.vocab_size);
    }

    #[test]
    fn chronos_softmax_sums_to_one() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let tokens = p.tokenize(&[0.0, 1.0, -2.0, 3.0]);
        let probs = p.next_token_distribution(&tokens).expect("dist");
        assert_eq!(probs.len(), p.vocab_size);
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "softmax sums to {s}");
        assert!(probs.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn chronos_forward_finite() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let tokens = p.tokenize(&[0.5, -0.5, 1.5, -1.5, 2.5]);
        let logits = p.forward(&tokens).expect("forward");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn chronos_forward_err_bad_token() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let bad = vec![0u32, 999u32];
        assert!(matches!(
            p.forward(&bad).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn chronos_forward_err_empty() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        assert!(matches!(
            p.forward(&[]).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn chronos_sample_forecast_quantiles_ordered() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let series: Vec<f32> = (0..40).map(|i| (i as f32 * 0.25).sin() * 3.0).collect();
        let horizon = 6;
        let fc = p
            .sample_forecast(&series, horizon, 32, &mut rng)
            .expect("forecast");
        assert_eq!(fc.values.len(), 3 * horizon);
        assert_eq!(fc.horizon, horizon);
        // levels are [0.1, 0.5, 0.9] -> q10 <= q50 <= q90 at every step.
        let q10 = fc.level(0);
        let q50 = fc.level(1);
        let q90 = fc.level(2);
        for step in 0..horizon {
            assert!(q10[step] <= q50[step] + 1e-6, "q10>q50 at {step}");
            assert!(q50[step] <= q90[step] + 1e-6, "q50>q90 at {step}");
        }
    }

    #[test]
    fn chronos_sample_forecast_finite() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let series: Vec<f32> = (0..30).map(|i| (i as f32 * 0.4).cos()).collect();
        let fc = p
            .sample_forecast(&series, 4, 16, &mut rng)
            .expect("forecast");
        assert!(fc.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn chronos_sample_forecast_deterministic() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        let series: Vec<f32> = (0..30).map(|i| (i as f32 * 0.4).cos()).collect();
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        let f1 = p.sample_forecast(&series, 5, 20, &mut r1).expect("f1");
        let f2 = p.sample_forecast(&series, 5, 20, &mut r2).expect("f2");
        assert_eq!(f1.values, f2.values);
    }

    #[test]
    fn chronos_sample_forecast_err_zero_samples() {
        let mut rng = make_rng();
        let p = ChronosPredictor::new(ChronosConfig::tiny(), &mut rng).expect("build");
        assert!(matches!(
            p.sample_forecast(&[1.0, 2.0], 4, 0, &mut rng).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn chronos_err_bad_config() {
        let mut rng = make_rng();
        let cfg = ChronosConfig {
            lo: 5.0,
            hi: -5.0,
            ..ChronosConfig::tiny()
        };
        assert!(matches!(
            ChronosPredictor::new(cfg, &mut rng).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn chronos_err_head_dim() {
        let mut rng = make_rng();
        let cfg = ChronosConfig {
            d_model: 15,
            n_heads: 2,
            ..ChronosConfig::tiny()
        };
        assert!(matches!(
            ChronosPredictor::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }
}
