//! TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.
//!
//! Reference: "TimeMixer: Decomposable Multiscale Mixing for Time Series
//! Forecasting", Wang et al., ICLR 2024.
//!
//! TimeMixer is a fully-MLP forecasting backbone that combines three ideas:
//!
//! 1. **Series decomposition** — each multivariate input is split into a
//!    `(seasonal, trend)` pair using a centred moving-average decomposition
//!    (reused from `crate::decomp::series_decomp`).
//! 2. **Multi-scale downsampling** — `n_scales` increasingly coarse views of
//!    the past are produced by average-pooling the time axis by a factor of
//!    `downsample_factor`. Each scale is independently decomposed into a
//!    seasonal/trend pair.
//! 3. **Past-Decomposable-Mixing (PDM)** — seasonal information is mixed from
//!    fine to coarse scales while trend information is mixed from coarse to
//!    fine scales using small per-scale MLPs that fuse neighbouring-scale
//!    components.
//! 4. **Future-Multipredictor-Mixing (FMM)** — each scale produces an
//!    independent length-`pred_len` forecast via a linear predictor on the
//!    sum of its mixed seasonal and trend pieces. The per-scale forecasts are
//!    combined by a learned (softmax-normalised) ensemble weighting.
//!
//! All tensors use the crate-wide row-major `[time, d_model]` layout with
//! `d_model` as the innermost axis. Pooling along the time axis skips any
//! final partial group (floor division).

use crate::decomp::{DecompResult, SeriesDecomp};
use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a TimeMixer forecaster.
#[derive(Debug, Clone)]
pub struct TimeMixerConfig {
    /// Input sequence length (time axis).
    pub seq_len: usize,
    /// Prediction horizon (time axis of the forecast).
    pub pred_len: usize,
    /// Channel embedding dimension.
    pub d_model: usize,
    /// Number of downsample scales (>= 1; scale 0 is finest = input itself).
    pub n_scales: usize,
    /// Downsample factor (>= 1; 1 = no downsampling, all scales same length).
    pub downsample_factor: usize,
    /// Moving-average kernel size used by series decomposition (odd recommended).
    pub moving_avg_kernel: usize,
}

impl TimeMixerConfig {
    /// Small configuration: `d_model = 8`, `n_scales = 3`,
    /// `downsample_factor = 2`, `moving_avg_kernel = 5`.
    #[must_use]
    pub fn tiny(seq_len: usize, pred_len: usize) -> Self {
        Self {
            seq_len,
            pred_len,
            d_model: 8,
            n_scales: 3,
            downsample_factor: 2,
            moving_avg_kernel: 5,
        }
    }
}

// ─── Per-position MLP weights ────────────────────────────────────────────────

/// A small per-channel MLP `d_model → d_hidden → d_model` applied position-wise.
#[derive(Debug, Clone)]
struct ChannelMlp {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d_in: usize,
    d_hidden: usize,
    d_out: usize,
}

impl ChannelMlp {
    fn new(d_in: usize, d_hidden: usize, d_out: usize, rng: &mut LcgRng) -> Self {
        let init_mat = |rng: &mut LcgRng, rows: usize, cols: usize| -> Vec<f32> {
            let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };
        Self {
            w1: init_mat(rng, d_hidden, d_in),
            b1: vec![0.0_f32; d_hidden],
            w2: init_mat(rng, d_out, d_hidden),
            b2: vec![0.0_f32; d_out],
            d_in,
            d_hidden,
            d_out,
        }
    }

    /// Apply to a single `d_in`-dimensional vector → `d_out`-dimensional vector.
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        let mut hidden = vec![0.0_f32; self.d_hidden];
        for (h, slot) in hidden.iter_mut().enumerate().take(self.d_hidden) {
            let row = &self.w1[h * self.d_in..(h + 1) * self.d_in];
            let mut acc = self.b1[h];
            for (xv, wv) in x.iter().zip(row.iter()) {
                acc += *xv * *wv;
            }
            // GELU activation (tanh approximation).
            *slot = gelu(acc);
        }
        for (o, slot) in out.iter_mut().enumerate().take(self.d_out) {
            let row = &self.w2[o * self.d_hidden..(o + 1) * self.d_hidden];
            let mut acc = self.b2[o];
            for (hv, wv) in hidden.iter().zip(row.iter()) {
                acc += *hv * *wv;
            }
            *slot = acc;
        }
    }

    /// Apply MLP position-wise over a `[n, d_in]` batch and append the result.
    fn apply_seq(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * self.d_out];
        let mut tmp = vec![0.0_f32; self.d_out];
        for ti in 0..n {
            self.apply(&x[ti * self.d_in..(ti + 1) * self.d_in], &mut tmp);
            out[ti * self.d_out..(ti + 1) * self.d_out].copy_from_slice(&tmp);
        }
        out
    }
}

#[inline]
fn gelu(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    let inner = c * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ─── Linear predictor ────────────────────────────────────────────────────────

/// Per-channel linear predictor `length_in × d → pred_len × d`.
///
/// Acts independently per channel: for channel `c`, the predictor maps the
/// length-`length_in` past signal to a length-`pred_len` future signal via a
/// shared `pred_len × length_in` matrix and bias.
#[derive(Debug, Clone)]
struct LinearPredictor {
    /// Per-prediction-step weight `[pred_len, length_in]` row-major.
    weight: Vec<f32>,
    /// Per-prediction-step bias `[pred_len]`.
    bias: Vec<f32>,
    /// Past length.
    length_in: usize,
    /// Future length.
    pred_len: usize,
    /// Channel dimension (shared mapping per channel).
    d_model: usize,
}

impl LinearPredictor {
    fn new(length_in: usize, pred_len: usize, d_model: usize, rng: &mut LcgRng) -> Self {
        let scale = (1.0_f32 / length_in.max(1) as f32).sqrt();
        let mut weight = vec![0.0_f32; pred_len * length_in];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        Self {
            weight,
            bias: vec![0.0_f32; pred_len],
            length_in,
            pred_len,
            d_model,
        }
    }

    /// `[length_in, d_model]` past → `[pred_len, d_model]` forecast.
    fn forward(&self, past: &[f32]) -> TsResult<Vec<f32>> {
        let n = self.length_in;
        let d = self.d_model;
        let expected = n * d;
        if past.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: past.len(),
            });
        }
        let mut out = vec![0.0_f32; self.pred_len * d];
        for pi in 0..self.pred_len {
            let row = &self.weight[pi * n..(pi + 1) * n];
            for ci in 0..d {
                let mut acc = self.bias[pi];
                for ti in 0..n {
                    acc += row[ti] * past[ti * d + ci];
                }
                out[pi * d + ci] = acc;
            }
        }
        Ok(out)
    }
}

// ─── TimeMixer model ─────────────────────────────────────────────────────────

/// TimeMixer multi-scale decomposable forecaster.
///
/// Decomposes the input into seasonal + trend, builds `n_scales` downsampled
/// views, mixes seasonal information fine→coarse and trend information
/// coarse→fine via per-scale MLPs (PDM), and ensembles per-scale linear
/// forecasts (FMM) into a final `[pred_len, d_model]` prediction.
#[derive(Debug, Clone)]
pub struct TimeMixer {
    /// Series decomposition block (shared across scales).
    decomp: SeriesDecomp,
    /// Per-scale seasonal mixer (fine→coarse): consumes `[seasonal_s, season_finer_pooled]`.
    ///
    /// Index `s` is used at scale `s` (s >= 1).
    season_mixers: Vec<ChannelMlp>,
    /// Per-scale trend mixer (coarse→fine): consumes `[trend_s, trend_coarser_upsampled]`.
    ///
    /// Index `s` is used at scale `s` (s < n_scales - 1).
    trend_mixers: Vec<ChannelMlp>,
    /// Per-scale linear predictor mapping the past at scale `s` to `pred_len`.
    predictors: Vec<LinearPredictor>,
    /// Per-scale ensemble logits (softmax-normalised at forward time).
    ensemble_logits: Vec<f32>,
    /// Cached per-scale past lengths.
    scale_lengths: Vec<usize>,
    /// Model configuration.
    cfg: TimeMixerConfig,
}

impl TimeMixer {
    /// Build a TimeMixer forecaster, initialising all weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`, or when the
    ///   pyramid would have an empty scale.
    /// - [`TsError::InvalidHorizon`] when `pred_len == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidPoolSize`] when `n_scales == 0`.
    /// - [`TsError::InvalidStride`] when `downsample_factor == 0`.
    /// - [`TsError::InvalidKernelSize`] when `moving_avg_kernel == 0`.
    pub fn new(cfg: TimeMixerConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.pred_len == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if cfg.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if cfg.n_scales == 0 {
            return Err(TsError::InvalidPoolSize(0));
        }
        if cfg.downsample_factor == 0 {
            return Err(TsError::InvalidStride(0));
        }
        if cfg.moving_avg_kernel == 0 {
            return Err(TsError::InvalidKernelSize(0));
        }

        // Determine per-scale lengths: floor division by downsample_factor.
        let mut scale_lengths = Vec::with_capacity(cfg.n_scales);
        scale_lengths.push(cfg.seq_len);
        for s in 1..cfg.n_scales {
            let prev = scale_lengths[s - 1];
            let next = if cfg.downsample_factor <= 1 {
                prev
            } else {
                prev / cfg.downsample_factor
            };
            if next == 0 {
                return Err(TsError::InvalidSequenceLength(cfg.seq_len));
            }
            scale_lengths.push(next);
        }

        let decomp = SeriesDecomp::new(cfg.moving_avg_kernel)?;
        let d = cfg.d_model;
        let d_hidden = d.max(4);

        // Season mixers: scale s in 1..n_scales consumes [seasonal_s, finer_pooled_s].
        // We use input dim 2*d_model → d_hidden → d_model.
        let season_mixers: Vec<ChannelMlp> = (0..cfg.n_scales.saturating_sub(1))
            .map(|_| ChannelMlp::new(2 * d, d_hidden, d, rng))
            .collect();
        // Trend mixers: scale s in 0..n_scales-1 consumes [trend_s, coarser_upsampled_s].
        let trend_mixers: Vec<ChannelMlp> = (0..cfg.n_scales.saturating_sub(1))
            .map(|_| ChannelMlp::new(2 * d, d_hidden, d, rng))
            .collect();

        let predictors: Vec<LinearPredictor> = scale_lengths
            .iter()
            .map(|&n_s| LinearPredictor::new(n_s, cfg.pred_len, d, rng))
            .collect();

        // Ensemble logits initialised to zero so the softmax starts at uniform.
        let ensemble_logits = vec![0.0_f32; cfg.n_scales];

        Ok(Self {
            decomp,
            season_mixers,
            trend_mixers,
            predictors,
            ensemble_logits,
            scale_lengths,
            cfg,
        })
    }

    /// Access the model configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &TimeMixerConfig {
        &self.cfg
    }

    /// Per-scale past lengths (length `n_scales`).
    #[must_use]
    #[inline]
    pub fn scale_lengths(&self) -> &[usize] {
        &self.scale_lengths
    }

    /// Downsample-and-decompose the input into `n_scales` `(seasonal, trend)` pairs.
    ///
    /// Scale 0 is the input itself decomposed; scale `s + 1` is obtained by
    /// average-pooling the previous scale's input by `downsample_factor` along
    /// the time axis (final partial group skipped), then re-decomposing.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn multi_scale_decompose(&self, x: &[f32]) -> TsResult<Vec<(Vec<f32>, Vec<f32>)>> {
        let d = self.cfg.d_model;
        let expected = self.cfg.seq_len * d;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // Build the downsampled raw inputs first, then decompose each.
        let mut raw_scales: Vec<Vec<f32>> = Vec::with_capacity(self.cfg.n_scales);
        raw_scales.push(x.to_vec());
        for s in 1..self.cfg.n_scales {
            let prev = &raw_scales[s - 1];
            let prev_len = self.scale_lengths[s - 1];
            let next_len = self.scale_lengths[s];
            let pooled = if self.cfg.downsample_factor <= 1 {
                prev.clone()
            } else {
                avg_pool_time(prev, prev_len, d, self.cfg.downsample_factor, next_len)
            };
            raw_scales.push(pooled);
        }

        let mut out: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(self.cfg.n_scales);
        for (s, raw) in raw_scales.iter().enumerate() {
            let n_s = self.scale_lengths[s];
            let DecompResult { trend, seasonal } = self.decomp.forward(raw, n_s, d)?;
            out.push((seasonal, trend));
        }
        Ok(out)
    }

    /// Past-Decomposable-Mixing (PDM).
    ///
    /// * Seasonal mixing: bottom-up (fine → coarse). The seasonal at scale
    ///   `s >= 1` is updated using its current seasonal and the average-pooled
    ///   seasonal from scale `s - 1` (downsampled to match the scale-`s`
    ///   length).
    /// * Trend mixing: top-down (coarse → fine). The trend at scale `s` (for
    ///   `s < n_scales - 1`) is updated using its current trend and the
    ///   nearest-neighbour-upsampled trend from scale `s + 1`.
    ///
    /// Per-scale shapes are preserved. The mixing maps are small position-wise
    /// MLPs with input `2 * d_model` and output `d_model`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `scales.len() != n_scales` or
    ///   when any per-scale `(seasonal, trend)` pair has the wrong length.
    pub fn pdm(&self, scales: &[(Vec<f32>, Vec<f32>)]) -> TsResult<Vec<(Vec<f32>, Vec<f32>)>> {
        if scales.len() != self.cfg.n_scales {
            return Err(TsError::DimensionMismatch {
                expected: self.cfg.n_scales,
                got: scales.len(),
            });
        }
        let d = self.cfg.d_model;
        for (s, (seasonal, trend)) in scales.iter().enumerate() {
            let expected = self.scale_lengths[s] * d;
            if seasonal.len() != expected {
                return Err(TsError::DimensionMismatch {
                    expected,
                    got: seasonal.len(),
                });
            }
            if trend.len() != expected {
                return Err(TsError::DimensionMismatch {
                    expected,
                    got: trend.len(),
                });
            }
        }

        let mut seasonals: Vec<Vec<f32>> = scales.iter().map(|(s, _)| s.clone()).collect();
        let mut trends: Vec<Vec<f32>> = scales.iter().map(|(_, t)| t.clone()).collect();

        // ── Seasonal: fine → coarse mixing. ──────────────────────────────────
        for s in 1..self.cfg.n_scales {
            let n_s = self.scale_lengths[s];
            let n_prev = self.scale_lengths[s - 1];
            // Downsample previous-scale seasonal to length n_s.
            let prev_pooled = if self.cfg.downsample_factor <= 1 {
                // Same length → identity copy already correct.
                seasonals[s - 1].clone()
            } else if n_prev == n_s {
                seasonals[s - 1].clone()
            } else {
                avg_pool_time(
                    &seasonals[s - 1],
                    n_prev,
                    d,
                    self.cfg.downsample_factor,
                    n_s,
                )
            };
            // Concatenate per-position and apply the season mixer at this scale.
            let mut concat = vec![0.0_f32; n_s * 2 * d];
            for ti in 0..n_s {
                concat[ti * 2 * d..ti * 2 * d + d]
                    .copy_from_slice(&seasonals[s][ti * d..(ti + 1) * d]);
                concat[ti * 2 * d + d..(ti + 1) * 2 * d]
                    .copy_from_slice(&prev_pooled[ti * d..(ti + 1) * d]);
            }
            let mixed = self.season_mixers[s - 1].apply_seq(&concat, n_s);
            // Residual add so the mixer learns deltas.
            for i in 0..n_s * d {
                seasonals[s][i] += mixed[i];
            }
        }

        // ── Trend: coarse → fine mixing. ─────────────────────────────────────
        for s in (0..self.cfg.n_scales - 1).rev() {
            let n_s = self.scale_lengths[s];
            let n_next = self.scale_lengths[s + 1];
            // Upsample next-scale trend to length n_s via nearest-neighbour repeat.
            let next_up = if n_next == n_s {
                trends[s + 1].clone()
            } else {
                nearest_upsample_time(&trends[s + 1], n_next, d, n_s)
            };
            let mut concat = vec![0.0_f32; n_s * 2 * d];
            for ti in 0..n_s {
                concat[ti * 2 * d..ti * 2 * d + d]
                    .copy_from_slice(&trends[s][ti * d..(ti + 1) * d]);
                concat[ti * 2 * d + d..(ti + 1) * 2 * d]
                    .copy_from_slice(&next_up[ti * d..(ti + 1) * d]);
            }
            let mixed = self.trend_mixers[s].apply_seq(&concat, n_s);
            for i in 0..n_s * d {
                trends[s][i] += mixed[i];
            }
        }

        let out: Vec<(Vec<f32>, Vec<f32>)> = seasonals.into_iter().zip(trends).collect();
        Ok(out)
    }

    /// Future-Multipredictor-Mixing (FMM).
    ///
    /// For each scale `s`, sum the per-scale seasonal and trend components
    /// (the mixed PDM outputs) and apply a linear predictor `length_s →
    /// pred_len` to produce a per-scale forecast `[pred_len, d_model]`. The
    /// per-scale forecasts are then combined into a single forecast via a
    /// softmax over the learned `ensemble_logits`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when shapes are inconsistent with
    ///   the configuration.
    pub fn fmm(&self, mixed_scales: &[(Vec<f32>, Vec<f32>)]) -> TsResult<Vec<f32>> {
        if mixed_scales.len() != self.cfg.n_scales {
            return Err(TsError::DimensionMismatch {
                expected: self.cfg.n_scales,
                got: mixed_scales.len(),
            });
        }
        let d = self.cfg.d_model;
        for (s, (seasonal, trend)) in mixed_scales.iter().enumerate() {
            let expected = self.scale_lengths[s] * d;
            if seasonal.len() != expected {
                return Err(TsError::DimensionMismatch {
                    expected,
                    got: seasonal.len(),
                });
            }
            if trend.len() != expected {
                return Err(TsError::DimensionMismatch {
                    expected,
                    got: trend.len(),
                });
            }
        }

        // Per-scale forecast and softmax ensemble weights.
        let mut weights = self.ensemble_logits.clone();
        softmax_row(&mut weights);

        let mut combined = vec![0.0_f32; self.cfg.pred_len * d];
        for (s, (seasonal, trend)) in mixed_scales.iter().enumerate() {
            // Sum seasonal + trend.
            let n_s = self.scale_lengths[s];
            let mut past = vec![0.0_f32; n_s * d];
            for i in 0..n_s * d {
                past[i] = seasonal[i] + trend[i];
            }
            let fc = self.predictors[s].forward(&past)?;
            let w_s = weights[s];
            for (c, f) in combined.iter_mut().zip(fc.iter()) {
                *c += w_s * *f;
            }
        }
        Ok(combined)
    }

    /// Full forward pass: `[seq_len, d_model]` → `[pred_len, d_model]`.
    ///
    /// Decomposes, mixes, and ensembles per-scale forecasts.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let scales = self.multi_scale_decompose(x)?;
        let mixed = self.pdm(&scales)?;
        self.fmm(&mixed)
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Average-pool the time axis of a `[t, d]` tensor by `factor`, producing a
/// `[out_len, d]` tensor (final partial group skipped).
fn avg_pool_time(x: &[f32], t: usize, d: usize, factor: usize, out_len: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_len * d];
    if factor == 0 {
        return out;
    }
    let inv = 1.0_f32 / factor as f32;
    for oi in 0..out_len {
        let base = oi * factor;
        for di in 0..d {
            let mut acc = 0.0_f32;
            for ki in 0..factor {
                let src = (base + ki).min(t.saturating_sub(1));
                acc += x[src * d + di];
            }
            out[oi * d + di] = acc * inv;
        }
    }
    out
}

/// Nearest-neighbour upsample the time axis of a `[t_in, d]` tensor to
/// length `t_out` via integer-rounded source indexing.
fn nearest_upsample_time(x: &[f32], t_in: usize, d: usize, t_out: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; t_out * d];
    if t_in == 0 || t_out == 0 {
        return out;
    }
    for oi in 0..t_out {
        // Map output index oi back to input by integer division.
        let src = (oi * t_in / t_out).min(t_in - 1);
        for di in 0..d {
            out[oi * d + di] = x[src * d + di];
        }
    }
    out
}

/// Numerically stable in-place softmax over a row.
fn softmax_row(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2025)
    }

    fn tiny(seq_len: usize, pred_len: usize) -> TimeMixerConfig {
        TimeMixerConfig {
            seq_len,
            pred_len,
            d_model: 4,
            n_scales: 3,
            downsample_factor: 2,
            moving_avg_kernel: 5,
        }
    }

    // 1. multi_scale_decompose returns n_scales (seasonal, trend) pairs.
    #[test]
    fn decompose_n_scales_pairs() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.3_f32; cfg.seq_len * cfg.d_model];
        let scales = model.multi_scale_decompose(&x).expect("decompose");
        assert_eq!(scales.len(), cfg.n_scales);
    }

    // 2. Each scale_s length equals floor(seq_len / df^s) * d_model.
    #[test]
    fn decompose_scale_lengths() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.0_f32; cfg.seq_len * cfg.d_model];
        let scales = model.multi_scale_decompose(&x).expect("decompose");
        let expected_lengths: [usize; 3] = [16, 8, 4];
        for (s, (seasonal, trend)) in scales.iter().enumerate() {
            let expected = expected_lengths[s] * cfg.d_model;
            assert_eq!(seasonal.len(), expected);
            assert_eq!(trend.len(), expected);
        }
    }

    // 3. seasonal + trend reconstructs the (downsampled) input at every scale.
    #[test]
    fn decompose_reconstructs_input() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x: Vec<f32> = (0..cfg.seq_len * cfg.d_model)
            .map(|i| (i as f32 * 0.17).sin() + (i as f32) * 0.005)
            .collect();
        let scales = model.multi_scale_decompose(&x).expect("decompose");
        // For scale 0 the reconstruction must equal the input exactly.
        for (i, &xi) in x.iter().enumerate() {
            let recon = scales[0].0[i] + scales[0].1[i];
            assert!((xi - recon).abs() < 1e-4, "idx={i}: x={xi} recon={recon}");
        }
        // For coarser scales reconstruction equals the pooled input at that scale.
        let d = cfg.d_model;
        let pooled = avg_pool_time(
            &x,
            cfg.seq_len,
            d,
            cfg.downsample_factor,
            cfg.seq_len / cfg.downsample_factor,
        );
        for (i, &pv) in pooled.iter().enumerate() {
            let recon = scales[1].0[i] + scales[1].1[i];
            assert!(
                (pv - recon).abs() < 1e-4,
                "idx={i}: pooled={pv} recon={recon}"
            );
        }
    }

    // 4. pdm preserves per-scale shapes.
    #[test]
    fn pdm_preserves_shapes() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.4_f32; cfg.seq_len * cfg.d_model];
        let scales = model.multi_scale_decompose(&x).expect("decompose");
        let mixed = model.pdm(&scales).expect("pdm");
        for (a, b) in mixed.iter().zip(scales.iter()) {
            assert_eq!(a.0.len(), b.0.len());
            assert_eq!(a.1.len(), b.1.len());
        }
    }

    // 5. fmm output length equals pred_len * d_model.
    #[test]
    fn fmm_output_length() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.3_f32; cfg.seq_len * cfg.d_model];
        let scales = model.multi_scale_decompose(&x).expect("decompose");
        let mixed = model.pdm(&scales).expect("pdm");
        let out = model.fmm(&mixed).expect("fmm");
        assert_eq!(out.len(), cfg.pred_len * cfg.d_model);
    }

    // 6. forward output length equals pred_len * d_model.
    #[test]
    fn forward_output_length() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.2_f32; cfg.seq_len * cfg.d_model];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.pred_len * cfg.d_model);
    }

    // 7. Deterministic given the same seed.
    #[test]
    fn deterministic_given_seed() {
        let cfg = tiny(16, 4);
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let model_a = TimeMixer::new(cfg.clone(), &mut rng_a).expect("build");
        let model_b = TimeMixer::new(cfg.clone(), &mut rng_b).expect("build");
        let x: Vec<f32> = (0..cfg.seq_len * cfg.d_model)
            .map(|i| (i as f32 * 0.09).cos())
            .collect();
        let out_a = model_a.forward(&x).expect("forward");
        let out_b = model_b.forward(&x).expect("forward");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-5, "non-deterministic: {a} vs {b}");
        }
    }

    // 8. n_scales = 1 (single-scale) works.
    #[test]
    fn single_scale_forecast() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            n_scales: 1,
            ..tiny(16, 4)
        };
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        assert_eq!(model.scale_lengths(), &[16]);
        let x = vec![0.5_f32; cfg.seq_len * cfg.d_model];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.pred_len * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 9. downsample_factor = 1 (no downsampling) yields all-same-length scales.
    #[test]
    fn no_downsampling_works() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            downsample_factor: 1,
            ..tiny(16, 4)
        };
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        for &n in model.scale_lengths() {
            assert_eq!(n, cfg.seq_len);
        }
        let x = vec![0.5_f32; cfg.seq_len * cfg.d_model];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.pred_len * cfg.d_model);
    }

    // 10. Constant input → near-zero seasonal at every scale.
    #[test]
    fn constant_input_zero_seasonal() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![2.5_f32; cfg.seq_len * cfg.d_model];
        let scales = model.multi_scale_decompose(&x).expect("decompose");
        for (seasonal, _trend) in &scales {
            for &v in seasonal {
                assert!(v.abs() < 1e-4, "seasonal not ~0 for constant: {v}");
            }
        }
    }

    // 11. err: seq_len == 0.
    #[test]
    fn err_seq_len_zero() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            seq_len: 0,
            ..tiny(16, 4)
        };
        assert!(matches!(
            TimeMixer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    // 12. err: pred_len == 0.
    #[test]
    fn err_pred_len_zero() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            pred_len: 0,
            ..tiny(16, 4)
        };
        assert!(matches!(
            TimeMixer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    // 13. err: d_model == 0.
    #[test]
    fn err_d_model_zero() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            d_model: 0,
            ..tiny(16, 4)
        };
        assert!(matches!(
            TimeMixer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 14. err: n_scales == 0.
    #[test]
    fn err_n_scales_zero() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            n_scales: 0,
            ..tiny(16, 4)
        };
        assert!(matches!(
            TimeMixer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidPoolSize(0)
        ));
    }

    // 15. err: downsample_factor == 0.
    #[test]
    fn err_downsample_factor_zero() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            downsample_factor: 0,
            ..tiny(16, 4)
        };
        assert!(matches!(
            TimeMixer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidStride(0)
        ));
    }

    // 16. err: moving_avg_kernel == 0.
    #[test]
    fn err_kernel_zero() {
        let mut rng = make_rng();
        let cfg = TimeMixerConfig {
            moving_avg_kernel: 0,
            ..tiny(16, 4)
        };
        assert!(matches!(
            TimeMixer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }

    // 17. err: x wrong length for multi_scale_decompose and forward.
    #[test]
    fn err_wrong_input_length() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg, &mut rng).expect("build");
        let bad = vec![0.0_f32; 13];
        assert!(matches!(
            model.multi_scale_decompose(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            model.forward(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 18. err: pdm input n_scales must match config.
    #[test]
    fn err_pdm_wrong_scale_count() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg, &mut rng).expect("build");
        let scales: Vec<(Vec<f32>, Vec<f32>)> = vec![(vec![0.0_f32; 0], vec![0.0_f32; 0])];
        assert!(matches!(
            model.pdm(&scales).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 19. err: fmm input n_scales must match config.
    #[test]
    fn err_fmm_wrong_scale_count() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg, &mut rng).expect("build");
        let scales: Vec<(Vec<f32>, Vec<f32>)> = vec![];
        assert!(matches!(
            model.fmm(&scales).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 20. Output is finite for random Gaussian input.
    #[test]
    fn forward_finite_random_input() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.seq_len * cfg.d_model];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite forward output"
        );
    }

    // 21. err: pdm with wrong per-scale length errors out.
    #[test]
    fn err_pdm_wrong_scale_length() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let scales: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_scales)
            .map(|_| (vec![0.0_f32; 1], vec![0.0_f32; 1]))
            .collect();
        assert!(matches!(
            model.pdm(&scales).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 22. err: fmm with wrong per-scale length errors out.
    #[test]
    fn err_fmm_wrong_scale_length() {
        let mut rng = make_rng();
        let cfg = tiny(16, 4);
        let model = TimeMixer::new(cfg.clone(), &mut rng).expect("build");
        let scales: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_scales)
            .map(|_| (vec![0.0_f32; 1], vec![0.0_f32; 1]))
            .collect();
        assert!(matches!(
            model.fmm(&scales).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }
}
