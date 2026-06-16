//! FMLP-Rec — Filter-enhanced MLP sequential recommender that mixes tokens
//! in the **frequency domain**.
//!
//! Reference: Kun Zhou, Hui Yu, Wayne Xin Zhao, Ji-Rong Wen, "Filter-enhanced
//! MLP is All You Need for Sequential Recommendation", WWW 2022.
//!
//! The model replaces self-attention with a sequence of *learnable* complex
//! filters multiplied in the frequency domain:
//!
//! ```text
//! Z = FFT(X)
//! Y = IFFT( Z ⊙ W_filter )          // (mixing across the sequence axis)
//! X' = LayerNorm( X + Y )
//! X' ' = LayerNorm( X' + FFN(X') )
//! ```
//!
//! In words: cast each feature column to a complex spectrum via a radix-2
//! Cooley-Tukey FFT, multiply it by a learnable complex filter, inverse-FFT
//! it back to the time domain, add a residual, normalise, then run a
//! position-wise feed-forward + LayerNorm.
//!
//! Implementation notes
//! --------------------
//! * The FFT is implemented inline (no dependency on `oxicuda-hdc`) following
//!   the bit-reversal + butterfly + rolling-twiddle convention used by
//!   `oxicuda-hdc::vector::hrr_fft`.
//! * The learnable filter is parameterised by *all* `max_seq_len` complex
//!   coefficients (the redundant negative-frequency half is updated freely
//!   too — the resulting iFFT output has a small imaginary residue that we
//!   drop, mirroring the behaviour reported by the authors when the model
//!   trains).
//! * Filters initialise to `(1, 0)` (the identity) so an untrained `FmlpRec`
//!   reduces to residual-LayerNorm + position-wise FFN, easing optimisation.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Approximate Gaussian Error Linear Unit (the GELU tanh variant used by
/// transformer-style models).
fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
}

/// Whether `n >= 1` is an exact power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// In-place radix-2 Cooley-Tukey FFT on split real/imaginary buffers of
/// equal power-of-two length. If `inverse == true`, performs the inverse
/// FFT with the standard `1/n` scaling.
fn fft_inplace(re: &mut [f32], im: &mut [f32], inverse: bool) -> RecsysResult<()> {
    if re.is_empty() || im.is_empty() {
        return Err(RecsysError::EmptyInput);
    }
    if re.len() != im.len() {
        return Err(RecsysError::DimensionMismatch {
            expected: re.len(),
            got: im.len(),
        });
    }
    let n = re.len();
    if !is_power_of_two(n) {
        return Err(RecsysError::InvalidConfig {
            msg: format!("fft length {n} is not a power of two"),
        });
    }

    // Bit-reversal permutation.
    let mut j = 0_usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let mut m = 2_usize;
    while m <= n {
        let half = m >> 1;
        let angle = sign * std::f64::consts::TAU / m as f64;
        let wm_re = angle.cos() as f32;
        let wm_im = angle.sin() as f32;
        let mut k = 0_usize;
        while k < n {
            let mut w_re = 1.0_f32;
            let mut w_im = 0.0_f32;
            for jj in 0..half {
                let idx_lo = k + jj;
                let idx_hi = k + jj + half;
                let re_hi = re.get(idx_hi).copied().unwrap_or(0.0);
                let im_hi = im.get(idx_hi).copied().unwrap_or(0.0);
                let t_re = w_re * re_hi - w_im * im_hi;
                let t_im = w_re * im_hi + w_im * re_hi;
                let u_re = re.get(idx_lo).copied().unwrap_or(0.0);
                let u_im = im.get(idx_lo).copied().unwrap_or(0.0);
                if let Some(slot) = re.get_mut(idx_lo) {
                    *slot = u_re + t_re;
                }
                if let Some(slot) = im.get_mut(idx_lo) {
                    *slot = u_im + t_im;
                }
                if let Some(slot) = re.get_mut(idx_hi) {
                    *slot = u_re - t_re;
                }
                if let Some(slot) = im.get_mut(idx_hi) {
                    *slot = u_im - t_im;
                }
                let new_w_re = w_re * wm_re - w_im * wm_im;
                let new_w_im = w_re * wm_im + w_im * wm_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
            k += m;
        }
        m <<= 1;
    }

    if inverse {
        let inv = 1.0_f32 / n as f32;
        for x in re.iter_mut() {
            *x *= inv;
        }
        for x in im.iter_mut() {
            *x *= inv;
        }
    }

    Ok(())
}

/// Numerically stable LayerNorm over a feature vector. Mutates `out` to
/// `gamma · normalize(x) + beta` where `normalize` zeroes the mean and
/// unit-variances (variance computed with the population estimator).
fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    if d == 0 {
        return Vec::new();
    }
    let n_inv = 1.0_f32 / d as f32;
    let mean: f32 = x.iter().copied().sum::<f32>() * n_inv;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * n_inv;
    let inv_std = 1.0_f32 / (var + eps).sqrt();
    let mut out = vec![0.0_f32; d];
    for k in 0..d {
        let xv = x.get(k).copied().unwrap_or(0.0);
        let g = gamma.get(k).copied().unwrap_or(1.0);
        let b = beta.get(k).copied().unwrap_or(0.0);
        if let Some(slot) = out.get_mut(k) {
            *slot = (xv - mean) * inv_std * g + b;
        }
    }
    out
}

/// One filter-MLP layer: complex filter + residual LayerNorm + position-wise
/// FFN + residual LayerNorm.
pub struct FilterMlpLayer {
    /// Filter real part, shape `max_seq_len × embed_dim`, row-major over the
    /// frequency index.
    pub filter_real: Vec<f32>,
    /// Filter imaginary part, same shape as `filter_real`.
    pub filter_imag: Vec<f32>,
    /// LayerNorm-1 gain, length `embed_dim`.
    pub ln1_gamma: Vec<f32>,
    /// LayerNorm-1 bias, length `embed_dim`.
    pub ln1_beta: Vec<f32>,
    /// FFN first projection weight, shape `hidden_dim × embed_dim` (row-major).
    pub w_ffn1: Vec<f32>,
    /// FFN first projection bias, length `hidden_dim`.
    pub b_ffn1: Vec<f32>,
    /// FFN second projection weight, shape `embed_dim × hidden_dim` (row-major).
    pub w_ffn2: Vec<f32>,
    /// FFN second projection bias, length `embed_dim`.
    pub b_ffn2: Vec<f32>,
    /// LayerNorm-2 gain, length `embed_dim`.
    pub ln2_gamma: Vec<f32>,
    /// LayerNorm-2 bias, length `embed_dim`.
    pub ln2_beta: Vec<f32>,
}

impl FilterMlpLayer {
    fn new(max_seq_len: usize, embed_dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> Self {
        let init = 0.02_f32;

        // Filters initialise to identity (real=1, imag=0).
        let filter_real = vec![1.0_f32; max_seq_len * embed_dim];
        let filter_imag = vec![0.0_f32; max_seq_len * embed_dim];

        let ln1_gamma = vec![1.0_f32; embed_dim];
        let ln1_beta = vec![0.0_f32; embed_dim];
        let ln2_gamma = vec![1.0_f32; embed_dim];
        let ln2_beta = vec![0.0_f32; embed_dim];

        let mut w_ffn1 = vec![0.0_f32; hidden_dim * embed_dim];
        rng.fill_normal(&mut w_ffn1);
        for v in w_ffn1.iter_mut() {
            *v *= init;
        }
        let b_ffn1 = vec![0.0_f32; hidden_dim];

        let mut w_ffn2 = vec![0.0_f32; embed_dim * hidden_dim];
        rng.fill_normal(&mut w_ffn2);
        for v in w_ffn2.iter_mut() {
            *v *= init;
        }
        let b_ffn2 = vec![0.0_f32; embed_dim];

        Self {
            filter_real,
            filter_imag,
            ln1_gamma,
            ln1_beta,
            w_ffn1,
            b_ffn1,
            w_ffn2,
            b_ffn2,
            ln2_gamma,
            ln2_beta,
        }
    }
}

/// FMLP-Rec configuration.
#[derive(Debug, Clone)]
pub struct FmlpRecConfig {
    /// Embedding dimension (`>= 1`).
    pub embed_dim: usize,
    /// Maximum sequence length — **must** be a power of two so the radix-2
    /// FFT applies.
    pub max_seq_len: usize,
    /// Number of stacked `FilterMlpLayer`s (`>= 1`).
    pub n_layers: usize,
    /// Hidden width of the position-wise feed-forward (`>= 1`).
    pub hidden_dim: usize,
    /// Number of distinct items in the catalogue (`>= 1`).
    pub n_items: usize,
}

/// Filter-enhanced MLP sequential recommender.
pub struct FmlpRec {
    /// Configuration the model was built from.
    pub cfg: FmlpRecConfig,
    /// Item-embedding table, shape `n_items × embed_dim` (row-major).
    pub item_embeds: Vec<f32>,
    /// Position embeddings, shape `max_seq_len × embed_dim` (row-major).
    pub pos_embeds: Vec<f32>,
    /// Stacked filter-MLP layers.
    pub layers: Vec<FilterMlpLayer>,
}

impl FmlpRec {
    /// Construct a FMLP-Rec with Transformer-style initialisation (`σ = 0.02`).
    /// Filters initialise to `(1, 0)` so the network is identity at
    /// initialisation modulo LayerNorm + (small) FFN contributions.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidEmbeddingDim`] when `embed_dim == 0`.
    /// - [`RecsysError::InvalidNumItems`] when `n_items == 0`.
    /// - [`RecsysError::InvalidConfig`] when `max_seq_len == 0`,
    ///   `max_seq_len` is not a power of two, `n_layers == 0`, or
    ///   `hidden_dim == 0`.
    pub fn new(cfg: FmlpRecConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: 0 });
        }
        if cfg.max_seq_len == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "max_seq_len must be >= 1".to_string(),
            });
        }
        if !is_power_of_two(cfg.max_seq_len) {
            return Err(RecsysError::InvalidConfig {
                msg: format!("max_seq_len {} must be a power of two", cfg.max_seq_len),
            });
        }
        if cfg.n_layers == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_layers must be >= 1".to_string(),
            });
        }
        if cfg.hidden_dim == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "hidden_dim must be >= 1".to_string(),
            });
        }

        let init = 0.02_f32;

        let mut item_embeds = vec![0.0_f32; cfg.n_items * cfg.embed_dim];
        rng.fill_normal(&mut item_embeds);
        for v in item_embeds.iter_mut() {
            *v *= init;
        }

        let mut pos_embeds = vec![0.0_f32; cfg.max_seq_len * cfg.embed_dim];
        rng.fill_normal(&mut pos_embeds);
        for v in pos_embeds.iter_mut() {
            *v *= init;
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(FilterMlpLayer::new(
                cfg.max_seq_len,
                cfg.embed_dim,
                cfg.hidden_dim,
                rng,
            ));
        }

        Ok(Self {
            cfg,
            item_embeds,
            pos_embeds,
            layers,
        })
    }

    /// Look up the embedding row for item `id` (length `embed_dim`).
    fn lookup_item(&self, id: usize) -> RecsysResult<&[f32]> {
        let d = self.cfg.embed_dim;
        let start = id.checked_mul(d).ok_or_else(|| RecsysError::Internal {
            msg: "item offset overflow".to_string(),
        })?;
        let end = start.checked_add(d).ok_or_else(|| RecsysError::Internal {
            msg: "item offset overflow".to_string(),
        })?;
        self.item_embeds
            .get(start..end)
            .ok_or(RecsysError::ItemOutOfBounds {
                idx: id,
                n: self.cfg.n_items,
            })
    }

    /// Apply one filter-MLP layer to `x` (shape `[seq_len][embed_dim]` packed
    /// row-major) and return the result with the same shape. `seq_len` must
    /// equal `max_seq_len`.
    fn apply_layer(&self, x: &[f32], layer: &FilterMlpLayer) -> RecsysResult<Vec<f32>> {
        let seq_len = self.cfg.max_seq_len;
        let d = self.cfg.embed_dim;
        if x.len() != seq_len * d {
            return Err(RecsysError::DimensionMismatch {
                expected: seq_len * d,
                got: x.len(),
            });
        }

        // -- frequency-domain mixing per feature column --
        let mut filtered = vec![0.0_f32; seq_len * d];
        for feat in 0..d {
            // Gather the per-feature column.
            let mut re = vec![0.0_f32; seq_len];
            let mut im = vec![0.0_f32; seq_len];
            for t in 0..seq_len {
                let val = x.get(t * d + feat).copied().unwrap_or(0.0);
                if let Some(slot) = re.get_mut(t) {
                    *slot = val;
                }
            }

            // Forward FFT.
            fft_inplace(&mut re, &mut im, false)?;

            // Multiply by complex filter (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
            for t in 0..seq_len {
                let a = re.get(t).copied().unwrap_or(0.0);
                let b = im.get(t).copied().unwrap_or(0.0);
                let c = layer.filter_real.get(t * d + feat).copied().unwrap_or(0.0);
                let dd = layer.filter_imag.get(t * d + feat).copied().unwrap_or(0.0);
                if let Some(slot) = re.get_mut(t) {
                    *slot = a * c - b * dd;
                }
                if let Some(slot) = im.get_mut(t) {
                    *slot = a * dd + b * c;
                }
            }

            // Inverse FFT — re[..] now carries the time-domain real signal.
            fft_inplace(&mut re, &mut im, true)?;

            // Scatter back, dropping tiny imaginary residuals.
            for t in 0..seq_len {
                let val = re.get(t).copied().unwrap_or(0.0);
                if let Some(slot) = filtered.get_mut(t * d + feat) {
                    *slot = val;
                }
            }
        }

        // -- residual + LayerNorm 1 (per position) --
        let mut h1 = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            let mut residual = vec![0.0_f32; d];
            for k in 0..d {
                let xv = x.get(pos * d + k).copied().unwrap_or(0.0);
                let fv = filtered.get(pos * d + k).copied().unwrap_or(0.0);
                if let Some(slot) = residual.get_mut(k) {
                    *slot = xv + fv;
                }
            }
            let normed = layer_norm(&residual, &layer.ln1_gamma, &layer.ln1_beta, 1e-5);
            for k in 0..d {
                if let Some(slot) = h1.get_mut(pos * d + k) {
                    *slot = normed.get(k).copied().unwrap_or(0.0);
                }
            }
        }

        // -- position-wise FFN --
        let hidden = self.cfg.hidden_dim;
        let mut h_ffn = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            // FFN1: hidden_dim × embed_dim · z.
            let mut mid = vec![0.0_f32; hidden];
            for o in 0..hidden {
                let row_start = o * d;
                let row_end = row_start + d;
                let row =
                    layer
                        .w_ffn1
                        .get(row_start..row_end)
                        .ok_or_else(|| RecsysError::Internal {
                            msg: "w_ffn1 row OOB".to_string(),
                        })?;
                let mut acc = layer.b_ffn1.get(o).copied().unwrap_or(0.0);
                for k in 0..d {
                    let w = row.get(k).copied().unwrap_or(0.0);
                    let z = h1.get(pos * d + k).copied().unwrap_or(0.0);
                    acc += w * z;
                }
                if let Some(slot) = mid.get_mut(o) {
                    *slot = gelu(acc);
                }
            }
            // FFN2: embed_dim × hidden_dim · mid.
            let mut out = vec![0.0_f32; d];
            for o in 0..d {
                let row_start = o * hidden;
                let row_end = row_start + hidden;
                let row =
                    layer
                        .w_ffn2
                        .get(row_start..row_end)
                        .ok_or_else(|| RecsysError::Internal {
                            msg: "w_ffn2 row OOB".to_string(),
                        })?;
                let mut acc = layer.b_ffn2.get(o).copied().unwrap_or(0.0);
                for k in 0..hidden {
                    let w = row.get(k).copied().unwrap_or(0.0);
                    let m = mid.get(k).copied().unwrap_or(0.0);
                    acc += w * m;
                }
                if let Some(slot) = out.get_mut(o) {
                    *slot = acc;
                }
            }
            // Residual h1 + FFN.
            let mut residual = vec![0.0_f32; d];
            for k in 0..d {
                let z = h1.get(pos * d + k).copied().unwrap_or(0.0);
                let f = out.get(k).copied().unwrap_or(0.0);
                if let Some(slot) = residual.get_mut(k) {
                    *slot = z + f;
                }
            }
            let normed = layer_norm(&residual, &layer.ln2_gamma, &layer.ln2_beta, 1e-5);
            for k in 0..d {
                if let Some(slot) = h_ffn.get_mut(pos * d + k) {
                    *slot = normed.get(k).copied().unwrap_or(0.0);
                }
            }
        }

        Ok(h_ffn)
    }

    /// Encode a session (already padded to `max_seq_len`) and return the
    /// per-position output of the final layer.
    fn encode_padded(&self, padded_ids: &[Option<usize>]) -> RecsysResult<Vec<f32>> {
        let seq_len = self.cfg.max_seq_len;
        let d = self.cfg.embed_dim;
        if padded_ids.len() != seq_len {
            return Err(RecsysError::DimensionMismatch {
                expected: seq_len,
                got: padded_ids.len(),
            });
        }

        let mut x = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            let pos_emb_start = pos * d;
            let pos_emb_end = pos_emb_start + d;
            let pos_e = self
                .pos_embeds
                .get(pos_emb_start..pos_emb_end)
                .ok_or_else(|| RecsysError::Internal {
                    msg: "pos embedding OOB".to_string(),
                })?;
            if let Some(Some(id)) = padded_ids.get(pos) {
                let item_e = self.lookup_item(*id)?;
                for k in 0..d {
                    let e = item_e.get(k).copied().unwrap_or(0.0);
                    let p = pos_e.get(k).copied().unwrap_or(0.0);
                    if let Some(slot) = x.get_mut(pos * d + k) {
                        *slot = e + p;
                    }
                }
            }
            // else: padding token → leave row at zero.
        }

        let mut h = x;
        for layer in &self.layers {
            h = self.apply_layer(&h, layer)?;
        }
        Ok(h)
    }

    /// Encode a session and score every catalogue item with a dot product
    /// against the **last-position** final-layer output.
    ///
    /// Sessions shorter than `max_seq_len` are right-zero-padded; longer
    /// sessions return an error.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `session_item_ids.is_empty()`.
    /// - [`RecsysError::InvalidConfig`] when the session length exceeds
    ///   `max_seq_len`.
    /// - [`RecsysError::ItemOutOfBounds`] when any id `>= n_items`.
    pub fn forward_session(&self, session_item_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        if session_item_ids.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        if session_item_ids.len() > self.cfg.max_seq_len {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "session length {} exceeds max_seq_len {}",
                    session_item_ids.len(),
                    self.cfg.max_seq_len
                ),
            });
        }
        for &id in session_item_ids {
            if id >= self.cfg.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: id,
                    n: self.cfg.n_items,
                });
            }
        }

        // Pad to max_seq_len; the **last** valid position is at index
        // `session_item_ids.len() - 1`.
        let mut padded: Vec<Option<usize>> = vec![None; self.cfg.max_seq_len];
        for (i, &id) in session_item_ids.iter().enumerate() {
            if let Some(slot) = padded.get_mut(i) {
                *slot = Some(id);
            }
        }
        let last_idx = session_item_ids.len() - 1;
        let h = self.encode_padded(&padded)?;
        let d = self.cfg.embed_dim;
        let last_start = last_idx * d;
        let last_end = last_start + d;
        let last_h = h
            .get(last_start..last_end)
            .ok_or_else(|| RecsysError::Internal {
                msg: "last position output OOB".to_string(),
            })?;

        let mut logits = vec![0.0_f32; self.cfg.n_items];
        for j in 0..self.cfg.n_items {
            let e_j = self.lookup_item(j)?;
            let mut acc = 0.0_f32;
            for k in 0..d {
                let e = e_j.get(k).copied().unwrap_or(0.0);
                let q = last_h.get(k).copied().unwrap_or(0.0);
                acc += e * q;
            }
            if let Some(slot) = logits.get_mut(j) {
                *slot = acc;
            }
        }
        Ok(logits)
    }

    /// Total number of learnable parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let layer_p: usize = self
            .layers
            .iter()
            .map(|l| {
                l.filter_real.len()
                    + l.filter_imag.len()
                    + l.ln1_gamma.len()
                    + l.ln1_beta.len()
                    + l.w_ffn1.len()
                    + l.b_ffn1.len()
                    + l.w_ffn2.len()
                    + l.b_ffn2.len()
                    + l.ln2_gamma.len()
                    + l.ln2_beta.len()
            })
            .sum();
        self.item_embeds.len() + self.pos_embeds.len() + layer_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_cfg() -> FmlpRecConfig {
        FmlpRecConfig {
            embed_dim: 4,
            max_seq_len: 8,
            n_layers: 2,
            hidden_dim: 8,
            n_items: 12,
        }
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "values differ: {a} vs {b}, tol = {tol}"
        );
    }

    #[test]
    fn fft_roundtrip_unit_impulse() {
        let n = 8;
        let mut re = vec![0.0_f32; n];
        let mut im = vec![0.0_f32; n];
        re[0] = 1.0;
        let orig_re = re.clone();
        fft_inplace(&mut re, &mut im, false).expect("fft_inplace should succeed");
        fft_inplace(&mut re, &mut im, true).expect("fft_inplace should succeed");
        for (a, b) in orig_re.iter().zip(re.iter()) {
            assert_close(*a, *b, 1e-4);
        }
        for v in &im {
            assert!(v.abs() < 1e-4);
        }
    }

    #[test]
    fn fft_roundtrip_sinusoid() {
        let n = 16;
        let mut re: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 3.0 * i as f32 / n as f32).sin())
            .collect();
        let mut im = vec![0.0_f32; n];
        let orig_re = re.clone();
        fft_inplace(&mut re, &mut im, false).expect("fft_inplace should succeed");
        fft_inplace(&mut re, &mut im, true).expect("fft_inplace should succeed");
        for (a, b) in orig_re.iter().zip(re.iter()) {
            assert_close(*a, *b, 1e-4);
        }
    }

    #[test]
    fn fft_roundtrip_gaussian_like() {
        let n = 16;
        let mut re: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32 - 8.0) / 3.0;
                (-(x * x)).exp()
            })
            .collect();
        let mut im = vec![0.0_f32; n];
        let orig_re = re.clone();
        fft_inplace(&mut re, &mut im, false).expect("fft_inplace should succeed");
        fft_inplace(&mut re, &mut im, true).expect("fft_inplace should succeed");
        for (a, b) in orig_re.iter().zip(re.iter()) {
            assert_close(*a, *b, 1e-4);
        }
    }

    #[test]
    fn fft_non_power_of_two_errors() {
        let mut re = vec![0.0_f32; 6];
        let mut im = vec![0.0_f32; 6];
        assert!(matches!(
            fft_inplace(&mut re, &mut im, false),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn identity_filter_zero_ffn_is_residual_layernorm_identity() {
        // With filter_real = 1, filter_imag = 0, and FFN weights = 0, the
        // filter pass yields the input itself (identity in frequency
        // domain), so layer output equals
        //   LayerNorm( LayerNorm(X + X) + GELU(LayerNorm(X+X)) · 0 + 0 ).
        // Since the FFN weights are zero, the second-residual is just
        // LayerNorm of LayerNorm(2·X) (LayerNorm is idempotent up to gamma=1,
        // beta=0). For a random X with gamma=1,beta=0 we can verify the per
        // position outputs equal layer_norm(2·X[pos]) → which is
        // layer_norm(X[pos]) (scale-invariant).
        let cfg = FmlpRecConfig {
            embed_dim: 4,
            max_seq_len: 8,
            n_layers: 1,
            hidden_dim: 4,
            n_items: 5,
        };
        let mut rng = LcgRng::new(101);
        let mut model = FmlpRec::new(cfg.clone(), &mut rng).expect("value should be present");
        // Zero the FFN weights and biases.
        let layer = &mut model.layers[0];
        for v in layer.w_ffn1.iter_mut() {
            *v = 0.0;
        }
        for v in layer.b_ffn1.iter_mut() {
            *v = 0.0;
        }
        for v in layer.w_ffn2.iter_mut() {
            *v = 0.0;
        }
        for v in layer.b_ffn2.iter_mut() {
            *v = 0.0;
        }

        // Build a random X.
        let seq_len = cfg.max_seq_len;
        let d = cfg.embed_dim;
        let mut rng_x = LcgRng::new(999);
        let mut x = vec![0.0_f32; seq_len * d];
        rng_x.fill_normal(&mut x);
        let out = model
            .apply_layer(&x, &model.layers[0])
            .expect("apply_layer should succeed");
        // Reference: per-position LayerNorm(X+X) (and then LayerNorm again
        // because the FFN adds zero residual).
        for pos in 0..seq_len {
            let mut residual = vec![0.0_f32; d];
            for (k, slot) in residual.iter_mut().enumerate().take(d) {
                let xv = x.get(pos * d + k).copied().unwrap_or(0.0);
                *slot = xv + xv;
            }
            let ones = vec![1.0_f32; d];
            let zeros = vec![0.0_f32; d];
            let ln1 = layer_norm(&residual, &ones, &zeros, 1e-5);
            // Apply LayerNorm a second time (residual = ln1 + 0).
            let ln2 = layer_norm(&ln1, &ones, &zeros, 1e-5);
            for (k, &expected) in ln2.iter().enumerate().take(d) {
                let got = out.get(pos * d + k).copied().unwrap_or(0.0);
                assert_close(got, expected, 1e-4);
            }
        }
    }

    #[test]
    fn session_shorter_than_max_works() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        let logits = model
            .forward_session(&[0_usize, 1, 2])
            .expect("forward_session should succeed");
        assert_eq!(logits.len(), 12);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn same_session_two_calls_returns_identical() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        let logits_a = model
            .forward_session(&[1_usize, 3, 5, 7])
            .expect("forward_session should succeed");
        let logits_b = model
            .forward_session(&[1_usize, 3, 5, 7])
            .expect("forward_session should succeed");
        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            assert_close(*a, *b, 1e-5);
        }
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let mut rng = LcgRng::new(7);
        let mut x = vec![0.0_f32; 16];
        rng.fill_normal(&mut x);
        let gamma = vec![1.0_f32; 16];
        let beta = vec![0.0_f32; 16];
        let out = layer_norm(&x, &gamma, &beta, 1e-5);
        let mean: f32 = out.iter().copied().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-4, "mean {mean} should be ~0");
        assert!((var - 1.0).abs() < 1e-3, "var {var} should be ~1");
    }

    #[test]
    fn err_max_seq_len_not_power_of_two() {
        let mut rng = make_rng();
        let cfg = FmlpRecConfig {
            embed_dim: 4,
            max_seq_len: 6,
            n_layers: 1,
            hidden_dim: 8,
            n_items: 5,
        };
        assert!(matches!(
            FmlpRec::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_n_layers_zero() {
        let mut rng = make_rng();
        let cfg = FmlpRecConfig {
            embed_dim: 4,
            max_seq_len: 8,
            n_layers: 0,
            hidden_dim: 8,
            n_items: 5,
        };
        assert!(matches!(
            FmlpRec::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn output_length_equals_n_items() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        let logits = model
            .forward_session(&[0_usize, 1])
            .expect("forward_session should succeed");
        assert_eq!(logits.len(), 12);
    }

    #[test]
    fn err_n_items_zero() {
        let mut rng = make_rng();
        let cfg = FmlpRecConfig {
            embed_dim: 4,
            max_seq_len: 8,
            n_layers: 1,
            hidden_dim: 8,
            n_items: 0,
        };
        assert!(matches!(
            FmlpRec::new(cfg, &mut rng),
            Err(RecsysError::InvalidNumItems { n: 0 })
        ));
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = FmlpRecConfig {
            embed_dim: 0,
            max_seq_len: 8,
            n_layers: 1,
            hidden_dim: 8,
            n_items: 5,
        };
        assert!(matches!(
            FmlpRec::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { d: 0 })
        ));
    }

    #[test]
    fn err_hidden_dim_zero() {
        let mut rng = make_rng();
        let cfg = FmlpRecConfig {
            embed_dim: 4,
            max_seq_len: 8,
            n_layers: 1,
            hidden_dim: 0,
            n_items: 5,
        };
        assert!(matches!(
            FmlpRec::new(cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_session_item_oor() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        // n_items = 12; id 12 is out of range.
        assert!(matches!(
            model.forward_session(&[1_usize, 12]),
            Err(RecsysError::ItemOutOfBounds { idx: 12, n: 12 })
        ));
    }

    #[test]
    fn err_empty_session() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        assert!(matches!(
            model.forward_session(&[]),
            Err(RecsysError::EmptyInput)
        ));
    }

    #[test]
    fn err_session_longer_than_max_seq_len() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        // max_seq_len = 8; a 9-long session must error.
        let long_session: Vec<usize> = (0..9).map(|i| i % 12).collect();
        assert!(matches!(
            model.forward_session(&long_session),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn deterministic_init_given_seed() {
        let mut rng_a = LcgRng::new(2026);
        let mut rng_b = LcgRng::new(2026);
        let model_a = FmlpRec::new(default_cfg(), &mut rng_a).expect("value should be present");
        let model_b = FmlpRec::new(default_cfg(), &mut rng_b).expect("value should be present");
        assert_eq!(model_a.item_embeds, model_b.item_embeds);
        assert_eq!(model_a.pos_embeds, model_b.pos_embeds);
        for (la, lb) in model_a.layers.iter().zip(model_b.layers.iter()) {
            assert_eq!(la.w_ffn1, lb.w_ffn1);
            assert_eq!(la.w_ffn2, lb.w_ffn2);
            assert_eq!(la.filter_real, lb.filter_real);
            assert_eq!(la.filter_imag, lb.filter_imag);
        }
    }

    #[test]
    fn weights_finite_after_init() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        for v in &model.item_embeds {
            assert!(v.is_finite());
        }
        for v in &model.pos_embeds {
            assert!(v.is_finite());
        }
        for layer in &model.layers {
            for v in layer
                .w_ffn1
                .iter()
                .chain(layer.w_ffn2.iter())
                .chain(layer.filter_real.iter())
                .chain(layer.filter_imag.iter())
                .chain(layer.ln1_gamma.iter())
                .chain(layer.ln1_beta.iter())
                .chain(layer.ln2_gamma.iter())
                .chain(layer.ln2_beta.iter())
            {
                assert!(v.is_finite(), "weight {v} not finite");
            }
        }
    }

    #[test]
    fn n_params_positive() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        assert!(model.n_params() > 0);
    }

    #[test]
    fn filter_init_is_one_zero() {
        let mut rng = make_rng();
        let model = FmlpRec::new(default_cfg(), &mut rng).expect("value should be present");
        for layer in &model.layers {
            for v in &layer.filter_real {
                assert!((v - 1.0).abs() < 1e-7);
            }
            for v in &layer.filter_imag {
                assert!(v.abs() < 1e-7);
            }
        }
    }
}
