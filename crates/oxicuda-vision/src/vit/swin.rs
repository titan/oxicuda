//! Swin Transformer block: window-based and shifted-window multi-head
//! self-attention with relative position bias.
//!
//! Reference: Liu et al. 2021 ICCV,
//! *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"*.
//!
//! ## Token layout
//! The input token sequence `x` is row-major spatial of shape
//! `(input_h * input_w, dim)`: token `(h, w)` lives at flat row index
//! `h * input_w + w`, and each row holds `dim` channels.
//!
//! ## Forward pass (pre-norm variant)
//! ```text
//! shortcut = x
//! h        = LN1(x)
//! h        = cyclic_shift(h, -M/2)         (only when SW-MSA)
//! windows  = window_partition(h)
//! attn     = per-window MHSA(+rel-pos-bias, +mask if SW-MSA)
//! merged   = window_reverse(attn)
//! merged   = cyclic_shift(merged, +M/2)    (only when SW-MSA)
//! x        = shortcut + merged
//! x        = x + MLP(LN2(x))
//! ```
//!
//! `window_partition` / `window_reverse` are exact inverses, and
//! `cyclic_shift(x, k)` followed by `cyclic_shift(_, -k)` is the identity.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::vit_block::{gelu_exact, layer_norm, linear, softmax_rows},
};

/// Mask penalty added to attention logits for token pairs that, after the
/// cyclic shift, originate from different image regions and therefore must
/// not attend to one another.
const MASK_PENALTY: f32 = -100.0;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a single Swin Transformer block.
#[derive(Debug, Clone, PartialEq)]
pub struct SwinConfig {
    /// Token / channel embedding dimension.
    pub dim: usize,
    /// Number of attention heads. Must divide `dim`.
    pub n_heads: usize,
    /// Window edge length `M` (windows are `M × M`).
    pub window_size: usize,
    /// Feature-map height in tokens. Must be divisible by `window_size`.
    pub input_h: usize,
    /// Feature-map width in tokens. Must be divisible by `window_size`.
    pub input_w: usize,
    /// Shift selector: `0` ⇒ W-MSA, `1` ⇒ SW-MSA (shift = `window_size / 2`).
    pub shift: usize,
    /// MLP hidden-dim multiplier: `hidden = mlp_ratio * dim`.
    pub mlp_ratio: usize,
}

impl SwinConfig {
    /// Create and validate a `SwinConfig`.
    ///
    /// # Errors
    /// - `dim == 0` → `InvalidEmbedDim`
    /// - `n_heads == 0` → `InvalidNumHeads`
    /// - `dim % n_heads != 0` → `HeadDimMismatch`
    /// - `window_size == 0` → `InvalidPatchSize`
    /// - `input_h % window_size != 0` or `input_w % window_size != 0`
    ///   → `InvalidPatchSize`
    /// - `shift > 1` → `Internal`
    /// - `mlp_ratio == 0` → `Internal`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        n_heads: usize,
        window_size: usize,
        input_h: usize,
        input_w: usize,
        shift: usize,
        mlp_ratio: usize,
    ) -> VisionResult<Self> {
        if dim == 0 {
            return Err(VisionError::InvalidEmbedDim(dim));
        }
        if n_heads == 0 {
            return Err(VisionError::InvalidNumHeads(n_heads));
        }
        if dim % n_heads != 0 {
            return Err(VisionError::HeadDimMismatch {
                n_heads,
                embed_dim: dim,
            });
        }
        if window_size == 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size: window_size,
                img_size: input_h,
            });
        }
        if input_h == 0 || input_w == 0 {
            return Err(VisionError::InvalidImageSize {
                height: input_h,
                width: input_w,
                channels: dim,
            });
        }
        if input_h % window_size != 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size: window_size,
                img_size: input_h,
            });
        }
        if input_w % window_size != 0 {
            return Err(VisionError::InvalidPatchSize {
                patch_size: window_size,
                img_size: input_w,
            });
        }
        if shift > 1 {
            return Err(VisionError::Internal(format!(
                "shift must be 0 (W-MSA) or 1 (SW-MSA), got {shift}"
            )));
        }
        if mlp_ratio == 0 {
            return Err(VisionError::Internal("mlp_ratio must be >= 1".to_string()));
        }
        Ok(Self {
            dim,
            n_heads,
            window_size,
            input_h,
            input_w,
            shift,
            mlp_ratio,
        })
    }

    /// Dimension per attention head.
    #[must_use]
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// MLP hidden dimension.
    #[must_use]
    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.mlp_ratio * self.dim
    }

    /// Number of tokens per window: `M²`.
    #[must_use]
    #[inline]
    pub fn window_tokens(&self) -> usize {
        self.window_size * self.window_size
    }

    /// Number of windows tiling the feature map: `(H/M) · (W/M)`.
    #[must_use]
    #[inline]
    pub fn n_windows(&self) -> usize {
        (self.input_h / self.window_size) * (self.input_w / self.window_size)
    }

    /// Length of one head's relative-position-bias table: `(2M-1)²`.
    #[must_use]
    #[inline]
    pub fn rel_table_len(&self) -> usize {
        let span = 2 * self.window_size - 1;
        span * span
    }

    /// Whether SW-MSA shifting is actually applied.
    ///
    /// `shift == 1` requests SW-MSA, but — following the reference Swin
    /// implementation — when a single window already covers the whole feature
    /// map (`window_size >= min(input_h, input_w)`) shifting is pointless and
    /// is disabled, collapsing the block back to W-MSA.
    #[must_use]
    #[inline]
    pub fn is_shifted(&self) -> bool {
        self.shift != 0 && self.window_size < self.input_h.min(self.input_w)
    }

    /// Cyclic-shift displacement used by SW-MSA: `M / 2` (0 when W-MSA or when
    /// the window spans the whole feature map; see [`Self::is_shifted`]).
    #[must_use]
    #[inline]
    pub fn shift_size(&self) -> usize {
        if self.is_shifted() {
            self.window_size / 2
        } else {
            0
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Learnable weights for one Swin Transformer block.
///
/// All tensors are flat row-major `Vec<f32>`.
pub struct SwinWeights {
    /// Fused QKV projection kernel `[3·dim, dim]`.
    pub qkv_weight: Vec<f32>,
    /// Fused QKV projection bias `[3·dim]`.
    pub qkv_bias: Vec<f32>,

    /// Output projection kernel `[dim, dim]`.
    pub proj_weight: Vec<f32>,
    /// Output projection bias `[dim]`.
    pub proj_bias: Vec<f32>,

    /// Relative-position-bias table `[n_heads · (2M-1)²]`
    /// (head-major: head `h` occupies `[h·(2M-1)², (h+1)·(2M-1)²)`).
    pub relative_position_bias_table: Vec<f32>,

    /// MLP first linear kernel `[hidden, dim]`.
    pub mlp_w1: Vec<f32>,
    /// MLP first linear bias `[hidden]`.
    pub mlp_b1: Vec<f32>,
    /// MLP second linear kernel `[dim, hidden]`.
    pub mlp_w2: Vec<f32>,
    /// MLP second linear bias `[dim]`.
    pub mlp_b2: Vec<f32>,

    /// LayerNorm 1 scale `[dim]` (init 1).
    pub ln1_gamma: Vec<f32>,
    /// LayerNorm 1 bias `[dim]` (init 0).
    pub ln1_beta: Vec<f32>,
    /// LayerNorm 2 scale `[dim]` (init 1).
    pub ln2_gamma: Vec<f32>,
    /// LayerNorm 2 bias `[dim]` (init 0).
    pub ln2_beta: Vec<f32>,
}

impl SwinWeights {
    /// Xavier-style default initialisation, mirroring `ViTBlockWeights`.
    ///
    /// - Attention & MLP weights: `N(0, 1/√dim)`
    /// - Relative-position-bias table: small `N(0, 0.02)`
    /// - Biases: zeros
    /// - LayerNorm scales: ones; biases: zeros
    pub fn default_init(cfg: &SwinConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.dim;
        let hidden = cfg.hidden_dim();
        let scale = 1.0 / (d as f32).sqrt();

        let fill_scaled = |rng: &mut LcgRng, n: usize, sc: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= sc;
            }
            v
        };

        let qkv_weight = fill_scaled(rng, 3 * d * d, scale);
        let qkv_bias = vec![0.0f32; 3 * d];
        let proj_weight = fill_scaled(rng, d * d, scale);
        let proj_bias = vec![0.0f32; d];

        let table_len = cfg.n_heads * cfg.rel_table_len();
        let relative_position_bias_table = fill_scaled(rng, table_len, 0.02);

        let mlp_w1 = fill_scaled(rng, hidden * d, scale);
        let mlp_b1 = vec![0.0f32; hidden];
        let mlp_w2 = fill_scaled(rng, d * hidden, scale);
        let mlp_b2 = vec![0.0f32; d];

        let ln1_gamma = vec![1.0f32; d];
        let ln1_beta = vec![0.0f32; d];
        let ln2_gamma = vec![1.0f32; d];
        let ln2_beta = vec![0.0f32; d];

        Self {
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            relative_position_bias_table,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
        }
    }
}

// ─── SwinBlock ─────────────────────────────────────────────────────────────────

/// A single pre-norm Swin Transformer block (W-MSA or SW-MSA).
pub struct SwinBlock {
    /// Block configuration.
    pub cfg: SwinConfig,
    /// Learnable weights.
    pub weights: SwinWeights,
}

impl SwinBlock {
    /// Construct a new block with Xavier-initialised weights.
    ///
    /// # Errors
    /// Propagates configuration validation errors from [`SwinConfig::new`]
    /// re-checking via the already-validated `cfg`.
    pub fn new(cfg: SwinConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        // `cfg` is expected to be validated by `SwinConfig::new`, but re-run the
        // structural invariants so a hand-built config cannot bypass them.
        let cfg = SwinConfig::new(
            cfg.dim,
            cfg.n_heads,
            cfg.window_size,
            cfg.input_h,
            cfg.input_w,
            cfg.shift,
            cfg.mlp_ratio,
        )?;
        let weights = SwinWeights::default_init(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Validate that `x` has the expected `(H·W, dim)` flat length.
    fn check_input_len(&self, x: &[f32]) -> VisionResult<()> {
        let expected = self.cfg.input_h * self.cfg.input_w * self.cfg.dim;
        if x.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        Ok(())
    }

    /// Partition a `(H·W, C)` token sequence into windows.
    ///
    /// Output layout is `(n_windows · M · M, C)`: windows are visited in
    /// row-major window order `(window_row, window_col)`, and within a window
    /// tokens are flattened in row-major `(row, col)` order.
    ///
    /// # Errors
    /// `DimensionMismatch` if `x.len() != H·W·C`.
    pub fn window_partition(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        self.check_input_len(x)?;
        let c = self.cfg.dim;
        let h = self.cfg.input_h;
        let w = self.cfg.input_w;
        let m = self.cfg.window_size;
        let win_rows = h / m;
        let win_cols = w / m;

        let mut out = vec![0.0f32; h * w * c];
        let mut dst = 0usize;
        for wr in 0..win_rows {
            for wc in 0..win_cols {
                for i in 0..m {
                    let row = wr * m + i;
                    for j in 0..m {
                        let col = wc * m + j;
                        let src = (row * w + col) * c;
                        out[dst..dst + c].copy_from_slice(&x[src..src + c]);
                        dst += c;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Exact inverse of [`Self::window_partition`]: scatter windows back to a
    /// `(H·W, C)` token sequence.
    ///
    /// # Errors
    /// `DimensionMismatch` if `windows.len() != H·W·C`.
    pub fn window_reverse(&self, windows: &[f32]) -> VisionResult<Vec<f32>> {
        let c = self.cfg.dim;
        let h = self.cfg.input_h;
        let w = self.cfg.input_w;
        let expected = h * w * c;
        if windows.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: windows.len(),
            });
        }
        let m = self.cfg.window_size;
        let win_rows = h / m;
        let win_cols = w / m;

        let mut out = vec![0.0f32; expected];
        let mut src = 0usize;
        for wr in 0..win_rows {
            for wc in 0..win_cols {
                for i in 0..m {
                    let row = wr * m + i;
                    for j in 0..m {
                        let col = wc * m + j;
                        let dst = (row * w + col) * c;
                        out[dst..dst + c].copy_from_slice(&windows[src..src + c]);
                        src += c;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Roll the feature map by `shift` tokens along both the H and W axes
    /// (with wrap-around). Positive `shift` moves content toward larger
    /// indices; negative `shift` toward smaller indices.
    ///
    /// `cyclic_shift(x, k)` then `cyclic_shift(_, -k)` is the identity.
    ///
    /// # Errors
    /// `DimensionMismatch` if `x.len() != H·W·C`.
    pub fn cyclic_shift(&self, x: &[f32], shift: i32) -> VisionResult<Vec<f32>> {
        self.check_input_len(x)?;
        let c = self.cfg.dim;
        let h = self.cfg.input_h as i32;
        let w = self.cfg.input_w as i32;

        let mut out = vec![0.0f32; x.len()];
        // Reduce the shift to the canonical positive residue per axis.
        let sh = shift.rem_euclid(h);
        let sw = shift.rem_euclid(w);
        for dr in 0..h {
            // Source row that lands on destination row `dr`.
            let sr = (dr - sh).rem_euclid(h);
            for dc in 0..w {
                let sc = (dc - sw).rem_euclid(w);
                let dst = ((dr * w + dc) as usize) * c;
                let src = ((sr * w + sc) as usize) * c;
                out[dst..dst + c].copy_from_slice(&x[src..src + c]);
            }
        }
        Ok(out)
    }

    /// Pairwise relative-position index table for one window.
    ///
    /// Length is `M² · M²`. For tokens `a` and `b` with 2-D coordinates
    /// `(ha, wa)` and `(hb, wb)`,
    /// `rel_h = ha - hb ∈ [-(M-1), M-1]`, `rel_w = wa - wb ∈ [-(M-1), M-1]`,
    /// and the index is `(rel_h + M-1) · (2M-1) + (rel_w + M-1)`,
    /// always `< (2M-1)²`.
    #[must_use]
    pub fn relative_position_index(&self) -> Vec<usize> {
        let m = self.cfg.window_size;
        let span = 2 * m - 1;
        let n = m * m;
        let mut idx = vec![0usize; n * n];
        for a in 0..n {
            let ha = a / m;
            let wa = a % m;
            for b in 0..n {
                let hb = b / m;
                let wb = b % m;
                let rel_h = ha as i64 - hb as i64 + (m as i64 - 1);
                let rel_w = wa as i64 - wb as i64 + (m as i64 - 1);
                idx[a * n + b] = (rel_h * span as i64 + rel_w) as usize;
            }
        }
        idx
    }

    /// SW-MSA attention mask: `n_windows × M² × M²`, each entry `0.0`
    /// (same region) or `MASK_PENALTY` (different region).
    ///
    /// Each spatial location is assigned a region id from the standard Swin
    /// `(0..-M, -M..-M/2, -M/2..)` slicing along H and W (nine regions). After
    /// the cyclic roll, two tokens in the same window may originate from
    /// different image regions; those pairs are masked.
    ///
    /// For a W-MSA block (`shift == 0`) the mask is all zeros.
    ///
    /// # Errors
    /// Currently infallible, but returns `VisionResult` for API symmetry and
    /// future-proofing.
    pub fn attention_mask(&self) -> VisionResult<Vec<f32>> {
        let n_windows = self.cfg.n_windows();
        let m = self.cfg.window_size;
        let win_tok = m * m;
        let mut mask = vec![0.0f32; n_windows * win_tok * win_tok];

        if !self.cfg.is_shifted() {
            // W-MSA (or window == whole map): no cross-region contamination.
            return Ok(mask);
        }

        let h = self.cfg.input_h;
        let w = self.cfg.input_w;
        let shift_size = self.cfg.shift_size();

        // Region id image: same construction as the reference Swin `img_mask`.
        // Three H-slices × three W-slices ⇒ up to nine distinct region ids.
        let h_slices = [
            (0usize, h - m),
            (h - m, h - shift_size),
            (h - shift_size, h),
        ];
        let w_slices = [
            (0usize, w - m),
            (w - m, w - shift_size),
            (w - shift_size, w),
        ];
        let mut region = vec![0usize; h * w];
        let mut region_id = 0usize;
        for &(h0, h1) in &h_slices {
            for &(w0, w1) in &w_slices {
                for r in h0..h1 {
                    for col in w0..w1 {
                        region[r * w + col] = region_id;
                    }
                }
                region_id += 1;
            }
        }

        // Partition the region-id image into windows in the same row-major
        // window order as `window_partition`, then build per-window masks.
        let win_rows = h / m;
        let win_cols = w / m;
        let mut win = 0usize;
        for wr in 0..win_rows {
            for wc in 0..win_cols {
                // Gather the M² region ids of this window (row-major in-window).
                let mut win_region = vec![0usize; win_tok];
                let mut t = 0usize;
                for i in 0..m {
                    let row = wr * m + i;
                    for j in 0..m {
                        let col = wc * m + j;
                        win_region[t] = region[row * w + col];
                        t += 1;
                    }
                }
                let base = win * win_tok * win_tok;
                for a in 0..win_tok {
                    for b in 0..win_tok {
                        if win_region[a] != win_region[b] {
                            mask[base + a * win_tok + b] = MASK_PENALTY;
                        }
                    }
                }
                win += 1;
            }
        }

        Ok(mask)
    }

    /// Forward pass: `(H·W, C) → (H·W, C)`.
    ///
    /// Pre-norm W-MSA / SW-MSA attention with relative position bias, residual
    /// connection, then pre-norm MLP (`Linear → GELU → Linear`) with a second
    /// residual connection.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != H·W·C`.
    /// - `NonFinite` if intermediate attention produces non-finite values.
    pub fn forward(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        self.check_input_len(x)?;
        let d = self.cfg.dim;
        let n_tok = self.cfg.input_h * self.cfg.input_w;
        let w = &self.weights;

        // Pre-norm 1.
        let normed = layer_norm(x, &w.ln1_gamma, &w.ln1_beta, n_tok, d, 1e-5);

        // Optional cyclic shift before partition (SW-MSA only).
        let shifted_block = self.cfg.is_shifted();
        let shift_size = self.cfg.shift_size();
        let shifted = if shifted_block {
            self.cyclic_shift(&normed, -(shift_size as i32))?
        } else {
            normed
        };

        // Window partition → per-window MHSA → reverse.
        let windows = self.window_partition(&shifted)?;
        let mask = if shifted_block {
            Some(self.attention_mask()?)
        } else {
            None
        };
        let attn = self.window_attention(&windows, mask.as_deref())?;
        let merged = self.window_reverse(&attn)?;

        // Optional reverse cyclic shift after reverse (SW-MSA only).
        let merged = if shifted_block {
            self.cyclic_shift(&merged, shift_size as i32)?
        } else {
            merged
        };

        // Residual 1.
        let mut h: Vec<f32> = x.iter().zip(merged.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm 2 + MLP + residual 2.
        let normed2 = layer_norm(&h, &w.ln2_gamma, &w.ln2_beta, n_tok, d, 1e-5);
        let hidden = self.cfg.hidden_dim();
        let mid = linear(&normed2, &w.mlp_w1, &w.mlp_b1, d, hidden);
        let mid: Vec<f32> = mid.into_iter().map(gelu_exact).collect();
        let mlp_out = linear(&mid, &w.mlp_w2, &w.mlp_b2, hidden, d);
        for (o, m_v) in h.iter_mut().zip(mlp_out.iter()) {
            *o += m_v;
        }

        if h.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("swin block output"));
        }
        Ok(h)
    }

    /// Per-window multi-head self-attention with relative position bias and an
    /// optional additive attention mask.
    ///
    /// `windows` is `(n_windows · M², C)` row-major. `mask`, when present, is
    /// `(n_windows · M² · M²)` row-major. Returns `(n_windows · M², C)`.
    fn window_attention(&self, windows: &[f32], mask: Option<&[f32]>) -> VisionResult<Vec<f32>> {
        let d = self.cfg.dim;
        let n_heads = self.cfg.n_heads;
        let head_dim = self.cfg.head_dim();
        let m = self.cfg.window_size;
        let win_tok = m * m;
        let n_windows = self.cfg.n_windows();
        let span = 2 * m - 1;
        let table_per_head = span * span;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let rel_index = self.relative_position_index();
        let w = &self.weights;

        let mut out = vec![0.0f32; n_windows * win_tok * d];
        // Per-window scratch for QKV split and attention scores.
        let mut q = vec![0.0f32; win_tok * d];
        let mut k = vec![0.0f32; win_tok * d];
        let mut v = vec![0.0f32; win_tok * d];
        let mut scores = vec![0.0f32; win_tok * win_tok];

        for win in 0..n_windows {
            let win_in = &windows[win * win_tok * d..(win + 1) * win_tok * d];
            // Fused QKV projection for this window: [win_tok, 3·d].
            let qkv = linear(win_in, &w.qkv_weight, &w.qkv_bias, d, 3 * d);
            for t in 0..win_tok {
                let src = &qkv[t * 3 * d..(t + 1) * 3 * d];
                q[t * d..(t + 1) * d].copy_from_slice(&src[..d]);
                k[t * d..(t + 1) * d].copy_from_slice(&src[d..2 * d]);
                v[t * d..(t + 1) * d].copy_from_slice(&src[2 * d..]);
            }

            let win_concat = &mut out[win * win_tok * d..(win + 1) * win_tok * d];
            let win_mask =
                mask.map(|mk| &mk[win * win_tok * win_tok..(win + 1) * win_tok * win_tok]);

            for head in 0..n_heads {
                let hd_off = head * head_dim;
                let table_off = head * table_per_head;

                // logits[a][b] = (Q_a · K_b)/√head_dim + rel_bias[a][b] (+ mask)
                for a in 0..win_tok {
                    for b in 0..win_tok {
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim {
                            dot += q[a * d + hd_off + dd] * k[b * d + hd_off + dd];
                        }
                        let bias =
                            w.relative_position_bias_table[table_off + rel_index[a * win_tok + b]];
                        let mut logit = dot * scale + bias;
                        if let Some(mk) = win_mask {
                            logit += mk[a * win_tok + b];
                        }
                        scores[a * win_tok + b] = logit;
                    }
                }

                softmax_rows(&mut scores, win_tok, win_tok);

                // A[a][dd] = Σ_b scores[a][b] · V[b, head]
                for a in 0..win_tok {
                    for dd in 0..head_dim {
                        let mut acc = 0.0f32;
                        for b in 0..win_tok {
                            acc += scores[a * win_tok + b] * v[b * d + hd_off + dd];
                        }
                        win_concat[a * d + hd_off + dd] = acc;
                    }
                }
            }
        }

        // Output projection [d, d] applied to all tokens of all windows.
        let projected = linear(&out, &w.proj_weight, &w.proj_bias, d, d);
        if projected.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("swin window attention"));
        }
        Ok(projected)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn wmsa_cfg() -> SwinConfig {
        // dim=32, 4 heads, window 2, 4×4 grid → 4 windows, W-MSA.
        SwinConfig::new(32, 4, 2, 4, 4, 0, 4).expect("valid config")
    }

    fn swmsa_cfg() -> SwinConfig {
        // dim=32, 4 heads, window 2, 4×4 grid → 4 windows, SW-MSA (shift=1).
        SwinConfig::new(32, 4, 2, 4, 4, 1, 4).expect("valid config")
    }

    fn random_input(cfg: &SwinConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut x = vec![0.0f32; cfg.input_h * cfg.input_w * cfg.dim];
        rng.fill_normal(&mut x);
        x
    }

    #[test]
    fn config_derived_quantities() {
        let cfg = wmsa_cfg();
        assert_eq!(cfg.head_dim(), 8);
        assert_eq!(cfg.hidden_dim(), 128);
        assert_eq!(cfg.window_tokens(), 4);
        assert_eq!(cfg.n_windows(), 4);
        assert_eq!(cfg.rel_table_len(), 9); // (2·2-1)² = 9
    }

    #[test]
    fn window_partition_length() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(1);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 2);
        let parts = block.window_partition(&x).expect("partition");
        assert_eq!(parts.len(), cfg.input_h * cfg.input_w * cfg.dim);
    }

    #[test]
    fn window_reverse_round_trip_exact() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(3);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 4);
        let parts = block.window_partition(&x).expect("partition");
        let back = block.window_reverse(&parts).expect("reverse");
        assert_eq!(back.len(), x.len());
        for (a, b) in back.iter().zip(x.iter()) {
            assert_eq!(a, b, "round-trip not exact");
        }
    }

    #[test]
    fn window_reverse_round_trip_nonsquare() {
        // Non-square feature map exercises distinct H and W tiling.
        let cfg = SwinConfig::new(16, 2, 2, 4, 6, 0, 2).expect("cfg");
        let mut rng = LcgRng::new(31);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 32);
        let parts = block.window_partition(&x).expect("partition");
        let back = block.window_reverse(&parts).expect("reverse");
        for (a, b) in back.iter().zip(x.iter()) {
            assert_eq!(a, b, "non-square round-trip not exact");
        }
    }

    #[test]
    fn n_windows_matches_formula() {
        let cfg = SwinConfig::new(16, 2, 2, 6, 8, 0, 2).expect("cfg");
        assert_eq!(cfg.n_windows(), (6 / 2) * (8 / 2));
    }

    #[test]
    fn cyclic_shift_round_trip_identity() {
        let cfg = swmsa_cfg();
        let mut rng = LcgRng::new(5);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 6);
        for k in [1i32, 2, 3, -1, -2] {
            let s = block.cyclic_shift(&x, k).expect("shift");
            let back = block.cyclic_shift(&s, -k).expect("unshift");
            for (a, b) in back.iter().zip(x.iter()) {
                assert_eq!(a, b, "cyclic shift round-trip failed for k={k}");
            }
        }
    }

    #[test]
    fn cyclic_shift_zero_is_identity() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(7);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 8);
        let s = block.cyclic_shift(&x, 0).expect("shift");
        assert_eq!(s, x);
    }

    #[test]
    fn relative_position_index_length_and_bounds() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(9);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let idx = block.relative_position_index();
        let m = cfg.window_size;
        let span = 2 * m - 1;
        assert_eq!(idx.len(), m * m * m * m, "length must be M⁴");
        assert!(idx.iter().all(|&v| v < span * span), "index out of bounds");
    }

    #[test]
    fn relative_position_index_diagonal_is_center() {
        // For a token attending to itself, rel_h=rel_w=0 → center index.
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(10);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let idx = block.relative_position_index();
        let m = cfg.window_size;
        let n = m * m;
        let span = 2 * m - 1;
        let center = (m - 1) * span + (m - 1);
        for a in 0..n {
            assert_eq!(idx[a * n + a], center, "diagonal must map to center bias");
        }
    }

    #[test]
    fn attention_mask_length() {
        let cfg = swmsa_cfg();
        let mut rng = LcgRng::new(11);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let mask = block.attention_mask().expect("mask");
        let win_tok = cfg.window_tokens();
        assert_eq!(mask.len(), cfg.n_windows() * win_tok * win_tok);
    }

    #[test]
    fn attention_mask_values_are_zero_or_penalty() {
        let cfg = swmsa_cfg();
        let mut rng = LcgRng::new(12);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let mask = block.attention_mask().expect("mask");
        assert!(
            mask.iter().all(|&v| v == 0.0 || v == MASK_PENALTY),
            "mask must be 0.0 or -100.0 only"
        );
    }

    #[test]
    fn single_window_config_mask_all_zero() {
        // H == W == M ⇒ one window; even SW-MSA mask is all zero (one region).
        let cfg = SwinConfig::new(16, 2, 4, 4, 4, 1, 2).expect("cfg");
        let mut rng = LcgRng::new(13);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        assert_eq!(cfg.n_windows(), 1);
        let mask = block.attention_mask().expect("mask");
        assert!(
            mask.iter().all(|&v| v == 0.0),
            "single window must be unmasked"
        );
    }

    #[test]
    fn wmsa_mask_all_zero() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(14);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let mask = block.attention_mask().expect("mask");
        assert!(
            mask.iter().all(|&v| v == 0.0),
            "W-MSA mask must be all zero"
        );
    }

    #[test]
    fn forward_output_shape_wmsa() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(15);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 16);
        let out = block.forward(&x).expect("forward");
        assert_eq!(out.len(), x.len());
    }

    #[test]
    fn forward_output_shape_swmsa() {
        let cfg = swmsa_cfg();
        let mut rng = LcgRng::new(17);
        let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 18);
        let out = block.forward(&x).expect("forward");
        assert_eq!(out.len(), x.len());
    }

    #[test]
    fn forward_finite_wmsa_and_swmsa() {
        for cfg in [wmsa_cfg(), swmsa_cfg()] {
            let mut rng = LcgRng::new(19);
            let block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
            let x = random_input(&cfg, 20);
            let out = block.forward(&x).expect("forward");
            assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        }
    }

    #[test]
    fn forward_changes_with_bias_table() {
        // Mutating the relative-position-bias table must change the output,
        // proving the bias is actually consumed by attention.
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(21);
        let mut block = SwinBlock::new(cfg.clone(), &mut rng).expect("block");
        let x = random_input(&cfg, 22);
        let before = block.forward(&x).expect("forward");
        for v in &mut block.weights.relative_position_bias_table {
            *v += 5.0;
        }
        let after = block.forward(&x).expect("forward");
        let diff: f32 = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-5, "bias table change did not affect output");
    }

    #[test]
    fn err_head_dim_mismatch() {
        let r = SwinConfig::new(32, 5, 2, 4, 4, 0, 4); // 32 % 5 != 0
        assert!(matches!(r, Err(VisionError::HeadDimMismatch { .. })));
    }

    #[test]
    fn err_input_not_divisible_by_window() {
        let r = SwinConfig::new(32, 4, 3, 4, 4, 0, 4); // 4 % 3 != 0
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn err_shift_too_large() {
        let r = SwinConfig::new(32, 4, 2, 4, 4, 2, 4); // shift > 1
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    #[test]
    fn err_mlp_ratio_zero() {
        let r = SwinConfig::new(32, 4, 2, 4, 4, 0, 0);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    #[test]
    fn err_window_size_zero() {
        let r = SwinConfig::new(32, 4, 0, 4, 4, 0, 4);
        assert!(matches!(r, Err(VisionError::InvalidPatchSize { .. })));
    }

    #[test]
    fn err_dim_zero() {
        let r = SwinConfig::new(0, 4, 2, 4, 4, 0, 4);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn err_forward_wrong_length() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(23);
        let block = SwinBlock::new(cfg, &mut rng).expect("block");
        let x = vec![0.0f32; 7]; // wrong
        let r = block.forward(&x);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_partition_wrong_length() {
        let cfg = wmsa_cfg();
        let mut rng = LcgRng::new(24);
        let block = SwinBlock::new(cfg, &mut rng).expect("block");
        let x = vec![0.0f32; 9]; // wrong
        let r = block.window_partition(&x);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = swmsa_cfg();
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let block_a = SwinBlock::new(cfg.clone(), &mut rng_a).expect("block");
        let block_b = SwinBlock::new(cfg.clone(), &mut rng_b).expect("block");
        let x = random_input(&cfg, 100);
        let out_a = block_a.forward(&x).expect("forward");
        let out_b = block_b.forward(&x).expect("forward");
        assert_eq!(out_a, out_b, "same seed must give identical output");
    }
}
