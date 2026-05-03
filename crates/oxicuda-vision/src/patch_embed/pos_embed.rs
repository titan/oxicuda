//! 2-D positional encodings for Vision Transformers.
//!
//! Provides:
//! - **`pos_2d_sincos`**: deterministic 2-D sinusoidal positional encoding
//!   as used in MAE / BEiT / DeiT. The first half of `dim` encodes the
//!   row (H) axis; the second half encodes the column (W) axis.
//! - **`LearnablePosEmbed`**: a simple learned position table.
//! - **`add_pos_embed`**: in-place addition of position embeddings to tokens.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── 2-D sinusoidal position encoding ────────────────────────────────────────

/// Compute a 2-D sinusoidal positional encoding for a `grid_h × grid_w` grid.
///
/// Each position `(h, w)` gets a `dim`-dimensional vector: the first `dim/2`
/// dimensions encode the row using the standard 1-D sinusoidal schedule, and
/// the second `dim/2` dimensions encode the column.
///
/// The temperature-based frequency schedule (Vaswani et al.) is used:
///
/// ```text
/// freq[k] = 1 / 10000^(2k / dim_half)
/// encoding[h, w, k]        = sin(h * freq[k])   for k in [0, dim/4)
/// encoding[h, w, k + dim/4] = cos(h * freq[k])   for k in [0, dim/4)
/// encoding[h, w, dim/2 + k]        = sin(w * freq[k])
/// encoding[h, w, dim/2 + k + dim/4] = cos(w * freq[k])
/// ```
///
/// Returns a flat `[grid_h * grid_w, dim]` `Vec<f32>` in row-major order.
/// `dim` must be divisible by 4.
pub fn pos_2d_sincos(grid_h: usize, grid_w: usize, dim: usize) -> VisionResult<Vec<f32>> {
    if dim == 0 || dim % 4 != 0 {
        return Err(VisionError::InvalidEmbedDim(dim));
    }
    if grid_h == 0 || grid_w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: grid_h,
            width: grid_w,
            channels: 1,
        });
    }

    let n = grid_h * grid_w;
    let dim_half = dim / 2; // split: first half H, second half W
    let dim_qtr = dim / 4; // sin/cos each get dim_qtr freqs

    let mut out = vec![0.0f32; n * dim];

    // Temperature = 10000^(2k / dim_half), k ∈ [0, dim_qtr)
    let freqs: Vec<f32> = (0..dim_qtr)
        .map(|k| 1.0 / 10000_f32.powf(2.0 * k as f32 / dim_half as f32))
        .collect();

    for h in 0..grid_h {
        for w in 0..grid_w {
            let pos = h * grid_w + w;
            let base = pos * dim;

            // H-axis encoding: indices [0, dim_qtr) sin, [dim_qtr, dim_half) cos
            for k in 0..dim_qtr {
                let angle = h as f32 * freqs[k];
                out[base + k] = angle.sin();
                out[base + dim_qtr + k] = angle.cos();
            }

            // W-axis encoding: indices [dim_half, dim_half+dim_qtr) sin, ...
            for k in 0..dim_qtr {
                let angle = w as f32 * freqs[k];
                out[base + dim_half + k] = angle.sin();
                out[base + dim_half + dim_qtr + k] = angle.cos();
            }
        }
    }

    Ok(out)
}

// ─── Learnable position embedding ────────────────────────────────────────────

/// Learnable position embedding table: `[n_positions, embed_dim]`.
///
/// Row `i` is the position embedding for the `i`-th token (index 0 is
/// conventionally the CLS token position).
#[derive(Debug, Clone)]
pub struct LearnablePosEmbed {
    /// Flat `[n_positions × embed_dim]` parameter table.
    pub table: Vec<f32>,
    /// Number of positions (including CLS if present).
    pub n_positions: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl LearnablePosEmbed {
    /// Create a learnable position embedding with small Gaussian init.
    pub fn new(n_positions: usize, embed_dim: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if n_positions == 0 {
            return Err(VisionError::EmptyInput("n_positions"));
        }
        let mut table = vec![0.0f32; n_positions * embed_dim];
        rng.fill_normal(&mut table);
        let scale = 0.02;
        for v in &mut table {
            *v *= scale;
        }
        Ok(Self {
            table,
            n_positions,
            embed_dim,
        })
    }

    /// Return the embedding for position `i` as a slice of length `embed_dim`.
    pub fn position_embedding(&self, i: usize) -> VisionResult<&[f32]> {
        if i >= self.n_positions {
            return Err(VisionError::DimensionMismatch {
                expected: self.n_positions - 1,
                got: i,
            });
        }
        let start = i * self.embed_dim;
        Ok(&self.table[start..start + self.embed_dim])
    }
}

// ─── add_pos_embed ────────────────────────────────────────────────────────────

/// Add positional embeddings to a token sequence in-place.
///
/// `tokens` is flat `[n_tokens × embed_dim]`.
/// `pos_embed` is flat `[n_tokens × embed_dim]` (or a prefix thereof).
///
/// Validates shape compatibility and returns an error on mismatch.
pub fn add_pos_embed(tokens: &mut [f32], pos_embed: &[f32], embed_dim: usize) -> VisionResult<()> {
    if tokens.len() != pos_embed.len() {
        return Err(VisionError::DimensionMismatch {
            expected: tokens.len(),
            got: pos_embed.len(),
        });
    }
    if embed_dim == 0 || tokens.len() % embed_dim != 0 {
        return Err(VisionError::InvalidEmbedDim(embed_dim));
    }
    for (t, p) in tokens.iter_mut().zip(pos_embed.iter()) {
        *t += p;
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn pos_2d_sincos_shape() {
        let pe = pos_2d_sincos(4, 4, 64).expect("ok");
        assert_eq!(pe.len(), 4 * 4 * 64); // 16 positions × 64 dims
    }

    #[test]
    fn pos_2d_sincos_finite() {
        let pe = pos_2d_sincos(8, 8, 64).expect("ok");
        assert!(pe.iter().all(|v| v.is_finite()), "non-finite pos embed");
    }

    #[test]
    fn pos_2d_sincos_in_range() {
        let pe = pos_2d_sincos(4, 4, 64).expect("ok");
        // sin/cos values are in [-1, 1]
        assert!(
            pe.iter().all(|&v| (-1.0f32..=1.0).contains(&v)),
            "out of [-1,1]"
        );
    }

    #[test]
    fn pos_2d_sincos_invalid_dim_not_div4() {
        let r = pos_2d_sincos(4, 4, 6); // 6 % 4 != 0
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(6))));
    }

    #[test]
    fn pos_2d_sincos_invalid_grid_zero() {
        let r = pos_2d_sincos(0, 4, 64);
        assert!(matches!(r, Err(VisionError::InvalidImageSize { .. })));
    }

    #[test]
    fn pos_2d_sincos_distinct_positions() {
        let pe = pos_2d_sincos(4, 4, 64).expect("ok");
        let embed_dim = 64;
        // Position (0,0) and (0,1) should differ
        let p00 = &pe[0..embed_dim];
        let p01 = &pe[embed_dim..2 * embed_dim];
        let diff: f32 = p00.iter().zip(p01.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "adjacent positions should differ; total diff={diff}"
        );
    }

    #[test]
    fn pos_2d_sincos_periodicity_check() {
        // The first dimension encodes frequency 1.0 (k=0, freq=1/10000^0=1),
        // so index 0 for position (h,w) is sin(h * 1.0).
        let pe = pos_2d_sincos(4, 1, 4).expect("ok"); // 4 rows, 1 col, dim=4
        // Position h=0: sin(0*1)=0
        assert!((pe[0] - 0.0_f32.sin()).abs() < 1e-6);
        // Position h=1: sin(1*1)=sin(1)
        assert!((pe[4] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn learnable_pos_embed_shape() {
        let mut rng = LcgRng::new(1);
        let lpe = LearnablePosEmbed::new(65, 64, &mut rng).expect("ok"); // 64 patches + CLS
        assert_eq!(lpe.table.len(), 65 * 64);
    }

    #[test]
    fn learnable_pos_embed_finite() {
        let mut rng = LcgRng::new(2);
        let lpe = LearnablePosEmbed::new(17, 32, &mut rng).expect("ok");
        assert!(lpe.table.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn learnable_pos_embed_access() {
        let mut rng = LcgRng::new(3);
        let lpe = LearnablePosEmbed::new(8, 16, &mut rng).expect("ok");
        let emb = lpe.position_embedding(3).expect("ok");
        assert_eq!(emb.len(), 16);
        assert_eq!(emb, &lpe.table[3 * 16..4 * 16]);
    }

    #[test]
    fn learnable_pos_embed_out_of_bounds_errors() {
        let mut rng = LcgRng::new(4);
        let lpe = LearnablePosEmbed::new(8, 16, &mut rng).expect("ok");
        let r = lpe.position_embedding(8);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn add_pos_embed_in_place() {
        let mut tokens = vec![1.0f32; 4 * 8]; // 4 tokens, dim=8
        let pos = vec![0.5f32; 4 * 8];
        add_pos_embed(&mut tokens, &pos, 8).expect("ok");
        assert!(tokens.iter().all(|&v| (v - 1.5).abs() < 1e-6));
    }

    #[test]
    fn add_pos_embed_shape_mismatch_errors() {
        let mut tokens = vec![1.0f32; 4 * 8];
        let pos = vec![0.5f32; 3 * 8]; // wrong size
        let r = add_pos_embed(&mut tokens, &pos, 8);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }
}
