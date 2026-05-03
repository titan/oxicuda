//! 1-D patch embedding for PatchTST and TimesNet.
//!
//! Extracts overlapping (or non-overlapping) fixed-length patches from a
//! univariate or multivariate `[T, C]` time series, then projects each patch
//! through a linear layer to produce a `[num_patches, d_model]` embedding.
//!
//! For PatchTST each variate is processed independently; the input to this
//! module is a single-variate `[T]` series (the caller loops over variates).
//! For convenience a multivariate wrapper is also provided.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

/// 1-D patch embedding parameters.
#[derive(Debug, Clone)]
pub struct PatchEmbed1d {
    /// Linear projection weight `[d_model, patch_len]`.
    pub weight: Vec<f32>,
    /// Linear projection bias `[d_model]`.
    pub bias: Vec<f32>,
    /// Length of each patch.
    pub patch_len: usize,
    /// Stride between patches.
    pub stride: usize,
    /// Output embedding dimension.
    pub d_model: usize,
}

impl PatchEmbed1d {
    /// Construct a `PatchEmbed1d` with Xavier-uniform initialised weight.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidPatchLen`] when `patch_len == 0`.
    /// - [`TsError::InvalidStride`] when `stride == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    pub fn new(
        patch_len: usize,
        stride: usize,
        d_model: usize,
        rng: &mut LcgRng,
    ) -> TsResult<Self> {
        if patch_len == 0 {
            return Err(TsError::InvalidPatchLen(0));
        }
        if stride == 0 {
            return Err(TsError::InvalidStride(0));
        }
        if d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }

        let scale = (6.0_f32 / (d_model + patch_len) as f32).sqrt();
        let mut weight = vec![0.0_f32; d_model * patch_len];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        let bias = vec![0.0_f32; d_model];

        Ok(Self {
            weight,
            bias,
            patch_len,
            stride,
            d_model,
        })
    }

    /// Compute the number of patches for a given sequence length.
    ///
    /// Returns `(T - patch_len) / stride + 1` when `T >= patch_len`,
    /// otherwise 0.
    #[must_use]
    pub fn num_patches(&self, t: usize) -> usize {
        if t < self.patch_len {
            return 0;
        }
        (t - self.patch_len) / self.stride + 1
    }

    /// Extract patches and project.
    ///
    /// # Arguments
    ///
    /// * `series` — `[T]` univariate time series.
    ///
    /// Returns `[num_patches, d_model]` flat row-major.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `series.len() < patch_len`.
    pub fn forward(&self, series: &[f32]) -> TsResult<Vec<f32>> {
        let t = series.len();
        if t < self.patch_len {
            return Err(TsError::InvalidSequenceLength(t));
        }

        let np = self.num_patches(t);
        let mut out = vec![0.0_f32; np * self.d_model];

        for p in 0..np {
            let t_start = p * self.stride;
            let patch = &series[t_start..t_start + self.patch_len];

            for di in 0..self.d_model {
                let w_row = &self.weight[di * self.patch_len..(di + 1) * self.patch_len];
                let val: f32 = w_row.iter().zip(patch.iter()).map(|(&w, &x)| w * x).sum();
                out[p * self.d_model + di] = val + self.bias[di];
            }
        }
        Ok(out)
    }

    /// Project all variates independently.
    ///
    /// # Arguments
    ///
    /// * `features` — `[T, C]` row-major multivariate series.
    /// * `t` — sequence length.
    /// * `c` — number of variates.
    ///
    /// Returns `[C, num_patches, d_model]` flat (variate-first layout).
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `features.len() != t * c`.
    /// - [`TsError::InvalidSequenceLength`] when `t < patch_len`.
    pub fn forward_mv(&self, features: &[f32], t: usize, c: usize) -> TsResult<Vec<f32>> {
        if features.len() != t * c {
            return Err(TsError::DimensionMismatch {
                expected: t * c,
                got: features.len(),
            });
        }
        let np = self.num_patches(t);
        let mut out = vec![0.0_f32; c * np * self.d_model];

        for ci in 0..c {
            let series: Vec<f32> = (0..t).map(|ti| features[ti * c + ci]).collect();
            let emb = self.forward(&series)?;
            let offset = ci * np * self.d_model;
            out[offset..offset + np * self.d_model].copy_from_slice(&emb);
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn patch_embed_num_patches_exact() {
        let mut rng = make_rng();
        let pe = PatchEmbed1d::new(16, 8, 64, &mut rng).expect("ok");
        // T=96: (96 - 16) / 8 + 1 = 11
        assert_eq!(pe.num_patches(96), 11);
    }

    #[test]
    fn patch_embed_num_patches_zero_when_too_short() {
        let mut rng = make_rng();
        let pe = PatchEmbed1d::new(16, 8, 64, &mut rng).expect("ok");
        assert_eq!(pe.num_patches(15), 0);
    }

    #[test]
    fn patch_embed_forward_shape() {
        let mut rng = make_rng();
        let pe = PatchEmbed1d::new(16, 8, 64, &mut rng).expect("ok");
        let series = vec![1.0_f32; 96];
        let out = pe.forward(&series).expect("ok");
        assert_eq!(out.len(), pe.num_patches(96) * 64);
    }

    #[test]
    fn patch_embed_forward_finite() {
        let mut rng = make_rng();
        let pe = PatchEmbed1d::new(8, 4, 32, &mut rng).expect("ok");
        let mut series = vec![0.0_f32; 64];
        rng.fill_normal(&mut series);
        let out = pe.forward(&series).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn patch_embed_mv_shape() {
        let mut rng = make_rng();
        let pe = PatchEmbed1d::new(16, 8, 64, &mut rng).expect("ok");
        let t = 96;
        let c = 4;
        let features = vec![0.5_f32; t * c];
        let out = pe.forward_mv(&features, t, c).expect("ok");
        let np = pe.num_patches(t);
        assert_eq!(out.len(), c * np * 64);
    }

    #[test]
    fn patch_embed_zero_patch_len_error() {
        let mut rng = make_rng();
        assert!(matches!(
            PatchEmbed1d::new(0, 4, 32, &mut rng).unwrap_err(),
            TsError::InvalidPatchLen(0)
        ));
    }

    #[test]
    fn patch_embed_zero_stride_error() {
        let mut rng = make_rng();
        assert!(matches!(
            PatchEmbed1d::new(8, 0, 32, &mut rng).unwrap_err(),
            TsError::InvalidStride(0)
        ));
    }

    #[test]
    fn patch_embed_zero_d_model_error() {
        let mut rng = make_rng();
        assert!(matches!(
            PatchEmbed1d::new(8, 4, 0, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    #[test]
    fn patch_embed_too_short_error() {
        let mut rng = make_rng();
        let pe = PatchEmbed1d::new(16, 8, 32, &mut rng).expect("ok");
        let series = vec![0.0_f32; 10]; // shorter than patch_len
        assert!(matches!(
            pe.forward(&series).unwrap_err(),
            TsError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn patch_embed_non_overlapping() {
        let mut rng = make_rng();
        // stride = patch_len → non-overlapping
        let pe = PatchEmbed1d::new(8, 8, 16, &mut rng).expect("ok");
        let series: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out = pe.forward(&series).expect("ok");
        // (32 - 8) / 8 + 1 = 4 patches
        assert_eq!(out.len(), 4 * 16);
    }
}
