//! Perpendicular Negative Guidance (Miyake et al. 2023).
//!
//! Removes the component of the negative prompt that is perpendicular
//! to the positive conditional noise, then applies CFG on the result.

use crate::error::{GenError, GenResult};

// ─── PerpNegGuidance ──────────────────────────────────────────────────────────

/// Perpendicular negative guidance combiner.
///
/// Computes: `ε̂_⊥ = ε_c - neg_scale * (ε_n · ε_c / |ε_c|²) * ε_c`
/// then applies CFG: `ε̂ = ε_u + scale * (ε̂_⊥ - ε_u)`
///
/// # Reference
/// Miyake et al., "Negative-prompt Inversion: Fast Image Inversion for
/// Editing with Text-Guided Diffusion Models", arXiv 2023.
#[derive(Debug, Clone)]
pub struct PerpNegGuidance {
    scale: f32,
    neg_scale: f32,
}

impl PerpNegGuidance {
    /// Create a new perpendicular negative guidance combiner.
    ///
    /// # Arguments
    /// - `scale`: CFG guidance scale (must be >= 1.0).
    /// - `neg_scale`: Negative prompt strength (must be >= 0.0).
    ///
    /// # Errors
    /// - `InvalidGuidanceScale` if `scale < 1.0`
    pub fn new(scale: f32, neg_scale: f32) -> GenResult<Self> {
        if scale < 1.0 {
            return Err(GenError::InvalidGuidanceScale(scale));
        }
        Ok(Self {
            scale,
            neg_scale: neg_scale.max(0.0),
        })
    }

    /// Apply perpendicular negative guidance.
    ///
    /// Algorithm:
    /// 1. Compute the projection of `neg` onto `cond`:
    ///    `proj = (neg · cond / |cond|²) * cond`
    /// 2. Remove the perpendicular component from `cond`:
    ///    `cond_perp = cond - neg_scale * proj`
    /// 3. Apply CFG: `out = uncond + scale * (cond_perp - uncond)`
    ///
    /// # Errors
    /// - `EmptyInput` if inputs are empty
    /// - `DimensionMismatch` if shapes differ
    pub fn apply(&self, cond: &[f32], uncond: &[f32], neg: &[f32]) -> GenResult<Vec<f32>> {
        if cond.is_empty() {
            return Err(GenError::EmptyInput("cond is empty"));
        }
        if cond.len() != uncond.len() {
            return Err(GenError::DimensionMismatch {
                expected: cond.len(),
                got: uncond.len(),
            });
        }
        if cond.len() != neg.len() {
            return Err(GenError::DimensionMismatch {
                expected: cond.len(),
                got: neg.len(),
            });
        }

        // Compute the projection coefficient: (neg · cond) / |cond|²
        let dot_neg_cond = Self::dot(neg, cond);
        let norm_sq_cond = Self::norm_sq(cond);

        // Compute perp-adjusted cond
        let cond_perp: Vec<f32> = if norm_sq_cond > 1e-10 {
            let proj_coeff = dot_neg_cond / norm_sq_cond;
            cond.iter()
                .map(|&c| c - self.neg_scale * proj_coeff * c)
                .collect()
        } else {
            cond.to_vec()
        };

        // Apply CFG: out = uncond + scale * (cond_perp - uncond)
        let result = uncond
            .iter()
            .zip(&cond_perp)
            .map(|(&u, &cp)| u + self.scale * (cp - u))
            .collect();
        Ok(result)
    }

    /// Apply perpendicular negative guidance with separate channel processing.
    ///
    /// Splits the inputs into `n_chunks` channels and applies perp-neg
    /// independently per channel for more localised control.
    pub fn apply_chunked(
        &self,
        cond: &[f32],
        uncond: &[f32],
        neg: &[f32],
        n_chunks: usize,
    ) -> GenResult<Vec<f32>> {
        if cond.is_empty() {
            return Err(GenError::EmptyInput("cond is empty"));
        }
        if n_chunks == 0 {
            return Err(GenError::EmptyInput("n_chunks must be > 0"));
        }
        if cond.len() % n_chunks != 0 {
            return Err(GenError::DimensionMismatch {
                expected: cond.len() - cond.len() % n_chunks,
                got: cond.len(),
            });
        }
        if cond.len() != uncond.len() || cond.len() != neg.len() {
            return Err(GenError::DimensionMismatch {
                expected: cond.len(),
                got: uncond.len(),
            });
        }
        let chunk_size = cond.len() / n_chunks;
        let mut result = Vec::with_capacity(cond.len());
        for i in 0..n_chunks {
            let lo = i * chunk_size;
            let hi = lo + chunk_size;
            let chunk = self.apply(&cond[lo..hi], &uncond[lo..hi], &neg[lo..hi])?;
            result.extend(chunk);
        }
        Ok(result)
    }

    /// Compute the dot product of two slices.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    /// Compute the squared L2 norm of a slice.
    fn norm_sq(a: &[f32]) -> f32 {
        a.iter().map(|&x| x * x).sum()
    }

    /// Return the CFG guidance scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Return the negative guidance strength.
    pub fn neg_scale(&self) -> f32 {
        self.neg_scale
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn zero_neg_scale_equals_cfg() {
        let guide = PerpNegGuidance::new(3.0, 0.0).unwrap();
        let cond = vec![1.0_f32, 2.0, 3.0];
        let uncond = vec![0.0_f32; 3];
        let neg = vec![5.0_f32; 3];
        let out = guide.apply(&cond, &uncond, &neg).unwrap();
        // neg_scale=0 → cond_perp = cond → out = uncond + scale*(cond - uncond) = scale*cond
        for (&o, &c) in out.iter().zip(&cond) {
            assert!((o - 3.0 * c).abs() < EPS, "{o} != 3*{c}");
        }
    }

    #[test]
    fn output_shape_matches_input() {
        let guide = PerpNegGuidance::new(2.0, 1.0).unwrap();
        let cond = vec![1.0_f32; 64];
        let uncond = vec![0.0_f32; 64];
        let neg = vec![0.5_f32; 64];
        let out = guide.apply(&cond, &uncond, &neg).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn invalid_scale_rejected() {
        assert!(PerpNegGuidance::new(0.5, 1.0).is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let guide = PerpNegGuidance::new(2.0, 1.0).unwrap();
        let cond = vec![0.0_f32; 8];
        let uncond = vec![0.0_f32; 4];
        let neg = vec![0.0_f32; 8];
        assert!(matches!(
            guide.apply(&cond, &uncond, &neg),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn output_is_finite() {
        let guide = PerpNegGuidance::new(5.0, 2.0).unwrap();
        let cond: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
        let uncond = vec![0.0_f32; 32];
        let neg: Vec<f32> = (0..32).map(|i| -(i as f32) / 32.0).collect();
        let out = guide.apply(&cond, &uncond, &neg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn parallel_neg_reduces_amplitude() {
        // When neg is proportional to cond, the perpendicular component should reduce it
        let guide = PerpNegGuidance::new(1.0, 1.0).unwrap();
        let cond = vec![1.0_f32; 4];
        let uncond = vec![0.0_f32; 4];
        let neg = vec![1.0_f32; 4]; // parallel to cond
        let out = guide.apply(&cond, &uncond, &neg).unwrap();
        // perp_cond = cond - 1.0 * (1/1)*cond = 0 → out = uncond + scale*0 = 0
        for &v in &out {
            assert!(v.abs() < EPS, "parallel neg should zero out cond: {v}");
        }
    }

    #[test]
    fn perp_neg_orthogonal_neg_unchanged() {
        // When neg is orthogonal to cond, the projection is 0, so cond is unchanged
        let guide = PerpNegGuidance::new(1.0, 1.0).unwrap();
        let cond = vec![1.0_f32, 0.0]; // along x-axis
        let uncond = vec![0.0_f32; 2];
        let neg = vec![0.0_f32, 1.0]; // along y-axis (orthogonal)
        let out = guide.apply(&cond, &uncond, &neg).unwrap();
        // dot(neg, cond) = 0 → proj = 0 → cond_perp = cond
        // out = 0 + 1*(cond - 0) = cond
        for (&o, &c) in out.iter().zip(&cond) {
            assert!((o - c).abs() < EPS, "{o} != {c}");
        }
    }

    #[test]
    fn chunked_equals_unchunked() {
        let guide = PerpNegGuidance::new(2.0, 0.5).unwrap();
        let cond: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let uncond = vec![0.0_f32; 8];
        let neg: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5).collect();
        let unchunked = guide.apply(&cond, &uncond, &neg).unwrap();
        // When chunked with chunk_size=full_len, result should be same
        // Actually chunked with n_chunks=1 is same as unchunked
        let chunked = guide.apply_chunked(&cond, &uncond, &neg, 1).unwrap();
        for (&u, &c) in unchunked.iter().zip(&chunked) {
            assert!((u - c).abs() < EPS, "chunked(1) should equal unchunked");
        }
    }

    #[test]
    fn scale_accessor() {
        let guide = PerpNegGuidance::new(3.5, 1.2).unwrap();
        assert!((guide.scale() - 3.5).abs() < EPS);
        assert!((guide.neg_scale() - 1.2).abs() < EPS);
    }
}
