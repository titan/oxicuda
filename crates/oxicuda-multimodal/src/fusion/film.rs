//! FiLM: Feature-wise Linear Modulation (Perez et al. 2018).
//!
//! Implements the conditioning mechanism from:
//! Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer."
//! AAAI 2018.
//!
//! A conditioning vector `z` (e.g. a language embedding) is mapped through two
//! learned affine projections to per-feature scale `γ(z)` and shift `β(z)`,
//! which then modulate a target feature map `x` (e.g. visual features):
//!
//! ```text
//! γ = W_γ · z + b_γ      (gain,  shape [d_feature])
//! β = W_β · z + b_β       (bias,  shape [d_feature])
//! FiLM(x | z) = γ ⊙ x + β
//! ```
//!
//! The modulation is applied **identically across spatial / token positions** of
//! `x` and varies only along the feature axis — the defining property of FiLM.
//! Initialising `W_γ = 0`, `b_γ = 1`, `W_β = 0`, `b_β = 0` yields the identity
//! transform (`γ = 1, β = 0`), provided by [`FilmGenerator::identity`].

use crate::error::{MmResult, MultiModalError};

/// FiLM parameter generator: maps a conditioning vector to `(γ, β)`.
///
/// Holds the two affine projections `W_γ, b_γ` and `W_β, b_β`.
#[derive(Debug, Clone)]
pub struct FilmGenerator {
    /// `W_γ`: `[d_cond × d_feature]` row-major.
    pub w_gamma: Vec<f32>,
    /// `b_γ`: `[d_feature]`.
    pub b_gamma: Vec<f32>,
    /// `W_β`: `[d_cond × d_feature]` row-major.
    pub w_beta: Vec<f32>,
    /// `b_β`: `[d_feature]`.
    pub b_beta: Vec<f32>,
    /// Conditioning-vector dimension.
    pub d_cond: usize,
    /// Modulated-feature dimension.
    pub d_feature: usize,
}

impl FilmGenerator {
    /// Create a generator with all-zero weights and biases.
    #[must_use]
    pub fn zeros(d_cond: usize, d_feature: usize) -> Self {
        Self {
            w_gamma: vec![0.0_f32; d_cond * d_feature],
            b_gamma: vec![0.0_f32; d_feature],
            w_beta: vec![0.0_f32; d_cond * d_feature],
            b_beta: vec![0.0_f32; d_feature],
            d_cond,
            d_feature,
        }
    }

    /// Create an identity generator (`γ ≡ 1`, `β ≡ 0` for any conditioning).
    ///
    /// `W_γ = W_β = 0`, `b_γ = 1`, `b_β = 0`.
    #[must_use]
    pub fn identity(d_cond: usize, d_feature: usize) -> Self {
        let mut g = Self::zeros(d_cond, d_feature);
        for v in g.b_gamma.iter_mut() {
            *v = 1.0;
        }
        g
    }

    /// Compute `(γ, β)` for a single conditioning vector `z` (`[d_cond]`).
    ///
    /// Returns `(gamma, beta)`, each `[d_feature]`.
    ///
    /// # Errors
    /// Returns [`MultiModalError::DimensionMismatch`] when `z.len() != d_cond`.
    pub fn generate(&self, z: &[f32]) -> MmResult<(Vec<f32>, Vec<f32>)> {
        if z.len() != self.d_cond {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_cond,
                got: z.len(),
            });
        }
        let mut gamma = self.b_gamma.clone();
        let mut beta = self.b_beta.clone();
        for i in 0..self.d_cond {
            let zi = z[i];
            let g_row = &self.w_gamma[i * self.d_feature..(i + 1) * self.d_feature];
            let b_row = &self.w_beta[i * self.d_feature..(i + 1) * self.d_feature];
            for f in 0..self.d_feature {
                gamma[f] += zi * g_row[f];
                beta[f] += zi * b_row[f];
            }
        }
        Ok((gamma, beta))
    }

    /// Apply FiLM modulation to a feature map `x` conditioned on `z`.
    ///
    /// `x` has shape `[n_positions × d_feature]` (row-major); the same `(γ, β)`
    /// is broadcast across all positions. Returns `[n_positions × d_feature]`.
    ///
    /// # Errors
    /// Returns [`MultiModalError`] for a `z` / `d_cond` or
    /// `x` / `n_positions·d_feature` mismatch.
    pub fn forward(&self, x: &[f32], z: &[f32], n_positions: usize) -> MmResult<Vec<f32>> {
        if x.len() != n_positions * self.d_feature {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_positions * self.d_feature,
                got: x.len(),
            });
        }
        if n_positions == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        let (gamma, beta) = self.generate(z)?;
        let out = apply_film(x, &gamma, &beta, n_positions)?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(MultiModalError::NanEncountered { location: "film" });
        }
        Ok(out)
    }
}

/// Apply precomputed FiLM parameters `γ ⊙ x + β` over a feature map.
///
/// `x` is `[n_positions × d_feature]`; `gamma` and `beta` are `[d_feature]`,
/// broadcast across positions.
///
/// # Errors
/// Returns [`MultiModalError::DimensionMismatch`] on any inconsistent length.
pub fn apply_film(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n_positions: usize,
) -> MmResult<Vec<f32>> {
    let d_feature = gamma.len();
    if beta.len() != d_feature {
        return Err(MultiModalError::DimensionMismatch {
            expected: d_feature,
            got: beta.len(),
        });
    }
    if x.len() != n_positions * d_feature {
        return Err(MultiModalError::DimensionMismatch {
            expected: n_positions * d_feature,
            got: x.len(),
        });
    }
    let mut out = vec![0.0_f32; x.len()];
    for p in 0..n_positions {
        let base = p * d_feature;
        for f in 0..d_feature {
            out[base + f] = gamma[f] * x[base + f] + beta[f];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passes_through() {
        let film_gen = FilmGenerator::identity(4, 3);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 positions × 3
        let z = vec![0.5_f32, -0.3, 0.7, 0.1];
        let out = film_gen.forward(&x, &z, 2).expect("forward should succeed");
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-6, "identity FiLM should be a no-op");
        }
    }

    #[test]
    fn output_shape_preserved() {
        let film_gen = FilmGenerator::zeros(8, 5);
        let n_pos = 7;
        let x = vec![0.3_f32; n_pos * 5];
        let z = vec![0.2_f32; 8];
        let out = film_gen
            .forward(&x, &z, n_pos)
            .expect("forward should succeed");
        assert_eq!(out.len(), n_pos * 5);
    }

    #[test]
    fn zero_generator_gives_beta_only() {
        // zeros → γ=0, β=0 → output all zeros regardless of x.
        let film_gen = FilmGenerator::zeros(4, 3);
        let x = vec![9.0_f32; 2 * 3];
        let z = vec![1.0_f32; 4];
        let out = film_gen.forward(&x, &z, 2).expect("forward should succeed");
        assert!(out.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn constant_gamma_beta_via_bias() {
        // Set b_gamma = 2, b_beta = 1, W=0 → out = 2x + 1.
        let mut film_gen = FilmGenerator::zeros(2, 3);
        film_gen.b_gamma = vec![2.0_f32; 3];
        film_gen.b_beta = vec![1.0_f32; 3];
        let x = vec![3.0_f32, 4.0, 5.0];
        let z = vec![10.0_f32, -7.0]; // W=0 → z irrelevant
        let out = film_gen.forward(&x, &z, 1).expect("forward should succeed");
        assert!((out[0] - 7.0).abs() < 1e-6); // 2*3+1
        assert!((out[1] - 9.0).abs() < 1e-6); // 2*4+1
        assert!((out[2] - 11.0).abs() < 1e-6); // 2*5+1
    }

    #[test]
    fn broadcast_same_across_positions() {
        // For uniform x rows, every output row must be identical.
        let mut film_gen = FilmGenerator::identity(2, 4);
        film_gen.b_beta = vec![0.5_f32; 4];
        let x = vec![1.0_f32; 3 * 4];
        let z = vec![0.0_f32; 2];
        let out = film_gen.forward(&x, &z, 3).expect("forward should succeed");
        let row0 = &out[0..4];
        for p in 1..3 {
            assert_eq!(&out[p * 4..(p + 1) * 4], row0);
        }
    }

    #[test]
    fn conditioning_affects_gamma() {
        // W_gamma row 0 = [1,0]; z[0]=3 → gamma[0] = b_gamma[0] + 3.
        let mut film_gen = FilmGenerator::zeros(2, 2);
        film_gen.w_gamma = vec![1.0_f32, 0.0, 0.0, 0.0]; // [d_cond=2 × d_feature=2]
        film_gen.b_gamma = vec![1.0_f32, 1.0];
        let z = vec![3.0_f32, 0.0];
        let (gamma, _beta) = film_gen.generate(&z).expect("generate should succeed");
        assert!((gamma[0] - 4.0).abs() < 1e-6, "gamma0={}", gamma[0]);
        assert!((gamma[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn generate_dim_mismatch_errors() {
        let film_gen = FilmGenerator::zeros(4, 3);
        let z = vec![0.0_f32; 3]; // wrong d_cond
        assert!(matches!(
            film_gen.generate(&z),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_x_dim_mismatch_errors() {
        let film_gen = FilmGenerator::zeros(4, 3);
        let x = vec![0.0_f32; 2 * 4]; // wrong feature dim
        let z = vec![0.0_f32; 4];
        assert!(matches!(
            film_gen.forward(&x, &z, 2),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_zero_positions_errors() {
        let film_gen = FilmGenerator::zeros(4, 3);
        let x: Vec<f32> = vec![];
        let z = vec![0.0_f32; 4];
        assert!(matches!(
            film_gen.forward(&x, &z, 0),
            Err(MultiModalError::EmptyInput)
        ));
    }

    #[test]
    fn apply_film_beta_mismatch_errors() {
        let x = vec![0.0_f32; 6];
        let gamma = vec![1.0_f32; 3];
        let beta = vec![0.0_f32; 2]; // wrong
        assert!(matches!(
            apply_film(&x, &gamma, &beta, 2),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn output_finite_for_large_input() {
        let mut film_gen = FilmGenerator::identity(2, 4);
        film_gen.b_gamma = vec![3.0_f32; 4];
        let x = vec![1e6_f32; 2 * 4];
        let z = vec![1.0_f32; 2];
        let out = film_gen.forward(&x, &z, 2).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
