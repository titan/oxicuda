//! Fisher Information Geometry for Gaussian families.
//!
//! The Fisher information metric gives a Riemannian structure to statistical manifolds.
//! For the family of isotropic Gaussians `N(μ, σ² I_d)` parameterised by `(μ, σ)`,
//! the Fisher information matrix is diagonal:
//!
//! ```text
//! F(μ, σ) = diag( [1/σ², …, 1/σ²]_d μ-block,  [2/σ²]_d σ-block )
//! ```
//!
//! The geodesic distance (Fisher-Rao metric) between two Gaussians with the same σ
//! reduces to: `d = sqrt(2) * || μ₁ - μ₂ || / σ`.
//!
//! # References
//! - Rao, C. R. (1945). Information and the accuracy attainable in the estimation
//!   of statistical parameters. *Bull. Calcutta Math. Soc.*, 37, 81–91.
//! - Amari, S. (2016). *Information Geometry and Its Applications*. Springer.

use crate::error::{ManifoldError, ManifoldResult};

/// Compute the diagonal of the Fisher information matrix for an isotropic Gaussian.
///
/// The parameterisation is `θ = (μ₁, …, μ_d, σ₁, …, σ_d)` where each `σ_i = σ`
/// (shared standard deviation). The Fisher metric block-diagonalises:
///
/// - **Mean block**: `F_{μᵢ, μᵢ} = 1 / σ²` for `i = 1, …, d`.
/// - **Sigma block**: `F_{σᵢ, σᵢ} = 2 / σ²` for `i = 1, …, d`.
///
/// # Returns
/// A vector of length `2d`: `[1/σ², …, 1/σ²  (d entries), 2/σ², …, 2/σ² (d entries)]`.
///
/// # Errors
/// [`ManifoldError::InvalidParameter`] if `sigma <= 0` or `d == 0`.
pub fn gaussian_fisher_info(sigma: f64, d: usize) -> ManifoldResult<Vec<f64>> {
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(ManifoldError::InvalidParameter {
            name: "sigma".into(),
            reason: format!("must be > 0 and finite, got {sigma}"),
        });
    }
    if d == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "d".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    let inv_s2 = 1.0 / (sigma * sigma);
    let mut f = Vec::with_capacity(2 * d);
    // mean block
    for _ in 0..d {
        f.push(inv_s2);
    }
    // sigma block
    for _ in 0..d {
        f.push(2.0 * inv_s2);
    }
    Ok(f)
}

/// Fisher-Rao geodesic distance between two Gaussian means with the same σ.
///
/// For `N(μ₁, σ² I_d)` and `N(μ₂, σ² I_d)`:
/// `d_FR(μ₁, μ₂) = sqrt(2) * || μ₁ - μ₂ ||₂ / σ`
///
/// # Arguments
/// - `params1`, `params2`: mean vectors of length `d`.
/// - `sigma`: common standard deviation.
///
/// # Errors
/// - [`ManifoldError::DimensionMismatch`] if `params1.len() != params2.len()`.
/// - [`ManifoldError::InvalidParameter`] if `sigma <= 0`.
pub fn fisher_rao_distance(params1: &[f64], params2: &[f64], sigma: f64) -> ManifoldResult<f64> {
    if params1.len() != params2.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: params1.len(),
            b: params2.len(),
        });
    }
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(ManifoldError::InvalidParameter {
            name: "sigma".into(),
            reason: format!("must be > 0 and finite, got {sigma}"),
        });
    }
    let sq_dist: f64 = params1
        .iter()
        .zip(params2.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    Ok((2.0 * sq_dist).sqrt() / sigma)
}

/// Natural gradient pre-conditioner: `F⁻¹ grad`.
///
/// For a diagonal Fisher matrix the natural gradient is simply `grad[i] / F[i]`.
#[derive(Debug, Clone)]
pub struct NaturalGradient {
    fisher_diag: Vec<f64>,
}

impl NaturalGradient {
    /// Construct a natural gradient operator from a diagonal Fisher matrix.
    ///
    /// # Errors
    /// [`ManifoldError::InvalidParameter`] if any diagonal entry is ≤ 0 or non-finite.
    pub fn new(fisher_diag: Vec<f64>) -> ManifoldResult<Self> {
        if fisher_diag.is_empty() {
            return Err(ManifoldError::InvalidParameter {
                name: "fisher_diag".into(),
                reason: "must be non-empty".into(),
            });
        }
        for (i, &f) in fisher_diag.iter().enumerate() {
            if f <= 0.0 || !f.is_finite() {
                return Err(ManifoldError::InvalidParameter {
                    name: format!("fisher_diag[{i}]"),
                    reason: format!("must be > 0 and finite, got {f}"),
                });
            }
        }
        Ok(Self { fisher_diag })
    }

    /// Apply the inverse Fisher metric: returns `F⁻¹ * grad`.
    ///
    /// # Errors
    /// [`ManifoldError::DimensionMismatch`] if `grad.len() != fisher_diag.len()`.
    pub fn apply(&self, grad: &[f64]) -> ManifoldResult<Vec<f64>> {
        if grad.len() != self.fisher_diag.len() {
            return Err(ManifoldError::DimensionMismatch {
                a: self.fisher_diag.len(),
                b: grad.len(),
            });
        }
        Ok(grad
            .iter()
            .zip(self.fisher_diag.iter())
            .map(|(&g, &f)| g / f)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_info_shape() {
        let f = gaussian_fisher_info(2.0, 3).expect("ok");
        assert_eq!(f.len(), 6, "2*d = 6 for d=3");
    }

    #[test]
    fn fisher_info_positive() {
        let f = gaussian_fisher_info(1.5, 4).expect("ok");
        for &v in &f {
            assert!(v > 0.0, "Fisher info must be positive");
        }
    }

    #[test]
    fn distance_zero_for_equal() {
        let mu = vec![1.0, 2.0, 3.0];
        let d = fisher_rao_distance(&mu, &mu, 1.0).expect("ok");
        assert!(d.abs() < 1.0e-12, "distance to self must be 0");
    }

    #[test]
    fn distance_symmetric() {
        let mu1 = vec![0.0, 0.0];
        let mu2 = vec![1.0, 1.0];
        let d12 = fisher_rao_distance(&mu1, &mu2, 1.0).expect("ok");
        let d21 = fisher_rao_distance(&mu2, &mu1, 1.0).expect("ok");
        assert!((d12 - d21).abs() < 1.0e-12, "distance must be symmetric");
    }

    #[test]
    fn natural_gradient_shape() {
        let f_diag = vec![2.0, 4.0, 1.0];
        let ng = NaturalGradient::new(f_diag).expect("ok");
        let grad = vec![1.0, 1.0, 1.0];
        let nat = ng.apply(&grad).expect("ok");
        assert_eq!(nat.len(), 3);
    }

    #[test]
    fn natural_gradient_scales() {
        // For Fisher diag = [2, 4], grad = [2, 4] → nat_grad = [1, 1]
        let ng = NaturalGradient::new(vec![2.0, 4.0]).expect("ok");
        let nat = ng.apply(&[2.0, 4.0]).expect("ok");
        assert!((nat[0] - 1.0).abs() < 1.0e-12);
        assert!((nat[1] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn natural_gradient_fisher_zero_error() {
        let result = NaturalGradient::new(vec![1.0, 0.0, 1.0]);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "zero Fisher entry should fail"
        );
    }

    #[test]
    fn sigma_0_error() {
        let result = gaussian_fisher_info(0.0, 2);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "sigma=0 should fail"
        );
        let result2 = fisher_rao_distance(&[0.0], &[1.0], 0.0);
        assert!(
            matches!(result2, Err(ManifoldError::InvalidParameter { .. })),
            "sigma=0 in distance should fail"
        );
    }

    #[test]
    fn fisher_info_values_correct() {
        // sigma=2, d=2 → F = [1/4, 1/4, 2/4, 2/4] = [0.25, 0.25, 0.5, 0.5]
        let f = gaussian_fisher_info(2.0, 2).expect("ok");
        assert!((f[0] - 0.25).abs() < 1.0e-12);
        assert!((f[1] - 0.25).abs() < 1.0e-12);
        assert!((f[2] - 0.5).abs() < 1.0e-12);
        assert!((f[3] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn distance_formula_correct() {
        // mu1=[0], mu2=[1], sigma=1 → d = sqrt(2) * 1 / 1 = sqrt(2)
        let d = fisher_rao_distance(&[0.0], &[1.0], 1.0).expect("ok");
        let expected = std::f64::consts::SQRT_2;
        assert!(
            (d - expected).abs() < 1.0e-10,
            "got {d}, expected {expected}"
        );
    }
}
