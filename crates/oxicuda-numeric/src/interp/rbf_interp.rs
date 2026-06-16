//! Radial Basis Function (RBF) interpolation of scattered data in ℝⁿ.
//!
//! Hardy 1971. Given `m` centres `{x_i} ⊂ ℝᵈ` with values `{f_i}`, the
//! interpolant is `s(x) = Σ_i λ_i φ(‖x − x_i‖)` where `φ` is a radial kernel.
//! The weights `λ` solve the dense linear system `Φ λ = f` with
//! `Φ_{ij} = φ(‖x_i − x_j‖)`, factored here via LU with partial pivoting (a
//! small Tikhonov term is added to the diagonal to stabilise near-duplicate
//! centres). This handles arbitrarily scattered nodes — unlike the tensor-grid
//! splines elsewhere in `interp/`.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

/// Radial kernel family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RbfKernel {
    /// Gaussian `φ(r) = exp(−(r / ε)²)`.
    Gaussian,
    /// Multiquadric `φ(r) = √(r² + ε²)`.
    Multiquadric,
    /// Inverse multiquadric `φ(r) = 1 / √(r² + ε²)`.
    InverseMultiquadric,
    /// Thin-plate spline `φ(r) = r² ln r` (the shape parameter is ignored).
    ThinPlate,
}

impl RbfKernel {
    /// Evaluate the kernel at radius `r` with shape parameter `eps`.
    #[must_use]
    pub fn eval(self, r: f64, eps: f64) -> f64 {
        match self {
            RbfKernel::Gaussian => {
                let z = r / eps;
                (-(z * z)).exp()
            }
            RbfKernel::Multiquadric => (r * r + eps * eps).sqrt(),
            RbfKernel::InverseMultiquadric => 1.0 / (r * r + eps * eps).sqrt(),
            RbfKernel::ThinPlate => {
                if r <= 0.0 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
        }
    }
}

/// A fitted RBF interpolator over `m` centres in `dim`-dimensional space.
#[derive(Debug, Clone)]
pub struct RbfInterpolator {
    /// Flattened centre coordinates, `[m × dim]` row-major.
    centers: Vec<f64>,
    /// Interpolation weights `λ`, length `m`.
    weights: Vec<f64>,
    /// Spatial dimension `dim`.
    dim: usize,
    /// Kernel family.
    kernel: RbfKernel,
    /// Shape parameter `ε`.
    eps: f64,
}

/// Euclidean distance between two `dim`-vectors stored contiguously.
fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

impl RbfInterpolator {
    /// Fit an interpolator to `m` points.
    ///
    /// `points` is `[m × dim]` row-major and `values` has length `m`. `eps` is
    /// the kernel shape parameter (ignored by [`RbfKernel::ThinPlate`]).
    ///
    /// # Errors
    /// Returns [`NumericError::EmptyInput`] if `m == 0` or `dim == 0`,
    /// [`NumericError::ShapeMismatch`] if `points.len() != m·dim`,
    /// [`NumericError::DimensionMismatch`] if `values.len() != m`,
    /// [`NumericError::InvalidParameter`] if `eps <= 0` for a kernel that needs
    /// it, and [`NumericError::SingularMatrix`] if the system is unsolvable.
    pub fn fit(
        points: &[f64],
        values: &[f64],
        m: usize,
        dim: usize,
        kernel: RbfKernel,
        eps: f64,
    ) -> NumericResult<Self> {
        if m == 0 || dim == 0 {
            return Err(NumericError::EmptyInput);
        }
        if points.len() != m * dim {
            return Err(NumericError::ShapeMismatch {
                expected: vec![m, dim],
                got: vec![points.len()],
            });
        }
        if values.len() != m {
            return Err(NumericError::DimensionMismatch {
                a: values.len(),
                b: m,
            });
        }
        let needs_eps = !matches!(kernel, RbfKernel::ThinPlate);
        if needs_eps && (!eps.is_finite() || eps <= 0.0) {
            return Err(NumericError::InvalidParameter(format!(
                "RBF shape parameter eps must be positive, got {eps}"
            )));
        }

        // Assemble the symmetric interpolation matrix Φ with a tiny ridge.
        let ridge = 1e-12;
        let mut mat = vec![0.0_f64; m * m];
        for i in 0..m {
            let xi = &points[i * dim..(i + 1) * dim];
            for j in 0..m {
                let xj = &points[j * dim..(j + 1) * dim];
                let r = distance(xi, xj);
                let mut phi = kernel.eval(r, eps);
                if i == j {
                    phi += ridge;
                }
                mat[i * m + j] = phi;
            }
        }

        let (lu, piv, _sign) = lu_decompose(&mat, m)?;
        let weights = lu_solve(&lu, &piv, m, values)?;

        Ok(Self {
            centers: points.to_vec(),
            weights,
            dim,
            kernel,
            eps,
        })
    }

    /// Evaluate the interpolant at a query point `x` (length `dim`).
    ///
    /// # Errors
    /// Returns [`NumericError::DimensionMismatch`] if `x.len() != dim`.
    pub fn eval(&self, x: &[f64]) -> NumericResult<f64> {
        if x.len() != self.dim {
            return Err(NumericError::DimensionMismatch {
                a: x.len(),
                b: self.dim,
            });
        }
        let m = self.weights.len();
        let mut acc = 0.0_f64;
        for i in 0..m {
            let xi = &self.centers[i * self.dim..(i + 1) * self.dim];
            let r = distance(x, xi);
            acc += self.weights[i] * self.kernel.eval(r, self.eps);
        }
        Ok(acc)
    }

    /// Number of centres in the fitted interpolator.
    #[must_use]
    pub fn n_centers(&self) -> usize {
        self.weights.len()
    }

    /// Spatial dimension of the interpolator.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_centers_gaussian() {
        // The interpolant must reproduce the data values at the centres.
        let pts = [0.0, 1.0, 2.0, 3.0, 4.0]; // 5 points in 1-D
        let vals = [0.0, 1.0, 4.0, 9.0, 16.0]; // x²
        let rbf = RbfInterpolator::fit(&pts, &vals, 5, 1, RbfKernel::Gaussian, 1.5)
            .expect("fit should succeed");
        for i in 0..5 {
            let y = rbf.eval(&[pts[i]]).expect("eval should succeed");
            assert!((y - vals[i]).abs() < 1e-6, "i={i}: {y} vs {}", vals[i]);
        }
    }

    #[test]
    fn interpolates_centers_multiquadric() {
        let pts = [0.0, 1.0, 2.0, 3.0];
        let vals = [1.0, 2.0, 0.5, 3.0];
        let rbf = RbfInterpolator::fit(&pts, &vals, 4, 1, RbfKernel::Multiquadric, 1.0)
            .expect("fit should succeed");
        for i in 0..4 {
            let y = rbf.eval(&[pts[i]]).expect("eval should succeed");
            assert!((y - vals[i]).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn smooth_function_interior_accuracy() {
        // Fit sin on a 1-D grid, check accuracy between nodes.
        let m = 11;
        let mut pts = vec![0.0; m];
        let mut vals = vec![0.0; m];
        for i in 0..m {
            let x = i as f64 * 0.3;
            pts[i] = x;
            vals[i] = x.sin();
        }
        let rbf = RbfInterpolator::fit(&pts, &vals, m, 1, RbfKernel::Multiquadric, 0.4)
            .expect("fit should succeed");
        for &x in &[0.15, 0.75, 1.35, 2.1] {
            let y = rbf.eval(&[x]).expect("eval should succeed");
            assert!((y - x.sin()).abs() < 1e-2, "x={x}: {y} vs {}", x.sin());
        }
    }

    #[test]
    fn two_dimensional() {
        // f(x, y) = x + y on 4 scattered points.
        let pts = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let vals = [0.0, 1.0, 1.0, 2.0];
        let rbf = RbfInterpolator::fit(&pts, &vals, 4, 2, RbfKernel::InverseMultiquadric, 1.0)
            .expect("fit should succeed");
        for i in 0..4 {
            let p = [pts[2 * i], pts[2 * i + 1]];
            let y = rbf.eval(&p).expect("eval should succeed");
            assert!((y - vals[i]).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn thin_plate_interpolates() {
        let pts = [0.0, 1.0, 2.0, 3.0, 4.0];
        let vals = [0.0, 1.0, 0.0, 1.0, 0.0];
        let rbf = RbfInterpolator::fit(&pts, &vals, 5, 1, RbfKernel::ThinPlate, 0.0)
            .expect("fit should succeed");
        for i in 0..5 {
            let y = rbf.eval(&[pts[i]]).expect("eval should succeed");
            assert!((y - vals[i]).abs() < 1e-5, "i={i}: {y}");
        }
    }

    #[test]
    fn eval_finite() {
        let pts = [0.0, 1.0, 2.0];
        let vals = [1.0, 2.0, 3.0];
        let rbf = RbfInterpolator::fit(&pts, &vals, 3, 1, RbfKernel::Gaussian, 1.0)
            .expect("fit should succeed");
        for &x in &[-1.0, 0.5, 1.5, 5.0] {
            assert!(rbf.eval(&[x]).expect("eval should succeed").is_finite());
        }
    }

    #[test]
    fn empty_input_error() {
        let res = RbfInterpolator::fit(&[], &[], 0, 1, RbfKernel::Gaussian, 1.0);
        assert!(matches!(res, Err(NumericError::EmptyInput)));
    }

    #[test]
    fn shape_mismatch_error() {
        // points length inconsistent with m·dim.
        let res = RbfInterpolator::fit(
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0],
            2,
            2,
            RbfKernel::Gaussian,
            1.0,
        );
        assert!(matches!(res, Err(NumericError::ShapeMismatch { .. })));
    }

    #[test]
    fn values_mismatch_error() {
        let res = RbfInterpolator::fit(
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0],
            3,
            1,
            RbfKernel::Gaussian,
            1.0,
        );
        assert!(matches!(res, Err(NumericError::DimensionMismatch { .. })));
    }

    #[test]
    fn bad_eps_error() {
        let res = RbfInterpolator::fit(&[0.0, 1.0], &[1.0, 2.0], 2, 1, RbfKernel::Gaussian, 0.0);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    #[test]
    fn query_dim_mismatch_error() {
        let pts = [0.0, 1.0, 2.0];
        let vals = [1.0, 2.0, 3.0];
        let rbf = RbfInterpolator::fit(&pts, &vals, 3, 1, RbfKernel::Gaussian, 1.0)
            .expect("fit should succeed");
        let res = rbf.eval(&[0.0, 0.0]); // wrong dimension
        assert!(matches!(res, Err(NumericError::DimensionMismatch { .. })));
    }

    #[test]
    fn accessors() {
        let pts = [0.0, 1.0, 2.0];
        let vals = [1.0, 2.0, 3.0];
        let rbf = RbfInterpolator::fit(&pts, &vals, 3, 1, RbfKernel::Gaussian, 1.0)
            .expect("fit should succeed");
        assert_eq!(rbf.n_centers(), 3);
        assert_eq!(rbf.dim(), 1);
    }
}
