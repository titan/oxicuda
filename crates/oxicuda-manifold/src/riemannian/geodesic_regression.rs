//! Geodesic regression on the SPD(d) manifold.
//!
//! Reference: Fletcher, P.T. (2013). "Geodesic Regression and the Theory of Least Squares on
//! Riemannian Manifolds." *International Journal of Computer Vision*, 105(2), pp. 171–185.
//!
//! Fits a geodesic `γ(t) = Exp_p(t · v)` on the symmetric positive-definite (SPD) cone
//! with the affine-invariant metric, minimising the sum of squared Riemannian distances:
//!
//! ```text
//! F(p, v) = Σᵢ d(γ(tᵢ), Yᵢ)²
//! ```
//!
//! where `p ∈ SPD(d)` is the base point, `v ∈ T_p SPD(d)` is the velocity (symmetric matrix),
//! and `γ(t) = Exp_p(t · v)` is the geodesic through `p` in direction `v`.
//!
//! # Algorithm
//! Gradient descent on the product manifold `SPD(d) × T_p SPD(d)`:
//! 1. For each sample i, compute the geodesic point `γᵢ = Exp_p(tᵢ v)`.
//! 2. Compute residual `rᵢ = Log_{γᵢ}(Yᵢ)` (tangent vector at γᵢ).
//! 3. Parallel-transport `rᵢ` back to `T_p M` via `PT_{γᵢ→p}(rᵢ)`.
//! 4. Accumulate gradients: `grad_v = -2 Σ tᵢ r̃ᵢ`, `grad_p = -2 Σ r̃ᵢ`.
//! 5. Update `v ← v - lr · grad_v` (Euclidean on tangent), then symmetrize.
//! 6. Update `p ← Exp_p(-lr · grad_p)` (Riemannian retraction).
//! 7. Repeat until convergence (‖grad_v‖_F + ‖grad_p‖_F < tol).

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::riemannian::spd::{spd_distance, spd_exp, spd_log, spd_project_symmetric};
use crate::riemannian::spd_kmeans::{FrechetMeanConfig, spd_frechet_mean};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for geodesic regression on SPD(d).
#[derive(Debug, Clone)]
pub struct GeodesicRegressionConfig {
    /// Matrix dimension d: SPD matrices are d×d.
    pub matrix_dim: usize,
    /// Gradient descent learning rate.
    pub learning_rate: f64,
    /// Maximum number of gradient descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the sum of Frobenius norms of gradients.
    pub tol: f64,
}

impl Default for GeodesicRegressionConfig {
    fn default() -> Self {
        Self {
            matrix_dim: 2,
            learning_rate: 0.01,
            max_iter: 500,
            tol: 1e-7,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted geodesic regression model on SPD(d).
#[derive(Debug, Clone)]
pub struct GeodesicRegressionFit {
    /// Base point p ∈ SPD(d), stored as flat d²-vector (row-major).
    pub base_point: Vec<f64>,
    /// Velocity v ∈ T_p SPD(d) (symmetric d×d matrix), flat d²-vector.
    pub velocity: Vec<f64>,
    /// Matrix dimension d.
    pub matrix_dim: usize,
    /// Final sum of squared Riemannian distances.
    pub final_sse: f64,
    /// Whether the algorithm converged within `tol` before `max_iter`.
    pub converged: bool,
    /// Number of gradient descent iterations performed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Dense n×n matrix multiply C = A B (row-major).
fn mat_mul_nn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

/// Frobenius norm of an n×n matrix.
fn frobenius_norm(m: &[f64]) -> f64 {
    m.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Scale a matrix in place by scalar s.
fn mat_scale_inplace(m: &mut [f64], s: f64) {
    for v in m.iter_mut() {
        *v *= s;
    }
}

/// Add scaled matrix: out += scale * m.
fn mat_add_scaled(out: &mut [f64], m: &[f64], scale: f64) {
    for (o, &mv) in out.iter_mut().zip(m.iter()) {
        *o += scale * mv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix square root via Jacobi eigendecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the symmetric positive-definite matrix square root M^{1/2} via Jacobi.
///
/// For M = V diag(λ) V^T (eigendecomposition), M^{1/2} = V diag(√λ) V^T.
/// Eigenvalues are clamped to `min_eig` to guarantee SPD output.
fn spd_matrix_sqrt(m: &[f64], n: usize, min_eig: f64) -> ManifoldResult<Vec<f64>> {
    let (w, v) = jacobi_eigh(m, n)?;
    let mut sq = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                let lam_sqrt = w[k].max(min_eig).sqrt();
                acc += v[i * n + k] * v[j * n + k] * lam_sqrt;
            }
            sq[i * n + j] = acc;
        }
    }
    Ok(sq)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gauss-Jordan matrix inversion
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the inverse of an n×n matrix via Gauss-Jordan elimination with partial pivoting.
fn mat_inverse_gauss_jordan(m: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    // Build augmented matrix [M | I]
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = m[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting: find row with largest absolute value in column `col`
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(ManifoldError::SingularMatrix(format!(
                "matrix is singular or near-singular at column {col}"
            )));
        }
        // Swap rows col and max_row
        if max_row != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }
        // Scale pivot row
        let pivot = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= pivot;
        }
        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            if factor.abs() < 1e-300 {
                continue;
            }
            for j in 0..2 * n {
                let sub = factor * aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= sub;
            }
        }
    }

    // Extract inverse from right half of augmented matrix
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel transport on SPD
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel-transport a tangent vector `x ∈ T_Q SPD(d)` to `T_P SPD(d)`.
///
/// Uses the formula: `PT_{Q→P}(X) = Γ X Γ^T`
/// where `Γ = (P Q^{-1})^{1/2}`.
///
/// Implementation:
/// 1. Compute `M = P Q^{-1}` via Gauss-Jordan inversion of Q.
/// 2. Compute `Γ = M^{1/2}` via Jacobi eigendecomposition.
/// 3. Return `Γ X Γ^T`.
fn parallel_transport_q_to_p(
    p: &[f64],
    q: &[f64],
    x: &[f64],
    n: usize,
) -> ManifoldResult<Vec<f64>> {
    // Compute Q^{-1}
    let q_inv = mat_inverse_gauss_jordan(q, n)?;
    // Compute M = P Q^{-1}
    let m = mat_mul_nn(p, &q_inv, n);
    // Compute Γ = M^{1/2} via Jacobi on M (M may not be symmetric but is positive for SPD)
    // For numerical safety, symmetrize before eigendecomp
    let m_sym = spd_project_symmetric(&m, n)?;
    let gamma = spd_matrix_sqrt(&m_sym, n, 1e-14)?;
    // Return Γ X Γ^T
    let gx = mat_mul_nn(&gamma, x, n);
    let gxgt = mat_mul_nn(&gx, &gamma, n); // Γ^T == Γ since Γ is symmetric (sym sqrt of sym matrix)
    Ok(gxgt)
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the sum of squared Riemannian distances for a fitted geodesic.
///
/// `SSE = Σᵢ d(γ(tᵢ), Yᵢ)²`
pub fn geodesic_regression_sse(
    fit: &GeodesicRegressionFit,
    t: &[f64],
    y: &[f64],
) -> ManifoldResult<f64> {
    let n = t.len();
    let d = fit.matrix_dim;
    if y.len() != n * d * d {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d, d],
            got: vec![y.len()],
        });
    }
    let mut sse = 0.0f64;
    for i in 0..n {
        // γ(tᵢ) = Exp_p(tᵢ · v)
        let mut tv = fit.velocity.clone();
        mat_scale_inplace(&mut tv, t[i]);
        let gamma_i = spd_exp(&fit.base_point, &tv, d)?;
        let yi = &y[i * d * d..(i + 1) * d * d];
        let dist = spd_distance(&gamma_i, yi, d)?;
        sse += dist * dist;
    }
    Ok(sse)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a geodesic regression model on SPD(d) data.
///
/// # Arguments
/// * `t`      — scalar time parameters, length `n_samples`
/// * `y`      — flat `[n_samples × d²]` row-major SPD matrices
/// * `config` — algorithm configuration
///
/// # Returns
/// A [`GeodesicRegressionFit`] containing the fitted base point and velocity.
///
/// # Errors
/// - `InvalidParameter` if `t.len() == 0`
/// - `ShapeMismatch` if `y.len() != t.len() * d * d`
/// - `ManifoldConstraint` if any Y_i is not SPD
pub fn geodesic_regression_fit(
    t: &[f64],
    y: &[f64],
    config: &GeodesicRegressionConfig,
) -> ManifoldResult<GeodesicRegressionFit> {
    let n = t.len();
    let d = config.matrix_dim;

    // ── Validation ────────────────────────────────────────────────────────────
    if n == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "t".into(),
            reason: "must have at least one sample".into(),
        });
    }
    if y.len() != n * d * d {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d, d],
            got: vec![y.len()],
        });
    }
    // Validate each Y_i is SPD (check positive eigenvalues via Jacobi)
    for i in 0..n {
        let yi = &y[i * d * d..(i + 1) * d * d];
        let (w, _) = jacobi_eigh(yi, d).map_err(|e| {
            ManifoldError::ManifoldConstraint(format!("Y[{i}] eigendecomposition failed: {e}"))
        })?;
        for &ev in &w {
            if ev < -1e-8 {
                return Err(ManifoldError::ManifoldConstraint(format!(
                    "Y[{i}] has negative eigenvalue {ev:.3e}: not SPD"
                )));
            }
        }
    }

    // ── Initialisation ────────────────────────────────────────────────────────
    // p₀ = Fréchet mean of all Y_i
    let frechet_cfg = FrechetMeanConfig {
        max_iter: 200,
        tol: 1e-8,
        step_size: 1.0,
    };
    let fm = spd_frechet_mean(y, n, d, &frechet_cfg)?;
    let mut p = fm.mean;

    // v₀ = mean of Log_p(Y_i) / t_i for |t_i| > 1e-6
    let mut v = vec![0.0f64; d * d];
    let mut v_count = 0usize;
    for i in 0..n {
        if t[i].abs() <= 1e-6 {
            continue;
        }
        let yi = &y[i * d * d..(i + 1) * d * d];
        match spd_log(&p, yi, d) {
            Ok(log_yi) => {
                let scale = 1.0 / t[i];
                mat_add_scaled(&mut v, &log_yi, scale);
                v_count += 1;
            }
            Err(_) => {
                // Skip samples where log map fails during init
            }
        }
    }
    if v_count > 0 {
        let inv_count = 1.0 / v_count as f64;
        mat_scale_inplace(&mut v, inv_count);
    }
    // Symmetrize the initial velocity
    v = spd_project_symmetric(&v, d)?;

    let mut converged = false;
    let mut iterations = 0usize;

    // ── Gradient descent loop ─────────────────────────────────────────────────
    for iter in 0..config.max_iter {
        iterations = iter + 1;

        let mut grad_v = vec![0.0f64; d * d];
        let mut grad_p = vec![0.0f64; d * d];

        for i in 0..n {
            // γᵢ = Exp_p(tᵢ · v)
            let mut tv = v.clone();
            mat_scale_inplace(&mut tv, t[i]);
            let gamma_i = match spd_exp(&p, &tv, d) {
                Ok(g) => g,
                Err(_) => continue, // skip numerically problematic samples
            };

            let yi = &y[i * d * d..(i + 1) * d * d];
            // rᵢ = Log_{γᵢ}(Yᵢ)
            let r_i = match spd_log(&gamma_i, yi, d) {
                Ok(r) => r,
                Err(_) => continue,
            };

            // r̃ᵢ = PT_{γᵢ→p}(rᵢ): parallel transport residual back to T_p M
            let r_tilde = match parallel_transport_q_to_p(&p, &gamma_i, &r_i, d) {
                Ok(rt) => rt,
                Err(_) => {
                    // Fallback: use residual without transport (approximate)
                    r_i.clone()
                }
            };

            // grad_v += -2 * tᵢ * r̃ᵢ
            mat_add_scaled(&mut grad_v, &r_tilde, -2.0 * t[i]);
            // grad_p += -2 * r̃ᵢ
            mat_add_scaled(&mut grad_p, &r_tilde, -2.0);
        }

        // Convergence check
        let norm_gv = frobenius_norm(&grad_v);
        let norm_gp = frobenius_norm(&grad_p);
        if norm_gv + norm_gp < config.tol {
            converged = true;
            break;
        }

        // Update v: Euclidean step on tangent space, then symmetrize
        mat_add_scaled(&mut v, &grad_v, -config.learning_rate);
        v = spd_project_symmetric(&v, d)?;

        // Update p: Riemannian retraction p ← Exp_p(-lr * grad_p_tangent)
        let mut step_p = grad_p.clone();
        mat_scale_inplace(&mut step_p, -config.learning_rate);
        step_p = spd_project_symmetric(&step_p, d)?;
        p = match spd_exp(&p, &step_p, d) {
            Ok(p_new) => p_new,
            Err(_) => p, // keep current p if retraction fails
        };
    }

    // Compute final SSE
    let fake_fit = GeodesicRegressionFit {
        base_point: p.clone(),
        velocity: v.clone(),
        matrix_dim: d,
        final_sse: 0.0,
        converged,
        iterations,
    };
    let final_sse = geodesic_regression_sse(&fake_fit, t, y).unwrap_or(f64::INFINITY);

    Ok(GeodesicRegressionFit {
        base_point: p,
        velocity: v,
        matrix_dim: d,
        final_sse,
        converged,
        iterations,
    })
}

/// Predict geodesic positions at new time values.
///
/// Returns `[n_new × d²]` row-major SPD matrices `γ(t_new[i]) = Exp_p(t_new[i] · v)`.
pub fn geodesic_regression_predict(
    fit: &GeodesicRegressionFit,
    t_new: &[f64],
) -> ManifoldResult<Vec<f64>> {
    let d = fit.matrix_dim;
    let n = t_new.len();
    let mut out = vec![0.0f64; n * d * d];
    for (i, &ti) in t_new.iter().enumerate() {
        let mut tv = fit.velocity.clone();
        mat_scale_inplace(&mut tv, ti);
        let gamma_i = spd_exp(&fit.base_point, &tv, d)?;
        out[i * d * d..(i + 1) * d * d].copy_from_slice(&gamma_i);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::linalg::jacobi_eig::jacobi_eigh;

    /// Check if a flat d×d matrix is SPD (symmetric + all eigenvalues positive).
    fn is_spd(m: &[f64], d: usize) -> bool {
        // Symmetry check
        for i in 0..d {
            for j in 0..d {
                if (m[i * d + j] - m[j * d + i]).abs() > 1e-6 {
                    return false;
                }
            }
        }
        match jacobi_eigh(m, d) {
            Ok((w, _)) => w.iter().all(|&ev| ev > 1e-10),
            Err(_) => false,
        }
    }

    /// Frobenius distance between two flat matrices.
    fn frob_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Build an identity matrix of size d×d.
    fn identity(d: usize) -> Vec<f64> {
        let mut m = vec![0.0f64; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }
        m
    }

    // ── Test 1: linear geodesic recovery ─────────────────────────────────────
    /// Y_i = Exp_I(tᵢ · V) for known V. Check recovered velocity is close to V.
    #[test]
    fn linear_geodesic_recovery_2x2() {
        let d = 2;
        let p0 = identity(d); // base point = I_2
        let v0 = vec![0.2, 0.1, 0.1, 0.3]; // symmetric velocity
        let t_vals: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
        let n = t_vals.len();

        // Generate Y_i = Exp_{I}(tᵢ · V)
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("Exp ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }

        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 800,
            tol: 1e-7,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("fit ok");

        let v_err = frob_dist(&fit.velocity, &v0);
        assert!(
            v_err < 0.2,
            "Velocity recovery error too large: {v_err:.4} (> 0.2)"
        );
    }

    // ── Test 2: SSE decreases from initial to final ───────────────────────────
    #[test]
    fn sse_decreases() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.3, 0.05, 0.05, 0.2];

        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }

        // Initial SSE using (frechet_mean, v=0)
        let frechet_cfg = FrechetMeanConfig::default();
        let fm = spd_frechet_mean(&y, n, d, &frechet_cfg).expect("frechet ok");
        let init_fit = GeodesicRegressionFit {
            base_point: fm.mean.clone(),
            velocity: vec![0.0f64; d * d],
            matrix_dim: d,
            final_sse: 0.0,
            converged: false,
            iterations: 0,
        };
        let initial_sse = geodesic_regression_sse(&init_fit, &t_vals, &y).expect("sse ok");

        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 300,
            tol: 1e-8,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("fit ok");

        assert!(
            fit.final_sse < initial_sse + 1e-6,
            "Final SSE ({:.6}) should be <= initial SSE ({:.6})",
            fit.final_sse,
            initial_sse
        );
    }

    // ── Test 3: predict at t=0 returns base_point ─────────────────────────────
    #[test]
    fn predict_at_t0_is_base_point() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.2, 0.1, 0.1, 0.3];

        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }

        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 300,
            tol: 1e-8,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("fit ok");
        let pred = geodesic_regression_predict(&fit, &[0.0]).expect("predict ok");

        let err = frob_dist(&pred[..d * d], &fit.base_point);
        assert!(
            err < 1e-6,
            "Predict at t=0 should return base_point, error={err:.2e}"
        );
    }

    // ── Test 4: predicted matrices are SPD ────────────────────────────────────
    #[test]
    fn predicted_matrices_are_spd() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.2, 0.05, 0.05, 0.15];

        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }

        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 200,
            tol: 1e-8,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("fit ok");

        let t_new = vec![0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5];
        let pred = geodesic_regression_predict(&fit, &t_new).expect("predict ok");
        for (idx, _ti) in t_new.iter().enumerate() {
            let pi = &pred[idx * d * d..(idx + 1) * d * d];
            assert!(
                is_spd(pi, d),
                "Predicted matrix at index {idx} is not SPD: {pi:?}"
            );
        }
    }

    // ── Test 5: validation — empty t → InvalidParameter ──────────────────────
    #[test]
    fn validation_empty_t() {
        let config = GeodesicRegressionConfig::default();
        let res = geodesic_regression_fit(&[], &[], &config);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "t");
            }
            other => panic!("Expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 6: validation — y shape mismatch → ShapeMismatch ────────────────
    #[test]
    fn validation_y_shape_mismatch() {
        let t = vec![0.0, 1.0, 2.0];
        let y_bad = vec![1.0f64; 5]; // wrong length (should be 3 * 2 * 2 = 12)
        let config = GeodesicRegressionConfig {
            matrix_dim: 2,
            ..Default::default()
        };
        let res = geodesic_regression_fit(&t, &y_bad, &config);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::ShapeMismatch { .. }) => {}
            other => panic!("Expected ShapeMismatch, got {other:?}"),
        }
    }

    // ── Test 7: validation — non-SPD Y_i → ManifoldConstraint ────────────────
    #[test]
    fn validation_non_spd_yi() {
        let d = 2;
        let t = vec![0.0, 1.0];
        let n = t.len();
        let mut y = vec![0.0f64; n * d * d];
        // Y[0] = identity (valid)
        y[0] = 1.0;
        y[3] = 1.0;
        // Y[1] = negative definite
        y[4] = -1.0;
        y[7] = -1.0;
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            ..Default::default()
        };
        let res = geodesic_regression_fit(&t, &y, &config);
        assert!(res.is_err());
        match res {
            Err(ManifoldError::ManifoldConstraint(_)) => {}
            other => panic!("Expected ManifoldConstraint, got {other:?}"),
        }
    }

    // ── Test 8: SSE is non-negative ───────────────────────────────────────────
    #[test]
    fn sse_nonneg() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.1, 0.0, 0.0, 0.1];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 100,
            tol: 1e-8,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        assert!(fit.final_sse >= 0.0, "SSE is negative: {}", fit.final_sse);
    }

    // ── Test 9: base_point is SPD ─────────────────────────────────────────────
    #[test]
    fn fitted_base_point_is_spd() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.15, 0.07, 0.07, 0.25];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 200,
            tol: 1e-8,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        assert!(is_spd(&fit.base_point, d), "Base point is not SPD");
    }

    // ── Test 10: geodesic_regression_sse matches final_sse field ─────────────
    #[test]
    fn sse_matches_final_sse_field() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.1, 0.05, 0.05, 0.15];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            ..Default::default()
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        let recomputed = geodesic_regression_sse(&fit, &t_vals, &y).expect("ok");
        assert!(
            (fit.final_sse - recomputed).abs() < 1e-10,
            "final_sse field ({}) does not match recomputed ({})",
            fit.final_sse,
            recomputed
        );
    }

    // ── Test 11: predict output length is correct ─────────────────────────────
    #[test]
    fn predict_output_length() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 1.0, 2.0];
        let n = t_vals.len();
        let mut y = vec![0.0f64; n * d * d];
        for i in 0..n {
            y[i * d * d] = 1.0 + i as f64 * 0.1;
            y[i * d * d + 3] = 1.0 + i as f64 * 0.1;
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            max_iter: 50,
            ..Default::default()
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        let t_new = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let pred = geodesic_regression_predict(&fit, &t_new).expect("ok");
        assert_eq!(pred.len(), t_new.len() * d * d);
    }

    // ── Test 12: parallel transport preserves symmetry ────────────────────────
    #[test]
    fn parallel_transport_preserves_symmetry() {
        let d = 2;
        let p = vec![2.0, 0.5, 0.5, 3.0];
        let q = vec![1.0, 0.3, 0.3, 2.0];
        let x = vec![0.5, 0.1, 0.1, -0.3]; // symmetric tangent at q
        let pt = parallel_transport_q_to_p(&p, &q, &x, d).expect("pt ok");
        // Result should be symmetric
        for i in 0..d {
            for j in i + 1..d {
                let diff = (pt[i * d + j] - pt[j * d + i]).abs();
                assert!(
                    diff < 1e-8,
                    "PT result not symmetric at ({i},{j}): {diff:.2e}"
                );
            }
        }
    }

    // ── Test 13: matrix_sqrt returns SPD matrix ────────────────────────────────
    #[test]
    fn matrix_sqrt_is_spd() {
        let d = 2;
        let m = vec![4.0, 1.0, 1.0, 3.0]; // SPD 2×2
        let sq = spd_matrix_sqrt(&m, d, 1e-14).expect("sqrt ok");
        // sq^2 should recover m
        let sq2 = mat_mul_nn(&sq, &sq, d);
        let err = frob_dist(&sq2, &m);
        assert!(err < 1e-8, "sqrt^2 != m, error={err:.2e}");
        assert!(is_spd(&sq, d), "Matrix sqrt is not SPD");
    }

    // ── Test 14: Gauss-Jordan inversion is correct ─────────────────────────────
    #[test]
    fn gauss_jordan_inversion() {
        let d = 2;
        let m = vec![3.0, 1.0, 1.0, 2.0];
        let inv = mat_inverse_gauss_jordan(&m, d).expect("inv ok");
        // M * M^{-1} should be I
        let prod = mat_mul_nn(&m, &inv, d);
        let ident = identity(d);
        let err = frob_dist(&prod, &ident);
        assert!(err < 1e-10, "M * inv(M) != I, error={err:.2e}");
    }

    // ── Test 15: single-sample fit doesn't panic ──────────────────────────────
    #[test]
    fn single_sample_fit() {
        let d = 2;
        let t = vec![1.0];
        let y = identity(d);
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            max_iter: 10,
            ..Default::default()
        };
        let res = geodesic_regression_fit(&t, &y, &config);
        assert!(res.is_ok(), "Single-sample fit failed: {res:?}");
    }

    // ── Test 16: iterations field is <= max_iter ──────────────────────────────
    #[test]
    fn iterations_le_max_iter() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 1.0, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.1, 0.0, 0.0, 0.1];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let max_iter = 50;
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            max_iter,
            ..Default::default()
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        assert!(
            fit.iterations <= max_iter,
            "iterations ({}) > max_iter ({})",
            fit.iterations,
            max_iter
        );
    }

    // ── Test 17: velocity is symmetric after fit ──────────────────────────────
    #[test]
    fn fitted_velocity_is_symmetric() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.2, 0.08, 0.08, 0.3];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            max_iter: 100,
            ..Default::default()
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        let v = &fit.velocity;
        for i in 0..d {
            for j in i + 1..d {
                let diff = (v[i * d + j] - v[j * d + i]).abs();
                assert!(
                    diff < 1e-10,
                    "Velocity not symmetric at ({i},{j}): {diff:.2e}"
                );
            }
        }
    }

    // ── Test 18: predict output is finite ────────────────────────────────────
    #[test]
    fn predict_output_finite() {
        let d = 2;
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0];
        let n = t_vals.len();
        let p0 = identity(d);
        let v0 = vec![0.1, 0.05, 0.05, 0.1];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            max_iter: 100,
            ..Default::default()
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        let t_new = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
        let pred = geodesic_regression_predict(&fit, &t_new).expect("ok");
        for &v in &pred {
            assert!(v.is_finite(), "Non-finite value in prediction: {v}");
        }
    }

    // ── Test 19: 3×3 SPD regression with diagonal data ───────────────────────
    #[test]
    fn regression_3x3_diagonal() {
        let d = 3;
        let p0: Vec<f64> = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        let v0: Vec<f64> = vec![
            0.1, 0.0, 0.0, //
            0.0, 0.2, 0.0, //
            0.0, 0.0, 0.15, //
        ];
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.005,
            max_iter: 500,
            tol: 1e-7,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        // Base point should be SPD and velocity close to v0
        assert!(is_spd(&fit.base_point, d), "Base point not SPD");
        let v_err = frob_dist(&fit.velocity, &v0);
        assert!(
            v_err < 0.5,
            "3×3 velocity error too large: {v_err:.4} (> 0.5)"
        );
    }

    // ── Test 20: perfect fit on geodesic data → SSE near zero ────────────────
    #[test]
    fn perfect_fit_sse_near_zero() {
        let d = 2;
        let p0 = identity(d);
        let v0 = vec![0.1, 0.05, 0.05, 0.2];
        let t_vals: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let n = t_vals.len();
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t_vals.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-10,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config).expect("ok");
        // Data lies on a geodesic, so SSE should be very small after convergence
        assert!(
            fit.final_sse < 1e-4,
            "Perfect geodesic data → SSE should be near 0, got {:.2e}",
            fit.final_sse
        );
    }

    // ── Test 21: RNG-generated noisy data regression ──────────────────────────
    #[test]
    fn noisy_geodesic_regression() {
        let d = 2;
        let mut rng = LcgRng::new(21);
        let p0 = identity(d);
        let v0 = vec![0.15, 0.06, 0.06, 0.22];
        let n = 15;
        let mut t_vals: Vec<f64> = Vec::with_capacity(n);
        let mut y = vec![0.0f64; n * d * d];
        for i in 0..n {
            let ti = i as f64 / (n as f64 - 1.0) * 2.0;
            t_vals.push(ti);
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi_clean = spd_exp(&p0, &tv, d).expect("ok");
            // Add small symmetric noise to stay SPD
            let noise_scale = 0.02;
            let n11 = rng.next_normal() * noise_scale;
            let n12 = rng.next_normal() * noise_scale * 0.3;
            let n22 = rng.next_normal() * noise_scale;
            let yi = vec![
                yi_clean[0] + n11,
                yi_clean[1] + n12,
                yi_clean[2] + n12,
                yi_clean[3] + n22,
            ];
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            learning_rate: 0.01,
            max_iter: 300,
            tol: 1e-7,
        };
        let fit = geodesic_regression_fit(&t_vals, &y, &config);
        assert!(fit.is_ok(), "Noisy geodesic regression failed: {fit:?}");
        let fit = fit.expect("fit should be present");
        assert!(is_spd(&fit.base_point, d), "Base point not SPD");
        assert!(fit.final_sse.is_finite(), "SSE not finite");
    }

    // ── Test 22: config stored in result (round-trip check) ───────────────────
    #[test]
    fn config_fields_are_used() {
        let d = 2;
        let t = vec![0.0, 1.0, 2.0];
        let n = t.len();
        let p0 = identity(d);
        let v0 = vec![0.1, 0.0, 0.0, 0.1];
        let mut y = vec![0.0f64; n * d * d];
        for (i, &ti) in t.iter().enumerate() {
            let mut tv = v0.clone();
            mat_scale_inplace(&mut tv, ti);
            let yi = spd_exp(&p0, &tv, d).expect("ok");
            y[i * d * d..(i + 1) * d * d].copy_from_slice(&yi);
        }
        let max_iter = 7;
        let config = GeodesicRegressionConfig {
            matrix_dim: d,
            max_iter,
            tol: 1e-20, // won't converge in 7 steps
            ..Default::default()
        };
        let fit = geodesic_regression_fit(&t, &y, &config).expect("ok");
        assert_eq!(fit.matrix_dim, d);
        assert!(
            fit.iterations <= max_iter,
            "Should not exceed max_iter={max_iter}"
        );
    }
}
