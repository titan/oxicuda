//! Riemannian Adam optimizer on Stiefel and Grassmann manifolds.
//!
//! Implements the Riemannian Adam algorithm (Becigneul & Ganea 2019; Liu et al. 2017),
//! adapting the Euclidean Adam algorithm to manifold-valued parameters via:
//!
//! 1. Compute Euclidean gradient `∇f(x)` via user-supplied callback.
//! 2. Project to tangent space: `g = project_tangent(x, ∇f(x))`.
//! 3. Maintain first/second moment estimates in the tangent space:
//!    - `m ← β₁ * transport(m) + (1-β₁) * g`
//!    - `v ← β₂ * transport(v) + (1-β₂) * g ⊙ g`
//! 4. Bias-corrected estimates: `m̂ = m / (1-β₁^t)`, `v̂ = v / (1-β₂^t)`
//! 5. Update step: `x_new = retract(x, -α * m̂ / (sqrt(v̂) + ε))`
//! 6. Transport moments to new tangent space (projection transport).

use crate::error::{ManifoldError, ManifoldResult};
use crate::riemannian::grassmann::{grassmann_project_tangent, grassmann_retract};
use crate::riemannian::stiefel::{stiefel_project_tangent, stiefel_retract_qr};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the Riemannian Adam optimizer.
#[derive(Debug, Clone)]
pub struct RiemannianAdamConfig {
    /// Learning rate (step size). Default: 0.01.
    pub lr: f64,
    /// First moment decay factor. Default: 0.9.
    pub beta1: f64,
    /// Second moment decay factor. Default: 0.999.
    pub beta2: f64,
    /// Numerical stability epsilon. Default: 1e-8.
    pub eps: f64,
    /// Maximum number of iterations. Default: 200.
    pub max_iter: usize,
    /// Convergence tolerance on tangent gradient norm. Default: 1e-6.
    pub tol: f64,
}

impl Default for RiemannianAdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

// ─── Manifold type ───────────────────────────────────────────────────────────

/// Specifies which Riemannian manifold the parameter lives on.
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldType {
    /// Stiefel manifold St(n, p) = { X ∈ ℝ^{n×p} : X^T X = I_p }.
    Stiefel { n: usize, p: usize },
    /// Grassmann manifold Gr(n, p) — p-dimensional subspaces of ℝ^n.
    Grassmann { n: usize, p: usize },
}

impl ManifoldType {
    /// The ambient dimension n * p.
    fn ambient_dim(&self) -> usize {
        match self {
            ManifoldType::Stiefel { n, p } | ManifoldType::Grassmann { n, p } => n * p,
        }
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Mutable optimizer state carried across iterations.
#[derive(Debug, Clone)]
pub struct RiemannianAdamState {
    /// Current point on the manifold (flattened row-major n×p matrix).
    pub point: Vec<f64>,
    /// First moment estimate (tangent vector at current point).
    pub m: Vec<f64>,
    /// Second moment estimate (element-wise squared, tangent vector).
    pub v: Vec<f64>,
    /// Iteration counter (1-indexed, incremented before bias correction).
    pub t: usize,
}

impl RiemannianAdamState {
    fn new(init: Vec<f64>, ambient_dim: usize) -> Self {
        Self {
            point: init,
            m: vec![0.0; ambient_dim],
            v: vec![0.0; ambient_dim],
            t: 0,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Result returned by the Riemannian Adam optimizer.
#[derive(Debug, Clone)]
pub struct RiemannianAdamResult {
    /// Optimized point on the manifold (flattened row-major n×p matrix).
    pub point: Vec<f64>,
    /// L2 norm of the Riemannian gradient at the final iterate.
    pub final_gradient_norm: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the optimizer converged (gradient norm < tol).
    pub converged: bool,
}

// ─── Internal vector helpers ─────────────────────────────────────────────────

/// Element-wise product: `a ⊙ b`.
pub fn vec_hadamard(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Element-wise `sqrt(v_i) + eps`.
pub fn vec_sqrt_eps(v: &[f64], eps: f64) -> Vec<f64> {
    v.iter().map(|x| x.sqrt() + eps).collect()
}

/// `alpha * x + beta * y` (AXPBY).
pub fn vec_axpby(alpha: f64, x: &[f64], beta: f64, y: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| alpha * xi + beta * yi)
        .collect()
}

/// L2 norm of a vector.
pub fn gradient_norm(g: &[f64]) -> f64 {
    g.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─── Manifold dispatch helpers ────────────────────────────────────────────────

/// Project an ambient vector `g` to the tangent space of `manifold` at `x`.
pub fn project_tangent_ambient(
    x: &[f64],
    g: &[f64],
    manifold: &ManifoldType,
) -> ManifoldResult<Vec<f64>> {
    match manifold {
        ManifoldType::Stiefel { n, p } => stiefel_project_tangent(x, g, *n, *p),
        ManifoldType::Grassmann { n, p } => grassmann_project_tangent(x, g, *n, *p),
    }
}

/// Apply retraction at `x` along tangent vector `xi` on `manifold`.
pub fn retract_ambient(x: &[f64], xi: &[f64], manifold: &ManifoldType) -> ManifoldResult<Vec<f64>> {
    match manifold {
        ManifoldType::Stiefel { n, p } => stiefel_retract_qr(x, xi, *n, *p),
        ManifoldType::Grassmann { n, p } => grassmann_retract(x, xi, *n, *p),
    }
}

// ─── Main optimizer ───────────────────────────────────────────────────────────

/// Optimize a function on a Riemannian manifold using the Riemannian Adam algorithm.
///
/// # Parameters
/// - `init`: Initial point on the manifold (flattened row-major n×p matrix).
/// - `grad_fn`: Returns the **Euclidean** gradient at a given point.
/// - `manifold`: Which manifold the parameter lives on.
/// - `config`: Optimizer hyperparameters.
///
/// # Returns
/// The optimized point together with convergence information.
///
/// # Errors
/// - `ManifoldError::EmptyInput` if `init` is empty.
/// - `ManifoldError::InvalidParameter` if hyperparameters are out of range.
pub fn riemannian_adam<F>(
    init: &[f64],
    grad_fn: F,
    manifold: ManifoldType,
    config: &RiemannianAdamConfig,
) -> ManifoldResult<RiemannianAdamResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    // ── Validate inputs ──────────────────────────────────────────────────────
    if init.is_empty() {
        return Err(ManifoldError::EmptyInput);
    }
    if config.lr < 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "lr".to_string(),
            reason: "learning rate must be non-negative".to_string(),
        });
    }
    if !(0.0..1.0).contains(&config.beta1) {
        return Err(ManifoldError::InvalidParameter {
            name: "beta1".to_string(),
            reason: "beta1 must be in [0, 1)".to_string(),
        });
    }
    if !(0.0..1.0).contains(&config.beta2) {
        return Err(ManifoldError::InvalidParameter {
            name: "beta2".to_string(),
            reason: "beta2 must be in [0, 1)".to_string(),
        });
    }
    if config.eps <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "eps".to_string(),
            reason: "eps must be positive".to_string(),
        });
    }
    let expected_dim = manifold.ambient_dim();
    if init.len() != expected_dim {
        return Err(ManifoldError::InvalidParameter {
            name: "init".to_string(),
            reason: format!(
                "init has length {} but manifold expects {}",
                init.len(),
                expected_dim
            ),
        });
    }

    // ── Initialise state ─────────────────────────────────────────────────────
    let mut state = RiemannianAdamState::new(init.to_vec(), expected_dim);
    let mut final_grad_norm = 0.0;
    let mut converged = false;

    for _iter in 0..config.max_iter {
        state.t += 1;
        let t = state.t;

        // Step 1: Euclidean gradient at current point.
        let euclid_g = grad_fn(&state.point);

        // Step 2: Riemannian gradient = project to tangent space at current point.
        let riem_g = project_tangent_ambient(&state.point, &euclid_g, &manifold)?;

        // Convergence check on Riemannian gradient norm.
        let g_norm = gradient_norm(&riem_g);
        final_grad_norm = g_norm;
        if g_norm < config.tol {
            converged = true;
            break;
        }

        // Short-circuit: if lr == 0, no update can happen; return immediately.
        if config.lr == 0.0 {
            break;
        }

        // Step 3a: First moment update — transport old m to current tangent space,
        // then blend with current Riemannian gradient.
        // Transport approximation: re-project m to tangent space at current point.
        let m_transported = project_tangent_ambient(&state.point, &state.m, &manifold)?;
        state.m = vec_axpby(config.beta1, &m_transported, 1.0 - config.beta1, &riem_g);

        // Step 3b: Second moment update — element-wise squared gradient.
        // Transport approximation: re-project v to tangent space at current point.
        // After projection, clamp to non-negative: the projection is a linear operator and can
        // map small positive values to tiny negatives due to floating-point; clamping prevents NaN
        // from sqrt later.
        let v_transported_raw = project_tangent_ambient(&state.point, &state.v, &manifold)?;
        let v_transported: Vec<f64> = v_transported_raw.into_iter().map(|x| x.max(0.0)).collect();
        let riem_g_sq = vec_hadamard(&riem_g, &riem_g);
        let v_updated = vec_axpby(config.beta2, &v_transported, 1.0 - config.beta2, &riem_g_sq);
        // Guarantee non-negativity of second moment (invariant must hold after each update).
        state.v = v_updated.into_iter().map(|x| x.max(0.0)).collect();

        // Step 4: Bias correction.
        let beta1_t = config.beta1.powi(t as i32);
        let beta2_t = config.beta2.powi(t as i32);
        let m_hat = state
            .m
            .iter()
            .map(|x| x / (1.0 - beta1_t))
            .collect::<Vec<_>>();
        // Clamp v_hat to non-negative before sqrt to guard against any residual floating-point drift.
        let v_hat: Vec<f64> = state
            .v
            .iter()
            .map(|x| (x / (1.0 - beta2_t)).max(0.0))
            .collect();

        // Step 5: Compute the update tangent vector.
        // direction = -α * m̂ / (sqrt(v̂) + ε)
        let denom = vec_sqrt_eps(&v_hat, config.eps);
        let update_tangent: Vec<f64> = m_hat
            .iter()
            .zip(denom.iter())
            .map(|(mi, di)| -config.lr * mi / di)
            .collect();

        // Step 6: Retract to new point.
        let new_point = retract_ambient(&state.point, &update_tangent, &manifold)?;

        // Update state — moment vectors are carried over; on the next iteration
        // they will be transported (projected) to the new tangent space.
        state.point = new_point;
    }

    Ok(RiemannianAdamResult {
        point: state.point,
        final_gradient_norm: final_grad_norm,
        n_iter: state.t,
        converged,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an orthonormal n×p matrix (first p standard basis vectors extended via QR).
    /// Returns the matrix as a flat row-major Vec<f64>.
    fn make_identity_stiefel(n: usize, p: usize) -> Vec<f64> {
        let mut y = vec![0.0_f64; n * p];
        for k in 0..p.min(n) {
            y[k * p + k] = 1.0;
        }
        y
    }

    /// Check that X^T X ≈ I_p within tolerance.
    fn check_orthonormality(x: &[f64], n: usize, p: usize, tol: f64) -> bool {
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0_f64;
                for r in 0..n {
                    acc += x[r * p + a] * x[r * p + b];
                }
                let expected = if a == b { 1.0 } else { 0.0 };
                if (acc - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the Rayleigh-quotient-style loss `f(X) = -tr(X^T A X)` for a diagonal A.
    ///
    /// On the Stiefel/Grassmann manifold the minimum of `-tr(X^T A X)` is achieved when
    /// the columns of X span the leading eigenspace of A. The gradient is `∇f = -2 A X`.
    /// This loss is manifold-compatible: sign changes in X columns do NOT affect the value,
    /// so the QR retraction sign convention does not interfere with measuring progress.
    fn rayleigh_loss(x: &[f64], a_diag: &[f64], n: usize, p: usize) -> f64 {
        // tr(X^T A X) = sum_{i,j} X_{ij}^2 * A_{ii}
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in 0..p {
                s += x[i * p + j] * x[i * p + j] * a_diag[i];
            }
        }
        -s
    }

    /// Euclidean gradient of `f(X) = -tr(X^T A X)`: `∇f = -2 A X` (diagonal A).
    fn rayleigh_grad(x: &[f64], a_diag: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut g = vec![0.0_f64; n * p];
        for i in 0..n {
            for j in 0..p {
                g[i * p + j] = -2.0 * a_diag[i] * x[i * p + j];
            }
        }
        g
    }

    // ─── Test 1: Loss decreases on Stiefel ───────────────────────────────────

    /// Verify that the Riemannian Adam optimizer reduces the Rayleigh-quotient loss on St(4,2).
    ///
    /// We use `f(X) = -tr(X^T diag(4,3,2,1) X)` (minimum = -7 at `X = ±[e1, e2]`).
    /// Starting from a 45° mixed point (Rayleigh = -5), the optimizer should decrease the loss.
    /// The initial point is chosen with negative-dominant first components to be consistent with
    /// the QR retraction sign convention (which returns Q columns aligned with negative-dominant
    /// directions of the input when the pivot is negative).
    #[test]
    fn radam_stiefel_decreases_loss() {
        let n = 4;
        let p = 2;
        let a_diag = vec![4.0_f64, 3.0, 2.0, 1.0];

        // 45° mix in (e1, e3) for col0 and (e2, e4) for col1 — NOT at the Rayleigh minimum.
        // Rayleigh at this point = -( (4+2)/2 + (3+1)/2 ) = -(3 + 2) = -5.
        // Columns are negative-dominant so QR retraction stays consistent.
        let c = (std::f64::consts::PI / 4.0_f64).cos(); // 1/sqrt(2)
        let s = (std::f64::consts::PI / 4.0_f64).sin(); // 1/sqrt(2)
        let mut init = vec![0.0_f64; n * p];
        // Row-major n×p matrix: element at (row r, col c) is init[r * p + c].
        init[0] = -c; // row0 col0
        init[2 * p] = -s; // row2 col0
        init[p + 1] = -c; // row1 col1
        init[3 * p + 1] = -s; // row3 col1

        let loss_init = rayleigh_loss(&init, &a_diag, n, p);

        let grad_fn = {
            let a = a_diag.clone();
            move |x: &[f64]| rayleigh_grad(x, &a, n, p)
        };
        let config = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 500,
            tol: 1e-9,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config)
            .expect("optimizer must not fail");

        let loss_final = rayleigh_loss(&result.point, &a_diag, n, p);
        assert!(
            loss_final < loss_init,
            "loss did not decrease: init={loss_init:.6} final={loss_final:.6}"
        );
    }

    // ─── Test 2: Loss decreases on Grassmann ─────────────────────────────────

    /// Same Rayleigh-quotient loss on Gr(4,2).
    /// Grassmann optimization is invariant to column sign flips, so any starting point works.
    #[test]
    fn radam_grassmann_decreases_loss() {
        let n = 4;
        let p = 2;
        let a_diag = vec![4.0_f64, 3.0, 2.0, 1.0];

        // Same 45° mixed starting point: Rayleigh = -5, minimum = -7.
        let c = (std::f64::consts::PI / 4.0_f64).cos();
        let s = (std::f64::consts::PI / 4.0_f64).sin();
        let mut init = vec![0.0_f64; n * p];
        // Row-major n×p matrix: element at (row r, col c) is init[r * p + c].
        init[0] = -c;
        init[2 * p] = -s;
        init[p + 1] = -c;
        init[3 * p + 1] = -s;

        let loss_init = rayleigh_loss(&init, &a_diag, n, p);

        let grad_fn = {
            let a = a_diag.clone();
            move |x: &[f64]| rayleigh_grad(x, &a, n, p)
        };
        let config = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 500,
            tol: 1e-9,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Grassmann { n, p }, &config)
            .expect("optimizer must not fail");

        let loss_final = rayleigh_loss(&result.point, &a_diag, n, p);
        assert!(
            loss_final < loss_init,
            "loss did not decrease: init={loss_init:.6} final={loss_final:.6}"
        );
    }

    // ─── Test 3: Stays on Stiefel ─────────────────────────────────────────────

    #[test]
    fn radam_stays_on_stiefel() {
        let n = 4;
        let p = 2;
        let a_diag = vec![4.0_f64, 3.0, 2.0, 1.0];
        let c = (std::f64::consts::PI / 4.0_f64).cos();
        let s = (std::f64::consts::PI / 4.0_f64).sin();
        let mut init = vec![0.0_f64; n * p];
        // Row-major n×p matrix: element at (row r, col c) is init[r * p + c].
        init[0] = -c;
        init[2 * p] = -s;
        init[p + 1] = -c;
        init[3 * p + 1] = -s;

        let grad_fn = {
            let a = a_diag.clone();
            move |x: &[f64]| rayleigh_grad(x, &a, n, p)
        };
        let config = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 150,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config)
            .expect("optimizer must not fail");

        assert!(
            check_orthonormality(&result.point, n, p, 1e-4),
            "result not on Stiefel manifold (X^T X ≠ I)"
        );
    }

    // ─── Test 4: Stays on Grassmann ───────────────────────────────────────────

    #[test]
    fn radam_stays_on_grassmann() {
        let n = 4;
        let p = 2;
        let a_diag = vec![4.0_f64, 3.0, 2.0, 1.0];
        let c = (std::f64::consts::PI / 4.0_f64).cos();
        let s = (std::f64::consts::PI / 4.0_f64).sin();
        let mut init = vec![0.0_f64; n * p];
        // Row-major n×p matrix: element at (row r, col c) is init[r * p + c].
        init[0] = -c;
        init[2 * p] = -s;
        init[p + 1] = -c;
        init[3 * p + 1] = -s;

        let grad_fn = {
            let a = a_diag.clone();
            move |x: &[f64]| rayleigh_grad(x, &a, n, p)
        };
        let config = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 150,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Grassmann { n, p }, &config)
            .expect("optimizer must not fail");

        assert!(
            check_orthonormality(&result.point, n, p, 1e-4),
            "result not on Grassmann manifold (X^T X ≠ I)"
        );
    }

    // ─── Test 5: Convergence flag on easy problem ─────────────────────────────

    #[test]
    fn radam_convergence_flag() {
        // Minimise ||X - X_init||²_F on Stiefel where X_init is already optimal.
        // Gradient at X_init is 2*(X - X_init) = 0, so it should converge immediately.
        let n = 4;
        let p = 2;
        let init = make_identity_stiefel(n, p);

        // Gradient is zero at the initial point => converges on first evaluation.
        let grad_fn = |_x: &[f64]| vec![0.0_f64; n * p];

        let config = RiemannianAdamConfig {
            lr: 0.01,
            tol: 1e-6,
            max_iter: 200,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config)
            .expect("optimizer must not fail");

        assert!(result.converged, "expected convergence flag to be true");
    }

    // ─── Test 6: Gradient norm decreases overall ──────────────────────────────

    #[test]
    fn radam_gradient_norm_decreasing() {
        // Compare the Riemannian gradient norm after 1 iteration vs. after many,
        // using the sign-invariant Rayleigh-quotient objective.
        let n = 4;
        let p = 2;
        let a_diag = vec![4.0_f64, 3.0, 2.0, 1.0];
        let c = (std::f64::consts::PI / 4.0_f64).cos();
        let s_val = (std::f64::consts::PI / 4.0_f64).sin();
        let mut init = vec![0.0_f64; n * p];
        // Row-major n×p matrix: element at (row r, col c) is init[r * p + c].
        init[0] = -c;
        init[2 * p] = -s_val;
        init[p + 1] = -c;
        init[3 * p + 1] = -s_val;

        let make_grad = |a: Vec<f64>| move |x: &[f64]| -> Vec<f64> { rayleigh_grad(x, &a, n, p) };

        // 1 iteration — captures initial Riemannian gradient norm.
        let config_short = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 1,
            tol: 1e-15,
            ..Default::default()
        };
        let result_short = riemannian_adam(
            &init,
            make_grad(a_diag.clone()),
            ManifoldType::Stiefel { n, p },
            &config_short,
        )
        .expect("optimizer must not fail");
        let norm_initial = result_short.final_gradient_norm;

        // 200 iterations — should reduce the Riemannian gradient norm substantially.
        let config_long = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 200,
            tol: 1e-15,
            ..Default::default()
        };
        let result_long = riemannian_adam(
            &init,
            make_grad(a_diag.clone()),
            ManifoldType::Stiefel { n, p },
            &config_long,
        )
        .expect("optimizer must not fail");
        let norm_final = result_long.final_gradient_norm;

        assert!(
            norm_final < norm_initial,
            "Riemannian gradient norm did not decrease: initial={norm_initial:.6} final={norm_final:.6}"
        );
    }

    // ─── Test 7: Output shape ─────────────────────────────────────────────────

    #[test]
    fn radam_result_shape() {
        let n = 5;
        let p = 3;
        let init = make_identity_stiefel(n, p);
        let grad_fn = |x: &[f64]| x.to_vec(); // gradient = x (arbitrary)
        let config = RiemannianAdamConfig {
            max_iter: 10,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config)
            .expect("optimizer must not fail");
        assert_eq!(result.point.len(), n * p, "output point has wrong length");
    }

    // ─── Test 8: Zero learning rate → point unchanged ─────────────────────────

    #[test]
    fn radam_zero_lr_no_change() {
        let n = 4;
        let p = 2;
        let init = make_identity_stiefel(n, p);
        let init_copy = init.clone();

        let grad_fn = |x: &[f64]| x.iter().map(|xi| xi * 0.5).collect::<Vec<_>>();
        let config = RiemannianAdamConfig {
            lr: 0.0,
            max_iter: 50,
            tol: 1e-12, // very tight so it won't converge via grad norm
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config)
            .expect("optimizer must not fail");

        for (a, b) in result.point.iter().zip(init_copy.iter()) {
            assert!((a - b).abs() < 1e-15, "point changed with lr=0: {a} != {b}");
        }
    }

    // ─── Test 9: Start from valid Stiefel point (QR of near-random matrix) ────

    #[test]
    fn radam_stiefel_identity_start_valid() {
        let n = 6;
        let p = 3;
        // Build a non-trivial starting point: QR of a structured matrix.
        // Use a Hadamard-like pattern (scaled to be full rank).
        let mut raw = vec![0.0_f64; n * p];
        for i in 0..n {
            for j in 0..p {
                // Simple structured values to ensure full rank.
                raw[i * p + j] = if (i + j) % 2 == 0 { 1.0 } else { 0.5 };
                if i == j {
                    raw[i * p + j] += 2.0;
                }
            }
        }
        // Orthonormalise via Gram-Schmidt (manual, to avoid importing QR directly in test).
        // Column 0: raw col 0
        let mut basis = vec![vec![0.0_f64; n]; p];
        for j in 0..p {
            let mut col: Vec<f64> = (0..n).map(|i| raw[i * p + j]).collect();
            // Subtract projections onto previous basis vectors.
            for prev in basis.iter().take(j) {
                let dot: f64 = col.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for i in 0..n {
                    col[i] -= dot * prev[i];
                }
            }
            let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            for i in 0..n {
                basis[j][i] = col[i] / norm;
            }
        }
        let mut init = vec![0.0_f64; n * p];
        for i in 0..n {
            for j in 0..p {
                init[i * p + j] = basis[j][i];
            }
        }
        assert!(
            check_orthonormality(&init, n, p, 1e-10),
            "start point must be on Stiefel"
        );

        let target: Vec<f64> = (0..n * p).map(|k| (k as f64) * 0.1).collect();
        let grad_fn = {
            let t = target.clone();
            move |x: &[f64]| {
                x.iter()
                    .zip(t.iter())
                    .map(|(xi, ti)| 2.0 * (xi - ti))
                    .collect()
            }
        };
        let config = RiemannianAdamConfig {
            lr: 0.02,
            max_iter: 100,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config)
            .expect("optimizer must not fail");

        // Result must still be on Stiefel.
        assert!(
            check_orthonormality(&result.point, n, p, 1e-4),
            "result not on Stiefel after optimization"
        );
    }

    // ─── Test 10: Empty init returns EmptyInput error ─────────────────────────

    #[test]
    fn empty_init_returns_error() {
        let grad_fn = |_x: &[f64]| vec![];
        let config = RiemannianAdamConfig::default();
        // We must use a manifold whose ambient dim is non-zero, but init is empty.
        // The EmptyInput check fires before the dim check.
        let result = riemannian_adam(&[], grad_fn, ManifoldType::Stiefel { n: 4, p: 2 }, &config);
        assert!(
            matches!(result, Err(ManifoldError::EmptyInput)),
            "expected EmptyInput error, got: {result:?}"
        );
    }

    // ─── Bonus test 11: Invalid lr returns error ──────────────────────────────

    #[test]
    fn radam_negative_lr_returns_error() {
        let n = 4;
        let p = 2;
        let init = make_identity_stiefel(n, p);
        let grad_fn = |x: &[f64]| x.to_vec();
        let config = RiemannianAdamConfig {
            lr: -0.01,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Stiefel { n, p }, &config);
        assert!(
            matches!(result, Err(ManifoldError::InvalidParameter { .. })),
            "expected InvalidParameter error for negative lr"
        );
    }

    // ─── Bonus test 12: Grassmann result shape ────────────────────────────────

    #[test]
    fn radam_grassmann_result_shape() {
        let n = 5;
        let p = 2;
        let init = make_identity_stiefel(n, p);
        let grad_fn = |x: &[f64]| x.to_vec();
        let config = RiemannianAdamConfig {
            max_iter: 10,
            ..Default::default()
        };
        let result = riemannian_adam(&init, grad_fn, ManifoldType::Grassmann { n, p }, &config)
            .expect("optimizer must not fail");
        assert_eq!(result.point.len(), n * p);
    }
}
