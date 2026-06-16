//! Ordinary differential equation integrators.

pub mod adams_bashforth_moulton;
pub mod bdf12;
pub mod dae;
pub mod dopri5;
pub mod explicit_euler;
pub mod heun;
pub mod imex_euler;
pub mod radau_iia;
pub mod rk4;
pub mod rk45;
pub mod rosenbrock_w;
pub mod sdirk;

use crate::error::{NumericError, NumericResult};

/// Forward-difference approximation of the Jacobian `∂f/∂y` of a vector field.
///
/// Returns the `n × n` Jacobian in row-major order, where entry `(i, k)` holds
/// `∂fᵢ/∂yₖ`. Used by the implicit integrators ([`radau_iia`], [`sdirk`]) when no
/// analytic Jacobian is supplied. The perturbation for column `k` is
/// `eps · max(|yₖ|, 1)`.
pub(crate) fn finite_diff_jacobian<F>(f: &F, t: f64, y: &[f64], eps: f64) -> NumericResult<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();
    let f0 = f(t, y);
    if f0.len() != n {
        return Err(NumericError::ShapeMismatch {
            expected: vec![n],
            got: vec![f0.len()],
        });
    }
    let mut jac = vec![0.0_f64; n * n];
    let mut yp = y.to_vec();
    for k in 0..n {
        let dk = eps * y[k].abs().max(1.0);
        yp[k] = y[k] + dk;
        let fk = f(t, &yp);
        yp[k] = y[k];
        if fk.len() != n {
            return Err(NumericError::ShapeMismatch {
                expected: vec![n],
                got: vec![fk.len()],
            });
        }
        for i in 0..n {
            jac[i * n + k] = (fk[i] - f0[i]) / dk;
        }
    }
    Ok(jac)
}
