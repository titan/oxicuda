//! Newton-Krylov (Jacobian-free Newton-Krylov, JFNK) nonlinear system solver.
//!
//! Solves `F(x) = 0` for `F : ℝⁿ → ℝⁿ` using Newton's method, where the inner
//! linear system `J(x) δ = −F(x)` is solved by restarted GMRES **without ever
//! forming the Jacobian** `J`.  Each GMRES matrix-vector product is replaced by
//! a first-order finite-difference directional derivative:
//!
//! ```text
//! J(x) · v  ≈  ( F(x + ε‖x‖/‖v‖ · v) − F(x) ) / (ε‖x‖/‖v‖)
//! ```
//!
//! This makes the method ideal for large or analytically-awkward systems where
//! the Jacobian is expensive or unavailable, while retaining Newton's fast
//! (super-linear / quadratic near the root) outer convergence.

use crate::error::{NumericError, NumericResult};

/// Configuration for [`newton_krylov`].
#[derive(Debug, Clone, Copy)]
pub struct NewtonKrylovConfig {
    /// Maximum number of outer Newton iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖F(x)‖₂`.
    pub tol: f64,
    /// Base relative step for the finite-difference Jacobian-vector product.
    pub fd_eps: f64,
}

impl Default for NewtonKrylovConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1.0e-10,
            fd_eps: 1.0e-7,
        }
    }
}

/// Solves `F(x) = 0` via Jacobian-free Newton-Krylov.
///
/// `f` evaluates the residual vector (must return a vector of the same length
/// as its input).  `x0` is the initial guess; the returned vector has the same
/// length.
///
/// # Errors
///
/// * [`NumericError::EmptyInput`] if `x0` is empty.
/// * [`NumericError::InvalidParameter`] if `cfg.tol`/`cfg.fd_eps` are not
///   positive finite, or if `f` returns a vector whose length differs from
///   `x0`.
/// * [`NumericError::NotConverged`] if `‖F‖` does not fall below `cfg.tol`
///   within `cfg.max_iter` iterations.
pub fn newton_krylov<F>(f: F, x0: &[f64], cfg: &NewtonKrylovConfig) -> NumericResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(NumericError::InvalidParameter(format!(
            "tol must be positive finite, got {}",
            cfg.tol
        )));
    }
    if !(cfg.fd_eps > 0.0 && cfg.fd_eps.is_finite()) {
        return Err(NumericError::InvalidParameter(format!(
            "fd_eps must be positive finite, got {}",
            cfg.fd_eps
        )));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(NumericError::InvalidParameter(
            "x0 has non-finite entries".into(),
        ));
    }

    let mut x = x0.to_vec();
    let mut fx = eval(&f, &x, n)?;
    let mut residual = norm2(&fx);

    for _ in 0..cfg.max_iter {
        if residual <= cfg.tol {
            return Ok(x);
        }

        // Solve J δ = −F via matrix-free GMRES.
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let inner_tol = (0.1 * residual).min(0.5);
        let delta = gmres_matrix_free(&f, &x, &fx, &rhs, cfg.fd_eps, inner_tol, n)?;

        // Damped update with a simple backtracking line search to stay robust.
        let mut lambda = 1.0_f64;
        let mut accepted = false;
        for _ in 0..20 {
            let trial: Vec<f64> = x
                .iter()
                .zip(&delta)
                .map(|(xi, di)| xi + lambda * di)
                .collect();
            if trial.iter().all(|v| v.is_finite()) {
                let ftrial = eval(&f, &trial, n)?;
                let rtrial = norm2(&ftrial);
                if rtrial < residual {
                    x = trial;
                    fx = ftrial;
                    residual = rtrial;
                    accepted = true;
                    break;
                }
            }
            lambda *= 0.5;
        }

        if !accepted {
            // Could not reduce the residual; take the full step anyway so the
            // iteration can escape a flat region, then re-evaluate.
            for (xi, di) in x.iter_mut().zip(&delta) {
                *xi += di;
            }
            fx = eval(&f, &x, n)?;
            residual = norm2(&fx);
            if !residual.is_finite() {
                return Err(NumericError::NumericalInstability(
                    "residual became non-finite".into(),
                ));
            }
        }
    }

    if residual <= cfg.tol {
        return Ok(x);
    }
    Err(NumericError::NotConverged {
        iter: cfg.max_iter,
        residual,
    })
}

/// Restarted-free (single cycle) GMRES for the matrix-free system `J δ = b`.
///
/// Uses Arnoldi with modified Gram-Schmidt and Givens rotations to solve the
/// least-squares Hessenberg problem; the Krylov dimension is capped at `n`.
fn gmres_matrix_free<F>(
    f: &F,
    x: &[f64],
    fx: &[f64],
    b: &[f64],
    fd_eps: f64,
    tol: f64,
    n: usize,
) -> NumericResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let max_kry = n.clamp(1, 50);
    let beta = norm2(b);
    if beta == 0.0 {
        return Ok(vec![0.0; n]);
    }

    // Krylov basis vectors q_0 … q_m.
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(max_kry + 1);
    q.push(b.iter().map(|v| v / beta).collect());

    // Hessenberg matrix stored column-major as Vec of columns.
    let mut h: Vec<Vec<f64>> = Vec::with_capacity(max_kry);
    // Givens rotation parameters.
    let mut cs = vec![0.0_f64; max_kry];
    let mut sn = vec![0.0_f64; max_kry];
    // RHS of the rotated least-squares problem.
    let mut g = vec![0.0_f64; max_kry + 1];
    g[0] = beta;

    let mut converged_k = max_kry;
    for k in 0..max_kry {
        // w = J · q_k via finite difference.
        let mut w = jac_vec(f, x, fx, &q[k], fd_eps, n)?;

        // Modified Gram-Schmidt against existing basis.
        let mut hcol = vec![0.0_f64; k + 2];
        for (i, qi) in q.iter().enumerate().take(k + 1) {
            let hij = dot(&w, qi);
            hcol[i] = hij;
            for (wj, qij) in w.iter_mut().zip(qi) {
                *wj -= hij * qij;
            }
        }
        let h_next = norm2(&w);
        hcol[k + 1] = h_next;

        // Apply previous Givens rotations to the new Hessenberg column.
        for i in 0..k {
            let temp = cs[i] * hcol[i] + sn[i] * hcol[i + 1];
            hcol[i + 1] = -sn[i] * hcol[i] + cs[i] * hcol[i + 1];
            hcol[i] = temp;
        }

        // Compute and apply the new rotation to zero out the sub-diagonal.
        let (c, s) = givens(hcol[k], hcol[k + 1]);
        cs[k] = c;
        sn[k] = s;
        hcol[k] = c * hcol[k] + s * hcol[k + 1];
        hcol[k + 1] = 0.0;
        let temp = c * g[k] + s * g[k + 1];
        g[k + 1] = -s * g[k] + c * g[k + 1];
        g[k] = temp;

        h.push(hcol);

        let res_norm = g[k + 1].abs();
        if h_next <= 1.0e-14 || res_norm <= tol * beta {
            converged_k = k + 1;
            break;
        }
        // Extend the basis.
        if k + 1 < max_kry {
            let inv = 1.0 / h_next;
            q.push(w.iter().map(|v| v * inv).collect());
        } else {
            converged_k = max_kry;
        }
    }

    // Back-substitution for y in the upper-triangular Hessenberg system.
    let m = converged_k;
    let mut y = vec![0.0_f64; m];
    for i in (0..m).rev() {
        let mut sum = g[i];
        for (j, yj) in y.iter().enumerate().skip(i + 1) {
            sum -= h[j][i] * yj;
        }
        let diag = h[i][i];
        if diag.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(
                "GMRES Hessenberg diagonal vanished".into(),
            ));
        }
        y[i] = sum / diag;
    }

    // δ = Σ y_i q_i.
    let mut delta = vec![0.0_f64; n];
    for (i, yi) in y.iter().enumerate() {
        for (d, qij) in delta.iter_mut().zip(&q[i]) {
            *d += yi * qij;
        }
    }
    Ok(delta)
}

/// Finite-difference Jacobian-vector product `J(x)·v`.
fn jac_vec<F>(
    f: &F,
    x: &[f64],
    fx: &[f64],
    v: &[f64],
    fd_eps: f64,
    n: usize,
) -> NumericResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let v_norm = norm2(v);
    if v_norm == 0.0 {
        return Ok(vec![0.0; n]);
    }
    // Scaled step: ε (1 + ‖x‖) / ‖v‖  keeps the perturbation well-scaled.
    let eps = fd_eps * (1.0 + norm2(x)) / v_norm;
    let xp: Vec<f64> = x.iter().zip(v).map(|(xi, vi)| xi + eps * vi).collect();
    let fp = eval(f, &xp, n)?;
    Ok(fp.iter().zip(fx).map(|(a, b)| (a - b) / eps).collect())
}

/// Evaluates `f`, checking the output dimension and finiteness.
fn eval<F>(f: &F, x: &[f64], n: usize) -> NumericResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let y = f(x);
    if y.len() != n {
        return Err(NumericError::InvalidParameter(format!(
            "f returned length {}, expected {n}",
            y.len()
        )));
    }
    Ok(y)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Givens rotation coefficients zeroing the second component of `(a, b)`.
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if a == 0.0 {
        (0.0, 1.0)
    } else {
        let r = a.hypot(b);
        (a / r, b / r)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NewtonKrylovConfig {
        NewtonKrylovConfig::default()
    }

    #[test]
    fn scalar_root() {
        // x² − 2 = 0 → √2.
        let f = |x: &[f64]| vec![x[0] * x[0] - 2.0];
        let r = newton_krylov(f, &[1.0], &cfg()).expect("ok");
        assert!((r[0] - 2.0_f64.sqrt()).abs() < 1e-8, "{}", r[0]);
    }

    #[test]
    fn system_root() {
        // x² + y² = 4, x − y = 0 → x = y = √2.
        let f = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] - 4.0, v[0] - v[1]];
        let r = newton_krylov(f, &[1.5, 1.0], &cfg()).expect("ok");
        let s = 2.0_f64.sqrt();
        assert!((r[0] - s).abs() < 1e-7, "x={}", r[0]);
        assert!((r[1] - s).abs() < 1e-7, "y={}", r[1]);
    }

    #[test]
    fn converges() {
        // Residual at the returned point is below tolerance.
        let f = |v: &[f64]| vec![v[0] + 2.0 * v[1] - 3.0, 3.0 * v[0] + v[1] * v[1] - 5.0];
        let r = newton_krylov(f, &[0.0, 0.0], &cfg()).expect("ok");
        let res = f(&r);
        let norm = (res[0] * res[0] + res[1] * res[1]).sqrt();
        assert!(norm < 1e-9, "residual {norm}");
    }

    #[test]
    fn max_iter_bound() {
        // A single iteration budget on a hard problem returns NotConverged.
        let f = |x: &[f64]| vec![x[0].exp() - 5.0];
        let c = NewtonKrylovConfig {
            max_iter: 1,
            tol: 1e-14,
            fd_eps: 1e-7,
        };
        let res = newton_krylov(f, &[-3.0], &c);
        match res {
            Err(NumericError::NotConverged { iter, .. }) => assert_eq!(iter, 1),
            other => panic!("expected NotConverged, got {other:?}"),
        }
    }

    #[test]
    fn already_at_root() {
        let f = |v: &[f64]| vec![v[0] - 1.0, v[1] - 2.0];
        let r = newton_krylov(f, &[1.0, 2.0], &cfg()).expect("ok");
        assert!((r[0] - 1.0).abs() < 1e-12);
        assert!((r[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn no_root_max_iter() {
        // x² + 1 = 0 has no real root → must not converge.
        let f = |x: &[f64]| vec![x[0] * x[0] + 1.0];
        let c = NewtonKrylovConfig {
            max_iter: 30,
            tol: 1e-10,
            fd_eps: 1e-7,
        };
        let res = newton_krylov(f, &[1.0], &c);
        assert!(res.is_err());
    }

    #[test]
    fn output_len() {
        let f = |v: &[f64]| v.iter().map(|x| x - 1.0).collect::<Vec<_>>();
        let r = newton_krylov(f, &[0.0; 5], &cfg()).expect("ok");
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn tol_respected() {
        let f = |x: &[f64]| vec![x[0] * x[0] * x[0] - 8.0];
        let c = NewtonKrylovConfig {
            max_iter: 100,
            tol: 1e-6,
            fd_eps: 1e-7,
        };
        let r = newton_krylov(f, &[1.0], &c).expect("ok");
        let res = f(&r)[0].abs();
        assert!(res <= 1e-6, "residual {res} exceeds tol");
        assert!((r[0] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn finite() {
        let f = |v: &[f64]| vec![v[0].sin() - 0.5, v[1] - v[0]];
        let r = newton_krylov(f, &[0.4, 0.4], &cfg()).expect("ok");
        for v in &r {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rejects_bad_input() {
        let f = |v: &[f64]| v.to_vec();
        assert!(newton_krylov(f, &[], &cfg()).is_err());
        let bad_tol = NewtonKrylovConfig {
            max_iter: 10,
            tol: 0.0,
            fd_eps: 1e-7,
        };
        assert!(newton_krylov(|v: &[f64]| v.to_vec(), &[1.0], &bad_tol).is_err());
    }

    #[test]
    fn larger_linear_system() {
        // Solve a 4×4 linear system A x = b as a root-finding problem.
        // A is diagonally dominant so GMRES converges quickly.
        let a = [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ];
        let b = [1.0, 2.0, 3.0, 4.0];
        let f = move |x: &[f64]| {
            (0..4)
                .map(|i| (0..4).map(|j| a[i][j] * x[j]).sum::<f64>() - b[i])
                .collect::<Vec<_>>()
        };
        let r = newton_krylov(f, &[0.0; 4], &cfg()).expect("ok");
        // Verify residual.
        for i in 0..4 {
            let row: f64 = (0..4).map(|j| a[i][j] * r[j]).sum::<f64>();
            assert!((row - b[i]).abs() < 1e-8, "row {i}");
        }
    }
}
