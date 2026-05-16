//! Smoothers used inside multigrid V-cycles.
//!
//! - `weighted_jacobi_smooth`: damped Jacobi for `-u''=f` on a 1D uniform grid.
//! - `gauss_seidel_1d_smooth`: in-place Gauss-Seidel sweep.

use crate::error::{PdeError, PdeResult};

/// One weighted Jacobi sweep on a 1D Poisson stencil with mesh spacing `h`.
///
/// Updates only interior `1..n-1` cells with the discrete operator
/// `(1/h^2)*(2 u_i - u_{i-1} - u_{i+1}) = f_i`.
pub fn weighted_jacobi_smooth(
    u: &mut [f64],
    f: &[f64],
    h: f64,
    omega: f64,
    n_sweeps: usize,
) -> PdeResult<()> {
    let n = u.len();
    if f.len() != n {
        return Err(PdeError::DimensionMismatch { a: f.len(), b: n });
    }
    if n < 3 {
        return Err(PdeError::InvalidGrid("smoother needs n >= 3".into()));
    }
    if h <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: "must be positive".into(),
        });
    }
    let h2 = h * h;
    for _ in 0..n_sweeps {
        let old = u.to_vec();
        for i in 1..n - 1 {
            // new = 0.5*(old[i-1] + old[i+1] + h^2*f[i])
            let new_val = 0.5 * (old[i - 1] + old[i + 1] + h2 * f[i]);
            u[i] = (1.0 - omega) * old[i] + omega * new_val;
        }
    }
    Ok(())
}

/// Gauss-Seidel red-black sweep on a 1D Poisson stencil.
pub fn gauss_seidel_1d_smooth(u: &mut [f64], f: &[f64], h: f64, n_sweeps: usize) -> PdeResult<()> {
    let n = u.len();
    if f.len() != n {
        return Err(PdeError::DimensionMismatch { a: f.len(), b: n });
    }
    if n < 3 {
        return Err(PdeError::InvalidGrid("smoother needs n >= 3".into()));
    }
    let h2 = h * h;
    for _ in 0..n_sweeps {
        for color in 0..2 {
            for i in 1..n - 1 {
                if i % 2 != color {
                    continue;
                }
                u[i] = 0.5 * (u[i - 1] + u[i + 1] + h2 * f[i]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_jacobi_const_rhs() {
        // -u'' = 2 on [0,1], u(0)=u(1)=0; exact u = x(1-x)
        let n = 17;
        let h = 1.0 / (n - 1) as f64;
        let mut u = vec![0.0; n];
        let f = vec![2.0; n];
        weighted_jacobi_smooth(&mut u, &f, h, 0.7, 1000).expect("ok");
        // check approximate solution at midpoint
        let mid = u[n / 2];
        assert!(mid > 0.0);
        let x_mid = 0.5;
        let exact = x_mid * (1.0 - x_mid);
        assert!((mid - exact).abs() < 0.1, "mid={mid} exact={exact}");
    }

    #[test]
    fn gs_1d_converges() {
        let n = 17;
        let h = 1.0 / (n - 1) as f64;
        let mut u = vec![0.0; n];
        let f = vec![2.0; n];
        gauss_seidel_1d_smooth(&mut u, &f, h, 2000).expect("ok");
        let x_mid = 0.5;
        let exact = x_mid * (1.0 - x_mid);
        assert!((u[n / 2] - exact).abs() < 1e-3);
    }
}
