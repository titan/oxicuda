//! 1D geometric multigrid V-cycle for `-u''(x) = f(x)` on `[0, L]` with Dirichlet 0.

use crate::error::{PdeError, PdeResult};
use crate::multigrid::restrict_prolong::{prolong_1d, restrict_1d};
use crate::multigrid::smoother::weighted_jacobi_smooth;

/// Compute the residual `r = f - A u` for the 1D Poisson operator with mesh size `h`.
fn residual_1d(u: &[f64], f: &[f64], h: f64) -> Vec<f64> {
    let n = u.len();
    let inv_h2 = 1.0 / (h * h);
    let mut r = vec![0.0; n];
    for i in 1..n - 1 {
        let lap = inv_h2 * (2.0 * u[i] - u[i - 1] - u[i + 1]);
        r[i] = f[i] - lap;
    }
    r
}

/// Apply one V-cycle of 1D geometric multigrid. Returns updated `u`.
///
/// - `n_pre`, `n_post`: number of pre/post smoothing sweeps.
/// - At the coarsest level (n <= 3), the system reduces to one unknown and is
///   solved directly.
pub fn v_cycle_1d(u: &mut [f64], f: &[f64], h: f64, n_pre: usize, n_post: usize) -> PdeResult<()> {
    let n = u.len();
    if f.len() != n {
        return Err(PdeError::DimensionMismatch { a: f.len(), b: n });
    }
    if n < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "v_cycle_1d: n={n} must be >=3"
        )));
    }
    if n % 2 == 0 {
        return Err(PdeError::InvalidGrid(format!(
            "v_cycle_1d: n={n} must be odd"
        )));
    }
    if n == 3 {
        // single interior node: u[1] = (h^2 * f[1] + u[0] + u[2]) / 2
        u[1] = 0.5 * (u[0] + u[2] + h * h * f[1]);
        return Ok(());
    }
    // pre-smooth
    weighted_jacobi_smooth(u, f, h, 2.0 / 3.0, n_pre)?;
    // residual
    let r = residual_1d(u, f, h);
    // restrict
    let r_coarse = restrict_1d(&r)?;
    let n_coarse = r_coarse.len();
    let mut e_coarse = vec![0.0; n_coarse];
    let h_coarse = h * 2.0;
    // recurse on coarse defect equation A e = r
    v_cycle_1d(&mut e_coarse, &r_coarse, h_coarse, n_pre, n_post)?;
    // prolong
    let e_fine = prolong_1d(&e_coarse)?;
    for i in 0..n {
        u[i] += e_fine[i];
    }
    // post-smooth
    weighted_jacobi_smooth(u, f, h, 2.0 / 3.0, n_post)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v_cycle_reduces_residual() {
        let n = 33;
        let h = 1.0 / (n - 1) as f64;
        let mut u = vec![0.0; n];
        let f: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * h;
                let pi = std::f64::consts::PI;
                pi * pi * (pi * x).sin()
            })
            .collect();
        // initial residual
        let initial_res = {
            let r = residual_1d(&u, &f, h);
            r.iter().map(|x| x * x).sum::<f64>().sqrt()
        };
        for _ in 0..8 {
            v_cycle_1d(&mut u, &f, h, 3, 3).expect("ok");
        }
        let final_res = {
            let r = residual_1d(&u, &f, h);
            r.iter().map(|x| x * x).sum::<f64>().sqrt()
        };
        // V-cycle should drop residual significantly
        assert!(
            final_res < 1e-3 * initial_res,
            "init={initial_res} final={final_res}"
        );
    }

    #[test]
    fn v_cycle_converges_to_analytic() {
        let n = 33;
        let h = 1.0 / (n - 1) as f64;
        let mut u = vec![0.0; n];
        let f = vec![2.0; n];
        // u(0)=u(1)=0, -u''=2 => u(x)=x(1-x)
        for _ in 0..15 {
            v_cycle_1d(&mut u, &f, h, 5, 5).expect("ok");
        }
        for (i, &ui) in u.iter().enumerate().take(n - 1).skip(1) {
            let x = i as f64 * h;
            let exact = x * (1.0 - x);
            assert!((ui - exact).abs() < 1e-3, "i={i} u={ui} exact={exact}");
        }
    }
}
