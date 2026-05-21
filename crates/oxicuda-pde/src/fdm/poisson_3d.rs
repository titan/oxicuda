//! 3D Poisson solver: `-Δu = f` on a rectangular box with Dirichlet BCs.
//!
//! # Mathematical formulation
//!
//! Solve
//!
//! ```text
//!     -Δu(x,y,z) = f(x,y,z),   (x,y,z) ∈ [0,Lx]×[0,Ly]×[0,Lz]
//!              u = g,           on ∂Ω
//! ```
//!
//! on a uniform tensor grid of size `(nx, ny, nz)` with isotropic spacing
//! `h = dx = dy = dz`.
//!
//! # Discretisation
//!
//! Using the standard 7-point Laplacian stencil:
//!
//! ```text
//!     (6·u(i,j,k) − u(i±1,j,k) − u(i,j±1,k) − u(i,j,k±1)) / h² = f(i,j,k)
//! ```
//!
//! # Solver
//!
//! Red-black (checkerboard) Gauss-Seidel with successive over-relaxation (SOR)
//! using factor `ω ∈ (0, 2)`. For ω = 1 this reduces to plain Gauss-Seidel; the
//! optimal ω for an `N³` grid is `≈ 2 / (1 + sin(π/N))` (≈ 1.5–1.9 for typical
//! `N`). The red-black ordering decouples odd/even checkerboard cells so that
//! each colour sweep updates independently of itself — this is the natural
//! parallel ordering for the 7-point stencil.
//!
//! Storage convention: `u[i + nx*(j + ny*k)]` (x is the fastest-varying index,
//! z the slowest).
//!
//! # Reference
//!
//! LeVeque, R. J., *Finite Difference Methods for Ordinary and Partial
//! Differential Equations: Steady-State and Time-Dependent Problems*,
//! SIAM, 2007 — Chapters 3 (Elliptic Equations) and 4 (Iterative Methods).

use crate::error::{PdeError, PdeResult};

/// Configuration for the 3D Poisson SOR-Gauss-Seidel solver.
#[derive(Debug, Clone, Copy)]
pub struct Poisson3dConfig {
    /// Number of grid points along x (including both boundary planes); `nx ≥ 3`.
    pub nx: usize,
    /// Number of grid points along y; `ny ≥ 3`.
    pub ny: usize,
    /// Number of grid points along z; `nz ≥ 3`.
    pub nz: usize,
    /// Isotropic grid spacing `h = dx = dy = dz > 0`.
    pub h: f64,
    /// Over-relaxation factor `ω ∈ (0, 2)`. `ω = 1` ⇒ plain Gauss-Seidel.
    pub omega: f64,
    /// Maximum number of SOR sweeps before bailing out.
    pub max_iter: usize,
    /// L2 residual tolerance (over the interior nodes) for early termination.
    pub tol: f64,
}

impl Default for Poisson3dConfig {
    fn default() -> Self {
        Self {
            nx: 17,
            ny: 17,
            nz: 17,
            h: 1.0 / 16.0,
            omega: 1.5,
            max_iter: 10_000,
            tol: 1.0e-8,
        }
    }
}

/// Result returned by [`solve_poisson_3d`].
#[derive(Debug, Clone)]
pub struct Poisson3dResult {
    /// Solution array of length `nx*ny*nz`, indexed as `i + nx*(j + ny*k)`.
    pub u: Vec<f64>,
    /// Number of SOR sweeps actually performed (≤ `max_iter`).
    pub iters: usize,
    /// Final L2 residual over the interior nodes.
    pub residual: f64,
}

impl Poisson3dResult {
    /// Number of SOR sweeps actually performed.
    #[must_use]
    pub fn iters(&self) -> usize {
        self.iters
    }

    /// Final L2 residual over the interior.
    #[must_use]
    pub fn residual(&self) -> f64 {
        self.residual
    }
}

#[inline]
fn idx3(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + nx * (j + ny * k)
}

fn validate_config(cfg: &Poisson3dConfig) -> PdeResult<()> {
    if cfg.nx < 3 || cfg.ny < 3 || cfg.nz < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "poisson 3d needs nx,ny,nz >= 3, got ({},{},{})",
            cfg.nx, cfg.ny, cfg.nz
        )));
    }
    if !(cfg.h.is_finite() && cfg.h > 0.0) {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: format!("must be a positive finite number, got {}", cfg.h),
        });
    }
    if !(cfg.omega.is_finite() && cfg.omega > 0.0 && cfg.omega < 2.0) {
        return Err(PdeError::InvalidParameter {
            name: "omega".into(),
            reason: format!("must lie in (0, 2), got {}", cfg.omega),
        });
    }
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "tol".into(),
            reason: format!("must be a non-negative finite number, got {}", cfg.tol),
        });
    }
    if cfg.max_iter == 0 {
        return Err(PdeError::InvalidParameter {
            name: "max_iter".into(),
            reason: "must be >= 1".into(),
        });
    }
    Ok(())
}

/// Solve `-Δu = f` on a uniform 3D box with Dirichlet boundary conditions.
///
/// On entry, `u_initial` must have length `nx*ny*nz` and must already contain
/// the prescribed boundary values on all six faces of the box; interior
/// entries are taken as the initial guess (commonly zero). The solver updates
/// only interior nodes and leaves the six boundary planes untouched.
///
/// `rhs[i + nx*(j + ny*k)]` is the source term at grid point `(i,j,k)`. Only
/// interior values of `rhs` are consulted by the solver; boundary entries may
/// be set to any finite value (typically zero).
///
/// # Errors
///
/// Returns [`PdeError::InvalidGrid`] / [`PdeError::InvalidParameter`] for
/// malformed configurations, and [`PdeError::ShapeMismatch`] if the length of
/// either input does not match `nx*ny*nz`.
pub fn solve_poisson_3d(
    rhs: &[f64],
    u_initial: &mut [f64],
    cfg: &Poisson3dConfig,
) -> PdeResult<Poisson3dResult> {
    validate_config(cfg)?;
    let n = cfg.nx * cfg.ny * cfg.nz;
    if rhs.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![rhs.len()],
        });
    }
    if u_initial.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u_initial.len()],
        });
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err(PdeError::NumericalInstability(
            "rhs contains non-finite values".into(),
        ));
    }
    if u_initial.iter().any(|v| !v.is_finite()) {
        return Err(PdeError::NumericalInstability(
            "u_initial contains non-finite values".into(),
        ));
    }

    let nx = cfg.nx;
    let ny = cfg.ny;
    let nz = cfg.nz;
    let h2 = cfg.h * cfg.h;
    // Stencil: 6 u_c − Σ neighbours = h² f  ⇒  u_c = (Σ neighbours + h² f) / 6
    let inv_diag = 1.0 / 6.0;
    let omega = cfg.omega;

    let mut u = u_initial.to_vec();
    let mut iters_done = 0usize;
    let mut last_res = f64::INFINITY;

    for it in 0..cfg.max_iter {
        // Two-colour red-black sweep covering all interior nodes.
        for colour in 0..2usize {
            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        if (i + j + k) & 1 != colour {
                            continue;
                        }
                        let c = idx3(i, j, k, nx, ny);
                        let sum_nbr = u[c - 1]
                            + u[c + 1]
                            + u[c - nx]
                            + u[c + nx]
                            + u[c - nx * ny]
                            + u[c + nx * ny];
                        let u_gs = inv_diag * (sum_nbr + h2 * rhs[c]);
                        u[c] = (1.0 - omega) * u[c] + omega * u_gs;
                    }
                }
            }
        }

        // L2 residual over interior nodes:
        //   r(i,j,k) = f − (6 u_c − Σ nbrs) / h²
        let mut acc = 0.0f64;
        let mut cnt = 0usize;
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let c = idx3(i, j, k, nx, ny);
                    let sum_nbr = u[c - 1]
                        + u[c + 1]
                        + u[c - nx]
                        + u[c + nx]
                        + u[c - nx * ny]
                        + u[c + nx * ny];
                    let lap = (6.0 * u[c] - sum_nbr) / h2;
                    let r = rhs[c] - lap;
                    acc += r * r;
                    cnt += 1;
                }
            }
        }
        let res = if cnt == 0 {
            0.0
        } else {
            (acc / cnt as f64).sqrt()
        };
        last_res = res;
        iters_done = it + 1;
        if res < cfg.tol {
            break;
        }
    }

    // Boundary planes are guaranteed unchanged because the sweep loops skip
    // them; reassert by copying the originals to be defensive against any
    // future stencil edits.
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let on_boundary =
                    i == 0 || i + 1 == nx || j == 0 || j + 1 == ny || k == 0 || k + 1 == nz;
                if on_boundary {
                    let c = idx3(i, j, k, nx, ny);
                    u[c] = u_initial[c];
                }
            }
        }
    }

    Ok(Poisson3dResult {
        u,
        iters: iters_done,
        residual: last_res,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_for(nx: usize, ny: usize, nz: usize) -> Poisson3dConfig {
        Poisson3dConfig {
            nx,
            ny,
            nz,
            h: 1.0 / (nx - 1) as f64,
            omega: 1.5,
            max_iter: 20_000,
            tol: 1.0e-9,
        }
    }

    fn is_boundary(i: usize, j: usize, k: usize, cfg: &Poisson3dConfig) -> bool {
        i == 0 || i + 1 == cfg.nx || j == 0 || j + 1 == cfg.ny || k == 0 || k + 1 == cfg.nz
    }

    fn stamp_boundary<F: Fn(usize, usize, usize) -> f64>(
        u: &mut [f64],
        cfg: &Poisson3dConfig,
        f: F,
    ) {
        for k in 0..cfg.nz {
            for j in 0..cfg.ny {
                for i in 0..cfg.nx {
                    if is_boundary(i, j, k, cfg) {
                        u[idx3(i, j, k, cfg.nx, cfg.ny)] = f(i, j, k);
                    }
                }
            }
        }
    }

    #[test]
    fn default_config_is_consistent() {
        let cfg = Poisson3dConfig::default();
        assert!(cfg.nx >= 3 && cfg.ny >= 3 && cfg.nz >= 3);
        assert!(cfg.h > 0.0 && cfg.h.is_finite());
        assert!(cfg.omega > 0.0 && cfg.omega < 2.0);
        assert!(cfg.tol >= 0.0);
        assert!(cfg.max_iter >= 1);
    }

    #[test]
    fn zero_rhs_zero_boundary_gives_zero_solution() {
        let cfg = cfg_for(5, 5, 5);
        let rhs = vec![0.0; cfg.nx * cfg.ny * cfg.nz];
        let mut u_init = vec![0.0; rhs.len()];
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        for &v in &res.u {
            assert!(v.abs() < 1.0e-12, "expected zero, got {}", v);
        }
        assert!(res.iters() >= 1);
        assert!(res.residual() < cfg.tol);
    }

    #[test]
    fn constant_rhs_with_zero_boundary_is_finite_and_centred() {
        let cfg = cfg_for(9, 9, 9);
        let rhs = vec![1.0; cfg.nx * cfg.ny * cfg.nz];
        let mut u_init = vec![0.0; rhs.len()];
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        let mid = idx3(cfg.nx / 2, cfg.ny / 2, cfg.nz / 2, cfg.nx, cfg.ny);
        let centre = res.u[mid];
        // Solution must be strictly positive and finite at the centre.
        assert!(centre.is_finite());
        assert!(centre > 0.0, "centre {} not positive", centre);
        // And the maximum should be attained at (or near) the centre for a
        // symmetric problem.
        let max_val = res.u.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!((max_val - centre).abs() < 0.05 * max_val + 1.0e-6);
    }

    #[test]
    fn linear_dirichlet_x_is_recovered() {
        // u = x has Δu = 0; with rhs = 0 and Dirichlet g = x, the discrete
        // solution should reproduce u(i,j,k) = x_i exactly.
        let cfg = cfg_for(7, 7, 7);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let mut u_init = vec![0.0; n];
        let rhs = vec![0.0; n];
        stamp_boundary(&mut u_init, &cfg, |i, _, _| i as f64 * cfg.h);
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        for k in 0..cfg.nz {
            for j in 0..cfg.ny {
                for i in 0..cfg.nx {
                    let want = i as f64 * cfg.h;
                    let got = res.u[idx3(i, j, k, cfg.nx, cfg.ny)];
                    assert!(
                        (got - want).abs() < 1.0e-6,
                        "i={i} j={j} k={k} got {got} want {want}",
                    );
                }
            }
        }
        assert!(res.residual() < 1.0e-8);
    }

    #[test]
    fn quadratic_manufactured_solution() {
        // Exact u(x,y,z) = x² + y² + z², so Δu = 6 and  −Δu = −6.
        // The 7-point stencil is *exact* for quadratics → residual is tiny.
        let cfg = cfg_for(9, 9, 9);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![-6.0; n];
        let mut u_init = vec![0.0; n];
        let exact = |i: usize, j: usize, k: usize| {
            let x = i as f64 * cfg.h;
            let y = j as f64 * cfg.h;
            let z = k as f64 * cfg.h;
            x * x + y * y + z * z
        };
        stamp_boundary(&mut u_init, &cfg, exact);
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        let mut max_err = 0.0f64;
        for k in 0..cfg.nz {
            for j in 0..cfg.ny {
                for i in 0..cfg.nx {
                    let want = exact(i, j, k);
                    let got = res.u[idx3(i, j, k, cfg.nx, cfg.ny)];
                    max_err = max_err.max((got - want).abs());
                }
            }
        }
        assert!(max_err < 1.0e-4, "max error {max_err} too large");
    }

    #[test]
    fn small_box_converges_within_max_iter() {
        let cfg = cfg_for(8, 8, 8);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![1.0; n];
        let mut u_init = vec![0.0; n];
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        assert!(res.iters() < cfg.max_iter, "iters={}", res.iters());
        assert!(res.residual() < cfg.tol);
    }

    #[test]
    fn rejects_zero_nx() {
        let mut cfg = cfg_for(5, 5, 5);
        cfg.nx = 0;
        let n = cfg.ny * cfg.nz; // anything that makes the mismatch detect later — but the validate fires first
        let rhs = vec![0.0; n];
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
    }

    #[test]
    fn rejects_omega_too_large() {
        let mut cfg = cfg_for(5, 5, 5);
        cfg.omega = 2.5;
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![0.0; n];
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rejects_omega_non_positive() {
        let mut cfg = cfg_for(5, 5, 5);
        cfg.omega = 0.0;
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![0.0; n];
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rejects_negative_h() {
        let mut cfg = cfg_for(5, 5, 5);
        cfg.h = -0.1;
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![0.0; n];
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rejects_wrong_rhs_length() {
        let cfg = cfg_for(5, 5, 5);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![0.0; n - 1];
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn boundary_is_preserved() {
        let cfg = cfg_for(6, 6, 6);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![2.0; n];
        let mut u_init = vec![0.0; n];
        stamp_boundary(&mut u_init, &cfg, |i, j, k| {
            (i + 10 * j + 100 * k) as f64 * 0.001
        });
        let snapshot = u_init.clone();
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        for k in 0..cfg.nz {
            for j in 0..cfg.ny {
                for i in 0..cfg.nx {
                    if is_boundary(i, j, k, &cfg) {
                        let c = idx3(i, j, k, cfg.nx, cfg.ny);
                        assert_eq!(res.u[c], snapshot[c]);
                    }
                }
            }
        }
    }

    #[test]
    fn pure_gauss_seidel_converges() {
        // ω = 1 ⇒ plain Gauss-Seidel — slower but must still converge.
        let mut cfg = cfg_for(7, 7, 7);
        cfg.omega = 1.0;
        cfg.max_iter = 30_000;
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![1.0; n];
        let mut u_init = vec![0.0; n];
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        assert!(res.residual() < cfg.tol);
        assert!(res.iters() < cfg.max_iter);
    }

    #[test]
    fn symmetric_input_yields_symmetric_solution() {
        // rhs and BCs invariant under (i,j,k) ↔ (nx-1-i, ny-1-j, nz-1-k):
        // the discrete solution must respect the same symmetry.
        let cfg = cfg_for(7, 7, 7);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![3.0; n];
        let mut u_init = vec![0.0; n];
        let res = solve_poisson_3d(&rhs, &mut u_init, &cfg).expect("solve ok");
        for k in 0..cfg.nz {
            for j in 0..cfg.ny {
                for i in 0..cfg.nx {
                    let a = res.u[idx3(i, j, k, cfg.nx, cfg.ny)];
                    let b = res.u[idx3(
                        cfg.nx - 1 - i,
                        cfg.ny - 1 - j,
                        cfg.nz - 1 - k,
                        cfg.nx,
                        cfg.ny,
                    )];
                    assert!(
                        (a - b).abs() < 1.0e-9,
                        "asymmetry at ({i},{j},{k}): {a} vs {b}",
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_zero_max_iter() {
        let mut cfg = cfg_for(5, 5, 5);
        cfg.max_iter = 0;
        let n = cfg.nx * cfg.ny * cfg.nz;
        let rhs = vec![0.0; n];
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_rhs() {
        let cfg = cfg_for(5, 5, 5);
        let n = cfg.nx * cfg.ny * cfg.nz;
        let mut rhs = vec![0.0; n];
        rhs[idx3(2, 2, 2, cfg.nx, cfg.ny)] = f64::NAN;
        let mut u = vec![0.0; n];
        assert!(matches!(
            solve_poisson_3d(&rhs, &mut u, &cfg),
            Err(PdeError::NumericalInstability(_))
        ));
    }
}
