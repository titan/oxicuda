//! Saddle-point (symmetric indefinite) system solvers.
//!
//! Many mixed finite-element discretisations (Stokes flow, mixed Poisson,
//! constrained optimisation) lead to a 2 × 2 block system
//!
//! ```text
//! [ A   Bᵀ ] [ u ]   [ f ]
//! [ B   0  ] [ p ] = [ g ]
//! ```
//!
//! with `A` symmetric positive definite on the kernel of `B` and `B` the
//! (rectangular) constraint operator. This module provides two classical
//! solvers, both implemented from scratch on the crate's
//! [`SparseCsr`] type with no external
//! linear-algebra dependency:
//!
//! * [`uzawa`] — the (preconditioned) Uzawa iteration, an outer fixed-point
//!   iteration on the pressure Schur complement `S = B A⁻¹ Bᵀ` that only ever
//!   solves with the `A` block. Robust and simple; one `A`-solve per outer
//!   step.
//! * [`minres`] — the Paige–Saunders MINRES method applied to the *full*
//!   augmented system viewed as one symmetric indefinite operator. It
//!   minimises the residual two-norm over a Krylov subspace using the
//!   symmetric Lanczos process and short three-term recurrences, so it needs
//!   only a handful of vectors of storage regardless of iteration count.
//!
//! # References
//!
//! * H. C. Elman, D. J. Silvester, A. J. Wathen, *Finite Elements and Fast
//!   Iterative Solvers*, 2nd ed., Oxford University Press, 2014, ch. 3–4.
//! * C. C. Paige and M. A. Saunders, "Solution of sparse indefinite systems
//!   of linear equations", SIAM J. Numer. Anal. 12(4), 617–629, 1975.
//! * Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM,
//!   2003, §6.7 (MINRES) and §8.4 (Uzawa).

use crate::error::{PdeError, PdeResult};
use crate::solver::cg::cg_solve;
use crate::solver::sparse::{SparseCsr, dot, norm2};

/// Configuration for the Uzawa outer iteration.
#[derive(Debug, Clone)]
pub struct UzawaConfig {
    /// Outer relaxation parameter `ω` on the pressure update
    /// `p ← p + ω (B u − g)`. Convergence requires `0 < ω < 2 / λ_max(S)`.
    pub omega: f64,
    /// Maximum number of outer Uzawa iterations.
    pub max_iter: usize,
    /// Relative tolerance on the combined residual two-norm.
    pub tol: f64,
    /// Maximum inner CG iterations for each `A`-solve.
    pub inner_max_iter: usize,
    /// Relative tolerance for the inner `A`-solve.
    pub inner_tol: f64,
}

impl Default for UzawaConfig {
    fn default() -> Self {
        Self {
            omega: 1.0,
            max_iter: 500,
            tol: 1.0e-9,
            inner_max_iter: 500,
            inner_tol: 1.0e-12,
        }
    }
}

/// Outcome of a saddle-point solve.
#[derive(Debug, Clone)]
pub struct SaddleResult {
    /// Primal (velocity / flux) block.
    pub u: Vec<f64>,
    /// Dual (pressure / multiplier) block.
    pub p: Vec<f64>,
    /// Number of outer iterations performed.
    pub iterations: usize,
    /// Final combined residual two-norm
    /// `‖(f − A u − Bᵀ p, g − B u)‖₂`.
    pub residual: f64,
    /// Whether the relative tolerance was reached.
    pub converged: bool,
}

/// Transpose-times-vector for a CSR matrix: returns `Bᵀ y`.
///
/// `B` is `m × n`; `y` has length `m`; the result has length `n`.
fn matvec_transpose(b: &SparseCsr, y: &[f64]) -> PdeResult<Vec<f64>> {
    if y.len() != b.n_rows {
        return Err(PdeError::DimensionMismatch {
            a: y.len(),
            b: b.n_rows,
        });
    }
    let mut out = vec![0.0_f64; b.n_cols];
    for (i, &yi) in y.iter().enumerate() {
        let lo = b.row_ptr[i];
        let hi = b.row_ptr[i + 1];
        for k in lo..hi {
            out[b.cols[k]] += b.vals[k] * yi;
        }
    }
    Ok(out)
}

/// Validate that the `(A, B)` block shapes are mutually consistent.
fn validate_blocks(
    a: &SparseCsr,
    b_block: &SparseCsr,
    f: &[f64],
    g: &[f64],
) -> PdeResult<(usize, usize)> {
    if a.n_rows != a.n_cols {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    let n = a.n_rows;
    let m = b_block.n_rows;
    if b_block.n_cols != n {
        return Err(PdeError::DimensionMismatch {
            a: b_block.n_cols,
            b: n,
        });
    }
    if f.len() != n {
        return Err(PdeError::DimensionMismatch { a: f.len(), b: n });
    }
    if g.len() != m {
        return Err(PdeError::DimensionMismatch { a: g.len(), b: m });
    }
    Ok((n, m))
}

/// Combined residual two-norm of a saddle-point state.
///
/// `r₁ = f − A u − Bᵀ p`, `r₂ = g − B u`.
fn combined_residual(
    a: &SparseCsr,
    b_block: &SparseCsr,
    f: &[f64],
    g: &[f64],
    u: &[f64],
    p: &[f64],
) -> PdeResult<f64> {
    let au = a.matvec(u)?;
    let btp = matvec_transpose(b_block, p)?;
    let bu = b_block.matvec(u)?;
    let mut acc = 0.0_f64;
    for i in 0..f.len() {
        let ri = f[i] - au[i] - btp[i];
        acc += ri * ri;
    }
    for i in 0..g.len() {
        let ri = g[i] - bu[i];
        acc += ri * ri;
    }
    Ok(acc.sqrt())
}

/// Solve the saddle-point system with the (inexact) Uzawa iteration.
///
/// The iteration is, with `p⁰` given (use zeros if unsure):
///
/// ```text
/// solve  A u^{k+1} = f − Bᵀ p^k        (inner CG)
/// update p^{k+1} = p^k + ω (B u^{k+1} − g)
/// ```
///
/// which is exactly a Richardson iteration on the pressure Schur complement
/// `S p = B A⁻¹ f − g`. It converges for `0 < ω < 2 / λ_max(S)`.
///
/// # Errors
///
/// * [`PdeError::DimensionMismatch`] when the block shapes or right-hand sides
///   are inconsistent.
/// * [`PdeError::InvalidParameter`] when `omega` is not positive or `tol`
///   is not positive.
/// * Inner-solve failures from [`cg_solve`].
pub fn uzawa(
    a: &SparseCsr,
    b_block: &SparseCsr,
    f: &[f64],
    g: &[f64],
    p0: &[f64],
    cfg: &UzawaConfig,
) -> PdeResult<SaddleResult> {
    let (n, m) = validate_blocks(a, b_block, f, g)?;
    if p0.len() != m {
        return Err(PdeError::DimensionMismatch { a: p0.len(), b: m });
    }
    if !(cfg.omega > 0.0 && cfg.omega.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "omega".into(),
            reason: "must be positive and finite".into(),
        });
    }
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "tol".into(),
            reason: "must be positive and finite".into(),
        });
    }
    let mut p = p0.to_vec();
    let mut u = vec![0.0_f64; n];
    let rhs_scale = (norm2(f) + norm2(g)).max(1.0);
    let mut residual = combined_residual(a, b_block, f, g, &u, &p)?;
    let mut converged = false;
    let mut iterations = 0_usize;
    for it in 0..cfg.max_iter {
        // rhs = f - Bᵀ p
        let btp = matvec_transpose(b_block, &p)?;
        let rhs: Vec<f64> = f.iter().zip(&btp).map(|(fi, bi)| fi - bi).collect();
        u = cg_solve(a, &rhs, &u, cfg.inner_max_iter, cfg.inner_tol)?;
        // p ← p + ω (B u − g)
        let bu = b_block.matvec(&u)?;
        for j in 0..m {
            p[j] += cfg.omega * (bu[j] - g[j]);
        }
        iterations = it + 1;
        residual = combined_residual(a, b_block, f, g, &u, &p)?;
        if residual / rhs_scale < cfg.tol {
            converged = true;
            break;
        }
    }
    Ok(SaddleResult {
        u,
        p,
        iterations,
        residual,
        converged,
    })
}

/// Configuration for MINRES on the augmented system.
#[derive(Debug, Clone, Copy)]
pub struct MinresConfig {
    /// Maximum number of MINRES iterations.
    pub max_iter: usize,
    /// Relative residual tolerance `‖r‖ / ‖rhs‖`.
    pub tol: f64,
}

impl Default for MinresConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1.0e-10,
        }
    }
}

/// Apply the augmented saddle-point operator to a stacked vector `[u; p]`.
///
/// Returns `[A u + Bᵀ p ; B u]` (the lower-right block is zero).
fn apply_augmented(a: &SparseCsr, b_block: &SparseCsr, x: &[f64]) -> PdeResult<Vec<f64>> {
    let n = a.n_rows;
    let m = b_block.n_rows;
    if x.len() != n + m {
        return Err(PdeError::DimensionMismatch {
            a: x.len(),
            b: n + m,
        });
    }
    let u = &x[..n];
    let p = &x[n..];
    let au = a.matvec(u)?;
    let btp = matvec_transpose(b_block, p)?;
    let bu = b_block.matvec(u)?;
    let mut out = vec![0.0_f64; n + m];
    for i in 0..n {
        out[i] = au[i] + btp[i];
    }
    out[n..(m + n)].copy_from_slice(&bu[..m]);
    Ok(out)
}

/// Solve the saddle-point system with MINRES on the symmetric indefinite
/// augmented operator
///
/// ```text
/// K = [ A   Bᵀ ]      rhs = [ f ]
///     [ B   0  ]            [ g ]
/// ```
///
/// MINRES (Paige–Saunders 1975) builds an orthonormal Krylov basis with the
/// symmetric Lanczos process and minimises `‖rhs − K x‖₂` using short
/// recurrences (constant storage). `K` need not be definite — only symmetric
/// — which is exactly the saddle-point case.
///
/// # Errors
///
/// * [`PdeError::DimensionMismatch`] when the block shapes are inconsistent.
/// * [`PdeError::InvalidParameter`] when `tol` is not positive.
/// * [`PdeError::NotConverged`] when the iteration limit is hit before the
///   tolerance is reached.
pub fn minres(
    a: &SparseCsr,
    b_block: &SparseCsr,
    f: &[f64],
    g: &[f64],
    cfg: &MinresConfig,
) -> PdeResult<SaddleResult> {
    let (n, m) = validate_blocks(a, b_block, f, g)?;
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "tol".into(),
            reason: "must be positive and finite".into(),
        });
    }
    let dim = n + m;
    let mut rhs = vec![0.0_f64; dim];
    rhs[..n].copy_from_slice(f);
    rhs[n..].copy_from_slice(g);
    let rhs_norm = norm2(&rhs).max(1.0);

    // Solution and the trivial-RHS short-circuit.
    let mut x = vec![0.0_f64; dim];
    let beta1 = norm2(&rhs);
    if beta1 == 0.0 || beta1 / rhs_norm < cfg.tol {
        let (u, p) = split(&x, n);
        return Ok(SaddleResult {
            u,
            p,
            iterations: 0,
            residual: beta1,
            converged: true,
        });
    }

    // ----------------------------------------------------------------------
    // MINRES (Paige–Saunders 1975), Stanford SOL reference recurrences.
    // Operator `K = apply_augmented`. Initial guess x0 = 0 ⇒ r0 = rhs.
    // Lanczos basis carried in `r1` (= previous, scaled), `r2` (= current),
    // `y` (work) and `v` (unit Lanczos vector). Direction vectors `w1, w2, w`.
    // ----------------------------------------------------------------------
    let mut r1 = rhs.clone();
    let mut r2 = rhs.clone();
    let mut beta = beta1;
    let mut oldb = 0.0_f64;

    // Running QR / Givens scalars.
    let mut dbar = 0.0_f64;
    let mut epsln = 0.0_f64;
    let mut phibar = beta1;
    let mut cs = -1.0_f64;
    let mut sn = 0.0_f64;

    let mut w = vec![0.0_f64; dim];
    let mut w1 = vec![0.0_f64; dim];

    let mut iterations = 0_usize;
    let mut converged = false;
    for itn in 1..=cfg.max_iter {
        iterations = itn;

        // --- Lanczos step ---
        // v = (1/beta) y  with  y = K r2/beta_prev produced last round; here we
        // recompute v from r2 to keep it exact: v = r2 / beta.
        let inv_beta = 1.0 / beta;
        let v: Vec<f64> = r2.iter().map(|yi| yi * inv_beta).collect();
        let mut y = apply_augmented(a, b_block, &v)?;
        if itn >= 2 {
            let c = beta / oldb;
            for i in 0..dim {
                y[i] -= c * r1[i];
            }
        }
        let alpha = dot(&v, &y)?;
        // y = y - (alpha/beta) r2
        let c2 = alpha / beta;
        for i in 0..dim {
            y[i] -= c2 * r2[i];
        }
        // Shift Lanczos history: r1 <- r2, r2 <- y.
        r1 = std::mem::replace(&mut r2, y.clone());
        oldb = beta;
        beta = norm2(&r2);

        // --- Apply previous Givens rotation, then form a new one ---
        // `oldeps` is the ε produced in the *previous* iteration (0 on the
        // first); it multiplies the two-steps-back direction vector `w1`.
        let oldeps = epsln;
        let delta = cs * dbar + sn * alpha; // δ_k
        let gbar = sn * dbar - cs * alpha; // \bar{γ}_k (diagonal before new rot.)
        epsln = sn * beta; // ε_{k+1}
        dbar = -cs * beta; // \bar{δ}_{k+1}

        // New plane rotation to annihilate `beta`.
        let (cs_new, sn_new, gamma) = sym_ortho(gbar, beta);
        cs = cs_new;
        sn = sn_new;
        let phi = cs * phibar; // τ_k
        phibar *= sn; // \bar{φ}_{k+1} (residual norm estimate)

        // --- Update solution ---
        if gamma > 1.0e-300 {
            let inv_gamma = 1.0 / gamma;
            let mut w_new = vec![0.0_f64; dim];
            for i in 0..dim {
                w_new[i] = (v[i] - oldeps * w1[i] - delta * w[i]) * inv_gamma;
            }
            for i in 0..dim {
                x[i] += phi * w_new[i];
            }
            w1 = std::mem::replace(&mut w, w_new);
        }

        // `phibar` is the running estimate of the residual two-norm.
        if phibar / rhs_norm < cfg.tol {
            converged = true;
            break;
        }
        // Lucky breakdown: invariant subspace reached.
        if beta <= 1.0e-300 {
            converged = phibar / rhs_norm < cfg.tol;
            break;
        }
    }

    let (u, p) = split(&x, n);
    let true_res = combined_residual(a, b_block, f, g, &u, &p)?;
    if converged || true_res / rhs_norm < cfg.tol {
        Ok(SaddleResult {
            u,
            p,
            iterations,
            residual: true_res,
            converged: true,
        })
    } else {
        Err(PdeError::NotConverged {
            iter: cfg.max_iter,
            residual: true_res,
        })
    }
}

/// Stable symmetric orthogonalisation (`SymOrtho`, Choi 2006): returns the
/// Givens rotation `(c, s)` and the resulting norm `r` such that
/// `[c s; s −c] · [a; b] = [r; 0]`. Avoids overflow in `√(a² + b²)`.
fn sym_ortho(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        if a == 0.0 {
            (1.0, 0.0, 0.0)
        } else {
            (a.signum(), 0.0, a.abs())
        }
    } else if a == 0.0 {
        (0.0, b.signum(), b.abs())
    } else if b.abs() > a.abs() {
        let t = a / b;
        let s = b.signum() / (1.0 + t * t).sqrt();
        let c = s * t;
        (c, s, b / s)
    } else {
        let t = b / a;
        let c = a.signum() / (1.0 + t * t).sqrt();
        let s = c * t;
        (c, s, a / c)
    }
}

/// Split a stacked `[u; p]` vector into its two blocks.
fn split(x: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    (x[..n].to_vec(), x[n..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small symmetric-positive-definite block (1-D Laplacian-like).
    fn spd_block(n: usize) -> SparseCsr {
        let mut row_ptr = vec![0_usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            if i > 0 {
                cols.push(i - 1);
                vals.push(-1.0);
            }
            cols.push(i);
            vals.push(4.0);
            if i + 1 < n {
                cols.push(i + 1);
                vals.push(-1.0);
            }
            row_ptr.push(cols.len());
        }
        SparseCsr::new(n, n, row_ptr, cols, vals).expect("valid spd block")
    }

    /// A full-rank `m × n` constraint block with `m < n` (rows of a discrete
    /// divergence): `B[j] = e_j − e_{j+1}` so `B` has rank `m`.
    fn constraint_block(m: usize, n: usize) -> SparseCsr {
        assert!(m < n);
        let mut row_ptr = vec![0_usize];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..m {
            cols.push(j);
            vals.push(1.0);
            cols.push(j + 1);
            vals.push(-1.0);
            row_ptr.push(cols.len());
        }
        SparseCsr::new(m, n, row_ptr, cols, vals).expect("valid constraint block")
    }

    /// Dense Gaussian elimination on the explicit augmented matrix — reference.
    fn dense_saddle_solve(a: &SparseCsr, b: &SparseCsr, f: &[f64], g: &[f64]) -> Vec<f64> {
        let n = a.n_rows;
        let m = b.n_rows;
        let dim = n + m;
        let mut k = vec![0.0_f64; dim * dim];
        // A block.
        for i in 0..n {
            for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
                k[i * dim + a.cols[idx]] += a.vals[idx];
            }
        }
        // B and Bᵀ blocks.
        for j in 0..m {
            for idx in b.row_ptr[j]..b.row_ptr[j + 1] {
                let col = b.cols[idx];
                let v = b.vals[idx];
                // B in lower-left: row (n+j), col (col).
                k[(n + j) * dim + col] += v;
                // Bᵀ in upper-right: row (col), col (n+j).
                k[col * dim + (n + j)] += v;
            }
        }
        let mut rhs = vec![0.0_f64; dim];
        rhs[..n].copy_from_slice(f);
        rhs[n..].copy_from_slice(g);
        gaussian_solve(&mut k, &mut rhs, dim)
    }

    fn gaussian_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
        for col in 0..n {
            let mut piv = col;
            let mut best = a[col * n + col].abs();
            for row in (col + 1)..n {
                let v = a[row * n + col].abs();
                if v > best {
                    best = v;
                    piv = row;
                }
            }
            if piv != col {
                for c in 0..n {
                    a.swap(col * n + c, piv * n + c);
                }
                b.swap(col, piv);
            }
            let diag = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / diag;
                if factor == 0.0 {
                    continue;
                }
                for c in col..n {
                    a[row * n + c] -= factor * a[col * n + c];
                }
                b[row] -= factor * b[col];
            }
        }
        let mut x = vec![0.0_f64; n];
        for col in (0..n).rev() {
            let mut acc = b[col];
            for c in (col + 1)..n {
                acc -= a[col * n + c] * x[c];
            }
            x[col] = acc / a[col * n + col];
        }
        x
    }

    #[test]
    fn transpose_matvec_correct() {
        // B = [[1, -1, 0], [0, 1, -1]]; Bᵀ y for y = [2, 3].
        let b = constraint_block(2, 3);
        let y = vec![2.0, 3.0];
        let bt = matvec_transpose(&b, &y).expect("ok");
        // Bᵀ = [[1,0],[-1,1],[0,-1]]; Bᵀ y = [2, -2+3, -3] = [2, 1, -3].
        assert!((bt[0] - 2.0).abs() < 1e-12);
        assert!((bt[1] - 1.0).abs() < 1e-12);
        assert!((bt[2] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn minres_matches_dense_reference() {
        let n = 6;
        let m = 3;
        let a = spd_block(n);
        let b = constraint_block(m, n);
        let f: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let g: Vec<f64> = (0..m).map(|j| (j as f64) * 0.25 - 0.1).collect();
        let cfg = MinresConfig {
            max_iter: 500,
            tol: 1.0e-11,
        };
        let res = minres(&a, &b, &f, &g, &cfg).expect("minres ok");
        assert!(res.converged);
        let exact = dense_saddle_solve(&a, &b, &f, &g);
        for (i, (got, want)) in res.u.iter().zip(&exact[..n]).enumerate() {
            assert!((got - want).abs() < 1e-6, "u[{i}] {got} vs {want}");
        }
        for (j, (got, want)) in res.p.iter().zip(&exact[n..]).enumerate() {
            assert!((got - want).abs() < 1e-6, "p[{j}] {got} vs {want}");
        }
    }

    #[test]
    fn minres_residual_actually_small() {
        let n = 8;
        let m = 4;
        let a = spd_block(n);
        let b = constraint_block(m, n);
        let f = vec![1.0_f64; n];
        let g = vec![0.0_f64; m];
        let res = minres(&a, &b, &f, &g, &MinresConfig::default()).expect("ok");
        assert!(res.converged);
        let real = combined_residual(&a, &b, &f, &g, &res.u, &res.p).expect("ok");
        assert!(real < 1e-7, "true residual {real}");
    }

    #[test]
    fn uzawa_matches_dense_reference() {
        let n = 6;
        let m = 3;
        let a = spd_block(n);
        let b = constraint_block(m, n);
        let f: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let g: Vec<f64> = (0..m).map(|j| (j as f64) * 0.25 - 0.1).collect();
        // S = B A⁻¹ Bᵀ; A ≈ 4I so λ_max(S) ≲ (1/4)·λ_max(B Bᵀ) ≲ (1/4)·4 = 1,
        // hence ω = 1 is comfortably inside (0, 2/λ_max).
        let cfg = UzawaConfig {
            omega: 1.0,
            max_iter: 2000,
            tol: 1.0e-9,
            ..UzawaConfig::default()
        };
        let res = uzawa(&a, &b, &f, &g, &vec![0.0; m], &cfg).expect("uzawa ok");
        assert!(res.converged, "residual {}", res.residual);
        let exact = dense_saddle_solve(&a, &b, &f, &g);
        for (i, (got, want)) in res.u.iter().zip(&exact[..n]).enumerate() {
            assert!((got - want).abs() < 1e-5, "u[{i}]");
        }
        for (j, (got, want)) in res.p.iter().zip(&exact[n..]).enumerate() {
            assert!((got - want).abs() < 1e-5, "p[{j}]");
        }
    }

    #[test]
    fn uzawa_and_minres_agree() {
        let n = 7;
        let m = 3;
        let a = spd_block(n);
        let b = constraint_block(m, n);
        let f: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.7).sin()).collect();
        let g: Vec<f64> = (0..m).map(|j| ((j as f64) * 1.1).cos() * 0.3).collect();
        let mr = minres(&a, &b, &f, &g, &MinresConfig::default()).expect("minres");
        let uz = uzawa(
            &a,
            &b,
            &f,
            &g,
            &vec![0.0; m],
            &UzawaConfig {
                omega: 1.0,
                max_iter: 3000,
                tol: 1.0e-9,
                ..UzawaConfig::default()
            },
        )
        .expect("uzawa");
        for (i, (a_i, b_i)) in mr.u.iter().zip(&uz.u).enumerate() {
            assert!((a_i - b_i).abs() < 1e-4, "u[{i}] disagree");
        }
        for (j, (a_j, b_j)) in mr.p.iter().zip(&uz.p).enumerate() {
            assert!((a_j - b_j).abs() < 1e-4, "p[{j}] disagree");
        }
    }

    #[test]
    fn rejects_inconsistent_shapes() {
        let a = spd_block(4);
        let b = constraint_block(2, 4);
        // f wrong length.
        assert!(matches!(
            minres(&a, &b, &[1.0, 2.0], &[0.0, 0.0], &MinresConfig::default()),
            Err(PdeError::DimensionMismatch { .. })
        ));
        // g wrong length.
        assert!(matches!(
            uzawa(
                &a,
                &b,
                &[1.0, 2.0, 3.0, 4.0],
                &[0.0, 0.0, 0.0],
                &[0.0, 0.0],
                &UzawaConfig::default()
            ),
            Err(PdeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_bad_parameters() {
        let a = spd_block(4);
        let b = constraint_block(2, 4);
        let f = vec![1.0; 4];
        let g = vec![0.0; 2];
        let bad_uzawa = UzawaConfig {
            omega: -1.0,
            ..UzawaConfig::default()
        };
        assert!(matches!(
            uzawa(&a, &b, &f, &g, &[0.0; 2], &bad_uzawa),
            Err(PdeError::InvalidParameter { .. })
        ));
        let bad_minres = MinresConfig {
            tol: 0.0,
            ..MinresConfig::default()
        };
        assert!(matches!(
            minres(&a, &b, &f, &g, &bad_minres),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn minres_solution_is_finite() {
        let n = 5;
        let m = 2;
        let a = spd_block(n);
        let b = constraint_block(m, n);
        let f = vec![0.3_f64; n];
        let g = vec![-0.2_f64; m];
        let res = minres(&a, &b, &f, &g, &MinresConfig::default()).expect("ok");
        assert!(res.u.iter().all(|v| v.is_finite()));
        assert!(res.p.iter().all(|v| v.is_finite()));
    }
}
