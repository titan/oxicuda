//! Mehrotra predictor-corrector interior-point method for second-order cone
//! programs (SOCP) with Nesterov-Todd (NT) scaling.
//!
//! Solves the primal-dual pair
//!
//! ```text
//!   (P)  min  cᵀx        (D)  max  bᵀy
//!        s.t. A x = b,        s.t. Aᵀy + s = c,
//!             x ∈ K,               s ∈ K,
//! ```
//!
//! where `K = K₁ × … × K_p` is a product of second-order (Lorentz) cones
//! `K_j = {(u₀, ū) ∈ ℝ × ℝ^{n_j−1} : ‖ū‖₂ ≤ u₀}`.  The KKT / central-path
//! conditions are `A x = b`, `Aᵀy + s = c`, `x ∘ s = μ e`, `x, s ∈ K`, where `∘`
//! is the Jordan product of the second-order-cone Euclidean Jordan algebra and
//! `e` its identity.
//!
//! # Nesterov-Todd scaling
//!
//! Each iteration the per-cone NT scaling matrix `W = Q_{w^{1/2}}` is formed,
//! where `Q_u = 2 u uᵀ − det(u) J` is the quadratic representation
//! (`J = diag(1,−1,…,−1)`) and the scaling point `w = Q_{x^{1/2}}
//! (Q_{x^{1/2}} s)^{-1/2}` is the unique point with `Q_w x = s`.  The symmetric
//! positive-definite `W` equalises the primal and dual iterates,
//! `λ = W s = W⁻¹ x = (Q_{x^{1/2}} s)^{1/2}`, turning the non-symmetric
//! complementarity `x ∘ s = μ e` into the symmetric Newton system in the scaled
//! variables.  Reducing to the normal equations gives `(A W² Aᵀ) Δy = …`, solved
//! with the crate's dense LU.
//!
//! # Predictor-corrector (Mehrotra 1992)
//!
//! 1. *Predictor (affine, σ = 0):* the right-hand side `−λ ∘ λ` collapses to the
//!    closed form `d = −λ`; solve for the affine direction and its cone step
//!    lengths.
//! 2. *Centering:* `σ = (μ_aff / μ)³`, clamped to `[0, 1]`.
//! 3. *Corrector:* right-hand side `σμ e − λ ∘ λ − Δx̂_aff ∘ Δŝ_aff` reuses the
//!    same factorisation.
//! 4. *Cone step-length guard:* the fraction-to-boundary rule keeps each iterate
//!    strictly inside its cone.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{dot, mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// One Newton direction `(Δx, Δy, Δs, Δx̂, Δŝ)` — primal/dual/slack steps plus
/// their NT-scaled counterparts.
type SocpDirection = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Solver status for [`mehrotra_socp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocpStatus {
    /// Primal/dual residuals and the duality gap all fell below the tolerance.
    Optimal,
}

/// Configuration for [`mehrotra_socp`].
#[derive(Debug, Clone, Copy)]
pub struct MehrotraSocpConfig {
    /// Maximum predictor-corrector iterations.
    pub max_iter: usize,
    /// Convergence tolerance applied to `‖r_p‖`, `‖r_d‖`, and `μ`.
    pub tol: f64,
    /// Fraction-to-boundary parameter in `(0, 1)` (typically `0.99`).
    pub step_fraction: f64,
}

impl Default for MehrotraSocpConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-9,
            step_fraction: 0.99,
        }
    }
}

/// Result returned by [`mehrotra_socp`] on success.
#[derive(Debug, Clone)]
pub struct MehrotraSocpResult {
    /// Primal solution `x ∈ K` (length `n`).
    pub x: Vec<f64>,
    /// Equality-constraint multiplier `y` (length `m`).
    pub y: Vec<f64>,
    /// Dual slack `s ∈ K` (length `n`).
    pub s: Vec<f64>,
    /// Iterations performed.
    pub iter: usize,
    /// Final duality measure `μ = ⟨x, s⟩ / p` (`p` = number of cones).
    pub mu: f64,
    /// Final primal residual `‖A x − b‖`.
    pub primal_res: f64,
    /// Final dual residual `‖Aᵀy + s − c‖`.
    pub dual_res: f64,
    /// Convergence status.
    pub status: SocpStatus,
}

// ───────────────────────── Jordan-algebra helpers (single cone) ─────────────

/// `det(u) = u₀² − ‖ū‖²` for one second-order cone.
fn soc_det(u: &[f64]) -> f64 {
    let mut s = u[0] * u[0];
    for &ui in &u[1..] {
        s -= ui * ui;
    }
    s
}

/// Jordan product `a ∘ b = (aᵀb, a₀ b̄ + b₀ ā)` for one cone.
fn soc_jprod(a: &[f64], b: &[f64]) -> Vec<f64> {
    let nj = a.len();
    let mut out = vec![0.0_f64; nj];
    let mut s = 0.0_f64;
    for i in 0..nj {
        s += a[i] * b[i];
    }
    out[0] = s;
    for i in 1..nj {
        out[i] = a[0] * b[i] + b[0] * a[i];
    }
    out
}

/// Jordan square root of an interior cone point: the unique `r ∈ int K` with
/// `r ∘ r = u`.
fn soc_sqrt(u: &[f64]) -> CvxResult<Vec<f64>> {
    let d = soc_det(u);
    if !d.is_finite() || d <= 0.0 || u[0] <= 0.0 {
        return Err(CvxError::ConeViolation(format!(
            "soc_sqrt needs an interior point (det={d}, u0={})",
            u[0]
        )));
    }
    let a = ((u[0] + d.sqrt()) * 0.5).sqrt();
    if a <= 0.0 || a.is_nan() {
        return Err(CvxError::NumericalInstability("soc_sqrt zero scale".into()));
    }
    let mut r = vec![0.0_f64; u.len()];
    r[0] = a;
    let inv = 1.0 / (2.0 * a);
    for i in 1..u.len() {
        r[i] = u[i] * inv;
    }
    Ok(r)
}

/// Jordan inverse `u⁻¹ = (1/det u)·(u₀, −ū)` for one cone.
fn soc_inv(u: &[f64]) -> CvxResult<Vec<f64>> {
    let d = soc_det(u);
    if d.abs() < 1e-300 {
        return Err(CvxError::SingularMatrix("soc_inv on zero-det point".into()));
    }
    let mut r = vec![0.0_f64; u.len()];
    r[0] = u[0] / d;
    for i in 1..u.len() {
        r[i] = -u[i] / d;
    }
    Ok(r)
}

/// Quadratic representation `Q_u = 2 u uᵀ − det(u) J` (row-major `nj × nj`).
fn q_rep(u: &[f64]) -> Vec<f64> {
    let nj = u.len();
    let det = soc_det(u);
    let mut q = vec![0.0_f64; nj * nj];
    for i in 0..nj {
        for j in 0..nj {
            q[i * nj + j] = 2.0 * u[i] * u[j];
        }
    }
    q[0] -= det; // J[0,0] = +1
    for i in 1..nj {
        q[i * nj + i] += det; // J[i,i] = −1
    }
    q
}

/// Arrow / multiplication matrix `Arw(u) = [[u₀, ūᵀ], [ū, u₀ I]]`
/// (row-major `nj × nj`); satisfies `Arw(u) v = u ∘ v`.
fn arrow(u: &[f64]) -> Vec<f64> {
    let nj = u.len();
    let mut m = vec![0.0_f64; nj * nj];
    m[0] = u[0];
    for i in 1..nj {
        m[i] = u[i]; // first row
        m[i * nj] = u[i]; // first column
        m[i * nj + i] = u[0]; // diagonal
    }
    m
}

// ───────────────────────── block-diagonal scaling ──────────────────────────

/// Full block-diagonal NT scaling.  Returns `(W, W⁻¹, λ)` where `W`, `W⁻¹` are
/// dense `n × n` (zero off the cone blocks) and `λ = W s = W⁻¹ x`.
fn build_scaling(
    x: &[f64],
    s: &[f64],
    cone_dims: &[usize],
    n: usize,
) -> CvxResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut w_full = vec![0.0_f64; n * n];
    let mut w_inv_full = vec![0.0_f64; n * n];
    let mut lambda = vec![0.0_f64; n];
    let mut off = 0usize;
    for &nj in cone_dims {
        let xj = &x[off..off + nj];
        let sj = &s[off..off + nj];
        let xh = soc_sqrt(xj)?;
        let qxh = q_rep(&xh);
        let g = mat_vec(&qxh, nj, nj, sj)?; // Q_{x^{1/2}} s
        let lam_j = soc_sqrt(&g)?; // λ = (Q_{x^{1/2}} s)^{1/2}
        let lam_inv = soc_inv(&lam_j)?; // = (Q_{x^{1/2}} s)^{-1/2}
        let w_pt = mat_vec(&qxh, nj, nj, &lam_inv)?; // scaling point w
        let wh = soc_sqrt(&w_pt)?;
        let wj = q_rep(&wh); // W block = Q_{w^{1/2}}
        let wh_inv = soc_inv(&wh)?;
        let winv_j = q_rep(&wh_inv); // W⁻¹ block = Q_{w^{-1/2}}
        for a in 0..nj {
            lambda[off + a] = lam_j[a];
            for b in 0..nj {
                w_full[(off + a) * n + off + b] = wj[a * nj + b];
                w_inv_full[(off + a) * n + off + b] = winv_j[a * nj + b];
            }
        }
        off += nj;
    }
    Ok((w_full, w_inv_full, lambda))
}

/// Largest `α ≥ 0` with `u + α·du ∈ K` for every cone (capped at a large
/// sentinel when unconstrained).  Pure cone step length, before the
/// fraction-to-boundary scaling.
fn max_step_in_cone(u: &[f64], du: &[f64], cone_dims: &[usize]) -> f64 {
    let mut alpha = f64::INFINITY;
    let mut off = 0usize;
    for &nj in cone_dims {
        let uj = &u[off..off + nj];
        let dj = &du[off..off + nj];
        // f(α) = det(u + α du) = a α² + 2 b α + c.
        let a = soc_det(dj);
        // b = ⟨u, du⟩_J = u₀ du₀ − ū·dū.
        let mut b = uj[0] * dj[0];
        for i in 1..nj {
            b -= uj[i] * dj[i];
        }
        let c = soc_det(uj); // > 0 for an interior point
        let mut cone_alpha = f64::INFINITY;
        // Candidate: u₀ + α du₀ = 0 (only binds when du₀ < 0).
        if dj[0] < 0.0 {
            let lin = -uj[0] / dj[0];
            if lin > 0.0 && lin < cone_alpha {
                cone_alpha = lin;
            }
        }
        // Candidate: smallest positive root of the determinant quadratic.
        if a.abs() < 1e-15 {
            // Linear: 2 b α + c = 0.
            if b < 0.0 {
                let r = -c / (2.0 * b);
                if r > 0.0 && r < cone_alpha {
                    cone_alpha = r;
                }
            }
        } else {
            let disc = b * b - a * c;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for &root in &[(-b - sq) / a, (-b + sq) / a] {
                    if root > 0.0 && root < cone_alpha {
                        cone_alpha = root;
                    }
                }
            }
        }
        if cone_alpha < alpha {
            alpha = cone_alpha;
        }
        off += nj;
    }
    alpha
}

// ───────────────────────── main solver ─────────────────────────────────────

/// Mehrotra predictor-corrector primal-dual interior-point SOCP solver.
///
/// # Parameters
/// * `a`         – `m × n` constraint matrix, row-major.
/// * `m`         – number of equality constraints.
/// * `n`         – number of primal variables (must equal `Σ cone_dims`).
/// * `b`         – right-hand side (length `m`).
/// * `c`         – linear objective (length `n`).
/// * `cone_dims` – dimensions of the product second-order cones (each `≥ 1`).
/// * `config`    – iteration limits / tolerances.
///
/// # Errors
/// * [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] for bad sizes.
/// * [`CvxError::InvalidParameter`] for an empty/invalid cone description or
///   non-positive tolerance / out-of-range step fraction.
/// * [`CvxError::Unbounded`] when an iterate diverges (primal unbounded / dual
///   infeasible).
/// * [`CvxError::Infeasible`] when the gap closes but feasibility cannot be met.
/// * [`CvxError::NotConverged`] when `max_iter` is exhausted.
pub fn mehrotra_socp(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    c: &[f64],
    cone_dims: &[usize],
    config: &MehrotraSocpConfig,
) -> CvxResult<MehrotraSocpResult> {
    // ── validation ──────────────────────────────────────────────────────────
    if n == 0 || m == 0 {
        return Err(CvxError::InvalidParameter(
            "SOCP requires n ≥ 1 and m ≥ 1".into(),
        ));
    }
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    if c.len() != n {
        return Err(CvxError::DimensionMismatch { a: c.len(), b: n });
    }
    if cone_dims.is_empty() {
        return Err(CvxError::InvalidParameter("cone_dims is empty".into()));
    }
    let cone_sum: usize = cone_dims.iter().sum();
    if cone_sum != n {
        return Err(CvxError::DimensionMismatch { a: cone_sum, b: n });
    }
    if cone_dims.contains(&0) {
        return Err(CvxError::InvalidParameter(
            "every cone dimension must be ≥ 1".into(),
        ));
    }
    if config.tol <= 0.0 || !config.tol.is_finite() {
        return Err(CvxError::InvalidParameter("tol must be positive".into()));
    }
    if !(config.step_fraction > 0.0 && config.step_fraction < 1.0) {
        return Err(CvxError::InvalidParameter(
            "step_fraction must be in (0, 1)".into(),
        ));
    }

    let p_cones = cone_dims.len();
    let blow_up = 1e12_f64;

    // ── strictly interior start: x = s = e per cone, y = 0 ──────────────────
    let mut x = vec![0.0_f64; n];
    let mut s = vec![0.0_f64; n];
    {
        let mut off = 0usize;
        for &nj in cone_dims {
            x[off] = 1.0;
            s[off] = 1.0;
            off += nj;
        }
    }
    let mut y = vec![0.0_f64; m];

    let mut last_rp = f64::INFINITY;
    let mut last_rd = f64::INFINITY;
    let mut last_mu = f64::INFINITY;

    for it in 0..config.max_iter {
        // ── residuals & duality measure ─────────────────────────────────────
        let ax = mat_vec(a, m, n, &x)?;
        let r_p: Vec<f64> = (0..m).map(|i| b[i] - ax[i]).collect();
        let aty = mat_t_vec(a, m, n, &y)?;
        let r_d: Vec<f64> = (0..n).map(|j| c[j] - aty[j] - s[j]).collect();
        let mu = dot(&x, &s)? / p_cones as f64;
        let np = norm2(&r_p);
        let nd = norm2(&r_d);
        last_rp = np;
        last_rd = nd;
        last_mu = mu;

        if np < config.tol && nd < config.tol && mu < config.tol {
            return Ok(MehrotraSocpResult {
                x,
                y,
                s,
                iter: it,
                mu,
                primal_res: np,
                dual_res: nd,
                status: SocpStatus::Optimal,
            });
        }

        // Divergence ⇒ primal unbounded / dual infeasible.
        if norm2(&x) > blow_up || norm2(&s) > blow_up || norm2(&y) > blow_up {
            return Err(CvxError::Unbounded(
                "iterate diverged; problem appears primal-unbounded / dual-infeasible".into(),
            ));
        }
        // Gap closed but feasibility unattainable ⇒ infeasible.
        if it >= 12 && mu < config.tol.max(1e-9) * 1e3 && (np > 1e-4 || nd > 1e-4) {
            return Err(CvxError::Infeasible(
                "duality gap vanished without reaching feasibility".into(),
            ));
        }

        // ── NT scaling ──────────────────────────────────────────────────────
        let (w, w_inv, lambda) = build_scaling(&x, &s, cone_dims, n)?;

        // Pre-compute W·r_d, W²·r_d and the per-row W·aᵢ (for N = A W² Aᵀ).
        let w_rd = mat_vec(&w, n, n, &r_d)?;
        let w2_rd = mat_vec(&w, n, n, &w_rd)?;
        let a_w2_rd = mat_vec(a, m, n, &w2_rd)?;
        let mut wa_rows: Vec<Vec<f64>> = Vec::with_capacity(m);
        for i in 0..m {
            let row = &a[i * n..i * n + n];
            wa_rows.push(mat_vec(&w, n, n, row)?);
        }
        // Normal-equations matrix N = A W² Aᵀ (+ tiny regularisation).
        let mut n_mat = vec![0.0_f64; m * m];
        for i in 0..m {
            for k in i..m {
                let v = dot(&wa_rows[i], &wa_rows[k])?;
                n_mat[i * m + k] = v;
                n_mat[k * m + i] = v;
            }
            n_mat[i * m + i] += 1e-12;
        }

        // Solve one predictor/corrector direction for a given complementarity
        // right-hand side `d` (= Arw(λ)⁻¹ · rhs_c).  Returns
        // (Δx, Δy, Δs, Δx̂, Δŝ).
        let solve_dir = |d: &[f64]| -> CvxResult<SocpDirection> {
            let w_d = mat_vec(&w, n, n, d)?;
            let a_w_d = mat_vec(a, m, n, &w_d)?;
            let rhs_y: Vec<f64> = (0..m).map(|i| r_p[i] - a_w_d[i] + a_w2_rd[i]).collect();
            let dy = solve_dense(&n_mat, m, &rhs_y)?;
            let at_dy = mat_t_vec(a, m, n, &dy)?;
            let w_at_dy = mat_vec(&w, n, n, &at_dy)?;
            // Δx̂ = W Aᵀ Δy + d − W r_d.
            let dx_hat: Vec<f64> = (0..n).map(|j| w_at_dy[j] + d[j] - w_rd[j]).collect();
            let dx = mat_vec(&w, n, n, &dx_hat)?;
            let ds_hat: Vec<f64> = (0..n).map(|j| d[j] - dx_hat[j]).collect();
            let ds = mat_vec(&w_inv, n, n, &ds_hat)?;
            Ok((dx, dy, ds, dx_hat, ds_hat))
        };

        // ── predictor (affine): d = −λ ──────────────────────────────────────
        let d_aff: Vec<f64> = lambda.iter().map(|v| -v).collect();
        let (dx_a, _dy_a, ds_a, dxh_a, dsh_a) = solve_dir(&d_aff)?;
        let alpha_p_aff = max_step_in_cone(&x, &dx_a, cone_dims).min(1.0);
        let alpha_d_aff = max_step_in_cone(&s, &ds_a, cone_dims).min(1.0);

        // μ_aff from the affine trial point.
        let xa: Vec<f64> = (0..n).map(|j| x[j] + alpha_p_aff * dx_a[j]).collect();
        let sa: Vec<f64> = (0..n).map(|j| s[j] + alpha_d_aff * ds_a[j]).collect();
        let mu_aff = dot(&xa, &sa)? / p_cones as f64;
        let sigma = if mu < 1e-300 {
            0.0
        } else {
            (mu_aff / mu).powi(3).clamp(0.0, 1.0)
        };

        // ── corrector: rhs_c = σμ e − λ∘λ − Δx̂_aff ∘ Δŝ_aff ─────────────────
        let mut rhs_c = vec![0.0_f64; n];
        {
            let mut off = 0usize;
            for &nj in cone_dims {
                let lam_j = &lambda[off..off + nj];
                let ll = soc_jprod(lam_j, lam_j);
                let cross = soc_jprod(&dxh_a[off..off + nj], &dsh_a[off..off + nj]);
                rhs_c[off] = sigma * mu - ll[0] - cross[0];
                for i in 1..nj {
                    rhs_c[off + i] = -ll[i] - cross[i];
                }
                off += nj;
            }
        }
        // d = Arw(λ)⁻¹ rhs_c (block-diagonal arrow system).
        let mut arw_full = vec![0.0_f64; n * n];
        {
            let mut off = 0usize;
            for &nj in cone_dims {
                let arw = arrow(&lambda[off..off + nj]);
                for aa in 0..nj {
                    for bb in 0..nj {
                        arw_full[(off + aa) * n + off + bb] = arw[aa * nj + bb];
                    }
                }
                off += nj;
            }
        }
        let d_cor = solve_dense(&arw_full, n, &rhs_c)?;
        let (dx, dy, ds, _dxh, _dsh) = solve_dir(&d_cor)?;

        // ── cone step-length guard (fraction-to-boundary) ───────────────────
        let alpha_p = (config.step_fraction * max_step_in_cone(&x, &dx, cone_dims)).min(1.0);
        let alpha_d = (config.step_fraction * max_step_in_cone(&s, &ds, cone_dims)).min(1.0);

        for j in 0..n {
            x[j] += alpha_p * dx[j];
            s[j] += alpha_d * ds[j];
        }
        for i in 0..m {
            y[i] += alpha_d * dy[i];
        }
    }

    Err(CvxError::NotConverged {
        iter: config.max_iter,
        residual: last_rp.max(last_rd).max(last_mu),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MehrotraSocpConfig {
        MehrotraSocpConfig {
            max_iter: 100,
            tol: 1e-10,
            step_fraction: 0.99,
        }
    }

    /// Check `(x_{1:})` lies inside its second-order cone: `‖x̄‖ ≤ x₀ (+ε)`.
    fn cone_ok(x: &[f64], cone_dims: &[usize], eps: f64) -> bool {
        let mut off = 0usize;
        for &nj in cone_dims {
            let xj = &x[off..off + nj];
            let mut nb = 0.0_f64;
            for &v in &xj[1..] {
                nb += v * v;
            }
            if nb.sqrt() > xj[0] + eps {
                return false;
            }
            off += nj;
        }
        true
    }

    // ── (a) analytic optimum: min x₀ s.t. x₁=1, x₂=0, x ∈ SOC(3) ────────────
    #[test]
    fn socp_analytic_min_first_coord() {
        // Optimum x* = (1, 1, 0), cᵀx* = 1 (cone active on the boundary).
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 2×3
        let b = vec![1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        let cone = [3usize];
        let res = mehrotra_socp(&a, 2, 3, &b, &c, &cone, &cfg()).expect("solves");
        assert_eq!(res.status, SocpStatus::Optimal);
        assert!((res.x[0] - 1.0).abs() < 1e-5, "x0={}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-5, "x1={}", res.x[1]);
        assert!(res.x[2].abs() < 1e-5, "x2={}", res.x[2]);
        let obj: f64 = c.iter().zip(res.x.iter()).map(|(ci, xi)| ci * xi).sum();
        assert!((obj - 1.0).abs() < 1e-5, "obj={obj}");
    }

    // ── (a') second analytic optimum: maximise x₁ inside the cone ───────────
    #[test]
    fn socp_analytic_max_inside_cone() {
        // min −x₁ s.t. x₀ = 1, x ∈ SOC(2) ⇒ |x₁| ≤ 1 ⇒ x* = (1, 1), obj = −1.
        let a = vec![1.0, 0.0]; // 1×2
        let b = vec![1.0];
        let c = vec![0.0, -1.0];
        let cone = [2usize];
        let res = mehrotra_socp(&a, 1, 2, &b, &c, &cone, &cfg()).expect("solves");
        assert!((res.x[0] - 1.0).abs() < 1e-5, "x0={}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-5, "x1={}", res.x[1]);
        let obj: f64 = c.iter().zip(res.x.iter()).map(|(ci, xi)| ci * xi).sum();
        assert!((obj + 1.0).abs() < 1e-5, "obj={obj}");
    }

    // ── (b) primal feasibility & cone membership at convergence ─────────────
    #[test]
    fn socp_primal_feasible_and_in_cone() {
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        let cone = [3usize];
        let res = mehrotra_socp(&a, 2, 3, &b, &c, &cone, &cfg()).expect("solves");
        // ‖A x − b‖ small.
        let ax = mat_vec(&a, 2, 3, &res.x).expect("ok");
        let rp = ((ax[0] - b[0]).powi(2) + (ax[1] - b[1]).powi(2)).sqrt();
        assert!(rp < 1e-6, "‖Ax−b‖={rp}");
        assert!(res.primal_res < 1e-6 && res.dual_res < 1e-6);
        assert!(cone_ok(&res.x, &cone, 1e-7), "x left the cone");
        assert!(cone_ok(&res.s, &cone, 1e-7), "s left the cone");
    }

    // ── (c) ‖x_{1:}‖ ≤ x₀ explicitly at the solution ────────────────────────
    #[test]
    fn socp_second_order_cone_constraint_holds() {
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        let cone = [3usize];
        let res = mehrotra_socp(&a, 2, 3, &b, &c, &cone, &cfg()).expect("solves");
        let nb = (res.x[1].powi(2) + res.x[2].powi(2)).sqrt();
        assert!(nb <= res.x[0] + 1e-7, "‖x̄‖={} > x0={}", nb, res.x[0]);
    }

    // ── (d) unbounded / infeasible is flagged ───────────────────────────────
    #[test]
    fn socp_unbounded_is_flagged() {
        // min −x₀ s.t. x₁ = 0, x ∈ SOC(2): x₀ → +∞ (primal unbounded).
        let a = vec![0.0, 1.0]; // 1×2 picks x₁
        let b = vec![0.0];
        let c = vec![-1.0, 0.0];
        let cone = [2usize];
        let res = mehrotra_socp(&a, 1, 2, &b, &c, &cone, &cfg());
        assert!(res.is_err(), "expected an error flag, got {res:?}");
    }

    #[test]
    fn socp_infeasible_is_flagged() {
        // x₀ = −1 forced, but x ∈ SOC keeps x₀ ≥ 0: primal infeasible.
        let a = vec![1.0, 0.0]; // 1×2 picks x₀
        let b = vec![-1.0];
        let c = vec![1.0, 0.0];
        let cone = [2usize];
        let res = mehrotra_socp(&a, 1, 2, &b, &c, &cone, &cfg());
        assert!(res.is_err(), "expected an error flag, got {res:?}");
    }

    // ── (e) dimension mismatches ────────────────────────────────────────────
    #[test]
    fn socp_shape_mismatch_a() {
        let a = vec![0.0, 1.0, 0.0]; // wrong length for 2×3
        let r = mehrotra_socp(&a, 2, 3, &[1.0, 0.0], &[1.0, 0.0, 0.0], &[3], &cfg());
        assert!(matches!(r, Err(CvxError::ShapeMismatch { .. })), "{r:?}");
    }

    #[test]
    fn socp_cone_dims_sum_mismatch() {
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // cone_dims sum to 2 ≠ n = 3.
        let r = mehrotra_socp(&a, 2, 3, &[1.0, 0.0], &[1.0, 0.0, 0.0], &[2], &cfg());
        assert!(
            matches!(r, Err(CvxError::DimensionMismatch { .. })),
            "{r:?}"
        );
    }

    #[test]
    fn socp_bad_step_fraction() {
        let a = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let bad = MehrotraSocpConfig {
            max_iter: 50,
            tol: 1e-9,
            step_fraction: 1.5,
        };
        let r = mehrotra_socp(&a, 2, 3, &[1.0, 0.0], &[1.0, 0.0, 0.0], &[3], &bad);
        assert!(matches!(r, Err(CvxError::InvalidParameter(_))), "{r:?}");
    }

    // ── NT scaling self-consistency: W s = λ = W⁻¹ x, and W·W⁻¹ = I ──────────
    #[test]
    fn nt_scaling_identities() {
        let x = vec![2.0, 1.0];
        let s = vec![3.0, 0.0];
        let cone = [2usize];
        let (w, w_inv, lambda) = build_scaling(&x, &s, &cone, 2).expect("ok");
        let ws = mat_vec(&w, 2, 2, &s).expect("ok");
        let winv_x = mat_vec(&w_inv, 2, 2, &x).expect("ok");
        for i in 0..2 {
            assert!((ws[i] - lambda[i]).abs() < 1e-9, "W s ≠ λ at {i}");
            assert!((winv_x[i] - lambda[i]).abs() < 1e-9, "W⁻¹ x ≠ λ at {i}");
        }
        // W · W⁻¹ = I.
        for col in 0..2 {
            let e: Vec<f64> = (0..2).map(|r| if r == col { 1.0 } else { 0.0 }).collect();
            let wi = mat_vec(&w_inv, 2, 2, &e).expect("ok");
            let wwi = mat_vec(&w, 2, 2, &wi).expect("ok");
            for (r, &val) in wwi.iter().enumerate().take(2) {
                let want = if r == col { 1.0 } else { 0.0 };
                assert!((val - want).abs() < 1e-9, "W W⁻¹ ≠ I");
            }
        }
        // ⟨x, s⟩ = ‖λ‖².
        let xs = dot(&x, &s).expect("ok");
        let ll = dot(&lambda, &lambda).expect("ok");
        assert!((xs - ll).abs() < 1e-9, "⟨x,s⟩ ≠ ‖λ‖²");
    }

    // ── max_step_in_cone behaves on a known geometry ────────────────────────
    #[test]
    fn cone_step_boundary() {
        // u = (1, 0), du = (0, 1): u + α du = (1, α) hits ‖·‖ ≤ u0 at α = 1.
        let alpha = max_step_in_cone(&[1.0, 0.0], &[0.0, 1.0], &[2]);
        assert!((alpha - 1.0).abs() < 1e-12, "alpha={alpha}");
        // du inside the cone ⇒ unbounded step.
        let big = max_step_in_cone(&[1.0, 0.0], &[1.0, 0.0], &[2]);
        assert!(
            big.is_infinite() || big > 1e6,
            "expected large step, got {big}"
        );
    }
}
