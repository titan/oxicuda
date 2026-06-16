//! Wasserstein-1 dual — Kantorovich-Rubinstein potentials (f64 API).
//!
//! By Kantorovich-Rubinstein duality the Wasserstein-1 distance between two
//! discrete measures `a` (on support `X`) and `b` (on support `Y`) equals
//!
//! ```text
//! W₁(a, b) = max_{f 1-Lipschitz}  Σᵢ f(xᵢ) aᵢ − Σⱼ f(yⱼ) bⱼ,
//! ```
//!
//! where "1-Lipschitz" is with respect to the ground metric `d`.  The maximiser
//! `f` (and its `c`-transform `g`) are the *Kantorovich potentials* — the
//! optimal 1-Lipschitz "critic" used, e.g., by Wasserstein GANs.  Unlike the
//! primal transport solvers in [`crate::wasserstein::w1`], this module returns
//! the dual potentials themselves, not merely the scalar distance.
//!
//! ## Entropic Kantorovich-Rubinstein
//! We solve the entropic-smoothed dual, whose potentials are obtained directly
//! from the Sinkhorn scaling vectors and converge to the true Kantorovich
//! potentials as `ε → 0` (Peyré & Cuturi 2019, §4).  With Gibbs kernel
//! `K = exp(−C / ε)` for the *metric* cost `C[i,j] = d(xᵢ, yⱼ)` (note: `W₁` uses
//! the distance, not its square), the scaling fixed point
//!
//! ```text
//! u ← a ./ (K v),   v ← b ./ (Kᵀ u)
//! ```
//!
//! yields the dual potentials `f = ε log u`, `g = ε log v`, and the transport
//! cost `⟨C, T⟩` with `T = diag(u) K diag(v)`.  As `ε → 0` this cost tends to
//! `W₁` from above and `f` becomes 1-Lipschitz.  A small `ε` therefore gives a
//! sharp, convergent estimate of both the distance and the potentials.
//!
//! References:
//! - Villani, C. (2009). *Optimal Transport: Old and New*, Ch. 5 (duality).
//! - Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport*, §4.

use crate::error::{OtError, OtResult};

/// Configuration for the entropic Wasserstein-1 dual solver.
#[derive(Debug, Clone)]
pub struct W1DualConfig {
    /// Entropic regularisation `ε > 0`.  Smaller ⇒ sharper (closer to true W₁)
    /// but slower / less numerically stable.
    pub reg: f64,
    /// Maximum number of Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum absolute change in `u`.
    pub tol: f64,
}

impl Default for W1DualConfig {
    fn default() -> Self {
        Self {
            reg: 0.01,
            max_iter: 2000,
            tol: 1e-12,
        }
    }
}

/// Result of a Wasserstein-1 dual solve.
#[derive(Debug, Clone)]
pub struct W1DualResult {
    /// Entropic-regularised Wasserstein-1 transport cost `⟨C, T⟩`.
    pub distance: f64,
    /// Source-side Kantorovich potential `f = ε log u`, length `|X|`.
    pub f: Vec<f64>,
    /// Target-side Kantorovich potential `g = ε log v`, length `|Y|`.
    pub g: Vec<f64>,
    /// Number of iterations performed.
    pub iters: usize,
}

fn validate(cfg: &W1DualConfig) -> OtResult<()> {
    if cfg.max_iter == 0 {
        return Err(OtError::Internal {
            msg: "w1_dual: max_iter must be ≥ 1".into(),
        });
    }
    if !(cfg.reg > 0.0 && cfg.reg.is_finite()) {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    Ok(())
}

/// Solve the entropic Wasserstein-1 dual for a metric cost matrix `cost`
/// (row-major `m × n`, with `cost[i,j] = d(xᵢ, yⱼ)`) and marginals `a`, `b`.
///
/// # Errors
/// - [`OtError::EmptyInput`] if `a` or `b` is empty.
/// - [`OtError::MarginalMismatch`] if `cost` is not `m × n`.
/// - [`OtError::BadEpsilon`] if `reg ≤ 0`.
/// - [`OtError::NegativeWeight`] if a marginal entry is negative.
/// - [`OtError::MassImbalance`] if the marginals carry different total mass.
pub fn w1_dual_from_cost(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    cfg: &W1DualConfig,
) -> OtResult<W1DualResult> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cost.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: m,
            b_len: n,
        });
    }
    validate(cfg)?;
    if a.iter().chain(b).any(|&v| v < 0.0) {
        return Err(OtError::NegativeWeight);
    }
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if (sum_a - sum_b).abs() > 1e-5 {
        return Err(OtError::MassImbalance {
            sum_a: sum_a as f32,
            sum_b: sum_b as f32,
        });
    }

    let reg = cfg.reg;
    // Stabilise the kernel by subtracting the global minimum cost.
    let cmin = cost.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut k = vec![0.0_f64; m * n];
    for idx in 0..m * n {
        k[idx] = (-(cost[idx] - cmin) / reg).exp();
    }

    let mut u = vec![1.0_f64; m];
    let mut v = vec![1.0_f64; n];
    let mut iters = 0_usize;
    for it in 0..cfg.max_iter {
        iters = it + 1;
        let u_old = u.clone();
        // u = a ./ (K v)
        for i in 0..m {
            let mut kv = 0.0_f64;
            let row = &k[i * n..i * n + n];
            for j in 0..n {
                kv += row[j] * v[j];
            }
            u[i] = a[i] / kv.max(f64::MIN_POSITIVE);
        }
        // v = b ./ (Kᵀ u)
        for j in 0..n {
            let mut ktu = 0.0_f64;
            for i in 0..m {
                ktu += k[i * n + j] * u[i];
            }
            v[j] = b[j] / ktu.max(f64::MIN_POSITIVE);
        }
        // Convergence: max change in u.
        let mut max_du = 0.0_f64;
        for i in 0..m {
            max_du = max_du.max((u[i] - u_old[i]).abs());
        }
        if max_du < cfg.tol {
            break;
        }
    }

    // Transport cost ⟨C, T⟩ with T = diag(u) K diag(v).
    let mut distance = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let t = u[i] * k[i * n + j] * v[j];
            distance += t * cost[i * n + j];
        }
    }

    // Potentials f = ε log u, g = ε log v.  The kernel was stabilised with the
    // shift `cmin`, so the recovered potentials satisfy `f_i + g_j ≤ C_ij −
    // cmin`; we fold `cmin` back into `g` so that the reported potentials are
    // dual-feasible for the *true* cost: `f_i + g_j ≤ C_ij`.
    let f: Vec<f64> = u
        .iter()
        .map(|&ui| reg * ui.max(f64::MIN_POSITIVE).ln())
        .collect();
    let g: Vec<f64> = v
        .iter()
        .map(|&vj| reg * vj.max(f64::MIN_POSITIVE).ln() + cmin)
        .collect();

    Ok(W1DualResult {
        distance,
        f,
        g,
        iters,
    })
}

/// Build the Euclidean metric cost between two point clouds and solve the
/// entropic Wasserstein-1 dual.
///
/// # Arguments
/// - `x`: source support, row-major `[m × dim]`.
/// - `a`: source weights, length `m`.
/// - `y`: target support, row-major `[n × dim]`.
/// - `b`: target weights, length `n`.
/// - `dim`: ambient dimension.
/// - `cfg`: solver configuration.
///
/// # Errors
/// - [`OtError::BadDim`] if `dim == 0`.
/// - [`OtError::EmptyInput`] if either cloud is empty.
/// - [`OtError::IncompatibleLength`] on shape mismatch.
/// - plus the errors of [`w1_dual_from_cost`].
pub fn w1_dual(
    x: &[f64],
    a: &[f64],
    y: &[f64],
    b: &[f64],
    dim: usize,
    cfg: &W1DualConfig,
) -> OtResult<W1DualResult> {
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if x.is_empty() || y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if !x.len().is_multiple_of(dim) || !y.len().is_multiple_of(dim) {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: y.len(),
        });
    }
    let m = x.len() / dim;
    let nt = y.len() / dim;
    if a.len() != m {
        return Err(OtError::IncompatibleLength { a: a.len(), b: m });
    }
    if b.len() != nt {
        return Err(OtError::IncompatibleLength { a: b.len(), b: nt });
    }

    let mut cost = vec![0.0_f64; m * nt];
    for i in 0..m {
        for j in 0..nt {
            let mut sq = 0.0_f64;
            for d in 0..dim {
                let diff = x[i * dim + d] - y[j * dim + d];
                sq += diff * diff;
            }
            cost[i * nt + j] = sq.sqrt();
        }
    }
    w1_dual_from_cost(&cost, a, b, cfg)
}

/// Exact 1D Wasserstein-1 between weighted samples via the closed form
/// `∫ |F_a − F_b|`, used as a reference oracle.
///
/// # Errors
/// - [`OtError::EmptyInput`] if either sample set is empty.
/// - [`OtError::IncompatibleLength`] if weights do not match supports.
pub fn w1_1d_exact(x: &[f64], a: &[f64], y: &[f64], b: &[f64]) -> OtResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if a.len() != x.len() {
        return Err(OtError::IncompatibleLength {
            a: a.len(),
            b: x.len(),
        });
    }
    if b.len() != y.len() {
        return Err(OtError::IncompatibleLength {
            a: b.len(),
            b: y.len(),
        });
    }
    // Merge all breakpoints, integrate |CDF_a − CDF_b| over the real line.
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(x.len() + y.len());
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    for (&xi, &ai) in x.iter().zip(a) {
        pts.push((xi, ai / sum_a));
    }
    for (&yj, &bj) in y.iter().zip(b) {
        pts.push((yj, -bj / sum_b));
    }
    pts.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut total = 0.0_f64;
    let mut cdf = 0.0_f64;
    for w in pts.windows(2) {
        cdf += w[0].1;
        let width = w[1].0 - w[0].0;
        total += cdf.abs() * width;
    }
    Ok(total)
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> W1DualConfig {
        W1DualConfig {
            reg: 0.005,
            max_iter: 5000,
            tol: 1e-13,
        }
    }

    #[test]
    fn two_diracs_distance() {
        // Single point at 0 vs single point at 3.5 in 1D → W1 = 3.5 exactly
        // (single coupling, entropy has no effect when m = n = 1).
        let x = vec![0.0_f64];
        let y = vec![3.5_f64];
        let a = vec![1.0_f64];
        let b = vec![1.0_f64];
        let r = w1_dual(&x, &a, &y, &b, 1, &cfg()).expect("ok");
        assert!((r.distance - 3.5).abs() < 1e-6, "dist={}", r.distance);
    }

    #[test]
    fn two_diracs_2d() {
        // (0,0) vs (3,4): Euclidean distance 5.
        let x = vec![0.0_f64, 0.0];
        let y = vec![3.0_f64, 4.0];
        let a = vec![1.0_f64];
        let b = vec![1.0_f64];
        let r = w1_dual(&x, &a, &y, &b, 2, &cfg()).expect("ok");
        assert!((r.distance - 5.0).abs() < 1e-4, "dist={}", r.distance);
    }

    #[test]
    fn zero_on_equal() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let a = vec![1.0_f64 / 3.0; 3];
        let r = w1_dual(&x, &a, &x, &a, 1, &cfg()).expect("ok");
        assert!(r.distance.abs() < 1e-3, "dist={}", r.distance);
    }

    #[test]
    fn matches_1d_closed_form() {
        // Compare entropic dual (small ε) to the exact ∫|F_a − F_b|.
        let x = vec![0.0_f64, 1.0, 2.0, 3.0];
        let a = vec![0.4_f64, 0.1, 0.3, 0.2];
        let y = vec![0.5_f64, 1.5, 2.5];
        let b = vec![0.2_f64, 0.5, 0.3];
        let exact = w1_1d_exact(&x, &a, &y, &b).expect("ok");
        let r = w1_dual(&x, &a, &y, &b, 1, &cfg()).expect("ok");
        assert!(
            (r.distance - exact).abs() < 5e-2,
            "dual={}, exact={}",
            r.distance,
            exact
        );
    }

    #[test]
    fn symmetry() {
        // Entropic W1 estimate is symmetric: cost(a→b) = cost(b→a).
        let x = vec![0.0_f64, 1.0];
        let a = vec![0.5_f64, 0.5];
        let y = vec![2.0_f64, 4.0];
        let b = vec![0.5_f64, 0.5];
        let r_ab = w1_dual(&x, &a, &y, &b, 1, &cfg()).expect("ok");
        let r_ba = w1_dual(&y, &b, &x, &a, 1, &cfg()).expect("ok");
        assert!(
            (r_ab.distance - r_ba.distance).abs() < 1e-4,
            "ab={}, ba={}",
            r_ab.distance,
            r_ba.distance
        );
    }

    #[test]
    fn distance_nonneg() {
        let x = vec![0.0_f64, 1.0, 5.0];
        let a = vec![0.3_f64, 0.3, 0.4];
        let y = vec![2.0_f64, 3.0];
        let b = vec![0.6_f64, 0.4];
        let r = w1_dual(&x, &a, &y, &b, 1, &cfg()).expect("ok");
        assert!(r.distance >= -1e-9, "dist={}", r.distance);
    }

    #[test]
    fn translation_increases_distance() {
        // Shifting the target further away strictly increases W1.
        let x = vec![0.0_f64, 1.0];
        let a = vec![0.5_f64, 0.5];
        let y_near = vec![2.0_f64, 3.0];
        let y_far = vec![5.0_f64, 6.0];
        let b = vec![0.5_f64, 0.5];
        let near = w1_dual(&x, &a, &y_near, &b, 1, &cfg())
            .expect("ok")
            .distance;
        let far = w1_dual(&x, &a, &y_far, &b, 1, &cfg()).expect("ok").distance;
        assert!(far > near + 1.0, "near={near}, far={far}");
    }

    #[test]
    fn potential_lengths_correct() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let a = vec![1.0_f64 / 3.0; 3];
        let y = vec![3.0_f64, 4.0];
        let b = vec![0.5_f64, 0.5];
        let r = w1_dual(&x, &a, &y, &b, 1, &cfg()).expect("ok");
        assert_eq!(r.f.len(), 3);
        assert_eq!(r.g.len(), 2);
        for v in r.f.iter().chain(&r.g) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn potentials_dual_feasible() {
        // The reported Kantorovich potentials must be (approximately) dual
        // feasible for the true metric cost: f_i + g_j ≤ d(x_i, y_j).  At small
        // ε the entropic smoothing allows only a tiny overshoot.
        let x = vec![0.0_f64, 2.0];
        let a = vec![0.5_f64, 0.5];
        let y = vec![1.0_f64, 3.0];
        let b = vec![0.5_f64, 0.5];
        let r = w1_dual(&x, &a, &y, &b, 1, &cfg()).expect("ok");
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                let d = (xi - yj).abs();
                assert!(
                    r.f[i] + r.g[j] <= d + 1e-2,
                    "dual infeasible: f[{i}]+g[{j}]={} > {d}",
                    r.f[i] + r.g[j]
                );
            }
        }
    }

    #[test]
    fn mass_imbalance_rejected() {
        let x = vec![0.0_f64];
        let a = vec![1.0_f64];
        let y = vec![1.0_f64];
        let b = vec![2.0_f64];
        let res = w1_dual(&x, &a, &y, &b, 1, &cfg());
        assert!(matches!(res, Err(OtError::MassImbalance { .. })));
    }

    #[test]
    fn bad_dim_rejected() {
        let res = w1_dual(&[0.0_f64], &[1.0_f64], &[0.0_f64], &[1.0_f64], 0, &cfg());
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn empty_rejected() {
        let res = w1_dual(&[], &[], &[0.0_f64], &[1.0_f64], 1, &cfg());
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn bad_eps_rejected() {
        let bad = W1DualConfig { reg: 0.0, ..cfg() };
        let res = w1_dual(&[0.0_f64], &[1.0_f64], &[1.0_f64], &[1.0_f64], 1, &bad);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }
}
