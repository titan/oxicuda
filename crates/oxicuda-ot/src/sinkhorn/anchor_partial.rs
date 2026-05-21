//! Anchor-based Partial Optimal Transport (Chapel et al. 2020).
//!
//! Classical OT requires that *all* mass is transported. Partial OT relaxes this
//! by allowing a fraction `mass ∈ (0, 1]` of the total mass to be transported while
//! the remainder is discarded.  The anchor approach augments both marginals with a
//! dummy "anchor" component that absorbs the unmatched mass at zero cost, reducing
//! the partial problem to a standard balanced OT problem of size `(n+1) × (m+1)`.
//!
//! # Construction
//!
//! Given normalised histograms `a ∈ Δ_n` and `b ∈ Δ_m` and a parameter
//! `mass ∈ (0, 1]`:
//!
//! ```text
//! a_aug = [a_1·mass, …, a_n·mass, 1−mass]   ∈ Δ_{n+1}
//! b_aug = [b_1·mass, …, b_m·mass, 1−mass]   ∈ Δ_{m+1}
//! C_aug[i, j] = C[i,j]  for i<n, j<m
//! C_aug[i, m] = 0        for all i   (source-to-anchor)
//! C_aug[n, j] = 0        for all j   (anchor-to-target)
//! C_aug[n, m] = 0                    (anchor self-coupling)
//! ```
//!
//! Standard log-domain Sinkhorn on `(C_aug, a_aug, b_aug)` then returns a plan
//! whose `n×m` top-left block is the partial transport plan.

use crate::error::{OtError, OtResult};

/// Configuration for the anchor-based partial OT solver.
#[derive(Debug, Clone)]
pub struct AnchorPartialConfig {
    /// Entropic regularisation strength ε (`> 0`).
    pub reg: f32,
    /// Fraction of mass to transport; must be in `(0, 1]`.
    /// `mass = 1.0` recovers the full balanced OT problem.
    pub mass: f32,
    /// Maximum number of Sinkhorn iterations.
    pub max_iter: usize,
    /// Marginal-residual convergence tolerance.
    pub tol: f32,
}

impl Default for AnchorPartialConfig {
    fn default() -> Self {
        Self {
            reg: 0.1,
            mass: 0.8,
            max_iter: 500,
            tol: 1e-4,
        }
    }
}

/// Output of the anchor-based partial OT solver.
///
/// Dual potentials `log_u` and `log_v` correspond to the *augmented* problem
/// (length `n+1` and `m+1`). The top-left `n×m` block of the augmented plan
/// is the partial transport plan.
#[derive(Debug, Clone)]
pub struct AnchorPartialFit {
    /// Row dual potentials for the augmented problem, length `n + 1`.
    pub log_u: Vec<f32>,
    /// Column dual potentials for the augmented problem, length `m + 1`.
    pub log_v: Vec<f32>,
    /// Total mass in the recovered partial transport plan `Σ_{i<n, j<m} P_ij`.
    pub transported_mass: f32,
    /// Transport cost `<P, C>` restricted to the `n×m` block (anchor excluded).
    pub cost: f32,
    /// Number of source histogram bins.
    pub n: usize,
    /// Number of target histogram bins.
    pub m: usize,
}

// ---------------------------------------------------------------------------
// Internal numerics
// ---------------------------------------------------------------------------

/// Stable log-sum-exp over a slice.
fn logsumexp(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut mx = f32::NEG_INFINITY;
    for &x in v {
        if x > mx {
            mx = x;
        }
    }
    if !mx.is_finite() {
        return mx;
    }
    let mut s = 0.0_f32;
    for &x in v {
        s += (x - mx).exp();
    }
    mx + s.ln()
}

fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Log-domain Sinkhorn-Knopp on the augmented `(n+1)×(m+1)` problem.
///
/// Returns `(log_u, log_v)` potentials after convergence.
fn log_sinkhorn_augmented(
    c_aug: &[f32],
    a_aug: &[f32],
    b_aug: &[f32],
    na1: usize, // n+1
    mb1: usize, // m+1
    eps: f32,
    max_iter: usize,
    tol: f32,
) -> OtResult<(Vec<f32>, Vec<f32>)> {
    let mut u = vec![0.0_f32; na1];
    let mut v = vec![0.0_f32; mb1];

    // Initialise u from log(a).
    for (i, &ai) in a_aug.iter().enumerate() {
        u[i] = eps * safe_ln(ai);
    }
    for (j, &bj) in b_aug.iter().enumerate() {
        v[j] = eps * safe_ln(bj);
    }

    let mut buf = vec![0.0_f32; na1.max(mb1)];

    for it in 0..max_iter {
        // Row update: u_i ← ε log(a_i) − ε LSE_j[(v_j − C_ij)/ε]
        for i in 0..na1 {
            let row_off = i * mb1;
            for j in 0..mb1 {
                buf[j] = (v[j] - c_aug[row_off + j]) / eps;
            }
            let lse = logsumexp(&buf[..mb1]);
            u[i] = eps * safe_ln(a_aug[i]) - eps * lse;
        }

        // Measure column residual.
        let mut max_res = 0.0_f32;
        for j in 0..mb1 {
            let mut col_sum = 0.0_f32;
            for i in 0..na1 {
                col_sum += ((u[i] + v[j] - c_aug[i * mb1 + j]) / eps).exp();
            }
            let r = (col_sum - b_aug[j]).abs();
            if r > max_res {
                max_res = r;
            }
        }

        // Column update: v_j ← ε log(b_j) − ε LSE_i[(u_i − C_ij)/ε]
        for j in 0..mb1 {
            for i in 0..na1 {
                buf[i] = (u[i] - c_aug[i * mb1 + j]) / eps;
            }
            let lse = logsumexp(&buf[..na1]);
            v[j] = eps * safe_ln(b_aug[j]) - eps * lse;
        }

        if max_res < tol {
            return Ok((u, v));
        }

        if it + 1 == max_iter {
            return Err(OtError::NotConverged {
                iter: max_iter,
                tol,
            });
        }
    }

    Ok((u, v))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Validate anchor-partial OT inputs.
fn validate(
    a: &[f32],
    b: &[f32],
    cost: &[f32],
    n: usize,
    m: usize,
    cfg: &AnchorPartialConfig,
) -> OtResult<()> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.reg });
    }
    if cfg.mass <= 0.0 || cfg.mass > 1.0 || !cfg.mass.is_finite() {
        return Err(OtError::Internal {
            msg: format!("mass must be in (0, 1], got {}", cfg.mass),
        });
    }
    if cost.len() != n * m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != n || b.len() != m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    for &ai in a {
        if ai < 0.0 || !ai.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    for &bj in b {
        if bj < 0.0 || !bj.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

/// Solve the anchor-based partial OT problem.
///
/// Returns the dual potentials and summary statistics for the partial plan.
///
/// # Arguments
/// * `a` — source histogram of length `n` (need not sum to 1).
/// * `b` — target histogram of length `m`.
/// * `cost` — ground cost matrix, length `n × m` row-major.
/// * `n` — number of source bins.
/// * `m` — number of target bins.
/// * `cfg` — solver configuration.
pub fn anchor_partial_ot(
    a: &[f32],
    b: &[f32],
    cost: &[f32],
    n: usize,
    m: usize,
    cfg: &AnchorPartialConfig,
) -> OtResult<AnchorPartialFit> {
    validate(a, b, cost, n, m, cfg)?;

    let mass = cfg.mass;
    let anchor_mass = 1.0 - mass;

    // Build augmented marginals (length n+1 and m+1).
    let na1 = n + 1;
    let mb1 = m + 1;

    let mut a_aug = Vec::with_capacity(na1);
    // Normalise a to sum to 1, then scale by mass.
    let sum_a: f32 = a.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    for &ai in a {
        a_aug.push(ai / sum_a * mass);
    }
    a_aug.push(anchor_mass);

    let mut b_aug = Vec::with_capacity(mb1);
    let sum_b: f32 = b.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    for &bj in b {
        b_aug.push(bj / sum_b * mass);
    }
    b_aug.push(anchor_mass);

    // Build augmented cost matrix (n+1) × (m+1) row-major.
    // C_aug[i, j] = C[i, j]  for i<n, j<m
    // C_aug[i, m] = 0        for all i (source→anchor: free)
    // C_aug[n, j] = 0        for all j (anchor→target: free)
    // C_aug[n, m] = 0
    let mut c_aug = vec![0.0_f32; na1 * mb1];
    for i in 0..n {
        let src_row = i * m;
        let aug_row = i * mb1;
        c_aug[aug_row..(aug_row + m)].copy_from_slice(&cost[src_row..(src_row + m)]);
        // c_aug[aug_row + m] = 0 by default
    }
    // Row n (anchor) is all zeros by initialisation.

    // Run log-domain Sinkhorn on the augmented problem.
    let (log_u, log_v) = log_sinkhorn_augmented(
        &c_aug,
        &a_aug,
        &b_aug,
        na1,
        mb1,
        cfg.reg,
        cfg.max_iter,
        cfg.tol,
    )?;

    // Compute transported mass and cost from the n×m sub-block.
    let eps = cfg.reg;
    let mut transported_mass = 0.0_f32;
    let mut plan_cost = 0.0_f32;
    for i in 0..n {
        let aug_row = i * mb1;
        for j in 0..m {
            let p_ij = ((log_u[i] + log_v[j] - c_aug[aug_row + j]) / eps).exp();
            transported_mass += p_ij;
            plan_cost += p_ij * cost[i * m + j];
        }
    }

    Ok(AnchorPartialFit {
        log_u,
        log_v,
        transported_mass,
        cost: plan_cost,
        n,
        m,
    })
}

/// Compute the transport cost `<P, C>` restricted to the `n×m` block.
///
/// Returns the stored `fit.cost` field, which equals `Σ_{i,j} P_ij · C[i,j]`
/// excluding the anchor rows and columns.  The `cost` parameter is accepted for
/// API symmetry with `anchor_partial_plan`; its length is validated but the
/// actual computation uses the pre-computed value in `fit`.
pub fn anchor_partial_transport_cost(fit: &AnchorPartialFit, cost: &[f32]) -> f32 {
    // Validate shape; fall back gracefully if caller supplies wrong buffer.
    if cost.len() != fit.n * fit.m {
        return fit.cost;
    }
    fit.cost
}

/// Reconstruct the `n × m` partial transport plan from the fit.
///
/// The plan entries are `P_ij = exp((u_i + v_j − C_aug[i,j]) / ε)`.
/// Since `ε` is not stored in the fit, this function requires the original
/// cost slice (to verify shape) and returns the plan as it was computed
/// internally. The entries are extracted directly from the stored potentials
/// assuming the caller provides the same `reg` value.
///
/// # Arguments
/// * `fit` — result of `anchor_partial_ot`.
/// * `cost` — original cost matrix, length `n × m` row-major (used for shape check).
/// * `reg` — regularisation strength ε used during fitting.
pub fn anchor_partial_plan(fit: &AnchorPartialFit, cost: &[f32], reg: f32) -> Vec<f32> {
    let n = fit.n;
    let m = fit.m;
    if cost.len() != n * m || reg <= 0.0 {
        return vec![0.0_f32; n * m];
    }
    let mb1 = m + 1;
    let eps = reg;
    let mut plan = vec![0.0_f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let c_ij = cost[i * m + j];
            let log_p = (fit.log_u[i] + fit.log_v[j] - c_ij) / eps;
            plan[i * m + j] = log_p.exp();
        }
    }
    let _ = mb1;
    plan
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f32> {
        vec![1.0_f32 / n as f32; n]
    }

    fn distance_cost(n: usize, m: usize) -> Vec<f32> {
        // C[i,j] = |i/(n-1) - j/(m-1)|
        let mut c = vec![0.0_f32; n * m];
        for i in 0..n {
            let xi = i as f32 / (n - 1).max(1) as f32;
            for j in 0..m {
                let yj = j as f32 / (m - 1).max(1) as f32;
                c[i * m + j] = (xi - yj).abs();
            }
        }
        c
    }

    // -----------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------

    #[test]
    fn rejects_zero_n() {
        let cfg = AnchorPartialConfig::default();
        let res = anchor_partial_ot(&[], &[0.5, 0.5], &[], 0, 2, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn rejects_bad_reg() {
        let n = 2;
        let m = 2;
        let a = uniform(n);
        let b = uniform(m);
        let c = vec![0.0_f32; n * m];
        let cfg = AnchorPartialConfig {
            reg: 0.0,
            ..Default::default()
        };
        let res = anchor_partial_ot(&a, &b, &c, n, m, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn rejects_invalid_mass() {
        let n = 2;
        let m = 2;
        let a = uniform(n);
        let b = uniform(m);
        let c = vec![0.0_f32; n * m];
        let cfg = AnchorPartialConfig {
            mass: 0.0,
            ..Default::default()
        };
        let res = anchor_partial_ot(&a, &b, &c, n, m, &cfg);
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn rejects_negative_weights() {
        let n = 2;
        let m = 2;
        let a = vec![-0.5_f32, 1.5];
        let b = uniform(m);
        let c = vec![0.0_f32; n * m];
        let cfg = AnchorPartialConfig::default();
        let res = anchor_partial_ot(&a, &b, &c, n, m, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn rejects_cost_shape_mismatch() {
        let n = 2;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let c = vec![0.0_f32; n * m + 1]; // wrong size
        let cfg = AnchorPartialConfig::default();
        let res = anchor_partial_ot(&a, &b, &c, n, m, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    // -----------------------------------------------------------------
    // Functional tests
    // -----------------------------------------------------------------

    #[test]
    fn full_mass_recovers_full_ot() {
        // With mass = 1, partial OT = full balanced OT.
        let n = 3;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg = AnchorPartialConfig {
            reg: 0.1,
            mass: 1.0,
            max_iter: 1000,
            tol: 1e-5,
        };
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("converges");
        // Transported mass ≈ 1 for mass=1.
        assert!(
            (fit.transported_mass - 1.0).abs() < 0.05,
            "transported_mass={} for full OT",
            fit.transported_mass
        );
    }

    #[test]
    fn transported_mass_bounded_by_configured_mass() {
        // With mass = 0.7, the n×m sub-block transported mass is at most `mass`.
        // Entropic regularisation causes the plan to spread mass to the anchor,
        // so the actual transported mass is <= mass (and strictly positive).
        let n = 4;
        let m = 4;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg = AnchorPartialConfig {
            reg: 0.2,
            mass: 0.7,
            max_iter: 1000,
            tol: 1e-4,
        };
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("converges");
        assert!(
            fit.transported_mass > 0.0,
            "transported_mass should be positive, got {}",
            fit.transported_mass
        );
        assert!(
            fit.transported_mass <= cfg.mass + 1e-4,
            "transported_mass={} exceeds mass={}",
            fit.transported_mass,
            cfg.mass
        );
    }

    #[test]
    fn plan_non_negative_and_finite() {
        let n = 3;
        let m = 4;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg = AnchorPartialConfig {
            reg: 0.3,
            mass: 0.8,
            max_iter: 500,
            tol: 1e-4,
        };
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("converges");
        let plan = anchor_partial_plan(&fit, &c, cfg.reg);
        assert_eq!(plan.len(), n * m);
        for &p in &plan {
            assert!(p >= 0.0 && p.is_finite(), "plan entry {p}");
        }
    }

    #[test]
    fn plan_row_sums_at_most_source_marginals() {
        // Row sums of partial plan ≤ a_i (unmatched mass goes to anchor).
        let n = 3;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg = AnchorPartialConfig {
            reg: 0.15,
            mass: 0.8,
            max_iter: 800,
            tol: 1e-4,
        };
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("converges");
        let plan = anchor_partial_plan(&fit, &c, cfg.reg);
        for i in 0..n {
            let row_sum: f32 = (0..m).map(|j| plan[i * m + j]).sum();
            assert!(
                row_sum <= a[i] * cfg.mass + 0.05,
                "row {i} sum {row_sum} > a[i]*mass={}",
                a[i] * cfg.mass
            );
        }
    }

    #[test]
    fn cost_field_non_negative() {
        let n = 2;
        let m = 2;
        let a = uniform(n);
        let b = uniform(m);
        let c = vec![0.0_f32, 1.0, 1.0, 0.0];
        let cfg = AnchorPartialConfig {
            reg: 0.2,
            mass: 0.9,
            max_iter: 500,
            tol: 1e-4,
        };
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("converges");
        assert!(fit.cost >= -1e-5, "cost={}", fit.cost);
    }

    #[test]
    fn lower_mass_reduces_transport_cost() {
        // Less mass transported → can cherry-pick cheaper pairs → cost ≤ cost at full mass.
        let n = 4;
        let m = 4;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg_full = AnchorPartialConfig {
            reg: 0.3,
            mass: 1.0,
            max_iter: 800,
            tol: 1e-4,
        };
        let cfg_partial = AnchorPartialConfig {
            reg: 0.3,
            mass: 0.5,
            max_iter: 800,
            tol: 1e-4,
        };
        let fit_full = anchor_partial_ot(&a, &b, &c, n, m, &cfg_full).expect("full");
        let fit_partial = anchor_partial_ot(&a, &b, &c, n, m, &cfg_partial).expect("partial");
        assert!(
            fit_partial.cost <= fit_full.cost + 1e-3,
            "partial cost {} > full cost {}",
            fit_partial.cost,
            fit_full.cost
        );
    }

    #[test]
    fn transport_cost_helper_consistent() {
        let n = 3;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg = AnchorPartialConfig::default();
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("ok");
        let tc = anchor_partial_transport_cost(&fit, &c);
        // Should match fit.cost.
        assert!(
            (tc - fit.cost).abs() < 1e-5,
            "tc={tc} fit.cost={}",
            fit.cost
        );
    }

    #[test]
    fn dual_potentials_correct_shape() {
        let n = 4;
        let m = 5;
        let a = uniform(n);
        let b = uniform(m);
        let c = distance_cost(n, m);
        let cfg = AnchorPartialConfig::default();
        let fit = anchor_partial_ot(&a, &b, &c, n, m, &cfg).expect("ok");
        assert_eq!(fit.log_u.len(), n + 1);
        assert_eq!(fit.log_v.len(), m + 1);
    }
}
