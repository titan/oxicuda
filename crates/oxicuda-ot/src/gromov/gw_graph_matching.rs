//! GW-Wasserstein hybrid for graph matching (Titouan et al. 2019).
//!
//! Combines Gromov-Wasserstein (structural matching via intra-domain
//! cost matrices) with a standard Wasserstein term over node features:
//!
//! ```text
//! L(T) = (1 − α) · <C_feat, T> + α · GW_cost(C_s, C_t, T)
//! ```
//!
//! where `C_feat[i,j] = ‖feat_s[i] − feat_t[j]‖²` and the GW cost uses
//! the squared Euclidean loss `L(x,y) = (x−y)²`.  The plan `T` is found by
//! the standard Frank-Wolfe / Bregman iteration with an inner Sinkhorn step.
//!
//! All arithmetic is `f64` throughout for numerical stability on real graphs.

use crate::error::{OtError, OtResult};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the GW-Wasserstein graph matching solver.
#[derive(Debug, Clone)]
pub struct GwGraphConfig {
    /// Mixing parameter α ∈ [0, 1]: 0 = pure Wasserstein, 1 = pure GW.
    pub alpha: f64,
    /// Sinkhorn entropic regularisation strength (must be > 0).
    pub eps: f64,
    /// Maximum outer (Frank-Wolfe) iterations.
    pub max_outer: usize,
    /// Maximum inner Sinkhorn iterations per outer step.
    pub inner_max_iter: usize,
    /// Convergence tolerance on ‖T_new − T‖_F.
    pub tol: f64,
}

impl Default for GwGraphConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            eps: 0.1,
            max_outer: 100,
            inner_max_iter: 100,
            tol: 1e-6,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Output of [`gw_graph_matching`].
#[derive(Debug, Clone)]
pub struct GwGraphResult {
    /// Transport plan, shape `n × m` in row-major order (length `n * m`).
    pub plan: Vec<f64>,
    /// Total combined cost `(1 − α) * feat_cost + α * gw_cost`.
    pub cost: f64,
    /// GW structural portion of the cost.
    pub gw_cost: f64,
    /// Feature Wasserstein portion of the cost.
    pub feat_cost: f64,
    /// Number of completed outer iterations.
    pub iters: usize,
}

// ─── Internal Sinkhorn (f64) ─────────────────────────────────────────────────

/// Stable log-sum-exp on a `f64` slice.
fn logsumexp_f64(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }
    let sum: f64 = slice.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Safe log (clamps zero to `f64::MIN_POSITIVE`).
fn safe_ln_f64(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Log-domain Sinkhorn-Knopp in `f64`.
///
/// Solves `min_P <cost, P> + ε KL(P ‖ a⊗b)` with marginals `a` (len `m`)
/// and `b` (len `n`). `cost` is row-major `m × n`.
///
/// Returns the plan as a flat `m × n` row-major vector.
fn sinkhorn_f64(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    eps: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    // Initialise dual potentials in log-space.
    let mut u: Vec<f64> = a.iter().map(|&ai| eps * safe_ln_f64(ai)).collect();
    let mut v: Vec<f64> = b.iter().map(|&bj| eps * safe_ln_f64(bj)).collect();

    let mut buf = vec![0.0_f64; m.max(n)];

    for _ in 0..max_iter {
        // Row update.
        for i in 0..m {
            let row_off = i * n;
            for j in 0..n {
                buf[j] = (v[j] - cost[row_off + j]) / eps;
            }
            let lse = logsumexp_f64(&buf[..n]);
            u[i] = eps * safe_ln_f64(a[i]) - eps * lse;
        }

        // Convergence check on column residuals.
        let mut max_res = 0.0_f64;
        for (j, &bj) in b.iter().enumerate() {
            let col_sum: f64 = u
                .iter()
                .enumerate()
                .map(|(i, &ui)| ((ui + v[j] - cost[i * n + j]) / eps).exp())
                .sum();
            let r = (col_sum - bj).abs();
            if r > max_res {
                max_res = r;
            }
        }
        if max_res < tol {
            // Final column update to symmetrize.
            for j in 0..n {
                for i in 0..m {
                    buf[i] = (u[i] - cost[i * n + j]) / eps;
                }
                let lse = logsumexp_f64(&buf[..m]);
                v[j] = eps * safe_ln_f64(b[j]) - eps * lse;
            }
            break;
        }

        // Column update.
        for j in 0..n {
            for i in 0..m {
                buf[i] = (u[i] - cost[i * n + j]) / eps;
            }
            let lse = logsumexp_f64(&buf[..m]);
            v[j] = eps * safe_ln_f64(b[j]) - eps * lse;
        }
    }

    // Materialise the plan.
    let mut plan = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            plan[i * n + j] = ((u[i] + v[j] - cost[i * n + j]) / eps).exp();
        }
    }
    plan
}

// ─── GW Gradient ─────────────────────────────────────────────────────────────

/// Compute the GW gradient `G_ij = −2 · Σ_{kl} C_s[i,k] · T[k,l] · C_t[j,l]`
/// in `O(n² m + n m²)` by splitting into two matrix products.
///
/// `c_s` and `c_t` are given as `&[Vec<f64>]` (n×n and m×m respectively).
/// `plan` is row-major `n × m`.
fn gw_gradient_f64(
    c_s: &[Vec<f64>],
    c_t: &[Vec<f64>],
    plan: &[f64],
    ns: usize,
    nt: usize,
) -> Vec<f64> {
    // Step 1: tmp[i,l] = Σ_k C_s[i,k] · T[k,l]  (ns × nt)
    let mut tmp = vec![0.0_f64; ns * nt];
    for (i, cs_row) in c_s.iter().enumerate() {
        for (k, &c_ik) in cs_row.iter().enumerate() {
            if c_ik == 0.0 {
                continue;
            }
            let plan_off = k * nt;
            for l in 0..nt {
                tmp[i * nt + l] += c_ik * plan[plan_off + l];
            }
        }
    }
    // Step 2: G[i,j] = −2 · Σ_l tmp[i,l] · C_t[j,l]  (ns × nt)
    let mut g = vec![0.0_f64; ns * nt];
    for i in 0..ns {
        for (j, ct_row) in c_t.iter().enumerate() {
            let acc: f64 = tmp[i * nt..i * nt + nt]
                .iter()
                .zip(ct_row.iter())
                .map(|(t, c)| t * c)
                .sum();
            g[i * nt + j] = -2.0 * acc;
        }
    }
    g
}

// ─── Feature cost ────────────────────────────────────────────────────────────

/// Compute the squared-Euclidean feature cost matrix `C_feat[i,j] = ‖feat_s[i] − feat_t[j]‖²`.
fn feature_cost_matrix(feat_s: &[Vec<f64>], feat_t: &[Vec<f64>]) -> Vec<f64> {
    let ns = feat_s.len();
    let nt = feat_t.len();
    let mut c = vec![0.0_f64; ns * nt];
    for (i, fi) in feat_s.iter().enumerate() {
        for (j, fj) in feat_t.iter().enumerate() {
            let sq: f64 = fi
                .iter()
                .zip(fj.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            c[i * nt + j] = sq;
        }
    }
    c
}

// ─── GW Frobenius cost ────────────────────────────────────────────────────────

/// Compute the full GW Frobenius cost
/// `Σ_{i,j,k,l} (C_s[i,k] − C_t[j,l])² · T[i,j] · T[k,l]`.
///
/// Expanded as `Σ C_s²_ik T_row_i T_row_k  +  Σ C_t²_jl T_col_j T_col_l
/// + Σ T_ij · G_ij` (where G is the GW gradient, containing the −2 factor).
pub fn gw_frobenius_cost(c_s: &[Vec<f64>], c_t: &[Vec<f64>], t: &[f64]) -> f64 {
    let ns = c_s.len();
    let nt = if ns > 0 { c_s[0].len() } else { 0 };
    let nm = c_t.len();
    if ns == 0 || nt == 0 || nm == 0 || t.is_empty() {
        return 0.0;
    }

    let row_sums: Vec<f64> = (0..ns)
        .map(|i| (0..nm).map(|j| t[i * nm + j]).sum::<f64>())
        .collect();
    let col_sums: Vec<f64> = (0..nm)
        .map(|j| (0..ns).map(|i| t[i * nm + j]).sum::<f64>())
        .collect();

    let mut term1 = 0.0_f64;
    for (i, cs_row) in c_s.iter().enumerate() {
        for (k, &v) in cs_row.iter().enumerate() {
            term1 += v * v * row_sums[i] * row_sums[k];
        }
    }
    let mut term2 = 0.0_f64;
    for (j, ct_row) in c_t.iter().enumerate() {
        for (l, &v) in ct_row.iter().enumerate() {
            term2 += v * v * col_sums[j] * col_sums[l];
        }
    }
    let g = gw_gradient_f64(c_s, c_t, t, ns, nm);
    let cross: f64 = t.iter().zip(g.iter()).map(|(ti, gi)| ti * gi).sum();
    term1 + term2 + cross
}

// ─── Final cost ──────────────────────────────────────────────────────────────

/// Compute `feat_cost = <C_feat, T>`.
fn inner_product(c: &[f64], t: &[f64]) -> f64 {
    c.iter().zip(t.iter()).map(|(ci, ti)| ci * ti).sum()
}

/// Frobenius norm of `a − b`.
fn frob_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let d = av - bv;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_gw_graph(
    a: &[f64],
    b: &[f64],
    c_s: &[Vec<f64>],
    c_t: &[Vec<f64>],
    feat_s: &[Vec<f64>],
    feat_t: &[Vec<f64>],
    config: &GwGraphConfig,
) -> OtResult<(usize, usize)> {
    let ns = a.len();
    let nt = b.len();

    if ns == 0 || nt == 0 {
        return Err(OtError::EmptyInput);
    }
    if feat_s.is_empty() || feat_t.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if !config.alpha.is_finite() || !(0.0..=1.0).contains(&config.alpha) {
        return Err(OtError::Internal {
            msg: format!("alpha must be in [0, 1], got {}", config.alpha),
        });
    }
    if config.eps <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: config.eps as f32,
        });
    }
    if c_s.len() != ns {
        return Err(OtError::MarginalMismatch {
            m: ns,
            n: nt,
            a_len: c_s.len(),
            b_len: ns,
        });
    }
    for row in c_s {
        if row.len() != ns {
            return Err(OtError::MarginalMismatch {
                m: ns,
                n: ns,
                a_len: row.len(),
                b_len: ns,
            });
        }
    }
    if c_t.len() != nt {
        return Err(OtError::MarginalMismatch {
            m: nt,
            n: nt,
            a_len: c_t.len(),
            b_len: nt,
        });
    }
    for row in c_t {
        if row.len() != nt {
            return Err(OtError::MarginalMismatch {
                m: nt,
                n: nt,
                a_len: row.len(),
                b_len: nt,
            });
        }
    }
    if feat_s.len() != ns {
        return Err(OtError::MarginalMismatch {
            m: ns,
            n: nt,
            a_len: feat_s.len(),
            b_len: ns,
        });
    }
    if feat_t.len() != nt {
        return Err(OtError::MarginalMismatch {
            m: ns,
            n: nt,
            a_len: feat_t.len(),
            b_len: nt,
        });
    }
    // Feature dimension consistency.
    let d_s = feat_s[0].len();
    let d_t = feat_t[0].len();
    if d_s == 0 || d_t == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    for row in feat_s {
        if row.len() != d_s {
            return Err(OtError::Internal {
                msg: "feat_s rows have inconsistent dimension".into(),
            });
        }
    }
    for row in feat_t {
        if row.len() != d_t {
            return Err(OtError::Internal {
                msg: "feat_t rows have inconsistent dimension".into(),
            });
        }
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
    Ok((ns, nt))
}

// ─── Main solver ─────────────────────────────────────────────────────────────

/// GW-Wasserstein graph matching.
///
/// Combines a Wasserstein feature term with Gromov-Wasserstein structural cost
/// via mixing parameter `α`:
///
/// ```text
/// L(T) = (1 − α) · <C_feat, T> + α · GW(C_s, C_t, T)
/// ```
///
/// # Arguments
/// - `a`, `b`: node weight vectors (probability simplex) for source and target.
/// - `c_s`: `ns × ns` source intra-domain distance matrix.
/// - `c_t`: `nt × nt` target intra-domain distance matrix.
/// - `feat_s`: `ns × d` source node features (one row per node).
/// - `feat_t`: `nt × d` target node features.
/// - `config`: solver hyperparameters.
pub fn gw_graph_matching(
    a: &[f64],
    b: &[f64],
    c_s: &[Vec<f64>],
    c_t: &[Vec<f64>],
    feat_s: &[Vec<f64>],
    feat_t: &[Vec<f64>],
    config: &GwGraphConfig,
) -> OtResult<GwGraphResult> {
    let (ns, nt) = validate_gw_graph(a, b, c_s, c_t, feat_s, feat_t, config)?;

    let alpha = config.alpha;
    let eps = config.eps;
    let tol = config.tol;

    // Pre-compute the feature cost matrix C_feat (ns × nt).
    let c_feat = feature_cost_matrix(feat_s, feat_t);

    // Initialise plan as outer product T_0 = a ⊗ b.
    let mut plan = vec![0.0_f64; ns * nt];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            plan[i * nt + j] = ai * bj;
        }
    }

    // Combined cost buffer (ns × nt).
    let mut m_cost = vec![0.0_f64; ns * nt];
    let mut completed = 0_usize;

    for _iter in 0..config.max_outer {
        // GW gradient at current plan.
        let g_gw = gw_gradient_f64(c_s, c_t, &plan, ns, nt);

        // Combined cost M = (1 − α) · C_feat + α · G_gw.
        for idx in 0..ns * nt {
            m_cost[idx] = (1.0 - alpha) * c_feat[idx] + alpha * g_gw[idx];
        }

        // Inner Sinkhorn step.
        let new_plan = sinkhorn_f64(&m_cost, a, b, ns, nt, eps, config.inner_max_iter, tol);

        let delta = frob_diff_f64(&plan, &new_plan);
        plan = new_plan;
        completed += 1;

        if delta < tol {
            break;
        }
    }

    // Compute final decomposed costs.
    let feat_cost = inner_product(&c_feat, &plan);
    let gw_cost_val = gw_frobenius_cost(c_s, c_t, &plan);
    let cost = (1.0 - alpha) * feat_cost + alpha * gw_cost_val;

    Ok(GwGraphResult {
        plan,
        cost,
        gw_cost: gw_cost_val,
        feat_cost,
        iters: completed,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn sym_cost(n: usize, f: impl Fn(usize, usize) -> f64) -> Vec<Vec<f64>> {
        (0..n).map(|i| (0..n).map(|j| f(i, j)).collect()).collect()
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn default_config() -> GwGraphConfig {
        GwGraphConfig {
            alpha: 0.5,
            eps: 0.1,
            max_outer: 100,
            inner_max_iter: 200,
            tol: 1e-6,
        }
    }

    // ─── alpha = 0: reduces to Wasserstein on features ────────────────────

    #[test]
    fn alpha_zero_close_to_wasserstein() {
        // With α = 0, the GW gradient is ignored and the combined cost is just
        // C_feat. The result must be a valid transport plan.
        let n = 3;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| if i == j { 0.0 } else { 1.0 });
        let c_t = sym_cost(n, |i, j| if i == j { 0.0 } else { 1.0 });
        // Features: source at [0,0],[1,0],[2,0]; target at [0,0],[1,0],[2,0] (identical).
        let feat_s: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let feat_t = feat_s.clone();
        let cfg = GwGraphConfig {
            alpha: 0.0,
            ..default_config()
        };
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat_s, &feat_t, &cfg).unwrap();
        // Plan must be non-negative.
        for &p in &res.plan {
            assert!(p >= -1e-9, "negative plan entry {p}");
        }
        // Row marginals must match a.
        for (i, &ai) in a.iter().enumerate().take(n) {
            let row: f64 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row, ai, 1e-3), "row {i} sum {row} != {ai}");
        }
    }

    // ─── alpha = 1: reduces to GW ─────────────────────────────────────────

    #[test]
    fn alpha_one_plan_is_valid() {
        let n = 3;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        let c_t = c_s.clone();
        let feat_s: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let feat_t = feat_s.clone();
        let cfg = GwGraphConfig {
            alpha: 1.0,
            eps: 0.05,
            max_outer: 100,
            inner_max_iter: 300,
            tol: 1e-5,
        };
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat_s, &feat_t, &cfg).unwrap();
        for &p in &res.plan {
            assert!(p >= -1e-8, "negative plan entry {p}");
        }
        let total: f64 = res.plan.iter().sum();
        assert!(approx(total, 1.0, 0.05), "total mass {total}");
    }

    // ─── Plan is non-negative and sums to 1 ──────────────────────────────

    #[test]
    fn plan_is_nonneg_and_sums_to_one() {
        let n = 4;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        let c_t = sym_cost(n, |i, j| (i as f64 - j as f64).abs() * 2.0);
        let feat_s: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, 0.0]).collect();
        let feat_t: Vec<Vec<f64>> = (0..n).map(|i| vec![0.0, i as f64]).collect();
        let cfg = default_config();
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat_s, &feat_t, &cfg).unwrap();
        for &p in &res.plan {
            assert!(p >= -1e-8);
        }
        let total: f64 = res.plan.iter().sum();
        assert!(approx(total, 1.0, 0.05), "total mass {total}");
    }

    // ─── Identical source and target → small cost ─────────────────────────

    #[test]
    fn identical_graphs_small_cost() {
        let n = 3;
        let a = uniform(n);
        let b = uniform(n);
        let c = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        let feat: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = GwGraphConfig {
            alpha: 0.5,
            eps: 0.05,
            max_outer: 150,
            inner_max_iter: 300,
            tol: 1e-6,
        };
        let res = gw_graph_matching(&a, &b, &c, &c, &feat, &feat, &cfg).expect("should converge");
        assert!(
            res.cost < 1.0,
            "cost {} should be small for identical graphs",
            res.cost
        );
    }

    // ─── Empty feat_s → error ─────────────────────────────────────────────

    #[test]
    fn empty_feat_s_returns_error() {
        let n = 3;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        let c_t = c_s.clone();
        let feat_t: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = default_config();
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &[], &feat_t, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    // ─── alpha out of [0,1] → error ──────────────────────────────────────

    #[test]
    fn invalid_alpha_returns_error() {
        let n = 2;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| if i == j { 0.0 } else { 1.0 });
        let c_t = c_s.clone();
        let feat: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = GwGraphConfig {
            alpha: 1.5,
            ..default_config()
        };
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat, &feat, &cfg);
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    // ─── gw_frobenius_cost is non-negative ───────────────────────────────

    #[test]
    fn gw_frobenius_cost_is_nonnegative() {
        let n = 3;
        let c = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        // Uniform plan.
        let t: Vec<f64> = vec![1.0 / (n * n) as f64; n * n];
        let val = gw_frobenius_cost(&c, &c, &t);
        assert!(val >= -1e-10, "frobenius cost negative: {val}");
    }

    // ─── Cost consistency: (1-α)*feat_cost + α*gw_cost ≈ cost ───────────

    #[test]
    fn cost_is_consistent_with_components() {
        let n = 3;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        let c_t = c_s.clone();
        let feat: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = default_config();
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat, &feat, &cfg).unwrap();
        let reconstructed = (1.0 - cfg.alpha) * res.feat_cost + cfg.alpha * res.gw_cost;
        assert!(
            approx(res.cost, reconstructed, 1e-8),
            "cost {} != reconstructed {}",
            res.cost,
            reconstructed
        );
    }

    // ─── Different feature spaces matched by structure ────────────────────

    #[test]
    fn different_features_different_structure_runs() {
        // Source: path graph 0-1-2 with identity features
        // Target: complete graph 3 nodes with different features
        let ns = 3;
        let nt = 3;
        let a = uniform(ns);
        let b = uniform(nt);
        let c_s = sym_cost(ns, |i, j| {
            if i == j {
                0.0
            } else if (i as i64 - j as i64).abs() == 1 {
                1.0
            } else {
                2.0
            }
        });
        let c_t = sym_cost(nt, |i, j| if i == j { 0.0 } else { 1.5 });
        let feat_s: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let feat_t: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.0, 2.0], vec![0.0, 3.0]];
        let cfg = default_config();
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat_s, &feat_t, &cfg).unwrap();
        assert!(res.cost.is_finite());
        assert!(res.iters >= 1);
    }

    // ─── eps ≤ 0 → error ─────────────────────────────────────────────────

    #[test]
    fn invalid_eps_returns_error() {
        let n = 2;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| if i == j { 0.0 } else { 1.0 });
        let c_t = c_s.clone();
        let feat: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = GwGraphConfig {
            eps: -0.1,
            ..default_config()
        };
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat, &feat, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    // ─── Iters recorded correctly ─────────────────────────────────────────

    #[test]
    fn iters_at_least_one() {
        let n = 2;
        let a = uniform(n);
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| if i == j { 0.0 } else { 1.0 });
        let feat: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = default_config();
        let res = gw_graph_matching(&a, &b, &c_s, &c_s, &feat, &feat, &cfg).unwrap();
        assert!(res.iters >= 1);
    }

    // ─── Negative weight → error ──────────────────────────────────────────

    #[test]
    fn negative_weight_returns_error() {
        let n = 2;
        let b = uniform(n);
        let c_s = sym_cost(n, |i, j| if i == j { 0.0 } else { 1.0 });
        let feat: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let cfg = default_config();
        // a has a negative weight.
        let a = vec![-0.5, 1.5];
        let res = gw_graph_matching(&a, &b, &c_s, &c_s, &feat, &feat, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    // ─── gw_frobenius_cost zero plan → 0 ─────────────────────────────────

    #[test]
    fn frobenius_cost_zero_plan() {
        let n = 3;
        let c = sym_cost(n, |i, j| (i as f64 - j as f64).abs());
        let t = vec![0.0_f64; n * n];
        let val = gw_frobenius_cost(&c, &c, &t);
        assert!(approx(val, 0.0, 1e-10));
    }

    // ─── Symmetric feature cost matrix ───────────────────────────────────

    #[test]
    fn feature_cost_symmetric_for_same_spaces() {
        // When feat_s == feat_t (1-D, integer positions), the cost is symmetric.
        let feat: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let c = feature_cost_matrix(&feat, &feat);
        let n = feat.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx(c[i * n + j], c[j * n + i], 1e-12),
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    // ─── Marginal mismatch → error ────────────────────────────────────────

    #[test]
    fn c_s_wrong_size_returns_error() {
        let ns = 3;
        let nt = 2;
        let a = uniform(ns);
        let b = uniform(nt);
        // c_s only has 2 rows instead of 3.
        let c_s = sym_cost(2, |i, j| if i == j { 0.0 } else { 1.0 });
        let c_t = sym_cost(nt, |i, j| if i == j { 0.0 } else { 1.0 });
        let feat_s: Vec<Vec<f64>> = (0..ns).map(|i| vec![i as f64]).collect();
        let feat_t: Vec<Vec<f64>> = (0..nt).map(|i| vec![i as f64]).collect();
        let cfg = default_config();
        let res = gw_graph_matching(&a, &b, &c_s, &c_t, &feat_s, &feat_t, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }
}
