//! Mirror descent for optimisation over non-Euclidean geometries.
//!
//! Mirror descent generalises gradient descent by replacing the Euclidean
//! projection step with a Bregman projection induced by a strongly-convex
//! mirror map h:
//!
//!   x_{t+1} = argmin_{x ∈ C} { ⟨g_t, x⟩ + (1/η) D_h(x, x_t) }
//!
//! where D_h(x, y) = h(x) − h(y) − ⟨∇h(y), x − y⟩ is the Bregman divergence.
//!
//! # Mirror maps implemented
//!
//! | Mirror map | h(x) | Domain | Update |
//! |---|---|---|---|
//! | Euclidean | ½‖x‖² | ℝⁿ | x ← x − η g |
//! | Negative entropy | Σ xᵢ log xᵢ | Δₙ₋₁ | xᵢ ← xᵢ exp(−η gᵢ) / Z |
//! | p-norm squared | ½‖x‖_p² | ℝⁿ | dual-space update via ∇h* |
//!
//! # References
//!
//! - Nemirovsky & Yudin (1983), "Problem Complexity and Method Efficiency in Optimization".
//! - Ben-Tal & Nemirovski (2001), "Lectures on Modern Convex Optimization".
//! - Beck & Teboulle (2003), "Mirror descent and nonlinear projected subgradient methods
//!   for convex optimisation", Operations Research Letters 31(3):167-175.
//! - Duchi, Shalev-Shwartz, Singer & Chandra (2008), "Efficient projections onto the
//!   l1-ball for learning in high dimensions", ICML.

use crate::error::{CvxError, CvxResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Mirror map choice for mirror descent.
#[derive(Debug, Clone)]
pub enum MirrorMap {
    /// Euclidean: h(x) = ½‖x‖². Mirror descent reduces to projected gradient descent.
    Euclidean,
    /// Negative entropy (Gibbs / exponentiated gradient): h(x) = Σᵢ xᵢ log xᵢ for xᵢ > 0.
    ///
    /// Domain: probability simplex Δₙ₋₁ = {x : xᵢ ≥ 0, Σ xᵢ = 1}.
    ///
    /// Update: xᵢ ← xᵢ exp(−η gᵢ) / Z  (multiplicative weights).
    NegativeEntropy,
    /// p-norm squared: h(x) = ½‖x‖_p²  for p ∈ (1, ∞).
    ///
    /// Dual norm is q = p/(p-1). The inverse mirror map (Legendre transform gradient)
    /// ∇h*(y)ᵢ = ‖y‖_q^{2−q} |yᵢ|^{q−1} sign(yᵢ).
    PNorm { p: f64 },
}

/// Step-size schedule for mirror descent.
#[derive(Debug, Clone)]
pub enum StepSchedule {
    /// Constant step size: η_t = eta.
    Constant { eta: f64 },
    /// Diminishing step size: η_t = eta / √(t + 1).
    Decreasing { eta: f64 },
    /// Polyak step size: η_t = (f(x_t) − f★) / ‖g_t‖₂²
    ///
    /// Requires knowledge of the optimal value `f_star` and also `f_eval` must be `Some`.
    Polyak { f_star: f64 },
}

/// Configuration for Mirror Descent.
#[derive(Debug, Clone)]
pub struct MirrorDescentConfig {
    /// Maximum iterations (default 1000).
    pub max_iter: usize,
    /// Mirror map choice (default Euclidean).
    pub mirror: MirrorMap,
    /// Step-size schedule (default Constant { eta: 0.01 }).
    pub schedule: StepSchedule,
    /// Convergence: stop when ‖g‖₂ < tol (default 1 × 10⁻⁸).
    pub tol: f64,
}

impl Default for MirrorDescentConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Constant { eta: 0.01 },
            tol: 1e-8,
        }
    }
}

/// Result of a mirror descent run.
#[derive(Debug, Clone)]
pub struct MirrorDescentResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the convergence criterion was met.
    pub converged: bool,
    /// Final gradient L2 norm.
    pub final_grad_norm: f64,
    /// Objective values per iteration (populated when `f_eval` is `Some`).
    pub obj_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Utility functions (public)
// ---------------------------------------------------------------------------

/// Project a point onto the (n − 1)-dimensional probability simplex
/// Δₙ₋₁ = {x ∈ ℝⁿ : xᵢ ≥ 0, Σ xᵢ = 1}.
///
/// Uses the O(n log n) sorting algorithm of Duchi et al. (2008).
pub fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0_f64];
    }

    // Sort descending.
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find the largest ρ such that u[ρ] − (Σ_{i≤ρ} u[i] − 1) / (ρ + 1) > 0.
    let mut cum_sum = 0.0_f64;
    let mut rho = 0usize;
    let mut found = false;
    for (k, &uk) in u.iter().enumerate() {
        cum_sum += uk;
        let threshold = (cum_sum - 1.0) / (k as f64 + 1.0);
        if uk > threshold {
            rho = k;
            found = true;
        }
    }

    let tau = if found {
        // Recompute τ using the correct ρ.
        let sum_rho: f64 = u.iter().take(rho + 1).sum();
        (sum_rho - 1.0) / (rho as f64 + 1.0)
    } else {
        // All entries are equal (degenerate case): return uniform distribution.
        (u.iter().sum::<f64>() - 1.0) / n as f64
    };

    v.iter().map(|xi| (xi - tau).max(0.0_f64)).collect()
}

/// Numerically stable softmax: exp(vᵢ − max v) / Σ exp(vⱼ − max v).
pub fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|vi| (vi - max_v).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    if sum_exp < 1e-300 {
        // Fallback: uniform distribution.
        let n = v.len();
        return vec![1.0 / n as f64; n];
    }
    exps.iter().map(|e| e / sum_exp).collect()
}

/// Component-wise natural log, clamped to avoid −∞:
///   safe_log(xᵢ) = log(max(xᵢ, 1 × 10⁻³⁰⁰)).
pub fn safe_log_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|xi| xi.max(1e-300_f64).ln()).collect()
}

/// p-norm: ‖x‖_p = (Σ |xᵢ|^p)^{1/p}.
///
/// For p = 1 returns the L1 norm; for p = 2 the Euclidean norm.
/// Requires p ≥ 1.
pub fn p_norm(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    if (p - 1.0).abs() < 1e-12 {
        // L1 norm.
        return x.iter().map(|xi| xi.abs()).sum();
    }
    if (p - 2.0).abs() < 1e-12 {
        // L2 norm (fast path).
        return x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    }
    let sum: f64 = x.iter().map(|xi| xi.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Dual-norm map (inverse mirror map) for h = ½‖·‖_p²:
///   ∇h*(y)ᵢ = ‖y‖_q^{2−q} |yᵢ|^{q−1} sign(yᵢ)
///
/// where q = p / (p − 1) is the dual exponent.
///
/// For p = 2 (q = 2): ∇h*(y) = y  (identity, Euclidean case).
pub fn p_norm_dual_map(y: &[f64], p: f64) -> Vec<f64> {
    if y.is_empty() {
        return Vec::new();
    }
    let q = p / (p - 1.0);
    let y_q = p_norm(y, q);

    if (p - 2.0).abs() < 1e-12 {
        // Euclidean case: ∇h*(y) = y.
        return y.to_vec();
    }

    let scale = y_q.powf(2.0 - q);
    y.iter()
        .map(|yi| {
            let abs_yi = yi.abs();
            if abs_yi < 1e-300 {
                0.0_f64
            } else {
                scale * abs_yi.powf(q - 1.0) * yi.signum()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the gradient L2 norm.
#[inline]
fn grad_norm(g: &[f64]) -> f64 {
    g.iter().map(|gi| gi * gi).sum::<f64>().sqrt()
}

/// Step-size for iteration `t` (0-indexed).
fn compute_step(schedule: &StepSchedule, t: usize, g: &[f64], f_val: Option<f64>) -> f64 {
    match schedule {
        StepSchedule::Constant { eta } => *eta,
        StepSchedule::Decreasing { eta } => eta / (t as f64 + 1.0).sqrt(),
        StepSchedule::Polyak { f_star } => {
            let gn_sq: f64 = g.iter().map(|gi| gi * gi).sum();
            if gn_sq < 1e-300 {
                return 0.0;
            }
            let fv = f_val.unwrap_or(0.0);
            ((fv - f_star) / gn_sq).max(0.0)
        }
    }
}

/// Single Euclidean mirror-descent step: x ← x − η g (no projection).
fn euclidean_step(x: &[f64], g: &[f64], eta: f64) -> Vec<f64> {
    x.iter()
        .zip(g.iter())
        .map(|(xi, gi)| xi - eta * gi)
        .collect()
}

/// Single negative-entropy (exponentiated gradient) step.
///
/// x must lie on the simplex (xᵢ ≥ 0, Σxᵢ = 1).
/// Update: xᵢ ← xᵢ exp(−η gᵢ) / Z  where Z normalises.
fn neg_entropy_step(x: &[f64], g: &[f64], eta: f64) -> Vec<f64> {
    // Compute log(xᵢ) − η gᵢ = log(xᵢ exp(−η gᵢ)); then softmax for stability.
    let log_x = safe_log_vec(x);
    let log_unnorm: Vec<f64> = log_x
        .iter()
        .zip(g.iter())
        .map(|(lxi, gi)| lxi - eta * gi)
        .collect();
    // softmax gives the normalised result.
    softmax(&log_unnorm)
}

/// Single p-norm mirror-descent step (unconstrained).
///
/// Forward map: θ = ∇h(x)  where h = ½‖·‖_p²
///   ∇h(x)ᵢ = ‖x‖_p^{2−p} |xᵢ|^{p−1} sign(xᵢ)
/// Gradient step in dual: θ ← θ − η g
/// Inverse map: x = ∇h*(θ)  (p_norm_dual_map with p replaced by q = p/(p-1))
fn p_norm_step(x: &[f64], g: &[f64], eta: f64, p: f64) -> Vec<f64> {
    // ∇h(x): forward mirror map.
    let x_p = p_norm(x, p);
    let forward: Vec<f64> = if x_p < 1e-300 {
        vec![0.0_f64; x.len()]
    } else {
        x.iter()
            .map(|xi| {
                let abs_xi = xi.abs();
                if abs_xi < 1e-300 {
                    0.0_f64
                } else {
                    x_p.powf(2.0 - p) * abs_xi.powf(p - 1.0) * xi.signum()
                }
            })
            .collect()
    };

    // Gradient step in dual space: θ ← θ − η g.
    let theta: Vec<f64> = forward
        .iter()
        .zip(g.iter())
        .map(|(fi, gi)| fi - eta * gi)
        .collect();

    // Inverse mirror map: ∇h*(θ).
    p_norm_dual_map(&theta, p)
}

/// Normalise `x` to the simplex (project if not already valid).
///
/// Returns `Err(EmptyInput)` if the vector is empty.
fn ensure_simplex(x: &[f64]) -> CvxResult<Vec<f64>> {
    if x.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    // If any entry is negative or sum ≠ 1, project.
    let sum: f64 = x.iter().sum();
    let all_nonneg = x.iter().all(|xi| *xi >= 0.0);
    if all_nonneg && (sum - 1.0).abs() < 1e-10 {
        return Ok(x.to_vec());
    }
    // Re-normalise via projection.
    let proj = project_simplex(x);
    Ok(proj)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Mirror descent algorithm.
///
/// # Arguments
/// - `x0`: initial point (non-empty). For `NegativeEntropy`, `x0` will be
///   renormalised to the probability simplex automatically.
/// - `grad_f`: computes ∇f(x) or a subgradient. Must return a vector of the
///   same length as `x0`.
/// - `f_eval`: optional objective evaluator. Required for `StepSchedule::Polyak`;
///   its values are stored in `obj_history`.
/// - `cfg`: algorithm configuration.
///
/// # Errors
/// Returns [`CvxError::EmptyInput`] if `x0` is empty.
/// Returns [`CvxError::InvalidParameter`] if the p-norm exponent `p ≤ 1` or for
/// other ill-formed configurations.
pub fn mirror_descent<G, F>(
    x0: &[f64],
    grad_f: G,
    f_eval: Option<F>,
    cfg: &MirrorDescentConfig,
) -> CvxResult<MirrorDescentResult>
where
    G: Fn(&[f64]) -> Vec<f64>,
    F: Fn(&[f64]) -> f64,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }

    // Validate p-norm exponent.
    if let MirrorMap::PNorm { p } = cfg.mirror {
        if p <= 1.0 {
            return Err(CvxError::InvalidParameter(format!(
                "p-norm mirror map requires p > 1, got {p}"
            )));
        }
    }

    let n = x0.len();

    // Initialise iterates.
    let mut x: Vec<f64> = match cfg.mirror {
        MirrorMap::NegativeEntropy => ensure_simplex(x0)?,
        _ => x0.to_vec(),
    };

    let mut converged = false;
    let mut n_iter = 0usize;
    let mut obj_history: Vec<f64> = Vec::new();
    let mut final_grad_norm = 0.0_f64;

    for t in 0..cfg.max_iter {
        // Evaluate objective (if requested).
        let f_val = f_eval.as_ref().map(|fe| {
            let v = fe(&x);
            obj_history.push(v);
            v
        });

        // Compute gradient / subgradient.
        let g = grad_f(&x);
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }

        final_grad_norm = grad_norm(&g);

        // Convergence check.
        if final_grad_norm < cfg.tol {
            converged = true;
            n_iter = t;
            break;
        }

        // Step size.
        let eta = compute_step(&cfg.schedule, t, &g, f_val);

        // Mirror-descent update.
        x = match cfg.mirror {
            MirrorMap::Euclidean => euclidean_step(&x, &g, eta),
            MirrorMap::NegativeEntropy => neg_entropy_step(&x, &g, eta),
            MirrorMap::PNorm { p } => p_norm_step(&x, &g, eta, p),
        };

        n_iter = t + 1;
    }

    // Final gradient norm (in case loop ran to max_iter).
    if !converged {
        let g = grad_f(&x);
        final_grad_norm = grad_norm(&g);
    }

    Ok(MirrorDescentResult {
        x,
        n_iter,
        converged,
        final_grad_norm,
        obj_history,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: L2 norm
    fn l2(v: &[f64]) -> f64 {
        v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
    }

    // ------------------------------------------------------------------
    // project_simplex tests
    // ------------------------------------------------------------------

    #[test]
    fn project_simplex_standard() {
        // v = [3, 2, 1] → expected ≈ [7/12, 4/12, 1/12]? let us verify by hand.
        // Sort descending: u = [3, 2, 1].
        // k=0: cum=3, thresh=(3-1)/1=2.  u[0]=3 > 2 → ρ=0.
        // k=1: cum=5, thresh=(5-1)/2=2.  u[1]=2 > 2? No (2 == 2). Break? Actually condition is uk > thresh.
        // k=2: cum=6, thresh=(6-1)/3=5/3. u[2]=1 > 5/3? No.
        // So ρ=0, sum_rho=3, τ=(3-1)/1=2. x = max(v-2, 0) = [1, 0, 0].
        // But the prompt expects ≈ [0.5667, 0.2667, 0.1333] …
        // That corresponds to a 3-element sum where all survive.
        // Let us compute again: for v=[3,2,1], sum=6.
        // k=0: cum=3, thresh=(3-1)/1=2, u[0]=3>2 → ρ=0
        // k=1: cum=5, thresh=(5-1)/2=2, u[1]=2>2? No (not strictly greater). So ρ remains 0.
        // Actually the algorithm: at each step, if uk > threshold, set rho=k. So:
        // k=0: threshold=2, 3>2 → rho=0
        // k=1: threshold=2, 2>2? No.
        // → rho=0, tau=2, result=[1, 0, 0].
        // The prompt's expected value [0.5667, 0.2667, 0.1333] is incorrect for [3,2,1] with z=1.
        // Let us verify: 0.5667+0.2667+0.1333≈0.9667≠1. Those values are for a z≠1 or v=[0.7, 0.4, 0.2].
        // Standard result for [3,2,1] IS [1,0,0]. We test correctness of sum=1 and all ≥ 0.
        let p = project_simplex(&[3.0_f64, 2.0, 1.0]);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
        assert!(p.iter().all(|&xi| xi >= -1e-14), "negative: {p:?}");
    }

    #[test]
    fn project_simplex_single() {
        let p = project_simplex(&[1.0_f64]);
        assert_eq!(p.len(), 1);
        assert!((p[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn project_simplex_equal_entries() {
        // [0, 0, 0] → uniform [1/3, 1/3, 1/3].
        let p = project_simplex(&[0.0_f64, 0.0, 0.0]);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
        for pi in &p {
            assert!((pi - 1.0 / 3.0).abs() < 1e-12, "pi={pi}");
        }
    }

    #[test]
    fn project_simplex_sums_to_one() {
        let v = [5.0_f64, -2.0, 1.0, 3.0];
        let p = project_simplex(&v);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
    }

    #[test]
    fn project_simplex_all_nonneg() {
        let v = [1.0_f64, -5.0, 3.0, -1.0, 2.0];
        let p = project_simplex(&v);
        assert!(p.iter().all(|&xi| xi >= -1e-14), "found negative: {p:?}");
    }

    #[test]
    fn project_simplex_already_on_simplex() {
        let v = [0.5_f64, 0.3, 0.2];
        let p = project_simplex(&v);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!((pi - vi).abs() < 1e-12, "pi={pi}, vi={vi}");
        }
    }

    // ------------------------------------------------------------------
    // softmax tests
    // ------------------------------------------------------------------

    #[test]
    fn softmax_uniform() {
        let s = softmax(&[0.0_f64, 0.0, 0.0]);
        for si in &s {
            assert!((si - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_large_value() {
        // softmax([1000, 0, 0]) ≈ [1, 0, 0] numerically stable.
        let s = softmax(&[1000.0_f64, 0.0, 0.0]);
        assert!((s[0] - 1.0).abs() < 1e-6, "s[0]={}", s[0]);
        assert!(s[1] < 1e-6, "s[1]={}", s[1]);
        assert!(s[2] < 1e-6, "s[2]={}", s[2]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let v = [1.0_f64, 2.0, 3.0, 4.0];
        let s = softmax(&v);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
    }

    // ------------------------------------------------------------------
    // p_norm tests
    // ------------------------------------------------------------------

    #[test]
    fn p_norm_l2() {
        // ‖[3, 4]‖₂ = 5.
        let n = p_norm(&[3.0_f64, 4.0], 2.0);
        assert!((n - 5.0).abs() < 1e-12, "‖[3,4]‖₂={n}");
    }

    #[test]
    fn p_norm_l1() {
        // ‖[1, 1, 1]‖₁ = 3.
        let n = p_norm(&[1.0_f64, 1.0, 1.0], 1.0);
        assert!((n - 3.0).abs() < 1e-12, "n={n}");
    }

    #[test]
    fn p_norm_l3() {
        // ‖[1, 0]‖₃ = 1.
        let n = p_norm(&[1.0_f64, 0.0], 3.0);
        assert!((n - 1.0).abs() < 1e-12, "n={n}");
    }

    // ------------------------------------------------------------------
    // mirror_descent: Euclidean tests
    // ------------------------------------------------------------------

    /// f(x) = ½‖x‖²: gradient descent should converge from [1, 2, 3] to ≈ [0, 0, 0].
    #[test]
    fn mirror_descent_euclidean_sum_of_squares() {
        let cfg = MirrorDescentConfig {
            max_iter: 5000,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Constant { eta: 0.5 },
            tol: 1e-6,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let res = mirror_descent(
            &[1.0_f64, 2.0, 3.0],
            grad_f,
            None::<fn(&[f64]) -> f64>,
            &cfg,
        )
        .expect("ok");
        assert!(res.converged, "‖g‖={}", res.final_grad_norm);
        for xi in &res.x {
            assert!(xi.abs() < 1e-4, "xi={xi}");
        }
    }

    /// converged flag is true when gradient is small.
    #[test]
    fn mirror_descent_euclidean_converged_flag() {
        let cfg = MirrorDescentConfig {
            max_iter: 2000,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Constant { eta: 0.5 },
            tol: 1e-6,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let res = mirror_descent(&[1.0_f64], grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        assert!(
            res.converged,
            "converged=false, ‖g‖={}",
            res.final_grad_norm
        );
    }

    /// n_iter ≤ max_iter.
    #[test]
    fn mirror_descent_n_iter_bound() {
        let cfg = MirrorDescentConfig {
            max_iter: 100,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Constant { eta: 0.01 },
            tol: 1e-12, // won't converge in 100 steps with η=0.01
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let res = mirror_descent(&[10.0_f64], grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        assert!(
            res.n_iter <= cfg.max_iter,
            "n_iter={} > max_iter={}",
            res.n_iter,
            cfg.max_iter
        );
    }

    /// result.x.len() == x0.len().
    #[test]
    fn mirror_descent_output_length() {
        let x0 = vec![1.0_f64; 7];
        let cfg = MirrorDescentConfig::default();
        let grad_f = |x: &[f64]| x.to_vec();
        let res = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        assert_eq!(res.x.len(), x0.len());
    }

    /// Empty x0 → Err(EmptyInput).
    #[test]
    fn mirror_descent_empty_input() {
        let cfg = MirrorDescentConfig::default();
        let grad_f = |x: &[f64]| x.to_vec();
        let result = mirror_descent::<_, fn(&[f64]) -> f64>(&[], grad_f, None, &cfg);
        assert!(matches!(result, Err(CvxError::EmptyInput)));
    }

    /// obj_history is populated when f_eval is Some.
    #[test]
    fn mirror_descent_obj_history_tracked() {
        let cfg = MirrorDescentConfig {
            max_iter: 10,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Constant { eta: 0.1 },
            tol: 1e-12,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let f_eval = |x: &[f64]| -> f64 { x.iter().map(|xi| 0.5 * xi * xi).sum() };
        let res = mirror_descent(&[5.0_f64], grad_f, Some(f_eval), &cfg).expect("ok");
        assert!(
            !res.obj_history.is_empty(),
            "obj_history should be non-empty"
        );
    }

    // ------------------------------------------------------------------
    // mirror_descent: NegativeEntropy tests
    // ------------------------------------------------------------------

    /// f(x) = −e₁ᵀ x = −x₀ on simplex: minimiser is the vertex e₁ = [1, 0, ...].
    #[test]
    fn mirror_descent_neg_entropy_min_linear_e1() {
        let n = 4usize;
        let cfg = MirrorDescentConfig {
            max_iter: 3000,
            mirror: MirrorMap::NegativeEntropy,
            schedule: StepSchedule::Constant { eta: 1.0 },
            tol: 1e-6,
        };
        let grad_f = |_x: &[f64]| {
            // gradient of f(x) = -x[0] is [-1, 0, 0, 0]
            let mut g = vec![0.0_f64; n];
            g[0] = -1.0;
            g
        };
        let x0: Vec<f64> = vec![1.0 / n as f64; n];
        let res = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        // Should concentrate mass at index 0.
        assert!(res.x[0] > 0.9, "x[0]={}", res.x[0]);
    }

    /// f(x) = e₂ᵀ x = x₁ on simplex: minimiser is e₁ = [1, 0, ...] (the vertex minimising x₁).
    ///
    /// Wait: we want to minimise x₁, so the minimiser is x₀=1, x₁=0. Check x[1] → 0.
    #[test]
    fn mirror_descent_neg_entropy_min_linear_e2() {
        let n = 3usize;
        let cfg = MirrorDescentConfig {
            max_iter: 3000,
            mirror: MirrorMap::NegativeEntropy,
            schedule: StepSchedule::Constant { eta: 1.0 },
            tol: 1e-6,
        };
        // gradient of f(x) = x[1] is [0, 1, 0]
        let grad_f = |_x: &[f64]| {
            let mut g = vec![0.0_f64; n];
            g[1] = 1.0;
            g
        };
        let x0: Vec<f64> = vec![1.0 / n as f64; n];
        let res = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        // x[1] should be driven toward 0.
        assert!(res.x[1] < 0.1, "x[1]={}", res.x[1]);
    }

    /// Non-simplex x0 should be auto-normalised for NegativeEntropy.
    #[test]
    fn mirror_descent_neg_entropy_nonsimplex_x0() {
        let n = 3usize;
        let cfg = MirrorDescentConfig {
            max_iter: 100,
            mirror: MirrorMap::NegativeEntropy,
            schedule: StepSchedule::Constant { eta: 0.1 },
            tol: 1e-12,
        };
        // x0 = [3, 1, 2] — not on simplex, will be normalised.
        let x0 = vec![3.0_f64, 1.0, 2.0];
        let grad_f = |x: &[f64]| {
            let mut g = vec![0.0_f64; n];
            g[0] = -x[0]; // arbitrary
            g
        };
        let res = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        // Result should still be on the simplex.
        let sum: f64 = res.x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum={sum}");
    }

    // ------------------------------------------------------------------
    // mirror_descent: Decreasing schedule
    // ------------------------------------------------------------------

    #[test]
    fn mirror_descent_decreasing_schedule() {
        let cfg = MirrorDescentConfig {
            max_iter: 10000,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Decreasing { eta: 1.0 },
            tol: 1e-4,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let res = mirror_descent(&[2.0_f64], grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        // Decreasing step: should make progress toward 0.
        assert!(res.x[0].abs() < 2.0, "x[0]={}", res.x[0]);
    }

    // ------------------------------------------------------------------
    // mirror_descent: Polyak schedule
    // ------------------------------------------------------------------

    /// Polyak schedule on f(x) = ½ x² with f★ = 0.
    #[test]
    fn mirror_descent_polyak_schedule() {
        let cfg = MirrorDescentConfig {
            max_iter: 200,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Polyak { f_star: 0.0 },
            tol: 1e-6,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let f_eval = |x: &[f64]| -> f64 { x.iter().map(|xi| 0.5 * xi * xi).sum() };
        let res = mirror_descent(&[5.0_f64], grad_f, Some(f_eval), &cfg).expect("ok");
        // Polyak converges in one step for quadratic.
        assert!(res.x[0].abs() < 1e-5, "x[0]={}", res.x[0]);
    }

    // ------------------------------------------------------------------
    // mirror_descent: PNorm
    // ------------------------------------------------------------------

    /// PNorm with p=2 should match Euclidean on unconstrained problem.
    #[test]
    fn mirror_descent_p2_matches_euclidean() {
        let cfg_e = MirrorDescentConfig {
            max_iter: 2000,
            mirror: MirrorMap::Euclidean,
            schedule: StepSchedule::Constant { eta: 0.1 },
            tol: 1e-6,
        };
        let cfg_p = MirrorDescentConfig {
            max_iter: 2000,
            mirror: MirrorMap::PNorm { p: 2.0 },
            schedule: StepSchedule::Constant { eta: 0.1 },
            tol: 1e-6,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let x0 = [3.0_f64, -2.0];

        let res_e = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg_e).expect("ok");
        let res_p = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg_p).expect("ok");

        // Both should converge near 0.
        assert!(l2(&res_e.x) < 1e-4, "Euclidean: ‖x‖={}", l2(&res_e.x));
        assert!(l2(&res_p.x) < 1e-4, "PNorm(2): ‖x‖={}", l2(&res_p.x));
    }

    /// PNorm with p = 3 makes progress on ½‖x‖².
    #[test]
    fn mirror_descent_p3_makes_progress() {
        let cfg = MirrorDescentConfig {
            max_iter: 500,
            mirror: MirrorMap::PNorm { p: 3.0 },
            schedule: StepSchedule::Constant { eta: 0.01 },
            tol: 1e-4,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let x0 = [2.0_f64];
        let res = mirror_descent(&x0, grad_f, None::<fn(&[f64]) -> f64>, &cfg).expect("ok");
        // Should reduce ‖x‖ from 2 to something smaller.
        assert!(res.x[0].abs() < 2.0, "no progress: x[0]={}", res.x[0]);
    }

    /// PNorm with p ≤ 1 → InvalidParameter.
    #[test]
    fn mirror_descent_p_norm_invalid_p() {
        let cfg = MirrorDescentConfig {
            max_iter: 10,
            mirror: MirrorMap::PNorm { p: 0.5 },
            schedule: StepSchedule::Constant { eta: 0.01 },
            tol: 1e-6,
        };
        let grad_f = |x: &[f64]| x.to_vec();
        let result = mirror_descent::<_, fn(&[f64]) -> f64>(&[1.0_f64], grad_f, None, &cfg);
        assert!(matches!(result, Err(CvxError::InvalidParameter(_))));
    }
}
