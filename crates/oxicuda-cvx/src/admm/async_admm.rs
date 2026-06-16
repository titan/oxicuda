//! Asynchronous consensus ADMM (Zhang & Kwok 2014; Wei & Ozdaglar 2013).
//!
//! Solves the separable consensus problem
//!
//! ```text
//! min_{x_1,…,x_N, z}  Σ_i f_i(x_i)   s.t.   x_i = z   ∀ i
//! ```
//!
//! without a global synchronisation barrier.  At each global round only a random
//! **active subset** `A_k ⊆ {1,…,N}` of agents performs its local proximal `x`- and
//! dual `u`-update; the remaining agents keep their previous iterate.  The master
//! refreshes the consensus variable `z` from the agents' **most recently received**
//! copies, which may be *stale* — but the staleness of every agent is bounded by a
//! constant `τ = max_delay`, which is exactly the *bounded-delay* assumption under
//! which Zhang & Kwok prove convergence for separable convex problems.
//!
//! Because no genuine threads are spawned, this is a faithful single-process
//! *simulation* of the asynchronous protocol: it reproduces the bookkeeping
//! (per-agent local clocks, bounded staleness, partial participation) so that the
//! numerical trajectory matches a real asynchronous deployment up to the order in
//! which messages are interleaved.
//!
//! Scaled-form local updates (penalty `ρ`):
//! ```text
//! x_i^{k+1} = prox_{f_i / ρ}( z̄_i − u_i^k )       (z̄_i = agent i's cached consensus)
//! u_i^{k+1} = u_i^k + x_i^{k+1} − z̄_i
//! z^{k+1}   = (1/N) Σ_i ( x̃_i + ũ_i )             (x̃, ũ possibly stale, delay ≤ τ)
//! ```
//!
//! # References
//!
//! - R. Zhang & J. T. Kwok (2014), "Asynchronous Distributed ADMM for Consensus
//!   Optimization", ICML.
//! - E. Wei & A. Ozdaglar (2013), "On the O(1/k) convergence of asynchronous
//!   distributed ADMM", IEEE GlobalSIP.

use crate::error::{CvxError, CvxResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for asynchronous consensus ADMM.
#[derive(Debug, Clone)]
pub struct AsyncAdmmConfig {
    /// Augmented-Lagrangian penalty `ρ > 0`.
    pub rho: f64,
    /// Maximum number of global rounds.
    pub max_iter: usize,
    /// Stop when the primal residual `√(Σ_i ‖x_i − z‖²)` falls below `tol`.
    pub tol: f64,
    /// Fraction of agents that participate in each round, in `(0, 1]`.
    ///
    /// `1.0` recovers fully-synchronous consensus ADMM. Smaller values model heavier
    /// asynchrony (fewer agents reporting per round).
    pub active_fraction: f64,
    /// Maximum staleness `τ ≥ 0` (in rounds) of any agent's contribution to `z`.
    ///
    /// A bound of `0` means the master always uses fresh copies (synchronous limit).
    pub max_delay: usize,
    /// RNG seed governing which agents are active each round.
    pub seed: u64,
}

impl Default for AsyncAdmmConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iter: 1000,
            tol: 1e-6,
            active_fraction: 0.5,
            max_delay: 3,
            seed: 0xA5A5,
        }
    }
}

/// Result of asynchronous consensus ADMM.
#[derive(Debug, Clone)]
pub struct AsyncAdmmResult {
    /// Final consensus variable `z ∈ ℝ^dim`.
    pub z: Vec<f64>,
    /// Final per-agent primal iterates `[n_agents × dim]`.
    pub x: Vec<Vec<f64>>,
    /// Number of global rounds performed.
    pub iter: usize,
    /// Final primal residual `√(Σ_i ‖x_i − z‖²)`.
    pub residual: f64,
    /// Whether the convergence criterion fired.
    pub converged: bool,
    /// Total number of local agent updates performed across all rounds.
    pub total_updates: usize,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Solve `min Σ_i f_i(x_i)  s.t.  x_i = z` with the bounded-delay asynchronous protocol.
///
/// * `n_agents` — number of agents `N ≥ 1`.
/// * `dim` — dimension of each `x_i` and of `z`.
/// * `x_init` — initial `[n_agents × dim]` iterate.
/// * `x_update` — local proximal update closure `(agent_id, z_cached, u_i, ρ) → new x_i`
///   returning `prox_{f_{agent_id}/ρ}(z_cached − u_i)`.
/// * `cfg` — configuration.
///
/// # Errors
/// * [`CvxError::InvalidParameter`] for `n_agents == 0`, `dim == 0`, `ρ ≤ 0`, `tol ≤ 0`,
///   or `active_fraction ∉ (0, 1]`.
/// * [`CvxError::DimensionMismatch`] if `x_init` has the wrong shape or any update
///   closure returns a wrong-length vector.
pub fn async_consensus_admm(
    n_agents: usize,
    dim: usize,
    x_init: Vec<Vec<f64>>,
    x_update: impl Fn(usize, &[f64], &[f64], f64) -> Vec<f64>,
    cfg: &AsyncAdmmConfig,
) -> CvxResult<AsyncAdmmResult> {
    // ── Validation ──────────────────────────────────────────────────────────
    if n_agents == 0 {
        return Err(CvxError::InvalidParameter("n_agents must be ≥ 1".into()));
    }
    if dim == 0 {
        return Err(CvxError::InvalidParameter("dim must be ≥ 1".into()));
    }
    if cfg.rho <= 0.0 || !cfg.rho.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "async ADMM: rho > 0 required, got {}",
            cfg.rho
        )));
    }
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "async ADMM: tol > 0 required, got {}",
            cfg.tol
        )));
    }
    if !(cfg.active_fraction > 0.0 && cfg.active_fraction <= 1.0) {
        return Err(CvxError::InvalidParameter(format!(
            "async ADMM: active_fraction ∈ (0, 1] required, got {}",
            cfg.active_fraction
        )));
    }
    if x_init.len() != n_agents {
        return Err(CvxError::DimensionMismatch {
            a: x_init.len(),
            b: n_agents,
        });
    }
    for xi in &x_init {
        if xi.len() != dim {
            return Err(CvxError::DimensionMismatch {
                a: xi.len(),
                b: dim,
            });
        }
    }

    let n_f = n_agents as f64;

    // ── State ───────────────────────────────────────────────────────────────
    let mut xs = x_init;
    let mut us: Vec<Vec<f64>> = vec![vec![0.0_f64; dim]; n_agents];

    // Master consensus variable (current).
    let mut z = vec![0.0_f64; dim];
    for xi in &xs {
        for j in 0..dim {
            z[j] += xi[j];
        }
    }
    for v in &mut z {
        *v /= n_f;
    }

    // Each agent caches the consensus value it last *received*. With bounded delay
    // the master, when refreshing z, uses (x̃_i, ũ_i) snapshots whose age ≤ τ.
    let mut z_cached: Vec<Vec<f64>> = vec![z.clone(); n_agents];
    // Snapshot ring per agent: the (x_i + u_i) value as seen by the master, plus the
    // round it was produced. We keep the latest snapshot and its age.
    let mut snapshot: Vec<Vec<f64>> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        let mut s = vec![0.0_f64; dim];
        for j in 0..dim {
            s[j] = xs[i][j] + us[i][j];
        }
        snapshot.push(s);
    }
    let mut snapshot_age: Vec<usize> = vec![0usize; n_agents];

    let active_count = ((cfg.active_fraction * n_f).round() as usize).clamp(1, n_agents);
    let mut rng = LcgRng::new(cfg.seed);

    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut final_iter = 0usize;
    let mut total_updates = 0usize;

    for iter in 0..cfg.max_iter {
        final_iter = iter + 1;

        // ── Pick the active subset (partial Fisher–Yates) ────────────────────
        let active = sample_subset(n_agents, active_count, &mut rng);

        // ── Local updates for active agents ──────────────────────────────────
        for &i in &active {
            let x_new = x_update(i, &z_cached[i], &us[i], cfg.rho);
            if x_new.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: x_new.len(),
                    b: dim,
                });
            }
            // Dual update against the consensus the agent actually used.
            for j in 0..dim {
                us[i][j] += x_new[j] - z_cached[i][j];
            }
            xs[i] = x_new;
            total_updates += 1;

            // Freshly produced snapshot for the master.
            for j in 0..dim {
                snapshot[i][j] = xs[i][j] + us[i][j];
            }
            snapshot_age[i] = 0;
        }

        // ── Age inactive agents; force a refresh if staleness hits the bound ──
        for i in 0..n_agents {
            if !active.contains(&i) {
                snapshot_age[i] += 1;
                if snapshot_age[i] > cfg.max_delay {
                    // Bounded-delay guarantee: a too-stale agent must report now.
                    let x_new = x_update(i, &z_cached[i], &us[i], cfg.rho);
                    if x_new.len() != dim {
                        return Err(CvxError::DimensionMismatch {
                            a: x_new.len(),
                            b: dim,
                        });
                    }
                    for j in 0..dim {
                        us[i][j] += x_new[j] - z_cached[i][j];
                    }
                    xs[i] = x_new;
                    total_updates += 1;
                    for j in 0..dim {
                        snapshot[i][j] = xs[i][j] + us[i][j];
                    }
                    snapshot_age[i] = 0;
                }
            }
        }

        // ── Master refresh: z = mean over (possibly stale) snapshots ─────────
        let mut z_new = vec![0.0_f64; dim];
        for s in &snapshot {
            for j in 0..dim {
                z_new[j] += s[j];
            }
        }
        for v in &mut z_new {
            *v /= n_f;
        }
        z = z_new;

        // Broadcast z to active agents only (they refresh their cache).
        for &i in &active {
            z_cached[i].copy_from_slice(&z);
        }

        // ── Primal residual against the *current* z ──────────────────────────
        let mut r_sq = 0.0_f64;
        for xi in &xs {
            for j in 0..dim {
                let d = xi[j] - z[j];
                r_sq += d * d;
            }
        }
        residual = r_sq.sqrt();
        if residual.is_finite() && residual < cfg.tol {
            // Refresh all caches before declaring convergence so z is consistent.
            for cache in &mut z_cached {
                cache.copy_from_slice(&z);
            }
            converged = true;
            break;
        }
    }

    Ok(AsyncAdmmResult {
        z,
        x: xs,
        iter: final_iter,
        residual,
        converged,
        total_updates,
    })
}

/// Sample `k` distinct indices from `[0, n)` via partial Fisher–Yates.
fn sample_subset(n: usize, k: usize, rng: &mut LcgRng) -> Vec<usize> {
    let k = k.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local prox for `f_i(x) = (β_i / 2) ‖x − a_i‖²`:
    /// `x_i = prox_{f_i/ρ}(z_cached − u_i) = (ρ (z_cached − u_i) + β_i a_i) / (ρ + β_i)`.
    fn quad_update(
        a: &[Vec<f64>],
        beta: f64,
        i: usize,
        z_cached: &[f64],
        u: &[f64],
        rho: f64,
    ) -> Vec<f64> {
        z_cached
            .iter()
            .zip(u.iter())
            .zip(a[i].iter())
            .map(|((zi, ui), ai)| {
                let v = zi - ui;
                (rho * v + beta * ai) / (rho + beta)
            })
            .collect()
    }

    fn make_init(n: usize, dim: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0_f64; dim]; n]
    }

    #[test]
    fn averaging_consensus_two_agents() {
        // f_0 = ½‖x − 0‖², f_1 = ½‖x − 10‖². Consensus optimum = mean = 5.
        let a = vec![vec![0.0_f64], vec![10.0_f64]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            max_delay: 0,
            max_iter: 2000,
            tol: 1e-9,
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 1.0, i, v, u, rho);
        let res = async_consensus_admm(2, 1, make_init(2, 1), prox, &cfg).expect("ok");
        assert!((res.z[0] - 5.0).abs() < 1e-4, "z = {}", res.z[0]);
        assert!(res.converged);
    }

    #[test]
    fn weighted_consensus_recovers_weighted_mean() {
        // f_0 = ½·1·(x−0)², f_1 = ½·3·(x−4)². Optimum = (1·0 + 3·4)/(1+3) = 3.
        let a = vec![vec![0.0_f64], vec![4.0_f64]];
        let betas = [1.0_f64, 3.0];
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            max_delay: 0,
            max_iter: 4000,
            tol: 1e-10,
            ..Default::default()
        };
        let prox =
            |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, betas[i], i, v, u, rho);
        let res = async_consensus_admm(2, 1, make_init(2, 1), prox, &cfg).expect("ok");
        assert!((res.z[0] - 3.0).abs() < 1e-3, "z = {}", res.z[0]);
    }

    #[test]
    fn asynchronous_partial_participation_converges() {
        // With only half the agents active per round it should still converge to mean.
        let a = vec![vec![0.0_f64], vec![6.0_f64], vec![12.0_f64], vec![18.0_f64]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 0.5,
            max_delay: 2,
            max_iter: 5000,
            tol: 1e-6,
            seed: 7,
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 1.0, i, v, u, rho);
        let res = async_consensus_admm(4, 1, make_init(4, 1), prox, &cfg).expect("ok");
        assert!((res.z[0] - 9.0).abs() < 1e-2, "z = {}", res.z[0]); // mean(0,6,12,18)=9
    }

    #[test]
    fn multidim_consensus() {
        let a = vec![vec![1.0_f64, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            max_delay: 0,
            max_iter: 3000,
            tol: 1e-9,
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 1.0, i, v, u, rho);
        let res = async_consensus_admm(2, 3, make_init(2, 3), prox, &cfg).expect("ok");
        for (j, zj) in res.z.iter().enumerate() {
            assert!((zj - 2.0).abs() < 1e-3, "z[{j}] = {zj}");
        }
    }

    #[test]
    fn residual_decreases_overall() {
        let a = vec![vec![0.0_f64], vec![10.0_f64]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            max_delay: 0,
            max_iter: 500,
            tol: 1e-12, // never met: forces full run
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 1.0, i, v, u, rho);
        let res = async_consensus_admm(2, 1, make_init(2, 1), prox, &cfg).expect("ok");
        // After many iterations the residual should be tiny even if "converged" is false.
        assert!(res.residual < 1e-5, "residual = {}", res.residual);
    }

    #[test]
    fn bounded_delay_forces_refresh() {
        // With active_fraction tiny and small max_delay, agents must still update
        // due to the staleness cap, so total_updates exceeds active-only count.
        let a = vec![vec![0.0_f64], vec![8.0_f64], vec![16.0_f64]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 0.34, // ⇒ 1 active agent per round
            max_delay: 1,
            max_iter: 50,
            tol: 1e-12,
            seed: 3,
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 1.0, i, v, u, rho);
        let res = async_consensus_admm(3, 1, make_init(3, 1), prox, &cfg).expect("ok");
        // 1 active + forced refreshes ⇒ more than 50 updates over 50 rounds.
        assert!(res.total_updates > 50, "updates = {}", res.total_updates);
    }

    #[test]
    fn sync_limit_matches_plain_consensus() {
        // active_fraction = 1, max_delay = 0 is exactly synchronous consensus ADMM:
        // z should equal the mean of the two targets for unit weights.
        let a = vec![vec![-4.0_f64], vec![4.0_f64]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            max_delay: 0,
            max_iter: 3000,
            tol: 1e-10,
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 1.0, i, v, u, rho);
        let res = async_consensus_admm(2, 1, make_init(2, 1), prox, &cfg).expect("ok");
        assert!(res.z[0].abs() < 1e-3, "z = {}", res.z[0]);
    }

    #[test]
    fn single_agent_is_its_own_minimiser() {
        let a = vec![vec![7.0_f64, -2.0]];
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            max_delay: 0,
            max_iter: 1000,
            tol: 1e-9,
            ..Default::default()
        };
        let prox = |i: usize, v: &[f64], u: &[f64], rho: f64| quad_update(&a, 5.0, i, v, u, rho);
        let res = async_consensus_admm(1, 2, make_init(1, 2), prox, &cfg).expect("ok");
        assert!((res.z[0] - 7.0).abs() < 1e-3);
        assert!((res.z[1] + 2.0).abs() < 1e-3);
    }

    #[test]
    fn zero_agents_errors() {
        let prox = |_i: usize, v: &[f64], _u: &[f64], _r: f64| v.to_vec();
        let err = async_consensus_admm(0, 1, Vec::new(), prox, &AsyncAdmmConfig::default());
        assert!(matches!(err, Err(CvxError::InvalidParameter(_))));
    }

    #[test]
    fn bad_active_fraction_errors() {
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.5,
            ..Default::default()
        };
        let prox = |_i: usize, v: &[f64], _u: &[f64], _r: f64| v.to_vec();
        let err = async_consensus_admm(2, 1, make_init(2, 1), prox, &cfg);
        assert!(matches!(err, Err(CvxError::InvalidParameter(_))));
    }

    #[test]
    fn wrong_init_shape_errors() {
        let cfg = AsyncAdmmConfig::default();
        let prox = |_i: usize, v: &[f64], _u: &[f64], _r: f64| v.to_vec();
        let err = async_consensus_admm(3, 2, make_init(2, 2), prox, &cfg);
        assert!(matches!(err, Err(CvxError::DimensionMismatch { .. })));
    }

    #[test]
    fn dimension_mismatch_from_update_closure() {
        let cfg = AsyncAdmmConfig {
            active_fraction: 1.0,
            ..Default::default()
        };
        // Update returns wrong length.
        let prox = |_i: usize, _v: &[f64], _u: &[f64], _r: f64| vec![0.0_f64, 0.0];
        let err = async_consensus_admm(2, 1, make_init(2, 1), prox, &cfg);
        assert!(matches!(err, Err(CvxError::DimensionMismatch { .. })));
    }

    #[test]
    fn sample_subset_distinct_and_bounded() {
        let mut rng = LcgRng::new(123);
        for _ in 0..20 {
            let s = sample_subset(10, 4, &mut rng);
            assert_eq!(s.len(), 4);
            let mut sorted = s.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 4, "indices must be distinct");
            assert!(s.iter().all(|&i| i < 10));
        }
    }
}
