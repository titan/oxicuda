//! Consensus ADMM (Boyd 2011) — distributed problem `min Σ_i f_i(x_i)  s.t.  x_i = z`.
//!
//! Each agent i maintains a private variable `x_i ∈ ℝ^d` and a dual multiplier `u_i ∈ ℝ^d`.
//! The global consensus variable `z ∈ ℝ^d` is the average of all `x_i + u_i`.
//!
//! Algorithm (scaled-form ADMM):
//! ```text
//! x_i^{k+1} = prox_{f_i / ρ}(z^k - u_i^k)
//!           = x_update(i, x_i^k, z^k, u_i^k, ρ)
//! z^{k+1}   = (1/N) Σ_i (x_i^{k+1} + u_i^k)
//! u_i^{k+1} = u_i^k + x_i^{k+1} − z^{k+1}
//! ```
//!
//! Reference: Boyd et al. (2011) §7.1 — "Distributed ADMM / Consensus".

use crate::error::{CvxError, CvxResult};

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Configuration for consensus ADMM.
#[derive(Debug, Clone)]
pub struct ConsensusAdmmConfig {
    /// Augmented Lagrangian penalty parameter ρ > 0.
    pub rho: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance: stop when primal residual < `tol`.
    pub tol: f64,
}

impl Default for ConsensusAdmmConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iter: 500,
            tol: 1e-6,
        }
    }
}

/// Result of consensus ADMM.
#[derive(Debug, Clone)]
pub struct ConsensusAdmmResult {
    /// Consensus variable `z ∈ ℝ^dim` (common agreement among agents).
    pub z: Vec<f64>,
    /// Number of iterations performed.
    pub iter: usize,
    /// Final primal residual `√(Σ_i ‖x_i − z‖²)`.
    pub residual: f64,
    /// Whether convergence criterion was met.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

/// Solve the consensus problem `min Σ_i f_i(x_i)  s.t.  x_i = z  ∀ i`.
///
/// # Arguments
/// - `n_agents`: number of agents N ≥ 1.
/// - `dim`: dimension of each `x_i` and `z` (d ≥ 1).
/// - `x_init`: initial `[n_agents × dim]` iterate (row i is agent i's starting point).
/// - `x_update`: proximal update closure `(agent_id, x_i, z, u_i, rho) → new x_i`.
///   Semantics: return `prox_{f_{agent_id} / rho}(z − u_i)`.
/// - `cfg`: algorithm configuration.
///
/// # Returns
/// [`ConsensusAdmmResult`] with the consensus variable, iteration count, residual, and
/// convergence flag.
///
/// # Errors
/// - [`CvxError::InvalidParameter`] if `n_agents == 0`, `dim == 0`, `rho ≤ 0`, or `tol ≤ 0`.
/// - [`CvxError::DimensionMismatch`] if `x_init.len() != n_agents` or any row has wrong length.
pub fn consensus_admm(
    n_agents: usize,
    dim: usize,
    x_init: Vec<Vec<f64>>,
    x_update: impl Fn(usize, &[f64], &[f64], &[f64], f64) -> Vec<f64>,
    cfg: &ConsensusAdmmConfig,
) -> CvxResult<ConsensusAdmmResult> {
    // --- Input validation ---
    if n_agents == 0 {
        return Err(CvxError::InvalidParameter("n_agents must be ≥ 1".into()));
    }
    if dim == 0 {
        return Err(CvxError::InvalidParameter("dim must be ≥ 1".into()));
    }
    if cfg.rho <= 0.0 || !cfg.rho.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "consensus ADMM: rho > 0 required, got {}",
            cfg.rho
        )));
    }
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "consensus ADMM: tol > 0 required, got {}",
            cfg.tol
        )));
    }
    if x_init.len() != n_agents {
        return Err(CvxError::DimensionMismatch {
            a: x_init.len(),
            b: n_agents,
        });
    }
    for (i, xi) in x_init.iter().enumerate() {
        if xi.len() != dim {
            return Err(CvxError::DimensionMismatch {
                a: xi.len(),
                b: dim,
            });
        }
        let _ = i;
    }

    // --- Initialise state ---
    let mut xs = x_init;
    let mut us: Vec<Vec<f64>> = vec![vec![0.0_f64; dim]; n_agents];

    // z = mean(xs[i]) as starting consensus point
    let mut z = vec![0.0_f64; dim];
    for xi in xs.iter() {
        for j in 0..dim {
            z[j] += xi[j];
        }
    }
    let n_f = n_agents as f64;
    for v in z.iter_mut() {
        *v /= n_f;
    }

    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut final_iter = 0usize;

    for iter in 0..cfg.max_iter {
        // ---- x-step: proximal update for each agent ----
        for i in 0..n_agents {
            let xi_new = x_update(i, &xs[i], &z, &us[i], cfg.rho);
            xs[i] = xi_new;
        }

        // ---- z-step: z = (1/N) Σ_i (x_i + u_i) ----
        let mut z_new = vec![0.0_f64; dim];
        for i in 0..n_agents {
            for j in 0..dim {
                z_new[j] += xs[i][j] + us[i][j];
            }
        }
        for v in z_new.iter_mut() {
            *v /= n_f;
        }

        // ---- u-step: u_i += x_i − z ----
        for i in 0..n_agents {
            for j in 0..dim {
                us[i][j] += xs[i][j] - z_new[j];
            }
        }

        // ---- convergence check: primal residual Σ_i ‖x_i − z‖² ----
        let mut r_sq = 0.0_f64;
        for xi in xs.iter() {
            for j in 0..dim {
                let d = xi[j] - z_new[j];
                r_sq += d * d;
            }
        }
        residual = r_sq.sqrt();
        z = z_new;
        final_iter = iter + 1;

        if residual < cfg.tol {
            converged = true;
            break;
        }
    }

    Ok(ConsensusAdmmResult {
        z,
        iter: final_iter,
        residual,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Boxed proximal-update closure used by the consensus tests:
    /// `(id, x_i, z, u, rho) -> x_i^{k+1}`.
    type ProxUpdate = Box<dyn Fn(usize, &[f64], &[f64], &[f64], f64) -> Vec<f64>>;

    #[test]
    fn convergence_on_quadratic() {
        // 3 agents in 2D: f_i(x) = 0.5*||x - b_i||^2
        // Consensus solution: z = mean(b_i)
        let bs = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let expected = vec![3.0_f64, 4.0]; // mean
        let cfg = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 1000,
            tol: 1e-9,
        };
        let x_init = bs.clone();
        let updates: Vec<ProxUpdate> = bs
            .iter()
            .map(|b| {
                let b_owned = b.clone();
                Box::new(
                    move |_id: usize, _xi: &[f64], z: &[f64], u: &[f64], rho: f64| -> Vec<f64> {
                        (0..b_owned.len())
                            .map(|j| (b_owned[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                            .collect()
                    },
                ) as ProxUpdate
            })
            .collect();
        let result = consensus_admm(
            3,
            2,
            x_init,
            |id, xi, z, u, rho| updates[id](id, xi, z, u, rho),
            &cfg,
        )
        .expect("should converge");
        assert!(result.converged, "should converge");
        assert!(
            (result.z[0] - expected[0]).abs() < 1e-5,
            "z[0]={}",
            result.z[0]
        );
        assert!(
            (result.z[1] - expected[1]).abs() < 1e-5,
            "z[1]={}",
            result.z[1]
        );
    }

    #[test]
    fn z_shape() {
        let dim = 5_usize;
        let cfg = ConsensusAdmmConfig::default();
        let x_init = vec![vec![0.0_f64; dim]; 2];
        let result =
            consensus_admm(2, dim, x_init, |_id, _xi, z, _u, _rho| z.to_vec(), &cfg).expect("ok");
        assert_eq!(result.z.len(), dim);
    }

    #[test]
    fn residual_decreases() {
        // Run few iters vs many iters; more iters → smaller residual.
        let b = vec![1.0_f64, 2.0];
        let cfg_few = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 3,
            tol: 1e-15,
        };
        let cfg_many = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 200,
            tol: 1e-15,
        };
        let x_init = vec![b.clone(), vec![0.0, 0.0]];
        let b1 = b.clone();
        let b2 = b.clone();
        let res_few = consensus_admm(
            2,
            2,
            x_init.clone(),
            move |id, _xi, z, u, rho| {
                let bv = if id == 0 { &b1 } else { &b2 };
                (0..bv.len())
                    .map(|j| (bv[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg_few,
        )
        .expect("ok");
        let b3 = b.clone();
        let b4 = b.clone();
        let res_many = consensus_admm(
            2,
            2,
            x_init,
            move |id, _xi, z, u, rho| {
                let bv = if id == 0 { &b3 } else { &b4 };
                (0..bv.len())
                    .map(|j| (bv[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg_many,
        )
        .expect("ok");
        assert!(
            res_many.residual <= res_few.residual + 1e-12,
            "many={}, few={}",
            res_many.residual,
            res_few.residual
        );
    }

    #[test]
    fn n_agents_1_trivial() {
        // Single agent: with N=1, constraint x=z is trivially satisfied in one step.
        // The algorithm immediately converges (residual=0) with z = x_new.
        let b = vec![3.0_f64, 7.0];
        let cfg = ConsensusAdmmConfig {
            rho: 2.0,
            max_iter: 300,
            tol: 1e-9,
        };
        let x_init = vec![vec![0.0_f64, 0.0]];
        let b_c = b.clone();
        let result = consensus_admm(
            1,
            2,
            x_init,
            move |_id, _xi, z, u, rho| {
                (0..b_c.len())
                    .map(|j| (b_c[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg,
        )
        .expect("ok");
        // With 1 agent: x = z = prox(z_prev), converges in 1 step.
        assert!(result.converged, "should converge");
        assert_eq!(result.z.len(), 2);
        // z should be finite and the algorithm should complete
        assert!(result.z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dim_mismatch_error() {
        // x_init[0] has dim=3 but declared dim=2 → DimensionMismatch.
        let cfg = ConsensusAdmmConfig::default();
        let x_init = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let result = consensus_admm(2, 2, x_init, |_id, _xi, z, _u, _rho| z.to_vec(), &cfg);
        match result {
            Err(CvxError::DimensionMismatch { .. }) => {}
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn max_iter_1_runs() {
        let cfg = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 1,
            tol: 1e-15,
        };
        let x_init = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let result =
            consensus_admm(2, 2, x_init, |_id, _xi, z, _u, _rho| z.to_vec(), &cfg).expect("ok");
        assert_eq!(result.iter, 1);
    }

    #[test]
    fn rho_affects_speed() {
        // Higher rho should converge faster (smaller residual in same iterations).
        let bs = [vec![1.0_f64, 0.0], vec![-1.0_f64, 0.0]];
        let cfg_low = ConsensusAdmmConfig {
            rho: 0.01,
            max_iter: 50,
            tol: 1e-15,
        };
        let cfg_high = ConsensusAdmmConfig {
            rho: 10.0,
            max_iter: 50,
            tol: 1e-15,
        };

        let b0l = bs[0].clone();
        let b1l = bs[1].clone();
        let res_low = consensus_admm(
            2,
            2,
            vec![bs[0].clone(), bs[1].clone()],
            move |id, _xi, z, u, rho| {
                let b = if id == 0 { &b0l } else { &b1l };
                (0..b.len())
                    .map(|j| (b[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg_low,
        )
        .expect("ok");

        let b0h = bs[0].clone();
        let b1h = bs[1].clone();
        let res_high = consensus_admm(
            2,
            2,
            vec![bs[0].clone(), bs[1].clone()],
            move |id, _xi, z, u, rho| {
                let b = if id == 0 { &b0h } else { &b1h };
                (0..b.len())
                    .map(|j| (b[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg_high,
        )
        .expect("ok");

        assert!(
            res_high.residual <= res_low.residual + 1e-8,
            "high_rho_residual={} should be ≤ low_rho_residual={}",
            res_high.residual,
            res_low.residual
        );
    }

    #[test]
    fn consensus_constraint_at_convergence() {
        // After convergence, each x_i should ≈ z.
        let bs = vec![vec![2.0_f64], vec![4.0], vec![6.0]];
        let cfg = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 1000,
            tol: 1e-9,
        };
        let x_init = bs.clone();
        let b0 = bs[0].clone();
        let b1 = bs[1].clone();
        let b2 = bs[2].clone();
        let result = consensus_admm(
            3,
            1,
            x_init,
            move |id, _xi, z, u, rho| {
                let b = match id {
                    0 => &b0,
                    1 => &b1,
                    _ => &b2,
                };
                (0..b.len())
                    .map(|j| (b[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg,
        )
        .expect("ok");
        assert!(result.converged);
        // residual is sqrt(sum ||x_i - z||^2) < tol
        assert!(result.residual < 1e-7, "residual={}", result.residual);
    }

    #[test]
    fn different_objectives_agree_on_z() {
        // f_0(x) = 0.5*(x-1)^2, f_1(x) = 0.5*(x-3)^2
        // Consensus: z = mean(1, 3) = 2.
        let b0 = vec![1.0_f64];
        let b1 = vec![3.0_f64];
        let cfg = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 500,
            tol: 1e-9,
        };
        let x_init = vec![b0.clone(), b1.clone()];
        let b0c = b0.clone();
        let b1c = b1.clone();
        let result = consensus_admm(
            2,
            1,
            x_init,
            move |id, _xi, z, u, rho| {
                let b = if id == 0 { &b0c } else { &b1c };
                (0..b.len())
                    .map(|j| (b[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg,
        )
        .expect("ok");
        assert!(result.converged);
        assert!((result.z[0] - 2.0).abs() < 1e-5, "z[0]={}", result.z[0]);
    }

    #[test]
    fn invalid_rho_error() {
        let cfg = ConsensusAdmmConfig {
            rho: -1.0,
            max_iter: 10,
            tol: 1e-6,
        };
        let result = consensus_admm(
            2,
            2,
            vec![vec![0.0; 2]; 2],
            |_id, _xi, z, _u, _rho| z.to_vec(),
            &cfg,
        );
        match result {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn n_agents_zero_error() {
        let cfg = ConsensusAdmmConfig::default();
        let result = consensus_admm(0, 2, vec![], |_id, _xi, z, _u, _rho| z.to_vec(), &cfg);
        match result {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn five_agents_convergence() {
        // 5 agents in 3D, all wanting their own centre.
        let targets: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64; 3]).collect();
        // expected z = [2.0, 2.0, 2.0] (mean of 0,1,2,3,4).
        let cfg = ConsensusAdmmConfig {
            rho: 1.0,
            max_iter: 2000,
            tol: 1e-9,
        };
        let x_init: Vec<Vec<f64>> = targets.to_vec();
        let tgts = targets.clone();
        let result = consensus_admm(
            5,
            3,
            x_init,
            move |id, _xi, z, u, rho| {
                let b = &tgts[id];
                (0..b.len())
                    .map(|j| (b[j] + rho * (z[j] - u[j])) / (1.0 + rho))
                    .collect()
            },
            &cfg,
        )
        .expect("ok");
        assert!(
            result.converged,
            "should converge, residual={}",
            result.residual
        );
        for &zj in &result.z {
            assert!((zj - 2.0).abs() < 1e-4, "z_j={zj}");
        }
    }
}
