//! Simulated Annealing (SA) for continuous box-constrained black-box minimisation.
//!
//! References:
//! - S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, "Optimization by Simulated Annealing",
//!   Science 220(4598):671-680, 1983.
//! - H. Szu & R. Hartley, "Fast Simulated Annealing", Physics Letters A 122(3-4):157-162,
//!   1987 (Cauchy / fast cooling).
//! - L. Ingber, "Very fast simulated re-annealing", Math. Comput. Modelling 12(8):967-973,
//!   1989.
//!
//! ## Overview
//! SA explores the search space with a temperature-controlled random walk. Each iteration:
//!
//! 1. A **neighbour** `x'` is generated from the current point `x` by adding Gaussian noise
//!    scaled by the current step size (proportional to the box width).
//! 2. The energy change `Δ = f(x') − f(x)` is computed.
//! 3. The neighbour is accepted with the **Metropolis criterion**
//!    `P = 1` if `Δ ≤ 0`, else `P = exp(−Δ / T)`. As the temperature `T` cools, uphill moves
//!    become increasingly unlikely and the walk settles into a minimum.
//!
//! The step size is annealed alongside the temperature so the walk transitions from coarse
//! global exploration to fine local refinement.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Temperature cooling schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoolingSchedule {
    /// Geometric / exponential cooling: `T_k = T₀·α^k` (`0 < α < 1`). The classic schedule.
    Geometric,
    /// Linear cooling: `T_k = T₀·(1 − k/K)`, reaching ~0 at the final iteration `K`.
    Linear,
    /// Logarithmic (Boltzmann) cooling: `T_k = T₀ / ln(k + e)`. Slow but with the strongest
    /// asymptotic convergence guarantee (Geman & Geman 1984).
    Logarithmic,
    /// Fast (Cauchy) cooling: `T_k = T₀ / (1 + k)` (Szu & Hartley 1987).
    Fast,
}

/// Hyper-parameters for a simulated-annealing run.
#[derive(Debug, Clone)]
pub struct SaConfig {
    /// Problem dimension n.
    pub n_dims: usize,
    /// Per-dimension search bounds `[lb, ub]` shared across all coordinates.
    pub bounds: (f64, f64),
    /// Initial temperature T₀ (> 0).
    pub t_init: f64,
    /// Geometric cooling factor α ∈ (0, 1) (used only by [`CoolingSchedule::Geometric`]).
    pub alpha: f64,
    /// Cooling schedule.
    pub schedule: CoolingSchedule,
    /// Number of outer cooling steps.
    pub max_iters: usize,
    /// Number of neighbour trials per temperature level (the inner Markov chain length).
    pub chain_length: usize,
    /// Initial neighbour step size as a fraction of the box width (e.g. 0.5).
    pub step_fraction: f64,
    /// Anneal the step size from `step_fraction` down to `step_fraction·step_decay` linearly
    /// over the run (in `(0, 1]`; 1.0 keeps the step size constant).
    pub step_decay: f64,
    /// Convergence threshold on the best energy; the run stops early once `best < tol`.
    pub tol: f64,
}

impl SaConfig {
    /// Build a default configuration for an `n`-dimensional problem on `bounds`.
    pub fn new(n_dims: usize, bounds: (f64, f64)) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        if bounds.0 >= bounds.1 {
            return Err(EvolError::InvalidParameter(
                "bounds: lower must be < upper".to_owned(),
            ));
        }
        Ok(Self {
            n_dims,
            bounds,
            t_init: 10.0,
            alpha: 0.95,
            schedule: CoolingSchedule::Geometric,
            max_iters: 1000,
            chain_length: 20,
            step_fraction: 0.5,
            step_decay: 0.01,
            tol: 1e-10,
        })
    }

    fn validate(&self) -> EvolResult<()> {
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        if self.bounds.0 >= self.bounds.1 {
            return Err(EvolError::InvalidParameter(
                "bounds: lower must be < upper".to_owned(),
            ));
        }
        if self.t_init <= 0.0 {
            return Err(EvolError::InvalidParameter("t_init must be > 0".to_owned()));
        }
        if !(0.0..1.0).contains(&self.alpha) || self.alpha <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "alpha must be in (0, 1)".to_owned(),
            ));
        }
        if self.chain_length == 0 {
            return Err(EvolError::InvalidParameter(
                "chain_length must be >= 1".to_owned(),
            ));
        }
        if self.step_fraction <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "step_fraction must be > 0".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.step_decay) || self.step_decay <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "step_decay must be in (0, 1]".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Result / final state of a simulated-annealing run.
#[derive(Debug, Clone)]
pub struct SaState {
    /// Best solution found.
    pub best_x: Vec<f64>,
    /// Best energy (objective value) found.
    pub best_energy: f64,
    /// Final current point of the walk.
    pub current_x: Vec<f64>,
    /// Final current energy of the walk.
    pub current_energy: f64,
    /// Total number of objective evaluations performed.
    pub n_evals: usize,
    /// Number of accepted moves over the run.
    pub n_accepted: usize,
}

/// Compute the temperature at cooling step `k` (0-based) for the given schedule.
fn temperature(cfg: &SaConfig, k: usize) -> f64 {
    let k_f = k as f64;
    match cfg.schedule {
        CoolingSchedule::Geometric => cfg.t_init * cfg.alpha.powf(k_f),
        CoolingSchedule::Linear => {
            let frac = 1.0 - k_f / cfg.max_iters.max(1) as f64;
            (cfg.t_init * frac).max(0.0)
        }
        CoolingSchedule::Logarithmic => cfg.t_init / (k_f + std::f64::consts::E).ln(),
        CoolingSchedule::Fast => cfg.t_init / (1.0 + k_f),
    }
}

/// Run simulated annealing to minimise `objective` starting at `x_init`.
///
/// `x_init` is clamped into the configured bounds. Returns the final [`SaState`] carrying the
/// best solution found together with run statistics.
///
/// # Errors
/// Returns `EvolError` if the configuration or initial point dimension is invalid.
pub fn simulated_annealing<F>(
    cfg: &SaConfig,
    x_init: Vec<f64>,
    objective: F,
    rng: &mut LcgRng,
) -> EvolResult<SaState>
where
    F: Fn(&[f64]) -> f64,
{
    cfg.validate()?;
    if x_init.len() != cfg.n_dims {
        return Err(EvolError::DimensionMismatch {
            expected: cfg.n_dims,
            got: x_init.len(),
        });
    }

    let (lb, ub) = cfg.bounds;
    let width = ub - lb;

    let mut current: Vec<f64> = x_init.iter().map(|&v| v.clamp(lb, ub)).collect();
    let mut current_energy = objective(&current);
    let mut best = current.clone();
    let mut best_energy = current_energy;

    let mut n_evals = 1usize;
    let mut n_accepted = 0usize;

    let iters = cfg.max_iters.max(1);
    'outer: for k in 0..iters {
        let temp = temperature(cfg, k).max(1e-300);

        // Anneal the proposal step size linearly from step_fraction to step_fraction·decay.
        let progress = k as f64 / iters as f64;
        let step_scale = cfg.step_fraction * (1.0 - progress * (1.0 - cfg.step_decay));
        let step = (step_scale * width).max(1e-300);

        for _ in 0..cfg.chain_length {
            // ── Generate a neighbour by Gaussian perturbation + reflection ────
            let candidate: Vec<f64> = current
                .iter()
                .map(|&xi| {
                    let proposed = xi + step * rng.next_normal();
                    reflect_into_bounds(proposed, lb, ub)
                })
                .collect();

            let cand_energy = objective(&candidate);
            n_evals += 1;

            // ── Metropolis acceptance ─────────────────────────────────────────
            let delta = cand_energy - current_energy;
            let accept = if delta <= 0.0 {
                true
            } else {
                let p = (-delta / temp).exp();
                rng.next_f64() < p
            };

            if accept {
                current = candidate;
                current_energy = cand_energy;
                n_accepted += 1;
                if current_energy < best_energy {
                    best_energy = current_energy;
                    best = current.clone();
                    if best_energy < cfg.tol {
                        break 'outer;
                    }
                }
            }
        }
    }

    Ok(SaState {
        best_x: best,
        best_energy,
        current_x: current,
        current_energy,
        n_evals,
        n_accepted,
    })
}

/// Reflect `v` back into `[lb, ub]` if it overshoots a boundary (folding random walk).
///
/// Reflection (as opposed to hard clamping) preserves the local exploration step length near
/// the boundary, avoiding the bias of accumulating probability mass exactly at `lb`/`ub`.
fn reflect_into_bounds(v: f64, lb: f64, ub: f64) -> f64 {
    let width = ub - lb;
    if width <= 0.0 {
        return lb;
    }
    let mut x = v;
    // Reflect repeatedly until inside (handles overshoots larger than the box).
    let mut guard = 0;
    while (x < lb || x > ub) && guard < 64 {
        if x < lb {
            x = lb + (lb - x);
        } else if x > ub {
            x = ub - (x - ub);
        }
        guard += 1;
    }
    x.clamp(lb, ub)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        10.0 * x.len() as f64
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (two_pi * xi).cos())
                .sum::<f64>()
    }

    #[test]
    fn config_rejects_bad_bounds() {
        assert!(SaConfig::new(2, (1.0, 1.0)).is_err());
        assert!(SaConfig::new(0, (-1.0, 1.0)).is_err());
    }

    #[test]
    fn reflect_keeps_inside() {
        assert!((reflect_into_bounds(6.0, -5.0, 5.0) - 4.0).abs() < 1e-12);
        assert!((reflect_into_bounds(-7.0, -5.0, 5.0) - (-3.0)).abs() < 1e-12);
        assert!((reflect_into_bounds(2.0, -5.0, 5.0) - 2.0).abs() < 1e-12);
        let v = reflect_into_bounds(100.0, -5.0, 5.0);
        assert!((-5.0..=5.0).contains(&v));
    }

    #[test]
    fn temperature_monotone_decreasing_geometric() {
        let cfg = SaConfig::new(2, (-5.0, 5.0)).expect("ok");
        let t0 = temperature(&cfg, 0);
        let t1 = temperature(&cfg, 1);
        let t10 = temperature(&cfg, 10);
        assert!(t1 < t0 && t10 < t1);
        assert!((t0 - cfg.t_init).abs() < 1e-12);
    }

    #[test]
    fn minimizes_sphere_5d_geometric() {
        let mut cfg = SaConfig::new(5, (-5.0, 5.0)).expect("ok");
        cfg.t_init = 5.0;
        cfg.alpha = 0.97;
        cfg.max_iters = 2000;
        cfg.chain_length = 30;
        let mut rng = LcgRng::new(42);
        let x0 = vec![4.0, -3.0, 3.5, -2.0, 2.5];
        let st = simulated_annealing(&cfg, x0, sphere, &mut rng).expect("ok");
        assert!(
            st.best_energy < 1e-3,
            "SA should minimise 5-D sphere below 1e-3, got {} at {:?}",
            st.best_energy,
            st.best_x
        );
        assert!(st.n_evals > 0 && st.n_accepted > 0);
    }

    #[test]
    fn minimizes_rastrigin_2d_escapes_local_minima() {
        // Rastrigin is highly multimodal; SA's uphill moves should let it find the basin.
        let mut cfg = SaConfig::new(2, (-5.12, 5.12)).expect("ok");
        cfg.t_init = 20.0;
        cfg.alpha = 0.98;
        cfg.max_iters = 3000;
        cfg.chain_length = 25;
        let mut rng = LcgRng::new(2024);
        let x0 = vec![4.5, -4.0];
        let st = simulated_annealing(&cfg, x0, rastrigin, &mut rng).expect("ok");
        assert!(
            st.best_energy < 1.0,
            "SA should escape Rastrigin local minima (best < 1.0), got {}",
            st.best_energy
        );
    }

    #[test]
    fn fast_schedule_also_minimizes_sphere() {
        let mut cfg = SaConfig::new(3, (-10.0, 10.0)).expect("ok");
        cfg.schedule = CoolingSchedule::Fast;
        cfg.t_init = 10.0;
        cfg.max_iters = 4000;
        cfg.chain_length = 20;
        let mut rng = LcgRng::new(7);
        let st = simulated_annealing(&cfg, vec![8.0, -7.0, 6.0], sphere, &mut rng).expect("ok");
        assert!(
            st.best_energy < 1e-2,
            "SA (fast schedule) should minimise sphere, got {}",
            st.best_energy
        );
    }

    #[test]
    fn linear_schedule_reaches_zero_temperature() {
        let cfg = SaConfig::new(2, (-5.0, 5.0)).expect("ok");
        let t_last = temperature(&cfg, cfg.max_iters);
        assert!(t_last <= 1e-9, "linear temp should reach ~0, got {t_last}");
    }

    #[test]
    fn rejects_wrong_init_dim() {
        let cfg = SaConfig::new(3, (-1.0, 1.0)).expect("ok");
        let mut rng = LcgRng::new(1);
        let r = simulated_annealing(&cfg, vec![0.0, 0.0], sphere, &mut rng);
        assert!(r.is_err());
    }

    #[test]
    fn best_never_worse_than_init() {
        let cfg = SaConfig::new(2, (-5.0, 5.0)).expect("ok");
        let mut rng = LcgRng::new(3);
        let x0 = vec![1.0, 1.0];
        let init_e = sphere(&x0);
        let st = simulated_annealing(&cfg, x0, sphere, &mut rng).expect("ok");
        assert!(st.best_energy <= init_e + 1e-12);
    }
}
