//! Firefly Algorithm for continuous optimization.
//!
//! Reference: X.-S. Yang, "Firefly Algorithm, Lévy Flights and Global Optimization",
//! Research and Development in Intelligent Systems XXVI, pp. 209-218, Springer, 2010.
//!
//! # Algorithm overview
//!
//! Each firefly has a brightness (light intensity) equal to the negated objective
//! value (lower objective = brighter = better for minimisation).  Fireflies move toward
//! brighter neighbours with an attractiveness that decays exponentially with squared
//! Euclidean distance.  Fireflies with no brighter neighbour perform a pure random walk.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Firefly Algorithm hyper-parameters.
#[derive(Debug, Clone)]
pub struct FireflyConfig {
    /// Number of fireflies.
    pub n_fireflies: usize,
    /// Maximum number of generations.
    pub max_iter: usize,
    /// Randomness step size coefficient (α).  Controls the scale of the random walk.
    /// Typical default: 0.2.
    pub alpha: f64,
    /// Attractiveness at zero distance (β₀).
    /// Typical default: 1.0.
    pub beta0: f64,
    /// Light absorption coefficient (γ).  Larger γ → faster attractiveness decay with distance.
    /// Typical default: 1.0.
    pub gamma: f64,
    /// Random seed for the internal LCG RNG.
    pub seed: u64,
}

impl FireflyConfig {
    /// Construct a `FireflyConfig` with sensible defaults.
    pub fn new(n_fireflies: usize, max_iter: usize) -> EvolResult<Self> {
        if n_fireflies == 0 {
            return Err(EvolError::SwarmEmpty);
        }
        if max_iter == 0 {
            return Err(EvolError::InvalidParameter(
                "max_iter must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_fireflies,
            max_iter,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 0,
        })
    }
}

/// Mutable state for a Firefly Algorithm run.
pub struct FireflyState {
    /// Current positions of all fireflies (n_fireflies × n_dims).
    pub positions: Vec<Vec<f64>>,
    /// Light intensity of each firefly: `light[i] = -f(x_i)`.  Higher = brighter = better.
    pub light: Vec<f64>,
    /// Per-dimension search bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Best decision variable vector seen across all iterations.
    pub best: Vec<f64>,
    /// Light intensity of the best solution (`-f(best)`).
    pub best_light: f64,
    /// Number of completed generations.
    pub generation: usize,
}

impl FireflyState {
    /// Initialise fireflies uniformly at random within `bounds` and evaluate initial brightness.
    pub fn new<F: Fn(&[f64]) -> f64>(
        bounds: Vec<(f64, f64)>,
        n_fireflies: usize,
        fitness_fn: &F,
        rng: &mut LcgRng,
    ) -> EvolResult<Self> {
        if bounds.is_empty() {
            return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
        }
        if n_fireflies == 0 {
            return Err(EvolError::SwarmEmpty);
        }

        let n_dims = bounds.len();
        let positions: Vec<Vec<f64>> = (0..n_fireflies)
            .map(|_| {
                (0..n_dims)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        lb + rng.next_f64() * (ub - lb)
                    })
                    .collect()
            })
            .collect();

        let light: Vec<f64> = positions.iter().map(|x| -fitness_fn(x)).collect();

        let (best_idx, &best_light) = light
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(EvolError::SwarmEmpty)?;

        let best = positions[best_idx].clone();

        Ok(Self {
            positions,
            light,
            bounds,
            best,
            best_light,
            generation: 0,
        })
    }
}

/// Compute squared Euclidean distance between two equal-length slices.
#[inline]
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum()
}

/// Execute one Firefly Algorithm generation step using explicit hyper-parameters.
///
/// For every firefly `i`, if any firefly `j` is brighter (`light[j] > light[i]`),
/// the firefly moves toward the brightest such neighbour with distance-decayed
/// attractiveness plus a random perturbation.  If no brighter neighbour exists, a
/// pure random walk is applied.  All positions are clipped to `bounds` after movement.
///
/// After moving, every firefly's objective is re-evaluated and lights are updated.
pub fn firefly_step<F: Fn(&[f64]) -> f64>(
    state: &mut FireflyState,
    fitness_fn: F,
    rng: &mut LcgRng,
) {
    // Default hyper-parameters used by the public step (for external callers).
    firefly_step_inner(state, &fitness_fn, rng, 0.2, 1.0, 1.0);
}

/// Internal step function with explicit hyper-parameter arguments.
#[allow(clippy::needless_range_loop)]
fn firefly_step_inner<F: Fn(&[f64]) -> f64>(
    state: &mut FireflyState,
    fitness_fn: &F,
    rng: &mut LcgRng,
    alpha: f64,
    beta0: f64,
    gamma: f64,
) {
    let n = state.positions.len();
    let n_dims = state.bounds.len();

    // Snapshot of positions and lights so that this generation's attraction is computed
    // from the *pre-step* brightness ranking.
    let positions_snap: Vec<Vec<f64>> = state.positions.clone();
    let light_snap: Vec<f64> = state.light.clone();

    for i in 0..n {
        // Find the brightest firefly j ≠ i that is strictly brighter than i.
        let brightest_j: Option<usize> = (0..n)
            .filter(|&j| j != i && light_snap[j] > light_snap[i])
            .max_by(|&a, &b| {
                light_snap[a]
                    .partial_cmp(&light_snap[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match brightest_j {
            Some(j) => {
                // Attractiveness decays with squared distance.
                let r2 = squared_distance(&positions_snap[i], &positions_snap[j]);
                let beta = beta0 * (-gamma * r2).exp();

                for d in 0..n_dims {
                    let rand_term = rng.next_f64() - 0.5; // uniform in [-0.5, 0.5]
                    let delta =
                        beta * (positions_snap[j][d] - positions_snap[i][d]) + alpha * rand_term;
                    let (lb, ub) = state.bounds[d];
                    state.positions[i][d] = (state.positions[i][d] + delta).clamp(lb, ub);
                }
            }
            None => {
                // No brighter neighbour: random walk only.
                for d in 0..n_dims {
                    let rand_term = rng.next_f64() - 0.5;
                    let (lb, ub) = state.bounds[d];
                    state.positions[i][d] =
                        (state.positions[i][d] + alpha * rand_term).clamp(lb, ub);
                }
            }
        }

        // Re-evaluate immediately after moving this firefly.
        let raw = fitness_fn(&state.positions[i]);
        state.light[i] = -raw;
        if state.light[i] > state.best_light {
            state.best_light = state.light[i];
            state.best = state.positions[i].clone();
        }
    }

    state.generation += 1;
}

/// Run the Firefly Algorithm to completion.
///
/// Returns the final `FireflyState`; `state.best` is the best decision variable vector
/// found and `state.best_light = -f(best)`.
pub fn firefly_run<F>(
    fitness_fn: F,
    bounds: &[(f64, f64)],
    cfg: &FireflyConfig,
) -> EvolResult<FireflyState>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
    }
    if cfg.n_fireflies == 0 {
        return Err(EvolError::SwarmEmpty);
    }
    if cfg.n_fireflies < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: cfg.n_fireflies,
            op: "Firefly",
        });
    }

    let mut rng = LcgRng::new(cfg.seed);
    let mut state = FireflyState::new(bounds.to_vec(), cfg.n_fireflies, &fitness_fn, &mut rng)?;

    for _ in 0..cfg.max_iter {
        firefly_step_inner(
            &mut state,
            &fitness_fn,
            &mut rng,
            cfg.alpha,
            cfg.beta0,
            cfg.gamma,
        );
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Objective functions ──────────────────────────────────────────────────

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| {
                let (xi, xj) = (w[0], w[1]);
                100.0 * (xj - xi * xi).powi(2) + (1.0 - xi).powi(2)
            })
            .sum()
    }

    fn ackley(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + std::f64::consts::E
            + 20.0
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
                .sum::<f64>()
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = FireflyConfig::new(20, 100).expect("new should succeed");
        assert_eq!(cfg.n_fireflies, 20);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.alpha - 0.2).abs() < 1e-12);
        assert!((cfg.beta0 - 1.0).abs() < 1e-12);
        assert!((cfg.gamma - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_config_new_zero_fireflies() {
        assert!(FireflyConfig::new(0, 100).is_err());
    }

    #[test]
    fn test_config_new_zero_iter() {
        assert!(FireflyConfig::new(10, 0).is_err());
    }

    // ── State construction tests ──────────────────────────────────────────────

    #[test]
    fn test_state_new_correct_sizes() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let mut rng = LcgRng::new(0);
        let state = FireflyState::new(bounds, 10, &sphere, &mut rng).expect("new should succeed");
        assert_eq!(state.positions.len(), 10);
        assert_eq!(state.light.len(), 10);
        assert_eq!(state.positions[0].len(), 3);
        assert_eq!(state.generation, 0);
    }

    #[test]
    fn test_state_new_best_is_max_light() {
        let bounds = vec![(-2.0_f64, 2.0_f64); 2];
        let mut rng = LcgRng::new(42);
        let state = FireflyState::new(bounds, 8, &sphere, &mut rng).expect("new should succeed");
        let max_light = state
            .light
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((state.best_light - max_light).abs() < 1e-12);
    }

    #[test]
    fn test_state_new_positions_within_bounds() {
        let bounds: Vec<(f64, f64)> = vec![(-3.0, 3.0), (1.0, 5.0), (-10.0, 0.0)];
        let mut rng = LcgRng::new(7);
        let state = FireflyState::new(bounds.clone(), 15, &sphere, &mut rng)
            .expect("value should be present");
        for pos in &state.positions {
            for (d, &x) in pos.iter().enumerate() {
                let (lb, ub) = bounds[d];
                assert!(x >= lb && x <= ub, "dim {d}: {x} not in [{lb}, {ub}]");
            }
        }
    }

    // ── Step tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_firefly_step_increments_generation() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 2];
        let mut rng = LcgRng::new(1);
        let mut state =
            FireflyState::new(bounds, 5, &sphere, &mut rng).expect("new should succeed");
        assert_eq!(state.generation, 0);
        firefly_step(&mut state, sphere, &mut rng);
        assert_eq!(state.generation, 1);
        firefly_step(&mut state, sphere, &mut rng);
        assert_eq!(state.generation, 2);
    }

    #[test]
    fn test_firefly_step_best_non_decreasing() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let mut rng = LcgRng::new(99);
        let mut state =
            FireflyState::new(bounds, 8, &sphere, &mut rng).expect("new should succeed");
        let initial_best_light = state.best_light;
        for _ in 0..20 {
            firefly_step(&mut state, sphere, &mut rng);
        }
        // best_light is -f(best); it should only increase (fitness decreases).
        assert!(
            state.best_light >= initial_best_light - 1e-12,
            "best_light decreased: {} < {}",
            state.best_light,
            initial_best_light
        );
    }

    #[test]
    fn test_firefly_step_positions_within_bounds() {
        let bounds: Vec<(f64, f64)> = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let mut rng = LcgRng::new(5);
        let mut state = FireflyState::new(bounds.clone(), 6, &sphere, &mut rng)
            .expect("value should be present");
        for _ in 0..30 {
            firefly_step(&mut state, sphere, &mut rng);
        }
        for pos in &state.positions {
            for (d, &x) in pos.iter().enumerate() {
                let (lb, ub) = bounds[d];
                assert!(x >= lb && x <= ub, "dim {d}: {x} out of [{lb},{ub}]");
            }
        }
    }

    // ── Full run tests ────────────────────────────────────────────────────────

    #[test]
    fn test_firefly_run_sphere_1d() {
        let bounds = vec![(-10.0_f64, 10.0_f64)];
        let cfg = FireflyConfig {
            n_fireflies: 20,
            max_iter: 300,
            alpha: 0.3,
            beta0: 1.0,
            gamma: 1.0,
            seed: 42,
        };
        let state = firefly_run(sphere, &bounds, &cfg).expect("firefly_run should succeed");
        // Should meaningfully reduce from 100 (max sphere at ±10).
        assert!(
            state.best_light > -50.0,
            "best_light = {}",
            state.best_light
        );
        assert_eq!(state.best.len(), 1);
    }

    #[test]
    fn test_firefly_run_sphere_5d() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 5];
        let cfg = FireflyConfig {
            n_fireflies: 25,
            max_iter: 500,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 0.5,
            seed: 7,
        };
        let state = firefly_run(sphere, &bounds, &cfg).expect("firefly_run should succeed");
        // Worst case: 5 * 25 = 125; should do better.
        let best_fitness = -state.best_light;
        assert!(best_fitness < 125.0, "best_fitness = {best_fitness}");
    }

    #[test]
    fn test_firefly_run_ackley_3d() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let cfg = FireflyConfig {
            n_fireflies: 20,
            max_iter: 400,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 11,
        };
        let worst = ackley(&[5.0, 5.0, 5.0]);
        let state = firefly_run(ackley, &bounds, &cfg).expect("firefly_run should succeed");
        let best_fitness = -state.best_light;
        assert!(best_fitness < worst, "best={best_fitness} worst={worst}");
    }

    #[test]
    fn test_firefly_run_rastrigin_2d() {
        let bounds = vec![(-5.12_f64, 5.12_f64); 2];
        let cfg = FireflyConfig {
            n_fireflies: 20,
            max_iter: 500,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 33,
        };
        let worst = rastrigin(&[5.12, 5.12]);
        let state = firefly_run(rastrigin, &bounds, &cfg).expect("firefly_run should succeed");
        let best_fitness = -state.best_light;
        assert!(best_fitness < worst, "best={best_fitness} worst={worst}");
    }

    #[test]
    fn test_firefly_run_rosenbrock_2d() {
        let bounds = vec![(-2.0_f64, 2.0_f64); 2];
        let cfg = FireflyConfig {
            n_fireflies: 20,
            max_iter: 1000,
            alpha: 0.1,
            beta0: 1.0,
            gamma: 0.5,
            seed: 77,
        };
        let state = firefly_run(rosenbrock, &bounds, &cfg).expect("firefly_run should succeed");
        // Rosenbrock global min = 0; should be < 200.
        let best_fitness = -state.best_light;
        assert!(best_fitness < 200.0, "best_fitness = {best_fitness}");
    }

    #[test]
    fn test_firefly_run_error_empty_bounds() {
        let cfg = FireflyConfig {
            n_fireflies: 5,
            max_iter: 10,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 0,
        };
        assert!(firefly_run(sphere, &[], &cfg).is_err());
    }

    #[test]
    fn test_firefly_run_best_within_bounds() {
        let bounds: Vec<(f64, f64)> = vec![(-3.0, 3.0), (-3.0, 3.0)];
        let cfg = FireflyConfig {
            n_fireflies: 15,
            max_iter: 200,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 21,
        };
        let state = firefly_run(sphere, &bounds, &cfg).expect("firefly_run should succeed");
        for (d, &x) in state.best.iter().enumerate() {
            let (lb, ub) = bounds[d];
            assert!(x >= lb && x <= ub, "dim {d}: {x} out of [{lb},{ub}]");
        }
    }

    #[test]
    fn test_firefly_run_deterministic() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let cfg = FireflyConfig {
            n_fireflies: 10,
            max_iter: 100,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 999,
        };
        let s1 = firefly_run(sphere, &bounds, &cfg).expect("firefly_run should succeed");
        let s2 = firefly_run(sphere, &bounds, &cfg).expect("firefly_run should succeed");
        assert_eq!(s1.best_light, s2.best_light, "runs not deterministic");
    }

    #[test]
    fn test_squared_distance_correctness() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 2.0];
        // distance = sqrt(1+4+4) = 3, squared = 9
        assert!((squared_distance(&a, &b) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_firefly_run_generation_count() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 2];
        let cfg = FireflyConfig {
            n_fireflies: 8,
            max_iter: 50,
            alpha: 0.2,
            beta0: 1.0,
            gamma: 1.0,
            seed: 3,
        };
        let state = firefly_run(sphere, &bounds, &cfg).expect("firefly_run should succeed");
        assert_eq!(state.generation, 50);
    }
}
