//! Artificial Bee Colony (ABC) algorithm for continuous optimization.
//!
//! Reference: D. Karaboga, "An Idea Based on Honey Bee Swarm for Numerical Optimization",
//! Technical Report TR06, Erciyes University, 2005.
//!
//! # Algorithm overview
//!
//! The colony consists of three phases executed every generation:
//!
//! 1. **Employed bees** — each bee exploits its food source by generating a neighbour
//!    candidate and applying greedy selection.
//! 2. **Onlooker bees** — each bee probabilistically selects a food source (fitness-
//!    proportional roulette) and applies the same exploit-and-select operation.
//! 3. **Scout bees** — any food source whose trial counter exceeds `limit` is abandoned
//!    and replaced by a randomly initialised new source.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// ABC algorithm hyper-parameters.
#[derive(Debug, Clone)]
pub struct AbcConfig {
    /// Number of employed bees (= number of food sources).
    pub n_bees: usize,
    /// Number of food sources (n_bees = n_food for the employed phase).
    pub n_food: usize,
    /// Abandonment threshold: a source not improved in `limit` trials is replaced.
    pub limit: usize,
    /// Maximum number of generations (each generation = employed + onlooker + scout phases).
    pub max_iter: usize,
    /// Random seed.
    pub seed: u64,
}

impl AbcConfig {
    /// Build a default `AbcConfig` with `n_food` food sources.
    ///
    /// Sets n_bees = n_food (employed bees only; onlooker bees match employed count),
    /// limit = n_food * n_dims (typical recommendation), max_iter = 1000.
    pub fn new(n_food: usize, n_dims: usize) -> EvolResult<Self> {
        if n_food == 0 {
            return Err(EvolError::SwarmEmpty);
        }
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_bees: n_food,
            n_food,
            limit: n_food * n_dims,
            max_iter: 1000,
            seed: 0,
        })
    }
}

/// Mutable state for an ABC run.
pub struct AbcState {
    /// Current food source positions (n_food × n_dims).
    pub food: Vec<Vec<f64>>,
    /// Fitness transform values: fitness_i = 1 / (1 + |f(x_i)|).
    pub fitness: Vec<f64>,
    /// Trial counters for each food source.
    pub trial: Vec<usize>,
    /// Decision variable bounds (per dimension).
    pub bounds: Vec<(f64, f64)>,
    /// Best known solution.
    pub best: Vec<f64>,
    /// Best known fitness (raw objective value, minimisation).
    pub best_fitness: f64,
    /// Current generation counter.
    pub generation: usize,
    /// Abandonment threshold: sources not improved in `limit` trials are replaced.
    pub limit: usize,
}

impl AbcState {
    /// Initialise a new ABC state.
    ///
    /// `raw_fitness_fn` must return the raw objective value (lower = better for minimisation).
    pub fn new<F: Fn(&[f64]) -> f64>(
        bounds: Vec<(f64, f64)>,
        n_food: usize,
        limit: usize,
        fitness_fn: &F,
        rng: &mut LcgRng,
    ) -> EvolResult<Self> {
        if bounds.is_empty() {
            return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
        }
        if n_food == 0 {
            return Err(EvolError::SwarmEmpty);
        }

        let n_dims = bounds.len();
        let food: Vec<Vec<f64>> = (0..n_food)
            .map(|_| {
                (0..n_dims)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        lb + rng.next_f64() * (ub - lb)
                    })
                    .collect()
            })
            .collect();

        let raw_vals: Vec<f64> = food.iter().map(|x| fitness_fn(x)).collect();
        let fitness: Vec<f64> = raw_vals.iter().map(|&v| fitness_transform(v)).collect();

        // Best source
        let (best_idx, &best_raw) = raw_vals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let best = food[best_idx].clone();

        Ok(Self {
            food,
            fitness,
            trial: vec![0; n_food],
            bounds,
            best,
            best_fitness: best_raw,
            generation: 0,
            limit,
        })
    }
}

/// Transform a raw objective value to a fitness value in (0, 1].
///
/// f_transformed = 1 / (1 + |f(x)|), so smaller raw values → larger fitness.
#[inline]
fn fitness_transform(raw: f64) -> f64 {
    1.0 / (1.0 + raw.abs())
}

/// Generate a candidate neighbour of food source `i` by mutating dimension `j`
/// using a randomly selected partner source `k` ≠ `i` and a random `phi ∈ [-1, 1]`.
///
/// v_ij = x_ij + phi * (x_ij - x_kj), clamped to bounds.
fn generate_candidate(
    i: usize,
    food: &[Vec<f64>],
    bounds: &[(f64, f64)],
    n_food: usize,
    rng: &mut LcgRng,
) -> Vec<f64> {
    let n_dims = food[i].len();

    // Choose partner k ≠ i
    let mut k = rng.next_usize(n_food - 1);
    if k >= i {
        k += 1;
    }

    // Random dimension to perturb
    let j = rng.next_usize(n_dims);

    // phi ∈ [-1, 1]
    let phi = rng.next_f64() * 2.0 - 1.0;

    let mut candidate = food[i].clone();
    let (lb, ub) = bounds[j];
    let new_val = food[i][j] + phi * (food[i][j] - food[k][j]);
    candidate[j] = new_val.clamp(lb, ub);
    candidate
}

/// Execute one ABC generation step.
///
/// Modifies `state` in-place; `fitness_fn` must return the raw objective value.
pub fn abc_step<F: Fn(&[f64]) -> f64>(state: &mut AbcState, fitness_fn: &F, rng: &mut LcgRng) {
    let n_food = state.food.len();
    let n_dims = state.bounds.len();
    let bounds = state.bounds.clone();

    // ── Phase 1: Employed bees ─────────────────────────────────────────────
    for i in 0..n_food {
        if n_food < 2 {
            break;
        }
        let candidate = generate_candidate(i, &state.food, &bounds, n_food, rng);
        let raw_cand = fitness_fn(&candidate);
        let fit_cand = fitness_transform(raw_cand);

        // Greedy selection
        if fit_cand >= state.fitness[i] {
            state.food[i] = candidate;
            state.fitness[i] = fit_cand;
            state.trial[i] = 0;
            if raw_cand < state.best_fitness {
                state.best_fitness = raw_cand;
                state.best = state.food[i].clone();
            }
        } else {
            state.trial[i] += 1;
        }
    }

    // ── Phase 2: Onlooker bees ──────────────────────────────────────────────
    // n_bees onlooker bees each select a food source proportional to fitness.
    let total_fit: f64 = state.fitness.iter().sum::<f64>().max(1e-300);
    let n_onlookers = n_food; // same as number of employed bees

    for _ in 0..n_onlookers {
        if n_food < 2 {
            break;
        }
        // Roulette wheel selection
        let r = rng.next_f64() * total_fit;
        let mut cumulative = 0.0;
        let mut selected = 0;
        for s in 0..n_food {
            cumulative += state.fitness[s];
            if cumulative >= r {
                selected = s;
                break;
            }
        }

        let candidate = generate_candidate(selected, &state.food, &bounds, n_food, rng);
        let raw_cand = fitness_fn(&candidate);
        let fit_cand = fitness_transform(raw_cand);

        if fit_cand >= state.fitness[selected] {
            state.food[selected] = candidate;
            state.fitness[selected] = fit_cand;
            state.trial[selected] = 0;
            if raw_cand < state.best_fitness {
                state.best_fitness = raw_cand;
                state.best = state.food[selected].clone();
            }
        } else {
            state.trial[selected] += 1;
        }
    }

    // ── Phase 3: Scout bees ─────────────────────────────────────────────────
    // Abandon and reinitialise food sources that have exceeded the trial limit.
    // Karaboga specifies at most one scout per cycle, but many implementations
    // process all exhausted sources.
    for i in 0..n_food {
        if state.trial[i] > state.limit {
            let new_source: Vec<f64> = (0..n_dims)
                .map(|d| {
                    let (lb, ub) = bounds[d];
                    lb + rng.next_f64() * (ub - lb)
                })
                .collect();
            let raw_new = fitness_fn(&new_source);
            state.food[i] = new_source;
            state.fitness[i] = fitness_transform(raw_new);
            state.trial[i] = 0;
            if raw_new < state.best_fitness {
                state.best_fitness = raw_new;
                state.best = state.food[i].clone();
            }
        }
    }

    state.generation += 1;
}

/// Run ABC optimization.
///
/// Returns the final `AbcState` containing `best` (best decision variable vector)
/// and `best_fitness` (raw objective value).
pub fn abc_run<F>(fitness_fn: F, bounds: &[(f64, f64)], cfg: &AbcConfig) -> EvolResult<AbcState>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
    }
    if cfg.n_food == 0 {
        return Err(EvolError::SwarmEmpty);
    }
    if cfg.n_food < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: cfg.n_food,
            op: "ABC",
        });
    }

    let mut rng = LcgRng::new(cfg.seed);
    let mut state = AbcState::new(
        bounds.to_vec(),
        cfg.n_food,
        cfg.limit,
        &fitness_fn,
        &mut rng,
    )?;

    for _ in 0..cfg.max_iter {
        abc_step(&mut state, &fitness_fn, &mut rng);
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper objective functions ─────────────────────────────────────────────

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
        let sum_sq = x.iter().map(|&xi| xi * xi).sum::<f64>();
        let sum_cos = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + std::f64::consts::E
            + 20.0
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    // ── Config / construction tests ───────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = AbcConfig::new(10, 3).unwrap();
        assert_eq!(cfg.n_food, 10);
        assert_eq!(cfg.n_bees, 10);
        assert_eq!(cfg.limit, 30);
    }

    #[test]
    fn test_config_new_zero_food() {
        assert!(AbcConfig::new(0, 3).is_err());
    }

    #[test]
    fn test_config_new_zero_dims() {
        assert!(AbcConfig::new(5, 0).is_err());
    }

    // ── State construction tests ──────────────────────────────────────────────

    #[test]
    fn test_state_new_correct_sizes() {
        let bounds = vec![(0.0f64, 1.0f64); 3];
        let mut rng = LcgRng::new(0);
        let state = AbcState::new(bounds, 5, 15, &sphere, &mut rng).unwrap();
        assert_eq!(state.food.len(), 5);
        assert_eq!(state.fitness.len(), 5);
        assert_eq!(state.trial.len(), 5);
        assert_eq!(state.food[0].len(), 3);
        assert_eq!(state.trial.iter().sum::<usize>(), 0);
    }

    #[test]
    fn test_state_new_fitness_in_range() {
        let bounds = vec![(-5.0f64, 5.0f64); 4];
        let mut rng = LcgRng::new(42);
        let state = AbcState::new(bounds, 8, 32, &sphere, &mut rng).unwrap();
        for &f in &state.fitness {
            assert!(f > 0.0 && f <= 1.0, "fitness out of range: {f}");
        }
    }

    #[test]
    fn test_state_new_best_valid() {
        let bounds = vec![(0.0f64, 1.0f64); 2];
        let mut rng = LcgRng::new(7);
        let state = AbcState::new(bounds, 4, 8, &sphere, &mut rng).unwrap();
        assert!(state.best_fitness.is_finite());
        assert!(state.best_fitness >= 0.0);
    }

    // ── Fitness transform test ────────────────────────────────────────────────

    #[test]
    fn test_fitness_transform() {
        // Raw value 0 → fitness 1.0
        assert!((fitness_transform(0.0) - 1.0).abs() < 1e-12);
        // Larger |raw| → smaller fitness
        assert!(fitness_transform(10.0) < fitness_transform(1.0));
        // Always in (0, 1]
        for v in &[-100.0f64, -1.0, 0.0, 1.0, 100.0] {
            let f = fitness_transform(*v);
            assert!(f > 0.0 && f <= 1.0, "f({v}) = {f}");
        }
    }

    // ── Step tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_abc_step_increments_generation() {
        let bounds = vec![(-5.0f64, 5.0f64); 3];
        let mut rng = LcgRng::new(1);
        let mut state = AbcState::new(bounds, 4, 12, &sphere, &mut rng).unwrap();
        assert_eq!(state.generation, 0);
        abc_step(&mut state, &sphere, &mut rng);
        assert_eq!(state.generation, 1);
        abc_step(&mut state, &sphere, &mut rng);
        assert_eq!(state.generation, 2);
    }

    #[test]
    fn test_abc_step_preserves_best() {
        let bounds = vec![(-5.0f64, 5.0f64); 2];
        let mut rng = LcgRng::new(100);
        let mut state = AbcState::new(bounds, 6, 12, &sphere, &mut rng).unwrap();
        let initial_best = state.best_fitness;
        for _ in 0..50 {
            abc_step(&mut state, &sphere, &mut rng);
        }
        // Best can only improve or stay the same
        assert!(state.best_fitness <= initial_best + 1e-12);
    }

    // ── Full run tests ────────────────────────────────────────────────────────

    #[test]
    fn test_abc_run_sphere_1d() {
        let bounds = vec![(-10.0f64, 10.0f64)];
        let cfg = AbcConfig {
            n_bees: 10,
            n_food: 10,
            limit: 10,
            max_iter: 200,
            seed: 42,
        };
        let state = abc_run(sphere, &bounds, &cfg).unwrap();
        assert!(
            state.best_fitness < 5.0,
            "best_fitness = {}",
            state.best_fitness
        );
        assert_eq!(state.best.len(), 1);
    }

    #[test]
    fn test_abc_run_sphere_5d() {
        let bounds = vec![(-5.0f64, 5.0f64); 5];
        let cfg = AbcConfig {
            n_bees: 20,
            n_food: 20,
            limit: 100,
            max_iter: 1000,
            seed: 7,
        };
        let state = abc_run(sphere, &bounds, &cfg).unwrap();
        let initial_random_expected_sq = 5.0f64 * 5.0 * 5.0; // rough upper bound
        assert!(
            state.best_fitness < initial_random_expected_sq,
            "best_fitness = {}",
            state.best_fitness
        );
    }

    #[test]
    fn test_abc_run_reduces_ackley() {
        let bounds = vec![(-5.0f64, 5.0f64); 3];
        let cfg = AbcConfig {
            n_bees: 20,
            n_food: 20,
            limit: 60,
            max_iter: 1000,
            seed: 11,
        };
        let worst_possible = ackley(&[5.0, 5.0, 5.0]);
        let state = abc_run(ackley, &bounds, &cfg).unwrap();
        assert!(
            state.best_fitness < worst_possible,
            "best={} worst={}",
            state.best_fitness,
            worst_possible
        );
    }

    #[test]
    fn test_abc_run_rosenbrock_2d() {
        let bounds = vec![(-2.0f64, 2.0f64); 2];
        let cfg = AbcConfig {
            n_bees: 20,
            n_food: 20,
            limit: 40,
            max_iter: 2000,
            seed: 77,
        };
        let state = abc_run(rosenbrock, &bounds, &cfg).unwrap();
        // Rosenbrock global min = 0 at (1,1). Should do better than 200.
        assert!(state.best_fitness < 200.0, "best = {}", state.best_fitness);
    }

    #[test]
    fn test_abc_run_rastrigin_3d_reduces() {
        let bounds = vec![(-5.12f64, 5.12f64); 3];
        let worst = rastrigin(&[5.12, 5.12, 5.12]);
        let cfg = AbcConfig {
            n_bees: 20,
            n_food: 20,
            limit: 60,
            max_iter: 1000,
            seed: 33,
        };
        let state = abc_run(rastrigin, &bounds, &cfg).unwrap();
        assert!(
            state.best_fitness < worst,
            "best={} worst={}",
            state.best_fitness,
            worst
        );
    }

    #[test]
    fn test_abc_run_error_empty_bounds() {
        let cfg = AbcConfig {
            n_bees: 5,
            n_food: 5,
            limit: 10,
            max_iter: 100,
            seed: 0,
        };
        assert!(abc_run(sphere, &[], &cfg).is_err());
    }

    #[test]
    fn test_abc_run_error_zero_food() {
        let bounds = vec![(0.0f64, 1.0f64); 2];
        let cfg = AbcConfig {
            n_bees: 0,
            n_food: 0,
            limit: 10,
            max_iter: 100,
            seed: 0,
        };
        assert!(abc_run(sphere, &bounds, &cfg).is_err());
    }

    #[test]
    fn test_abc_run_best_within_bounds() {
        let bounds: Vec<(f64, f64)> = vec![(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)];
        let cfg = AbcConfig {
            n_bees: 15,
            n_food: 15,
            limit: 45,
            max_iter: 500,
            seed: 21,
        };
        let state = abc_run(sphere, &bounds, &cfg).unwrap();
        for (d, &x) in state.best.iter().enumerate() {
            let (lb, ub) = bounds[d];
            assert!(x >= lb && x <= ub, "dim {d}: {x} out of [{lb}, {ub}]");
        }
    }

    #[test]
    fn test_abc_run_deterministic() {
        let bounds = vec![(-5.0f64, 5.0f64); 3];
        let cfg = AbcConfig {
            n_bees: 10,
            n_food: 10,
            limit: 30,
            max_iter: 100,
            seed: 999,
        };
        let s1 = abc_run(sphere, &bounds, &cfg).unwrap();
        let s2 = abc_run(sphere, &bounds, &cfg).unwrap();
        assert_eq!(s1.best_fitness, s2.best_fitness, "runs not deterministic");
    }
}
