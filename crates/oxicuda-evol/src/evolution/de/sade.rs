//! Self-adaptive Differential Evolution (SaDE) implementation.
//!
//! Adapts both mutation strategy probabilities and crossover rate (CR) means
//! per learning period LP. The scale factor F is sampled from Cauchy(0.5, 0.1)
//! each trial.
//!
//! Reference: A.K. Qin & P.N. Suganthan, "Self-adaptive Differential Evolution
//! Algorithm for Numerical Optimization", IEEE CEC 2005.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of mutation strategies in SaDE.
const N_STRATEGIES: usize = 4;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sample from Cauchy(center, scale) via the tangent formula.
fn sample_cauchy(center: f64, scale: f64, rng: &mut LcgRng) -> f64 {
    let u = (rng.next_f64()).clamp(1e-6, 1.0 - 1e-6);
    center + scale * (std::f64::consts::PI * (u - 0.5)).tan()
}

/// Sample from N(crm, 0.1) clamped to [0, 1].
fn sample_cr(crm: f64, rng: &mut LcgRng) -> f64 {
    let n = rng.next_normal();
    (crm + 0.1 * n).clamp(0.0, 1.0)
}

/// Select a strategy index by sampling from cumulative `strategy_probs`.
fn select_strategy(probs: &[f64], rng: &mut LcgRng) -> usize {
    let u = rng.next_f64();
    let mut cum = 0.0;
    for (k, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return k;
        }
    }
    N_STRATEGIES - 1
}

/// Pick `count` distinct indices from `[0, n)`, excluding all indices in `exclude`.
///
/// Uses Fisher-Yates partial shuffle on the eligible pool.
fn pick_distinct(n: usize, exclude: &[usize], count: usize, rng: &mut LcgRng) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).filter(|i| !exclude.contains(i)).collect();
    let take = count.min(pool.len());
    for i in 0..take {
        let j = i + rng.next_usize(pool.len() - i);
        pool.swap(i, j);
    }
    pool[..take].to_vec()
}

/// Binomial crossover: ensure at least one gene is taken from `donor` (j_rand).
fn crossover(target: &[f64], donor: &[f64], cr: f64, rng: &mut LcgRng) -> Vec<f64> {
    let dim = target.len();
    let j_rand = rng.next_usize(dim);
    (0..dim)
        .map(|d| {
            if d == j_rand || rng.next_f64() < cr {
                donor[d]
            } else {
                target[d]
            }
        })
        .collect()
}

/// Clamp a value to the search bounds.
#[inline]
fn clamp_bounds(v: f64, lb: f64, ub: f64) -> f64 {
    v.clamp(lb, ub)
}

/// Compute the median of a mutable slice (sorts in place).
fn median_of(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.5;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) * 0.5
    }
}

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Hyper-parameters for a SaDE run.
#[derive(Debug, Clone)]
pub struct SaDeConfig {
    /// Problem dimension.
    pub n_dims: usize,
    /// Population size (must be ≥ 6).
    pub pop_size: usize,
    /// Learning period (LP) in generations. Default 50.
    pub learning_period: usize,
    /// Maximum objective evaluations.
    pub max_evals: usize,
    /// Convergence threshold on best fitness.
    pub tol: f64,
    /// Search bounds applied to every dimension.
    pub bounds: (f64, f64),
    /// RNG seed.
    pub seed: u64,
}

impl SaDeConfig {
    /// Build a sensible default `SaDeConfig` for dimension `n_dims`.
    ///
    /// pop_size = 10·n_dims, LP = 50, max_evals = 100 000, tol = 1e-6,
    /// bounds = (−5, 5), seed = 42.
    pub fn default_for(n_dims: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_dims,
            pop_size: 10 * n_dims,
            learning_period: 50,
            max_evals: 100_000,
            tol: 1e-6,
            bounds: (-5.0, 5.0),
            seed: 42,
        })
    }

    /// Validate all fields.
    fn validate(&self) -> EvolResult<()> {
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        if self.pop_size < 6 {
            return Err(EvolError::PopulationTooSmall {
                size: self.pop_size,
                op: "SaDE",
            });
        }
        if self.max_evals == 0 {
            return Err(EvolError::InvalidParameter(
                "max_evals must be >= 1".to_owned(),
            ));
        }
        if self.bounds.0 >= self.bounds.1 {
            return Err(EvolError::InvalidParameter(
                "bounds: lower must be < upper".to_owned(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public state
// ---------------------------------------------------------------------------

/// Mutable SaDE population state (completely separate from `DeState`).
pub struct SaDeState {
    /// Population matrix: `pop_size × n_dims`.
    pub population: Vec<Vec<f64>>,
    /// Current fitness of each individual.
    pub fitness: Vec<f64>,
    /// Index of the best individual.
    pub best_idx: usize,
    /// Total objective evaluations consumed.
    pub n_evals: usize,
    /// Generation counter (0-based).
    pub generation: usize,

    // --- Strategy self-adaptation ---
    /// Probability of choosing each of the N_STRATEGIES strategies.
    strategy_probs: Vec<f64>,
    /// Per-strategy mean crossover rate CRm_k.
    cr_means: Vec<f64>,
    /// Success count per strategy in the current learning period.
    success_counts: Vec<usize>,
    /// Failure count per strategy in the current learning period.
    failure_counts: Vec<usize>,
    /// List of successful CRs per strategy in the current learning period.
    success_cr: Vec<Vec<f64>>,
}

impl SaDeState {
    /// Randomly initialise the population uniformly in bounds and evaluate.
    pub fn new<F>(config: &SaDeConfig, obj: F, rng: &mut LcgRng) -> EvolResult<Self>
    where
        F: Fn(&[f64]) -> f64,
    {
        config.validate()?;

        let (lb, ub) = config.bounds;
        let range = ub - lb;

        let population: Vec<Vec<f64>> = (0..config.pop_size)
            .map(|_| {
                (0..config.n_dims)
                    .map(|_| lb + rng.next_f64() * range)
                    .collect()
            })
            .collect();

        let fitness: Vec<f64> = population.iter().map(|ind| obj(ind)).collect();
        let n_evals = config.pop_size;

        let best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or(EvolError::EmptyPopulation)?;

        // Initialise all strategies with equal probability
        let strategy_probs = vec![1.0 / N_STRATEGIES as f64; N_STRATEGIES];
        // Initialise CR means at 0.5
        let cr_means = vec![0.5; N_STRATEGIES];

        Ok(Self {
            population,
            fitness,
            best_idx,
            n_evals,
            generation: 0,
            strategy_probs,
            cr_means,
            success_counts: vec![0; N_STRATEGIES],
            failure_counts: vec![0; N_STRATEGIES],
            success_cr: vec![Vec::new(); N_STRATEGIES],
        })
    }

    // -----------------------------------------------------------------------
    // Per-strategy mutation helpers
    // -----------------------------------------------------------------------

    /// S1: DE/rand/1/bin — `v = r1 + F*(r2 − r3)`.
    fn mutate_s1(&self, target: usize, f: f64, rng: &mut LcgRng, cfg: &SaDeConfig) -> Vec<f64> {
        let (lb, ub) = cfg.bounds;
        let r = pick_distinct(cfg.pop_size, &[target], 3, rng);
        (0..cfg.n_dims)
            .map(|d| {
                clamp_bounds(
                    self.population[r[0]][d]
                        + f * (self.population[r[1]][d] - self.population[r[2]][d]),
                    lb,
                    ub,
                )
            })
            .collect()
    }

    /// S2: DE/current-to-best/1/bin — `v = x + F*(best − x) + F*(r1 − r2)`.
    fn mutate_s2(&self, target: usize, f: f64, rng: &mut LcgRng, cfg: &SaDeConfig) -> Vec<f64> {
        let (lb, ub) = cfg.bounds;
        let r = pick_distinct(cfg.pop_size, &[target, self.best_idx], 2, rng);
        (0..cfg.n_dims)
            .map(|d| {
                clamp_bounds(
                    self.population[target][d]
                        + f * (self.population[self.best_idx][d] - self.population[target][d])
                        + f * (self.population[r[0]][d] - self.population[r[1]][d]),
                    lb,
                    ub,
                )
            })
            .collect()
    }

    /// S3: DE/rand/2/bin — `v = r1 + F*(r2 − r3) + F*(r4 − r5)`.
    fn mutate_s3(&self, target: usize, f: f64, rng: &mut LcgRng, cfg: &SaDeConfig) -> Vec<f64> {
        let (lb, ub) = cfg.bounds;
        let r = pick_distinct(cfg.pop_size, &[target], 5, rng);
        (0..cfg.n_dims)
            .map(|d| {
                clamp_bounds(
                    self.population[r[0]][d]
                        + f * (self.population[r[1]][d] - self.population[r[2]][d])
                        + f * (self.population[r[3]][d] - self.population[r[4]][d]),
                    lb,
                    ub,
                )
            })
            .collect()
    }

    /// S4: DE/current-to-rand/1 — `v = x + F*(r1 − x) + F*(r2 − r3)` (no crossover).
    fn mutate_s4(&self, target: usize, f: f64, rng: &mut LcgRng, cfg: &SaDeConfig) -> Vec<f64> {
        let (lb, ub) = cfg.bounds;
        let r = pick_distinct(cfg.pop_size, &[target], 3, rng);
        (0..cfg.n_dims)
            .map(|d| {
                clamp_bounds(
                    self.population[target][d]
                        + f * (self.population[r[0]][d] - self.population[target][d])
                        + f * (self.population[r[1]][d] - self.population[r[2]][d]),
                    lb,
                    ub,
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // LP (learning period) update
    // -----------------------------------------------------------------------

    fn update_strategy_probs(&mut self) {
        // Compute ns/(ns+nf) for each strategy; guard divide-by-zero.
        let mut raw: Vec<f64> = (0..N_STRATEGIES)
            .map(|k| {
                let ns = self.success_counts[k] as f64;
                let nf = self.failure_counts[k] as f64;
                let denom = ns + nf;
                if denom > 0.0 { ns / denom } else { 0.0 }
            })
            .collect();

        let total: f64 = raw.iter().sum();
        if total > 0.0 {
            for p in raw.iter_mut() {
                *p = (*p / total).clamp(0.01, 0.99);
            }
            // Re-normalise after clipping
            let sum2: f64 = raw.iter().sum();
            for p in raw.iter_mut() {
                *p /= sum2;
            }
            self.strategy_probs = raw;
        }
        // If total == 0 (no trials yet), keep existing probs.
    }

    fn update_cr_means(&mut self) {
        for k in 0..N_STRATEGIES {
            if !self.success_cr[k].is_empty() {
                self.cr_means[k] = median_of(&mut self.success_cr[k]);
            }
        }
    }

    fn reset_lp_counters(&mut self) {
        for k in 0..N_STRATEGIES {
            self.success_counts[k] = 0;
            self.failure_counts[k] = 0;
            self.success_cr[k].clear();
        }
    }

    // -----------------------------------------------------------------------
    // Public step / run
    // -----------------------------------------------------------------------

    /// Execute one SaDE generation. Returns the best fitness after the step.
    pub fn step<F>(&mut self, config: &SaDeConfig, obj: F, rng: &mut LcgRng) -> EvolResult<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let pop_size = config.pop_size;

        for target in 0..pop_size {
            // --- Sample strategy, F, and CR ---
            let strat = select_strategy(&self.strategy_probs, rng);
            let f_val = sample_cauchy(0.5, 0.1, rng).clamp(f64::MIN_POSITIVE, 2.0);
            let cr_val = sample_cr(self.cr_means[strat], rng);

            // --- Mutation ---
            let donor = match strat {
                0 => self.mutate_s1(target, f_val, rng, config),
                1 => self.mutate_s2(target, f_val, rng, config),
                2 => self.mutate_s3(target, f_val, rng, config),
                _ => self.mutate_s4(target, f_val, rng, config),
            };

            // --- Crossover (S4 is already a final trial; S1-S3 apply binomial) ---
            let trial = if strat < 3 {
                crossover(&self.population[target], &donor, cr_val, rng)
            } else {
                donor // S4: no crossover
            };

            // --- Evaluate and select ---
            let trial_fit = obj(&trial);
            self.n_evals += 1;

            if trial_fit <= self.fitness[target] {
                self.population[target] = trial;
                self.fitness[target] = trial_fit;
                self.success_counts[strat] += 1;
                if strat < 3 {
                    self.success_cr[strat].push(cr_val);
                }
                // Update best
                if trial_fit < self.fitness[self.best_idx] {
                    self.best_idx = target;
                }
            } else {
                self.failure_counts[strat] += 1;
            }
        }

        self.generation += 1;

        // --- LP update ---
        if self.generation.is_multiple_of(config.learning_period) {
            self.update_strategy_probs();
            self.update_cr_means();
            self.reset_lp_counters();
        }

        Ok(self.fitness[self.best_idx])
    }

    /// Run SaDE to convergence or budget exhaustion.
    ///
    /// Returns `(best_individual, best_fitness)`.
    pub fn run<F>(config: &SaDeConfig, obj: F) -> EvolResult<(Vec<f64>, f64)>
    where
        F: Fn(&[f64]) -> f64 + Clone,
    {
        config.validate()?;
        let mut rng = LcgRng::new(config.seed);
        let mut state = SaDeState::new(config, obj.clone(), &mut rng)?;

        while state.n_evals < config.max_evals {
            let best_f = state.step(config, obj.clone(), &mut rng)?;
            if best_f < config.tol {
                let best = state.population[state.best_idx].clone();
                return Ok((best, best_f));
            }
        }

        let best = state.population[state.best_idx].clone();
        let best_f = state.fitness[state.best_idx];
        Ok((best, best_f))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| {
                let a = 1.0 - w[0];
                let b = w[1] - w[0] * w[0];
                a * a + 100.0 * b * b
            })
            .sum()
    }

    fn shifted_sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum()
    }

    // Test 1: SaDeState::new initialises correctly on sphere
    #[test]
    fn test_new_initialises_correctly() {
        let config = SaDeConfig::default_for(3).expect("valid config");
        let mut rng = LcgRng::new(1);
        let state = SaDeState::new(&config, sphere, &mut rng).expect("state");
        assert_eq!(state.population.len(), config.pop_size);
        assert_eq!(state.population[0].len(), 3);
        assert_eq!(state.fitness.len(), config.pop_size);
        assert!(state.fitness.iter().all(|f| f.is_finite()));
        assert!(state.n_evals > 0);
        // strategy_probs sum to 1
        let sum: f64 = state.strategy_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // Test 2: step reduces best fitness over 100 generations on sphere
    #[test]
    fn test_step_reduces_fitness_sphere() {
        let config = SaDeConfig::default_for(5).expect("valid config");
        let mut rng = LcgRng::new(2);
        let mut state = SaDeState::new(&config, sphere, &mut rng).expect("state");
        let initial_best = state.fitness[state.best_idx];
        for _ in 0..100 {
            state.step(&config, sphere, &mut rng).expect("step ok");
        }
        let final_best = state.fitness[state.best_idx];
        assert!(
            final_best < initial_best,
            "fitness should decrease: {final_best} vs {initial_best}"
        );
    }

    // Test 3: run converges on sphere n=2
    #[test]
    fn test_run_converges_sphere_2d() {
        let mut config = SaDeConfig::default_for(2).expect("valid config");
        config.max_evals = 50_000;
        config.tol = 1e-6;
        let (_, fit) = SaDeState::run(&config, sphere).expect("run ok");
        assert!(fit < 1e-4, "sphere should converge: got {fit}");
    }

    // Test 4: run on Rosenbrock finds fitness < 0.01 for n=2
    #[test]
    fn test_run_rosenbrock_2d() {
        let mut config = SaDeConfig::default_for(2).expect("valid config");
        config.max_evals = 100_000;
        config.tol = 1e-3;
        config.bounds = (-5.0, 5.0);
        config.seed = 99;
        let (_, fit) = SaDeState::run(&config, rosenbrock).expect("run ok");
        assert!(fit < 0.01, "Rosenbrock should converge: got {fit}");
    }

    // Test 5: run on shifted sphere finds solution ≈ (1,...,1)
    #[test]
    fn test_run_shifted_sphere() {
        let mut config = SaDeConfig::default_for(3).expect("valid config");
        config.bounds = (-3.0, 5.0);
        config.max_evals = 80_000;
        config.tol = 1e-5;
        config.seed = 7;
        let (sol, fit) = SaDeState::run(&config, shifted_sphere).expect("run ok");
        assert!(fit < 1e-3, "shifted sphere should converge: got {fit}");
        for &v in &sol {
            assert!((v - 1.0).abs() < 0.1, "solution should be near 1: got {v}");
        }
    }

    // Test 6: pop_size < 6 returns error
    #[test]
    fn test_error_pop_too_small() {
        let config = SaDeConfig {
            n_dims: 3,
            pop_size: 4,
            learning_period: 50,
            max_evals: 1000,
            tol: 1e-6,
            bounds: (-5.0, 5.0),
            seed: 1,
        };
        let mut rng = LcgRng::new(1);
        let res = SaDeState::new(&config, sphere, &mut rng);
        assert!(res.is_err());
        assert!(matches!(res, Err(EvolError::PopulationTooSmall { .. })));
    }

    // Test 7: n_dims == 0 returns error
    #[test]
    fn test_error_ndims_zero() {
        let res = SaDeConfig::default_for(0);
        assert!(res.is_err());
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    // Test 8: strategy probabilities sum ≈ 1.0 after initialisation
    #[test]
    fn test_strategy_probs_sum_to_one_init() {
        let config = SaDeConfig::default_for(4).expect("valid config");
        let mut rng = LcgRng::new(3);
        let state = SaDeState::new(&config, sphere, &mut rng).expect("state");
        let sum: f64 = state.strategy_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "probs sum = {sum}");
        assert_eq!(state.strategy_probs.len(), N_STRATEGIES);
    }

    // Test 9: LP update — after LP generations, strategy_probs still sum ≈ 1.0
    #[test]
    fn test_strategy_probs_after_lp_update() {
        let mut config = SaDeConfig::default_for(4).expect("valid config");
        config.learning_period = 10;
        let mut rng = LcgRng::new(4);
        let mut state = SaDeState::new(&config, sphere, &mut rng).expect("state");
        // Run for exactly LP generations to trigger one update
        for _ in 0..config.learning_period {
            state.step(&config, sphere, &mut rng).expect("step");
        }
        let sum: f64 = state.strategy_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "probs sum after LP = {sum}");
        // Each probability should stay in a sane range
        for &p in &state.strategy_probs {
            assert!(p > 0.0 && p <= 1.0, "prob out of range: {p}");
        }
    }

    // Test 10: Different seeds give different results
    #[test]
    fn test_different_seeds_give_different_results() {
        let mut cfg_a = SaDeConfig::default_for(3).expect("valid config");
        cfg_a.max_evals = 5_000;
        cfg_a.seed = 100;

        let mut cfg_b = cfg_a.clone();
        cfg_b.seed = 200;

        let (sol_a, _) = SaDeState::run(&cfg_a, sphere).expect("run a");
        let (sol_b, _) = SaDeState::run(&cfg_b, sphere).expect("run b");

        // At least one dimension should differ between runs
        let all_same = sol_a
            .iter()
            .zip(sol_b.iter())
            .all(|(a, b)| (a - b).abs() < 1e-15);
        assert!(
            !all_same,
            "different seeds should produce different solutions"
        );
    }
}
