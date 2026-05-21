//! Memetic Algorithm (MA): genetic algorithm hybridised with local search.
//!
//! A memetic algorithm augments a population-based global search with a local-search
//! refinement (hill-climbing) applied to offspring before selection. Two inheritance
//! modes are supported:
//!
//! - **Lamarckian**: the locally-improved genome replaces the original in the population.
//! - **Baldwinian**: fitness is evaluated at the locally-improved point, but the genome
//!   in the population is kept at the original (unmodified) position.
//!
//! # Reference
//! Moscato, P. (1989). "On Evolution, Search, Optimisation, Genetic Algorithms and
//! Martial Arts: Towards Memetic Algorithms." Caltech Concurrent Computation Program
//! Report 826.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Specifies whether locally-improved solutions update the genome in the population
/// (Lamarckian inheritance) or only the evaluated fitness (Baldwinian inheritance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inheritance {
    /// The genome is replaced in-place with the locally-improved solution.
    Lamarckian,
    /// The genome stays unchanged; only the fitness reflects the improved point.
    Baldwinian,
}

/// Configuration for a memetic algorithm run.
#[derive(Debug, Clone)]
pub struct MemeticConfig {
    /// Population size (must be ≥ 2).
    pub pop_size: usize,
    /// Number of generations to run.
    pub n_gens: usize,
    /// Initial Gaussian mutation standard deviation.
    pub sigma_init: f64,
    /// Per-gene mutation probability in `(0, 1]`.
    pub p_mut: f64,
    /// Number of coordinate hill-climbing steps per offspring per generation.
    pub local_search_iters: usize,
    /// Step size used during hill-climbing local search.
    pub local_search_step: f64,
    /// Inheritance mode for locally-improved solutions.
    pub inheritance: Inheritance,
    /// Problem dimensionality (must be ≥ 1).
    pub n_dims: usize,
    /// Lower search bound applied to every dimension.
    pub lb: f64,
    /// Upper search bound applied to every dimension.
    pub ub: f64,
}

impl MemeticConfig {
    /// Validate the configuration and return an error on the first violation.
    pub fn validate(&self) -> EvolResult<()> {
        if self.pop_size < 2 {
            return Err(EvolError::InvalidParameter(
                "MemeticConfig: pop_size must be >= 2".to_owned(),
            ));
        }
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "MemeticConfig: n_dims must be >= 1".to_owned(),
            ));
        }
        if self.lb >= self.ub {
            return Err(EvolError::InvalidParameter(format!(
                "MemeticConfig: lb ({}) must be < ub ({})",
                self.lb, self.ub
            )));
        }
        if self.sigma_init <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "MemeticConfig: sigma_init must be > 0".to_owned(),
            ));
        }
        if !(self.p_mut > 0.0 && self.p_mut <= 1.0) {
            return Err(EvolError::InvalidParameter(format!(
                "MemeticConfig: p_mut ({}) must be in (0, 1]",
                self.p_mut
            )));
        }
        if self.local_search_step <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "MemeticConfig: local_search_step must be > 0".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Results of a completed memetic algorithm run.
#[derive(Debug, Clone)]
pub struct MemeticResult {
    /// The best genome found across all generations.
    pub best_genome: Vec<f64>,
    /// The fitness value of `best_genome`.
    pub best_fitness: f64,
    /// Best fitness recorded at the end of each generation (length == `n_gens`).
    pub history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// k-tournament selection over a slice of fitness values. Returns the index of the winner.
fn tournament_k3(fitness: &[f64], rng: &mut LcgRng) -> usize {
    let n = fitness.len();
    let a = rng.next_usize(n);
    let b = rng.next_usize(n);
    let c = rng.next_usize(n);
    let ab = if fitness[a] <= fitness[b] { a } else { b };
    if fitness[ab] <= fitness[c] { ab } else { c }
}

/// Gaussian mutation: each gene is perturbed with probability `p_mut`.
fn gaussian_mutate(
    genome: &[f64],
    sigma: f64,
    p_mut: f64,
    lb: f64,
    ub: f64,
    rng: &mut LcgRng,
) -> Vec<f64> {
    genome
        .iter()
        .map(|&g| {
            if rng.next_f64() < p_mut {
                (g + sigma * rng.next_normal()).max(lb).min(ub)
            } else {
                g
            }
        })
        .collect()
}

/// Coordinate-wise hill-climbing local search.
///
/// Iterates for `iters` steps; each step picks a random dimension, perturbs it
/// by `±step`, and keeps the perturbed point if it is strictly better.
///
/// Returns `(improved_genome, improved_fitness)`.
fn local_search<F: Fn(&[f64]) -> f64>(
    genome: &[f64],
    current_fitness: f64,
    iters: usize,
    step: f64,
    lb: f64,
    ub: f64,
    fitness_fn: &F,
    rng: &mut LcgRng,
) -> (Vec<f64>, f64) {
    let n = genome.len();
    let mut best_g = genome.to_vec();
    let mut best_f = current_fitness;

    for _ in 0..iters {
        let dim = rng.next_usize(n);
        let sign = if rng.next_bool() { 1.0 } else { -1.0 };
        let mut candidate = best_g.clone();
        candidate[dim] = (candidate[dim] + sign * step).max(lb).min(ub);
        let cand_f = fitness_fn(&candidate);
        if cand_f < best_f {
            best_f = cand_f;
            best_g = candidate;
        }
    }

    (best_g, best_f)
}

/// (μ+λ) selection: merge `parents` and `offspring`, keep the best `pop_size`.
fn mu_plus_lambda_select(
    pop: &[Vec<f64>],
    pop_fit: &[f64],
    offspring: &[Vec<f64>],
    off_fit: &[f64],
    pop_size: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Merge all individuals with their fitnesses.
    let mut combined: Vec<(f64, &Vec<f64>)> = pop
        .iter()
        .zip(pop_fit.iter())
        .map(|(g, &f)| (f, g))
        .chain(offspring.iter().zip(off_fit.iter()).map(|(g, &f)| (f, g)))
        .collect();

    // Partial sort: keep only the best `pop_size`.
    combined.sort_unstable_by(|(fa, _), (fb, _)| {
        fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(pop_size);

    let new_pop: Vec<Vec<f64>> = combined.iter().map(|(_, g)| (*g).clone()).collect();
    let new_fit: Vec<f64> = combined.iter().map(|(f, _)| *f).collect();
    (new_pop, new_fit)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run a memetic algorithm on a minimisation objective `fitness_fn`.
///
/// # Algorithm
/// 1. Initialise population uniformly in `[lb, ub]^n_dims`.
/// 2. For each generation:
///    a. Evaluate all individuals.
///    b. Produce `pop_size` offspring via tournament selection (k=3) + Gaussian mutation.
///    c. Apply coordinate hill-climbing local search to each offspring.
///    d. Apply inheritance policy (Lamarckian updates genome; Baldwinian keeps genome).
///    e. (μ+λ) selection: retain the best `pop_size` from parents ∪ offspring.
///    f. Record the best fitness in `history`.
///
/// # Errors
/// Returns `EvolError::InvalidParameter` if the configuration is invalid.
pub fn memetic_run<F>(
    config: &MemeticConfig,
    fitness_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<MemeticResult>
where
    F: Fn(&[f64]) -> f64,
{
    config.validate()?;

    let range = config.ub - config.lb;
    let n = config.n_dims;

    // --- Initialise population ---
    let mut pop: Vec<Vec<f64>> = (0..config.pop_size)
        .map(|_| (0..n).map(|_| config.lb + rng.next_f64() * range).collect())
        .collect();

    let mut pop_fit: Vec<f64> = pop.iter().map(|g| fitness_fn(g)).collect();

    let mut history = Vec::with_capacity(config.n_gens);

    // Track global best.
    let mut global_best_f = pop_fit.iter().copied().fold(f64::INFINITY, f64::min);
    let mut global_best_g = pop[pop_fit
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)]
    .clone();

    // --- Generational loop ---
    for _ in 0..config.n_gens {
        let mut offspring = Vec::with_capacity(config.pop_size);
        let mut off_fit = Vec::with_capacity(config.pop_size);

        for _ in 0..config.pop_size {
            // Tournament selection of parent.
            let parent_idx = tournament_k3(&pop_fit, rng);
            let parent_genome = &pop[parent_idx];

            // Gaussian mutation.
            let mutant = gaussian_mutate(
                parent_genome,
                config.sigma_init,
                config.p_mut,
                config.lb,
                config.ub,
                rng,
            );
            let mutant_fit = fitness_fn(&mutant);

            // Local search.
            let (ls_genome, ls_fit) = local_search(
                &mutant,
                mutant_fit,
                config.local_search_iters,
                config.local_search_step,
                config.lb,
                config.ub,
                &fitness_fn,
                rng,
            );

            // Inheritance policy.
            match config.inheritance {
                Inheritance::Lamarckian => {
                    // Replace genome with locally-improved version.
                    offspring.push(ls_genome);
                    off_fit.push(ls_fit);
                }
                Inheritance::Baldwinian => {
                    // Keep original mutated genome; use improved fitness for selection.
                    offspring.push(mutant);
                    off_fit.push(ls_fit);
                }
            }
        }

        // (μ+λ) selection.
        let (new_pop, new_fit) =
            mu_plus_lambda_select(&pop, &pop_fit, &offspring, &off_fit, config.pop_size);
        pop = new_pop;
        pop_fit = new_fit;

        // Track best.
        let gen_best_f = pop_fit.iter().copied().fold(f64::INFINITY, f64::min);
        if gen_best_f < global_best_f {
            global_best_f = gen_best_f;
            let best_idx = pop_fit
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            global_best_g = pop[best_idx].clone();
        }

        history.push(global_best_f);
    }

    Ok(MemeticResult {
        best_genome: global_best_g,
        best_fitness: global_best_f,
        history,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&v| v * v).sum()
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

    fn default_config() -> MemeticConfig {
        MemeticConfig {
            pop_size: 30,
            n_gens: 200,
            sigma_init: 0.3,
            p_mut: 0.3,
            local_search_iters: 20,
            local_search_step: 0.05,
            inheritance: Inheritance::Lamarckian,
            n_dims: 5,
            lb: -5.0,
            ub: 5.0,
        }
    }

    // ── Config validation tests ────────────────────────────────────────────

    #[test]
    fn test_config_pop_size_zero_errors() {
        let mut cfg = default_config();
        cfg.pop_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_pop_size_one_errors() {
        let mut cfg = default_config();
        cfg.pop_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_n_dims_zero_errors() {
        let mut cfg = default_config();
        cfg.n_dims = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_bounds_errors() {
        let mut cfg = default_config();
        cfg.lb = 1.0;
        cfg.ub = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_equal_bounds_errors() {
        let mut cfg = default_config();
        cfg.lb = 2.0;
        cfg.ub = 2.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_sigma_zero_errors() {
        let mut cfg = default_config();
        cfg.sigma_init = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_p_mut_zero_errors() {
        let mut cfg = default_config();
        cfg.p_mut = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_p_mut_above_one_errors() {
        let mut cfg = default_config();
        cfg.p_mut = 1.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_local_search_step_zero_errors() {
        let mut cfg = default_config();
        cfg.local_search_step = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ── Functional tests ───────────────────────────────────────────────────

    #[test]
    fn test_lamarckian_history_length_equals_n_gens() {
        let cfg = default_config();
        let mut rng = LcgRng::new(42);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.history.len(), cfg.n_gens);
    }

    #[test]
    fn test_baldwinian_run_completes() {
        let mut cfg = default_config();
        cfg.inheritance = Inheritance::Baldwinian;
        let mut rng = LcgRng::new(99);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.history.len(), cfg.n_gens);
        assert!(result.best_fitness.is_finite());
    }

    #[test]
    fn test_history_is_non_increasing() {
        let cfg = default_config();
        let mut rng = LcgRng::new(13);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        for w in result.history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-14,
                "history increased: {} → {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_lamarckian_sphere_converges_5d() {
        let mut cfg = default_config();
        cfg.n_gens = 1000;
        cfg.pop_size = 50;
        cfg.sigma_init = 0.5;
        cfg.local_search_iters = 50;
        cfg.local_search_step = 0.01;
        let mut rng = LcgRng::new(7);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        assert!(
            result.best_fitness < 1e-3,
            "Sphere 5D did not converge: best = {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_best_genome_within_bounds() {
        let cfg = default_config();
        let mut rng = LcgRng::new(21);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        for &g in &result.best_genome {
            assert!(
                g >= cfg.lb && g <= cfg.ub,
                "gene {g} outside [{}, {}]",
                cfg.lb,
                cfg.ub
            );
        }
    }

    #[test]
    fn test_best_genome_length_equals_n_dims() {
        let cfg = default_config();
        let mut rng = LcgRng::new(33);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.best_genome.len(), cfg.n_dims);
    }

    #[test]
    fn test_local_search_does_not_worsen_fitness() {
        // Run local_search in isolation and verify fitness never increases.
        let genome = vec![2.0, -3.0, 1.5];
        let f0 = sphere(&genome);
        let mut rng = LcgRng::new(5);
        let (_, f1) = local_search(&genome, f0, 100, 0.1, -5.0, 5.0, &sphere, &mut rng);
        assert!(f1 <= f0 + 1e-14, "local search worsened: {f0} → {f1}");
    }

    #[test]
    fn test_rosenbrock_runs_without_panic() {
        let mut cfg = default_config();
        cfg.n_dims = 2;
        cfg.lb = -2.0;
        cfg.ub = 2.0;
        cfg.n_gens = 100;
        let mut rng = LcgRng::new(111);
        let result = memetic_run(&cfg, rosenbrock, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zero_local_search_iters_still_runs() {
        let mut cfg = default_config();
        cfg.local_search_iters = 0;
        let mut rng = LcgRng::new(77);
        // Should complete normally — 0 iters of local search == pure GA.
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        assert!(result.best_fitness.is_finite());
    }

    #[test]
    fn test_single_generation_produces_result() {
        let mut cfg = default_config();
        cfg.n_gens = 1;
        let mut rng = LcgRng::new(88);
        let result = memetic_run(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.history.len(), 1);
    }
}
