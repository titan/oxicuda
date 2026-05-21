//! Cooperative and Competitive Coevolution.
//!
//! Coevolutionary algorithms decompose the search space across multiple species
//! (subpopulations), each of which evolves a partial solution. Fitness evaluation
//! requires combining representatives from all species.
//!
//! # Cooperative mode
//! Species collaborate: each individual's fitness is assessed by concatenating its
//! genes with the current best (or random collaborators) from every other species
//! and calling the joint fitness function.
//!
//! # Competitive mode
//! Species compete adversarially: each species' individual gains fitness by
//! performing well when the other species use their *worst* representatives,
//! modelling a predator-prey or zero-sum dynamic.
//!
//! # References
//! - Potter, M. A. & De Jong, K. A. (1994). "A cooperative coevolutionary approach
//!   to function optimization." *Proc. PPSN III*, pp. 249–257.
//! - Rosin, C. D. & Belew, R. K. (1997). "New methods for competitive coevolution."
//!   *Evolutionary Computation*, 5(1), 1–29.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the coevolutionary interaction model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoevolMode {
    /// Species collaborate: the joint genome = concatenation of best from each species.
    /// Fitness rewards cooperation.
    Cooperative,
    /// Species compete: an individual's fitness is how well it does when other species
    /// use their *worst* representative (adversarial selection pressure).
    Competitive,
}

/// Configuration for a coevolutionary run.
#[derive(Debug, Clone)]
pub struct CoevolConfig {
    /// Cooperative or Competitive dynamics.
    pub mode: CoevolMode,
    /// Number of subpopulations (species). Must be ≥ 1.
    pub n_species: usize,
    /// Population size per species. Must be ≥ 2.
    pub pop_per_species: usize,
    /// Number of decision variables per species. Must be ≥ 1.
    pub genes_per_species: usize,
    /// Total number of generations.
    pub n_gens: usize,
    /// Standard deviation for Gaussian mutation.
    pub sigma_mut: f64,
    /// Per-gene mutation probability ∈ (0, 1].
    pub p_mut: f64,
    /// Lower bound for all genes.
    pub lb: f64,
    /// Upper bound for all genes.
    pub ub: f64,
    /// Number of random collaborator evaluations per individual per generation.
    /// Must be ≥ 1. When > 1, the fitness is averaged across random collaborators.
    pub n_collaborators: usize,
}

impl CoevolConfig {
    /// Validate all configuration parameters; return the first error found.
    pub fn validate(&self) -> EvolResult<()> {
        if self.n_species == 0 {
            return Err(EvolError::InvalidParameter(
                "CoevolConfig: n_species must be >= 1".to_owned(),
            ));
        }
        if self.pop_per_species < 2 {
            return Err(EvolError::InvalidParameter(
                "CoevolConfig: pop_per_species must be >= 2".to_owned(),
            ));
        }
        if self.genes_per_species == 0 {
            return Err(EvolError::InvalidParameter(
                "CoevolConfig: genes_per_species must be >= 1".to_owned(),
            ));
        }
        if self.lb >= self.ub {
            return Err(EvolError::InvalidParameter(format!(
                "CoevolConfig: lb ({}) must be < ub ({})",
                self.lb, self.ub
            )));
        }
        if self.sigma_mut <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "CoevolConfig: sigma_mut must be > 0".to_owned(),
            ));
        }
        if !(self.p_mut > 0.0 && self.p_mut <= 1.0) {
            return Err(EvolError::InvalidParameter(format!(
                "CoevolConfig: p_mut ({}) must be in (0, 1]",
                self.p_mut
            )));
        }
        if self.n_collaborators == 0 {
            return Err(EvolError::InvalidParameter(
                "CoevolConfig: n_collaborators must be >= 1".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Results of a completed coevolutionary run.
#[derive(Debug, Clone)]
pub struct CoevolResult {
    /// Concatenation of the best individual from each species.
    /// Length = n_species * genes_per_species.
    pub best_combined: Vec<f64>,
    /// Joint fitness of `best_combined`.
    pub best_fitness: f64,
    /// Best individual (gene vector) for each species.
    pub species_bests: Vec<Vec<f64>>,
    /// Best combined fitness recorded after each generation (length == n_gens).
    pub history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// k=3 tournament selection index (minimization).
fn tournament_k3(fitness: &[f64], rng: &mut LcgRng) -> usize {
    let n = fitness.len();
    let a = rng.next_usize(n);
    let b = rng.next_usize(n);
    let c = rng.next_usize(n);
    let ab = if fitness[a] <= fitness[b] { a } else { b };
    if fitness[ab] <= fitness[c] { ab } else { c }
}

/// Return the index of the individual with the lowest fitness (best in minimization).
fn argmin_fit(fitness: &[f64]) -> usize {
    fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Return the index of the individual with the highest fitness (worst in minimization).
fn argmax_fit(fitness: &[f64]) -> usize {
    fitness
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Gaussian mutation clamped to `[lb, ub]`.
fn gauss_mutate(
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
                (g + sigma * rng.next_normal()).clamp(lb, ub)
            } else {
                g
            }
        })
        .collect()
}

/// (μ+λ) selection: merge parents and offspring, keep the best `keep` individuals.
fn mu_plus_lambda(
    pop: &[Vec<f64>],
    pop_fit: &[f64],
    offspring: &[Vec<f64>],
    off_fit: &[f64],
    keep: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut combined: Vec<(f64, Vec<f64>)> = pop
        .iter()
        .zip(pop_fit.iter())
        .map(|(g, &f)| (f, g.clone()))
        .chain(
            offspring
                .iter()
                .zip(off_fit.iter())
                .map(|(g, &f)| (f, g.clone())),
        )
        .collect();
    combined.sort_unstable_by(|(fa, _), (fb, _)| {
        fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(keep);
    let new_pop: Vec<Vec<f64>> = combined.iter().map(|(_, g)| g.clone()).collect();
    let new_fit: Vec<f64> = combined.iter().map(|(f, _)| *f).collect();
    (new_pop, new_fit)
}

/// Build the joint genome by concatenating one representative from each species.
/// `reps[s]` is the index into `pops[s]` to use for species `s`.
fn build_joint(pops: &[Vec<Vec<f64>>], reps: &[usize]) -> Vec<f64> {
    let mut joint = Vec::new();
    for (s, pop) in pops.iter().enumerate() {
        joint.extend_from_slice(&pop[reps[s]]);
    }
    joint
}

/// Evaluate a single individual `(species=s, index=ind_idx)` in cooperative mode.
/// The joint genome is: concatenation of `individual` (for species s) with the
/// best individual from every other species. When `n_collaborators > 1`, the fitness
/// is averaged across multiple random choices of collaborators from each other species.
fn eval_cooperative<F: Fn(&[f64]) -> f64>(
    ind_genes: &[f64],
    s: usize,
    pops: &[Vec<Vec<f64>>],
    bests: &[usize],
    n_collabs: usize,
    genes_per_species: usize,
    fitness_fn: &F,
    rng: &mut LcgRng,
) -> f64 {
    if n_collabs == 1 {
        // Use the current best from each other species
        let mut joint = Vec::with_capacity(pops.len() * genes_per_species);
        for (t, pop) in pops.iter().enumerate() {
            if t == s {
                joint.extend_from_slice(ind_genes);
            } else {
                joint.extend_from_slice(&pop[bests[t]]);
            }
        }
        fitness_fn(&joint)
    } else {
        // Average across n_collabs random collaborator sets
        let mut total = 0.0;
        for _ in 0..n_collabs {
            let mut joint = Vec::with_capacity(pops.len() * genes_per_species);
            for (t, pop) in pops.iter().enumerate() {
                if t == s {
                    joint.extend_from_slice(ind_genes);
                } else {
                    let collab_idx = rng.next_usize(pop.len());
                    joint.extend_from_slice(&pop[collab_idx]);
                }
            }
            total += fitness_fn(&joint);
        }
        total / n_collabs as f64
    }
}

/// Evaluate a single individual `(species=s, index=ind_idx)` in competitive mode.
/// Each species uses its *worst* representative from every other species (adversarial).
fn eval_competitive<F: Fn(&[f64]) -> f64>(
    ind_genes: &[f64],
    s: usize,
    pops: &[Vec<Vec<f64>>],
    worsts: &[usize],
    genes_per_species: usize,
    fitness_fn: &F,
) -> f64 {
    let mut joint = Vec::with_capacity(pops.len() * genes_per_species);
    for (t, pop) in pops.iter().enumerate() {
        if t == s {
            joint.extend_from_slice(ind_genes);
        } else {
            joint.extend_from_slice(&pop[worsts[t]]);
        }
    }
    fitness_fn(&joint)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run cooperative or competitive coevolution, minimizing `fitness_fn`.
///
/// # Algorithm (Cooperative)
/// 1. Initialise each species' subpopulation independently in `[lb, ub]^genes_per_species`.
/// 2. For each generation:
///    a. Evaluate each individual by concatenating its genes with the current best (or random
///    collaborators) from every other species.
///    b. Tournament select (k=3) + Gaussian mutate to produce offspring.
///    c. (μ+λ) selection within each species.
///    d. Track the best combined genome and record in history.
///
/// # Algorithm (Competitive)
/// Same as cooperative but each species' individual is evaluated against the *worst*
/// individual from every other species (adversarial pressure).
///
/// # Errors
/// Returns `EvolError::InvalidParameter` if the configuration is invalid.
pub fn coevolve<F>(
    config: &CoevolConfig,
    fitness_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<CoevolResult>
where
    F: Fn(&[f64]) -> f64,
{
    config.validate()?;

    let n_s = config.n_species;
    let ps = config.pop_per_species;
    let gs = config.genes_per_species;
    let range = config.ub - config.lb;

    // ── Initialise subpopulations ─────────────────────────────────────────────
    let mut pops: Vec<Vec<Vec<f64>>> = (0..n_s)
        .map(|_| {
            (0..ps)
                .map(|_| {
                    (0..gs)
                        .map(|_| config.lb + rng.next_f64() * range)
                        .collect()
                })
                .collect()
        })
        .collect();

    // ── Compute best/worst indices per species ────────────────────────────────
    // We need an initial fitness evaluation to determine bests.
    // For first evaluation, use the first individual from each other species.
    let mut bests: Vec<usize> = vec![0; n_s];
    let mut worsts: Vec<usize> = vec![0; n_s];

    // Initial fitness: evaluate every individual with initial bests = index 0 for each species
    let mut fitnesses: Vec<Vec<f64>> = (0..n_s)
        .map(|s| {
            (0..ps)
                .map(|ind| match config.mode {
                    CoevolMode::Cooperative => eval_cooperative(
                        &pops[s][ind],
                        s,
                        &pops,
                        &bests,
                        config.n_collaborators,
                        gs,
                        &fitness_fn,
                        rng,
                    ),
                    CoevolMode::Competitive => {
                        eval_competitive(&pops[s][ind], s, &pops, &worsts, gs, &fitness_fn)
                    }
                })
                .collect()
        })
        .collect();

    // Update bests and worsts after initial evaluation
    for s in 0..n_s {
        bests[s] = argmin_fit(&fitnesses[s]);
        worsts[s] = argmax_fit(&fitnesses[s]);
    }

    let mut history = Vec::with_capacity(config.n_gens);
    let mut global_best_f = f64::INFINITY;
    let mut global_best_combined: Vec<f64> = build_joint(&pops, &bests);

    // ── Main loop ─────────────────────────────────────────────────────────────
    for _gen in 0..config.n_gens {
        // Evolve each species independently
        for s in 0..n_s {
            let pop_size = pops[s].len();
            let mut offspring = Vec::with_capacity(pop_size);
            let mut off_fit = Vec::with_capacity(pop_size);

            for _ in 0..pop_size {
                let parent_idx = tournament_k3(&fitnesses[s], rng);
                let child = gauss_mutate(
                    &pops[s][parent_idx],
                    config.sigma_mut,
                    config.p_mut,
                    config.lb,
                    config.ub,
                    rng,
                );
                let child_fit = match config.mode {
                    CoevolMode::Cooperative => eval_cooperative(
                        &child,
                        s,
                        &pops,
                        &bests,
                        config.n_collaborators,
                        gs,
                        &fitness_fn,
                        rng,
                    ),
                    CoevolMode::Competitive => {
                        eval_competitive(&child, s, &pops, &worsts, gs, &fitness_fn)
                    }
                };
                offspring.push(child);
                off_fit.push(child_fit);
            }

            let (new_pop, new_fit) =
                mu_plus_lambda(&pops[s], &fitnesses[s], &offspring, &off_fit, pop_size);
            pops[s] = new_pop;
            fitnesses[s] = new_fit;

            // Update best and worst for this species
            bests[s] = argmin_fit(&fitnesses[s]);
            worsts[s] = argmax_fit(&fitnesses[s]);
        }

        // Evaluate the combined best genome
        let combined = build_joint(&pops, &bests);
        let combined_fit = fitness_fn(&combined);

        if combined_fit < global_best_f {
            global_best_f = combined_fit;
            global_best_combined = combined;
        }

        history.push(global_best_f);
    }

    // Collect per-species bests
    let species_bests: Vec<Vec<f64>> = (0..n_s).map(|s| pops[s][bests[s]].clone()).collect();

    Ok(CoevolResult {
        best_combined: global_best_combined,
        best_fitness: global_best_f,
        species_bests,
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

    fn default_config() -> CoevolConfig {
        CoevolConfig {
            mode: CoevolMode::Cooperative,
            n_species: 3,
            pop_per_species: 20,
            genes_per_species: 2,
            n_gens: 30,
            sigma_mut: 0.3,
            p_mut: 0.3,
            lb: -5.0,
            ub: 5.0,
            n_collaborators: 1,
        }
    }

    // ── Config validation ──────────────────────────────────────────────────────

    #[test]
    fn test_config_n_species_zero_errors() {
        let mut cfg = default_config();
        cfg.n_species = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_pop_per_species_one_errors() {
        let mut cfg = default_config();
        cfg.pop_per_species = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_genes_per_species_zero_errors() {
        let mut cfg = default_config();
        cfg.genes_per_species = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_n_collaborators_zero_errors() {
        let mut cfg = default_config();
        cfg.n_collaborators = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_bounds_errors() {
        let mut cfg = default_config();
        cfg.lb = 2.0;
        cfg.ub = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_sigma_zero_errors() {
        let mut cfg = default_config();
        cfg.sigma_mut = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_p_mut_zero_errors() {
        let mut cfg = default_config();
        cfg.p_mut = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ── Cooperative mode ───────────────────────────────────────────────────────

    #[test]
    fn test_cooperative_species_bests_length() {
        let cfg = default_config();
        let mut rng = LcgRng::new(42);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.species_bests.len(), cfg.n_species);
    }

    #[test]
    fn test_cooperative_combined_length() {
        let cfg = default_config();
        let mut rng = LcgRng::new(7);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(
            result.best_combined.len(),
            cfg.n_species * cfg.genes_per_species
        );
    }

    #[test]
    fn test_cooperative_history_length() {
        let cfg = default_config();
        let mut rng = LcgRng::new(11);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.history.len(), cfg.n_gens);
    }

    #[test]
    fn test_cooperative_history_non_increasing() {
        let cfg = default_config();
        let mut rng = LcgRng::new(13);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        for w in result.history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "history increased: {} → {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_cooperative_separable_sphere_converges() {
        // Each species handles one dimension; the joint sphere should converge toward 0
        // Species 0: x[0..2], Species 1: x[2..4], Species 2: x[4..6]
        let mut cfg = default_config();
        cfg.n_gens = 80;
        cfg.pop_per_species = 30;
        cfg.sigma_mut = 0.5;
        let mut rng = LcgRng::new(99);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert!(
            result.best_fitness < 5.0,
            "cooperative sphere did not converge: best = {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_cooperative_multi_collaborators_runs() {
        let mut cfg = default_config();
        cfg.n_collaborators = 3;
        let mut rng = LcgRng::new(21);
        let result = coevolve(&cfg, sphere, &mut rng);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.best_fitness.is_finite());
    }

    #[test]
    fn test_cooperative_single_species_degenerates_to_ga() {
        // With a single species, coevolution reduces to a standard GA
        let mut cfg = default_config();
        cfg.n_species = 1;
        cfg.genes_per_species = 4;
        let mut rng = LcgRng::new(55);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.best_combined.len(), 4);
        assert_eq!(result.species_bests.len(), 1);
        assert!(result.best_fitness.is_finite());
    }

    // ── Competitive mode ───────────────────────────────────────────────────────

    #[test]
    fn test_competitive_runs_without_error() {
        let mut cfg = default_config();
        cfg.mode = CoevolMode::Competitive;
        let mut rng = LcgRng::new(33);
        let result = coevolve(&cfg, sphere, &mut rng);
        assert!(result.is_ok(), "competitive coevolve failed: {:?}", result);
        let r = result.unwrap();
        assert!(r.best_fitness.is_finite());
    }

    #[test]
    fn test_competitive_history_length() {
        let mut cfg = default_config();
        cfg.mode = CoevolMode::Competitive;
        let mut rng = LcgRng::new(77);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.history.len(), cfg.n_gens);
    }

    #[test]
    fn test_competitive_combined_genome_length() {
        let mut cfg = default_config();
        cfg.mode = CoevolMode::Competitive;
        cfg.n_species = 4;
        cfg.genes_per_species = 3;
        let mut rng = LcgRng::new(88);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        assert_eq!(result.best_combined.len(), 4 * 3);
    }

    #[test]
    fn test_result_history_consistent_with_best_fitness() {
        let cfg = default_config();
        let mut rng = LcgRng::new(123);
        let result = coevolve(&cfg, sphere, &mut rng).unwrap();
        // The last history entry should equal best_fitness
        let last = *result.history.last().unwrap();
        assert!(
            (last - result.best_fitness).abs() < 1e-10,
            "history last ({last}) != best_fitness ({})",
            result.best_fitness
        );
    }
}
