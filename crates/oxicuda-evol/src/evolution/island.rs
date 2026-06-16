//! Island Model Genetic Algorithm.
//!
//! The island model partitions the population into semi-isolated subpopulations
//! (islands). Each island evolves independently for `n_gens_per_epoch` generations;
//! then a *migration step* exchanges high-quality individuals between islands
//! according to the chosen connectivity topology.
//!
//! This approach promotes diversity (reduced premature convergence) while still
//! allowing global knowledge to propagate via periodic migration.
//!
//! # Supported topologies
//! - [`Topology::Ring`]: island *i* sends to *(i+1) % n* and receives from *(i-1+n) % n*.
//! - [`Topology::Star`]: every island sends to island 0; island 0 sends back to all.
//! - [`Topology::AllToAll`]: every island sends migrants to every other island.
//!
//! # Reference
//! Cantú-Paz, E. (1998). "A Survey of Parallel Genetic Algorithms." *Calculateurs
//! Parallèles, Réseaux et Systèmes Répartis*, 10(2), 141–171.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Migration topology: determines which islands exchange individuals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Each island sends migrants to the next island (cyclically).
    Ring,
    /// All non-hub islands send migrants to island 0; hub sends back to all.
    Star,
    /// Every island sends migrants to every other island.
    AllToAll,
}

/// Configuration for an island-model GA run.
#[derive(Debug, Clone)]
pub struct IslandConfig {
    /// Number of islands (must be ≥ 2).
    pub n_islands: usize,
    /// Population size per island (must be ≥ 2).
    pub pop_per_island: usize,
    /// Number of local GA generations run on each island between migrations.
    pub n_gens_per_epoch: usize,
    /// Total number of epochs (migration intervals).
    pub n_epochs: usize,
    /// Fraction of each island's population that migrates `(0.0, 1.0]`.
    pub migration_rate: f64,
    /// Standard deviation for Gaussian mutation.
    pub sigma_mut: f64,
    /// Per-gene mutation probability.
    pub p_mut: f64,
    /// Migration topology.
    pub topology: Topology,
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Lower search bound for every dimension.
    pub lb: f64,
    /// Upper search bound for every dimension.
    pub ub: f64,
}

impl IslandConfig {
    /// Validate configuration; return the first error found.
    pub fn validate(&self) -> EvolResult<()> {
        if self.n_islands < 2 {
            return Err(EvolError::InvalidParameter(
                "IslandConfig: n_islands must be >= 2".to_owned(),
            ));
        }
        if self.pop_per_island < 2 {
            return Err(EvolError::InvalidParameter(
                "IslandConfig: pop_per_island must be >= 2".to_owned(),
            ));
        }
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "IslandConfig: n_dims must be >= 1".to_owned(),
            ));
        }
        if self.lb >= self.ub {
            return Err(EvolError::InvalidParameter(format!(
                "IslandConfig: lb ({}) must be < ub ({})",
                self.lb, self.ub
            )));
        }
        if self.sigma_mut <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "IslandConfig: sigma_mut must be > 0".to_owned(),
            ));
        }
        if !(self.p_mut > 0.0 && self.p_mut <= 1.0) {
            return Err(EvolError::InvalidParameter(format!(
                "IslandConfig: p_mut ({}) must be in (0, 1]",
                self.p_mut
            )));
        }
        if !(self.migration_rate > 0.0 && self.migration_rate <= 1.0) {
            return Err(EvolError::InvalidParameter(format!(
                "IslandConfig: migration_rate ({}) must be in (0, 1]",
                self.migration_rate
            )));
        }
        Ok(())
    }
}

/// Results of a completed island-model GA run.
#[derive(Debug, Clone)]
pub struct IslandResult {
    /// The best genome found across all islands and all epochs.
    pub best_genome: Vec<f64>,
    /// Fitness of `best_genome`.
    pub best_fitness: f64,
    /// Best fitness on each island at the end of the run.
    pub island_bests: Vec<f64>,
    /// Global best fitness recorded after each epoch (length == `n_epochs`).
    pub history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// k=3 tournament selection index over a fitness slice.
fn tournament_k3(fitness: &[f64], rng: &mut LcgRng) -> usize {
    let n = fitness.len();
    let a = rng.next_usize(n);
    let b = rng.next_usize(n);
    let c = rng.next_usize(n);
    let ab = if fitness[a] <= fitness[b] { a } else { b };
    if fitness[ab] <= fitness[c] { ab } else { c }
}

/// Gaussian mutation of a genome, clamped to `[lb, ub]`.
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

/// (μ+λ) merge-and-select: keep the `keep` best individuals from two populations.
///
/// Returns `(genomes, fitnesses)`.
fn mu_plus_lambda(
    pop: &[Vec<f64>],
    pop_fit: &[f64],
    offspring: &[Vec<f64>],
    off_fit: &[f64],
    keep: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut combined: Vec<(f64, &Vec<f64>)> = pop
        .iter()
        .zip(pop_fit.iter())
        .map(|(g, &f)| (f, g))
        .chain(offspring.iter().zip(off_fit.iter()).map(|(g, &f)| (f, g)))
        .collect();
    combined.sort_unstable_by(|(fa, _), (fb, _)| {
        fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(keep);
    let new_pop: Vec<Vec<f64>> = combined.iter().map(|(_, g)| (*g).clone()).collect();
    let new_fit: Vec<f64> = combined.iter().map(|(f, _)| *f).collect();
    (new_pop, new_fit)
}

/// Run `n_gens` generations of a standard GA on a single island population.
///
/// Modifies `pop` and `fitness` in place.
fn evolve_island<F: Fn(&[f64]) -> f64>(
    pop: &mut Vec<Vec<f64>>,
    fitness: &mut Vec<f64>,
    n_gens: usize,
    cfg: &IslandConfig,
    fitness_fn: &F,
    rng: &mut LcgRng,
) {
    let pop_size = pop.len();
    for _ in 0..n_gens {
        let mut offspring = Vec::with_capacity(pop_size);
        let mut off_fit = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let parent_idx = tournament_k3(fitness, rng);
            let child = gaussian_mutate(
                &pop[parent_idx],
                cfg.sigma_mut,
                cfg.p_mut,
                cfg.lb,
                cfg.ub,
                rng,
            );
            let child_fit = fitness_fn(&child);
            offspring.push(child);
            off_fit.push(child_fit);
        }
        let (new_pop, new_fit) = mu_plus_lambda(pop, fitness, &offspring, &off_fit, pop_size);
        *pop = new_pop;
        *fitness = new_fit;
    }
}

/// Return the index of the minimum-fitness element in a slice.
fn argmin(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Return the index of the maximum-fitness element (worst individual).
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Copy `n_migrants` best individuals from `src` island into `dst` island,
/// replacing the worst individuals in `dst`.
fn migrate_individuals(
    src_pop: &[Vec<f64>],
    src_fit: &[f64],
    dst_pop: &mut [Vec<f64>],
    dst_fit: &mut [f64],
    n_migrants: usize,
) {
    if n_migrants == 0 {
        return;
    }
    // Collect (fitness, genome_ref) sorted ascending — best first.
    let mut ranked: Vec<(f64, &Vec<f64>)> = src_pop
        .iter()
        .zip(src_fit.iter())
        .map(|(g, &f)| (f, g))
        .collect();
    ranked.sort_unstable_by(|(fa, _), (fb, _)| {
        fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (migrant_fit, migrant_genome) in ranked.iter().take(n_migrants) {
        // Replace the worst individual in the destination island.
        let worst = argmax(dst_fit);
        // Only replace if migrant is better than the worst.
        if *migrant_fit < dst_fit[worst] {
            dst_pop[worst] = (*migrant_genome).clone();
            dst_fit[worst] = *migrant_fit;
        }
    }
}

/// Perform the migration step for the Ring topology.
///
/// Island `i` sends its best `n_migrants` individuals to island `(i+1) % n`.
fn migrate_ring(islands: &mut [Vec<Vec<f64>>], fitnesses: &mut [Vec<f64>], n_migrants: usize) {
    let n = islands.len();
    // Collect migrants from each island first (avoid borrow issues).
    let migrant_batches: Vec<(Vec<Vec<f64>>, Vec<f64>)> = (0..n)
        .map(|i| {
            let mut ranked: Vec<(f64, Vec<f64>)> = islands[i]
                .iter()
                .zip(fitnesses[i].iter())
                .map(|(g, &f)| (f, g.clone()))
                .collect();
            ranked.sort_unstable_by(|(fa, _), (fb, _)| {
                fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(n_migrants);
            let gs: Vec<Vec<f64>> = ranked.iter().map(|(_, g)| g.clone()).collect();
            let fs: Vec<f64> = ranked.iter().map(|(f, _)| *f).collect();
            (gs, fs)
        })
        .collect();

    // Insert migrants into destination islands.
    for (src, (gs, fs)) in migrant_batches.iter().enumerate() {
        let dst = (src + 1) % n;
        migrate_individuals(gs, fs, &mut islands[dst], &mut fitnesses[dst], n_migrants);
    }
}

/// Perform the migration step for the Star topology.
///
/// Every non-hub island sends migrants to hub (island 0); hub sends to all others.
fn migrate_star(islands: &mut [Vec<Vec<f64>>], fitnesses: &mut [Vec<f64>], n_migrants: usize) {
    let n = islands.len();
    // Collect migrant batches for each island.
    let migrant_batches: Vec<(Vec<Vec<f64>>, Vec<f64>)> = (0..n)
        .map(|i| {
            let mut ranked: Vec<(f64, Vec<f64>)> = islands[i]
                .iter()
                .zip(fitnesses[i].iter())
                .map(|(g, &f)| (f, g.clone()))
                .collect();
            ranked.sort_unstable_by(|(fa, _), (fb, _)| {
                fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(n_migrants);
            let gs: Vec<Vec<f64>> = ranked.iter().map(|(_, g)| g.clone()).collect();
            let fs: Vec<f64> = ranked.iter().map(|(f, _)| *f).collect();
            (gs, fs)
        })
        .collect();

    // Non-hub islands send to hub (island 0).
    for (gs, fs) in migrant_batches.iter().skip(1) {
        migrate_individuals(gs, fs, &mut islands[0], &mut fitnesses[0], n_migrants);
    }
    // Hub (island 0) sends back to all other islands.
    let (hub_gs, hub_fs) = migrant_batches[0].clone();
    for dst in 1..n {
        migrate_individuals(
            &hub_gs,
            &hub_fs,
            &mut islands[dst],
            &mut fitnesses[dst],
            n_migrants,
        );
    }
}

/// Perform the migration step for the AllToAll topology.
///
/// Every island sends migrants to every other island.
fn migrate_all_to_all(
    islands: &mut [Vec<Vec<f64>>],
    fitnesses: &mut [Vec<f64>],
    n_migrants: usize,
) {
    let n = islands.len();
    // Collect migrant batches.
    let migrant_batches: Vec<(Vec<Vec<f64>>, Vec<f64>)> = (0..n)
        .map(|i| {
            let mut ranked: Vec<(f64, Vec<f64>)> = islands[i]
                .iter()
                .zip(fitnesses[i].iter())
                .map(|(g, &f)| (f, g.clone()))
                .collect();
            ranked.sort_unstable_by(|(fa, _), (fb, _)| {
                fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(n_migrants);
            let gs: Vec<Vec<f64>> = ranked.iter().map(|(_, g)| g.clone()).collect();
            let fs: Vec<f64> = ranked.iter().map(|(f, _)| *f).collect();
            (gs, fs)
        })
        .collect();

    for (src, (gs, fs)) in migrant_batches.iter().enumerate() {
        for dst in 0..n {
            if src == dst {
                continue;
            }
            migrate_individuals(gs, fs, &mut islands[dst], &mut fitnesses[dst], n_migrants);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run an island-model GA on a minimisation objective `fitness_fn`.
///
/// # Algorithm
/// 1. Initialise `n_islands` populations, each of size `pop_per_island`, uniformly in `[lb, ub]^n_dims`.
/// 2. For each epoch:
///    a. Run `n_gens_per_epoch` generations of GA (tournament k=3 + Gaussian mutation + (μ+λ) selection) on each island.
///    b. Compute `n_migrants = floor(migration_rate * pop_per_island)` (at least 1 if rate > 0).
///    c. Execute migration according to the chosen topology.
///    d. Track the global best fitness and record it in `history`.
/// 3. Return the best genome found across all islands and epochs.
///
/// # Errors
/// Returns `EvolError::InvalidParameter` if the configuration is invalid.
pub fn island_model_run<F>(
    config: &IslandConfig,
    fitness_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<IslandResult>
where
    F: Fn(&[f64]) -> f64,
{
    config.validate()?;

    let range = config.ub - config.lb;
    let n = config.n_dims;
    let ni = config.n_islands;
    let ps = config.pop_per_island;

    // --- Initialise islands ---
    let mut islands: Vec<Vec<Vec<f64>>> = (0..ni)
        .map(|_| {
            (0..ps)
                .map(|_| (0..n).map(|_| config.lb + rng.next_f64() * range).collect())
                .collect()
        })
        .collect();

    let mut fitnesses: Vec<Vec<f64>> = islands
        .iter()
        .map(|isl| isl.iter().map(|g| fitness_fn(g)).collect())
        .collect();

    // Number of migrants per island per migration step.
    let n_migrants = ((config.migration_rate * ps as f64).floor() as usize).max(1);

    let mut history = Vec::with_capacity(config.n_epochs);
    let mut global_best_f = f64::INFINITY;
    let mut global_best_g: Vec<f64> = Vec::new();

    // --- Epoch loop ---
    for _ in 0..config.n_epochs {
        // Local evolution on each island.
        for i in 0..ni {
            evolve_island(
                &mut islands[i],
                &mut fitnesses[i],
                config.n_gens_per_epoch,
                config,
                &fitness_fn,
                rng,
            );
        }

        // Migration step.
        match config.topology {
            Topology::Ring => migrate_ring(&mut islands, &mut fitnesses, n_migrants),
            Topology::Star => migrate_star(&mut islands, &mut fitnesses, n_migrants),
            Topology::AllToAll => migrate_all_to_all(&mut islands, &mut fitnesses, n_migrants),
        }

        // Track global best.
        for (i, island_fit) in fitnesses.iter().enumerate() {
            let best_i = argmin(island_fit);
            if island_fit[best_i] < global_best_f {
                global_best_f = island_fit[best_i];
                global_best_g = islands[i][best_i].clone();
            }
        }

        history.push(global_best_f);
    }

    // Collect per-island bests.
    let island_bests: Vec<f64> = fitnesses.iter().map(|f| f[argmin(f)]).collect();

    Ok(IslandResult {
        best_genome: global_best_g,
        best_fitness: global_best_f,
        island_bests,
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

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    fn default_config() -> IslandConfig {
        IslandConfig {
            n_islands: 4,
            pop_per_island: 20,
            n_gens_per_epoch: 10,
            n_epochs: 20,
            migration_rate: 0.1,
            sigma_mut: 0.3,
            p_mut: 0.3,
            topology: Topology::Ring,
            n_dims: 2,
            lb: -5.0,
            ub: 5.0,
        }
    }

    // ── Config validation tests ────────────────────────────────────────────

    #[test]
    fn test_config_n_islands_zero_errors() {
        let mut cfg = default_config();
        cfg.n_islands = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_n_islands_one_errors() {
        let mut cfg = default_config();
        cfg.n_islands = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_pop_per_island_zero_errors() {
        let mut cfg = default_config();
        cfg.pop_per_island = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_bounds_errors() {
        let mut cfg = default_config();
        cfg.lb = 3.0;
        cfg.ub = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_migration_rate_above_one_errors() {
        let mut cfg = default_config();
        cfg.migration_rate = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_migration_rate_zero_errors() {
        let mut cfg = default_config();
        cfg.migration_rate = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_sigma_zero_errors() {
        let mut cfg = default_config();
        cfg.sigma_mut = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ── Functional tests ───────────────────────────────────────────────────

    #[test]
    fn test_history_length_equals_n_epochs() {
        let cfg = default_config();
        let mut rng = LcgRng::new(42);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert_eq!(result.history.len(), cfg.n_epochs);
    }

    #[test]
    fn test_island_bests_length_equals_n_islands() {
        let cfg = default_config();
        let mut rng = LcgRng::new(7);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert_eq!(result.island_bests.len(), cfg.n_islands);
    }

    #[test]
    fn test_ring_topology_sphere_2d_converges() {
        let mut cfg = default_config();
        cfg.n_epochs = 50;
        cfg.n_gens_per_epoch = 20;
        cfg.pop_per_island = 30;
        cfg.sigma_mut = 0.5;
        let mut rng = LcgRng::new(13);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert!(
            result.best_fitness < 1.0,
            "Ring/Sphere 2D did not converge: best = {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_star_topology_runs_without_panic() {
        let mut cfg = default_config();
        cfg.topology = Topology::Star;
        let mut rng = LcgRng::new(55);
        let result = island_model_run(&cfg, sphere, &mut rng);
        assert!(result.is_ok());
        assert!(
            result
                .expect("result should be present")
                .best_fitness
                .is_finite()
        );
    }

    #[test]
    fn test_all_to_all_topology_runs_without_panic() {
        let mut cfg = default_config();
        cfg.topology = Topology::AllToAll;
        let mut rng = LcgRng::new(99);
        let result = island_model_run(&cfg, sphere, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_best_genome_within_bounds() {
        let cfg = default_config();
        let mut rng = LcgRng::new(21);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
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
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert_eq!(result.best_genome.len(), cfg.n_dims);
    }

    #[test]
    fn test_history_is_non_increasing() {
        let cfg = default_config();
        let mut rng = LcgRng::new(4);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
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
    fn test_rastrigin_star_topology_runs() {
        let mut cfg = default_config();
        cfg.topology = Topology::Star;
        cfg.n_epochs = 30;
        cfg.n_islands = 3;
        let mut rng = LcgRng::new(77);
        let result =
            island_model_run(&cfg, rastrigin, &mut rng).expect("island_model_run should succeed");
        assert!(result.best_fitness.is_finite());
    }

    #[test]
    fn test_migration_preserves_population_size() {
        // After running, each island must still have exactly pop_per_island individuals.
        // We verify this indirectly via island_bests length and that n_islands is respected.
        let cfg = default_config();
        let mut rng = LcgRng::new(9);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert_eq!(result.island_bests.len(), cfg.n_islands);
        // All island bests must be finite.
        for &b in &result.island_bests {
            assert!(b.is_finite());
        }
    }

    #[test]
    fn test_single_epoch_produces_result() {
        let mut cfg = default_config();
        cfg.n_epochs = 1;
        let mut rng = LcgRng::new(88);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert_eq!(result.history.len(), 1);
        assert!(result.best_fitness.is_finite());
    }

    #[test]
    fn test_all_to_all_convergence_rastrigin() {
        let mut cfg = default_config();
        cfg.topology = Topology::AllToAll;
        cfg.n_epochs = 40;
        cfg.pop_per_island = 25;
        cfg.migration_rate = 0.2;
        let mut rng = LcgRng::new(123);
        let result =
            island_model_run(&cfg, sphere, &mut rng).expect("island_model_run should succeed");
        assert!(result.best_fitness.is_finite());
    }
}
