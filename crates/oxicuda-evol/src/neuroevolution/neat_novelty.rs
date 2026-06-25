//! NEAT + Novelty Search: abandoning objectives in favour of behavioural novelty.
//!
//! Reference: J. Lehman & K. O. Stanley, "Abandoning Objectives: Evolution through the
//! Search for Novelty Alone", Evolutionary Computation 19(2):189-223, 2011 (ECAL 2008 /
//! ALife 2008 origins). Blend variant after J.-B. Mouret, "Novelty-Based Multiobjectivization",
//! New Horizons in Evolutionary Robotics, 2011.
//!
//! ## Idea
//! Pure fitness-driven search is deceptive on tasks where the gradient of the objective
//! leads away from the global optimum (e.g. a maze whose exit requires first moving
//! *away* from the goal). Novelty search abandons the objective entirely: each individual
//! is rewarded for exhibiting a **behaviour** that differs from behaviours already seen.
//!
//! Concretely:
//! - Every individual produces a **behaviour characterization** (BC) — a fixed-length
//!   real vector (e.g. its final position in a maze, or a behaviour fingerprint). The BC
//!   is supplied by the caller because it is task-specific.
//! - The **novelty** (a.k.a. *sparseness*) of an individual is the mean Euclidean distance
//!   to its `k` nearest neighbours in BC space, where neighbours are drawn from the
//!   **current population ∪ a persistent novelty archive**.
//! - Individuals whose novelty exceeds a threshold are inserted into the archive. The
//!   threshold adapts to the recent add-rate so the archive grows at a controlled pace.
//! - Selection is driven by novelty. An optional **blend** `score = (1-ρ)·fitness + ρ·novelty`
//!   (Mouret 2011) lets novelty and the objective cooperate.
//!
//! ## Reuse of the NEAT machinery
//! This module does **not** re-implement mutation, crossover or speciation. It computes a
//! novelty (or blended) score per genome, writes it into [`Genome::fitness`], then delegates
//! to the existing [`NeatState::speciate`] and [`NeatState::reproduce`] — which select on
//! the `fitness` field. The result is genuine NEAT topology evolution driven by novelty.

use crate::neuroevolution::neat::{Genome, NeatConfig, NeatState};
use crate::{EvolError, EvolResult, handle::LcgRng};

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Hyper-parameters for a NEAT + Novelty Search run.
#[derive(Debug, Clone)]
pub struct NeatNoveltyConfig {
    /// Number of nearest neighbours `k` averaged to compute sparseness (must be >= 1).
    pub k_nearest: usize,
    /// Novelty threshold above which an individual is added to the archive (>= 0).
    pub archive_threshold: f64,
    /// Blend weight ρ ∈ [0, 1]: `score = (1-ρ)·fitness + ρ·novelty`.
    ///
    /// `ρ = 1` is pure novelty search (objective ignored); `ρ = 0` is pure fitness.
    pub blend_rho: f64,
    /// Hard cap on the archive size. When exceeded the oldest entries are evicted.
    pub max_archive: usize,
    /// Whether to adapt `archive_threshold` to the recent add-rate.
    pub dynamic_threshold: bool,
    /// Number of recent generations over which the add-rate is measured.
    pub adapt_window: usize,
    /// If more than this many individuals are added across the window, raise the threshold.
    pub add_rate_high: usize,
    /// If fewer than this many individuals are added across the window, lower the threshold.
    pub add_rate_low: usize,
    /// Multiplicative factor (> 1) used to raise / lower the threshold.
    pub threshold_factor: f64,
    /// Minimum BC dimensionality enforced for every supplied behaviour vector (>= 1).
    pub bc_dim: usize,
}

impl NeatNoveltyConfig {
    /// Build a sensible default configuration for behaviour vectors of dimension `bc_dim`.
    ///
    /// Defaults follow Lehman & Stanley 2011: `k = 15`, dynamic threshold enabled.
    pub fn new(bc_dim: usize) -> Self {
        Self {
            k_nearest: 15,
            archive_threshold: 6.0,
            blend_rho: 1.0,
            max_archive: 2500,
            dynamic_threshold: true,
            adapt_window: 5,
            add_rate_high: 4,
            add_rate_low: 1,
            threshold_factor: 1.05,
            bc_dim,
        }
    }

    /// Validate all fields, returning the crate error type on any inconsistency.
    pub fn validate(&self) -> EvolResult<()> {
        if self.bc_dim == 0 {
            return Err(EvolError::InvalidParameter(
                "bc_dim must be >= 1".to_string(),
            ));
        }
        if self.k_nearest == 0 {
            return Err(EvolError::InvalidParameter(
                "k_nearest must be >= 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.blend_rho) {
            return Err(EvolError::InvalidParameter(
                "blend_rho must lie in [0, 1]".to_string(),
            ));
        }
        if self.archive_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "archive_threshold must be >= 0".to_string(),
            ));
        }
        if self.max_archive == 0 {
            return Err(EvolError::InvalidParameter(
                "max_archive must be >= 1".to_string(),
            ));
        }
        if self.dynamic_threshold {
            if self.adapt_window == 0 {
                return Err(EvolError::InvalidParameter(
                    "adapt_window must be >= 1 when dynamic_threshold is set".to_string(),
                ));
            }
            if self.add_rate_low > self.add_rate_high {
                return Err(EvolError::InvalidParameter(
                    "add_rate_low must be <= add_rate_high".to_string(),
                ));
            }
            if self.threshold_factor <= 1.0 || !self.threshold_factor.is_finite() {
                return Err(EvolError::InvalidParameter(
                    "threshold_factor must be > 1".to_string(),
                ));
            }
        }
        Ok(())
    }
}

// ─── Behaviour-space helpers ────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length behaviour vectors.
///
/// Returns [`EvolError::DimensionMismatch`] if the lengths differ. The squared form is used
/// internally for all neighbour ranking (monotone in the true distance) to avoid redundant
/// `sqrt`s; the final sparseness applies one `sqrt` per neighbour.
fn bc_sq_distance(a: &[f64], b: &[f64]) -> EvolResult<f64> {
    if a.len() != b.len() {
        return Err(EvolError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mut acc = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    Ok(acc)
}

/// Mean of the `k` smallest values in `dists`, taking the square root of each first.
///
/// `dists` holds *squared* distances. The `k` smallest squared distances are the `k`
/// smallest true distances (the map is monotone), so a partial selection on squared
/// distances followed by `sqrt` of the chosen `k` yields the exact mean-kNN distance.
fn mean_k_smallest_sqrt(mut dists: Vec<f64>, k: usize) -> f64 {
    if dists.is_empty() || k == 0 {
        return 0.0;
    }
    let k = k.min(dists.len());
    // Partial sort: ascending by squared distance. `k` is small relative to the population,
    // so a full unstable sort is both simple and adequate; it keeps determinism trivial.
    dists.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let sum: f64 = dists.iter().take(k).map(|d| d.max(0.0).sqrt()).sum();
    sum / k as f64
}

// ─── Novelty archive ────────────────────────────────────────────────────────────

/// A persistent archive of behaviour characterizations judged novel in past generations.
///
/// The archive together with the current population forms the reference set against which
/// each individual's sparseness is measured (Lehman & Stanley 2011, §3).
#[derive(Debug, Clone, Default)]
pub struct NoveltyArchive {
    /// Stored behaviour vectors, oldest first.
    behaviours: Vec<Vec<f64>>,
    /// Per-generation add counts, most recent at the back; bounded by `adapt_window`.
    recent_adds: Vec<usize>,
}

impl NoveltyArchive {
    /// Create an empty archive.
    pub fn new() -> Self {
        Self {
            behaviours: Vec::new(),
            recent_adds: Vec::new(),
        }
    }

    /// Number of behaviours currently stored.
    pub fn len(&self) -> usize {
        self.behaviours.len()
    }

    /// Whether the archive holds no behaviours.
    pub fn is_empty(&self) -> bool {
        self.behaviours.is_empty()
    }

    /// Read-only view of the stored behaviour vectors.
    pub fn behaviours(&self) -> &[Vec<f64>] {
        &self.behaviours
    }

    /// Total number of additions recorded across the current adaptation window.
    pub fn recent_add_count(&self) -> usize {
        self.recent_adds.iter().copied().sum()
    }

    /// Push one behaviour, evicting the oldest entry if `max_archive` would be exceeded.
    fn push(&mut self, bc: Vec<f64>, max_archive: usize) {
        self.behaviours.push(bc);
        while self.behaviours.len() > max_archive && !self.behaviours.is_empty() {
            self.behaviours.remove(0);
        }
    }

    /// Record that `added` individuals entered the archive this generation, sliding the
    /// `adapt_window`-length history.
    fn record_generation(&mut self, added: usize, adapt_window: usize) {
        self.recent_adds.push(added);
        while self.recent_adds.len() > adapt_window.max(1) {
            self.recent_adds.remove(0);
        }
    }
}

// ─── Novelty engine ──────────────────────────────────────────────────────────────

/// Compute the novelty (sparseness) of `bc` relative to `population_bcs ∪ archive`.
///
/// Returns the **mean Euclidean distance to the `k` nearest neighbours**, where the
/// neighbour pool is every behaviour in `population_bcs` (excluding `bc` itself, matched by
/// reference identity is not used — see below) plus every behaviour in `archive`.
///
/// To make the "self" exclusion well-defined when `bc` is also one of `population_bcs`,
/// the caller passes `self_index`: the index of `bc` within `population_bcs`, or `None` if
/// `bc` is not a member (e.g. when scoring an archive candidate). The element at
/// `self_index` is skipped so an individual never counts its own zero distance.
///
/// # Errors
/// - [`EvolError::EmptyPopulation`] if the neighbour pool is empty.
/// - [`EvolError::InvalidParameter`] if `k == 0`.
/// - [`EvolError::DimensionMismatch`] if any reference vector has a different length to `bc`.
pub fn compute_novelty(
    bc: &[f64],
    population_bcs: &[Vec<f64>],
    archive: &NoveltyArchive,
    k: usize,
    self_index: Option<usize>,
) -> EvolResult<f64> {
    if k == 0 {
        return Err(EvolError::InvalidParameter(
            "k_nearest must be >= 1".to_string(),
        ));
    }

    let mut dists: Vec<f64> = Vec::with_capacity(population_bcs.len() + archive.len());
    for (i, other) in population_bcs.iter().enumerate() {
        if Some(i) == self_index {
            continue;
        }
        dists.push(bc_sq_distance(bc, other)?);
    }
    for other in archive.behaviours() {
        dists.push(bc_sq_distance(bc, other)?);
    }

    if dists.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }

    Ok(mean_k_smallest_sqrt(dists, k))
}

/// Compute the novelty of every individual in `population_bcs` (against the population and
/// the archive), returning a parallel vector of sparseness values.
///
/// Each individual excludes its own behaviour from its neighbour pool.
pub fn compute_population_novelty(
    population_bcs: &[Vec<f64>],
    archive: &NoveltyArchive,
    cfg: &NeatNoveltyConfig,
) -> EvolResult<Vec<f64>> {
    cfg.validate()?;
    if population_bcs.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    for (i, bc) in population_bcs.iter().enumerate() {
        if bc.len() != cfg.bc_dim {
            return Err(EvolError::DimensionMismatch {
                expected: cfg.bc_dim,
                got: bc.len(),
            });
        }
        // Defensive: archive vectors must also match the configured BC dimensionality.
        if i == 0 {
            for arc in archive.behaviours() {
                if arc.len() != cfg.bc_dim {
                    return Err(EvolError::DimensionMismatch {
                        expected: cfg.bc_dim,
                        got: arc.len(),
                    });
                }
            }
        }
    }

    let mut out = Vec::with_capacity(population_bcs.len());
    for (i, bc) in population_bcs.iter().enumerate() {
        out.push(compute_novelty(
            bc,
            population_bcs,
            archive,
            cfg.k_nearest,
            Some(i),
        )?);
    }
    Ok(out)
}

// ─── Top-level NEAT + Novelty driver ─────────────────────────────────────────────

/// NEAT evolution driven by behavioural novelty (with optional fitness blend).
///
/// Wraps a [`NeatState`] (the standard NEAT population, innovation tracker and speciation)
/// and a [`NoveltyArchive`]. Each generation:
///
/// 1. The caller supplies a behaviour characterization (and optional fitness) per genome.
/// 2. [`NeatNovelty::compute_scores`] computes per-genome novelty and the blended selection
///    score `(1-ρ)·fitness + ρ·novelty`.
/// 3. [`NeatNovelty::update_archive`] inserts sufficiently novel behaviours and adapts the
///    threshold to the recent add-rate.
/// 4. The blended score is written into each [`Genome::fitness`] and the NEAT speciation +
///    reproduction machinery selects on it, producing the next generation.
pub struct NeatNovelty {
    /// Underlying NEAT state (population, species, innovation tracker, generation counter).
    pub neat: NeatState,
    /// Standard NEAT structural-evolution hyper-parameters.
    pub neat_cfg: NeatConfig,
    /// Novelty-search hyper-parameters.
    pub novelty_cfg: NeatNoveltyConfig,
    /// The persistent novelty archive.
    pub archive: NoveltyArchive,
    /// Current (possibly adapted) archive threshold.
    current_threshold: f64,
    /// Best fitness ever observed (for reporting on blended / objective-aware runs).
    best_fitness: f64,
    /// Number of distinct behaviour cells visited across all generations (coarse coverage).
    distinct_behaviours: usize,
    /// Internal coarse behaviour grid used only to count distinct visited cells.
    visited_cells: std::collections::HashSet<Vec<i64>>,
    /// Resolution of the coarse coverage grid (BC units per cell).
    coverage_resolution: f64,
}

impl NeatNovelty {
    /// Construct a new NEAT + Novelty driver with a freshly initialised minimal population.
    ///
    /// # Errors
    /// Propagates configuration validation failures from [`NeatNoveltyConfig::validate`] and
    /// rejects a zero population size.
    pub fn new(
        neat_cfg: NeatConfig,
        novelty_cfg: NeatNoveltyConfig,
        rng: &mut LcgRng,
    ) -> EvolResult<Self> {
        novelty_cfg.validate()?;
        if neat_cfg.pop_size == 0 {
            return Err(EvolError::InvalidParameter(
                "pop_size must be >= 1".to_string(),
            ));
        }
        let neat = NeatState::new(&neat_cfg, rng);
        let current_threshold = novelty_cfg.archive_threshold;
        Ok(Self {
            neat,
            neat_cfg,
            novelty_cfg,
            archive: NoveltyArchive::new(),
            current_threshold,
            best_fitness: f64::NEG_INFINITY,
            distinct_behaviours: 0,
            visited_cells: std::collections::HashSet::new(),
            coverage_resolution: 1.0,
        })
    }

    /// Override the coarse coverage-grid resolution used by [`Self::distinct_behaviours`].
    ///
    /// Purely diagnostic — does not influence selection. A smaller value counts more cells.
    pub fn set_coverage_resolution(&mut self, resolution: f64) -> EvolResult<()> {
        if resolution <= 0.0 || !resolution.is_finite() {
            return Err(EvolError::InvalidParameter(
                "coverage_resolution must be a positive, finite value".to_string(),
            ));
        }
        self.coverage_resolution = resolution;
        Ok(())
    }

    /// Population size (number of genomes).
    pub fn pop_size(&self) -> usize {
        self.neat.population.len()
    }

    /// Current generation index.
    pub fn generation(&self) -> usize {
        self.neat.generation
    }

    /// The current (possibly adapted) archive insertion threshold.
    pub fn current_threshold(&self) -> f64 {
        self.current_threshold
    }

    /// Best raw fitness observed across all generations (or `-inf` if never supplied).
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Count of distinct coarse behaviour cells visited across all generations.
    ///
    /// This is the exploration metric used to demonstrate that novelty search visits more
    /// of the behaviour space than fitness search on a deceptive task.
    pub fn distinct_behaviours(&self) -> usize {
        self.distinct_behaviours
    }

    /// Quantise a behaviour vector onto the coarse coverage grid.
    fn cell_of(&self, bc: &[f64]) -> Vec<i64> {
        bc.iter()
            .map(|&v| (v / self.coverage_resolution).floor() as i64)
            .collect()
    }

    /// Compute per-genome novelty and the blended selection score.
    ///
    /// `population_bcs[i]` is the behaviour of genome `i`. `fitnesses` is optional: when
    /// `None`, pure novelty is used regardless of `blend_rho`. Returns `(novelties, scores)`.
    ///
    /// # Errors
    /// - [`EvolError::EmptyPopulation`] if `population_bcs` is empty.
    /// - [`EvolError::DimensionMismatch`] if `population_bcs.len()` differs from the genome
    ///   count, if any BC has the wrong dimension, or if `fitnesses` has the wrong length.
    pub fn compute_scores(
        &self,
        population_bcs: &[Vec<f64>],
        fitnesses: Option<&[f64]>,
    ) -> EvolResult<(Vec<f64>, Vec<f64>)> {
        let n = self.neat.population.len();
        if population_bcs.len() != n {
            return Err(EvolError::DimensionMismatch {
                expected: n,
                got: population_bcs.len(),
            });
        }
        if let Some(fit) = fitnesses
            && fit.len() != n
        {
            return Err(EvolError::DimensionMismatch {
                expected: n,
                got: fit.len(),
            });
        }

        let novelties =
            compute_population_novelty(population_bcs, &self.archive, &self.novelty_cfg)?;

        // Blended score. When fitness is absent we fall back to pure novelty (ρ forced to 1).
        let rho = if fitnesses.is_some() {
            self.novelty_cfg.blend_rho
        } else {
            1.0
        };
        let scores: Vec<f64> = novelties
            .iter()
            .enumerate()
            .map(|(i, &nov)| {
                let fit = fitnesses.map(|f| f[i]).unwrap_or(0.0);
                (1.0 - rho) * fit + rho * nov
            })
            .collect();

        Ok((novelties, scores))
    }

    /// Insert sufficiently novel behaviours into the archive and adapt the threshold.
    ///
    /// An individual is added when its novelty `>= current_threshold`. As a guarantee that
    /// the archive can always seed itself, the single most-novel individual of the first
    /// generation (when the archive is empty) is also added.
    ///
    /// When `dynamic_threshold` is set, after recording the generation's add-count the
    /// threshold is multiplied by `threshold_factor` if the windowed add-rate exceeds
    /// `add_rate_high`, or divided by it if the rate is below `add_rate_low` (Lehman &
    /// Stanley 2011, §3: "if the archive grows too fast … raise the threshold").
    ///
    /// Returns the number of behaviours added this generation.
    ///
    /// # Errors
    /// [`EvolError::DimensionMismatch`] if `novelties.len()` differs from `population_bcs.len()`.
    pub fn update_archive(
        &mut self,
        population_bcs: &[Vec<f64>],
        novelties: &[f64],
    ) -> EvolResult<usize> {
        if novelties.len() != population_bcs.len() {
            return Err(EvolError::DimensionMismatch {
                expected: population_bcs.len(),
                got: novelties.len(),
            });
        }
        for bc in population_bcs {
            if bc.len() != self.novelty_cfg.bc_dim {
                return Err(EvolError::DimensionMismatch {
                    expected: self.novelty_cfg.bc_dim,
                    got: bc.len(),
                });
            }
        }

        let mut added = 0usize;
        // Seed the archive on the very first non-trivial generation with the most-novel BC.
        let seed_idx = if self.archive.is_empty() && !novelties.is_empty() {
            let mut best = 0usize;
            for i in 1..novelties.len() {
                if novelties[i] > novelties[best] {
                    best = i;
                }
            }
            Some(best)
        } else {
            None
        };

        for (i, bc) in population_bcs.iter().enumerate() {
            let novel_enough = novelties[i] >= self.current_threshold;
            let is_seed = Some(i) == seed_idx;
            if novel_enough || is_seed {
                self.archive.push(bc.clone(), self.novelty_cfg.max_archive);
                added += 1;
            }
        }

        // Adapt the threshold to the recent add-rate.
        if self.novelty_cfg.dynamic_threshold {
            self.archive
                .record_generation(added, self.novelty_cfg.adapt_window);
            let windowed = self.archive.recent_add_count();
            if windowed > self.novelty_cfg.add_rate_high {
                self.current_threshold *= self.novelty_cfg.threshold_factor;
            } else if windowed < self.novelty_cfg.add_rate_low {
                self.current_threshold /= self.novelty_cfg.threshold_factor;
                // Never let the threshold collapse to zero / negative.
                if self.current_threshold < f64::MIN_POSITIVE {
                    self.current_threshold = f64::MIN_POSITIVE;
                }
            }
        } else {
            self.archive.record_generation(added, 1);
        }

        Ok(added)
    }

    /// Update the coarse coverage statistics from this generation's behaviours.
    fn record_coverage(&mut self, population_bcs: &[Vec<f64>]) {
        for bc in population_bcs {
            let cell = self.cell_of(bc);
            if self.visited_cells.insert(cell) {
                self.distinct_behaviours += 1;
            }
        }
    }

    /// Execute one full novelty-driven generation.
    ///
    /// 1. Compute novelty + blended score from the caller-supplied behaviours / fitnesses.
    /// 2. Update the archive (insertion + threshold adaptation) and coverage statistics.
    /// 3. Write the blended score into each [`Genome::fitness`] and run the NEAT
    ///    speciation + reproduction step, advancing to the next generation.
    ///
    /// Returns the `(novelties, scores)` of the generation *before* reproduction, so the
    /// caller can inspect what drove selection.
    ///
    /// # Errors
    /// Propagates dimension / population errors from [`Self::compute_scores`] and
    /// [`Self::update_archive`], plus any NEAT reproduction error.
    pub fn step(
        &mut self,
        population_bcs: &[Vec<f64>],
        fitnesses: Option<&[f64]>,
        rng: &mut LcgRng,
    ) -> EvolResult<(Vec<f64>, Vec<f64>)> {
        let (novelties, scores) = self.compute_scores(population_bcs, fitnesses)?;

        // Track best raw fitness for reporting (does not affect selection under pure novelty).
        if let Some(fit) = fitnesses {
            for &f in fit {
                if f > self.best_fitness {
                    self.best_fitness = f;
                }
            }
        }

        self.update_archive(population_bcs, &novelties)?;
        self.record_coverage(population_bcs);

        // Drive NEAT selection by the blended novelty score via the `fitness` field.
        for (genome, &score) in self.neat.population.iter_mut().zip(scores.iter()) {
            genome.fitness = score;
        }
        self.neat.speciate(&self.neat_cfg);
        self.neat.reproduce(&self.neat_cfg, rng)?;

        Ok((novelties, scores))
    }

    /// Borrow the current population of genomes.
    pub fn population(&self) -> &[Genome] {
        &self.neat.population
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Helper: build a default novelty config with a given BC dimension and k.
    fn cfg(bc_dim: usize, k: usize) -> NeatNoveltyConfig {
        let mut c = NeatNoveltyConfig::new(bc_dim);
        c.k_nearest = k;
        c
    }

    #[test]
    fn novelty_equals_mean_of_k_smallest_distances() {
        // Hand-computed example. bc = origin in 1-D.
        // Neighbours at distances 1, 2, 3, 10 (population) and 5 (archive).
        // k = 3 nearest distances are {1, 2, 3}; mean = 2.0.
        let bc = vec![0.0];
        let population: Vec<Vec<f64>> = vec![
            vec![0.0], // self (index 0) — must be skipped
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![10.0],
        ];
        let mut archive = NoveltyArchive::new();
        archive.push(vec![5.0], 100);

        let nov =
            compute_novelty(&bc, &population, &archive, 3, Some(0)).expect("novelty must compute");
        assert!(
            (nov - 2.0).abs() < 1e-12,
            "expected mean of 3 nearest = 2.0, got {nov}"
        );

        // k = 4 nearest distances are {1, 2, 3, 5}; mean = 11/4 = 2.75.
        let nov4 =
            compute_novelty(&bc, &population, &archive, 4, Some(0)).expect("novelty must compute");
        assert!(
            (nov4 - 2.75).abs() < 1e-12,
            "expected mean of 4 nearest = 2.75, got {nov4}"
        );
    }

    #[test]
    fn novelty_2d_hand_check() {
        // 2-D: bc at origin; neighbours forming a 3-4-5 triangle pattern.
        // points (3,4) dist 5, (1,0) dist 1, (0,2) dist 2.
        let bc = vec![0.0, 0.0];
        let population = vec![
            vec![0.0, 0.0], // self
            vec![3.0, 4.0], // dist 5
            vec![1.0, 0.0], // dist 1
            vec![0.0, 2.0], // dist 2
        ];
        let archive = NoveltyArchive::new();
        // k = 2 nearest: {1, 2}, mean = 1.5.
        let nov =
            compute_novelty(&bc, &population, &archive, 2, Some(0)).expect("novelty must compute");
        assert!((nov - 1.5).abs() < 1e-12, "expected 1.5, got {nov}");
    }

    #[test]
    fn archive_grows_for_novel_and_not_for_duplicates() {
        // Behaviours far apart should add; identical-to-archive behaviours should not
        // (their nearest-neighbour distance against the archived copy is 0, below any
        // positive threshold). We use k = 1 so "nearest neighbour" is unambiguous: a
        // behaviour that coincides with an archived copy has sparseness exactly 0.
        let mut c = cfg(1, 1);
        c.archive_threshold = 1.0;
        c.dynamic_threshold = false;

        let neat_cfg = NeatConfig::new(2, 1);
        let mut rng = LcgRng::new(7);
        let mut nn = NeatNovelty::new(neat_cfg, c, &mut rng).expect("construct");
        // The BC vectors below are supplied directly to the public archive/novelty API;
        // they need not match the genome count for these archive-only assertions.

        // Round 1: three well-separated behaviours → all genuinely novel and archived.
        let bcs1: Vec<Vec<f64>> = vec![vec![0.0], vec![10.0], vec![20.0]];
        let nov1 =
            compute_population_novelty(&bcs1, &nn.archive, &nn.novelty_cfg).expect("novelty");
        // With k = 1 the nearest neighbours are at distance 10 each → all clear threshold 1.
        for &v in &nov1 {
            assert!(v >= 1.0, "well-separated behaviour must be novel, got {v}");
        }
        let added1 = nn.update_archive(&bcs1, &nov1).expect("update");
        assert_eq!(added1, 3, "all three novel behaviours must be archived");
        let size_after_1 = nn.archive.len();
        assert_eq!(size_after_1, 3);

        // Round 2: behaviours identical to archived ones → nearest neighbour is the
        // archived copy at distance 0 → novelty 0 → no growth.
        let bcs2: Vec<Vec<f64>> = vec![vec![0.0], vec![10.0], vec![20.0]];
        let nov2 =
            compute_population_novelty(&bcs2, &nn.archive, &nn.novelty_cfg).expect("novelty");
        for &v in &nov2 {
            assert!(
                v.abs() < 1e-9,
                "duplicate behaviour must have novelty 0, got {v}"
            );
        }
        let added2 = nn.update_archive(&bcs2, &nov2).expect("update");
        assert_eq!(added2, 0, "duplicates must not grow the archive");
        assert_eq!(
            nn.archive.len(),
            size_after_1,
            "archive size must be unchanged for duplicate behaviours"
        );

        // Round 3: one fresh behaviour far from everything → archive grows by exactly 1.
        let bcs3: Vec<Vec<f64>> = vec![vec![100.0]];
        let nov3 =
            compute_population_novelty(&bcs3, &nn.archive, &nn.novelty_cfg).expect("novelty");
        assert!(
            nov3[0] >= 1.0,
            "distant behaviour must be novel, got {}",
            nov3[0]
        );
        let added3 = nn.update_archive(&bcs3, &nov3).expect("update");
        assert_eq!(
            added3, 1,
            "one genuinely novel behaviour must grow the archive by 1"
        );
        assert_eq!(nn.archive.len(), size_after_1 + 1);
    }

    #[test]
    fn dynamic_threshold_rises_with_high_add_rate() {
        // Many novel additions per generation should raise the threshold.
        let mut c = cfg(1, 1);
        c.archive_threshold = 0.5;
        c.dynamic_threshold = true;
        c.adapt_window = 2;
        c.add_rate_high = 2;
        c.add_rate_low = 1;
        c.threshold_factor = 2.0;

        let neat_cfg = NeatConfig::new(2, 1);
        let mut rng = LcgRng::new(1);
        let mut nn = NeatNovelty::new(neat_cfg, c, &mut rng).expect("construct");
        let start = nn.current_threshold();

        // Feed two generations of highly-novel, well-separated behaviours.
        for g in 0..2 {
            let base = (g as f64) * 1000.0;
            let bcs: Vec<Vec<f64>> = vec![
                vec![base],
                vec![base + 100.0],
                vec![base + 200.0],
                vec![base + 300.0],
            ];
            let nov =
                compute_population_novelty(&bcs, &nn.archive, &nn.novelty_cfg).expect("novelty");
            nn.update_archive(&bcs, &nov).expect("update");
        }
        assert!(
            nn.current_threshold() > start,
            "threshold should rise with high add-rate: start {start}, now {}",
            nn.current_threshold()
        );
    }

    #[test]
    fn dynamic_threshold_falls_with_low_add_rate() {
        // Zero additions per generation should lower the threshold.
        let mut c = cfg(1, 1);
        c.archive_threshold = 100.0; // so nothing clears the bar
        c.dynamic_threshold = true;
        c.adapt_window = 2;
        c.add_rate_high = 5;
        c.add_rate_low = 1;
        c.threshold_factor = 2.0;

        let neat_cfg = NeatConfig::new(2, 1);
        let mut rng = LcgRng::new(2);
        let mut nn = NeatNovelty::new(neat_cfg, c, &mut rng).expect("construct");

        // First seed the archive so future generations are not auto-seeded.
        let seed: Vec<Vec<f64>> = vec![vec![0.0], vec![0.1]];
        let nov0 =
            compute_population_novelty(&seed, &nn.archive, &nn.novelty_cfg).expect("novelty");
        nn.update_archive(&seed, &nov0).expect("update");
        let after_seed = nn.current_threshold();

        // Now feed tiny-novelty behaviours that never clear the (huge) threshold.
        for _ in 0..2 {
            let bcs: Vec<Vec<f64>> = vec![vec![0.0], vec![0.05]];
            let nov =
                compute_population_novelty(&bcs, &nn.archive, &nn.novelty_cfg).expect("novelty");
            let added = nn.update_archive(&bcs, &nov).expect("update");
            assert_eq!(added, 0, "nothing should clear the high threshold");
        }
        assert!(
            nn.current_threshold() < after_seed,
            "threshold should fall with low add-rate: after_seed {after_seed}, now {}",
            nn.current_threshold()
        );
    }

    #[test]
    fn determinism_with_fixed_seed() {
        // Two runs with the same seed and the same behaviour stream must agree exactly.
        fn run(seed: u64) -> (usize, f64) {
            let novelty_cfg = cfg(1, 2);
            let mut neat_cfg = NeatConfig::new(2, 1);
            neat_cfg.pop_size = 12;
            neat_cfg.max_generations = 4;
            let mut rng = LcgRng::new(seed);
            let mut nn = NeatNovelty::new(neat_cfg, novelty_cfg, &mut rng).expect("construct");

            for g in 0..4 {
                let n = nn.pop_size();
                // Deterministic behaviour stream derived from generation + index.
                let bcs: Vec<Vec<f64>> = (0..n).map(|i| vec![(g * 31 + i * 7) as f64]).collect();
                nn.step(&bcs, None, &mut rng).expect("step");
            }
            (nn.archive.len(), nn.current_threshold())
        }

        let a = run(123_456);
        let b = run(123_456);
        assert_eq!(a.0, b.0, "archive size must be deterministic");
        assert!(
            (a.1 - b.1).abs() < 1e-12,
            "threshold must be deterministic: {} vs {}",
            a.1,
            b.1
        );

        // A different seed should (almost surely) differ in topology evolution; we only
        // assert the run does not error and produces a population.
        let c = run(987_654);
        assert!(c.0 <= a.0 + 1000);
    }

    #[test]
    fn step_evolves_population_and_advances_generation() {
        let novelty_cfg = cfg(2, 3);
        let mut neat_cfg = NeatConfig::new(3, 2);
        neat_cfg.pop_size = 20;
        let mut rng = LcgRng::new(42);
        let mut nn = NeatNovelty::new(neat_cfg, novelty_cfg, &mut rng).expect("construct");
        assert_eq!(nn.generation(), 0);

        for g in 0..5 {
            let n = nn.pop_size();
            let bcs: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let a = ((g + i) % 7) as f64;
                    let b = ((g * 2 + i) % 5) as f64;
                    vec![a, b]
                })
                .collect();
            let (nov, scores) = nn.step(&bcs, None, &mut rng).expect("step");
            assert_eq!(nov.len(), n);
            assert_eq!(scores.len(), n);
            // Pure novelty: score must equal novelty exactly.
            for (s, v) in scores.iter().zip(nov.iter()) {
                assert!((s - v).abs() < 1e-12);
            }
        }
        assert_eq!(
            nn.generation(),
            5,
            "five steps must advance five generations"
        );
        assert_eq!(nn.pop_size(), 20, "population size is preserved");
    }

    #[test]
    fn blend_combines_fitness_and_novelty() {
        // With ρ = 0.5 the score must be the average of fitness and novelty.
        let mut novelty_cfg = cfg(1, 1);
        novelty_cfg.blend_rho = 0.5;
        novelty_cfg.dynamic_threshold = false;
        let mut neat_cfg = NeatConfig::new(2, 1);
        neat_cfg.pop_size = 3;
        let mut rng = LcgRng::new(5);
        let nn = NeatNovelty::new(neat_cfg, novelty_cfg, &mut rng).expect("construct");

        let bcs: Vec<Vec<f64>> = vec![vec![0.0], vec![4.0], vec![8.0]];
        let fits = vec![10.0, 20.0, 30.0];
        let (nov, scores) = nn
            .compute_scores(&bcs, Some(&fits))
            .expect("scores must compute");
        for i in 0..3 {
            let expected = 0.5 * fits[i] + 0.5 * nov[i];
            assert!(
                (scores[i] - expected).abs() < 1e-12,
                "blend mismatch at {i}: {} vs {expected}",
                scores[i]
            );
        }
    }

    #[test]
    fn deceptive_task_novelty_explores_more_than_fitness() {
        // ── Deceptive toy task ──────────────────────────────────────────────────
        // Behaviour space is the 1-D interval [0, GOAL]. A deceptive **trap** sits at the
        // left edge b = 0. Fitness = closeness to the trap (`fitness = GOAL - b`, maximised),
        // so a pure-fitness search is rewarded for staying pinned at the trap and is never
        // pushed to explore. The interesting region (large b) is only reached by valuing the
        // *novelty* of behaviours that differ from those already visited.
        //
        // Both searches share an identical generative model so the only difference is the
        // SELECTION SIGNAL. Each generation produces offspring by taking a small SYMMETRIC
        // (±) step from each selected parent behaviour, clamped to [0, GOAL]. Fitness keeps
        // the *smallest*-b parents: the symmetric step plus the left clamp at the trap pins
        // its population against b = 0, so it explores only a sliver of the space. Novelty
        // keeps the *most novel* (frontier) parents, so half of their symmetric steps push
        // the frontier ever rightward across the whole interval.

        const GOAL: f64 = 60.0;
        const GENS: usize = 40;
        const N: usize = 16;
        const KEEP: usize = 4;
        const STEP: f64 = 1.5;

        // Generate this generation's behaviours by taking a symmetric step around each
        // retained parent. The clamp at 0 makes the trap an absorbing edge for any search
        // that selects toward it.
        fn offspring(parents: &[f64], rng: &mut LcgRng) -> Vec<Vec<f64>> {
            (0..N)
                .map(|i| {
                    let p = parents[i % parents.len()];
                    let step = (rng.next_f64() - 0.5) * 2.0 * STEP; // ∈ [-STEP, STEP]
                    vec![(p + step).clamp(0.0, GOAL)]
                })
                .collect()
        }

        // ── Fitness-driven run: keep the KEEP smallest-b individuals (closest to trap) ──
        let fitness_distinct = {
            let mut rng = LcgRng::new(2024);
            let mut parents = vec![0.0f64; KEEP];
            let mut visited: std::collections::HashSet<i64> = std::collections::HashSet::new();
            for _ in 0..GENS {
                let bcs = offspring(&parents, &mut rng);
                for bc in &bcs {
                    visited.insert(bc[0].floor() as i64);
                }
                // Maximise fitness = GOAL - b  ⇔  keep the smallest-b individuals.
                let mut idx: Vec<usize> = (0..N).collect();
                idx.sort_by(|&a, &b| {
                    bcs[a][0]
                        .partial_cmp(&bcs[b][0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                parents = idx[..KEEP].iter().map(|&j| bcs[j][0]).collect();
            }
            visited.len()
        };

        // ── Novelty-driven run: keep the KEEP most-novel individuals (the frontier) ──
        let novelty_distinct = {
            let mut rng = LcgRng::new(2024);
            let mut parents = vec![0.0f64; KEEP];
            let mut archive = NoveltyArchive::new();
            let novelty_cfg = {
                let mut cc = NeatNoveltyConfig::new(1);
                cc.k_nearest = 3;
                cc.archive_threshold = 1.0;
                cc.dynamic_threshold = false;
                cc
            };
            let mut visited: std::collections::HashSet<i64> = std::collections::HashSet::new();
            for _ in 0..GENS {
                let bcs = offspring(&parents, &mut rng);
                for bc in &bcs {
                    visited.insert(bc[0].floor() as i64);
                }
                let nov =
                    compute_population_novelty(&bcs, &archive, &novelty_cfg).expect("novelty");
                for (i, bc) in bcs.iter().enumerate() {
                    if nov[i] >= novelty_cfg.archive_threshold {
                        archive.push(bc.clone(), novelty_cfg.max_archive);
                    }
                }
                // Keep the most-novel individuals; ties broken toward larger b (the frontier).
                let mut idx: Vec<usize> = (0..N).collect();
                idx.sort_by(|&a, &b| {
                    nov[b]
                        .partial_cmp(&nov[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(
                            bcs[b][0]
                                .partial_cmp(&bcs[a][0])
                                .unwrap_or(std::cmp::Ordering::Equal),
                        )
                });
                parents = idx[..KEEP].iter().map(|&j| bcs[j][0]).collect();
            }
            visited.len()
        };

        // Fitness search is pinned near the trap and visits only a handful of cells; novelty
        // search marches across the interval and visits many more distinct behaviours.
        assert!(
            novelty_distinct > fitness_distinct,
            "novelty search must visit more distinct behaviours than fitness search: \
             novelty={novelty_distinct} fitness={fitness_distinct}"
        );
        // Sanity: novelty search should reach deep into the interval, fitness should not.
        assert!(
            novelty_distinct >= 20,
            "novelty search expected to explore broadly, got {novelty_distinct}"
        );
        assert!(
            fitness_distinct <= 10,
            "fitness search expected to stay pinned near the trap, got {fitness_distinct}"
        );
    }

    // ── Error-path tests ────────────────────────────────────────────────────────

    #[test]
    fn error_k_larger_than_available_is_clamped_not_panicking() {
        // k larger than the neighbour pool must be handled gracefully (clamped to pool size).
        let bc = vec![0.0];
        let population = vec![vec![0.0], vec![1.0], vec![2.0]]; // 2 usable neighbours
        let archive = NoveltyArchive::new();
        // k = 10 but only 2 neighbours: mean of {1, 2} = 1.5.
        let nov = compute_novelty(&bc, &population, &archive, 10, Some(0))
            .expect("k>pool must clamp, not error");
        assert!(
            (nov - 1.5).abs() < 1e-12,
            "expected 1.5 (clamped), got {nov}"
        );
    }

    #[test]
    fn error_empty_population_pool() {
        // A single-element population that is the self → empty neighbour pool → error.
        let bc = vec![0.0];
        let population = vec![vec![0.0]];
        let archive = NoveltyArchive::new();
        let err = compute_novelty(&bc, &population, &archive, 1, Some(0));
        assert!(
            matches!(err, Err(EvolError::EmptyPopulation)),
            "got {err:?}"
        );

        // compute_population_novelty on an empty slice also errors.
        let c = cfg(1, 1);
        let arc = NoveltyArchive::new();
        let err2 = compute_population_novelty(&[], &arc, &c);
        assert!(
            matches!(err2, Err(EvolError::EmptyPopulation)),
            "got {err2:?}"
        );
    }

    #[test]
    fn error_bc_dimension_mismatch() {
        // Mismatched BC lengths within compute_novelty.
        let bc = vec![0.0, 0.0];
        let population = vec![vec![1.0]]; // wrong length
        let archive = NoveltyArchive::new();
        let err = compute_novelty(&bc, &population, &archive, 1, None);
        assert!(
            matches!(err, Err(EvolError::DimensionMismatch { .. })),
            "got {err:?}"
        );

        // Mismatch detected by compute_population_novelty against the configured bc_dim.
        let c = cfg(2, 1);
        let arc = NoveltyArchive::new();
        let bad: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0]]; // second is wrong dim
        let err2 = compute_population_novelty(&bad, &arc, &c);
        assert!(
            matches!(err2, Err(EvolError::DimensionMismatch { .. })),
            "got {err2:?}"
        );
    }

    #[test]
    fn error_config_validation_and_score_length() {
        // k = 0 must be rejected at config validation.
        let mut bad = NeatNoveltyConfig::new(1);
        bad.k_nearest = 0;
        assert!(bad.validate().is_err());

        // blend_rho out of range.
        let mut bad2 = NeatNoveltyConfig::new(1);
        bad2.blend_rho = 1.5;
        assert!(bad2.validate().is_err());

        // bc_dim = 0.
        let bad3 = NeatNoveltyConfig::new(0);
        assert!(bad3.validate().is_err());

        // compute_scores with a wrong-length fitness slice → DimensionMismatch.
        let novelty_cfg = cfg(1, 1);
        let mut neat_cfg = NeatConfig::new(2, 1);
        neat_cfg.pop_size = 3;
        let mut rng = LcgRng::new(9);
        let nn = NeatNovelty::new(neat_cfg, novelty_cfg, &mut rng).expect("construct");
        let bcs: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let err = nn.compute_scores(&bcs, Some(&[1.0, 2.0])); // length 2 != 3
        assert!(
            matches!(err, Err(EvolError::DimensionMismatch { .. })),
            "got {err:?}"
        );

        // update_archive with mismatched novelty length.
        let mut nn2 = {
            let nc = cfg(1, 1);
            let mut ncfg = NeatConfig::new(2, 1);
            ncfg.pop_size = 3;
            let mut r = LcgRng::new(10);
            NeatNovelty::new(ncfg, nc, &mut r).expect("construct")
        };
        let err3 = nn2.update_archive(&bcs, &[0.1, 0.2]); // length 2 != 3
        assert!(
            matches!(err3, Err(EvolError::DimensionMismatch { .. })),
            "got {err3:?}"
        );
    }
}
