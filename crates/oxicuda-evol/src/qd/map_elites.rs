//! MAP-Elites: Illuminating Search Spaces by Mapping Elites.
//!
//! Reference: Mouret & Clune 2015 GECCO.
//! "Illuminating Search Spaces by Mapping Elites."
//!
//! ## IMPORTANT — Maximization
//! MAP-Elites is a **maximization** algorithm: higher fitness values are better.
//! This is a deliberate contrast to the rest of the `oxicuda-evol` crate, which
//! minimizes fitness. When using MAP-Elites alongside other algorithms, take care
//! to negate your objective if you have a minimization problem.
//!
//! ## Overview
//! The algorithm maintains a grid of "elite" solutions. Each cell in the grid
//! corresponds to a region of behavioral descriptor space. Only the best solution
//! seen for each cell is retained. The algorithm:
//!
//! 1. **Initializes** a random population, placing each solution into its cell
//!    (replacing a prior occupant only if the new solution has higher fitness).
//! 2. **Iterates** by picking a random occupied cell, applying Gaussian mutation
//!    to its genome, evaluating the offspring, and trying to place it.
//!
//! The result is an `MapElitesArchive` — a coverage map of high-quality, diverse solutions.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Hyper-parameters for a MAP-Elites run.
#[derive(Debug, Clone)]
pub struct MapElitesConfig {
    /// Genome dimensionality (must be >= 1).
    pub genome_dim: usize,
    /// Uniform search bounds for all genome dimensions.
    pub genome_bounds: (f64, f64),
    /// Lower bounds for each descriptor dimension (length = n_descriptor_dims).
    pub descriptor_min: Vec<f64>,
    /// Upper bounds for each descriptor dimension (must be > descriptor_min elementwise).
    pub descriptor_max: Vec<f64>,
    /// Number of bins per descriptor dimension (length = n_descriptor_dims, all >= 1).
    pub n_bins: Vec<usize>,
    /// Number of random genomes to evaluate during initialization.
    pub n_init: usize,
    /// Number of mutation-based iterations after initialization.
    pub n_iters: usize,
    /// Standard deviation of Gaussian mutation applied to each genome dimension.
    pub sigma: f64,
    /// RNG seed.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Archive types
// ---------------------------------------------------------------------------

/// A single elite stored in one cell of the MAP-Elites archive.
#[derive(Debug, Clone)]
pub struct Elite {
    /// The genome of this elite solution.
    pub genome: Vec<f64>,
    /// The fitness of this elite (higher is better — MAP-Elites maximizes).
    pub fitness: f64,
    /// The behavioral descriptor of this elite.
    pub descriptor: Vec<f64>,
}

/// The MAP-Elites archive: a multi-dimensional grid of elites.
pub struct MapElitesArchive {
    /// Flattened row-major grid of cells; `None` means unoccupied.
    pub cells: Vec<Option<Elite>>,
    /// Number of bins per descriptor dimension.
    pub n_bins: Vec<usize>,
    /// Lower bounds for each descriptor dimension.
    pub descriptor_min: Vec<f64>,
    /// Upper bounds for each descriptor dimension.
    pub descriptor_max: Vec<f64>,
    /// Number of descriptor dimensions.
    n_descriptor_dims: usize,
}

impl MapElitesArchive {
    /// Create a new empty archive from descriptor space configuration.
    fn new(n_bins: Vec<usize>, descriptor_min: Vec<f64>, descriptor_max: Vec<f64>) -> Self {
        let total_cells: usize = n_bins.iter().product();
        let n_descriptor_dims = n_bins.len();
        Self {
            cells: vec![None; total_cells],
            n_bins,
            descriptor_min,
            descriptor_max,
            n_descriptor_dims,
        }
    }

    /// Compute the flat (row-major) cell index for a descriptor vector.
    ///
    /// Each dimension is binned as:
    /// `bin_j = floor((d_j - min_j) / (max_j - min_j) * n_bins_j).clamp(0, n_bins_j - 1)`
    ///
    /// The flat index is: `sum_j( bin_j * product(n_bins[j+1..]) )`
    pub fn cell_index(&self, descriptor: &[f64]) -> usize {
        let mut idx = 0_usize;
        let mut stride = 1_usize;

        // Compute strides: stride for dimension j = product of n_bins[j+1..]
        // We accumulate from right to left.
        // Precompute suffix products.
        let nd = self.n_descriptor_dims;
        let mut strides = vec![1_usize; nd];
        for j in (0..nd).rev() {
            strides[j] = stride;
            stride *= self.n_bins[j];
        }

        for j in 0..nd {
            let d_j = descriptor[j];
            let min_j = self.descriptor_min[j];
            let max_j = self.descriptor_max[j];
            let n_bins_j = self.n_bins[j];

            let raw_bin = ((d_j - min_j) / (max_j - min_j) * n_bins_j as f64).floor() as isize;
            let bin_j = raw_bin.clamp(0, n_bins_j as isize - 1) as usize;
            idx += bin_j * strides[j];
        }
        idx
    }

    /// Fraction of cells that are occupied (in [0, 1]).
    pub fn coverage(&self) -> f64 {
        let total = self.cells.len();
        if total == 0 {
            return 0.0;
        }
        let occupied = self.cells.iter().filter(|c| c.is_some()).count();
        occupied as f64 / total as f64
    }

    /// Sum of fitness values over all occupied cells.
    pub fn qd_score(&self) -> f64 {
        self.cells
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|e| e.fitness)
            .sum()
    }

    /// The highest-fitness elite across all occupied cells, or `None` if the archive is empty.
    pub fn best(&self) -> Option<&Elite> {
        self.cells.iter().filter_map(|c| c.as_ref()).max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Number of occupied cells.
    pub fn n_elites(&self) -> usize {
        self.cells.iter().filter(|c| c.is_some()).count()
    }

    /// Try to insert an elite into the archive.
    ///
    /// Replaces the existing occupant only if `candidate.fitness > current.fitness`.
    fn try_insert(&mut self, cell_idx: usize, candidate: Elite) {
        let should_insert = match &self.cells[cell_idx] {
            None => true,
            Some(existing) => candidate.fitness > existing.fitness,
        };
        if should_insert {
            self.cells[cell_idx] = Some(candidate);
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_config(cfg: &MapElitesConfig) -> EvolResult<()> {
    if cfg.genome_dim == 0 {
        return Err(EvolError::InvalidParameter(
            "genome_dim must be >= 1".to_owned(),
        ));
    }
    if cfg.genome_bounds.0 >= cfg.genome_bounds.1 {
        return Err(EvolError::InvalidParameter(
            "genome_bounds: lower must be < upper".to_owned(),
        ));
    }
    let nd = cfg.n_bins.len();
    if nd != cfg.descriptor_min.len() || nd != cfg.descriptor_max.len() {
        return Err(EvolError::DimensionMismatch {
            expected: nd,
            got: if cfg.descriptor_min.len() != nd {
                cfg.descriptor_min.len()
            } else {
                cfg.descriptor_max.len()
            },
        });
    }
    if cfg.n_bins.contains(&0) {
        return Err(EvolError::InvalidParameter(
            "all n_bins must be >= 1".to_owned(),
        ));
    }
    for j in 0..nd {
        if cfg.descriptor_min[j] >= cfg.descriptor_max[j] {
            return Err(EvolError::InvalidParameter(
                "descriptor bounds: min must be < max".to_owned(),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Main algorithm
// ---------------------------------------------------------------------------

/// Run MAP-Elites and return the filled archive.
///
/// ## Maximization
/// MAP-Elites **maximizes** fitness. A candidate replaces the current cell occupant
/// if and only if `candidate.fitness > current.fitness`. Pass negated objective values
/// if you have a minimization problem.
///
/// ## Arguments
/// - `cfg`: algorithm configuration
/// - `fitness_fn`: maps a genome to a scalar fitness (maximize)
/// - `descriptor_fn`: maps a genome to a behavioral descriptor vector
///
/// ## Errors
/// Returns `EvolError` if the config is invalid.
pub fn map_elites<F, D>(
    cfg: &MapElitesConfig,
    fitness_fn: F,
    descriptor_fn: D,
) -> EvolResult<MapElitesArchive>
where
    F: Fn(&[f64]) -> f64,
    D: Fn(&[f64]) -> Vec<f64>,
{
    validate_config(cfg)?;

    let mut rng = LcgRng::new(cfg.seed);
    let (lb, ub) = cfg.genome_bounds;
    let range = ub - lb;

    let mut archive = MapElitesArchive::new(
        cfg.n_bins.clone(),
        cfg.descriptor_min.clone(),
        cfg.descriptor_max.clone(),
    );

    // --- Initialization phase ---
    for _ in 0..cfg.n_init {
        let genome: Vec<f64> = (0..cfg.genome_dim)
            .map(|_| lb + rng.next_f64() * range)
            .collect();
        let fitness = fitness_fn(&genome);
        let descriptor = descriptor_fn(&genome);
        let cell_idx = archive.cell_index(&descriptor);
        archive.try_insert(
            cell_idx,
            Elite {
                genome,
                fitness,
                descriptor,
            },
        );
    }

    // --- Iteration phase ---
    for _ in 0..cfg.n_iters {
        // Collect occupied cell indices
        let occupied: Vec<usize> = (0..archive.cells.len())
            .filter(|&i| archive.cells[i].is_some())
            .collect();

        if occupied.is_empty() {
            continue;
        }

        // Pick a random occupied cell
        let chosen_cell = occupied[rng.next_usize(occupied.len())];
        let parent_genome = archive.cells[chosen_cell]
            .as_ref()
            .expect("occupied cell must have elite")
            .genome
            .clone();

        // Apply Gaussian mutation and clamp to genome bounds
        let offspring_genome: Vec<f64> = parent_genome
            .iter()
            .map(|&g| (g + cfg.sigma * rng.next_normal()).clamp(lb, ub))
            .collect();

        let fitness = fitness_fn(&offspring_genome);
        let descriptor = descriptor_fn(&offspring_genome);
        let cell_idx = archive.cell_index(&descriptor);

        archive.try_insert(
            cell_idx,
            Elite {
                genome: offspring_genome,
                fitness,
                descriptor,
            },
        );
    }

    Ok(archive)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: negative sphere (maximization — higher is better)
    fn neg_sphere(x: &[f64]) -> f64 {
        -x.iter().map(|v| v * v).sum::<f64>()
    }

    // Test 1: Single-cell grid — archive has exactly 1 occupied cell with best fitness
    #[test]
    fn test_single_cell_grid() {
        let cfg = MapElitesConfig {
            genome_dim: 3,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![1],
            n_init: 200,
            n_iters: 0,
            sigma: 0.1,
            seed: 1,
        };
        // fitness = -||x||^2 (maximization: closer to 0 is better)
        let fitness_fn = |x: &[f64]| -x.iter().map(|v| v * v).sum::<f64>();
        let descriptor_fn = |_: &[f64]| vec![0.5]; // all in same cell

        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");
        assert_eq!(
            archive.n_elites(),
            1,
            "single-cell grid must have exactly 1 elite"
        );
        let elite = archive.best().expect("must have best");
        // The stored elite should be the best fitness seen (highest, i.e. closest to 0)
        assert!(elite.fitness <= 0.0, "fitness should be <= 0 (neg sphere)");
    }

    // Test 2 (LOAD-BEARING): Coverage non-decreasing
    #[test]
    fn test_coverage_non_decreasing() {
        // 10×10 2-D grid.
        // We compare runs that share the same RNG prefix: with the same seed and the
        // same n_init, the first n_init initialization steps produce the same archive.
        // Adding more n_iters can only place solutions in empty cells or improve
        // existing ones (try_insert is strictly-better-only) — coverage is monotone.
        let base_cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![10, 10],
            n_init: 50,
            n_iters: 0,
            sigma: 0.2,
            seed: 2,
        };

        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0, (x[1] + 1.0) / 2.0];

        // c1: 50 init only
        let c1 = {
            let cfg = MapElitesConfig {
                n_init: 50,
                n_iters: 0,
                ..base_cfg.clone()
            };
            map_elites(&cfg, fitness_fn, descriptor_fn)
                .expect("run ok")
                .coverage()
        };
        // c2: 50 init + 50 iters — archive can only grow
        let c2 = {
            let cfg = MapElitesConfig {
                n_init: 50,
                n_iters: 50,
                ..base_cfg.clone()
            };
            map_elites(&cfg, fitness_fn, descriptor_fn)
                .expect("run ok")
                .coverage()
        };
        // c3: 50 init + 200 iters — even more opportunity
        let c3 = {
            let cfg = MapElitesConfig {
                n_init: 50,
                n_iters: 200,
                ..base_cfg.clone()
            };
            map_elites(&cfg, fitness_fn, descriptor_fn)
                .expect("run ok")
                .coverage()
        };
        // c4: 100 init + 200 iters — more init = at least as many cells
        let c4 = {
            let cfg = MapElitesConfig {
                n_init: 100,
                n_iters: 200,
                ..base_cfg.clone()
            };
            map_elites(&cfg, fitness_fn, descriptor_fn)
                .expect("run ok")
                .coverage()
        };

        assert!(
            c2 >= c1,
            "coverage must not decrease (0→50 iters): {c1} -> {c2}"
        );
        assert!(
            c3 >= c2,
            "coverage must not decrease (50→200 iters): {c2} -> {c3}"
        );
        // c4 uses same seed and same 50-init prefix + more — at least as much coverage
        // (the 100-init run processes all 50 init samples plus 50 more, reaching at
        // least the same cells as the 50-init run)
        assert!(
            c4 >= c1,
            "more evaluations must not reduce coverage: {c1} -> {c4}"
        );
    }

    // Test 3 (LOAD-BEARING): QD-score non-decreasing
    #[test]
    fn test_qd_score_non_decreasing() {
        // With the same seed and same n_init, the first n_init samples are identical.
        // Adding n_iters mutation steps can ONLY:
        //   (a) fill a previously empty cell — adds a positive fitness value, or
        //   (b) replace an existing occupant with strictly higher fitness — increases sum.
        // Therefore QD score (sum of cell fitnesses) is non-decreasing as n_iters
        // increases, PROVIDED the fitness function is non-negative.
        //
        // We use fitness = exp(-||x||^2) ∈ (0,1] so that every occupied cell
        // contributes a positive amount and the sum can only grow.
        let base_cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![10, 10],
            n_init: 50,
            n_iters: 0,
            sigma: 0.2,
            seed: 3,
        };

        // Non-negative fitness: Gaussian centered at origin
        let gauss_fitness = |x: &[f64]| -> f64 {
            let sq: f64 = x.iter().map(|v| v * v).sum();
            (-sq).exp() // in (0, 1], maximized at origin
        };
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0, (x[1] + 1.0) / 2.0];

        // q1: 50 init, 0 iters
        let q1 = {
            let cfg = MapElitesConfig {
                n_init: 50,
                n_iters: 0,
                ..base_cfg.clone()
            };
            map_elites(&cfg, gauss_fitness, descriptor_fn)
                .expect("run ok")
                .qd_score()
        };
        // q2: 50 init, 50 iters — same init prefix, more mutation → score >= q1
        let q2 = {
            let cfg = MapElitesConfig {
                n_init: 50,
                n_iters: 50,
                ..base_cfg.clone()
            };
            map_elites(&cfg, gauss_fitness, descriptor_fn)
                .expect("run ok")
                .qd_score()
        };
        // q3: 50 init, 200 iters — even more mutation → score >= q2
        let q3 = {
            let cfg = MapElitesConfig {
                n_init: 50,
                n_iters: 200,
                ..base_cfg.clone()
            };
            map_elites(&cfg, gauss_fitness, descriptor_fn)
                .expect("run ok")
                .qd_score()
        };

        // QD score is non-decreasing as n_iters increases (same n_init, same seed)
        assert!(
            q2 >= q1,
            "QD-score must not decrease (0→50 iters): {q1} -> {q2}"
        );
        assert!(
            q3 >= q2,
            "QD-score must not decrease (50→200 iters): {q2} -> {q3}"
        );
    }

    // Test 4: Only-if-better replacement
    #[test]
    fn test_only_if_better_replacement() {
        // Start with n_init=0, n_iters=0, then manually insert a high-fitness elite
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![5],
            n_init: 0,
            n_iters: 0,
            sigma: 0.1,
            seed: 4,
        };
        let fitness_fn = |_: &[f64]| 0.0;
        let descriptor_fn = |_: &[f64]| vec![0.5];
        let mut archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");

        // Manually insert a high-fitness elite
        let high_fitness_elite = Elite {
            genome: vec![0.1, 0.1],
            fitness: 100.0,
            descriptor: vec![0.5],
        };
        let cell = archive.cell_index(&[0.5]);
        archive.try_insert(cell, high_fitness_elite);
        assert_eq!(archive.cells[cell].as_ref().expect("elite").fitness, 100.0);

        // Try to insert a lower-fitness candidate
        let low_fitness_elite = Elite {
            genome: vec![0.9, 0.9],
            fitness: 50.0,
            descriptor: vec![0.5],
        };
        archive.try_insert(cell, low_fitness_elite);
        assert_eq!(
            archive.cells[cell].as_ref().expect("elite").fitness,
            100.0,
            "lower-fitness should not replace high-fitness elite"
        );
    }

    // Test 5: Descriptor binning boundary cases
    #[test]
    fn test_descriptor_binning_boundary() {
        let archive = MapElitesArchive::new(vec![10], vec![0.0], vec![1.0]);

        // descriptor=0.0 -> bin 0
        assert_eq!(
            archive.cell_index(&[0.0]),
            0,
            "descriptor=0.0 should be bin 0"
        );

        // descriptor=0.99 -> floor(0.99*10) = floor(9.9) = 9
        assert_eq!(
            archive.cell_index(&[0.99]),
            9,
            "descriptor=0.99 should be bin 9"
        );

        // descriptor=1.0 -> floor(1.0*10) = 10, clamped to 9
        assert_eq!(
            archive.cell_index(&[1.0]),
            9,
            "descriptor=1.0 should clamp to bin 9"
        );

        // descriptor=0.5 -> floor(0.5*10) = floor(5.0) = 5
        assert_eq!(
            archive.cell_index(&[0.5]),
            5,
            "descriptor=0.5 should be bin 5"
        );
    }

    // Test 6: 2-D fitness near (0,0) has highest fitness
    #[test]
    fn test_2d_neg_sphere_best_near_origin() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![-1.0, -1.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![5, 5],
            n_init: 200,
            n_iters: 300,
            sigma: 0.1,
            seed: 6,
        };
        let fitness_fn = |x: &[f64]| -(x[0] * x[0] + x[1] * x[1]);
        let descriptor_fn = |x: &[f64]| vec![x[0], x[1]];

        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");

        // The best fitness should be near 0.0 (i.e., genome near origin)
        if let Some(best) = archive.best() {
            assert!(
                best.fitness > -0.5,
                "best fitness near origin should be > -0.5, got {}",
                best.fitness
            );
        }
    }

    // Test 7: Determinism — same seed gives identical archive
    #[test]
    fn test_determinism_same_seed() {
        let cfg = MapElitesConfig {
            genome_dim: 3,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![4, 4],
            n_init: 100,
            n_iters: 50,
            sigma: 0.1,
            seed: 777,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0, (x[1] + 1.0) / 2.0];

        let archive_a = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run a");
        let archive_b = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run b");

        assert_eq!(
            archive_a.n_elites(),
            archive_b.n_elites(),
            "same seed: n_elites must match"
        );
        for (ca, cb) in archive_a.cells.iter().zip(archive_b.cells.iter()) {
            match (ca, cb) {
                (None, None) => {}
                (Some(ea), Some(eb)) => {
                    assert!(
                        (ea.fitness - eb.fitness).abs() < 1e-15,
                        "same seed: fitness must match"
                    );
                }
                _ => panic!("same seed: cell occupancy must match"),
            }
        }
    }

    // Test 8: Different seeds give different archives
    #[test]
    fn test_different_seeds_give_different_archives() {
        let base_cfg = MapElitesConfig {
            genome_dim: 3,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![4, 4],
            n_init: 100,
            n_iters: 50,
            sigma: 0.1,
            seed: 100,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0, (x[1] + 1.0) / 2.0];

        let archive_a = map_elites(&base_cfg, fitness_fn, descriptor_fn).expect("run a");
        let cfg_b = MapElitesConfig {
            seed: 200,
            ..base_cfg.clone()
        };
        let archive_b = map_elites(&cfg_b, fitness_fn, descriptor_fn).expect("run b");

        // At least one cell should differ
        let any_diff = archive_a
            .cells
            .iter()
            .zip(archive_b.cells.iter())
            .any(|(ca, cb)| match (ca, cb) {
                (Some(ea), Some(eb)) => (ea.fitness - eb.fitness).abs() > 1e-15,
                (None, Some(_)) | (Some(_), None) => true,
                (None, None) => false,
            });
        assert!(
            any_diff,
            "different seeds should produce different archives"
        );
    }

    // Test 9: Gaussian mutation stays within bounds
    #[test]
    fn test_mutation_stays_within_bounds() {
        let cfg = MapElitesConfig {
            genome_dim: 5,
            genome_bounds: (-2.0, 2.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![10],
            n_init: 50,
            n_iters: 200,
            sigma: 5.0, // large sigma to stress-test clamping
            seed: 9,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| {
            let d = x[0].clamp(cfg.genome_bounds.0, cfg.genome_bounds.1);
            vec![(d + 2.0) / 4.0]
        };

        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");
        let (lb, ub) = cfg.genome_bounds;
        for cell in archive.cells.iter().filter_map(|c| c.as_ref()) {
            for &v in &cell.genome {
                assert!(
                    v >= lb && v <= ub,
                    "genome value {v} out of bounds [{lb}, {ub}]"
                );
            }
        }
    }

    // Test 10: genome_dim=0 => InvalidParameter
    #[test]
    fn test_error_genome_dim_zero() {
        let cfg = MapElitesConfig {
            genome_dim: 0,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![5],
            n_init: 10,
            n_iters: 10,
            sigma: 0.1,
            seed: 1,
        };
        let res = map_elites(&cfg, |_| 0.0, |_| vec![0.5]);
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    // Test 11: Mismatched descriptor dims => DimensionMismatch
    #[test]
    fn test_error_mismatched_descriptor_dims() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0], // length 2
            descriptor_max: vec![1.0, 1.0], // length 2
            n_bins: vec![5],                // length 1 — mismatch
            n_init: 10,
            n_iters: 0,
            sigma: 0.1,
            seed: 1,
        };
        let res = map_elites(&cfg, |_| 0.0, |_| vec![0.5]);
        assert!(matches!(res, Err(EvolError::DimensionMismatch { .. })));
    }

    // Test 12: n_bins=[0,5] => InvalidParameter
    #[test]
    fn test_error_zero_bins() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![0, 5],
            n_init: 10,
            n_iters: 0,
            sigma: 0.1,
            seed: 1,
        };
        let res = map_elites(&cfg, |_| 0.0, |_| vec![0.5, 0.5]);
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    // Test 13: Inverted descriptor bounds => InvalidParameter
    #[test]
    fn test_error_inverted_descriptor_bounds() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![1.0],
            descriptor_max: vec![0.0], // inverted
            n_bins: vec![5],
            n_init: 10,
            n_iters: 0,
            sigma: 0.1,
            seed: 1,
        };
        let res = map_elites(&cfg, |_| 0.0, |_| vec![0.5]);
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    // Test 14: Inverted genome bounds => InvalidParameter
    #[test]
    fn test_error_inverted_genome_bounds() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (1.0, -1.0), // inverted
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![5],
            n_init: 10,
            n_iters: 0,
            sigma: 0.1,
            seed: 1,
        };
        let res = map_elites(&cfg, |_| 0.0, |_| vec![0.5]);
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    // Test 15: n_init=0, n_iters=0 => empty archive (no crash)
    #[test]
    fn test_empty_run_no_crash() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![5],
            n_init: 0,
            n_iters: 0,
            sigma: 0.1,
            seed: 1,
        };
        let archive = map_elites(&cfg, |_| 0.0, |_| vec![0.5]).expect("should not error");
        assert_eq!(archive.n_elites(), 0, "archive should be empty");
        assert!(
            archive.best().is_none(),
            "best should be None on empty archive"
        );
    }

    // Test 16: coverage() returns value in [0, 1]
    #[test]
    fn test_coverage_in_range() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![5, 5],
            n_init: 100,
            n_iters: 50,
            sigma: 0.2,
            seed: 16,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0, (x[1] + 1.0) / 2.0];
        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");
        let cov = archive.coverage();
        assert!((0.0..=1.0).contains(&cov), "coverage out of [0,1]: {cov}");
    }

    // Test 17: qd_score() is sum of occupied cell fitnesses
    #[test]
    fn test_qd_score_is_sum_of_fitnesses() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![10],
            n_init: 100,
            n_iters: 0,
            sigma: 0.1,
            seed: 17,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0];
        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");

        let manual_sum: f64 = archive
            .cells
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|e| e.fitness)
            .sum();
        assert!(
            (archive.qd_score() - manual_sum).abs() < 1e-12,
            "qd_score mismatch: {} vs {}",
            archive.qd_score(),
            manual_sum
        );
    }

    // Test 18: best() returns None on empty, Some with correct highest fitness on non-empty
    #[test]
    fn test_best_empty_and_nonempty() {
        // Empty
        let cfg_empty = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![5],
            n_init: 0,
            n_iters: 0,
            sigma: 0.1,
            seed: 18,
        };
        let empty_archive = map_elites(&cfg_empty, |_| 0.0, |_| vec![0.5]).expect("run");
        assert!(empty_archive.best().is_none());

        // Non-empty
        let cfg_full = MapElitesConfig {
            n_init: 100,
            n_iters: 50,
            ..cfg_empty.clone()
        };
        let archive = map_elites(&cfg_full, neg_sphere, |x: &[f64]| vec![(x[0] + 1.0) / 2.0])
            .expect("run ok");
        if archive.n_elites() > 0 {
            let best = archive.best().expect("non-empty archive must have best");
            for cell in archive.cells.iter().filter_map(|c| c.as_ref()) {
                assert!(
                    best.fitness >= cell.fitness - 1e-15,
                    "best fitness {} should be >= all cell fitnesses {}",
                    best.fitness,
                    cell.fitness
                );
            }
        }
    }

    // Test 19: cell_index correct on a 3-D grid (verify manually)
    #[test]
    fn test_cell_index_3d() {
        let archive =
            MapElitesArchive::new(vec![3, 4, 5], vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]);
        // descriptor = [0.0, 0.0, 0.0] -> bins [0, 0, 0] -> idx = 0*20 + 0*5 + 0 = 0
        assert_eq!(archive.cell_index(&[0.0, 0.0, 0.0]), 0);

        // descriptor = [1.0, 1.0, 1.0] -> bins [2, 3, 4] (clamped) -> idx = 2*20 + 3*5 + 4 = 40+15+4 = 59
        assert_eq!(archive.cell_index(&[1.0, 1.0, 1.0]), 59);

        // descriptor = [0.5, 0.5, 0.5]
        // bin_0 = floor(0.5*3) = floor(1.5) = 1
        // bin_1 = floor(0.5*4) = floor(2.0) = 2
        // bin_2 = floor(0.5*5) = floor(2.5) = 2
        // idx = 1*20 + 2*5 + 2 = 20+10+2 = 32
        assert_eq!(archive.cell_index(&[0.5, 0.5, 0.5]), 32);
    }

    // Test 20: n_elites() equals occupied cell count
    #[test]
    fn test_n_elites_equals_occupied_count() {
        let cfg = MapElitesConfig {
            genome_dim: 2,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![5, 5],
            n_init: 200,
            n_iters: 100,
            sigma: 0.2,
            seed: 20,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 1.0) / 2.0, (x[1] + 1.0) / 2.0];
        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");

        let manual_count = archive.cells.iter().filter(|c| c.is_some()).count();
        assert_eq!(
            archive.n_elites(),
            manual_count,
            "n_elites must equal occupied count"
        );
    }

    // Test 21: After init, all genome values in elites are within bounds
    #[test]
    fn test_init_genomes_within_bounds() {
        let cfg = MapElitesConfig {
            genome_dim: 4,
            genome_bounds: (-3.0, 3.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![5, 5],
            n_init: 300,
            n_iters: 0,
            sigma: 0.1,
            seed: 21,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| vec![(x[0] + 3.0) / 6.0, (x[1] + 3.0) / 6.0];
        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");
        let (lb, ub) = cfg.genome_bounds;
        for cell in archive.cells.iter().filter_map(|c| c.as_ref()) {
            for &v in &cell.genome {
                assert!(
                    v >= lb && v <= ub,
                    "init genome value {v} out of bounds [{lb}, {ub}]"
                );
            }
        }
    }

    // Test 22: With large sigma, mutations still clamp correctly (no out-of-bounds genome)
    #[test]
    fn test_large_sigma_clamp_correct() {
        let cfg = MapElitesConfig {
            genome_dim: 3,
            genome_bounds: (-1.0, 1.0),
            descriptor_min: vec![0.0],
            descriptor_max: vec![1.0],
            n_bins: vec![10],
            n_init: 50,
            n_iters: 500,
            sigma: 100.0, // extremely large sigma
            seed: 22,
        };
        let fitness_fn = neg_sphere;
        let descriptor_fn = |x: &[f64]| {
            let d = ((x[0] + 1.0) / 2.0).clamp(0.0, 0.9999);
            vec![d]
        };
        let archive = map_elites(&cfg, fitness_fn, descriptor_fn).expect("run ok");
        let (lb, ub) = cfg.genome_bounds;
        for cell in archive.cells.iter().filter_map(|c| c.as_ref()) {
            for &v in &cell.genome {
                assert!(
                    v >= lb && v <= ub,
                    "mutated genome value {v} out of bounds [{lb}, {ub}]"
                );
            }
        }
    }
}
