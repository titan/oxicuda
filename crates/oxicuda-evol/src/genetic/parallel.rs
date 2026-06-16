//! Parallel and Cellular Genetic Algorithm variants.
//!
//! # Overview
//! Two spatially-structured GA models that simulate the topology of parallel GAs
//! in pure sequential Rust (no threads, no GPU):
//!
//! - **Master-slave GA**: a `(μ+λ)` strategy with the API surface of a real
//!   master-slave parallel system. The master maintains one population, λ
//!   offspring are generated and evaluated each generation, and the top μ
//!   survive.
//!
//! - **Cellular GA (cGA)**: individuals live on a toroidal 2-D grid. Each
//!   cell's neighbourhood is limited to its Von Neumann (4) or Moore (8)
//!   adjacent cells; selection, reproduction, and replacement are all local.
//!   This creates emergent diversity and slows premature convergence
//!   compared to panmictic GAs.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─── Master-slave GA ─────────────────────────────────────────────────────────

/// Configuration for the master-slave `(μ+λ)` parallel GA.
#[derive(Debug, Clone)]
pub struct MasterSlaveConfig {
    /// μ — parent (master) population size.
    pub pop_size: usize,
    /// λ — number of offspring generated per generation.
    pub n_offspring: usize,
    /// Total number of generations to run.
    pub n_gens: usize,
    /// Gaussian mutation standard deviation.
    pub sigma_mut: f64,
    /// Probability of applying Gaussian mutation to each gene.
    pub p_mut: f64,
    /// Number of decision variables (genome length).
    pub n_dims: usize,
    /// Lower bound for initialisation and clamping.
    pub lb: f64,
    /// Upper bound for initialisation and clamping.
    pub ub: f64,
}

/// Output of [`master_slave_ga`].
#[derive(Debug, Clone)]
pub struct MasterSlaveResult {
    /// Best genome found across all generations.
    pub best_genome: Vec<f64>,
    /// Fitness of [`Self::best_genome`].
    pub best_fitness: f64,
    /// Best-fitness-per-generation history (length == `n_gens`).
    pub history: Vec<f64>,
}

/// Validate a [`MasterSlaveConfig`].
fn validate_ms(cfg: &MasterSlaveConfig) -> EvolResult<()> {
    if cfg.pop_size == 0 {
        return Err(EvolError::InvalidParameter(
            "pop_size must be >= 1".to_owned(),
        ));
    }
    if cfg.n_offspring == 0 {
        return Err(EvolError::InvalidParameter(
            "n_offspring must be >= 1".to_owned(),
        ));
    }
    if cfg.n_dims == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if cfg.lb >= cfg.ub {
        return Err(EvolError::InvalidParameter(format!(
            "bounds ({}, {}) invalid: lb must be < ub",
            cfg.lb, cfg.ub
        )));
    }
    Ok(())
}

/// Gaussian-mutate `genome` in-place: each gene is perturbed by `N(0, sigma)`
/// with probability `p_mut`; results are clamped to `[lb, ub]`.
fn gaussian_mutate(genome: &mut [f64], sigma: f64, p_mut: f64, lb: f64, ub: f64, rng: &mut LcgRng) {
    for gene in genome.iter_mut() {
        if rng.next_f64() < p_mut {
            *gene = (*gene + sigma * rng.next_normal()).clamp(lb, ub);
        }
    }
}

/// Produce one random genome uniformly sampled in `[lb, ub]^n_dims`.
fn random_genome(n_dims: usize, lb: f64, ub: f64, rng: &mut LcgRng) -> Vec<f64> {
    let range = ub - lb;
    (0..n_dims).map(|_| lb + rng.next_f64() * range).collect()
}

/// k-tournament selection over a flat fitness slice; returns the index of the winner.
fn tournament_idx(fitnesses: &[f64], k: usize, rng: &mut LcgRng) -> usize {
    let n = fitnesses.len();
    let mut best = rng.next_usize(n);
    for _ in 1..k {
        let candidate = rng.next_usize(n);
        if fitnesses[candidate] < fitnesses[best] {
            best = candidate;
        }
    }
    best
}

/// Run the master-slave `(μ+λ)` GA.
///
/// Each generation:
/// 1. Generate λ offspring by tournament-selecting one parent and applying
///    Gaussian mutation.
/// 2. Evaluate all offspring.
/// 3. Pool parents + offspring, keep the μ fittest (truncation selection).
///
/// The function is entirely sequential and deterministic for a given `rng`.
pub fn master_slave_ga<F>(
    config: &MasterSlaveConfig,
    fitness_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<MasterSlaveResult>
where
    F: Fn(&[f64]) -> f64,
{
    validate_ms(config)?;

    let MasterSlaveConfig {
        pop_size,
        n_offspring,
        n_gens,
        sigma_mut,
        p_mut,
        n_dims,
        lb,
        ub,
    } = *config;

    // Initialise population.
    let mut genomes: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| random_genome(n_dims, lb, ub, rng))
        .collect();
    let mut fitnesses: Vec<f64> = genomes.iter().map(|g| fitness_fn(g)).collect();

    let mut best_genome = genomes[0].clone();
    let mut best_fitness = fitnesses[0];
    for (g, &f) in genomes.iter().zip(fitnesses.iter()) {
        if f < best_fitness {
            best_fitness = f;
            best_genome = g.clone();
        }
    }

    let mut history = Vec::with_capacity(n_gens);

    for _gen in 0..n_gens {
        // Generate λ offspring.
        let mut offspring_genomes: Vec<Vec<f64>> = Vec::with_capacity(n_offspring);
        let mut offspring_fitnesses: Vec<f64> = Vec::with_capacity(n_offspring);

        for _ in 0..n_offspring {
            let parent_idx = tournament_idx(&fitnesses, 2, rng);
            let mut child = genomes[parent_idx].clone();
            gaussian_mutate(&mut child, sigma_mut, p_mut, lb, ub, rng);
            let f = fitness_fn(&child);
            offspring_genomes.push(child);
            offspring_fitnesses.push(f);
        }

        // Pool parents + offspring, truncate to top μ.
        genomes.extend(offspring_genomes);
        fitnesses.extend(offspring_fitnesses);

        // Sort by fitness ascending and keep the best pop_size.
        let mut indexed: Vec<usize> = (0..genomes.len()).collect();
        indexed.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(pop_size);

        // Rebuild from sorted survivors.
        let (new_genomes, new_fitnesses): (Vec<Vec<f64>>, Vec<f64>) = indexed
            .into_iter()
            .map(|idx| (genomes[idx].clone(), fitnesses[idx]))
            .unzip();
        genomes = new_genomes;
        fitnesses = new_fitnesses;

        // Track global best.
        for (g, &f) in genomes.iter().zip(fitnesses.iter()) {
            if f < best_fitness {
                best_fitness = f;
                best_genome = g.clone();
            }
        }
        history.push(best_fitness);
    }

    Ok(MasterSlaveResult {
        best_genome,
        best_fitness,
        history,
    })
}

// ─── Cellular GA ─────────────────────────────────────────────────────────────

/// Neighbourhood topology for the cellular GA grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighbourhood {
    /// Von Neumann neighbourhood: North, South, East, West (4 cells).
    VonNeumann,
    /// Moore neighbourhood: N, S, E, W, NE, NW, SE, SW (8 cells).
    Moore,
}

/// Configuration for the cellular GA.
#[derive(Debug, Clone)]
pub struct CellularGaConfig {
    /// Number of rows in the toroidal grid.
    pub grid_rows: usize,
    /// Number of columns in the toroidal grid.
    pub grid_cols: usize,
    /// Total number of generations.
    pub n_gens: usize,
    /// Neighbourhood topology for local selection.
    pub neighbourhood: Neighbourhood,
    /// Gaussian mutation standard deviation.
    pub sigma_mut: f64,
    /// Per-gene mutation probability.
    pub p_mut: f64,
    /// Number of decision variables (genome length).
    pub n_dims: usize,
    /// Lower bound.
    pub lb: f64,
    /// Upper bound.
    pub ub: f64,
}

/// Output of [`cellular_ga`].
#[derive(Debug, Clone)]
pub struct CellularGaResult {
    /// Best genome found.
    pub best_genome: Vec<f64>,
    /// Fitness of [`Self::best_genome`].
    pub best_fitness: f64,
    /// Final fitness of every grid cell, indexed `[row][col]`.
    pub grid_fitness: Vec<Vec<f64>>,
    /// Global best fitness per generation (length == `n_gens`).
    pub history: Vec<f64>,
}

/// Validate a [`CellularGaConfig`].
fn validate_cga(cfg: &CellularGaConfig) -> EvolResult<()> {
    if cfg.grid_rows == 0 {
        return Err(EvolError::InvalidParameter(
            "grid_rows must be >= 1".to_owned(),
        ));
    }
    if cfg.grid_cols == 0 {
        return Err(EvolError::InvalidParameter(
            "grid_cols must be >= 1".to_owned(),
        ));
    }
    if cfg.n_dims == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if cfg.lb >= cfg.ub {
        return Err(EvolError::InvalidParameter(format!(
            "bounds ({}, {}) invalid: lb must be < ub",
            cfg.lb, cfg.ub
        )));
    }
    Ok(())
}

/// Collect the toroidal neighbourhood (including the cell itself) for cell
/// `(row, col)` on a `rows × cols` grid.
///
/// For `VonNeumann` this yields 5 cells (self + 4 cardinal directions).
/// For `Moore` this yields 9 cells (self + 8 directions).
fn neighbourhood_cells(
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    nbh: Neighbourhood,
) -> Vec<(usize, usize)> {
    let offsets: &[(i64, i64)] = match nbh {
        Neighbourhood::VonNeumann => &[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
        Neighbourhood::Moore => &[
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ],
    };
    offsets
        .iter()
        .map(|&(dr, dc)| {
            let nr = ((row as i64 + dr).rem_euclid(rows as i64)) as usize;
            let nc = ((col as i64 + dc).rem_euclid(cols as i64)) as usize;
            (nr, nc)
        })
        .collect()
}

/// Run the cellular GA on a toroidal grid.
///
/// Each generation performs a row-by-row scan. For every cell `(i, j)`:
/// 1. Collect the cell's neighbourhood (toroidal).
/// 2. Tournament-select two parents from the neighbourhood.
/// 3. Produce one child by Gaussian mutation of the better parent.
/// 4. Replace the cell's genome if the child has lower (better) fitness.
pub fn cellular_ga<F>(
    config: &CellularGaConfig,
    fitness_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<CellularGaResult>
where
    F: Fn(&[f64]) -> f64,
{
    validate_cga(config)?;

    let CellularGaConfig {
        grid_rows,
        grid_cols,
        n_gens,
        neighbourhood,
        sigma_mut,
        p_mut,
        n_dims,
        lb,
        ub,
    } = config.clone();

    let n_cells = grid_rows * grid_cols;

    // Flatten the grid into a single Vec (row-major).
    let mut genomes: Vec<Vec<f64>> = (0..n_cells)
        .map(|_| random_genome(n_dims, lb, ub, rng))
        .collect();
    let mut fitnesses: Vec<f64> = genomes.iter().map(|g| fitness_fn(g)).collect();

    // Global best tracking.
    let init_best_idx = fitnesses
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut best_genome = genomes[init_best_idx].clone();
    let mut best_fitness = fitnesses[init_best_idx];

    let mut history = Vec::with_capacity(n_gens);

    for _gen in 0..n_gens {
        // Row-by-row scan (each cell updated once per generation).
        for row in 0..grid_rows {
            for col in 0..grid_cols {
                let nbh_cells = neighbourhood_cells(row, col, grid_rows, grid_cols, neighbourhood);

                // Tournament-select 2 distinct parents from neighbourhood.
                let nbh_len = nbh_cells.len();
                let idx_a = rng.next_usize(nbh_len);
                let mut idx_b = rng.next_usize(nbh_len);
                // Allow re-draw once to try to get a distinct parent.
                if idx_b == idx_a && nbh_len > 1 {
                    idx_b = rng.next_usize(nbh_len);
                }

                let (ra, ca) = nbh_cells[idx_a];
                let (rb, cb) = nbh_cells[idx_b];
                let fit_a = fitnesses[ra * grid_cols + ca];
                let fit_b = fitnesses[rb * grid_cols + cb];

                // Better parent is the one with lower fitness.
                let (par_row, par_col) = if fit_a <= fit_b { (ra, ca) } else { (rb, cb) };

                let mut child = genomes[par_row * grid_cols + par_col].clone();
                gaussian_mutate(&mut child, sigma_mut, p_mut, lb, ub, rng);
                let child_fit = fitness_fn(&child);

                let cell_idx = row * grid_cols + col;
                if child_fit < fitnesses[cell_idx] {
                    genomes[cell_idx] = child;
                    fitnesses[cell_idx] = child_fit;
                }
            }
        }

        // Update global best.
        for (g, &f) in genomes.iter().zip(fitnesses.iter()) {
            if f < best_fitness {
                best_fitness = f;
                best_genome = g.clone();
            }
        }
        history.push(best_fitness);
    }

    // Build grid_fitness[row][col].
    let grid_fitness: Vec<Vec<f64>> = (0..grid_rows)
        .map(|r| {
            (0..grid_cols)
                .map(|c| fitnesses[r * grid_cols + c])
                .collect()
        })
        .collect();

    Ok(CellularGaResult {
        best_genome,
        best_fitness,
        grid_fitness,
        history,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn ms_default_config() -> MasterSlaveConfig {
        MasterSlaveConfig {
            pop_size: 30,
            n_offspring: 20,
            n_gens: 500,
            sigma_mut: 0.1,
            p_mut: 0.5,
            n_dims: 5,
            lb: -5.0,
            ub: 5.0,
        }
    }

    fn cga_default_config() -> CellularGaConfig {
        CellularGaConfig {
            grid_rows: 5,
            grid_cols: 5,
            n_gens: 300,
            neighbourhood: Neighbourhood::VonNeumann,
            sigma_mut: 0.1,
            p_mut: 0.5,
            n_dims: 4,
            lb: -5.0,
            ub: 5.0,
        }
    }

    // ── Master-slave tests ───────────────────────────────────────────────────

    #[test]
    fn ms_converges_sphere_5d() {
        let mut rng = LcgRng::new(42);
        let cfg = ms_default_config();
        let res = master_slave_ga(&cfg, sphere, &mut rng).expect("should succeed");
        assert!(
            res.best_fitness < 1e-3,
            "best_fitness = {} (expected < 1e-3)",
            res.best_fitness
        );
    }

    #[test]
    fn ms_history_length_equals_n_gens() {
        let mut rng = LcgRng::new(7);
        let cfg = ms_default_config();
        let res = master_slave_ga(&cfg, sphere, &mut rng).expect("master_slave_ga should succeed");
        assert_eq!(res.history.len(), cfg.n_gens);
    }

    #[test]
    fn ms_genome_within_bounds() {
        let mut rng = LcgRng::new(99);
        let cfg = ms_default_config();
        let res = master_slave_ga(&cfg, sphere, &mut rng).expect("master_slave_ga should succeed");
        for &gene in &res.best_genome {
            assert!(
                gene >= cfg.lb && gene <= cfg.ub,
                "gene {gene} out of [{}, {}]",
                cfg.lb,
                cfg.ub
            );
        }
    }

    #[test]
    fn ms_pop_size_zero_returns_error() {
        let mut rng = LcgRng::new(1);
        let mut cfg = ms_default_config();
        cfg.pop_size = 0;
        let res = master_slave_ga(&cfg, sphere, &mut rng);
        assert!(
            matches!(res, Err(EvolError::InvalidParameter(_))),
            "expected InvalidParameter"
        );
    }

    #[test]
    fn ms_n_offspring_zero_returns_error() {
        let mut rng = LcgRng::new(2);
        let mut cfg = ms_default_config();
        cfg.n_offspring = 0;
        let res = master_slave_ga(&cfg, sphere, &mut rng);
        assert!(
            matches!(res, Err(EvolError::InvalidParameter(_))),
            "expected InvalidParameter"
        );
    }

    #[test]
    fn ms_n_offspring_larger_than_pop_size_works() {
        let mut rng = LcgRng::new(55);
        let mut cfg = ms_default_config();
        cfg.pop_size = 5;
        cfg.n_offspring = 100;
        cfg.n_gens = 200;
        let res = master_slave_ga(&cfg, sphere, &mut rng).expect("should succeed");
        assert!(res.best_fitness < 1.0);
    }

    #[test]
    fn ms_history_is_monotone_non_increasing() {
        let mut rng = LcgRng::new(13);
        let cfg = ms_default_config();
        let res = master_slave_ga(&cfg, sphere, &mut rng).expect("master_slave_ga should succeed");
        for w in res.history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "history not non-increasing: {} then {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn ms_seed_reproducibility() {
        let cfg = ms_default_config();
        let res1 =
            master_slave_ga(&cfg, sphere, &mut LcgRng::new(77)).expect("value should be present");
        let res2 =
            master_slave_ga(&cfg, sphere, &mut LcgRng::new(77)).expect("value should be present");
        assert_eq!(res1.best_fitness, res2.best_fitness);
        assert_eq!(res1.best_genome, res2.best_genome);
    }

    #[test]
    fn ms_invalid_bounds_returns_error() {
        let mut cfg = ms_default_config();
        cfg.lb = 5.0;
        cfg.ub = -5.0; // inverted
        let res = master_slave_ga(&cfg, sphere, &mut LcgRng::new(0));
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    // ── Cellular GA tests ────────────────────────────────────────────────────

    #[test]
    fn cga_von_neumann_converges_sphere() {
        let mut rng = LcgRng::new(42);
        let cfg = cga_default_config();
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("should succeed");
        assert!(
            res.best_fitness < 0.5,
            "best_fitness = {} (expected < 0.5)",
            res.best_fitness
        );
    }

    #[test]
    fn cga_moore_neighbourhood_converges() {
        let mut rng = LcgRng::new(13);
        let mut cfg = cga_default_config();
        cfg.neighbourhood = Neighbourhood::Moore;
        cfg.n_gens = 400;
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("should succeed");
        assert!(
            res.best_fitness < 1.0,
            "best_fitness = {} (expected < 1.0)",
            res.best_fitness
        );
    }

    #[test]
    fn cga_grid_fitness_shape() {
        let mut rng = LcgRng::new(7);
        let cfg = cga_default_config();
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("cellular_ga should succeed");
        assert_eq!(res.grid_fitness.len(), cfg.grid_rows);
        for row in &res.grid_fitness {
            assert_eq!(row.len(), cfg.grid_cols);
        }
    }

    #[test]
    fn cga_history_length_equals_n_gens() {
        let mut rng = LcgRng::new(3);
        let cfg = cga_default_config();
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("cellular_ga should succeed");
        assert_eq!(res.history.len(), cfg.n_gens);
    }

    #[test]
    fn cga_genome_within_bounds() {
        let mut rng = LcgRng::new(88);
        let cfg = cga_default_config();
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("cellular_ga should succeed");
        for &gene in &res.best_genome {
            assert!(
                gene >= cfg.lb && gene <= cfg.ub,
                "gene {gene} out of [{}, {}]",
                cfg.lb,
                cfg.ub
            );
        }
    }

    #[test]
    fn cga_grid_rows_zero_returns_error() {
        let mut cfg = cga_default_config();
        cfg.grid_rows = 0;
        let res = cellular_ga(&cfg, sphere, &mut LcgRng::new(0));
        assert!(matches!(res, Err(EvolError::InvalidParameter(_))));
    }

    #[test]
    fn cga_best_fitness_le_grid_min() {
        let mut rng = LcgRng::new(22);
        let cfg = cga_default_config();
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("cellular_ga should succeed");
        let grid_min = res
            .grid_fitness
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            res.best_fitness <= grid_min + 1e-12,
            "best_fitness {} > grid_min {}",
            res.best_fitness,
            grid_min
        );
    }

    #[test]
    fn cga_toroidal_wrapping_2x2() {
        // On a 2×2 grid with Von Neumann neighbourhood every cell should see
        // all 4 other distinct cells (wrapping makes them all adjacent).
        // We verify that any cell in the 2×2 grid has exactly 5 entries in its
        // neighbourhood (self + 4, but on a 2×2 some wrap to the same cell).
        let rows = 2;
        let cols = 2;
        let nbh = neighbourhood_cells(0, 0, rows, cols, Neighbourhood::VonNeumann);
        // Should yield 5 cells (offsets may alias on a 2×2 grid — that is fine
        // but the count must equal the number of offsets).
        assert_eq!(nbh.len(), 5, "VonNeumann neighbourhood size = 5");
        // Each cell coordinate must be in [0, rows) × [0, cols).
        for (r, c) in &nbh {
            assert!(*r < rows && *c < cols);
        }
    }

    #[test]
    fn cga_seed_reproducibility() {
        let cfg = cga_default_config();
        let res1 =
            cellular_ga(&cfg, sphere, &mut LcgRng::new(41)).expect("value should be present");
        let res2 =
            cellular_ga(&cfg, sphere, &mut LcgRng::new(41)).expect("value should be present");
        assert_eq!(res1.best_fitness, res2.best_fitness);
        assert_eq!(res1.best_genome, res2.best_genome);
    }

    #[test]
    fn cga_history_non_increasing() {
        let mut rng = LcgRng::new(99);
        let cfg = cga_default_config();
        let res = cellular_ga(&cfg, sphere, &mut rng).expect("cellular_ga should succeed");
        for w in res.history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "history not non-increasing: {} then {}",
                w[0],
                w[1]
            );
        }
    }
}
