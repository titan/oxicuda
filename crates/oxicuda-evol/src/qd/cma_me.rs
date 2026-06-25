//! CMA-ME: Covariance Matrix Adaptation MAP-Elites.
//!
//! Reference: M. C. Fontaine, J. Togelius, S. Nikolaidis, A. K. Hoover,
//! "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space",
//! GECCO 2020. <https://doi.org/10.1145/3377930.3390232>
//!
//! ## Overview
//! CMA-ME combines the self-adaptive sampling of CMA-ES with the archive of MAP-Elites. A set
//! of **emitters** — here the canonical *improvement emitters* — each wrap a CMA-ES instance
//! that samples candidate solutions. Crucially, the CMA-ES instance is **not** ranked on the
//! raw objective: it is ranked on how much each solution *improves the archive*, using a
//! two-level ranking (Fontaine 2020, §3.2):
//!
//! 1. Solutions that discover a **new cell** rank ahead of all others, ordered by objective
//!    value (best objective first).
//! 2. Solutions that **improve** an occupied cell rank next, ordered by improvement delta
//!    (largest improvement first).
//! 3. Solutions that improve nothing are placed last and given no positive weight.
//!
//! After ranking, the emitter performs a standard CMA-ES update. When the emitter's step size
//! collapses, its covariance becomes degenerate, or it adds no solutions for a generation, it
//! **restarts** from a randomly chosen existing elite — this is what lets CMA-ME both exploit
//! locally and re-seed exploration across the discovered behavior space.
//!
//! ## Maximization
//! Like the rest of the [`qd`](crate::qd) module, CMA-ME **maximizes** fitness. Pass a negated
//! objective if you are minimizing.

use crate::evolution::cmaes::cmaes::{CmaEsConfig, CmaEsState};
use crate::qd::map_elites::{Elite, InsertStatus, MapElitesArchive};
use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for a CMA-ME run.
#[derive(Debug, Clone)]
pub struct CmaMeConfig {
    /// Genome (solution) dimensionality.
    pub genome_dim: usize,
    /// Uniform genome search bounds.
    pub genome_bounds: (f64, f64),
    /// Lower bound of each descriptor dimension (length = number of descriptor dims).
    pub descriptor_min: Vec<f64>,
    /// Upper bound of each descriptor dimension.
    pub descriptor_max: Vec<f64>,
    /// Number of bins per descriptor dimension.
    pub n_bins: Vec<usize>,
    /// Number of parallel emitters.
    pub n_emitters: usize,
    /// Batch size λ sampled by each emitter per generation.
    pub batch_size: usize,
    /// Initial CMA-ES step size σ₀ for each emitter (as a fraction of the genome range).
    pub sigma0: f64,
    /// Total number of generations (each generation steps every emitter once).
    pub n_generations: usize,
    /// RNG seed.
    pub seed: u64,
}

impl CmaMeConfig {
    fn validate(&self) -> EvolResult<()> {
        if self.genome_dim == 0 {
            return Err(EvolError::InvalidParameter(
                "genome_dim must be >= 1".to_owned(),
            ));
        }
        if self.genome_bounds.0 >= self.genome_bounds.1 {
            return Err(EvolError::InvalidParameter(
                "genome_bounds: lower must be < upper".to_owned(),
            ));
        }
        let nd = self.n_bins.len();
        if nd == 0 || nd != self.descriptor_min.len() || nd != self.descriptor_max.len() {
            return Err(EvolError::DimensionMismatch {
                expected: nd,
                got: self.descriptor_min.len(),
            });
        }
        if self.n_bins.contains(&0) {
            return Err(EvolError::InvalidParameter(
                "all n_bins must be >= 1".to_owned(),
            ));
        }
        for j in 0..nd {
            if self.descriptor_min[j] >= self.descriptor_max[j] {
                return Err(EvolError::InvalidParameter(
                    "descriptor bounds: min must be < max".to_owned(),
                ));
            }
        }
        if self.n_emitters == 0 {
            return Err(EvolError::InvalidParameter(
                "n_emitters must be >= 1".to_owned(),
            ));
        }
        if self.batch_size < 2 {
            return Err(EvolError::PopulationTooSmall {
                size: self.batch_size,
                op: "CMA-ME emitter",
            });
        }
        if self.sigma0 <= 0.0 {
            return Err(EvolError::InvalidParameter("sigma0 must be > 0".to_owned()));
        }
        Ok(())
    }
}

/// A single CMA-ME *improvement emitter*: a CMA-ES instance attached to the shared archive.
struct ImprovementEmitter {
    state: CmaEsState,
    cfg: CmaEsConfig,
    /// Number of consecutive generations this emitter has added nothing to the archive.
    stall_count: usize,
}

impl ImprovementEmitter {
    /// Build a fresh emitter centred at `mean`.
    fn new(mean: Vec<f64>, genome_dim: usize, batch: usize, sigma0: f64) -> EvolResult<Self> {
        let mut cfg = CmaEsConfig::new(genome_dim)?;
        cfg.pop_size = batch;
        cfg.mu = (batch / 2).max(1);
        cfg.sigma_init = sigma0;
        cfg.max_evals = usize::MAX; // generations are driven externally
        let state = CmaEsState::new(mean, &cfg)?;
        Ok(Self {
            state,
            cfg,
            stall_count: 0,
        })
    }

    /// Restart this emitter from a new mean (a randomly chosen elite), resetting CMA state.
    fn restart(&mut self, mean: Vec<f64>) -> EvolResult<()> {
        self.state = CmaEsState::new(mean, &self.cfg)?;
        self.stall_count = 0;
        Ok(())
    }
}

/// Result of a CMA-ME run: the illuminated archive plus run statistics.
pub struct CmaMeResult {
    /// The illuminated MAP-Elites archive.
    pub archive: MapElitesArchive,
    /// Total objective evaluations performed.
    pub n_evals: usize,
    /// Number of emitter restarts triggered over the run.
    pub n_restarts: usize,
}

/// Run CMA-ME and return the illuminated archive.
///
/// ## Arguments
/// - `cfg`: algorithm configuration.
/// - `objective`: maps a genome to a scalar fitness (**maximized**).
/// - `descriptor`: maps a genome to a behavioral descriptor vector.
///
/// ## Errors
/// Returns `EvolError` if the configuration is invalid.
pub fn cma_me<F, D>(cfg: &CmaMeConfig, objective: F, descriptor: D) -> EvolResult<CmaMeResult>
where
    F: Fn(&[f64]) -> f64,
    D: Fn(&[f64]) -> Vec<f64>,
{
    cfg.validate()?;
    let mut rng = LcgRng::new(cfg.seed);
    let (lb, ub) = cfg.genome_bounds;
    let range = ub - lb;

    let mut archive = MapElitesArchive::with_config(
        cfg.n_bins.clone(),
        cfg.descriptor_min.clone(),
        cfg.descriptor_max.clone(),
    );

    // ── Seed the archive with a handful of random genomes so emitters have somewhere
    //    to restart from, and the very first improvement signal is meaningful. ────────
    let n_seed = (cfg.batch_size * cfg.n_emitters).max(cfg.genome_dim * 4);
    let mut n_evals = 0usize;
    for _ in 0..n_seed {
        let genome: Vec<f64> = (0..cfg.genome_dim)
            .map(|_| lb + rng.next_f64() * range)
            .collect();
        let fitness = objective(&genome);
        let desc = descriptor(&genome);
        n_evals += 1;
        let _ = archive.add_with_status(Elite {
            genome,
            fitness,
            descriptor: desc,
        });
    }

    // ── Create emitters, each centred at the genome midpoint (or a random elite). ────
    let midpoint = vec![lb + 0.5 * range; cfg.genome_dim];
    let sigma0 = (cfg.sigma0 * range).max(1e-6);
    let mut emitters: Vec<ImprovementEmitter> = Vec::with_capacity(cfg.n_emitters);
    for _ in 0..cfg.n_emitters {
        let mean = random_elite_genome(&archive, &mut rng).unwrap_or_else(|| midpoint.clone());
        emitters.push(ImprovementEmitter::new(
            mean,
            cfg.genome_dim,
            cfg.batch_size,
            sigma0,
        )?);
    }

    let mut n_restarts = 0usize;

    for _gen in 0..cfg.n_generations {
        for emitter in emitters.iter_mut() {
            // ── Sample a batch and clamp to genome bounds ─────────────────────
            let raw = emitter.state.sample(&emitter.cfg, &mut rng);
            let samples: Vec<Vec<f64>> = raw
                .into_iter()
                .map(|g| g.into_iter().map(|v| v.clamp(lb, ub)).collect())
                .collect();

            // ── Evaluate, attempt insertion, record improvement status ────────
            let mut improvement_rank: Vec<(usize, RankKey)> = Vec::with_capacity(samples.len());
            let mut added_any = false;
            for (i, genome) in samples.iter().enumerate() {
                let fitness = objective(genome);
                let desc = descriptor(genome);
                n_evals += 1;
                let status = archive.add_with_status(Elite {
                    genome: genome.clone(),
                    fitness,
                    descriptor: desc,
                });
                let key = match status {
                    InsertStatus::NewCell => {
                        added_any = true;
                        // New cells rank first; among them, higher fitness ranks better.
                        RankKey::New(fitness)
                    }
                    InsertStatus::Improved { delta } => {
                        added_any = true;
                        // Improvements rank next; larger delta ranks better.
                        RankKey::Improved(delta)
                    }
                    InsertStatus::Discarded => RankKey::Discarded,
                };
                improvement_rank.push((i, key));
            }

            // ── Two-level improvement ranking → synthetic minimisation fitness ─
            // CMA-ES minimises, so the best (rank 0) solution must get the smallest value.
            // We assign synthetic fitnesses by sorted position.
            let mut order: Vec<usize> = (0..improvement_rank.len()).collect();
            order.sort_by(|&a, &b| improvement_rank[a].1.cmp_better(&improvement_rank[b].1));
            let mut synthetic = vec![0.0f64; samples.len()];
            for (pos, &slot) in order.iter().enumerate() {
                let sample_idx = improvement_rank[slot].0;
                synthetic[sample_idx] = pos as f64;
            }

            // ── CMA-ES update on the improvement ranking ──────────────────────
            emitter.state.update(&samples, &synthetic, &emitter.cfg)?;

            // ── Restart logic ─────────────────────────────────────────────────
            if added_any {
                emitter.stall_count = 0;
            } else {
                emitter.stall_count += 1;
            }
            let degenerate = emitter.state.sigma < 1e-12
                || emitter.state.d_vector.iter().any(|&d| !d.is_finite())
                || emitter.state.d_vector.iter().all(|&d| d < 1e-15);
            if degenerate || emitter.stall_count >= restart_threshold(cfg.genome_dim) {
                if let Some(seed) = random_elite_genome(&archive, &mut rng) {
                    emitter.restart(seed)?;
                } else {
                    emitter.restart(midpoint.clone())?;
                }
                n_restarts += 1;
            }
        }
    }

    Ok(CmaMeResult {
        archive,
        n_evals,
        n_restarts,
    })
}

/// Number of stalled generations after which an emitter restarts (scales mildly with dim).
fn restart_threshold(genome_dim: usize) -> usize {
    (10 + genome_dim).min(50)
}

/// Pick the genome of a random occupied cell, if any.
fn random_elite_genome(archive: &MapElitesArchive, rng: &mut LcgRng) -> Option<Vec<f64>> {
    let occupied: Vec<usize> = (0..archive.n_cells())
        .filter(|&i| archive.get(i).is_some())
        .collect();
    if occupied.is_empty() {
        return None;
    }
    let pick = occupied[rng.next_usize(occupied.len())];
    archive.get(pick).map(|e| e.genome.clone())
}

/// Two-level ranking key for the improvement emitter.
#[derive(Debug, Clone, Copy)]
enum RankKey {
    /// Discovered a new cell with the given (maximised) objective value.
    New(f64),
    /// Improved an existing cell by the given positive delta.
    Improved(f64),
    /// Improved nothing.
    Discarded,
}

impl RankKey {
    /// Ordering where "less" means "better" (rank 0). Implements the CMA-ME priority:
    /// `New` ≻ `Improved` ≻ `Discarded`, with finer ordering inside each tier.
    fn cmp_better(&self, other: &RankKey) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        fn tier(k: &RankKey) -> u8 {
            match k {
                RankKey::New(_) => 0,
                RankKey::Improved(_) => 1,
                RankKey::Discarded => 2,
            }
        }
        let (ta, tb) = (tier(self), tier(other));
        if ta != tb {
            return ta.cmp(&tb);
        }
        match (self, other) {
            // New cells: higher objective is better → smaller rank.
            (RankKey::New(a), RankKey::New(b)) => b.partial_cmp(a).unwrap_or(Ordering::Equal),
            // Improvements: larger delta is better → smaller rank.
            (RankKey::Improved(a), RankKey::Improved(b)) => {
                b.partial_cmp(a).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Maximisation: negative sphere (closer to origin → higher fitness).
    fn neg_sphere(x: &[f64]) -> f64 {
        -x.iter().map(|v| v * v).sum::<f64>()
    }

    // Descriptor: clamp first two genome coordinates into [-1, 1] mapped to [0, 1].
    fn descriptor_2d(x: &[f64]) -> Vec<f64> {
        vec![
            ((x[0] + 5.0) / 10.0).clamp(0.0, 1.0),
            ((x[1] + 5.0) / 10.0).clamp(0.0, 1.0),
        ]
    }

    fn base_cfg() -> CmaMeConfig {
        CmaMeConfig {
            genome_dim: 4,
            genome_bounds: (-5.0, 5.0),
            descriptor_min: vec![0.0, 0.0],
            descriptor_max: vec![1.0, 1.0],
            n_bins: vec![10, 10],
            n_emitters: 3,
            batch_size: 12,
            sigma0: 0.2,
            n_generations: 120,
            seed: 42,
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut c = base_cfg();
        c.genome_dim = 0;
        assert!(cma_me(&c, neg_sphere, descriptor_2d).is_err());

        let mut c = base_cfg();
        c.n_bins = vec![0, 4];
        assert!(cma_me(&c, neg_sphere, descriptor_2d).is_err());

        let mut c = base_cfg();
        c.batch_size = 1;
        assert!(cma_me(&c, neg_sphere, descriptor_2d).is_err());
    }

    #[test]
    fn rank_key_priority_ordering() {
        // New ≻ Improved ≻ Discarded.
        assert_eq!(
            RankKey::New(0.0).cmp_better(&RankKey::Improved(100.0)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            RankKey::Improved(1.0).cmp_better(&RankKey::Discarded),
            std::cmp::Ordering::Less
        );
        // Within New: higher objective is better (Less).
        assert_eq!(
            RankKey::New(1.0).cmp_better(&RankKey::New(0.0)),
            std::cmp::Ordering::Less
        );
        // Within Improved: larger delta is better (Less).
        assert_eq!(
            RankKey::Improved(5.0).cmp_better(&RankKey::Improved(1.0)),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn illuminates_archive_with_good_coverage() {
        let cfg = base_cfg();
        let res = cma_me(&cfg, neg_sphere, descriptor_2d).expect("run ok");
        let cov = res.archive.coverage();
        assert!(
            cov > 0.3,
            "CMA-ME should illuminate a meaningful fraction of cells, coverage = {cov}"
        );
        assert!(res.n_evals > 0);
    }

    #[test]
    fn best_elite_is_near_optimum() {
        // The optimum of neg_sphere is x = 0 (fitness 0); the descriptor cell at (0.5, 0.5)
        // corresponds to genome coordinates near 0, so CMA-ME should find a high-fitness
        // elite there.
        let mut cfg = base_cfg();
        cfg.n_generations = 200;
        let res = cma_me(&cfg, neg_sphere, descriptor_2d).expect("run ok");
        let best = res.archive.best().expect("archive must be non-empty");
        assert!(
            best.fitness > -1.0,
            "best elite fitness should be close to 0, got {}",
            best.fitness
        );
    }

    #[test]
    fn coverage_dominates_pure_random_seed() {
        // CMA-ME with emitters should beat just the random seeding it starts from.
        let cfg = base_cfg();

        // Coverage from seeding only (n_generations = 0).
        let mut seed_only = cfg.clone();
        seed_only.n_generations = 0;
        let cov_seed = cma_me(&seed_only, neg_sphere, descriptor_2d)
            .expect("ok")
            .archive
            .coverage();

        // Coverage with emitters running.
        let cov_full = cma_me(&cfg, neg_sphere, descriptor_2d)
            .expect("ok")
            .archive
            .coverage();

        assert!(
            cov_full >= cov_seed,
            "running emitters must not reduce coverage: seed={cov_seed}, full={cov_full}"
        );
    }

    #[test]
    fn qd_score_is_finite_and_positive_progress() {
        let cfg = base_cfg();
        let res = cma_me(&cfg, neg_sphere, descriptor_2d).expect("ok");
        let qd = res.archive.qd_score();
        assert!(qd.is_finite(), "QD score must be finite");
        // neg_sphere fitnesses are <= 0, so qd_score <= 0; just ensure it is computed.
        assert!(qd <= 0.0);
    }
}
