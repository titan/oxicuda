//! Differential Evolution (DE) implementation.
//!
//! Supports DE/rand/1, DE/best/1, DE/rand-to-best/1, DE/rand/2, DE/current-to-best/2,
//! and jDE (self-adaptive F and CR per individual).
//!
//! Reference: R. Storn & K. Price, "Differential Evolution — A Simple and Efficient
//! Heuristic for Global Optimization over Continuous Spaces", J. Global Optim. 1997.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Mutation strategy variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeStrategy {
    /// `v = pop[r1] + F * (pop[r2] - pop[r3])` — three random, mutually distinct donors.
    Rand1,
    /// `v = best + F * (pop[r1] - pop[r2])` — base is the current best individual.
    Best1,
    /// `v = pop[r1] + F * (best - pop[r1]) + F * (pop[r2] - pop[r3])`.
    RandToBest1,
    /// `v = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])` — five donors.
    Rand2,
    /// `v = x + F * (best - x) + F * (pop[r1] - pop[r2])`.
    CurrentToBest2,
}

/// Hyper-parameters for a DE run.
#[derive(Debug, Clone)]
pub struct DeConfig {
    /// Problem dimension.
    pub n_dims: usize,
    /// Population size (must be ≥ 4 for `Rand1`; ≥ 6 for `Rand2`).
    pub pop_size: usize,
    /// Mutation scale factor F ∈ (0, 2]. Default 0.8.
    pub f: f64,
    /// Crossover rate CR ∈ [0, 1]. Default 0.9.
    pub cr: f64,
    /// Mutation strategy.
    pub strategy: DeStrategy,
    /// Maximum objective evaluations.
    pub max_evals: usize,
    /// Convergence threshold on best fitness.
    pub tol: f64,
    /// jDE self-adaptive F and CR.
    pub adaptive: bool,
}

impl DeConfig {
    /// Build a default `DeConfig` for dimension `n_dims`.
    pub fn default_for(n_dims: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_dims,
            pop_size: 10 * n_dims,
            f: 0.8,
            cr: 0.9,
            strategy: DeStrategy::Rand1,
            max_evals: 100_000,
            tol: 1e-6,
            adaptive: false,
        })
    }
}

/// Mutable DE population state.
pub struct DeState {
    /// Population matrix: `pop_size × n_dims` (row-major).
    pub population: Vec<Vec<f64>>,
    /// Current fitness of each individual.
    pub fitness: Vec<f64>,
    /// Per-individual F values for jDE.
    pub f_vals: Vec<f64>,
    /// Per-individual CR values for jDE.
    pub cr_vals: Vec<f64>,
    /// Search bounds shared by all dimensions.
    pub bounds: (f64, f64),
    /// Total evaluations consumed.
    pub n_evals: usize,
}

impl DeState {
    /// Randomly initialise the population and evaluate.
    pub fn new(cfg: &DeConfig, bounds: (f64, f64), rng: &mut LcgRng) -> EvolResult<Self> {
        let min_size = match cfg.strategy {
            DeStrategy::Rand2 => 6,
            _ => 4,
        };
        if cfg.pop_size < min_size {
            return Err(EvolError::PopulationTooSmall {
                size: cfg.pop_size,
                op: "DE mutation",
            });
        }
        if bounds.0 >= bounds.1 {
            return Err(EvolError::InvalidParameter(
                "bounds: lower must be < upper".to_owned(),
            ));
        }
        let range = bounds.1 - bounds.0;
        let population: Vec<Vec<f64>> = (0..cfg.pop_size)
            .map(|_| {
                (0..cfg.n_dims)
                    .map(|_| bounds.0 + rng.next_f64() * range)
                    .collect()
            })
            .collect();

        Ok(Self {
            fitness: vec![f64::INFINITY; cfg.pop_size],
            f_vals: vec![cfg.f; cfg.pop_size],
            cr_vals: vec![cfg.cr; cfg.pop_size],
            population,
            bounds,
            n_evals: 0,
        })
    }

    /// Return indices of `count` distinct individuals all different from `exclude`.
    fn distinct_indices(&self, count: usize, exclude: usize, rng: &mut LcgRng) -> Vec<usize> {
        let n = self.population.len();
        let mut all: Vec<usize> = (0..n).filter(|&i| i != exclude).collect();
        // Fisher-Yates partial shuffle
        for i in 0..count.min(all.len()) {
            let j = i + rng.next_usize(all.len() - i);
            all.swap(i, j);
        }
        all[..count.min(all.len())].to_vec()
    }

    /// Build a mutant vector for individual `target` using the configured strategy.
    fn mutate(
        &self,
        target: usize,
        f_scale: f64,
        best_idx: usize,
        rng: &mut LcgRng,
        cfg: &DeConfig,
    ) -> Vec<f64> {
        let n = cfg.n_dims;
        let (lb, ub) = self.bounds;
        match cfg.strategy {
            DeStrategy::Rand1 => {
                let r = self.distinct_indices(3, target, rng);
                let (r1, r2, r3) = (r[0], r[1], r[2]);
                (0..n)
                    .map(|d| {
                        let v = self.population[r1][d]
                            + f_scale * (self.population[r2][d] - self.population[r3][d]);
                        v.max(lb).min(ub)
                    })
                    .collect()
            }
            DeStrategy::Best1 => {
                let r = self.distinct_indices(2, target, rng);
                let (r1, r2) = (r[0], r[1]);
                (0..n)
                    .map(|d| {
                        let v = self.population[best_idx][d]
                            + f_scale * (self.population[r1][d] - self.population[r2][d]);
                        v.max(lb).min(ub)
                    })
                    .collect()
            }
            DeStrategy::RandToBest1 => {
                let r = self.distinct_indices(3, target, rng);
                let (r1, r2, r3) = (r[0], r[1], r[2]);
                (0..n)
                    .map(|d| {
                        let v = self.population[r1][d]
                            + f_scale * (self.population[best_idx][d] - self.population[r1][d])
                            + f_scale * (self.population[r2][d] - self.population[r3][d]);
                        v.max(lb).min(ub)
                    })
                    .collect()
            }
            DeStrategy::Rand2 => {
                let r = self.distinct_indices(5, target, rng);
                let (r1, r2, r3, r4, r5) = (r[0], r[1], r[2], r[3], r[4]);
                (0..n)
                    .map(|d| {
                        let v = self.population[r1][d]
                            + f_scale * (self.population[r2][d] - self.population[r3][d])
                            + f_scale * (self.population[r4][d] - self.population[r5][d]);
                        v.max(lb).min(ub)
                    })
                    .collect()
            }
            DeStrategy::CurrentToBest2 => {
                let r = self.distinct_indices(2, target, rng);
                let (r1, r2) = (r[0], r[1]);
                (0..n)
                    .map(|d| {
                        let v = self.population[target][d]
                            + f_scale * (self.population[best_idx][d] - self.population[target][d])
                            + f_scale * (self.population[r1][d] - self.population[r2][d]);
                        v.max(lb).min(ub)
                    })
                    .collect()
            }
        }
    }

    /// Execute one DE generation (evaluate all unevaluated individuals + one selection step).
    pub fn step<F: Fn(&[f64]) -> f64>(
        &mut self,
        f: &F,
        cfg: &DeConfig,
        rng: &mut LcgRng,
    ) -> EvolResult<()> {
        let pop_size = cfg.pop_size;
        let n = cfg.n_dims;

        // Evaluate any un-evaluated individuals (first call)
        for i in 0..pop_size {
            if self.fitness[i].is_infinite() {
                self.fitness[i] = f(&self.population[i]);
                self.n_evals += 1;
            }
        }

        // Find current best
        let best_idx = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for target in 0..pop_size {
            // jDE: adapt F and CR
            let (fi, cri) = if cfg.adaptive {
                let fi = if rng.next_f64() < 0.1 {
                    0.1 + rng.next_f64() * 0.9 // U(0.1, 1.0)
                } else {
                    self.f_vals[target]
                };
                let cri = if rng.next_f64() < 0.1 {
                    rng.next_f64() // U(0, 1)
                } else {
                    self.cr_vals[target]
                };
                (fi, cri)
            } else {
                (cfg.f, cfg.cr)
            };

            let mutant = self.mutate(target, fi, best_idx, rng, cfg);

            // Binomial crossover
            let jrand = rng.next_usize(n);
            let trial: Vec<f64> = (0..n)
                .map(|d| {
                    if d == jrand || rng.next_f64() < cri {
                        mutant[d]
                    } else {
                        self.population[target][d]
                    }
                })
                .collect();

            let trial_fit = f(&trial);
            self.n_evals += 1;

            // Greedy selection
            if trial_fit <= self.fitness[target] {
                self.population[target] = trial;
                self.fitness[target] = trial_fit;
                if cfg.adaptive {
                    self.f_vals[target] = fi;
                    self.cr_vals[target] = cri;
                }
            }
        }

        Ok(())
    }

    /// Run DE to convergence.
    pub fn run<F: Fn(&[f64]) -> f64>(
        &mut self,
        f: F,
        cfg: &DeConfig,
        rng: &mut LcgRng,
    ) -> EvolResult<(Vec<f64>, f64)> {
        while self.n_evals < cfg.max_evals {
            self.step(&f, cfg, rng)?;
            let (best, best_f) = self.best()?;
            if best_f < cfg.tol {
                return Ok((best.clone(), best_f));
            }
        }
        let (best, best_f) = self.best()?;
        Ok((best.clone(), best_f))
    }

    /// Return a reference to the best individual and its fitness.
    pub fn best(&self) -> EvolResult<(&Vec<f64>, f64)> {
        let (best_idx, &best_f) = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(EvolError::EmptyPopulation)?;
        Ok((&self.population[best_idx], best_f))
    }
}
