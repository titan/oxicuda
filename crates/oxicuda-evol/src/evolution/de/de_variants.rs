//! Simplified high-level DE variant API with three named mutation strategies.
//!
//! This module provides a clean, ergonomic wrapper around the core DE algorithm,
//! exposing exactly the three most commonly used named variants:
//!
//! - **Best1Bin** — DE/best/1/bin: mutant = best + F*(r1 - r2)
//! - **Rand2Bin** — DE/rand/2/bin: mutant = r1 + F*(r2-r3) + F*(r4-r5)
//! - **CurrentToBest1Bin** — DE/current-to-best/1/bin: mutant = xi + F*(best-xi) + F*(r1-r2)
//!
//! # Example
//!
//! ```
//! use oxicuda_evol::evolution::de::de_variants::{De, DeConfig, DeVariant};
//! use oxicuda_evol::handle::LcgRng;
//!
//! let cfg = DeConfig {
//!     variant: DeVariant::Best1Bin,
//!     f: 0.8,
//!     cr: 0.9,
//!     pop_size: 20,
//!     max_iter: 200,
//! };
//! let mut rng = LcgRng::new(42);
//! let bounds = vec![(-5.0_f64, 5.0_f64); 3];
//! let de = De::new(cfg).expect("new should succeed");
//! let best = de.optimize(|x| x.iter().map(|v| v * v).sum(), &bounds, 3, &mut rng).expect("value should be present");
//! assert_eq!(best.len(), 3);
//! ```

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Named DE mutation strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeVariant {
    /// DE/best/1/bin — uses the global best as base vector.
    Best1Bin,
    /// DE/rand/2/bin — uses two difference pairs applied to a random base.
    /// Requires `pop_size >= 6` (checked in `De::new`).
    Rand2Bin,
    /// DE/current-to-best/1/bin — blends current individual toward the best.
    CurrentToBest1Bin,
}

/// Hyper-parameters for the simplified DE API.
#[derive(Debug, Clone)]
pub struct DeConfig {
    /// Mutation strategy variant.
    pub variant: DeVariant,
    /// Mutation scale factor F ∈ (0, 2].
    pub f: f64,
    /// Binomial crossover rate CR ∈ [0, 1].
    pub cr: f64,
    /// Population size. Minimum is 4; Rand2Bin requires ≥ 6.
    pub pop_size: usize,
    /// Maximum number of generations.
    pub max_iter: usize,
}

/// High-level Differential Evolution optimiser.
#[derive(Debug)]
pub struct De {
    config: DeConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pick `count` indices that are all mutually distinct and all differ from `exclude`.
///
/// Uses a partial Fisher-Yates shuffle on the candidate pool.
fn distinct_indices(
    pool_size: usize,
    exclude: usize,
    count: usize,
    rng: &mut LcgRng,
) -> Vec<usize> {
    let mut candidates: Vec<usize> = (0..pool_size).filter(|&i| i != exclude).collect();
    for i in 0..count.min(candidates.len()) {
        let j = i + rng.next_usize(candidates.len() - i);
        candidates.swap(i, j);
    }
    candidates[..count.min(candidates.len())].to_vec()
}

/// Clip `v` into `[lb, ub]`.
#[inline]
fn clip(v: f64, lb: f64, ub: f64) -> f64 {
    v.max(lb).min(ub)
}

// ─────────────────────────────────────────────────────────────────────────────
// De implementation
// ─────────────────────────────────────────────────────────────────────────────

impl De {
    /// Construct a new `De` optimiser, validating all configuration parameters.
    ///
    /// # Errors
    ///
    /// - `PopulationTooSmall` if `pop_size < 4` (or < 6 for `Rand2Bin`)
    /// - `InvalidParameter` if `f <= 0` or `cr < 0` or `cr > 1`
    pub fn new(config: DeConfig) -> EvolResult<Self> {
        let min_pop = match config.variant {
            DeVariant::Rand2Bin => 6,
            _ => 4,
        };
        if config.pop_size < min_pop {
            return Err(EvolError::PopulationTooSmall {
                size: config.pop_size,
                op: "DE mutation",
            });
        }
        if config.f <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "mutation scale F must be > 0".to_owned(),
            ));
        }
        if config.cr < 0.0 || config.cr > 1.0 {
            return Err(EvolError::InvalidParameter(
                "crossover rate CR must be in [0, 1]".to_owned(),
            ));
        }
        Ok(Self { config })
    }

    /// Run the DE optimisation and return the best solution found.
    ///
    /// # Arguments
    ///
    /// * `obj_fn` — Objective function to **minimise**; maps `&[f64]` → `f64`.
    /// * `bounds` — Per-dimension `(lower, upper)` bounds; must have length `dim`.
    /// * `dim`    — Problem dimension (number of decision variables).
    /// * `rng`    — Mutable LCG RNG used for all stochastic choices.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter` if `dim == 0`
    /// - `DimensionMismatch` if `bounds.len() != dim`
    pub fn optimize(
        &self,
        obj_fn: impl Fn(&[f64]) -> f64,
        bounds: &[(f64, f64)],
        dim: usize,
        rng: &mut LcgRng,
    ) -> EvolResult<Vec<f64>> {
        if dim == 0 {
            return Err(EvolError::InvalidParameter(
                "problem dimension must be >= 1".to_owned(),
            ));
        }
        if bounds.len() != dim {
            return Err(EvolError::DimensionMismatch {
                expected: dim,
                got: bounds.len(),
            });
        }

        let pop_size = self.config.pop_size;
        let f_scale = self.config.f;
        let cr = self.config.cr;

        // ── Initialise population uniformly within bounds ──────────────────
        let mut population: Vec<Vec<f64>> = (0..pop_size)
            .map(|_| {
                (0..dim)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        lb + rng.next_f64() * (ub - lb)
                    })
                    .collect()
            })
            .collect();

        // ── Evaluate initial population ────────────────────────────────────
        let mut fitness: Vec<f64> = population.iter().map(|x| obj_fn(x)).collect();

        // ── Track best individual ──────────────────────────────────────────
        let mut best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // ── Main generational loop ─────────────────────────────────────────
        for _iter in 0..self.config.max_iter {
            for target in 0..pop_size {
                // ── Build mutant vector ──────────────────────────────────
                let mutant =
                    self.build_mutant(target, best_idx, &population, dim, f_scale, bounds, rng);

                // ── Binomial crossover ───────────────────────────────────
                // At least one dimension must come from the mutant (jrand).
                let jrand = rng.next_usize(dim);
                let trial: Vec<f64> = (0..dim)
                    .map(|d| {
                        if d == jrand || rng.next_f64() < cr {
                            mutant[d]
                        } else {
                            population[target][d]
                        }
                    })
                    .collect();

                // ── Greedy selection ─────────────────────────────────────
                let trial_fit = obj_fn(&trial);
                if trial_fit <= fitness[target] {
                    population[target] = trial;
                    fitness[target] = trial_fit;
                    // Update best if this improves on the global best
                    if trial_fit < fitness[best_idx] {
                        best_idx = target;
                    }
                }
            }

            // Re-sync best_idx in case selection never updated it but fitness
            // values shifted due to tied comparisons.
            best_idx = fitness
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        Ok(population[best_idx].clone())
    }

    /// Construct a mutant vector for individual `target` according to the chosen variant.
    fn build_mutant(
        &self,
        target: usize,
        best_idx: usize,
        population: &[Vec<f64>],
        dim: usize,
        f_scale: f64,
        bounds: &[(f64, f64)],
        rng: &mut LcgRng,
    ) -> Vec<f64> {
        let pop_size = population.len();

        match self.config.variant {
            // DE/best/1/bin: v = best + F*(r1 - r2)
            DeVariant::Best1Bin => {
                let donors = distinct_indices(pop_size, target, 2, rng);
                let (r1, r2) = (donors[0], donors[1]);
                (0..dim)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        let v = population[best_idx][d]
                            + f_scale * (population[r1][d] - population[r2][d]);
                        clip(v, lb, ub)
                    })
                    .collect()
            }

            // DE/rand/2/bin: v = r1 + F*(r2-r3) + F*(r4-r5)
            // Requires pop_size >= 6; validated in new(), so safe here.
            DeVariant::Rand2Bin => {
                // Need 5 donors distinct from target and from each other.
                let donors = distinct_indices(pop_size, target, 5, rng);
                let (r1, r2, r3, r4, r5) = (donors[0], donors[1], donors[2], donors[3], donors[4]);
                (0..dim)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        let v = population[r1][d]
                            + f_scale * (population[r2][d] - population[r3][d])
                            + f_scale * (population[r4][d] - population[r5][d]);
                        clip(v, lb, ub)
                    })
                    .collect()
            }

            // DE/current-to-best/1/bin: v = xi + F*(best - xi) + F*(r1 - r2)
            DeVariant::CurrentToBest1Bin => {
                let donors = distinct_indices(pop_size, target, 2, rng);
                let (r1, r2) = (donors[0], donors[1]);
                (0..dim)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        let v = population[target][d]
                            + f_scale * (population[best_idx][d] - population[target][d])
                            + f_scale * (population[r1][d] - population[r2][d]);
                        clip(v, lb, ub)
                    })
                    .collect()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: sum of squares — global min = 0 at origin.
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn default_cfg(variant: DeVariant) -> DeConfig {
        DeConfig {
            variant,
            f: 0.8,
            cr: 0.9,
            pop_size: 20,
            max_iter: 200,
        }
    }

    fn default_bounds(dim: usize) -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0); dim]
    }

    // ── Test 1: output is within bounds for all dimensions ─────────────────
    #[test]
    fn output_in_bounds() {
        let cfg = default_cfg(DeVariant::Best1Bin);
        let mut rng = LcgRng::new(1);
        let bounds = default_bounds(4);
        let de = De::new(cfg).expect("valid config");
        let best = de
            .optimize(sphere, &bounds, 4, &mut rng)
            .expect("optimize ok");
        for (d, &v) in best.iter().enumerate() {
            let (lb, ub) = bounds[d];
            assert!(v >= lb && v <= ub, "dim {d}: {v} outside [{lb}, {ub}]");
        }
    }

    // ── Test 2: result contains no NaN or Inf values ───────────────────────
    #[test]
    fn output_finite() {
        let cfg = default_cfg(DeVariant::CurrentToBest1Bin);
        let mut rng = LcgRng::new(2);
        let bounds = default_bounds(3);
        let de = De::new(cfg).expect("valid config");
        let best = de
            .optimize(sphere, &bounds, 3, &mut rng)
            .expect("optimize ok");
        for &v in &best {
            assert!(v.is_finite(), "non-finite value in result: {v}");
        }
    }

    // ── Test 3: F very small ≈ 0 — mutation barely moves; algorithm runs ───
    #[test]
    fn f_near_zero_no_mutation() {
        let cfg = DeConfig {
            variant: DeVariant::Best1Bin,
            f: 1e-10,
            cr: 0.5,
            pop_size: 10,
            max_iter: 50,
        };
        let mut rng = LcgRng::new(3);
        let bounds = default_bounds(2);
        let de = De::new(cfg).expect("valid config");
        let best = de.optimize(sphere, &bounds, 2, &mut rng).expect("runs ok");
        assert_eq!(best.len(), 2);
    }

    // ── Test 4: CR=1 means trial always uses mutant — algorithm runs ───────
    #[test]
    fn cr_1_full_crossover() {
        let cfg = DeConfig {
            variant: DeVariant::Rand2Bin,
            f: 0.5,
            cr: 1.0,
            pop_size: 12,
            max_iter: 50,
        };
        let mut rng = LcgRng::new(4);
        let bounds = default_bounds(3);
        let de = De::new(cfg).expect("valid config");
        let best = de.optimize(sphere, &bounds, 3, &mut rng).expect("runs ok");
        assert_eq!(best.len(), 3);
    }

    // ── Test 5: all three variants complete without error ──────────────────
    #[test]
    fn all_variants_work() {
        let variants = [
            DeVariant::Best1Bin,
            DeVariant::Rand2Bin,
            DeVariant::CurrentToBest1Bin,
        ];
        for (seed, variant) in variants.into_iter().enumerate() {
            let pop_size = if variant == DeVariant::Rand2Bin {
                12
            } else {
                10
            };
            let cfg = DeConfig {
                variant,
                f: 0.7,
                cr: 0.8,
                pop_size,
                max_iter: 30,
            };
            let mut rng = LcgRng::new(seed as u64 + 100);
            let bounds = default_bounds(2);
            let de = De::new(cfg).expect("valid config");
            let best = de
                .optimize(sphere, &bounds, 2, &mut rng)
                .expect("variant should run");
            assert_eq!(best.len(), 2, "variant {variant:?} returned wrong dim");
        }
    }

    // ── Test 6: dim=0 yields InvalidParameter ─────────────────────────────
    #[test]
    fn dim_0_error() {
        let cfg = default_cfg(DeVariant::Best1Bin);
        let mut rng = LcgRng::new(6);
        let de = De::new(cfg).expect("valid config");
        let result = de.optimize(sphere, &[], 0, &mut rng);
        match result {
            Err(EvolError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 7: pop_size=3 < 4 yields PopulationTooSmall ──────────────────
    #[test]
    fn pop_size_lt_4_error() {
        let cfg = DeConfig {
            variant: DeVariant::Best1Bin,
            f: 0.8,
            cr: 0.9,
            pop_size: 3,
            max_iter: 10,
        };
        let result = De::new(cfg);
        match result {
            Err(EvolError::PopulationTooSmall { .. }) => {}
            other => panic!("expected PopulationTooSmall, got {other:?}"),
        }
    }

    // ── Test 8: max_iter=1 completes without error ─────────────────────────
    #[test]
    fn max_iter_1_ok() {
        let cfg = DeConfig {
            variant: DeVariant::CurrentToBest1Bin,
            f: 0.8,
            cr: 0.9,
            pop_size: 8,
            max_iter: 1,
        };
        let mut rng = LcgRng::new(8);
        let bounds = default_bounds(2);
        let de = De::new(cfg).expect("valid config");
        let best = de
            .optimize(sphere, &bounds, 2, &mut rng)
            .expect("single iter ok");
        assert_eq!(best.len(), 2);
    }

    // ── Test 9: better than random initial solution for 2D sphere ─────────
    #[test]
    fn better_than_random_init() {
        let cfg = DeConfig {
            variant: DeVariant::Best1Bin,
            f: 0.8,
            cr: 0.9,
            pop_size: 20,
            max_iter: 100,
        };
        let mut rng = LcgRng::new(9);
        let bounds = vec![(-5.0_f64, 5.0_f64); 2];
        let de = De::new(cfg).expect("valid config");
        let best = de
            .optimize(sphere, &bounds, 2, &mut rng)
            .expect("optimize ok");
        let fit: f64 = sphere(&best);
        assert!(
            fit < 0.1,
            "expected best fitness < 0.1 for 2D sphere after 100 iters, got {fit}"
        );
    }

    // ── Test 10: bounds mismatch yields DimensionMismatch ─────────────────
    #[test]
    fn bounds_dim_mismatch_error() {
        let cfg = default_cfg(DeVariant::Best1Bin);
        let mut rng = LcgRng::new(10);
        let bounds = vec![(-1.0, 1.0); 2]; // only 2 bounds for dim=3
        let de = De::new(cfg).expect("valid config");
        let result = de.optimize(sphere, &bounds, 3, &mut rng);
        match result {
            Err(EvolError::DimensionMismatch {
                expected: 3,
                got: 2,
            }) => {}
            other => panic!("expected DimensionMismatch{{3,2}}, got {other:?}"),
        }
    }

    // ── Test 11: Rand2Bin requires pop_size >= 6 ──────────────────────────
    #[test]
    fn rand2bin_requires_pop_ge_6() {
        let cfg = DeConfig {
            variant: DeVariant::Rand2Bin,
            f: 0.5,
            cr: 0.9,
            pop_size: 5,
            max_iter: 10,
        };
        let result = De::new(cfg);
        match result {
            Err(EvolError::PopulationTooSmall { .. }) => {}
            other => panic!("expected PopulationTooSmall, got {other:?}"),
        }
    }
}
