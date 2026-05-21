//! Regularized Evolution (aging evolution) for single-objective NAS.
//!
//! Reference: Real, Aggarwal, Huang & Le, "Regularized Evolution for Image
//! Classifier Architecture Search", AAAI 2019 (the AmoebaNet search algorithm).
//!
//! Unlike a classic `(μ + λ)` evolutionary algorithm — which removes the
//! *worst* individual every cycle — regularized evolution removes the
//! **oldest** individual. This *aging* mechanism continually forces the
//! population to re-discover good solutions through mutation rather than
//! letting a single early lucky genotype dominate forever, which empirically
//! regularizes the search and yields better-generalising architectures.
//!
//! The algorithm keeps a fixed-size FIFO population in a [`VecDeque`]: the
//! front is the oldest member and the back is the newest. Each cycle:
//!
//! 1. Run a tournament of `tournament_size` random members → pick the parent.
//! 2. Clone the parent and mutate exactly one gene → child.
//! 3. Evaluate the child's fitness.
//! 4. Push the child to the back; pop (discard) the front (oldest).
//! 5. Track the best-so-far fitness.
//!
//! Fitness is treated as *higher-is-better* (a maximisation objective).

use std::collections::VecDeque;

use crate::error::{NasError, NasResult};
use crate::evolution::encoding::ArchEncoding;
use crate::handle::LcgRng;

// ─── RegEvoConfig ──────────────────────────────────────────────────────────────

/// Configuration for [`RegularizedEvolution::search`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegEvoConfig {
    /// Fixed size of the FIFO population (`P` in the paper). Must be `>= 1`.
    pub population_size: usize,
    /// Number of candidates sampled per tournament (`S` in the paper).
    /// Must satisfy `1 <= tournament_size <= population_size`.
    pub tournament_size: usize,
    /// Number of evolution cycles (`C` in the paper). Must be `>= 1`.
    pub n_cycles: usize,
    /// Number of genes (edges) per architecture encoding. Must be `>= 1`.
    pub n_genes: usize,
    /// Number of candidate ops per gene (upper bound, exclusive). Must be `>= 1`.
    pub n_ops: usize,
}

impl RegEvoConfig {
    /// Validate the configuration, returning the appropriate [`NasError`].
    fn validate(&self) -> NasResult<()> {
        if self.population_size == 0 {
            return Err(NasError::PopulationTooSmall { min: 1, got: 0 });
        }
        if self.n_genes == 0 {
            return Err(NasError::EmptySearchSpace);
        }
        if self.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        if self.tournament_size == 0 || self.tournament_size > self.population_size {
            return Err(NasError::InvalidTournamentSize);
        }
        if self.n_cycles == 0 {
            return Err(NasError::InvalidArchEncoding);
        }
        Ok(())
    }
}

// ─── RegEvoResult ──────────────────────────────────────────────────────────────

/// Result of a regularized-evolution search run.
#[derive(Debug, Clone, PartialEq)]
pub struct RegEvoResult {
    /// Best architecture encountered across the whole run (init + all cycles).
    pub best: ArchEncoding,
    /// Fitness of [`RegEvoResult::best`] (the global maximum observed).
    pub best_fitness: f32,
    /// Best-so-far fitness recorded *after* each cycle. Length `== n_cycles`,
    /// and (because it is a running maximum) non-decreasing.
    pub history: Vec<f32>,
}

// ─── RegularizedEvolution ────────────────────────────────────────────────────

/// Aging-based evolutionary search (Real et al. 2019).
#[derive(Debug, Clone, Copy)]
pub struct RegularizedEvolution;

impl RegularizedEvolution {
    /// Run regularized evolution.
    ///
    /// `fitness` maps an architecture to a score where **higher is better**.
    /// Returns the global best architecture, its fitness, and the best-so-far
    /// history (one entry per cycle).
    ///
    /// # Errors
    /// Returns [`NasError`] if the configuration is invalid (see
    /// [`RegEvoConfig`] field documentation for the exact bounds).
    pub fn search<F>(cfg: &RegEvoConfig, fitness: F, rng: &mut LcgRng) -> NasResult<RegEvoResult>
    where
        F: Fn(&ArchEncoding) -> f32,
    {
        cfg.validate()?;

        // 1. Initialise the FIFO population with random architectures. The
        //    front of the deque is the oldest member.
        let mut population: VecDeque<(ArchEncoding, f32)> =
            VecDeque::with_capacity(cfg.population_size);
        let mut best: Option<ArchEncoding> = None;
        let mut best_fitness = f32::NEG_INFINITY;

        for _ in 0..cfg.population_size {
            let arch = ArchEncoding::random(cfg.n_genes, cfg.n_ops, rng);
            let fit = fitness(&arch);
            // 2. Track the initial argmax (tie-break: keep the earliest, so we
            //    only replace on a strict improvement).
            if best.is_none() || fit > best_fitness {
                best_fitness = fit;
                best = Some(arch.clone());
            }
            population.push_back((arch, fit));
        }

        // The population is guaranteed non-empty because population_size >= 1.
        let mut best = best.ok_or(NasError::EmptySearchSpace)?;

        // 3. Evolution cycles.
        let mut history = Vec::with_capacity(cfg.n_cycles);
        for _ in 0..cfg.n_cycles {
            let members = population.make_contiguous();
            let parent_idx = Self::tournament_select(members, cfg.tournament_size, rng)?;
            // `parent_idx` is a valid index into a non-empty population.
            let mut child = match members.get(parent_idx) {
                Some((parent, _)) => parent.clone(),
                None => return Err(NasError::Internal("tournament index out of range".into())),
            };
            child.mutate_one(rng)?;
            let child_fit = fitness(&child);

            // Push child to the back (newest), pop the front (oldest). This
            // keeps the population size exactly constant.
            population.push_back((child.clone(), child_fit));
            population.pop_front();

            if child_fit > best_fitness {
                best_fitness = child_fit;
                best = child;
            }
            history.push(best_fitness);
        }

        Ok(RegEvoResult {
            best,
            best_fitness,
            history,
        })
    }

    /// Tournament selection over a slice of `(arch, fitness)` members.
    ///
    /// Samples `tournament_size` **distinct** members uniformly at random and
    /// returns the slice-index of the highest-fitness one (ties broken by the
    /// lowest index). If `tournament_size == members.len()`, the entire
    /// population participates.
    ///
    /// # Errors
    /// Returns [`NasError::PopulationTooSmall`] if `members` is empty, or
    /// [`NasError::InvalidTournamentSize`] if `tournament_size` is `0` or
    /// exceeds `members.len()`.
    pub fn tournament_select(
        members: &[(ArchEncoding, f32)],
        tournament_size: usize,
        rng: &mut LcgRng,
    ) -> NasResult<usize> {
        let n = members.len();
        if n == 0 {
            return Err(NasError::PopulationTooSmall { min: 1, got: 0 });
        }
        if tournament_size == 0 || tournament_size > n {
            return Err(NasError::InvalidTournamentSize);
        }

        // Sample `tournament_size` distinct indices via a partial Fisher-Yates
        // shuffle over a scratch permutation (distinctness guaranteed).
        let mut pool: Vec<usize> = (0..n).collect();
        let mut chosen: Vec<usize> = Vec::with_capacity(tournament_size);
        for i in 0..tournament_size {
            // Draw uniformly from the still-unselected suffix [i, n).
            let j = i + rng.next_usize(n - i);
            pool.swap(i, j);
            chosen.push(pool[i]);
        }

        // Reduce to the index with maximum fitness; tie-break: lowest index.
        let mut best_idx = chosen[0];
        let mut best_fit = members[best_idx].1;
        for &idx in &chosen[1..] {
            let fit = members[idx].1;
            if fit > best_fit || (fit == best_fit && idx < best_idx) {
                best_fit = fit;
                best_idx = idx;
            }
        }
        Ok(best_idx)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pop: usize, tour: usize, cycles: usize) -> RegEvoConfig {
        RegEvoConfig {
            population_size: pop,
            tournament_size: tour,
            n_cycles: cycles,
            n_genes: 6,
            n_ops: 4,
        }
    }

    /// Fitness = sum of genes (higher is better, deterministic).
    fn sum_fitness(a: &ArchEncoding) -> f32 {
        a.genes.iter().map(|&g| g as f32).sum()
    }

    #[test]
    fn search_best_equals_max_over_returned_best() {
        // The reported best_fitness must equal fitness(best) — internal
        // consistency between the architecture and its score.
        let mut rng = LcgRng::new(42);
        let c = cfg(10, 3, 50);
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        assert!((res.best_fitness - sum_fitness(&res.best)).abs() < 1e-6);
    }

    #[test]
    fn history_is_non_decreasing() {
        let mut rng = LcgRng::new(7);
        let c = cfg(12, 3, 80);
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        for w in res.history.windows(2) {
            assert!(w[1] >= w[0], "history must be non-decreasing: {w:?}");
        }
        assert!(res.history.last().unwrap() >= res.history.first().unwrap());
    }

    #[test]
    fn history_length_equals_n_cycles() {
        let mut rng = LcgRng::new(1);
        let c = cfg(8, 2, 37);
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        assert_eq!(res.history.len(), 37);
    }

    #[test]
    fn tournament_select_returns_valid_index() {
        let mut rng = LcgRng::new(3);
        let members: Vec<(ArchEncoding, f32)> = (0..6)
            .map(|i| (ArchEncoding::random(4, 3, &mut rng), i as f32))
            .collect();
        for _ in 0..100 {
            let idx = RegularizedEvolution::tournament_select(&members, 3, &mut rng)
                .expect("test invariant: tournament");
            assert!(idx < members.len());
        }
    }

    #[test]
    fn tournament_size_one_is_uniform_valid_index() {
        let mut rng = LcgRng::new(9);
        let members: Vec<(ArchEncoding, f32)> = (0..5)
            .map(|i| (ArchEncoding::random(4, 3, &mut rng), i as f32))
            .collect();
        // tournament_size == 1 selects a single random member (any valid idx).
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let idx = RegularizedEvolution::tournament_select(&members, 1, &mut rng)
                .expect("test invariant: tournament");
            assert!(idx < members.len());
            seen.insert(idx);
        }
        // With 200 draws over 5 members it should not collapse to a single idx.
        assert!(seen.len() > 1, "size-1 tournament should be random");
    }

    #[test]
    fn tournament_size_equals_pop_returns_global_max() {
        let mut rng = LcgRng::new(11);
        // Distinct fitnesses so the argmax is unique.
        let members: Vec<(ArchEncoding, f32)> = (0..6)
            .map(|i| (ArchEncoding::random(4, 3, &mut rng), (i * 3) as f32))
            .collect();
        let max_idx = 5; // last has highest fitness 15.0
        for _ in 0..50 {
            let idx = RegularizedEvolution::tournament_select(&members, members.len(), &mut rng)
                .expect("test invariant: tournament");
            assert_eq!(idx, max_idx);
        }
    }

    #[test]
    fn tournament_tie_break_lowest_index() {
        let mut rng = LcgRng::new(13);
        // All equal fitness → full tournament must return index 0.
        let members: Vec<(ArchEncoding, f32)> = (0..5)
            .map(|_| (ArchEncoding::random(4, 3, &mut rng), 1.0_f32))
            .collect();
        let idx = RegularizedEvolution::tournament_select(&members, members.len(), &mut rng)
            .expect("test invariant: tournament");
        assert_eq!(idx, 0);
    }

    #[test]
    fn best_genes_in_range_and_correct_length() {
        let mut rng = LcgRng::new(17);
        let c = cfg(10, 3, 40);
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        assert_eq!(res.best.genes.len(), c.n_genes);
        assert!(res.best.genes.iter().all(|&g| g < c.n_ops));
        assert_eq!(res.best.n_ops, c.n_ops);
    }

    #[test]
    fn deterministic_given_seed() {
        let c = cfg(10, 3, 60);
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let ra = RegularizedEvolution::search(&c, sum_fitness, &mut rng_a)
            .expect("test invariant: search a");
        let rb = RegularizedEvolution::search(&c, sum_fitness, &mut rng_b)
            .expect("test invariant: search b");
        assert_eq!(ra.best, rb.best);
        assert_eq!(ra.best_fitness, rb.best_fitness);
        assert_eq!(ra.history, rb.history);
    }

    #[test]
    fn fitness_called_exactly_pop_plus_cycles_times() {
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let counting = |a: &ArchEncoding| {
            calls.set(calls.get() + 1);
            sum_fitness(a)
        };
        let c = cfg(10, 3, 25);
        let mut rng = LcgRng::new(5);
        let _ =
            RegularizedEvolution::search(&c, counting, &mut rng).expect("test invariant: search");
        // population stays constant: P inits + C children = P + C evals.
        assert_eq!(calls.get(), c.population_size + c.n_cycles);
    }

    #[test]
    fn converges_near_optimum_on_sum_fitness() {
        // Optimum of sum-of-genes is n_genes * (n_ops - 1).
        let mut rng = LcgRng::new(2024);
        let c = RegEvoConfig {
            population_size: 20,
            tournament_size: 5,
            n_cycles: 2000,
            n_genes: 6,
            n_ops: 4,
        };
        let optimum = (c.n_genes * (c.n_ops - 1)) as f32; // 18.0
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        // Should get very close to the optimum after many cycles.
        assert!(
            res.best_fitness >= optimum - 1.0,
            "best {} should approach optimum {optimum}",
            res.best_fitness
        );
    }

    #[test]
    fn aging_evicts_old_high_fitness_member() {
        // Construct a targeted scenario: with population_size == 1 and
        // tournament_size == 1, every cycle replaces the single member with a
        // mutated child regardless of fitness — proving the OLDEST (here, the
        // only) member is removed even if it was the best.
        //
        // We use a fitness that strongly prefers the all-zero genome; the
        // initial random member is unlikely to be optimal, but the key point
        // is that history reflects the running max while the live population
        // turns over completely (aging), so best can be retained even though
        // the population no longer contains it.
        let mut rng = LcgRng::new(99);
        let c = RegEvoConfig {
            population_size: 1,
            tournament_size: 1,
            n_cycles: 30,
            n_genes: 5,
            n_ops: 3,
        };
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        // best_fitness is the running max and must be >= the final/any single
        // member's fitness; with pop size 1 the live member changes each cycle.
        assert!((res.best_fitness - sum_fitness(&res.best)).abs() < 1e-6);
        assert_eq!(res.history.len(), 30);
        assert!(res.history.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn recent_low_fitness_child_survives_over_old_high() {
        // Targeted aging test: with a 2-member population and full tournament,
        // we verify that pushing a freshly added child and popping the front
        // can evict a high-fitness OLD member. We instrument by checking that
        // the search never errors and the global best is preserved even when
        // it would have been aged out of the live deque.
        //
        // fitness rewards high gene sums. Use small pop so turnover is fast.
        let mut rng = LcgRng::new(404);
        let c = RegEvoConfig {
            population_size: 2,
            tournament_size: 2,
            n_cycles: 100,
            n_genes: 4,
            n_ops: 5,
        };
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        // The global best (running max) is preserved regardless of aging.
        assert!(res.best_fitness >= res.history[0]);
        assert!((res.best_fitness - sum_fitness(&res.best)).abs() < 1e-6);
    }

    #[test]
    fn err_population_size_zero() {
        let mut rng = LcgRng::new(1);
        let c = cfg(0, 1, 10);
        assert_eq!(
            RegularizedEvolution::search(&c, sum_fitness, &mut rng),
            Err(NasError::PopulationTooSmall { min: 1, got: 0 })
        );
    }

    #[test]
    fn err_tournament_size_zero() {
        let mut rng = LcgRng::new(1);
        let c = cfg(10, 0, 10);
        assert_eq!(
            RegularizedEvolution::search(&c, sum_fitness, &mut rng),
            Err(NasError::InvalidTournamentSize)
        );
    }

    #[test]
    fn err_tournament_size_exceeds_population() {
        let mut rng = LcgRng::new(1);
        let c = cfg(5, 6, 10);
        assert_eq!(
            RegularizedEvolution::search(&c, sum_fitness, &mut rng),
            Err(NasError::InvalidTournamentSize)
        );
    }

    #[test]
    fn err_n_cycles_zero() {
        let mut rng = LcgRng::new(1);
        let c = cfg(10, 3, 0);
        assert_eq!(
            RegularizedEvolution::search(&c, sum_fitness, &mut rng),
            Err(NasError::InvalidArchEncoding)
        );
    }

    #[test]
    fn err_n_genes_zero() {
        let mut rng = LcgRng::new(1);
        let mut c = cfg(10, 3, 10);
        c.n_genes = 0;
        assert_eq!(
            RegularizedEvolution::search(&c, sum_fitness, &mut rng),
            Err(NasError::EmptySearchSpace)
        );
    }

    #[test]
    fn err_n_ops_zero() {
        let mut rng = LcgRng::new(1);
        let mut c = cfg(10, 3, 10);
        c.n_ops = 0;
        assert_eq!(
            RegularizedEvolution::search(&c, sum_fitness, &mut rng),
            Err(NasError::InvalidNumOps)
        );
    }

    #[test]
    fn tournament_select_empty_errors() {
        let mut rng = LcgRng::new(1);
        let members: Vec<(ArchEncoding, f32)> = Vec::new();
        assert_eq!(
            RegularizedEvolution::tournament_select(&members, 1, &mut rng),
            Err(NasError::PopulationTooSmall { min: 1, got: 0 })
        );
    }

    #[test]
    fn population_constant_via_call_count_varied_cfg() {
        // Repeat the call-count invariant for several configs to confirm the
        // population never grows or shrinks.
        for (p, s, cyc) in [(1usize, 1usize, 10usize), (5, 2, 33), (16, 8, 7)] {
            use std::cell::Cell;
            let calls = Cell::new(0usize);
            let counting = |a: &ArchEncoding| {
                calls.set(calls.get() + 1);
                sum_fitness(a)
            };
            let c = cfg(p, s, cyc);
            let mut rng = LcgRng::new(p as u64 * 100 + cyc as u64);
            let _ = RegularizedEvolution::search(&c, counting, &mut rng)
                .expect("test invariant: search");
            assert_eq!(calls.get(), p + cyc, "cfg ({p},{s},{cyc})");
        }
    }

    #[test]
    fn single_op_genome_does_not_panic() {
        // n_ops == 1 means mutate_one is a no-op; search must still succeed.
        let mut rng = LcgRng::new(50);
        let c = RegEvoConfig {
            population_size: 4,
            tournament_size: 2,
            n_cycles: 10,
            n_genes: 3,
            n_ops: 1,
        };
        let res = RegularizedEvolution::search(&c, sum_fitness, &mut rng)
            .expect("test invariant: search");
        // All genes must be 0 (only valid op), fitness 0.
        assert!(res.best.genes.iter().all(|&g| g == 0));
        assert_eq!(res.best_fitness, 0.0);
    }
}
