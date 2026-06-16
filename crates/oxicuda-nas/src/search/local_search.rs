//! Local Search NAS — hill-climbing over the discrete architecture space.
//!
//! Reference: White, Nolen & Savani, "Local Search is State of the Art for
//! Neural Architecture Search Benchmarks", AAAI 2021 / ICLR 2021 workshop.
//!
//! Local search is a deceptively strong NAS baseline: starting from one
//! architecture, it repeatedly examines the *neighbourhood* reachable by a
//! single-operation change, moves to the best strictly-improving neighbour, and
//! stops when no neighbour improves (a **local optimum**) or the iteration
//! budget is spent. On cell-based benchmarks this best-improvement hill climb
//! matches or beats far more elaborate search strategies.
//!
//! The genome is the shared [`ArchEncoding`] (one op-index per edge). A
//! single-op perturbation changes exactly one gene to one of the `n_ops - 1`
//! alternative operations, so the neighbourhood of an architecture has exactly
//! `n_genes * (n_ops - 1)` members ([`single_op_neighbors`]).
//!
//! The objective is supplied as a closure `Fn(&ArchEncoding) -> f32` where
//! **higher is better**. It may wrap a trained-accuracy lookup, a FLOP/latency
//! penalty, or — to avoid any supernet training at all — a zero-cost proxy from
//! [`crate::proxy::zero_cost`] applied to signals materialised for the
//! architecture.

use crate::error::{NasError, NasResult};
use crate::evolution::encoding::ArchEncoding;
use crate::handle::LcgRng;

// ─── ArchSpace ─────────────────────────────────────────────────────────────────

/// Discrete architecture space for the [`ArchEncoding`] genome: `n_genes`
/// edges, each choosing one of `n_ops` candidate operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchSpace {
    /// Number of edges (genes) per architecture. Must be `>= 1`.
    pub n_genes: usize,
    /// Number of candidate ops per edge (upper bound, exclusive). Must be `>= 1`.
    pub n_ops: usize,
}

impl ArchSpace {
    /// Construct an architecture space.
    #[must_use]
    pub fn new(n_genes: usize, n_ops: usize) -> Self {
        Self { n_genes, n_ops }
    }

    /// Validate the dimensions.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `n_genes == 0`.
    /// - [`NasError::InvalidNumOps`] if `n_ops == 0`.
    pub fn validate(&self) -> NasResult<()> {
        if self.n_genes == 0 {
            return Err(NasError::EmptySearchSpace);
        }
        if self.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        Ok(())
    }

    /// Number of single-op neighbours of any architecture in this space:
    /// `n_genes * (n_ops - 1)`.
    #[must_use]
    pub fn neighborhood_size(&self) -> usize {
        self.n_genes * self.n_ops.saturating_sub(1)
    }

    /// Sample a uniformly random architecture from this space.
    #[must_use]
    pub fn sample(&self, rng: &mut LcgRng) -> ArchEncoding {
        ArchEncoding::random(self.n_genes, self.n_ops, rng)
    }
}

// ─── Neighbourhood ─────────────────────────────────────────────────────────────

/// Enumerate the single-op-perturbation neighbourhood of `arch`.
///
/// For every gene position `p` and every alternative op value `v != arch[p]`,
/// emit the architecture obtained by setting gene `p` to `v` and leaving all
/// other genes unchanged. Each neighbour therefore differs from `arch` in
/// **exactly one** gene, and the returned vector has length
/// `arch.len() * (arch.n_ops - 1)` (empty when `n_ops <= 1`).
///
/// The enumeration order is deterministic: outer loop over positions ascending,
/// inner loop over op values ascending (skipping the current value).
#[must_use]
pub fn single_op_neighbors(arch: &ArchEncoding) -> Vec<ArchEncoding> {
    let n_ops = arch.n_ops;
    if n_ops <= 1 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(arch.genes.len() * (n_ops - 1));
    for (pos, &cur) in arch.genes.iter().enumerate() {
        for v in 0..n_ops {
            if v == cur {
                continue;
            }
            let mut genes = arch.genes.clone();
            genes[pos] = v;
            out.push(ArchEncoding { genes, n_ops });
        }
    }
    out
}

// ─── Config / Result ───────────────────────────────────────────────────────────

/// Configuration for [`LocalSearchNas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalSearchConfig {
    /// Maximum number of hill-climbing rounds. Each round evaluates the whole
    /// single-op neighbourhood of the current architecture and (on a strict
    /// improvement) moves to the best neighbour. `0` performs no rounds and
    /// returns the start architecture (its score is still computed once).
    pub max_iters: usize,
}

impl LocalSearchConfig {
    /// Construct a config with the given iteration budget.
    #[must_use]
    pub fn new(max_iters: usize) -> Self {
        Self { max_iters }
    }
}

/// Result of a local-search run.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Best (local-optimum or budget-terminated) architecture found.
    pub best: ArchEncoding,
    /// Objective value of [`SearchResult::best`].
    pub best_score: f32,
    /// Accepted-score trajectory: the start score followed by the score after
    /// each accepted improvement. Length `== 1 + (number of accepted moves)`
    /// and, because only strict improvements are accepted, strictly increasing
    /// (hence non-decreasing).
    pub trajectory: Vec<f32>,
    /// Total number of objective evaluations spent (start + all neighbours).
    pub evals: usize,
    /// Number of hill-climbing rounds actually performed (`<= max_iters`).
    pub iters: usize,
}

impl SearchResult {
    /// `true` if the run terminated at a local optimum (no neighbour strictly
    /// improved) rather than by exhausting the iteration budget.
    #[must_use]
    pub fn converged(&self, max_iters: usize) -> bool {
        self.iters < max_iters
    }
}

// ─── LocalSearchNas ─────────────────────────────────────────────────────────────

/// Best-improvement hill-climbing architecture search (White et al. 2021).
#[derive(Debug, Clone, Copy)]
pub struct LocalSearchNas {
    config: LocalSearchConfig,
}

impl LocalSearchNas {
    /// Create a searcher from its configuration.
    #[must_use]
    pub fn new(config: LocalSearchConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &LocalSearchConfig {
        &self.config
    }

    /// Run local search from a uniformly random start architecture in `space`.
    ///
    /// `objective` maps an architecture to a score where **higher is better**.
    ///
    /// # Errors
    /// Returns [`NasError::EmptySearchSpace`] / [`NasError::InvalidNumOps`] for
    /// an invalid `space` (see [`ArchSpace::validate`]).
    pub fn search<F>(
        &self,
        space: &ArchSpace,
        objective: F,
        rng: &mut LcgRng,
    ) -> NasResult<SearchResult>
    where
        F: Fn(&ArchEncoding) -> f32,
    {
        space.validate()?;
        let start = space.sample(rng);
        self.search_from(start, objective)
    }

    /// Run local search from an explicit start architecture.
    ///
    /// Useful for restarts, warm-starts, or deterministic tests. The start is
    /// evaluated once to seed the trajectory; with `max_iters == 0` the start is
    /// returned unchanged.
    ///
    /// # Errors
    /// Returns [`NasError::EmptySearchSpace`] if `start` has no genes.
    pub fn search_from<F>(&self, start: ArchEncoding, objective: F) -> NasResult<SearchResult>
    where
        F: Fn(&ArchEncoding) -> f32,
    {
        if start.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }

        let mut current = start;
        let mut current_score = objective(&current);
        let mut evals = 1usize;
        let mut trajectory = vec![current_score];
        let mut iters = 0usize;

        while iters < self.config.max_iters {
            let neighbors = single_op_neighbors(&current);
            if neighbors.is_empty() {
                // Degenerate space (n_ops <= 1): the current point is, trivially,
                // a local optimum.
                break;
            }

            // Best-improvement: scan the whole neighbourhood, keep the strictly
            // best neighbour. Ties resolve to the lowest enumeration index
            // (first-found) for determinism.
            let mut best_idx: Option<usize> = None;
            let mut best_score = current_score;
            for (k, nb) in neighbors.iter().enumerate() {
                let s = objective(nb);
                evals += 1;
                if s > best_score {
                    best_score = s;
                    best_idx = Some(k);
                }
            }
            iters += 1;

            match best_idx {
                Some(k) => {
                    current = neighbors[k].clone();
                    current_score = best_score;
                    trajectory.push(current_score);
                }
                None => break, // local optimum: no neighbour improves
            }
        }

        Ok(SearchResult {
            best: current,
            best_score: current_score,
            trajectory,
            evals,
            iters,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Separable objective: `-Σ_p |gene_p - target_p|`. Strictly maximised
    /// (value 0) only at `genes == target`; every gene is independent, so the
    /// landscape has a single global optimum and no spurious local optima.
    fn separable_objective(target: &[usize]) -> impl Fn(&ArchEncoding) -> f32 + '_ {
        move |a: &ArchEncoding| {
            -a.genes
                .iter()
                .zip(target.iter())
                .map(|(&g, &t)| (g as f32 - t as f32).abs())
                .sum::<f32>()
        }
    }

    fn arch(genes: &[usize], n_ops: usize) -> ArchEncoding {
        ArchEncoding {
            genes: genes.to_vec(),
            n_ops,
        }
    }

    #[test]
    fn neighborhood_count_is_single_op() {
        let a = arch(&[0, 1, 2, 0], 4);
        let nbrs = single_op_neighbors(&a);
        // n_genes * (n_ops - 1) = 4 * 3 = 12.
        assert_eq!(nbrs.len(), 12);
        assert_eq!(nbrs.len(), ArchSpace::new(4, 4).neighborhood_size());
    }

    #[test]
    fn neighbors_differ_in_exactly_one_gene_and_are_valid() {
        let a = arch(&[0, 1, 2, 3], 5);
        for nb in single_op_neighbors(&a) {
            assert_eq!(nb.genes.len(), a.genes.len());
            assert_eq!(nb.n_ops, a.n_ops);
            assert!(nb.genes.iter().all(|&g| g < nb.n_ops));
            let diff = a
                .genes
                .iter()
                .zip(nb.genes.iter())
                .filter(|(x, y)| x != y)
                .count();
            assert_eq!(diff, 1, "exactly one gene must change");
        }
    }

    #[test]
    fn neighborhood_empty_when_single_op() {
        let a = arch(&[0, 0, 0], 1);
        assert!(single_op_neighbors(&a).is_empty());
    }

    #[test]
    fn reaches_global_optimum_on_separable_objective() {
        // From any start, best-improvement converges to the unique optimum.
        let target = [3usize, 0, 2, 1, 4, 2];
        let space = ArchSpace::new(target.len(), 5);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(50));
        let mut rng = LcgRng::new(2024);
        let res = searcher
            .search(&space, separable_objective(&target), &mut rng)
            .expect("test invariant: search");
        assert_eq!(res.best.genes, target.to_vec());
        assert!(res.best_score.abs() < 1e-6, "score = {}", res.best_score);
    }

    #[test]
    fn terminates_at_verified_local_optimum() {
        // With a generous budget the run stops at a local optimum: assert that
        // no single-op neighbour of the returned arch strictly improves.
        let target = [1usize, 2, 0, 3];
        let space = ArchSpace::new(target.len(), 4);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(100));
        let mut rng = LcgRng::new(7);
        let obj = separable_objective(&target);
        let res = searcher
            .search(&space, &obj, &mut rng)
            .expect("test invariant: search");
        assert!(res.converged(100), "should stop at a local optimum");
        for nb in single_op_neighbors(&res.best) {
            assert!(
                obj(&nb) <= res.best_score + 1e-7,
                "neighbour improved over a claimed local optimum"
            );
        }
    }

    #[test]
    fn trajectory_is_non_decreasing() {
        let target = [4usize, 4, 0, 0, 2];
        let space = ArchSpace::new(target.len(), 5);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(40));
        let mut rng = LcgRng::new(123);
        let res = searcher
            .search(&space, separable_objective(&target), &mut rng)
            .expect("test invariant: search");
        for w in res.trajectory.windows(2) {
            assert!(w[1] >= w[0], "trajectory must be non-decreasing: {w:?}");
        }
        // best_score must equal the last accepted trajectory point.
        assert_eq!(
            *res.trajectory.last().expect("last should succeed"),
            res.best_score
        );
    }

    #[test]
    fn zero_budget_returns_start() {
        let start = arch(&[2, 0, 1], 4);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(0));
        let target = [0usize, 0, 0];
        let res = searcher
            .search_from(start.clone(), separable_objective(&target))
            .expect("test invariant: search_from");
        assert_eq!(res.best, start);
        assert_eq!(res.iters, 0);
        assert_eq!(res.evals, 1); // only the start is scored
        assert_eq!(res.trajectory.len(), 1);
    }

    #[test]
    fn empty_space_errors() {
        let space = ArchSpace::new(0, 4);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(10));
        let mut rng = LcgRng::new(1);
        assert_eq!(
            searcher.search(&space, |_| 0.0, &mut rng),
            Err(NasError::EmptySearchSpace)
        );
    }

    #[test]
    fn zero_ops_errors() {
        let space = ArchSpace::new(4, 0);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(10));
        let mut rng = LcgRng::new(1);
        assert_eq!(
            searcher.search(&space, |_| 0.0, &mut rng),
            Err(NasError::InvalidNumOps)
        );
    }

    #[test]
    fn search_from_empty_genome_errors() {
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(5));
        let empty = ArchEncoding {
            genes: Vec::new(),
            n_ops: 3,
        };
        assert_eq!(
            searcher.search_from(empty, |_| 1.0),
            Err(NasError::EmptySearchSpace)
        );
    }

    #[test]
    fn deterministic_given_seed() {
        let target = [2usize, 1, 3, 0, 1];
        let space = ArchSpace::new(target.len(), 4);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(30));
        let mut rng_a = LcgRng::new(555);
        let mut rng_b = LcgRng::new(555);
        let ra = searcher
            .search(&space, separable_objective(&target), &mut rng_a)
            .expect("a");
        let rb = searcher
            .search(&space, separable_objective(&target), &mut rng_b)
            .expect("b");
        assert_eq!(ra, rb);
    }

    #[test]
    fn single_op_space_returns_start_immediately() {
        // n_ops == 1: neighbourhood empty, so the start is the local optimum.
        let space = ArchSpace::new(5, 1);
        let searcher = LocalSearchNas::new(LocalSearchConfig::new(10));
        let mut rng = LcgRng::new(9);
        let res = searcher
            .search(&space, |_| 1.0, &mut rng)
            .expect("test invariant: search");
        assert_eq!(res.iters, 0);
        assert!(res.best.genes.iter().all(|&g| g == 0));
    }
}
