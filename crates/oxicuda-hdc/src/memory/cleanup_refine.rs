//! Cleanup memory with iterative refinement over an item memory.
//!
//! A *cleanup memory* maps a noisy hypervector back onto the clean prototype it most
//! resembles. The single-shot version is exactly [`ItemMemory::query`], which performs one
//! nearest-neighbour lookup. This module implements the *iterative* fixed-point variant used in
//! Vector Symbolic Architectures: the cleaned estimate is fed back into the item memory
//! repeatedly until the retrieved symbol stabilises, sharpening a weak / noisy read into a
//! confident one.
//!
//! Two refinement schemes are offered, selected by [`CleanupMode`]:
//!
//! * **Replace** — at every step the running estimate is *overwritten* by the stored
//!   hypervector of the retrieved symbol. This is a hard projection onto the codebook and is a
//!   genuine contraction: once the same symbol is returned twice in a row the estimate is, by
//!   construction, exactly that stored vector, so a Replace fixed point is reached in at most two
//!   iterations.
//!
//! * **Bundle** — the running estimate is *blended* toward the retrieved hypervector by a
//!   majority-vote bundle of `{estimate, stored}` (see [`bundle_binary`]). Each step nudges the
//!   estimate roughly halfway to the prototype rather than snapping to it, which is more robust
//!   when several prototypes are near-equidistant: the gradual pull lets the dominant attractor
//!   win over a few iterations instead of committing to a possibly-spurious first guess.
//!
//! In both modes the loop terminates early (`converged = true`) the first time two consecutive
//! iterations return the *same* symbol id; otherwise it runs the full `max_iter` budget and
//! reports `converged = false`. The returned similarity is the cosine between the final estimate
//! and the stored hypervector of the winning symbol, so a clean recovery yields a value close to
//! `+1`.
//!
//! Binary hypervectors are `±1` vectors stored as `Vec<i8>`; cosine for such vectors reduces to
//! `dot / D` (see [`cosine_binary`]).
//!
//! # References
//!
//! * P. Kanerva, "Hyperdimensional Computing: An Introduction to Computing in Distributed
//!   Representation with High-Dimensional Random Vectors," *Cognitive Computation* 1(2):139–159,
//!   2009 — cleanup / item memory as nearest-codebook recall.
//! * T. C. Stewart, T. Bekolay, and C. Eliasmith, "Neural representations of compositional
//!   structures: representing and manipulating vector spaces with spiking neurons," *Connection
//!   Science* 23(2):145–153, 2011 — iterative cleanup as a fixed-point operation that restores
//!   noisy symbolic reads.

use crate::distance::cosine::cosine_binary;
use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::memory::item_memory::ItemMemory;
use crate::ops::bundling::bundle_binary;

/// Refinement scheme used by [`cleanup`].
///
/// See the module-level documentation for the convergence behaviour of each mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupMode {
    /// Hard projection: replace the running estimate with the retrieved stored hypervector.
    Replace,
    /// Soft projection: majority-vote bundle the running estimate toward the retrieved
    /// hypervector so it converges gradually.
    Bundle,
}

/// Configuration for an iterative cleanup run.
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// Maximum number of query/refine iterations to perform (must be `>= 1`).
    pub max_iter: usize,
    /// Refinement scheme.
    pub mode: CleanupMode,
}

impl CleanupConfig {
    /// Create a configuration with the given iteration budget and refinement `mode`.
    ///
    /// `max_iter` is the upper bound on the number of query/refine steps. A value of `1` performs
    /// a single nearest-neighbour query (equivalent to [`ItemMemory::query`]) and then stops.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `max_iter == 0` (an empty iteration budget is rejected).
    pub fn new(max_iter: usize, mode: CleanupMode) -> HdcResult<Self> {
        if max_iter == 0 {
            return Err(HdcError::EmptyInput);
        }
        Ok(Self { max_iter, mode })
    }

    /// Convenience constructor for the hard [`CleanupMode::Replace`] scheme.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `max_iter == 0`.
    pub fn replace(max_iter: usize) -> HdcResult<Self> {
        Self::new(max_iter, CleanupMode::Replace)
    }

    /// Convenience constructor for the soft [`CleanupMode::Bundle`] scheme.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `max_iter == 0`.
    pub fn bundle(max_iter: usize) -> HdcResult<Self> {
        Self::new(max_iter, CleanupMode::Bundle)
    }
}

/// Outcome of an iterative cleanup run.
#[derive(Debug, Clone)]
pub struct CleanupResult {
    /// Symbol id of the winning (nearest) prototype.
    pub id: usize,
    /// The final cleaned hypervector estimate.
    pub hv: Vec<i8>,
    /// Number of query/refine iterations actually performed.
    pub n_iter: usize,
    /// `true` if the retrieved id stabilised before exhausting `max_iter`.
    pub converged: bool,
    /// Cosine similarity between [`CleanupResult::hv`] and the stored hypervector of [`CleanupResult::id`].
    pub final_similarity: f32,
}

/// Clean up a noisy binary hypervector by iterative item-memory refinement.
///
/// Starting from `query`, the function repeatedly looks the running estimate up in `mem`,
/// retrieves the stored hypervector of the nearest symbol, and refines the estimate toward it
/// according to `cfg.mode` (see [`CleanupMode`] and the module documentation). The loop runs at
/// most `cfg.max_iter` times and stops early as soon as the same symbol id is returned on two
/// consecutive iterations.
///
/// The dimension of `query` must equal the dimension of the hypervectors held in `mem`; this is
/// checked against the first retrieved stored vector.
///
/// # Errors
///
/// - [`HdcError::EmptyItemMemory`] if `mem` holds no items.
/// - [`HdcError::EmptyInput`] if `query` is empty.
/// - [`HdcError::DimensionMismatch`] if `query.len()` differs from the stored hypervector
///   dimension.
/// - Any error propagated from the underlying query, bundling, or cosine routines.
pub fn cleanup(
    mem: &ItemMemory,
    query: &[i8],
    cfg: &CleanupConfig,
    rng: &mut LcgRng,
) -> HdcResult<CleanupResult> {
    if mem.is_empty() {
        return Err(HdcError::EmptyItemMemory);
    }
    if query.is_empty() {
        return Err(HdcError::EmptyInput);
    }

    // Running estimate; refined in place across iterations.
    let mut estimate: Vec<i8> = query.to_vec();
    let mut prev_id: Option<usize> = None;
    let mut last_id: usize = 0;
    let mut n_iter: usize = 0;
    let mut converged = false;

    for _ in 0..cfg.max_iter {
        // Single nearest-neighbour lookup against the codebook.
        let id = mem.query(&estimate)?;
        let stored = mem.get(id)?;

        // Validate the query dimension against the stored prototype dimension on the first hit.
        // (`ItemMemory::query` already guards its own length, but we surface a clear error tied
        // to the caller-supplied `query` rather than relying on internal state.)
        if estimate.len() != stored.len() {
            return Err(HdcError::DimensionMismatch {
                expected: stored.len(),
                got: estimate.len(),
            });
        }

        n_iter += 1;
        last_id = id;

        // Stabilised: the same symbol two iterations running is a fixed point.
        if prev_id == Some(id) {
            converged = true;
            break;
        }

        // Refine the estimate toward the retrieved prototype.
        match cfg.mode {
            CleanupMode::Replace => {
                estimate.clear();
                estimate.extend_from_slice(stored);
            }
            CleanupMode::Bundle => {
                let blended = bundle_binary(&[estimate.clone(), stored.to_vec()], rng)?;
                estimate = blended;
            }
        }

        prev_id = Some(id);
    }

    // Similarity of the final estimate against the winning prototype.
    let winning = mem.get(last_id)?;
    let final_similarity = cosine_binary(&estimate, winning)?;

    Ok(CleanupResult {
        id: last_id,
        hv: estimate,
        n_iter,
        converged,
        final_similarity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::binary::random_binary;

    /// Build an item memory of `n` random prototypes of dimension `dim`, seeded for determinism.
    fn build_memory(n: usize, dim: usize, rng: &mut LcgRng) -> ItemMemory {
        let mut mem = ItemMemory::new(dim).expect("new item memory");
        for id in 0..n {
            mem.add_random(id, rng).expect("add_random");
        }
        mem
    }

    /// Flip `count` bits of `hv` at deterministic, distinct positions drawn from `rng`.
    fn flip_bits(hv: &[i8], count: usize, rng: &mut LcgRng) -> Vec<i8> {
        let mut noisy = hv.to_vec();
        let dim = noisy.len();
        let mut flipped = 0usize;
        // Bounded attempts so even with collisions we terminate deterministically.
        let mut attempts = 0usize;
        while flipped < count && attempts < count * 16 {
            let pos = rng.next_usize(dim);
            // Only count a flip the first time a position is toggled relative to the original.
            if noisy[pos] == hv[pos] {
                noisy[pos] = -noisy[pos];
                flipped += 1;
            }
            attempts += 1;
        }
        noisy
    }

    #[test]
    fn config_rejects_zero_iter() {
        let err = CleanupConfig::new(0, CleanupMode::Replace);
        assert!(matches!(err, Err(HdcError::EmptyInput)));
        // Non-zero budgets are accepted for both modes.
        assert!(CleanupConfig::new(1, CleanupMode::Replace).is_ok());
        assert!(CleanupConfig::bundle(5).is_ok());
        assert!(CleanupConfig::replace(3).is_ok());
    }

    #[test]
    fn exact_match_cleans_to_itself() {
        let mut rng = LcgRng::new(101);
        let mem = build_memory(8, 512, &mut rng);
        let target = mem.get(3).expect("get").to_vec();
        let cfg = CleanupConfig::replace(5).expect("cfg");
        let mut run_rng = LcgRng::new(202);
        let res = cleanup(&mem, &target, &cfg, &mut run_rng).expect("cleanup");
        assert_eq!(res.id, 3);
        assert_eq!(res.hv, target);
        assert!(res.converged);
        assert!(
            (res.final_similarity - 1.0_f32).abs() < 1e-6,
            "sim={}",
            res.final_similarity
        );
    }

    #[test]
    fn noisy_replace_recovers_id_and_improves_cosine() {
        let mut rng = LcgRng::new(303);
        let dim = 1024;
        let mem = build_memory(10, dim, &mut rng);
        let target = mem.get(6).expect("get").to_vec();

        // Corrupt ~20% of the bits.
        let mut noise_rng = LcgRng::new(404);
        let noisy = flip_bits(&target, dim / 5, &mut noise_rng);

        // Cosine of the noisy input vs the true prototype (the baseline to beat).
        let noisy_sim = cosine_binary(&noisy, &target).expect("cosine noisy");

        let cfg = CleanupConfig::replace(5).expect("cfg");
        let mut run_rng = LcgRng::new(505);
        let res = cleanup(&mem, &noisy, &cfg, &mut run_rng).expect("cleanup");

        assert_eq!(res.id, 6, "should recover the correct symbol");
        assert!(
            res.final_similarity > noisy_sim,
            "final {} should beat noisy {}",
            res.final_similarity,
            noisy_sim
        );
        // Replace lands exactly on the prototype.
        assert!((res.final_similarity - 1.0_f32).abs() < 1e-6);
        assert!(res.converged);
    }

    #[test]
    fn noisy_bundle_recovers_id_and_improves_cosine() {
        let mut rng = LcgRng::new(606);
        let dim = 1024;
        let mem = build_memory(10, dim, &mut rng);
        let target = mem.get(2).expect("get").to_vec();

        let mut noise_rng = LcgRng::new(707);
        let noisy = flip_bits(&target, dim / 5, &mut noise_rng);
        let noisy_sim = cosine_binary(&noisy, &target).expect("cosine noisy");

        let cfg = CleanupConfig::bundle(8).expect("cfg");
        let mut run_rng = LcgRng::new(808);
        let res = cleanup(&mem, &noisy, &cfg, &mut run_rng).expect("cleanup");

        assert_eq!(res.id, 2, "bundle should still recover the correct symbol");
        assert!(
            res.final_similarity > noisy_sim,
            "final {} should beat noisy {}",
            res.final_similarity,
            noisy_sim
        );
    }

    #[test]
    fn replace_converges_within_two_iters_on_exact() {
        let mut rng = LcgRng::new(909);
        let mem = build_memory(6, 256, &mut rng);
        let target = mem.get(1).expect("get").to_vec();
        let cfg = CleanupConfig::replace(10).expect("cfg");
        let mut run_rng = LcgRng::new(111);
        let res = cleanup(&mem, &target, &cfg, &mut run_rng).expect("cleanup");
        // First query already returns id 1; the second confirms it -> converged at iter 2.
        assert!(res.converged);
        assert!(res.n_iter <= 2, "n_iter={}", res.n_iter);
    }

    #[test]
    fn single_iter_does_not_flag_convergence() {
        // With max_iter == 1 there is no consecutive pair to compare, so convergence is false
        // even though the read is correct.
        let mut rng = LcgRng::new(222);
        let mem = build_memory(5, 256, &mut rng);
        let target = mem.get(4).expect("get").to_vec();
        let cfg = CleanupConfig::replace(1).expect("cfg");
        let mut run_rng = LcgRng::new(333);
        let res = cleanup(&mem, &target, &cfg, &mut run_rng).expect("cleanup");
        assert_eq!(res.id, 4);
        assert_eq!(res.n_iter, 1);
        assert!(!res.converged);
        // The single read landed on the exact prototype.
        assert!((res.final_similarity - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn empty_memory_errors() {
        let mem = ItemMemory::new(128).expect("new");
        let query = {
            let mut r = LcgRng::new(444);
            random_binary(128, &mut r).expect("rand")
        };
        let cfg = CleanupConfig::replace(3).expect("cfg");
        let mut run_rng = LcgRng::new(555);
        let err = cleanup(&mem, &query, &cfg, &mut run_rng);
        assert!(matches!(err, Err(HdcError::EmptyItemMemory)));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut rng = LcgRng::new(666);
        let mem = build_memory(4, 256, &mut rng);
        // Query of the wrong length.
        let mut q_rng = LcgRng::new(777);
        let bad_query = random_binary(128, &mut q_rng).expect("rand");
        let cfg = CleanupConfig::replace(3).expect("cfg");
        let mut run_rng = LcgRng::new(888);
        let err = cleanup(&mem, &bad_query, &cfg, &mut run_rng);
        assert!(matches!(err, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_query_errors() {
        let mut rng = LcgRng::new(999);
        let mem = build_memory(4, 256, &mut rng);
        let cfg = CleanupConfig::replace(3).expect("cfg");
        let mut run_rng = LcgRng::new(1010);
        let err = cleanup(&mem, &[], &cfg, &mut run_rng);
        assert!(matches!(err, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let dim = 512;
        // Two independent but identically-seeded runs must agree bit-for-bit.
        let build = |memseed: u64, noiseseed: u64, runseed: u64| -> CleanupResult {
            let mut rng = LcgRng::new(memseed);
            let mem = build_memory(12, dim, &mut rng);
            let target = mem.get(7).expect("get").to_vec();
            let mut noise_rng = LcgRng::new(noiseseed);
            let noisy = flip_bits(&target, dim / 5, &mut noise_rng);
            let cfg = CleanupConfig::bundle(6).expect("cfg");
            let mut run_rng = LcgRng::new(runseed);
            cleanup(&mem, &noisy, &cfg, &mut run_rng).expect("cleanup")
        };
        let a = build(2024, 4048, 8096);
        let b = build(2024, 4048, 8096);
        assert_eq!(a.id, b.id);
        assert_eq!(a.hv, b.hv);
        assert_eq!(a.n_iter, b.n_iter);
        assert_eq!(a.converged, b.converged);
        assert!((a.final_similarity - b.final_similarity).abs() < 1e-9);
    }
}
