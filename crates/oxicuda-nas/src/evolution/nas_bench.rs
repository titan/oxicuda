//! Multi-trial NAS-Bench-style reproducibility hooks.
//!
//! References:
//! - Ying, Klein, Christiansen, Real, Murphy & Hutter, "NAS-Bench-101: Towards
//!   Reproducible Neural Architecture Search", ICML 2019.
//!   <https://github.com/google-research/nasbench>
//! - Dong & Yang, "NAS-Bench-201: Extending the Scope of Reproducible Neural
//!   Architecture Search", ICLR 2020.
//!
//! Reproducible NAS demands two things this module supplies:
//!
//! 1. **Deterministic per-architecture seeds.** Tabular benchmarks store several
//!    independent training *trials* per architecture (NAS-Bench-101 keeps 3).
//!    To reproduce a run, the seed used for trial `t` of architecture `a` must
//!    be a pure function of `(a, t)` and a global run seed — never of wall-clock
//!    time or iteration order. [`derive_arch_seed`] is that function: a
//!    splittable-hash mixing of the encoded genome with the trial index and the
//!    base seed.
//!
//! 2. **A per-architecture result cache.** Every architecture must be evaluated
//!    *at most once* per trial: a search that re-proposes an architecture must
//!    get the identical cached result, and the running query count must reflect
//!    only genuinely-new evaluations. [`NasBenchCache`] keys
//!    [`TrialResult`]s by the canonical genome bytes and trial index, so the
//!    same architecture queried twice is a cache hit with zero extra
//!    evaluations.
//!
//! Together these make a search reproducible bit-for-bit given the run seed, and
//! let a benchmark report *query budget* (unique-arch evaluations) rather than a
//! noisy wall-clock proxy.

use std::collections::HashMap;

use crate::error::{NasError, NasResult};
use crate::evolution::encoding::ArchEncoding;
use crate::handle::LcgRng;

// ─── Deterministic seeding ──────────────────────────────────────────────────────

/// Canonical key bytes for an architecture: its genome packed little-endian.
///
/// Two [`ArchEncoding`]s map to the same key **iff** they have identical genes
/// (the op count is not part of the key — only the realised architecture is).
#[must_use]
pub fn arch_key(arch: &ArchEncoding) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(arch.genes.len() * 8);
    for &g in &arch.genes {
        bytes.extend_from_slice(&(g as u64).to_le_bytes());
    }
    bytes
}

/// Derive a deterministic 64-bit seed for trial `trial` of `arch` under run
/// seed `base_seed`.
///
/// The mixing uses the SplitMix64 finaliser, a high-quality avalanche hash, fed
/// a stream that folds in the base seed, the trial index, and every gene. This
/// guarantees: different architectures, trials, or run seeds (almost surely)
/// give different seeds; and the mapping is a pure, stable function so a run can
/// be replayed exactly.
#[must_use]
pub fn derive_arch_seed(arch: &ArchEncoding, trial: u32, base_seed: u64) -> u64 {
    // SplitMix64 finaliser.
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    // Golden-ratio increment keeps the stream well-distributed.
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut acc = base_seed
        .wrapping_add(mix(base_seed ^ (trial as u64).wrapping_mul(GAMMA)))
        .wrapping_add(mix((arch.genes.len() as u64).wrapping_mul(GAMMA)));
    for (i, &g) in arch.genes.iter().enumerate() {
        let lane = (g as u64)
            .wrapping_add((i as u64).wrapping_mul(GAMMA))
            .wrapping_add(0x1234_5678_9abc_def0);
        acc = mix(acc ^ lane).wrapping_add(GAMMA);
    }
    mix(acc)
}

/// A deterministic [`LcgRng`] for trial `trial` of `arch`, seeded via
/// [`derive_arch_seed`]. Two calls with identical arguments produce identical
/// random streams.
#[must_use]
pub fn arch_rng(arch: &ArchEncoding, trial: u32, base_seed: u64) -> LcgRng {
    LcgRng::new(derive_arch_seed(arch, trial, base_seed))
}

// ─── TrialResult ───────────────────────────────────────────────────────────────

/// The recorded outcome of one training trial of one architecture, mirroring a
/// NAS-Bench table row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrialResult {
    /// Held-out validation accuracy in `[0, 1]`.
    pub val_accuracy: f32,
    /// Held-out test accuracy in `[0, 1]`.
    pub test_accuracy: f32,
    /// Wall-clock training time for the trial (seconds); a *recorded* value, not
    /// a measured-on-this-host one.
    pub train_time: f32,
    /// Trial index this row belongs to.
    pub trial: u32,
}

impl TrialResult {
    /// Construct a validated trial result.
    ///
    /// # Errors
    /// - [`NasError::NanInArchParams`] if either accuracy is non-finite or
    ///   outside `[0, 1]`, or `train_time` is non-finite or negative.
    pub fn new(
        val_accuracy: f32,
        test_accuracy: f32,
        train_time: f32,
        trial: u32,
    ) -> NasResult<Self> {
        let acc_ok = |a: f32| a.is_finite() && (0.0..=1.0).contains(&a);
        if !acc_ok(val_accuracy) || !acc_ok(test_accuracy) {
            return Err(NasError::NanInArchParams);
        }
        if !train_time.is_finite() || train_time < 0.0 {
            return Err(NasError::NanInArchParams);
        }
        Ok(Self {
            val_accuracy,
            test_accuracy,
            train_time,
            trial,
        })
    }
}

// ─── NasBenchCache ───────────────────────────────────────────────────────────

/// Per-architecture, per-trial result cache with query accounting.
///
/// The cache is the reproducibility backbone of a tabular NAS run: it ensures an
/// architecture is *evaluated* at most once per trial and tracks how many
/// genuinely-new evaluations a search spent (`unique_queries`) versus how often
/// it re-proposed a known architecture (`cache_hits`).
#[derive(Debug, Clone, Default)]
pub struct NasBenchCache {
    table: HashMap<(Vec<u8>, u32), TrialResult>,
    unique_queries: u64,
    cache_hits: u64,
}

impl NasBenchCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or overwrite) a recorded trial result for `arch`.
    ///
    /// Overwriting is allowed (re-recording the same row) and does **not** alter
    /// the query counters — those track *lookups*, not table population.
    pub fn record(&mut self, arch: &ArchEncoding, result: TrialResult) {
        self.table.insert((arch_key(arch), result.trial), result);
    }

    /// Look up a previously-recorded result without touching the query counters.
    #[must_use]
    pub fn peek(&self, arch: &ArchEncoding, trial: u32) -> Option<TrialResult> {
        self.table.get(&(arch_key(arch), trial)).copied()
    }

    /// Query the benchmark for trial `trial` of `arch`, evaluating lazily.
    ///
    /// If a result is cached, it is returned and `cache_hits` is incremented (no
    /// evaluation happens). Otherwise `evaluate(arch, derived_seed)` is called
    /// exactly once, the result recorded, `unique_queries` incremented, and the
    /// fresh result returned. `base_seed` feeds [`derive_arch_seed`] so the
    /// evaluation is reproducible.
    ///
    /// This is the central reproducibility hook: the *same* search trace, run
    /// twice with the same `base_seed`, performs the identical set of unique
    /// evaluations and returns identical results.
    ///
    /// # Errors
    /// Propagates any error from `evaluate`.
    pub fn query<F>(
        &mut self,
        arch: &ArchEncoding,
        trial: u32,
        base_seed: u64,
        mut evaluate: F,
    ) -> NasResult<TrialResult>
    where
        F: FnMut(&ArchEncoding, u64) -> NasResult<TrialResult>,
    {
        let key = (arch_key(arch), trial);
        if let Some(r) = self.table.get(&key) {
            self.cache_hits += 1;
            return Ok(*r);
        }
        let seed = derive_arch_seed(arch, trial, base_seed);
        let result = evaluate(arch, seed)?;
        self.table.insert(key, result);
        self.unique_queries += 1;
        Ok(result)
    }

    /// Mean validation accuracy across all recorded trials of `arch`.
    ///
    /// Returns `None` if no trial of `arch` is cached. This mirrors the standard
    /// NAS-Bench "average over the stored trials" query.
    #[must_use]
    pub fn mean_val_accuracy(&self, arch: &ArchEncoding) -> Option<f32> {
        let key = arch_key(arch);
        let mut sum = 0.0_f32;
        let mut n = 0u32;
        for ((k, _), r) in &self.table {
            if *k == key {
                sum += r.val_accuracy;
                n += 1;
            }
        }
        if n == 0 { None } else { Some(sum / n as f32) }
    }

    /// Number of unique `(architecture, trial)` evaluations performed via
    /// [`Self::query`] — the reproducible query budget of a search.
    #[must_use]
    pub fn unique_queries(&self) -> u64 {
        self.unique_queries
    }

    /// Number of [`Self::query`] calls that hit the cache (re-proposed arches).
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// Number of distinct `(architecture, trial)` rows stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// `true` if no rows are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(genes: &[usize]) -> ArchEncoding {
        ArchEncoding {
            genes: genes.to_vec(),
            n_ops: 8,
        }
    }

    #[test]
    fn seed_is_deterministic_and_pure() {
        let a = arch(&[0, 3, 5, 1]);
        let s1 = derive_arch_seed(&a, 0, 12345);
        let s2 = derive_arch_seed(&a, 0, 12345);
        assert_eq!(s1, s2, "seed must be a pure function of its inputs");
    }

    #[test]
    fn seed_varies_with_trial_arch_and_base() {
        let a = arch(&[0, 3, 5, 1]);
        let b = arch(&[0, 3, 5, 2]);
        let s_a0 = derive_arch_seed(&a, 0, 42);
        let s_a1 = derive_arch_seed(&a, 1, 42);
        let s_b0 = derive_arch_seed(&b, 0, 42);
        let s_a0_other = derive_arch_seed(&a, 0, 43);
        assert_ne!(s_a0, s_a1, "different trials must differ");
        assert_ne!(s_a0, s_b0, "different architectures must differ");
        assert_ne!(s_a0, s_a0_other, "different base seeds must differ");
    }

    #[test]
    fn seed_distinguishes_gene_order() {
        // Permuted genome must (almost surely) give a different seed.
        let a = arch(&[1, 2, 3]);
        let b = arch(&[3, 2, 1]);
        assert_ne!(derive_arch_seed(&a, 0, 7), derive_arch_seed(&b, 0, 7));
    }

    #[test]
    fn arch_rng_streams_match_for_same_inputs() {
        let a = arch(&[2, 2, 0, 7]);
        let mut r1 = arch_rng(&a, 1, 999);
        let mut r2 = arch_rng(&a, 1, 999);
        for _ in 0..64 {
            assert_eq!(r1.next_u32(), r2.next_u32());
        }
    }

    #[test]
    fn cache_hit_returns_identical_and_counts() {
        let mut cache = NasBenchCache::new();
        let a = arch(&[0, 1, 2]);
        let mut calls = 0u32;
        let r1 = cache
            .query(&a, 0, 100, |_, _| {
                calls += 1;
                TrialResult::new(0.9, 0.88, 1.0, 0)
            })
            .expect("first query");
        // Re-propose the same architecture: must be a cache hit, no new eval.
        let r2 = cache
            .query(&a, 0, 100, |_, _| {
                calls += 1;
                TrialResult::new(0.0, 0.0, 0.0, 0)
            })
            .expect("second query");
        assert_eq!(r1, r2, "cache must return the identical result");
        assert_eq!(calls, 1, "evaluate must run exactly once");
        assert_eq!(cache.unique_queries(), 1);
        assert_eq!(cache.cache_hits(), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn query_uses_derived_seed() {
        let mut cache = NasBenchCache::new();
        let a = arch(&[3, 1, 4, 1, 5]);
        let trial = 2u32;
        let base = 2024u64;
        let expected_seed = derive_arch_seed(&a, trial, base);
        let mut seen_seed = 0u64;
        cache
            .query(&a, trial, base, |_, seed| {
                seen_seed = seed;
                TrialResult::new(0.5, 0.5, 1.0, trial)
            })
            .expect("query");
        assert_eq!(seen_seed, expected_seed);
    }

    #[test]
    fn distinct_trials_are_separate_rows() {
        let mut cache = NasBenchCache::new();
        let a = arch(&[0, 0, 0]);
        for t in 0..3u32 {
            cache
                .query(&a, t, 1, |_, _| {
                    TrialResult::new(0.8 + 0.01 * t as f32, 0.8, 1.0, t)
                })
                .expect("query");
        }
        assert_eq!(cache.unique_queries(), 3);
        assert_eq!(cache.cache_hits(), 0);
        assert_eq!(cache.len(), 3);
        let mean = cache.mean_val_accuracy(&a).expect("mean present");
        // (0.80 + 0.81 + 0.82) / 3 = 0.81
        assert!((mean - 0.81).abs() < 1e-5, "mean = {mean}");
    }

    #[test]
    fn full_run_is_reproducible() {
        // Two independent runs over the same proposal trace with the same base
        // seed must produce identical accuracies and identical query budgets.
        fn run() -> (Vec<f32>, u64) {
            let mut cache = NasBenchCache::new();
            // A synthetic "oracle": accuracy is a deterministic function of the
            // first random draw from the arch-derived RNG. The arch identity is
            // already folded into `seed` by `derive_arch_seed`, so the encoding
            // argument itself is intentionally unused here.
            let evaluate = |_: &ArchEncoding, seed: u64| {
                let mut rng = LcgRng::new(seed);
                let acc = 0.5 + 0.4 * rng.next_f32();
                TrialResult::new(acc.clamp(0.0, 1.0), acc.clamp(0.0, 1.0), 1.0, 0)
            };
            // Proposal trace re-proposes arch #1 twice on purpose.
            let proposals = [
                arch(&[0, 1, 2]),
                arch(&[3, 3, 3]),
                arch(&[0, 1, 2]), // duplicate
                arch(&[7, 0, 7]),
            ];
            let mut accs = Vec::new();
            for p in &proposals {
                let r = cache.query(p, 0, 555, evaluate).expect("query");
                accs.push(r.val_accuracy);
            }
            (accs, cache.unique_queries())
        }
        let (a1, q1) = run();
        let (a2, q2) = run();
        assert_eq!(a1, a2, "runs must be bit-reproducible");
        assert_eq!(q1, q2);
        assert_eq!(q1, 3, "the duplicate must not cost an extra evaluation");
        // The duplicate proposal got the identical cached accuracy.
        assert_eq!(a1[0], a1[2]);
    }

    #[test]
    fn peek_does_not_disturb_counters() {
        let mut cache = NasBenchCache::new();
        let a = arch(&[1, 2, 3]);
        cache.record(&a, TrialResult::new(0.9, 0.9, 1.0, 0).expect("result"));
        assert!(cache.peek(&a, 0).is_some());
        assert!(cache.peek(&a, 1).is_none());
        assert_eq!(cache.cache_hits(), 0);
        assert_eq!(cache.unique_queries(), 0);
    }

    #[test]
    fn trial_result_rejects_bad_values() {
        assert_eq!(
            TrialResult::new(1.5, 0.5, 1.0, 0),
            Err(NasError::NanInArchParams)
        );
        assert_eq!(
            TrialResult::new(0.5, 0.5, -1.0, 0),
            Err(NasError::NanInArchParams)
        );
        assert_eq!(
            TrialResult::new(f32::NAN, 0.5, 1.0, 0),
            Err(NasError::NanInArchParams)
        );
    }

    #[test]
    fn mean_absent_for_unknown_arch() {
        let cache = NasBenchCache::new();
        assert!(cache.mean_val_accuracy(&arch(&[9, 9])).is_none());
        assert!(cache.is_empty());
    }
}
