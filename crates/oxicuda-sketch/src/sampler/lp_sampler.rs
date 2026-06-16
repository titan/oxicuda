//! L0 / Lp samplers for turnstile streams (Jowhari, Sağlam & Tardos, STOC 2011;
//! Cormode & Firmani, *Distributed & Parallel Databases* 2014).
//!
//! An **L0-sampler** maintains a linear sketch of a vector `x ∈ ℝⁿ` defined by a
//! turnstile stream of updates `(i, Δ)` (`x_i += Δ`, deletions allowed) and, on
//! demand, returns a **near-uniformly random non-zero coordinate** `(i, x_i)` of
//! `x` — uniform over the *support* `{i : x_i ≠ 0}` regardless of the magnitudes
//! of the entries (that is the "L0" / `p → 0` regime).
//!
//! # Construction
//!
//! The sketch is the JST/Cormode–Firmani three-part recipe:
//!
//! 1. **Geometric subsampling levels.** A 2-universal hash `h : [n] → [n]`
//!    assigns each coordinate `i` to all levels `0 ≤ ℓ ≤ ⌊log₂ h_max/h(i)⌋`; more
//!    precisely coordinate `i` *survives* to level `ℓ` iff `h(i) < n / 2^ℓ`.
//!    Level `0` therefore contains every coordinate and each successive level
//!    keeps roughly half as many, so for a support of size `k` there is — with
//!    constant probability — some level holding **exactly one** surviving
//!    non-zero (the level near `ℓ ≈ log₂ k`).
//!
//! 2. **Per-level 1-sparse recovery.** Each level keeps three running sums over
//!    its surviving coordinates:
//!    ```text
//!    w = Σ x_i           (plain weight / count)
//!    p = Σ (i+1) · x_i   (index-weighted sum; uses i+1 to keep index 0 usable)
//!    z = Σ x_i · r^{i+1} (mod q)   (Rabin–Karp fingerprint over a prime field)
//!    ```
//!    If a level is **exactly 1-sparse** — a single non-zero `i*` with value
//!    `v` survives — then `w = v`, `p = (i*+1)·v`, hence
//!    `i* = p/w − 1` and `v = w`, and the fingerprint must satisfy
//!    `z ≡ v · r^{i*+1} (mod q)`.  The fingerprint is what lets us *verify*
//!    1-sparsity: if two or more non-zeros collide on a level, the recovered
//!    `(i*, v)` candidate will (with overwhelming probability over the random
//!    `r`) fail the `z` check and the level is rejected as a *collision*.
//!
//! 3. **Recovery scan.** To draw a sample, scan the levels from the sparsest
//!    (highest `ℓ`) toward the densest and return the first level that passes the
//!    1-sparse verification.  Because the surviving coordinate of that level is a
//!    uniformly random member of the support (the hash `h` is oblivious to the
//!    values), the returned index is near-uniform over the support.
//!
//! The whole sketch is **linear** in `x`: two sketches built with the same
//! parameters can be merged by summing their per-level `(w, p, z)` triples, so
//! the structure composes over distributed stream shards.
//!
//! All arithmetic for the fingerprint is carried out modulo the Mersenne prime
//! `2⁶¹ − 1` reused from [`crate::hash::twouniv::PRIME_MERSENNE_61`]; the
//! coordinate values are tracked exactly as signed 64-bit integers so the
//! turnstile (add-then-delete) algebra is exact and a coordinate that returns to
//! zero genuinely leaves the support.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::{PRIME_MERSENNE_61, TwoUniversal};

/// A single geometric subsampling level holding a 1-sparse recovery sketch.
///
/// Tracks the three running sums needed to detect and decode a level that holds
/// exactly one surviving non-zero coordinate.  All sums are exact: `w` and `p`
/// over signed 128-bit accumulators (to tolerate large turnstile cancellation),
/// the fingerprint `z` over the prime field.
#[derive(Debug, Clone)]
struct LevelSketch {
    /// `w = Σ x_i` over the coordinates surviving to this level.
    weight: i128,
    /// `p = Σ (i+1) · x_i`.
    index_weight: i128,
    /// `z = Σ x_i · r^{i+1}  (mod q)` — Rabin–Karp fingerprint.
    fingerprint: u64,
    /// Number of *updates* applied to this level (diagnostics only).
    updates: u64,
}

impl LevelSketch {
    fn new() -> Self {
        Self {
            weight: 0,
            index_weight: 0,
            fingerprint: 0,
            updates: 0,
        }
    }

    /// Apply `x[index] += delta` to this level's sketch.
    fn apply(&mut self, index: usize, delta: i64, pow_table: &PowerTable) {
        self.weight += delta as i128;
        // (index + 1) keeps index 0 distinguishable from "no contribution".
        self.index_weight += (index as i128 + 1) * delta as i128;
        // z += delta * r^{index+1}  (mod q).  Signed delta handled via the field.
        let term = mod_mul(pow_table.pow(index + 1), field_signed(delta));
        self.fingerprint = mod_add(self.fingerprint, term);
        self.updates += 1;
    }

    /// Attempt to decode this level as exactly 1-sparse.
    ///
    /// Returns `Some((index, value))` iff the level holds a single non-zero whose
    /// recovered index/value is consistent with all three sums (including the
    /// fingerprint).  Returns `None` for an empty level (`w = 0` and `p = 0`) or
    /// a collision (≥ 2 surviving non-zeros that fail verification).
    fn recover_one_sparse(&self, n: usize, pow_table: &PowerTable) -> Option<(usize, i64)> {
        // Empty level: nothing survived (or everything cancelled to zero).
        if self.weight == 0 {
            // A genuinely empty level also has p = 0 and z = 0.  If p ≠ 0 while
            // w = 0 the level cannot be 1-sparse (a lone non-zero has w = v ≠ 0),
            // so it is a collision; either way there is nothing to recover.
            return None;
        }
        // Candidate value v = w and candidate index i* = p/w − 1.
        let value = self.weight;
        // p must be exactly divisible by w for a clean single coordinate.
        if self.index_weight % value != 0 {
            return None;
        }
        let idx_plus_one = self.index_weight / value;
        if idx_plus_one < 1 {
            return None;
        }
        let index_i128 = idx_plus_one - 1;
        if index_i128 < 0 || index_i128 >= n as i128 {
            return None;
        }
        let index = index_i128 as usize;
        // Value must fit back into i64 (it always does for a real single coord).
        if value > i64::MAX as i128 || value < i64::MIN as i128 {
            return None;
        }
        let value_i64 = value as i64;
        // Fingerprint verification: z ?= v · r^{index+1} (mod q).
        let expected = mod_mul(pow_table.pow(index + 1), field_signed(value_i64));
        if expected != self.fingerprint {
            return None;
        }
        Some((index, value_i64))
    }

    /// Linearly merge `other` into `self` (sum the three sums in the field).
    fn merge(&mut self, other: &LevelSketch) {
        self.weight += other.weight;
        self.index_weight += other.index_weight;
        self.fingerprint = mod_add(self.fingerprint, other.fingerprint);
        self.updates += other.updates;
    }
}

/// Precomputed table of `r^j mod q` for `j ∈ [0, n]`, where `r` is the random
/// fingerprint base and `q = 2⁶¹ − 1`.
#[derive(Debug, Clone)]
struct PowerTable {
    powers: Vec<u64>,
}

impl PowerTable {
    /// Build `r^0 … r^count` modulo the field.
    fn new(r: u64, count: usize) -> Self {
        let mut powers = Vec::with_capacity(count + 1);
        let mut cur = 1u64 % PRIME_MERSENNE_61;
        powers.push(cur);
        for _ in 0..count {
            cur = mod_mul(cur, r);
            powers.push(cur);
        }
        Self { powers }
    }

    #[inline]
    fn pow(&self, j: usize) -> u64 {
        self.powers[j]
    }
}

/// An L0 / Lp sampler over a turnstile-streamed vector of length `n`.
///
/// Built with [`LpSampler::new`]; updated with [`LpSampler::update`]; queried
/// with [`LpSampler::sample`].  See the module docs for the algorithm.
#[derive(Debug, Clone)]
pub struct LpSampler {
    /// Logical vector length (coordinate universe size).
    n: usize,
    /// Number of geometric subsampling levels (`⌈log₂ n⌉ + 1`).
    n_levels: usize,
    /// 2-universal hash mapping a coordinate to a value in `[0, n)` used to
    /// decide the deepest level each coordinate survives to.
    level_hash: TwoUniversal,
    /// Fingerprint base `r ∈ [2, q − 1]`.
    base: u64,
    /// Per-level 1-sparse sketches, index `ℓ ∈ [0, n_levels)`.
    levels: Vec<LevelSketch>,
    /// Shared `r^j` table.
    pow_table: PowerTable,
    /// Seed retained so independent-but-mergeable instances can be recognised.
    seed: u64,
}

impl LpSampler {
    /// Construct a fresh sampler for a length-`n` vector, seeded by `seed`.
    ///
    /// # Errors
    /// Returns [`SketchError::InvalidParameter`] if `n == 0`.
    pub fn new(n: usize, seed: u64) -> SketchResult<Self> {
        if n == 0 {
            return Err(SketchError::InvalidParameter {
                name: "n".to_string(),
                reason: "vector length must be positive".to_string(),
            });
        }
        let mut rng = LcgRng::new(seed);
        // Level hash maps coordinates uniformly into [0, n); a coordinate
        // survives to level ℓ iff h(i) < n / 2^ℓ.
        let level_hash = TwoUniversal::new(&mut rng, n as u64);
        // Number of levels: enough that the sparsest level keeps ≤ 1 expected
        // coordinate even for a full vector.  ⌈log₂ n⌉ + 1 suffices.
        let n_levels = (usize::BITS - (n.saturating_sub(1)).leading_zeros()) as usize + 1;
        // Fingerprint base in [2, q-1].
        let base = 2 + rng.next_u64() % (PRIME_MERSENNE_61 - 3);
        let pow_table = PowerTable::new(base, n);
        let levels = vec![LevelSketch::new(); n_levels];
        Ok(Self {
            n,
            n_levels,
            level_hash,
            base,
            levels,
            pow_table,
            seed,
        })
    }

    /// Logical vector length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the vector universe is empty (always `false`, `n ≥ 1`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of geometric subsampling levels.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// The deepest level coordinate `i` survives to: the largest `ℓ` with
    /// `h(i) < n / 2^ℓ`, i.e. `ℓ = ⌊log₂( n / (h(i)+1) )⌋` clamped to the level
    /// range.  Coordinate `i` is then a member of levels `0 … survive_level(i)`.
    fn survive_level(&self, i: usize) -> usize {
        let h = self.level_hash.hash(i as u64);
        // Largest ℓ with h < n / 2^ℓ  ⇔  2^ℓ < n / h  (for h ≥ 1).
        // Use (h+1) to keep h = 0 finite and bound by the level count.
        let ratio = self.n as u64 / (h + 1);
        // floor(log2(ratio)) gives the deepest surviving level.
        let depth = if ratio == 0 {
            0
        } else {
            (u64::BITS - 1 - ratio.leading_zeros()) as usize
        };
        depth.min(self.n_levels - 1)
    }

    /// Apply a turnstile update `x[index] += delta`.
    ///
    /// `delta` may be negative (deletion/decrement).  A zero `delta` is a no-op.
    ///
    /// # Errors
    /// Returns [`SketchError::IndexOutOfBounds`] if `index ≥ n`.
    pub fn update(&mut self, index: usize, delta: i64) -> SketchResult<()> {
        if index >= self.n {
            return Err(SketchError::IndexOutOfBounds { index, len: self.n });
        }
        if delta == 0 {
            return Ok(());
        }
        let deepest = self.survive_level(index);
        // Coordinate i contributes to every level 0..=deepest.
        for level in 0..=deepest {
            self.levels[level].apply(index, delta, &self.pow_table);
        }
        Ok(())
    }

    /// Draw a near-uniform random non-zero coordinate `(index, value)` of the
    /// current vector, or `None` if no 1-sparse level can be found (which happens
    /// only when the support is empty or, with small probability, when every
    /// level collided).
    ///
    /// Scans levels from the sparsest (highest `ℓ`) downward and returns the
    /// first that decodes and verifies as exactly 1-sparse.
    #[must_use]
    pub fn sample(&self) -> Option<(usize, i64)> {
        for level in (0..self.n_levels).rev() {
            if let Some(found) = self.levels[level].recover_one_sparse(self.n, &self.pow_table) {
                return Some(found);
            }
        }
        None
    }

    /// Decode a *specific* level as 1-sparse, exposing the per-level recovery for
    /// testing and diagnostics.
    ///
    /// Returns `Some((index, value))` iff level `level` currently holds exactly
    /// one verified non-zero, `None` for an empty or colliding level.
    ///
    /// # Errors
    /// Returns [`SketchError::IndexOutOfBounds`] if `level ≥ n_levels`.
    pub fn recover_level(&self, level: usize) -> SketchResult<Option<(usize, i64)>> {
        if level >= self.n_levels {
            return Err(SketchError::IndexOutOfBounds {
                index: level,
                len: self.n_levels,
            });
        }
        Ok(self.levels[level].recover_one_sparse(self.n, &self.pow_table))
    }

    /// Whether level `0` (which contains the whole vector) currently has any
    /// non-zero weight; a cheap "is the support non-empty?" probe.
    ///
    /// Note this is necessary but not sufficient (the full weight could cancel to
    /// zero while the support is non-empty); use [`LpSampler::sample`] for a true
    /// emptiness decision via the fingerprinted levels.
    #[must_use]
    pub fn level_zero_weight(&self) -> i128 {
        self.levels[0].weight
    }

    /// Linearly merge another sampler built with the **same** `(n, seed)` into
    /// `self` by summing the per-level sketches.
    ///
    /// # Errors
    /// Returns [`SketchError::ShapeMismatch`] on an `n` mismatch and
    /// [`SketchError::InvalidParameter`] on a seed mismatch (different hashes /
    /// fingerprint bases make a level-wise sum meaningless).
    pub fn merge(&mut self, other: &Self) -> SketchResult<()> {
        if self.n != other.n {
            return Err(SketchError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![other.n],
            });
        }
        if self.seed != other.seed || self.base != other.base {
            return Err(SketchError::InvalidParameter {
                name: "seed".to_string(),
                reason: "merge requires identical (n, seed) so hashes/bases match".to_string(),
            });
        }
        for (a, b) in self.levels.iter_mut().zip(other.levels.iter()) {
            a.merge(b);
        }
        Ok(())
    }
}

// ─── Prime-field (2⁶¹ − 1) arithmetic helpers ──────────────────────────────────

/// Reduce a `u128` modulo the Mersenne prime `2⁶¹ − 1`.
#[inline]
fn mod_reduce(x: u128) -> u64 {
    let p = PRIME_MERSENNE_61 as u128;
    let r = (x & p) + (x >> 61);
    let r = if r >= p { r - p } else { r };
    r as u64
}

/// `(a + b) mod q`.
#[inline]
fn mod_add(a: u64, b: u64) -> u64 {
    let s = a as u128 + b as u128;
    mod_reduce(s)
}

/// `(a * b) mod q`.
#[inline]
fn mod_mul(a: u64, b: u64) -> u64 {
    mod_reduce((a as u128).wrapping_mul(b as u128))
}

/// Map a signed value into the field `[0, q)`: negative deltas become `q − |Δ|`.
#[inline]
fn field_signed(delta: i64) -> u64 {
    let q = PRIME_MERSENNE_61;
    if delta >= 0 {
        (delta as u64) % q
    } else {
        // -|delta| mod q
        let m = (delta.unsigned_abs()) % q;
        if m == 0 { 0 } else { q - m }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Reference dense vector model to drive the sampler in tests.
    struct DenseVec {
        data: Vec<i64>,
    }

    impl DenseVec {
        fn new(n: usize) -> Self {
            Self { data: vec![0; n] }
        }
        fn add(&mut self, i: usize, d: i64) {
            self.data[i] += d;
        }
    }

    // (a) On a vector with k non-zeros, sample() returns a coordinate that IS in
    //     the support (always non-zero, with the correct value).
    #[test]
    fn sample_returns_support_member() {
        let n = 1024;
        // Insert k = 12 non-zeros at scattered indices with various values.
        let entries = [
            (3usize, 5i64),
            (17, -2),
            (100, 9),
            (255, 1),
            (256, 7),
            (511, -4),
            (600, 3),
            (700, 8),
            (888, -6),
            (900, 2),
            (1000, 10),
            (1023, -1),
        ];
        // Try several seeds; for each, the sample must be a genuine support pair.
        let mut any_found = false;
        for seed in 0..40u64 {
            let mut sampler = LpSampler::new(n, seed).expect("ok");
            let mut model = DenseVec::new(n);
            for &(i, v) in &entries {
                sampler.update(i, v).expect("update");
                model.add(i, v);
            }
            if let Some((idx, val)) = sampler.sample() {
                any_found = true;
                // The returned coordinate must be one of the true non-zeros and
                // carry its exact value.
                let truth = model.data[idx];
                assert_ne!(truth, 0, "sampled a zero coordinate {idx}");
                assert_eq!(val, truth, "sampled value wrong at {idx}");
                assert!(
                    entries.iter().any(|&(i, v)| i == idx && v == val),
                    "({idx},{val}) is not an inserted entry"
                );
            }
        }
        assert!(any_found, "no seed produced a sample over the support");
    }

    // (b) 1-sparse recovery EXACTLY recovers a single non-zero's index AND value.
    #[test]
    fn one_sparse_exact_recovery() {
        let n = 2048;
        // A single coordinate ⇒ it survives to level 0 for sure (level 0 holds
        // everything), so recovering level 0 must yield it exactly.
        for &(idx, val) in &[(0usize, 7i64), (1, -3), (1000, 42), (2047, -99), (123, 1)] {
            let mut sampler = LpSampler::new(n, 12345).expect("ok");
            sampler.update(idx, val).expect("update");
            let rec = sampler
                .recover_level(0)
                .expect("level in range")
                .expect("level 0 is 1-sparse for a single non-zero");
            assert_eq!(rec, (idx, val), "exact recovery failed");
            // And sample() must agree.
            assert_eq!(sampler.sample(), Some((idx, val)));
        }
    }

    // (c) The fingerprint correctly DETECTS a not-1-sparse level: ≥ 2 non-zeros
    //     that land on the same level register as a collision (not falsely
    //     recovered as a phantom single coordinate).
    #[test]
    fn fingerprint_detects_collision() {
        // Force a guaranteed collision at level 0: level 0 always holds ALL
        // coordinates, so any two non-zeros collide there.  Recovery of level 0
        // must therefore be rejected whenever ≥ 2 non-zeros exist — UNLESS the
        // index-weight happens to be divisible and the fingerprint matches, which
        // the random base makes vanishingly unlikely.  We assert no FALSE
        // recovery: if level 0 returns Some, it must be wrong, which must not
        // happen.
        let n = 4096;
        let mut false_recover = 0;
        for seed in 0..200u64 {
            let mut sampler = LpSampler::new(n, seed).expect("ok");
            // Two non-zeros: their average index*value could fool plain w,p but
            // not the fingerprint.
            sampler.update(10, 5).expect("u");
            sampler.update(20, 5).expect("u");
            // Level 0 holds both ⇒ it is 2-sparse ⇒ must NOT decode to a single.
            if sampler.recover_level(0).expect("range").is_some() {
                // A 2-sparse vector (both coordinates present at level 0) has NO
                // valid single-coordinate decode, so ANY `Some` here is a false
                // positive that the fingerprint should have rejected.
                false_recover += 1;
            }
        }
        assert_eq!(
            false_recover, 0,
            "fingerprint failed to reject a 2-sparse level {false_recover} times"
        );
    }

    // The midpoint trap: two symmetric coordinates whose (i+1)*v sum is divisible
    // by w, so w/p alone would decode a phantom centre index. The fingerprint
    // must still reject it.
    #[test]
    fn fingerprint_rejects_phantom_midpoint() {
        let n = 1024;
        let mut rejected = 0;
        let trials = 200;
        for seed in 0..trials {
            let mut sampler = LpSampler::new(n, seed).expect("ok");
            // x[100]=1, x[300]=1 ⇒ w=2, p=(101)+(301)=402 ⇒ p/w=201 ⇒ phantom
            // index 200 with value 2.  Without the fingerprint this would decode!
            sampler.update(100, 1).expect("u");
            sampler.update(300, 1).expect("u");
            match sampler.recover_level(0).expect("range") {
                None => rejected += 1,
                Some((idx, val)) => {
                    // If it ever decodes, it must NOT be the phantom (200, 2).
                    assert!(
                        !(idx == 200 && val == 2),
                        "phantom midpoint (200,2) falsely recovered (seed {seed})"
                    );
                }
            }
        }
        // The fingerprint should reject the overwhelming majority.
        assert!(
            rejected > trials * 9 / 10,
            "phantom rejected only {rejected}/{trials} times"
        );
    }

    // (d) TURNSTILE: adding then deleting a coordinate removes it from the
    //     support; a then-empty vector yields None.
    #[test]
    fn turnstile_delete_empties() {
        let n = 512;
        let mut sampler = LpSampler::new(n, 999).expect("ok");
        sampler.update(42, 8).expect("u");
        // Right now the support is {42}; sample must find it.
        assert_eq!(sampler.sample(), Some((42, 8)));
        // Delete it.
        sampler.update(42, -8).expect("u");
        // Vector is now all-zero ⇒ no sample.
        assert_eq!(sampler.sample(), None, "deleted coordinate still sampled");
        assert_eq!(sampler.level_zero_weight(), 0);

        // Partial deletion keeps the coordinate in the support with the residual.
        sampler.update(7, 10).expect("u");
        sampler.update(7, -3).expect("u");
        assert_eq!(sampler.sample(), Some((7, 7)), "residual value wrong");
    }

    // (e) Over many independent sampler instances the returned coordinate is
    //     roughly UNIFORM over the support: each non-zero gets hit, and it is not
    //     always the same index.
    #[test]
    fn approximately_uniform_over_support() {
        let n = 256;
        // A modest support; we check coverage and non-degeneracy across seeds.
        let support_indices = [5usize, 50, 120, 200, 250];
        let mut hit: HashSet<usize> = HashSet::new();
        let mut samples = 0usize;
        let mut counts = std::collections::HashMap::<usize, usize>::new();
        for seed in 0..600u64 {
            let mut sampler = LpSampler::new(n, seed).expect("ok");
            for &i in &support_indices {
                sampler.update(i, 1).expect("u");
            }
            if let Some((idx, _)) = sampler.sample() {
                assert!(
                    support_indices.contains(&idx),
                    "sample {idx} outside support"
                );
                hit.insert(idx);
                *counts.entry(idx).or_insert(0) += 1;
                samples += 1;
            }
        }
        // Every support member should be reachable across the seeds.
        assert_eq!(
            hit.len(),
            support_indices.len(),
            "not all support members were ever sampled: hit {hit:?}"
        );
        // Not degenerate: no single index dominates *all* samples.
        let max_count = counts.values().copied().max().unwrap_or(0);
        assert!(
            (max_count as f64) < 0.9 * samples as f64,
            "sampling collapsed onto one index ({max_count}/{samples})"
        );
        assert!(samples > 100, "too few successful samples: {samples}");
    }

    // (f) All-zero input ⇒ None.
    #[test]
    fn all_zero_yields_none() {
        let sampler = LpSampler::new(1024, 7).expect("ok");
        assert_eq!(sampler.sample(), None);
        // Also after add/remove that nets to zero across several coords.
        let mut s2 = LpSampler::new(1024, 8).expect("ok");
        s2.update(1, 5).expect("u");
        s2.update(2, -3).expect("u");
        s2.update(1, -5).expect("u");
        s2.update(2, 3).expect("u");
        assert_eq!(s2.sample(), None, "balanced turnstile not empty");
    }

    // Parameter validation.
    #[test]
    fn invalid_parameters_error() {
        assert!(matches!(
            LpSampler::new(0, 1),
            Err(SketchError::InvalidParameter { .. })
        ));
        let mut s = LpSampler::new(16, 1).expect("ok");
        assert!(matches!(
            s.update(16, 1),
            Err(SketchError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            s.recover_level(usize::MAX),
            Err(SketchError::IndexOutOfBounds { .. })
        ));
    }

    // Linear merge: sketching two disjoint halves and merging equals sketching
    // the whole stream.
    #[test]
    fn merge_is_linear() {
        let n = 1024;
        let seed = 555;
        let mut whole = LpSampler::new(n, seed).expect("ok");
        let mut a = LpSampler::new(n, seed).expect("ok");
        let mut b = LpSampler::new(n, seed).expect("ok");
        let updates = [(3i64, 10usize), (5, 200), (-2, 10), (7, 500), (1, 999)];
        for (k, &(delta, idx)) in updates.iter().enumerate() {
            whole.update(idx, delta).expect("u");
            if k % 2 == 0 {
                a.update(idx, delta).expect("u");
            } else {
                b.update(idx, delta).expect("u");
            }
        }
        a.merge(&b).expect("merge");
        // The merged per-level state must match the whole stream exactly, so the
        // samples from both must agree.
        assert_eq!(a.sample(), whole.sample(), "merge ≠ whole");
        // Merge guards.
        let other_n = LpSampler::new(2048, seed).expect("ok");
        assert!(a.merge(&other_n).is_err(), "n mismatch must error");
        let other_seed = LpSampler::new(n, seed + 1).expect("ok");
        assert!(a.merge(&other_seed).is_err(), "seed mismatch must error");
    }

    // The level structure: a single coordinate survives to a contiguous prefix of
    // levels starting at 0, and deeper levels keep progressively fewer coords.
    #[test]
    fn level_structure_is_geometric() {
        let n = 4096;
        let sampler = LpSampler::new(n, 314).expect("ok");
        // Count how many coordinates survive to each level.
        let mut per_level = vec![0usize; sampler.n_levels()];
        for i in 0..n {
            let deepest = sampler.survive_level(i);
            for (lvl, slot) in per_level.iter_mut().enumerate() {
                if lvl <= deepest {
                    *slot += 1;
                }
            }
        }
        // Level 0 holds everything.
        assert_eq!(per_level[0], n);
        // Each level is (weakly) sparser than the previous, and the counts decay
        // roughly geometrically — the last level keeps a small handful.
        for w in per_level.windows(2) {
            assert!(w[1] <= w[0], "level counts not monotone: {per_level:?}");
        }
        assert!(
            *per_level.last().expect("nonempty") <= n / 4 + 4,
            "deepest level not sparse enough: {per_level:?}"
        );
    }

    // Field arithmetic sanity (the fingerprint correctness rests on these).
    #[test]
    fn field_helpers_consistent() {
        // field_signed for a negative value plus the positive must vanish.
        let s = mod_add(field_signed(-5), field_signed(5));
        assert_eq!(s, 0, "signed field add ≠ 0");
        // mod_mul matches a direct 128-bit reference for a few values.
        let a = 123_456_789u64;
        let b = 987_654_321u64;
        let direct = ((a as u128 * b as u128) % PRIME_MERSENNE_61 as u128) as u64;
        assert_eq!(mod_mul(a, b), direct);
    }
}
