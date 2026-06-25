//! Reference-model log-probability caching for a stationary reference policy.
//!
//! In DPO / IPO / KTO / PPO-RLHF the reference (frozen SFT) policy is evaluated
//! on the *same* prompts and responses every optimisation step. Because the
//! reference is stationary its log-probabilities never change, so re-running a
//! forward pass through it each step is pure waste. The standard optimisation is
//! to compute the reference log-probs **once** (keyed by an example identifier)
//! and look them up on every subsequent step.
//!
//! This module provides a deterministic, insertion-ordered cache keyed by a
//! `u64` example id. It stores either a single scalar reference log-prob (the
//! sequence-level log-prob used by DPO) or a per-token log-prob vector (used by
//! token-level KL penalties). A simple hit/miss counter lets callers verify the
//! cache is actually saving forward passes.
//!
//! The cache is a plain in-memory map (no eviction): the reference set is fixed
//! for a training run, so unbounded growth is bounded by the dataset size. A
//! `RefLogProbCache::capacity` guard rejects insertions beyond a caller-set
//! limit so a mis-keyed loop cannot exhaust memory silently.

use std::collections::HashMap;

use crate::error::{RlhfError, RlhfResult};

// ── Cached entry ────────────────────────────────────────────────────────────

/// A cached reference log-probability entry.
#[derive(Debug, Clone, PartialEq)]
pub enum RefLogProb {
    /// Sequence-level scalar reference log-prob (DPO-style).
    Scalar(f32),
    /// Per-token reference log-prob vector (token-level KL).
    PerToken(Vec<f32>),
}

// ── Cache ───────────────────────────────────────────────────────────────────

/// Insertion-ordered cache of stationary reference log-probs keyed by example
/// id, with hit/miss accounting and an optional capacity bound.
#[derive(Debug, Clone)]
pub struct RefLogProbCache {
    entries: HashMap<u64, RefLogProb>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl RefLogProbCache {
    /// Create an empty cache bounded to `capacity` entries (`0` → unbounded).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cumulative cache hits recorded by [`RefLogProbCache::get_or_compute_scalar`]
    /// / [`RefLogProbCache::get_or_compute_per_token`].
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cumulative cache misses (entries that triggered a compute).
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Whether an entry exists for `id` (does not affect hit/miss counters).
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.entries.contains_key(&id)
    }

    fn check_capacity(&self, id: u64) -> RlhfResult<()> {
        if self.capacity != 0
            && !self.entries.contains_key(&id)
            && self.entries.len() >= self.capacity
        {
            return Err(RlhfError::Internal {
                msg: format!(
                    "reference cache capacity {} exceeded (id {id})",
                    self.capacity
                ),
            });
        }
        Ok(())
    }

    /// Insert / overwrite a scalar reference log-prob.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::NanEncountered`] for a non-finite value and
    /// [`RlhfError::Internal`] when inserting a *new* id past `capacity`.
    pub fn insert_scalar(&mut self, id: u64, logp: f32) -> RlhfResult<()> {
        if !logp.is_finite() {
            return Err(RlhfError::NanEncountered);
        }
        self.check_capacity(id)?;
        self.entries.insert(id, RefLogProb::Scalar(logp));
        Ok(())
    }

    /// Insert / overwrite a per-token reference log-prob vector.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::EmptyInput`] for an empty vector,
    /// [`RlhfError::NanEncountered`] for any non-finite element, and
    /// [`RlhfError::Internal`] when inserting a *new* id past `capacity`.
    pub fn insert_per_token(&mut self, id: u64, logps: Vec<f32>) -> RlhfResult<()> {
        if logps.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if logps.iter().any(|x| !x.is_finite()) {
            return Err(RlhfError::NanEncountered);
        }
        self.check_capacity(id)?;
        self.entries.insert(id, RefLogProb::PerToken(logps));
        Ok(())
    }

    /// Fetch a scalar reference log-prob for `id`, computing and caching it via
    /// `compute` on a miss.
    ///
    /// On a hit the cached value is returned and the hit counter is incremented;
    /// on a miss `compute()` is invoked (simulating a reference forward pass),
    /// the result is validated and stored, and the miss counter is incremented.
    ///
    /// # Errors
    ///
    /// Propagates any error from `compute`, [`RlhfError::NanEncountered`] for a
    /// non-finite computed value, a type mismatch if `id` already holds a
    /// per-token entry, and capacity errors from [`RefLogProbCache::insert_scalar`].
    pub fn get_or_compute_scalar<F>(&mut self, id: u64, compute: F) -> RlhfResult<f32>
    where
        F: FnOnce() -> RlhfResult<f32>,
    {
        if let Some(entry) = self.entries.get(&id) {
            return match entry {
                RefLogProb::Scalar(v) => {
                    self.hits += 1;
                    Ok(*v)
                }
                RefLogProb::PerToken(_) => Err(RlhfError::Internal {
                    msg: format!("id {id} cached as per-token, requested scalar"),
                }),
            };
        }
        let value = compute()?;
        self.insert_scalar(id, value)?;
        self.misses += 1;
        Ok(value)
    }

    /// Fetch a per-token reference log-prob vector for `id`, computing and
    /// caching it via `compute` on a miss. Semantics mirror
    /// [`RefLogProbCache::get_or_compute_scalar`].
    ///
    /// # Errors
    ///
    /// Propagates any error from `compute`, validation errors from
    /// [`RefLogProbCache::insert_per_token`], and a type mismatch if `id`
    /// already holds a scalar entry.
    pub fn get_or_compute_per_token<F>(&mut self, id: u64, compute: F) -> RlhfResult<Vec<f32>>
    where
        F: FnOnce() -> RlhfResult<Vec<f32>>,
    {
        if let Some(entry) = self.entries.get(&id) {
            return match entry {
                RefLogProb::PerToken(v) => {
                    self.hits += 1;
                    Ok(v.clone())
                }
                RefLogProb::Scalar(_) => Err(RlhfError::Internal {
                    msg: format!("id {id} cached as scalar, requested per-token"),
                }),
            };
        }
        let value = compute()?;
        self.insert_per_token(id, value.clone())?;
        self.misses += 1;
        Ok(value)
    }

    /// Clear all entries; counters are preserved.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // 1. First access misses, second hits, value is stable.
    #[test]
    fn miss_then_hit() {
        let mut cache = RefLogProbCache::new(0);
        let calls = Cell::new(0);
        let v1 = cache
            .get_or_compute_scalar(42, || {
                calls.set(calls.get() + 1);
                Ok(-1.5)
            })
            .expect("first");
        let v2 = cache
            .get_or_compute_scalar(42, || {
                calls.set(calls.get() + 1);
                Ok(-1.5)
            })
            .expect("second");
        assert!((v1 - v2).abs() < 1e-9);
        assert_eq!(calls.get(), 1, "compute should run exactly once");
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 1);
    }

    // 2. Cached value is returned even if compute would give something else.
    #[test]
    fn cache_returns_stored_not_recomputed() {
        let mut cache = RefLogProbCache::new(0);
        let _ = cache.get_or_compute_scalar(1, || Ok(-2.0)).expect("first");
        // Second compute returns a different value; cache must ignore it.
        let v = cache.get_or_compute_scalar(1, || Ok(99.0)).expect("second");
        assert!(
            (v - (-2.0)).abs() < 1e-9,
            "must return cached -2.0, got {v}"
        );
    }

    // 3. Per-token caching round-trips and counts hits.
    #[test]
    fn per_token_roundtrip() {
        let mut cache = RefLogProbCache::new(0);
        let v1 = cache
            .get_or_compute_per_token(7, || Ok(vec![-0.1, -0.2, -0.3]))
            .expect("first");
        let v2 = cache
            .get_or_compute_per_token(7, || Ok(vec![1.0, 1.0, 1.0]))
            .expect("second");
        assert_eq!(v1, vec![-0.1, -0.2, -0.3]);
        assert_eq!(v2, v1, "second must return cached vector");
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    // 4. Distinct ids are cached independently.
    #[test]
    fn distinct_ids_independent() {
        let mut cache = RefLogProbCache::new(0);
        let a = cache.get_or_compute_scalar(1, || Ok(-1.0)).expect("a");
        let b = cache.get_or_compute_scalar(2, || Ok(-2.0)).expect("b");
        assert!((a - (-1.0)).abs() < 1e-9);
        assert!((b - (-2.0)).abs() < 1e-9);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.misses(), 2);
    }

    // 5. Capacity bound rejects new ids past the limit.
    #[test]
    fn capacity_bound_enforced() {
        let mut cache = RefLogProbCache::new(2);
        cache.insert_scalar(1, -1.0).expect("1");
        cache.insert_scalar(2, -2.0).expect("2");
        assert!(matches!(
            cache.insert_scalar(3, -3.0),
            Err(RlhfError::Internal { .. })
        ));
        // Overwriting an existing id is still allowed at capacity.
        cache
            .insert_scalar(2, -2.5)
            .expect("overwrite within capacity");
        assert_eq!(cache.len(), 2);
    }

    // 6. Type mismatch errors (scalar requested for per-token entry).
    #[test]
    fn type_mismatch_errors() {
        let mut cache = RefLogProbCache::new(0);
        cache.insert_per_token(5, vec![-0.5]).expect("insert");
        assert!(matches!(
            cache.get_or_compute_scalar(5, || Ok(-1.0)),
            Err(RlhfError::Internal { .. })
        ));
        let mut cache2 = RefLogProbCache::new(0);
        cache2.insert_scalar(6, -1.0).expect("insert");
        assert!(matches!(
            cache2.get_or_compute_per_token(6, || Ok(vec![-1.0])),
            Err(RlhfError::Internal { .. })
        ));
    }

    // 7. NaN / infinite values rejected.
    #[test]
    fn non_finite_rejected() {
        let mut cache = RefLogProbCache::new(0);
        assert!(matches!(
            cache.insert_scalar(1, f32::NAN),
            Err(RlhfError::NanEncountered)
        ));
        assert!(matches!(
            cache.insert_per_token(2, vec![-1.0, f32::INFINITY]),
            Err(RlhfError::NanEncountered)
        ));
    }

    // 8. Empty per-token vector rejected.
    #[test]
    fn empty_per_token_rejected() {
        let mut cache = RefLogProbCache::new(0);
        assert!(matches!(
            cache.insert_per_token(1, vec![]),
            Err(RlhfError::EmptyInput)
        ));
    }

    // 9. compute error propagates and is not cached.
    #[test]
    fn compute_error_propagates() {
        let mut cache = RefLogProbCache::new(0);
        let res = cache.get_or_compute_scalar(1, || Err(RlhfError::EmptyInput));
        assert!(matches!(res, Err(RlhfError::EmptyInput)));
        assert!(!cache.contains(1), "failed compute must not insert");
        assert_eq!(cache.misses(), 0);
    }

    // 10. Repeated access over a "training loop" yields exactly one miss per id.
    #[test]
    fn loop_yields_one_miss_per_id() {
        let mut cache = RefLogProbCache::new(0);
        let computes = Cell::new(0);
        for _step in 0..10 {
            for id in 0..4_u64 {
                let _ = cache
                    .get_or_compute_scalar(id, || {
                        computes.set(computes.get() + 1);
                        Ok(-(id as f32))
                    })
                    .expect("get");
            }
        }
        assert_eq!(computes.get(), 4, "exactly one forward pass per unique id");
        assert_eq!(cache.misses(), 4);
        assert_eq!(cache.hits(), 36, "remaining 9 steps * 4 ids hit the cache");
    }

    // 11. clear empties entries but preserves counters.
    #[test]
    fn clear_preserves_counters() {
        let mut cache = RefLogProbCache::new(0);
        let _ = cache.get_or_compute_scalar(1, || Ok(-1.0)).expect("get");
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.misses(), 1, "counters survive clear");
    }
}
