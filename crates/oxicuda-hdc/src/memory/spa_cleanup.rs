//! Semantic Pointer Architecture (SPA) cleanup memory.
//!
//! Reference: C. Eliasmith, "How to Build a Brain: A Neural Architecture for Biological
//! Cognition" (Oxford University Press, 2013), chapter on the Semantic Pointer Architecture.
//!
//! A *cleanup memory* stores a set of clean prototype vectors — *semantic pointers*, typically
//! unit-norm real-valued vectors. Given a noisy query (for example the imperfect result of an
//! unbinding), it returns the closest clean prototype. Beyond hard winner-take-all, it also
//! supports an *associative* (soft) readout that superposes every prototype whose similarity
//! clears a threshold, weighted by how far it clears it:
//!
//! ```text
//! cosine(q, pᵢ)        = q̂ · pᵢ                          (pᵢ unit-norm)
//! cleanup(q)           = argmaxᵢ cosine(q, pᵢ)            (hard winner)
//! cleanup_threshold(q) = Σᵢ max(cosine(q, pᵢ) − τ, 0) · pᵢ  (thresholded-linear soft readout)
//! ```
//!
//! Unlike [`crate::memory::item_memory`] (which performs nearest-neighbour over `±1` binary
//! hypervectors keyed by symbol), `SpaCleanup` operates on real (`f32`) unit vectors and offers
//! a similarity threshold, a soft weighted-superposition output, and top-`k` retrieval. Dot
//! products are accumulated in `f64` to reduce rounding error and cast back to `f32`.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// A cleanup memory over unit-norm real-valued semantic pointers.
#[derive(Debug, Clone)]
pub struct SpaCleanup {
    dim: usize,
    labels: Vec<usize>,
    /// Each stored pointer, normalised to unit L2 norm on insertion.
    pointers: Vec<Vec<f32>>,
}

impl SpaCleanup {
    /// Create an empty cleanup memory for pointers of dimension `dim`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    pub fn new(dim: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            dim,
            labels: Vec::new(),
            pointers: Vec::new(),
        })
    }

    /// Add a labelled prototype, stored as a unit-norm copy.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `pointer.len() != dim`.
    /// - [`HdcError::DivisionByZero`] if `pointer` is non-finite or has (near-)zero norm.
    pub fn add(&mut self, label: usize, pointer: &[f32]) -> HdcResult<()> {
        if pointer.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: pointer.len(),
            });
        }
        let mut norm_sq = 0f64;
        for &x in pointer {
            if !x.is_finite() {
                return Err(HdcError::DivisionByZero);
            }
            norm_sq += (x as f64) * (x as f64);
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-12 {
            return Err(HdcError::DivisionByZero);
        }
        let inv = 1.0 / norm;
        let unit: Vec<f32> = pointer.iter().map(|&x| ((x as f64) * inv) as f32).collect();
        self.labels.push(label);
        self.pointers.push(unit);
        Ok(())
    }

    /// Add a labelled prototype drawn as a random unit vector (uniform `[-1, 1]`, normalised).
    pub fn add_random(&mut self, label: usize, rng: &mut LcgRng) {
        let mut v: Vec<f32> = (0..self.dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let mut norm = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        // Re-draw the (astronomically unlikely) all-zero vector so we always store a unit one.
        while norm < 1e-12 {
            for x in v.iter_mut() {
                *x = rng.next_f32() * 2.0 - 1.0;
            }
            norm = v
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
        }
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x = ((*x as f64) * inv) as f32;
        }
        self.labels.push(label);
        self.pointers.push(v);
    }

    /// Number of stored prototypes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pointers.len()
    }

    /// True when no prototypes are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    /// The pointer dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Normalise `query` to unit norm in `f64`, returning the unit components.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`HdcError::DivisionByZero`] if `query` is non-finite or has (near-)zero norm.
    fn unit_query(&self, query: &[f32]) -> HdcResult<Vec<f64>> {
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut norm_sq = 0f64;
        for &x in query {
            if !x.is_finite() {
                return Err(HdcError::DivisionByZero);
            }
            norm_sq += (x as f64) * (x as f64);
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-12 {
            return Err(HdcError::DivisionByZero);
        }
        let inv = 1.0 / norm;
        Ok(query.iter().map(|&x| (x as f64) * inv).collect())
    }

    /// Cosine between a unit `query` (as `f64`) and the stored unit pointer at `idx`.
    fn cosine_at(&self, query_unit: &[f64], idx: usize) -> f32 {
        let dot: f64 = query_unit
            .iter()
            .zip(self.pointers[idx].iter())
            .map(|(&q, &p)| q * (p as f64))
            .sum();
        dot as f32
    }

    /// Cosine similarity of `query` against every stored pointer, in insertion order.
    ///
    /// Each pair is `(label, cosine)`, with `cosine = q̂ · pᵢ` for the unit-normalised query.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`HdcError::DivisionByZero`] if `query` has (near-)zero norm.
    pub fn similarities(&self, query: &[f32]) -> HdcResult<Vec<(usize, f32)>> {
        let query_unit = self.unit_query(query)?;
        Ok(self
            .labels
            .iter()
            .enumerate()
            .map(|(idx, &label)| (label, self.cosine_at(&query_unit, idx)))
            .collect())
    }

    /// Hard winner-take-all cleanup: the label of the highest-cosine prototype.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyItemMemory`] if nothing is stored.
    /// - [`HdcError::DimensionMismatch`] / [`HdcError::DivisionByZero`] from query validation.
    pub fn cleanup(&self, query: &[f32]) -> HdcResult<usize> {
        if self.pointers.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let query_unit = self.unit_query(query)?;
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for idx in 0..self.pointers.len() {
            let score = self.cosine_at(&query_unit, idx);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Ok(self.labels[best_idx])
    }

    /// Return the *cleaned* vector: the stored unit pointer of the hard winner.
    ///
    /// # Errors
    ///
    /// Same as [`cleanup`](Self::cleanup).
    pub fn cleanup_vector(&self, query: &[f32]) -> HdcResult<Vec<f32>> {
        if self.pointers.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let query_unit = self.unit_query(query)?;
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for idx in 0..self.pointers.len() {
            let score = self.cosine_at(&query_unit, idx);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Ok(self.pointers[best_idx].clone())
    }

    /// Soft / associative cleanup: superpose all prototypes whose cosine clears `threshold`.
    ///
    /// Each qualifying pointer is weighted by `(cosine − threshold).max(0)` — a
    /// thresholded-linear readout — and the (un-normalised) sum is returned. If no prototype
    /// clears the threshold, a zero vector of length `dim` is returned.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyItemMemory`] if nothing is stored.
    /// - [`HdcError::DimensionMismatch`] / [`HdcError::DivisionByZero`] from query validation.
    pub fn cleanup_threshold(&self, query: &[f32], threshold: f32) -> HdcResult<Vec<f32>> {
        if self.pointers.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let query_unit = self.unit_query(query)?;
        let mut acc = vec![0f64; self.dim];
        for idx in 0..self.pointers.len() {
            let score = self.cosine_at(&query_unit, idx);
            let weight = (score - threshold).max(0.0) as f64;
            if weight > 0.0 {
                for (slot, &p) in acc.iter_mut().zip(self.pointers[idx].iter()) {
                    *slot += weight * (p as f64);
                }
            }
        }
        Ok(acc.into_iter().map(|v| v as f32).collect())
    }

    /// The `k` highest-cosine prototypes, sorted descending (ties broken by ascending label).
    ///
    /// `k` is clamped to the number of stored prototypes.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyItemMemory`] if nothing is stored.
    /// - [`HdcError::DimensionMismatch`] / [`HdcError::DivisionByZero`] from query validation.
    pub fn top_k(&self, query: &[f32], k: usize) -> HdcResult<Vec<(usize, f32)>> {
        if self.pointers.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let mut scored = self.similarities(query)?;
        // Descending by cosine; ties broken by ascending label for a stable, deterministic order.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k.min(scored.len()));
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0x5BAC_1EA0_0001)
    }

    /// Build a random unit vector of length `dim`.
    fn random_unit(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let norm = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        for x in v.iter_mut() {
            *x = ((*x as f64) / norm) as f32;
        }
        v
    }

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt() as f32
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum();
        let na = a
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let nb = b
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        (dot / (na * nb)) as f32
    }

    #[test]
    fn new_rejects_zero_dim() {
        assert!(matches!(SpaCleanup::new(0), Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn add_rejects_wrong_len() {
        let mut mem = SpaCleanup::new(8).expect("new");
        let res = mem.add(0, &[1.0f32; 4]);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn add_rejects_zero_vector() {
        let mut mem = SpaCleanup::new(8).expect("new");
        let res = mem.add(0, &[0.0f32; 8]);
        assert!(matches!(res, Err(HdcError::DivisionByZero)));
    }

    #[test]
    fn stored_pointers_are_normalised() {
        let mut mem = SpaCleanup::new(16).expect("new");
        // A clearly non-unit input gets normalised on store.
        let raw = vec![3.0f32; 16];
        mem.add(7, &raw).expect("add");
        let sims = mem.similarities(&raw).expect("sim");
        assert_eq!(sims.len(), 1);
        // Self-cosine of the (now unit) prototype against the (unit-normalised) query ≈ 1.
        assert!((sims[0].1 - 1.0).abs() < 1e-5, "self-cos={}", sims[0].1);
    }

    #[test]
    fn cleanup_recovers_exact_prototype() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(256).expect("new");
        let mut protos = Vec::new();
        for label in 0..4 {
            let p = random_unit(256, &mut rng);
            mem.add(label, &p).expect("add");
            protos.push(p);
        }
        // Query == prototype 2 → returns label 2.
        let label = mem.cleanup(&protos[2]).expect("cleanup");
        assert_eq!(label, 2);
    }

    #[test]
    fn cleanup_recovers_from_noisy_query() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(256).expect("new");
        let mut protos = Vec::new();
        for label in 0..4 {
            let p = random_unit(256, &mut rng);
            mem.add(label, &p).expect("add");
            protos.push(p);
        }
        // Add modest noise to prototype 1 (high-D random vectors are near-orthogonal).
        let mut noisy = protos[1].clone();
        for slot in noisy.iter_mut() {
            // Small deterministic perturbation derived from the RNG.
            *slot += (rng.next_f32() - 0.5) * 0.2;
        }
        let label = mem.cleanup(&noisy).expect("cleanup");
        assert_eq!(label, 1, "noisy query did not clean up to prototype 1");
    }

    #[test]
    fn cleanup_vector_returns_unit_prototype() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(128).expect("new");
        let p = random_unit(128, &mut rng);
        mem.add(3, &p).expect("add");
        let cleaned = mem.cleanup_vector(&p).expect("cleanup_vector");
        assert_eq!(cleaned.len(), 128);
        assert!(
            (l2_norm(&cleaned) - 1.0).abs() < 1e-5,
            "norm={}",
            l2_norm(&cleaned)
        );
        assert!(cosine(&cleaned, &p) > 0.999, "cleaned != stored prototype");
    }

    #[test]
    fn similarities_length_and_self_cosine() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(256).expect("new");
        let mut protos = Vec::new();
        for label in 0..3 {
            let p = random_unit(256, &mut rng);
            mem.add(label, &p).expect("add");
            protos.push(p);
        }
        let sims = mem.similarities(&protos[0]).expect("sim");
        assert_eq!(sims.len(), 3);
        // First entry is prototype 0 with self-cosine ≈ 1.
        assert_eq!(sims[0].0, 0);
        assert!((sims[0].1 - 1.0).abs() < 1e-5, "self-cos={}", sims[0].1);
    }

    #[test]
    fn cleanup_threshold_isolates_single_pointer() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(256).expect("new");
        let mut protos = Vec::new();
        for label in 0..4 {
            let p = random_unit(256, &mut rng);
            mem.add(label, &p).expect("add");
            protos.push(p);
        }
        // High threshold: only the (self-similar) prototype 2 clears it.
        let out = mem.cleanup_threshold(&protos[2], 0.5).expect("threshold");
        assert_eq!(out.len(), 256);
        // The soft readout points along prototype 2.
        let sim = cosine(&out, &protos[2]);
        assert!(sim > 0.95, "thresholded readout cosine to proto2 = {sim}");
    }

    #[test]
    fn cleanup_threshold_zero_when_above_all() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(256).expect("new");
        let mut protos = Vec::new();
        for label in 0..4 {
            let p = random_unit(256, &mut rng);
            mem.add(label, &p).expect("add");
            protos.push(p);
        }
        // Threshold above any achievable cosine (max is 1.0 for self) → zero vector.
        let out = mem.cleanup_threshold(&protos[0], 1.5).expect("threshold");
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 0.0), "expected all-zero readout");
    }

    #[test]
    fn top_k_sorted_descending_and_clamped() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(256).expect("new");
        let mut protos = Vec::new();
        for label in 0..5 {
            let p = random_unit(256, &mut rng);
            mem.add(label, &p).expect("add");
            protos.push(p);
        }
        let top = mem.top_k(&protos[3], 3).expect("top_k");
        assert_eq!(top.len(), 3);
        // Descending order.
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1, "not descending: {:?}", top);
        }
        // The best match is prototype 3 (queried with itself).
        assert_eq!(top[0].0, 3);
        // Clamp: requesting more than stored returns all.
        let all = mem.top_k(&protos[0], 99).expect("top_k clamp");
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn empty_memory_queries_error() {
        let mem = SpaCleanup::new(16).expect("new");
        let q = vec![1.0f32; 16];
        assert!(matches!(mem.cleanup(&q), Err(HdcError::EmptyItemMemory)));
        assert!(matches!(
            mem.cleanup_vector(&q),
            Err(HdcError::EmptyItemMemory)
        ));
        assert!(matches!(
            mem.cleanup_threshold(&q, 0.0),
            Err(HdcError::EmptyItemMemory)
        ));
        assert!(matches!(mem.top_k(&q, 3), Err(HdcError::EmptyItemMemory)));
    }

    #[test]
    fn similarities_rejects_zero_query() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(16).expect("new");
        mem.add_random(0, &mut rng);
        let res = mem.similarities(&[0.0f32; 16]);
        assert!(matches!(res, Err(HdcError::DivisionByZero)));
    }

    #[test]
    fn add_random_determinism() {
        let mut r1 = LcgRng::new(0x1111_2222);
        let mut r2 = LcgRng::new(0x1111_2222);
        let mut m1 = SpaCleanup::new(64).expect("m1");
        let mut m2 = SpaCleanup::new(64).expect("m2");
        for label in 0..4 {
            m1.add_random(label, &mut r1);
            m2.add_random(label, &mut r2);
        }
        // Same seed → identical stored pointers, hence identical similarity profile.
        let q = vec![1.0f32; 64];
        let s1 = m1.similarities(&q).expect("s1");
        let s2 = m2.similarities(&q).expect("s2");
        assert_eq!(s1, s2);
    }

    #[test]
    fn add_random_produces_unit_vectors() {
        let mut rng = rng();
        let mut mem = SpaCleanup::new(128).expect("new");
        mem.add_random(0, &mut rng);
        // Self-query of the stored pointer yields cosine ≈ 1, confirming unit norm.
        let stored = mem.cleanup_vector(&vec_one(128)).expect("cleanup_vector");
        assert!(
            (l2_norm(&stored) - 1.0).abs() < 1e-5,
            "norm={}",
            l2_norm(&stored)
        );
    }

    fn vec_one(dim: usize) -> Vec<f32> {
        vec![1.0f32; dim]
    }
}
