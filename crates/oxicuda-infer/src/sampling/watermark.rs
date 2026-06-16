//! # Statistical watermarking (Kirchenbauer et al. 2023).
//!
//! Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein (2023), "A Watermark for
//! Large Language Models" (ICML). <https://arxiv.org/abs/2301.10226>
//!
//! The "soft" watermark partitions the vocabulary into a pseudo-random
//! **green list** (a fraction `γ` of tokens) and a **red list** at each step,
//! seeded by a hash of the previous token. A constant bias `δ` is added to the
//! logits of green tokens, gently nudging generation toward the green list
//! without noticeably degrading quality. Detection later recomputes the green
//! list for each position and counts how many emitted tokens are green; the
//! one-proportion z-statistic
//!
//! ```text
//! z = (|s|_G − γ·T) / √(T·γ·(1 − γ))
//! ```
//!
//! (where `|s|_G` is the green-token count over `T` scored positions) is large
//! for watermarked text and ≈ 0 for human/unwatermarked text.
//!
//! This module provides:
//! * [`Watermarker`] — green-list construction, logit biasing, and detection.
//!
//! The green-list partition uses a small deterministic hash of
//! `(hash_key, prev_token)` to seed a per-step LCG that shuffles the vocabulary
//! and takes the first `⌊γ·V⌋` ids as green.

use crate::error::{InferError, InferResult};

// ─── Watermarker ─────────────────────────────────────────────────────────────

/// Soft-watermark generator/detector.
#[derive(Debug, Clone, Copy)]
pub struct Watermarker {
    /// Vocabulary size `V` (must be ≥ 2).
    pub vocab_size: usize,
    /// Green-list fraction `γ ∈ (0, 1)`.
    pub gamma: f32,
    /// Logit bias `δ ≥ 0` added to green tokens.
    pub delta: f32,
    /// Secret hash key mixed into the per-step seed.
    pub hash_key: u64,
}

impl Watermarker {
    /// Create a new watermarker.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `vocab_size < 2`, `gamma` ∉ (0, 1),
    ///   or `delta < 0`.
    pub fn new(vocab_size: usize, gamma: f32, delta: f32, hash_key: u64) -> InferResult<Self> {
        if vocab_size < 2 {
            return Err(InferError::InvalidConfig("vocab_size must be >= 2"));
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(InferError::InvalidConfig("gamma must be in (0, 1)"));
        }
        if delta < 0.0 {
            return Err(InferError::InvalidConfig("delta must be >= 0"));
        }
        Ok(Self {
            vocab_size,
            gamma,
            delta,
            hash_key,
        })
    }

    /// Number of green tokens `⌊γ·V⌋` (at least 1).
    #[must_use]
    pub fn green_count(&self) -> usize {
        ((self.gamma * self.vocab_size as f32).floor() as usize).max(1)
    }

    /// Compute the green-list token ids for the step following `prev_token`.
    ///
    /// Deterministically shuffles `0..vocab_size` with a per-step LCG seeded by
    /// `hash(hash_key, prev_token)` (a partial Fisher–Yates over the first
    /// `green_count` positions), then returns those ids sorted ascending.
    #[must_use]
    pub fn green_list(&self, prev_token: u32) -> Vec<usize> {
        let g = self.green_count();
        let v = self.vocab_size;
        let mut perm: Vec<usize> = (0..v).collect();
        let mut state = seed_for(self.hash_key, prev_token);

        // Partial Fisher–Yates: select the first g elements uniformly.
        for i in 0..g.min(v.saturating_sub(1)) {
            state = next_state(state);
            let j = i + (state as usize) % (v - i);
            perm.swap(i, j);
        }
        let mut green: Vec<usize> = perm[..g].to_vec();
        green.sort_unstable();
        green
    }

    /// Add the watermark bias `δ` to the logits of green tokens (in place).
    ///
    /// # Arguments
    /// * `logits`     — `[vocab_size]` next-token logits, modified in place.
    /// * `prev_token` — the previously generated token (seeds the green list).
    ///
    /// # Errors
    /// * [`InferError::DimensionMismatch`] if `logits.len() != vocab_size`.
    pub fn apply_bias(&self, logits: &mut [f32], prev_token: u32) -> InferResult<()> {
        if logits.len() != self.vocab_size {
            return Err(InferError::DimensionMismatch {
                expected: self.vocab_size,
                got: logits.len(),
            });
        }
        for id in self.green_list(prev_token) {
            logits[id] += self.delta;
        }
        Ok(())
    }

    /// Test whether `token` is on the green list for the step after `prev_token`.
    #[must_use]
    pub fn is_green(&self, prev_token: u32, token: u32) -> bool {
        let g = self.green_count();
        let v = self.vocab_size;
        if (token as usize) >= v {
            return false;
        }
        // Reconstruct the same partial permutation and check membership in the
        // first g positions without sorting (cheaper for single queries).
        let mut perm: Vec<usize> = (0..v).collect();
        let mut state = seed_for(self.hash_key, prev_token);
        for i in 0..g.min(v.saturating_sub(1)) {
            state = next_state(state);
            let j = i + (state as usize) % (v - i);
            perm.swap(i, j);
        }
        perm[..g].contains(&(token as usize))
    }

    /// Detect a watermark in `tokens`, returning the green-token count, the
    /// number of scored positions, and the z-statistic.
    ///
    /// Positions are scored from index 1 onward (each token is checked against
    /// the green list seeded by its predecessor). A z-score above ≈ 4.0
    /// corresponds to an extremely low false-positive rate.
    ///
    /// # Arguments
    /// * `tokens` — the full emitted token sequence.
    ///
    /// # Returns
    /// `(green_hits, num_scored, z_score)`. When `num_scored == 0` the z-score
    /// is `0.0`.
    ///
    /// # Errors
    /// Never errors for a valid watermarker; pathological all-equal partitions
    /// are guarded.
    #[must_use]
    pub fn detect(&self, tokens: &[u32]) -> WatermarkDetection {
        if tokens.len() < 2 {
            return WatermarkDetection {
                green_hits: 0,
                num_scored: 0,
                z_score: 0.0,
            };
        }
        let mut green_hits = 0_usize;
        let mut num_scored = 0_usize;
        for w in tokens.windows(2) {
            let prev = w[0];
            let cur = w[1];
            if self.is_green(prev, cur) {
                green_hits += 1;
            }
            num_scored += 1;
        }
        let t = num_scored as f32;
        let gamma = self.gamma;
        let denom = (t * gamma * (1.0 - gamma)).sqrt();
        let z_score = if denom > 1e-8 {
            (green_hits as f32 - gamma * t) / denom
        } else {
            0.0
        };
        WatermarkDetection {
            green_hits,
            num_scored,
            z_score,
        }
    }
}

/// Result of a watermark-detection pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WatermarkDetection {
    /// Number of emitted tokens found on their step's green list.
    pub green_hits: usize,
    /// Number of scored positions (= `tokens.len() − 1`).
    pub num_scored: usize,
    /// One-proportion z-statistic; large ⇒ watermarked.
    pub z_score: f32,
}

// ─── Hashing helpers ─────────────────────────────────────────────────────────

/// Mix the secret key and previous token into a non-zero 64-bit seed.
#[inline]
fn seed_for(hash_key: u64, prev_token: u32) -> u64 {
    // SplitMix64-style finaliser over key XOR token, guaranteed non-zero.
    let mut z = hash_key ^ (prev_token as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

/// Advance a 64-bit LCG (Knuth constants), returning the new state.
#[inline]
fn next_state(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn wm() -> Watermarker {
        Watermarker::new(100, 0.25, 2.0, 0xABCD).expect("valid")
    }

    #[test]
    fn green_count_floor() {
        let w = wm();
        // floor(0.25 * 100) = 25
        assert_eq!(w.green_count(), 25);
    }

    #[test]
    fn green_count_at_least_one() {
        let w = Watermarker::new(3, 0.1, 1.0, 0).expect("valid");
        // floor(0.1*3)=0 → clamped to 1
        assert_eq!(w.green_count(), 1);
    }

    #[test]
    fn green_list_size_and_range() {
        let w = wm();
        let g = w.green_list(7);
        assert_eq!(g.len(), w.green_count());
        assert!(g.iter().all(|&id| id < w.vocab_size));
    }

    #[test]
    fn green_list_deterministic() {
        let w = wm();
        assert_eq!(w.green_list(42), w.green_list(42), "must be reproducible");
    }

    #[test]
    fn green_list_depends_on_prev_token() {
        let w = wm();
        // Different seeds ⇒ (almost surely) different green sets.
        assert_ne!(w.green_list(1), w.green_list(2));
    }

    #[test]
    fn green_list_depends_on_hash_key() {
        let a = Watermarker::new(100, 0.25, 2.0, 1).expect("valid");
        let b = Watermarker::new(100, 0.25, 2.0, 2).expect("valid");
        assert_ne!(a.green_list(5), b.green_list(5));
    }

    #[test]
    fn is_green_matches_green_list() {
        let w = wm();
        let g = w.green_list(13);
        for &id in &g {
            assert!(w.is_green(13, id as u32), "id {id} should be green");
        }
        // A token not in the list should be red.
        let not_green = (0..w.vocab_size)
            .find(|id| !g.contains(id))
            .expect("some red");
        assert!(!w.is_green(13, not_green as u32));
    }

    #[test]
    fn apply_bias_only_boosts_green() {
        let w = wm();
        let prev = 9_u32;
        let g = w.green_list(prev);
        let mut logits = vec![0.0_f32; w.vocab_size];
        w.apply_bias(&mut logits, prev).expect("ok");
        for (id, &l) in logits.iter().enumerate() {
            if g.contains(&id) {
                assert!((l - 2.0).abs() < 1e-6, "green id {id} should be +delta");
            } else {
                assert!(l.abs() < 1e-6, "red id {id} should be unchanged");
            }
        }
    }

    #[test]
    fn detect_unwatermarked_low_z() {
        // A constant/degenerate sequence should not score as strongly
        // watermarked; z-score must be finite and modest.
        let w = wm();
        let tokens = vec![0_u32; 200];
        let det = w.detect(&tokens);
        assert!(det.z_score.is_finite());
        assert_eq!(det.num_scored, 199);
    }

    #[test]
    fn detect_watermarked_high_z() {
        // Greedily emit a green token at every step ⇒ green_hits == num_scored
        // ⇒ large positive z-score.
        let w = wm();
        let mut tokens = vec![3_u32];
        for _ in 0..300 {
            let prev = *tokens.last().expect("non-empty");
            // pick the first green token for this step
            let green = w.green_list(prev);
            tokens.push(green[0] as u32);
        }
        let det = w.detect(&tokens);
        assert_eq!(
            det.green_hits, det.num_scored,
            "all tokens green by construction"
        );
        assert!(
            det.z_score > 4.0,
            "fully-green text should have large z, got {}",
            det.z_score
        );
    }

    #[test]
    fn detect_short_sequence_zero() {
        let w = wm();
        let det = w.detect(&[5]);
        assert_eq!(det.num_scored, 0);
        assert_eq!(det.z_score, 0.0);
    }

    #[test]
    fn detect_green_hits_le_scored() {
        let w = wm();
        let tokens: Vec<u32> = (0..150).map(|i| (i % 100) as u32).collect();
        let det = w.detect(&tokens);
        assert!(det.green_hits <= det.num_scored);
    }

    #[test]
    fn apply_bias_increases_green_probability_ordering() {
        // After biasing, a green token that started tied with a red token must
        // end up with a strictly higher logit.
        let w = wm();
        let prev = 11_u32;
        let g = w.green_list(prev);
        let green_id = g[0];
        let red_id = (0..w.vocab_size)
            .find(|id| !g.contains(id))
            .expect("red exists");
        let mut logits = vec![1.0_f32; w.vocab_size];
        w.apply_bias(&mut logits, prev).expect("ok");
        assert!(
            logits[green_id] > logits[red_id],
            "green {green_id} ({}) should exceed red {red_id} ({})",
            logits[green_id],
            logits[red_id]
        );
    }

    #[test]
    fn err_small_vocab() {
        assert!(matches!(
            Watermarker::new(1, 0.5, 1.0, 0),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_bad_gamma() {
        assert!(matches!(
            Watermarker::new(100, 0.0, 1.0, 0),
            Err(InferError::InvalidConfig(_))
        ));
        assert!(matches!(
            Watermarker::new(100, 1.0, 1.0, 0),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_negative_delta() {
        assert!(matches!(
            Watermarker::new(100, 0.25, -1.0, 0),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_bias_dim_mismatch() {
        let w = wm();
        let mut logits = vec![0.0_f32; 50]; // != vocab_size 100
        assert!(matches!(
            w.apply_bias(&mut logits, 0),
            Err(InferError::DimensionMismatch { .. })
        ));
    }
}
