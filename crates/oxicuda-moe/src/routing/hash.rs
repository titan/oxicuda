//! Hash routing: deterministic, stateless token-to-expert mapping.
//!
//! Implements the routing mechanism from:
//! Roller et al. "Hash Layers For Large Sparse Models." NeurIPS 2021.
//!
//! Unlike learned routers (top-k softmax, Switch), a hash layer assigns each
//! token to an expert purely as a function of the token id. There are **no
//! learned parameters** and **no softmax / logits**: the mapping is a fixed
//! hash of the token id, which makes it fully reproducible per token regardless
//! of batch composition or ordering — ideal for cache-friendly serving where
//! a given token always lands on the same expert (and thus the same device).
//!
//! The hash is a `SplitMix64` finalizer applied to a mix of `(seed, token_id)`.
//! SplitMix64 is a well-distributed 64-bit avalanche finalizer (Steele,
//! Lea & Flood, OOPSLA 2014), so taking `hash % n_experts` yields a balanced,
//! deterministic assignment. We deliberately do **not** use the stateful
//! `LcgRng` from [`crate::handle`]: routing must depend only on the token id,
//! never on how many tokens were routed before it.
//!
//! Multi-hash routing (`n_hashes > 1`) assigns each token to several **distinct**
//! experts by perturbing the SplitMix64 input with a per-hash salt and rejecting
//! collisions. This mirrors the multi-hash variant in the paper, used to give
//! each token a small fixed set of experts.

use crate::error::{MoeError, MoeResult};

/// Configuration for hash routing.
#[derive(Debug, Clone)]
pub struct HashRoutingConfig {
    /// Total number of experts. Must be `> 0`.
    pub n_experts: usize,
    /// Number of distinct experts each token is mapped to. Must be in
    /// `1..=n_experts`. Defaults to `1` via [`HashRoutingConfig::default`].
    pub n_hashes: usize,
    /// Seed mixed into the hash, allowing different deterministic mappings.
    pub seed: u64,
}

impl Default for HashRoutingConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            n_hashes: 1,
            seed: 0,
        }
    }
}

/// Hash router: a stateless, parameter-free deterministic token→expert map.
#[derive(Debug, Clone)]
pub struct HashRouter {
    /// Routing configuration.
    pub config: HashRoutingConfig,
}

/// SplitMix64 finalizer (avalanche mixing of a 64-bit word).
///
/// This is the finalizing step of the SplitMix64 generator: a sequence of
/// xor-shift / odd-multiply rounds with strong avalanche, so a single-bit
/// change in the input flips ~half the output bits. Used here as a pure hash.
#[inline]
#[must_use]
fn splitmix64_finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hash a `(seed, token_id, salt)` triple to a 64-bit value.
///
/// The token id and salt are folded into the seed with distinct odd
/// multipliers (the SplitMix64 increment and the golden-ratio constant) so
/// that different salts produce statistically independent streams, then run
/// through the SplitMix64 finalizer.
#[inline]
#[must_use]
fn hash_token(seed: u64, token_id: u64, salt: u64) -> u64 {
    let mut z = seed;
    z = z.wrapping_add(token_id.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = z.wrapping_add(salt.wrapping_mul(0xff51_afd7_ed55_8ccd));
    splitmix64_finalize(z)
}

impl HashRouter {
    /// Create a new hash router, validating the configuration.
    ///
    /// # Errors
    /// * [`MoeError::InvalidExpertCount`] if `n_experts == 0`.
    /// * [`MoeError::InvalidTopK`] if `n_hashes == 0` or `n_hashes > n_experts`
    ///   (reusing the top-k error since `n_hashes` plays the same "experts per
    ///   token" role and must satisfy the same bound).
    pub fn new(config: HashRoutingConfig) -> MoeResult<Self> {
        if config.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: config.n_experts,
            });
        }
        if config.n_hashes == 0 || config.n_hashes > config.n_experts {
            return Err(MoeError::InvalidTopK {
                k: config.n_hashes,
                n_experts: config.n_experts,
            });
        }
        Ok(Self { config })
    }

    /// Map a single token id to its (distinct) expert ids.
    ///
    /// Returns a vector of length `n_hashes` containing distinct expert ids in
    /// `[0, n_experts)`. The result depends only on `(seed, token_id)`, never on
    /// surrounding tokens.
    #[must_use]
    pub fn route_token(&self, token_id: usize) -> Vec<usize> {
        let n_experts = self.config.n_experts;
        let n_hashes = self.config.n_hashes;
        let token_word = token_id as u64;

        // Fast path: single hash → single expert.
        if n_hashes == 1 {
            let h = hash_token(self.config.seed, token_word, 0);
            return vec![(h % n_experts as u64) as usize];
        }

        // Multi-hash: draw distinct experts by salting the hash and rejecting
        // collisions. `n_hashes <= n_experts` is guaranteed by `new`, so this
        // terminates: each new salt is a fresh independent draw, and we only
        // need `n_hashes` distinct values out of `n_experts`.
        let mut experts: Vec<usize> = Vec::with_capacity(n_hashes);
        let mut salt: u64 = 0;
        while experts.len() < n_hashes {
            let h = hash_token(self.config.seed, token_word, salt);
            let candidate = (h % n_experts as u64) as usize;
            if !experts.contains(&candidate) {
                experts.push(candidate);
            }
            salt = salt.wrapping_add(1);
        }
        experts
    }

    /// Route a batch of token ids to their expert ids.
    ///
    /// Returns one inner vector per input token, each of length `n_hashes`.
    ///
    /// # Errors
    /// * [`MoeError::EmptyInput`] if `token_ids` is empty.
    pub fn route(&self, token_ids: &[usize]) -> MoeResult<Vec<Vec<usize>>> {
        if token_ids.is_empty() {
            return Err(MoeError::EmptyInput);
        }
        Ok(token_ids
            .iter()
            .map(|&token_id| self.route_token(token_id))
            .collect())
    }

    /// Compute the per-expert token load (counts) for a batch of token ids.
    ///
    /// Each token contributes `+1` to every expert it is routed to, so with
    /// `n_hashes > 1` the total count is `token_ids.len() * n_hashes`. Returns a
    /// vector of length `n_experts`.
    ///
    /// # Errors
    /// * [`MoeError::EmptyInput`] if `token_ids` is empty.
    pub fn expert_load(&self, token_ids: &[usize]) -> MoeResult<Vec<usize>> {
        if token_ids.is_empty() {
            return Err(MoeError::EmptyInput);
        }
        let mut load = vec![0_usize; self.config.n_experts];
        for &token_id in token_ids {
            for expert in self.route_token(token_id) {
                // `route_token` guarantees `expert < n_experts`, so this index
                // is always in bounds; guard defensively all the same.
                if let Some(slot) = load.get_mut(expert) {
                    *slot += 1;
                }
            }
        }
        Ok(load)
    }

    /// Number of trainable parameters (always `0` — hash routing is parameter-free).
    #[must_use]
    pub fn param_count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(n_experts: usize, n_hashes: usize, seed: u64) -> HashRouter {
        HashRouter::new(HashRoutingConfig {
            n_experts,
            n_hashes,
            seed,
        })
        .expect("valid hash router config")
    }

    // --- Construction / validation ---

    #[test]
    fn default_config_has_single_hash() {
        let cfg = HashRoutingConfig::default();
        assert_eq!(cfg.n_hashes, 1);
        assert!(cfg.n_experts > 0);
    }

    #[test]
    fn new_zero_experts_errors() {
        let err = HashRouter::new(HashRoutingConfig {
            n_experts: 0,
            n_hashes: 1,
            seed: 0,
        });
        assert!(matches!(err, Err(MoeError::InvalidExpertCount { .. })));
    }

    #[test]
    fn new_zero_hashes_errors() {
        let err = HashRouter::new(HashRoutingConfig {
            n_experts: 8,
            n_hashes: 0,
            seed: 0,
        });
        assert!(matches!(err, Err(MoeError::InvalidTopK { .. })));
    }

    #[test]
    fn new_too_many_hashes_errors() {
        let err = HashRouter::new(HashRoutingConfig {
            n_experts: 4,
            n_hashes: 5,
            seed: 0,
        });
        assert!(matches!(err, Err(MoeError::InvalidTopK { .. })));
    }

    #[test]
    fn param_count_is_zero() {
        assert_eq!(router(8, 1, 0).param_count(), 0);
    }

    // --- Determinism / statelessness ---

    #[test]
    fn deterministic_same_id_same_expert() {
        let r = router(16, 1, 123);
        let a = r.route_token(42);
        let b = r.route_token(42);
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_across_router_instances() {
        let a = router(16, 1, 7).route_token(99);
        let b = router(16, 1, 7).route_token(99);
        assert_eq!(a, b);
    }

    #[test]
    fn order_independent_stateless() {
        // Routing token 5 must not depend on tokens routed before it.
        let r = router(8, 1, 55);
        let solo = r.route_token(5);

        let forward = r.route(&[1, 2, 3, 4, 5]).expect("route ok");
        let reverse = r.route(&[5, 4, 3, 2, 1]).expect("route ok");

        assert_eq!(forward[4], solo); // token 5 is last forward
        assert_eq!(reverse[0], solo); // token 5 is first reverse
        assert_eq!(forward[4], reverse[0]);
    }

    #[test]
    fn route_matches_route_token() {
        let r = router(12, 2, 314);
        let ids = [0_usize, 7, 42, 1000, 999_999];
        let batch = r.route(&ids).expect("route ok");
        for (slot, &id) in ids.iter().enumerate() {
            assert_eq!(batch[slot], r.route_token(id));
        }
    }

    // --- Range / shape ---

    #[test]
    fn all_experts_in_range() {
        let n_experts = 13_usize; // non-power-of-two to exercise the modulo
        let r = router(n_experts, 1, 2024);
        for id in 0..5000_usize {
            for &e in &r.route_token(id) {
                assert!(e < n_experts, "expert {e} out of range for id {id}");
            }
        }
    }

    #[test]
    fn n_hashes_one_yields_single_expert() {
        let r = router(8, 1, 0);
        for id in 0..200_usize {
            assert_eq!(r.route_token(id).len(), 1);
        }
    }

    #[test]
    fn n_hashes_distinct_experts() {
        let n_hashes = 3_usize;
        let r = router(8, n_hashes, 17);
        for id in 0..500_usize {
            let experts = r.route_token(id);
            assert_eq!(experts.len(), n_hashes);
            let mut sorted = experts.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), n_hashes, "experts not distinct for id {id}");
        }
    }

    #[test]
    fn n_hashes_equal_n_experts_is_permutation() {
        // With n_hashes == n_experts, every token must cover all experts.
        let n = 6_usize;
        let r = router(n, n, 88);
        for id in 0..100_usize {
            let mut experts = r.route_token(id);
            experts.sort_unstable();
            let expected: Vec<usize> = (0..n).collect();
            assert_eq!(experts, expected);
        }
    }

    // --- Different seed → different mapping (statistical) ---

    #[test]
    fn different_seed_changes_mapping() {
        let a = router(32, 1, 1);
        let b = router(32, 1, 2);
        let mut differ = 0_usize;
        let n = 1000_usize;
        for id in 0..n {
            if a.route_token(id) != b.route_token(id) {
                differ += 1;
            }
        }
        // With 32 experts, two independent maps should disagree on ~31/32 ids.
        // Require a comfortably loose lower bound to avoid flakiness.
        assert!(
            differ > n / 2,
            "seeds produced near-identical mappings: {differ}/{n} differ"
        );
    }

    // --- Load balance ---

    #[test]
    fn balanced_load_every_expert_used() {
        let n_experts = 16_usize;
        let r = router(n_experts, 1, 4242);
        let ids: Vec<usize> = (0..16_000).collect();
        let load = r.expert_load(&ids).expect("load ok");
        assert_eq!(load.len(), n_experts);
        assert!(load.iter().all(|&c| c > 0), "some expert got zero tokens");
        assert_eq!(load.iter().sum::<usize>(), ids.len());
    }

    #[test]
    fn balanced_load_max_min_ratio_bounded() {
        let n_experts = 8_usize;
        let r = router(n_experts, 1, 909);
        let ids: Vec<usize> = (0..40_000).collect();
        let load = r.expert_load(&ids).expect("load ok");
        let max = *load.iter().max().expect("non-empty load");
        let min = *load.iter().min().expect("non-empty load");
        assert!(min > 0);
        // A good avalanche hash over 40k ids / 8 buckets should be very even;
        // allow a generous 1.2x slack for finite-sample fluctuation.
        let ratio = max as f64 / min as f64;
        assert!(ratio < 1.2, "load imbalance too high: max/min = {ratio}");
    }

    #[test]
    fn multi_hash_load_counts_each_assignment() {
        let n_experts = 10_usize;
        let n_hashes = 3_usize;
        let r = router(n_experts, n_hashes, 31);
        let ids: Vec<usize> = (0..5000).collect();
        let load = r.expert_load(&ids).expect("load ok");
        assert_eq!(load.iter().sum::<usize>(), ids.len() * n_hashes);
        assert!(load.iter().all(|&c| c > 0));
    }

    // --- Error paths on the batch API ---

    #[test]
    fn route_empty_errors() {
        let r = router(8, 1, 0);
        assert!(matches!(r.route(&[]), Err(MoeError::EmptyInput)));
    }

    #[test]
    fn expert_load_empty_errors() {
        let r = router(8, 1, 0);
        assert!(matches!(r.expert_load(&[]), Err(MoeError::EmptyInput)));
    }

    // --- Distinct from learned routers: no softmax, raw modulo mapping ---

    #[test]
    fn mapping_is_raw_modulo_of_hash() {
        // Confirms there is no softmax/logit step: the expert is exactly
        // `hash % n_experts`, recomputed independently here.
        let seed = 555_u64;
        let n_experts = 7_usize;
        let r = router(n_experts, 1, seed);
        for id in [0_usize, 1, 2, 3, 100, 12_345] {
            let expected = (hash_token(seed, id as u64, 0) % n_experts as u64) as usize;
            assert_eq!(r.route_token(id), vec![expected]);
        }
    }

    #[test]
    fn splitmix64_finalize_avalanches() {
        // Flipping the lowest bit of the input should change many output bits;
        // a healthy avalanche flips roughly half of the 64.
        let a = splitmix64_finalize(0x0000_0000_0000_0000);
        let b = splitmix64_finalize(0x0000_0000_0000_0001);
        let changed = (a ^ b).count_ones();
        assert!(
            (16..=48).contains(&changed),
            "avalanche outside healthy band: {changed} bits changed"
        );
    }

    #[test]
    fn distinct_token_ids_can_share_expert_but_vary() {
        // Sanity: the map is not the identity / not constant. Over a small id
        // window with many experts we should see at least two distinct experts.
        let r = router(16, 1, 2718);
        let experts: Vec<usize> = (0..32_usize).map(|id| r.route_token(id)[0]).collect();
        let first = experts[0];
        assert!(
            experts.iter().any(|&e| e != first),
            "mapping appears constant"
        );
    }
}
