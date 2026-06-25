//! Cold-start handling via content-based fallback.
//!
//! Collaborative filtering fails for **cold** users and items that have no (or
//! too few) interactions: their latent factors are untrained, so their scores
//! are noise. The standard remedy is a *content-based* fallback that scores
//! items from side information (genre / brand / text embedding) and a hybrid
//! *switching* blend that defers to the content model exactly when the
//! collaborative signal is missing.
//!
//! References:
//! - Schein, Popescul, Ungar, Pennock, "Methods and Metrics for Cold-Start
//!   Recommendations", SIGIR 2002 (cold-start taxonomy).
//! - Burke, "Hybrid Recommender Systems: Survey and Experiments", UMUAI 2002
//!   (switching / weighted hybridisation).
//!
//! # Components
//!
//! * **Item–item content KNN.** Cosine similarity over the supplied item feature
//!   matrix, with optional L2 normalisation.
//! * **Content user profile.** The (interaction-weighted) mean of the feature
//!   vectors of the items a user has consumed; scoring a candidate is its cosine
//!   to the profile. This generalises to a *cold user* the moment they have a
//!   single interaction — no model retraining needed.
//! * **Switching hybrid.** `score = collab` for warm (user, item) pairs and
//!   `score = content` when either side is cold (interaction count below a
//!   threshold), with a smooth interpolation in between governed by a warmth
//!   weight.
//!
//! All maths is FP32 and deterministic; no randomness is used.

use crate::error::{RecsysError, RecsysResult};

/// L2 norm of a slice.
#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity of two equal-length vectors (`0` if either is the zero
/// vector).
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot / (na * nb)
}

/// Content-based recommender over a dense item-feature matrix.
#[derive(Debug, Clone)]
pub struct ContentRecommender {
    /// Number of items.
    n_items: usize,
    /// Feature dimension.
    feat_dim: usize,
    /// Row-major `[n_items × feat_dim]` (optionally L2-normalised) features.
    feats: Vec<f32>,
}

impl ContentRecommender {
    /// Build from a flat `[n_items × feat_dim]` feature matrix. When
    /// `normalize` is set, each row is L2-normalised so dot products become
    /// cosine similarities.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumItems`] if `n_items == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] if `feat_dim == 0`.
    /// - [`RecsysError::DimensionMismatch`] if `feats.len() != n_items · feat_dim`.
    pub fn new(
        n_items: usize,
        feat_dim: usize,
        feats: &[f32],
        normalize: bool,
    ) -> RecsysResult<Self> {
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if feat_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: feat_dim });
        }
        if feats.len() != n_items * feat_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: n_items * feat_dim,
                got: feats.len(),
            });
        }
        let mut owned = feats.to_vec();
        if normalize {
            for row in owned.chunks_mut(feat_dim) {
                let n = norm(row);
                if n > 1e-12 {
                    for x in row.iter_mut() {
                        *x /= n;
                    }
                }
            }
        }
        Ok(Self {
            n_items,
            feat_dim,
            feats: owned,
        })
    }

    /// Number of items.
    #[must_use]
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Feature dimension.
    #[must_use]
    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }

    /// Feature row for `item`.
    fn row(&self, item: usize) -> RecsysResult<&[f32]> {
        if item >= self.n_items {
            return Err(RecsysError::ItemOutOfBounds {
                idx: item,
                n: self.n_items,
            });
        }
        Ok(&self.feats[item * self.feat_dim..(item + 1) * self.feat_dim])
    }

    /// Cosine similarity between two items by id.
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] for an out-of-range id.
    pub fn item_similarity(&self, a: usize, b: usize) -> RecsysResult<f32> {
        Ok(cosine(self.row(a)?, self.row(b)?))
    }

    /// Top-`k` most similar items to `item` (excluding itself), highest first.
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] for an out-of-range id.
    /// - [`RecsysError::InvalidK`] if `k == 0`.
    pub fn similar_items(&self, item: usize, k: usize) -> RecsysResult<Vec<(usize, f32)>> {
        if k == 0 {
            return Err(RecsysError::InvalidK { k, n: self.n_items });
        }
        let q = self.row(item)?;
        let mut sims: Vec<(usize, f32)> = (0..self.n_items)
            .filter(|&j| j != item)
            .map(|j| {
                let s = cosine(q, &self.feats[j * self.feat_dim..(j + 1) * self.feat_dim]);
                (j, s)
            })
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        Ok(sims)
    }

    /// Build a content **user profile**: the interaction-weighted mean of the
    /// feature vectors of the user's history. `history` is a list of
    /// `(item, weight)` pairs (use weight `1.0` for an unweighted history).
    /// Returns the zero vector for an empty history (a fully cold user).
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] for an out-of-range item id.
    pub fn user_profile(&self, history: &[(usize, f32)]) -> RecsysResult<Vec<f32>> {
        let mut profile = vec![0.0_f32; self.feat_dim];
        let mut wsum = 0.0_f32;
        for &(item, w) in history {
            let row = self.row(item)?;
            for (p, &x) in profile.iter_mut().zip(row.iter()) {
                *p += w * x;
            }
            wsum += w;
        }
        if wsum.abs() > 1e-12 {
            for p in &mut profile {
                *p /= wsum;
            }
        }
        Ok(profile)
    }

    /// Score a candidate `item` for a user described by `history` as the cosine
    /// between the candidate features and the user's content profile.
    ///
    /// # Errors
    /// - [`RecsysError::ItemOutOfBounds`] for an out-of-range id.
    pub fn score_for_user(&self, history: &[(usize, f32)], item: usize) -> RecsysResult<f32> {
        let profile = self.user_profile(history)?;
        Ok(cosine(&profile, self.row(item)?))
    }

    /// Recommend the top-`k` items for a content profile, excluding the items
    /// already in the user's history.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidK`] if `k == 0`.
    /// - Propagates [`Self::user_profile`].
    pub fn recommend(&self, history: &[(usize, f32)], k: usize) -> RecsysResult<Vec<(usize, f32)>> {
        if k == 0 {
            return Err(RecsysError::InvalidK { k, n: self.n_items });
        }
        let profile = self.user_profile(history)?;
        let seen: std::collections::BTreeSet<usize> = history.iter().map(|&(i, _)| i).collect();
        let mut scored: Vec<(usize, f32)> = (0..self.n_items)
            .filter(|j| !seen.contains(j))
            .map(|j| {
                let s = cosine(
                    &profile,
                    &self.feats[j * self.feat_dim..(j + 1) * self.feat_dim],
                );
                (j, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}

/// Configuration for the switching hybrid that blends a collaborative score
/// with a content fallback based on interaction warmth.
#[derive(Debug, Clone)]
pub struct ColdStartConfig {
    /// Interaction count at/above which a user or item is fully *warm*
    /// (collaborative score trusted entirely).
    pub warm_threshold: usize,
}

impl Default for ColdStartConfig {
    fn default() -> Self {
        Self { warm_threshold: 5 }
    }
}

impl ColdStartConfig {
    /// Validate.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidConfig`] if `warm_threshold == 0`.
    pub fn validate(&self) -> RecsysResult<()> {
        if self.warm_threshold == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "warm_threshold must be >= 1".into(),
            });
        }
        Ok(())
    }

    /// Warmth weight `w ∈ [0, 1]` for an interaction count: linearly ramps from
    /// `0` (cold) to `1` at `warm_threshold`.
    #[must_use]
    pub fn warmth(&self, count: usize) -> f32 {
        (count as f32 / self.warm_threshold as f32).min(1.0)
    }
}

/// Switching/weighted hybrid recommender: collaborative score when warm,
/// content score when cold, smoothly interpolated by the minimum of the user
/// and item warmth.
#[derive(Debug, Clone)]
pub struct ColdStartHybrid {
    cfg: ColdStartConfig,
}

impl ColdStartHybrid {
    /// Build from a validated configuration.
    ///
    /// # Errors
    /// Propagates [`ColdStartConfig::validate`].
    pub fn new(cfg: ColdStartConfig) -> RecsysResult<Self> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Blend a collaborative score with a content score given the user's and
    /// item's interaction counts. The collaborative term receives weight
    /// `α = min(warmth(user_count), warmth(item_count))` and the content term
    /// `1 − α`, so the result is purely content-based when *either* side is fully
    /// cold and purely collaborative once both sides are warm.
    #[must_use]
    pub fn blend(
        &self,
        collab_score: f32,
        content_score: f32,
        user_count: usize,
        item_count: usize,
    ) -> f32 {
        let alpha = self.cfg.warmth(user_count).min(self.cfg.warmth(item_count));
        alpha * collab_score + (1.0 - alpha) * content_score
    }

    /// The warmth weight applied to the collaborative term for a `(user, item)`
    /// pair (exposed for inspection / testing).
    #[must_use]
    pub fn collab_weight(&self, user_count: usize, item_count: usize) -> f32 {
        self.cfg.warmth(user_count).min(self.cfg.warmth(item_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn toy() -> ContentRecommender {
        // 4 items in 3-d feature space: items 0 & 1 nearly identical, 2 & 3 differ.
        let feats = vec![
            1.0, 0.0, 0.0, // item 0
            0.9, 0.1, 0.0, // item 1 (close to 0)
            0.0, 1.0, 0.0, // item 2
            0.0, 0.0, 1.0, // item 3
        ];
        ContentRecommender::new(4, 3, &feats, true).expect("ok")
    }

    #[test]
    fn rejects_bad_shape() {
        assert!(ContentRecommender::new(0, 3, &[], true).is_err());
        assert!(ContentRecommender::new(2, 0, &[], true).is_err());
        assert!(ContentRecommender::new(2, 3, &[1.0, 2.0], true).is_err());
    }

    #[test]
    fn similar_items_ranks_nearest_first() {
        let cr = toy();
        let sims = cr.similar_items(0, 3).expect("sims");
        assert_eq!(sims[0].0, 1, "item 1 is closest to item 0");
        // similarity to item 1 must exceed similarity to 2 and 3.
        assert!(sims[0].1 > sims[1].1);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn user_profile_and_recommend_for_cold_user() {
        let cr = toy();
        // A "cold" user with a single interaction on item 0: content recsys
        // should still recommend its nearest neighbour, item 1, first.
        let history = vec![(0usize, 1.0_f32)];
        let recs = cr.recommend(&history, 3).expect("recs");
        assert_eq!(recs[0].0, 1, "nearest unseen item should rank first");
        assert!(!recs.iter().any(|&(i, _)| i == 0), "history excluded");
    }

    #[test]
    fn empty_history_gives_zero_profile() {
        let cr = toy();
        let profile = cr.user_profile(&[]).expect("profile");
        assert!(profile.iter().all(|&x| x == 0.0));
        // Scoring against a zero profile yields cosine 0 everywhere.
        let s = cr.score_for_user(&[], 2).expect("score");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn weighted_profile_tracks_dominant_interaction() {
        let cr = toy();
        // Heavily weight item 2 vs lightly item 3 ⇒ profile closer to item 2.
        let history = vec![(2usize, 5.0_f32), (3usize, 0.1)];
        let s2 = cr.score_for_user(&history, 2).expect("s2");
        let s3 = cr.score_for_user(&history, 3).expect("s3");
        assert!(s2 > s3, "dominant interaction should win: {s2} vs {s3}");
    }

    #[test]
    fn hybrid_switches_on_warmth() {
        let hybrid = ColdStartHybrid::new(ColdStartConfig { warm_threshold: 4 }).expect("ok");
        // Fully cold item (count 0) ⇒ pure content score regardless of user.
        let cold = hybrid.blend(10.0, -3.0, 100, 0);
        assert!(
            (cold + 3.0).abs() < 1e-6,
            "cold item ⇒ content score, got {cold}"
        );
        // Both warm ⇒ pure collaborative score.
        let warm = hybrid.blend(10.0, -3.0, 4, 8);
        assert!(
            (warm - 10.0).abs() < 1e-6,
            "warm ⇒ collab score, got {warm}"
        );
        // Half-warm interpolates.
        let half = hybrid.blend(10.0, 0.0, 2, 100);
        assert!(
            (half - 5.0).abs() < 1e-6,
            "half warmth ⇒ midpoint, got {half}"
        );
    }

    #[test]
    fn hybrid_rejects_zero_threshold() {
        assert!(ColdStartHybrid::new(ColdStartConfig { warm_threshold: 0 }).is_err());
    }

    #[test]
    fn random_features_normalize_to_unit_norm() {
        let mut rng = LcgRng::new(31);
        let n = 20usize;
        let d = 6usize;
        let feats: Vec<f32> = (0..n * d).map(|_| rng.next_normal()).collect();
        let cr = ContentRecommender::new(n, d, &feats, true).expect("ok");
        // Self-similarity of a normalised row must be ≈ 1.
        for i in 0..n {
            let s = cr.item_similarity(i, i).expect("sim");
            assert!((s - 1.0).abs() < 1e-5, "row {i} self-sim {s}");
        }
    }
}
