//! Multi-pass online HD classifier with an explicit *forgetting factor* (recency-weighted
//! centroids / exponentially-weighted prototype learning).
//!
//! This is a *streaming* hyperdimensional classifier for **non-stationary** data: when the
//! statistics of a class drift over time (concept drift), evidence collected long ago should
//! count for less than evidence collected recently. It generalises the single-pass HD model
//! in [`crate::classifier::hd_classifier`] (integer majority-vote accumulators, no decay) and
//! the AdaptHD retraining model in [`crate::learning::adaptive_hd`] (error-corrective updates,
//! no decay) by keeping **real-valued** per-class accumulators that *decay geometrically* on
//! every update.
//!
//! # Forgetting / EWMA update
//!
//! Each class `c` owns a real prototype vector `μ_c ∈ ℝ^D`. When a new binary hypervector
//! `x ∈ {−1,+1}^D` arrives labelled with class `c`, **only that class** is updated with an
//! exponentially-weighted moving average (EWMA):
//!
//! ```text
//! μ_c ← λ · μ_c + (1 − λ) · x        with forgetting factor  λ ∈ (0, 1)
//! ```
//!
//! Unrolling `t` consecutive updates `x_1, …, x_t` to class `c` (starting from `μ_c = 0`)
//! gives the recency-weighted centroid
//!
//! ```text
//! μ_c = (1 − λ) · Σ_{k=1}^{t} λ^{t−k} · x_k
//! ```
//!
//! so the most recent sample `x_t` carries weight `(1 − λ)`, the one before it `(1 − λ)·λ`,
//! and a sample seen `n` steps ago is down-weighted by `λ^n`. The *effective memory* spans
//! roughly `1 / (1 − λ)` samples. `λ → 1` would mean "never forget" (and reduces to a scaled
//! running sum), while small `λ` reacts almost instantly to the newest sample; we therefore
//! require `0 < λ < 1` strictly. Because a single `update` touches only the labelled class,
//! the decay is applied *per class on its own timeline* — exactly the recency-weighted centroid
//! behaviour that lets old class definitions fade when a stream's concept changes.
//!
//! # Classification
//!
//! A query `q ∈ {−1,+1}^D` is assigned the class whose real prototype maximises cosine
//! similarity. For a `±1` query `‖q‖ = √D`, so
//!
//! ```text
//! cos(μ_c, q) = ⟨μ_c, q⟩ / (‖μ_c‖ · √D)
//! ```
//!
//! is computed inline (with a guard for an all-zero / never-trained prototype, whose
//! similarity is taken as `0`) and `argmax_c` is returned, ties broken toward the lowest index.
//!
//! # References
//!
//! * A. Hernández-Cano et al., *OnlineHD: Robust, Efficient, and Single-Pass Online Learning
//!   Using Hyperdimensional System*, DATE 2021 — single-pass streaming prototypes.
//! * M. Imani et al., *AdaptHD: Adaptive Efficient Training for Brain-Inspired
//!   Hyperdimensional Computing*, BioCAS/DATE 2019 — iterative prototype refinement; here
//!   extended with an exponential forgetting factor.
//! * S. Roberts, *Control Chart Tests Based on Geometric Moving Averages*, Technometrics 1959
//!   — the EWMA / exponential-forgetting recursion `μ ← λμ + (1−λ)x`.

use crate::error::{HdcError, HdcResult};

/// Configuration for a [`ForgettingHdClassifier`].
///
/// The forgetting factor [`lambda`](ForgettingHdConfig::lambda) is the core knob: it sets how
/// fast old evidence decays in the per-class EWMA prototype update
/// `μ_c ← λ·μ_c + (1−λ)·x`.
#[derive(Debug, Clone)]
pub struct ForgettingHdConfig {
    /// Number of classes (must be `≥ 1`).
    pub n_classes: usize,
    /// Hypervector dimension `D` (must be `≥ 1`).
    pub dim: usize,
    /// Forgetting factor `λ`, strictly inside `(0, 1)`. Larger `λ` remembers longer
    /// (effective memory `≈ 1 / (1 − λ)` samples); `λ → 1` would never forget and is
    /// disallowed, as is `λ ≤ 0`.
    pub lambda: f32,
    /// Number of passes over a training batch performed by [`ForgettingHdClassifier::fit`]
    /// (must be `≥ 1`).
    pub epochs: usize,
}

impl Default for ForgettingHdConfig {
    /// A two-class, 10 000-dimensional configuration with `λ = 0.9` and a single pass — a
    /// reasonable streaming default that remembers on the order of ten recent samples.
    fn default() -> Self {
        Self {
            n_classes: 2,
            dim: 10_000,
            lambda: 0.9,
            epochs: 1,
        }
    }
}

impl ForgettingHdConfig {
    /// Build a validated configuration.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `n_classes == 0`.
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::InvalidProbability`] if `lambda` is not finite or not strictly inside
    ///   `(0, 1)` (the value is reported back in the error).
    /// - [`HdcError::EmptyInput`] if `epochs == 0`.
    pub fn new(n_classes: usize, dim: usize, lambda: f32, epochs: usize) -> HdcResult<Self> {
        if n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if !lambda.is_finite() || lambda <= 0.0 || lambda >= 1.0 {
            return Err(HdcError::InvalidProbability(lambda as f64));
        }
        if epochs == 0 {
            return Err(HdcError::EmptyInput);
        }
        Ok(Self {
            n_classes,
            dim,
            lambda,
            epochs,
        })
    }
}

/// Streaming HD classifier with real-valued, exponentially-decaying class prototypes.
///
/// Each class keeps an `f32` accumulator that is updated by the EWMA rule
/// `μ_c ← λ·μ_c + (1−λ)·x` so that recent evidence dominates. Classification is argmax cosine
/// over those real prototypes against a `±1` query. See the [module documentation](self) for
/// the full forgetting math.
pub struct ForgettingHdClassifier {
    /// Validated configuration (`n_classes`, `dim`, `λ`, `epochs`).
    cfg: ForgettingHdConfig,
    /// Real-valued per-class prototype accumulators (`n_classes` rows, each length `dim`).
    accumulators: Vec<Vec<f32>>,
    /// Total EWMA weight applied to each class so far, `Σ (1−λ)·λ^{t−k}`. Starts at `0` and
    /// approaches `1` as a class receives many updates; `0` flags a never-trained class.
    total_weight: Vec<f32>,
}

impl ForgettingHdClassifier {
    /// Create a classifier with zero-initialised prototypes.
    ///
    /// # Errors
    ///
    /// Never fails for an already-validated [`ForgettingHdConfig`]; the `Result` is kept for
    /// API symmetry and forward compatibility.
    pub fn new(cfg: ForgettingHdConfig) -> HdcResult<Self> {
        let accumulators = vec![vec![0f32; cfg.dim]; cfg.n_classes];
        let total_weight = vec![0f32; cfg.n_classes];
        Ok(Self {
            cfg,
            accumulators,
            total_weight,
        })
    }

    /// Apply one EWMA / forgetting update of `class` toward the binary hypervector `hv`.
    ///
    /// The labelled class decays and absorbs the new sample,
    /// `μ_c[i] ← λ·μ_c[i] + (1 − λ)·hv[i]`, so its older contents are down-weighted by `λ` and
    /// the freshest sample enters with weight `(1 − λ)`. **No other class is touched** — each
    /// prototype decays only on its own update timeline, which is what makes these
    /// recency-weighted centroids. The running `total_weight` is updated
    /// with the same recursion.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ClassNotFound`] if `class >= n_classes`.
    /// - [`HdcError::DimensionMismatch`] if `hv.len() != dim`.
    pub fn update(&mut self, class: usize, hv: &[i8]) -> HdcResult<()> {
        if class >= self.cfg.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        if hv.len() != self.cfg.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.dim,
                got: hv.len(),
            });
        }
        let lambda = self.cfg.lambda;
        let gain = 1.0 - lambda;
        let acc = &mut self.accumulators[class];
        for (slot, &bit) in acc.iter_mut().zip(hv.iter()) {
            *slot = lambda * *slot + gain * bit as f32;
        }
        // Same EWMA recursion on the scalar weight: w ← λ·w + (1−λ).
        self.total_weight[class] = lambda * self.total_weight[class] + gain;
        Ok(())
    }

    /// Cosine similarity between class `c`'s real prototype and a `±1` query, computed inline.
    ///
    /// For a binary query `‖q‖ = √D`, so the similarity is `⟨μ_c, q⟩ / (‖μ_c‖ · √D)`. A
    /// never-trained (all-zero) prototype has undefined direction and yields `0.0` via the
    /// division-by-zero guard. The caller guarantees `query.len() == dim`.
    fn cosine_to_class(&self, query: &[i8], c: usize) -> f32 {
        let proto = &self.accumulators[c];
        let mut dot = 0f64;
        let mut sq = 0f64;
        for (&p, &q) in proto.iter().zip(query.iter()) {
            let pv = p as f64;
            dot += pv * q as f64;
            sq += pv * pv;
        }
        let norm = sq.sqrt();
        let denom = norm * (self.cfg.dim as f64).sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            (dot / denom) as f32
        }
    }

    /// Classify `query` as the class of maximum cosine similarity over the real prototypes.
    ///
    /// Ties (including the degenerate case where every prototype is empty and all similarities
    /// are `0`) resolve to the lowest class index — so an untrained classifier deterministically
    /// returns class `0`. See `cosine_to_class` for the metric.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    pub fn classify(&self, query: &[i8]) -> HdcResult<usize> {
        if query.len() != self.cfg.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        let mut best = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for c in 0..self.cfg.n_classes {
            let sim = self.cosine_to_class(query, c);
            if sim > best_sim {
                best_sim = sim;
                best = c;
            }
        }
        Ok(best)
    }

    /// Cosine similarity of `query` to a specific class prototype.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn class_similarity(&self, query: &[i8], class: usize) -> HdcResult<f32> {
        if query.len() != self.cfg.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        if class >= self.cfg.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(self.cosine_to_class(query, class))
    }

    /// Multi-pass online training: run `cfg.epochs` passes over `(hvs, labels)`, calling
    /// [`update`](Self::update) for every `(hv, label)` pair in order.
    ///
    /// Because each pass re-applies the forgetting update, later epochs (and, within an epoch,
    /// later samples) dominate the resulting prototypes — the deliberate recency bias of this
    /// model. To learn a *stationary* batch, present samples in a fixed order and use enough
    /// epochs that every class is revisited after the others.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `hvs.len() != labels.len()`.
    /// - [`HdcError::EmptyInput`] if `hvs` is empty.
    /// - [`HdcError::ClassNotFound`] if any label is `>= n_classes`.
    /// - [`HdcError::DimensionMismatch`] if any HV has the wrong dimension.
    pub fn fit(&mut self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<()> {
        if hvs.len() != labels.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: labels.len(),
            });
        }
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        // Validate the whole batch up front so a bad sample fails before any mutation.
        for (hv, &label) in hvs.iter().zip(labels.iter()) {
            if label >= self.cfg.n_classes {
                return Err(HdcError::ClassNotFound(label));
            }
            if hv.len() != self.cfg.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: self.cfg.dim,
                    got: hv.len(),
                });
            }
        }
        for _ in 0..self.cfg.epochs {
            for (hv, &label) in hvs.iter().zip(labels.iter()) {
                self.update(label, hv)?;
            }
        }
        Ok(())
    }

    /// Classification accuracy on a labelled evaluation set, in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `hvs` is empty.
    /// - [`HdcError::DimensionMismatch`] if `hvs.len() != labels.len()`.
    /// - Propagates errors from [`classify`](Self::classify).
    pub fn accuracy(&self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<f32> {
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if hvs.len() != labels.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: labels.len(),
            });
        }
        let mut correct = 0usize;
        for (hv, &label) in hvs.iter().zip(labels.iter()) {
            if self.classify(hv)? == label {
                correct += 1;
            }
        }
        Ok(correct as f32 / hvs.len() as f32)
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.cfg.n_classes
    }

    /// Hypervector dimension `D`.
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Forgetting factor `λ`.
    pub fn lambda(&self) -> f32 {
        self.cfg.lambda
    }

    /// Borrow the real-valued prototype accumulator of `class`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn prototype(&self, class: usize) -> HdcResult<&[f32]> {
        if class >= self.cfg.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(&self.accumulators[class])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    /// Build a noisy copy of `centroid` by flipping each `±1` bit with probability `flip_frac`.
    fn noisy_copy(centroid: &[i8], flip_frac: f32, rng: &mut LcgRng) -> Vec<i8> {
        let mut sample = centroid.to_vec();
        for slot in sample.iter_mut() {
            if rng.next_f32() < flip_frac {
                *slot = -*slot;
            }
        }
        sample
    }

    #[test]
    fn config_rejects_zero_classes() {
        assert!(matches!(
            ForgettingHdConfig::new(0, 512, 0.8, 1),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn config_rejects_zero_dim() {
        assert!(matches!(
            ForgettingHdConfig::new(2, 0, 0.8, 1),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn config_rejects_lambda_out_of_range() {
        // λ ≤ 0
        assert!(matches!(
            ForgettingHdConfig::new(2, 512, 0.0, 1),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            ForgettingHdConfig::new(2, 512, -0.1, 1),
            Err(HdcError::InvalidProbability(_))
        ));
        // λ ≥ 1
        assert!(matches!(
            ForgettingHdConfig::new(2, 512, 1.0, 1),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            ForgettingHdConfig::new(2, 512, 1.5, 1),
            Err(HdcError::InvalidProbability(_))
        ));
        // non-finite λ
        assert!(matches!(
            ForgettingHdConfig::new(2, 512, f32::NAN, 1),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    #[test]
    fn config_rejects_zero_epochs() {
        assert!(matches!(
            ForgettingHdConfig::new(2, 512, 0.8, 0),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn config_accepts_valid() {
        let cfg = ForgettingHdConfig::new(3, 1024, 0.85, 4).expect("valid config");
        assert_eq!(cfg.n_classes, 3);
        assert_eq!(cfg.dim, 1024);
        assert!((cfg.lambda - 0.85).abs() < 1e-6);
        assert_eq!(cfg.epochs, 4);
    }

    #[test]
    fn two_class_separable_training_classifies_originals() {
        // Two distinct random prototypes; train on noisy copies; the clean originals must
        // classify to their own class.
        let mut rng = LcgRng::new(11);
        let dim = 2048;
        let p0 = random_binary(dim, &mut rng).expect("p0");
        let p1 = random_binary(dim, &mut rng).expect("p1");
        let cfg = ForgettingHdConfig::new(2, dim, 0.8, 5).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");

        // Interleave classes so each is revisited after the other across the multi-pass fit.
        let mut hvs = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..20 {
            hvs.push(noisy_copy(&p0, 0.08, &mut rng));
            labels.push(0usize);
            hvs.push(noisy_copy(&p1, 0.08, &mut rng));
            labels.push(1usize);
        }
        clf.fit(&hvs, &labels).expect("fit");

        assert_eq!(clf.classify(&p0).expect("classify p0"), 0);
        assert_eq!(clf.classify(&p1).expect("classify p1"), 1);
    }

    #[test]
    fn forgetting_makes_recent_prototype_win() {
        // Core λ test: train class 0 on prototype P0, then stream MANY class-0 updates of a
        // DIFFERENT prototype P0'. Exponential forgetting must decay the old P0 evidence so
        // that the recent P0' becomes the closer match.
        let mut rng = LcgRng::new(23);
        let dim = 2048;
        let p_old = random_binary(dim, &mut rng).expect("p_old");
        let p_new = random_binary(dim, &mut rng).expect("p_new");

        let cfg = ForgettingHdConfig::new(2, dim, 0.7, 1).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");

        // Establish the old concept for class 0.
        for _ in 0..30 {
            clf.update(0, &p_old).expect("update old");
        }
        let sim_old_before = clf.class_similarity(&p_old, 0).expect("sim old before");
        assert!(
            sim_old_before > 0.5,
            "class 0 should match the old concept first: {sim_old_before}"
        );

        // Drift: feed many recent updates of the NEW concept.
        for _ in 0..60 {
            clf.update(0, &p_new).expect("update new");
        }

        // Recency: the prototype is now closer to P0' than to the faded P0.
        let sim_new = clf.class_similarity(&p_new, 0).expect("sim new");
        let sim_old_after = clf.class_similarity(&p_old, 0).expect("sim old after");
        assert!(
            sim_new > sim_old_after,
            "forgetting failed: sim_new={sim_new} should exceed sim_old_after={sim_old_after}"
        );
        // And the old evidence genuinely decayed versus its pre-drift value.
        assert!(
            sim_old_after < sim_old_before,
            "old concept did not fade: before={sim_old_before} after={sim_old_after}"
        );
    }

    #[test]
    fn fit_length_mismatch_errors() {
        let mut rng = LcgRng::new(31);
        let dim = 512;
        let cfg = ForgettingHdConfig::new(2, dim, 0.8, 2).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");
        let hvs = vec![
            random_binary(dim, &mut rng).expect("hv0"),
            random_binary(dim, &mut rng).expect("hv1"),
        ];
        let labels = vec![0usize]; // shorter than hvs
        assert!(matches!(
            clf.fit(&hvs, &labels),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_empty_errors() {
        let dim = 512;
        let cfg = ForgettingHdConfig::new(2, dim, 0.8, 2).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");
        assert!(matches!(clf.fit(&[], &[]), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn classify_dim_mismatch_errors() {
        let mut rng = LcgRng::new(41);
        let dim = 512;
        let cfg = ForgettingHdConfig::new(2, dim, 0.8, 2).expect("cfg");
        let clf = ForgettingHdClassifier::new(cfg).expect("new");
        let bad = random_binary(256, &mut rng).expect("bad");
        assert!(matches!(
            clf.classify(&bad),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn update_class_out_of_range_errors() {
        let mut rng = LcgRng::new(47);
        let dim = 512;
        let cfg = ForgettingHdConfig::new(2, dim, 0.8, 2).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");
        let hv = random_binary(dim, &mut rng).expect("hv");
        assert!(matches!(
            clf.update(5, &hv),
            Err(HdcError::ClassNotFound(5))
        ));
    }

    #[test]
    fn update_dim_mismatch_errors() {
        let mut rng = LcgRng::new(53);
        let dim = 512;
        let cfg = ForgettingHdConfig::new(2, dim, 0.8, 2).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");
        let bad = random_binary(128, &mut rng).expect("bad");
        assert!(matches!(
            clf.update(0, &bad),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        // Two independent runs with the same seed must yield byte-identical prototypes.
        fn run(seed: u64) -> Vec<f32> {
            let mut rng = LcgRng::new(seed);
            let dim = 1024;
            let p0 = random_binary(dim, &mut rng).expect("p0");
            let p1 = random_binary(dim, &mut rng).expect("p1");
            let cfg = ForgettingHdConfig::new(2, dim, 0.75, 3).expect("cfg");
            let mut clf = ForgettingHdClassifier::new(cfg).expect("new");
            let mut hvs = Vec::new();
            let mut labels = Vec::new();
            for _ in 0..10 {
                hvs.push(noisy_copy(&p0, 0.1, &mut rng));
                labels.push(0usize);
                hvs.push(noisy_copy(&p1, 0.1, &mut rng));
                labels.push(1usize);
            }
            clf.fit(&hvs, &labels).expect("fit");
            clf.prototype(0).expect("proto").to_vec()
        }
        let a = run(99);
        let b = run(99);
        assert_eq!(a, b);
    }

    #[test]
    fn accuracy_on_separable_set_is_one() {
        // Trivially separable: clean centroids only, classify the very samples trained on.
        let mut rng = LcgRng::new(61);
        let dim = 1024;
        let p0 = random_binary(dim, &mut rng).expect("p0");
        let p1 = random_binary(dim, &mut rng).expect("p1");
        let cfg = ForgettingHdConfig::new(2, dim, 0.85, 6).expect("cfg");
        let mut clf = ForgettingHdClassifier::new(cfg).expect("new");

        let hvs = vec![p0.clone(), p1.clone()];
        let labels = vec![0usize, 1usize];
        clf.fit(&hvs, &labels).expect("fit");

        let acc = clf.accuracy(&hvs, &labels).expect("accuracy");
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "accuracy should be 1.0, got {acc}"
        );
    }

    #[test]
    fn accessors_report_config() {
        let cfg = ForgettingHdConfig::new(4, 768, 0.9, 2).expect("cfg");
        let clf = ForgettingHdClassifier::new(cfg).expect("new");
        assert_eq!(clf.n_classes(), 4);
        assert_eq!(clf.dim(), 768);
        assert!((clf.lambda() - 0.9).abs() < 1e-6);
        assert_eq!(clf.prototype(0).expect("proto").len(), 768);
        assert!(matches!(clf.prototype(9), Err(HdcError::ClassNotFound(9))));
    }
}
