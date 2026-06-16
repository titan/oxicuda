//! Random-subspace bagging ensemble of HD classifiers.
//!
//! A single prototype-based HD classifier ([`crate::classifier::hd_classifier::HdClassifier`])
//! builds one prototype hypervector per class from the components it is shown. Its decision
//! boundary therefore depends on the particular random basis / coordinate set that happened to
//! carry the discriminative signal. This module reduces that variance with an ensemble of `M`
//! classifiers, each trained on an **independent random subspace** of the hypervector
//! coordinates (the random-subspace / attribute-bagging method of Ho 1998 applied to VSA), and
//! aggregates their decisions by majority vote with a summed-similarity tie-break.
//!
//! Because each member projects every hypervector onto a different (seed-determined) subset of
//! the `D` dimensions, the members' prototypes are built from decorrelated views of the data;
//! averaging their votes lowers the variance of the prediction relative to any one member. A
//! member that retains *all* `D` dimensions degenerates to the plain single classifier, so an
//! ensemble of one member with `feature_fraction == 1.0` reproduces a single
//! [`crate::classifier::hd_classifier::HdClassifier`] exactly.

use crate::classifier::hd_classifier::HdClassifier;
use crate::distance::cosine::cosine_binary;
use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::permutation::random_permutation;

/// Salt mixed into a member seed before deriving its random subspace, keeping the subspace RNG
/// independent of the (separately seeded) prototype-build RNG.
const SUBSPACE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Configuration for an [`HdEnsemble`].
#[derive(Debug, Clone)]
pub struct HdEnsembleConfig {
    /// Number of ensemble members (independent HD classifiers), must be ≥ 1.
    pub n_members: usize,
    /// Number of classes, must be ≥ 1.
    pub n_classes: usize,
    /// Hypervector dimension of the inputs, must be ≥ 1.
    pub dim: usize,
    /// Fraction of the `dim` coordinates each member retains, in `(0, 1]`.
    pub feature_fraction: f32,
    /// Base RNG seed; member `m` derives its subspace and tie-break RNGs from `seed + m`.
    pub seed: u64,
}

impl HdEnsembleConfig {
    /// Create a configuration with sensible defaults (`feature_fraction = 0.5`, `seed = 0`).
    #[must_use]
    pub fn new(n_members: usize, n_classes: usize, dim: usize) -> Self {
        Self {
            n_members,
            n_classes,
            dim,
            feature_fraction: 0.5,
            seed: 0,
        }
    }

    /// Set the per-member feature (coordinate) retention fraction.
    #[must_use]
    pub fn with_feature_fraction(mut self, feature_fraction: f32) -> Self {
        self.feature_fraction = feature_fraction;
        self
    }

    /// Set the base RNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// One ensemble member: a coordinate subspace plus the classifier trained on it.
struct EnsembleMember {
    /// Sorted indices of the retained coordinates (length = projected dimension).
    subset: Vec<usize>,
    /// Per-class prototype classifier operating in the projected subspace.
    classifier: HdClassifier,
    /// Deterministic seed for this member's prototype-build tie-breaks.
    build_seed: u64,
}

/// Random-subspace bagging ensemble of prototype HD classifiers.
pub struct HdEnsemble {
    config: HdEnsembleConfig,
    members: Vec<EnsembleMember>,
    trained: bool,
}

/// Project a full-dimension hypervector onto the coordinates listed in `subset`.
fn project(sample: &[i8], subset: &[usize]) -> Vec<i8> {
    subset.iter().map(|&d| sample[d]).collect()
}

impl HdEnsemble {
    /// Build an untrained ensemble from a configuration.
    ///
    /// Each member's coordinate subspace is fixed here (deterministically from the seed); call
    /// [`Self::train`] before predicting.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `n_members == 0` or `n_classes == 0`.
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::InvalidProbability`] if `feature_fraction` is not in `(0, 1]`.
    pub fn new(config: HdEnsembleConfig) -> HdcResult<Self> {
        if config.n_members == 0 || config.n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if config.dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if !(config.feature_fraction > 0.0 && config.feature_fraction <= 1.0) {
            return Err(HdcError::InvalidProbability(config.feature_fraction as f64));
        }

        let dim = config.dim;
        let k = ((config.feature_fraction * dim as f32).round() as usize).clamp(1, dim);

        let mut members = Vec::with_capacity(config.n_members);
        for m in 0..config.n_members {
            let build_seed = config.seed.wrapping_add(m as u64);
            let subset = if k >= dim {
                // Full basis: identity projection (keeps member ≡ plain classifier).
                (0..dim).collect()
            } else {
                let mut subset_rng = LcgRng::new(build_seed ^ SUBSPACE_SALT);
                let perm = random_permutation(dim, &mut subset_rng)?;
                let mut chosen: Vec<usize> = perm[..k].to_vec();
                chosen.sort_unstable();
                chosen
            };
            let classifier = HdClassifier::new(config.n_classes, subset.len())?;
            members.push(EnsembleMember {
                subset,
                classifier,
                build_seed,
            });
        }

        Ok(Self {
            config,
            members,
            trained: false,
        })
    }

    /// Number of ensemble members.
    #[must_use]
    pub fn n_members(&self) -> usize {
        self.members.len()
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.config.n_classes
    }

    /// Train every member on its coordinate subspace.
    ///
    /// `samples[i]` is a full-dimension hypervector with label `labels[i]`. Calling `train`
    /// again retrains from scratch (member accumulators are reset first), so the call is
    /// idempotent for a fixed dataset.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `samples` is empty.
    /// - [`HdcError::DimensionMismatch`] if `samples.len() != labels.len()` or any sample's
    ///   length differs from the configured `dim`.
    /// - [`HdcError::ClassNotFound`] if any label is `>= n_classes`.
    pub fn train(&mut self, samples: &[Vec<i8>], labels: &[usize]) -> HdcResult<()> {
        if samples.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if samples.len() != labels.len() {
            return Err(HdcError::DimensionMismatch {
                expected: samples.len(),
                got: labels.len(),
            });
        }
        for s in samples {
            if s.len() != self.config.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: self.config.dim,
                    got: s.len(),
                });
            }
        }
        for &label in labels {
            if label >= self.config.n_classes {
                return Err(HdcError::ClassNotFound(label));
            }
        }

        let n_classes = self.config.n_classes;
        for member in &mut self.members {
            // Reset to a fresh classifier so repeated training does not double-count.
            member.classifier = HdClassifier::new(n_classes, member.subset.len())?;
            for (sample, &label) in samples.iter().zip(labels.iter()) {
                let proj = project(sample, &member.subset);
                member.classifier.add_example(label, &proj)?;
            }
            let mut build_rng = LcgRng::new(member.build_seed);
            member.classifier.build_prototypes(&mut build_rng)?;
        }
        self.trained = true;
        Ok(())
    }

    /// Per-class cosine similarities of `sample` to one member's prototypes.
    fn member_sims(&self, member: &EnsembleMember, sample: &[i8]) -> HdcResult<Vec<f32>> {
        let proj = project(sample, &member.subset);
        let mut sims = Vec::with_capacity(self.config.n_classes);
        for c in 0..self.config.n_classes {
            let proto = member.classifier.prototype(c)?;
            sims.push(cosine_binary(&proj, proto)?);
        }
        Ok(sims)
    }

    /// Aggregate member decisions into `(vote_counts, summed_similarities)` per class.
    fn aggregate(&self, sample: &[i8]) -> HdcResult<(Vec<usize>, Vec<f32>)> {
        if !self.trained {
            return Err(HdcError::PrototypeNotBuilt);
        }
        if sample.len() != self.config.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.config.dim,
                got: sample.len(),
            });
        }
        let n_classes = self.config.n_classes;
        let mut votes = vec![0usize; n_classes];
        let mut scores = vec![0f32; n_classes];
        for member in &self.members {
            let sims = self.member_sims(member, sample)?;
            // Member's vote = argmax similarity (lowest index on ties).
            let mut best_c = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (c, &s) in sims.iter().enumerate() {
                if s > best {
                    best = s;
                    best_c = c;
                }
            }
            votes[best_c] += 1;
            for (acc, &s) in scores.iter_mut().zip(sims.iter()) {
                *acc += s;
            }
        }
        Ok((votes, scores))
    }

    /// Predict the class of `sample` by majority vote across members.
    ///
    /// Ties in the vote count are broken by the higher summed cosine similarity (and, failing
    /// that, the lower class index).
    ///
    /// # Errors
    ///
    /// - [`HdcError::PrototypeNotBuilt`] if [`Self::train`] has not been called.
    /// - [`HdcError::DimensionMismatch`] if `sample.len()` differs from the configured `dim`.
    pub fn predict(&self, sample: &[i8]) -> HdcResult<usize> {
        let (votes, scores) = self.aggregate(sample)?;
        let mut best_c = 0usize;
        let mut best_votes = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.config.n_classes {
            let better =
                votes[c] > best_votes || (votes[c] == best_votes && scores[c] > best_score);
            if better {
                best_votes = votes[c];
                best_score = scores[c];
                best_c = c;
            }
        }
        Ok(best_c)
    }

    /// Return the per-class summed cosine similarity across all members.
    ///
    /// Higher scores indicate greater confidence; this is the soft analogue of [`Self::predict`]
    /// and uses the same aggregation.
    ///
    /// # Errors
    ///
    /// - [`HdcError::PrototypeNotBuilt`] if [`Self::train`] has not been called.
    /// - [`HdcError::DimensionMismatch`] if `sample.len()` differs from the configured `dim`.
    pub fn predict_scores(&self, sample: &[i8]) -> HdcResult<Vec<f32>> {
        let (_, scores) = self.aggregate(sample)?;
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::binary::random_binary;

    /// Copy `base`, flipping `n_flips` deterministic positions (a noisy class exemplar).
    fn noisy(base: &[i8], n_flips: usize, rng: &mut LcgRng) -> Vec<i8> {
        let mut v = base.to_vec();
        for _ in 0..n_flips {
            let idx = rng.next_usize(v.len());
            v[idx] = -v[idx];
        }
        v
    }

    /// Build a 2-class, linearly separable toy set of binary hypervectors.
    fn toy_dataset(dim: usize, per_class: usize, seed: u64) -> (Vec<Vec<i8>>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let base0 = random_binary(dim, &mut rng).expect("base0");
        let base1 = random_binary(dim, &mut rng).expect("base1");
        let mut samples = Vec::new();
        let mut labels = Vec::new();
        let flips = dim / 20; // 5% noise — well separated
        for _ in 0..per_class {
            samples.push(noisy(&base0, flips, &mut rng));
            labels.push(0usize);
            samples.push(noisy(&base1, flips, &mut rng));
            labels.push(1usize);
        }
        (samples, labels)
    }

    fn accuracy_single(dim: usize, samples: &[Vec<i8>], labels: &[usize], seed: u64) -> f32 {
        let mut clf = HdClassifier::new(2, dim).expect("single");
        for (s, &l) in samples.iter().zip(labels.iter()) {
            clf.add_example(l, s).expect("add");
        }
        let mut rng = LcgRng::new(seed);
        clf.build_prototypes(&mut rng).expect("build");
        let mut correct = 0usize;
        for (s, &l) in samples.iter().zip(labels.iter()) {
            if clf.classify(s).expect("classify") == l {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }

    #[test]
    fn ensemble_matches_or_beats_single_and_fits_training() {
        // (a) On a separable toy set: ensemble accuracy ≥ single, and all training points right.
        let dim = 512;
        let (samples, labels) = toy_dataset(dim, 8, 100);
        let cfg = HdEnsembleConfig::new(5, 2, dim)
            .with_feature_fraction(0.5)
            .with_seed(42);
        let mut ens = HdEnsemble::new(cfg).expect("new");
        ens.train(&samples, &labels).expect("train");

        let mut correct = 0usize;
        for (s, &l) in samples.iter().zip(labels.iter()) {
            if ens.predict(s).expect("predict") == l {
                correct += 1;
            }
        }
        let ens_acc = correct as f32 / samples.len() as f32;
        let single_acc = accuracy_single(dim, &samples, &labels, 42);
        assert!(
            ens_acc >= single_acc - 1e-6,
            "ensemble {ens_acc} should be ≥ single {single_acc}"
        );
        assert!(
            (ens_acc - 1.0).abs() < 1e-6,
            "ensemble should fit separable training set: acc={ens_acc}"
        );
    }

    #[test]
    fn single_member_full_basis_equals_single_classifier() {
        // (b) M = 1 with feature_fraction = 1.0 reproduces a plain HdClassifier.
        let dim = 256;
        let (samples, labels) = toy_dataset(dim, 5, 7);
        let seed = 314;

        let cfg = HdEnsembleConfig::new(1, 2, dim)
            .with_feature_fraction(1.0)
            .with_seed(seed);
        let mut ens = HdEnsemble::new(cfg).expect("new");
        ens.train(&samples, &labels).expect("train");

        let mut single = HdClassifier::new(2, dim).expect("single");
        for (s, &l) in samples.iter().zip(labels.iter()) {
            single.add_example(l, s).expect("add");
        }
        let mut rng = LcgRng::new(seed); // member 0's build seed == config.seed + 0
        single.build_prototypes(&mut rng).expect("build");

        let mut probe_rng = LcgRng::new(9090);
        for _ in 0..50 {
            let q = random_binary(dim, &mut probe_rng).expect("probe");
            assert_eq!(
                ens.predict(&q).expect("ens"),
                single.classify(&q).expect("single"),
                "M=1 full-basis ensemble must match the single classifier"
            );
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        // (c) Identical config + data ⇒ identical predictions.
        let dim = 384;
        let (samples, labels) = toy_dataset(dim, 6, 55);
        let cfg = HdEnsembleConfig::new(4, 2, dim).with_seed(2024);

        let mut a = HdEnsemble::new(cfg.clone()).expect("a");
        let mut b = HdEnsemble::new(cfg).expect("b");
        a.train(&samples, &labels).expect("train a");
        b.train(&samples, &labels).expect("train b");

        let mut probe_rng = LcgRng::new(31337);
        for _ in 0..40 {
            let q = random_binary(dim, &mut probe_rng).expect("probe");
            assert_eq!(a.predict(&q).expect("a"), b.predict(&q).expect("b"));
            assert_eq!(
                a.predict_scores(&q).expect("a"),
                b.predict_scores(&q).expect("b")
            );
        }
    }

    #[test]
    fn predict_before_train_errors() {
        // (d) Untrained model → PrototypeNotBuilt.
        let cfg = HdEnsembleConfig::new(3, 2, 128);
        let ens = HdEnsemble::new(cfg).expect("new");
        let q = vec![1i8; 128];
        assert!(matches!(ens.predict(&q), Err(HdcError::PrototypeNotBuilt)));
        assert!(matches!(
            ens.predict_scores(&q),
            Err(HdcError::PrototypeNotBuilt)
        ));
    }

    #[test]
    fn construction_and_label_validation() {
        // (d cont.) zero members / classes / dim and out-of-range labels are rejected.
        assert!(matches!(
            HdEnsemble::new(HdEnsembleConfig::new(0, 2, 64)),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            HdEnsemble::new(HdEnsembleConfig::new(3, 0, 64)),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            HdEnsemble::new(HdEnsembleConfig::new(3, 2, 0)),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            HdEnsemble::new(HdEnsembleConfig::new(3, 2, 64).with_feature_fraction(0.0)),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            HdEnsemble::new(HdEnsembleConfig::new(3, 2, 64).with_feature_fraction(1.5)),
            Err(HdcError::InvalidProbability(_))
        ));

        let mut ens = HdEnsemble::new(HdEnsembleConfig::new(2, 2, 64)).expect("new");
        let samples = vec![vec![1i8; 64], vec![-1i8; 64]];
        let labels = vec![0usize, 5usize]; // 5 >= n_classes
        assert!(matches!(
            ens.train(&samples, &labels),
            Err(HdcError::ClassNotFound(5))
        ));
    }

    #[test]
    fn dimension_mismatch_errors() {
        // (e) Wrong-length training/query vectors are rejected.
        let dim = 128;
        let cfg = HdEnsembleConfig::new(2, 2, dim);
        let mut ens = HdEnsemble::new(cfg).expect("new");

        let bad_samples = vec![vec![1i8; dim], vec![1i8; dim - 1]];
        let labels = vec![0usize, 1usize];
        assert!(matches!(
            ens.train(&bad_samples, &labels),
            Err(HdcError::DimensionMismatch { .. })
        ));

        // Mismatched sample/label counts.
        let ok_samples = vec![vec![1i8; dim], vec![-1i8; dim]];
        assert!(matches!(
            ens.train(&ok_samples, &[0usize]),
            Err(HdcError::DimensionMismatch { .. })
        ));

        // Train correctly, then query with the wrong dimension.
        ens.train(&ok_samples, &labels).expect("train");
        let bad_q = vec![1i8; dim + 3];
        assert!(matches!(
            ens.predict(&bad_q),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn predict_scores_length_and_accessors() {
        let dim = 256;
        let (samples, labels) = toy_dataset(dim, 4, 12);
        let cfg = HdEnsembleConfig::new(3, 2, dim);
        let mut ens = HdEnsemble::new(cfg).expect("new");
        assert_eq!(ens.n_members(), 3);
        assert_eq!(ens.n_classes(), 2);
        ens.train(&samples, &labels).expect("train");
        let scores = ens.predict_scores(&samples[0]).expect("scores");
        assert_eq!(scores.len(), 2);
    }
}
