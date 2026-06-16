//! Adaptive hyperdimensional learning (Imani et al., DATE 2019 — "AdaptHD"; see also
//! OnlineHD, Hernández-Cano 2021).
//!
//! The baseline HD classifier in [`crate::classifier::hd_classifier`] stores one *binary*
//! prototype per class, formed by a single majority-vote pass over the training data. AdaptHD
//! improves accuracy by keeping **real-valued** (`f32`) class prototypes and *iteratively
//! retraining* them: each epoch streams the training set and, for every sample, performs a
//! confidence-weighted error-corrective update.
//!
//! For an encoded query `x` whose true class is `c`, let `δ_y = cos(x, model_y)` be the cosine
//! similarity to class `y`'s prototype and let `p = argmax_y δ_y` be the prediction. AdaptHD
//! applies the update
//!
//! ```text
//! model_c ← model_c + η · (1 − δ_c) · x        (pull the correct class toward x)
//! model_p ← model_p − η · (1 − δ_p) · x        (push the wrongly-favoured class away)  if p ≠ c
//! ```
//!
//! The `(1 − δ)` weighting makes confidently-correct samples contribute little and hard /
//! misclassified samples contribute strongly, which is the core adaptive mechanism that lets
//! the model converge in a handful of epochs. Inputs are the crate-standard binary `Vec<i8>`
//! hypervectors in `{−1, +1}`; prototypes accumulate in `f32`.

use crate::error::{HdcError, HdcResult};

/// Configuration for an [`AdaptiveHdClassifier`].
#[derive(Debug, Clone)]
pub struct AdaptiveHdConfig {
    /// Number of classes (must be ≥ 1).
    pub n_classes: usize,
    /// Hypervector dimension `D` (must be ≥ 1).
    pub dim: usize,
    /// Learning rate `η` (> 0) applied to each confidence-weighted update.
    pub learning_rate: f32,
    /// Number of retraining epochs over the training set (≥ 1).
    pub epochs: usize,
}

impl Default for AdaptiveHdConfig {
    fn default() -> Self {
        Self {
            n_classes: 2,
            dim: 10_000,
            learning_rate: 0.035,
            epochs: 20,
        }
    }
}

/// Adaptive HD classifier with real-valued prototypes and iterative retraining.
pub struct AdaptiveHdClassifier {
    cfg: AdaptiveHdConfig,
    /// Real-valued class prototypes (`n_classes` rows, each length `dim`).
    prototypes: Vec<Vec<f32>>,
    /// Cached L2 norms of each prototype (kept in sync after every mutation).
    norms: Vec<f32>,
}

impl AdaptiveHdClassifier {
    /// Create a new classifier with zero-initialised prototypes.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `cfg.n_classes == 0`.
    /// - [`HdcError::ZeroDimension`] if `cfg.dim == 0`.
    /// - [`HdcError::InvalidProbability`] (reused) if `cfg.learning_rate <= 0`.
    /// - [`HdcError::InvalidNgramOrder`] (reused) if `cfg.epochs == 0`.
    pub fn new(cfg: AdaptiveHdConfig) -> HdcResult<Self> {
        if cfg.n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if cfg.dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if cfg.learning_rate <= 0.0 {
            return Err(HdcError::InvalidProbability(cfg.learning_rate as f64));
        }
        if cfg.epochs == 0 {
            return Err(HdcError::InvalidNgramOrder(cfg.epochs));
        }
        let prototypes = vec![vec![0f32; cfg.dim]; cfg.n_classes];
        let norms = vec![0f32; cfg.n_classes];
        Ok(Self {
            cfg,
            prototypes,
            norms,
        })
    }

    /// Cosine similarity between a binary query and class `c`'s real-valued prototype.
    /// For a `±1` query, `‖x‖ = √D`; the prototype norm is cached.
    fn cosine_to_class(&self, query: &[i8], c: usize) -> f32 {
        let dot: f32 = self.prototypes[c]
            .iter()
            .zip(query.iter())
            .map(|(&p, &q)| p * q as f32)
            .sum();
        let denom = self.norms[c] * (self.cfg.dim as f32).sqrt();
        if denom < f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    /// Recompute and cache the L2 norm of class `c`'s prototype.
    fn refresh_norm(&mut self, c: usize) {
        let sum_sq: f32 = self.prototypes[c].iter().map(|&v| v * v).sum();
        self.norms[c] = sum_sq.sqrt();
    }

    /// Add `scale · query` to class `c`'s prototype and refresh its cached norm.
    fn update_prototype(&mut self, c: usize, query: &[i8], scale: f32) {
        for (p, &q) in self.prototypes[c].iter_mut().zip(query.iter()) {
            *p += scale * q as f32;
        }
        self.refresh_norm(c);
    }

    /// Initialise prototypes with one additive pass (the standard "single-pass" HD model)
    /// before adaptive retraining. Each sample's HV is added to its class prototype.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `hvs.len() != labels.len()` or any HV dimension
    ///   is wrong.
    /// - [`HdcError::ClassNotFound`] if any label is `>= n_classes`.
    pub fn init_single_pass(&mut self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<()> {
        self.check_dataset(hvs, labels)?;
        for (hv, &label) in hvs.iter().zip(labels.iter()) {
            for (p, &q) in self.prototypes[label].iter_mut().zip(hv.iter()) {
                *p += q as f32;
            }
        }
        for c in 0..self.cfg.n_classes {
            self.refresh_norm(c);
        }
        Ok(())
    }

    /// Run one adaptive retraining epoch over the dataset, returning the number of
    /// misclassifications encountered during the pass.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] / [`HdcError::ClassNotFound`] for invalid data.
    pub fn train_epoch(&mut self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<usize> {
        self.check_dataset(hvs, labels)?;
        let eta = self.cfg.learning_rate;
        let mut errors = 0usize;
        for (hv, &true_class) in hvs.iter().zip(labels.iter()) {
            // Predict with the current model.
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for c in 0..self.cfg.n_classes {
                let sim = self.cosine_to_class(hv, c);
                if sim > best_sim {
                    best_sim = sim;
                    best = c;
                }
            }
            let delta_true = self.cosine_to_class(hv, true_class);
            // Pull the correct class toward the sample, weighted by its current miss margin.
            self.update_prototype(true_class, hv, eta * (1.0 - delta_true));
            if best != true_class {
                errors += 1;
                let delta_pred = self.cosine_to_class(hv, best);
                // Push the wrongly-favoured class away from the sample.
                self.update_prototype(best, hv, -eta * (1.0 - delta_pred));
            }
        }
        Ok(errors)
    }

    /// Full fit: single-pass initialisation followed by `cfg.epochs` adaptive epochs.
    /// Returns the per-epoch misclassification counts.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if there are no samples.
    /// - [`HdcError::DimensionMismatch`] / [`HdcError::ClassNotFound`] for invalid data.
    pub fn fit(&mut self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<Vec<usize>> {
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        self.init_single_pass(hvs, labels)?;
        let mut history = Vec::with_capacity(self.cfg.epochs);
        for _ in 0..self.cfg.epochs {
            let errors = self.train_epoch(hvs, labels)?;
            history.push(errors);
        }
        Ok(history)
    }

    /// Classify a query hypervector by argmax cosine over the real-valued prototypes.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `query` has the wrong dimension.
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

    /// Cosine similarity of a query to a specific class prototype.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `query` has the wrong dimension.
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

    /// Classification accuracy on a labelled evaluation set, in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if there are no samples.
    /// - Propagates errors from [`classify`](AdaptiveHdClassifier::classify).
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

    /// Validate a `(hvs, labels)` dataset against the configured shape.
    fn check_dataset(&self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<()> {
        if hvs.len() != labels.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: labels.len(),
            });
        }
        for hv in hvs {
            if hv.len() != self.cfg.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: self.cfg.dim,
                    got: hv.len(),
                });
            }
        }
        for &label in labels {
            if label >= self.cfg.n_classes {
                return Err(HdcError::ClassNotFound(label));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    fn cfg(n_classes: usize, dim: usize) -> AdaptiveHdConfig {
        AdaptiveHdConfig {
            n_classes,
            dim,
            learning_rate: 0.05,
            epochs: 15,
        }
    }

    /// Build a noisy dataset: each class has a centroid HV, samples flip a fraction of bits.
    fn noisy_dataset(
        n_classes: usize,
        per_class: usize,
        dim: usize,
        flip_frac: f32,
        rng: &mut LcgRng,
    ) -> (Vec<Vec<i8>>, Vec<usize>) {
        let centroids: Vec<Vec<i8>> = (0..n_classes)
            .map(|_| random_binary(dim, rng).expect("centroid"))
            .collect();
        let mut hvs = Vec::new();
        let mut labels = Vec::new();
        for (c, centroid) in centroids.iter().enumerate() {
            for _ in 0..per_class {
                let mut sample = centroid.clone();
                for slot in sample.iter_mut() {
                    if rng.next_f32() < flip_frac {
                        *slot = -*slot;
                    }
                }
                hvs.push(sample);
                labels.push(c);
            }
        }
        (hvs, labels)
    }

    #[test]
    fn config_default_valid() {
        let c = AdaptiveHdConfig::default();
        assert_eq!(c.n_classes, 2);
        assert!(c.learning_rate > 0.0);
        assert!(c.epochs >= 1);
    }

    #[test]
    fn new_rejects_bad_config() {
        assert!(matches!(
            AdaptiveHdClassifier::new(AdaptiveHdConfig {
                n_classes: 0,
                ..cfg(2, 64)
            }),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            AdaptiveHdClassifier::new(AdaptiveHdConfig {
                dim: 0,
                ..cfg(2, 64)
            }),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            AdaptiveHdClassifier::new(AdaptiveHdConfig {
                learning_rate: 0.0,
                ..cfg(2, 64)
            }),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            AdaptiveHdClassifier::new(AdaptiveHdConfig {
                epochs: 0,
                ..cfg(2, 64)
            }),
            Err(HdcError::InvalidNgramOrder(0))
        ));
    }

    #[test]
    fn single_pass_then_classify_clean_centroids() {
        let mut r = LcgRng::new(1);
        let dim = 2000;
        let (hvs, labels) = noisy_dataset(3, 1, dim, 0.0, &mut r);
        let mut clf = AdaptiveHdClassifier::new(cfg(3, dim)).expect("new");
        clf.init_single_pass(&hvs, &labels).expect("init");
        for (hv, &label) in hvs.iter().zip(labels.iter()) {
            assert_eq!(clf.classify(hv).expect("classify"), label);
        }
    }

    #[test]
    fn fit_returns_epoch_history() {
        let mut r = LcgRng::new(2);
        let dim = 1500;
        let (hvs, labels) = noisy_dataset(2, 10, dim, 0.1, &mut r);
        let mut clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        let history = clf.fit(&hvs, &labels).expect("fit");
        assert_eq!(history.len(), 15);
    }

    #[test]
    fn adaptive_training_reduces_errors() {
        // Error count in the last epoch should not exceed the first epoch.
        let mut r = LcgRng::new(3);
        let dim = 2000;
        let (hvs, labels) = noisy_dataset(3, 15, dim, 0.2, &mut r);
        let mut clf = AdaptiveHdClassifier::new(cfg(3, dim)).expect("new");
        let history = clf.fit(&hvs, &labels).expect("fit");
        let first = history[0];
        let last = *history.last().expect("history nonempty");
        assert!(
            last <= first,
            "adaptive training did not reduce errors: first={first} last={last}"
        );
    }

    #[test]
    fn fit_achieves_high_train_accuracy() {
        let mut r = LcgRng::new(4);
        let dim = 4000;
        let (hvs, labels) = noisy_dataset(4, 12, dim, 0.15, &mut r);
        let mut clf = AdaptiveHdClassifier::new(cfg(4, dim)).expect("new");
        clf.fit(&hvs, &labels).expect("fit");
        let acc = clf.accuracy(&hvs, &labels).expect("accuracy");
        assert!(acc > 0.9, "train accuracy too low: {acc}");
    }

    #[test]
    fn classify_wrong_dim_errors() {
        let mut r = LcgRng::new(5);
        let dim = 256;
        let (hvs, labels) = noisy_dataset(2, 2, dim, 0.0, &mut r);
        let mut clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        clf.init_single_pass(&hvs, &labels).expect("init");
        let bad = random_binary(128, &mut r).expect("bad");
        assert!(matches!(
            clf.classify(&bad),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn train_epoch_bad_label_errors() {
        let mut r = LcgRng::new(6);
        let dim = 256;
        let hvs = vec![random_binary(dim, &mut r).expect("hv")];
        let labels = vec![5usize]; // out of range for 2 classes
        let mut clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        assert!(matches!(
            clf.train_epoch(&hvs, &labels),
            Err(HdcError::ClassNotFound(5))
        ));
    }

    #[test]
    fn class_similarity_self_is_high() {
        // A prototype trained on a single clean sample should be highly similar to it.
        let mut r = LcgRng::new(7);
        let dim = 2000;
        let sample = random_binary(dim, &mut r).expect("sample");
        let mut clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        clf.init_single_pass(std::slice::from_ref(&sample), &[0])
            .expect("init");
        let sim = clf.class_similarity(&sample, 0).expect("sim");
        assert!(sim > 0.95, "self-similarity low: {sim}");
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn class_similarity_bad_class_errors() {
        let mut r = LcgRng::new(8);
        let dim = 256;
        let sample = random_binary(dim, &mut r).expect("sample");
        let clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        assert!(matches!(
            clf.class_similarity(&sample, 9),
            Err(HdcError::ClassNotFound(9))
        ));
    }

    #[test]
    fn accuracy_empty_errors() {
        let dim = 64;
        let clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        assert!(matches!(clf.accuracy(&[], &[]), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn fit_empty_errors() {
        let dim = 64;
        let mut clf = AdaptiveHdClassifier::new(cfg(2, dim)).expect("new");
        assert!(matches!(clf.fit(&[], &[]), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn n_classes_accessor() {
        let dim = 64;
        let clf = AdaptiveHdClassifier::new(cfg(5, dim)).expect("new");
        assert_eq!(clf.n_classes(), 5);
    }
}
