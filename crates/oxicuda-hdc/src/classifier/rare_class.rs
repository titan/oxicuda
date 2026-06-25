//! HD classifier with regularisation against rare classes (long-tail / imbalanced HD learning).
//!
//! # Motivation
//!
//! In hyperdimensional computing (HDC), a class is represented by a *prototype*
//! hypervector built by bundling (summing) the training examples of that class.
//! Classification picks the class whose prototype is most cosine-similar to the
//! query. Under **class imbalance** this comparison is biased toward *frequent*
//! classes for two reasons:
//!
//! 1. **Magnitude.** An un-normalised accumulator `acc_c = sum_i x_i` grows with
//!    the number of examples `N_c`. If a downstream score uses the raw dot
//!    product (rather than a true cosine) the frequent class is favoured purely
//!    by magnitude.
//! 2. **Sharpness.** A prototype averaged over many examples concentrates more
//!    tightly around the class mean, so a frequent prototype tends to achieve a
//!    higher *cosine* with in-distribution queries than a rare prototype built
//!    from only a handful of (noisier) examples. Frequent classes therefore
//!    "swamp" the long tail.
//!
//! # Regularisation
//!
//! This module applies two complementary corrections:
//!
//! * **(a) Per-class L2 normalisation.** Each prototype is stored as the unit
//!   vector `proto_c = acc_c / ||acc_c||`. This removes the count-dependent
//!   magnitude entirely, so the cosine of a `±1` query `q` against `proto_c` is
//!   simply `dot(proto_c, q) / (||proto_c|| * ||q||) = dot(proto_c, q) / sqrt(D)`
//!   (since `||proto_c|| = 1` and `||q|| = sqrt(D)` for a `±1` hypervector).
//!
//! * **(b) Inverse-frequency logit bias.** A per-class additive bias is added to
//!   the cosine score at classification time:
//!
//!   ```text
//!   bias_c = alpha * ln(N_total / N_c)
//!   ```
//!
//!   where `N_total = sum_c N_c` and `N_c` is the number of training examples of
//!   class `c`. Because `ln(N_total / N_c)` is *large* for rare classes
//!   (`N_c` small) and *small* for frequent classes (`N_c` large), this term
//!   boosts the long tail. `alpha >= 0` controls the regularisation strength;
//!   `alpha = 0` disables the bias and recovers a plain normalised-prototype
//!   classifier.
//!
//! This is the HDC analogue of **logit adjustment** for long-tail recognition
//! (Menon et al., "Long-tail learning via logit adjustment", ICLR 2021), which
//! shifts each logit by `tau * ln(pi_c)` (a balanced prior); here we use the
//! complementary inverse-frequency form `-ln(pi_c) = ln(N_total / N_c)` (up to an
//! additive constant). It is closely related to inverse-frequency class
//! weighting and the *effective-number* re-weighting of the class-balanced loss
//! (Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
//! CVPR 2019).
//!
//! # Score
//!
//! For a query `q in {-1,+1}^D` the score of a *seen* class `c` (with `N_c > 0`)
//! is
//!
//! ```text
//! score_c(q) = dot(proto_c, q) / sqrt(D) + bias_c .
//! ```
//!
//! Classes never seen during training (`N_c = 0`) receive `score = -inf` and are
//! never predicted. The prediction is `argmax_c score_c(q)`.
//!
//! Crucially, the cosine term does **not** depend on `alpha`, so the score
//! *margin* between two classes splits cleanly:
//!
//! ```text
//! score_rare(q) - score_freq(q)
//!     = [cos_rare - cos_freq] + alpha * [ln(N/N_rare) - ln(N/N_freq)] .
//! ```
//!
//! For `N_rare < N_freq` the bracketed bias difference is strictly positive, so
//! increasing `alpha` *strictly* increases the rare class's margin — the
//! mechanism by which the regulariser rescues the long tail.

use crate::error::{HdcError, HdcResult};

/// Configuration for a [`RareClassClassifier`].
///
/// Holds the fixed problem geometry (`n_classes`, `dim`) together with the
/// inverse-frequency regularisation strength `alpha`.
#[derive(Debug, Clone)]
pub struct RareClassConfig {
    /// Number of classes (must be `> 0`).
    n_classes: usize,
    /// Hypervector dimension (must be `> 0`).
    dim: usize,
    /// Inverse-frequency logit-bias strength `alpha >= 0`.
    ///
    /// The per-class bias is `alpha * ln(N_total / N_c)`. `alpha = 0` disables
    /// the bias (plain normalised-prototype classifier); larger `alpha` boosts
    /// rare classes more aggressively.
    alpha: f32,
}

impl RareClassConfig {
    /// Create a validated configuration.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if `n_classes == 0`.
    /// * [`HdcError::ZeroDimension`] if `dim == 0`.
    /// * [`HdcError::InvalidProbability`] if `alpha` is not finite or `alpha < 0`.
    pub fn new(n_classes: usize, dim: usize, alpha: f32) -> HdcResult<Self> {
        if n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(HdcError::InvalidProbability(alpha as f64));
        }
        Ok(Self {
            n_classes,
            dim,
            alpha,
        })
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Hypervector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Inverse-frequency regularisation strength `alpha`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

/// HD classifier with inverse-frequency regularisation for imbalanced data.
///
/// Training accumulates real-valued (`f32`) per-class sums of `±1` example
/// hypervectors. [`build`](Self::build) then forms **unit-L2 normalised**
/// prototypes (so the count is not encoded in the magnitude) and the per-class
/// inverse-frequency bias `bias_c = alpha * ln(N_total / N_c)`. At classify
/// time the score is `cosine(proto_c, query) + bias_c`, with unseen classes
/// (`N_c = 0`) excluded.
///
/// See the [module documentation](crate::classifier::rare_class) for the full
/// derivation and references (Menon 2021; Cui 2019).
pub struct RareClassClassifier {
    /// Number of classes.
    n_classes: usize,
    /// Hypervector dimension.
    dim: usize,
    /// Inverse-frequency regularisation strength.
    alpha: f32,
    /// Per-class real-valued accumulators (sum of `±1` training HVs as `f32`).
    accumulators: Vec<Vec<f32>>,
    /// Per-class training example counts `N_c`.
    counts: Vec<usize>,
    /// Cached per-class unit-L2 prototypes (`acc_c / ||acc_c||`); zero if unseen.
    prototypes: Vec<Vec<f32>>,
    /// Cached per-class inverse-frequency bias `alpha * ln(N_total / N_c)`.
    biases: Vec<f32>,
    /// Whether [`build`](Self::build) has run since the last mutation.
    built: bool,
}

impl RareClassClassifier {
    /// Create a new classifier from a [`RareClassConfig`].
    ///
    /// All accumulators, counts, prototypes and biases start at zero and the
    /// classifier is not yet `built`.
    pub fn new(cfg: RareClassConfig) -> HdcResult<Self> {
        let n_classes = cfg.n_classes;
        let dim = cfg.dim;
        Ok(Self {
            n_classes,
            dim,
            alpha: cfg.alpha,
            accumulators: vec![vec![0f32; dim]; n_classes],
            counts: vec![0usize; n_classes],
            prototypes: vec![vec![0f32; dim]; n_classes],
            biases: vec![0f32; n_classes],
            built: false,
        })
    }

    /// Add one training example `hv` (`±1` hypervector) for class `class`.
    ///
    /// The example is summed (as `f32`) into the class accumulator and the class
    /// count is incremented. This invalidates any previously built prototypes
    /// and biases (sets `built = false`); call [`build`](Self::build) again
    /// before classifying.
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    /// * [`HdcError::DimensionMismatch`] if `hv.len() != dim`.
    pub fn add_example(&mut self, class: usize, hv: &[i8]) -> HdcResult<()> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        for (a, &v) in self.accumulators[class].iter_mut().zip(hv.iter()) {
            *a += v as f32;
        }
        self.counts[class] += 1;
        self.built = false;
        Ok(())
    }

    /// Build unit-L2 prototypes and inverse-frequency biases from the
    /// accumulators.
    ///
    /// For every class `c` with `N_c > 0`:
    ///
    /// * the prototype is `acc_c / ||acc_c||_2` (unit L2 norm); if the
    ///   accumulator is exactly all-zero it is left as zeros;
    /// * the bias is `bias_c = alpha * ln(N_total / N_c)` where
    ///   `N_total = sum_c N_c`.
    ///
    /// Classes with `N_c = 0` keep a zero prototype and a zero bias and are
    /// excluded from classification.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if no training examples have been added
    ///   (`N_total == 0`), so no prototype can be formed.
    pub fn build(&mut self) -> HdcResult<()> {
        let total: usize = self.counts.iter().sum();
        if total == 0 {
            return Err(HdcError::EmptyInput);
        }
        let total_f = total as f32;
        for c in 0..self.n_classes {
            if self.counts[c] == 0 {
                // Unseen class: zero prototype, zero bias.
                for p in self.prototypes[c].iter_mut() {
                    *p = 0.0;
                }
                self.biases[c] = 0.0;
                continue;
            }
            // L2 norm of the accumulator (computed in f64 for stability).
            let norm_sq: f64 = self.accumulators[c]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum();
            let norm = norm_sq.sqrt();
            if norm < f64::EPSILON {
                // All-zero accumulator: leave prototype as zeros.
                for p in self.prototypes[c].iter_mut() {
                    *p = 0.0;
                }
            } else {
                let inv = 1.0 / norm;
                for (p, &a) in self.prototypes[c]
                    .iter_mut()
                    .zip(self.accumulators[c].iter())
                {
                    *p = (a as f64 * inv) as f32;
                }
            }
            // Inverse-frequency logit bias: alpha * ln(N_total / N_c).
            self.biases[c] = self.alpha * (total_f / self.counts[c] as f32).ln();
        }
        self.built = true;
        Ok(())
    }

    /// Compute the regularised score of every class for `query`.
    ///
    /// The score of a *seen* class `c` (`N_c > 0`) is
    /// `dot(proto_c, query) / sqrt(D) + bias_c`. Because each `proto_c` is unit
    /// L2 norm and `query in {-1,+1}^D` has norm `sqrt(D)`, this is exactly the
    /// cosine similarity plus the inverse-frequency bias.
    ///
    /// *Unseen* classes (`N_c = 0`) receive [`f32::NEG_INFINITY`] so they can
    /// never be the `argmax`.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if [`build`](Self::build) has not been called
    ///   yet (call `build()` first).
    /// * [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    pub fn scores(&self, query: &[i8]) -> HdcResult<Vec<f32>> {
        if !self.built {
            // No prototypes available yet — caller must build() first.
            return Err(HdcError::EmptyInput);
        }
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let sqrt_d = (self.dim as f64).sqrt();
        let mut out = vec![f32::NEG_INFINITY; self.n_classes];
        for (c, slot) in out.iter_mut().enumerate() {
            if self.counts[c] == 0 {
                // Never predict an unseen class (leave score at -inf).
                continue;
            }
            let dot: f64 = self.prototypes[c]
                .iter()
                .zip(query.iter())
                .map(|(&p, &q)| (p as f64) * (q as f64))
                .sum();
            let cosine = (dot / sqrt_d) as f32;
            *slot = cosine + self.biases[c];
        }
        Ok(out)
    }

    /// Classify `query`: return the seen class with the highest regularised
    /// score `cosine(proto_c, query) + bias_c`.
    ///
    /// Unseen classes (`N_c = 0`) are skipped and can never be returned. Because
    /// [`build`](Self::build) requires at least one training example, at least
    /// one class is always seen, so a valid class is always returned.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if [`build`](Self::build) has not been called
    ///   yet (call `build()` first).
    /// * [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    pub fn classify(&self, query: &[i8]) -> HdcResult<usize> {
        let scores = self.scores(query)?;
        let mut best_class = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (c, &s) in scores.iter().enumerate() {
            if s > best_score {
                best_score = s;
                best_class = c;
            }
        }
        Ok(best_class)
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Hypervector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Inverse-frequency regularisation strength `alpha`.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Number of training examples `N_c` accumulated for `class`.
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn class_count(&self, class: usize) -> HdcResult<usize> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(self.counts[class])
    }

    /// Inverse-frequency bias `alpha * ln(N_total / N_c)` of `class`.
    ///
    /// Returns `0.0` for unseen classes and before [`build`](Self::build).
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn class_bias(&self, class: usize) -> HdcResult<f32> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(self.biases[class])
    }

    /// Unit-L2 prototype hypervector of `class`.
    ///
    /// Returns all-zeros for an unseen class or before [`build`](Self::build).
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn prototype(&self, class: usize) -> HdcResult<&[f32]> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(&self.prototypes[class])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    /// Flip a fraction of a `±1` hypervector to create a correlated noisy copy.
    /// `flip_prob` in [0, 1]: probability of flipping each coordinate.
    fn noisy_copy(base: &[i8], flip_prob: f32, rng: &mut LcgRng) -> Vec<i8> {
        base.iter()
            .map(|&v| {
                let u = rng.next_u32() as f64 / 2f64.powi(32);
                if u < flip_prob as f64 { -v } else { v }
            })
            .collect()
    }

    /// Build a hypervector that shares `shared` leading coordinates with `base`
    /// and uses fresh random `±1` values for the remaining `dim - shared`. The
    /// shared block makes two prototypes non-orthogonal (positively correlated).
    fn correlated_with(base: &[i8], shared: usize, rng: &mut LcgRng) -> Vec<i8> {
        let dim = base.len();
        let mut out = vec![0i8; dim];
        for i in 0..dim {
            if i < shared {
                out[i] = base[i];
            } else {
                out[i] = if rng.next_bool() { 1 } else { -1 };
            }
        }
        out
    }

    #[test]
    fn config_rejects_zero_classes() {
        let err = RareClassConfig::new(0, 512, 1.0);
        assert!(matches!(err, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn config_rejects_zero_dim() {
        let err = RareClassConfig::new(3, 0, 1.0);
        assert!(matches!(err, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn config_rejects_negative_and_nan_alpha() {
        let neg = RareClassConfig::new(2, 512, -0.5);
        assert!(matches!(neg, Err(HdcError::InvalidProbability(_))));
        let nan = RareClassConfig::new(2, 512, f32::NAN);
        assert!(matches!(nan, Err(HdcError::InvalidProbability(_))));
        let inf = RareClassConfig::new(2, 512, f32::INFINITY);
        assert!(matches!(inf, Err(HdcError::InvalidProbability(_))));
    }

    #[test]
    fn balanced_two_class_classifies_correctly() {
        let mut rng = LcgRng::new(101);
        let dim = 1024;
        let cfg = RareClassConfig::new(2, dim, 1.0).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");

        let proto0 = random_binary(dim, &mut rng).expect("proto0");
        let proto1 = random_binary(dim, &mut rng).expect("proto1");

        // Balanced counts: 8 examples each (noisy copies).
        for _ in 0..8 {
            let n0 = noisy_copy(&proto0, 0.1, &mut rng);
            let n1 = noisy_copy(&proto1, 0.1, &mut rng);
            clf.add_example(0, &n0).expect("add 0");
            clf.add_example(1, &n1).expect("add 1");
        }
        clf.build().expect("build");

        assert_eq!(clf.classify(&proto0).expect("c0"), 0);
        assert_eq!(clf.classify(&proto1).expect("c1"), 1);
        // Balanced => equal biases.
        let b0 = clf.class_bias(0).expect("b0");
        let b1 = clf.class_bias(1).expect("b1");
        assert!((b0 - b1).abs() < 1e-6, "b0={b0} b1={b1}");
    }

    #[test]
    fn classify_before_build_errors() {
        let dim = 512;
        let cfg = RareClassConfig::new(2, dim, 1.0).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");
        let mut rng = LcgRng::new(7);
        let p = random_binary(dim, &mut rng).expect("p");
        clf.add_example(0, &p).expect("add");
        // Not built yet.
        assert!(matches!(clf.classify(&p), Err(HdcError::EmptyInput)));
        assert!(matches!(clf.scores(&p), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn build_requires_examples() {
        let cfg = RareClassConfig::new(3, 512, 1.0).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");
        assert!(matches!(clf.build(), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn unseen_class_never_predicted() {
        let mut rng = LcgRng::new(202);
        let dim = 512;
        // 3 classes but only classes 0 and 1 get examples; class 2 is unseen.
        let cfg = RareClassConfig::new(3, dim, 2.0).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");

        let proto0 = random_binary(dim, &mut rng).expect("proto0");
        let proto1 = random_binary(dim, &mut rng).expect("proto1");
        for _ in 0..4 {
            clf.add_example(0, &proto0).expect("add 0");
            clf.add_example(1, &proto1).expect("add 1");
        }
        clf.build().expect("build");

        let scores = clf.scores(&proto0).expect("scores");
        assert_eq!(scores[2], f32::NEG_INFINITY, "unseen class must be -inf");
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
        // Even a query equal to nothing in particular must not pick class 2.
        let pred = clf.classify(&proto1).expect("pred");
        assert_ne!(pred, 2);
        assert_eq!(clf.class_count(2).expect("cnt2"), 0);
        assert_eq!(clf.class_bias(2).expect("bias2"), 0.0);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut rng = LcgRng::new(9);
        let dim = 512;
        let cfg = RareClassConfig::new(2, dim, 1.0).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");
        let p = random_binary(dim, &mut rng).expect("p");
        // Wrong-length add.
        let short = vec![1i8; dim - 1];
        assert!(matches!(
            clf.add_example(0, &short),
            Err(HdcError::DimensionMismatch { .. })
        ));
        // Out-of-range class.
        assert!(matches!(
            clf.add_example(5, &p),
            Err(HdcError::ClassNotFound(5))
        ));
        for _ in 0..3 {
            clf.add_example(0, &p).expect("add 0");
            clf.add_example(1, &p).expect("add 1");
        }
        clf.build().expect("build");
        assert!(matches!(
            clf.classify(&short),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    /// KEY IMBALANCE TEST.
    ///
    /// A frequent class (many examples) and a rare class (few examples) with
    /// *non-orthogonal* prototypes (they share a leading block, so their cosines
    /// against a `P_rare` query are close). With `alpha = 0` the rare class's
    /// margin over the frequent class is determined purely by cosine. Increasing
    /// `alpha` adds `alpha * (ln(N/N_rare) - ln(N/N_freq)) > 0` to that margin,
    /// so the margin must strictly increase — proving the inverse-frequency
    /// regulariser boosts the long tail and can flip the decision.
    #[test]
    fn imbalance_bias_boosts_rare_class() {
        let mut rng = LcgRng::new(303);
        let dim = 2048;

        // Build two correlated base patterns: P_rare and P_freq share the first
        // `shared` coordinates, the rest are independent random ±1.
        let p_rare = random_binary(dim, &mut rng).expect("p_rare");
        let shared = dim / 2; // strong positive correlation between the two
        let p_freq = correlated_with(&p_rare, shared, &mut rng);

        // Class 0 = rare (few examples), class 1 = frequent (many examples).
        let n_rare = 3usize;
        let n_freq = 60usize;

        // Helper to build a classifier at a given alpha with identical data.
        let build_clf = |alpha: f32| -> RareClassClassifier {
            let cfg = RareClassConfig::new(2, dim, alpha).expect("cfg");
            let mut clf = RareClassClassifier::new(cfg).expect("new");
            // Use a fixed inner seed so both classifiers see identical examples.
            let mut inner = LcgRng::new(424_242);
            for _ in 0..n_rare {
                let ex = noisy_copy(&p_rare, 0.15, &mut inner);
                clf.add_example(0, &ex).expect("add rare");
            }
            for _ in 0..n_freq {
                let ex = noisy_copy(&p_freq, 0.15, &mut inner);
                clf.add_example(1, &ex).expect("add freq");
            }
            clf.build().expect("build");
            clf
        };

        let clf_a0 = build_clf(0.0);
        let clf_hi = build_clf(2.5);

        // Query equals the rare prototype: we want the rare class to win once
        // regularised. Use P_rare itself as the query.
        let query = &p_rare;

        let s0 = clf_a0.scores(query).expect("scores a0");
        let s_hi = clf_hi.scores(query).expect("scores hi");

        // Cosine terms are identical across alpha (bias is additive only):
        // recover them by subtracting the (known) biases.
        let margin_a0 = s0[0] - s0[1]; // rare - freq at alpha = 0
        let margin_hi = s_hi[0] - s_hi[1]; // rare - freq at alpha = 2.5

        // (1) The regulariser STRICTLY increases the rare-vs-freq margin.
        assert!(
            margin_hi > margin_a0 + 1e-3,
            "margin must strictly increase: a0={margin_a0} hi={margin_hi}"
        );

        // (2) Bias ordering: rare class has the larger inverse-frequency bias.
        let b_rare = clf_hi.class_bias(0).expect("b_rare");
        let b_freq = clf_hi.class_bias(1).expect("b_freq");
        assert!(
            b_rare > b_freq,
            "rare bias must exceed freq bias: rare={b_rare} freq={b_freq}"
        );
        // At alpha = 0 both biases vanish.
        assert_eq!(clf_a0.class_bias(0).expect("b0"), 0.0);
        assert_eq!(clf_a0.class_bias(1).expect("b1"), 0.0);

        // (3) The bias difference equals exactly the margin increase
        // (cosine terms cancel): margin_hi - margin_a0 == b_rare - b_freq.
        let delta_margin = margin_hi - margin_a0;
        let delta_bias = b_rare - b_freq;
        assert!(
            (delta_margin - delta_bias).abs() < 1e-2,
            "margin gain {delta_margin} should equal bias gap {delta_bias}"
        );

        // (4) With enough regularisation the rare class actually WINS for the
        // rare query, demonstrating a flipped/secured decision.
        assert_eq!(
            clf_hi.classify(query).expect("classify hi"),
            0,
            "rare class must win the rare query under regularisation"
        );
    }

    #[test]
    fn class_count_and_bias_correctness() {
        let mut rng = LcgRng::new(404);
        let dim = 512;
        let alpha = 1.5f32;
        let cfg = RareClassConfig::new(2, dim, alpha).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");

        let p0 = random_binary(dim, &mut rng).expect("p0");
        let p1 = random_binary(dim, &mut rng).expect("p1");
        let n0 = 2usize;
        let n1 = 8usize;
        for _ in 0..n0 {
            clf.add_example(0, &p0).expect("add 0");
        }
        for _ in 0..n1 {
            clf.add_example(1, &p1).expect("add 1");
        }
        clf.build().expect("build");

        assert_eq!(clf.class_count(0).expect("c0"), n0);
        assert_eq!(clf.class_count(1).expect("c1"), n1);

        let total = (n0 + n1) as f32;
        let expected_b0 = alpha * (total / n0 as f32).ln();
        let expected_b1 = alpha * (total / n1 as f32).ln();
        assert!((clf.class_bias(0).expect("b0") - expected_b0).abs() < 1e-5);
        assert!((clf.class_bias(1).expect("b1") - expected_b1).abs() < 1e-5);
        // Out-of-range accessors error.
        assert!(matches!(
            clf.class_count(9),
            Err(HdcError::ClassNotFound(9))
        ));
        assert!(matches!(clf.class_bias(9), Err(HdcError::ClassNotFound(9))));
        assert!(matches!(clf.prototype(9), Err(HdcError::ClassNotFound(9))));
    }

    #[test]
    fn prototype_is_unit_norm() {
        let mut rng = LcgRng::new(505);
        let dim = 1024;
        let cfg = RareClassConfig::new(2, dim, 0.0).expect("cfg");
        let mut clf = RareClassClassifier::new(cfg).expect("new");
        let p0 = random_binary(dim, &mut rng).expect("p0");
        for _ in 0..5 {
            let ex = noisy_copy(&p0, 0.2, &mut rng);
            clf.add_example(0, &ex).expect("add");
        }
        // class 1 seen too (build needs only one, but exercise a 2nd unit norm).
        let p1 = random_binary(dim, &mut rng).expect("p1");
        clf.add_example(1, &p1).expect("add1");
        clf.build().expect("build");

        let proto0 = clf.prototype(0).expect("proto0");
        let norm: f64 = proto0
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "prototype L2 norm = {norm}");
    }

    #[test]
    fn determinism_same_seed_same_result() {
        let dim = 768;
        let run = |seed: u64| -> (usize, Vec<f32>) {
            let mut rng = LcgRng::new(seed);
            let cfg = RareClassConfig::new(3, dim, 1.2).expect("cfg");
            let mut clf = RareClassClassifier::new(cfg).expect("new");
            let p0 = random_binary(dim, &mut rng).expect("p0");
            let p1 = random_binary(dim, &mut rng).expect("p1");
            let p2 = random_binary(dim, &mut rng).expect("p2");
            for _ in 0..2 {
                clf.add_example(0, &p0).expect("a0");
            }
            for _ in 0..5 {
                clf.add_example(1, &p1).expect("a1");
            }
            for _ in 0..9 {
                clf.add_example(2, &p2).expect("a2");
            }
            clf.build().expect("build");
            let pred = clf.classify(&p0).expect("pred");
            let scores = clf.scores(&p0).expect("scores");
            (pred, scores)
        };
        let (pred_a, scores_a) = run(909);
        let (pred_b, scores_b) = run(909);
        assert_eq!(pred_a, pred_b);
        assert_eq!(scores_a.len(), scores_b.len());
        for (x, y) in scores_a.iter().zip(scores_b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "scores must be bit-identical");
        }
    }
}
