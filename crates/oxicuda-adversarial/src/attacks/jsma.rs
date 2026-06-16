//! Jacobian-based Saliency Map Attack (JSMA).
//!
//! Targeted, sparse (L0) adversarial attack from
//! Papernot, McDaniel, Jha, Fredrikson, Celik & Swami (2016),
//! *"The Limitations of Deep Learning in Adversarial Settings"*, EuroS&P.
//!
//! JSMA constructs an *adversarial saliency map* from the forward Jacobian
//! `∂logit_c / ∂x_f` of the classifier and greedily perturbs the input
//! feature(s) that most increase the chosen target class' logit while
//! decreasing the sum of all other class logits. Because only a small number
//! of input features are touched, the resulting perturbation has a small
//! L0 (sparsity) norm rather than a small L∞ / L2 norm.
//!
//! # Saliency map (increasing variant)
//!
//! For target class `t` and feature `f` define
//!
//! ```text
//! alpha_f = ∂logit_t / ∂x_f                       (target sensitivity)
//! beta_f  = Σ_{c ≠ t} ∂logit_c / ∂x_f             (rest sensitivity)
//! ```
//!
//! A feature is *salient for increasing the target logit* only when
//! `alpha_f > 0` (increasing `x_f` raises the target logit) **and**
//! `beta_f < 0` (increasing `x_f` lowers the other logits). The saliency
//! score is then `alpha_f · |beta_f|`; otherwise the score is `0`. The most
//! salient unsaturated feature is moved by `+theta` and clamped, mirroring
//! the increasing-perturbation single-feature variant of the original paper.
//!
//! # Model interface
//!
//! Following the crate convention (see [`crate::attacks::deepfool()`]) the
//! classifier is supplied as closures rather than a trait object:
//!
//! * `jacobian: Fn(&[f32]) -> Vec<f32>` returns the flattened
//!   `n_classes × n_features` row-major Jacobian `∂logit_c / ∂x_f`.
//! * `predict:  Fn(&[f32]) -> usize` returns the argmax (predicted class).

use crate::error::{AdvError, AdvResult};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Hyperparameters for [`Jsma`].
///
/// Construct with [`Jsma::new`] which validates every field up-front.
#[derive(Debug, Clone, Copy)]
pub struct JsmaConfig {
    /// Per-step perturbation magnitude added to the chosen feature
    /// (must be `> 0` and finite).
    pub theta: f32,
    /// Maximum fraction of input features that may be modified, in `(0, 1]`.
    pub gamma: f32,
    /// Box lower bound (inclusive) applied after each perturbation.
    pub clamp_min: f32,
    /// Box upper bound (inclusive) applied after each perturbation.
    pub clamp_max: f32,
    /// Number of output classes (must be `>= 2`).
    pub n_classes: usize,
    /// Target class index the attack drives the prediction toward
    /// (must satisfy `target_class < n_classes`).
    pub target_class: usize,
}

// ─── Attack ──────────────────────────────────────────────────────────────────

/// Jacobian-based Saliency Map Attack.
#[derive(Debug, Clone)]
pub struct Jsma {
    cfg: JsmaConfig,
}

impl Jsma {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidAlpha`]      — non-finite or non-positive `theta`.
    /// * [`AdvError::InvalidLossWeight`] — degenerate box (`clamp_min >= clamp_max`
    ///   or non-finite bounds).
    /// * [`AdvError::Internal`]          — `gamma` outside `(0, 1]`, `n_classes < 2`,
    ///   or `target_class >= n_classes`.
    pub fn new(cfg: JsmaConfig) -> AdvResult<Self> {
        if !(cfg.theta.is_finite() && cfg.theta > 0.0) {
            return Err(AdvError::InvalidAlpha { alpha: cfg.theta });
        }
        if !(cfg.gamma.is_finite() && cfg.gamma > 0.0 && cfg.gamma <= 1.0) {
            return Err(AdvError::Internal(
                "jsma: gamma must be in (0, 1]".to_owned(),
            ));
        }
        if !(cfg.clamp_min.is_finite() && cfg.clamp_max.is_finite())
            || cfg.clamp_min >= cfg.clamp_max
        {
            return Err(AdvError::InvalidLossWeight {
                weight: cfg.clamp_max - cfg.clamp_min,
            });
        }
        if cfg.n_classes < 2 {
            return Err(AdvError::Internal(
                "jsma: n_classes must be >= 2".to_owned(),
            ));
        }
        if cfg.target_class >= cfg.n_classes {
            return Err(AdvError::Internal(
                "jsma: target_class must be < n_classes".to_owned(),
            ));
        }
        Ok(Self { cfg })
    }

    /// Borrow the validated configuration.
    #[must_use]
    pub fn config(&self) -> &JsmaConfig {
        &self.cfg
    }

    /// Compute the increasing-variant adversarial saliency map for the target
    /// class given a flattened `n_classes × n_features` row-major Jacobian.
    ///
    /// The returned vector has length `n_features`; entry `f` holds
    /// `alpha_f · |beta_f|` when `alpha_f > 0 && beta_f < 0`, else `0`.
    ///
    /// # Errors
    /// * [`AdvError::Internal`]          — `n_features == 0`.
    /// * [`AdvError::DimensionMismatch`] — `jac.len() != n_classes * n_features`.
    /// * [`AdvError::NanEncountered`]    — a Jacobian entry is non-finite.
    pub fn saliency_map(&self, jac: &[f32], n_features: usize) -> AdvResult<Vec<f32>> {
        if n_features == 0 {
            return Err(AdvError::Internal(
                "jsma: n_features must be > 0".to_owned(),
            ));
        }
        let expected = self.cfg.n_classes * n_features;
        if jac.len() != expected {
            return Err(AdvError::DimensionMismatch {
                expected,
                got: jac.len(),
            });
        }
        if jac.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "jsma:saliency_map",
            });
        }

        let target = self.cfg.target_class;
        let target_row = target * n_features;
        let mut scores = vec![0.0_f32; n_features];

        for (f, score) in scores.iter_mut().enumerate() {
            // alpha = ∂logit_target / ∂x_f.
            let alpha = jac[target_row + f];
            // beta = Σ_{c ≠ target} ∂logit_c / ∂x_f.
            let mut beta = 0.0_f32;
            for c in 0..self.cfg.n_classes {
                if c == target {
                    continue;
                }
                beta += jac[c * n_features + f];
            }
            // Salient only when raising x_f raises the target and lowers the rest.
            *score = if alpha > 0.0 && beta < 0.0 {
                alpha * beta.abs()
            } else {
                0.0
            };
        }
        Ok(scores)
    }

    /// Run the JSMA targeted attack on flat input `input`.
    ///
    /// Loop: if `predict(x) == target_class` stop; otherwise compute the
    /// Jacobian, derive the saliency map, pick the most-salient feature that
    /// has not yet saturated at `clamp_max`, add `+theta` to it, clamp into
    /// `[clamp_min, clamp_max]`, and mark it modified. The loop stops once the
    /// modified-feature count would exceed `ceil(gamma · n_features)` or no
    /// positive-saliency unsaturated feature remains.
    ///
    /// Returns the (possibly unchanged) adversarial example, same length as
    /// `input`.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]        — `input.is_empty()`.
    /// * [`AdvError::DimensionMismatch`] — `jacobian(x)` has the wrong length.
    /// * [`AdvError::NanEncountered`]    — a Jacobian entry is non-finite.
    pub fn attack<J, P>(&self, input: &[f32], jacobian: J, predict: P) -> AdvResult<Vec<f32>>
    where
        J: Fn(&[f32]) -> Vec<f32>,
        P: Fn(&[f32]) -> usize,
    {
        if input.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        let n_features = input.len();
        // Feature budget: number of features we are allowed to modify.
        let max_features = feature_budget(self.cfg.gamma, n_features);

        let mut x = input.to_vec();
        // Track which features have already been picked (each modified once in
        // the single-feature increasing variant) so we never re-select them.
        let mut modified = vec![false; n_features];
        let mut modified_count = 0_usize;

        loop {
            // Success: prediction already matches the target class.
            if predict(&x) == self.cfg.target_class {
                break;
            }
            // Feature budget exhausted.
            if modified_count >= max_features {
                break;
            }

            let jac = jacobian(&x);
            let scores = self.saliency_map(&jac, n_features)?;

            // Pick the highest-scoring feature that is neither already modified
            // nor saturated at the upper clamp (so +theta can still move it).
            let mut best_idx: Option<usize> = None;
            let mut best_score = 0.0_f32;
            for (f, &s) in scores.iter().enumerate() {
                if modified[f] {
                    continue;
                }
                if x[f] >= self.cfg.clamp_max {
                    continue;
                }
                if s > best_score {
                    best_score = s;
                    best_idx = Some(f);
                }
            }

            match best_idx {
                Some(f) => {
                    // Increasing perturbation toward the target class.
                    x[f] = (x[f] + self.cfg.theta).clamp(self.cfg.clamp_min, self.cfg.clamp_max);
                    modified[f] = true;
                    modified_count += 1;
                }
                // No positive-saliency unsaturated feature remains.
                None => break,
            }
        }

        Ok(x)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Number of features that may be modified: `ceil(gamma · n_features)`,
/// clamped to `[1, n_features]` so at least one feature is always eligible
/// for a non-empty input with positive `gamma`.
#[inline]
fn feature_budget(gamma: f32, n_features: usize) -> usize {
    let raw = (gamma * n_features as f32).ceil();
    let budget = if raw < 1.0 { 1 } else { raw as usize };
    budget.min(n_features)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Default valid config: 3 classes, target class 0, generous box.
    fn cfg_target0() -> JsmaConfig {
        JsmaConfig {
            theta: 0.1,
            gamma: 1.0,
            clamp_min: 0.0,
            clamp_max: 1.0,
            n_classes: 3,
            target_class: 0,
        }
    }

    /// Build a constant-Jacobian closure (linear model) of shape
    /// `n_classes × n_features` row-major.
    fn const_jac(jac: Vec<f32>) -> impl Fn(&[f32]) -> Vec<f32> {
        move |_x: &[f32]| jac.clone()
    }

    // ── saliency_map shape & non-negativity ───────────────────────────────────

    #[test]
    fn saliency_map_length_matches_features() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let n_features = 4;
        // 3 classes × 4 features.
        let jac = vec![0.0_f32; 3 * n_features];
        let s = jsma
            .saliency_map(&jac, n_features)
            .expect("saliency_map should succeed");
        assert_eq!(s.len(), n_features);
    }

    #[test]
    fn saliency_map_non_negative() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let n_features = 3;
        // Arbitrary mixed-sign Jacobian.
        let jac = vec![
            0.5_f32, -0.2, 0.9, // target (class 0)
            -0.3, 0.4, -0.7, // class 1
            0.1, -0.8, 0.2, // class 2
        ];
        let s = jsma
            .saliency_map(&jac, n_features)
            .expect("saliency_map should succeed");
        assert!(s.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn saliency_zero_when_alpha_nonpositive() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let n_features = 2;
        // Feature 0: alpha = -1 (≤ 0) → score 0 even though beta < 0.
        let jac = vec![
            -1.0_f32, 1.0, // target row
            -1.0_f32, -1.0, // class 1
            -1.0_f32, -1.0, // class 2
        ];
        let s = jsma
            .saliency_map(&jac, n_features)
            .expect("saliency_map should succeed");
        assert!((s[0]).abs() < 1e-9);
    }

    #[test]
    fn saliency_zero_when_beta_nonnegative() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let n_features = 2;
        // Feature 1: alpha = 1 (> 0) but beta = +2 (≥ 0) → score 0.
        let jac = vec![
            1.0_f32, 1.0, // target row
            1.0_f32, 1.0, // class 1
            1.0_f32, 1.0, // class 2
        ];
        let s = jsma
            .saliency_map(&jac, n_features)
            .expect("saliency_map should succeed");
        assert!((s[1]).abs() < 1e-9);
    }

    #[test]
    fn saliency_score_is_alpha_times_abs_beta() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let n_features = 1;
        // alpha = 2, beta = (-1) + (-3) = -4 → score = 2 * 4 = 8.
        let jac = vec![
            2.0_f32,  // target
            -1.0_f32, // class 1
            -3.0_f32, // class 2
        ];
        let s = jsma
            .saliency_map(&jac, n_features)
            .expect("saliency_map should succeed");
        assert!((s[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn saliency_picks_most_salient_feature() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let n_features = 3;
        // Feature 2 has the strongest (alpha>0, beta<0) signal.
        let jac = vec![
            0.2_f32, 0.5, 1.0, // target row (alpha)
            -0.1, -0.5, -1.0, // class 1
            -0.1, -0.5, -1.0, // class 2
        ];
        let s = jsma
            .saliency_map(&jac, n_features)
            .expect("saliency_map should succeed");
        let argmax = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .expect("value should be present");
        assert_eq!(argmax, 2);
    }

    // ── attack output / clamping / monotonicity ───────────────────────────────

    #[test]
    fn attack_output_length_equals_input() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.5_f32; 5];
        // Jacobian: feature 0 salient for target. Never predicts target → runs.
        let jac = vec![
            1.0_f32, 0.0, 0.0, 0.0, 0.0, // target
            -1.0, 0.0, 0.0, 0.0, 0.0, // class 1
            -1.0, 0.0, 0.0, 0.0, 0.0, // class 2
        ];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 1_usize)
            .expect("value should be present");
        assert_eq!(y.len(), input.len());
    }

    #[test]
    fn attack_clamps_within_box() {
        let cfg = JsmaConfig {
            theta: 0.5,
            ..cfg_target0()
        };
        let jsma = Jsma::new(cfg).expect("new should succeed");
        let input = vec![0.9_f32; 3];
        // All features salient for the target.
        let jac = vec![
            1.0_f32, 1.0, 1.0, // target
            -1.0, -1.0, -1.0, // class 1
            -1.0, -1.0, -1.0, // class 2
        ];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 1_usize)
            .expect("value should be present");
        for &v in &y {
            assert!((0.0..=1.0).contains(&v), "out of [0,1]: {v}");
        }
    }

    #[test]
    fn attack_increases_feature_zero_on_linear_model() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.1_f32; 4];
        // Only feature 0 has positive target gradient and negative rest gradient.
        let jac = vec![
            1.0_f32, 0.0, 0.0, 0.0, // target
            -1.0, 0.0, 0.0, 0.0, // class 1
            -1.0, 0.0, 0.0, 0.0, // class 2
        ];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 2_usize)
            .expect("value should be present");
        assert!(y[0] > input[0], "feature 0 should increase");
        // Other features untouched.
        for f in 1..4 {
            assert!((y[f] - input[f]).abs() < 1e-9);
        }
    }

    #[test]
    fn attack_monotone_perturbation_direction() {
        // Every modified feature only ever moves up by +theta (never down).
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.2_f32, 0.2, 0.2];
        let jac = vec![
            1.0_f32, 1.0, 1.0, // target
            -1.0, -1.0, -1.0, // class 1
            -1.0, -1.0, -1.0, // class 2
        ];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 1_usize)
            .expect("value should be present");
        for (a, b) in y.iter().zip(input.iter()) {
            assert!(*a >= *b - 1e-9, "feature moved downward");
        }
    }

    #[test]
    fn attack_stops_when_prediction_already_target() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.3_f32; 4];
        let jac = vec![1.0_f32; 3 * 4];
        // predict already returns the target class → no modification.
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 0_usize)
            .expect("value should be present");
        assert_eq!(y, input);
    }

    #[test]
    fn attack_target_already_winning_no_change() {
        // Distinct from the previous test: target is non-zero and the predictor
        // returns it immediately.
        let cfg = JsmaConfig {
            target_class: 2,
            ..cfg_target0()
        };
        let jsma = Jsma::new(cfg).expect("new should succeed");
        let input = vec![0.4_f32, 0.6, 0.1];
        let jac = vec![1.0_f32; 3 * 3];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 2_usize)
            .expect("value should be present");
        assert_eq!(y, input);
    }

    #[test]
    fn attack_respects_feature_budget() {
        // gamma = 0.5 over 4 features → at most ceil(2) = 2 modified.
        let cfg = JsmaConfig {
            gamma: 0.5,
            theta: 0.1,
            ..cfg_target0()
        };
        let jsma = Jsma::new(cfg).expect("new should succeed");
        let input = vec![0.1_f32; 4];
        // All features salient; predictor never reaches target so the budget
        // is the binding stop condition.
        let jac = vec![
            1.0_f32, 1.0, 1.0, 1.0, // target
            -1.0, -1.0, -1.0, -1.0, // class 1
            -1.0, -1.0, -1.0, -1.0, // class 2
        ];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 1_usize)
            .expect("value should be present");
        let changed = y
            .iter()
            .zip(input.iter())
            .filter(|(a, b)| (**a - **b).abs() > 1e-9)
            .count();
        let allowed = (0.5_f32 * 4.0).ceil() as usize;
        assert!(changed <= allowed, "changed {changed} > allowed {allowed}");
    }

    #[test]
    fn attack_stops_when_no_positive_saliency() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.5_f32; 3];
        // All-zero Jacobian → no salient feature → immediate stop, unchanged.
        let jac = vec![0.0_f32; 3 * 3];
        let y = jsma
            .attack(&input, const_jac(jac), |_x| 1_usize)
            .expect("value should be present");
        assert_eq!(y, input);
    }

    #[test]
    fn attack_deterministic_given_fixed_closures() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.1_f32, 0.2, 0.3, 0.4];
        let jac = vec![
            1.0_f32, 0.5, 0.2, 0.1, // target
            -1.0, -0.5, -0.2, -0.1, // class 1
            -1.0, -0.5, -0.2, -0.1, // class 2
        ];
        let y1 = jsma
            .attack(&input, const_jac(jac.clone()), |_x| 1_usize)
            .expect("value should be present");
        let y2 = jsma
            .attack(&input, const_jac(jac), |_x| 1_usize)
            .expect("value should be present");
        assert_eq!(y1, y2);
    }

    #[test]
    fn attack_flips_prediction_when_threshold_reached() {
        // Single-feature increasing variant: each feature is modified at most
        // once. The predictor declares the target won as soon as feature 0 has
        // been bumped above its starting value, so the attack must terminate on
        // success after exactly one modification.
        let cfg = JsmaConfig {
            theta: 0.25,
            ..cfg_target0()
        };
        let jsma = Jsma::new(cfg).expect("new should succeed");
        let input = vec![0.0_f32; 2];
        let jac = vec![
            1.0_f32, 0.0, // target
            -1.0, 0.0, // class 1
            -1.0, 0.0, // class 2
        ];
        // Target wins once feature 0 has moved off zero (a single +theta step).
        let predict = |x: &[f32]| {
            if x[0] > 1e-6 { 0_usize } else { 1_usize }
        };
        let y = jsma
            .attack(&input, const_jac(jac), predict)
            .expect("value should be present");
        // Exactly one +theta step applied to feature 0; feature 1 untouched.
        assert!((y[0] - 0.25).abs() < 1e-6, "feature 0 = {}", y[0]);
        assert!((y[1]).abs() < 1e-9);
    }

    // ── error paths ────────────────────────────────────────────────────────────

    #[test]
    fn err_theta_non_positive() {
        let cfg = JsmaConfig {
            theta: 0.0,
            ..cfg_target0()
        };
        assert!(matches!(
            Jsma::new(cfg).unwrap_err(),
            AdvError::InvalidAlpha { .. }
        ));
        let cfg_neg = JsmaConfig {
            theta: -0.1,
            ..cfg_target0()
        };
        assert!(matches!(
            Jsma::new(cfg_neg).unwrap_err(),
            AdvError::InvalidAlpha { .. }
        ));
    }

    #[test]
    fn err_gamma_zero_or_too_large() {
        let cfg0 = JsmaConfig {
            gamma: 0.0,
            ..cfg_target0()
        };
        assert!(matches!(
            Jsma::new(cfg0).unwrap_err(),
            AdvError::Internal(_)
        ));
        let cfg_big = JsmaConfig {
            gamma: 1.5,
            ..cfg_target0()
        };
        assert!(matches!(
            Jsma::new(cfg_big).unwrap_err(),
            AdvError::Internal(_)
        ));
    }

    #[test]
    fn err_degenerate_box() {
        let cfg = JsmaConfig {
            clamp_min: 1.0,
            clamp_max: 1.0,
            ..cfg_target0()
        };
        assert!(matches!(
            Jsma::new(cfg).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
        let cfg_inv = JsmaConfig {
            clamp_min: 1.0,
            clamp_max: 0.0,
            ..cfg_target0()
        };
        assert!(matches!(
            Jsma::new(cfg_inv).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
    }

    #[test]
    fn err_n_classes_too_small() {
        let cfg = JsmaConfig {
            n_classes: 1,
            target_class: 0,
            ..cfg_target0()
        };
        assert!(matches!(Jsma::new(cfg).unwrap_err(), AdvError::Internal(_)));
    }

    #[test]
    fn err_target_class_out_of_range() {
        let cfg = JsmaConfig {
            n_classes: 3,
            target_class: 3,
            ..cfg_target0()
        };
        assert!(matches!(Jsma::new(cfg).unwrap_err(), AdvError::Internal(_)));
    }

    #[test]
    fn err_jacobian_wrong_length() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        // 3 classes × 4 features expected = 12; supply 11.
        let jac = vec![0.0_f32; 11];
        assert!(matches!(
            jsma.saliency_map(&jac, 4).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn err_empty_input() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input: Vec<f32> = vec![];
        assert_eq!(
            jsma.attack(&input, const_jac(vec![]), |_x| 0_usize)
                .unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn err_saliency_zero_features() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        assert!(matches!(
            jsma.saliency_map(&[], 0).unwrap_err(),
            AdvError::Internal(_)
        ));
    }

    #[test]
    fn err_saliency_nan_jacobian() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let jac = vec![f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            jsma.saliency_map(&jac, 2).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn attack_propagates_jacobian_dim_mismatch() {
        let jsma = Jsma::new(cfg_target0()).expect("value should be present");
        let input = vec![0.5_f32; 4];
        // Jacobian closure returns the wrong length → DimensionMismatch.
        let bad = |_x: &[f32]| vec![0.0_f32; 5];
        assert!(matches!(
            jsma.attack(&input, bad, |_x| 1_usize).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn feature_budget_ceil_and_clamped() {
        assert_eq!(feature_budget(1.0, 4), 4);
        assert_eq!(feature_budget(0.5, 4), 2);
        assert_eq!(feature_budget(0.25, 4), 1);
        // Tiny gamma still allows at least one feature.
        assert_eq!(feature_budget(0.01, 4), 1);
        // Never exceeds the feature count.
        assert_eq!(feature_budget(1.0, 3), 3);
    }

    #[test]
    fn config_accessor_round_trips() {
        let cfg = cfg_target0();
        let jsma = Jsma::new(cfg).expect("new should succeed");
        assert_eq!(jsma.config().n_classes, cfg.n_classes);
        assert_eq!(jsma.config().target_class, cfg.target_class);
    }
}
