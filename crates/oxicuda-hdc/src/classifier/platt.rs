//! Confidence-calibrated cosine output via Platt scaling (1-D logistic calibration).
//!
//! A hyperdimensional (HD) classifier scores a query against each class prototype
//! with a cosine similarity in `[-1, 1]`. That raw score is a useful *ranking*
//! signal but is **not** a calibrated posterior probability: a cosine of `0.7`
//! does not mean "70 % chance of being this class". Platt scaling fixes this by
//! fitting a one-dimensional logistic (sigmoid) link from raw score `s` to a
//! probability, learned on a held-out set of `(score, binary_label)` pairs.
//!
//! # Model
//!
//! Following Platt's original parameterization (Platt, 1999) we fit two scalars
//! `A` and `B` such that
//!
//! ```text
//! P(y = 1 | s) = 1 / (1 + exp(A * s + B)).
//! ```
//!
//! Note the **plus** sign inside the exponent: this is Platt's convention, not
//! the more common `1 / (1 + exp(-(w*s + c)))`. A direct consequence is that for
//! a *well-behaved* score where a larger `s` should mean a larger
//! `P(y = 1 | s)`, the fitted slope `A` comes out **negative** (so that
//! `A * s + B` *decreases* as `s` grows, pushing the sigmoid toward 1). This
//! sign convention is documented and respected throughout: [`PlattScaler::a`]
//! returns the raw fitted `A` (negative for a normally-oriented classifier), and
//! [`PlattScaler::predict_proba`] computes the numerically stable sigmoid of
//! `-(A * s + B)` so that probability *increases* with score.
//!
//! # Fitting (Lin–Lin–Weng, 2007)
//!
//! Naively maximizing the Bernoulli log-likelihood with hard `0/1` targets
//! overfits and can diverge when the data are separable. We therefore use the
//! safe-target, Newton-with-backtracking procedure of Lin, Lin & Weng (2007),
//! which is the algorithm shipped in LIBSVM. Given `n_pos` positive and `n_neg`
//! negative examples we replace the hard labels by smoothed targets
//!
//! ```text
//! t_i = (n_pos + 1) / (n_pos + 2)   for a positive example, and
//! t_i = 1 / (n_neg + 2)             for a negative example,
//! ```
//!
//! and minimize the regularized cross-entropy objective
//!
//! ```text
//! F(A, B) = - sum_i [ t_i * log(p_i) + (1 - t_i) * log(1 - p_i) ],
//! ```
//!
//! with `p_i = 1 / (1 + exp(A * s_i + B))`. The minimization is Newton's method:
//! at each step we assemble the `2 x 2` Hessian `H` and gradient `g`, solve
//! `H d = -g`, and accept a step length `lambda` (halved by backtracking line
//! search until the objective decreases). Iteration stops when the gradient norm
//! falls below the requested tolerance or the line-search step underflows.
//!
//! All sums in the gradient/Hessian and the objective are evaluated in the
//! branch-free numerically stable form (computing `p` from `exp(-|fApB|)`) so the
//! routine never overflows for large `|A * s + B|`.
//!
//! # References
//!
//! - J. C. Platt, "Probabilistic Outputs for Support Vector Machines and
//!   Comparisons to Regularized Likelihood Methods", in *Advances in Large
//!   Margin Classifiers*, MIT Press, 1999, pp. 61–74.
//! - H.-T. Lin, C.-J. Lin & R. C. Weng, "A Note on Platt's Probabilistic Outputs
//!   for Support Vector Machines", *Machine Learning* 68(3):267–276, 2007.
//!
//! # Example
//!
//! ```
//! use oxicuda_hdc::classifier::platt::{fit, PlattConfig};
//! use oxicuda_hdc::error::HdcResult;
//!
//! fn calibrate() -> HdcResult<()> {
//!     // Cleanly separable held-out scores: positives near +0.8, negatives near -0.8.
//!     let scores = [
//!         0.82f32, 0.78, 0.80, 0.79, 0.81, -0.79, -0.81, -0.80, -0.78, -0.82,
//!     ];
//!     let labels = [
//!         true, true, true, true, true, false, false, false, false, false,
//!     ];
//!     let scaler = fit(&scores, &labels, &PlattConfig::default())?;
//!
//!     // A high raw score now maps to a high *probability*.
//!     assert!(scaler.predict_proba(0.8) > 0.8);
//!     assert!(scaler.predict_proba(-0.8) < 0.2);
//!     Ok(())
//! }
//! calibrate().expect("calibration example");
//! ```

use crate::error::{HdcError, HdcResult};

/// Configuration for the Platt-scaling Newton optimizer.
///
/// The defaults mirror the constants used by LIBSVM's `sigmoid_train`
/// (`max_iter = 100`, `min_step = 1e-10`, `tol = 1e-5`) and are appropriate for
/// the modestly sized held-out calibration sets typical of HD classifiers.
#[derive(Debug, Clone)]
pub struct PlattConfig {
    /// Maximum number of Newton iterations before giving up (must be `>= 1`).
    pub max_iter: usize,
    /// Minimum line-search step length; below this the backtracking search is
    /// considered to have stalled and the current iterate is accepted.
    pub min_step: f64,
    /// Convergence tolerance on the gradient infinity-style norm; iteration stops
    /// once both gradient components are smaller than this in magnitude.
    pub tol: f64,
}

impl Default for PlattConfig {
    /// LIBSVM-compatible defaults: `max_iter = 100`, `min_step = 1e-10`,
    /// `tol = 1e-5`.
    fn default() -> Self {
        Self {
            max_iter: 100,
            min_step: 1e-10,
            tol: 1e-5,
        }
    }
}

impl PlattConfig {
    /// Build a configuration from an iteration cap and a gradient tolerance,
    /// keeping the default `min_step`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `max_iter == 0` (at least one Newton step
    ///   must be permitted).
    /// - [`HdcError::InvalidProbability`] if `tol` is not a finite, strictly
    ///   positive number.
    pub fn new(max_iter: usize, tol: f64) -> HdcResult<Self> {
        if max_iter == 0 {
            return Err(HdcError::EmptyInput);
        }
        if !tol.is_finite() || tol <= 0.0 {
            return Err(HdcError::InvalidProbability(tol));
        }
        Ok(Self {
            max_iter,
            min_step: 1e-10,
            tol,
        })
    }
}

/// A fitted one-dimensional Platt calibrator.
///
/// Holds the two scalars of the logistic link `P(y = 1 | s) = 1 / (1 + exp(a*s + b))`.
/// Build one with [`fit`] and apply it with [`PlattScaler::predict_proba`] /
/// [`PlattScaler::predict_proba_many`].
///
/// Recall the sign convention (see the module docs): for a normally oriented
/// classifier `a` is **negative**, because a higher score must drive the
/// `exp(a*s + b)` term toward zero so the probability tends to one.
#[derive(Debug, Clone)]
pub struct PlattScaler {
    /// Fitted slope `A` of the logistic link (negative for a normal classifier).
    a: f64,
    /// Fitted intercept `B` of the logistic link.
    b: f64,
}

impl PlattScaler {
    /// The fitted slope `A` of `1 / (1 + exp(A*s + B))`.
    ///
    /// For a classifier where higher scores indicate the positive class this is
    /// negative.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// The fitted intercept `B` of `1 / (1 + exp(A*s + B))`.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Map a single raw score to a calibrated probability in `[0, 1]`.
    ///
    /// Computes the numerically stable sigmoid of `-(a*score + b)`. Equivalently
    /// this is `1 / (1 + exp(a*score + b))` evaluated without overflow: for the
    /// non-negative branch of the exponent argument we use `exp(-x)/(1+exp(-x))`,
    /// for the negative branch `1/(1+exp(x))`. The result is finally clamped to
    /// `[0, 1]` to absorb any floating-point rounding at the extremes.
    pub fn predict_proba(&self, score: f32) -> f32 {
        // f_ap_b = A*s + B is the exponent argument of Platt's parameterization.
        let f_ap_b = self.a * (score as f64) + self.b;
        // Stable sigmoid of -(A*s + B) so that probability rises with the score.
        let p = if f_ap_b >= 0.0 {
            // P = 1/(1+exp(f)) = exp(-f)/(1+exp(-f)); exp(-f) in (0, 1].
            let e = (-f_ap_b).exp();
            e / (1.0 + e)
        } else {
            // P = 1/(1+exp(f)); exp(f) in (0, 1).
            1.0 / (1.0 + f_ap_b.exp())
        };
        (p as f32).clamp(0.0, 1.0)
    }

    /// Map a slice of raw scores to calibrated probabilities, preserving order.
    ///
    /// The returned vector has exactly the same length as `scores`; element `i`
    /// is `self.predict_proba(scores[i])`.
    pub fn predict_proba_many(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.predict_proba(s)).collect()
    }
}

/// Evaluate `p = 1 / (1 + exp(f_ap_b))` in a branch-free, overflow-safe way.
///
/// Returns the pair `(p, one_minus_p)` so callers building the gradient and
/// Hessian avoid the catastrophic cancellation of `1.0 - p` near the extremes.
#[inline]
fn stable_p(f_ap_b: f64) -> (f64, f64) {
    if f_ap_b >= 0.0 {
        let e = (-f_ap_b).exp(); // in (0, 1]
        let p = e / (1.0 + e); // = 1/(1+exp(f))
        let one_minus_p = 1.0 / (1.0 + e); // = exp(f)/(1+exp(f))
        (p, one_minus_p)
    } else {
        let e = f_ap_b.exp(); // in (0, 1)
        let p = 1.0 / (1.0 + e);
        let one_minus_p = e / (1.0 + e);
        (p, one_minus_p)
    }
}

/// Compute the regularized cross-entropy objective `F(A, B)` in stable form.
///
/// `F = - sum_i [ t_i*log(p_i) + (1 - t_i)*log(1 - p_i) ]`, rewritten as
/// `sum_i [ t_i * f_i + log(1 + exp(-f_i)) ]` with `f_i = A*s_i + B`, which is the
/// algebraically identical but overflow-free expression used by Lin–Lin–Weng.
fn objective(scores: &[f32], targets: &[f64], a: f64, b: f64) -> f64 {
    let mut acc = 0.0f64;
    for (&s, &t) in scores.iter().zip(targets.iter()) {
        let f = a * (s as f64) + b;
        // log(1 + exp(-f)) computed stably for either sign of f.
        let log_term = if f >= 0.0 {
            (1.0 + (-f).exp()).ln()
        } else {
            -f + (1.0 + f.exp()).ln()
        };
        acc += t * f + log_term;
    }
    acc
}

/// Fit a Platt calibrator from held-out `(score, label)` pairs.
///
/// Implements the Lin–Lin–Weng (2007) safe-target Newton procedure described in
/// the module documentation. `labels[i] == true` marks a positive example,
/// `false` a negative one.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `scores` is empty.
/// - [`HdcError::DimensionMismatch`] if `scores` and `labels` differ in length.
///
/// # Degenerate inputs
///
/// All-positive or all-negative label sets are handled gracefully: the safe
/// targets keep the objective finite and the optimizer converges to a calibrator
/// that returns near-1 (resp. near-0) probabilities. The routine never panics.
pub fn fit(scores: &[f32], labels: &[bool], cfg: &PlattConfig) -> HdcResult<PlattScaler> {
    if scores.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if scores.len() != labels.len() {
        return Err(HdcError::DimensionMismatch {
            expected: scores.len(),
            got: labels.len(),
        });
    }

    // --- Prior counts and Lin–Lin–Weng safe targets ---------------------------
    let mut n_pos = 0usize;
    for &l in labels {
        if l {
            n_pos += 1;
        }
    }
    let n_neg = labels.len() - n_pos;

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;
    // Smoothed targets: positives -> (n_pos+1)/(n_pos+2), negatives -> 1/(n_neg+2).
    let hi_target = (n_pos_f + 1.0) / (n_pos_f + 2.0);
    let lo_target = 1.0 / (n_neg_f + 2.0);
    let targets: Vec<f64> = labels
        .iter()
        .map(|&l| if l { hi_target } else { lo_target })
        .collect();

    // --- Initialization (Lin–Lin–Weng) ---------------------------------------
    let mut a = 0.0f64;
    // B = log((n_neg + 1)/(n_pos + 1)); finite for any counts because of the +1.
    let mut b = ((n_neg_f + 1.0) / (n_pos_f + 1.0)).ln();

    let mut fval = objective(scores, &targets, a, b);

    // Tiny ridge on the Hessian for invertibility, as in LIBSVM's sigmoid_train.
    let sigma = 1e-12f64;

    for _ in 0..cfg.max_iter {
        // Assemble gradient g = (g0, g1) and Hessian H = [[h11, h12], [h12, h22]].
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h12 = 0.0f64;
        let mut g0 = 0.0f64;
        let mut g1 = 0.0f64;

        for (&s, &t) in scores.iter().zip(targets.iter()) {
            let sf = s as f64;
            let f = a * sf + b;
            let (p, q) = stable_p(f); // q = 1 - p
            // d2 = p * (1 - p) is the per-point Bernoulli variance (Hessian curvature).
            let d2 = p * q;
            // First derivative of the per-point objective wrt f = A*s + B:
            //   F_i = t*f + log(1 + exp(-f))  =>  dF_i/df = t - p,
            // where p = 1/(1 + exp(f)) (note exp(-f)/(1+exp(-f)) = p). This is the
            // LIBSVM `sigmoid_train` accumulation.
            let d1 = t - p;
            h11 += sf * sf * d2;
            h22 += d2;
            h12 += sf * d2;
            g0 += sf * d1;
            g1 += d1;
        }

        // Convergence test on the gradient.
        if g0.abs() < cfg.tol && g1.abs() < cfg.tol {
            break;
        }

        // Solve the 2x2 Newton system H * d = -g for the search direction d.
        let det = h11 * h22 - h12 * h12;
        if det == 0.0 || !det.is_finite() {
            // Singular curvature (e.g. all scores identical): nothing more to do.
            break;
        }
        // d = -H^{-1} g.
        let d_a = -(h22 * g0 - h12 * g1) / det;
        let d_b = -(h11 * g1 - h12 * g0) / det;
        // Directional derivative g·d (negative for a descent direction).
        let gd = g0 * d_a + g1 * d_b;

        // --- Backtracking line search --------------------------------------
        let mut step = 1.0f64;
        let mut improved = false;
        while step >= cfg.min_step {
            let new_a = a + step * d_a;
            let new_b = b + step * d_b;
            let new_f = objective(scores, &targets, new_a, new_b);
            // Sufficient-decrease (Armijo, c1 = 1e-4) acceptance.
            if new_f < fval + 1e-4 * step * gd {
                a = new_a;
                b = new_b;
                fval = new_f;
                improved = true;
                break;
            }
            step *= 0.5;
        }

        if !improved {
            // Line search underflowed past min_step: accept current iterate.
            break;
        }
    }

    Ok(PlattScaler { a, b })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference logistic probability for a known `(A, B)` directly from Platt's
    /// parameterization, used to cross-check `predict_proba`.
    fn reference_proba(a: f64, b: f64, s: f64) -> f64 {
        1.0 / (1.0 + (a * s + b).exp())
    }

    #[test]
    fn config_new_validates_max_iter() {
        assert!(PlattConfig::new(0, 1e-5).is_err());
        assert!(PlattConfig::new(1, 1e-5).is_ok());
        let cfg = PlattConfig::new(42, 1e-4).expect("valid config");
        assert_eq!(cfg.max_iter, 42);
        assert!((cfg.tol - 1e-4).abs() < 1e-18);
    }

    #[test]
    fn config_new_validates_tol() {
        assert!(PlattConfig::new(10, 0.0).is_err());
        assert!(PlattConfig::new(10, -1.0).is_err());
        assert!(PlattConfig::new(10, f64::NAN).is_err());
        assert!(PlattConfig::new(10, f64::INFINITY).is_err());
        assert!(PlattConfig::new(10, 1e-8).is_ok());
    }

    #[test]
    fn config_default_matches_libsvm() {
        let cfg = PlattConfig::default();
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.min_step - 1e-10).abs() < 1e-18);
        assert!((cfg.tol - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn fit_separable_calibrates_and_slope_is_negative() {
        // Positives clustered near +0.8, negatives near -0.8 (cleanly separable).
        let scores = [
            0.82f32, 0.78, 0.80, 0.79, 0.81, -0.79, -0.81, -0.80, -0.78, -0.82,
        ];
        let labels = [
            true, true, true, true, true, false, false, false, false, false,
        ];
        let scaler = fit(&scores, &labels, &PlattConfig::default()).expect("fit");

        // Sign convention: A must be negative so higher score -> higher P(positive).
        assert!(
            scaler.a() < 0.0,
            "expected negative slope, got a = {}",
            scaler.a()
        );

        let p_pos = scaler.predict_proba(0.8);
        let p_neg = scaler.predict_proba(-0.8);
        assert!(p_pos > 0.8, "positive score proba too low: {p_pos}");
        assert!(p_neg < 0.2, "negative score proba too high: {p_neg}");

        // predict_proba must agree with the closed-form logistic for the fit (A,B).
        let expected_pos = reference_proba(scaler.a(), scaler.b(), 0.8);
        assert!(
            ((p_pos as f64) - expected_pos).abs() < 1e-5,
            "predict_proba disagrees with closed form: {p_pos} vs {expected_pos}"
        );
    }

    #[test]
    fn predict_proba_is_monotonic_in_score() {
        let scores = [0.9f32, 0.6, 0.7, -0.6, -0.9, 0.8, -0.7, 0.5, -0.5, -0.8];
        let labels = [
            true, true, true, false, false, true, false, true, false, false,
        ];
        let scaler = fit(&scores, &labels, &PlattConfig::default()).expect("fit");

        let grid: [f32; 11] = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let probs = scaler.predict_proba_many(&grid);
        for w in probs.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "probability not monotonically non-decreasing: {} then {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn proba_always_in_unit_interval() {
        let scores = [0.95f32, -0.95, 0.10, -0.10, 0.50, -0.50];
        let labels = [true, false, true, false, true, false];
        let scaler = fit(&scores, &labels, &PlattConfig::default()).expect("fit");

        // Probe extreme and intermediate scores, including values outside [-1, 1].
        let probe: [f32; 9] = [-100.0, -1.0, -0.3, 0.0, 0.3, 1.0, 5.0, 100.0, -5.0];
        for &s in &probe {
            let p = scaler.predict_proba(s);
            assert!(
                (0.0..=1.0).contains(&p),
                "probability {p} out of [0, 1] for score {s}"
            );
            assert!(p.is_finite(), "probability not finite for score {s}");
        }
    }

    #[test]
    fn degenerate_all_positive_gives_high_proba() {
        let scores = [0.7f32, 0.6, 0.8, 0.65, 0.75];
        let labels = [true, true, true, true, true];
        let scaler = fit(&scores, &labels, &PlattConfig::default()).expect("fit");
        // With no negatives the safe target is (n_pos+1)/(n_pos+2) ~= 0.857.
        let p = scaler.predict_proba(0.7);
        assert!(
            p > 0.5,
            "all-positive set should yield a high probability, got {p}"
        );
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    #[test]
    fn degenerate_all_negative_gives_low_proba() {
        let scores = [-0.7f32, -0.6, -0.8, -0.65, -0.75];
        let labels = [false, false, false, false, false];
        let scaler = fit(&scores, &labels, &PlattConfig::default()).expect("fit");
        let p = scaler.predict_proba(-0.7);
        assert!(
            p < 0.5,
            "all-negative set should yield a low probability, got {p}"
        );
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    #[test]
    fn fit_rejects_length_mismatch() {
        let scores = [0.1f32, 0.2, 0.3];
        let labels = [true, false];
        match fit(&scores, &labels, &PlattConfig::default()) {
            Err(HdcError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn fit_rejects_empty_input() {
        let scores: [f32; 0] = [];
        let labels: [bool; 0] = [];
        assert!(matches!(
            fit(&scores, &labels, &PlattConfig::default()),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn predict_proba_many_preserves_length() {
        let scores = [0.3f32, -0.3, 0.9, -0.9];
        let labels = [true, false, true, false];
        let scaler = fit(&scores, &labels, &PlattConfig::default()).expect("fit");

        let query = [0.0f32, 0.5, -0.5, 0.25, -0.25, 1.0];
        let out = scaler.predict_proba_many(&query);
        assert_eq!(out.len(), query.len());
        for (i, (&s, &p)) in query.iter().zip(out.iter()).enumerate() {
            assert!(
                (p - scaler.predict_proba(s)).abs() < 1e-12,
                "batch element {i} disagrees with scalar predict_proba"
            );
        }
    }

    #[test]
    fn fit_is_deterministic() {
        let scores = [0.4f32, -0.4, 0.9, -0.7, 0.2, -0.2, 0.6, -0.9, 0.1, -0.5];
        let labels = [
            true, false, true, false, true, false, true, false, true, false,
        ];
        let cfg = PlattConfig::default();
        let s1 = fit(&scores, &labels, &cfg).expect("fit 1");
        let s2 = fit(&scores, &labels, &cfg).expect("fit 2");
        assert_eq!(
            s1.a().to_bits(),
            s2.a().to_bits(),
            "slope not bit-identical"
        );
        assert_eq!(
            s1.b().to_bits(),
            s2.b().to_bits(),
            "intercept not bit-identical"
        );
    }

    #[test]
    fn single_point_fit_does_not_panic() {
        // Minimal viable input: one example. Safe targets keep it well-defined.
        let scores = [0.5f32];
        let labels = [true];
        let scaler = fit(&scores, &labels, &PlattConfig::new(50, 1e-6).expect("cfg"))
            .expect("single-point fit");
        let p = scaler.predict_proba(0.5);
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }
}
