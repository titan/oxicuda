//! Doubly-Robust Policy Learner (AIPW scores + welfare-maximizing PolicyTree).
//!
//! References:
//! - Athey, S. & Wager, S. (2021). "Policy Learning with Observational Data."
//!   *Econometrica* 89(1):133–161.
//! - Zhou, Z., Athey, S. & Wager, S. (2023). "Offline Multi-Action Policy
//!   Learning: Generalization and Optimization." *Operations Research*,
//!   71(1):148–183.
//!
//! # Algorithm
//!
//! Given a multi-action observational dataset `(X_i, T_i, Y_i)` with the
//! caller's K-fold cross-fitted estimates of the **outcome model**
//! `m̂(X, a) = E[Y | X, T=a]` (`m_hat`) and the **propensity** `ê(X, a) =
//! P(T=a | X)` (`e_hat`), we form per-sample, per-action **doubly-robust
//! AIPW scores**:
//!
//! ```text
//!   Γ_i(a) = m̂(X_i, a) + ( Y_i − m̂(X_i, T_i) ) · 𝟙{T_i = a} / ê(X_i, a)
//! ```
//!
//! These are the same scores used in AIPW estimation of E[Y(a)] but tabulated
//! per-action; their consistency only requires that *either* the outcome
//! model *or* the propensity be correctly specified (the "double-robustness"
//! property), and the K-fold cross-fitting decouples the nuisance estimation
//! from the policy fit (Chernozhukov et al. 2018).
//!
//! Stage 2 fits a depth-`policy_depth` welfare-maximizing
//! [`crate::forest::policy_tree::PolicyTree`] on `(X, Γ)` — i.e. an exact
//! exhaustive search over axis-aligned splits and per-leaf action assignments
//! that maximises in-sample welfare `Σ_i Γ_{i,π(X_i)}`.  This delivers the
//! Athey-Wager (2021) regret guarantee for the depth class.
//!
//! # Propensity clipping
//!
//! Because the inverse-propensity weight `1/ê(X_i, a)` can blow up when
//! `ê` is tiny, we **clip** every propensity to `[clip, 1 − clip]` before
//! division (with a default clip ≈ 0.01).  This is standard practice; see
//! Crump-Hotz-Imbens-Mitnik 2009 and Athey-Wager 2021 §4.1.

use crate::error::{CausalError, CausalResult};
use crate::forest::policy_tree::{PolicyTree, PolicyTreeConfig};

/// Configuration for [`DrPolicy::fit`].
#[derive(Debug, Clone)]
pub struct DrPolicyConfig {
    /// Maximum depth of the welfare-maximizing policy tree.  `0` ⇒ single
    /// leaf (constant policy).
    pub policy_depth: usize,
    /// Minimum samples on each side of any candidate policy-tree split.
    /// Must be ≥ 1.
    pub min_leaf_samples: usize,
    /// Number of available actions (must be ≥ 2).
    pub n_actions: usize,
}

/// Output of [`DrPolicy::fit`].
#[derive(Debug, Clone)]
pub struct DrPolicyResult {
    /// The fitted welfare-maximizing policy tree.
    pub policy_tree: PolicyTree,
    /// The doubly-robust AIPW scores `Γ_i(a)` used to fit the tree;
    /// row-major `n_samples × n_actions`.
    pub dr_scores: Vec<f32>,
    /// In-sample empirical welfare `Σ_i Γ_{i,π(X_i)}` of the fitted tree.
    pub train_welfare: f64,
    /// Configuration used (retained for diagnostics).
    pub cfg: DrPolicyConfig,
}

/// Doubly-robust policy learner — a zero-state façade so callers can write
/// `DrPolicy::fit(...)` / `DrPolicy::predict(...)` rather than carry a
/// hand-rolled struct.
pub struct DrPolicy;

impl DrPolicy {
    /// Compute the AIPW doubly-robust scores
    ///
    /// ```text
    ///   Γ_i(a) = m̂(X_i, a) + (Y_i − m̂(X_i, T_i)) · 𝟙{T_i = a} / ê(X_i, a)
    /// ```
    ///
    /// `m_hat` and `e_hat` are row-major `n_samples × n_actions` (the caller's
    /// K-fold cross-fitted outcome and propensity predictions).  Each entry of
    /// `e_hat` is clamped to `[propensity_clip, 1 − propensity_clip]` before
    /// the inverse is taken.
    ///
    /// # Errors
    /// - [`CausalError::InvalidParameter`] when `n_actions < 2`,
    ///   `propensity_clip ∉ (0, 0.5)`, `n_samples == 0`, or a treatment index
    ///   is out of range `[0, n_actions)`.
    /// - [`CausalError::DimensionMismatch`] when any slice has the wrong
    ///   length.
    pub fn build_dr_scores(
        treatments: &[usize],
        outcomes: &[f32],
        m_hat: &[f32],
        e_hat: &[f32],
        n_samples: usize,
        n_actions: usize,
        propensity_clip: f32,
    ) -> CausalResult<Vec<f32>> {
        validate_score_inputs(
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples,
            n_actions,
            propensity_clip,
        )?;

        let mut scores = vec![0.0_f32; n_samples * n_actions];
        for i in 0..n_samples {
            let t_i = treatments[i];
            let m_t = m_hat[i * n_actions + t_i];
            let resid = outcomes[i] - m_t;
            for a in 0..n_actions {
                let m_a = m_hat[i * n_actions + a];
                let e_a_raw = e_hat[i * n_actions + a];
                // Clip propensity to bound the inverse-propensity weight.
                let e_a = clip_propensity(e_a_raw, propensity_clip);
                let indicator = if t_i == a { 1.0_f32 } else { 0.0_f32 };
                scores[i * n_actions + a] = m_a + resid * indicator / e_a;
            }
        }
        Ok(scores)
    }

    /// Build the doubly-robust scores, then fit a welfare-maximising
    /// [`PolicyTree`] of depth `cfg.policy_depth` on `(features, scores)`.
    ///
    /// `propensity_clip` defaults to `0.01` — the caller can build the
    /// scores manually with [`Self::build_dr_scores`] and an explicit clip
    /// for finer control.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] /
    /// [`CausalError::DimensionMismatch`] under the same conditions as
    /// [`Self::build_dr_scores`], plus the
    /// [`crate::forest::policy_tree::PolicyTree::fit`] errors.
    pub fn fit(
        features: &[f32],
        treatments: &[usize],
        outcomes: &[f32],
        m_hat: &[f32],
        e_hat: &[f32],
        n_samples: usize,
        n_features: usize,
        cfg: DrPolicyConfig,
    ) -> CausalResult<DrPolicyResult> {
        // Validate feature dimensions before doing any score work.
        if n_features == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "n_features must be >= 1".to_string(),
            });
        }
        if features.len() != n_samples * n_features {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_features,
                got: features.len(),
            });
        }

        let propensity_clip = DEFAULT_PROPENSITY_CLIP;
        let dr_scores = Self::build_dr_scores(
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples,
            cfg.n_actions,
            propensity_clip,
        )?;

        let pt_cfg = PolicyTreeConfig {
            depth: cfg.policy_depth,
            n_actions: cfg.n_actions,
            min_leaf_samples: cfg.min_leaf_samples,
        };
        let pt_res = PolicyTree::fit(features, &dr_scores, n_samples, n_features, &pt_cfg)?;

        Ok(DrPolicyResult {
            policy_tree: pt_res.tree,
            dr_scores,
            train_welfare: pt_res.train_welfare,
            cfg,
        })
    }

    /// Predict the action assigned to a feature vector by the fitted policy
    /// tree.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] if a split along the
    /// traversed path references a feature index out of range for `x`.
    pub fn predict(result: &DrPolicyResult, x: &[f32]) -> CausalResult<usize> {
        result.policy_tree.predict(x)
    }
}

/// Default propensity-clipping floor: cross-fitted propensities are clamped to
/// `[0.01, 0.99]` before inverting in the AIPW score.
pub const DEFAULT_PROPENSITY_CLIP: f32 = 0.01;

// =====================================================================
// validation
// =====================================================================

#[inline]
fn clip_propensity(p: f32, clip: f32) -> f32 {
    if p < clip {
        clip
    } else if p > 1.0 - clip {
        1.0 - clip
    } else {
        p
    }
}

fn validate_score_inputs(
    treatments: &[usize],
    outcomes: &[f32],
    m_hat: &[f32],
    e_hat: &[f32],
    n_samples: usize,
    n_actions: usize,
    propensity_clip: f32,
) -> CausalResult<()> {
    if n_actions < 2 {
        return Err(CausalError::InvalidParameter {
            reason: format!("n_actions must be >= 2, got {n_actions}"),
        });
    }
    if n_samples == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "n_samples must be >= 1".to_string(),
        });
    }
    if !(propensity_clip > 0.0 && propensity_clip < 0.5) {
        return Err(CausalError::InvalidParameter {
            reason: format!("propensity_clip must be in (0, 0.5), got {propensity_clip}"),
        });
    }
    if treatments.len() != n_samples {
        return Err(CausalError::DimensionMismatch {
            expected: n_samples,
            got: treatments.len(),
        });
    }
    if outcomes.len() != n_samples {
        return Err(CausalError::DimensionMismatch {
            expected: n_samples,
            got: outcomes.len(),
        });
    }
    if m_hat.len() != n_samples * n_actions {
        return Err(CausalError::DimensionMismatch {
            expected: n_samples * n_actions,
            got: m_hat.len(),
        });
    }
    if e_hat.len() != n_samples * n_actions {
        return Err(CausalError::DimensionMismatch {
            expected: n_samples * n_actions,
            got: e_hat.len(),
        });
    }
    for (i, &t) in treatments.iter().enumerate() {
        if t >= n_actions {
            return Err(CausalError::InvalidParameter {
                reason: format!("treatments[{i}] = {t} >= n_actions {n_actions}"),
            });
        }
    }
    Ok(())
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::policy_tree::PolicyNode;

    /// Container bundling the fixed dataset returned by [`hand_example_2actions`].
    struct HandExample {
        features: Vec<f32>,
        treatments: Vec<usize>,
        outcomes: Vec<f32>,
        m_hat: Vec<f32>,
        e_hat: Vec<f32>,
        n_samples: usize,
        n_features: usize,
    }

    /// Build a hand example with 4 samples × 2 actions where action 1 is best
    /// when `x[0] <= 0.5` and action 0 is best when `x[0] > 0.5`.
    fn hand_example_2actions() -> HandExample {
        // 4 samples; 1 feature.
        let n_samples = 4usize;
        let n_features = 1usize;
        let features = vec![0.1_f32, 0.2, 0.7, 0.9];
        // Use a uniform 50/50 propensity (so e_hat = 0.5 everywhere) and an
        // outcome model that mirrors the ground-truth CATE: m̂(x, 1) > m̂(x, 0)
        // for x <= 0.5, the reverse for x > 0.5.
        let m_hat = vec![
            0.0, 2.0, // sample 0: x=0.1 → a=1 better
            0.0, 2.0, // sample 1: x=0.2 → a=1 better
            2.0, 0.0, // sample 2: x=0.7 → a=0 better
            2.0, 0.0, // sample 3: x=0.9 → a=0 better
        ];
        let e_hat = vec![0.5_f32; n_samples * 2];
        // Treatments and outcomes — the outcomes are consistent with m̂.
        let treatments = vec![0usize, 1, 0, 1];
        let outcomes = vec![0.0_f32, 2.0, 2.0, 0.0];
        HandExample {
            features,
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples,
            n_features,
        }
    }

    #[test]
    fn build_dr_scores_length() {
        let HandExample {
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            ..
        } = hand_example_2actions();
        let scores = DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, n, 2, 0.01)
            .expect("build_dr_scores should succeed");
        assert_eq!(scores.len(), n * 2);
    }

    #[test]
    fn aipw_identity_treated_arm() {
        // For a row with T_i=a: Γ_i(a) = m̂_a + (Y - m̂_a) / ê_a.
        // For a' != a:           Γ_i(a') = m̂_{a'}.
        // Use the hand example, sample 0: T=0, so Γ_0(0) = 0 + (0 − 0)/0.5 = 0.
        // Γ_0(1) = 2 (just m̂_1).
        let HandExample {
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            ..
        } = hand_example_2actions();
        let scores = DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, n, 2, 0.01)
            .expect("build_dr_scores should succeed");
        // sample 0: T=0, Y=0, m̂_0=0, m̂_1=2, ê=0.5
        assert!((scores[0] - 0.0).abs() < 1e-6, "Γ_0(0) = {}", scores[0]);
        assert!((scores[1] - 2.0).abs() < 1e-6, "Γ_0(1) = {}", scores[1]);
    }

    #[test]
    fn aipw_identity_treated_arm_nontrivial_residual() {
        // sample 1: T=1, Y=2, m̂_0=0, m̂_1=2, ê=0.5
        // Γ_1(0) = m̂_0 = 0.
        // Γ_1(1) = m̂_1 + (Y - m̂_1)/ê_1 = 2 + 0/0.5 = 2.
        // Now build a custom example where residual is nonzero.
        let n_samples = 1usize;
        let n_actions = 2usize;
        let treatments = vec![1usize];
        let outcomes = vec![3.0_f32]; // Y_i = 3, but m̂(x, 1) = 2 → residual = 1.
        let m_hat = vec![5.0_f32, 2.0]; // m̂_0 = 5, m̂_1 = 2
        let e_hat = vec![0.4_f32, 0.6]; // ê_0 = 0.4, ê_1 = 0.6
        let scores = DrPolicy::build_dr_scores(
            &treatments,
            &outcomes,
            &m_hat,
            &e_hat,
            n_samples,
            n_actions,
            0.01,
        )
        .expect("value should be present");
        // Γ_0(0) = m̂_0 = 5 (since T != 0).
        assert!((scores[0] - 5.0).abs() < 1e-6, "Γ_0(0) = {}", scores[0]);
        // Γ_0(1) = m̂_1 + (Y − m̂_1) · 1 / ê_1 = 2 + 1 / 0.6 = 2 + 1.666... = 3.666...
        let expected = 2.0_f32 + 1.0 / 0.6;
        assert!(
            (scores[1] - expected).abs() < 1e-4,
            "Γ_0(1) = {} vs expected {}",
            scores[1],
            expected
        );
    }

    #[test]
    fn propensity_clipping_replaces_zero() {
        // ê = 0 should be clipped to `propensity_clip`; the result must be
        // finite (no NaN/Inf).
        let treatments = vec![0usize];
        let outcomes = vec![1.0_f32];
        let m_hat = vec![0.0_f32, 0.0];
        let e_hat = vec![0.0_f32, 1.0]; // ê for a=0 is zero!
        let clip = 0.05_f32;
        let scores = DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 2, clip)
            .expect("build_dr_scores should succeed");
        // Γ_0(0) = m̂_0 + (Y − m̂_0)·1 / clipped(ê_0)
        //        = 0 + 1 / 0.05 = 20.
        assert!((scores[0] - 20.0).abs() < 1e-4, "Γ_0(0) = {}", scores[0]);
        // Γ_0(1) = m̂_1 = 0 (no indicator, since T != 1).
        assert!((scores[1] - 0.0).abs() < 1e-6, "Γ_0(1) = {}", scores[1]);
        assert!(scores.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn propensity_clipping_replaces_one() {
        // ê = 1.0 should be clipped to `1 − clip`.
        let treatments = vec![0usize];
        let outcomes = vec![1.0_f32];
        let m_hat = vec![0.0_f32, 0.0];
        let e_hat = vec![1.0_f32, 0.0]; // ê_0 = 1.0
        let clip = 0.1_f32;
        let scores = DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 2, clip)
            .expect("build_dr_scores should succeed");
        // Γ_0(0) = 0 + 1 / (1 - 0.1) = 1 / 0.9
        let expected = 1.0_f32 / 0.9;
        assert!(
            (scores[0] - expected).abs() < 1e-5,
            "Γ_0(0) = {} vs expected {}",
            scores[0],
            expected
        );
    }

    #[test]
    fn fit_returns_non_empty_tree() {
        let HandExample {
            features,
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            n_features: d,
        } = hand_example_2actions();
        let cfg = DrPolicyConfig {
            policy_depth: 1,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        let res = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        // The tree should have splits (depth 1 over a clearly-separable
        // dataset).
        assert!(matches!(res.policy_tree.root, PolicyNode::Split { .. }));
        assert_eq!(res.dr_scores.len(), n * 2);
        assert_eq!(res.policy_tree.n_actions, 2);
    }

    #[test]
    fn predict_consistent_with_policy_tree() {
        let HandExample {
            features,
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            n_features: d,
        } = hand_example_2actions();
        let cfg = DrPolicyConfig {
            policy_depth: 1,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        let res = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        for i in 0..n {
            let row = &features[i * d..(i + 1) * d];
            let via_dr = DrPolicy::predict(&res, row).expect("predict should succeed");
            let via_pt = res
                .policy_tree
                .predict(row)
                .expect("predict should succeed");
            assert_eq!(via_dr, via_pt);
        }
    }

    #[test]
    fn deterministic_fit() {
        let HandExample {
            features,
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            n_features: d,
        } = hand_example_2actions();
        let cfg = DrPolicyConfig {
            policy_depth: 2,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        let a = DrPolicy::fit(
            &features,
            &treatments,
            &outcomes,
            &m_hat,
            &e_hat,
            n,
            d,
            cfg.clone(),
        )
        .expect("value should be present");
        let b = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        assert_eq!(a.dr_scores, b.dr_scores);
        assert!((a.train_welfare - b.train_welfare).abs() < 1e-12);
        assert_eq!(a.policy_tree.root, b.policy_tree.root);
    }

    #[test]
    fn train_welfare_matches_policy_welfare_on_dr_scores() {
        let HandExample {
            features,
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            n_features: d,
        } = hand_example_2actions();
        let cfg = DrPolicyConfig {
            policy_depth: 1,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        let res = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        let preds = res
            .policy_tree
            .predict_batch(&features, n, d)
            .expect("predict_batch should succeed");
        let w = PolicyTree::policy_welfare(&res.dr_scores, n, 2, &preds)
            .expect("policy_welfare should succeed");
        assert!((w - res.train_welfare).abs() < 1e-6);
    }

    #[test]
    fn min_leaf_samples_respected() {
        // Build a slightly larger dataset.
        let n = 8usize;
        let d = 1usize;
        let features: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        // 2 actions; pretend cross-fit gave good m̂ and uniform ê.
        let m_hat: Vec<f32> = (0..n)
            .flat_map(|i| {
                let x = i as f32 / n as f32;
                vec![
                    if x > 0.5 { 5.0 } else { 0.0 },
                    if x <= 0.5 { 5.0 } else { 0.0 },
                ]
            })
            .collect();
        let e_hat = vec![0.5_f32; n * 2];
        let treatments = vec![0usize; n];
        let outcomes = vec![1.0_f32; n];
        let min_leaf = 3;
        let cfg = DrPolicyConfig {
            policy_depth: 2,
            min_leaf_samples: min_leaf,
            n_actions: 2,
        };
        let res = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        let rows: Vec<usize> = (0..n).collect();
        let smallest = min_leaf_size(&res.policy_tree.root, &rows, &features, d);
        assert!(
            smallest >= min_leaf,
            "smallest leaf {smallest} < min_leaf {min_leaf}"
        );
    }

    fn min_leaf_size(
        node: &PolicyNode,
        rows: &[usize],
        features: &[f32],
        n_features: usize,
    ) -> usize {
        match node {
            PolicyNode::Leaf { .. } => rows.len(),
            PolicyNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let mut lr = Vec::new();
                let mut rr = Vec::new();
                for &i in rows {
                    if features[i * n_features + feature] <= *threshold {
                        lr.push(i);
                    } else {
                        rr.push(i);
                    }
                }
                min_leaf_size(left, &lr, features, n_features)
                    .min(min_leaf_size(right, &rr, features, n_features))
            }
        }
    }

    #[test]
    fn err_n_actions_too_small() {
        let treatments = vec![0usize];
        let outcomes = vec![1.0_f32];
        let m_hat = vec![1.0_f32];
        let e_hat = vec![0.5_f32];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 1, 0.01),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_propensity_clip_zero() {
        let treatments = vec![0usize];
        let outcomes = vec![1.0_f32];
        let m_hat = vec![0.0_f32, 0.0];
        let e_hat = vec![0.5_f32, 0.5];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 2, 0.0),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_propensity_clip_too_large() {
        let treatments = vec![0usize];
        let outcomes = vec![1.0_f32];
        let m_hat = vec![0.0_f32, 0.0];
        let e_hat = vec![0.5_f32, 0.5];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 2, 0.6),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_propensity_clip_negative() {
        let treatments = vec![0usize];
        let outcomes = vec![1.0_f32];
        let m_hat = vec![0.0_f32, 0.0];
        let e_hat = vec![0.5_f32, 0.5];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 2, -0.01),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_treatment_out_of_range() {
        let treatments = vec![5usize]; // 5 >= 2 (n_actions)
        let outcomes = vec![1.0_f32];
        let m_hat = vec![0.0_f32, 0.0];
        let e_hat = vec![0.5_f32, 0.5];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 1, 2, 0.01),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_treatments_wrong_length() {
        let treatments = vec![0usize; 3]; // should be 2
        let outcomes = vec![1.0_f32; 2];
        let m_hat = vec![0.0_f32; 4];
        let e_hat = vec![0.5_f32; 4];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 2, 2, 0.01),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_outcomes_wrong_length() {
        let treatments = vec![0usize, 1];
        let outcomes = vec![1.0_f32]; // length 1, should be 2
        let m_hat = vec![0.0_f32; 4];
        let e_hat = vec![0.5_f32; 4];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 2, 2, 0.01),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_m_hat_wrong_length() {
        let treatments = vec![0usize, 1];
        let outcomes = vec![1.0_f32, 1.0];
        let m_hat = vec![0.0_f32; 3]; // should be 4
        let e_hat = vec![0.5_f32; 4];
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 2, 2, 0.01),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_e_hat_wrong_length() {
        let treatments = vec![0usize, 1];
        let outcomes = vec![1.0_f32, 1.0];
        let m_hat = vec![0.0_f32; 4];
        let e_hat = vec![0.5_f32; 3]; // should be 4
        assert!(matches!(
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, 2, 2, 0.01),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_features_wrong_length() {
        let HandExample {
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            ..
        } = hand_example_2actions();
        // Build features of wrong size.
        let features = vec![0.1_f32, 0.2]; // should be n=4
        let cfg = DrPolicyConfig {
            policy_depth: 1,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        assert!(matches!(
            DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, 1, cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_n_samples_zero() {
        let cfg = DrPolicyConfig {
            policy_depth: 1,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        assert!(matches!(
            DrPolicy::fit(&[], &[], &[], &[], &[], 0, 1, cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn policy_depth_zero_yields_single_leaf() {
        let HandExample {
            features,
            treatments,
            outcomes,
            m_hat,
            e_hat,
            n_samples: n,
            n_features: d,
        } = hand_example_2actions();
        let cfg = DrPolicyConfig {
            policy_depth: 0,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        let res = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        assert!(matches!(res.policy_tree.root, PolicyNode::Leaf { .. }));
        // All predictions identical (constant policy).
        let preds = res
            .policy_tree
            .predict_batch(&features, n, d)
            .expect("predict_batch should succeed");
        let first = preds[0];
        for &a in &preds {
            assert_eq!(a, first);
        }
    }

    #[test]
    fn constant_outcome_yields_valid_policy() {
        let n = 6usize;
        let d = 1usize;
        let features: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let treatments = vec![0usize; n];
        let outcomes = vec![3.0_f32; n]; // constant outcome
        let m_hat = vec![3.0_f32; n * 2]; // m̂(x, a) = 3 for any a
        let e_hat = vec![0.5_f32; n * 2];
        let cfg = DrPolicyConfig {
            policy_depth: 1,
            min_leaf_samples: 1,
            n_actions: 2,
        };
        let res = DrPolicy::fit(&features, &treatments, &outcomes, &m_hat, &e_hat, n, d, cfg)
            .expect("fit should succeed");
        // With constant scores Γ(a) = 3 for all (i, a), policy is arbitrary
        // but every predicted action must be in range.
        let preds = res
            .policy_tree
            .predict_batch(&features, n, d)
            .expect("predict_batch should succeed");
        for &a in &preds {
            assert!(a < 2, "action {a} out of range");
        }
        // Welfare = n · 3 = 18.
        assert!(
            (res.train_welfare - (n as f64 * 3.0)).abs() < 1e-4,
            "welfare = {}",
            res.train_welfare
        );
    }

    #[test]
    fn uniform_propensity_dr_score_form() {
        // With ê(X, a) = 1/n_actions uniformly, the DR score reduces to
        //   Γ_i(a) = m̂_a + n_actions · (Y_i − m̂_{T_i}) · 𝟙{T_i = a}.
        // Verify for a single sample.
        let n = 1usize;
        let n_actions = 3usize;
        let treatments = vec![1usize];
        let outcomes = vec![7.0_f32];
        let m_hat = vec![1.0_f32, 4.0, 3.0]; // m̂_0=1, m̂_1=4, m̂_2=3
        let e_hat = vec![1.0_f32 / 3.0; 3];
        let scores =
            DrPolicy::build_dr_scores(&treatments, &outcomes, &m_hat, &e_hat, n, n_actions, 0.01)
                .expect("value should be present");
        // Γ_0(0) = m̂_0 = 1.
        assert!((scores[0] - 1.0).abs() < 1e-5, "Γ_0(0) = {}", scores[0]);
        // Γ_0(1) = m̂_1 + (Y − m̂_1)·1 / ê_1 = 4 + (7-4)/(1/3) = 4 + 9 = 13.
        assert!((scores[1] - 13.0).abs() < 1e-4, "Γ_0(1) = {}", scores[1]);
        // Γ_0(2) = m̂_2 = 3.
        assert!((scores[2] - 3.0).abs() < 1e-5, "Γ_0(2) = {}", scores[2]);
    }

    #[test]
    fn config_clone_works() {
        let cfg = DrPolicyConfig {
            policy_depth: 2,
            min_leaf_samples: 3,
            n_actions: 4,
        };
        let cloned = cfg.clone();
        assert_eq!(cfg.policy_depth, cloned.policy_depth);
        assert_eq!(cfg.min_leaf_samples, cloned.min_leaf_samples);
        assert_eq!(cfg.n_actions, cloned.n_actions);
    }
}
