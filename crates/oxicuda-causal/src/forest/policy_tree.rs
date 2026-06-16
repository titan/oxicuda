//! Welfare-maximizing policy trees (exact shallow-tree search).
//!
//! Reference: Athey & Wager (2021) "Policy Learning with Observational Data",
//! Econometrica 89(1):133–161; algorithmic form per the `policytree` package of
//! Zhou, Athey & Wager (2023) "Offline Multi-Action Policy Learning".
//!
//! Given per-sample, per-action **doubly-robust reward scores** `Γ_{i,a}`, a
//! policy `π` maps a feature vector to an action. The empirical welfare of a
//! policy is `Σ_i Γ_{i,π(x_i)}`. We learn a depth-`L` decision tree (axis-
//! aligned threshold splits) that **exactly maximizes** this empirical welfare
//! by exhaustive recursive search over every feature and every candidate
//! threshold (midpoints of consecutive sorted unique values). This exhaustive
//! search is what gives the Athey–Wager regret guarantee, so it is performed
//! without approximation.
//!
//! Complexity is `O(depth · n_features · n_samples² · n_actions)` — acceptable
//! for the shallow trees (`L ∈ {1, 2}`) for which the method is intended.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`PolicyTree::fit`].
#[derive(Debug, Clone)]
pub struct PolicyTreeConfig {
    /// Maximum tree depth (number of split layers). `0` yields a single leaf.
    pub depth: usize,
    /// Number of available actions (must be ≥ 2).
    pub n_actions: usize,
    /// Minimum number of samples that must fall on each side of a split.
    pub min_leaf_samples: usize,
}

/// A node of a learned policy tree.
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyNode {
    /// Terminal node assigning a fixed action.
    Leaf {
        /// Action index assigned to samples reaching this leaf.
        action: usize,
    },
    /// Internal split: go `left` if `x[feature] <= threshold`, else `right`.
    Split {
        /// Feature index used by the split.
        feature: usize,
        /// Threshold value (split point).
        threshold: f32,
        /// Subtree for `x[feature] <= threshold`.
        left: Box<PolicyNode>,
        /// Subtree for `x[feature] > threshold`.
        right: Box<PolicyNode>,
    },
}

/// A learned policy tree.
#[derive(Debug, Clone)]
pub struct PolicyTree {
    /// Root node.
    pub root: PolicyNode,
    /// Configured maximum depth.
    pub depth: usize,
    /// Number of actions.
    pub n_actions: usize,
}

/// Output of [`PolicyTree::fit`].
#[derive(Debug, Clone)]
pub struct PolicyTreeResult {
    /// The fitted tree.
    pub tree: PolicyTree,
    /// In-sample welfare `Σ_i Γ_{i,π(x_i)}` achieved on the training data.
    pub train_welfare: f64,
}

impl PolicyTree {
    /// Fit a welfare-maximizing policy tree.
    ///
    /// `features` is row-major `n_samples × n_features`; `scores` is row-major
    /// `n_samples × n_actions` (doubly-robust reward per action).
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] for `n_actions < 2`,
    /// `min_leaf_samples == 0`, `n_samples < 1`, or `n_features < 1`; and
    /// [`CausalError::DimensionMismatch`] if the buffer lengths do not match
    /// `n_samples * n_features` / `n_samples * n_actions`.
    pub fn fit(
        features: &[f32],
        scores: &[f32],
        n_samples: usize,
        n_features: usize,
        cfg: &PolicyTreeConfig,
    ) -> CausalResult<PolicyTreeResult> {
        if cfg.n_actions < 2 {
            return Err(CausalError::InvalidParameter {
                reason: format!("n_actions must be >= 2, got {}", cfg.n_actions),
            });
        }
        if cfg.min_leaf_samples == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "min_leaf_samples must be >= 1".to_string(),
            });
        }
        if n_samples < 1 {
            return Err(CausalError::InvalidParameter {
                reason: "n_samples must be >= 1".to_string(),
            });
        }
        if n_features < 1 {
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
        if scores.len() != n_samples * cfg.n_actions {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * cfg.n_actions,
                got: scores.len(),
            });
        }

        let rows: Vec<usize> = (0..n_samples).collect();
        let ctx = FitContext {
            features,
            scores,
            n_features,
            n_actions: cfg.n_actions,
            min_leaf_samples: cfg.min_leaf_samples,
        };
        let (root, welfare) = fit_node(&ctx, &rows, cfg.depth);

        Ok(PolicyTreeResult {
            tree: PolicyTree {
                root,
                depth: cfg.depth,
                n_actions: cfg.n_actions,
            },
            train_welfare: welfare,
        })
    }

    /// Predict the action for a single feature vector by traversing the tree.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] if `x.len()` does not match
    /// the feature dimensionality used during fitting (inferred from the split
    /// features encountered along the traversed path; an out-of-range index is
    /// reported here).
    pub fn predict(&self, x: &[f32]) -> CausalResult<usize> {
        let mut node = &self.root;
        loop {
            match node {
                PolicyNode::Leaf { action } => return Ok(*action),
                PolicyNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    let value = *x.get(*feature).ok_or(CausalError::DimensionMismatch {
                        expected: *feature + 1,
                        got: x.len(),
                    })?;
                    node = if value <= *threshold { left } else { right };
                }
            }
        }
    }

    /// Predict actions for a batch of `n_samples × n_features` row-major data.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] if `features.len()` does not
    /// match `n_samples * n_features`, or if a split references an out-of-range
    /// feature for the given `n_features`.
    pub fn predict_batch(
        &self,
        features: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> CausalResult<Vec<usize>> {
        if features.len() != n_samples * n_features {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_features,
                got: features.len(),
            });
        }
        let mut out = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let row = &features[s * n_features..(s + 1) * n_features];
            out.push(self.predict(row)?);
        }
        Ok(out)
    }

    /// Empirical welfare `Σ_i scores[i][actions[i]]` of an explicit assignment.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] for `n_actions < 1`,
    /// [`CausalError::DimensionMismatch`] if `scores.len() != n_samples *
    /// n_actions` or `actions.len() != n_samples`, and again
    /// [`CausalError::InvalidParameter`] if any chosen action is out of range.
    pub fn policy_welfare(
        scores: &[f32],
        n_samples: usize,
        n_actions: usize,
        actions: &[usize],
    ) -> CausalResult<f64> {
        if n_actions < 1 {
            return Err(CausalError::InvalidParameter {
                reason: "n_actions must be >= 1".to_string(),
            });
        }
        if scores.len() != n_samples * n_actions {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_actions,
                got: scores.len(),
            });
        }
        if actions.len() != n_samples {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples,
                got: actions.len(),
            });
        }
        let mut welfare = 0.0_f64;
        for (i, &a) in actions.iter().enumerate() {
            if a >= n_actions {
                return Err(CausalError::InvalidParameter {
                    reason: format!("action {a} at sample {i} >= n_actions {n_actions}"),
                });
            }
            welfare += scores[i * n_actions + a] as f64;
        }
        Ok(welfare)
    }
}

/// Immutable context threaded through the recursive fit.
struct FitContext<'a> {
    features: &'a [f32],
    scores: &'a [f32],
    n_features: usize,
    n_actions: usize,
    min_leaf_samples: usize,
}

/// Best leaf action and its welfare for a set of rows: `a* = argmax_a Σ Γ_{i,a}`.
fn best_leaf(ctx: &FitContext, rows: &[usize]) -> (usize, f64) {
    let mut best_action = 0usize;
    let mut best_sum = f64::NEG_INFINITY;
    for a in 0..ctx.n_actions {
        let mut sum = 0.0_f64;
        for &i in rows {
            sum += ctx.scores[i * ctx.n_actions + a] as f64;
        }
        if sum > best_sum {
            best_sum = sum;
            best_action = a;
        }
    }
    // With at least 2 actions and any rows, best_sum is finite. The empty-rows
    // case yields a welfare of 0 across all actions (best_sum == 0).
    if !best_sum.is_finite() {
        best_sum = 0.0;
    }
    (best_action, best_sum)
}

/// Recursively fit the welfare-maximizing subtree for `rows` with
/// `depth_left` remaining split layers. Returns `(node, welfare)`.
fn fit_node(ctx: &FitContext, rows: &[usize], depth_left: usize) -> (PolicyNode, f64) {
    let (leaf_action, leaf_welfare) = best_leaf(ctx, rows);
    let leaf = PolicyNode::Leaf {
        action: leaf_action,
    };

    // Leaf case: no remaining depth, or too few samples to split into two
    // valid children.
    if depth_left == 0 || rows.len() < 2 * ctx.min_leaf_samples {
        return (leaf, leaf_welfare);
    }

    let mut best: Option<(usize, f32, PolicyNode, PolicyNode, f64)> = None;

    for feature in 0..ctx.n_features {
        // Sorted unique feature values among `rows`.
        let mut values: Vec<f32> = rows
            .iter()
            .map(|&i| ctx.features[i * ctx.n_features + feature])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        if values.len() < 2 {
            continue; // constant feature → no split possible
        }

        // Candidate thresholds = midpoints of consecutive sorted unique values.
        for w in values.windows(2) {
            let threshold = 0.5 * (w[0] + w[1]);

            let mut left_rows = Vec::new();
            let mut right_rows = Vec::new();
            for &i in rows {
                let v = ctx.features[i * ctx.n_features + feature];
                if v <= threshold {
                    left_rows.push(i);
                } else {
                    right_rows.push(i);
                }
            }
            if left_rows.len() < ctx.min_leaf_samples || right_rows.len() < ctx.min_leaf_samples {
                continue;
            }

            let (left_node, left_welfare) = fit_node(ctx, &left_rows, depth_left - 1);
            let (right_node, right_welfare) = fit_node(ctx, &right_rows, depth_left - 1);
            let total = left_welfare + right_welfare;

            let improves = match &best {
                Some((_, _, _, _, best_total)) => total > *best_total,
                None => true,
            };
            if improves {
                best = Some((feature, threshold, left_node, right_node, total));
            }
        }
    }

    match best {
        // Only split when it strictly beats the leaf welfare.
        Some((feature, threshold, left_node, right_node, total)) if total > leaf_welfare => (
            PolicyNode::Split {
                feature,
                threshold,
                left: Box::new(left_node),
                right: Box::new(right_node),
            },
            total,
        ),
        _ => (leaf, leaf_welfare),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum realised depth of a tree (a single leaf has depth 0).
    fn node_depth(node: &PolicyNode) -> usize {
        match node {
            PolicyNode::Leaf { .. } => 0,
            PolicyNode::Split { left, right, .. } => 1 + node_depth(left).max(node_depth(right)),
        }
    }

    /// Smallest leaf size (sample count) reachable for the given rows.
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
    fn depth_zero_picks_global_best_action() {
        // Action 1 has the highest column sum overall.
        let n = 4;
        let n_actions = 2;
        let n_features = 1;
        let features = vec![0.1_f32, 0.2, 0.3, 0.4];
        let scores = vec![
            1.0, 5.0, // sample 0
            1.0, 5.0, // sample 1
            1.0, 5.0, // sample 2
            1.0, 5.0, // sample 3
        ];
        let cfg = PolicyTreeConfig {
            depth: 0,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        // Single leaf with action 1.
        assert!(matches!(res.tree.root, PolicyNode::Leaf { action: 1 }));
        assert!((res.train_welfare - 20.0).abs() < 1e-6);
        // Every sample maps to action 1.
        for &x in &features {
            assert_eq!(res.tree.predict(&[x]).expect("predict should succeed"), 1);
        }
    }

    /// Build a dataset where action 1 is best for x[0] <= 0.5 and action 0 is
    /// best for x[0] > 0.5.
    fn split_dataset() -> (Vec<f32>, Vec<f32>, usize, usize, usize) {
        let n_actions = 2;
        let n_features = 1;
        // x values straddling 0.5
        let xs = [0.1_f32, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let n = xs.len();
        let mut features = Vec::with_capacity(n * n_features);
        let mut scores = Vec::with_capacity(n * n_actions);
        for &x in &xs {
            features.push(x);
            if x <= 0.5 {
                // action 1 better here
                scores.push(0.0); // action 0
                scores.push(3.0); // action 1
            } else {
                // action 0 better here
                scores.push(3.0); // action 0
                scores.push(0.0); // action 1
            }
        }
        (features, scores, n, n_features, n_actions)
    }

    #[test]
    fn depth_one_splits_near_half() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        match &res.tree.root {
            PolicyNode::Split {
                feature, threshold, ..
            } => {
                assert_eq!(*feature, 0);
                assert!(
                    (*threshold - 0.5).abs() < 0.11,
                    "threshold {threshold} not near 0.5"
                );
            }
            other => panic!("expected a split, got {other:?}"),
        }
        // Welfare should be the maximum possible: 3 per sample = 24.
        assert!((res.train_welfare - 24.0).abs() < 1e-6);
        // Beats any constant policy: a constant action earns 3 on half = 12.
        let const0: Vec<usize> = vec![0; n];
        let const1: Vec<usize> = vec![1; n];
        let w0 = PolicyTree::policy_welfare(&scores, n, n_actions, &const0)
            .expect("policy_welfare should succeed");
        let w1 = PolicyTree::policy_welfare(&scores, n, n_actions, &const1)
            .expect("policy_welfare should succeed");
        assert!(res.train_welfare > w0);
        assert!(res.train_welfare > w1);
    }

    #[test]
    fn welfare_non_decreasing_with_depth() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let mut last = f64::NEG_INFINITY;
        for depth in 0..=2 {
            let cfg = PolicyTreeConfig {
                depth,
                n_actions,
                min_leaf_samples: 1,
            };
            let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
                .expect("fit should succeed with valid test inputs");
            assert!(
                res.train_welfare >= last - 1e-9,
                "welfare decreased at depth {depth}: {} < {}",
                res.train_welfare,
                last
            );
            last = res.train_welfare;
        }
    }

    #[test]
    fn predict_matches_leaf_assignment() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        // x=0.1 → action 1; x=0.9 → action 0.
        assert_eq!(res.tree.predict(&[0.1]).expect("predict should succeed"), 1);
        assert_eq!(res.tree.predict(&[0.9]).expect("predict should succeed"), 0);
    }

    #[test]
    fn predict_batch_length() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        let preds = res
            .tree
            .predict_batch(&features, n, n_features)
            .expect("predict_batch should succeed with valid inputs");
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn policy_welfare_hand_example() {
        // 2 samples, 2 actions.
        let scores = vec![
            1.0, 2.0, // sample 0: action0=1, action1=2
            3.0, 0.5, // sample 1: action0=3, action1=0.5
        ];
        let actions = vec![1usize, 0usize]; // pick 2.0 + 3.0
        let w = PolicyTree::policy_welfare(&scores, 2, 2, &actions)
            .expect("policy_welfare should succeed");
        assert!((w - 5.0).abs() < 1e-6);
    }

    #[test]
    fn min_leaf_samples_respected() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let min_leaf = 3;
        let cfg = PolicyTreeConfig {
            depth: 2,
            n_actions,
            min_leaf_samples: min_leaf,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        let rows: Vec<usize> = (0..n).collect();
        let smallest = min_leaf_size(&res.tree.root, &rows, &features, n_features);
        assert!(
            smallest >= min_leaf,
            "smallest leaf {smallest} < min_leaf {min_leaf}"
        );
    }

    #[test]
    fn three_actions_works() {
        // 6 samples, 3 actions; action best depends on a single feature region.
        let n_actions = 3;
        let n_features = 1;
        let xs = [0.1_f32, 0.2, 0.3, 0.7, 0.8, 0.9];
        let n = xs.len();
        let mut features = Vec::new();
        let mut scores = Vec::new();
        for &x in &xs {
            features.push(x);
            if x <= 0.5 {
                scores.extend_from_slice(&[0.0, 0.0, 5.0]); // action 2 best
            } else {
                scores.extend_from_slice(&[5.0, 0.0, 0.0]); // action 0 best
            }
        }
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        assert_eq!(res.tree.predict(&[0.1]).expect("predict should succeed"), 2);
        assert_eq!(res.tree.predict(&[0.9]).expect("predict should succeed"), 0);
        assert!((res.train_welfare - 30.0).abs() < 1e-6);
    }

    #[test]
    fn identical_scores_returns_valid_action() {
        // All actions identical per sample ⇒ welfare equals the row-sum of any
        // single action; a leaf is returned (no split helps).
        let n = 5;
        let n_actions = 3;
        let n_features = 1;
        let features = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let mut scores = Vec::new();
        for _ in 0..n {
            scores.extend_from_slice(&[2.0, 2.0, 2.0]);
        }
        let cfg = PolicyTreeConfig {
            depth: 2,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        // Welfare == sum over samples of any single action's score = 5 * 2 = 10.
        assert!((res.train_welfare - 10.0).abs() < 1e-6);
        // Predicts a valid action everywhere.
        let preds = res
            .tree
            .predict_batch(&features, n, n_features)
            .expect("predict_batch should succeed with valid inputs");
        for &a in &preds {
            assert!(a < n_actions);
        }
        // No split improves on the leaf, so the root is a leaf.
        assert!(matches!(res.tree.root, PolicyNode::Leaf { .. }));
    }

    #[test]
    fn threshold_split_at_midpoint() {
        // Two unique x values 0.0 and 1.0 ⇒ threshold should be their midpoint.
        let n_actions = 2;
        let n_features = 1;
        let features = vec![0.0_f32, 0.0, 1.0, 1.0];
        let scores = vec![
            0.0, 4.0, // x=0 → action 1
            0.0, 4.0, // x=0 → action 1
            4.0, 0.0, // x=1 → action 0
            4.0, 0.0, // x=1 → action 0
        ];
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions,
            min_leaf_samples: 1,
        };
        let res =
            PolicyTree::fit(&features, &scores, 4, n_features, &cfg).expect("fit should succeed");
        match res.tree.root {
            PolicyNode::Split {
                feature, threshold, ..
            } => {
                assert_eq!(feature, 0);
                assert!((threshold - 0.5).abs() < 1e-6, "threshold {threshold}");
            }
            other => panic!("expected split, got {other:?}"),
        }
    }

    #[test]
    fn tree_depth_within_config() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        for depth in 0..=2 {
            let cfg = PolicyTreeConfig {
                depth,
                n_actions,
                min_leaf_samples: 1,
            };
            let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
                .expect("fit should succeed with valid test inputs");
            assert!(
                node_depth(&res.tree.root) <= depth,
                "realised depth exceeds {depth}"
            );
        }
    }

    #[test]
    fn deterministic_fit() {
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let cfg = PolicyTreeConfig {
            depth: 2,
            n_actions,
            min_leaf_samples: 1,
        };
        let a = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        let b = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        assert_eq!(a.tree.root, b.tree.root);
        assert!((a.train_welfare - b.train_welfare).abs() < 1e-12);
    }

    #[test]
    fn err_n_actions_too_small() {
        let features = vec![0.1_f32, 0.2];
        let scores = vec![1.0_f32, 1.0];
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions: 1,
            min_leaf_samples: 1,
        };
        assert!(matches!(
            PolicyTree::fit(&features, &scores, 2, 1, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_min_leaf_zero() {
        let features = vec![0.1_f32, 0.2];
        let scores = vec![1.0_f32, 1.0, 1.0, 1.0];
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions: 2,
            min_leaf_samples: 0,
        };
        assert!(matches!(
            PolicyTree::fit(&features, &scores, 2, 1, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_features_length_mismatch() {
        let features = vec![0.1_f32, 0.2, 0.3]; // should be 2*1=2
        let scores = vec![1.0_f32, 1.0, 1.0, 1.0];
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions: 2,
            min_leaf_samples: 1,
        };
        assert!(matches!(
            PolicyTree::fit(&features, &scores, 2, 1, &cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_scores_length_mismatch() {
        let features = vec![0.1_f32, 0.2];
        let scores = vec![1.0_f32, 1.0, 1.0]; // should be 2*2=4
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions: 2,
            min_leaf_samples: 1,
        };
        assert!(matches!(
            PolicyTree::fit(&features, &scores, 2, 1, &cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_predict_wrong_length_x() {
        // Build a tree that splits on feature 0, then predict with empty x.
        let (features, scores, n, n_features, n_actions) = split_dataset();
        let cfg = PolicyTreeConfig {
            depth: 1,
            n_actions,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, n, n_features, &cfg)
            .expect("fit should succeed with valid test inputs");
        // Only triggers if the root is a split (it is for this dataset).
        assert!(matches!(res.tree.root, PolicyNode::Split { .. }));
        assert!(matches!(
            res.tree.predict(&[]),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_policy_welfare_actions_wrong_length() {
        let scores = vec![1.0_f32, 2.0, 3.0, 4.0];
        let actions = vec![0usize]; // should be length 2
        assert!(matches!(
            PolicyTree::policy_welfare(&scores, 2, 2, &actions),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_policy_welfare_action_out_of_range() {
        let scores = vec![1.0_f32, 2.0, 3.0, 4.0];
        let actions = vec![0usize, 5usize]; // action 5 invalid
        assert!(matches!(
            PolicyTree::policy_welfare(&scores, 2, 2, &actions),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn single_sample_yields_leaf() {
        // n_samples = 1 is valid; cannot split, returns a leaf with best action.
        let features = vec![0.5_f32];
        let scores = vec![1.0_f32, 7.0]; // action 1 best
        let cfg = PolicyTreeConfig {
            depth: 2,
            n_actions: 2,
            min_leaf_samples: 1,
        };
        let res = PolicyTree::fit(&features, &scores, 1, 1, &cfg).expect("fit should succeed");
        assert!(matches!(res.tree.root, PolicyNode::Leaf { action: 1 }));
        assert!((res.train_welfare - 7.0).abs() < 1e-6);
    }
}
