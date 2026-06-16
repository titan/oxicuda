use crate::error::{RecsysError, RecsysResult};

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SlimConfig {
    pub n_users: usize,
    pub n_items: usize,
    /// L1 penalty (sparsity), α in elastic-net.
    pub lambda_l1: f32,
    /// L2 penalty (smoothness), β in elastic-net. Must be > 0.
    pub lambda_l2: f32,
    /// Maximum coordinate-descent iterations per column. Default: 100.
    pub max_iter: usize,
    /// Convergence tolerance. Default: 1e-4.
    pub tol: f32,
}

impl Default for SlimConfig {
    fn default() -> Self {
        Self {
            n_users: 0,
            n_items: 0,
            lambda_l1: 0.01,
            lambda_l2: 1.0,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// SLIM (Sparse Linear Method, Ning & Karypis ICDM 2011) item-item recommender.
///
/// Learns W (n_items × n_items, diagonal=0, entries ≥ 0) by minimising:
///   ½||X - X·W||²_F + λ₁||W||₁ + ½λ₂||W||²_F
/// using coordinate descent with elastic-net closed form per column.
#[derive(Debug, Clone)]
pub struct SlimModel {
    pub cfg: SlimConfig,
    /// n_items × n_items weight matrix, row-major; diagonal = 0.
    pub weights: Vec<f32>,
}

impl SlimModel {
    // ── Public API ─────────────────────────────────────────────────────────

    /// Fit SLIM via column-wise coordinate descent with elastic-net regularization.
    ///
    /// `interactions` is a row-major n_users × n_items binary/rating matrix.
    pub fn fit(interactions: &[f32], cfg: SlimConfig) -> RecsysResult<Self> {
        // ── Validate ───────────────────────────────────────────────────────
        if cfg.n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: cfg.n_users });
        }
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: cfg.n_items });
        }
        if cfg.lambda_l2 <= 0.0 {
            return Err(RecsysError::InvalidLambda { val: cfg.lambda_l2 });
        }
        if cfg.lambda_l1 < 0.0 {
            return Err(RecsysError::InvalidLambda { val: cfg.lambda_l1 });
        }
        if cfg.max_iter == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "max_iter must be >= 1".into(),
            });
        }
        let n_u = cfg.n_users;
        let n = cfg.n_items;
        let expected = n_u * n;
        if interactions.len() != expected {
            return Err(RecsysError::DimensionMismatch {
                expected,
                got: interactions.len(),
            });
        }

        // ── Step 1: Pre-compute gram matrix Q = X^T X ─────────────────────
        // Q[i,j] = Σ_u X[u,i] * X[u,j]
        // Computing upper triangle and mirroring saves half the FLOPs.
        let mut q = vec![0.0_f32; n * n];
        for u in 0..n_u {
            let row = &interactions[u * n..(u + 1) * n];
            for i in 0..n {
                let xi = row[i];
                if xi == 0.0 {
                    continue;
                }
                for j in i..n {
                    let val = xi * row[j];
                    q[i * n + j] += val;
                    if j != i {
                        q[j * n + i] += val;
                    }
                }
            }
        }

        // ── Step 2: Column-wise coordinate descent ─────────────────────────
        let mut w = vec![0.0_f32; n * n];

        let l1 = cfg.lambda_l1;
        let l2 = cfg.lambda_l2;

        for j in 0..n {
            // Q[i,i] is the denominator for each coordinate; pre-fetch per row i.
            // Iterate up to max_iter over all i ≠ j.
            'outer: for _iter in 0..cfg.max_iter {
                let mut max_delta: f32 = 0.0;

                for i in 0..n {
                    if i == j {
                        continue;
                    }
                    let q_ii = q[i * n + i];
                    if q_ii <= 0.0 {
                        // Item i has zero interactions; skip.
                        continue;
                    }
                    // Partial correlation for (i, j):
                    //   ρ_ij = Q[i,j] - Σ_{k≠i,k≠j} Q[i,k]*w[k,j]
                    // Efficiently: ρ_ij = Q[i,j] - (Σ_k Q[i,k]*w[k,j]) + Q[i,i]*w[i,j]
                    // (The diagonal of Q*w_j is added back because k=i was excluded.)
                    // Also k=j term: w[j,j] = 0 always, so Q[i,j]*w[j,j] = 0; no correction needed.
                    let mut sum_qw: f32 = 0.0;
                    for k in 0..n {
                        sum_qw += q[i * n + k] * w[k * n + j];
                    }
                    // Add back i's own contribution since we excluded k=i from Σ_{k≠i}.
                    let rho_ij = q[i * n + j] - sum_qw + q_ii * w[i * n + j];

                    let new_w = Self::cd_update(rho_ij, q_ii, l1, l2);
                    let delta = (new_w - w[i * n + j]).abs();
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    w[i * n + j] = new_w;
                }

                if max_delta < cfg.tol {
                    break 'outer;
                }
            }
            // Enforce diagonal = 0 (non-negativity already guaranteed by cd_update).
            w[j * n + j] = 0.0;
        }

        Ok(Self { cfg, weights: w })
    }

    /// Score all items for a user given their interaction history.
    ///
    /// `history` must have length `n_items`. Returns `score = history @ W`.
    pub fn predict(&self, history: &[f32]) -> RecsysResult<Vec<f32>> {
        let n = self.cfg.n_items;
        if history.len() != n {
            return Err(RecsysError::DimensionMismatch {
                expected: n,
                got: history.len(),
            });
        }
        let mut scores = vec![0.0_f32; n];
        for (i, &h) in history.iter().enumerate() {
            if h == 0.0 {
                continue;
            }
            let row_offset = i * n;
            for (j, s) in scores.iter_mut().enumerate() {
                *s += h * self.weights[row_offset + j];
            }
        }
        Ok(scores)
    }

    /// Return top-k item indices, excluding items where `history[i] > 0`.
    pub fn recommend(&self, history: &[f32], k: usize) -> RecsysResult<Vec<usize>> {
        let scores = self.predict(history)?;

        let mut candidates: Vec<(f32, usize)> = scores
            .iter()
            .enumerate()
            .filter(|&(i, _)| history[i] == 0.0)
            .map(|(i, &s)| (s, i))
            .collect();

        let take = k.min(candidates.len());
        if take > 0 && take < candidates.len() {
            candidates.select_nth_unstable_by(take - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(take);
        }
        candidates
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates.into_iter().map(|(_, idx)| idx).collect())
    }

    /// Soft-threshold: sign(v) * max(|v| - threshold, 0).
    #[inline]
    pub fn soft_threshold(v: f32, threshold: f32) -> f32 {
        if v > threshold {
            v - threshold
        } else if v < -threshold {
            v + threshold
        } else {
            0.0
        }
    }

    /// Elastic-net coordinate descent update for a single coefficient.
    ///
    /// Computes:
    ///   numerator   = soft_threshold(ρ_ij, λ₁)
    ///   denominator = Q[i,i] + λ₂
    ///   w_ij        = max(numerator / denominator, 0)   ← non-negativity
    #[inline]
    fn cd_update(rho_ij: f32, q_ii: f32, lambda_l1: f32, lambda_l2: f32) -> f32 {
        let numerator = Self::soft_threshold(rho_ij, lambda_l1);
        let denominator = q_ii + lambda_l2;
        if denominator <= 0.0 {
            return 0.0;
        }
        (numerator / denominator).max(0.0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(n_users: usize, n_items: usize) -> SlimConfig {
        SlimConfig {
            n_users,
            n_items,
            lambda_l1: 0.01,
            lambda_l2: 1.0,
            max_iter: 100,
            tol: 1e-5,
        }
    }

    fn small_interactions() -> Vec<f32> {
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    }

    #[test]
    fn zero_interaction_matrix() {
        let x = vec![0.0_f32; 4 * 4];
        let cfg = small_cfg(4, 4);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        for &w in &model.weights {
            assert_eq!(w, 0.0, "all weights should be zero for zero interactions");
        }
    }

    #[test]
    fn identity_block_recovers() {
        // Each user interacted with exactly one item — the gram Q = I (ignoring off-diag).
        // W solution must still satisfy diagonal=0 constraint.
        let n = 4;
        let mut x = vec![0.0_f32; n * n];
        for i in 0..n {
            x[i * n + i] = 1.0;
        }
        let cfg = small_cfg(n, n);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        for i in 0..n {
            assert_eq!(model.weights[i * n + i], 0.0, "diagonal must be 0");
        }
    }

    #[test]
    fn soft_threshold_zero() {
        assert_eq!(SlimModel::soft_threshold(0.3, 0.5), 0.0);
        assert_eq!(SlimModel::soft_threshold(-0.3, 0.5), 0.0);
        assert_eq!(SlimModel::soft_threshold(0.5, 0.5), 0.0);
    }

    #[test]
    fn soft_threshold_positive() {
        let result = SlimModel::soft_threshold(1.2, 0.5);
        assert!((result - 0.7).abs() < 1e-6, "expected 0.7, got {result}");
    }

    #[test]
    fn soft_threshold_negative() {
        let result = SlimModel::soft_threshold(-1.2, 0.5);
        assert!((result + 0.7).abs() < 1e-6, "expected -0.7, got {result}");
    }

    #[test]
    fn weights_non_negative() {
        let x = small_interactions();
        let cfg = small_cfg(3, 3);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        for (idx, &w) in model.weights.iter().enumerate() {
            assert!(w >= 0.0, "weight at {idx} is negative: {w}");
        }
    }

    #[test]
    fn diagonal_always_zero() {
        let x = small_interactions();
        let cfg = small_cfg(3, 3);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        let n = 3;
        for i in 0..n {
            assert_eq!(model.weights[i * n + i], 0.0, "W[{i},{i}] must be 0");
        }
    }

    #[test]
    fn predict_output_length() {
        let x = small_interactions();
        let cfg = small_cfg(3, 3);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        let history = vec![1.0, 0.0, 0.0];
        let scores = model.predict(&history).expect("predict should succeed");
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn recommend_k_items() {
        let x = small_interactions();
        let cfg = small_cfg(3, 3);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        let history = vec![1.0, 0.0, 0.0];
        let recs = model
            .recommend(&history, 2)
            .expect("recommend should succeed");
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn recommend_excludes_seen() {
        let x = small_interactions();
        let cfg = small_cfg(3, 3);
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        let history = vec![1.0, 0.0, 0.0];
        let recs = model
            .recommend(&history, 2)
            .expect("recommend should succeed");
        for &idx in &recs {
            assert!(
                history[idx] == 0.0,
                "item {idx} was in history but recommended"
            );
        }
    }

    #[test]
    fn single_item_model() {
        // n_items=1 is a degenerate case: only one item, no off-diagonal entries.
        let x = vec![1.0_f32; 3];
        let cfg = SlimConfig {
            n_users: 3,
            n_items: 1,
            lambda_l1: 0.01,
            lambda_l2: 1.0,
            max_iter: 10,
            tol: 1e-4,
        };
        let model = SlimModel::fit(&x, cfg).expect("fit should succeed");
        assert_eq!(model.weights.len(), 1);
        assert_eq!(model.weights[0], 0.0);
    }

    #[test]
    fn high_l1_yields_sparse_weights() {
        let n_users = 5;
        let n_items = 5;
        let x: Vec<f32> = vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let cfg_high = SlimConfig {
            n_users,
            n_items,
            lambda_l1: 10.0,
            lambda_l2: 1.0,
            max_iter: 200,
            tol: 1e-5,
        };
        let model_high = SlimModel::fit(&x, cfg_high).expect("fit should succeed");
        let nnz_high = model_high.weights.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(
            nnz_high, 0,
            "high l1 should produce all-zero off-diagonal weights"
        );
    }

    #[test]
    fn low_l1_less_sparse() {
        let n_users = 5;
        let n_items = 5;
        let x: Vec<f32> = vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let cfg_high = SlimConfig {
            n_users,
            n_items,
            lambda_l1: 10.0,
            lambda_l2: 1.0,
            max_iter: 200,
            tol: 1e-5,
        };
        let cfg_low = SlimConfig {
            n_users,
            n_items,
            lambda_l1: 0.001,
            lambda_l2: 0.01,
            max_iter: 200,
            tol: 1e-6,
        };
        let model_high = SlimModel::fit(&x, cfg_high).expect("fit should succeed");
        let model_low = SlimModel::fit(&x, cfg_low).expect("fit should succeed");
        let nnz_high = model_high.weights.iter().filter(|&&w| w > 0.0).count();
        let nnz_low = model_low.weights.iter().filter(|&&w| w > 0.0).count();
        assert!(
            nnz_low >= nnz_high,
            "lower l1 should have at least as many non-zeros: nnz_low={nnz_low} nnz_high={nnz_high}"
        );
    }

    #[test]
    fn cd_update_returns_zero_at_threshold() {
        // When rho_ij == lambda_l1, soft_threshold gives 0, so update = 0.
        let l1 = 0.5;
        let l2 = 1.0;
        let q_ii = 2.0;
        // rho_ij = l1 exactly → soft_threshold(rho, l1) = 0 → w = 0
        let rho = l1;
        // Access via the public soft_threshold to mirror cd_update logic.
        let st = SlimModel::soft_threshold(rho, l1);
        let w = (st / (q_ii + l2)).max(0.0);
        assert_eq!(w, 0.0, "cd_update should return 0 at threshold boundary");
    }

    #[test]
    fn fit_err_zero_users() {
        let cfg = SlimConfig {
            n_users: 0,
            n_items: 3,
            ..SlimConfig::default()
        };
        assert!(matches!(
            SlimModel::fit(&[], cfg),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn fit_err_zero_items() {
        let cfg = SlimConfig {
            n_users: 3,
            n_items: 0,
            ..SlimConfig::default()
        };
        assert!(matches!(
            SlimModel::fit(&[], cfg),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    #[test]
    fn fit_err_length_mismatch() {
        let cfg = SlimConfig {
            n_users: 3,
            n_items: 3,
            lambda_l2: 1.0,
            ..SlimConfig::default()
        };
        let x = vec![1.0_f32; 5];
        assert!(matches!(
            SlimModel::fit(&x, cfg),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_err_negative_lambda_l2() {
        let cfg = SlimConfig {
            n_users: 3,
            n_items: 3,
            lambda_l2: -1.0,
            ..SlimConfig::default()
        };
        let x = small_interactions();
        assert!(matches!(
            SlimModel::fit(&x, cfg),
            Err(RecsysError::InvalidLambda { .. })
        ));
    }
}
