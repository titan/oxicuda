//! EASE (Embarrassingly Shallow Autoencoders, Steck 2019 WWW) and
//! EASER (EASE with elastic-net regularization, Steck 2020 RecSys)
//! closed-form item-item recommender.
//!
//! EASE finds an item-item weight matrix W minimizing:
//!   ||X - X*W||²_F + λ||W||²_F  subject to diag(W) = 0
//!
//! Closed form: G = (X^T X + λI)^{-1}, then
//!   W_ij = -G_ij / G_jj  for i ≠ j,  W_ii = 0.
//!
//! EASER adds an L1 term to each off-diagonal entry, solved via
//! coordinate descent with soft-thresholding.

use crate::error::{RecsysError, RecsysResult};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for EASE / EASER.
#[derive(Debug, Clone)]
pub struct EaseConfig {
    /// Number of users.
    pub n_users: usize,
    /// Number of items.
    pub n_items: usize,
    /// L2 regularization strength (λ). Must be > 0. Default: 500.0.
    pub lambda: f32,
    /// L1 regularization strength for EASER (λ₁). Default: 0.0 (= pure EASE).
    pub lambda_l1: f32,
    /// Maximum coordinate-descent iterations for EASER. Default: 50.
    pub l1_iter: usize,
    /// Convergence tolerance for EASER. Default: 1e-4.
    pub l1_tol: f32,
}

impl Default for EaseConfig {
    fn default() -> Self {
        Self {
            n_users: 0,
            n_items: 0,
            lambda: 500.0,
            lambda_l1: 0.0,
            l1_iter: 50,
            l1_tol: 1e-4,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// Trained EASE / EASER model.
///
/// `weights` stores the n_items × n_items item-item weight matrix W
/// in row-major order with `W[i,i] = 0` (diagonal constraint).
#[derive(Debug, Clone)]
pub struct Ease {
    pub cfg: EaseConfig,
    /// Item-item weight matrix W, shape n_items × n_items, row-major.
    pub weights: Vec<f32>,
}

impl Ease {
    // ── Public fitting API ─────────────────────────────────────────────────

    /// Fit an EASE (or EASER when `cfg.lambda_l1 > 0`) model from a
    /// user-item interaction matrix `interactions` (n_users × n_items, row-major).
    ///
    /// # Errors
    ///
    /// Returns [`RecsysError::InvalidNumUsers`] / [`RecsysError::InvalidNumItems`]
    /// when dimensions are zero, [`RecsysError::InvalidLambda`] when `λ ≤ 0`,
    /// [`RecsysError::DimensionMismatch`] when `interactions.len()` is wrong,
    /// and [`RecsysError::NotPositiveDefinite`] if the gram matrix is singular.
    pub fn fit(interactions: &[f32], cfg: EaseConfig) -> RecsysResult<Self> {
        // ── Validate ───────────────────────────────────────────────────────
        if cfg.n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: cfg.n_users });
        }
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: cfg.n_items });
        }
        if cfg.lambda <= 0.0 {
            return Err(RecsysError::InvalidLambda { val: cfg.lambda });
        }
        let expected_len = cfg.n_users * cfg.n_items;
        if interactions.len() != expected_len {
            return Err(RecsysError::DimensionMismatch {
                expected: expected_len,
                got: interactions.len(),
            });
        }

        // ── Step 1: Gram matrix G = X^T X + λI ────────────────────────────
        let gram = Self::compute_gram(interactions, cfg.n_users, cfg.n_items, cfg.lambda);

        // ── Step 2: Invert G via Cholesky decomposition ────────────────────
        let g_inv = Self::invert_via_cholesky(gram, cfg.n_items)?;

        // ── Step 3: Build weight matrix W ──────────────────────────────────
        let n = cfg.n_items;
        let mut weights = vec![0.0_f32; n * n];
        for j in 0..n {
            let g_inv_jj = g_inv[j * n + j];
            for i in 0..n {
                if i == j {
                    weights[i * n + j] = 0.0;
                } else {
                    weights[i * n + j] = -g_inv[i * n + j] / g_inv_jj;
                }
            }
        }

        // ── Step 4 (EASER): coordinate-descent L1 refinement ──────────────
        if cfg.lambda_l1 > 0.0 {
            Self::easer_coordinate_descent(&mut weights, &g_inv, n, &cfg);
        }

        Ok(Self { cfg, weights })
    }

    /// Predict scores for all items for a single user.
    ///
    /// `user_row` must have length `n_items`.  Returns `scores = user_row @ W`
    /// of length `n_items`.
    pub fn predict(&self, user_row: &[f32]) -> RecsysResult<Vec<f32>> {
        let n = self.cfg.n_items;
        if user_row.len() != n {
            return Err(RecsysError::DimensionMismatch {
                expected: n,
                got: user_row.len(),
            });
        }
        let mut scores = vec![0.0_f32; n];
        // scores[j] = Σ_i user_row[i] * W[i,j]
        for (i, &x_i) in user_row.iter().enumerate() {
            if x_i == 0.0 {
                continue;
            }
            let row_offset = i * n;
            for (j, s) in scores.iter_mut().enumerate() {
                *s += x_i * self.weights[row_offset + j];
            }
        }
        Ok(scores)
    }

    /// Predict scores for a batch of users.
    ///
    /// `user_matrix` is row-major with shape `n_users × n_items`.
    /// Returns a score matrix of shape `n_users × n_items`, row-major.
    pub fn predict_batch(&self, user_matrix: &[f32], n_users: usize) -> RecsysResult<Vec<f32>> {
        let n = self.cfg.n_items;
        let expected = n_users * n;
        if user_matrix.len() != expected {
            return Err(RecsysError::DimensionMismatch {
                expected,
                got: user_matrix.len(),
            });
        }
        let mut scores = vec![0.0_f32; n_users * n];
        for u in 0..n_users {
            let row = &user_matrix[u * n..(u + 1) * n];
            let score_row = self.predict(row)?;
            scores[u * n..(u + 1) * n].copy_from_slice(&score_row);
        }
        Ok(scores)
    }

    /// Return the top-`k` item indices for a user, sorted by score descending.
    ///
    /// When `exclude_interacted = true`, items where `user_row[i] > 0` are
    /// removed from the candidate set before ranking.
    pub fn recommend_top_k(
        &self,
        user_row: &[f32],
        k: usize,
        exclude_interacted: bool,
    ) -> RecsysResult<Vec<usize>> {
        let scores = self.predict(user_row)?;

        // Build (score, index) pairs, filtering interacted items if requested.
        let mut candidates: Vec<(f32, usize)> = scores
            .iter()
            .enumerate()
            .filter(|&(i, _)| !exclude_interacted || user_row[i] == 0.0)
            .map(|(i, &s)| (s, i))
            .collect();

        // Partial sort: bring the top-k to the front (largest first).
        let take = k.min(candidates.len());
        // Use select_nth_unstable_by to avoid a full sort on large item sets.
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

    // ── Linear-algebra helpers (pub for testing) ───────────────────────────

    /// Compute gram matrix `G = X^T X + λI` of shape n_items × n_items.
    ///
    /// This is the main bottleneck: O(n_users · n_items²) FLOPs.
    pub fn compute_gram(
        interactions: &[f32],
        n_users: usize,
        n_items: usize,
        lambda: f32,
    ) -> Vec<f32> {
        let mut g = vec![0.0_f32; n_items * n_items];

        // G += X^T X  (column-major outer product accumulation)
        for u in 0..n_users {
            let row = &interactions[u * n_items..(u + 1) * n_items];
            for i in 0..n_items {
                let xi = row[i];
                if xi == 0.0 {
                    continue;
                }
                // Upper triangle (including diagonal); symmetrise below.
                for j in i..n_items {
                    let val = xi * row[j];
                    g[i * n_items + j] += val;
                    if j != i {
                        g[j * n_items + i] += val;
                    }
                }
            }
        }

        // Add ridge: G += λI
        for k in 0..n_items {
            g[k * n_items + k] += lambda;
        }
        g
    }

    /// In-place Cholesky-Banachiewicz decomposition of a positive-definite matrix.
    ///
    /// On success the lower-triangular factor L is written into `a` (the
    /// upper triangle is zeroed).  Returns [`RecsysError::NotPositiveDefinite`]
    /// if a diagonal pivot is non-positive.
    pub fn cholesky(a: &mut [f32], n: usize) -> RecsysResult<()> {
        for i in 0..n {
            // Off-diagonal entries L[i,j] for j < i
            for j in 0..i {
                let mut sum: f32 = a[i * n + j];
                for k in 0..j {
                    sum -= a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = sum / a[j * n + j];
            }
            // Diagonal entry L[i,i]
            let mut diag_sum: f32 = a[i * n + i];
            for k in 0..i {
                diag_sum -= a[i * n + k] * a[i * n + k];
            }
            if diag_sum <= 0.0 {
                return Err(RecsysError::NotPositiveDefinite);
            }
            a[i * n + i] = diag_sum.sqrt();
            // Zero the upper triangle
            for j in (i + 1)..n {
                a[i * n + j] = 0.0;
            }
        }
        Ok(())
    }

    /// Solve `L x = b` in-place (forward substitution, L is lower triangular).
    ///
    /// Result is written into `b`.
    pub fn forward_substitution(l: &[f32], b: &mut [f32], n: usize) {
        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= l[i * n + k] * b[k];
            }
            b[i] = sum / l[i * n + i];
        }
    }

    /// Solve `L^T x = b` in-place (backward substitution, L is lower triangular).
    ///
    /// Result is written into `b`.
    pub fn backward_substitution(l: &[f32], b: &mut [f32], n: usize) {
        // Traverse from bottom to top
        let mut i = n;
        loop {
            if i == 0 {
                break;
            }
            i -= 1;
            let mut sum = b[i];
            for k in (i + 1)..n {
                // L^T[i,k] = L[k,i]
                sum -= l[k * n + i] * b[k];
            }
            b[i] = sum / l[i * n + i];
        }
    }

    /// Invert a positive-definite matrix whose Gram form is provided in `gram`,
    /// using Cholesky factorisation followed by column-by-column triangular solves.
    ///
    /// Returns G⁻¹ as an n × n row-major matrix.
    pub fn invert_via_cholesky(mut gram: Vec<f32>, n: usize) -> RecsysResult<Vec<f32>> {
        // Decompose in-place: gram → L (lower triangular)
        Self::cholesky(&mut gram, n)?;

        let mut g_inv = vec![0.0_f32; n * n];

        for j in 0..n {
            // Construct e_j
            let mut col = vec![0.0_f32; n];
            col[j] = 1.0;

            // Forward: L y = e_j
            Self::forward_substitution(&gram, &mut col, n);
            // Backward: L^T x = y
            Self::backward_substitution(&gram, &mut col, n);

            // Write column j of G_inv
            for i in 0..n {
                g_inv[i * n + j] = col[i];
            }
        }
        Ok(g_inv)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// EASER coordinate-descent refinement (Steck 2020).
    ///
    /// Applies soft-thresholding to each off-diagonal entry of W using
    /// the closed-form EASE initialisation as the starting point.
    fn easer_coordinate_descent(weights: &mut [f32], g_inv: &[f32], n: usize, cfg: &EaseConfig) {
        let lambda_l1 = cfg.lambda_l1;
        for _iter in 0..cfg.l1_iter {
            let mut max_delta: f32 = 0.0;
            for j in 0..n {
                let g_inv_jj = g_inv[j * n + j];
                let threshold = lambda_l1 / (2.0 * g_inv_jj);
                for i in 0..n {
                    if i == j {
                        continue;
                    }
                    // EASE-derived residual for entry (i,j)
                    let r = -(g_inv[i * n + j] / g_inv_jj);
                    // Soft-threshold
                    let new_w = soft_threshold(r, threshold);
                    let old_w = weights[i * n + j];
                    let delta = (new_w - old_w).abs();
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    weights[i * n + j] = new_w;
                }
            }
            if max_delta < cfg.l1_tol {
                break;
            }
        }
    }
}

// ── Module-level helpers ───────────────────────────────────────────────────────

/// Soft-thresholding operator: sign(x) * max(0, |x| - threshold).
#[inline]
fn soft_threshold(x: f32, threshold: f32) -> f32 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small binary interaction matrix.
    /// 3 users × 3 items:
    ///   user 0: items 0, 1
    ///   user 1: items 1, 2
    ///   user 2: items 0, 2
    fn small_interactions() -> Vec<f32> {
        vec![
            1.0, 1.0, 0.0, // user 0
            0.0, 1.0, 1.0, // user 1
            1.0, 0.0, 1.0, // user 2
        ]
    }

    fn default_cfg(n_users: usize, n_items: usize) -> EaseConfig {
        EaseConfig {
            n_users,
            n_items,
            lambda: 500.0,
            ..EaseConfig::default()
        }
    }

    // ── Basic fitting ──────────────────────────────────────────────────────

    #[test]
    fn ease_fit_small() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        assert_eq!(model.weights.len(), 9);
    }

    #[test]
    fn ease_weights_diagonal_zero() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        let n = 3;
        for i in 0..n {
            assert_eq!(model.weights[i * n + i], 0.0, "W[{i},{i}] should be 0");
        }
    }

    #[test]
    fn ease_weights_shape_correct() {
        let n_items = 5;
        let n_users = 4;
        let x: Vec<f32> = vec![1.0; n_users * n_items];
        let cfg = EaseConfig {
            n_users,
            n_items,
            lambda: 10.0,
            ..EaseConfig::default()
        };
        let model = Ease::fit(&x, cfg).unwrap();
        assert_eq!(model.weights.len(), n_items * n_items);
    }

    // ── Prediction ─────────────────────────────────────────────────────────

    #[test]
    fn ease_predict_shape() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        let user_row = vec![1.0, 0.0, 0.0];
        let scores = model.predict(&user_row).unwrap();
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn ease_predict_batch_shape() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        let scores = model.predict_batch(&x, 3).unwrap();
        assert_eq!(scores.len(), 9);
    }

    // ── Recommendations ────────────────────────────────────────────────────

    #[test]
    fn ease_recommend_top_k_length() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        let user_row = vec![1.0, 0.0, 0.0];
        let recs = model.recommend_top_k(&user_row, 2, false).unwrap();
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn ease_recommend_excludes_interacted() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        // User has interacted with items 0 and 1
        let user_row = vec![1.0, 1.0, 0.0];
        let recs = model.recommend_top_k(&user_row, 1, true).unwrap();
        for &idx in &recs {
            assert!(
                user_row[idx] == 0.0,
                "Recommendation {idx} should not be an already-interacted item"
            );
        }
    }

    #[test]
    fn ease_recommend_not_excluding() {
        let x = small_interactions();
        let cfg = default_cfg(3, 3);
        let model = Ease::fit(&x, cfg).unwrap();
        // All candidates considered — length should still be k
        let user_row = vec![1.0, 1.0, 0.0];
        let recs = model.recommend_top_k(&user_row, 3, false).unwrap();
        assert_eq!(recs.len(), 3);
    }

    // ── Regularization effects ─────────────────────────────────────────────

    #[test]
    fn ease_high_lambda_forces_small_weights() {
        let x = small_interactions();
        let cfg = EaseConfig {
            n_users: 3,
            n_items: 3,
            lambda: 1_000_000.0,
            ..EaseConfig::default()
        };
        let model = Ease::fit(&x, cfg).unwrap();
        for (i, &w) in model.weights.iter().enumerate() {
            // Skip diagonal
            if i % 4 == 0 {
                continue;
            }
            assert!(
                w.abs() < 0.01,
                "Weight {w} too large at index {i} with high lambda"
            );
        }
    }

    #[test]
    fn ease_low_lambda_allows_larger_weights() {
        let x = small_interactions();
        let cfg = EaseConfig {
            n_users: 3,
            n_items: 3,
            lambda: 1.0,
            ..EaseConfig::default()
        };
        let model = Ease::fit(&x, cfg).unwrap();
        let max_off_diag = model
            .weights
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 != 0)
            .map(|(_, &w)| w.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_off_diag > 0.01,
            "Expected larger weights with low lambda, got max={max_off_diag}"
        );
    }

    // ── Gram matrix ────────────────────────────────────────────────────────

    #[test]
    fn gram_matrix_symmetric() {
        let x = small_interactions();
        let g = Ease::compute_gram(&x, 3, 3, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (g[i * 3 + j] - g[j * 3 + i]).abs() < 1e-6,
                    "G[{i},{j}]={} ≠ G[{j},{i}]={}",
                    g[i * 3 + j],
                    g[j * 3 + i]
                );
            }
        }
    }

    #[test]
    fn gram_matrix_diagonal_includes_lambda() {
        let x = small_interactions();
        let lambda = 7.0;
        let g = Ease::compute_gram(&x, 3, 3, lambda);
        for i in 0..3 {
            assert!(
                g[i * 3 + i] >= lambda,
                "G[{i},{i}]={} should be >= lambda={lambda}",
                g[i * 3 + i]
            );
        }
    }

    // ── Cholesky ───────────────────────────────────────────────────────────

    #[test]
    fn cholesky_identity_gives_identity() {
        let lambda = 4.0;
        // A = I + lambda*I = (1+lambda)*I
        let n = 3;
        let scale = 1.0 + lambda;
        let mut a: Vec<f32> = (0..n * n)
            .map(|k| if k % (n + 1) == 0 { scale } else { 0.0 })
            .collect();
        Ease::cholesky(&mut a, n).unwrap();
        for i in 0..n {
            let diag = a[i * n + i];
            assert!(
                (diag - scale.sqrt()).abs() < 1e-5,
                "L[{i},{i}] = {diag}, expected {}",
                scale.sqrt()
            );
        }
    }

    #[test]
    fn cholesky_fails_non_positive_definite() {
        let n = 2;
        // A = [[1, 2],[2, 1]] — not positive definite (det < 0)
        let mut a = vec![1.0_f32, 2.0, 2.0, 1.0];
        let result = Ease::cholesky(&mut a, n);
        assert!(
            result.is_err(),
            "Cholesky should fail on non-positive-definite matrix"
        );
    }

    // ── Matrix inversion ───────────────────────────────────────────────────

    #[test]
    fn invert_cholesky_product_is_identity() {
        // G = X^T X + λI for the small example — then G * G_inv ≈ I
        let x = small_interactions();
        let lambda = 500.0;
        let gram = Ease::compute_gram(&x, 3, 3, lambda);
        let gram_orig = gram.clone();
        let g_inv = Ease::invert_via_cholesky(gram, 3).unwrap();

        // Check G * G_inv ≈ I
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let dot: f32 = (0..n)
                    .map(|k| gram_orig[i * n + k] * g_inv[k * n + j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-3,
                    "G*G_inv[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    // ── EASER ──────────────────────────────────────────────────────────────

    #[test]
    fn easer_lambda_l1_zero_matches_ease() {
        let x = small_interactions();
        let cfg_ease = default_cfg(3, 3);
        let cfg_easer = EaseConfig {
            n_users: 3,
            n_items: 3,
            lambda: 500.0,
            lambda_l1: 0.0,
            ..EaseConfig::default()
        };
        let ease = Ease::fit(&x, cfg_ease).unwrap();
        let easer = Ease::fit(&x, cfg_easer).unwrap();

        for (i, (&w_e, &w_r)) in ease.weights.iter().zip(easer.weights.iter()).enumerate() {
            assert!(
                (w_e - w_r).abs() < 1e-5,
                "Weight mismatch at {i}: EASE={w_e} EASER={w_r}"
            );
        }
    }

    #[test]
    fn easer_l1_reduces_weights() {
        let x = small_interactions();
        let cfg_ease = default_cfg(3, 3);
        let cfg_easer = EaseConfig {
            n_users: 3,
            n_items: 3,
            lambda: 500.0,
            lambda_l1: 1_000.0, // strong L1 → heavy sparsification
            l1_iter: 100,
            l1_tol: 1e-6,
        };
        let ease = Ease::fit(&x, cfg_ease).unwrap();
        let easer = Ease::fit(&x, cfg_easer).unwrap();

        let ease_sum: f32 = ease.weights.iter().map(|w| w.abs()).sum();
        let easer_sum: f32 = easer.weights.iter().map(|w| w.abs()).sum();
        assert!(
            easer_sum <= ease_sum,
            "EASER L1 sum {easer_sum} should be ≤ EASE sum {ease_sum}"
        );
    }

    // ── Error cases ────────────────────────────────────────────────────────

    #[test]
    fn err_n_items_zero() {
        let cfg = EaseConfig {
            n_users: 3,
            n_items: 0,
            lambda: 1.0,
            ..EaseConfig::default()
        };
        let result = Ease::fit(&[], cfg);
        assert!(
            matches!(result, Err(RecsysError::InvalidNumItems { .. })),
            "Expected InvalidNumItems error"
        );
    }

    #[test]
    fn err_lambda_negative() {
        let cfg = EaseConfig {
            n_users: 3,
            n_items: 3,
            lambda: -1.0,
            ..EaseConfig::default()
        };
        let x = small_interactions();
        let result = Ease::fit(&x, cfg);
        assert!(
            matches!(result, Err(RecsysError::InvalidLambda { .. })),
            "Expected InvalidLambda error"
        );
    }

    #[test]
    fn err_interaction_length_mismatch() {
        let cfg = EaseConfig {
            n_users: 3,
            n_items: 3,
            lambda: 1.0,
            ..EaseConfig::default()
        };
        // Provide only 5 values instead of 3*3=9
        let x = vec![1.0_f32; 5];
        let result = Ease::fit(&x, cfg);
        assert!(
            matches!(result, Err(RecsysError::DimensionMismatch { .. })),
            "Expected DimensionMismatch error"
        );
    }
}
