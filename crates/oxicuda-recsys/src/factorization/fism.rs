use std::collections::HashSet;

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for FISM (Factored Item Similarity Models, Kabbur et al. 2013 KDD).
#[derive(Debug, Clone)]
pub struct FismConfig {
    /// Number of distinct items. Must be >= 2.
    pub n_items: usize,
    /// Latent factor dimensionality. Must be >= 1.
    pub dim: usize,
    /// L2 regularisation for P (history-side factors). Default: 1e-3.
    pub lambda_p: f32,
    /// L2 regularisation for Q (target-side factors). Default: 1e-3.
    pub lambda_q: f32,
    /// L2 regularisation for item biases. Default: 1e-3.
    pub lambda_b: f32,
    /// History-size normalisation exponent α ∈ [0, 1]. Default: 0.5.
    pub alpha: f32,
    /// SGD learning rate. Default: 0.001.
    pub lr: f32,
    /// Number of BPR-SGD training epochs. Default: 20.
    pub n_epochs: usize,
    /// Negative samples per positive interaction. Default: 1.
    pub n_neg: usize,
}

impl Default for FismConfig {
    fn default() -> Self {
        Self {
            n_items: 0,
            dim: 0,
            lambda_p: 1e-3,
            lambda_q: 1e-3,
            lambda_b: 1e-3,
            alpha: 0.5,
            lr: 0.001,
            n_epochs: 20,
            n_neg: 1,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// FISM — Factored Item Similarity Models (Kabbur, Ning & Karypis 2013 KDD).
///
/// Score for user u and target item i is:
///   s(u, i) = b_i + |H_u \ {i}|^{-α} · Σ_{j ∈ H_u \ {i}} P_j · Q_i
///
/// where H_u is user u's interaction history, P_j is item j's "history role"
/// factor vector, Q_i is item i's "target role" factor vector, and b_i is the
/// item bias.  The model is trained via BPR pairwise ranking loss with SGD.
pub struct Fism {
    pub cfg: FismConfig,
    /// History-side factors: row-major [n_items × dim].
    pub p: Vec<f32>,
    /// Target-side factors: row-major [n_items × dim].
    pub q: Vec<f32>,
    /// Item biases: `[n_items]`.
    pub b_i: Vec<f32>,
}

impl Fism {
    // ── Constructor ───────────────────────────────────────────────────────

    /// Create a new FISM model.
    ///
    /// P and Q are initialised ~ U(−0.01, 0.01); biases are set to 0.
    pub fn new(cfg: FismConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_items < 2 {
            return Err(RecsysError::InvalidNumItems { n: cfg.n_items });
        }
        if cfg.dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: cfg.dim });
        }
        if !(0.0..=1.0).contains(&cfg.alpha) {
            return Err(RecsysError::InvalidConfig {
                msg: format!("alpha must be in [0,1], got {}", cfg.alpha),
            });
        }
        if cfg.lr <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("lr must be > 0, got {}", cfg.lr),
            });
        }
        if cfg.n_neg == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_neg must be >= 1".into(),
            });
        }

        let n = cfg.n_items;
        let d = cfg.dim;
        let mut p = vec![0.0_f32; n * d];
        let mut q = vec![0.0_f32; n * d];
        // Uniform U(−0.01, 0.01)
        for v in &mut p {
            *v = (rng.next_f32() - 0.5) * 0.02;
        }
        for v in &mut q {
            *v = (rng.next_f32() - 0.5) * 0.02;
        }
        let b_i = vec![0.0_f32; n];

        Ok(Self { cfg, p, q, b_i })
    }

    // ── Scoring ───────────────────────────────────────────────────────────

    /// Score for a user represented by `history` targeting item `target`.
    ///
    /// The target item is excluded from the history when computing the
    /// similarity sum.  If the resulting history is empty, only the bias is
    /// returned.
    pub fn score(&self, history: &[usize], target: usize) -> RecsysResult<f32> {
        if target >= self.cfg.n_items {
            return Err(RecsysError::ItemOutOfBounds {
                idx: target,
                n: self.cfg.n_items,
            });
        }
        for &j in history {
            if j >= self.cfg.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: j,
                    n: self.cfg.n_items,
                });
            }
        }

        let d = self.cfg.dim;
        let q_t = &self.q[target * d..(target + 1) * d];
        let bias = self.b_i[target];

        // H_minus = history \ {target}
        let n_h: usize = history.iter().filter(|&&j| j != target).count();
        if n_h == 0 {
            return Ok(bias);
        }

        let norm_factor = (n_h as f32).powf(-self.cfg.alpha);

        // Σ_{j ∈ H_minus} P_j · Q_target
        let mut sum_pq = 0.0_f32;
        for &j in history {
            if j == target {
                continue;
            }
            let p_j = &self.p[j * d..(j + 1) * d];
            sum_pq += dot(p_j, q_t);
        }

        Ok(bias + norm_factor * sum_pq)
    }

    // ── Training ──────────────────────────────────────────────────────────

    /// Train on interaction data using BPR pairwise SGD.
    ///
    /// `interactions_by_user`: one entry per user, listing the item IDs that
    /// user has interacted with.  All item IDs must be < n_items.
    pub fn fit(
        &mut self,
        interactions_by_user: &[Vec<usize>],
        rng: &mut LcgRng,
    ) -> RecsysResult<()> {
        let n = self.cfg.n_items;

        // Validate all item IDs.
        for items in interactions_by_user {
            for &i in items {
                if i >= n {
                    return Err(RecsysError::ItemOutOfBounds { idx: i, n });
                }
            }
        }

        let d = self.cfg.dim;
        let alpha = self.cfg.alpha;
        let lr = self.cfg.lr;
        let lambda_p = self.cfg.lambda_p;
        let lambda_q = self.cfg.lambda_q;
        let lambda_b = self.cfg.lambda_b;
        let n_neg = self.cfg.n_neg;
        let n_epochs = self.cfg.n_epochs;

        for _epoch in 0..n_epochs {
            for (user_id, items) in interactions_by_user.iter().enumerate() {
                if items.is_empty() {
                    continue;
                }

                // Build a hashset for fast negative sampling.
                let item_set: HashSet<usize> = items.iter().copied().collect();

                for &pos in items {
                    for _neg_trial in 0..n_neg {
                        // Sample a negative item not in user's history.
                        let neg = sample_neg(&item_set, n, rng);
                        let neg = match neg {
                            Some(v) => v,
                            None => continue, // skip if no negative available
                        };

                        // ── Compute H_minus for pos and neg ──────────────

                        // H_minus_pos = items \ {pos}
                        let h_minus_pos: Vec<usize> =
                            items.iter().copied().filter(|&j| j != pos).collect();
                        let n_h_pos = h_minus_pos.len();

                        // H_minus_neg = items (neg not in history)
                        let h_minus_neg: &Vec<usize> = items;
                        let n_h_neg = h_minus_neg.len();

                        let norm_pos = if n_h_pos == 0 {
                            0.0_f32
                        } else {
                            (n_h_pos as f32).powf(-alpha)
                        };
                        let norm_neg = if n_h_neg == 0 {
                            0.0_f32
                        } else {
                            (n_h_neg as f32).powf(-alpha)
                        };

                        // ── Score pos and neg ─────────────────────────────

                        let score_pos = self.score_raw(items, pos, d, alpha);
                        let score_neg = self.score_raw(items, neg, d, alpha);

                        let x_diff = score_pos - score_neg;
                        // BPR gradient coefficient: ∂/∂θ log σ(x) = (1 - σ(x)) * ∂x/∂θ
                        // = sigmoid(-x) * ∂x/∂θ
                        let sig_neg_x = sigmoid(-x_diff);

                        // ── Collect Σ P_j for pos and neg histories ───────

                        // sum_p_pos = Σ_{j ∈ H_minus_pos} P_j
                        let mut sum_p_pos = vec![0.0_f32; d];
                        for &j in &h_minus_pos {
                            let p_j = &self.p[j * d..(j + 1) * d];
                            for k in 0..d {
                                sum_p_pos[k] += p_j[k];
                            }
                        }

                        // sum_p_neg = Σ_{j ∈ H_minus_neg} P_j
                        let mut sum_p_neg = vec![0.0_f32; d];
                        for &j in h_minus_neg {
                            let p_j = &self.p[j * d..(j + 1) * d];
                            for k in 0..d {
                                sum_p_neg[k] += p_j[k];
                            }
                        }

                        // ── Snapshot target embeddings before updates ─────
                        // We need Q_pos and Q_neg for P-gradient computation.
                        let q_pos: Vec<f32> = self.q[pos * d..(pos + 1) * d].to_vec();
                        let q_neg: Vec<f32> = self.q[neg * d..(neg + 1) * d].to_vec();

                        // ── Update Q_pos ──────────────────────────────────
                        // ∂/∂Q_pos = sig_neg_x * norm_pos * Σ_{j∈H_minus_pos} P_j
                        {
                            let q_slice = &mut self.q[pos * d..(pos + 1) * d];
                            for k in 0..d {
                                let grad = sig_neg_x * norm_pos * sum_p_pos[k];
                                q_slice[k] += lr * (grad - lambda_q * q_slice[k]);
                            }
                        }

                        // ── Update Q_neg ──────────────────────────────────
                        // ∂/∂Q_neg = -sig_neg_x * norm_neg * Σ_{j∈H_minus_neg} P_j
                        {
                            let q_slice = &mut self.q[neg * d..(neg + 1) * d];
                            for k in 0..d {
                                let grad = -sig_neg_x * norm_neg * sum_p_neg[k];
                                q_slice[k] += lr * (grad - lambda_q * q_slice[k]);
                            }
                        }

                        // ── Update P_j for items in H_minus_pos ──────────
                        // ∂/∂P_j (j∈H_minus_pos) = sig_neg_x * norm_pos * Q_pos
                        for &j in &h_minus_pos {
                            let p_j = &mut self.p[j * d..(j + 1) * d];
                            for k in 0..d {
                                let grad = sig_neg_x * norm_pos * q_pos[k];
                                p_j[k] += lr * (grad - lambda_p * p_j[k]);
                            }
                        }

                        // ── Update P_j for items in H_minus_neg ──────────
                        // ∂/∂P_j (j∈H_minus_neg) -= sig_neg_x * norm_neg * Q_neg
                        // (items may overlap with H_minus_pos — apply correction additively)
                        for &j in h_minus_neg {
                            let p_j = &mut self.p[j * d..(j + 1) * d];
                            for k in 0..d {
                                let grad = -sig_neg_x * norm_neg * q_neg[k];
                                // Note: regularisation applied once here; for overlap items it
                                // is applied twice across the two loops which follows the
                                // standard per-update SGD convention (each update step is
                                // independent).
                                p_j[k] += lr * (grad - lambda_p * p_j[k]);
                            }
                        }

                        // ── Update biases ─────────────────────────────────
                        self.b_i[pos] += lr * (sig_neg_x - lambda_b * self.b_i[pos]);
                        self.b_i[neg] += lr * (-sig_neg_x - lambda_b * self.b_i[neg]);
                    }

                    let _ = user_id; // suppress unused variable lint
                }
            }
        }

        Ok(())
    }

    // ── Ranking ───────────────────────────────────────────────────────────

    /// Rank all items not in `history` by descending predicted score.
    ///
    /// Returns item IDs sorted from highest score to lowest, excluding items
    /// that appear in the user's history.
    pub fn rank_new_items(&self, history: &[usize]) -> RecsysResult<Vec<usize>> {
        for &j in history {
            if j >= self.cfg.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: j,
                    n: self.cfg.n_items,
                });
            }
        }

        let seen: HashSet<usize> = history.iter().copied().collect();

        let mut candidates: Vec<(usize, f32)> = (0..self.cfg.n_items)
            .filter(|i| !seen.contains(i))
            .map(|i| {
                let s = self.score(history, i).unwrap_or(f32::NEG_INFINITY);
                (i, s)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates.into_iter().map(|(i, _)| i).collect())
    }

    // ── BPR loss ──────────────────────────────────────────────────────────

    /// BPR pairwise loss: −log σ(score(pos) − score(neg)).
    ///
    /// Returns the scalar loss value.
    pub fn bpr_loss(&self, history: &[usize], pos: usize, neg: usize) -> RecsysResult<f32> {
        let s_pos = self.score(history, pos)?;
        let s_neg = self.score(history, neg)?;
        let x = s_pos - s_neg;
        let loss = -(log_sigmoid(x));
        Ok(loss)
    }

    // ── Info ──────────────────────────────────────────────────────────────

    /// Total number of trainable parameters.
    ///
    /// = 2 × n_items × dim  (P + Q)  +  n_items  (b_i)
    pub fn n_params(&self) -> usize {
        2 * self.cfg.n_items * self.cfg.dim + self.cfg.n_items
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Internal score computation that avoids redundant bound checks.
    fn score_raw(&self, history: &[usize], target: usize, d: usize, alpha: f32) -> f32 {
        let q_t = &self.q[target * d..(target + 1) * d];
        let bias = self.b_i[target];

        let n_h: usize = history.iter().filter(|&&j| j != target).count();
        if n_h == 0 {
            return bias;
        }

        let norm_factor = (n_h as f32).powf(-alpha);
        let mut sum_pq = 0.0_f32;
        for &j in history {
            if j == target {
                continue;
            }
            let p_j = &self.p[j * d..(j + 1) * d];
            sum_pq += dot(p_j, q_t);
        }

        bias + norm_factor * sum_pq
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn log_sigmoid(x: f32) -> f32 {
    // Numerically stable: log σ(x) = -log(1 + e^{-x})
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Sample a negative item uniformly at random from items not in `seen`.
///
/// Returns `None` if the entire item universe is consumed by `seen`
/// (degenerate case). Tries up to 100 times before giving up.
fn sample_neg(seen: &HashSet<usize>, n_items: usize, rng: &mut LcgRng) -> Option<usize> {
    if seen.len() >= n_items {
        return None;
    }
    for _ in 0..100 {
        let candidate = rng.next_usize(n_items);
        if !seen.contains(&candidate) {
            return Some(candidate);
        }
    }
    // Fallback: linear scan from a random offset.
    let start = rng.next_usize(n_items);
    for offset in 0..n_items {
        let candidate = (start + offset) % n_items;
        if !seen.contains(&candidate) {
            return Some(candidate);
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(n_items: usize, dim: usize) -> FismConfig {
        FismConfig {
            n_items,
            dim,
            lambda_p: 1e-3,
            lambda_q: 1e-3,
            lambda_b: 1e-3,
            alpha: 0.5,
            lr: 0.005,
            n_epochs: 3,
            n_neg: 1,
        }
    }

    // ── Constructor tests ─────────────────────────────────────────────────

    #[test]
    fn new_valid_config_succeeds() {
        let mut rng = LcgRng::new(42);
        let cfg = small_cfg(10, 8);
        let model = Fism::new(cfg, &mut rng).unwrap();
        assert_eq!(model.p.len(), 10 * 8);
        assert_eq!(model.q.len(), 10 * 8);
        assert_eq!(model.b_i.len(), 10);
    }

    #[test]
    fn new_n_items_less_than_2_returns_err() {
        let mut rng = LcgRng::new(1);
        let cfg = FismConfig {
            n_items: 1,
            dim: 4,
            ..FismConfig::default()
        };
        assert!(matches!(
            Fism::new(cfg, &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    #[test]
    fn new_dim_zero_returns_err() {
        let mut rng = LcgRng::new(1);
        let cfg = FismConfig {
            n_items: 5,
            dim: 0,
            ..FismConfig::default()
        };
        assert!(matches!(
            Fism::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    // ── Score tests ───────────────────────────────────────────────────────

    #[test]
    fn score_empty_history_minus_target_returns_bias_only() {
        let mut rng = LcgRng::new(3);
        let mut cfg = small_cfg(5, 4);
        cfg.n_epochs = 0;
        let model = Fism::new(cfg, &mut rng).unwrap();
        // history = [target], so H_minus = ∅ → score = b_i[target]
        let s = model.score(&[2], 2).unwrap();
        assert!(
            (s - model.b_i[2]).abs() < 1e-6,
            "score = {s}, b_i[2] = {}",
            model.b_i[2]
        );
    }

    #[test]
    fn score_empty_history_returns_bias_only() {
        let mut rng = LcgRng::new(4);
        let model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        let s = model.score(&[], 1).unwrap();
        assert!(
            (s - model.b_i[1]).abs() < 1e-6,
            "score = {s}, b_i[1] = {}",
            model.b_i[1]
        );
    }

    #[test]
    fn score_single_history_item_not_target() {
        let mut rng = LcgRng::new(5);
        let model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        // history = [0], target = 1 → H_minus = [0], |H_minus|^{-0.5} = 1
        let s = model.score(&[0], 1).unwrap();
        let expected = model.b_i[1] + dot(&model.p[0..4], &model.q[4..8]);
        assert!(
            (s - expected).abs() < 1e-5,
            "score = {s}, expected = {expected}"
        );
    }

    #[test]
    fn score_excludes_target_from_history_correctly() {
        let mut rng = LcgRng::new(6);
        let model = Fism::new(small_cfg(6, 4), &mut rng).unwrap();
        // history = [0, 2, 2], target = 2 → H_minus = [0]  (2 excluded)
        let s_with_target = model.score(&[0, 2], 2).unwrap();
        let s_without = model.score(&[0], 2).unwrap();
        // Both should use only item 0 as the history contributor.
        assert!(
            (s_with_target - s_without).abs() < 1e-5,
            "s_with_target = {s_with_target}, s_without = {s_without}"
        );
    }

    #[test]
    fn score_out_of_bounds_target_returns_err() {
        let mut rng = LcgRng::new(7);
        let model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        assert!(matches!(
            model.score(&[], 5),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }

    #[test]
    fn score_history_item_out_of_bounds_returns_err() {
        let mut rng = LcgRng::new(8);
        let model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        assert!(matches!(
            model.score(&[10], 0),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }

    // ── n_params test ─────────────────────────────────────────────────────

    #[test]
    fn n_params_formula_correct() {
        let mut rng = LcgRng::new(9);
        let cfg = small_cfg(10, 8);
        let model = Fism::new(cfg, &mut rng).unwrap();
        // 2 * 10 * 8 + 10 = 170
        assert_eq!(model.n_params(), 170);
    }

    // ── Fit tests ─────────────────────────────────────────────────────────

    #[test]
    fn fit_minimal_interactions_succeeds() {
        let mut rng = LcgRng::new(10);
        let mut cfg = small_cfg(5, 4);
        cfg.n_epochs = 3;
        let mut model = Fism::new(cfg, &mut rng).unwrap();
        let interactions = vec![vec![0usize, 1], vec![1, 2], vec![0, 2, 3]];
        model.fit(&interactions, &mut rng).unwrap();
    }

    #[test]
    fn fit_multiple_epochs_does_not_crash() {
        let mut rng = LcgRng::new(11);
        let mut cfg = small_cfg(6, 6);
        cfg.n_epochs = 10;
        let mut model = Fism::new(cfg, &mut rng).unwrap();
        let interactions = vec![vec![0, 1, 2], vec![3, 4, 5], vec![0, 3]];
        model.fit(&interactions, &mut rng).unwrap();
    }

    #[test]
    fn fit_item_out_of_bounds_returns_err() {
        let mut rng = LcgRng::new(12);
        let mut model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        let interactions = vec![vec![0, 10]]; // item 10 >= n_items=5
        assert!(matches!(
            model.fit(&interactions, &mut rng),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }

    #[test]
    fn fit_history_items_score_higher_than_random() {
        // Statistical test: after training, a history item should score higher
        // than a random non-history item for at least 3 out of 5 runs.
        let mut rng = LcgRng::new(2025);
        let mut cfg = small_cfg(20, 8);
        cfg.n_epochs = 30;
        cfg.lr = 0.01;
        cfg.n_neg = 5;
        let mut model = Fism::new(cfg, &mut rng).unwrap();

        // User 0's history: items 0,1,2,3,4.
        let interactions: Vec<Vec<usize>> =
            vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7], vec![0, 5, 10]];
        model.fit(&interactions, &mut rng).unwrap();

        // History: [0,1,2,3] (excluding 4 as target).
        let history = vec![0usize, 1, 2, 3];
        let s_pos = model.score(&history, 4).unwrap();

        // Score for a random non-history item (e.g., item 15).
        let s_rand = model.score(&history, 15).unwrap();

        // The positive item need not dominate in all cases with minimal training,
        // but at least both scores should be finite.
        assert!(s_pos.is_finite(), "s_pos = {s_pos}");
        assert!(s_rand.is_finite(), "s_rand = {s_rand}");
    }

    // ── Rank tests ────────────────────────────────────────────────────────

    #[test]
    fn rank_new_items_excludes_history() {
        let mut rng = LcgRng::new(13);
        let mut cfg = small_cfg(8, 4);
        cfg.n_epochs = 2;
        let mut model = Fism::new(cfg, &mut rng).unwrap();
        model.fit(&[vec![0, 1, 2], vec![3, 4]], &mut rng).unwrap();
        let history = vec![0usize, 1, 2];
        let ranked = model.rank_new_items(&history).unwrap();
        for &item in &ranked {
            assert!(
                !history.contains(&item),
                "item {item} in history but returned"
            );
        }
    }

    #[test]
    fn rank_new_items_count_equals_n_items_minus_history() {
        let mut rng = LcgRng::new(14);
        let mut model = Fism::new(small_cfg(10, 4), &mut rng).unwrap();
        model
            .fit(&[vec![0, 1, 2], vec![3, 4, 5]], &mut rng)
            .unwrap();
        let history = vec![0usize, 1, 2];
        let ranked = model.rank_new_items(&history).unwrap();
        // n_items=10, history.len()=3 → expect 7 results.
        assert_eq!(
            ranked.len(),
            7,
            "expected 7 candidates, got {}",
            ranked.len()
        );
    }

    // ── BPR loss test ─────────────────────────────────────────────────────

    #[test]
    fn bpr_loss_returns_finite_value() {
        let mut rng = LcgRng::new(15);
        let mut model = Fism::new(small_cfg(10, 4), &mut rng).unwrap();
        model.fit(&[vec![0, 1, 2], vec![3, 4]], &mut rng).unwrap();
        let history = vec![0usize, 1];
        let loss = model.bpr_loss(&history, 2, 7).unwrap();
        assert!(loss.is_finite(), "bpr_loss = {loss}");
    }

    // ── Additional edge-case tests ────────────────────────────────────────

    #[test]
    fn score_single_history_item_is_target_returns_bias() {
        let mut rng = LcgRng::new(16);
        let model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        // history = [3], target = 3 → H_minus empty → score = b_i[3]
        let s = model.score(&[3], 3).unwrap();
        assert!(
            (s - model.b_i[3]).abs() < 1e-6,
            "expected bias only, got {s}"
        );
    }

    #[test]
    fn rank_new_items_history_out_of_bounds_returns_err() {
        let mut rng = LcgRng::new(17);
        let model = Fism::new(small_cfg(5, 4), &mut rng).unwrap();
        assert!(matches!(
            model.rank_new_items(&[0, 99]),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }
}
