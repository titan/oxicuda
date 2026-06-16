//! R2D2 — Differentiable Ridge Regression Base Learner (Bertinetto et al., NeurIPS 2019).
//!
//! Uses closed-form ridge regression as the meta-learned classifier head.
//! The solution W* = (Φ^T Φ + λI)^{-1} Φ^T Y is differentiable w.r.t. the embedding
//! weights Φ, enabling end-to-end meta-training of the feature extractor.
//!
//! Reference: <https://arxiv.org/abs/1805.09567>

use crate::episode::types::FewShotEpisode;
use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for R2D2.
#[derive(Debug, Clone)]
pub struct R2D2Config {
    /// Input feature dimension (embedding output dim = feat_dim).
    pub feat_dim: usize,
    /// Number of classes for N-way classification.
    pub n_way: usize,
    /// Ridge regularisation coefficient λ > 0.
    pub lambda: f32,
    /// If true and n_support < feat_dim, apply the Woodbury identity for efficiency.
    pub use_woodbury: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Weights
// ─────────────────────────────────────────────────────────────────────────────

/// Linear embedding projection weights (input → φ space) for R2D2.
#[derive(Debug, Clone)]
pub struct R2D2Weights {
    /// Row-major [feat_dim × input_dim]: w[j * input_dim + k] = W[j, k].
    pub embedding_w: Vec<f32>,
    /// Bias vector `[feat_dim]`.
    pub embedding_b: Vec<f32>,
    /// Dimension of the raw input before projection.
    pub input_dim: usize,
    /// Projected feature dimension (== feat_dim in config).
    pub feat_dim: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main struct
// ─────────────────────────────────────────────────────────────────────────────

/// R2D2 few-shot base learner.
pub struct R2D2 {
    /// Configuration.
    pub config: R2D2Config,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky-Banachiewicz factorisation of symmetric positive-definite A (n × n, row-major)
/// in-place.  On return `a` holds the lower-triangular factor L (A = L Lᵀ).
fn cholesky_in_place(a: &mut [f32], n: usize) -> MetaResult<()> {
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(MetaError::Internal {
                        msg: format!(
                            "Cholesky failed: non-positive pivot {s:.6e} at diagonal ({i},{i})"
                        ),
                    });
                }
                a[i * n + i] = s.sqrt();
            } else {
                let diag = a[j * n + j];
                if diag.abs() < 1e-15 {
                    return Err(MetaError::Internal {
                        msg: format!("Cholesky failed: near-zero diagonal {diag:.6e} at ({j},{j})"),
                    });
                }
                a[i * n + j] = s / diag;
            }
        }
    }
    Ok(())
}

/// Forward substitution: solve L y = b where L is lower-triangular (n × n, row-major).
fn forward_sub(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    y
}

/// Back substitution: solve Lᵀ x = y where L is lower-triangular (n × n, row-major).
fn back_sub(l: &[f32], y: &[f32], n: usize) -> Vec<f32> {
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    x
}

/// Solve A X = B via previously computed Cholesky factor L (n × n), for m RHS columns.
/// `b` is n × m row-major; returns x of shape n × m row-major.
fn cholesky_solve_multi(l: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut x = vec![0.0_f32; n * m];
    for col in 0..m {
        let b_col: Vec<f32> = (0..n).map(|r| b[r * m + col]).collect();
        let y = forward_sub(l, &b_col, n);
        let xc = back_sub(l, &y, n);
        for r in 0..n {
            x[r * m + col] = xc[r];
        }
    }
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// impl R2D2
// ─────────────────────────────────────────────────────────────────────────────

impl R2D2 {
    /// Construct a new R2D2 instance after validating configuration.
    pub fn new(config: R2D2Config) -> MetaResult<Self> {
        if config.n_way < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: config.n_way,
            });
        }
        if config.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim {
                dim: config.feat_dim,
            });
        }
        if config.lambda <= 0.0 {
            return Err(MetaError::InvalidLr { lr: config.lambda });
        }
        Ok(Self { config })
    }

    /// Initialise embedding weights with Kaiming uniform for the linear+ReLU layer.
    ///
    /// Kaiming fan-in limit = sqrt(2 / input_dim).
    /// Weights sampled in [-limit, +limit) using the LCG range trick.
    pub fn init_weights(input_dim: usize, feat_dim: usize, rng: &mut LcgRng) -> R2D2Weights {
        let limit = (2.0_f32 / input_dim.max(1) as f32).sqrt();
        let mut embedding_w = vec![0.0_f32; feat_dim * input_dim];
        for v in embedding_w.iter_mut() {
            // next_f32() -> [0, 0.5), so (next_f32() - 0.25) * 2 * limit -> [-0.5*limit*2, 0.5*limit*2)
            // i.e., approximately symmetric about 0 within [-limit, limit)
            *v = (rng.next_f32() - 0.25) * 2.0 * limit;
        }
        let embedding_b = vec![0.0_f32; feat_dim];
        R2D2Weights {
            embedding_w,
            embedding_b,
            input_dim,
            feat_dim,
        }
    }

    /// Apply the linear embedding followed by ReLU.
    ///
    /// `x`: n × input_dim row-major.  Returns φ(x): n × feat_dim row-major.
    pub fn embed(weights: &R2D2Weights, x: &[f32], n: usize) -> MetaResult<Vec<f32>> {
        let id = weights.input_dim;
        let fd = weights.feat_dim;
        if x.len() != n * id {
            return Err(MetaError::DimensionMismatch {
                expected: n * id,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; n * fd];
        for i in 0..n {
            let xi = &x[i * id..(i + 1) * id];
            for j in 0..fd {
                let w_row = &weights.embedding_w[j * id..(j + 1) * id];
                let val: f32 = w_row
                    .iter()
                    .zip(xi.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + weights.embedding_b[j];
                out[i * fd + j] = val.max(0.0); // ReLU
            }
        }
        Ok(out)
    }

    /// Encode integer labels to one-hot matrix.
    ///
    /// `labels`: (n,), values in 0..n_way.  Returns Y: n × n_way row-major.
    pub fn one_hot(labels: &[u32], n_way: usize) -> Vec<f32> {
        let n = labels.len();
        let mut y = vec![0.0_f32; n * n_way];
        for (i, &lbl) in labels.iter().enumerate() {
            let c = lbl as usize;
            if c < n_way {
                y[i * n_way + c] = 1.0;
            }
        }
        y
    }

    /// Closed-form ridge regression: W* = (Φᵀ Φ + λI)^{-1} Φᵀ Y.
    ///
    /// Uses Cholesky decomposition of the (feat_dim × feat_dim) Gram matrix.
    /// Efficient when n_support ≥ feat_dim.
    ///
    /// - `phi`: n_support × feat_dim, row-major.
    /// - `y`: n_support × n_way, row-major (one-hot).
    ///
    /// Returns W*: feat_dim × n_way, row-major.
    pub fn ridge_solve(
        phi: &[f32],
        y: &[f32],
        n_support: usize,
        feat_dim: usize,
        n_way: usize,
        lambda: f32,
    ) -> MetaResult<Vec<f32>> {
        if phi.len() != n_support * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * feat_dim,
                got: phi.len(),
            });
        }
        if y.len() != n_support * n_way {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * n_way,
                got: y.len(),
            });
        }
        if n_support == 0 {
            return Err(MetaError::EmptySupport);
        }

        let fd = feat_dim;

        // ── A = Φᵀ Φ + λI  (fd × fd) ──────────────────────────────────────
        let mut a = vec![0.0_f32; fd * fd];
        for row in phi.chunks(fd) {
            for i in 0..fd {
                for j in 0..fd {
                    a[i * fd + j] += row[i] * row[j];
                }
            }
        }
        for i in 0..fd {
            a[i * fd + i] += lambda;
        }

        // ── Cholesky of A ──────────────────────────────────────────────────
        cholesky_in_place(&mut a, fd)?;

        // ── B = Φᵀ Y  (fd × n_way) ─────────────────────────────────────────
        let mut b = vec![0.0_f32; fd * n_way];
        for (k, phi_row) in phi.chunks(fd).enumerate() {
            let y_row = &y[k * n_way..(k + 1) * n_way];
            for i in 0..fd {
                for c in 0..n_way {
                    b[i * n_way + c] += phi_row[i] * y_row[c];
                }
            }
        }

        // ── Solve A W* = B ─────────────────────────────────────────────────
        let w_star = cholesky_solve_multi(&a, &b, fd, n_way);
        Ok(w_star)
    }

    /// Woodbury-identity ridge regression: W* = Φᵀ (Φ Φᵀ + λI)^{-1} Y.
    ///
    /// Efficient when n_support < feat_dim (inner matrix is n_support × n_support).
    ///
    /// Returns W*: feat_dim × n_way, row-major.
    pub fn ridge_solve_woodbury(
        phi: &[f32],
        y: &[f32],
        n_support: usize,
        feat_dim: usize,
        n_way: usize,
        lambda: f32,
    ) -> MetaResult<Vec<f32>> {
        if phi.len() != n_support * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * feat_dim,
                got: phi.len(),
            });
        }
        if y.len() != n_support * n_way {
            return Err(MetaError::DimensionMismatch {
                expected: n_support * n_way,
                got: y.len(),
            });
        }
        if n_support == 0 {
            return Err(MetaError::EmptySupport);
        }

        let ns = n_support;
        let fd = feat_dim;

        // ── K = Φ Φᵀ + λI  (ns × ns) ──────────────────────────────────────
        let mut k_mat = vec![0.0_f32; ns * ns];
        for i in 0..ns {
            let phi_i = &phi[i * fd..(i + 1) * fd];
            for j in 0..ns {
                let phi_j = &phi[j * fd..(j + 1) * fd];
                let dot: f32 = phi_i.iter().zip(phi_j.iter()).map(|(&a, &b)| a * b).sum();
                k_mat[i * ns + j] = dot;
            }
        }
        for i in 0..ns {
            k_mat[i * ns + i] += lambda;
        }

        // ── Cholesky of K ──────────────────────────────────────────────────
        cholesky_in_place(&mut k_mat, ns)?;

        // ── Solve K Z = Y  →  Z (ns × n_way) ──────────────────────────────
        let z = cholesky_solve_multi(&k_mat, y, ns, n_way);

        // ── W* = Φᵀ Z  (fd × n_way) ────────────────────────────────────────
        let mut w_star = vec![0.0_f32; fd * n_way];
        for i in 0..ns {
            let phi_row = &phi[i * fd..(i + 1) * fd];
            let z_row = &z[i * n_way..(i + 1) * n_way];
            for j in 0..fd {
                for c in 0..n_way {
                    w_star[j * n_way + c] += phi_row[j] * z_row[c];
                }
            }
        }
        Ok(w_star)
    }

    /// Embed support, solve ridge regression, embed queries, and return logits.
    ///
    /// Returns scores: n_query × n_way row-major.
    pub fn predict(&self, weights: &R2D2Weights, episode: &FewShotEpisode) -> MetaResult<Vec<f32>> {
        let cfg = &episode.config;
        let n_support = cfg.n_way * cfg.k_shot;
        let n_query = cfg.n_way * cfg.n_query;
        let n_way = cfg.n_way;

        // Embed support
        let phi = Self::embed(weights, &episode.support_x, n_support)?;

        // One-hot encode support labels
        let y_oh = Self::one_hot(&episode.support_y, n_way);

        // Solve ridge regression
        let w_star = if self.config.use_woodbury && n_support < weights.feat_dim {
            Self::ridge_solve_woodbury(
                &phi,
                &y_oh,
                n_support,
                weights.feat_dim,
                n_way,
                self.config.lambda,
            )?
        } else {
            Self::ridge_solve(
                &phi,
                &y_oh,
                n_support,
                weights.feat_dim,
                n_way,
                self.config.lambda,
            )?
        };

        // Embed queries
        let phi_q = Self::embed(weights, &episode.query_x, n_query)?;
        let fd = weights.feat_dim;

        // Scores = Φ_q @ W*  →  n_query × n_way
        let mut scores = vec![0.0_f32; n_query * n_way];
        for i in 0..n_query {
            let phi_row = &phi_q[i * fd..(i + 1) * fd];
            for c in 0..n_way {
                let s: f32 = phi_row
                    .iter()
                    .enumerate()
                    .map(|(j, &pj)| pj * w_star[j * n_way + c])
                    .sum();
                scores[i * n_way + c] = s;
            }
        }
        Ok(scores)
    }

    /// Full episode evaluation: embed → solve → predict → argmax → accuracy ∈ [0, 1].
    pub fn evaluate_episode(
        &self,
        weights: &R2D2Weights,
        episode: &FewShotEpisode,
    ) -> MetaResult<f32> {
        let scores = self.predict(weights, episode)?;
        let n_way = episode.config.n_way;
        let n_query = episode.config.n_way * episode.config.n_query;

        let mut n_correct = 0_usize;
        for i in 0..n_query {
            let row = &scores[i * n_way..(i + 1) * n_way];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if pred == episode.query_y[i] as usize {
                n_correct += 1;
            }
        }
        Ok(n_correct as f32 / n_query as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::types::EpisodeConfig;
    use crate::handle::LcgRng;

    fn default_config() -> R2D2Config {
        R2D2Config {
            feat_dim: 8,
            n_way: 3,
            lambda: 0.1,
            use_woodbury: false,
        }
    }

    fn make_episode(
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        feat_dim: usize,
    ) -> FewShotEpisode {
        let mut rng = LcgRng::new(42);
        let n_support = n_way * k_shot;
        let n_q = n_way * n_query;
        let support_x: Vec<f32> = (0..n_support * feat_dim).map(|_| rng.next_f32()).collect();
        let support_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
            .collect();
        let query_x: Vec<f32> = (0..n_q * feat_dim).map(|_| rng.next_f32()).collect();
        let query_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, n_query))
            .collect();
        FewShotEpisode {
            config: EpisodeConfig {
                n_way,
                k_shot,
                n_query,
                feat_dim,
            },
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn new_valid_config_succeeds() {
        let r = R2D2::new(default_config());
        assert!(r.is_ok());
    }

    #[test]
    fn new_n_way_one_fails() {
        let mut cfg = default_config();
        cfg.n_way = 1;
        assert!(matches!(R2D2::new(cfg), Err(MetaError::InvalidNWay { .. })));
    }

    #[test]
    fn new_lambda_zero_fails() {
        let mut cfg = default_config();
        cfg.lambda = 0.0;
        assert!(matches!(R2D2::new(cfg), Err(MetaError::InvalidLr { .. })));
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    #[test]
    fn embed_output_shape() {
        let mut rng = LcgRng::new(1);
        let weights = R2D2::init_weights(16, 8, &mut rng);
        let x: Vec<f32> = (0..5 * 16).map(|_| rng.next_f32()).collect();
        let out = R2D2::embed(&weights, &x, 5).expect("embed should succeed");
        assert_eq!(out.len(), 5 * 8, "embed shape should be n*feat_dim");
    }

    #[test]
    fn embed_applies_relu() {
        let mut rng = LcgRng::new(2);
        // Large negative bias to force all pre-activations negative before ReLU
        let mut weights = R2D2::init_weights(4, 4, &mut rng);
        for b in weights.embedding_b.iter_mut() {
            *b = -1000.0;
        }
        let x: Vec<f32> = vec![1.0; 4];
        let out = R2D2::embed(&weights, &x, 1).expect("embed should succeed");
        for &v in &out {
            assert!(v >= 0.0, "ReLU should clip negatives to zero");
        }
    }

    #[test]
    fn embed_dimension_mismatch_error() {
        let mut rng = LcgRng::new(3);
        let weights = R2D2::init_weights(4, 8, &mut rng);
        let x = vec![0.0_f32; 3 * 5]; // wrong input_dim (5 ≠ 4)
        assert!(matches!(
            R2D2::embed(&weights, &x, 3),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── One-hot ───────────────────────────────────────────────────────────────

    #[test]
    fn one_hot_correct_index() {
        let labels = vec![2_u32];
        let oh = R2D2::one_hot(&labels, 3);
        assert_eq!(oh, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn one_hot_all_zeros_except_label() {
        let labels = vec![0_u32, 1, 2];
        let oh = R2D2::one_hot(&labels, 3);
        for (i, row) in oh.chunks(3).enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let expected = if c == i { 1.0 } else { 0.0 };
                assert_eq!(v, expected, "one_hot[{i},{c}] = {v}, expected {expected}");
            }
        }
    }

    // ── Ridge solve ───────────────────────────────────────────────────────────

    #[test]
    fn ridge_solve_identity_phi() {
        // Φ = I (4×4), Y = I (4×4), lambda small → W* ≈ (I+λI)^{-1} I = 1/(1+λ) I
        let n = 4;
        let lambda = 0.01_f32;
        let phi: Vec<f32> = (0..n * n)
            .map(|k| if k / n == k % n { 1.0_f32 } else { 0.0 })
            .collect();
        let y: Vec<f32> = phi.clone();
        let w = R2D2::ridge_solve(&phi, &y, n, n, n, lambda).expect("ridge_solve should succeed");
        // Diagonal entries should be ≈ 1/(1+lambda)
        let expected_diag = 1.0 / (1.0 + lambda);
        for i in 0..n {
            let wii = w[i * n + i];
            assert!(
                (wii - expected_diag).abs() < 1e-4,
                "W*[{i},{i}] = {wii}, expected ~{expected_diag}"
            );
        }
    }

    #[test]
    fn ridge_solve_large_lambda_shrinks_weights() {
        let n = 4;
        let phi: Vec<f32> = (0..n * n)
            .map(|k| if k / n == k % n { 1.0_f32 } else { 0.0 })
            .collect();
        let y = phi.clone();
        let w_small =
            R2D2::ridge_solve(&phi, &y, n, n, n, 0.01).expect("ridge_solve should succeed");
        let w_large =
            R2D2::ridge_solve(&phi, &y, n, n, n, 100.0).expect("ridge_solve should succeed");
        let norm_small: f32 = w_small.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_large: f32 = w_large.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            norm_large < norm_small,
            "larger lambda should produce smaller weight norm: {norm_large} vs {norm_small}"
        );
    }

    #[test]
    fn ridge_solve_n_support_one_edge_case() {
        // n_support=1, feat_dim=4, n_way=2
        let phi = vec![1.0_f32, 0.0, 0.0, 0.0];
        let y = vec![1.0_f32, 0.0]; // class 0
        let w = R2D2::ridge_solve(&phi, &y, 1, 4, 2, 1.0);
        assert!(w.is_ok(), "single support example should work");
        let w = w.expect("w should be present");
        assert!(w.iter().all(|v| v.is_finite()), "weights should be finite");
    }

    #[test]
    fn ridge_solve_output_shape() {
        let feat_dim = 6;
        let n_way = 3;
        let n_support = 9;
        let mut rng = LcgRng::new(77);
        let phi: Vec<f32> = (0..n_support * feat_dim).map(|_| rng.next_f32()).collect();
        let labels: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, 3))
            .collect();
        let y = R2D2::one_hot(&labels, n_way);
        let w = R2D2::ridge_solve(&phi, &y, n_support, feat_dim, n_way, 0.1)
            .expect("ridge_solve should succeed");
        assert_eq!(w.len(), feat_dim * n_way);
    }

    // ── Woodbury solve ────────────────────────────────────────────────────────

    #[test]
    fn woodbury_matches_standard_for_overdetermined() {
        // When n_support == feat_dim, both should give the same answer
        let n = 4;
        let lambda = 0.5_f32;
        let phi: Vec<f32> = (0..n * n)
            .map(|k| if k / n == k % n { 1.0_f32 } else { 0.5 })
            .collect();
        let labels: Vec<u32> = (0..n as u32).collect();
        let y = R2D2::one_hot(&labels, n);
        let w_std =
            R2D2::ridge_solve(&phi, &y, n, n, n, lambda).expect("ridge_solve should succeed");
        let w_wood = R2D2::ridge_solve_woodbury(&phi, &y, n, n, n, lambda)
            .expect("ridge_solve_woodbury should succeed");
        for (a, b) in w_std.iter().zip(w_wood.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "woodbury and standard should match: {a} vs {b}"
            );
        }
    }

    #[test]
    fn woodbury_output_shape() {
        let n_support = 3;
        let feat_dim = 8;
        let n_way = 3;
        let mut rng = LcgRng::new(11);
        let phi: Vec<f32> = (0..n_support * feat_dim).map(|_| rng.next_f32()).collect();
        let labels: Vec<u32> = (0..n_way as u32).collect();
        let y = R2D2::one_hot(&labels, n_way);
        let w = R2D2::ridge_solve_woodbury(&phi, &y, n_support, feat_dim, n_way, 0.1)
            .expect("ridge_solve_woodbury should succeed");
        assert_eq!(w.len(), feat_dim * n_way);
    }

    // ── Weight initialisation ─────────────────────────────────────────────────

    #[test]
    fn init_weights_not_all_zeros() {
        let mut rng = LcgRng::new(13);
        let w = R2D2::init_weights(16, 8, &mut rng);
        let any_nonzero = w.embedding_w.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "weights should not all be zero after init");
    }

    // ── Predict / evaluate ────────────────────────────────────────────────────

    #[test]
    fn predict_output_shape() {
        let cfg = default_config();
        let r2d2 = R2D2::new(cfg.clone()).expect("value should be present");
        let mut rng = LcgRng::new(99);
        let weights = R2D2::init_weights(4, cfg.feat_dim, &mut rng);
        let episode = make_episode(cfg.n_way, 2, 3, 4);
        let scores = r2d2
            .predict(&weights, &episode)
            .expect("predict should succeed");
        let n_query = cfg.n_way * 3;
        assert_eq!(scores.len(), n_query * cfg.n_way, "scores shape");
    }

    #[test]
    fn evaluate_episode_range() {
        let cfg = default_config();
        let r2d2 = R2D2::new(cfg.clone()).expect("value should be present");
        let mut rng = LcgRng::new(42);
        let weights = R2D2::init_weights(4, cfg.feat_dim, &mut rng);
        let episode = make_episode(cfg.n_way, 2, 3, 4);
        let acc = r2d2
            .evaluate_episode(&weights, &episode)
            .expect("evaluate_episode should succeed");
        assert!((0.0..=1.0).contains(&acc), "accuracy must be in [0,1]");
    }

    #[test]
    fn one_shot_works() {
        let mut cfg = default_config();
        cfg.n_way = 2;
        let r2d2 = R2D2::new(cfg.clone()).expect("value should be present");
        let mut rng = LcgRng::new(7);
        let weights = R2D2::init_weights(4, cfg.feat_dim, &mut rng);
        let episode = make_episode(2, 1, 2, 4); // k_shot=1
        let acc = r2d2
            .evaluate_episode(&weights, &episode)
            .expect("evaluate_episode should succeed");
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn binary_n_way_works() {
        let mut cfg = default_config();
        cfg.n_way = 2;
        let r2d2 = R2D2::new(cfg.clone()).expect("value should be present");
        let mut rng = LcgRng::new(8);
        let weights = R2D2::init_weights(4, cfg.feat_dim, &mut rng);
        let episode = make_episode(2, 3, 2, 4);
        let scores = r2d2
            .predict(&weights, &episode)
            .expect("predict should succeed");
        assert_eq!(scores.len(), 4 * 2);
    }

    #[test]
    fn woodbury_mode_works() {
        // n_support (n_way*k_shot=2*1=2) < feat_dim (8) → Woodbury path
        let cfg = R2D2Config {
            feat_dim: 8,
            n_way: 2,
            lambda: 0.1,
            use_woodbury: true,
        };
        let r2d2 = R2D2::new(cfg).expect("new should succeed");
        let mut rng = LcgRng::new(55);
        let weights = R2D2::init_weights(4, 8, &mut rng);
        let episode = make_episode(2, 1, 2, 4);
        let acc = r2d2
            .evaluate_episode(&weights, &episode)
            .expect("evaluate_episode should succeed");
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn cholesky_error_on_near_singular() {
        // Collinear rows → Gram almost rank-1, with tiny lambda should fail or succeed gracefully
        let feat_dim = 3;
        let n_support = 2;
        let phi = vec![1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // identical rows
        let y = vec![1.0_f32, 0.0, 0.0, 0.0]; // 2×2 one-hot
        let result = R2D2::ridge_solve(&phi, &y, n_support, feat_dim, 2, 1e-20);
        match result {
            Ok(_) | Err(MetaError::Internal { .. }) => {}
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }
}
