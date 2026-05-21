//! MetaOptNet — Differentiable Convex Solver Base Learner (Lee et al., CVPR 2019).
//!
//! Uses a closed-form convex solver (ridge regression or multi-class linear SVM)
//! as the base learner for few-shot classification.  The solutions are
//! differentiable w.r.t. the support features, enabling end-to-end
//! meta-training of the feature extractor.
//!
//! Reference: <https://arxiv.org/abs/1904.03758>

use crate::error::{MetaError, MetaResult};

// ──────────────────────────────────────────────────────────────────────────────
// Solver variant
// ──────────────────────────────────────────────────────────────────────────────

/// Which closed-form base learner to use.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaOptNetSolver {
    /// Closed-form ridge regression (RR-MetaOptNet).
    Ridge,
    /// Multi-class linear SVM solved via one-vs-all subgradient descent.
    Svm,
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for MetaOptNet.
#[derive(Debug, Clone)]
pub struct MetaOptNetConfig {
    /// N-way classification.
    pub n_way: usize,
    /// K-shot support.
    pub k_shot: usize,
    /// Feature dimension from backbone.
    pub feat_dim: usize,
    /// Regularisation: λ for ridge, C⁻¹ for SVM.
    pub reg_lambda: f32,
    /// Which convex solver to use.
    pub solver: MetaOptNetSolver,
    /// Maximum coordinate-descent iterations for SVM.
    pub max_svm_iter: usize,
    /// Convergence tolerance (squared gradient norm) for SVM.
    pub svm_tol: f32,
}

// ──────────────────────────────────────────────────────────────────────────────
// Weights produced by the inner solver
// ──────────────────────────────────────────────────────────────────────────────

/// Linear classifier weights produced by the inner convex solver.
#[derive(Debug, Clone)]
pub struct MetaOptNetWeights {
    /// n_way × feat_dim row-major weight matrix.
    pub w: Vec<f32>,
    /// n_way bias vector.
    pub b: Vec<f32>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Result
// ──────────────────────────────────────────────────────────────────────────────

/// Outputs of a full MetaOptNet forward pass.
#[derive(Debug, Clone)]
pub struct MetaOptNetResult {
    /// Classifier weights returned by the inner solver.
    pub weights: MetaOptNetWeights,
    /// Cross-entropy of the solved classifier on the query set.
    pub query_loss: f32,
    /// Fraction of correctly-classified query examples.
    pub query_accuracy: f32,
}

// ──────────────────────────────────────────────────────────────────────────────
// Main struct
// ──────────────────────────────────────────────────────────────────────────────

/// MetaOptNet few-shot base learner.
pub struct MetaOptNet {
    pub cfg: MetaOptNetConfig,
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically-stable soft-max.
fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for e in exps.iter_mut() {
        *e *= inv;
    }
    exps
}

/// Forward-substitution: solve L y = b where L is lower-triangular.
fn forward_sub(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }
    y
}

/// Back-substitution: solve L^T y = b where L is lower-triangular.
fn back_sub(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * y[j];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }
    y
}

// ──────────────────────────────────────────────────────────────────────────────
// impl MetaOptNet
// ──────────────────────────────────────────────────────────────────────────────

impl MetaOptNet {
    /// Construct a new MetaOptNet instance after validating configuration.
    pub fn new(cfg: MetaOptNetConfig) -> MetaResult<Self> {
        if cfg.n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way: cfg.n_way });
        }
        if cfg.k_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: cfg.k_shot });
        }
        if cfg.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.feat_dim });
        }
        Ok(Self { cfg })
    }

    // ── Cholesky factorisation ─────────────────────────────────────────────────

    /// Cholesky-Banachiewicz in-place factorisation of a symmetric
    /// positive-definite matrix A (n × n, row-major) into L such that A = L Lᵀ.
    ///
    /// On success `a` is overwritten with the lower-triangular Cholesky factor.
    pub fn cholesky(a: &mut [f32], n: usize) -> MetaResult<()> {
        for i in 0..n {
            for j in 0..=i {
                let mut s = a[i * n + j];
                for k in 0..j {
                    s -= a[i * n + k] * a[j * n + k];
                }
                if i == j {
                    if s <= 0.0 {
                        return Err(MetaError::Internal {
                            msg: format!("Cholesky failed: non-positive pivot {s} at ({i},{i})"),
                        });
                    }
                    a[i * n + i] = s.sqrt();
                } else {
                    let diag = a[j * n + j];
                    if diag.abs() < 1e-15 {
                        return Err(MetaError::Internal {
                            msg: format!("Cholesky failed: near-zero diagonal at ({j},{j})"),
                        });
                    }
                    a[i * n + j] = s / diag;
                }
            }
        }
        Ok(())
    }

    /// Solve A x = B where A = L Lᵀ (Cholesky), for multiple right-hand sides.
    ///
    /// - `l`: n × n lower-triangular Cholesky factor (row-major)
    /// - `b`: n × m right-hand side matrix (row-major)
    ///
    /// Returns x of shape n × m (row-major).
    pub fn cholesky_solve(l: &[f32], b: &[f32], n: usize, m: usize) -> Vec<f32> {
        let mut x = vec![0.0_f32; n * m];
        for col in 0..m {
            // Extract column of B
            let b_col: Vec<f32> = (0..n).map(|r| b[r * m + col]).collect();
            // Forward sub: L y = b_col
            let y = forward_sub(l, &b_col, n);
            // Back sub: Lᵀ x = y
            let xc = back_sub(l, &y, n);
            // Store column into row-major x
            for r in 0..n {
                x[r * m + col] = xc[r];
            }
        }
        x
    }

    // ── Ridge regression ───────────────────────────────────────────────────────

    /// Closed-form ridge regression base learner.
    ///
    /// Solves: W = (ΦᵀΦ + λI)⁻¹ Φᵀ Y
    ///
    /// where Φ ∈ ℝ^{n_support × feat_dim} and Y ∈ {0,1}^{n_support × n_way}
    /// is the one-hot label matrix.
    ///
    /// The system is solved via Cholesky decomposition of the (feat_dim × feat_dim)
    /// Gram matrix, which is efficient when feat_dim ≪ n_support.
    pub fn ridge_solve(
        support_feats: &[f32],
        support_labels: &[usize],
        cfg: &MetaOptNetConfig,
    ) -> MetaResult<MetaOptNetWeights> {
        let n_s = support_labels.len();
        let fd = cfg.feat_dim;
        let nw = cfg.n_way;

        if n_s == 0 {
            return Err(MetaError::EmptySupport);
        }
        if support_feats.len() != n_s * fd {
            return Err(MetaError::DimensionMismatch {
                expected: n_s * fd,
                got: support_feats.len(),
            });
        }

        // ── Build Gram matrix A = ΦᵀΦ + λI  (fd × fd) ───────────────────────
        let mut gram = vec![0.0_f32; fd * fd];
        // ΦᵀΦ
        for row_feat in support_feats.chunks(fd) {
            for i in 0..fd {
                for j in 0..fd {
                    gram[i * fd + j] += row_feat[i] * row_feat[j];
                }
            }
        }
        // + λI
        for i in 0..fd {
            gram[i * fd + i] += cfg.reg_lambda;
        }

        // ── Cholesky decomposition of A ────────────────────────────────────────
        Self::cholesky(&mut gram, fd)?;
        // `gram` is now the lower-triangular Cholesky factor L

        // ── Build ΦᵀY  (fd × n_way) ──────────────────────────────────────────
        let mut phy = vec![0.0_f32; fd * nw];
        for (s, feat) in support_feats.chunks(fd).enumerate() {
            let lbl = support_labels[s];
            if lbl >= nw {
                return Err(MetaError::Internal {
                    msg: format!("label {lbl} >= n_way {nw}"),
                });
            }
            for (i, &fi) in feat.iter().enumerate() {
                phy[i * nw + lbl] += fi;
            }
        }

        // ── Solve A W = ΦᵀY  →  W ∈ ℝ^{fd × nw} ────────────────────────────
        // cholesky_solve expects l (fd×fd) and b (fd×nw)
        let w_col_major = Self::cholesky_solve(&gram, &phy, fd, nw);
        // w_col_major is (fd × nw) row-major: w_col_major[i*nw + c] = W[i,c]

        // Repack into classifier: weights[c * fd .. (c+1)*fd] = W[:,c]
        let mut w = vec![0.0_f32; nw * fd];
        for c in 0..nw {
            for i in 0..fd {
                w[c * fd + i] = w_col_major[i * nw + c];
            }
        }
        let b = vec![0.0_f32; nw];

        Ok(MetaOptNetWeights { w, b })
    }

    // ── Linear SVM (one-vs-all) ────────────────────────────────────────────────

    /// Multi-class linear SVM base learner via one-vs-all subgradient descent.
    ///
    /// For each class c, solves:
    ///   min_{w_c, b_c}  ½‖w_c‖² + C Σᵢ max(0, 1 − yᵢ(Φᵢ w_c + b_c))
    ///
    /// where yᵢ ∈ {+1, −1} and C = 1 / reg_lambda.
    pub fn svm_solve(
        support_feats: &[f32],
        support_labels: &[usize],
        cfg: &MetaOptNetConfig,
    ) -> MetaResult<MetaOptNetWeights> {
        let n_s = support_labels.len();
        let fd = cfg.feat_dim;
        let nw = cfg.n_way;
        let c_svm = 1.0 / cfg.reg_lambda.max(1e-8);

        if n_s == 0 {
            return Err(MetaError::EmptySupport);
        }
        if support_feats.len() != n_s * fd {
            return Err(MetaError::DimensionMismatch {
                expected: n_s * fd,
                got: support_feats.len(),
            });
        }

        let mut w_all = vec![0.0_f32; nw * fd];
        let mut b_all = vec![0.0_f32; nw];

        for (cls, b_cls) in b_all.iter_mut().enumerate() {
            // Binary labels: +1 for this class, -1 for all others
            let y_c: Vec<f32> = support_labels
                .iter()
                .map(|&l| if l == cls { 1.0_f32 } else { -1.0_f32 })
                .collect();

            let mut wc = vec![0.0_f32; fd];
            let mut bc = 0.0_f32;

            for iter in 0..cfg.max_svm_iter {
                let step = 1.0 / (1.0 + iter as f32);

                // Subgradient: g_w = w_c − C Σ_{m_i<1} y_i Φ_i
                let mut g_w = wc.clone();
                let mut g_b = 0.0_f32;

                for (s, feat) in support_feats.chunks(fd).enumerate() {
                    let dot: f32 = feat.iter().zip(wc.iter()).map(|(&fi, &wi)| fi * wi).sum();
                    let margin = y_c[s] * (dot + bc);
                    if margin < 1.0 {
                        let yci = y_c[s];
                        for (gwi, &fi) in g_w.iter_mut().zip(feat.iter()) {
                            *gwi -= c_svm * yci * fi;
                        }
                        g_b -= c_svm * yci;
                    }
                }

                // Check convergence
                let g_norm_sq: f32 = g_w.iter().map(|&v| v * v).sum::<f32>() + g_b * g_b;
                if g_norm_sq < cfg.svm_tol * cfg.svm_tol {
                    break;
                }

                // Update
                for (wi, &gi) in wc.iter_mut().zip(g_w.iter()) {
                    *wi -= step * gi;
                }
                bc -= step * g_b;
            }

            let base = cls * fd;
            w_all[base..base + fd].copy_from_slice(&wc);
            *b_cls = bc;
        }

        Ok(MetaOptNetWeights { w: w_all, b: b_all })
    }

    // ── Query prediction ───────────────────────────────────────────────────────

    /// Evaluate a linear classifier on the query set.
    ///
    /// Returns `(mean_cross_entropy, accuracy)`.
    pub fn predict_query(
        weights: &MetaOptNetWeights,
        query_feats: &[f32],
        query_labels: &[usize],
        n_way: usize,
        feat_dim: usize,
    ) -> MetaResult<(f32, f32)> {
        let n_q = query_labels.len();
        if n_q == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        if query_feats.len() != n_q * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_q * feat_dim,
                got: query_feats.len(),
            });
        }
        if weights.w.len() != n_way * feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_way * feat_dim,
                got: weights.w.len(),
            });
        }

        let mut total_loss = 0.0_f32;
        let mut n_correct = 0_usize;

        for (q, feat) in query_feats.chunks(feat_dim).enumerate() {
            let lbl = query_labels[q];
            let logits: Vec<f32> = (0..n_way)
                .map(|c| {
                    let row = &weights.w[c * feat_dim..(c + 1) * feat_dim];
                    let dot: f32 = row.iter().zip(feat.iter()).map(|(&wi, &xi)| wi * xi).sum();
                    dot + weights.b[c]
                })
                .collect();

            let probs = softmax_slice(&logits);
            let log_p = probs[lbl].max(1e-38_f32).ln();
            if !log_p.is_finite() {
                return Err(MetaError::NanEncountered {
                    context: "predict_query log probability".into(),
                });
            }
            total_loss -= log_p;

            let pred = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if pred == lbl {
                n_correct += 1;
            }
        }

        Ok((total_loss / n_q as f32, n_correct as f32 / n_q as f32))
    }

    // ── Full forward pass ──────────────────────────────────────────────────────

    /// Full MetaOptNet forward: solve inner problem on support, then evaluate on query.
    pub fn forward(
        &self,
        support_feats: &[f32],
        support_labels: &[usize],
        query_feats: &[f32],
        query_labels: &[usize],
    ) -> MetaResult<MetaOptNetResult> {
        // Solve
        let weights = match self.cfg.solver {
            MetaOptNetSolver::Ridge => Self::ridge_solve(support_feats, support_labels, &self.cfg)?,
            MetaOptNetSolver::Svm => Self::svm_solve(support_feats, support_labels, &self.cfg)?,
        };

        // Evaluate
        let (query_loss, query_accuracy) = Self::predict_query(
            &weights,
            query_feats,
            query_labels,
            self.cfg.n_way,
            self.cfg.feat_dim,
        )?;

        Ok(MetaOptNetResult {
            weights,
            query_loss,
            query_accuracy,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ridge_cfg() -> MetaOptNetConfig {
        MetaOptNetConfig {
            n_way: 3,
            k_shot: 2,
            feat_dim: 4,
            reg_lambda: 0.1,
            solver: MetaOptNetSolver::Ridge,
            max_svm_iter: 100,
            svm_tol: 1e-4,
        }
    }

    fn default_svm_cfg() -> MetaOptNetConfig {
        MetaOptNetConfig {
            solver: MetaOptNetSolver::Svm,
            ..default_ridge_cfg()
        }
    }

    /// Orthogonal one-hot support: class c has feature e_c (standard basis).
    fn orthogonal_support(n_way: usize, k_shot: usize, feat_dim: usize) -> (Vec<f32>, Vec<usize>) {
        let n_s = n_way * k_shot;
        let mut feats = vec![0.0_f32; n_s * feat_dim];
        let mut labels = Vec::with_capacity(n_s);
        for c in 0..n_way {
            for k in 0..k_shot {
                let idx = (c * k_shot + k) * feat_dim;
                feats[idx + (c % feat_dim)] = 1.0 + k as f32 * 0.01;
            }
            for _ in 0..k_shot {
                labels.push(c);
            }
        }
        (feats, labels)
    }

    // ── Cholesky ───────────────────────────────────────────────────────────────

    #[test]
    fn cholesky_identity() {
        let n = 3;
        let mut a: Vec<f32> = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        MetaOptNet::cholesky(&mut a, n).unwrap();
        // L should also be identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (a[i * n + j] - expected).abs() < 1e-5,
                    "Chol(I)[{i},{j}] = {} ≠ {expected}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_spd_matrix() {
        // A = [[4, 2], [2, 3]]  → L = [[2, 0], [1, √2]]
        let n = 2;
        let mut a = vec![4.0_f32, 2.0, 2.0, 3.0];
        MetaOptNet::cholesky(&mut a, n).unwrap();
        assert!((a[0] - 2.0).abs() < 1e-5, "L[0,0] = {}", a[0]);
        assert!((a[2] - 1.0).abs() < 1e-5, "L[1,0] = {}", a[2]);
        assert!((a[3] - 2.0_f32.sqrt()).abs() < 1e-5, "L[1,1] = {}", a[3]);
        // Upper-triangular part should still be A's original value but we only use lower
        // No assertion needed for upper part.
    }

    #[test]
    fn cholesky_solve_basic() {
        // A = I_2, b = [3, 5]  →  x = [3, 5]
        let n = 2;
        let l = vec![1.0_f32, 0.0, 0.0, 1.0]; // Cholesky of I
        let b = vec![3.0_f32, 5.0]; // single RHS column
        let x = MetaOptNet::cholesky_solve(&l, &b, n, 1);
        assert!((x[0] - 3.0).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!((x[1] - 5.0).abs() < 1e-5, "x[1] = {}", x[1]);
    }

    // ── Ridge ──────────────────────────────────────────────────────────────────

    #[test]
    fn ridge_solve_output_shape() {
        let cfg = default_ridge_cfg();
        let (feats, labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let w = MetaOptNet::ridge_solve(&feats, &labels, &cfg).unwrap();
        assert_eq!(w.w.len(), cfg.n_way * cfg.feat_dim);
        assert_eq!(w.b.len(), cfg.n_way);
    }

    #[test]
    fn ridge_solve_finite() {
        let cfg = default_ridge_cfg();
        let (feats, labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let w = MetaOptNet::ridge_solve(&feats, &labels, &cfg).unwrap();
        assert!(w.w.iter().all(|v| v.is_finite()));
        assert!(w.b.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ridge_separable_data() {
        // Orthogonal one-hot class features → ridge should perfectly separate
        let cfg = MetaOptNetConfig {
            n_way: 3,
            k_shot: 2,
            feat_dim: 3,
            reg_lambda: 1e-4,
            solver: MetaOptNetSolver::Ridge,
            max_svm_iter: 100,
            svm_tol: 1e-4,
        };
        let (s_feats, s_labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let w = MetaOptNet::ridge_solve(&s_feats, &s_labels, &cfg).unwrap();

        // Query = the same prototypes (one-hot)
        let q_feats: Vec<f32> = (0..cfg.n_way)
            .flat_map(|c| (0..cfg.feat_dim).map(move |j| if j == c { 1.0_f32 } else { 0.0_f32 }))
            .collect();
        let q_labels: Vec<usize> = (0..cfg.n_way).collect();

        let (_, acc) =
            MetaOptNet::predict_query(&w, &q_feats, &q_labels, cfg.n_way, cfg.feat_dim).unwrap();
        assert!(
            acc >= 0.9,
            "Ridge on separable data should achieve ≥90% accuracy, got {acc}"
        );
    }

    // ── SVM ────────────────────────────────────────────────────────────────────

    #[test]
    fn svm_solve_output_shape() {
        let cfg = default_svm_cfg();
        let (feats, labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let w = MetaOptNet::svm_solve(&feats, &labels, &cfg).unwrap();
        assert_eq!(w.w.len(), cfg.n_way * cfg.feat_dim);
        assert_eq!(w.b.len(), cfg.n_way);
    }

    #[test]
    fn svm_solve_finite() {
        let cfg = default_svm_cfg();
        let (feats, labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let w = MetaOptNet::svm_solve(&feats, &labels, &cfg).unwrap();
        assert!(w.w.iter().all(|v| v.is_finite()));
        assert!(w.b.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn svm_convergence() {
        // Run SVM and verify loss decreases over iterations by checking
        // the final predictor has lower hinge loss than initial.
        let cfg = MetaOptNetConfig {
            n_way: 2,
            k_shot: 3,
            feat_dim: 4,
            reg_lambda: 0.1,
            solver: MetaOptNetSolver::Svm,
            max_svm_iter: 500,
            svm_tol: 1e-6,
        };
        let (feats, labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let w_zero = MetaOptNetWeights {
            w: vec![0.0_f32; cfg.n_way * cfg.feat_dim],
            b: vec![0.0_f32; cfg.n_way],
        };
        let q_feats: Vec<f32> = (0..cfg.n_way)
            .flat_map(|c| (0..cfg.feat_dim).map(move |j| if j == c { 1.0_f32 } else { 0.0_f32 }))
            .collect();
        let q_labels: Vec<usize> = (0..cfg.n_way).collect();

        let (loss_zero, _) =
            MetaOptNet::predict_query(&w_zero, &q_feats, &q_labels, cfg.n_way, cfg.feat_dim)
                .unwrap();
        let w_svm = MetaOptNet::svm_solve(&feats, &labels, &cfg).unwrap();
        let (loss_svm, _) =
            MetaOptNet::predict_query(&w_svm, &q_feats, &q_labels, cfg.n_way, cfg.feat_dim)
                .unwrap();

        assert!(
            loss_svm <= loss_zero + 1e-3,
            "SVM should reduce loss: {loss_svm} > {loss_zero}"
        );
    }

    // ── predict_query ──────────────────────────────────────────────────────────

    #[test]
    fn predict_query_all_correct_when_distinct_classes() {
        // Perfect one-hot prototypes as weights → argmax selects correct class
        let n_way = 3;
        let feat_dim = 3;
        let mut w = vec![0.0_f32; n_way * feat_dim];
        for c in 0..n_way {
            w[c * feat_dim + c] = 10.0; // strong diagonal
        }
        let b = vec![0.0_f32; n_way];
        let weights = MetaOptNetWeights { w, b };

        let q_feats: Vec<f32> = (0..n_way)
            .flat_map(|c| (0..feat_dim).map(move |j| if j == c { 1.0_f32 } else { 0.0_f32 }))
            .collect();
        let q_labels: Vec<usize> = (0..n_way).collect();

        let (_loss, acc) =
            MetaOptNet::predict_query(&weights, &q_feats, &q_labels, n_way, feat_dim).unwrap();
        assert!((acc - 1.0).abs() < 1e-5, "Perfect classifier: acc={acc}");
    }

    // ── Forward ────────────────────────────────────────────────────────────────

    #[test]
    fn forward_ridge_runs() {
        let cfg = default_ridge_cfg();
        let net = MetaOptNet::new(cfg.clone()).unwrap();
        let (s_feats, s_labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let q_feats: Vec<f32> = (0..cfg.n_way)
            .flat_map(|c| {
                (0..cfg.feat_dim).map(move |j| {
                    if j == c % cfg.feat_dim {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                })
            })
            .collect();
        let q_labels: Vec<usize> = (0..cfg.n_way).collect();
        let result = net
            .forward(&s_feats, &s_labels, &q_feats, &q_labels)
            .unwrap();
        assert!(result.query_loss.is_finite());
        assert!(result.query_accuracy >= 0.0 && result.query_accuracy <= 1.0);
    }

    #[test]
    fn forward_svm_runs() {
        let cfg = default_svm_cfg();
        let net = MetaOptNet::new(cfg.clone()).unwrap();
        let (s_feats, s_labels) = orthogonal_support(cfg.n_way, cfg.k_shot, cfg.feat_dim);
        let q_feats: Vec<f32> = (0..cfg.n_way)
            .flat_map(|c| {
                (0..cfg.feat_dim).map(move |j| {
                    if j == c % cfg.feat_dim {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                })
            })
            .collect();
        let q_labels: Vec<usize> = (0..cfg.n_way).collect();
        let result = net
            .forward(&s_feats, &s_labels, &q_feats, &q_labels)
            .unwrap();
        assert!(result.query_loss.is_finite());
        assert!(result.query_accuracy >= 0.0 && result.query_accuracy <= 1.0);
    }

    // ── Error paths ────────────────────────────────────────────────────────────

    #[test]
    fn err_n_way_one() {
        let mut cfg = default_ridge_cfg();
        cfg.n_way = 1;
        let result = MetaOptNet::new(cfg);
        assert!(matches!(result, Err(MetaError::InvalidNWay { .. })));
    }

    #[test]
    fn err_k_shot_zero() {
        let mut cfg = default_ridge_cfg();
        cfg.k_shot = 0;
        let result = MetaOptNet::new(cfg);
        assert!(matches!(result, Err(MetaError::InvalidKShot { .. })));
    }

    #[test]
    fn err_feat_dim_zero() {
        let mut cfg = default_ridge_cfg();
        cfg.feat_dim = 0;
        let result = MetaOptNet::new(cfg);
        assert!(matches!(result, Err(MetaError::InvalidFeatDim { .. })));
    }

    #[test]
    fn err_singular_gram() {
        // Collinear features: all support rows are identical → Gram is rank-1
        // even with a tiny lambda, Cholesky will fail
        let cfg = MetaOptNetConfig {
            n_way: 2,
            k_shot: 2,
            feat_dim: 3,
            reg_lambda: 1e-20, // near-zero regularisation → near-singular
            solver: MetaOptNetSolver::Ridge,
            max_svm_iter: 50,
            svm_tol: 1e-3,
        };
        // All rows identical (collinear)
        let feats = vec![
            1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let labels = vec![0_usize, 0, 1, 1];
        let result = MetaOptNet::ridge_solve(&feats, &labels, &cfg);
        // Should either succeed (with tiny lambda Gram might not be singular in f32)
        // or fail with Internal.  Either outcome is acceptable; the important thing
        // is no panic.
        match result {
            Ok(_) | Err(MetaError::Internal { .. }) => {}
            Err(e) => panic!("unexpected error variant: {e:?}"),
        }
    }

    #[test]
    fn err_support_length_mismatch() {
        let cfg = default_ridge_cfg();
        let labels = vec![0_usize, 1, 2];
        // feats length does not match n_support * feat_dim
        let feats = vec![0.0_f32; 5]; // wrong
        let result = MetaOptNet::ridge_solve(&feats, &labels, &cfg);
        assert!(matches!(result, Err(MetaError::DimensionMismatch { .. })));
    }
}
