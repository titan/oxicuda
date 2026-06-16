//! DeepEMD — Few-Shot Image Classification with Differentiable Earth Mover's
//! Distance (Zhang, Cai, Lin & Shen, CVPR 2020).
//!
//! Rather than collapsing an image into a single global embedding, DeepEMD keeps
//! a *set* of local feature vectors (e.g. spatial cells of a feature map) and
//! measures the distance between two images as the **Earth Mover's Distance**
//! between their local-feature distributions.  The ground cost between two local
//! vectors is `1 − cosine_similarity`, and the optimal transport plan is found
//! with entropic (Sinkhorn) regularisation, which makes the whole pipeline
//! differentiable.
//!
//! ```text
//! C_ij = 1 − cos(a_i, b_j)                        (ground cost, n_local × n_local)
//! K    = exp(−C / ε)                              (Gibbs kernel)
//! u    = a ⊘ (K v),   v = b ⊘ (Kᵀ u)             (Sinkhorn iterations)
//! T    = diag(u) K diag(v)                        (transport plan)
//! EMD  = ⟨T, C⟩                                   (Frobenius inner product)
//! ```
//!
//! Reference: <https://arxiv.org/abs/2003.06777>

use crate::error::{MetaError, MetaResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for DeepEMD.
#[derive(Debug, Clone)]
pub struct DeepEmdConfig {
    /// Dimension of each local feature vector.
    pub feat_dim: usize,
    /// Number of local features per image (`n_local`).
    pub n_local: usize,
    /// Number of Sinkhorn iterations (`≥ 1`).
    pub sinkhorn_iters: usize,
    /// Entropic regularisation strength `ε > 0`.
    pub epsilon: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main struct
// ─────────────────────────────────────────────────────────────────────────────

/// DeepEMD few-shot distance / classifier over local features.
pub struct DeepEmd {
    /// Configuration.
    cfg: DeepEmdConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a slice.
fn softmax_stable(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 1e-38 { 1.0 / sum } else { 1.0 };
    for e in exps.iter_mut() {
        *e *= inv;
    }
    exps
}

// ─────────────────────────────────────────────────────────────────────────────
// impl DeepEmd
// ─────────────────────────────────────────────────────────────────────────────

impl DeepEmd {
    /// Construct and validate a DeepEMD instance.
    pub fn new(cfg: DeepEmdConfig) -> MetaResult<Self> {
        if cfg.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.feat_dim });
        }
        if cfg.n_local < 1 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!("n_local must be >= 1, got {}", cfg.n_local),
            });
        }
        if cfg.sinkhorn_iters < 1 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!("sinkhorn_iters must be >= 1, got {}", cfg.sinkhorn_iters),
            });
        }
        if cfg.epsilon <= 0.0 || cfg.epsilon.is_nan() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!("epsilon must be > 0, got {}", cfg.epsilon),
            });
        }
        Ok(Self { cfg })
    }

    /// Ground cost matrix `C_ij = 1 − cosine(a_i, b_j)`, shape
    /// `n_local × n_local`, row-major.
    ///
    /// `feats_a`, `feats_b`: `n_local × feat_dim`, row-major.  Cosine uses an
    /// `ε`-guarded denominator so zero vectors do not divide by zero.
    pub fn cost_matrix(&self, feats_a: &[f32], feats_b: &[f32]) -> MetaResult<Vec<f32>> {
        let n = self.cfg.n_local;
        let d = self.cfg.feat_dim;
        let expected = n * d;
        if feats_a.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: feats_a.len(),
            });
        }
        if feats_b.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: feats_b.len(),
            });
        }

        // Pre-compute L2 norms.
        let norm_a: Vec<f32> = (0..n)
            .map(|i| {
                feats_a[i * d..(i + 1) * d]
                    .iter()
                    .map(|&v| v * v)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();
        let norm_b: Vec<f32> = (0..n)
            .map(|j| {
                feats_b[j * d..(j + 1) * d]
                    .iter()
                    .map(|&v| v * v)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let mut cost = vec![0.0_f32; n * n];
        for i in 0..n {
            let a_i = &feats_a[i * d..(i + 1) * d];
            for j in 0..n {
                let b_j = &feats_b[j * d..(j + 1) * d];
                let mut dot = 0.0_f32;
                for t in 0..d {
                    dot += a_i[t] * b_j[t];
                }
                let denom = norm_a[i] * norm_b[j] + 1e-8;
                let cos = (dot / denom).clamp(-1.0, 1.0);
                cost[i * n + j] = 1.0 - cos;
            }
        }
        Ok(cost)
    }

    /// Validate, and if necessary normalise, a marginal weight vector to sum 1.
    fn prepare_weights(&self, weights: &[f32]) -> MetaResult<Vec<f32>> {
        let n = self.cfg.n_local;
        if weights.len() != n {
            return Err(MetaError::DimensionMismatch {
                expected: n,
                got: weights.len(),
            });
        }
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "sinkhorn marginal sum is non-positive or non-finite".into(),
            });
        }
        // Normalise so marginals form a probability vector (accepts uniform or
        // any non-negative input summing to a positive value).
        let inv = 1.0 / sum;
        Ok(weights.iter().map(|&w| w * inv).collect())
    }

    /// Sinkhorn transport plan `T` for cost `C` with marginals
    /// `a_weights` / `b_weights`.
    ///
    /// `K = exp(−C/ε)`, alternate `u = a ⊘ (K v)`, `v = b ⊘ (Kᵀ u)` for
    /// `sinkhorn_iters`, then `T = diag(u) K diag(v)`.  Marginals are normalised
    /// to sum 1 if they do not already.
    ///
    /// Returns `T`: `n_local × n_local`, row-major.
    pub fn sinkhorn(
        &self,
        cost: &[f32],
        a_weights: &[f32],
        b_weights: &[f32],
    ) -> MetaResult<Vec<f32>> {
        let n = self.cfg.n_local;
        let expected = n * n;
        if cost.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: cost.len(),
            });
        }
        let a = self.prepare_weights(a_weights)?;
        let b = self.prepare_weights(b_weights)?;

        let inv_eps = 1.0 / self.cfg.epsilon;
        // Gibbs kernel K = exp(-C/eps).
        let kernel: Vec<f32> = cost.iter().map(|&c| (-c * inv_eps).exp()).collect();

        let mut u = vec![1.0_f32; n];
        let mut v = vec![1.0_f32; n];

        for _ in 0..self.cfg.sinkhorn_iters {
            // u = a ⊘ (K v)
            for i in 0..n {
                let mut kv = 0.0_f32;
                for j in 0..n {
                    kv += kernel[i * n + j] * v[j];
                }
                u[i] = a[i] / (kv + 1e-38);
            }
            // v = b ⊘ (Kᵀ u)
            for j in 0..n {
                let mut ktu = 0.0_f32;
                for i in 0..n {
                    ktu += kernel[i * n + j] * u[i];
                }
                v[j] = b[j] / (ktu + 1e-38);
            }
        }

        // T = diag(u) K diag(v)
        let mut plan = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                plan[i * n + j] = u[i] * kernel[i * n + j] * v[j];
            }
        }
        Ok(plan)
    }

    /// Earth Mover's Distance between two local-feature sets:
    /// `EMD = ⟨T, C⟩` for the Sinkhorn plan `T` and cost `C`, using uniform
    /// marginals `1/n_local`.
    pub fn emd_distance(&self, feats_a: &[f32], feats_b: &[f32]) -> MetaResult<f32> {
        let n = self.cfg.n_local;
        let cost = self.cost_matrix(feats_a, feats_b)?;
        let uniform = vec![1.0_f32 / n as f32; n];
        let plan = self.sinkhorn(&cost, &uniform, &uniform)?;
        let emd: f32 = plan
            .iter()
            .zip(cost.iter())
            .map(|(&t, &c)| t * c)
            .sum::<f32>();
        if !emd.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "emd_distance produced a non-finite value".into(),
            });
        }
        Ok(emd)
    }

    /// Classify a query's local features against `n_way` per-class prototype
    /// local-feature sets.  Lower EMD ⇒ closer ⇒ higher logit; logits `= −EMD`,
    /// then softmax over classes.
    ///
    /// `query_feats`: `n_local × feat_dim`, row-major.
    /// `class_feats`: `n_way × n_local × feat_dim`, row-major.
    /// Returns a probability distribution of length `n_way`.
    pub fn classify(
        &self,
        query_feats: &[f32],
        class_feats: &[f32],
        n_way: usize,
    ) -> MetaResult<Vec<f32>> {
        if n_way < 1 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        let n = self.cfg.n_local;
        let d = self.cfg.feat_dim;
        let per_class = n * d;
        if query_feats.len() != per_class {
            return Err(MetaError::DimensionMismatch {
                expected: per_class,
                got: query_feats.len(),
            });
        }
        let expected = n_way * per_class;
        if class_feats.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: class_feats.len(),
            });
        }

        let mut logits = vec![0.0_f32; n_way];
        for c in 0..n_way {
            let cf = &class_feats[c * per_class..(c + 1) * per_class];
            let emd = self.emd_distance(query_feats, cf)?;
            logits[c] = -emd;
        }
        Ok(softmax_stable(&logits))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn default_config() -> DeepEmdConfig {
        // Well-converged Sinkhorn (small epsilon, many iterations) so the
        // entropic transport plan closely approximates the true optimal plan;
        // this makes the EMD invariants (self ≈ 0, symmetry) hold tightly.
        DeepEmdConfig {
            feat_dim: 4,
            n_local: 3,
            sinkhorn_iters: 1000,
            epsilon: 0.01,
        }
    }

    fn random_feats(n_local: usize, feat_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n_local * feat_dim)
            .map(|_| {
                // Map to roughly [-1, 1) to get varied cosine values.
                let u = rng.next_u32() as f32 / (u32::MAX as f32 + 1.0);
                u * 2.0 - 1.0
            })
            .collect()
    }

    // ── Construction validation ─────────────────────────────────────────────

    #[test]
    fn new_valid_config_succeeds() {
        assert!(DeepEmd::new(default_config()).is_ok());
    }

    #[test]
    fn new_feat_dim_zero_errs() {
        let mut cfg = default_config();
        cfg.feat_dim = 0;
        assert!(matches!(
            DeepEmd::new(cfg),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn new_n_local_zero_errs() {
        let mut cfg = default_config();
        cfg.n_local = 0;
        assert!(matches!(
            DeepEmd::new(cfg),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_sinkhorn_iters_zero_errs() {
        let mut cfg = default_config();
        cfg.sinkhorn_iters = 0;
        assert!(matches!(
            DeepEmd::new(cfg),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_epsilon_non_positive_errs() {
        let mut cfg = default_config();
        cfg.epsilon = 0.0;
        assert!(matches!(
            DeepEmd::new(cfg),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    // ── cost_matrix ─────────────────────────────────────────────────────────

    #[test]
    fn cost_matrix_shape() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 1);
        let b = random_feats(3, 4, 2);
        let cost = emd.cost_matrix(&a, &b).expect("cost_matrix should succeed");
        assert_eq!(cost.len(), 3 * 3);
    }

    #[test]
    fn cost_matrix_in_zero_two_range() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 3);
        let b = random_feats(3, 4, 4);
        let cost = emd.cost_matrix(&a, &b).expect("cost_matrix should succeed");
        for &c in &cost {
            assert!((0.0..=2.0).contains(&c), "cost {c} outside [0,2]");
        }
    }

    #[test]
    fn cost_self_diagonal_is_zero() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 5);
        let cost = emd.cost_matrix(&a, &a).expect("cost_matrix should succeed");
        for i in 0..3 {
            assert!(
                cost[i * 3 + i].abs() < 1e-5,
                "self cosine should give zero diagonal cost, got {}",
                cost[i * 3 + i]
            );
        }
    }

    #[test]
    fn cost_matrix_wrong_length_errs() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = vec![0.0_f32; 5];
        let b = random_feats(3, 4, 6);
        assert!(matches!(
            emd.cost_matrix(&a, &b),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── sinkhorn ────────────────────────────────────────────────────────────

    #[test]
    fn sinkhorn_marginals_match() {
        let cfg = DeepEmdConfig {
            feat_dim: 4,
            n_local: 4,
            sinkhorn_iters: 200,
            epsilon: 0.1,
        };
        let emd = DeepEmd::new(cfg.clone()).expect("value should be present");
        let a = random_feats(cfg.n_local, cfg.feat_dim, 11);
        let b = random_feats(cfg.n_local, cfg.feat_dim, 12);
        let cost = emd.cost_matrix(&a, &b).expect("cost_matrix should succeed");
        let uniform = vec![1.0_f32 / cfg.n_local as f32; cfg.n_local];
        let plan = emd
            .sinkhorn(&cost, &uniform, &uniform)
            .expect("sinkhorn should succeed");
        let n = cfg.n_local;
        for i in 0..n {
            let row_sum: f32 = (0..n).map(|j| plan[i * n + j]).sum();
            assert!(
                (row_sum - uniform[i]).abs() < 1e-3,
                "row {i} sum {row_sum} != marginal {}",
                uniform[i]
            );
        }
        for j in 0..n {
            let col_sum: f32 = (0..n).map(|i| plan[i * n + j]).sum();
            assert!(
                (col_sum - uniform[j]).abs() < 1e-3,
                "col {j} sum {col_sum} != marginal {}",
                uniform[j]
            );
        }
    }

    #[test]
    fn sinkhorn_plan_non_negative() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 13);
        let b = random_feats(3, 4, 14);
        let cost = emd.cost_matrix(&a, &b).expect("cost_matrix should succeed");
        let uniform = vec![1.0_f32 / 3.0; 3];
        let plan = emd
            .sinkhorn(&cost, &uniform, &uniform)
            .expect("sinkhorn should succeed");
        assert!(
            plan.iter().all(|&t| t >= 0.0),
            "transport plan must be >= 0"
        );
    }

    #[test]
    fn sinkhorn_cost_wrong_length_errs() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let cost = vec![0.0_f32; 5];
        let uniform = vec![1.0_f32 / 3.0; 3];
        assert!(matches!(
            emd.sinkhorn(&cost, &uniform, &uniform),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn sinkhorn_weights_wrong_length_errs() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 15);
        let cost = emd.cost_matrix(&a, &a).expect("cost_matrix should succeed");
        let bad = vec![0.5_f32; 2];
        let uniform = vec![1.0_f32 / 3.0; 3];
        assert!(matches!(
            emd.sinkhorn(&cost, &bad, &uniform),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn larger_epsilon_more_uniform_plan() {
        // A higher epsilon → more entropic regularisation → plan closer to the
        // independent coupling a·bᵀ (more uniform). Measure variance of the plan.
        let base = DeepEmdConfig {
            feat_dim: 4,
            n_local: 4,
            sinkhorn_iters: 200,
            epsilon: 0.05,
        };
        let big = DeepEmdConfig {
            epsilon: 5.0,
            ..base.clone()
        };
        let emd_small = DeepEmd::new(base.clone()).expect("value should be present");
        let emd_big = DeepEmd::new(big).expect("new should succeed");
        let a = random_feats(base.n_local, base.feat_dim, 21);
        let b = random_feats(base.n_local, base.feat_dim, 22);
        let cost = emd_small
            .cost_matrix(&a, &b)
            .expect("cost_matrix should succeed");
        let uniform = vec![1.0_f32 / base.n_local as f32; base.n_local];
        let plan_small = emd_small
            .sinkhorn(&cost, &uniform, &uniform)
            .expect("sinkhorn should succeed");
        let plan_big = emd_big
            .sinkhorn(&cost, &uniform, &uniform)
            .expect("sinkhorn should succeed");
        let var = |p: &[f32]| {
            let mean = p.iter().sum::<f32>() / p.len() as f32;
            p.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / p.len() as f32
        };
        assert!(
            var(&plan_big) < var(&plan_small),
            "larger epsilon should give a more uniform (lower-variance) plan: big={}, small={}",
            var(&plan_big),
            var(&plan_small)
        );
    }

    // ── emd_distance ────────────────────────────────────────────────────────

    #[test]
    fn emd_distance_non_negative() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 31);
        let b = random_feats(3, 4, 32);
        let d = emd
            .emd_distance(&a, &b)
            .expect("emd_distance should succeed");
        assert!(d >= 0.0, "EMD must be >= 0, got {d}");
    }

    #[test]
    fn emd_self_is_zero() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 33);
        let d = emd
            .emd_distance(&a, &a)
            .expect("emd_distance should succeed");
        assert!(d.abs() < 1e-4, "EMD(a,a) should be ~0, got {d}");
    }

    #[test]
    fn emd_symmetric() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 34);
        let b = random_feats(3, 4, 35);
        let dab = emd
            .emd_distance(&a, &b)
            .expect("emd_distance should succeed");
        let dba = emd
            .emd_distance(&b, &a)
            .expect("emd_distance should succeed");
        assert!(
            (dab - dba).abs() < 1e-4,
            "EMD should be symmetric: {dab} vs {dba}"
        );
    }

    #[test]
    fn single_local_feature_emd_equals_cost() {
        // n_local = 1: the only transport plan is T=[1], so EMD == C[0,0].
        let cfg = DeepEmdConfig {
            feat_dim: 3,
            n_local: 1,
            sinkhorn_iters: 5,
            epsilon: 0.1,
        };
        let emd = DeepEmd::new(cfg).expect("new should succeed");
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0]; // orthogonal -> cos 0 -> cost 1
        let cost = emd.cost_matrix(&a, &b).expect("cost_matrix should succeed");
        let d = emd
            .emd_distance(&a, &b)
            .expect("emd_distance should succeed");
        assert!(
            (d - cost[0]).abs() < 1e-5,
            "n_local=1 EMD {d} should equal cost {}",
            cost[0]
        );
    }

    #[test]
    fn emd_distance_wrong_length_errs() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = vec![0.0_f32; 7];
        let b = random_feats(3, 4, 36);
        assert!(matches!(
            emd.emd_distance(&a, &b),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── classify ────────────────────────────────────────────────────────────

    #[test]
    fn classify_softmax_sums_to_one() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let n_way = 3;
        let q = random_feats(3, 4, 41);
        let class_feats: Vec<f32> = (0..n_way)
            .flat_map(|c| random_feats(3, 4, 50 + c as u64))
            .collect();
        let probs = emd
            .classify(&q, &class_feats, n_way)
            .expect("classify should succeed");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
        assert_eq!(probs.len(), n_way);
    }

    #[test]
    fn classify_picks_most_similar_class() {
        // Class 1's local features are identical to the query → EMD 0 → top prob.
        let cfg = DeepEmdConfig {
            feat_dim: 3,
            n_local: 2,
            sinkhorn_iters: 50,
            epsilon: 0.1,
        };
        let emd = DeepEmd::new(cfg.clone()).expect("value should be present");
        let n_way = 3;
        let query = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let per_class = cfg.n_local * cfg.feat_dim;
        let mut class_feats = vec![0.0_f32; n_way * per_class];
        // class 0: dissimilar
        class_feats[0..per_class].copy_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        // class 1: identical to query
        class_feats[per_class..2 * per_class].copy_from_slice(&query);
        // class 2: dissimilar
        class_feats[2 * per_class..3 * per_class].copy_from_slice(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let probs = emd
            .classify(&query, &class_feats, n_way)
            .expect("classify should succeed");
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .expect("value should be present");
        assert_eq!(best, 1, "class with identical local features must win");
    }

    #[test]
    fn classify_class_feats_wrong_length_errs() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let q = random_feats(3, 4, 42);
        let class_feats = vec![0.0_f32; 5];
        assert!(matches!(
            emd.classify(&q, &class_feats, 3),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn classify_query_wrong_length_errs() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let q = vec![0.0_f32; 5];
        let class_feats: Vec<f32> = (0..3).flat_map(|c| random_feats(3, 4, 60 + c)).collect();
        assert!(matches!(
            emd.classify(&q, &class_feats, 3),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── Determinism ─────────────────────────────────────────────────────────

    #[test]
    fn deterministic_emd() {
        let emd = DeepEmd::new(default_config()).expect("value should be present");
        let a = random_feats(3, 4, 71);
        let b = random_feats(3, 4, 72);
        let d1 = emd
            .emd_distance(&a, &b)
            .expect("emd_distance should succeed");
        let d2 = emd
            .emd_distance(&a, &b)
            .expect("emd_distance should succeed");
        assert_eq!(d1, d2, "EMD must be deterministic");
    }
}
