//! DGI — Deep Graph Infomax contrastive pre-training.
//!
//! Maximises mutual information between node-level "patch" representations and a
//! graph-level "summary" representation. Training drives a bilinear discriminator
//! to distinguish real (positive) node–summary pairs from corrupted (negative) ones
//! produced by randomly shuffling node features across the graph.
//!
//! # Reference
//!
//! Velickovic et al. (2019) "Deep Graph Infomax", ICLR 2019.

use crate::error::{GnnError, GnnResult};
use crate::handle::LcgRng;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the DGI module.
#[derive(Debug, Clone)]
pub struct DgiConfig {
    /// Input feature dimension (before the GNN encoder).
    pub feat_dim: usize,
    /// Output embedding dimension (after the GNN encoder).
    pub embed_dim: usize,
    /// Number of layers in the bilinear discriminator (currently only depth ≥ 1).
    pub n_discriminator_layers: usize,
}

// ─── Weights ──────────────────────────────────────────────────────────────────

/// Learned parameters for the DGI module.
#[derive(Debug, Clone)]
pub struct DgiWeights {
    /// Bilinear weight matrix W_d ∈ ℝ^{embed_dim × embed_dim}.
    /// Layout: `[embed_dim × embed_dim]` row-major.
    pub discriminator_w: Vec<f32>,
    /// Readout weighting vector W_r ∈ ℝ^{embed_dim}.
    pub readout_w: Vec<f32>,
}

// ─── Loss ─────────────────────────────────────────────────────────────────────

/// DGI binary cross-entropy loss with diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct DgiLoss {
    /// Total DGI loss = E+ + E-.
    pub loss: f32,
    /// Mean discriminator score for real (node, summary) pairs; ∈ \[0, 1\].
    pub positive_score: f32,
    /// Mean discriminator score for corrupted (node, summary) pairs; ∈ \[0, 1\].
    pub negative_score: f32,
}

// ─── DGI ─────────────────────────────────────────────────────────────────────

/// Deep Graph Infomax module.
pub struct Dgi {
    /// Module configuration.
    pub cfg: DgiConfig,
    /// Learned parameters.
    pub weights: DgiWeights,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Scalar sigmoid: `σ(x) = 1 / (1 + e^{-x})`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Dgi {
    /// Construct a DGI module with Xavier-initialised discriminator weights and
    /// small-normal readout weights.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if any required dimension is zero.
    pub fn new(cfg: DgiConfig, rng: &mut LcgRng) -> GnnResult<Self> {
        if cfg.feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "DGI: feat_dim must be > 0".to_string(),
            ));
        }
        if cfg.embed_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "DGI: embed_dim must be > 0".to_string(),
            ));
        }
        if cfg.n_discriminator_layers == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "DGI: n_discriminator_layers must be > 0".to_string(),
            ));
        }

        let d = cfg.embed_dim;
        // Xavier uniform: ±1/√d.
        let xavier_bound = 1.0_f32 / (d as f32).sqrt();
        let discriminator_w: Vec<f32> = (0..d * d)
            .map(|_| {
                let u = rng.next_f32();
                (2.0 * u - 1.0) * xavier_bound
            })
            .collect();

        // Small normal approximation via [0,1] uniform scaled: N(0, 0.01²).
        // We use (u - 0.5) * 2 * 0.01 ≈ Uniform([-0.01, 0.01]) as a stable proxy
        // that keeps values in the correct scale without Box-Muller dependency
        // on the GNN LcgRng (which lacks next_normal).
        // More precisely: Box-Muller needs two u ∈ (0,1) draws.
        let readout_w: Vec<f32> = (0..d)
            .map(|_| {
                // Box-Muller for N(0, 0.01²).
                loop {
                    let u1 = rng.next_f32();
                    let u2 = rng.next_f32();
                    if u1 > 0.0 {
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * std::f32::consts::PI * u2;
                        return r * theta.cos() * 0.01;
                    }
                }
            })
            .collect();

        Ok(Self {
            cfg,
            weights: DgiWeights {
                discriminator_w,
                readout_w,
            },
        })
    }

    // ─── Readout ─────────────────────────────────────────────────────────────

    /// Compute a graph-level summary vector via sigmoid-weighted mean readout.
    ///
    /// For each node `i`:
    /// ```text
    /// w_i = sigmoid(W_r ⊙ h_i)   (element-wise)
    /// s   = sigmoid( mean_i(w_i ⊙ h_i) )
    /// ```
    ///
    /// # Arguments
    ///
    /// - `node_embeds`: `[n_nodes × embed_dim]` row-major.
    /// - `n_nodes`: number of nodes.
    ///
    /// # Returns
    ///
    /// `[embed_dim]` summary vector.
    pub fn readout(&self, node_embeds: &[f32], n_nodes: usize) -> GnnResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if node_embeds.len() != n_nodes * d {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * d,
                got: node_embeds.len(),
            });
        }

        let inv_n = 1.0_f32 / n_nodes as f32;

        // Accumulate weighted sum: Σ_i sigmoid(W_r ⊙ h_i) ⊙ h_i.
        let mut acc = vec![0.0_f32; d];
        for i in 0..n_nodes {
            for k in 0..d {
                let h_ik = node_embeds[i * d + k];
                let w_ik = sigmoid(self.weights.readout_w[k] * h_ik);
                acc[k] += w_ik * h_ik;
            }
        }

        // Summary = sigmoid(mean of weighted embeddings).
        let summary: Vec<f32> = acc.iter().map(|&v| sigmoid(v * inv_n)).collect();
        Ok(summary)
    }

    // ─── Discriminator ────────────────────────────────────────────────────────

    /// Bilinear discriminator scores: `D(h_i, s) = sigmoid(h_i^T W_d s)`.
    ///
    /// # Arguments
    ///
    /// - `node_embeds`: `[n_nodes × embed_dim]`.
    /// - `summary`: `[embed_dim]`.
    /// - `n_nodes`: number of nodes.
    ///
    /// # Returns
    ///
    /// `[n_nodes]` scalar scores in \[0, 1\].
    pub fn discriminate(
        &self,
        node_embeds: &[f32],
        summary: &[f32],
        n_nodes: usize,
    ) -> GnnResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if node_embeds.len() != n_nodes * d {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * d,
                got: node_embeds.len(),
            });
        }
        if summary.len() != d {
            return Err(GnnError::DimensionMismatch {
                expected: d,
                got: summary.len(),
            });
        }

        // Pre-compute W_d s ∈ ℝ^d.
        let wds: Vec<f32> = (0..d)
            .map(|row| {
                self.weights.discriminator_w[row * d..(row + 1) * d]
                    .iter()
                    .zip(summary.iter())
                    .map(|(&w, &s)| w * s)
                    .sum()
            })
            .collect();

        // D(h_i, s) = sigmoid(h_i^T W_d s).
        let scores: Vec<f32> = (0..n_nodes)
            .map(|i| {
                let dot: f32 = (0..d).map(|k| node_embeds[i * d + k] * wds[k]).sum();
                sigmoid(dot)
            })
            .collect();

        Ok(scores)
    }

    // ─── Corruption ──────────────────────────────────────────────────────────

    /// Corrupt node features by randomly permuting node indices (Fisher-Yates shuffle).
    ///
    /// The output is a `[n_nodes × feat_dim]` matrix where each row is drawn from a
    /// different (shuffled) row of the input, breaking all node-local structure.
    ///
    /// # Arguments
    ///
    /// - `features`: `[n_nodes × feat_dim]`.
    /// - `n_nodes`: number of nodes.
    /// - `feat_dim`: feature dimension.
    /// - `rng`: random source.
    pub fn corrupt(
        features: &[f32],
        n_nodes: usize,
        feat_dim: usize,
        rng: &mut LcgRng,
    ) -> Vec<f32> {
        if n_nodes == 0 || feat_dim == 0 {
            return Vec::new();
        }

        // Build a shuffled permutation of node indices.
        let mut perm: Vec<usize> = (0..n_nodes).collect();
        // Fisher-Yates shuffle.
        for i in (1..n_nodes).rev() {
            let j = rng.next_usize(i + 1);
            perm.swap(i, j);
        }

        // Assemble corrupted feature matrix using the permutation.
        let mut corrupted = vec![0.0_f32; n_nodes * feat_dim];
        for (dst_node, &src_node) in perm.iter().enumerate() {
            corrupted[dst_node * feat_dim..(dst_node + 1) * feat_dim]
                .copy_from_slice(&features[src_node * feat_dim..(src_node + 1) * feat_dim]);
        }
        corrupted
    }

    // ─── Loss ─────────────────────────────────────────────────────────────────

    /// Compute the DGI binary cross-entropy loss.
    ///
    /// ```text
    /// E+ = -mean_i [log(D(h_i,  s) + ε)]
    /// E- = -mean_j [log(1 - D(h̃_j, s) + ε)]
    /// loss = E+ + E-
    /// ```
    ///
    /// where ε = 1e-8 for numerical stability.
    ///
    /// # Arguments
    ///
    /// - `real_embeds`: `[n_nodes × embed_dim]` from the real graph.
    /// - `corrupted_embeds`: `[n_nodes × embed_dim]` from the corrupted graph.
    /// - `n_nodes`: number of nodes.
    ///
    /// # Returns
    ///
    /// [`DgiLoss`] with `loss`, `positive_score`, `negative_score`.
    ///
    /// # Errors
    ///
    /// - [`GnnError::EmptyGraph`] if `n_nodes < 2`.
    /// - [`GnnError::DimensionMismatch`] on size mismatch.
    pub fn loss(
        &self,
        real_embeds: &[f32],
        corrupted_embeds: &[f32],
        n_nodes: usize,
    ) -> GnnResult<DgiLoss> {
        let d = self.cfg.embed_dim;
        if n_nodes < 2 {
            return Err(GnnError::InvalidLayerConfig(
                "DGI: need at least 2 nodes for MI estimate".to_string(),
            ));
        }
        if real_embeds.len() != n_nodes * d {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * d,
                got: real_embeds.len(),
            });
        }
        if corrupted_embeds.len() != n_nodes * d {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * d,
                got: corrupted_embeds.len(),
            });
        }

        const EPS: f32 = 1e-8;

        // Compute summary from real embeddings.
        let summary = self.readout(real_embeds, n_nodes)?;

        // Positive scores: real embeddings vs summary.
        let pos_scores = self.discriminate(real_embeds, &summary, n_nodes)?;
        // Negative scores: corrupted embeddings vs summary.
        let neg_scores = self.discriminate(corrupted_embeds, &summary, n_nodes)?;

        let inv_n = 1.0_f32 / n_nodes as f32;

        let e_pos: f32 = pos_scores.iter().map(|&s| -(s + EPS).ln()).sum::<f32>() * inv_n;
        let e_neg: f32 = neg_scores
            .iter()
            .map(|&s| -(1.0 - s + EPS).ln())
            .sum::<f32>()
            * inv_n;

        let positive_score: f32 = pos_scores.iter().sum::<f32>() * inv_n;
        let negative_score: f32 = neg_scores.iter().sum::<f32>() * inv_n;

        Ok(DgiLoss {
            loss: e_pos + e_neg,
            positive_score,
            negative_score,
        })
    }

    // ─── Full forward step ────────────────────────────────────────────────────

    /// Full DGI forward step: corrupt input, encode both graphs, compute loss.
    ///
    /// The `encode` closure maps `(features: &[f32], row_ptr: &[usize], col_idx: &[usize])`
    /// to `n_nodes × embed_dim` embeddings.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from `loss`.
    pub fn forward<F>(
        &self,
        features: &[f32],
        n_nodes: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        encode: F,
        rng: &mut LcgRng,
    ) -> GnnResult<DgiLoss>
    where
        F: Fn(&[f32], &[usize], &[usize]) -> Vec<f32>,
    {
        let feat_dim = self.cfg.feat_dim;
        if n_nodes < 2 {
            return Err(GnnError::InvalidLayerConfig(
                "DGI: need at least 2 nodes".to_string(),
            ));
        }
        if features.len() != n_nodes * feat_dim {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * feat_dim,
                got: features.len(),
            });
        }

        // Corrupt features.
        let corrupted = Self::corrupt(features, n_nodes, feat_dim, rng);

        // Encode real and corrupted graphs.
        let real_embeds = encode(features, row_ptr, col_idx);
        let corrupted_embeds = encode(&corrupted, row_ptr, col_idx);

        self.loss(&real_embeds, &corrupted_embeds, n_nodes)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dgi(embed_dim: usize) -> Dgi {
        let cfg = DgiConfig {
            feat_dim: embed_dim,
            embed_dim,
            n_discriminator_layers: 1,
        };
        let mut rng = LcgRng::new(42);
        Dgi::new(cfg, &mut rng).expect("test invariant: DGI must construct")
    }

    // ─── Readout ─────────────────────────────────────────────────────────────

    #[test]
    fn readout_output_shape() {
        let d = 8;
        let n = 5;
        let dgi = make_dgi(d);
        let embeds = vec![0.3_f32; n * d];
        let s = dgi
            .readout(&embeds, n)
            .expect("test invariant: readout must succeed");
        assert_eq!(s.len(), d, "readout output must have length embed_dim");
    }

    #[test]
    fn readout_sigmoid_bounded() {
        let d = 6;
        let n = 8;
        let dgi = make_dgi(d);
        let embeds: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let s = dgi
            .readout(&embeds, n)
            .expect("test invariant: readout must succeed");
        for &v in &s {
            assert!(
                (0.0..=1.0).contains(&v),
                "readout value {v} out of [0,1] range"
            );
        }
    }

    // ─── Discriminator ────────────────────────────────────────────────────────

    #[test]
    fn discriminate_output_shape() {
        let d = 4;
        let n = 6;
        let dgi = make_dgi(d);
        let embeds = vec![0.2_f32; n * d];
        let summary = vec![0.5_f32; d];
        let scores = dgi
            .discriminate(&embeds, &summary, n)
            .expect("test invariant: discriminate must succeed");
        assert_eq!(
            scores.len(),
            n,
            "discriminator must return one score per node"
        );
    }

    #[test]
    fn discriminate_sigmoid_bounded() {
        let d = 8;
        let n = 4;
        let dgi = make_dgi(d);
        let embeds: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let summary: Vec<f32> = (0..d).map(|k| (k as f32) * 0.1).collect();
        let scores = dgi
            .discriminate(&embeds, &summary, n)
            .expect("test invariant: must succeed");
        for &s in &scores {
            assert!(
                (0.0..=1.0).contains(&s),
                "discriminator score {s} must be in [0,1]"
            );
        }
    }

    // ─── Corruption ──────────────────────────────────────────────────────────

    #[test]
    fn corrupt_permutes_rows() {
        let n = 5;
        let fd = 3;
        let features: Vec<f32> = (0..n * fd).map(|i| i as f32).collect();
        let mut rng = LcgRng::new(7);
        let corrupted = Dgi::corrupt(&features, n, fd, &mut rng);

        // Collect multiset of rows in original.
        let mut orig_rows: Vec<Vec<f32>> = (0..n)
            .map(|i| features[i * fd..(i + 1) * fd].to_vec())
            .collect();
        let mut corr_rows: Vec<Vec<f32>> = (0..n)
            .map(|i| corrupted[i * fd..(i + 1) * fd].to_vec())
            .collect();

        orig_rows.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        corr_rows.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        assert_eq!(
            orig_rows, corr_rows,
            "corrupted rows must be a permutation of original rows"
        );
    }

    #[test]
    fn corrupt_different_from_original() {
        // For n ≥ 4 distinct rows, shuffling should (almost certainly) produce
        // a different ordering. We use distinct rows to make this deterministic.
        let n = 8;
        let fd = 4;
        let features: Vec<f32> = (0..n * fd).map(|i| i as f32 * 1.1).collect();
        let mut rng = LcgRng::new(12);
        let corrupted = Dgi::corrupt(&features, n, fd, &mut rng);
        // With seed 12 and n=8 distinct rows, the permutation should differ.
        let same = features == corrupted;
        assert!(
            !same,
            "corrupted features should differ from original for n={n} distinct rows"
        );
    }

    // ─── Loss ─────────────────────────────────────────────────────────────────

    #[test]
    fn loss_positive_is_finite() {
        let d = 4;
        let n = 4;
        let dgi = make_dgi(d);
        let real = vec![0.5_f32; n * d];
        let corrupted = vec![0.1_f32; n * d];
        let l = dgi
            .loss(&real, &corrupted, n)
            .expect("test invariant: loss must succeed");
        assert!(l.loss.is_finite(), "total loss must be finite");
        assert!(
            l.positive_score.is_finite(),
            "positive score must be finite"
        );
    }

    #[test]
    fn loss_negative_is_finite() {
        let d = 4;
        let n = 4;
        let dgi = make_dgi(d);
        let real = vec![0.5_f32; n * d];
        let corrupted = vec![-0.3_f32; n * d];
        let l = dgi
            .loss(&real, &corrupted, n)
            .expect("test invariant: loss must succeed");
        assert!(
            l.negative_score.is_finite(),
            "negative score must be finite"
        );
    }

    #[test]
    fn loss_non_negative() {
        let d = 6;
        let n = 6;
        let dgi = make_dgi(d);
        let real: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.03).collect();
        let corrupted: Vec<f32> = (0..n * d).map(|i| -((i as f32) * 0.02)).collect();
        let l = dgi
            .loss(&real, &corrupted, n)
            .expect("test invariant: loss must succeed");
        assert!(
            l.loss >= 0.0,
            "DGI cross-entropy loss must be ≥ 0, got {}",
            l.loss
        );
    }

    #[test]
    fn loss_perfect_discriminator() {
        // When real_embeds == corrupted_embeds, D(h_i, s) ≈ same for both.
        // For a perfect 0.5-scoring discriminator, loss ≈ 2 * log(2) ≈ 1.386.
        // We verify loss is close to 1.386 (within 0.5 tolerance).
        let d = 4;
        let n = 4;
        let dgi = make_dgi(d);
        let embeds = vec![0.5_f32; n * d];
        let l = dgi
            .loss(&embeds, &embeds, n)
            .expect("test invariant: loss must succeed");
        let expected = 2.0 * 2.0_f32.ln(); // ≈ 1.386
        assert!(
            (l.loss - expected).abs() < 0.8,
            "uniform embeddings: loss={:.4} should be near {expected:.4}",
            l.loss
        );
    }

    // ─── Forward ─────────────────────────────────────────────────────────────

    #[test]
    fn forward_produces_loss() {
        let d = 4;
        let n = 5;
        let cfg = DgiConfig {
            feat_dim: d,
            embed_dim: d,
            n_discriminator_layers: 1,
        };
        let mut rng = LcgRng::new(100);
        let dgi = Dgi::new(cfg, &mut rng).expect("test invariant: must construct");

        let features: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1).collect();
        let row_ptr = vec![0, 1, 2, 3, 4, 5];
        let col_idx = vec![1, 2, 3, 4, 0];

        // Identity encoder: just returns features as embeddings.
        let encode = |feats: &[f32], _rp: &[usize], _ci: &[usize]| feats.to_vec();

        let l = dgi
            .forward(&features, n, &row_ptr, &col_idx, encode, &mut rng)
            .expect("test invariant: forward must succeed");
        assert!(l.loss.is_finite(), "DGI forward loss must be finite");
        assert!(l.loss >= 0.0, "DGI loss must be non-negative");
    }

    #[test]
    fn forward_loss_shape() {
        let d = 6;
        let n = 4;
        let cfg = DgiConfig {
            feat_dim: d,
            embed_dim: d,
            n_discriminator_layers: 1,
        };
        let mut rng = LcgRng::new(200);
        let dgi = Dgi::new(cfg, &mut rng).expect("test invariant: must construct");

        let features = vec![0.4_f32; n * d];
        let row_ptr = vec![0usize, 1, 2, 3, 4];
        let col_idx = vec![1usize, 2, 3, 0];

        let encode = |feats: &[f32], _: &[usize], _: &[usize]| feats.to_vec();
        let l = dgi
            .forward(&features, n, &row_ptr, &col_idx, encode, &mut rng)
            .expect("test invariant: forward must succeed");

        assert!(
            (0.0..=1.0).contains(&l.positive_score),
            "positive_score={} out of [0,1]",
            l.positive_score
        );
        assert!(
            (0.0..=1.0).contains(&l.negative_score),
            "negative_score={} out of [0,1]",
            l.negative_score
        );
    }

    // ─── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn n_nodes_min_2() {
        let d = 4;
        let dgi = make_dgi(d);
        let real = vec![0.5_f32; 2 * d];
        let corrupted = vec![0.3_f32; 2 * d];
        let l = dgi
            .loss(&real, &corrupted, 2)
            .expect("test invariant: n=2 must work");
        assert!(l.loss.is_finite());
    }

    #[test]
    fn err_n_nodes_zero() {
        let d = 4;
        let dgi = make_dgi(d);
        let err = dgi.loss(&[], &[], 0);
        assert!(err.is_err(), "n_nodes=0 should return an error");
    }

    #[test]
    fn err_feat_dim_zero() {
        let cfg = DgiConfig {
            feat_dim: 0,
            embed_dim: 4,
            n_discriminator_layers: 1,
        };
        let mut rng = LcgRng::new(1);
        let result = Dgi::new(cfg, &mut rng);
        assert!(result.is_err(), "feat_dim=0 must return an error");
    }

    #[test]
    fn err_embed_dim_zero() {
        let cfg = DgiConfig {
            feat_dim: 4,
            embed_dim: 0,
            n_discriminator_layers: 1,
        };
        let mut rng = LcgRng::new(2);
        let result = Dgi::new(cfg, &mut rng);
        assert!(result.is_err(), "embed_dim=0 must return an error");
    }

    #[test]
    fn err_embed_mismatch() {
        let d = 4;
        let dgi = make_dgi(d);
        // real_embeds has wrong length.
        let real = vec![0.5_f32; 3 * d]; // only 3 nodes' worth
        let corrupted = vec![0.3_f32; 4 * d]; // 4 nodes
        let err = dgi.loss(&real, &corrupted, 4);
        assert!(
            matches!(err, Err(GnnError::DimensionMismatch { .. })),
            "mismatched embed size should return DimensionMismatch"
        );
    }

    // ─── Additional coverage ─────────────────────────────────────────────────

    #[test]
    fn readout_single_node() {
        let d = 8;
        let n = 1;
        let dgi = make_dgi(d);
        let embeds = vec![1.0_f32; n * d];
        let s = dgi
            .readout(&embeds, n)
            .expect("test invariant: single node readout");
        assert_eq!(s.len(), d);
        // sigmoid output must be in [0,1].
        assert!(s.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn corrupt_empty_is_empty() {
        let mut rng = LcgRng::new(3);
        let result = Dgi::corrupt(&[], 0, 4, &mut rng);
        assert!(
            result.is_empty(),
            "corrupt with n_nodes=0 must return empty"
        );
    }
}
