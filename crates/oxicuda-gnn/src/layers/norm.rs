//! Graph normalisation layers: GraphNorm (Cai 2021) and PairNorm (Zhao 2020).
//!
//! Standard BatchNorm/LayerNorm are sub-optimal on graphs. Two graph-specific
//! normalisations address distinct problems:
//!
//! - **GraphNorm** (Cai et al. 2021, ICML) normalises each feature channel over
//!   the nodes of a single graph, but subtracts a *learnable* fraction `α_k` of
//!   the channel mean rather than the full mean. This preserves discriminative
//!   information that mean-subtraction would otherwise destroy, and accelerates
//!   training convergence:
//!
//!   ```text
//!   μ_k = mean_i x_{i,k}
//!   σ_k = sqrt( mean_i (x_{i,k} − α_k μ_k)^2 + ε )
//!   GraphNorm(x_{i,k}) = γ_k · (x_{i,k} − α_k μ_k) / σ_k + β_k
//!   ```
//!
//! - **PairNorm** (Zhao & Akoglu 2020, ICLR) combats over-smoothing in deep GNNs
//!   by keeping the total pairwise feature distance constant across layers. It
//!   centres node features then rescales them so the mean squared row-norm equals
//!   a target scale `s²`:
//!
//!   ```text
//!   x̃_i = x_i − mean_j x_j               (centre)
//!   PairNorm(x_i) = s · x̃_i / sqrt( mean_j ||x̃_j||² )   (scale)
//!   ```

use crate::error::{GnnError, GnnResult};

/// Configuration / learnable parameters for a [`GraphNorm`] layer.
#[derive(Debug, Clone)]
pub struct GraphNorm {
    feat_dim: usize,
    /// Learnable per-channel mean-shift coefficients `α_k` (length `feat_dim`).
    alpha: Vec<f32>,
    /// Learnable per-channel scale `γ_k` (length `feat_dim`).
    gamma: Vec<f32>,
    /// Learnable per-channel shift `β_k` (length `feat_dim`).
    beta: Vec<f32>,
    /// Numerical-stability epsilon added inside the square root.
    eps: f32,
}

impl GraphNorm {
    /// Construct a GraphNorm with the standard init `α = γ = 1`, `β = 0`.
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] if `feat_dim == 0`.
    pub fn new(feat_dim: usize) -> GnnResult<Self> {
        if feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphNorm: feat_dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            feat_dim,
            alpha: vec![1.0; feat_dim],
            gamma: vec![1.0; feat_dim],
            beta: vec![0.0; feat_dim],
            eps: 1e-5,
        })
    }

    /// Construct with explicit learnable parameters.
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] for `feat_dim == 0` or `eps <= 0`, and
    /// [`GnnError::DimensionMismatch`] if any parameter vector length differs
    /// from `feat_dim`.
    pub fn with_params(
        feat_dim: usize,
        alpha: Vec<f32>,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        eps: f32,
    ) -> GnnResult<Self> {
        if feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphNorm: feat_dim must be > 0".to_string(),
            ));
        }
        if eps <= 0.0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphNorm: eps must be > 0".to_string(),
            ));
        }
        for v in [&alpha, &gamma, &beta] {
            if v.len() != feat_dim {
                return Err(GnnError::DimensionMismatch {
                    expected: feat_dim,
                    got: v.len(),
                });
            }
        }
        Ok(Self {
            feat_dim,
            alpha,
            gamma,
            beta,
            eps,
        })
    }

    /// Apply GraphNorm to the `[n_nodes × feat_dim]` features of a single graph.
    ///
    /// # Errors
    ///
    /// [`GnnError::EmptyGraph`] if `n_nodes == 0`, and
    /// [`GnnError::NodeFeatureMismatch`] if `x.len() != n_nodes * feat_dim`.
    pub fn forward(&self, x: &[f32], n_nodes: usize) -> GnnResult<Vec<f32>> {
        let d = self.feat_dim;
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if x.len() != n_nodes * d {
            return Err(GnnError::NodeFeatureMismatch(n_nodes, x.len() / d.max(1)));
        }
        let inv_n = 1.0 / n_nodes as f32;

        // Per-channel mean μ_k.
        let mut mean = vec![0.0_f32; d];
        for i in 0..n_nodes {
            for k in 0..d {
                mean[k] += x[i * d + k];
            }
        }
        for m in &mut mean {
            *m *= inv_n;
        }

        // Centred-by-α residual r_{i,k} = x_{i,k} − α_k μ_k, and its variance.
        let mut var = vec![0.0_f32; d];
        let mut out = vec![0.0_f32; n_nodes * d];
        for i in 0..n_nodes {
            for k in 0..d {
                let r = x[i * d + k] - self.alpha[k] * mean[k];
                out[i * d + k] = r; // stash residual; rescale below
                var[k] += r * r;
            }
        }
        for v in &mut var {
            *v *= inv_n;
        }

        for i in 0..n_nodes {
            for k in 0..d {
                let denom = (var[k] + self.eps).sqrt();
                out[i * d + k] = self.gamma[k] * out[i * d + k] / denom + self.beta[k];
            }
        }
        Ok(out)
    }

    /// Feature dimension.
    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }
}

/// Scaling mode for [`PairNorm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairNormMode {
    /// Centre then rescale to target mean-squared row norm (`PN`).
    Standard,
    /// Centre then per-node L2-normalise to scale `s` (`PN-SI`, scale-individual).
    ScaleIndividual,
}

/// PairNorm normalisation layer (no learnable parameters; only a target scale).
#[derive(Debug, Clone)]
pub struct PairNorm {
    feat_dim: usize,
    /// Target scale `s` (controls the overall feature magnitude after norm).
    scale: f32,
    mode: PairNormMode,
    eps: f32,
}

impl PairNorm {
    /// Construct a PairNorm with target scale `s` and the given mode.
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] if `feat_dim == 0` or `scale <= 0`.
    pub fn new(feat_dim: usize, scale: f32, mode: PairNormMode) -> GnnResult<Self> {
        if feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "PairNorm: feat_dim must be > 0".to_string(),
            ));
        }
        if scale <= 0.0 {
            return Err(GnnError::InvalidLayerConfig(
                "PairNorm: scale must be > 0".to_string(),
            ));
        }
        Ok(Self {
            feat_dim,
            scale,
            mode,
            eps: 1e-6,
        })
    }

    /// Apply PairNorm to `[n_nodes × feat_dim]` features of a single graph.
    ///
    /// # Errors
    ///
    /// [`GnnError::EmptyGraph`] if `n_nodes == 0`, and
    /// [`GnnError::NodeFeatureMismatch`] if `x.len() != n_nodes * feat_dim`.
    pub fn forward(&self, x: &[f32], n_nodes: usize) -> GnnResult<Vec<f32>> {
        let d = self.feat_dim;
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if x.len() != n_nodes * d {
            return Err(GnnError::NodeFeatureMismatch(n_nodes, x.len() / d.max(1)));
        }
        let inv_n = 1.0 / n_nodes as f32;

        // Step 1: centre — subtract the per-channel mean.
        let mut mean = vec![0.0_f32; d];
        for i in 0..n_nodes {
            for k in 0..d {
                mean[k] += x[i * d + k];
            }
        }
        for m in &mut mean {
            *m *= inv_n;
        }
        let mut centred = vec![0.0_f32; n_nodes * d];
        for i in 0..n_nodes {
            for k in 0..d {
                centred[i * d + k] = x[i * d + k] - mean[k];
            }
        }

        // Step 2: rescale.
        let out = match self.mode {
            PairNormMode::Standard => {
                // mean squared row norm
                let mut msr = 0.0_f32;
                for i in 0..n_nodes {
                    let mut row_sq = 0.0_f32;
                    for k in 0..d {
                        let v = centred[i * d + k];
                        row_sq += v * v;
                    }
                    msr += row_sq;
                }
                msr *= inv_n;
                let denom = (msr + self.eps).sqrt();
                let factor = self.scale / denom;
                centred.iter().map(|&v| v * factor).collect()
            }
            PairNormMode::ScaleIndividual => {
                let mut out = vec![0.0_f32; n_nodes * d];
                for i in 0..n_nodes {
                    let mut row_sq = 0.0_f32;
                    for k in 0..d {
                        let v = centred[i * d + k];
                        row_sq += v * v;
                    }
                    let denom = (row_sq + self.eps).sqrt();
                    let factor = self.scale / denom;
                    for k in 0..d {
                        out[i * d + k] = centred[i * d + k] * factor;
                    }
                }
                out
            }
        };
        Ok(out)
    }

    /// Feature dimension.
    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }

    /// Target scale `s`.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphnorm_build_and_dim() {
        let gn = GraphNorm::new(4).expect("build");
        assert_eq!(gn.feat_dim(), 4);
    }

    #[test]
    fn graphnorm_zero_dim_errors() {
        assert!(GraphNorm::new(0).is_err());
    }

    #[test]
    fn graphnorm_output_shape() {
        let gn = GraphNorm::new(3).expect("build");
        let x = vec![1.0_f32; 5 * 3];
        let out = gn.forward(&x, 5).expect("forward");
        assert_eq!(out.len(), 5 * 3);
    }

    #[test]
    fn graphnorm_zero_mean_per_channel_default() {
        // With α=1 the residual r = x − μ has zero mean, so output (γ=1,β=0) has
        // zero per-channel mean too.
        let gn = GraphNorm::new(2).expect("build");
        // node feats: [1,10], [3,20], [5,30] → per-channel means 3, 20
        let x = vec![1.0_f32, 10.0, 3.0, 20.0, 5.0, 30.0];
        let out = gn.forward(&x, 3).expect("forward");
        for k in 0..2 {
            let mean: f32 = (0..3).map(|i| out[i * 2 + k]).sum::<f32>() / 3.0;
            assert!(
                mean.abs() < 1e-4,
                "channel {k} mean should be ~0, got {mean}"
            );
        }
    }

    #[test]
    fn graphnorm_unit_variance_default() {
        let gn = GraphNorm::new(1).expect("build");
        let x = vec![2.0_f32, 4.0, 6.0, 8.0];
        let out = gn.forward(&x, 4).expect("forward");
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        // variance should be close to 1 (eps makes it slightly under)
        assert!((var - 1.0).abs() < 1e-2, "variance ~1 expected, got {var}");
    }

    #[test]
    fn graphnorm_alpha_zero_keeps_mean() {
        // α=0 means no mean subtraction; constant features stay constant after
        // normalisation by their own magnitude.
        let gn = GraphNorm::with_params(2, vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 0.0], 1e-5)
            .expect("build");
        let x = vec![3.0_f32, 3.0, 3.0, 3.0]; // 2 nodes, both [3,3]
        let out = gn.forward(&x, 2).expect("forward");
        // residual = x (α=0), var per channel = 9, denom = 3 → out = 1
        for v in &out {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1, got {v}");
        }
    }

    #[test]
    fn graphnorm_beta_shifts_output() {
        let gn = GraphNorm::with_params(1, vec![1.0], vec![1.0], vec![5.0], 1e-5).expect("build");
        let x = vec![1.0_f32, 2.0, 3.0];
        let out = gn.forward(&x, 3).expect("forward");
        let mean: f32 = out.iter().sum::<f32>() / 3.0;
        assert!((mean - 5.0).abs() < 1e-3, "beta shift failed, mean {mean}");
    }

    #[test]
    fn graphnorm_param_length_mismatch_errors() {
        let err = GraphNorm::with_params(3, vec![1.0, 1.0], vec![1.0; 3], vec![0.0; 3], 1e-5);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn graphnorm_feature_mismatch_errors() {
        let gn = GraphNorm::new(3).expect("build");
        let err = gn.forward(&[1.0_f32; 7], 3); // 7 not multiple of 3 nodes*dim
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn graphnorm_empty_graph_errors() {
        let gn = GraphNorm::new(2).expect("build");
        assert!(matches!(gn.forward(&[], 0), Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn graphnorm_output_finite() {
        let gn = GraphNorm::new(4).expect("build");
        let x: Vec<f32> = (0..6 * 4).map(|i| (i as f32) * 0.37 - 4.0).collect();
        let out = gn.forward(&x, 6).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pairnorm_build_and_accessors() {
        let pn = PairNorm::new(4, 1.0, PairNormMode::Standard).expect("build");
        assert_eq!(pn.feat_dim(), 4);
        assert!((pn.scale() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn pairnorm_invalid_params_error() {
        assert!(PairNorm::new(0, 1.0, PairNormMode::Standard).is_err());
        assert!(PairNorm::new(4, 0.0, PairNormMode::Standard).is_err());
        assert!(PairNorm::new(4, -1.0, PairNormMode::Standard).is_err());
    }

    #[test]
    fn pairnorm_centres_features() {
        // After centring the per-channel mean must be ~0.
        let pn = PairNorm::new(2, 1.0, PairNormMode::Standard).expect("build");
        let x = vec![1.0_f32, 5.0, 3.0, 7.0, 5.0, 9.0]; // means 3, 7
        let out = pn.forward(&x, 3).expect("forward");
        for k in 0..2 {
            let mean: f32 = (0..3).map(|i| out[i * 2 + k]).sum::<f32>() / 3.0;
            assert!(
                mean.abs() < 1e-4,
                "channel {k} mean should be ~0, got {mean}"
            );
        }
    }

    #[test]
    fn pairnorm_standard_target_msr() {
        // After Standard PairNorm with scale s, the mean squared row norm ≈ s².
        let s = 2.0_f32;
        let pn = PairNorm::new(3, s, PairNormMode::Standard).expect("build");
        let x: Vec<f32> = (0..4 * 3).map(|i| i as f32 * 0.5).collect();
        let out = pn.forward(&x, 4).expect("forward");
        let mut msr = 0.0_f32;
        for i in 0..4 {
            for k in 0..3 {
                let v = out[i * 3 + k];
                msr += v * v;
            }
        }
        msr /= 4.0;
        assert!((msr - s * s).abs() < 1e-2, "msr {msr} should be ~{}", s * s);
    }

    #[test]
    fn pairnorm_scale_individual_unit_rows() {
        // PN-SI: each centred row is L2-normalised to scale s.
        let s = 1.0_f32;
        let pn = PairNorm::new(3, s, PairNormMode::ScaleIndividual).expect("build");
        let x: Vec<f32> = (0..3 * 3).map(|i| i as f32 + 1.0).collect();
        let out = pn.forward(&x, 3).expect("forward");
        for i in 0..3 {
            let norm: f32 = (0..3).map(|k| out[i * 3 + k].powi(2)).sum::<f32>().sqrt();
            // rows that aren't all-zero after centring should have norm ~s
            assert!(norm <= s + 1e-3, "row {i} norm {norm} exceeds scale {s}");
        }
    }

    #[test]
    fn pairnorm_feature_mismatch_errors() {
        let pn = PairNorm::new(3, 1.0, PairNormMode::Standard).expect("build");
        let err = pn.forward(&[1.0_f32; 5], 3);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn pairnorm_empty_graph_errors() {
        let pn = PairNorm::new(2, 1.0, PairNormMode::Standard).expect("build");
        assert!(matches!(pn.forward(&[], 0), Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn pairnorm_output_finite_both_modes() {
        for mode in [PairNormMode::Standard, PairNormMode::ScaleIndividual] {
            let pn = PairNorm::new(4, 1.5, mode).expect("build");
            let x: Vec<f32> = (0..5 * 4).map(|i| (i as f32) * 0.21 - 2.0).collect();
            let out = pn.forward(&x, 5).expect("forward");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "mode {mode:?} non-finite"
            );
        }
    }
}
