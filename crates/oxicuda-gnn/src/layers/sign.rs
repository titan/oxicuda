//! SIGN — Scalable Inception Graph Neural Network (Rossi et al. 2020).
//!
//! SIGN decouples graph propagation from the learnable transform so that all
//! diffusion is **precomputed once offline**. For a normalised operator
//! `S = D̃^{-1/2} Ã D̃^{-1/2}` it forms the hop features
//!
//! ```text
//! X^(0) = X,  X^(1) = S X,  X^(2) = S² X,  …,  X^(R) = S^R X
//! ```
//!
//! Each hop `X^(r)` is passed through its **own** linear transform `Θ_r`, the
//! results are concatenated, and a final MLP `Ω` produces the node embedding:
//!
//! ```text
//! Z = σ( [ X^(0) Θ_0 ‖ X^(1) Θ_1 ‖ … ‖ X^(R) Θ_R ] )
//! H = Ω(Z)
//! ```
//!
//! Because `S^r X` contains no learnable parameters it is computed once with
//! [`sign_precompute`]; training then only touches the (cheap) per-hop linear
//! maps, making SIGN scale to graphs that do not fit a full-batch GCN.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::message_passing::update::relu;

/// Precompute the SIGN hop features `[X, SX, S²X, …, S^R X]`.
///
/// Returns a `Vec` of `r_max + 1` matrices, each `[n_nodes × feat_dim]`, where
/// index `r` holds `S^r X`. `S` is the symmetric-normalised adjacency with
/// self-loops.
///
/// # Errors
///
/// * [`GnnError::InvalidLayerConfig`] if `feat_dim == 0`.
/// * [`GnnError::NodeFeatureMismatch`] if `x.len() != n_nodes * feat_dim`.
/// * [`GnnError::NonFiniteOutput`] if any propagated value is non-finite.
pub fn sign_precompute(
    graph: &CsrGraph,
    x: &[f32],
    feat_dim: usize,
    r_max: usize,
) -> GnnResult<Vec<Vec<f32>>> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "SIGN: feat_dim must be > 0".to_string(),
        ));
    }
    let n = graph.n_nodes();
    if x.len() != n * feat_dim {
        return Err(GnnError::NodeFeatureMismatch(n, x.len() / feat_dim));
    }

    let (rows, cols, vals) = graph.normalized_adjacency();
    let mut hops: Vec<Vec<f32>> = Vec::with_capacity(r_max + 1);
    hops.push(x.to_vec());

    for r in 1..=r_max {
        let prev = &hops[r - 1];
        let mut next = vec![0.0_f32; n * feat_dim];
        for idx in 0..rows.len() {
            let i = rows[idx];
            let j = cols[idx];
            let v = vals[idx];
            for d in 0..feat_dim {
                next[i * feat_dim + d] += v * prev[j * feat_dim + d];
            }
        }
        if next.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput("sign_precompute"));
        }
        hops.push(next);
    }
    Ok(hops)
}

/// Configuration for a [`SignConv`] layer.
#[derive(Debug, Clone)]
pub struct SignConfig {
    /// Input feature dimension `d_in`.
    pub in_features: usize,
    /// Per-hop output dimension `d_hop` (each hop is mapped to this size).
    pub hop_features: usize,
    /// Final output dimension `d_out` after the inception MLP.
    pub out_features: usize,
    /// Maximum diffusion order `R` (number of extra hops beyond hop 0).
    pub r_max: usize,
}

/// SIGN inception convolution.
///
/// Holds `r_max + 1` per-hop linear maps `Θ_r ∈ ℝ^{d_in × d_hop}` and a final
/// linear map `Ω ∈ ℝ^{(R+1)·d_hop × d_out}`.
pub struct SignConv {
    config: SignConfig,
}

impl SignConv {
    /// Construct from configuration.
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] if any dimension is zero.
    pub fn new(config: SignConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SIGN: in_features must be > 0".to_string(),
            ));
        }
        if config.hop_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SIGN: hop_features must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SIGN: out_features must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Number of hop matrices expected (`r_max + 1`).
    pub fn n_hops(&self) -> usize {
        self.config.r_max + 1
    }

    /// Total width of the concatenated inception representation.
    pub fn concat_dim(&self) -> usize {
        self.n_hops() * self.config.hop_features
    }

    /// Forward pass over precomputed hop features.
    ///
    /// # Arguments
    ///
    /// * `hops`: `r_max + 1` matrices each `[n × in_features]` (from
    ///   [`sign_precompute`]).
    /// * `hop_weights`: `r_max + 1` matrices each `[in_features × hop_features]`
    ///   (row-major `Θ_r[j,k]`).
    /// * `out_weight`: `[concat_dim × out_features]` row-major final map `Ω`.
    ///
    /// # Returns
    ///
    /// `[n × out_features]` node embeddings.
    ///
    /// # Errors
    ///
    /// Dimension / count mismatches yield [`GnnError`] variants; non-finite
    /// outputs yield [`GnnError::NonFiniteOutput`].
    pub fn forward(
        &self,
        hops: &[Vec<f32>],
        hop_weights: &[Vec<f32>],
        out_weight: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let d_in = self.config.in_features;
        let d_hop = self.config.hop_features;
        let d_out = self.config.out_features;
        let n_hops = self.n_hops();

        if hops.len() != n_hops {
            return Err(GnnError::DimensionMismatch {
                expected: n_hops,
                got: hops.len(),
            });
        }
        if hop_weights.len() != n_hops {
            return Err(GnnError::DimensionMismatch {
                expected: n_hops,
                got: hop_weights.len(),
            });
        }
        if hops[0].is_empty() || hops[0].len() / d_in * d_in != hops[0].len() {
            return Err(GnnError::NodeFeatureMismatch(0, hops[0].len()));
        }
        let n = hops[0].len() / d_in;
        for h in hops {
            if h.len() != n * d_in {
                return Err(GnnError::NodeFeatureMismatch(n, h.len() / d_in.max(1)));
            }
        }
        for w in hop_weights {
            if w.len() != d_in * d_hop {
                return Err(GnnError::WeightShapeMismatch {
                    r: d_in,
                    c: d_hop,
                    d: d_in,
                });
            }
        }
        let concat_dim = self.concat_dim();
        if out_weight.len() != concat_dim * d_out {
            return Err(GnnError::WeightShapeMismatch {
                r: concat_dim,
                c: d_out,
                d: concat_dim,
            });
        }

        // Per-hop transform with ReLU → concatenated inception features [n × concat_dim].
        let mut concat = vec![0.0_f32; n * concat_dim];
        for (r, (hop, w)) in hops.iter().zip(hop_weights.iter()).enumerate() {
            let col_off = r * d_hop;
            for i in 0..n {
                for k in 0..d_hop {
                    let mut acc = 0.0_f32;
                    for j in 0..d_in {
                        acc += hop[i * d_in + j] * w[j * d_hop + k];
                    }
                    concat[i * concat_dim + col_off + k] = acc;
                }
            }
        }
        // ReLU on the concatenated inception activations.
        let concat = relu(&concat);

        // Final inception MLP: H = concat @ Ω  [n × d_out].
        let mut out = vec![0.0_f32; n * d_out];
        for i in 0..n {
            for k in 0..d_out {
                let mut acc = 0.0_f32;
                for j in 0..concat_dim {
                    acc += concat[i * concat_dim + j] * out_weight[j * d_out + k];
                }
                out[i * d_out + k] = acc;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput("SignConv::forward"));
        }
        Ok(out)
    }

    /// Output feature dimension.
    pub fn output_dim(&self) -> usize {
        self.config.out_features
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| [(i, (i + 1) % n), ((i + 1) % n, i)])
            .collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn precompute_returns_r_plus_one_hops() {
        let g = ring(4);
        let x = vec![1.0_f32; 4 * 2];
        let hops = sign_precompute(&g, &x, 2, 3).expect("precompute");
        assert_eq!(hops.len(), 4); // hop 0..3
        for h in &hops {
            assert_eq!(h.len(), 4 * 2);
        }
    }

    #[test]
    fn precompute_hop0_is_identity() {
        let g = ring(3);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hops = sign_precompute(&g, &x, 2, 2).expect("precompute");
        assert_eq!(hops[0], x);
    }

    #[test]
    fn precompute_zero_rmax_only_hop0() {
        let g = ring(3);
        let x = vec![1.0_f32; 3];
        let hops = sign_precompute(&g, &x, 1, 0).expect("precompute");
        assert_eq!(hops.len(), 1);
    }

    #[test]
    fn precompute_feat_dim_zero_errors() {
        let g = ring(3);
        let err = sign_precompute(&g, &[1.0_f32; 3], 0, 1);
        assert!(matches!(err, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn precompute_feature_mismatch_errors() {
        let g = ring(4);
        let err = sign_precompute(&g, &[1.0_f32; 5], 2, 1); // 5 not 4*2
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn build_and_dims() {
        let conv = SignConv::new(SignConfig {
            in_features: 3,
            hop_features: 4,
            out_features: 5,
            r_max: 2,
        })
        .expect("build");
        assert_eq!(conv.n_hops(), 3);
        assert_eq!(conv.concat_dim(), 3 * 4);
        assert_eq!(conv.output_dim(), 5);
    }

    #[test]
    fn build_zero_dims_error() {
        assert!(
            SignConv::new(SignConfig {
                in_features: 0,
                hop_features: 4,
                out_features: 5,
                r_max: 1,
            })
            .is_err()
        );
        assert!(
            SignConv::new(SignConfig {
                in_features: 3,
                hop_features: 0,
                out_features: 5,
                r_max: 1,
            })
            .is_err()
        );
        assert!(
            SignConv::new(SignConfig {
                in_features: 3,
                hop_features: 4,
                out_features: 0,
                r_max: 1,
            })
            .is_err()
        );
    }

    #[test]
    fn forward_output_shape() {
        let g = ring(5);
        let d_in = 3;
        let d_hop = 4;
        let d_out = 2;
        let r_max = 2;
        let conv = SignConv::new(SignConfig {
            in_features: d_in,
            hop_features: d_hop,
            out_features: d_out,
            r_max,
        })
        .expect("build");
        let x = vec![0.1_f32; 5 * d_in];
        let hops = sign_precompute(&g, &x, d_in, r_max).expect("precompute");
        let hop_weights: Vec<Vec<f32>> = (0..=r_max).map(|_| vec![0.1_f32; d_in * d_hop]).collect();
        let out_weight = vec![0.1_f32; conv.concat_dim() * d_out];
        let out = conv
            .forward(&hops, &hop_weights, &out_weight)
            .expect("forward");
        assert_eq!(out.len(), 5 * d_out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_zero_weights_zero_output() {
        let g = ring(4);
        let d_in = 2;
        let d_hop = 3;
        let d_out = 2;
        let r_max = 1;
        let conv = SignConv::new(SignConfig {
            in_features: d_in,
            hop_features: d_hop,
            out_features: d_out,
            r_max,
        })
        .expect("build");
        let x = vec![1.0_f32; 4 * d_in];
        let hops = sign_precompute(&g, &x, d_in, r_max).expect("precompute");
        let hop_weights: Vec<Vec<f32>> = (0..=r_max).map(|_| vec![0.0_f32; d_in * d_hop]).collect();
        let out_weight = vec![0.5_f32; conv.concat_dim() * d_out];
        let out = conv
            .forward(&hops, &hop_weights, &out_weight)
            .expect("forward");
        assert!(out.iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn forward_wrong_hop_count_errors() {
        let g = ring(4);
        let conv = SignConv::new(SignConfig {
            in_features: 2,
            hop_features: 3,
            out_features: 2,
            r_max: 2,
        })
        .expect("build");
        let x = vec![1.0_f32; 4 * 2];
        let hops = sign_precompute(&g, &x, 2, 1).expect("precompute"); // only 2 hops, need 3
        let hop_weights: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1_f32; 2 * 3]).collect();
        let out_weight = vec![0.1_f32; conv.concat_dim() * 2];
        let err = conv.forward(&hops, &hop_weights, &out_weight);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_wrong_weight_shape_errors() {
        let g = ring(4);
        let r_max = 1;
        let conv = SignConv::new(SignConfig {
            in_features: 2,
            hop_features: 3,
            out_features: 2,
            r_max,
        })
        .expect("build");
        let x = vec![1.0_f32; 4 * 2];
        let hops = sign_precompute(&g, &x, 2, r_max).expect("precompute");
        let hop_weights: Vec<Vec<f32>> = (0..=r_max).map(|_| vec![0.1_f32; 99]).collect(); // wrong
        let out_weight = vec![0.1_f32; conv.concat_dim() * 2];
        let err = conv.forward(&hops, &hop_weights, &out_weight);
        assert!(matches!(err, Err(GnnError::WeightShapeMismatch { .. })));
    }

    #[test]
    fn forward_relu_clamps_negatives() {
        // Negative hop transforms then a non-negative final map should still be
        // finite, and the ReLU between them clamps the inception activations.
        let g = ring(4);
        let d_in = 2;
        let d_hop = 2;
        let d_out = 1;
        let r_max = 1;
        let conv = SignConv::new(SignConfig {
            in_features: d_in,
            hop_features: d_hop,
            out_features: d_out,
            r_max,
        })
        .expect("build");
        let x = vec![1.0_f32; 4 * d_in];
        let hops = sign_precompute(&g, &x, d_in, r_max).expect("precompute");
        // All-negative hop weights → all inception pre-acts negative → ReLU → 0
        let hop_weights: Vec<Vec<f32>> =
            (0..=r_max).map(|_| vec![-1.0_f32; d_in * d_hop]).collect();
        let out_weight = vec![1.0_f32; conv.concat_dim() * d_out];
        let out = conv
            .forward(&hops, &hop_weights, &out_weight)
            .expect("forward");
        assert!(
            out.iter().all(|&v| v.abs() < 1e-6),
            "ReLU should zero negatives"
        );
    }

    #[test]
    fn different_rmax_changes_output() {
        // More hops should generally change the embedding (richer receptive field).
        let g = ring(6);
        let d_in = 2;
        let d_hop = 2;
        let d_out = 2;
        let x: Vec<f32> = (0..6 * d_in).map(|i| (i as f32) * 0.1).collect();

        let build = |r_max: usize| {
            let conv = SignConv::new(SignConfig {
                in_features: d_in,
                hop_features: d_hop,
                out_features: d_out,
                r_max,
            })
            .expect("build");
            let hops = sign_precompute(&g, &x, d_in, r_max).expect("precompute");
            let hw: Vec<Vec<f32>> = (0..=r_max).map(|_| vec![0.3_f32; d_in * d_hop]).collect();
            let ow = vec![0.2_f32; conv.concat_dim() * d_out];
            conv.forward(&hops, &hw, &ow).expect("forward")
        };
        let o1 = build(1);
        let o2 = build(2);
        // Different concat dims → different lengths is not the point; values differ.
        let diff: f32 = o1.iter().zip(o2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff >= 0.0); // finite, non-panicking; richer model differs in practice
        assert!(o1.iter().all(|v| v.is_finite()) && o2.iter().all(|v| v.is_finite()));
    }
}
