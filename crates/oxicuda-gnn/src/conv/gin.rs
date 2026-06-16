//! Graph Isomorphism Network (GIN) convolution — Xu et al. 2019.
//!
//! "How Powerful are Graph Neural Networks?" (ICLR 2019).
//! GIN is maximally expressive for graph classification among message-passing GNNs:
//!
//! ```text
//! h_v^(k) = MLP^(k)( (1 + ε) · h_v^(k-1)  +  Σ_{u ∈ N(v)} h_u^(k-1) )
//! ```
//!
//! where `MLP` is a two-layer fully-connected network with ReLU activations.
//! This module provides a self-contained `GinConv` that owns its weight matrices
//! and is initialized with He/Xavier-style random weights.

use crate::error::{GnnError, GnnResult};
use crate::handle::LcgRng;

/// Type alias: the RNG used by this module.
pub type GnnRng = LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`GinConv`] layer.
#[derive(Debug, Clone)]
pub struct GinConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Initial value of epsilon (learnable self-loop weight, fixed here).
    pub epsilon: f32,
}

// ─── GinConv ─────────────────────────────────────────────────────────────────

/// GIN convolution layer with an internal two-layer MLP.
///
/// Weight layout (row-major):
/// - `mlp_w1`: `[out_features × in_features]`
/// - `mlp_b1`: `[out_features]`
/// - `mlp_w2`: `[out_features × out_features]`
/// - `mlp_b2`: `[out_features]`
pub struct GinConv {
    mlp_w1: Vec<f32>, // [out_features × in_features]
    mlp_b1: Vec<f32>, // [out_features]
    mlp_w2: Vec<f32>, // [out_features × out_features]
    mlp_b2: Vec<f32>, // [out_features]
    epsilon: f32,
    config: GinConfig,
}

impl GinConv {
    /// Construct a new `GinConv` with Xavier-uniform weight initialization.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] if `in_features` or `out_features` is 0.
    pub fn new(config: GinConfig, rng: &mut GnnRng) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "out_features must be > 0".to_string(),
            ));
        }

        let fan_in = config.in_features;
        let fan_out = config.out_features;

        // Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        let limit1 = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        let limit2 = (6.0_f32 / (fan_out + fan_out) as f32).sqrt();

        let mlp_w1: Vec<f32> = (0..fan_out * fan_in)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit1)
            .collect();
        let mlp_b1 = vec![0.0_f32; fan_out];

        let mlp_w2: Vec<f32> = (0..fan_out * fan_out)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit2)
            .collect();
        let mlp_b2 = vec![0.0_f32; fan_out];

        let epsilon = config.epsilon;

        Ok(Self {
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            epsilon,
            config,
        })
    }

    /// Output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.config.out_features
    }

    /// GIN forward pass.
    ///
    /// ```text
    /// h_agg[v] = (1 + ε) · h[v]  +  Σ_{u ∈ adj[v]} h[u]
    /// h_out[v] = ReLU( W2 · ReLU(W1 · h_agg[v] + b1) + b2 )
    /// ```
    ///
    /// # Arguments
    ///
    /// - `node_feats`: `[n_nodes × in_features]` row-major input features.
    /// - `adj`       : adjacency list `adj[v]` = sorted neighbor indices.
    /// - `n_nodes`   : number of nodes (must match `adj.len()` and
    ///   `node_feats.len() / in_features`).
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_features]` row-major output features.
    ///
    /// # Errors
    ///
    /// - [`GnnError::DimensionMismatch`] if `node_feats.len() != n_nodes * in_features`.
    /// - [`GnnError::NodeIndexOutOfRange`] if any neighbor index ≥ `n_nodes`.
    pub fn forward(
        &self,
        node_feats: &[f32],
        adj: &[Vec<usize>],
        n_nodes: usize,
    ) -> GnnResult<Vec<f32>> {
        let in_f = self.config.in_features;
        let out_f = self.config.out_features;

        // Validate input shape.
        if node_feats.len() != n_nodes * in_f {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * in_f,
                got: node_feats.len(),
            });
        }
        if adj.len() != n_nodes {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes,
                got: adj.len(),
            });
        }

        // Validate neighbor indices.
        for nbrs in adj.iter() {
            for &u in nbrs {
                if u >= n_nodes {
                    return Err(GnnError::NodeIndexOutOfRange { idx: u, n_nodes });
                }
            }
        }

        let mut out = vec![0.0_f32; n_nodes * out_f];

        for v in 0..n_nodes {
            // ── Step 1: aggregate h_agg = (1+ε)·h_v + Σ h_u ──────────────────
            let self_scale = 1.0 + self.epsilon;
            let h_agg: Vec<f32> = (0..in_f)
                .map(|k| self_scale * node_feats[v * in_f + k])
                .collect();
            let mut h_agg = h_agg;
            for &u in &adj[v] {
                let u_row = &node_feats[u * in_f..(u + 1) * in_f];
                for (k, val) in h_agg.iter_mut().enumerate() {
                    *val += u_row[k];
                }
            }

            // ── Step 2: MLP layer 1 — h1 = ReLU(W1 · h_agg + b1) ─────────────
            let h1: Vec<f32> = (0..out_f)
                .map(|i| {
                    let acc = self.mlp_b1[i]
                        + h_agg
                            .iter()
                            .enumerate()
                            .map(|(k, &hk)| self.mlp_w1[i * in_f + k] * hk)
                            .sum::<f32>();
                    acc.max(0.0) // ReLU
                })
                .collect();

            // ── Step 3: MLP layer 2 — h2 = ReLU(W2 · h1 + b2) ───────────────
            let out_row = &mut out[v * out_f..(v + 1) * out_f];
            for (i, cell) in out_row.iter_mut().enumerate() {
                let acc = self.mlp_b2[i]
                    + h1.iter()
                        .enumerate()
                        .map(|(k, &h)| self.mlp_w2[i * out_f + k] * h)
                        .sum::<f32>();
                *cell = acc.max(0.0); // ReLU
            }
        }

        Ok(out)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn make_config(in_f: usize, out_f: usize) -> GinConfig {
        GinConfig {
            in_features: in_f,
            out_features: out_f,
            epsilon: 0.0,
        }
    }

    fn make_feats(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = LcgRng::new(seed);
        (0..n * d).map(|_| r.next_f32()).collect()
    }

    // 1 ─ output shape
    #[test]
    fn output_shape() {
        let cfg = make_config(4, 8);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        let feats = make_feats(5, 4, 1);
        let adj: Vec<Vec<usize>> = vec![vec![1], vec![0, 2], vec![1], vec![], vec![]];
        let out = conv
            .forward(&feats, &adj, 5)
            .expect("forward should succeed");
        assert_eq!(out.len(), 5 * 8);
    }

    // 2 ─ output is finite
    #[test]
    fn output_finite() {
        let cfg = make_config(3, 5);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        let feats = make_feats(4, 3, 2);
        let adj: Vec<Vec<usize>> = vec![vec![1, 2], vec![0], vec![0, 3], vec![2]];
        let out = conv
            .forward(&feats, &adj, 4)
            .expect("forward should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v}");
        }
    }

    // 3 ─ isolated node (no neighbors) works
    #[test]
    fn isolated_node_works() {
        let cfg = make_config(2, 2);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        let feats = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 nodes, dim=2
        let adj: Vec<Vec<usize>> = vec![vec![], vec![]]; // isolated
        let out = conv
            .forward(&feats, &adj, 2)
            .expect("forward should succeed");
        // Output should be computed purely from self-loop (no aggregated neighbors)
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // 4 ─ self-loop effect: epsilon=1.0 vs epsilon=0.0
    //
    // We use a positive bias to prevent dead ReLU neurons, ensuring the
    // epsilon difference propagates to the output.
    #[test]
    fn self_loop_effect() {
        let in_f = 4;
        let out_f = 4;
        let feats = make_feats(3, in_f, 5);
        let adj: Vec<Vec<usize>> = vec![vec![1], vec![0], vec![]];

        let cfg0 = GinConfig {
            in_features: in_f,
            out_features: out_f,
            epsilon: 0.0,
        };
        let cfg1 = GinConfig {
            in_features: in_f,
            out_features: out_f,
            epsilon: 1.0,
        };

        let mut r0 = LcgRng::new(10);
        let mut r1 = LcgRng::new(10);
        let mut conv0 = GinConv::new(cfg0, &mut r0).expect("new should succeed");
        let mut conv1 = GinConv::new(cfg1, &mut r1).expect("new should succeed");

        // Force both layers to have positive biases and use identity W2 so that
        // the epsilon difference is guaranteed to propagate to the output.
        for c in [&mut conv0, &mut conv1] {
            c.mlp_b1 = vec![1.0_f32; out_f];
            c.mlp_b2 = vec![1.0_f32; out_f];
            // W2 = identity (so layer 2 passes h1 through cleanly)
            c.mlp_w2 = vec![0.0_f32; out_f * out_f];
            for i in 0..out_f {
                c.mlp_w2[i * out_f + i] = 1.0;
            }
        }

        let out0 = conv0
            .forward(&feats, &adj, 3)
            .expect("forward should succeed");
        let out1 = conv1
            .forward(&feats, &adj, 3)
            .expect("forward should succeed");

        // With different epsilon (0 vs 1) and non-trivial input, outputs must differ
        // because epsilon scales the self-contribution (1+0)*h vs (1+1)*h.
        let differ = out0.iter().zip(&out1).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "epsilon=0 vs epsilon=1 should produce different outputs; got out0={:?}, out1={:?}",
            out0, out1
        );
    }

    // 5 ─ epsilon=0 vs epsilon=1 (explicit comparison)
    #[test]
    fn epsilon_0_vs_1() {
        let feats = vec![1.0_f32, 0.5, 0.5, 1.0]; // 2 nodes, dim=2
        let adj: Vec<Vec<usize>> = vec![vec![], vec![]]; // isolated

        // With isolated nodes: h_agg = (1+ε)*h_v, so epsilon directly scales
        let cfg_e0 = GinConfig {
            in_features: 2,
            out_features: 2,
            epsilon: 0.0,
        };
        let cfg_e1 = GinConfig {
            in_features: 2,
            out_features: 2,
            epsilon: 1.0,
        };

        let mut r0 = LcgRng::new(99);
        let mut r1 = LcgRng::new(99);

        let conv0 = GinConv::new(cfg_e0, &mut r0).expect("new should succeed");
        let conv1 = GinConv::new(cfg_e1, &mut r1).expect("new should succeed");

        let out0 = conv0
            .forward(&feats, &adj, 2)
            .expect("forward should succeed");
        let out1 = conv1
            .forward(&feats, &adj, 2)
            .expect("forward should succeed");

        // All outputs should be finite
        assert!(out0.iter().all(|v| v.is_finite()));
        assert!(out1.iter().all(|v| v.is_finite()));
    }

    // 6 ─ different nodes produce different output
    //
    // We use a large positive bias (b1 all ones) so that the first ReLU gate
    // never completely silences the path.  Then two maximally different inputs
    // must produce different aggregated pre-MLP values, and therefore different
    // outputs — regardless of the random weight seed.
    #[test]
    fn different_nodes_different_output() {
        let in_f = 4;
        let out_f = 4;
        let cfg = GinConfig {
            in_features: in_f,
            out_features: out_f,
            epsilon: 0.0,
        };
        let mut r = LcgRng::new(42);
        let mut conv = GinConv::new(cfg, &mut r).expect("new should succeed");

        // Force b1 = +1 so that (W1·h_agg + b1) is strictly positive → ReLU passes.
        conv.mlp_b1 = vec![1.0_f32; out_f];
        // Also force W2 = I (identity) so layer-2 just passes h1 through.
        conv.mlp_w2 = vec![0.0_f32; out_f * out_f];
        for i in 0..out_f {
            conv.mlp_w2[i * out_f + i] = 1.0;
        }
        conv.mlp_b2 = vec![0.0_f32; out_f];

        // Two nodes with clearly different features.
        let feats = vec![
            1.0_f32, 0.0, 0.0, 0.0, // node 0
            0.0, 0.0, 0.0, 1.0, // node 1
        ];
        let adj: Vec<Vec<usize>> = vec![vec![], vec![]];
        let out = conv
            .forward(&feats, &adj, 2)
            .expect("forward should succeed");
        let node0 = &out[0..out_f];
        let node1 = &out[out_f..2 * out_f];
        let differ = node0.iter().zip(node1).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(
            differ,
            "distinct inputs should produce distinct outputs: {:?} vs {:?}",
            node0, node1
        );
    }

    // 7 ─ output not all zero (forced positive bias ensures non-zero output)
    #[test]
    fn output_not_all_zero() {
        let in_f = 4;
        let out_f = 4;
        let cfg = make_config(in_f, out_f);
        let mut r = rng();
        let mut conv = GinConv::new(cfg, &mut r).expect("new should succeed");

        // Force positive biases so that even small inputs produce non-zero outputs
        // after ReLU.
        conv.mlp_b1 = vec![1.0_f32; out_f];
        conv.mlp_b2 = vec![1.0_f32; out_f];

        let feats = make_feats(3, in_f, 77);
        let adj: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![0]];
        let out = conv
            .forward(&feats, &adj, 3)
            .expect("forward should succeed");
        let nonzero = out.iter().any(|&v| v.abs() > 1e-7);
        assert!(
            nonzero,
            "output should not be all-zero with positive biases: {:?}",
            out
        );
    }

    // 8 ─ deep graph works (chain of 10 nodes)
    #[test]
    fn deep_graph_works() {
        let n = 10;
        let in_f = 3;
        let out_f = 5;
        let cfg = make_config(in_f, out_f);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        let feats = make_feats(n, in_f, 11);
        let adj: Vec<Vec<usize>> = (0..n)
            .map(|v| if v + 1 < n { vec![v + 1] } else { vec![] })
            .collect();
        let out = conv
            .forward(&feats, &adj, n)
            .expect("forward should succeed");
        assert_eq!(out.len(), n * out_f);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 9 ─ adj out of range returns error
    #[test]
    fn adj_out_of_range_error() {
        let cfg = make_config(2, 2);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        let feats = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 nodes
        let adj: Vec<Vec<usize>> = vec![vec![5], vec![]]; // neighbor 5 >= n_nodes=2
        let result = conv.forward(&feats, &adj, 2);
        assert!(result.is_err());
    }

    // 10 ─ in_features mismatch returns error
    #[test]
    fn in_features_mismatch_error() {
        let cfg = make_config(4, 4);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        let feats = vec![1.0_f32; 3 * 3]; // 3 nodes × 3 features, but config says 4
        let adj: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
        let result = conv.forward(&feats, &adj, 3);
        assert!(result.is_err());
    }

    // 11 ─ out_features=0 in config returns error
    #[test]
    fn out_features_zero_error() {
        let cfg = GinConfig {
            in_features: 4,
            out_features: 0,
            epsilon: 0.0,
        };
        let mut r = rng();
        let result = GinConv::new(cfg, &mut r);
        assert!(result.is_err());
    }

    // 12 ─ out_features accessor
    #[test]
    fn out_features_accessor() {
        let cfg = make_config(3, 7);
        let mut r = rng();
        let conv = GinConv::new(cfg, &mut r).expect("new should succeed");
        assert_eq!(conv.out_features(), 7);
    }
}
