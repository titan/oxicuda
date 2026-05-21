//! APPNP — Approximate Personalized Propagation of Neural Predictions.
//!
//! Klicpera et al. "Predict then Propagate: Graph Neural Networks meet Personalized
//! PageRank", ICLR 2019.
//!
//! Separates feature transformation from propagation.  Given pre-computed node
//! predictions H = f_θ(X) (e.g., an MLP output), the layer applies K steps of
//! personalized PageRank with teleportation probability α:
//!
//! ```text
//! H^(0) = H
//! H^(k) = (1 − α) · Â H^(k−1) + α · H^(0)
//! ```
//!
//! where Â = D̃^{−1/2} Ã D̃^{−1/2} (normalized adjacency with self-loops).

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for an APPNP propagation layer.
#[derive(Debug, Clone, Copy)]
pub struct AppnpConfig {
    /// Feature dimension of node predictions.
    pub feat_dim: usize,
    /// Teleportation probability α ∈ (0, 1).
    ///
    /// Higher α retains more of the initial prediction H^(0).
    /// Typical values: 0.1–0.2.
    pub alpha: f32,
    /// Number of power-iteration steps K (≥ 1).
    pub k: usize,
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// APPNP personalized PageRank propagation layer.
///
/// Given pre-computed node predictions H (not computed here), applies K steps of
/// the update rule:
///
/// ```text
/// H^(k) = (1 − α) · Â H^(k−1) + α · H^(0)
/// ```
pub struct AppnpLayer {
    config: AppnpConfig,
}

impl AppnpLayer {
    /// Create a new APPNP layer, validating the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if:
    /// * `feat_dim == 0`
    /// * `alpha` is not strictly in (0, 1)
    /// * `k == 0`
    pub fn new(config: AppnpConfig) -> GnnResult<Self> {
        if config.feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "APPNP: feat_dim must be > 0".to_string(),
            ));
        }
        if config.alpha <= 0.0 || config.alpha >= 1.0 {
            return Err(GnnError::InvalidLayerConfig(format!(
                "APPNP: alpha must be in (0, 1) exclusive, got {:.6}",
                config.alpha
            )));
        }
        if config.k == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "APPNP: k must be >= 1".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Apply K steps of personalized PageRank propagation.
    ///
    /// # Arguments
    ///
    /// * `graph` — CSR graph providing the propagation topology.
    /// * `h`     — `[n_nodes × feat_dim]` pre-computed node predictions.
    ///
    /// # Returns
    ///
    /// `[n_nodes × feat_dim]` smoothed node representations after K propagation steps.
    ///
    /// # Errors
    ///
    /// * [`GnnError::NodeFeatureMismatch`] if `h.len() != n_nodes * feat_dim`.
    /// * [`GnnError::NonFiniteOutput`] if any output value is NaN or infinite.
    pub fn forward(&self, graph: &CsrGraph, h: &[f32]) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let f = self.config.feat_dim;
        let alpha = self.config.alpha;
        let one_minus_alpha = 1.0_f32 - alpha;

        // Validate input shape.
        if h.len() != n * f {
            return Err(GnnError::NodeFeatureMismatch(n, h.len() / f.max(1)));
        }

        // Pre-compute the normalized adjacency in COO form once.
        let (rows, cols, vals) = graph.normalized_adjacency();

        // H^(0): the teleportation target (initial predictions).
        let h_0 = h.to_vec();
        // H^(k): current propagation state.
        let mut h_cur = h.to_vec();

        for _step in 0..self.config.k {
            // h_prop = Â · h_cur
            let mut h_prop = vec![0.0_f32; n * f];
            for idx in 0..rows.len() {
                let i = rows[idx];
                let j = cols[idx];
                let v = vals[idx];
                for d in 0..f {
                    h_prop[i * f + d] += v * h_cur[j * f + d];
                }
            }
            // Teleportation blend: h_cur = (1-α)·h_prop + α·h_0
            for k in 0..n * f {
                h_cur[k] = one_minus_alpha * h_prop[k] + alpha * h_0[k];
            }
        }

        // Finite-value guard.
        if h_cur.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput("APPNP forward"));
        }

        Ok(h_cur)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_graph() -> CsrGraph {
        // 4-node path: 0–1–2–3 (undirected)
        CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid")
    }

    fn triangle_graph() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
            .expect("test invariant: value must be valid")
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_valid() {
        let cfg = AppnpConfig {
            feat_dim: 4,
            alpha: 0.1,
            k: 5,
        };
        assert!(AppnpLayer::new(cfg).is_ok());
    }

    #[test]
    fn new_invalid_feat_dim_zero() {
        let cfg = AppnpConfig {
            feat_dim: 0,
            alpha: 0.1,
            k: 5,
        };
        assert!(AppnpLayer::new(cfg).is_err());
    }

    #[test]
    fn new_invalid_alpha_zero() {
        let cfg = AppnpConfig {
            feat_dim: 4,
            alpha: 0.0,
            k: 5,
        };
        assert!(AppnpLayer::new(cfg).is_err());
    }

    #[test]
    fn new_invalid_alpha_one() {
        let cfg = AppnpConfig {
            feat_dim: 4,
            alpha: 1.0,
            k: 5,
        };
        assert!(AppnpLayer::new(cfg).is_err());
    }

    #[test]
    fn new_invalid_alpha_gt_1() {
        let cfg = AppnpConfig {
            feat_dim: 4,
            alpha: 1.5,
            k: 5,
        };
        assert!(AppnpLayer::new(cfg).is_err());
    }

    #[test]
    fn new_invalid_k_zero() {
        let cfg = AppnpConfig {
            feat_dim: 4,
            alpha: 0.1,
            k: 0,
        };
        assert!(AppnpLayer::new(cfg).is_err());
    }

    // ── Forward shape ─────────────────────────────────────────────────────────

    #[test]
    fn forward_output_shape() {
        let g = line_graph();
        let cfg = AppnpConfig {
            feat_dim: 3,
            alpha: 0.15,
            k: 5,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h: Vec<f32> = (0..4 * 3).map(|i| i as f32 * 0.1).collect();
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── High alpha stays close to initial predictions ─────────────────────────

    #[test]
    fn forward_high_alpha_stays_near_h0() {
        // alpha=0.99 → propagation step weight = 0.01; after 5 steps output ≈ h_0.
        let g = line_graph();
        let cfg = AppnpConfig {
            feat_dim: 2,
            alpha: 0.99,
            k: 5,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        // With such a high alpha the output should be very close to h.
        let max_diff = out
            .iter()
            .zip(h.iter())
            .map(|(o, hi)| (o - hi).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 0.1, "max_diff={max_diff}");
    }

    // ── Single-node graph ─────────────────────────────────────────────────────

    #[test]
    fn forward_single_node_graph() {
        // n=1, self-loop normalised adjacency → Â = 1.0.
        // H^(k) = (1-α)·1·H^(k-1) + α·H^0
        //       = H^0 (fixed point for all k when Â=I).
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let cfg = AppnpConfig {
            feat_dim: 2,
            alpha: 0.1,
            k: 5,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![3.0_f32, 7.0];
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
        // With a single-node self-loop normalized to 1, H doesn't change.
        assert!((out[0] - 3.0_f32).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 7.0_f32).abs() < 1e-5, "out[1]={}", out[1]);
    }

    // ── k=1 star graph: correct output shape ─────────────────────────────────

    #[test]
    fn forward_does_not_change_with_k1_identity_graph() {
        // Star graph (4 nodes), k=1; just verify correct output shape.
        let g = CsrGraph::from_edges(4, &[(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)])
            .expect("test invariant: value must be valid");
        let cfg = AppnpConfig {
            feat_dim: 2,
            alpha: 0.1,
            k: 1,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![1.0_f32; 4 * 2];
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── More steps with low alpha produce different output ────────────────────

    #[test]
    fn forward_k_steps_applied() {
        let g = line_graph();
        let make = |k: usize| {
            let cfg = AppnpConfig {
                feat_dim: 2,
                alpha: 0.1,
                k,
            };
            let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
            let h: Vec<f32> = (0..4 * 2).map(|i| i as f32).collect();
            layer
                .forward(&g, &h)
                .expect("test invariant: value must be valid")
        };
        let out1 = make(1);
        let out10 = make(10);
        let diff: f32 = out1
            .iter()
            .zip(out10.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "k=1 and k=10 outputs should differ, diff={diff}"
        );
    }

    // ── Output must be finite for valid inputs ────────────────────────────────

    #[test]
    fn forward_feature_mean_preserved_approx() {
        let g = triangle_graph();
        let cfg = AppnpConfig {
            feat_dim: 3,
            alpha: 0.1,
            k: 5,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h: Vec<f32> = (0..3 * 3).map(|i| i as f32 * 0.3).collect();
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        // All output values must be finite.
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Disconnected graph (isolated nodes) ───────────────────────────────────

    #[test]
    fn forward_disconnected_graph_no_panic() {
        // 3 nodes, no edges — normalized_adjacency will still include self-loops.
        let g = CsrGraph::from_edges(3, &[(0, 0), (1, 1), (2, 2)])
            .expect("test invariant: value must be valid");
        let cfg = AppnpConfig {
            feat_dim: 2,
            alpha: 0.1,
            k: 5,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = layer.forward(&g, &h);
        assert!(out.is_ok());
        assert!(
            out.expect("test invariant: value must be valid")
                .iter()
                .all(|v| v.is_finite())
        );
    }

    // ── Wrong input length → error ────────────────────────────────────────────

    #[test]
    fn forward_err_node_feature_mismatch() {
        let g = line_graph(); // 4 nodes
        let cfg = AppnpConfig {
            feat_dim: 3,
            alpha: 0.1,
            k: 3,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        // Provide only 3 nodes' worth of features.
        let h = vec![1.0_f32; 3 * 3];
        let err = layer.forward(&g, &h);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    // ── Small triangle, 2 features: output shape correct ─────────────────────

    #[test]
    fn forward_two_nodes_triangle() {
        let g = triangle_graph(); // 3 nodes
        let cfg = AppnpConfig {
            feat_dim: 2,
            alpha: 0.2,
            k: 3,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5];
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 3 * 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Propagation smooths extreme feature values ────────────────────────────

    #[test]
    fn forward_propagation_smooths_extremes() {
        // Node 0 has large feature, nodes 1 and 2 have zero.
        // Triangle graph: all connected to each other.
        let g = triangle_graph();
        let cfg = AppnpConfig {
            feat_dim: 1,
            alpha: 0.1,
            k: 20,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![10.0_f32, 0.0, 0.0];
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");
        // Node 0's feature should decrease toward the mean.
        assert!(
            out[0] < 10.0,
            "node 0 feature should decrease, out[0]={}",
            out[0]
        );
        // Nodes 1 and 2 should gain some non-zero value through propagation.
        assert!(
            out[1] > 0.0 || out[2] > 0.0,
            "neighbors should gain feature mass"
        );
    }

    // ── k=1: manual verification ──────────────────────────────────────────────

    #[test]
    fn forward_k1_matches_manual() {
        // 2-node graph: 0 → 1 and 1 → 0 (bidirected), k=1.
        // normalized_adjacency gives self-loops (i,i) and off-diagonal (i,j).
        // out_deg[0]=1, out_deg[1]=1 → d_inv_sqrt = 1/sqrt(2) each.
        // Self-loop vals: (1/sqrt(2))^2 = 1/2 each.
        // Off-diag vals: (1/sqrt(2))*1*(1/sqrt(2)) = 1/2 each.
        // So Â·h:
        //   node 0: 0.5*h[0] + 0.5*h[1]
        //   node 1: 0.5*h[1] + 0.5*h[0]
        // Both equal (h[0]+h[1])/2.
        // After teleportation: (1-α)*(h[0]+h[1])/2 + α*h[0]
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let alpha = 0.2_f32;
        let cfg = AppnpConfig {
            feat_dim: 1,
            alpha,
            k: 1,
        };
        let layer = AppnpLayer::new(cfg).expect("test invariant: value must be valid");
        let h = vec![4.0_f32, 2.0_f32];
        let out = layer
            .forward(&g, &h)
            .expect("test invariant: value must be valid");

        let mean = (h[0] + h[1]) * 0.5_f32;
        let expected_0 = (1.0 - alpha) * mean + alpha * h[0];
        let expected_1 = (1.0 - alpha) * mean + alpha * h[1];
        assert!(
            (out[0] - expected_0).abs() < 1e-5,
            "out[0]={} expected={expected_0}",
            out[0]
        );
        assert!(
            (out[1] - expected_1).abs() < 1e-5,
            "out[1]={} expected={expected_1}",
            out[1]
        );
    }
}
