//! SortPooling readout — the `SortPool` layer of DGCNN (Zhang et al. 2018, AAAI
//! "An End-to-End Deep Learning Architecture for Graph Classification").
//!
//! SortPooling turns a variable-size graph into a **fixed-size** `k × feat_dim`
//! representation so that a downstream 1-D CNN / MLP can consume it:
//!
//! 1. Sort all nodes by their feature rows in descending lexicographic order,
//!    using the **last** feature channel as the primary key (in DGCNN this
//!    channel is the finest continuous Weisfeiler-Leman "colour"). Ties are
//!    broken by earlier channels, then by node index for determinism.
//! 2. Keep the first `k` rows. If the graph has more than `k` nodes the tail is
//!    truncated; if it has fewer, the output is zero-padded up to `k` rows.
//!
//! The result is a deterministic, permutation-invariant `[k × feat_dim]` matrix
//! (row-major) that establishes a consistent node ordering across graphs.

use crate::error::{GnnError, GnnResult};

/// Configuration for a [`SortPool`] readout.
#[derive(Debug, Clone, Copy)]
pub struct SortPoolConfig {
    /// Feature dimension `feat_dim` of each node row.
    pub feat_dim: usize,
    /// Fixed number of retained rows `k` (truncate or zero-pad to this many).
    pub k: usize,
}

/// SortPooling readout layer.
#[derive(Debug, Clone)]
pub struct SortPool {
    config: SortPoolConfig,
}

impl SortPool {
    /// Construct a SortPool readout.
    ///
    /// # Errors
    ///
    /// [`GnnError::InvalidLayerConfig`] if `feat_dim == 0` or `k == 0`.
    pub fn new(config: SortPoolConfig) -> GnnResult<Self> {
        if config.feat_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SortPool: feat_dim must be > 0".to_string(),
            ));
        }
        if config.k == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "SortPool: k must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Sort node indices descending by the last feature channel, breaking ties by
    /// earlier channels (also descending) and finally by ascending node index.
    fn sorted_order(&self, x: &[f32], n_nodes: usize) -> Vec<usize> {
        let fd = self.config.feat_dim;
        let mut order: Vec<usize> = (0..n_nodes).collect();
        order.sort_by(|&a, &b| {
            // Primary key: last channel, then channels fd-2 .. 0, all descending.
            for c in (0..fd).rev() {
                let va = x[a * fd + c];
                let vb = x[b * fd + c];
                match vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            // Final tie-break: ascending node index for determinism.
            a.cmp(&b)
        });
        order
    }

    /// Apply SortPooling, returning a fixed `[k × feat_dim]` row-major matrix.
    ///
    /// # Arguments
    ///
    /// * `x`: `[n_nodes × feat_dim]` node features.
    /// * `n_nodes`: node count of the graph.
    ///
    /// # Errors
    ///
    /// [`GnnError::EmptyGraph`] if `n_nodes == 0`, and
    /// [`GnnError::NodeFeatureMismatch`] if `x.len() != n_nodes * feat_dim`.
    pub fn forward(&self, x: &[f32], n_nodes: usize) -> GnnResult<Vec<f32>> {
        let fd = self.config.feat_dim;
        let k = self.config.k;
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if x.len() != n_nodes * fd {
            return Err(GnnError::NodeFeatureMismatch(n_nodes, x.len() / fd.max(1)));
        }

        let order = self.sorted_order(x, n_nodes);
        let mut out = vec![0.0_f32; k * fd]; // zero-padded by construction
        let take = k.min(n_nodes);
        for (row, &node) in order.iter().take(take).enumerate() {
            for c in 0..fd {
                out[row * fd + c] = x[node * fd + c];
            }
        }
        Ok(out)
    }

    /// Output length of the flattened representation (`k * feat_dim`).
    pub fn output_len(&self) -> usize {
        self.config.k * self.config.feat_dim
    }

    /// Retained-row count `k`.
    pub fn k(&self) -> usize {
        self.config.k
    }

    /// Feature dimension.
    pub fn feat_dim(&self) -> usize {
        self.config.feat_dim
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(feat_dim: usize, k: usize) -> SortPool {
        SortPool::new(SortPoolConfig { feat_dim, k }).expect("test invariant: value must be valid")
    }

    #[test]
    fn build_and_accessors() {
        let p = pool(3, 4);
        assert_eq!(p.feat_dim(), 3);
        assert_eq!(p.k(), 4);
        assert_eq!(p.output_len(), 12);
    }

    #[test]
    fn zero_config_errors() {
        assert!(SortPool::new(SortPoolConfig { feat_dim: 0, k: 4 }).is_err());
        assert!(SortPool::new(SortPoolConfig { feat_dim: 3, k: 0 }).is_err());
    }

    #[test]
    fn output_shape_is_k_times_d() {
        let p = pool(2, 3);
        let x = vec![1.0_f32; 5 * 2];
        let out = p.forward(&x, 5).expect("forward");
        assert_eq!(out.len(), 3 * 2);
    }

    #[test]
    fn sorts_descending_by_last_channel() {
        // single channel; nodes [3, 1, 2] should sort to [3, 2, 1].
        let p = pool(1, 3);
        let x = vec![3.0_f32, 1.0, 2.0];
        let out = p.forward(&x, 3).expect("forward");
        assert_eq!(out, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn truncates_to_k_when_more_nodes() {
        // 4 nodes, k=2 → keep two largest by last channel.
        let p = pool(1, 2);
        let x = vec![10.0_f32, 40.0, 20.0, 30.0];
        let out = p.forward(&x, 4).expect("forward");
        assert_eq!(out, vec![40.0, 30.0]);
    }

    #[test]
    fn zero_pads_when_fewer_nodes() {
        // 2 nodes, k=4 → two rows then two zero rows.
        let p = pool(2, 4);
        // node0 = [1,5], node1 = [2,9]; last channel 5 vs 9 → node1 first
        let x = vec![1.0_f32, 5.0, 2.0, 9.0];
        let out = p.forward(&x, 2).expect("forward");
        assert_eq!(out.len(), 8);
        assert_eq!(&out[0..2], &[2.0, 9.0]); // node1
        assert_eq!(&out[2..4], &[1.0, 5.0]); // node0
        assert!(
            out[4..].iter().all(|&v| v == 0.0),
            "tail must be zero-padded"
        );
    }

    #[test]
    fn last_channel_is_primary_key() {
        // Two channels; ordering must follow the LAST channel, not the first.
        let p = pool(2, 2);
        // node0 = [9, 1], node1 = [0, 8]; last channel 1 vs 8 → node1 first
        let x = vec![9.0_f32, 1.0, 0.0, 8.0];
        let out = p.forward(&x, 2).expect("forward");
        assert_eq!(&out[0..2], &[0.0, 8.0], "node1 (last=8) must come first");
        assert_eq!(&out[2..4], &[9.0, 1.0]);
    }

    #[test]
    fn ties_broken_by_earlier_channel() {
        // Equal last channel → compare previous channel (descending).
        let p = pool(2, 2);
        // node0 = [1, 5], node1 = [9, 5]; last equal (5), prev 1 vs 9 → node1 first
        let x = vec![1.0_f32, 5.0, 9.0, 5.0];
        let out = p.forward(&x, 2).expect("forward");
        assert_eq!(&out[0..2], &[9.0, 5.0]);
        assert_eq!(&out[2..4], &[1.0, 5.0]);
    }

    #[test]
    fn full_tie_broken_by_node_index() {
        // Identical rows → deterministic ascending node-index order.
        let p = pool(1, 3);
        let x = vec![7.0_f32, 7.0, 7.0];
        let out = p.forward(&x, 3).expect("forward");
        assert_eq!(out, vec![7.0, 7.0, 7.0]); // stable, all equal
    }

    #[test]
    fn empty_graph_errors() {
        let p = pool(2, 3);
        assert!(matches!(p.forward(&[], 0), Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn feature_mismatch_errors() {
        let p = pool(3, 2);
        let err = p.forward(&[1.0_f32; 7], 3); // 7 != 3*3
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn exact_k_nodes_no_padding() {
        let p = pool(2, 3);
        let x: Vec<f32> = (0..3 * 2).map(|i| i as f32).collect();
        let out = p.forward(&x, 3).expect("forward");
        // all rows populated, none are entirely zero unless input was zero
        assert_eq!(out.len(), 6);
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn permutation_invariant() {
        // Same node set in two orders must yield identical SortPool output.
        let p = pool(2, 3);
        let x1 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // nodes A,B,C
        let x2 = vec![5.0_f32, 6.0, 1.0, 2.0, 3.0, 4.0]; // nodes C,A,B
        let o1 = p.forward(&x1, 3).expect("forward");
        let o2 = p.forward(&x2, 3).expect("forward");
        assert_eq!(o1, o2, "SortPool must be permutation invariant");
    }

    #[test]
    fn output_finite_and_negative_values() {
        let p = pool(3, 4);
        let x: Vec<f32> = (0..5 * 3).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let out = p.forward(&x, 5).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
