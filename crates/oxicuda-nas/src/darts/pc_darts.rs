//! PC-DARTS: Partial-Channel connections for memory-efficient DARTS.
//!
//! Reference: Xu, Xie, Zhang, Chen, Qi, Tian & Xiong, "PC-DARTS: Partial
//! Channel Connections for Memory-Efficient Architecture Search", ICLR 2020.
//!
//! PC-DARTS adds two ideas on top of vanilla DARTS:
//!
//! 1. **Partial-channel sampling.** Instead of sending *all* channels through
//!    the (memory-hungry) candidate operations, only a random fraction `1/K`
//!    of the channels pass through the mixed op each step; the remaining
//!    `(K-1)/K` channels bypass via an identity shortcut and are concatenated
//!    back. This cuts the search-time memory/compute of the operation mixture
//!    by roughly `K×`.
//! 2. **Edge normalization.** Partial sampling makes the per-edge operation
//!    choice noisy from step to step. To stabilise it, PC-DARTS adds an extra
//!    set of learnable per-edge weights `β`. For every destination node, the
//!    `β` of its incoming edges are softmax-normalised and multiply the
//!    op-mixed edge outputs, so the network learns *which edges matter* in a
//!    way that is decoupled from the noisy per-step channel sampling.
//!
//! # Topology
//!
//! This mirrors the DARTS cell topology: a cell has 2 input nodes and
//! `n_nodes` intermediate nodes. Intermediate node `i` (0-indexed) receives
//! one edge from each of the `2 + i` preceding nodes (the 2 cell inputs plus
//! all earlier intermediate nodes). Hence the total edge count is
//!
//! ```text
//! n_edges = Σ_{i=0}^{n_nodes-1} (2 + i)
//! ```
//!
//! # Candidate-op set (self-contained)
//!
//! `DartsCell`/`MixedOp` operate on full `[C·H·W]` spatial feature maps with
//! per-op weight tensors and cannot cleanly accept a *channel subset* of
//! scalar node values, so PC-DARTS uses its own minimal, fully-connected
//! candidate-op set applied per channel to the *selected* channels only:
//!
//! | idx | op       | definition (on masked channel value `x_c`)            |
//! |-----|----------|-------------------------------------------------------|
//! | 0   | identity | `x_c`                                                 |
//! | 1   | zero     | `0`                                                   |
//! | 2   | scale    | `s · x_c` (per-edge learned scalar `s`)               |
//! | 3   | average  | mean of all selected channel values (broadcast)       |
//!
//! `n_ops` controls how many of these (cyclically) are mixed per edge; the
//! op-mixing weights come from softmax over the per-edge `alpha` logits.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::mixed_op::softmax;

// ─── PcDartsConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`PcDarts`] cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcDartsConfig {
    /// Number of intermediate nodes. Must be `>= 1`.
    pub n_nodes: usize,
    /// Number of channels per node. Must be `>= 1`.
    pub n_channels: usize,
    /// Partial-channel divisor `K`: exactly `floor(n_channels / K)` channels
    /// are routed through the ops, the rest bypass via identity. Must satisfy
    /// `1 <= partial_k <= n_channels` (`K == 1` ⇒ all channels go through ops).
    pub partial_k: usize,
    /// Number of candidate ops mixed per edge. Must be `>= 1`.
    pub n_ops: usize,
    /// Whether this models a reduction cell (recorded only; topology is shared).
    pub reduction: bool,
}

impl PcDartsConfig {
    fn validate(&self) -> NasResult<()> {
        if self.n_nodes == 0 {
            return Err(NasError::InvalidNumNodes { min: 1, got: 0 });
        }
        if self.n_channels == 0 {
            return Err(NasError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if self.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        if self.partial_k == 0 || self.partial_k > self.n_channels {
            return Err(NasError::InvalidWeightShape);
        }
        Ok(())
    }
}

// ─── PcDarts ───────────────────────────────────────────────────────────────────

/// A PC-DARTS searchable cell with partial-channel sampling + edge
/// normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct PcDarts {
    /// Per-edge op logits, row-major `[n_edges][n_ops]` flattened.
    alpha: Vec<f32>,
    /// Per-edge edge-normalization weights `β`, length `n_edges`.
    beta: Vec<f32>,
    /// Per-edge learned scale for the "scale" candidate op, length `n_edges`.
    edge_scale: Vec<f32>,
    /// Cached configuration.
    config: PcDartsConfig,
    /// Cached edge count `Σ (2 + i)`.
    n_edges: usize,
}

impl PcDarts {
    /// Construct a PC-DARTS cell, initialising `alpha`/`beta`/`scale` from the
    /// supplied RNG (`N(0, 0.01)` for logits, `N(1, 0.01)` for the scale).
    ///
    /// # Errors
    /// Returns [`NasError`] if the config is out of range (see
    /// [`PcDartsConfig`]).
    pub fn new(cfg: PcDartsConfig, rng: &mut LcgRng) -> NasResult<Self> {
        cfg.validate()?;
        let n_edges = (0..cfg.n_nodes).map(|i| 2 + i).sum::<usize>();

        let mut alpha = vec![0.0_f32; n_edges * cfg.n_ops];
        rng.fill_normal(&mut alpha);
        alpha.iter_mut().for_each(|v| *v *= 0.01);

        let mut beta = vec![0.0_f32; n_edges];
        rng.fill_normal(&mut beta);
        beta.iter_mut().for_each(|v| *v *= 0.01);

        let mut edge_scale = vec![0.0_f32; n_edges];
        rng.fill_normal(&mut edge_scale);
        // Centre the scale around 1.0 so "scale" starts near identity.
        edge_scale.iter_mut().for_each(|v| *v = 1.0 + *v * 0.01);

        Ok(Self {
            alpha,
            beta,
            edge_scale,
            config: cfg,
            n_edges,
        })
    }

    /// Number of edges in the cell: `Σ_{i=0}^{n_nodes-1} (2 + i)` (DARTS
    /// topology — each intermediate node connects to the 2 cell inputs plus
    /// every preceding intermediate node).
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// Read-only view of the configuration.
    #[must_use]
    pub fn config(&self) -> &PcDartsConfig {
        &self.config
    }

    /// Number of channels routed through the candidate ops:
    /// `floor(n_channels / partial_k)`.
    #[must_use]
    fn n_selected(&self) -> usize {
        self.config.n_channels / self.config.partial_k
    }

    /// Sample a boolean channel mask of length `n_channels` with exactly
    /// `floor(n_channels / partial_k)` entries set to `true` (those channels
    /// are routed through the ops; the rest bypass via identity).
    ///
    /// Distinct channel indices are chosen via a partial Fisher-Yates shuffle.
    /// `partial_k == 1` selects every channel.
    ///
    /// # Errors
    /// Infallible for a validly-constructed [`PcDarts`]; returns [`NasError`]
    /// only on internal inconsistency.
    pub fn sample_channel_mask(&self, rng: &mut LcgRng) -> NasResult<Vec<bool>> {
        let n = self.config.n_channels;
        let keep = self.n_selected();
        let mut mask = vec![false; n];

        // Partial Fisher-Yates: select `keep` distinct indices uniformly.
        let mut perm: Vec<usize> = (0..n).collect();
        for i in 0..keep {
            let j = i + rng.next_usize(n - i);
            perm.swap(i, j);
            match mask.get_mut(perm[i]) {
                Some(slot) => *slot = true,
                None => return Err(NasError::Internal("channel index out of range".into())),
            }
        }
        Ok(mask)
    }

    /// Per-edge op-mixing weights: softmax over `alpha[edge]` for each edge.
    /// Returns a flattened `[n_edges][n_ops]` vector; each length-`n_ops` row
    /// sums to 1.
    ///
    /// # Errors
    /// Returns [`NasError`] only on internal inconsistency.
    pub fn op_weights(&self) -> NasResult<Vec<f32>> {
        let n_ops = self.config.n_ops;
        let mut out = Vec::with_capacity(self.n_edges * n_ops);
        for e in 0..self.n_edges {
            let start = e * n_ops;
            let row = self
                .alpha
                .get(start..start + n_ops)
                .ok_or_else(|| NasError::Internal("alpha row out of range".into()))?;
            out.extend(softmax(row));
        }
        Ok(out)
    }

    /// Edge-normalized weights `β`: for each destination node, softmax over
    /// that node's incoming edges' `β`. Returns a flattened length-`n_edges`
    /// vector where every per-destination-node group sums to 1.
    ///
    /// Edges are laid out by destination node in increasing order, so node `i`
    /// owns the contiguous block of its `2 + i` incoming edges.
    ///
    /// # Errors
    /// Returns [`NasError`] only on internal inconsistency.
    pub fn edge_normalized_weights(&self) -> NasResult<Vec<f32>> {
        let mut out = Vec::with_capacity(self.n_edges);
        let mut base = 0usize;
        for i in 0..self.config.n_nodes {
            let n_in = 2 + i;
            let group = self
                .beta
                .get(base..base + n_in)
                .ok_or_else(|| NasError::Internal("beta group out of range".into()))?;
            out.extend(softmax(group));
            base += n_in;
        }
        Ok(out)
    }

    /// Apply candidate op `op_idx` (cyclic over the 4-op set) to the masked
    /// channel value `x`, given the per-edge scale and the mean of selected
    /// channel values.
    #[inline]
    fn apply_op(op_idx: usize, x: f32, scale: f32, sel_mean: f32) -> f32 {
        match op_idx % 4 {
            0 => x,         // identity
            1 => 0.0,       // zero
            2 => scale * x, // learned scale
            _ => sel_mean,  // average (broadcast mean of selected channels)
        }
    }

    /// Forward one cell.
    ///
    /// `inputs` is the concatenation of the two input nodes, each of length
    /// `n_channels` (so `inputs.len() == 2 * n_channels`). For each
    /// intermediate node, every incoming edge:
    ///
    /// * routes the *selected* channels (per the freshly sampled mask) through
    ///   the candidate-op mixture (weighted by the op softmax), and
    /// * passes the *unselected* channels through unchanged (identity bypass);
    ///
    /// then the edge output is scaled by the edge-normalized `β` for that
    /// destination node and summed across incoming edges. The result is the
    /// channel vector of the **last** intermediate node (length `n_channels`).
    ///
    /// # Errors
    /// Returns [`NasError::DimensionMismatch`] if `inputs.len() != 2 *
    /// n_channels`.
    pub fn forward(&self, inputs: &[f32], rng: &mut LcgRng) -> NasResult<Vec<f32>> {
        let c = self.config.n_channels;
        if inputs.len() != 2 * c {
            return Err(NasError::DimensionMismatch {
                expected: 2 * c,
                got: inputs.len(),
            });
        }

        let op_w = self.op_weights()?;
        let edge_w = self.edge_normalized_weights()?;
        let n_ops = self.config.n_ops;

        // Node value store: index 0,1 = inputs; 2.. = intermediates.
        let mut nodes: Vec<Vec<f32>> = Vec::with_capacity(self.config.n_nodes + 2);
        nodes.push(inputs[..c].to_vec());
        nodes.push(inputs[c..2 * c].to_vec());

        let mut edge_idx = 0usize;
        for i in 0..self.config.n_nodes {
            let n_in = 2 + i;
            let mut node_val = vec![0.0_f32; c];

            for src in 0..n_in {
                // Fresh partial-channel mask per edge (PC-DARTS samples per op).
                let mask = self.sample_channel_mask(rng)?;
                let src_val = nodes
                    .get(src)
                    .ok_or_else(|| NasError::Internal("source node missing".into()))?;

                // Mean over selected channels for the "average" op.
                let mut sel_sum = 0.0_f32;
                let mut sel_cnt = 0u32;
                for (ch, &m) in mask.iter().enumerate() {
                    if m {
                        sel_sum += src_val[ch];
                        sel_cnt += 1;
                    }
                }
                let sel_mean = if sel_cnt > 0 {
                    sel_sum / sel_cnt as f32
                } else {
                    0.0
                };

                let scale = self.edge_scale[edge_idx];
                let beta = edge_w[edge_idx];
                let op_row = &op_w[edge_idx * n_ops..edge_idx * n_ops + n_ops];

                // Per-channel edge output.
                let mut edge_out = vec![0.0_f32; c];
                for (ch, slot) in edge_out.iter_mut().enumerate() {
                    if mask[ch] {
                        // Selected: mixture over the candidate ops.
                        let mut mixed = 0.0_f32;
                        for (op_idx, &w) in op_row.iter().enumerate() {
                            mixed += w * Self::apply_op(op_idx, src_val[ch], scale, sel_mean);
                        }
                        *slot = mixed;
                    } else {
                        // Unselected: identity bypass (passed through unchanged).
                        *slot = src_val[ch];
                    }
                }

                // Edge-normalized accumulation into the node value.
                for (acc, &ev) in node_val.iter_mut().zip(edge_out.iter()) {
                    *acc += beta * ev;
                }
                edge_idx += 1;
            }
            nodes.push(node_val);
        }

        // Output = the last intermediate node's channels.
        nodes
            .pop()
            .ok_or_else(|| NasError::Internal("no intermediate node produced".into()))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_nodes: usize, n_channels: usize, partial_k: usize, n_ops: usize) -> PcDartsConfig {
        PcDartsConfig {
            n_nodes,
            n_channels,
            partial_k,
            n_ops,
            reduction: false,
        }
    }

    fn make(n_nodes: usize, n_channels: usize, partial_k: usize, n_ops: usize) -> PcDarts {
        let mut rng = LcgRng::new(42);
        PcDarts::new(cfg(n_nodes, n_channels, partial_k, n_ops), &mut rng)
            .expect("test invariant: construct")
    }

    #[test]
    fn channel_mask_length_and_count() {
        let pc = make(4, 16, 4, 3);
        let mut rng = LcgRng::new(7);
        let mask = pc.sample_channel_mask(&mut rng).expect("mask");
        assert_eq!(mask.len(), 16);
        assert_eq!(mask.iter().filter(|&&b| b).count(), 16 / 4);
    }

    #[test]
    fn channel_mask_partial_k_one_all_true() {
        let pc = make(3, 12, 1, 3);
        let mut rng = LcgRng::new(7);
        let mask = pc.sample_channel_mask(&mut rng).expect("mask");
        assert_eq!(mask.len(), 12);
        assert!(mask.iter().all(|&b| b));
    }

    #[test]
    fn channel_mask_fraction_k2() {
        let pc = make(3, 16, 2, 3);
        let mut rng = LcgRng::new(11);
        let mask = pc.sample_channel_mask(&mut rng).expect("mask");
        assert_eq!(mask.iter().filter(|&&b| b).count(), 8);
    }

    #[test]
    fn channel_mask_fraction_k4() {
        let pc = make(3, 20, 4, 3);
        let mut rng = LcgRng::new(11);
        let mask = pc.sample_channel_mask(&mut rng).expect("mask");
        assert_eq!(mask.iter().filter(|&&b| b).count(), 5);
    }

    #[test]
    fn channel_mask_floor_division() {
        // 17 / 4 = 4 (floor).
        let pc = make(2, 17, 4, 3);
        let mut rng = LcgRng::new(3);
        let mask = pc.sample_channel_mask(&mut rng).expect("mask");
        assert_eq!(mask.iter().filter(|&&b| b).count(), 4);
    }

    #[test]
    fn op_weights_rows_sum_to_one() {
        let pc = make(4, 16, 4, 5);
        let w = pc.op_weights().expect("op weights");
        let n_ops = pc.config().n_ops;
        for e in 0..pc.n_edges() {
            let s: f32 = w[e * n_ops..e * n_ops + n_ops].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {e} sum = {s}");
        }
    }

    #[test]
    fn op_weights_length() {
        let pc = make(4, 16, 4, 5);
        let w = pc.op_weights().expect("op weights");
        assert_eq!(w.len(), pc.n_edges() * 5);
    }

    #[test]
    fn op_weights_non_negative() {
        let pc = make(4, 16, 4, 5);
        let w = pc.op_weights().expect("op weights");
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn edge_normalized_groups_sum_to_one() {
        let pc = make(4, 16, 4, 3);
        let w = pc.edge_normalized_weights().expect("edge weights");
        let mut base = 0usize;
        for i in 0..pc.config().n_nodes {
            let n_in = 2 + i;
            let s: f32 = w[base..base + n_in].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "group {i} sum = {s}");
            base += n_in;
        }
    }

    #[test]
    fn edge_normalized_length() {
        let pc = make(4, 16, 4, 3);
        let w = pc.edge_normalized_weights().expect("edge weights");
        assert_eq!(w.len(), pc.n_edges());
    }

    #[test]
    fn n_edges_formula_matches() {
        // n_nodes=4 → 2+3+4+5 = 14
        assert_eq!(make(4, 8, 2, 3).n_edges(), 14);
        // n_nodes=2 → 2+3 = 5
        assert_eq!(make(2, 8, 2, 3).n_edges(), 5);
        // n_nodes=1 → 2
        assert_eq!(make(1, 8, 2, 3).n_edges(), 2);
        // n_nodes=5 → 2+3+4+5+6 = 20
        assert_eq!(make(5, 8, 2, 3).n_edges(), 20);
    }

    #[test]
    fn forward_output_length() {
        let pc = make(4, 16, 4, 4);
        let mut rng = LcgRng::new(5);
        let inputs = vec![0.5_f32; 32];
        let out = pc.forward(&inputs, &mut rng).expect("forward");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn forward_finite() {
        let pc = make(4, 16, 4, 4);
        let mut rng = LcgRng::new(5);
        let inputs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let out = pc.forward(&inputs, &mut rng).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_deterministic_given_rng() {
        let pc = make(4, 16, 4, 4);
        let inputs = vec![0.3_f32; 32];
        let mut rng_a = LcgRng::new(77);
        let mut rng_b = LcgRng::new(77);
        let a = pc.forward(&inputs, &mut rng_a).expect("forward a");
        let b = pc.forward(&inputs, &mut rng_b).expect("forward b");
        assert_eq!(a, b);
    }

    #[test]
    fn forward_partial_k_one_finite() {
        // With K=1 all channels go through ops; still must produce finite out.
        let pc = make(3, 12, 1, 4);
        let mut rng = LcgRng::new(9);
        let inputs = vec![0.7_f32; 24];
        let out = pc.forward(&inputs, &mut rng).expect("forward");
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn different_seeds_give_different_masks() {
        let pc = make(3, 16, 2, 3);
        let mut rng_a = LcgRng::new(1);
        let mut rng_b = LcgRng::new(2);
        let ma = pc.sample_channel_mask(&mut rng_a).expect("mask a");
        let mb = pc.sample_channel_mask(&mut rng_b).expect("mask b");
        // Same count, but the selected indices should differ (high prob.).
        assert_ne!(ma, mb);
    }

    #[test]
    fn err_n_nodes_zero() {
        let mut rng = LcgRng::new(1);
        assert_eq!(
            PcDarts::new(cfg(0, 16, 4, 3), &mut rng),
            Err(NasError::InvalidNumNodes { min: 1, got: 0 })
        );
    }

    #[test]
    fn err_n_channels_zero() {
        let mut rng = LcgRng::new(1);
        assert_eq!(
            PcDarts::new(cfg(4, 0, 1, 3), &mut rng),
            Err(NasError::DimensionMismatch {
                expected: 1,
                got: 0
            })
        );
    }

    #[test]
    fn err_partial_k_zero() {
        let mut rng = LcgRng::new(1);
        assert_eq!(
            PcDarts::new(cfg(4, 16, 0, 3), &mut rng),
            Err(NasError::InvalidWeightShape)
        );
    }

    #[test]
    fn err_partial_k_exceeds_channels() {
        let mut rng = LcgRng::new(1);
        assert_eq!(
            PcDarts::new(cfg(4, 16, 17, 3), &mut rng),
            Err(NasError::InvalidWeightShape)
        );
    }

    #[test]
    fn err_n_ops_zero() {
        let mut rng = LcgRng::new(1);
        assert_eq!(
            PcDarts::new(cfg(4, 16, 4, 0), &mut rng),
            Err(NasError::InvalidNumOps)
        );
    }

    #[test]
    fn err_forward_wrong_input_length() {
        let pc = make(4, 16, 4, 3);
        let mut rng = LcgRng::new(1);
        let inputs = vec![0.0_f32; 30]; // expected 32
        assert_eq!(
            pc.forward(&inputs, &mut rng),
            Err(NasError::DimensionMismatch {
                expected: 32,
                got: 30
            })
        );
    }

    #[test]
    fn partial_k_equals_channels_selects_one() {
        // K == n_channels → floor(n/n) = 1 channel selected.
        let pc = make(2, 8, 8, 3);
        let mut rng = LcgRng::new(4);
        let mask = pc.sample_channel_mask(&mut rng).expect("mask");
        assert_eq!(mask.iter().filter(|&&b| b).count(), 1);
    }
}
