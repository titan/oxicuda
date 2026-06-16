//! NODE: Neural Oblivious Decision Ensembles (Popov et al. 2019).
//!
//! Each oblivious tree has `depth` levels; all nodes at a given level share
//! the same (feature, threshold) split.  The soft version replaces hard
//! binary decisions with sigmoid-smoothed probabilities and uses entmax-1.5
//! for sparse feature selection.

use crate::attention::sparsemax::entmax15;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── NodeConfig ───────────────────────────────────────────────────────────────

/// Configuration for a `NodeEnsemble`.
pub struct NodeConfig {
    /// Number of trees in the ensemble.
    pub n_trees: usize,
    /// Depth of each tree (2–8).
    pub depth: usize,
    /// Number of input features.
    pub input_dim: usize,
    /// Output dimension (1 for regression, `n_classes` for classification).
    pub output_dim: usize,
}

// ─── NodeTree ─────────────────────────────────────────────────────────────────

/// A single soft oblivious decision tree.
pub struct NodeTree {
    // Feature selection logits: [depth * input_dim]
    feature_logits: Vec<f32>,
    // Threshold per level: [depth]
    thresholds: Vec<f32>,
    // Leaf values: [2^depth * output_dim]
    leaf_values: Vec<f32>,
    // Sigmoid sharpness
    beta: f32,
    depth: usize,
    input_dim: usize,
    output_dim: usize,
}

impl NodeTree {
    /// Construct a new `NodeTree` with small random initialisations.
    pub fn new(
        depth: usize,
        input_dim: usize,
        output_dim: usize,
        rng: &mut LcgRng,
    ) -> TabularResult<Self> {
        if depth == 0 {
            return Err(TabularError::InvalidTreeDepth { depth: 0 });
        }
        if input_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }

        let n_leaves = 1usize << depth; // 2^depth

        let mut feature_logits = vec![0.0_f32; depth * input_dim];
        rng.fill_normal_scaled(&mut feature_logits, 0.01);

        let mut thresholds = vec![0.0_f32; depth];
        rng.fill_normal_scaled(&mut thresholds, 0.1);

        let mut leaf_values = vec![0.0_f32; n_leaves * output_dim];
        rng.fill_normal_scaled(&mut leaf_values, 0.01);

        Ok(Self {
            feature_logits,
            thresholds,
            leaf_values,
            beta: 1.0,
            depth,
            input_dim,
            output_dim,
        })
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        1 << self.depth
    }

    /// Forward pass: compute ensemble output for a single input vector `x [input_dim]`.
    ///
    /// Returns `[output_dim]`.
    pub fn forward(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        let n_leaves = self.n_leaves();
        let mut leaf_probs = vec![1.0_f32; n_leaves];

        for level in 0..self.depth {
            // Sparse feature selection via entmax-1.5
            let level_logits =
                &self.feature_logits[level * self.input_dim..(level + 1) * self.input_dim];
            let feat_probs = entmax15(level_logits)?;

            // Soft feature value: weighted sum
            let selected_x: f32 = feat_probs
                .iter()
                .zip(x.iter())
                .map(|(&p, &xi)| p * xi)
                .sum();

            // Soft binary split
            let b_i = sigmoid_sharp(self.beta, selected_x - self.thresholds[level]);

            // Update leaf probabilities
            // leaf j goes left (bit level = 0) with (1 - b_i), right (bit = 1) with b_i
            for (leaf, lp) in leaf_probs.iter_mut().enumerate() {
                let bit = (leaf >> (self.depth - 1 - level)) & 1;
                let factor = if bit == 1 { b_i } else { 1.0 - b_i };
                *lp *= factor;
            }
        }

        // Output = weighted sum over leaves
        let mut out = vec![0.0_f32; self.output_dim];
        for (leaf, &lp) in leaf_probs.iter().enumerate() {
            let leaf_base = leaf * self.output_dim;
            for (d, ov) in out.iter_mut().enumerate() {
                *ov += lp * self.leaf_values[leaf_base + d];
            }
        }
        Ok(out)
    }
}

#[inline(always)]
fn sigmoid_sharp(beta: f32, x: f32) -> f32 {
    1.0 / (1.0 + (-beta * x).exp())
}

// ─── NodeEnsemble ─────────────────────────────────────────────────────────────

/// An ensemble of soft oblivious decision trees.
pub struct NodeEnsemble {
    trees: Vec<NodeTree>,
    config: NodeConfig,
}

impl NodeEnsemble {
    /// Construct a new `NodeEnsemble` with `cfg.n_trees` trees.
    pub fn new(cfg: NodeConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.n_trees == 0 {
            return Err(TabularError::InvalidTreeCount { n: 0 });
        }
        if cfg.depth == 0 {
            return Err(TabularError::InvalidTreeDepth { depth: 0 });
        }

        let mut trees = Vec::with_capacity(cfg.n_trees);
        for _ in 0..cfg.n_trees {
            trees.push(NodeTree::new(
                cfg.depth,
                cfg.input_dim,
                cfg.output_dim,
                rng,
            )?);
        }
        Ok(Self { trees, config: cfg })
    }

    /// Forward pass for a single sample `x [input_dim]`.
    ///
    /// Returns the mean of tree outputs: `[output_dim]`.
    pub fn forward(&self, x: &[f32]) -> TabularResult<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }
        let mut agg = vec![0.0_f32; self.config.output_dim];
        for tree in &self.trees {
            let out = tree.forward(x)?;
            for (a, &v) in agg.iter_mut().zip(out.iter()) {
                *a += v;
            }
        }
        let n = self.config.n_trees as f32;
        for a in &mut agg {
            *a /= n;
        }
        Ok(agg)
    }

    /// Batch forward: `x` is flat `[batch_size * input_dim]`.
    ///
    /// Returns `[batch_size * output_dim]`.
    pub fn forward_batch(&self, x: &[f32], batch_size: usize) -> TabularResult<Vec<f32>> {
        let in_d = self.config.input_dim;
        if x.len() != batch_size * in_d {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * in_d,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(batch_size * self.config.output_dim);
        for b in 0..batch_size {
            let row = &x[b * in_d..(b + 1) * in_d];
            let pred = self.forward(row)?;
            out.extend_from_slice(&pred);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn node_tree_forward_shape() {
        let mut rng = LcgRng::new(42);
        let tree = NodeTree::new(3, 8, 2, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 8];
        let out = tree.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn node_ensemble_forward_shape() {
        let mut rng = LcgRng::new(7);
        let cfg = NodeConfig {
            n_trees: 5,
            depth: 3,
            input_dim: 8,
            output_dim: 1,
        };
        let ensemble = NodeEnsemble::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.1_f32; 8];
        let out = ensemble.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn node_n_leaves() {
        let mut rng = LcgRng::new(1);
        let tree = NodeTree::new(4, 4, 1, &mut rng).expect("new should succeed");
        assert_eq!(tree.n_leaves(), 16);
    }
}
