//! DARTS network: stacked cells with global average pooling and a linear classifier.

use crate::darts::cell::DartsCell;
use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::primitives::OpWeights;

// ─── DartsNetwork ────────────────────────────────────────────────────────────

/// A stacked DARTS network.
///
/// Architecture: `[n_normal + n_reduction]` cells interleaved according to
/// `reduction_at` positions, followed by global average pooling and a linear
/// classifier head.
#[derive(Debug, Clone)]
pub struct DartsNetwork {
    /// All cells in order (normal or reduction).
    pub cells: Vec<DartsCell>,
    /// Which cell indices are reduction cells.
    pub reduction_at: Vec<usize>,
    /// Number of output classes.
    pub n_classes: usize,
    /// Number of channels in each cell.
    pub in_ch: usize,
    /// Classifier weights: `[n_classes * (n_nodes * in_ch)]`.
    pub classifier: Vec<f32>,
}

impl DartsNetwork {
    /// Build a DARTS network with the given layer count and reduction positions.
    #[must_use]
    pub fn new(
        n_layers: usize,
        in_ch: usize,
        n_nodes: usize,
        n_classes: usize,
        reduction_at: Vec<usize>,
        rng: &mut LcgRng,
    ) -> Self {
        let mut cells = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let is_reduction = reduction_at.contains(&i);
            cells.push(DartsCell::new(n_nodes, in_ch, is_reduction, rng));
        }
        // Classifier maps from pooled feature dim to n_classes
        // After concatenation of n_nodes node outputs: dim = n_nodes * in_ch
        let feat_dim = n_nodes * in_ch;
        let mut classifier = vec![0.0_f32; n_classes * feat_dim];
        rng.fill_normal(&mut classifier);
        classifier.iter_mut().for_each(|v| *v *= 0.01);

        Self {
            cells,
            reduction_at,
            n_classes,
            in_ch,
            classifier,
        }
    }

    /// Forward pass through all cells, then global average pool, then linear.
    ///
    /// # Arguments
    /// * `input` — `[in_ch * H * W]`
    /// * `h`, `w` — spatial dimensions
    /// * `all_op_weights` — `[n_cells][n_edges][n_ops]` weight tensors
    pub fn forward_cpu(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        all_op_weights: &[Vec<Vec<OpWeights>>],
    ) -> NasResult<Vec<f32>> {
        if all_op_weights.len() != self.cells.len() {
            return Err(NasError::DimensionMismatch {
                expected: self.cells.len(),
                got: all_op_weights.len(),
            });
        }

        // s0 = s1 = input (both inputs to the first cell are the stem output)
        let mut s0 = input.to_vec();
        let mut s1 = input.to_vec();

        for (i, cell) in self.cells.iter().enumerate() {
            let cell_out = cell.forward_cpu(
                &[s0.clone(), s1.clone()],
                self.in_ch,
                h,
                w,
                &all_op_weights[i],
            )?;
            s0 = s1;
            s1 = cell_out;
        }

        // s1 is now [n_nodes * in_ch * H * W]
        // Global average pooling: average over H*W for each channel
        let n_nodes = self.cells.first().map(|c| c.n_nodes).unwrap_or(4);
        let feat_dim = n_nodes * self.in_ch;
        let spatial = h * w;

        let mut pooled = vec![0.0_f32; feat_dim];
        for (c, p) in pooled.iter_mut().enumerate() {
            let start = c * spatial;
            let end = start + spatial;
            if end <= s1.len() {
                *p = s1[start..end].iter().sum::<f32>() / spatial as f32;
            }
        }

        // Linear classifier
        let mut logits = vec![0.0_f32; self.n_classes];
        for (cls, logit) in logits.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (f, &pf) in pooled.iter().enumerate() {
                let w_idx = cls * feat_dim + f;
                acc += self.classifier.get(w_idx).copied().unwrap_or(0.0) * pf;
            }
            *logit = acc;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn darts_network_construction() {
        let mut rng = LcgRng::new(7);
        let net = DartsNetwork::new(4, 8, 4, 10, vec![1, 3], &mut rng);
        assert_eq!(net.cells.len(), 4);
        assert_eq!(net.n_classes, 10);
    }
}
