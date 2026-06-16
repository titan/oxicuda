//! Dynamic Graph CNN (DGCNN) EdgeConv layer.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn linear_relu(input: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut acc = b[i];
        for j in 0..in_dim {
            acc += w[i * in_dim + j] * input[j];
        }
        out[i] = acc.max(0.0);
    }
    out
}

fn sq_dist(a: &[f32], ai: usize, b: &[f32], bi: usize, c: usize) -> f32 {
    let mut d = 0.0_f32;
    for ch in 0..c {
        let diff = a[ai * c + ch] - b[bi * c + ch];
        d += diff * diff;
    }
    d
}

// ─── EdgeConv ────────────────────────────────────────────────────────────────

/// Configuration for an EdgeConv layer.
#[derive(Debug, Clone)]
pub struct EdgeConvConfig {
    pub k: usize,
    pub mlp_channels: Vec<usize>,
}

/// EdgeConv layer from DGCNN.
///
/// For each point i, finds k nearest in feature space, computes edge features
/// `concat(feat_i, feat_j - feat_i)` for each neighbor j, applies MLP,
/// then max-pools across k neighbors.
pub struct EdgeConv {
    config: EdgeConvConfig,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
}

impl EdgeConv {
    /// Create a new EdgeConv layer.
    pub fn new(config: EdgeConvConfig, in_channels: usize, rng: &mut LcgRng) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Edge feature dimension: 2 * in_channels (concat of feat_i and feat_j - feat_i)
        let mut prev = 2 * in_channels;
        for &out_ch in &config.mlp_channels {
            let mut w = vec![0.0_f32; out_ch * prev];
            rng.fill_xavier_uniform(&mut w, prev, out_ch);
            weights.push(w);
            biases.push(vec![0.0_f32; out_ch]);
            prev = out_ch;
        }

        Self {
            config,
            weights,
            biases,
        }
    }

    /// Forward pass: `features [n×c_in]` → `[n×c_out]`.
    ///
    /// Builds dynamic kNN graph in feature space.
    pub fn forward(&self, features: &[f32], n: usize, c_in: usize) -> Geom3dResult<Vec<f32>> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if features.len() != n * c_in {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * c_in,
                got: features.len(),
            });
        }
        let k = self.config.k.min(n);
        let c_out = self.config.mlp_channels.last().copied().unwrap_or(c_in);

        let mut out_feat = vec![0.0_f32; n * c_out];

        for i in 0..n {
            // Find k nearest in feature space
            let mut dists: Vec<(f32, usize)> = (0..n)
                .map(|j| (sq_dist(features, i, features, j, c_in), j))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Skip self (idx 0) and take k neighbors
            let neighbors: Vec<usize> = dists.iter().skip(1).take(k).map(|&(_, j)| j).collect();
            let actual_k = neighbors.len();

            let edge_dim = 2 * c_in;
            let mut pooled = vec![f32::NEG_INFINITY; c_out];

            for &j in &neighbors {
                // Edge feature: concat(feat_i, feat_j - feat_i)
                let mut edge = vec![0.0_f32; edge_dim];
                for ch in 0..c_in {
                    edge[ch] = features[i * c_in + ch];
                    edge[c_in + ch] = features[j * c_in + ch] - features[i * c_in + ch];
                }

                // Apply MLP
                let mut h = edge;
                let mut cur_in = edge_dim;
                for (layer_idx, &out_ch) in self.config.mlp_channels.iter().enumerate() {
                    h = linear_relu(
                        &h,
                        &self.weights[layer_idx],
                        &self.biases[layer_idx],
                        cur_in,
                        out_ch,
                    );
                    cur_in = out_ch;
                }

                // Max pool
                for ch in 0..c_out {
                    if h[ch] > pooled[ch] {
                        pooled[ch] = h[ch];
                    }
                }
            }

            // Handle case where no neighbors (shouldn't happen given k.min(n))
            if actual_k == 0 {
                for v in &mut pooled {
                    if *v == f32::NEG_INFINITY {
                        *v = 0.0;
                    }
                }
            }

            // Replace any remaining -inf with 0
            for v in &mut pooled {
                if *v == f32::NEG_INFINITY {
                    *v = 0.0;
                }
            }

            out_feat[i * c_out..(i + 1) * c_out].copy_from_slice(&pooled);
        }

        Ok(out_feat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edgeconv_output_shape() {
        let n = 16;
        let c_in = 3;
        let c_out = 16;
        let mut rng = LcgRng::new(42);
        let cfg = EdgeConvConfig {
            k: 4,
            mlp_channels: vec![8, c_out],
        };
        let ec = EdgeConv::new(cfg, c_in, &mut rng);
        let feat: Vec<f32> = (0..n * c_in).map(|i| i as f32 * 0.01).collect();
        let out = ec.forward(&feat, n, c_in).expect("forward should succeed");
        assert_eq!(out.len(), n * c_out);
    }

    #[test]
    fn edgeconv_finite_output() {
        let n = 8;
        let c_in = 4;
        let mut rng = LcgRng::new(42);
        let cfg = EdgeConvConfig {
            k: 3,
            mlp_channels: vec![8],
        };
        let ec = EdgeConv::new(cfg, c_in, &mut rng);
        let mut feat_rng = LcgRng::new(99);
        let mut feat = vec![0.0_f32; n * c_in];
        feat_rng.fill_normal(&mut feat);
        let out = ec.forward(&feat, n, c_in).expect("forward should succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "EdgeConv output must be finite"
        );
    }

    #[test]
    fn edgeconv_empty_error() {
        let mut rng = LcgRng::new(0);
        let cfg = EdgeConvConfig {
            k: 4,
            mlp_channels: vec![8],
        };
        let ec = EdgeConv::new(cfg, 3, &mut rng);
        assert!(ec.forward(&[], 0, 3).is_err());
    }
}
