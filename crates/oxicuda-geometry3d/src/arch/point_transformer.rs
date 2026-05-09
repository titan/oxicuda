//! Point Transformer layer for 3D point cloud processing.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn linear_no_relu(input: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_d];
    for i in 0..out_d {
        let mut acc = b[i];
        for j in 0..in_d {
            acc += w[i * in_d + j] * input[j];
        }
        out[i] = acc;
    }
    out
}

fn linear_relu(input: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let mut out = linear_no_relu(input, w, b, in_d, out_d);
    for v in &mut out {
        *v = v.max(0.0);
    }
    out
}

/// Softmax over a slice.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / (sum + 1e-12)).collect()
}

fn sq_dist_3d(xyz: &[f32], i: usize, j: usize) -> f32 {
    let dx = xyz[i * 3] - xyz[j * 3];
    let dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
    let dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];
    dx * dx + dy * dy + dz * dz
}

// ─── Point Transformer Layer ─────────────────────────────────────────────────

/// Configuration for a Point Transformer layer.
#[derive(Debug, Clone)]
pub struct PointTransformerConfig {
    pub k: usize,
    pub d_model: usize,
    pub n_heads: usize,
}

/// Point Transformer layer implementing vector self-attention.
///
/// For each point i, finds k neighbors. Computes position encoding
/// `δ_ij = MLP(xyz_i - xyz_j)`. Attention weight
/// `α_ij = softmax_j(γ(φ(feat_i) - ψ(feat_j) + δ_ij))`.
/// Output = `Σ_j α_ij ⊙ (α_linear(feat_j) + δ_ij)`.
pub struct PointTransformerLayer {
    config: PointTransformerConfig,
    // φ: query projection [d_model × d_model]
    phi_w: Vec<f32>,
    phi_b: Vec<f32>,
    // ψ: key projection [d_model × d_model]
    psi_w: Vec<f32>,
    psi_b: Vec<f32>,
    // α: value projection [d_model × d_model]
    alpha_w: Vec<f32>,
    alpha_b: Vec<f32>,
    // γ: relation attention MLP: d_model → d_model → d_model
    gamma_w1: Vec<f32>,
    gamma_b1: Vec<f32>,
    gamma_w2: Vec<f32>,
    gamma_b2: Vec<f32>,
    // Position encoding MLP: 3 → d_model → d_model
    pos_w1: Vec<f32>,
    pos_b1: Vec<f32>,
    pos_w2: Vec<f32>,
    pos_b2: Vec<f32>,
}

impl PointTransformerLayer {
    /// Create a new Point Transformer layer.
    pub fn new(config: PointTransformerConfig, rng: &mut LcgRng) -> Self {
        let d = config.d_model;

        let mut phi_w = vec![0.0_f32; d * d];
        rng.fill_xavier_uniform(&mut phi_w, d, d);
        let phi_b = vec![0.0_f32; d];

        let mut psi_w = vec![0.0_f32; d * d];
        rng.fill_xavier_uniform(&mut psi_w, d, d);
        let psi_b = vec![0.0_f32; d];

        let mut alpha_w = vec![0.0_f32; d * d];
        rng.fill_xavier_uniform(&mut alpha_w, d, d);
        let alpha_b = vec![0.0_f32; d];

        // γ MLP
        let mut gamma_w1 = vec![0.0_f32; d * d];
        rng.fill_xavier_uniform(&mut gamma_w1, d, d);
        let gamma_b1 = vec![0.0_f32; d];
        let mut gamma_w2 = vec![0.0_f32; d * d];
        rng.fill_xavier_uniform(&mut gamma_w2, d, d);
        let gamma_b2 = vec![0.0_f32; d];

        // Position encoding MLP: 3 → d → d
        let mut pos_w1 = vec![0.0_f32; d * 3];
        rng.fill_xavier_uniform(&mut pos_w1, 3, d);
        let pos_b1 = vec![0.0_f32; d];
        let mut pos_w2 = vec![0.0_f32; d * d];
        rng.fill_xavier_uniform(&mut pos_w2, d, d);
        let pos_b2 = vec![0.0_f32; d];

        Self {
            config,
            phi_w,
            phi_b,
            psi_w,
            psi_b,
            alpha_w,
            alpha_b,
            gamma_w1,
            gamma_b1,
            gamma_w2,
            gamma_b2,
            pos_w1,
            pos_b1,
            pos_w2,
            pos_b2,
        }
    }

    /// Forward pass: `xyz [n×3]`, `features [n×d]` → `out [n×d]`.
    pub fn forward(&self, xyz: &[f32], features: &[f32], n: usize) -> Geom3dResult<Vec<f32>> {
        let d = self.config.d_model;
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if xyz.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: xyz.len(),
            });
        }
        if features.len() != n * d {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * d,
                got: features.len(),
            });
        }

        let k = self.config.k.min(n);
        let mut out = vec![0.0_f32; n * d];

        for i in 0..n {
            let feat_i = &features[i * d..(i + 1) * d];

            // φ(feat_i): query
            let q = linear_no_relu(feat_i, &self.phi_w, &self.phi_b, d, d);

            // Find k neighbors
            let mut dists: Vec<(f32, usize)> = (0..n).map(|j| (sq_dist_3d(xyz, i, j), j)).collect();
            dists.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let neighbors: Vec<usize> = dists.iter().take(k).map(|&(_, j)| j).collect();

            // Compute attention weights and values
            let mut attn_scores: Vec<Vec<f32>> = Vec::with_capacity(k);
            let mut values: Vec<Vec<f32>> = Vec::with_capacity(k);

            for &j in &neighbors {
                let feat_j = &features[j * d..(j + 1) * d];

                // ψ(feat_j): key
                let key = linear_no_relu(feat_j, &self.psi_w, &self.psi_b, d, d);

                // Position encoding δ_ij = MLP(xyz_i - xyz_j)
                let rel_pos = [
                    xyz[i * 3] - xyz[j * 3],
                    xyz[i * 3 + 1] - xyz[j * 3 + 1],
                    xyz[i * 3 + 2] - xyz[j * 3 + 2],
                ];
                let pos_h = linear_relu(&rel_pos, &self.pos_w1, &self.pos_b1, 3, d);
                let delta = linear_no_relu(&pos_h, &self.pos_w2, &self.pos_b2, d, d);

                // Relation: γ(φ(feat_i) - ψ(feat_j) + δ_ij)
                let mut relation = vec![0.0_f32; d];
                for ch in 0..d {
                    relation[ch] = q[ch] - key[ch] + delta[ch];
                }
                let gamma_h = linear_relu(&relation, &self.gamma_w1, &self.gamma_b1, d, d);
                let attn_vec = linear_relu(&gamma_h, &self.gamma_w2, &self.gamma_b2, d, d);
                attn_scores.push(attn_vec);

                // Value: α_linear(feat_j) + δ_ij
                let v_linear = linear_no_relu(feat_j, &self.alpha_w, &self.alpha_b, d, d);
                let mut val = vec![0.0_f32; d];
                for ch in 0..d {
                    val[ch] = v_linear[ch] + delta[ch];
                }
                values.push(val);
            }

            // Softmax attention per dimension (vector attention)
            // For each dimension ch, softmax over k neighbors
            let mut out_i = vec![0.0_f32; d];
            for ch in 0..d {
                let scores_ch: Vec<f32> = attn_scores.iter().map(|a| a[ch]).collect();
                let weights = softmax(&scores_ch);
                for (idx, &w) in weights.iter().enumerate() {
                    out_i[ch] += w * values[idx][ch];
                }
            }

            out[i * d..(i + 1) * d].copy_from_slice(&out_i);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_transformer_output_shape() {
        let n = 8;
        let d = 16;
        let mut rng = LcgRng::new(42);
        let cfg = PointTransformerConfig {
            k: 4,
            d_model: d,
            n_heads: 1,
        };
        let layer = PointTransformerLayer::new(cfg, &mut rng);
        let xyz: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.1).collect();
        let feat: Vec<f32> = vec![0.1; n * d];
        let out = layer.forward(&xyz, &feat, n).unwrap();
        assert_eq!(out.len(), n * d);
    }

    #[test]
    fn point_transformer_finite() {
        let n = 6;
        let d = 8;
        let mut rng = LcgRng::new(42);
        let cfg = PointTransformerConfig {
            k: 3,
            d_model: d,
            n_heads: 1,
        };
        let layer = PointTransformerLayer::new(cfg, &mut rng);
        let mut feat_rng = LcgRng::new(1);
        let xyz: Vec<f32> = (0..n * 3).map(|i| (i as f32) * 0.5).collect();
        let mut feat = vec![0.0_f32; n * d];
        feat_rng.fill_normal(&mut feat);
        let out = layer.forward(&xyz, &feat, n).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "Point Transformer output must be finite"
        );
    }

    #[test]
    fn point_transformer_empty_error() {
        let mut rng = LcgRng::new(0);
        let cfg = PointTransformerConfig {
            k: 4,
            d_model: 8,
            n_heads: 1,
        };
        let layer = PointTransformerLayer::new(cfg, &mut rng);
        assert!(layer.forward(&[], &[], 0).is_err());
    }
}
