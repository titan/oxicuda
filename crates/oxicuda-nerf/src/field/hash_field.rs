//! Instant-NGP style neural field: hash grid + tiny MLP decoder.

use crate::encoding::hash_grid::{HashGrid, HashGridConfig};
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── HashField ───────────────────────────────────────────────────────────────

/// Instant-NGP style neural field: multi-resolution hash grid + 2-layer MLP.
///
/// Architecture:
/// - HashGrid query → `[n_levels * F]` features
/// - Concat with `dir_enc` → `[(n_levels*F) + dir_enc_dim]`
/// - Linear + ReLU → `hidden_dim`
/// - Linear → `(1 + color_dim)`
/// - sigma = ReLU(output\[0\]), color_feat = output\[1..\]
#[derive(Debug, Clone)]
pub struct HashField {
    /// Multi-resolution hash grid.
    pub grid: HashGrid,
    /// MLP layer 1 weights: `[hidden_dim * (n_levels*F + dir_enc_dim)]`.
    mlp_w1: Vec<f32>,
    /// MLP layer 1 bias: `[hidden_dim]`.
    mlp_b1: Vec<f32>,
    /// MLP layer 2 weights: `[(1 + color_dim) * hidden_dim]`.
    mlp_w2: Vec<f32>,
    /// MLP layer 2 bias: `[(1 + color_dim)]`.
    mlp_b2: Vec<f32>,
    /// Hidden layer width.
    hidden_dim: usize,
    /// Dimension of encoded view direction.
    dir_enc_dim: usize,
    /// Number of output color features.
    color_dim: usize,
}

impl HashField {
    /// Create a new `HashField`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidHashConfig` or `InvalidFeatureDim` for bad parameters.
    pub fn new(
        grid_cfg: HashGridConfig,
        hidden_dim: usize,
        dir_enc_dim: usize,
        color_dim: usize,
        rng: &mut LcgRng,
    ) -> NerfResult<Self> {
        if hidden_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        if color_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        let grid = HashGrid::new(grid_cfg, rng)?;
        let grid_feat_dim = grid.output_dim();
        let in_dim = grid_feat_dim + dir_enc_dim;
        let out_dim = 1 + color_dim;

        let mut init = |fan_in: usize, fan_out: usize| -> (Vec<f32>, Vec<f32>) {
            let s = (2.0_f32 / fan_in as f32).sqrt();
            let mut w = vec![0.0_f32; fan_out * fan_in];
            for v in w.iter_mut() {
                let (a, _) = rng.next_normal_pair();
                *v = a * s;
            }
            (w, vec![0.0_f32; fan_out])
        };

        let (mlp_w1, mlp_b1) = init(in_dim, hidden_dim);
        let (mlp_w2, mlp_b2) = init(hidden_dim, out_dim);

        Ok(Self {
            grid,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            hidden_dim,
            dir_enc_dim,
            color_dim,
        })
    }

    /// Query the hash field at a 3D world point with encoded view direction.
    ///
    /// Returns `(sigma: f32, color_feat: Vec<f32>)` where color_feat has `color_dim` elements.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `dir_enc.len() != dir_enc_dim`.
    pub fn forward(&self, xyz: [f32; 3], dir_enc: &[f32]) -> NerfResult<(f32, Vec<f32>)> {
        if dir_enc.len() != self.dir_enc_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.dir_enc_dim,
                got: dir_enc.len(),
            });
        }

        // Hash grid feature lookup
        let grid_feat = self.grid.query(xyz)?;

        // Concatenate with direction encoding
        let mut input = Vec::with_capacity(grid_feat.len() + self.dir_enc_dim);
        input.extend_from_slice(&grid_feat);
        input.extend_from_slice(dir_enc);

        let in_dim = input.len();
        let h = self.hidden_dim;
        let out_dim = 1 + self.color_dim;

        // Layer 1: FC + ReLU
        let mut hidden = vec![0.0_f32; h];
        for (o, (wo, &bi)) in hidden
            .iter_mut()
            .zip(self.mlp_w1.chunks(in_dim).zip(self.mlp_b1.iter()))
        {
            *o = (wo
                .iter()
                .zip(input.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + bi)
                .max(0.0);
        }

        // Layer 2: FC, no activation here
        let mut out = vec![0.0_f32; out_dim];
        for (o, (wo, &bi)) in out
            .iter_mut()
            .zip(self.mlp_w2.chunks(h).zip(self.mlp_b2.iter()))
        {
            *o = wo
                .iter()
                .zip(hidden.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + bi;
        }

        let sigma = out[0].max(0.0);
        let color_feat = out[1..].to_vec();

        Ok((sigma, color_feat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash_field(seed: u64) -> HashField {
        let cfg = HashGridConfig {
            n_levels: 4,
            n_features_per_level: 2,
            log2_hashmap_size: 8,
            base_resolution: 4,
            max_resolution: 32,
        };
        let mut rng = LcgRng::new(seed);
        HashField::new(cfg, 16, 8, 3, &mut rng).unwrap()
    }

    #[test]
    fn forward_output_types() {
        let hf = make_hash_field(42);
        let dir_enc = vec![0.1_f32; 8];
        let (sigma, color) = hf.forward([0.5, 0.3, 0.7], &dir_enc).unwrap();
        assert!(sigma >= 0.0);
        assert_eq!(color.len(), 3);
    }

    #[test]
    fn wrong_dir_enc_dim() {
        let hf = make_hash_field(99);
        let dir_enc = vec![0.0_f32; 5]; // Wrong size (expected 8)
        assert!(hf.forward([0.0, 0.0, 0.0], &dir_enc).is_err());
    }
}
