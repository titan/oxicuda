//! Sparse 3D convolution over sparse tensors.

use std::collections::HashMap;

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

/// Sparse tensor: list of occupied coordinates with corresponding features.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub coords: Vec<[i32; 3]>,
    pub features: Vec<f32>, // [len(coords) × c_in] row-major
    pub c_in: usize,
}

/// Configuration for a sparse 3D convolution.
#[derive(Debug, Clone)]
pub struct SparseConv3dConfig {
    pub kernel_size: usize,
    pub c_in: usize,
    pub c_out: usize,
}

/// Sparse 3D convolution layer.
///
/// Weight shape: `[kernel_size^3 × c_in × c_out]`.
pub struct SparseConv3d {
    config: SparseConv3dConfig,
    weight: Vec<f32>, // [k^3 × c_in × c_out]
    bias: Vec<f32>,   // [c_out]
}

impl SparseConv3d {
    /// Create a new Sparse 3D convolution layer with Xavier-uniform initialization.
    pub fn new(config: SparseConv3dConfig, rng: &mut LcgRng) -> Self {
        let k = config.kernel_size;
        let k3 = k * k * k;
        let fan_in = k3 * config.c_in;
        let fan_out = k3 * config.c_out;

        let mut weight = vec![0.0_f32; k3 * config.c_in * config.c_out];
        rng.fill_xavier_uniform(&mut weight, fan_in, fan_out);
        let bias = vec![0.0_f32; config.c_out];

        Self {
            config,
            weight,
            bias,
        }
    }

    /// Forward pass over a `SparseTensor`.
    ///
    /// For each input coord + kernel offset: add contribution to output coord
    /// (if occupied or not — all offsets generate output). Uses HashMap for
    /// accumulation. Returns a new `SparseTensor`.
    pub fn forward(&self, input: &SparseTensor) -> Geom3dResult<SparseTensor> {
        let n = input.coords.len();
        if n == 0 {
            return Ok(SparseTensor {
                coords: Vec::new(),
                features: Vec::new(),
                c_in: self.config.c_out,
            });
        }
        if input.features.len() != n * self.config.c_in {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * self.config.c_in,
                got: input.features.len(),
            });
        }
        if input.c_in != self.config.c_in {
            return Err(Geom3dError::DimensionMismatch {
                expected: self.config.c_in,
                got: input.c_in,
            });
        }

        let k = self.config.kernel_size as i32;
        let half = k / 2;
        let c_in = self.config.c_in;
        let c_out = self.config.c_out;
        let _k3 = (k * k * k) as usize;

        // Accumulate into HashMap: coord → Vec<f32> of c_out values
        let mut acc: HashMap<[i32; 3], Vec<f32>> = HashMap::new();

        for (pt_idx, &coord) in input.coords.iter().enumerate() {
            let feat = &input.features[pt_idx * c_in..(pt_idx + 1) * c_in];

            for kx in 0..k {
                for ky in 0..k {
                    for kz in 0..k {
                        let out_coord = [
                            coord[0] + kx - half,
                            coord[1] + ky - half,
                            coord[2] + kz - half,
                        ];

                        let kernel_idx = ((kx * k + ky) * k + kz) as usize;

                        let out_vals = acc.entry(out_coord).or_insert_with(|| {
                            let mut v = vec![0.0_f32; c_out];
                            // Add bias on first creation
                            v.copy_from_slice(&self.bias);
                            v
                        });

                        for (co, oval) in out_vals.iter_mut().enumerate() {
                            for (ci, &fi) in feat.iter().enumerate() {
                                let w_idx = kernel_idx * c_in * c_out + ci * c_out + co;
                                *oval += self.weight[w_idx] * fi;
                            }
                        }
                    }
                }
            }
        }

        // Build output SparseTensor (sorted by coord for determinism)
        let mut entries: Vec<([i32; 3], Vec<f32>)> = acc.into_iter().collect();
        entries.sort_unstable_by_key(|(c, _)| *c);

        let m = entries.len();
        let mut out_coords = Vec::with_capacity(m);
        let mut out_features = Vec::with_capacity(m * c_out);

        for (coord, feats) in entries {
            out_coords.push(coord);
            out_features.extend_from_slice(&feats);
        }

        Ok(SparseTensor {
            coords: out_coords,
            features: out_features,
            c_in: c_out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_conv3d_output_nonempty() {
        let mut rng = LcgRng::new(42);
        let cfg = SparseConv3dConfig {
            kernel_size: 3,
            c_in: 4,
            c_out: 8,
        };
        let conv = SparseConv3d::new(cfg, &mut rng);
        let input = SparseTensor {
            coords: vec![[0, 0, 0], [1, 1, 1]],
            features: vec![1.0_f32; 2 * 4],
            c_in: 4,
        };
        let out = conv.forward(&input).unwrap();
        assert!(!out.coords.is_empty());
        assert_eq!(out.c_in, 8);
        assert_eq!(out.features.len(), out.coords.len() * 8);
    }

    #[test]
    fn sparse_conv3d_empty_input() {
        let mut rng = LcgRng::new(0);
        let cfg = SparseConv3dConfig {
            kernel_size: 3,
            c_in: 4,
            c_out: 8,
        };
        let conv = SparseConv3d::new(cfg, &mut rng);
        let input = SparseTensor {
            coords: vec![],
            features: vec![],
            c_in: 4,
        };
        let out = conv.forward(&input).unwrap();
        assert!(out.coords.is_empty());
    }

    #[test]
    fn sparse_conv3d_finite_output() {
        let mut rng = LcgRng::new(42);
        let cfg = SparseConv3dConfig {
            kernel_size: 3,
            c_in: 2,
            c_out: 4,
        };
        let conv = SparseConv3d::new(cfg, &mut rng);
        let input = SparseTensor {
            coords: vec![[0, 0, 0]],
            features: vec![1.0, -1.0],
            c_in: 2,
        };
        let out = conv.forward(&input).unwrap();
        assert!(out.features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_conv3d_cin_mismatch_error() {
        let mut rng = LcgRng::new(0);
        let cfg = SparseConv3dConfig {
            kernel_size: 3,
            c_in: 4,
            c_out: 8,
        };
        let conv = SparseConv3d::new(cfg, &mut rng);
        let input = SparseTensor {
            coords: vec![[0, 0, 0]],
            features: vec![1.0; 3], // wrong: 3 != 4
            c_in: 4,
        };
        assert!(conv.forward(&input).is_err());
    }
}
