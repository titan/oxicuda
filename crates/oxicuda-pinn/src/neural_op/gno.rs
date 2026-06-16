//! Graph Neural Operator: kernel-based message passing for irregular grids.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Configuration for Graph Neural Operator.
pub struct GnoConfig {
    /// Input feature/coordinate dimensionality.
    pub d_in: usize,
    /// Output feature dimensionality.
    pub d_out: usize,
    /// Hidden layer sizes for the kernel MLP (processes relative positions).
    pub kernel_hidden: Vec<usize>,
    /// Neighborhood radius for graph construction.
    pub radius: f32,
}

/// Graph Neural Operator.
pub struct Gno {
    kernel_w: Vec<Vec<f32>>,
    kernel_b: Vec<Vec<f32>>,
    config: GnoConfig,
}

impl Gno {
    /// Construct a new GNO.
    pub fn new(config: GnoConfig, rng: &mut LcgRng) -> Self {
        let d_in = config.d_in;
        let d_out = config.d_out;

        // Kernel MLP: relative_position [d_in] → [kernel_hidden...] → [d_out × d_in]
        let kernel_output_dim = d_out * d_in;
        let mut layer_sizes = vec![d_in];
        layer_sizes.extend_from_slice(&config.kernel_hidden);
        layer_sizes.push(kernel_output_dim);

        let mut kernel_w = Vec::new();
        let mut kernel_b = Vec::new();
        for win in layer_sizes.windows(2) {
            let d_i = win[0];
            let d_o = win[1];
            let scale = (2.0 / d_i as f32).sqrt();
            let w: Vec<f32> = (0..d_o * d_i)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect();
            let b = vec![0.0_f32; d_o];
            kernel_w.push(w);
            kernel_b.push(b);
        }

        Self {
            kernel_w,
            kernel_b,
            config,
        }
    }

    /// Apply the kernel MLP to a relative position vector.
    fn kernel_mlp(&self, rel_pos: &[f32]) -> Vec<f32> {
        let mut x = rel_pos.to_vec();
        let n_layers = self.kernel_w.len();
        for (l, (w, b)) in self.kernel_w.iter().zip(self.kernel_b.iter()).enumerate() {
            let d_in = x.len();
            let d_out = b.len();
            let out: Vec<f32> = (0..d_out)
                .map(|i| {
                    let dot: f32 = (0..d_in).map(|j| w[i * d_in + j] * x[j]).sum();
                    dot + b[i]
                })
                .collect();
            // Tanh on all but last
            x = if l < n_layers - 1 {
                out.into_iter().map(|v| v.tanh()).collect()
            } else {
                out
            };
        }
        x
    }

    /// Squared Euclidean distance between two `d_in`-dimensional points.
    fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum()
    }

    /// Forward pass: `coords [n × d_in]`, `features [n × d_in]` → `[n × d_out]`.
    ///
    /// For each node `i`:
    /// 1. Find neighbors `j` within `radius`.
    /// 2. Compute kernel `K(x_i - x_j; θ)` → `[d_out × d_in]`.
    /// 3. Aggregate: `out_i = mean_j( K(x_i-x_j) · feat_j )`.
    pub fn forward(&self, coords: &[f32], features: &[f32], n: usize) -> PinnResult<Vec<f32>> {
        let d_in = self.config.d_in;
        let d_out = self.config.d_out;
        let radius_sq = self.config.radius * self.config.radius;

        if coords.len() != n * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: n * d_in,
                got: coords.len(),
            });
        }
        if features.len() != n * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: n * d_in,
                got: features.len(),
            });
        }

        let mut output = vec![0.0_f32; n * d_out];

        for i in 0..n {
            let xi = &coords[i * d_in..(i + 1) * d_in];
            let mut agg = vec![0.0_f32; d_out];
            let mut count = 0_usize;

            for j in 0..n {
                let xj = &coords[j * d_in..(j + 1) * d_in];
                let dist_sq = Self::sq_dist(xi, xj);

                if dist_sq <= radius_sq {
                    // Relative position
                    let rel: Vec<f32> =
                        xi.iter().zip(xj.iter()).map(|(&ai, &bi)| ai - bi).collect();
                    // Kernel output: [d_out × d_in] flat
                    let k_out = self.kernel_mlp(&rel);
                    // Apply kernel to feature_j: k_mat [d_out × d_in] * feat_j [d_in] → [d_out]
                    let feat_j = &features[j * d_in..(j + 1) * d_in];
                    for row in 0..d_out {
                        let dot: f32 = (0..d_in)
                            .map(|col| k_out[row * d_in + col] * feat_j[col])
                            .sum();
                        agg[row] += dot;
                    }
                    count += 1;
                }
            }

            // Mean aggregation
            if count > 0 {
                for val in &mut agg {
                    *val /= count as f32;
                }
            }
            output[i * d_out..(i + 1) * d_out].copy_from_slice(&agg);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(radius: f32) -> GnoConfig {
        GnoConfig {
            d_in: 2,
            d_out: 3,
            kernel_hidden: vec![16],
            radius,
        }
    }

    #[test]
    fn gno_construct_no_panic() {
        let mut rng = LcgRng::new(1);
        let _gno = Gno::new(make_config(1.0), &mut rng);
    }

    #[test]
    fn gno_forward_shape() {
        let mut rng = LcgRng::new(2);
        let gno = Gno::new(make_config(2.0), &mut rng);
        let n = 5;
        let coords = vec![0.0_f32; n * 2];
        let feats = vec![1.0_f32; n * 2];
        let out = gno
            .forward(&coords, &feats, n)
            .expect("GNO forward should succeed for valid input coordinates and features");
        assert_eq!(out.len(), n * 3);
    }

    #[test]
    fn gno_forward_finite() {
        let mut rng = LcgRng::new(3);
        let gno = Gno::new(make_config(1.0), &mut rng);
        let n = 4;
        let coords: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.3).collect();
        let feats: Vec<f32> = vec![0.1_f32; n * 2];
        let out = gno
            .forward(&coords, &feats, n)
            .expect("GNO forward should succeed and produce finite values");
        assert!(out.iter().all(|v| v.is_finite()), "GNO output not finite");
    }

    #[test]
    fn gno_zero_radius_only_self() {
        // With radius = 0, only self-connections (dist_sq = 0 when i == j)
        let mut rng = LcgRng::new(4);
        let gno = Gno::new(make_config(0.0), &mut rng);
        let n = 3;
        let coords = vec![1.0_f32, 0.0, 5.0, 0.0, 10.0, 0.0]; // 3 points far apart
        let feats = vec![1.0_f32; n * 2];
        let out = gno
            .forward(&coords, &feats, n)
            .expect("GNO forward with zero radius should succeed with only self-connections");
        // Each point only aggregates from itself (count=1 per node)
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gno_dim_mismatch_error() {
        let mut rng = LcgRng::new(5);
        let gno = Gno::new(make_config(1.0), &mut rng);
        let result = gno.forward(&[0.0; 5], &[0.0; 10], 5); // coords len != n * d_in
        assert!(result.is_err());
    }

    #[test]
    fn gno_large_radius_includes_all() {
        let mut rng = LcgRng::new(6);
        let gno = Gno::new(make_config(1e6), &mut rng);
        let n = 4;
        let coords: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
        let feats = vec![1.0_f32; n * 2];
        let out = gno
            .forward(&coords, &feats, n)
            .expect("GNO forward with large radius including all nodes should succeed");
        assert_eq!(out.len(), n * 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
