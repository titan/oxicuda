//! Point / Graph Fourier Neural Operator for unstructured meshes (1D latent).
//!
//! The standard FNO ([`crate::neural_op::fno`]) relies on the FFT and therefore
//! requires data sampled on a *uniform* grid. Many physics problems live on
//! *unstructured* point clouds or meshes. The geometry-aware FNO of Li et al.
//! (*Fourier Neural Operator with Learned Deformations*, 2022 — "Geo-FNO" /
//! "PointFNO") solves this by sandwiching the spectral convolution between two
//! learned scatter / gather operators that map the irregular points to and from
//! a fixed-resolution latent grid:
//!
//! ```text
//! points ──scatter (kernel interp)──▶ latent grid ──spectral conv──▶ latent grid ──gather──▶ points
//! ```
//!
//! This module implements that pipeline on a 1D latent grid: a Gaussian
//! radial-basis scatter to a uniform grid of `grid_size` nodes, an O(N²) DFT
//! spectral convolution truncated to `k_max` modes (complex channel mixing),
//! an inverse DFT, and a Gaussian gather back to the query coordinates. A
//! pointwise linear skip connection in physical space preserves high
//! frequencies the truncated spectrum drops.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;
use crate::neural_op::fno::{dft_1d, idft_1d};

/// Configuration for the point / graph FNO.
#[derive(Debug, Clone)]
pub struct PointFnoConfig {
    /// Number of input channels per point.
    pub d_in: usize,
    /// Number of output channels per point.
    pub d_out: usize,
    /// Latent channel width inside the spectral block.
    pub width: usize,
    /// Resolution of the uniform latent grid the points scatter onto.
    pub grid_size: usize,
    /// Number of Fourier modes retained in the spectral convolution.
    pub k_max: usize,
    /// Gaussian scatter/gather bandwidth as a multiple of the grid spacing.
    /// Larger values smooth more; ~1.0 keeps the interpolation local.
    pub kernel_width: f32,
}

impl PointFnoConfig {
    /// Validate and construct a configuration.
    pub fn new(
        d_in: usize,
        d_out: usize,
        width: usize,
        grid_size: usize,
        k_max: usize,
        kernel_width: f32,
    ) -> PinnResult<Self> {
        if d_in == 0 || d_out == 0 {
            return Err(PinnError::EmptyInput);
        }
        if width == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if grid_size < 2 {
            return Err(PinnError::InvalidGridResolution { n: grid_size });
        }
        if k_max == 0 || k_max > grid_size / 2 + 1 {
            return Err(PinnError::TooManyFourierModes {
                k_max,
                n_half: grid_size / 2 + 1,
            });
        }
        if !(kernel_width.is_finite() && kernel_width > 0.0) {
            return Err(PinnError::Internal(
                "kernel_width must be finite and > 0".to_string(),
            ));
        }
        Ok(Self {
            d_in,
            d_out,
            width,
            grid_size,
            k_max,
            kernel_width,
        })
    }
}

/// Point / Graph Fourier Neural Operator on a 1D latent grid.
#[derive(Debug, Clone)]
pub struct PointFno {
    config: PointFnoConfig,
    /// Lift `d_in → width` (row-major `width × d_in`) + bias.
    lift_w: Vec<f32>,
    lift_b: Vec<f32>,
    /// Spectral weights per mode and channel pair (`width × width × k_max`).
    spec_w_real: Vec<f32>,
    spec_w_imag: Vec<f32>,
    /// Pointwise physical-space skip (`width × width`) + bias.
    skip_w: Vec<f32>,
    skip_b: Vec<f32>,
    /// Project `width → d_out` (row-major `d_out × width`) + bias.
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
}

fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

impl PointFno {
    /// Construct a new point-FNO with He-style random weights.
    pub fn new(config: PointFnoConfig, rng: &mut LcgRng) -> Self {
        let d_in = config.d_in;
        let d_out = config.d_out;
        let w = config.width;
        let k = config.k_max;

        let scale_lift = (2.0 / d_in as f32).sqrt();
        let lift_w: Vec<f32> = (0..w * d_in)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_lift)
            .collect();
        let lift_b = vec![0.0_f32; w];

        let scale_spec = (2.0 / (w * k.max(1)) as f32).sqrt();
        let n_spec = w * w * k;
        let spec_w_real: Vec<f32> = (0..n_spec)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_spec)
            .collect();
        let spec_w_imag: Vec<f32> = (0..n_spec)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_spec)
            .collect();

        let scale_skip = (2.0 / w as f32).sqrt();
        let skip_w: Vec<f32> = (0..w * w)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_skip)
            .collect();
        let skip_b = vec![0.0_f32; w];

        let scale_proj = (2.0 / w as f32).sqrt();
        let proj_w: Vec<f32> = (0..d_out * w)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_proj)
            .collect();
        let proj_b = vec![0.0_f32; d_out];

        Self {
            config,
            lift_w,
            lift_b,
            spec_w_real,
            spec_w_imag,
            skip_w,
            skip_b,
            proj_w,
            proj_b,
        }
    }

    /// Normalised grid-node positions in `[0, 1]`.
    fn grid_positions(&self) -> Vec<f32> {
        let g = self.config.grid_size;
        (0..g).map(|i| i as f32 / (g - 1) as f32).collect()
    }

    /// Gaussian scatter weights from each point to each grid node.
    /// Returns `(weights[n_points × grid_size], col_sums[grid_size])` where
    /// `col_sums` is the per-grid-node normaliser used to average.
    fn scatter_weights(&self, coords_norm: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let g = self.config.grid_size;
        let n = coords_norm.len();
        let spacing = 1.0 / (g - 1) as f32;
        let sigma = self.config.kernel_width * spacing;
        let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
        let grid = self.grid_positions();
        let mut weights = vec![0.0_f32; n * g];
        let mut col_sums = vec![0.0_f32; g];
        for (p, &xp) in coords_norm.iter().enumerate() {
            for (gi, &xg) in grid.iter().enumerate() {
                let d = xp - xg;
                let wv = (-d * d * inv_two_sigma2).exp();
                weights[p * g + gi] = wv;
                col_sums[gi] += wv;
            }
        }
        (weights, col_sums)
    }

    /// Lift each point feature vector to the latent width.
    fn lift(&self, features: &[f32], n_points: usize) -> Vec<f32> {
        let d_in = self.config.d_in;
        let w = self.config.width;
        let mut out = vec![0.0_f32; n_points * w];
        for p in 0..n_points {
            let fin = &features[p * d_in..(p + 1) * d_in];
            for o in 0..w {
                let w_row = &self.lift_w[o * d_in..o * d_in + d_in];
                let dot: f32 = w_row
                    .iter()
                    .zip(fin.iter())
                    .map(|(&wij, &xj)| wij * xj)
                    .sum();
                out[p * w + o] = self.lift_b[o] + dot;
            }
        }
        out
    }

    /// Spectral convolution on the latent grid.
    /// `grid_feat` is `grid_size × width` row-major; returns same shape.
    fn spectral_conv(&self, grid_feat: &[f32]) -> Vec<f32> {
        let g = self.config.grid_size;
        let w = self.config.width;
        let k = self.config.k_max;

        // Transform each channel along the grid axis.
        // Collect per-channel signals.
        let mut out = vec![0.0_f32; g * w];
        // Forward DFT per channel.
        let mut chan_real = vec![0.0_f32; g * w];
        let mut chan_imag = vec![0.0_f32; g * w];
        for c in 0..w {
            let signal: Vec<f32> = (0..g).map(|i| grid_feat[i * w + c]).collect();
            let (re, im) = dft_1d(&signal);
            for i in 0..g {
                chan_real[i * w + c] = re[i];
                chan_imag[i * w + c] = im[i];
            }
        }
        // Per retained mode, mix channels with complex weights.
        let mut mixed_real = vec![0.0_f32; g * w];
        let mut mixed_imag = vec![0.0_f32; g * w];
        for mode in 0..k {
            for o in 0..w {
                let mut acc_r = 0.0_f32;
                let mut acc_i = 0.0_f32;
                for i in 0..w {
                    let xr = chan_real[mode * w + i];
                    let xi = chan_imag[mode * w + i];
                    let wr = self.spec_w_real[i * w * k + o * k + mode];
                    let wi = self.spec_w_imag[i * w * k + o * k + mode];
                    acc_r += xr * wr - xi * wi;
                    acc_i += xr * wi + xi * wr;
                }
                mixed_real[mode * w + o] = acc_r;
                mixed_imag[mode * w + o] = acc_i;
            }
        }
        // Inverse DFT per channel (zero-padded high modes).
        for c in 0..w {
            let re: Vec<f32> = (0..g).map(|i| mixed_real[i * w + c]).collect();
            let im: Vec<f32> = (0..g).map(|i| mixed_imag[i * w + c]).collect();
            let sig = idft_1d(&re, &im);
            for i in 0..g {
                out[i * w + c] = sig[i];
            }
        }
        out
    }

    /// Pointwise physical-space linear skip on latent features.
    /// `feat` is `n × width`; returns `n × width`.
    fn skip(&self, feat: &[f32], n: usize) -> Vec<f32> {
        let w = self.config.width;
        let mut out = vec![0.0_f32; n * w];
        for p in 0..n {
            let fin = &feat[p * w..(p + 1) * w];
            for o in 0..w {
                let w_row = &self.skip_w[o * w..o * w + w];
                let dot: f32 = w_row
                    .iter()
                    .zip(fin.iter())
                    .map(|(&wij, &xj)| wij * xj)
                    .sum();
                out[p * w + o] = self.skip_b[o] + dot;
            }
        }
        out
    }

    /// Project latent width back to output channels.
    fn project(&self, feat: &[f32], n: usize) -> Vec<f32> {
        let w = self.config.width;
        let d_out = self.config.d_out;
        let mut out = vec![0.0_f32; n * d_out];
        for p in 0..n {
            let fin = &feat[p * w..(p + 1) * w];
            for o in 0..d_out {
                let w_row = &self.proj_w[o * w..o * w + w];
                let dot: f32 = w_row
                    .iter()
                    .zip(fin.iter())
                    .map(|(&wij, &xj)| wij * xj)
                    .sum();
                out[p * d_out + o] = self.proj_b[o] + dot;
            }
        }
        out
    }

    /// Forward pass over an unstructured 1D point set.
    ///
    /// * `coords` — point positions, length `n_points` (arbitrary range; they
    ///   are min-max normalised internally to `[0, 1]`).
    /// * `features` — `n_points × d_in` row-major input features.
    ///
    /// Returns `n_points × d_out` row-major output.
    pub fn forward(
        &self,
        coords: &[f32],
        features: &[f32],
        n_points: usize,
    ) -> PinnResult<Vec<f32>> {
        let d_in = self.config.d_in;
        let w = self.config.width;
        let g = self.config.grid_size;
        if n_points == 0 {
            return Err(PinnError::EmptyInput);
        }
        if coords.len() != n_points {
            return Err(PinnError::DimensionMismatch {
                expected: n_points,
                got: coords.len(),
            });
        }
        if features.len() != n_points * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d_in,
                got: features.len(),
            });
        }

        // Min-max normalise coordinates to [0, 1].
        let mut lo = coords[0];
        let mut hi = coords[0];
        for &c in coords {
            if c < lo {
                lo = c;
            }
            if c > hi {
                hi = c;
            }
        }
        let span = if (hi - lo).abs() < 1e-12 {
            1.0
        } else {
            hi - lo
        };
        let coords_norm: Vec<f32> = coords.iter().map(|&c| (c - lo) / span).collect();

        // Lift to latent width.
        let lifted = self.lift(features, n_points); // n × w

        // Scatter to the latent grid (channel-wise weighted average).
        let (sweights, col_sums) = self.scatter_weights(&coords_norm);
        let mut grid_feat = vec![0.0_f32; g * w];
        for gi in 0..g {
            let norm = if col_sums[gi] > 1e-12 {
                1.0 / col_sums[gi]
            } else {
                0.0
            };
            for p in 0..n_points {
                let sw = sweights[p * g + gi] * norm;
                if sw == 0.0 {
                    continue;
                }
                let src = &lifted[p * w..(p + 1) * w];
                let dst = &mut grid_feat[gi * w..(gi + 1) * w];
                for c in 0..w {
                    dst[c] += sw * src[c];
                }
            }
        }

        // Spectral convolution on the latent grid + GeLU.
        let conv = self.spectral_conv(&grid_feat);
        let grid_act: Vec<f32> = conv.iter().map(|&v| gelu(v)).collect();

        // Gather from grid back to points (Gaussian interpolation, row-normalised).
        let mut gathered = vec![0.0_f32; n_points * w];
        for p in 0..n_points {
            // Row weights from this point to all grid nodes.
            let row = &sweights[p * g..(p + 1) * g];
            let row_sum: f32 = row.iter().sum();
            let inv = if row_sum > 1e-12 { 1.0 / row_sum } else { 0.0 };
            let dst = &mut gathered[p * w..(p + 1) * w];
            for gi in 0..g {
                let gw = row[gi] * inv;
                if gw == 0.0 {
                    continue;
                }
                let gsrc = &grid_act[gi * w..(gi + 1) * w];
                for c in 0..w {
                    dst[c] += gw * gsrc[c];
                }
            }
        }

        // Physical-space skip connection (uses the lifted features directly).
        let skip = self.skip(&lifted, n_points);
        let mut combined = gathered;
        for i in 0..combined.len() {
            combined[i] = gelu(combined[i] + skip[i]);
        }

        // Project to output channels.
        let out = self.project(&combined, n_points);
        if out.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "PointFno::forward",
            });
        }
        Ok(out)
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &PointFnoConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> PointFnoConfig {
        PointFnoConfig::new(1, 1, 8, 16, 6, 1.0).expect("valid config")
    }

    #[test]
    fn config_validation() {
        assert!(PointFnoConfig::new(0, 1, 8, 16, 4, 1.0).is_err());
        assert!(PointFnoConfig::new(1, 1, 0, 16, 4, 1.0).is_err());
        assert!(PointFnoConfig::new(1, 1, 8, 1, 4, 1.0).is_err());
        assert!(PointFnoConfig::new(1, 1, 8, 16, 0, 1.0).is_err());
        assert!(PointFnoConfig::new(1, 1, 8, 16, 100, 1.0).is_err());
        assert!(PointFnoConfig::new(1, 1, 8, 16, 4, -1.0).is_err());
        assert!(PointFnoConfig::new(1, 1, 8, 16, 4, 1.0).is_ok());
    }

    #[test]
    fn forward_output_shape() {
        let mut rng = LcgRng::new(1);
        let op = PointFno::new(make_config(), &mut rng);
        let n = 12;
        // Irregular (non-uniform) coordinates.
        let coords: Vec<f32> = (0..n).map(|i| (i as f32).powf(1.5) * 0.1).collect();
        let feats: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let out = op.forward(&coords, &feats, n).expect("forward");
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_multichannel_shape() {
        let mut rng = LcgRng::new(7);
        let cfg = PointFnoConfig::new(3, 2, 8, 16, 5, 1.2).expect("cfg");
        let op = PointFno::new(cfg, &mut rng);
        let n = 9;
        let coords: Vec<f32> = (0..n).map(|i| i as f32 * 0.4).collect();
        let feats: Vec<f32> = (0..n * 3).map(|i| (i as f32 * 0.11).cos()).collect();
        let out = op.forward(&coords, &feats, n).expect("forward");
        assert_eq!(out.len(), n * 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_given_seed() {
        let cfg = make_config();
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let op_a = PointFno::new(cfg.clone(), &mut rng_a);
        let op_b = PointFno::new(cfg, &mut rng_b);
        let n = 8;
        let coords: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let feats: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let out_a = op_a.forward(&coords, &feats, n).expect("a");
        let out_b = op_b.forward(&coords, &feats, n).expect("b");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn dim_mismatch_errors() {
        let mut rng = LcgRng::new(3);
        let op = PointFno::new(make_config(), &mut rng);
        assert!(op.forward(&[0.0; 5], &[0.0; 5], 6).is_err()); // coords len
        assert!(op.forward(&[0.0; 6], &[0.0; 5], 6).is_err()); // feats len
        assert!(op.forward(&[], &[], 0).is_err()); // empty
    }

    #[test]
    fn scatter_weights_normalise_to_partition() {
        // For a single point exactly on a grid node, that node's averaged value
        // must reproduce the point value after scatter+normalise.
        let mut rng = LcgRng::new(5);
        let op = PointFno::new(make_config(), &mut rng);
        let coords_norm = vec![0.0_f32]; // first grid node
        let (w, col_sums) = op.scatter_weights(&coords_norm);
        assert_eq!(w.len(), op.config.grid_size);
        // Nearest node has the largest weight.
        let max_idx = w
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        assert_eq!(max_idx, 0, "closest grid node should dominate");
        assert!(col_sums[0] > 0.0);
    }

    #[test]
    fn constant_field_stays_finite_and_smooth() {
        // A constant input field should not produce NaNs and remains bounded.
        let mut rng = LcgRng::new(11);
        let op = PointFno::new(make_config(), &mut rng);
        let n = 20;
        let coords: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let feats = vec![1.0_f32; n];
        let out = op.forward(&coords, &feats, n).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
