//! Slimmable networks: width multipliers with adaptive batch-norm statistics.
//!
//! A `SlimmableNet` can be evaluated at four width multipliers: [0.25, 0.5, 0.75, 1.0].
//! Each width has its own batch-norm running statistics (mean + var per channel),
//! while the conv weights are shared across widths (sliced to the active width).

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

/// The four canonical width multipliers for slimmable networks.
pub const WIDTH_MULTIPLIERS: &[f32] = &[0.25, 0.5, 0.75, 1.0];

/// Batch-norm statistics for a single width setting.
#[derive(Debug, Clone)]
pub struct BnStats {
    /// Running mean: `[n_channels]`.
    pub running_mean: Vec<f32>,
    /// Running variance: `[n_channels]`.
    pub running_var: Vec<f32>,
    /// Number of channels this BN is for.
    pub n_channels: usize,
}

impl BnStats {
    /// Initialise with mean=0, var=1.
    #[must_use]
    pub fn new(n_channels: usize) -> Self {
        Self {
            running_mean: vec![0.0_f32; n_channels],
            running_var: vec![1.0_f32; n_channels],
            n_channels,
        }
    }

    /// Update running statistics from a batch of activations `[n_channels, n_elements]`.
    ///
    /// Uses exponential moving average with `momentum = 0.1`.
    pub fn update(&mut self, activations: &[f32], momentum: f32) -> NasResult<()> {
        let n = self.n_channels;
        if activations.len() % n != 0 {
            return Err(NasError::DimensionMismatch {
                expected: n,
                got: activations.len() % n,
            });
        }
        let spatial = activations.len() / n;
        if spatial == 0 {
            return Ok(());
        }
        for c in 0..n {
            let slice: Vec<f32> = (0..spatial).map(|s| activations[c * spatial + s]).collect();
            let mean = slice.iter().sum::<f32>() / spatial as f32;
            let var = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / spatial as f32;
            self.running_mean[c] = (1.0 - momentum) * self.running_mean[c] + momentum * mean;
            self.running_var[c] = (1.0 - momentum) * self.running_var[c] + momentum * var;
        }
        Ok(())
    }

    /// Apply batch normalisation (inference mode) to `[n_channels * spatial]`.
    pub fn normalize(&self, activations: &[f32], eps: f32) -> NasResult<Vec<f32>> {
        let n = self.n_channels;
        if activations.len() % n != 0 {
            return Err(NasError::DimensionMismatch {
                expected: n,
                got: activations.len() % n,
            });
        }
        let spatial = activations.len() / n;
        let mut out = vec![0.0_f32; activations.len()];
        for c in 0..n {
            let mean = self.running_mean[c];
            let std = (self.running_var[c] + eps).sqrt();
            for s in 0..spatial {
                let idx = c * spatial + s;
                out[idx] = (activations[idx] - mean) / std;
            }
        }
        Ok(out)
    }
}

// ─── SlimmableNet ─────────────────────────────────────────────────────────────

/// A slimmable network layer with width-adaptive BN statistics.
///
/// Conv weights are stored at full width and sliced at runtime.
#[derive(Debug, Clone)]
pub struct SlimmableNet {
    /// Full-width conv weight: `[out_ch * in_ch * k * k]`.
    pub full_weight: Vec<f32>,
    /// Full output channel count (width multiplier = 1.0).
    pub max_out_ch: usize,
    /// Full input channel count.
    pub max_in_ch: usize,
    /// Kernel size.
    pub kernel: usize,
    /// Per-width BN statistics: indexed by multiplier index 0..4.
    pub bn_stats: Vec<BnStats>,
}

impl SlimmableNet {
    /// Construct a slimmable layer with randomly initialised weights.
    #[must_use]
    pub fn new(max_in_ch: usize, max_out_ch: usize, kernel: usize, rng: &mut LcgRng) -> Self {
        let weight_len = max_out_ch * max_in_ch * kernel * kernel;
        let mut full_weight = vec![0.0_f32; weight_len];
        rng.fill_normal(&mut full_weight);
        full_weight.iter_mut().for_each(|v| *v *= 0.01);

        let bn_stats = WIDTH_MULTIPLIERS
            .iter()
            .map(|&m| {
                let n_ch = ((max_out_ch as f32 * m).ceil() as usize).max(1);
                BnStats::new(n_ch)
            })
            .collect();

        Self {
            full_weight,
            max_out_ch,
            max_in_ch,
            kernel,
            bn_stats,
        }
    }

    /// Validate that `multiplier` is one of the canonical values.
    pub fn check_multiplier(multiplier: f32) -> NasResult<usize> {
        WIDTH_MULTIPLIERS
            .iter()
            .enumerate()
            .find(|&(_, &m)| (m - multiplier).abs() < 1e-6)
            .map(|(i, _)| i)
            .ok_or(NasError::InvalidWidthMultiplier { value: multiplier })
    }

    /// Forward pass at a given width multiplier.
    ///
    /// Slices the conv weight to `[active_out_ch, active_in_ch, k, k]`,
    /// performs a naive conv, then applies the corresponding BN stats.
    pub fn forward(
        &self,
        input: &[f32],
        in_ch: usize,
        h: usize,
        w: usize,
        multiplier: f32,
    ) -> NasResult<Vec<f32>> {
        let m_idx = Self::check_multiplier(multiplier)?;
        let active_in = ((in_ch as f32 * multiplier).ceil() as usize).max(1);
        let active_out = ((self.max_out_ch as f32 * multiplier).ceil() as usize).max(1);
        let k = self.kernel;
        let pad = k / 2;
        let _ = pad;

        // Naive conv
        let mut out = vec![0.0_f32; active_out * h * w];
        for oc in 0..active_out {
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = 0.0_f32;
                    for ic in 0..active_in {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = oy as isize + ky as isize - (k / 2) as isize;
                                let ix = ox as isize + kx as isize - (k / 2) as isize;
                                if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let in_val = if ic < in_ch {
                                    input[ic * h * w + iy as usize * w + ix as usize]
                                } else {
                                    0.0
                                };
                                // weight idx: oc * max_in_ch * k * k + ic * k * k + ky * k + kx
                                let w_idx = oc * self.max_in_ch * k * k + ic * k * k + ky * k + kx;
                                let w_val = self.full_weight.get(w_idx).copied().unwrap_or(0.0);
                                acc += in_val * w_val;
                            }
                        }
                    }
                    out[oc * h * w + oy * w + ox] = acc;
                }
            }
        }

        // BN normalisation
        let bn = &self.bn_stats[m_idx];
        // Extend bn if the active_out doesn't match (approximate)
        if bn.n_channels != active_out {
            // Just return the raw conv output without BN (skip for mismatched channels)
            return Ok(out);
        }
        bn.normalize(&out, 1e-5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slimmable_forward_all_multipliers() {
        let mut rng = LcgRng::new(99);
        let net = SlimmableNet::new(4, 8, 3, &mut rng);
        let input = vec![0.5_f32; 4 * 4 * 4];
        for &m in WIDTH_MULTIPLIERS {
            let out = net
                .forward(&input, 4, 4, 4, m)
                .expect("test invariant: slimmable forward");
            assert!(!out.is_empty(), "empty output for m={m}");
        }
    }

    #[test]
    fn invalid_multiplier_errors() {
        let mut rng = LcgRng::new(1);
        let net = SlimmableNet::new(4, 4, 3, &mut rng);
        let input = vec![1.0_f32; 4 * 4 * 4];
        assert!(net.forward(&input, 4, 4, 4, 0.3).is_err());
    }

    #[test]
    fn bn_stats_update_and_normalize() {
        let mut bn = BnStats::new(2);
        let acts = vec![1.0_f32, 2.0, 3.0, 4.0]; // [2 ch, 2 spatial]
        bn.update(&acts, 0.1).expect("test invariant: bn update");
        let normed = bn
            .normalize(&acts, 1e-5)
            .expect("test invariant: bn normalize");
        assert_eq!(normed.len(), 4);
    }
}
