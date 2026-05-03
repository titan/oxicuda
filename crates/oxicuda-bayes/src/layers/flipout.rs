//! Flipout layers for efficient Bayesian approximate inference.
//!
//! Uses ±1 random sign perturbations to decorrelate mini-batch samples,
//! enabling efficient MC integration without full weight re-sampling per sample.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

/// Flipout linear layer.
///
/// Forward pass: `out = x @ W_mu^T + (x * r) @ W_delta^T * s + bias`
/// where `r ∈ {-1,+1}^{in}` and `s ∈ {-1,+1}^{out}` are sampled per forward call.
#[derive(Debug, Clone)]
pub struct FlipoutLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Mean weights `[out × in]`.
    pub w_mu: Vec<f32>,
    /// Perturbation weights `[out × in]` (typically σ * N(0,1)).
    pub w_delta: Vec<f32>,
    /// Bias vector `[out]`.
    pub bias: Vec<f32>,
}

impl FlipoutLinear {
    /// Create a new FlipoutLinear layer with random initialization.
    ///
    /// - `w_mu` ~ N(0, 0.1 / sqrt(in_features))
    /// - `w_delta` ~ |N(0, 0.05 / sqrt(in_features))|  (abs ensures positive perturbation)
    /// - `bias` = 0
    pub fn new(in_features: usize, out_features: usize, rng: &mut LcgRng) -> Self {
        let scale = 0.1 / (in_features as f32).sqrt();
        let delta_scale = 0.05 / (in_features as f32).sqrt();
        let n = out_features * in_features;

        let mut w_mu = vec![0.0_f32; n];
        rng.fill_normal(&mut w_mu);
        for v in w_mu.iter_mut() {
            *v *= scale;
        }

        let mut w_delta = vec![0.0_f32; n];
        rng.fill_normal(&mut w_delta);
        for v in w_delta.iter_mut() {
            // Take absolute value to ensure non-negative perturbation magnitudes
            *v = (*v * delta_scale).abs();
        }

        let bias = vec![0.0_f32; out_features];

        Self {
            in_features,
            out_features,
            w_mu,
            w_delta,
            bias,
        }
    }

    /// Compute the deterministic mean output: `x @ W_mu^T + bias`.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if `x.len() != in_features`.
    pub fn forward_mean(&self, x: &[f32]) -> BayesResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(BayesError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = self.bias.clone();
        for (oc, o) in out.iter_mut().enumerate() {
            for (ic, &xi) in x.iter().enumerate() {
                *o += self.w_mu[oc * self.in_features + ic] * xi;
            }
        }
        Ok(out)
    }

    /// Single stochastic forward pass with Flipout perturbation.
    ///
    /// Samples `r[i] ∈ {-1,+1}` and `s[j] ∈ {-1,+1}` from LCG random bits.
    /// `out = x @ W_mu^T + (x * r) @ W_delta^T * s + bias`.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if `x.len() != in_features`.
    pub fn forward(&self, x: &[f32], rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(BayesError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }

        // Sample r[i] ∈ {-1, +1} for each input
        let r: Vec<f32> = (0..self.in_features)
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();

        // Sample s[j] ∈ {-1, +1} for each output
        let s: Vec<f32> = (0..self.out_features)
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();

        // x_perturbed[i] = x[i] * r[i]
        let x_perturbed: Vec<f32> = x.iter().zip(r.iter()).map(|(&xi, &ri)| xi * ri).collect();

        // mean output
        let mut out = self.forward_mean(x)?;

        // perturbation: (x_perturbed) @ W_delta^T * s
        for (oc, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (ic, &xpi) in x_perturbed.iter().enumerate() {
                acc += self.w_delta[oc * self.in_features + ic] * xpi;
            }
            *o += acc * s[oc];
        }

        Ok(out)
    }

    /// Multiple independent stochastic forward passes.
    ///
    /// Returns `n_samples` independent output vectors.
    ///
    /// # Errors
    /// Returns `BayesError::InsufficientSamples` if `n_samples == 0`,
    /// or propagates errors from `forward`.
    pub fn forward_samples(
        &self,
        x: &[f32],
        n_samples: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Vec<Vec<f32>>> {
        if n_samples == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }
        (0..n_samples).map(|_| self.forward(x, rng)).collect()
    }
}

/// Flipout 2D convolutional layer.
///
/// Analogous to `FlipoutLinear` but applies to feature maps.
#[derive(Debug, Clone)]
pub struct FlipoutConv2d {
    /// Number of output channels.
    pub out_channels: usize,
    /// Number of input channels.
    pub in_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Mean weights `[out_channels × in_channels × kH × kW]`.
    pub w_mu: Vec<f32>,
    /// Perturbation weights, same shape as `w_mu`.
    pub w_delta: Vec<f32>,
    /// Bias `[out_channels]`.
    pub bias: Vec<f32>,
}

impl FlipoutConv2d {
    /// Number of weight parameters.
    #[must_use]
    pub fn n_weights(&self) -> usize {
        self.out_channels * self.in_channels * self.kernel_h * self.kernel_w
    }

    /// Create a new FlipoutConv2d with random initialization.
    pub fn new(
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let n = out_channels * in_channels * kernel_h * kernel_w;
        let fan_in = (in_channels * kernel_h * kernel_w) as f32;
        let scale = 0.1 / fan_in.sqrt();

        let mut w_mu = vec![0.0_f32; n];
        rng.fill_normal(&mut w_mu);
        for v in w_mu.iter_mut() {
            *v *= scale;
        }

        let mut w_delta = vec![0.0_f32; n];
        rng.fill_normal(&mut w_delta);
        for v in w_delta.iter_mut() {
            *v = (*v * scale * 0.5).abs();
        }

        Self {
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            w_mu,
            w_delta,
            bias: vec![0.0_f32; out_channels],
        }
    }

    /// Single stochastic forward pass with Flipout perturbation on feature maps.
    ///
    /// Input: `[C_in × H × W]`.
    /// Output: `[C_out × H_out × W_out]` (stride=1, no padding).
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if input has wrong size.
    pub fn forward(
        &self,
        input: &[f32],
        input_h: usize,
        input_w: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Vec<f32>> {
        let expected = self.in_channels * input_h * input_w;
        if input.len() != expected {
            return Err(BayesError::DimensionMismatch {
                expected,
                got: input.len(),
            });
        }

        let out_h = input_h.saturating_sub(self.kernel_h) + 1;
        let out_w = input_w.saturating_sub(self.kernel_w) + 1;
        let out_size = self.out_channels * out_h * out_w;

        // Sample r for each input element, s for each output element
        let r: Vec<f32> = (0..input.len())
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let s: Vec<f32> = (0..out_size)
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();

        let mut out = vec![0.0_f32; out_size];

        for oc in 0..self.out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc_mu = self.bias[oc];
                    let mut acc_delta = 0.0_f32;
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh + kh;
                                let iw = ow + kw;
                                let in_idx = ic * input_h * input_w + ih * input_w + iw;
                                let w_idx = oc * self.in_channels * self.kernel_h * self.kernel_w
                                    + ic * self.kernel_h * self.kernel_w
                                    + kh * self.kernel_w
                                    + kw;
                                acc_mu += self.w_mu[w_idx] * input[in_idx];
                                acc_delta += self.w_delta[w_idx] * input[in_idx] * r[in_idx];
                            }
                        }
                    }
                    let out_idx = oc * out_h * out_w + oh * out_w + ow;
                    out[out_idx] = acc_mu + acc_delta * s[out_idx];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flipout_linear_forward_shape() {
        let mut rng = LcgRng::new(42);
        let layer = FlipoutLinear::new(8, 4, &mut rng);
        let x = vec![1.0_f32; 8];
        let out = layer
            .forward(&x, &mut rng)
            .expect("test invariant: forward must succeed");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn flipout_linear_forward_samples_count() {
        let mut rng = LcgRng::new(7);
        let layer = FlipoutLinear::new(4, 2, &mut rng);
        let x = vec![0.5_f32; 4];
        let samples = layer
            .forward_samples(&x, 10, &mut rng)
            .expect("test invariant: forward_samples must succeed");
        assert_eq!(samples.len(), 10);
        for s in &samples {
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn flipout_linear_dim_mismatch() {
        let mut rng = LcgRng::new(1);
        let layer = FlipoutLinear::new(4, 2, &mut rng);
        assert!(layer.forward(&[0.0; 3], &mut rng).is_err());
    }

    #[test]
    fn flipout_conv2d_forward_shape() {
        let mut rng = LcgRng::new(13);
        let conv = FlipoutConv2d::new(2, 1, 3, 3, &mut rng);
        let input = vec![1.0_f32; 25]; // 1×5×5
        let out = conv
            .forward(&input, 5, 5, &mut rng)
            .expect("test invariant: FlipoutConv2d::forward must succeed");
        assert_eq!(out.len(), 18); // 2*(5-3+1)*(5-3+1) = 2*3*3=18
    }

    #[test]
    fn flipout_linear_zero_samples_error() {
        let mut rng = LcgRng::new(0);
        let layer = FlipoutLinear::new(4, 2, &mut rng);
        let x = vec![0.0_f32; 4];
        assert!(layer.forward_samples(&x, 0, &mut rng).is_err());
    }
}
