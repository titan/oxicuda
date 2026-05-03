//! Bayesian 2D convolution layer via Bayes-by-Backprop (BBB).
//!
//! Per-filter mean and log-variance parameters; reparameterization trick
//! for stochastic forward passes.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;
use crate::layers::bayes_linear::softplus;
use crate::variational::reparam::gaussian_sample;

/// Bayesian 2D convolution layer.
///
/// Weight shape: `[out_channels × in_channels × kH × kW]`.
/// For each parameter w_i: q(w_i) = N(w_mu_i, softplus(w_rho_i)²).
/// Prior: p(w) = N(0, prior_sigma²).
#[derive(Debug, Clone)]
pub struct BayesConv2d {
    /// Number of output channels (filters).
    pub out_channels: usize,
    /// Number of input channels.
    pub in_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Weight means, shape `[out_channels × in_channels × kH × kW]`.
    pub w_mu: Vec<f32>,
    /// Weight rho (σ = softplus(ρ)), same shape as `w_mu`.
    pub w_log_var: Vec<f32>,
    /// Bias means, shape `[out_channels]`.
    pub b_mu: Vec<f32>,
    /// Bias rho, shape `[out_channels]`.
    pub b_log_var: Vec<f32>,
    /// Prior standard deviation.
    pub prior_sigma: f32,
}

impl BayesConv2d {
    /// Total number of weight parameters.
    #[must_use]
    pub fn n_weights(&self) -> usize {
        self.out_channels * self.in_channels * self.kernel_h * self.kernel_w
    }

    /// Create a new BayesConv2d layer.
    ///
    /// Initializes w_mu ~ N(0, 0.05), w_rho = -4.0.
    ///
    /// # Errors
    /// Returns `BayesError::InvalidPriorVariance` if `prior_sigma <= 0`.
    pub fn new(
        out_channels: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        prior_sigma: f32,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if prior_sigma <= 0.0 || !prior_sigma.is_finite() {
            return Err(BayesError::InvalidPriorVariance);
        }
        let n = out_channels * in_channels * kernel_h * kernel_w;
        let mut w_mu = vec![0.0_f32; n];
        rng.fill_normal(&mut w_mu);
        for v in w_mu.iter_mut() {
            *v *= 0.05;
        }
        let w_log_var = vec![-4.0_f32; n];
        let mut b_mu = vec![0.0_f32; out_channels];
        rng.fill_normal(&mut b_mu);
        for v in b_mu.iter_mut() {
            *v *= 0.01;
        }
        let b_log_var = vec![-4.0_f32; out_channels];
        Ok(Self {
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            w_mu,
            w_log_var,
            b_mu,
            b_log_var,
            prior_sigma,
        })
    }

    /// Sample weights from q(W) and apply 2D convolution.
    ///
    /// Input: `[C_in × H × W]` flattened as row-major.
    /// Output: `[C_out × H_out × W_out]` flattened, with zero-padding, stride=1.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if input shape is inconsistent.
    pub fn forward_sample(
        &self,
        input: &[f32],
        input_h: usize,
        input_w: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Vec<f32>> {
        let expected_input = self.in_channels * input_h * input_w;
        if input.len() != expected_input {
            return Err(BayesError::DimensionMismatch {
                expected: expected_input,
                got: input.len(),
            });
        }

        // Output spatial dimensions with same-padding (floor): stride=1, no pad
        let out_h = input_h.saturating_sub(self.kernel_h) + 1;
        let out_w = input_w.saturating_sub(self.kernel_w) + 1;
        let out_size = self.out_channels * out_h * out_w;
        let mut out = vec![0.0_f32; out_size];

        // Sample all weights at once
        let mut w_sampled = vec![0.0_f32; self.n_weights()];
        for (i, (&mu, &lv)) in self.w_mu.iter().zip(self.w_log_var.iter()).enumerate() {
            w_sampled[i] = gaussian_sample(mu, lv, rng);
        }

        // Conv2D with sampled weights
        for oc in 0..self.out_channels {
            // Sample bias
            let b = self.b_mu[oc] + gaussian_sample(0.0, self.b_log_var[oc], rng);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = b;
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
                                acc += w_sampled[w_idx] * input[in_idx];
                            }
                        }
                    }
                    out[oc * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }
        Ok(out)
    }

    /// Deterministic forward pass using mean weights.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if input shape is inconsistent.
    pub fn forward_mean(
        &self,
        input: &[f32],
        input_h: usize,
        input_w: usize,
    ) -> BayesResult<Vec<f32>> {
        let expected_input = self.in_channels * input_h * input_w;
        if input.len() != expected_input {
            return Err(BayesError::DimensionMismatch {
                expected: expected_input,
                got: input.len(),
            });
        }
        let out_h = input_h.saturating_sub(self.kernel_h) + 1;
        let out_w = input_w.saturating_sub(self.kernel_w) + 1;
        let out_size = self.out_channels * out_h * out_w;
        let mut out = vec![0.0_f32; out_size];

        for oc in 0..self.out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = self.b_mu[oc];
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
                                acc += self.w_mu[w_idx] * input[in_idx];
                            }
                        }
                    }
                    out[oc * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }
        Ok(out)
    }

    /// KL divergence KL(q(W) ‖ p(W)) summed over all parameters.
    ///
    /// # Errors
    /// Returns `BayesError::NanEncountered` if computation produces non-finite values.
    pub fn kl_divergence(&self) -> BayesResult<f32> {
        let prior_var = self.prior_sigma * self.prior_sigma;
        let log_prior_sigma = self.prior_sigma.ln();
        let mut kl = 0.0_f32;

        for (&mu, &rho) in self.w_mu.iter().zip(self.w_log_var.iter()) {
            let sigma = softplus(rho);
            let sigma_sq = sigma * sigma;
            let log_sigma = sigma.ln();
            kl += log_prior_sigma - log_sigma + (sigma_sq + mu * mu) / (2.0 * prior_var) - 0.5;
        }
        for (&mu, &rho) in self.b_mu.iter().zip(self.b_log_var.iter()) {
            let sigma = softplus(rho);
            let sigma_sq = sigma * sigma;
            let log_sigma = sigma.ln();
            kl += log_prior_sigma - log_sigma + (sigma_sq + mu * mu) / (2.0 * prior_var) - 0.5;
        }

        if !kl.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "BayesConv2d::kl_divergence: non-finite result",
            });
        }
        Ok(kl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayes_conv2d_new_valid() {
        let mut rng = LcgRng::new(42);
        let conv = BayesConv2d::new(2, 1, 3, 3, 1.0, &mut rng)
            .expect("test invariant: BayesConv2d::new must succeed");
        assert_eq!(conv.n_weights(), 18);
    }

    #[test]
    fn bayes_conv2d_forward_mean_shape() {
        let mut rng = LcgRng::new(1);
        let conv = BayesConv2d::new(2, 1, 3, 3, 1.0, &mut rng)
            .expect("test invariant: BayesConv2d::new must succeed");
        let input = vec![0.0_f32; 25]; // 1×5×5
        let out = conv
            .forward_mean(&input, 5, 5)
            .expect("test invariant: forward_mean must succeed");
        // out_h = 5-3+1=3, out_w=3, 2 channels → 2*3*3=18
        assert_eq!(out.len(), 18);
    }

    #[test]
    fn bayes_conv2d_kl_positive() {
        let mut rng = LcgRng::new(9);
        let conv = BayesConv2d::new(2, 1, 3, 3, 1.0, &mut rng)
            .expect("test invariant: BayesConv2d::new must succeed");
        let kl = conv
            .kl_divergence()
            .expect("test invariant: kl_divergence must succeed");
        assert!(kl >= 0.0, "KL must be non-negative, got {kl}");
    }

    #[test]
    fn bayes_conv2d_invalid_prior() {
        let mut rng = LcgRng::new(0);
        assert!(BayesConv2d::new(2, 1, 3, 3, 0.0, &mut rng).is_err());
    }
}
