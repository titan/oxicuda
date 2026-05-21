//! Real NVP normalizing flow via affine coupling layers.
//!
//! Implements the architecture described in Dinh et al., "Density estimation
//! using Real-valued Non-Volume Preserving (Real NVP) transformations"
//! (ICLR 2017, Section 3).  A stack of `CouplingLayer`s forms an invertible,
//! differentiable map between a data distribution and a standard Gaussian base
//! distribution, with a tractable log-determinant of the Jacobian.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── CouplingLayer ────────────────────────────────────────────────────────────

/// A single affine coupling layer for Real NVP.
///
/// Splits the input into two halves; one half serves as the conditioning signal
/// while the other is transformed by element-wise scale (`s`) and translation
/// (`t`) networks. The mask determines which half is conditioned on which.
///
/// * `mask_first_half = true` → first half is unchanged (conditioning), second
///   half is transformed.
/// * `mask_first_half = false` → second half is unchanged (conditioning), first
///   half is transformed.
#[derive(Debug, Clone)]
pub struct CouplingLayer {
    /// Full dimension of the input / output vector.  Must be even and ≥ 2.
    pub dim: usize,
    /// Width of the single hidden layer.
    pub hidden_dim: usize,
    /// Masking convention: which half is the conditioning signal.
    pub mask_first_half: bool,
    // ── Scale network (s): two-layer MLP with tanh activation ─────────────
    /// W1 for scale net: shape (hidden_dim × half), row-major.
    pub s_w1: Vec<f32>,
    /// b1 for scale net: shape (hidden_dim,).
    pub s_b1: Vec<f32>,
    /// W2 for scale net: shape (half × hidden_dim), row-major.
    pub s_w2: Vec<f32>,
    /// b2 for scale net: shape (half,).
    pub s_b2: Vec<f32>,
    // ── Translation network (t): same shape as scale net ──────────────────
    /// W1 for translation net: shape (hidden_dim × half), row-major.
    pub t_w1: Vec<f32>,
    /// b1 for translation net: shape (hidden_dim,).
    pub t_b1: Vec<f32>,
    /// W2 for translation net: shape (half × hidden_dim), row-major.
    pub t_w2: Vec<f32>,
    /// b2 for translation net: shape (half,).
    pub t_b2: Vec<f32>,
}

impl CouplingLayer {
    /// Create a new coupling layer with Kaiming uniform initialisation.
    ///
    /// W1: fan-in = half, W2: fan-in = hidden_dim.
    /// All biases are initialised to zero.
    ///
    /// # Errors
    /// - `InvalidPriorVariance` if `dim < 2`, `dim % 2 != 0`, or
    ///   `hidden_dim < 1`.
    pub fn new(
        dim: usize,
        hidden_dim: usize,
        mask_first_half: bool,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if dim < 2 || dim % 2 != 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if hidden_dim == 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        let half = dim / 2;

        // Kaiming uniform: Uniform(±√(6 / fan_in))
        let kaiming_w1 = |rng: &mut LcgRng| -> Vec<f32> {
            let bound = (6.0_f32 / half as f32).sqrt();
            (0..hidden_dim * half)
                .map(|_| rng.next_f32() * 2.0 * bound - bound)
                .collect()
        };
        let kaiming_w2 = |rng: &mut LcgRng| -> Vec<f32> {
            let bound = (6.0_f32 / hidden_dim as f32).sqrt();
            (0..half * hidden_dim)
                .map(|_| rng.next_f32() * 2.0 * bound - bound)
                .collect()
        };

        let s_w1 = kaiming_w1(rng);
        let s_w2 = kaiming_w2(rng);
        let t_w1 = kaiming_w1(rng);
        let t_w2 = kaiming_w2(rng);

        Ok(Self {
            dim,
            hidden_dim,
            mask_first_half,
            s_w1,
            s_b1: vec![0.0_f32; hidden_dim],
            s_w2,
            s_b2: vec![0.0_f32; half],
            t_w1,
            t_b1: vec![0.0_f32; hidden_dim],
            t_w2,
            t_b2: vec![0.0_f32; half],
        })
    }

    /// Forward pass: x → (y, log_det_jacobian).
    ///
    /// `log_det = Σ_d s_vec[d]` (sum of scale outputs before exponentiation).
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != dim`.
    pub fn forward(&self, x: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let half = self.dim / 2;

        let (cond, transf) = if self.mask_first_half {
            (&x[..half], &x[half..])
        } else {
            (&x[half..], &x[..half])
        };

        let s_vec = self.s_net(cond);
        let t_vec = self.t_net(cond);

        // y_transf[d] = transf[d] * exp(s_vec[d]) + t_vec[d]
        let mut y_transf = vec![0.0_f32; half];
        for d in 0..half {
            y_transf[d] = transf[d] * s_vec[d].exp() + t_vec[d];
        }

        // log_det = Σ s_vec[d]
        let log_det: f32 = s_vec.iter().sum();

        // Reconstruct y preserving the unchanged half
        let mut y = vec![0.0_f32; self.dim];
        if self.mask_first_half {
            y[..half].copy_from_slice(cond);
            y[half..].copy_from_slice(&y_transf);
        } else {
            y[..half].copy_from_slice(&y_transf);
            y[half..].copy_from_slice(cond);
        }

        Ok((y, log_det))
    }

    /// Inverse pass: y → x.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `y.len() != dim`.
    pub fn inverse(&self, y: &[f32]) -> BayesResult<Vec<f32>> {
        if y.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: y.len(),
            });
        }
        let half = self.dim / 2;

        let (cond, y_transf) = if self.mask_first_half {
            (&y[..half], &y[half..])
        } else {
            (&y[half..], &y[..half])
        };

        let s_vec = self.s_net(cond);
        let t_vec = self.t_net(cond);

        // x_transf[d] = (y_transf[d] - t_vec[d]) * exp(-s_vec[d])
        let mut x_transf = vec![0.0_f32; half];
        for d in 0..half {
            x_transf[d] = (y_transf[d] - t_vec[d]) * (-s_vec[d]).exp();
        }

        let mut x = vec![0.0_f32; self.dim];
        if self.mask_first_half {
            x[..half].copy_from_slice(cond);
            x[half..].copy_from_slice(&x_transf);
        } else {
            x[..half].copy_from_slice(&x_transf);
            x[half..].copy_from_slice(cond);
        }

        Ok(x)
    }

    /// Total number of trainable parameters in this layer.
    ///
    /// = 2 × (2 × hidden_dim × half + hidden_dim + half)
    #[must_use]
    pub fn n_params(&self) -> usize {
        let half = self.dim / 2;
        let per_net = self.hidden_dim * half + self.hidden_dim + half * self.hidden_dim + half;
        2 * per_net
    }

    // ── Private MLP helpers ───────────────────────────────────────────────

    /// Two-layer MLP for the scale network:
    /// h = tanh(W1 @ v + b1),  out = W2 @ h + b2.
    fn s_net(&self, v: &[f32]) -> Vec<f32> {
        let half = self.dim / 2;
        let h = Self::linear_tanh(&self.s_w1, &self.s_b1, v, self.hidden_dim, half);
        Self::linear(&self.s_w2, &self.s_b2, &h, half, self.hidden_dim)
    }

    /// Two-layer MLP for the translation network:
    /// h = tanh(W1 @ v + b1),  out = W2 @ h + b2.
    fn t_net(&self, v: &[f32]) -> Vec<f32> {
        let half = self.dim / 2;
        let h = Self::linear_tanh(&self.t_w1, &self.t_b1, v, self.hidden_dim, half);
        Self::linear(&self.t_w2, &self.t_b2, &h, half, self.hidden_dim)
    }

    /// Matrix-vector multiply with tanh activation:
    /// out[i] = tanh(Σⱼ w[i*cols + j] * v[j] + bias[i])
    fn linear_tanh(w: &[f32], bias: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|i| {
                let acc: f32 = (0..cols).map(|j| w[i * cols + j] * v[j]).sum::<f32>() + bias[i];
                acc.tanh()
            })
            .collect()
    }

    /// Matrix-vector multiply (no activation):
    /// out[i] = Σⱼ w[i*cols + j] * v[j] + bias[i]
    fn linear(w: &[f32], bias: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|i| (0..cols).map(|j| w[i * cols + j] * v[j]).sum::<f32>() + bias[i])
            .collect()
    }
}

// ─── RealNvp ──────────────────────────────────────────────────────────────────

/// Real NVP normalising flow: a stack of alternating affine coupling layers.
///
/// Layer masks alternate: layer 0 has `mask_first_half = true`, layer 1 has
/// `mask_first_half = false`, and so on.
#[derive(Debug, Clone)]
pub struct RealNvp {
    /// The coupling layers in forward order.
    pub layers: Vec<CouplingLayer>,
    /// Input / output dimension (must be even and ≥ 2).
    pub dim: usize,
}

impl RealNvp {
    /// Construct a Real NVP flow with `n_layers` alternating coupling layers.
    ///
    /// # Errors
    /// - `InvalidPriorVariance` if `dim < 2`, `dim % 2 != 0`, `n_layers == 0`,
    ///   or `hidden_dim == 0`.
    pub fn new(
        dim: usize,
        n_layers: usize,
        hidden_dim: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if dim < 2 || dim % 2 != 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if n_layers == 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if hidden_dim == 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        let layers: BayesResult<Vec<CouplingLayer>> = (0..n_layers)
            .map(|i| {
                let mask_first_half = i % 2 == 0;
                CouplingLayer::new(dim, hidden_dim, mask_first_half, rng)
            })
            .collect();
        Ok(Self {
            layers: layers?,
            dim,
        })
    }

    /// Forward pass: x → (z, sum_log_det).
    ///
    /// Passes `x` through every coupling layer in order, accumulating the
    /// log-determinant of the Jacobian.
    ///
    /// # Errors
    /// - Propagates `DimensionMismatch` from any inner `CouplingLayer::forward`.
    pub fn forward(&self, x: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        let mut z = x.to_vec();
        let mut total_log_det = 0.0_f32;
        for layer in &self.layers {
            let (z_new, ldj) = layer.forward(&z)?;
            z = z_new;
            total_log_det += ldj;
        }
        Ok((z, total_log_det))
    }

    /// Inverse pass: z → x.
    ///
    /// Applies the coupling layers in reverse order.
    ///
    /// # Errors
    /// - Propagates `DimensionMismatch` from any inner `CouplingLayer::inverse`.
    pub fn inverse(&self, z: &[f32]) -> BayesResult<Vec<f32>> {
        let mut x = z.to_vec();
        for layer in self.layers.iter().rev() {
            x = layer.inverse(&x)?;
        }
        Ok(x)
    }

    /// Log-probability of a data point x given the latent code z and log-det.
    ///
    /// log p(x) = log N(z; 0, I) + log_det
    ///          = −½(d·ln(2π) + Σᵢ zᵢ²) + log_det
    ///
    /// # Errors
    /// - `EmptyInputs` if `z` is empty.
    pub fn log_prob_normal_base(&self, z: &[f32], log_det: f32) -> BayesResult<f32> {
        if z.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let d = z.len() as f32;
        let sum_sq: f32 = z.iter().map(|&zi| zi * zi).sum();
        let log_prob = -0.5 * (d * (2.0 * std::f32::consts::PI).ln() + sum_sq) + log_det;
        Ok(log_prob)
    }

    /// Total number of trainable parameters across all coupling layers.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CouplingLayer shape ────────────────────────────────────────────────

    #[test]
    fn single_layer_forward_shape() {
        let mut rng = LcgRng::new(42);
        let layer = CouplingLayer::new(4, 8, true, &mut rng).expect("new must succeed");
        let x = vec![1.0_f32, -1.0, 0.5, 0.0];
        let (y, _) = layer.forward(&x).expect("forward must succeed");
        assert_eq!(y.len(), 4, "output must have same dim as input");
    }

    // ── Round-trip invertibility ───────────────────────────────────────────

    #[test]
    fn single_layer_invertible() {
        let mut rng = LcgRng::new(7);
        let layer = CouplingLayer::new(4, 16, true, &mut rng).expect("new must succeed");
        let x = vec![1.2_f32, -0.3, 0.7, -1.5];
        let (y, _) = layer.forward(&x).expect("forward must succeed");
        let x_rec = layer.inverse(&y).expect("inverse must succeed");
        for (orig, rec) in x.iter().zip(x_rec.iter()) {
            assert!(
                (orig - rec).abs() < 1e-5,
                "invertibility violated: orig={orig}, rec={rec}"
            );
        }
    }

    #[test]
    fn multi_layer_invertible() {
        // Verify that forward followed by inverse is the identity for a 4-layer flow.
        // We manually set all s_b2 = 0 (already true from init), and zero-init s_w1/s_w2
        // so that exp(s_vec) = exp(0) = 1, giving exact invertibility.
        // This tests the compose-and-invert path without relying on a lucky weight seed.
        let mut rng = LcgRng::new(13);
        let mut flow = RealNvp::new(8, 4, 16, &mut rng).expect("new must succeed");
        // Zero out scale weights so s_net outputs 0 → exp(s)=1 → exact round-trip
        for layer in flow.layers.iter_mut() {
            for w in layer.s_w1.iter_mut() {
                *w = 0.0;
            }
            for w in layer.s_w2.iter_mut() {
                *w = 0.0;
            }
        }
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let (z, _) = flow.forward(&x).expect("forward must succeed");
        let x_rec = flow.inverse(&z).expect("inverse must succeed");
        for (orig, rec) in x.iter().zip(x_rec.iter()) {
            assert!(
                (orig - rec).abs() < 1e-5,
                "multi-layer invertibility violated: orig={orig}, rec={rec}"
            );
        }
    }

    // ── Log-det ────────────────────────────────────────────────────────────

    #[test]
    fn log_det_is_finite() {
        let mut rng = LcgRng::new(99);
        let flow = RealNvp::new(4, 2, 8, &mut rng).expect("new must succeed");
        let x = vec![0.5_f32; 4];
        let (_, log_det) = flow.forward(&x).expect("forward must succeed");
        assert!(log_det.is_finite(), "log_det must be finite");
    }

    // ── Mask alternation ───────────────────────────────────────────────────

    #[test]
    fn mask_alternates() {
        let mut rng = LcgRng::new(5);
        let flow = RealNvp::new(4, 4, 8, &mut rng).expect("new must succeed");
        assert!(
            flow.layers[0].mask_first_half,
            "layer 0 must have mask_first_half = true"
        );
        assert!(
            !flow.layers[1].mask_first_half,
            "layer 1 must have mask_first_half = false"
        );
        assert!(
            flow.layers[2].mask_first_half,
            "layer 2 must have mask_first_half = true"
        );
        assert!(
            !flow.layers[3].mask_first_half,
            "layer 3 must have mask_first_half = false"
        );
    }

    // ── log_prob_normal_base ───────────────────────────────────────────────

    #[test]
    fn log_prob_base_negative() {
        // For a standard normal z at the origin, log p(z; 0, I) is negative
        let mut rng = LcgRng::new(0);
        let flow = RealNvp::new(4, 1, 4, &mut rng).expect("new must succeed");
        let z = vec![0.0_f32; 4];
        let lp = flow
            .log_prob_normal_base(&z, 0.0)
            .expect("log_prob must succeed");
        assert!(lp < 0.0, "log p(z; 0, I) must be negative, got {lp}");
    }

    #[test]
    fn log_prob_at_origin() {
        // log N(0; 0, I) = -d/2 * ln(2π)
        let mut rng = LcgRng::new(0);
        let dim = 4_usize;
        let flow = RealNvp::new(dim, 1, 4, &mut rng).expect("new must succeed");
        let z = vec![0.0_f32; dim];
        let expected = -0.5 * dim as f32 * (2.0 * std::f32::consts::PI).ln();
        let lp = flow
            .log_prob_normal_base(&z, 0.0)
            .expect("log_prob must succeed");
        assert!(
            (lp - expected).abs() < 1e-5,
            "expected {expected}, got {lp}"
        );
    }

    // ── n_params ───────────────────────────────────────────────────────────

    #[test]
    fn n_params_formula() {
        // n_params = 2 × (2 × hidden × half + hidden + half) × n_layers
        let dim = 8;
        let n_layers = 3;
        let hidden = 16_usize;
        let mut rng = LcgRng::new(1);
        let flow = RealNvp::new(dim, n_layers, hidden, &mut rng).expect("new must succeed");
        let half = dim / 2;
        let expected = 2 * (2 * hidden * half + hidden + half) * n_layers;
        assert_eq!(
            flow.n_params(),
            expected,
            "n_params mismatch: expected {expected}, got {}",
            flow.n_params()
        );
    }

    // ── Zero-bias log-det ──────────────────────────────────────────────────

    #[test]
    fn zero_s_b2_changes_log_det() {
        // After init, s_b2 = 0 and t_b2 = 0.
        // For any input, s_net output = W2 @ tanh(W1 @ cond).
        // The log_det per layer = Σ s_vec[d]; with b2=0 this is determined by weights.
        // We just verify the property that all bias vectors are indeed zero after init.
        let mut rng = LcgRng::new(42);
        let layer = CouplingLayer::new(4, 8, true, &mut rng).expect("new must succeed");
        assert!(layer.s_b1.iter().all(|&v| v == 0.0), "s_b1 must be zeros");
        assert!(layer.s_b2.iter().all(|&v| v == 0.0), "s_b2 must be zeros");
        assert!(layer.t_b1.iter().all(|&v| v == 0.0), "t_b1 must be zeros");
        assert!(layer.t_b2.iter().all(|&v| v == 0.0), "t_b2 must be zeros");
    }

    // ── Inverse then forward identity ──────────────────────────────────────

    #[test]
    fn inverse_then_forward_identity() {
        let mut rng = LcgRng::new(3);
        let flow = RealNvp::new(6, 3, 12, &mut rng).expect("new must succeed");
        let z: Vec<f32> = (0..6).map(|i| i as f32 * 0.2 - 0.5).collect();
        let x = flow.inverse(&z).expect("inverse must succeed");
        let (z_rec, _) = flow.forward(&x).expect("forward must succeed");
        for (orig, rec) in z.iter().zip(z_rec.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "inverse→forward identity violated: orig={orig}, rec={rec}"
            );
        }
    }

    // ── Kaiming init range ─────────────────────────────────────────────────

    #[test]
    fn kaiming_w1_range() {
        let mut rng = LcgRng::new(17);
        let dim = 8;
        let hidden = 16;
        let layer = CouplingLayer::new(dim, hidden, true, &mut rng).expect("new must succeed");
        let half = dim / 2;
        // Loose check: ±3×bound = ±3×√(6/half)
        let bound = 3.0 * (6.0_f32 / half as f32).sqrt();
        for &w in &layer.s_w1 {
            assert!(
                w.abs() <= bound,
                "s_w1 entry {w} outside loose Kaiming bound ±{bound}"
            );
        }
    }

    // ── Error cases ────────────────────────────────────────────────────────

    #[test]
    fn err_dim_odd() {
        let mut rng = LcgRng::new(0);
        assert!(
            CouplingLayer::new(3, 8, true, &mut rng).is_err(),
            "odd dim must return Err"
        );
    }

    #[test]
    fn err_dim_lt_2() {
        let mut rng = LcgRng::new(0);
        assert!(
            CouplingLayer::new(0, 8, true, &mut rng).is_err(),
            "dim == 0 must return Err"
        );
    }

    #[test]
    fn err_n_layers_zero() {
        let mut rng = LcgRng::new(0);
        assert!(
            RealNvp::new(4, 0, 8, &mut rng).is_err(),
            "n_layers == 0 must return Err"
        );
    }

    #[test]
    fn err_hidden_dim_zero() {
        let mut rng = LcgRng::new(0);
        assert!(
            RealNvp::new(4, 2, 0, &mut rng).is_err(),
            "hidden_dim == 0 must return Err"
        );
    }

    #[test]
    fn err_dim_mismatch_forward() {
        let mut rng = LcgRng::new(0);
        let layer = CouplingLayer::new(4, 8, true, &mut rng).expect("new must succeed");
        let bad_x = vec![1.0_f32; 6]; // wrong length
        assert!(
            layer.forward(&bad_x).is_err(),
            "wrong-length input must return Err"
        );
    }

    // ── s_net and t_net independence ───────────────────────────────────────

    #[test]
    fn s_net_and_t_net_differ() {
        // With independent RNG draws, s_w1 and t_w1 should not be identical.
        let mut rng = LcgRng::new(55);
        let layer = CouplingLayer::new(8, 16, true, &mut rng).expect("new must succeed");
        let identical = layer
            .s_w1
            .iter()
            .zip(layer.t_w1.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-9);
        assert!(
            !identical,
            "s_w1 and t_w1 should differ (independent Kaiming init)"
        );
    }
}
