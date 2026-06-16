//! Inverse Autoregressive Flow (IAF) for flexible variational posteriors.
//!
//! Implements Kingma et al. (2016), "Improving Variational Inference with
//! Inverse Autoregressive Flow", NeurIPS 2016.
//!
//! # Key idea
//!
//! IAF transforms a base Gaussian sample `z₀ ~ N(0, I)` through a sequence of
//! invertible autoregressive transformations:
//!
//! ```text
//! z_{t+1} = σ_t ⊙ z_t + (1 − σ_t) ⊙ μ_t
//! ```
//!
//! where `σ_t = sigmoid(s_t)` and `μ_t` are the outputs of an autoregressive
//! network (MADE-style) conditioned on the previous `z_{t}` and a context
//! vector `h` from the encoder.
//!
//! The log-determinant of the Jacobian is:
//!
//! ```text
//! log |det J_t| = Σ_j log σ_{t,j}
//! ```
//!
//! # CPU-only implementation
//!
//! This module provides a pure-CPU reference using masked autoregressive
//! networks with triangular weight masking.  GPU PTX generation is out of
//! scope here; use `variational/real_nvp.rs` as the GPU-path complement.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── MADE-style autoregressive network ────────────────────────────────────────

/// A single-hidden-layer MADE autoregressive network.
///
/// Given input `z ∈ ℝ^d` and optional context `h ∈ ℝ^c`, outputs
/// `(μ, log_s) ∈ ℝ^d × ℝ^d` such that `μ_j` and `log_s_j` depend only on
/// `z_{<j}` (autoregressive constraint enforced by binary masks).
#[derive(Debug, Clone)]
pub struct MadeNet {
    /// Dimensionality of the flow variable.
    pub dim: usize,
    /// Context vector dimensionality (0 = no context).
    pub context_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    // ── Layer 1 (input → hidden): W_h1 ────────────────────────────────────
    /// Weight matrix `[hidden_dim × (dim + context_dim)]`, row-major.
    pub w_h1: Vec<f32>,
    /// Bias `[hidden_dim]`.
    pub b_h1: Vec<f32>,
    // ── Autoregressive mask for W_h1 ──────────────────────────────────────
    /// Binary mask `[hidden_dim × dim]` for the z→hidden path, row-major.
    /// `mask[i, j] = 1` iff `m_h[i] >= j` (MADE ordering `m_h[i]` ≥ j).
    pub mask_h1: Vec<f32>,
    // ── Layer 2 (hidden → μ + log_s): W_out ─────────────────────────────
    /// Weight matrix `[(2*dim) × hidden_dim]`, row-major.
    pub w_out: Vec<f32>,
    /// Bias `[2*dim]`.
    pub b_out: Vec<f32>,
    // ── Autoregressive mask for W_out ─────────────────────────────────────
    /// Binary mask `[(2*dim) × hidden_dim]`, row-major.
    /// `mask[k, i] = 1` iff `m_h[i] < (k % dim)`.
    pub mask_out: Vec<f32>,
}

impl MadeNet {
    /// Construct a new MADE network with Kaiming uniform initialisation.
    ///
    /// The MADE mask is computed using a simple monotone assignment
    /// `m_h[i] = (i % (dim - 1)) + 1` for the hidden units, ensuring
    /// coverage across all autoregressive levels.
    ///
    /// # Errors
    /// - `BayesError::InvalidConfig` if `dim < 2` or `hidden_dim < 1`.
    pub fn new(
        dim: usize,
        context_dim: usize,
        hidden_dim: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if dim < 2 {
            return Err(BayesError::InvalidConfig("IAF dim must be >= 2".into()));
        }
        if hidden_dim == 0 {
            return Err(BayesError::InvalidConfig("hidden_dim must be >= 1".into()));
        }
        let in_dim = dim + context_dim;
        let out_dim = 2 * dim;

        // Kaiming uniform initialisation.
        let kaiming = |fan_in: usize, size: usize, rng: &mut LcgRng| -> Vec<f32> {
            let bound = (6.0_f32 / fan_in as f32).sqrt();
            (0..size)
                .map(|_| rng.next_f32() * 2.0 * bound - bound)
                .collect()
        };

        let w_h1 = kaiming(in_dim, hidden_dim * in_dim, rng);
        let b_h1 = vec![0.0_f32; hidden_dim];
        let w_out = kaiming(hidden_dim, out_dim * hidden_dim, rng);
        let b_out = vec![0.0_f32; out_dim];

        // MADE masks: assign hidden ordering m_h[i] ∈ {1, ..., dim-1}.
        let m_h: Vec<usize> = (0..hidden_dim).map(|i| (i % (dim - 1)) + 1).collect();

        // Mask for W_h1 (hidden × dim portion of the input): hidden[i] gets
        // z[j] only if m_h[i] >= j+1 (1-indexed feature ordering).
        // Context inputs are unmasked (always visible).
        let mut mask_h1 = vec![0.0_f32; hidden_dim * dim];
        for i in 0..hidden_dim {
            for j in 0..dim {
                if m_h[i] > j {
                    mask_h1[i * dim + j] = 1.0;
                }
            }
        }

        // Mask for W_out: output[k] (where k < 2*dim, feature = k % dim)
        // gets hidden[i] only if m_h[i] < feature_k + 1.
        let mut mask_out = vec![0.0_f32; out_dim * hidden_dim];
        for k in 0..out_dim {
            let feat_k = k % dim; // 0-indexed feature
            for i in 0..hidden_dim {
                if m_h[i] < feat_k + 1 {
                    mask_out[k * hidden_dim + i] = 1.0;
                }
            }
        }

        Ok(Self {
            dim,
            context_dim,
            hidden_dim,
            w_h1,
            b_h1,
            w_out,
            b_out,
            mask_h1,
            mask_out,
        })
    }

    /// Forward pass: given `z` and optional `context`, return `(mu, log_s)`.
    ///
    /// Both `z` and the returned vectors have length `dim`.
    /// `context` must have length `context_dim` (or be empty if `context_dim == 0`).
    ///
    /// # Errors
    /// - `BayesError::DimensionMismatch` if lengths disagree.
    /// - `BayesError::NanEncountered` if NaN appears in intermediate results.
    pub fn forward(&self, z: &[f32], context: &[f32]) -> BayesResult<(Vec<f32>, Vec<f32>)> {
        if z.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: z.len(),
            });
        }
        if context.len() != self.context_dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.context_dim,
                got: context.len(),
            });
        }

        // Concatenate [z | context] as input.
        let mut inp = Vec::with_capacity(self.dim + self.context_dim);
        inp.extend_from_slice(z);
        inp.extend_from_slice(context);
        let in_dim = inp.len();

        // Hidden layer: h = tanh(W_h1 ⊙_mask · [z | ctx] + b_h1).
        // For the z-portion of the input we apply the mask; for context, no mask.
        let mut h = vec![0.0_f32; self.hidden_dim];
        for (i, h_i) in h.iter_mut().enumerate() {
            let mut acc = self.b_h1[i];
            // z-inputs (columns 0..dim): masked.
            let z_slice = &inp[..self.dim];
            let w_row = &self.w_h1[i * in_dim..i * in_dim + self.dim];
            let m_row = &self.mask_h1[i * self.dim..(i + 1) * self.dim];
            for j in 0..self.dim {
                acc += w_row[j] * m_row[j] * z_slice[j];
            }
            // Context inputs (columns dim..): unmasked.
            let ctx_slice = &inp[self.dim..];
            let w_ctx = &self.w_h1[i * in_dim + self.dim..i * in_dim + in_dim];
            for (wj, xj) in w_ctx.iter().zip(ctx_slice.iter()) {
                acc += wj * xj;
            }
            *h_i = acc.tanh();
        }

        // Output layer: out = W_out ⊙_mask · h + b_out.
        let out_dim = 2 * self.dim;
        let mut out = vec![0.0_f32; out_dim];
        for (k, out_k) in out.iter_mut().enumerate() {
            let mut acc = self.b_out[k];
            let w_row = &self.w_out[k * self.hidden_dim..(k + 1) * self.hidden_dim];
            let m_row = &self.mask_out[k * self.hidden_dim..(k + 1) * self.hidden_dim];
            for ((wj, mj), hj) in w_row.iter().zip(m_row.iter()).zip(h.iter()) {
                acc += wj * mj * hj;
            }
            *out_k = acc;
        }

        if out.iter().any(|v| v.is_nan()) {
            return Err(BayesError::NanEncountered {
                location: "MadeNet::forward",
            });
        }

        let mu = out[..self.dim].to_vec();
        let log_s = out[self.dim..].to_vec();
        Ok((mu, log_s))
    }
}

// ─── IafStep ──────────────────────────────────────────────────────────────────

/// A single IAF transformation step.
///
/// Applies:
/// ```text
/// z' = σ ⊙ z + (1 − σ) ⊙ μ,   σ = sigmoid(log_s + gate_bias)
/// ```
/// and tracks the accumulated log-determinant.
#[derive(Debug, Clone)]
pub struct IafStep {
    /// Underlying MADE network.
    pub made: MadeNet,
    /// Scalar gate bias added to `log_s` for stability (typically 2.0).
    pub gate_bias: f32,
}

impl IafStep {
    /// Construct a new IAF step.
    ///
    /// # Errors
    /// Propagates errors from [`MadeNet::new`].
    pub fn new(
        dim: usize,
        context_dim: usize,
        hidden_dim: usize,
        gate_bias: f32,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        Ok(Self {
            made: MadeNet::new(dim, context_dim, hidden_dim, rng)?,
            gate_bias,
        })
    }

    /// Apply the IAF transformation to `z`.
    ///
    /// Returns `(z_new, log_det)` where `log_det = Σ log σ_j`.
    ///
    /// # Errors
    /// Propagates errors from [`MadeNet::forward`].
    pub fn forward(&self, z: &[f32], context: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        let (mu, log_s) = self.made.forward(z, context)?;
        let dim = z.len();
        let mut z_new = vec![0.0_f32; dim];
        let mut log_det = 0.0_f32;
        for j in 0..dim {
            let sigma = sigmoid(log_s[j] + self.gate_bias);
            z_new[j] = sigma * z[j] + (1.0 - sigma) * mu[j];
            log_det += sigma.max(1e-10).ln();
        }
        Ok((z_new, log_det))
    }
}

// ─── IafFlow ──────────────────────────────────────────────────────────────────

/// A multi-step Inverse Autoregressive Flow.
///
/// Applies `n_steps` IAF transformations sequentially.  Each step uses the
/// same context vector `h` from the encoder.  Between steps the dimensions of
/// `z` are *permuted* (reversed) to allow all dimensions to influence each
/// other after one full pass.
#[derive(Debug, Clone)]
pub struct IafFlow {
    /// Ordered list of IAF steps.
    pub steps: Vec<IafStep>,
}

impl IafFlow {
    /// Create a new IAF flow with `n_steps` steps.
    ///
    /// # Errors
    /// - `BayesError::InvalidConfig` if `n_steps == 0`.
    /// - Propagates errors from [`IafStep::new`].
    pub fn new(
        n_steps: usize,
        dim: usize,
        context_dim: usize,
        hidden_dim: usize,
        gate_bias: f32,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if n_steps == 0 {
            return Err(BayesError::InvalidConfig("IAF n_steps must be >= 1".into()));
        }
        let steps = (0..n_steps)
            .map(|_| IafStep::new(dim, context_dim, hidden_dim, gate_bias, rng))
            .collect::<BayesResult<Vec<_>>>()?;
        Ok(Self { steps })
    }

    /// Forward pass: transform `z₀` through all IAF steps.
    ///
    /// Returns `(z_T, log_det_total)` where `log_det_total = Σ_t log_det_t`.
    ///
    /// # Errors
    /// - `BayesError::DimensionMismatch` if `z.len()` does not match flow dim.
    /// - `BayesError::NanEncountered` if NaN appears.
    pub fn forward(&self, z0: &[f32], context: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        let mut z = z0.to_vec();
        let mut log_det_total = 0.0_f32;
        for (t, step) in self.steps.iter().enumerate() {
            let (z_new, ld) = step.forward(&z, context)?;
            if z_new.iter().any(|v| v.is_nan()) || ld.is_nan() {
                return Err(BayesError::NanEncountered {
                    location: "IafFlow::forward",
                });
            }
            z = z_new;
            log_det_total += ld;
            // Permute (reverse) dimensions between steps to improve mixing.
            if t + 1 < self.steps.len() {
                z.reverse();
            }
        }
        Ok((z, log_det_total))
    }

    /// Sample from the IAF posterior: first draw `z₀ ~ N(0, I)`, then
    /// forward-pass through the flow.
    ///
    /// Returns `(z_T, log_det_total)`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::forward`].
    pub fn sample(&self, context: &[f32], rng: &mut LcgRng) -> BayesResult<(Vec<f32>, f32)> {
        let dim = self.steps[0].made.dim;
        let mut z0 = vec![0.0_f32; dim];
        rng.fill_normal(&mut z0);
        self.forward(&z0, context)
    }

    /// Compute the ELBO contribution from a single sample:
    ///
    /// ```text
    /// ELBO_IAF = log p(x|z_T) + log p(z_T) − log q(z₀) + log|det J|
    ///          = log p(x|z_T) − Σ_j ½z₀_j² − Σ_t log_det_t
    /// ```
    ///
    /// This function computes the negative KL term
    /// `−log q(z_T) = Σ_j(½ + log σ_j(z₀)) − ½‖z₀‖²`
    /// using the log-det already accumulated.
    ///
    /// Returns the accumulated `log_det` (positive = volume expansion).
    pub fn neg_kl_term(z0_log_prob: f32, log_det: f32) -> f32 {
        // log p(z_T) − log q(z_T) = log p(z_T) − (log q(z_0) − log|det J|)
        // For standard Gaussian base p(z_T) we don't compute it here; callers
        // must add the log-likelihood term.  We expose log_det for them.
        -z0_log_prob + log_det
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Sigmoid function (numerically stable).
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Standard Gaussian log-probability for a vector `z`.
pub fn standard_normal_log_prob(z: &[f32]) -> f32 {
    let d = z.len() as f32;
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let sum_sq: f32 = z.iter().map(|&v| v * v).sum();
    -0.5 * (d * log_2pi + sum_sq)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── 1. MADE forward returns correct output dimensions ─────────────────────
    #[test]
    fn made_forward_shape() {
        let mut rng = make_rng();
        let net = MadeNet::new(4, 2, 8, &mut rng).expect("new should succeed");
        let z = vec![0.1_f32, 0.2, 0.3, 0.4];
        let ctx = vec![0.5_f32, 0.6];
        let (mu, log_s) = net.forward(&z, &ctx).expect("forward should succeed");
        assert_eq!(mu.len(), 4);
        assert_eq!(log_s.len(), 4);
    }

    // ── 2. MADE output is finite ──────────────────────────────────────────────
    #[test]
    fn made_forward_finite() {
        let mut rng = make_rng();
        let net = MadeNet::new(6, 0, 16, &mut rng).expect("new should succeed");
        let z: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
        let (mu, log_s) = net.forward(&z, &[]).expect("forward should succeed");
        assert!(mu.iter().all(|v| v.is_finite()));
        assert!(log_s.iter().all(|v| v.is_finite()));
    }

    // ── 3. MADE autoregressive mask: mu[0] is constant (no input ─────────────
    //    can influence the first output in a strict autoregressive net).
    #[test]
    fn made_first_output_constant() {
        let mut rng = make_rng();
        let net = MadeNet::new(4, 0, 8, &mut rng).expect("new should succeed");
        let z_a = vec![0.0_f32, 0.5, -0.5, 1.0];
        let z_b = vec![9.9_f32, 0.5, -0.5, 1.0]; // first element differs
        let (mu_a, _) = net.forward(&z_a, &[]).expect("forward should succeed");
        let (mu_b, _) = net.forward(&z_b, &[]).expect("forward should succeed");
        // mu[0] and log_s[0] must not depend on z[0].
        assert!(
            (mu_a[0] - mu_b[0]).abs() < 1e-5,
            "mu[0] should be independent of z[0]: {} vs {}",
            mu_a[0],
            mu_b[0]
        );
    }

    // ── 4. IafStep forward returns correct shape and finite values ────────────
    #[test]
    fn iaf_step_forward_shape_finite() {
        let mut rng = make_rng();
        let step = IafStep::new(4, 0, 8, 2.0, &mut rng).expect("new should succeed");
        let z = vec![0.3_f32, -0.1, 0.7, -0.5];
        let (z_new, log_det) = step.forward(&z, &[]).expect("forward should succeed");
        assert_eq!(z_new.len(), 4);
        assert!(log_det.is_finite(), "log_det={log_det}");
        assert!(z_new.iter().all(|v| v.is_finite()));
    }

    // ── 5. IafStep log_det is negative (sigma in (0,1), log sigma < 0) ────────
    #[test]
    fn iaf_step_log_det_sign() {
        let mut rng = make_rng();
        let step = IafStep::new(4, 0, 8, 2.0, &mut rng).expect("new should succeed");
        let z = vec![0.1_f32; 4];
        let (_, log_det) = step.forward(&z, &[]).expect("forward should succeed");
        // log_det = Σ log σ_j; σ ∈ (0,1) so log σ < 0 → total < 0.
        assert!(log_det <= 0.0, "log_det={log_det} should be ≤ 0");
    }

    // ── 6. IafFlow forward returns correct shape ───────────────────────────────
    #[test]
    fn iaf_flow_forward_shape() {
        let mut rng = make_rng();
        let flow = IafFlow::new(3, 4, 2, 8, 2.0, &mut rng).expect("new should succeed");
        let z0 = vec![0.0_f32; 4];
        let ctx = vec![0.1_f32, 0.2];
        let (z_t, ld) = flow.forward(&z0, &ctx).expect("forward should succeed");
        assert_eq!(z_t.len(), 4);
        assert!(ld.is_finite());
    }

    // ── 7. IafFlow sample draws without error ─────────────────────────────────
    #[test]
    fn iaf_flow_sample() {
        let mut rng = make_rng();
        let flow = IafFlow::new(2, 4, 0, 8, 2.0, &mut rng).expect("new should succeed");
        let mut rng2 = LcgRng::new(99);
        let (z_t, ld) = flow.sample(&[], &mut rng2).expect("sample should succeed");
        assert_eq!(z_t.len(), 4);
        assert!(ld.is_finite());
        assert!(z_t.iter().all(|v| v.is_finite()));
    }

    // ── 8. Multiple IAF samples are different (stochastic) ────────────────────
    #[test]
    fn iaf_flow_samples_differ() {
        let mut rng = make_rng();
        let flow = IafFlow::new(2, 4, 0, 8, 2.0, &mut rng).expect("new should succeed");
        let mut rng2 = LcgRng::new(1);
        let mut rng3 = LcgRng::new(2);
        let (s1, _) = flow.sample(&[], &mut rng2).expect("sample should succeed");
        let (s2, _) = flow.sample(&[], &mut rng3).expect("sample should succeed");
        let diff: f32 = s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "samples should differ: diff={diff}");
    }

    // ── 9. IafFlow: invalid config returns errors ─────────────────────────────
    #[test]
    fn iaf_flow_invalid_config() {
        let mut rng = make_rng();
        // dim < 2
        assert!(IafFlow::new(2, 1, 0, 8, 2.0, &mut rng).is_err());
        // n_steps = 0
        assert!(IafFlow::new(0, 4, 0, 8, 2.0, &mut rng).is_err());
    }

    // ── 10. standard_normal_log_prob is correct for origin ────────────────────
    #[test]
    fn standard_normal_log_prob_origin() {
        let z = vec![0.0_f32; 4];
        let lp = standard_normal_log_prob(&z);
        // log p(0^d) = -d/2 * log(2π) = -4/2 * log(2π) = -2 * log(2π)
        let expected = -2.0 * (2.0 * std::f32::consts::PI).ln(); // -d/2 * log(2π)
        assert!((lp - expected).abs() < 0.01, "lp={lp} expected≈{expected}");
    }

    // ── 11. MadeNet dimension-mismatch errors ──────────────────────────────────
    #[test]
    fn made_dim_mismatch_error() {
        let mut rng = make_rng();
        let net = MadeNet::new(4, 2, 8, &mut rng).expect("new should succeed");
        // wrong z length
        assert!(net.forward(&[0.1, 0.2], &[0.5, 0.6]).is_err());
        // wrong context length
        assert!(net.forward(&[0.1, 0.2, 0.3, 0.4], &[0.5]).is_err());
    }
}
