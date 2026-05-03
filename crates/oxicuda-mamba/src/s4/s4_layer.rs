//! S4 sequence layer using the DPLR convolutional formulation.
//!
//! The S4 layer computes the SSM output as a length-`L` convolution:
//!
//! ```text
//! y[:, d] = naive_conv1d(u[:, d], K[d]) + D_skip[d] · u[:, d]
//! ```
//!
//! where `K[d]` is the SSM kernel for channel `d` (see [`crate::s4::dplr`]),
//! and `D_skip` is a per-channel direct feed-through scalar.
//!
//! ## Layout
//!
//! All input/output tensors are flat row-major `[L × D]`: element `(t, d)` is
//! at index `t * D + d`.  This follows the convention used by
//! [`crate::ssm::ssm_kernel`] and allows straightforward slicing.
//!
//! ## Convolution
//!
//! [`naive_conv1d`] implements causal (non-circular) direct-sum convolution in
//! `O(L² )` time:
//!
//! ```text
//! y[t] = Σ_{k=0}^{min(t, K-1)}  kernel[k] · x[t-k]
//! ```
//!
//! For production use, this should be replaced by an FFT-based `O(L log L)`
//! convolution; the naive version is the correctness reference.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::s4::dplr::Dplr;

// ─── S4Config ────────────────────────────────────────────────────────────────

/// Configuration for an S4 sequence layer.
#[derive(Debug, Clone)]
pub struct S4Config {
    /// Input / output feature dimension `D`.
    pub d_model: usize,
    /// SSM state dimension `N`.
    pub d_state: usize,
    /// Expected sequence length `L`.
    pub seq_len: usize,
    /// ZOH discretization step `Δ > 0` (default `0.001`).
    pub delta: f32,
    /// If `true`, process both forward and backward passes and average their outputs.
    pub bidirectional: bool,
}

impl S4Config {
    /// Create a new `S4Config` with `delta = 0.001` and `bidirectional = false`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`]  — if `d_model == 0`.
    /// * [`MambaError::InvalidSsmOrder`]  — if `d_state == 0`.
    /// * [`MambaError::InvalidSeqLen`]    — if `seq_len == 0`.
    pub fn new(d_model: usize, d_state: usize, seq_len: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        Ok(Self {
            d_model,
            d_state,
            seq_len,
            delta: 0.001_f32,
            bidirectional: false,
        })
    }

    /// Override the discretization step `Δ`.
    ///
    /// # Errors
    ///
    /// [`MambaError::NonPositiveDelta`] if `delta ≤ 0`.
    pub fn with_delta(mut self, delta: f32) -> MambaResult<Self> {
        if delta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(delta));
        }
        self.delta = delta;
        Ok(self)
    }

    /// Override the bidirectional flag.
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }
}

// ─── S4Weights ───────────────────────────────────────────────────────────────

/// Learnable weights for an S4 layer.
#[derive(Debug, Clone)]
pub struct S4Weights {
    /// Output projection matrix `C`, flat `[D × N]`.
    pub c_proj: Vec<f32>,
    /// Direct feed-through scalar per channel `D`.
    pub d_skip: Vec<f32>,
}

impl S4Weights {
    /// Allocate zero-initialised weights for the given config.
    pub fn zeros(config: &S4Config) -> Self {
        Self {
            c_proj: vec![0.0_f32; config.d_model * config.d_state],
            d_skip: vec![0.0_f32; config.d_model],
        }
    }

    /// Allocate weights with entries sampled from `N(0, 1)` using `rng`.
    pub fn random(config: &S4Config, rng: &mut LcgRng) -> Self {
        let mut c_proj = vec![0.0_f32; config.d_model * config.d_state];
        let mut d_skip = vec![0.0_f32; config.d_model];
        rng.fill_normal(&mut c_proj);
        rng.fill_normal(&mut d_skip);
        Self { c_proj, d_skip }
    }
}

// ─── S4Layer ─────────────────────────────────────────────────────────────────

/// S4 sequence-to-sequence layer.
///
/// Applies a multi-channel SSM convolution with HiPPO-LegS initialization
/// followed by a per-channel skip connection.
pub struct S4Layer {
    config: S4Config,
    /// One `Dplr` per output channel, length `D`.
    dplr: Vec<Dplr>,
    weights: S4Weights,
}

impl S4Layer {
    /// Initialize an `S4Layer` with HiPPO-LegS DPLR for each channel
    /// and zero weights.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`S4Config`] validation or [`Dplr::from_hippo`].
    pub fn new(config: S4Config) -> MambaResult<Self> {
        let d = config.d_model;
        let n = config.d_state;
        let mut dplr = Vec::with_capacity(d);
        for _ in 0..d {
            dplr.push(Dplr::from_hippo(n)?);
        }
        let weights = S4Weights::zeros(&config);
        Ok(Self {
            config,
            dplr,
            weights,
        })
    }

    /// Return a reference to the layer configuration.
    #[inline]
    pub fn config(&self) -> &S4Config {
        &self.config
    }

    /// Forward pass: `u [L × D]` → `y [L × D]`.
    ///
    /// For each channel `d`:
    /// 1. Slice `u[:, d]` from the input.
    /// 2. Compute the SSM kernel `K[d]` of length `L`.
    /// 3. `y[:, d] = naive_conv1d(u[:, d], K[d]) + D_skip[d] · u[:, d]`.
    ///
    /// Input layout: `u[t * D + d]`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `u.len() ≠ L * D`.
    pub fn forward(&self, u: &[f32]) -> MambaResult<Vec<f32>> {
        let l = self.config.seq_len;
        let d = self.config.d_model;
        let n = self.config.d_state;
        let expected_len = l * d;

        if u.len() != expected_len {
            return Err(MambaError::DimensionMismatch {
                expected: expected_len,
                got: u.len(),
            });
        }

        let mut y = vec![0.0_f32; expected_len];

        for ch in 0..d {
            // Extract the channel slice u[:, ch] → length L.
            let mut u_ch = Vec::with_capacity(l);
            for t in 0..l {
                u_ch.push(u[t * d + ch]);
            }

            // Build C vector for this channel: c_proj[ch * N .. ch * N + N].
            let c_start = ch * n;
            let c = &self.weights.c_proj[c_start..c_start + n];

            // Compute SSM kernel of length L.
            let kernel = self.dplr[ch].ssm_kernel(c, self.config.delta, l)?;

            // Causal convolution.
            let conv_out = naive_conv1d(&u_ch, &kernel);

            // Accumulate: y[:, ch] = conv_out + D_skip[ch] * u[:, ch]
            let d_skip = self.weights.d_skip[ch];
            for t in 0..l {
                y[t * d + ch] = conv_out[t] + d_skip * u_ch[t];
            }

            // ── Bidirectional: process reversed and average ──────────────────
            if self.config.bidirectional {
                // Reverse input for the backward pass.
                let u_rev: Vec<f32> = u_ch.iter().rev().copied().collect();
                let kernel_rev = self.dplr[ch].ssm_kernel(c, self.config.delta, l)?;
                let conv_rev = naive_conv1d(&u_rev, &kernel_rev);
                // Un-reverse and average with forward output.
                for t in 0..l {
                    let fwd = y[t * d + ch];
                    let bwd = conv_rev[l - 1 - t] + d_skip * u_ch[t];
                    y[t * d + ch] = 0.5 * (fwd + bwd);
                }
            }
        }

        Ok(y)
    }
}

// ─── naive_conv1d ────────────────────────────────────────────────────────────

/// Causal 1-D convolution (direct sum, `O(L² )` reference implementation).
///
/// ```text
/// y[t] = Σ_{k=0}^{min(t, K-1)}  kernel[k] · x[t - k]
/// ```
///
/// * If `kernel` is empty or `x` is empty, returns a zero vector of length `x.len()`.
pub fn naive_conv1d(x: &[f32], kernel: &[f32]) -> Vec<f32> {
    let l = x.len();
    let k_len = kernel.len();
    let mut y = vec![0.0_f32; l];
    if k_len == 0 {
        return y;
    }
    for t in 0..l {
        let max_k = k_len.min(t + 1);
        let mut acc = 0.0_f32;
        for k in 0..max_k {
            acc += kernel[k] * x[t - k];
        }
        y[t] = acc;
    }
    y
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── S4Config ─────────────────────────────────────────────────────────────

    /// Valid config is accepted and fields are set correctly.
    #[test]
    fn s4_config_valid() {
        let cfg = S4Config::new(8, 16, 64).expect("valid config");
        assert_eq!(cfg.d_model, 8);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.seq_len, 64);
        assert!((cfg.delta - 0.001).abs() < EPS);
        assert!(!cfg.bidirectional);
    }

    /// d_model=0 returns InvalidModelDim.
    #[test]
    fn s4_config_zero_d_model() {
        let err = S4Config::new(0, 4, 8).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    /// d_state=0 returns InvalidSsmOrder.
    #[test]
    fn s4_config_zero_d_state() {
        let err = S4Config::new(4, 0, 8).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    /// seq_len=0 returns InvalidSeqLen.
    #[test]
    fn s4_config_zero_seq_len() {
        let err = S4Config::new(4, 4, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    /// Negative delta returns NonPositiveDelta.
    #[test]
    fn s4_config_negative_delta() {
        let err = S4Config::new(4, 4, 8)
            .expect("valid config")
            .with_delta(-0.1_f32)
            .expect_err("should fail for negative delta");
        assert!(matches!(err, MambaError::NonPositiveDelta(_)));
    }

    /// with_bidirectional sets the flag.
    #[test]
    fn s4_config_bidirectional_flag() {
        let cfg = S4Config::new(4, 4, 8)
            .expect("valid")
            .with_bidirectional(true);
        assert!(cfg.bidirectional);
    }

    // ── S4Weights ─────────────────────────────────────────────────────────────

    /// Zeros weights have correct shapes.
    #[test]
    fn s4_weights_zeros_shape() {
        let cfg = S4Config::new(6, 8, 32).expect("valid config");
        let w = S4Weights::zeros(&cfg);
        assert_eq!(w.c_proj.len(), 6 * 8, "c_proj length mismatch");
        assert_eq!(w.d_skip.len(), 6, "d_skip length mismatch");
        assert!(w.c_proj.iter().all(|&v| v == 0.0), "c_proj should be zeros");
        assert!(w.d_skip.iter().all(|&v| v == 0.0), "d_skip should be zeros");
    }

    /// Random weights are all finite.
    #[test]
    fn s4_weights_random_finite() {
        let cfg = S4Config::new(4, 8, 16).expect("valid config");
        let mut rng = LcgRng::new(42);
        let w = S4Weights::random(&cfg, &mut rng);
        assert!(
            w.c_proj.iter().all(|v| v.is_finite()),
            "c_proj not all finite"
        );
        assert!(
            w.d_skip.iter().all(|v| v.is_finite()),
            "d_skip not all finite"
        );
    }

    // ── naive_conv1d ──────────────────────────────────────────────────────────

    /// Identity kernel [1.0]: y == x.
    #[test]
    fn naive_conv1d_identity_kernel() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0_f32];
        let y = naive_conv1d(&x, &kernel);
        assert_eq!(y.len(), x.len());
        for (i, (&yi, &xi)) in y.iter().zip(x.iter()).enumerate() {
            assert!((yi - xi).abs() < EPS, "y[{i}]={yi} expected {xi}");
        }
    }

    /// Scalar-multiplication kernel [0.5]: y == x * 0.5.
    #[test]
    fn naive_conv1d_single_step() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let kernel = vec![0.5_f32];
        let y = naive_conv1d(&x, &kernel);
        let expected = [0.5_f32, 1.0, 1.5];
        for (i, (&yi, &ei)) in y.iter().zip(expected.iter()).enumerate() {
            assert!((yi - ei).abs() < EPS, "y[{i}]={yi} expected {ei}");
        }
    }

    /// Zero kernel produces all zeros.
    #[test]
    fn naive_conv1d_zero_kernel() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let kernel = vec![0.0_f32, 0.0, 0.0];
        let y = naive_conv1d(&x, &kernel);
        assert!(
            y.iter().all(|&v| v.abs() < EPS),
            "zero kernel should produce zeros"
        );
    }

    /// Manual 2-tap kernel: y[t] = x[t] + 0.5*x[t-1] (causal).
    ///
    /// x = [1, 2, 3, 4], kernel = [1, 0.5]
    /// y[0] = 1*1 = 1
    /// y[1] = 1*2 + 0.5*1 = 2.5
    /// y[2] = 1*3 + 0.5*2 = 4.0
    /// y[3] = 1*4 + 0.5*3 = 5.5
    #[test]
    fn naive_conv1d_two_tap() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let kernel = vec![1.0_f32, 0.5];
        let y = naive_conv1d(&x, &kernel);
        let expected = [1.0_f32, 2.5, 4.0, 5.5];
        for (i, (&yi, &ei)) in y.iter().zip(expected.iter()).enumerate() {
            assert!((yi - ei).abs() < EPS, "y[{i}]={yi} expected {ei}");
        }
    }

    // ── S4Layer ───────────────────────────────────────────────────────────────

    /// Output has the same [L × D] shape as input.
    #[test]
    fn s4_layer_forward_shape() {
        let cfg = S4Config::new(4, 8, 16).expect("valid config");
        let layer = S4Layer::new(cfg).expect("valid layer");
        let u = vec![0.5_f32; 16 * 4];
        let y = layer.forward(&u).expect("forward");
        assert_eq!(y.len(), 16 * 4, "output shape mismatch");
    }

    /// All output values are finite for random input.
    #[test]
    fn s4_layer_forward_finite() {
        let cfg = S4Config::new(4, 8, 32).expect("valid config");
        let layer = S4Layer::new(cfg).expect("valid layer");
        let mut rng = LcgRng::new(99);
        let mut u = vec![0.0_f32; 32 * 4];
        rng.fill_normal(&mut u);
        let y = layer.forward(&u).expect("forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite");
        }
    }

    /// Zero input with zero D_skip gives zero output.
    #[test]
    fn s4_layer_zero_input() {
        let cfg = S4Config::new(3, 4, 8).expect("valid config");
        // S4Layer::new initializes D_skip to zero.
        let layer = S4Layer::new(cfg).expect("valid layer");
        let u = vec![0.0_f32; 8 * 3];
        let y = layer.forward(&u).expect("forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.abs() < EPS, "y[{i}]={v} should be zero for zero input");
        }
    }

    /// L=64, D=4, N=4 produces finite output (medium-scale smoke test).
    #[test]
    fn s4_layer_large() {
        let cfg = S4Config::new(4, 4, 64).expect("valid config");
        let layer = S4Layer::new(cfg).expect("valid layer");
        let mut rng = LcgRng::new(2025);
        let mut u = vec![0.0_f32; 64 * 4];
        rng.fill_normal(&mut u);
        let y = layer.forward(&u).expect("forward");
        assert_eq!(y.len(), 64 * 4);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite for large config");
        }
    }

    /// Wrong input length returns DimensionMismatch.
    #[test]
    fn s4_layer_wrong_input_len() {
        let cfg = S4Config::new(4, 4, 8).expect("valid config");
        let layer = S4Layer::new(cfg).expect("valid layer");
        let u = vec![0.0_f32; 10]; // should be 8*4=32
        let err = layer.forward(&u).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }

    /// Bidirectional layer produces finite output.
    #[test]
    fn s4_layer_bidirectional_finite() {
        let cfg = S4Config::new(4, 4, 16)
            .expect("valid config")
            .with_bidirectional(true);
        let layer = S4Layer::new(cfg).expect("valid layer");
        let mut rng = LcgRng::new(77);
        let mut u = vec![0.0_f32; 16 * 4];
        rng.fill_normal(&mut u);
        let y = layer.forward(&u).expect("forward");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "bidirectional y[{i}]={v} not finite");
        }
    }
}
