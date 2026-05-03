//! Core SSM forward-pass kernel operating on pre-discretized parameters.
//!
//! Implements the Mamba-style selective SSM recurrence:
//!
//! ```text
//! h[b, t, d, n] = Ā[d, n] * h[b, t-1, d, n] + B̄[b, t, d, n] * u[b, t, d]
//! y[b, t, d]    = Σ_n  C[b, t, d, n] * h[b, t, d, n]
//! ```
//!
//! where `(Ā, B̄)` are obtained by ZOH-discretizing the learned diagonal A
//! with the per-sequence `Δ` step.
//!
//! Batch and SSM-channel dimensions are treated independently, and the
//! recurrence over time is computed sequentially (CPU reference).  This
//! module is the correctness baseline; GPU acceleration is a separate concern.

use crate::error::{MambaError, MambaResult};
use crate::ssm::discretize::{Discretization, discretize};

// ─── SsmConfig ───────────────────────────────────────────────────────────────

/// Configuration for the SSM kernel dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmConfig {
    /// Batch size `B`.
    pub batch: usize,
    /// Sequence length `L`.
    pub seq_len: usize,
    /// Number of SSM channels `D` (model width after input projection).
    pub d_model: usize,
    /// State dimension per channel `N`.
    pub d_state: usize,
}

impl SsmConfig {
    /// Create a new `SsmConfig`, validating that all dimensions are > 0.
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — proxied through specific error
    ///   variants:
    ///   - [`MambaError::InvalidSsmOrder`]  if `d_state == 0`
    ///   - [`MambaError::InvalidModelDim`]  if `d_model == 0`
    ///   - [`MambaError::InvalidSeqLen`]    if `seq_len == 0`
    ///   - zero `batch` uses [`MambaError::Internal`]
    pub fn new(batch: usize, seq_len: usize, d_model: usize, d_state: usize) -> MambaResult<Self> {
        if batch == 0 {
            return Err(MambaError::Internal("batch size must be > 0".into()));
        }
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(d_model));
        }
        if d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(d_state));
        }
        Ok(Self {
            batch,
            seq_len,
            d_model,
            d_state,
        })
    }

    /// Total number of elements in the `u` / `y` tensor: `B * L * D`.
    #[inline]
    pub fn u_numel(&self) -> usize {
        self.batch * self.seq_len * self.d_model
    }

    /// Total number of elements in a `B_proj` or `C_proj` tensor: `B * L * D * N`.
    #[inline]
    pub fn bc_numel(&self) -> usize {
        self.batch * self.seq_len * self.d_model * self.d_state
    }

    /// Flat index into `u` / `y` tensor (row-major layout: B, L, D).
    #[inline]
    pub fn u_idx(&self, b: usize, t: usize, d: usize) -> usize {
        b * (self.seq_len * self.d_model) + t * self.d_model + d
    }

    /// Flat index into `B_proj` / `C_proj` tensors (row-major: B, L, D, N).
    #[inline]
    pub fn bc_idx(&self, b: usize, t: usize, d: usize, n: usize) -> usize {
        b * (self.seq_len * self.d_model * self.d_state)
            + t * (self.d_model * self.d_state)
            + d * self.d_state
            + n
    }

    /// Flat index into the per-batch state tensor (B, D, N).
    #[inline]
    pub fn state_idx(&self, b: usize, d: usize, n: usize) -> usize {
        b * (self.d_model * self.d_state) + d * self.d_state + n
    }
}

// ─── SsmKernel ───────────────────────────────────────────────────────────────

/// Core SSM forward kernel.
///
/// Holds the learned diagonal A matrix and the kernel configuration.
/// The A matrix is shared across the batch and time dimensions (input-
/// independent), while B and C projections are input-dependent and supplied
/// per forward call.
#[derive(Debug)]
pub struct SsmKernel {
    config: SsmConfig,
    /// Diagonal of A, shape `[D × N]`, laid out as `[d * N + n]`.
    a_diag: Vec<f32>,
}

impl SsmKernel {
    /// Create a new `SsmKernel`.
    ///
    /// # Arguments
    ///
    /// * `config`  — Validated kernel configuration.
    /// * `a_init`  — Optional slice of length `D * N` for the A diagonal.
    ///   When `None`, all entries are initialised to `-0.5` (a mild stable decay).
    ///
    /// # Errors
    ///
    /// * [`MambaError::WeightShapeMismatch`] — if `a_init.len() ≠ D * N`.
    pub fn new(config: SsmConfig, a_init: Option<&[f32]>) -> MambaResult<Self> {
        let expected_len = config.d_model * config.d_state;
        let a_diag = match a_init {
            None => vec![-0.5_f32; expected_len],
            Some(src) => {
                if src.len() != expected_len {
                    return Err(MambaError::WeightShapeMismatch {
                        name: "a_diag",
                        expected: vec![config.d_model, config.d_state],
                        got: vec![src.len()],
                    });
                }
                src.to_vec()
            }
        };
        Ok(Self { config, a_diag })
    }

    /// Return a reference to the kernel configuration.
    #[inline]
    pub fn config(&self) -> &SsmConfig {
        &self.config
    }

    /// Return a reference to the diagonal A vector `[D * N]`.
    #[inline]
    pub fn a_diag(&self) -> &[f32] {
        &self.a_diag
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Forward pass
    // ─────────────────────────────────────────────────────────────────────────

    /// Run the SSM forward pass.
    ///
    /// # Layout
    ///
    /// * `u`       — Input, flat `[B * L * D]` (row-major: batch, seq, channel).
    /// * `b_proj`  — Input-dependent B, flat `[B * L * D * N]`.
    /// * `c_proj`  — Input-dependent C, flat `[B * L * D * N]`.
    /// * `delta`   — Positive time-step used to discretize A via ZOH.
    ///
    /// # Returns
    ///
    /// Output tensor `y`, flat `[B * L * D]`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::NonPositiveDelta`]   — if `delta ≤ 0`.
    /// * [`MambaError::DimensionMismatch`]  — if any input has wrong length.
    pub fn forward(
        &self,
        u: &[f32],
        b_proj: &[f32],
        c_proj: &[f32],
        delta: f32,
    ) -> MambaResult<Vec<f32>> {
        let cfg = &self.config;

        // ── Input shape validation ────────────────────────────────────────────
        let expected_u = cfg.u_numel();
        if u.len() != expected_u {
            return Err(MambaError::DimensionMismatch {
                expected: expected_u,
                got: u.len(),
            });
        }
        let expected_bc = cfg.bc_numel();
        if b_proj.len() != expected_bc {
            return Err(MambaError::DimensionMismatch {
                expected: expected_bc,
                got: b_proj.len(),
            });
        }
        if c_proj.len() != expected_bc {
            return Err(MambaError::DimensionMismatch {
                expected: expected_bc,
                got: c_proj.len(),
            });
        }
        if delta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(delta));
        }

        // ── Pre-discretize A with ZOH ─────────────────────────────────────────
        // a_diag is [D * N]; discretize all D*N entries at once.
        // All B entries in the A→Ā formula are set to 1 (the formula's B factor
        // is absorbed into b_proj, so we only need Ā[d, n] here).
        let ones_b = vec![1.0_f32; self.a_diag.len()];
        let (a_bar_all, _) = discretize(&self.a_diag, &ones_b, delta, Discretization::Zoh)?;
        // a_bar_all[d * N + n] = exp(delta * A[d, n])

        // ── Allocate outputs and initial state ───────────────────────────────
        let mut y = vec![0.0_f32; expected_u];
        // Hidden state h: [B × D × N], initialised to zero.
        let state_size = cfg.batch * cfg.d_model * cfg.d_state;
        let mut h = vec![0.0_f32; state_size];

        // ── Recurrence over time ─────────────────────────────────────────────
        for t in 0..cfg.seq_len {
            for b in 0..cfg.batch {
                for d in 0..cfg.d_model {
                    let u_val = u[cfg.u_idx(b, t, d)];

                    // Update each state dimension n
                    let mut y_val = 0.0_f32;
                    for n in 0..cfg.d_state {
                        let a_bar = a_bar_all[d * cfg.d_state + n];

                        // B_bar[b,t,d,n] = (Ā[d,n] - 1) / A[d,n] * B_proj[b,t,d,n]
                        // Using the ZOH formula; the near-zero branch is handled
                        // inside `discretize`, but here we replicate the logic
                        // directly to avoid a per-step allocation.
                        let a_val = self.a_diag[d * cfg.d_state + n];
                        let b_proj_val = b_proj[cfg.bc_idx(b, t, d, n)];
                        let b_bar = if a_val.abs() < 1e-6_f32 {
                            delta * b_proj_val
                        } else {
                            (a_bar - 1.0) / a_val * b_proj_val
                        };

                        let c_val = c_proj[cfg.bc_idx(b, t, d, n)];
                        let h_prev = h[cfg.state_idx(b, d, n)];

                        // h[t] = Ā * h[t-1] + B̄ * u[t]
                        let h_new = a_bar * h_prev + b_bar * u_val;
                        h[cfg.state_idx(b, d, n)] = h_new;

                        // y[t] += C[b,t,d,n] * h[t]
                        y_val += c_val * h_new;
                    }
                    y[cfg.u_idx(b, t, d)] = y_val;
                }
            }
        }

        Ok(y)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── SsmConfig ─────────────────────────────────────────────────────────────

    /// Valid configuration is accepted.
    #[test]
    fn config_valid() {
        let cfg = SsmConfig::new(2, 8, 4, 4).expect("valid config");
        assert_eq!(cfg.batch, 2);
        assert_eq!(cfg.seq_len, 8);
        assert_eq!(cfg.d_model, 4);
        assert_eq!(cfg.d_state, 4);
    }

    /// Zero batch size must fail.
    #[test]
    fn config_zero_batch() {
        let err = SsmConfig::new(0, 8, 4, 4).expect_err("should fail");
        assert!(matches!(err, MambaError::Internal(_)));
    }

    /// Zero sequence length must fail.
    #[test]
    fn config_zero_seq_len() {
        let err = SsmConfig::new(2, 0, 4, 4).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    /// Zero d_model must fail.
    #[test]
    fn config_zero_d_model() {
        let err = SsmConfig::new(2, 8, 0, 4).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    /// Zero d_state must fail.
    #[test]
    fn config_zero_d_state() {
        let err = SsmConfig::new(2, 8, 4, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    // ── SsmKernel construction ────────────────────────────────────────────────

    /// Default A diagonal is all `-0.5`.
    #[test]
    fn a_diag_default_init() {
        let cfg = SsmConfig::new(1, 4, 3, 2).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");
        assert_eq!(kernel.a_diag().len(), 3 * 2);
        assert!(
            kernel.a_diag().iter().all(|&v| (v - (-0.5)).abs() < 1e-7),
            "default A diagonal should be -0.5"
        );
    }

    /// Custom A diagonal is accepted when length matches.
    #[test]
    fn a_diag_custom_init_accepted() {
        let cfg = SsmConfig::new(1, 4, 2, 3).expect("valid config");
        let a_init = vec![-1.0_f32; 2 * 3];
        let kernel = SsmKernel::new(cfg, Some(&a_init)).expect("valid kernel");
        assert!(kernel.a_diag().iter().all(|&v| (v + 1.0).abs() < 1e-7));
    }

    /// Wrong A diagonal length returns WeightShapeMismatch.
    #[test]
    fn a_diag_wrong_length_fails() {
        let cfg = SsmConfig::new(1, 4, 2, 3).expect("valid config");
        let a_init = vec![-1.0_f32; 5]; // should be 6
        let err = SsmKernel::new(cfg, Some(&a_init)).expect_err("should fail");
        assert!(matches!(err, MambaError::WeightShapeMismatch { .. }));
    }

    // ── Forward pass: output shape ────────────────────────────────────────────

    /// forward() returns B*L*D elements.
    #[test]
    fn kernel_output_shape() {
        let b = 2_usize;
        let l = 8_usize;
        let d = 4_usize;
        let n = 4_usize;
        let cfg = SsmConfig::new(b, l, d, n).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");

        let u = vec![0.1_f32; b * l * d];
        let b_proj = vec![0.1_f32; b * l * d * n];
        let c_proj = vec![0.1_f32; b * l * d * n];
        let y = kernel
            .forward(&u, &b_proj, &c_proj, 0.1)
            .expect("forward pass");
        assert_eq!(y.len(), b * l * d, "output shape mismatch");
    }

    // ── Forward pass: all finite ──────────────────────────────────────────────

    /// Forward pass with random inputs produces all finite outputs.
    #[test]
    fn kernel_output_finite() {
        let b = 3_usize;
        let l = 16_usize;
        let d = 6_usize;
        let n = 4_usize;
        let cfg = SsmConfig::new(b, l, d, n).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");

        let mut rng = LcgRng::new(42);
        let numel_u = b * l * d;
        let numel_bc = b * l * d * n;
        let mut u = vec![0.0_f32; numel_u];
        let mut bp = vec![0.0_f32; numel_bc];
        let mut cp = vec![0.0_f32; numel_bc];
        rng.fill_normal(&mut u);
        rng.fill_normal(&mut bp);
        rng.fill_normal(&mut cp);

        let y = kernel.forward(&u, &bp, &cp, 0.05).expect("forward pass");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite");
        }
    }

    // ── Forward pass: zero input ──────────────────────────────────────────────

    /// Zero u + zero B_proj → zero output (state stays zero).
    #[test]
    fn kernel_zero_input() {
        let cfg = SsmConfig::new(2, 6, 4, 3).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");

        let numel_u = cfg.u_numel();
        let numel_bc = cfg.bc_numel();
        let u = vec![0.0_f32; numel_u];
        let b_proj = vec![0.0_f32; numel_bc];
        let c_proj = vec![1.0_f32; numel_bc]; // C can be nonzero; h=0 so y=0
        let y = kernel
            .forward(&u, &b_proj, &c_proj, 0.1)
            .expect("forward pass");
        for (i, &v) in y.iter().enumerate() {
            assert!((v).abs() < 1e-7, "y[{i}]={v} should be zero");
        }
    }

    // ── Forward pass: manual single-step recurrence ───────────────────────────

    /// B=1, L=2, D=1, N=1: verify recurrence step by step.
    ///
    /// With A = -1.0, delta = 0.5:
    ///   Ā = exp(-0.5) ≈ 0.60653
    ///   B̄ = (Ā - 1) / A = (0.60653 - 1) / (-1) ≈ 0.39347
    ///
    /// B_proj = [[b0, b1]] = [[1.0, 1.0]], C_proj = [[1.0, 1.0]], u = [[u0, u1]] = [[1.0, 1.0]]
    ///
    /// h[0] = Ā * 0 + B̄ * b0 * u0 = 0.39347
    /// y[0] = C[0] * h[0]           = 0.39347
    ///
    /// h[1] = Ā * h[0] + B̄ * b1 * u1 = 0.60653 * 0.39347 + 0.39347 ≈ 0.63230
    /// y[1] = C[1] * h[1]              ≈ 0.63230
    #[test]
    fn kernel_unit_c_and_b_manual_recurrence() {
        let cfg = SsmConfig::new(1, 2, 1, 1).expect("valid config");
        let a_init = [-1.0_f32]; // A[0,0] = -1.0
        let kernel = SsmKernel::new(cfg, Some(&a_init)).expect("valid kernel");

        let delta = 0.5_f32;
        let a_bar = (-0.5_f32).exp();
        let b_bar = (a_bar - 1.0) / (-1.0_f32); // = 1 - exp(-0.5)

        let u = vec![1.0_f32, 1.0]; // [L=2, D=1]
        let b_proj = vec![1.0_f32, 1.0]; // [L=2, D=1, N=1]
        let c_proj = vec![1.0_f32, 1.0];

        let y = kernel
            .forward(&u, &b_proj, &c_proj, delta)
            .expect("forward pass");

        let h0 = b_bar * 1.0 * 1.0;
        let expected_y0 = 1.0 * h0;
        let h1 = a_bar * h0 + b_bar * 1.0 * 1.0;
        let expected_y1 = 1.0 * h1;

        assert!(
            (y[0] - expected_y0).abs() < 1e-5,
            "y[0]={} expected {expected_y0}",
            y[0]
        );
        assert!(
            (y[1] - expected_y1).abs() < 1e-5,
            "y[1]={} expected {expected_y1}",
            y[1]
        );
    }

    // ── Forward pass: stable A → bounded states ───────────────────────────────

    /// Stable A (all negative) → states remain bounded even for long sequences.
    #[test]
    fn kernel_stable_a_bounded_states() {
        let b = 1_usize;
        let l = 256_usize;
        let d = 2_usize;
        let n = 4_usize;
        let cfg = SsmConfig::new(b, l, d, n).expect("valid config");

        // Strongly negative A → fast decay
        let a_init = vec![-2.0_f32; d * n];
        let kernel = SsmKernel::new(cfg, Some(&a_init)).expect("valid kernel");

        let u = vec![1.0_f32; b * l * d];
        let b_proj = vec![1.0_f32; b * l * d * n];
        let c_proj = vec![1.0_f32; b * l * d * n];
        let y = kernel
            .forward(&u, &b_proj, &c_proj, 0.05)
            .expect("forward pass");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite");
            // Steady-state bound: |y| ≤ N * (1 / (1 - exp(-2*0.05))) * (B̄ * u)
            // which is finite for stable A; just check no explosion
            assert!(v.abs() < 1e4, "y[{i}]={v} seems too large");
        }
    }

    // ── Forward pass: batch independence ─────────────────────────────────────

    /// The same input placed in two different batch slots produces identical outputs.
    #[test]
    fn kernel_batch_independence() {
        let l = 8_usize;
        let d = 3_usize;
        let n = 2_usize;
        let cfg = SsmConfig::new(2, l, d, n).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");

        let mut rng = LcgRng::new(99);
        let single_u = {
            let mut v = vec![0.0_f32; l * d];
            rng.fill_normal(&mut v);
            v
        };
        let single_bc = {
            let mut v = vec![0.0_f32; l * d * n];
            rng.fill_normal(&mut v);
            v
        };

        // Replicate single input into batch of 2 (same data in both slots)
        let u: Vec<f32> = single_u.iter().chain(single_u.iter()).copied().collect();
        let b_proj: Vec<f32> = single_bc.iter().chain(single_bc.iter()).copied().collect();
        let c_proj: Vec<f32> = single_bc.iter().chain(single_bc.iter()).copied().collect();

        let y = kernel
            .forward(&u, &b_proj, &c_proj, 0.1)
            .expect("forward pass");

        // y[batch=0, t, d] must equal y[batch=1, t, d]
        let stride = l * d;
        let y_b0 = &y[..stride];
        let y_b1 = &y[stride..];
        for (i, (&v0, &v1)) in y_b0.iter().zip(y_b1.iter()).enumerate() {
            assert!(
                (v0 - v1).abs() < 1e-5,
                "batch independence violated at i={i}: y_b0={v0}, y_b1={v1}"
            );
        }
    }

    // ── Forward pass: large kernel ────────────────────────────────────────────

    /// B=4, L=64, D=8, N=4 completes without OOM and produces finite output.
    #[test]
    fn kernel_large_finite() {
        let b = 4_usize;
        let l = 64_usize;
        let d = 8_usize;
        let n = 4_usize;
        let cfg = SsmConfig::new(b, l, d, n).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");

        let mut rng = LcgRng::new(2024);
        let mut u = vec![0.0_f32; cfg.u_numel()];
        let mut bp = vec![0.0_f32; cfg.bc_numel()];
        let mut cp = vec![0.0_f32; cfg.bc_numel()];
        rng.fill_normal(&mut u);
        rng.fill_normal(&mut bp);
        rng.fill_normal(&mut cp);

        let y = kernel.forward(&u, &bp, &cp, 0.1).expect("forward pass");
        assert_eq!(y.len(), b * l * d);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite for large kernel");
        }
    }

    // ── Forward pass: error cases ─────────────────────────────────────────────

    /// Non-positive delta returns NonPositiveDelta.
    #[test]
    fn kernel_forward_non_positive_delta() {
        let cfg = SsmConfig::new(1, 4, 2, 2).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");
        let u = vec![0.0_f32; cfg.u_numel()];
        let bc = vec![0.0_f32; cfg.bc_numel()];
        let err = kernel.forward(&u, &bc, &bc, 0.0).expect_err("should fail");
        assert!(matches!(err, MambaError::NonPositiveDelta(_)));
    }

    /// Wrong u length returns DimensionMismatch.
    #[test]
    fn kernel_forward_wrong_u_length() {
        let cfg = SsmConfig::new(1, 4, 2, 2).expect("valid config");
        let kernel = SsmKernel::new(cfg, None).expect("valid kernel");
        let u = vec![0.0_f32; 3]; // should be 4*2=8
        let bc = vec![0.0_f32; cfg.bc_numel()];
        let err = kernel.forward(&u, &bc, &bc, 0.1).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }
}
