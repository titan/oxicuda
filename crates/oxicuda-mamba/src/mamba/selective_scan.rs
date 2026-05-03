//! Pure-Rust reference implementation of the Mamba selective scan (S6 model).
//!
//! # Theory (Gu & Dao, 2023)
//!
//! Unlike S4 where A, B, C are fixed, Mamba makes B, C, Δ input-dependent:
//!
//! ```text
//! Given u: [B, L, D]:
//!   Δ         = softplus(linear(u))          — positive, input-dependent step size
//!   A         = -exp(a_log)                   — negative definite (stable)
//!   A_bar[d,n] = exp(Δ[b,t,d] * A[d,n])     — per-element ZOH discretization
//!   B_bar[d,n] = Δ[b,t,d] * B_proj[b,t,n]   — input-dependent B (ZOH simplified)
//!   h[b,t,d,n] = A_bar * h_prev + B_bar * u[b,t,d]
//!   y[b,t,d]   = Σ_n C_proj[b,t,n] * h[b,t,d,n]
//! ```
//!
//! The B_bar formula uses the simplified ZOH: `Δ * B` (as in the paper, the
//! full `(exp(ΔA) - 1)/A * B` is approximated by `Δ * B` when forming B_bar
//! from the projected B, since A absorbs the normalization in A_log).

use crate::error::{MambaError, MambaResult};

// ─── SelectiveScanConfig ─────────────────────────────────────────────────────

/// Configuration for Mamba's selective scan (S6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectiveScanConfig {
    /// Batch size `B`.
    pub batch: usize,
    /// Sequence length `L`.
    pub seq_len: usize,
    /// Model dimension `D` (number of SSM channels).
    pub d_model: usize,
    /// State size `N` (hidden state dimension per channel).
    pub d_state: usize,
}

impl SelectiveScanConfig {
    /// Create a new `SelectiveScanConfig`, validating that all dimensions are > 0.
    ///
    /// # Errors
    ///
    /// - [`MambaError::Internal`]         — if `batch == 0`
    /// - [`MambaError::InvalidSeqLen`]    — if `seq_len == 0`
    /// - [`MambaError::InvalidModelDim`]  — if `d_model == 0`
    /// - [`MambaError::InvalidSsmOrder`]  — if `d_state == 0`
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

    /// Total elements in `u` / `y`: `B * L * D`.
    #[inline]
    pub fn u_numel(&self) -> usize {
        self.batch * self.seq_len * self.d_model
    }

    /// Total elements in `b_proj` / `c_proj`: `B * L * N`.
    #[inline]
    pub fn bc_numel(&self) -> usize {
        self.batch * self.seq_len * self.d_state
    }

    /// Flat index into `u` / `delta` / `y` layout `[B, L, D]`.
    #[inline]
    pub fn u_idx(&self, b: usize, t: usize, d: usize) -> usize {
        b * (self.seq_len * self.d_model) + t * self.d_model + d
    }

    /// Flat index into `b_proj` / `c_proj` layout `[B, L, N]`.
    #[inline]
    pub fn bc_idx(&self, b: usize, t: usize, n: usize) -> usize {
        b * (self.seq_len * self.d_state) + t * self.d_state + n
    }

    /// Flat index into `a_log` layout `[D, N]`.
    #[inline]
    pub fn a_idx(&self, d: usize, n: usize) -> usize {
        d * self.d_state + n
    }

    /// Flat index into hidden state `[B, D, N]`.
    #[inline]
    pub fn h_idx(&self, b: usize, d: usize, n: usize) -> usize {
        b * (self.d_model * self.d_state) + d * self.d_state + n
    }
}

// ─── Softplus ────────────────────────────────────────────────────────────────

/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// For `x > 20` returns `x` (avoids overflow in `exp`).
/// For `x < -20` returns `0.0` (avoids underflow).
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

// ─── selective_scan ──────────────────────────────────────────────────────────

/// Mamba selective scan (S6) — pure-Rust CPU reference.
///
/// # Inputs (all row-major flat `f32` slices)
///
/// * `u`      — `[B, L, D]`  input sequence
/// * `delta`  — `[B, L, D]`  raw step sizes (passed through `softplus` internally)
/// * `a_log`  — `[D, N]`     `log(-A)`, so `A = -exp(a_log)` (ensures `A < 0`)
/// * `b_proj` — `[B, L, N]`  input-dependent B projection
/// * `c_proj` — `[B, L, N]`  input-dependent C projection
///
/// # Output
///
/// `y` — `[B, L, D]`, length `B * L * D`.
///
/// # Errors
///
/// - [`MambaError::DimensionMismatch`] if any input slice has wrong length.
pub fn selective_scan(
    u: &[f32],
    delta: &[f32],
    a_log: &[f32],
    b_proj: &[f32],
    c_proj: &[f32],
    config: &SelectiveScanConfig,
) -> MambaResult<Vec<f32>> {
    let cfg = config;

    // ── Validate input shapes ──────────────────────────────────────────────────
    let expected_u = cfg.u_numel();
    if u.len() != expected_u {
        return Err(MambaError::DimensionMismatch {
            expected: expected_u,
            got: u.len(),
        });
    }
    if delta.len() != expected_u {
        return Err(MambaError::DimensionMismatch {
            expected: expected_u,
            got: delta.len(),
        });
    }
    let expected_a = cfg.d_model * cfg.d_state;
    if a_log.len() != expected_a {
        return Err(MambaError::DimensionMismatch {
            expected: expected_a,
            got: a_log.len(),
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

    // ── Allocate output and hidden state ──────────────────────────────────────
    let mut y = vec![0.0_f32; expected_u];
    // Hidden state h: [B, D, N], initialised to zero.
    let mut h = vec![0.0_f32; cfg.batch * cfg.d_model * cfg.d_state];

    // ── Recurrence over (batch, time, channel, state) ─────────────────────────
    for t in 0..cfg.seq_len {
        for b in 0..cfg.batch {
            for d in 0..cfg.d_model {
                let u_val = u[cfg.u_idx(b, t, d)];
                // Apply softplus to get positive Δ
                let delta_raw = delta[cfg.u_idx(b, t, d)];
                let dt = softplus(delta_raw);

                let mut y_val = 0.0_f32;
                for n in 0..cfg.d_state {
                    // A = -exp(a_log[d, n])  (always negative, stable)
                    let a_val = -(a_log[cfg.a_idx(d, n)].exp());
                    // A_bar = exp(Δ * A) — ZOH discretization of diagonal A
                    let a_bar = (dt * a_val).exp();
                    // B_bar = Δ * B_proj — simplified ZOH for B in the selective setting
                    let b_bar = dt * b_proj[cfg.bc_idx(b, t, n)];
                    // State update: h[t] = A_bar * h[t-1] + B_bar * u[t]
                    let h_prev = h[cfg.h_idx(b, d, n)];
                    let h_new = a_bar * h_prev + b_bar * u_val;
                    h[cfg.h_idx(b, d, n)] = h_new;
                    // Output accumulation: y[t] += C[t,n] * h[t,d,n]
                    y_val += c_proj[cfg.bc_idx(b, t, n)] * h_new;
                }
                y[cfg.u_idx(b, t, d)] = y_val;
            }
        }
    }

    Ok(y)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const EPS: f32 = 1e-5;

    // ── softplus ──────────────────────────────────────────────────────────────

    #[test]
    fn softplus_large_x() {
        // x=100 should return ≈100 (avoids exp overflow)
        let v = softplus(100.0);
        assert!((v - 100.0).abs() < 1e-3, "softplus(100)={v}, expected ≈100");
    }

    #[test]
    fn softplus_small_x() {
        // x=-100 should return ≈0
        let v = softplus(-100.0);
        assert!(v.abs() < 1e-6, "softplus(-100)={v}, expected ≈0");
    }

    #[test]
    fn softplus_zero() {
        // softplus(0) = log(2)
        let expected = std::f32::consts::LN_2;
        let v = softplus(0.0);
        assert!(
            (v - expected).abs() < EPS,
            "softplus(0)={v}, expected ln(2)={expected}"
        );
    }

    #[test]
    fn softplus_positive() {
        // softplus(x) > 0 for all x
        let xs = [-50.0_f32, -1.0, 0.0, 1.0, 10.0, 50.0];
        for x in xs {
            let v = softplus(x);
            assert!(v >= 0.0, "softplus({x})={v} should be >= 0");
        }
    }

    // ── SelectiveScanConfig ───────────────────────────────────────────────────

    #[test]
    fn config_valid() {
        let cfg = SelectiveScanConfig::new(2, 8, 4, 16).expect("valid config");
        assert_eq!(cfg.batch, 2);
        assert_eq!(cfg.seq_len, 8);
        assert_eq!(cfg.d_model, 4);
        assert_eq!(cfg.d_state, 16);
    }

    #[test]
    fn config_zero_batch() {
        let err = SelectiveScanConfig::new(0, 8, 4, 16).expect_err("should fail");
        assert!(matches!(err, MambaError::Internal(_)));
    }

    #[test]
    fn config_zero_d_state() {
        let err = SelectiveScanConfig::new(1, 8, 4, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    // ── selective_scan: output shape ──────────────────────────────────────────

    #[test]
    fn scan_output_shape() {
        let b = 2_usize;
        let l = 4_usize;
        let d = 3_usize;
        let n = 8_usize;
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("valid config");
        let u = vec![0.0_f32; b * l * d];
        let delta = vec![0.5_f32; b * l * d];
        let a_log = vec![0.0_f32; d * n];
        let b_proj = vec![0.1_f32; b * l * n];
        let c_proj = vec![0.1_f32; b * l * n];
        let y = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("selective scan");
        assert_eq!(y.len(), b * l * d, "output should have B*L*D elements");
    }

    // ── selective_scan: finiteness ────────────────────────────────────────────

    #[test]
    fn scan_output_finite() {
        let b = 2_usize;
        let l = 8_usize;
        let d = 4_usize;
        let n = 8_usize;
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("valid config");
        let mut rng = LcgRng::new(42);
        let mut u = vec![0.0_f32; b * l * d];
        let mut delta = vec![0.0_f32; b * l * d];
        let mut a_log = vec![0.0_f32; d * n];
        let mut b_proj = vec![0.0_f32; b * l * n];
        let mut c_proj = vec![0.0_f32; b * l * n];
        rng.fill_normal(&mut u);
        rng.fill_normal(&mut delta);
        rng.fill_normal(&mut a_log);
        rng.fill_normal(&mut b_proj);
        rng.fill_normal(&mut c_proj);
        let y = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("selective scan");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} is not finite");
        }
    }

    // ── selective_scan: zero input ────────────────────────────────────────────

    #[test]
    fn scan_zero_input() {
        // u=zeros, b_proj=zeros → state stays 0 → y=zeros
        let b = 2_usize;
        let l = 4_usize;
        let d = 3_usize;
        let n = 4_usize;
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("valid config");
        let u = vec![0.0_f32; b * l * d];
        let delta = vec![0.5_f32; b * l * d];
        let a_log = vec![0.0_f32; d * n];
        let b_proj = vec![0.0_f32; b * l * n];
        // c_proj can be nonzero: since h=0, y will still be 0
        let c_proj = vec![1.0_f32; b * l * n];
        let y = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("selective scan");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.abs() < 1e-7, "y[{i}]={v} should be zero for zero input");
        }
    }

    // ── selective_scan: manual unit test ─────────────────────────────────────

    /// B=1, L=2, D=1, N=1 — verify by hand:
    ///
    /// a_log=[0.0] → A = -exp(0.0) = -1.0
    /// delta_raw = [0.5, 0.5] → Δ = softplus(0.5) ≈ 0.97408...
    /// A_bar = exp(Δ * (-1.0)) = exp(-Δ)
    /// B_bar = Δ * b_proj
    ///
    /// t=0: h = 0 + B_bar * u_0; y_0 = C_0 * h
    /// t=1: h = A_bar * h + B_bar * u_1; y_1 = C_1 * h
    #[test]
    fn scan_unit_test_manual() {
        let cfg = SelectiveScanConfig::new(1, 2, 1, 1).expect("valid config");
        let u = vec![1.0_f32, 0.5_f32]; // [B=1, L=2, D=1]
        let delta = vec![0.5_f32, 0.5_f32]; // raw delta
        let a_log = vec![0.0_f32]; // a_log[0,0] = 0 → A = -1.0
        let b_proj = vec![1.0_f32, 1.0_f32]; // [B=1, L=2, N=1]
        let c_proj = vec![1.0_f32, 1.0_f32]; // [B=1, L=2, N=1]

        let dt = softplus(0.5_f32);
        let a_bar = (-dt).exp(); // exp(dt * (-1.0))
        let b_bar = dt * 1.0_f32; // dt * b_proj

        // t=0
        let h0 = b_bar * 1.0_f32; // A_bar * 0 + B_bar * u_0
        let y0_expected = 1.0 * h0; // C_0 * h0

        // t=1
        let h1 = a_bar * h0 + b_bar * 0.5_f32; // A_bar * h0 + B_bar * u_1
        let y1_expected = 1.0 * h1; // C_1 * h1

        let y = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("selective scan");

        assert_eq!(y.len(), 2);
        assert!(
            (y[0] - y0_expected).abs() < 1e-5,
            "y[0]={}, expected {y0_expected}",
            y[0]
        );
        assert!(
            (y[1] - y1_expected).abs() < 1e-5,
            "y[1]={}, expected {y1_expected}",
            y[1]
        );
    }

    // ── selective_scan: long sequence stability ───────────────────────────────

    #[test]
    fn scan_stable_for_long_sequences() {
        let b = 1_usize;
        let l = 512_usize;
        let d = 4_usize;
        let n = 8_usize;
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("valid config");
        let u = vec![0.1_f32; b * l * d];
        // delta raw slightly positive → softplus gives a reasonable Δ
        let delta = vec![0.0_f32; b * l * d];
        // a_log = 0 → A = -1 (stable, unit decay)
        let a_log = vec![0.0_f32; d * n];
        let b_proj = vec![0.01_f32; b * l * n];
        let c_proj = vec![1.0_f32; b * l * n];
        let y = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("selective scan");
        assert_eq!(y.len(), b * l * d);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}]={v} not finite at L=512");
        }
    }

    // ── selective_scan: batch independence ────────────────────────────────────

    #[test]
    fn scan_batch_independence() {
        let l = 6_usize;
        let d = 3_usize;
        let n = 4_usize;
        // Create single-batch data
        let cfg1 = SelectiveScanConfig::new(1, l, d, n).expect("valid config");
        let mut rng = LcgRng::new(77);
        let mut u_single = vec![0.0_f32; l * d];
        let mut delta_single = vec![0.0_f32; l * d];
        let mut b_single = vec![0.0_f32; l * n];
        let mut c_single = vec![0.0_f32; l * n];
        rng.fill_normal(&mut u_single);
        rng.fill_normal(&mut delta_single);
        rng.fill_normal(&mut b_single);
        rng.fill_normal(&mut c_single);
        let a_log = vec![0.5_f32; d * n]; // any valid a_log

        let y_single = selective_scan(
            &u_single,
            &delta_single,
            &a_log,
            &b_single,
            &c_single,
            &cfg1,
        )
        .expect("single batch scan");

        // Now replicate into batch of 2
        let cfg2 = SelectiveScanConfig::new(2, l, d, n).expect("valid config");
        let u2: Vec<f32> = u_single.iter().chain(u_single.iter()).copied().collect();
        let delta2: Vec<f32> = delta_single
            .iter()
            .chain(delta_single.iter())
            .copied()
            .collect();
        let b2: Vec<f32> = b_single.iter().chain(b_single.iter()).copied().collect();
        let c2: Vec<f32> = c_single.iter().chain(c_single.iter()).copied().collect();

        let y_batch = selective_scan(&u2, &delta2, &a_log, &b2, &c2, &cfg2).expect("batch scan");

        let stride = l * d;
        let y_b0 = &y_batch[..stride];
        let y_b1 = &y_batch[stride..];

        for (i, (&v0, &v1)) in y_b0.iter().zip(y_b1.iter()).enumerate() {
            assert!(
                (v0 - v1).abs() < 1e-5,
                "batch independence violated at i={i}: y_b0={v0}, y_b1={v1}"
            );
        }
        // Also verify against single-batch reference
        for (i, (&vs, &vb)) in y_single.iter().zip(y_b0.iter()).enumerate() {
            assert!(
                (vs - vb).abs() < 1e-5,
                "single vs batch mismatch at i={i}: single={vs}, batch={vb}"
            );
        }
    }

    // ── selective_scan: error on wrong shapes ─────────────────────────────────

    #[test]
    fn scan_error_wrong_u_len() {
        let cfg = SelectiveScanConfig::new(1, 4, 2, 4).expect("valid config");
        let u = vec![0.0_f32; 5]; // wrong: should be 1*4*2=8
        let delta = vec![0.0_f32; 8];
        let a_log = vec![0.0_f32; 8];
        let b_proj = vec![0.0_f32; 16];
        let c_proj = vec![0.0_f32; 16];
        let err =
            selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }

    #[test]
    fn scan_error_wrong_a_log_len() {
        let cfg = SelectiveScanConfig::new(1, 4, 2, 4).expect("valid config");
        let u = vec![0.0_f32; 8];
        let delta = vec![0.0_f32; 8];
        let a_log = vec![0.0_f32; 5]; // wrong: should be 2*4=8
        let b_proj = vec![0.0_f32; 16];
        let c_proj = vec![0.0_f32; 16];
        let err =
            selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }
}
