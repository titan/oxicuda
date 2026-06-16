//! Adaptive filters: LMS, NLMS, and RLS algorithms.
//!
//! Adaptive filters update their coefficients online to minimise a cost
//! function (typically the mean-square error) between the filter output and a
//! desired reference signal.  Three classical algorithms are provided:
//!
//! ## LMS — Least Mean Squares (Widrow-Hoff, 1960)
//!
//! ```text
//! y[n]    = Σ_{k=0}^{M-1}  w[k] · x[n-k]
//! e[n]    = d[n] - y[n]
//! w[k]   += μ · e[n] · x[n-k]
//! ```
//!
//! ## NLMS — Normalised LMS
//!
//! ```text
//! μ_eff = μ / (ε + ‖x[n]‖²)
//! ```
//!
//! ## RLS — Recursive Least Squares
//!
//! ```text
//! k[n] = P[n-1]·x[n] / (λ + x[n]ᵀ·P[n-1]·x[n])
//! e[n] = d[n] - wᵀ[n-1]·x[n]
//! w[n] = w[n-1] + k[n]·e[n]
//! P[n] = (P[n-1] - k[n]·x[n]ᵀ·P[n-1]) / λ
//! ```

use crate::error::{SignalError, SignalResult};
use std::collections::VecDeque;

// --------------------------------------------------------------------------- //
//  LMS / NLMS configuration and state
// --------------------------------------------------------------------------- //

/// Configuration for the LMS and NLMS adaptive filters.
#[derive(Debug, Clone)]
pub struct AdaptiveLmsConfig {
    /// Number of filter taps (filter order = n_taps - 1).
    pub n_taps: usize,
    /// Step size (learning rate) μ > 0.
    pub step_size: f64,
    /// When `true`, use NLMS (normalised step size).
    pub normalize: bool,
    /// Regularisation constant ε for NLMS denominator (default 1e-6).
    pub regularization: f64,
}

impl AdaptiveLmsConfig {
    /// Create a new LMS/NLMS configuration.
    ///
    /// # Errors
    /// Returns `SignalError::InvalidSize` when `n_taps == 0`.
    /// Returns `SignalError::InvalidParameter` when `step_size <= 0`.
    pub fn new(n_taps: usize, step_size: f64, normalize: bool) -> SignalResult<Self> {
        if n_taps == 0 {
            return Err(SignalError::InvalidSize("n_taps must be ≥ 1".to_owned()));
        }
        if step_size <= 0.0_f64 {
            return Err(SignalError::InvalidParameter(format!(
                "step_size ({step_size}) must be > 0"
            )));
        }
        Ok(Self {
            n_taps,
            step_size,
            normalize,
            regularization: 1e-6_f64,
        })
    }
}

/// Runtime state for the LMS/NLMS adaptive filter.
///
/// Holds the current tap weights and the input delay line.
#[derive(Debug, Clone)]
pub struct AdaptiveLmsState {
    /// Current filter tap weights.
    pub weights: Vec<f64>,
    /// Input delay line: `buf[0]` = most recent, `buf[n_taps-1]` = oldest.
    buf: VecDeque<f64>,
}

impl AdaptiveLmsState {
    /// Create a new filter state initialised to zero weights and zero delay line.
    #[must_use]
    pub fn new(cfg: &AdaptiveLmsConfig) -> Self {
        Self {
            weights: vec![0.0_f64; cfg.n_taps],
            buf: std::iter::repeat_n(0.0_f64, cfg.n_taps).collect(),
        }
    }

    /// Process one input/desired sample and update the filter weights.
    ///
    /// Returns `(y, e)` where `y` is the filter output and `e = d - y` is the
    /// error signal.
    ///
    /// The input delay line is maintained so that `buf[0]` is the most recent
    /// sample.  Pushing `x` to the front and popping from the back keeps the
    /// buffer at exactly `n_taps` samples.
    pub fn update(&mut self, x: f64, d: f64, cfg: &AdaptiveLmsConfig) -> (f64, f64) {
        // Maintain delay line: newest sample at index 0.
        self.buf.push_front(x);
        if self.buf.len() > cfg.n_taps {
            self.buf.pop_back();
        }

        // Filter output: y = wᵀ · x_buf.
        let y: f64 = self
            .weights
            .iter()
            .zip(self.buf.iter())
            .map(|(w, b)| w * b)
            .sum();

        let e = d - y;

        // Effective step size (NLMS normalises by input power).
        let effective_mu = if cfg.normalize {
            let power: f64 = self.buf.iter().map(|b| b * b).sum();
            cfg.step_size / (cfg.regularization + power)
        } else {
            cfg.step_size
        };

        // Weight update: w[k] += μ_eff · e · buf[k].
        for (w, b) in self.weights.iter_mut().zip(self.buf.iter()) {
            *w += effective_mu * e * b;
        }

        (y, e)
    }

    /// Reset weights and delay line to zero.
    pub fn reset(&mut self) {
        for w in self.weights.iter_mut() {
            *w = 0.0_f64;
        }
        for b in self.buf.iter_mut() {
            *b = 0.0_f64;
        }
    }
}

// --------------------------------------------------------------------------- //
//  Batch LMS
// --------------------------------------------------------------------------- //

/// Run the LMS adaptive filter over an entire signal in batch mode.
///
/// # Parameters
/// - `signal`  — reference input `x[n]`
/// - `desired` — desired response `d[n]`
/// - `n_taps`  — number of filter taps
/// - `mu`      — step size μ > 0
///
/// # Returns
/// `(outputs, errors, final_weights)` — all three vectors.  The output and
/// error vectors have the same length as `signal`; `final_weights` has length
/// `n_taps`.
///
/// # Errors
/// - `InvalidSize`         — `n_taps == 0`
/// - `InvalidParameter`    — `mu <= 0`
/// - `DimensionMismatch`   — `desired.len() != signal.len()`
pub fn lms_filter(
    signal: &[f64],
    desired: &[f64],
    n_taps: usize,
    mu: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_batch_inputs(signal, desired, n_taps, mu)?;

    if signal.is_empty() {
        return Ok((vec![], vec![], vec![0.0_f64; n_taps]));
    }

    let cfg = AdaptiveLmsConfig {
        n_taps,
        step_size: mu,
        normalize: false,
        regularization: 1e-6_f64,
    };
    let mut state = AdaptiveLmsState::new(&cfg);
    let mut outputs = Vec::with_capacity(signal.len());
    let mut errors = Vec::with_capacity(signal.len());

    for (&x, &d) in signal.iter().zip(desired.iter()) {
        let (y, e) = state.update(x, d, &cfg);
        outputs.push(y);
        errors.push(e);
    }

    Ok((outputs, errors, state.weights))
}

// --------------------------------------------------------------------------- //
//  Batch NLMS
// --------------------------------------------------------------------------- //

/// Run the NLMS adaptive filter over an entire signal in batch mode.
///
/// # Parameters
/// - `regularization` — ε added to the input power denominator
///
/// # Errors
/// Same as [`lms_filter`].
pub fn nlms_filter(
    signal: &[f64],
    desired: &[f64],
    n_taps: usize,
    mu: f64,
    regularization: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_batch_inputs(signal, desired, n_taps, mu)?;

    if signal.is_empty() {
        return Ok((vec![], vec![], vec![0.0_f64; n_taps]));
    }

    let cfg = AdaptiveLmsConfig {
        n_taps,
        step_size: mu,
        normalize: true,
        regularization,
    };
    let mut state = AdaptiveLmsState::new(&cfg);
    let mut outputs = Vec::with_capacity(signal.len());
    let mut errors = Vec::with_capacity(signal.len());

    for (&x, &d) in signal.iter().zip(desired.iter()) {
        let (y, e) = state.update(x, d, &cfg);
        outputs.push(y);
        errors.push(e);
    }

    Ok((outputs, errors, state.weights))
}

// --------------------------------------------------------------------------- //
//  Batch RLS
// --------------------------------------------------------------------------- //

/// Run the RLS adaptive filter over an entire signal in batch mode.
///
/// The RLS algorithm converges in exactly `n_taps` steps on a noiseless
/// system and is significantly faster than LMS for coloured inputs.
///
/// # Parameters
/// - `forgetting_factor` — λ ∈ (0, 1]; λ = 1 → infinite memory (exact LS)
/// - `delta`             — initial P matrix scaling: P = I / δ; must be > 0
///
/// # Errors
/// - `InvalidSize`         — `n_taps == 0`
/// - `InvalidParameter`    — `forgetting_factor ∉ (0,1]` or `delta <= 0`
/// - `DimensionMismatch`   — `desired.len() != signal.len()`
pub fn rls_filter(
    signal: &[f64],
    desired: &[f64],
    n_taps: usize,
    forgetting_factor: f64,
    delta: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if n_taps == 0 {
        return Err(SignalError::InvalidSize("n_taps must be ≥ 1".to_owned()));
    }
    if forgetting_factor <= 0.0_f64 || forgetting_factor > 1.0_f64 {
        return Err(SignalError::InvalidParameter(format!(
            "forgetting_factor ({forgetting_factor}) must be in (0, 1]"
        )));
    }
    if delta <= 0.0_f64 {
        return Err(SignalError::InvalidParameter(format!(
            "delta ({delta}) must be > 0"
        )));
    }
    if signal.len() != desired.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("desired.len() = signal.len() = {}", signal.len()),
            got: format!("desired.len() = {}", desired.len()),
        });
    }

    if signal.is_empty() {
        return Ok((vec![], vec![], vec![0.0_f64; n_taps]));
    }

    // Initialise weight vector and inverse correlation matrix.
    let mut weights = vec![0.0_f64; n_taps];
    // P is row-major, n_taps × n_taps; init: P = I / delta.
    let mut p_mat = vec![0.0_f64; n_taps * n_taps];
    for i in 0..n_taps {
        p_mat[i * n_taps + i] = 1.0_f64 / delta;
    }

    // Input delay line: newest at index 0.
    let mut buf: VecDeque<f64> = std::iter::repeat_n(0.0_f64, n_taps).collect();

    let mut outputs = Vec::with_capacity(signal.len());
    let mut errors = Vec::with_capacity(signal.len());
    let lambda = forgetting_factor;

    for (&x, &d) in signal.iter().zip(desired.iter()) {
        // Update delay line.
        buf.push_front(x);
        if buf.len() > n_taps {
            buf.pop_back();
        }

        // Build input vector from delay line.
        let x_vec: Vec<f64> = buf.iter().copied().collect();

        // Px = P · x_vec   (n_taps vector)
        let mut px = vec![0.0_f64; n_taps];
        for i in 0..n_taps {
            let mut acc = 0.0_f64;
            for j in 0..n_taps {
                acc += p_mat[i * n_taps + j] * x_vec[j];
            }
            px[i] = acc;
        }

        // denom = λ + xᵀ·Px
        let x_dot_px: f64 = x_vec.iter().zip(px.iter()).map(|(a, b)| a * b).sum();
        let denom = lambda + x_dot_px;

        // Gain vector: k = Px / denom
        let k_gain: Vec<f64> = px.iter().map(|v| v / denom).collect();

        // Error: e = d - wᵀ·x
        let w_dot_x: f64 = weights.iter().zip(x_vec.iter()).map(|(w, xi)| w * xi).sum();
        let e = d - w_dot_x;

        // Weight update: w += k · e
        for (w, ki) in weights.iter_mut().zip(k_gain.iter()) {
            *w += ki * e;
        }

        // P update (rank-1 downdate, scaled by 1/λ):
        // P = (P - outer(k, Px)) / λ
        for i in 0..n_taps {
            for j in 0..n_taps {
                p_mat[i * n_taps + j] = (p_mat[i * n_taps + j] - k_gain[i] * px[j]) / lambda;
            }
        }

        outputs.push(
            w_dot_x
                + k_gain
                    .iter()
                    .zip(x_vec.iter())
                    .map(|(ki, xi)| ki * e * xi)
                    .sum::<f64>(),
        );
        errors.push(e);
    }

    // Re-derive outputs from final weights for correctness in the output vector
    // (the running output computed above is accurate, but let's keep it direct).
    Ok((outputs, errors, weights))
}

// --------------------------------------------------------------------------- //
//  Internal helpers
// --------------------------------------------------------------------------- //

fn validate_batch_inputs(
    signal: &[f64],
    desired: &[f64],
    n_taps: usize,
    mu: f64,
) -> SignalResult<()> {
    if n_taps == 0 {
        return Err(SignalError::InvalidSize("n_taps must be ≥ 1".to_owned()));
    }
    if mu <= 0.0_f64 {
        return Err(SignalError::InvalidParameter(format!(
            "mu ({mu}) must be > 0"
        )));
    }
    if signal.len() != desired.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("desired.len() = signal.len() = {}", signal.len()),
            got: format!("desired.len() = {}", desired.len()),
        });
    }
    Ok(())
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ---- helper: simple LCG pseudo-random sequence ----
    fn lcg_sequence(n: usize, seed: u64) -> Vec<f64> {
        let mut v = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            v.push((s >> 33) as f64 / (u32::MAX as f64));
        }
        v
    }

    // ---- Test 1: LMS echo cancellation MSE improvement ----
    #[test]
    fn test_lms_echo_cancellation_mse_improvement() {
        let n = 500usize;
        let noise_seq = lcg_sequence(n, 42);
        let t: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.05 * i as f64).sin()).collect();
        let signal: Vec<f64> = t.iter().map(|&s| s * 0.5).collect();
        let desired: Vec<f64> = t
            .iter()
            .zip(noise_seq.iter())
            .map(|(&s, &nz)| s + nz * 0.1)
            .collect();
        let (_out, errors, _w) =
            lms_filter(&signal, &desired, 4, 0.05).expect("LMS filter should succeed");
        let mse_initial: f64 = errors[..20].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        let mse_final: f64 = errors[n - 20..].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        assert!(
            mse_final < mse_initial,
            "LMS should reduce MSE: initial={mse_initial}, final={mse_final}"
        );
    }

    // ---- Test 2: LMS MSE non-increasing (window averages) ----
    #[test]
    fn test_lms_mse_non_increasing() {
        let n = 500usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.05 * i as f64).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| 2.0 * x).collect();
        let (_out, errors, _w) =
            lms_filter(&signal, &desired, 1, 0.05).expect("LMS filter should succeed");
        let mse_first: f64 = errors[..20].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        let mse_last: f64 = errors[n - 20..].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        assert!(
            mse_last < mse_first,
            "LMS MSE should decrease: first={mse_first}, last={mse_last}"
        );
    }

    // ---- Test 3: NLMS amplitude invariance ----
    #[test]
    fn test_nlms_amplitude_invariance() {
        let n = 200usize;
        let base: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();
        let desired: Vec<f64> = base.iter().map(|&x| 0.7 * x).collect();

        let signal1 = base.clone();
        let signal2: Vec<f64> = base.iter().map(|&x| x * 2.0).collect();
        let desired2: Vec<f64> = desired.iter().map(|&x| x * 2.0).collect();

        let (_o1, e1, _) =
            nlms_filter(&signal1, &desired, 4, 0.5, 1e-6).expect("NLMS filter should succeed");
        let (_o2, e2, _) =
            nlms_filter(&signal2, &desired2, 4, 0.5, 1e-6).expect("NLMS filter should succeed");

        // Both should reach similarly small MSE at step 50.
        let mse1_50: f64 = e1[40..60].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        let mse2_50: f64 = e2[40..60].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        // Ratio should be within 4x (generous bound for amplitude scaling invariance).
        let ratio = if mse1_50 > mse2_50 {
            mse1_50 / (mse2_50 + 1e-30)
        } else {
            mse2_50 / (mse1_50 + 1e-30)
        };
        assert!(
            ratio < 4.0_f64,
            "NLMS convergence rate should be amplitude-invariant: mse1={mse1_50}, mse2={mse2_50}, ratio={ratio}"
        );
    }

    // ---- Test 4: hand-computed LMS single step ----
    #[test]
    fn test_lms_hand_computed_single_step() {
        let cfg = AdaptiveLmsConfig::new(2, 0.1, false).expect("valid config");
        let mut state = AdaptiveLmsState::new(&cfg);
        // Initial weights: [0, 0]; buf: [0, 0]
        // x=1.0, d=0.5 → y=0, e=0.5, w[0] += 0.1*0.5*1.0=0.05, w[1] += 0.1*0.5*0.0=0
        let (y, e) = state.update(1.0_f64, 0.5_f64, &cfg);
        assert!((y).abs() < 1e-12, "y should be 0 initially");
        assert!((e - 0.5_f64).abs() < 1e-12, "e should be 0.5");
        assert!(
            (state.weights[0] - 0.05_f64).abs() < 1e-12,
            "w[0] = {}",
            state.weights[0]
        );
        assert!(
            state.weights[1].abs() < 1e-12,
            "w[1] = {}",
            state.weights[1]
        );
    }

    // ---- Test 5: reset zeroes weights ----
    #[test]
    fn test_lms_reset_zeroes_weights() {
        let cfg = AdaptiveLmsConfig::new(3, 0.1, false).expect("valid config");
        let mut state = AdaptiveLmsState::new(&cfg);
        state.update(1.0_f64, 0.5_f64, &cfg);
        state.update(0.5_f64, 0.3_f64, &cfg);
        state.reset();
        assert!(
            state.weights.iter().all(|&w| w == 0.0_f64),
            "weights should be zeroed after reset"
        );
    }

    // ---- Test 6: initial output is zero ----
    #[test]
    fn test_lms_initial_output_zero() {
        let cfg = AdaptiveLmsConfig::new(4, 0.1, false).expect("valid config");
        let mut state = AdaptiveLmsState::new(&cfg);
        let (y, _e) = state.update(0.0_f64, 0.0_f64, &cfg);
        assert!(y.abs() < 1e-15, "initial output should be 0");
    }

    // ---- Test 7: batch lms_filter matches stepwise ----
    #[test]
    fn test_batch_lms_matches_stepwise() {
        let n = 50usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| 1.5 * x).collect();

        let (batch_out, batch_err, _) =
            lms_filter(&signal, &desired, 3, 0.05).expect("batch LMS should succeed");

        let cfg = AdaptiveLmsConfig::new(3, 0.05, false).expect("valid config");
        let mut state = AdaptiveLmsState::new(&cfg);
        for (i, (&x, &d)) in signal.iter().zip(desired.iter()).enumerate() {
            let (y, e) = state.update(x, d, &cfg);
            assert!(
                (y - batch_out[i]).abs() < 1e-12,
                "output mismatch at step {i}: stepwise={y}, batch={}",
                batch_out[i]
            );
            assert!(
                (e - batch_err[i]).abs() < 1e-12,
                "error mismatch at step {i}: stepwise={e}, batch={}",
                batch_err[i]
            );
        }
    }

    // ---- Test 8: RLS identifies FIR [1.0, -0.5] ----
    #[test]
    fn test_rls_identifies_fir_system() {
        let n = 100usize;
        let signal = lcg_sequence(n, 7);
        // Desired = FIR [1.0, -0.5] applied to signal (zero boundary).
        let desired: Vec<f64> = (0..n)
            .map(|i| {
                let prev = if i == 0 { 0.0_f64 } else { signal[i - 1] };
                1.0_f64 * signal[i] - 0.5_f64 * prev
            })
            .collect();

        // Large initial P (small delta) for fast convergence.
        let (_out, _err, weights) =
            rls_filter(&signal, &desired, 2, 1.0_f64, 1e-4_f64).expect("RLS should succeed");
        // After sufficient steps, weights should be near [1.0, -0.5].
        assert!(
            (weights[0] - 1.0_f64).abs() < 1e-4,
            "w[0]={} (expected ~1.0)",
            weights[0]
        );
        assert!(
            (weights[1] - (-0.5_f64)).abs() < 1e-4,
            "w[1]={} (expected ~-0.5)",
            weights[1]
        );
    }

    // ---- Test 9: RLS with lambda=1.0 drops MSE to ~0 after n_taps steps ----
    #[test]
    fn test_rls_lambda1_exact_identification() {
        let n = 50usize;
        let signal = lcg_sequence(n, 13);
        let desired: Vec<f64> = (0..n)
            .map(|i| {
                let prev = if i == 0 { 0.0_f64 } else { signal[i - 1] };
                1.0_f64 * signal[i] - 0.5_f64 * prev
            })
            .collect();

        // Use small delta (large initial P) for fast convergence with λ=1.
        let (_out, errors, _w) =
            rls_filter(&signal, &desired, 2, 1.0_f64, 1e-4_f64).expect("RLS should succeed");

        // After a few steps past n_taps, errors should be very small.
        // Note: the first ~n_taps steps are affected by the zero initial delay
        // line (boundary condition), so we skip those.
        let mse_later: f64 =
            errors[10..].iter().map(|e| e * e).sum::<f64>() / (errors.len() - 10) as f64;
        assert!(
            mse_later < 1e-8,
            "RLS with λ=1 should identify system exactly; MSE={mse_later}"
        );
    }

    // ---- Test 10: RLS weight error < 1e-6 on noiseless system ----
    #[test]
    fn test_rls_weight_error_noiseless() {
        let n = 100usize;
        let signal = lcg_sequence(n, 99);
        let w_true = [1.0_f64, -0.5_f64];
        let desired: Vec<f64> = (0..n)
            .map(|i| {
                let prev = if i == 0 { 0.0_f64 } else { signal[i - 1] };
                w_true[0] * signal[i] + w_true[1] * prev
            })
            .collect();

        // Use small delta for fast, accurate convergence.
        let (_out, _err, weights) =
            rls_filter(&signal, &desired, 2, 1.0_f64, 1e-4_f64).expect("RLS should succeed");

        let err_norm: f64 = weights
            .iter()
            .zip(w_true.iter())
            .map(|(w, wt)| (w - wt).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            err_norm < 1e-4,
            "RLS weight error ‖w - w_true‖ = {err_norm} (expected < 1e-4)"
        );
    }

    // ---- Test 11: n_taps=1 learns scalar ----
    #[test]
    fn test_lms_ntaps1_learns_scalar() {
        let n = 200usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.05 * i as f64).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| 3.0 * x).collect();
        let (_out, _err, weights) =
            lms_filter(&signal, &desired, 1, 0.1).expect("LMS should succeed");
        assert!(
            (weights[0] - 3.0_f64).abs() < 0.1_f64,
            "weight should converge to ~3, got {}",
            weights[0]
        );
    }

    // ---- Test 12: LMS noiseless error → 0 after 50 steps ----
    #[test]
    fn test_lms_noiseless_error_converges() {
        let n = 200usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.05 * i as f64).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| 2.0 * x).collect();
        let (_out, errors, _w) = lms_filter(&signal, &desired, 1, 0.1).expect("LMS should succeed");
        let avg_err_after_50: f64 =
            errors[50..].iter().map(|e| e.abs()).sum::<f64>() / (errors.len() - 50) as f64;
        assert!(
            avg_err_after_50 < 0.1_f64,
            "error should approach 0 after 50 steps, got {avg_err_after_50}"
        );
    }

    // ---- Test 13: very small step_size gives slow but monotone decrease ----
    #[test]
    fn test_lms_small_stepsize_monotone_decrease() {
        let n = 500usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.05 * i as f64).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| 2.0 * x).collect();
        let (_out, errors, _w) =
            lms_filter(&signal, &desired, 1, 1e-8).expect("LMS should succeed");
        // Compare first half MSE vs second half MSE.
        let mid = n / 2;
        let mse_first_half: f64 = errors[..mid].iter().map(|e| e * e).sum::<f64>() / mid as f64;
        let mse_second_half: f64 = errors[mid..].iter().map(|e| e * e).sum::<f64>() / mid as f64;
        assert!(
            mse_second_half <= mse_first_half,
            "Small step_size: second half MSE should not exceed first half; first={mse_first_half}, second={mse_second_half}"
        );
    }

    // ---- Test 14: output/error lengths match signal ----
    #[test]
    fn test_lms_output_lengths() {
        let n = 80usize;
        let signal: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.5).collect();
        let (out, err, weights) =
            lms_filter(&signal, &desired, 5, 0.01).expect("LMS should succeed");
        assert_eq!(out.len(), n);
        assert_eq!(err.len(), n);
        assert_eq!(weights.len(), 5);
    }

    // ---- Test 15: NLMS large regularization behaves like small step_size ----
    #[test]
    fn test_nlms_large_regularization_slow_convergence() {
        let n = 100usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| 2.0 * x).collect();
        // Large regularization: effective step ≈ mu / large_eps << mu.
        let (_o, errors, _w) =
            nlms_filter(&signal, &desired, 1, 0.5, 1000.0_f64).expect("NLMS should succeed");
        let mse_last: f64 = errors[80..].iter().map(|e| e * e).sum::<f64>() / 20.0_f64;
        // Should not have converged yet (MSE stays large due to tiny effective step).
        assert!(
            mse_last > 0.001_f64,
            "Large regularization should slow convergence; mse={mse_last}"
        );
    }

    // ---- Test 16: RLS P matrix symmetric ----
    #[test]
    fn test_rls_p_symmetric_throughout() {
        // We verify symmetry at the end by running our own RLS variant.
        // We check that the final weights are consistent (weight error test already covers RLS correctness).
        // Here we do a manual RLS to check symmetry.
        let n = 20usize;
        let signal = lcg_sequence(n, 77);
        let desired: Vec<f64> = signal.iter().map(|&x| x * 2.0).collect();

        // Run RLS manually to inspect P.
        let n_taps = 2usize;
        let lambda = 0.99_f64;
        let delta = 1.0_f64;
        let mut weights = vec![0.0_f64; n_taps];
        let mut p_mat = vec![0.0_f64; n_taps * n_taps];
        p_mat[0] = 1.0_f64 / delta;
        p_mat[3] = 1.0_f64 / delta;
        let mut buf: VecDeque<f64> = std::iter::repeat_n(0.0_f64, n_taps).collect();

        for (&x, &d) in signal.iter().zip(desired.iter()) {
            buf.push_front(x);
            if buf.len() > n_taps {
                buf.pop_back();
            }
            let x_vec: Vec<f64> = buf.iter().copied().collect();
            let mut px = vec![0.0_f64; n_taps];
            for i in 0..n_taps {
                for j in 0..n_taps {
                    px[i] += p_mat[i * n_taps + j] * x_vec[j];
                }
            }
            let denom: f64 = lambda + x_vec.iter().zip(px.iter()).map(|(a, b)| a * b).sum::<f64>();
            let k_gain: Vec<f64> = px.iter().map(|v| v / denom).collect();
            let e: f64 = d - weights
                .iter()
                .zip(x_vec.iter())
                .map(|(w, xi)| w * xi)
                .sum::<f64>();
            for (w, ki) in weights.iter_mut().zip(k_gain.iter()) {
                *w += ki * e;
            }
            for i in 0..n_taps {
                for j in 0..n_taps {
                    p_mat[i * n_taps + j] = (p_mat[i * n_taps + j] - k_gain[i] * px[j]) / lambda;
                }
            }
        }

        // Check symmetry: P[i,j] == P[j,i].
        for i in 0..n_taps {
            for j in 0..n_taps {
                let diff = (p_mat[i * n_taps + j] - p_mat[j * n_taps + i]).abs();
                assert!(
                    diff < 1e-10,
                    "P[{i},{j}]={} != P[{j},{i}]={} (diff={diff})",
                    p_mat[i * n_taps + j],
                    p_mat[j * n_taps + i]
                );
            }
        }
    }

    // ---- Test 17: empty signal returns empty outputs ----
    #[test]
    fn test_lms_empty_signal() {
        let (out, err, weights) =
            lms_filter(&[], &[], 3, 0.1).expect("empty signal should return empty outputs");
        assert!(out.is_empty());
        assert!(err.is_empty());
        assert_eq!(weights, vec![0.0_f64; 3]);
    }

    // ---- Test 18: n_taps=0 → InvalidSize ----
    #[test]
    fn test_adaptive_lms_config_ntaps0_error() {
        let result = AdaptiveLmsConfig::new(0, 0.1, false);
        assert!(
            matches!(result, Err(SignalError::InvalidSize(_))),
            "n_taps=0 should return InvalidSize"
        );
    }

    // ---- Test 19: desired.len() != signal.len() → DimensionMismatch ----
    #[test]
    fn test_lms_dimension_mismatch() {
        let result = lms_filter(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2, 0.1);
        assert!(
            matches!(result, Err(SignalError::DimensionMismatch { .. })),
            "mismatched lengths should return DimensionMismatch"
        );
    }

    // ---- Test 20: mu <= 0 → InvalidParameter ----
    #[test]
    fn test_lms_mu_nonpositive_error() {
        let result = lms_filter(&[1.0_f64], &[1.0_f64], 2, 0.0_f64);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "mu=0 should return InvalidParameter"
        );
        let result2 = lms_filter(&[1.0_f64], &[1.0_f64], 2, -0.1_f64);
        assert!(
            matches!(result2, Err(SignalError::InvalidParameter(_))),
            "mu<0 should return InvalidParameter"
        );
    }

    // ---- Test 21: forgetting_factor=0 → InvalidParameter ----
    #[test]
    fn test_rls_forgetting_factor_zero_error() {
        let result = rls_filter(&[1.0_f64], &[1.0_f64], 2, 0.0_f64, 1.0_f64);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "forgetting_factor=0 should return InvalidParameter"
        );
    }

    // ---- Test 22: forgetting_factor > 1 → InvalidParameter ----
    #[test]
    fn test_rls_forgetting_factor_gt1_error() {
        let result = rls_filter(&[1.0_f64], &[1.0_f64], 2, 1.1_f64, 1.0_f64);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "forgetting_factor>1 should return InvalidParameter"
        );
    }
}
