//! Intrinsic Plasticity (IP) — Triesch (2005), "A Gradient Rule for the
//! Intrinsic Plasticity of a Neuron".
//!
//! IP adapts a neuron's transfer-function gain `a` and bias `b` so that the
//! distribution of its output firing rate approaches a target exponential
//! distribution with fixed mean `μ` (the maximum-entropy distribution on the
//! non-negative reals with a fixed mean). Because the exponential places most
//! mass near zero with an occasional large response, IP acts as a homeostatic
//! mechanism that keeps the average activity at `μ` while encouraging sparse,
//! information-rich responses.
//!
//! ## Transfer function and gradient rule
//!
//! ```text
//! y    = σ(a·x + b)
//! Δb   = η · ( 1 − (2 + 1/μ)·y + (1/μ)·y² )
//! Δa   = η/a + Δb·x
//! ```
//!
//! where `x` is the total synaptic input to the neuron and `σ` is the logistic
//! sigmoid. The `b` update is applied first, then the `a` update reuses the same
//! `Δb`. The gain `a` is held away from zero (`|a| ≥ 1e-6`) so that the `η/a`
//! term stays finite. The sigmoid is evaluated with a numerically stable
//! two-branch form to avoid overflow for large `|a·x + b|`.

use crate::error::{SnnError, SnnResult};

/// Intrinsic-plasticity configuration.
#[derive(Debug, Clone, Copy)]
pub struct IpConfig {
    /// Learning rate `η`.
    pub eta: f32,
    /// Target mean firing rate `μ ∈ (0, 1)`; small values (≈ 0.1) give sparse,
    /// exponential-like output statistics.
    pub mu: f32,
}

impl Default for IpConfig {
    fn default() -> Self {
        Self {
            eta: 0.001,
            mu: 0.1,
        }
    }
}

/// Mutable intrinsic-plasticity state: per-neuron gain `a` and bias `b`.
#[derive(Debug, Clone)]
pub struct IpState {
    /// Transfer-function gain `a_i` per neuron (initialised to `1`).
    pub a: Vec<f32>,
    /// Transfer-function bias `b_i` per neuron (initialised to `0`).
    pub b: Vec<f32>,
}

impl IpState {
    /// Allocate state for `n` neurons with `a = 1`, `b = 0`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            a: vec![1.0_f32; n],
            b: vec![0.0_f32; n],
        }
    }
}

/// Smallest permitted magnitude for the gain `a`, guarding the `η/a` term.
const A_FLOOR: f32 = 1e-6;

/// Numerically stable logistic activation `σ(a·x + b) = 1 / (1 + exp(−(a·x+b)))`.
///
/// Uses the same two-branch form as [`crate::surrogate::sigmoid::stable_sigmoid`]
/// to avoid overflow at large negative arguments.
#[must_use]
pub fn ip_activation(x: f32, a: f32, b: f32) -> f32 {
    let z = a * x + b;
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Validate `cfg` and slice lengths used by [`ip_step`].
fn validate_ip(state: &IpState, inputs: &[f32], cfg: &IpConfig, y_out: &[f32]) -> SnnResult<()> {
    if !cfg.eta.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "eta".into(),
            val: cfg.eta,
        });
    }
    if !cfg.mu.is_finite() || cfg.mu <= 0.0 || cfg.mu >= 1.0 {
        return Err(SnnError::OutOfRange {
            name: "mu".into(),
            val: cfg.mu,
        });
    }
    let n = state.a.len();
    if state.b.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: state.b.len(),
        });
    }
    if inputs.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: inputs.len(),
        });
    }
    if y_out.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: y_out.len(),
        });
    }
    Ok(())
}

/// Advance the intrinsic-plasticity state by one step.
///
/// For each neuron computes `y = σ(a·x + b)`, writes it to `y_out`, then updates
/// `b` and `a` with Triesch's gradient rule using the just-computed `y`. The
/// gain `a` is kept away from zero before the `η/a` division.
///
/// # Errors
/// Returns `SnnError` if `eta` is non-finite, `mu ∉ (0, 1)`, or any slice length
/// does not match `state.a`.
pub fn ip_step(
    state: &mut IpState,
    inputs: &[f32],
    cfg: &IpConfig,
    y_out: &mut [f32],
) -> SnnResult<()> {
    validate_ip(state, inputs, cfg, y_out)?;
    let inv_mu = 1.0 / cfg.mu;
    for (((a, b), &x), y) in state
        .a
        .iter_mut()
        .zip(state.b.iter_mut())
        .zip(inputs.iter())
        .zip(y_out.iter_mut())
    {
        // Guard the gain away from zero so η/a stays finite.
        if a.abs() < A_FLOOR {
            *a = A_FLOOR.copysign(*a);
        }
        let y_val = ip_activation(x, *a, *b);
        *y = y_val;

        // Triesch gradient rule: update b first, then a reuses Δb.
        let delta_b = cfg.eta * (1.0 - (2.0 + inv_mu) * y_val + inv_mu * y_val * y_val);
        *b += delta_b;
        let delta_a = cfg.eta / *a + delta_b * x;
        *a += delta_a;

        // Keep a away from zero after the update as well.
        if a.abs() < A_FLOOR {
            *a = A_FLOOR.copysign(*a);
        }
    }
    Ok(())
}

/// Run intrinsic plasticity over a sequence of input vectors.
///
/// `input_sequence` is a list of per-step input vectors, all of the same length
/// matching `state.a`. Returns the per-step output `y` vectors in the same
/// shape.
///
/// # Errors
/// Returns `SnnError::EmptyInput` if `input_sequence` is empty, `SnnError`
/// (length variants) if any inner vector has the wrong length, or the same
/// validation errors as [`ip_step`].
pub fn ip_run(
    state: &mut IpState,
    input_sequence: &[Vec<f32>],
    cfg: &IpConfig,
) -> SnnResult<Vec<Vec<f32>>> {
    if input_sequence.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    let n = state.a.len();
    for step in input_sequence {
        if step.len() != n {
            return Err(SnnError::IncompatibleLength {
                a: n,
                b: step.len(),
            });
        }
    }
    let mut outputs = Vec::with_capacity(input_sequence.len());
    let mut y_buf = vec![0.0_f32; n];
    for step in input_sequence {
        ip_step(state, step, cfg, &mut y_buf)?;
        outputs.push(y_buf.clone());
    }
    Ok(outputs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic varied input generator in `[-1, 1)`.
    fn varied_inputs(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 % 13) as f32 / 13.0 - 0.5) * 2.0)
            .collect()
    }

    fn manual_sigmoid(x: f32, a: f32, b: f32) -> f32 {
        1.0 / (1.0 + (-(a * x + b)).exp())
    }

    #[test]
    fn state_init_a_one_b_zero() {
        let s = IpState::new(6);
        assert_eq!(s.a.len(), 6);
        assert_eq!(s.b.len(), 6);
        assert!(s.a.iter().all(|&a| (a - 1.0).abs() < 1e-9));
        assert!(s.b.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn activation_matches_manual_sigmoid() {
        for &(x, a, b) in &[
            (0.0_f32, 1.0_f32, 0.0_f32),
            (0.5, 2.0, -0.3),
            (-0.7, 1.5, 0.2),
            (1.0, 0.5, 0.5),
        ] {
            let got = ip_activation(x, a, b);
            let want = manual_sigmoid(x, a, b);
            assert!((got - want).abs() < 1e-6, "got={got} want={want}");
        }
    }

    #[test]
    fn activation_numerically_stable_for_extremes() {
        for &(x, a, b) in &[
            (1e6_f32, 1.0_f32, 0.0_f32),
            (-1e6, 1.0, 0.0),
            (1.0, 1e6, 1e6),
            (1.0, -1e6, -1e6),
        ] {
            let y = ip_activation(x, a, b);
            assert!(y.is_finite(), "activation not finite: {y}");
            assert!((0.0..=1.0).contains(&y), "activation out of (0,1): {y}");
        }
    }

    #[test]
    fn constant_input_drives_mean_toward_mu() {
        let cfg = IpConfig { eta: 0.01, mu: 0.1 };
        let n = 1;
        let mut state = IpState::new(n);
        let x = vec![1.0_f32; n];
        let mut y_buf = vec![0.0_f32; n];

        // Output before adaptation.
        ip_step(&mut state, &x, &cfg, &mut y_buf).expect("step");
        let y_before = y_buf[0];

        for _ in 0..5000 {
            ip_step(&mut state, &x, &cfg, &mut y_buf).expect("step");
        }
        let y_after = y_buf[0];
        assert!(
            (y_after - cfg.mu).abs() < (y_before - cfg.mu).abs(),
            "mean did not move toward μ: before={y_before}, after={y_after}, μ={}",
            cfg.mu
        );
    }

    #[test]
    fn gain_stays_away_from_zero() {
        let cfg = IpConfig { eta: 0.05, mu: 0.1 };
        let mut state = IpState::new(1);
        state.a[0] = 1e-9; // start essentially zero
        let x = vec![0.5_f32; 1];
        let mut y_buf = vec![0.0_f32; 1];
        for _ in 0..200 {
            ip_step(&mut state, &x, &cfg, &mut y_buf).expect("step");
            assert!(
                state.a[0].abs() >= A_FLOOR - 1e-12,
                "a too small: {}",
                state.a[0]
            );
            assert!(state.a[0].is_finite() && y_buf[0].is_finite());
        }
    }

    #[test]
    fn bias_decreases_when_output_persistently_high() {
        // Large positive input with a large gain keeps y ≈ 1 ≫ μ, so the bias
        // term 1 − (2 + 1/μ)·y + (1/μ)·y² evaluated near y=1 is 1 − 2 = −1 < 0,
        // pushing b down to lower excitability.
        let cfg = IpConfig { eta: 0.01, mu: 0.1 };
        let mut state = IpState::new(1);
        state.a[0] = 5.0;
        state.b[0] = 5.0; // start with high excitability
        let x = vec![3.0_f32; 1];
        let mut y_buf = vec![0.0_f32; 1];
        let b_start = state.b[0];
        for _ in 0..100 {
            ip_step(&mut state, &x, &cfg, &mut y_buf).expect("step");
        }
        assert!(
            state.b[0] < b_start,
            "bias should drop when y persistently high: {} -> {}",
            b_start,
            state.b[0]
        );
    }

    #[test]
    fn ip_run_shape_correct() {
        let cfg = IpConfig::default();
        let n = 4;
        let mut state = IpState::new(n);
        let seq: Vec<Vec<f32>> = (0..7).map(|_| varied_inputs(n)).collect();
        let out = ip_run(&mut state, &seq, &cfg).expect("run");
        assert_eq!(out.len(), 7);
        assert!(out.iter().all(|row| row.len() == n));
        // Outputs are valid probabilities.
        for row in &out {
            for &y in row {
                assert!((0.0..=1.0).contains(&y));
            }
        }
    }

    #[test]
    fn ip_run_rejects_empty() {
        let cfg = IpConfig::default();
        let mut state = IpState::new(3);
        let seq: Vec<Vec<f32>> = Vec::new();
        let err = ip_run(&mut state, &seq, &cfg);
        assert!(matches!(err, Err(SnnError::EmptyInput)));
    }

    #[test]
    fn ip_run_rejects_inconsistent_inner_length() {
        let cfg = IpConfig::default();
        let mut state = IpState::new(3);
        let seq = vec![vec![0.0_f32; 3], vec![0.0_f32; 2]];
        let err = ip_run(&mut state, &seq, &cfg);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn ip_step_rejects_bad_mu() {
        let mut state = IpState::new(2);
        let x = vec![0.0_f32; 2];
        let mut y_buf = vec![0.0_f32; 2];
        let too_low = IpConfig { eta: 0.01, mu: 0.0 };
        assert!(matches!(
            ip_step(&mut state, &x, &too_low, &mut y_buf),
            Err(SnnError::OutOfRange { .. })
        ));
        let too_high = IpConfig { eta: 0.01, mu: 1.0 };
        assert!(matches!(
            ip_step(&mut state, &x, &too_high, &mut y_buf),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn ip_step_rejects_length_mismatch() {
        let cfg = IpConfig::default();
        let mut state = IpState::new(2);
        let x = vec![0.0_f32; 3];
        let mut y_buf = vec![0.0_f32; 2];
        let err = ip_step(&mut state, &x, &cfg, &mut y_buf);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn convergence_varied_input_mean_near_mu() {
        // Feed a fixed deterministic varied input cycled over many steps and
        // check the empirical mean of y settles near μ.
        let cfg = IpConfig {
            eta: 0.02,
            mu: 0.15,
        };
        let n = 16;
        let mut state = IpState::new(n);
        let base = varied_inputs(n);
        let mut y_buf = vec![0.0_f32; n];

        // Warm-up adaptation.
        for _ in 0..8000 {
            ip_step(&mut state, &base, &cfg, &mut y_buf).expect("step");
        }
        // Measure empirical mean over a long run.
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for _ in 0..4000 {
            ip_step(&mut state, &base, &cfg, &mut y_buf).expect("step");
            for &y in &y_buf {
                sum += y as f64;
                count += 1;
            }
        }
        let empirical_mean = (sum / count as f64) as f32;
        assert!(
            (empirical_mean - cfg.mu).abs() < 0.05,
            "empirical mean {empirical_mean} not within 0.05 of μ={}",
            cfg.mu
        );
    }
}
