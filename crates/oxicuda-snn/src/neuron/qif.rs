//! Quadratic Integrate-and-Fire (QIF) neuron and the canonical Theta neuron.
//!
//! These are the canonical *Type-I* excitable models: near the spiking
//! bifurcation the firing rate grows continuously from zero (a saddle-node on an
//! invariant circle / SNIC bifurcation). The QIF voltage model and the
//! Ermentrout–Kopell theta phase model are exactly equivalent under the
//! transform `θ = 2·atan(v)`.
//!
//! ## QIF dynamics (explicit Euler)
//!
//! ```text
//! τ_m · dv/dt = (v − v_rest)(v − v_c) + R·I
//! v_{t+1}     = v_t + (dt/τ_m) · [ (v_t − v_rest)(v_t − v_c) + r_in·I_t ]
//! s_{t+1}     = 1 if v_{t+1} ≥ v_peak else 0
//! v_{t+1}     ← v_reset if s_{t+1} = 1                       (Hard reset)
//! ```
//!
//! Here `v_rest < v_c`, with `v_c` the critical/threshold voltage. With no input
//! a state above `v_c` diverges (and is registered as a spike), while a state
//! below `v_c` decays back toward `v_rest`. `r_in` is the input resistance `R`.
//! Because the quadratic term diverges in finite time, the post-update voltage
//! is clamped to `[-1e6, v_peak]` to guard against `NaN`/overflow before the
//! threshold test.
//!
//! ## Theta neuron (Ermentrout–Kopell canonical form)
//!
//! ```text
//! dθ/dt = (1/τ) · [ (1 − cos θ) + (1 + cos θ)·I ]
//! ```
//!
//! A spike is emitted when `θ` crosses `π` from below; the phase is then wrapped
//! by subtracting `2π` so it remains in `(−π, π]`. For constant `I > 0` the rest
//! state disappears and the phase circulates, producing repetitive firing; for
//! `I ≤ 0` the phase settles at a stable fixed point and is silent. The phase and
//! the QIF voltage are related by `v = tan(θ/2)` and `θ = 2·atan(v)`.

use crate::error::{SnnError, SnnResult};
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// QIF
// ─────────────────────────────────────────────────────────────────────────────

/// QIF configuration; `tau_m` and `dt` must be strictly positive and finite.
#[derive(Debug, Clone, Copy)]
pub struct QifConfig {
    /// Membrane time constant `τ_m` in the same time units as `dt`.
    pub tau_m: f32,
    /// Resting potential `v_rest`; the stable fixed point below `v_c`.
    pub v_rest: f32,
    /// Critical/threshold voltage `v_c`; above it (no input) the neuron diverges.
    pub v_c: f32,
    /// Peak voltage at which a spike is registered.
    pub v_peak: f32,
    /// Reset potential applied (hard reset) after a spike.
    pub v_reset: f32,
    /// Input resistance `R` scaling the injected current.
    pub r_in: f32,
    /// Integration time step.
    pub dt: f32,
}

impl Default for QifConfig {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            v_rest: -65.0,
            v_c: -50.0,
            v_peak: 30.0,
            v_reset: -65.0,
            r_in: 1.0,
            dt: 0.1,
        }
    }
}

/// Mutable QIF state (membrane potential per neuron).
#[derive(Debug, Clone)]
pub struct QifState {
    /// Membrane potential `v_i` for each neuron, length `n`.
    pub v: Vec<f32>,
}

impl QifState {
    /// Allocate state for `n` neurons with `v` initialised to zero.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            v: vec![0.0_f32; n],
        }
    }

    /// Allocate state for `n` neurons with `v` initialised to `v_rest`.
    #[must_use]
    pub fn with_rest(n: usize, v_rest: f32) -> Self {
        Self { v: vec![v_rest; n] }
    }
}

/// Lower clamp guarding against the quadratic term diverging to `−∞`.
const QIF_V_FLOOR: f32 = -1e6;

/// Validate `cfg` and slice lengths used by [`qif_step`].
fn validate_qif(
    state: &QifState,
    current: &[f32],
    cfg: &QifConfig,
    spikes_out: &[f32],
) -> SnnResult<()> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_m });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    if !cfg.v_peak.is_finite() {
        return Err(SnnError::BadThreshold { v_th: cfg.v_peak });
    }
    let n = state.v.len();
    if current.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: current.len(),
        });
    }
    if spikes_out.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: spikes_out.len(),
        });
    }
    Ok(())
}

/// Advance the QIF state by one timestep using a hard reset.
///
/// `current` is the input current `I_t`, length must match `state.v`.
/// `spikes_out` receives `1.0` where a spike occurred, `0.0` elsewhere.
///
/// # Errors
/// Returns `SnnError` if `tau_m`/`dt` are non-positive or non-finite, `v_peak`
/// is non-finite, or any slice length does not match `state.v`.
pub fn qif_step(
    state: &mut QifState,
    current: &[f32],
    cfg: &QifConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate_qif(state, current, cfg, spikes_out)?;
    let scale = cfg.dt / cfg.tau_m;
    for ((v, &i_in), s_out) in state
        .v
        .iter_mut()
        .zip(current.iter())
        .zip(spikes_out.iter_mut())
    {
        let drift = (*v - cfg.v_rest) * (*v - cfg.v_c) + cfg.r_in * i_in;
        let mut v_new = *v + scale * drift;
        // Guard NaN / overflow before the threshold test: NaN collapses to the
        // peak (treated as a spike), otherwise clamp into [QIF_V_FLOOR, v_peak].
        if v_new.is_nan() {
            v_new = cfg.v_peak;
        } else {
            v_new = v_new.clamp(QIF_V_FLOOR, cfg.v_peak);
        }
        let spike = if v_new >= cfg.v_peak {
            1.0_f32
        } else {
            0.0_f32
        };
        *v = if spike > 0.0 { cfg.v_reset } else { v_new };
        *s_out = spike;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Theta neuron
// ─────────────────────────────────────────────────────────────────────────────

/// Theta-neuron configuration; `tau` and `dt` must be strictly positive.
#[derive(Debug, Clone, Copy)]
pub struct ThetaConfig {
    /// Phase time constant `τ`.
    pub tau: f32,
    /// Integration time step.
    pub dt: f32,
}

impl Default for ThetaConfig {
    fn default() -> Self {
        Self { tau: 1.0, dt: 0.05 }
    }
}

/// Mutable theta-neuron state (phase per neuron).
#[derive(Debug, Clone)]
pub struct ThetaState {
    /// Phase `θ_i ∈ (−π, π]` for each neuron, length `n`.
    pub theta: Vec<f32>,
}

impl ThetaState {
    /// Allocate state for `n` neurons with `θ` initialised to zero.
    ///
    /// Zero is the rest phase for sub-threshold (negative) drive.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            theta: vec![0.0_f32; n],
        }
    }
}

/// Validate `cfg` and slice lengths used by [`theta_step`].
fn validate_theta(
    state: &ThetaState,
    current: &[f32],
    cfg: &ThetaConfig,
    spikes_out: &[f32],
) -> SnnResult<()> {
    if cfg.tau <= 0.0 || !cfg.tau.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    let n = state.theta.len();
    if current.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: current.len(),
        });
    }
    if spikes_out.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: spikes_out.len(),
        });
    }
    Ok(())
}

/// Advance the theta-neuron state by one timestep (explicit Euler).
///
/// A spike is emitted when `θ` crosses `π` from below during the step; the phase
/// is then wrapped by subtracting `2π` to keep it in `(−π, π]`. `spikes_out`
/// receives `1.0` where a spike occurred, `0.0` elsewhere.
///
/// # Errors
/// Returns `SnnError` if `tau`/`dt` are non-positive or non-finite, or any slice
/// length does not match `state.theta`.
pub fn theta_step(
    state: &mut ThetaState,
    current: &[f32],
    cfg: &ThetaConfig,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    validate_theta(state, current, cfg, spikes_out)?;
    let scale = cfg.dt / cfg.tau;
    for ((theta, &i_in), s_out) in state
        .theta
        .iter_mut()
        .zip(current.iter())
        .zip(spikes_out.iter_mut())
    {
        let cos_t = theta.cos();
        let dtheta = scale * ((1.0 - cos_t) + (1.0 + cos_t) * i_in);
        let theta_new = *theta + dtheta;
        // Spike when the phase passes through/above π from below within the step.
        let spike = if theta_new >= PI && *theta < PI {
            1.0_f32
        } else {
            0.0_f32
        };
        *theta = wrap_theta(theta_new);
        *s_out = spike;
    }
    Ok(())
}

/// Wrap a phase into the half-open interval `(−π, π]`.
fn wrap_theta(theta: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut t = theta % two_pi;
    if t > PI {
        t -= two_pi;
    } else if t <= -PI {
        t += two_pi;
    }
    t
}

/// Map a theta phase to the equivalent QIF voltage `v = tan(θ/2)`.
#[must_use]
pub fn theta_to_voltage(theta: f32) -> f32 {
    (theta * 0.5).tan()
}

/// Map a QIF voltage to the equivalent theta phase `θ = 2·atan(v)`.
#[must_use]
pub fn voltage_to_theta(v: f32) -> f32 {
    2.0 * v.atan()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── QIF ──────────────────────────────────────────────────────────────────

    #[test]
    fn qif_state_new_zeros_and_with_rest() {
        let s = QifState::new(4);
        assert_eq!(s.v.len(), 4);
        assert!(s.v.iter().all(|&v| v == 0.0));
        let r = QifState::with_rest(3, -65.0);
        assert!(r.v.iter().all(|&v| (v - (-65.0)).abs() < 1e-6));
    }

    #[test]
    fn qif_below_threshold_decays_toward_rest() {
        let cfg = QifConfig::default();
        // Start between v_rest and v_c (still below v_c → should decay to v_rest).
        let mut state = QifState::new(1);
        state.v[0] = -55.0;
        let current = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        for _ in 0..2000 {
            qif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        }
        assert!(
            (state.v[0] - cfg.v_rest).abs() < 1.0,
            "v={} should approach v_rest={}",
            state.v[0],
            cfg.v_rest
        );
    }

    #[test]
    fn qif_strong_input_produces_spikes() {
        let cfg = QifConfig::default();
        let mut state = QifState::with_rest(1, cfg.v_rest);
        let current = vec![500.0_f32; 1]; // strong drive past v_c
        let mut spikes = vec![0.0_f32; 1];
        let mut count = 0_usize;
        for _ in 0..2000 {
            qif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            count += spikes[0] as usize;
        }
        assert!(count > 0, "expected repetitive spiking, got {count}");
    }

    #[test]
    fn qif_reset_to_v_reset_after_spike() {
        let cfg = QifConfig {
            v_reset: -60.0,
            ..Default::default()
        };
        let mut state = QifState::new(1);
        state.v[0] = cfg.v_peak; // force an immediate spike
        let current = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        qif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0);
        assert!(
            (state.v[0] - cfg.v_reset).abs() < 1e-5,
            "v={} should equal v_reset={}",
            state.v[0],
            cfg.v_reset
        );
    }

    #[test]
    fn qif_rejects_bad_tau() {
        let cfg = QifConfig {
            tau_m: 0.0,
            ..Default::default()
        };
        let mut state = QifState::new(2);
        let current = vec![0.0_f32; 2];
        let mut spikes = vec![0.0_f32; 2];
        let err = qif_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn qif_rejects_bad_dt() {
        let cfg = QifConfig {
            dt: -1.0,
            ..Default::default()
        };
        let mut state = QifState::new(2);
        let current = vec![0.0_f32; 2];
        let mut spikes = vec![0.0_f32; 2];
        let err = qif_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }

    #[test]
    fn qif_rejects_length_mismatch() {
        let cfg = QifConfig::default();
        let mut state = QifState::new(2);
        let current = vec![0.0_f32; 3];
        let mut spikes = vec![0.0_f32; 2];
        let err = qif_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn qif_no_nan_with_huge_input() {
        let cfg = QifConfig::default();
        let mut state = QifState::with_rest(1, cfg.v_rest);
        let current = vec![1e12_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        for _ in 0..100 {
            qif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            assert!(
                state.v[0].is_finite(),
                "v became non-finite: {}",
                state.v[0]
            );
            assert!(state.v[0] <= cfg.v_peak + 1e-3 && state.v[0] >= QIF_V_FLOOR - 1.0);
        }
    }

    // ── Theta neuron ─────────────────────────────────────────────────────────

    #[test]
    fn theta_zero_current_rest_no_spikes() {
        let cfg = ThetaConfig::default();
        let mut state = ThetaState::new(1); // θ = 0 (rest)
        let current = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let mut count = 0_usize;
        for _ in 0..1000 {
            theta_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            count += spikes[0] as usize;
        }
        assert_eq!(count, 0, "zero current should not spike");
        assert!(state.theta[0].abs() < 1e-3, "θ should stay near rest 0");
    }

    #[test]
    fn theta_negative_current_no_spikes() {
        let cfg = ThetaConfig::default();
        let mut state = ThetaState::new(1);
        state.theta[0] = 0.5;
        let current = vec![-1.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let mut count = 0_usize;
        for _ in 0..1000 {
            theta_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            count += spikes[0] as usize;
        }
        assert_eq!(count, 0, "negative current should be silent");
    }

    #[test]
    fn theta_positive_current_repetitive_spiking() {
        let cfg = ThetaConfig::default();
        let mut state = ThetaState::new(1);
        let current = vec![1.0_f32; 1]; // above bifurcation (I > 0)
        let mut spikes = vec![0.0_f32; 1];
        let mut count = 0_usize;
        for _ in 0..2000 {
            theta_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            count += spikes[0] as usize;
        }
        assert!(
            count > 0,
            "positive current should produce spikes, got {count}"
        );
    }

    #[test]
    fn theta_spike_wraps_into_interval() {
        let cfg = ThetaConfig::default();
        let mut state = ThetaState::new(1);
        // Just below π with strong drive so the step crosses π.
        state.theta[0] = PI - 0.01;
        let current = vec![5.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        theta_step(&mut state, &current, &cfg, &mut spikes).expect("step");
        assert_eq!(spikes[0], 1.0, "should spike crossing π");
        assert!(
            state.theta[0] > -PI && state.theta[0] <= PI,
            "θ={} must be in (−π, π]",
            state.theta[0]
        );
    }

    #[test]
    fn theta_stays_wrapped_over_long_run() {
        let cfg = ThetaConfig::default();
        let mut state = ThetaState::new(1);
        let current = vec![2.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        for _ in 0..5000 {
            theta_step(&mut state, &current, &cfg, &mut spikes).expect("step");
            assert!(
                state.theta[0] > -PI - 1e-4 && state.theta[0] <= PI + 1e-4,
                "θ left (−π, π]: {}",
                state.theta[0]
            );
        }
    }

    #[test]
    fn theta_voltage_round_trip() {
        for &v in &[-3.0_f32, -0.5, 0.0, 0.75, 2.0] {
            let theta = voltage_to_theta(v);
            let back = theta_to_voltage(theta);
            assert!((back - v).abs() < 1e-4, "round-trip v={v} -> {back}");
        }
        // And the other direction within (−π, π).
        for &t in &[-2.0_f32, -0.3, 0.0, 1.1, 2.5] {
            let v = theta_to_voltage(t);
            let back = voltage_to_theta(v);
            assert!((back - t).abs() < 1e-4, "round-trip θ={t} -> {back}");
        }
    }

    #[test]
    fn theta_rejects_bad_tau() {
        let cfg = ThetaConfig { tau: 0.0, dt: 0.05 };
        let mut state = ThetaState::new(1);
        let current = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let err = theta_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadTau { .. })));
    }

    #[test]
    fn theta_rejects_length_mismatch() {
        let cfg = ThetaConfig::default();
        let mut state = ThetaState::new(2);
        let current = vec![0.0_f32; 1];
        let mut spikes = vec![0.0_f32; 2];
        let err = theta_step(&mut state, &current, &cfg, &mut spikes);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }
}
