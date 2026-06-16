//! Temporal-contrast (event-camera) spike encoding.
//!
//! Reference: Brandli, Berner, Yang, Liu & Delbruck — *"A 240×180 130 dB 3 µs
//! Latency Global Shutter Spatiotemporal Vision Sensor"* (IEEE JSSC, 2014), the
//! DAVIS dynamic-vision-sensor pixel model; and Lichtsteiner, Posch & Delbruck
//! (2008) for the original DVS temporal-contrast pixel.
//!
//! Each input channel is encoded asynchronously by tracking a *memorised
//! reference level* and emitting a signed event whenever the (optionally
//! log-transformed) signal departs from that reference by more than a fixed
//! contrast threshold `θ`:
//!
//! ```text
//! f(x)   = ln(x + ε)            if log_domain else x         # pixel transform
//! Δ      = f(x_t) − ref
//! if Δ ≥ +θ:   emit ON  (+1), ref ← ref + θ                 # brightness up
//! if Δ ≤ −θ:   emit OFF (−1), ref ← ref − θ                 # brightness down
//! else:        no event (0)                                  # inside dead-band
//! ```
//!
//! The `±θ` dead-band gives the threshold-crossing **hysteresis** that
//! suppresses noise events: small fluctuations within `(ref−θ, ref+θ)` produce
//! no spikes. Because the reference advances by exactly `θ` per event, a large
//! step in the signal generates a *burst* of same-sign events — one per `θ` of
//! contrast crossed — mirroring the integrate-and-fire behaviour of the real
//! sensor pixel.
//!
//! Output layout matches the other encoders: a flat `(t_steps × n)`
//! **time-major** buffer, with `+1.0` for an ON event, `−1.0` for an OFF event,
//! and `0.0` for no event at `out[t*n + i]`.

use crate::error::{SnnError, SnnResult};

/// Configuration for the temporal-contrast encoder.
#[derive(Debug, Clone, Copy)]
pub struct TemporalContrastConfig {
    /// Contrast threshold `θ` (half-width of the dead-band); must be `> 0`.
    pub theta: f32,
    /// Encode in the logarithmic (contrast) domain `ln(x + ε)` when `true`,
    /// matching the real sensor's photoreceptor; otherwise the linear signal.
    pub log_domain: bool,
    /// Small offset `ε` keeping `ln(x + ε)` finite for non-positive inputs.
    pub epsilon: f32,
    /// Cap on the number of same-sign events emitted from a single large step
    /// (prevents an unbounded burst for a huge jump); must be `≥ 1`.
    pub max_events_per_step: u32,
}

impl Default for TemporalContrastConfig {
    /// DVS-like defaults: `θ = 0.15` log-contrast, log domain on, `ε = 1e-3`,
    /// burst capped at 64 events per step.
    fn default() -> Self {
        Self {
            theta: 0.15,
            log_domain: true,
            epsilon: 1e-3,
            max_events_per_step: 64,
        }
    }
}

/// Per-channel encoder state: the memorised reference level.
#[derive(Debug, Clone)]
pub struct TemporalContrastState {
    /// Reference (memorised) transformed level `ref_i` per channel.
    pub reference: Vec<f32>,
    /// `true` once the reference has been initialised from the first sample.
    initialised: bool,
}

impl TemporalContrastState {
    /// Allocate state for `n` channels with an uninitialised reference; the
    /// reference is set from the first sample seen by [`temporal_contrast_step`].
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            reference: vec![0.0_f32; n],
            initialised: false,
        }
    }

    /// `true` if the reference has been seeded from a first sample.
    #[must_use]
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }
}

fn validate_cfg(cfg: &TemporalContrastConfig) -> SnnResult<()> {
    if cfg.theta <= 0.0 || !cfg.theta.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "theta".into(),
            val: cfg.theta,
        });
    }
    if cfg.max_events_per_step == 0 {
        return Err(SnnError::OutOfRange {
            name: "max_events_per_step".into(),
            val: cfg.max_events_per_step as f32,
        });
    }
    if cfg.log_domain && (!cfg.epsilon.is_finite() || cfg.epsilon <= 0.0) {
        return Err(SnnError::OutOfRange {
            name: "epsilon".into(),
            val: cfg.epsilon,
        });
    }
    Ok(())
}

#[inline]
fn transform(x: f32, cfg: &TemporalContrastConfig) -> f32 {
    if cfg.log_domain {
        (x + cfg.epsilon).max(cfg.epsilon).ln()
    } else {
        x
    }
}

/// Encode one frame `values` (length `n`) into a signed-event row `events_out`.
///
/// `events_out` receives `+1.0` (ON), `−1.0` (OFF), or `0.0` (no event) per
/// channel. On the very first call the reference is seeded from `values` and no
/// events are produced. Subsequent calls emit the **net** signed event for each
/// channel — its sign indicates ON/OFF and is non-zero whenever at least one
/// threshold was crossed; the reference advances by `θ` per crossing (capped by
/// `max_events_per_step`).
///
/// # Errors
/// [`SnnError::EmptyInput`], [`SnnError::OutOfRange`] for invalid config or
/// non-finite inputs, [`SnnError::IncompatibleLength`] for length mismatches.
pub fn temporal_contrast_step(
    state: &mut TemporalContrastState,
    values: &[f32],
    cfg: &TemporalContrastConfig,
    events_out: &mut [f32],
) -> SnnResult<()> {
    validate_cfg(cfg)?;
    if values.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    let n = state.reference.len();
    if values.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: values.len(),
        });
    }
    if events_out.len() != n {
        return Err(SnnError::IncompatibleLength {
            a: n,
            b: events_out.len(),
        });
    }
    for &v in values {
        if !v.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "value".into(),
                val: v,
            });
        }
    }

    if !state.initialised {
        for (r, &v) in state.reference.iter_mut().zip(values.iter()) {
            *r = transform(v, cfg);
        }
        state.initialised = true;
        for e in events_out.iter_mut() {
            *e = 0.0;
        }
        return Ok(());
    }

    for ((r, &v), e) in state
        .reference
        .iter_mut()
        .zip(values.iter())
        .zip(events_out.iter_mut())
    {
        let f = transform(v, cfg);
        let mut net = 0i64;
        let mut count = 0u32;
        // Integrate-and-fire across the dead-band: emit one event per θ crossed.
        loop {
            let delta = f - *r;
            if delta >= cfg.theta && count < cfg.max_events_per_step {
                *r += cfg.theta;
                net += 1;
                count += 1;
            } else if delta <= -cfg.theta && count < cfg.max_events_per_step {
                *r -= cfg.theta;
                net -= 1;
                count += 1;
            } else {
                break;
            }
        }
        *e = match net.signum() {
            1 => 1.0,
            -1 => -1.0,
            _ => 0.0,
        };
    }
    Ok(())
}

/// Encode a full `(t_steps × n)` time-major signal into a `(t_steps × n)`
/// signed-event train.
///
/// Equivalent to repeatedly calling [`temporal_contrast_step`] on each frame
/// `signal[t*n .. t*n + n]`. The first frame seeds the reference and yields an
/// all-zero event row.
///
/// # Errors
/// [`SnnError::EmptyInput`], [`SnnError::BadDim`], [`SnnError::BadTimesteps`],
/// [`SnnError::BadShape`], [`SnnError::OutOfRange`] for invalid inputs.
pub fn temporal_contrast_encode(
    signal: &[f32],
    t_steps: usize,
    n: usize,
    cfg: &TemporalContrastConfig,
    out: &mut [f32],
) -> SnnResult<()> {
    validate_cfg(cfg)?;
    if n == 0 {
        return Err(SnnError::BadDim { got: n });
    }
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    if signal.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: signal.len(),
        });
    }
    if out.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: out.len(),
        });
    }
    let mut state = TemporalContrastState::new(n);
    for t in 0..t_steps {
        let frame = &signal[t * n..(t + 1) * n];
        let row = &mut out[t * n..(t + 1) * n];
        temporal_contrast_step(&mut state, frame, cfg, row)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin_cfg(theta: f32) -> TemporalContrastConfig {
        TemporalContrastConfig {
            theta,
            log_domain: false,
            epsilon: 1e-3,
            max_events_per_step: 64,
        }
    }

    #[test]
    fn rejects_non_positive_theta() {
        let cfg = lin_cfg(0.0);
        let mut state = TemporalContrastState::new(2);
        let mut out = vec![0.0_f32; 2];
        assert!(matches!(
            temporal_contrast_step(&mut state, &[0.0, 0.0], &cfg, &mut out),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_zero_max_events() {
        let cfg = TemporalContrastConfig {
            max_events_per_step: 0,
            ..lin_cfg(0.1)
        };
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        assert!(matches!(
            temporal_contrast_step(&mut state, &[0.0], &cfg, &mut out),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_empty_input() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(0);
        let mut out: Vec<f32> = Vec::new();
        assert!(matches!(
            temporal_contrast_step(&mut state, &[], &cfg, &mut out),
            Err(SnnError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_non_finite_value() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(1);
        // Seed first.
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.0], &cfg, &mut out).expect("seed");
        assert!(matches!(
            temporal_contrast_step(&mut state, &[f32::NAN], &cfg, &mut out),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_length_mismatch() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(2);
        let mut out = vec![0.0_f32; 2];
        assert!(matches!(
            temporal_contrast_step(&mut state, &[0.0, 0.0, 0.0], &cfg, &mut out),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn first_frame_seeds_and_emits_no_events() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(3);
        assert!(!state.is_initialised());
        let mut out = vec![9.0_f32; 3];
        temporal_contrast_step(&mut state, &[0.2, 0.5, 0.9], &cfg, &mut out).expect("seed");
        assert!(state.is_initialised());
        for &e in &out {
            assert_eq!(e, 0.0, "first frame must emit no events");
        }
        // Reference equals the (linear) first sample.
        assert!((state.reference[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rising_signal_emits_on_event() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.0], &cfg, &mut out).expect("seed");
        // +0.15 > θ ⇒ ON event.
        temporal_contrast_step(&mut state, &[0.15], &cfg, &mut out).expect("step");
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn falling_signal_emits_off_event() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.5], &cfg, &mut out).expect("seed");
        temporal_contrast_step(&mut state, &[0.3], &cfg, &mut out).expect("step");
        assert_eq!(out[0], -1.0);
    }

    #[test]
    fn small_fluctuation_inside_deadband_no_event() {
        // Hysteresis: changes smaller than θ produce no events.
        let cfg = lin_cfg(0.2);
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.5], &cfg, &mut out).expect("seed");
        for &v in &[0.55_f32, 0.45, 0.6, 0.4, 0.5] {
            temporal_contrast_step(&mut state, &[v], &cfg, &mut out).expect("step");
            assert_eq!(out[0], 0.0, "value {v} inside dead-band must not spike");
        }
    }

    #[test]
    fn reference_advances_by_theta_per_event() {
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.0], &cfg, &mut out).expect("seed");
        // Single crossing: ref goes 0.0 → 0.1.
        temporal_contrast_step(&mut state, &[0.12], &cfg, &mut out).expect("step");
        assert_eq!(out[0], 1.0);
        assert!(
            (state.reference[0] - 0.1).abs() < 1e-6,
            "ref={}",
            state.reference[0]
        );
    }

    #[test]
    fn large_step_bursts_multiple_crossings() {
        // A jump of 0.95 with θ = 0.1 should advance the reference by many θ's
        // (≈ 9 crossings) so the reference ends close to the new value.
        let cfg = lin_cfg(0.1);
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.0], &cfg, &mut out).expect("seed");
        temporal_contrast_step(&mut state, &[0.95], &cfg, &mut out).expect("step");
        assert_eq!(out[0], 1.0, "net event sign should be ON");
        // Reference should have caught up to within one θ of the signal.
        assert!(
            (0.95 - state.reference[0]).abs() <= cfg.theta + 1e-6,
            "ref={} did not catch up",
            state.reference[0]
        );
    }

    #[test]
    fn burst_respects_max_events_cap() {
        // With a tiny θ and a small cap, the reference advances by at most
        // cap·θ even for an enormous jump.
        let cfg = TemporalContrastConfig {
            theta: 0.01,
            log_domain: false,
            epsilon: 1e-3,
            max_events_per_step: 3,
        };
        let mut state = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut state, &[0.0], &cfg, &mut out).expect("seed");
        temporal_contrast_step(&mut state, &[100.0], &cfg, &mut out).expect("step");
        assert_eq!(out[0], 1.0);
        // ref advanced by exactly cap·θ = 0.03.
        assert!(
            (state.reference[0] - 0.03).abs() < 1e-6,
            "ref={} should be cap·θ",
            state.reference[0]
        );
    }

    #[test]
    fn log_domain_detects_relative_contrast() {
        // In log domain a fixed *ratio* crosses threshold regardless of absolute
        // level: doubling 0.1→0.2 and 1.0→2.0 both equal ln(2) of contrast.
        let cfg = TemporalContrastConfig {
            theta: 0.5,
            log_domain: true,
            epsilon: 1e-6,
            max_events_per_step: 64,
        };
        let mut s_low = TemporalContrastState::new(1);
        let mut s_high = TemporalContrastState::new(1);
        let mut out = vec![0.0_f32; 1];
        temporal_contrast_step(&mut s_low, &[0.1], &cfg, &mut out).expect("seed");
        temporal_contrast_step(&mut s_low, &[0.2], &cfg, &mut out).expect("step");
        let low_event = out[0];
        temporal_contrast_step(&mut s_high, &[1.0], &cfg, &mut out).expect("seed");
        temporal_contrast_step(&mut s_high, &[2.0], &cfg, &mut out).expect("step");
        let high_event = out[0];
        // ln(2) ≈ 0.693 > θ=0.5 ⇒ both emit an ON event.
        assert_eq!(low_event, 1.0);
        assert_eq!(high_event, 1.0);
    }

    #[test]
    fn full_encode_shape_and_first_row_zero() {
        let cfg = lin_cfg(0.1);
        let t = 5usize;
        let n = 2usize;
        // Channel 0 ramps up, channel 1 stays constant.
        let mut signal = vec![0.0_f32; t * n];
        for tt in 0..t {
            signal[tt * n] = 0.2 * tt as f32; // 0, .2, .4, .6, .8
            signal[tt * n + 1] = 0.5; // constant
        }
        let mut out = vec![0.0_f32; t * n];
        temporal_contrast_encode(&signal, t, n, &cfg, &mut out).expect("encode");
        // First row seeds ⇒ all zero.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        // Ramping channel 0 emits ON events on later frames.
        let ch0_events: usize = (1..t).filter(|&tt| out[tt * n] == 1.0).count();
        assert!(ch0_events >= 1, "ramp should produce ON events");
        // Constant channel 1 never fires after seeding.
        for tt in 1..t {
            assert_eq!(out[tt * n + 1], 0.0, "constant channel must stay silent");
        }
    }

    #[test]
    fn encode_rejects_bad_shapes() {
        let cfg = lin_cfg(0.1);
        let signal = vec![0.0_f32; 6];
        let mut out = vec![0.0_f32; 6];
        assert!(matches!(
            temporal_contrast_encode(&signal, 0, 2, &cfg, &mut out),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            temporal_contrast_encode(&signal, 3, 0, &cfg, &mut out),
            Err(SnnError::BadDim { .. })
        ));
        let mut bad_out = vec![0.0_f32; 4];
        assert!(matches!(
            temporal_contrast_encode(&signal, 3, 2, &cfg, &mut bad_out),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn all_outputs_are_finite_signed_units() {
        let cfg = TemporalContrastConfig::default();
        let t = 20usize;
        let n = 3usize;
        let mut signal = vec![0.0_f32; t * n];
        for tt in 0..t {
            for i in 0..n {
                // Mixed oscillatory + ramp signal, kept positive for log domain.
                signal[tt * n + i] =
                    0.5 + 0.4 * ((tt as f32 * 0.3 + i as f32).sin()) + 0.01 * tt as f32;
            }
        }
        let mut out = vec![0.0_f32; t * n];
        temporal_contrast_encode(&signal, t, n, &cfg, &mut out).expect("encode");
        for &e in &out {
            assert!(e.is_finite(), "event not finite: {e}");
            assert!(
                e == -1.0 || e == 0.0 || e == 1.0,
                "event not in {{-1,0,1}}: {e}"
            );
        }
    }
}
