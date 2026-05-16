//! Phase encoding via an oscillatory reference signal.
//!
//! Each input value `x_i ∈ [0, 1]` controls the phase offset of a sinusoid of
//! angular frequency `ω`. A spike is emitted whenever the phase
//! `φ_i(t) = ω · t + 2π · x_i` crosses an integer multiple of `2π`.
//!
//! Concretely, we monitor `s_i(t) = cos(φ_i(t))` and detect rising
//! zero-crossings: a spike fires at the first step `t` such that
//! `s_i(t) ≥ 0` and `s_i(t − 1) < 0`. This emits at most one spike per
//! oscillation cycle, regardless of the time step.
//!
//! The expected firing rate is therefore `ω / (2π)` and is independent of the
//! offset, while the precise inter-spike timing is shifted by an amount
//! proportional to `x_i` — the information channel exploited by phase coding.

use crate::error::{SnnError, SnnResult};

/// Encode `values` into a phase-coded spike train of length `t_steps` using
/// reference angular frequency `omega`.
///
/// `omega` must be strictly positive; values are wrapped/aliased to `[0, 1]`
/// for safety but callers should keep them in range to preserve the encoding
/// semantics. The output buffer is row-major and must have length
/// `t_steps * n`.
///
/// Errors: [`SnnError::EmptyInput`], [`SnnError::BadTimesteps`],
/// [`SnnError::BadShape`], and [`SnnError::OutOfRange`] for non-finite or
/// non-positive `omega`.
pub fn phase_encode(values: &[f32], t_steps: usize, omega: f32, out: &mut [f32]) -> SnnResult<()> {
    if values.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    if !omega.is_finite() || omega <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "omega".into(),
            val: omega,
        });
    }
    let n = values.len();
    if out.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: out.len(),
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
    for slot in out.iter_mut() {
        *slot = 0.0_f32;
    }
    let two_pi = 2.0 * std::f32::consts::PI;
    // Precompute initial reference values at t = -1 so that t = 0 can be a
    // valid spike step if the phase rises through zero immediately.
    let mut prev_cos = Vec::with_capacity(n);
    for &x in values {
        let phi = -omega + two_pi * x;
        prev_cos.push(phi.cos());
    }
    for t in 0..t_steps {
        let row = &mut out[t * n..(t + 1) * n];
        let t_f = t as f32;
        for ((slot, &x), prev) in row.iter_mut().zip(values.iter()).zip(prev_cos.iter_mut()) {
            let phi = omega * t_f + two_pi * x;
            let cur = phi.cos();
            // Rising zero-crossing: cur ≥ 0 and prev < 0.
            *slot = if *prev < 0.0 && cur >= 0.0 {
                1.0_f32
            } else {
                0.0_f32
            };
            *prev = cur;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_spikes(train: &[f32], t_steps: usize, n: usize, i: usize) -> usize {
        let mut c = 0;
        for t in 0..t_steps {
            if train[t * n + i] == 1.0 {
                c += 1;
            }
        }
        c
    }

    #[test]
    fn rate_proportional_to_omega() {
        // ω / (2π) cycles per step — the expected number of spikes in T steps.
        let omega = 0.6_f32; // ~9.55% per step
        let t_steps = 5000_usize;
        let values = vec![0.2_f32, 0.5, 0.8];
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        phase_encode(&values, t_steps, omega, &mut out).expect("encode");
        let expected_total = (t_steps as f32) * omega / (2.0 * std::f32::consts::PI);
        for i in 0..n {
            let c = count_spikes(&out, t_steps, n, i) as f32;
            assert!(
                (c - expected_total).abs() < 5.0,
                "neuron {i}: spikes={c} expected≈{expected_total}"
            );
        }
    }

    #[test]
    fn shape_correct() {
        let values = vec![0.1_f32; 2];
        let t_steps = 32_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        phase_encode(&values, t_steps, 0.5, &mut out).expect("encode");
        assert_eq!(out.len(), t_steps * n);
        // Output must contain only 0/1.
        for &v in &out {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn rejects_invalid_arguments() {
        let mut buf = vec![0.0_f32; 8];
        assert!(matches!(
            phase_encode(&[], 8, 0.5, &mut buf),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            phase_encode(&[0.5_f32; 2], 0, 0.5, &mut buf),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            phase_encode(&[0.5_f32; 2], 8, 0.0, &mut buf),
            Err(SnnError::OutOfRange { .. })
        ));
        let mut wrong = vec![0.0_f32; 4];
        assert!(matches!(
            phase_encode(&[0.5_f32; 2], 8, 0.5, &mut wrong),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn distinct_phase_offsets_yield_distinct_spike_times() {
        let values = vec![0.0_f32, 0.5];
        let omega = 0.4_f32;
        let t_steps = 64_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        phase_encode(&values, t_steps, omega, &mut out).expect("encode");
        let mut times_a: Vec<usize> = Vec::new();
        let mut times_b: Vec<usize> = Vec::new();
        for t in 0..t_steps {
            if out[t * n] == 1.0 {
                times_a.push(t);
            }
            if out[t * n + 1] == 1.0 {
                times_b.push(t);
            }
        }
        assert!(!times_a.is_empty());
        assert!(!times_b.is_empty());
        // First spikes should differ when offsets differ.
        assert_ne!(times_a[0], times_b[0]);
    }
}
