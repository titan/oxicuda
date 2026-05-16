#![allow(clippy::needless_range_loop)]
//! Time-To-First-Spike (TTFS) latency coding.
//!
//! High-amplitude inputs spike early; low-amplitude inputs spike late. The
//! encoding is deterministic, sparse (exactly one spike per neuron) and
//! latency-monotone.
//!
//! ```text
//! t_spike(i) = floor((1 − clamp(x_i, 0, 1)) · (T − 1)).
//! ```

use crate::error::{SnnError, SnnResult};

/// Encode `values` into a one-spike-per-neuron TTFS train of length
/// `t_steps`. Values outside `[0, 1]` are clamped before computing the
/// latency.
///
/// Errors: [`SnnError::EmptyInput`], [`SnnError::BadTimesteps`],
/// [`SnnError::BadShape`] for length mismatches and
/// [`SnnError::OutOfRange`] for non-finite values.
pub fn ttfs_encode(values: &[f32], t_steps: usize, out: &mut [f32]) -> SnnResult<()> {
    if values.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
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
    let scale = (t_steps as f32 - 1.0).max(0.0);
    for (i, &v) in values.iter().enumerate() {
        let clamped = v.clamp(0.0, 1.0);
        let t_spike_f = ((1.0 - clamped) * scale).floor();
        let mut t_spike = t_spike_f as usize;
        if t_spike >= t_steps {
            t_spike = t_steps - 1;
        }
        out[t_spike * n + i] = 1.0_f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_input_spikes_at_t_zero() {
        let values = vec![1.0_f32; 4];
        let t_steps = 8_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        ttfs_encode(&values, t_steps, &mut out).expect("encode");
        for i in 0..n {
            assert!((out[i] - 1.0).abs() < 1e-6, "i={i}");
        }
        // No spike anywhere else.
        for t in 1..t_steps {
            for i in 0..n {
                assert!(out[t * n + i] == 0.0);
            }
        }
    }

    #[test]
    fn zero_input_spikes_at_last_step() {
        let values = vec![0.0_f32; 3];
        let t_steps = 5_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        ttfs_encode(&values, t_steps, &mut out).expect("encode");
        for i in 0..n {
            assert!((out[(t_steps - 1) * n + i] - 1.0).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn exactly_one_spike_per_neuron() {
        let values = vec![0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let t_steps = 16_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        ttfs_encode(&values, t_steps, &mut out).expect("encode");
        for i in 0..n {
            let mut count = 0_usize;
            for t in 0..t_steps {
                if out[t * n + i] == 1.0 {
                    count += 1;
                }
            }
            assert_eq!(count, 1, "neuron {i} fired {count} times");
        }
    }

    #[test]
    fn higher_value_fires_earlier() {
        let values = vec![0.2_f32, 0.8];
        let t_steps = 16_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        ttfs_encode(&values, t_steps, &mut out).expect("encode");
        let find = |i: usize| -> usize {
            for t in 0..t_steps {
                if out[t * n + i] == 1.0 {
                    return t;
                }
            }
            t_steps
        };
        assert!(find(1) < find(0), "higher value did not fire earlier");
    }

    #[test]
    fn clamps_out_of_range_values() {
        let values = vec![-0.5_f32, 1.5];
        let t_steps = 4_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        ttfs_encode(&values, t_steps, &mut out).expect("encode");
        // -0.5 → clamped 0 → t = T-1; 1.5 → clamped 1 → t = 0.
        assert_eq!(out[(t_steps - 1) * n], 1.0);
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut buf = vec![0.0_f32; 8];
        assert!(matches!(
            ttfs_encode(&[], 4, &mut buf),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            ttfs_encode(&[0.5_f32; 2], 0, &mut buf),
            Err(SnnError::BadTimesteps { .. })
        ));
        let mut bad = vec![0.0_f32; 3];
        assert!(matches!(
            ttfs_encode(&[0.5_f32; 2], 4, &mut bad),
            Err(SnnError::BadShape { .. })
        ));
        let mut buf_one = vec![0.0_f32; 4];
        assert!(matches!(
            ttfs_encode(&[f32::NAN], 4, &mut buf_one),
            Err(SnnError::OutOfRange { .. })
        ));
    }
}
