//! Bernoulli rate coding.
//!
//! Each input value `x_i ∈ [0, 1]` is interpreted as a per-step spike
//! probability: at every step `t`, neuron `i` emits a spike with probability
//! `x_i`. The expected firing rate is therefore `x_i / dt` (with `dt = 1`
//! implicit), and the empirical rate over `T` steps converges to `x_i` as
//! `T → ∞`.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Encode `values ∈ [0, 1]^n` into a Bernoulli rate-coded spike train of
/// length `t_steps`.
///
/// The output buffer must have length `t_steps * n` and is filled row-major.
///
/// Errors: [`SnnError::EmptyInput`] for empty inputs,
/// [`SnnError::BadTimesteps`] for `t_steps == 0`,
/// [`SnnError::BadShape`] when the output buffer length is wrong, and
/// [`SnnError::OutOfRange`] for values outside `[0, 1]`.
pub fn rate_encode(
    values: &[f32],
    t_steps: usize,
    rng: &mut LcgRng,
    out: &mut [f32],
) -> SnnResult<()> {
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
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(SnnError::OutOfRange {
                name: "value".into(),
                val: v,
            });
        }
    }
    for t in 0..t_steps {
        let row = &mut out[t * n..(t + 1) * n];
        for (slot, &p) in row.iter_mut().zip(values.iter()) {
            let u = rng.next_f32();
            *slot = if u < p { 1.0_f32 } else { 0.0_f32 };
        }
    }
    Ok(())
}

/// Average a rate-coded spike train across the time axis to recover an
/// estimate of the encoded probabilities.
///
/// Errors: [`SnnError::BadTimesteps`] / [`SnnError::BadDim`] for zero-sized
/// axes and [`SnnError::BadShape`] for buffer-length mismatches.
pub fn rate_decode(spike_train: &[f32], t_steps: usize, n: usize) -> SnnResult<Vec<f32>> {
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    if n == 0 {
        return Err(SnnError::BadDim { got: n });
    }
    if spike_train.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: spike_train.len(),
        });
    }
    let mut out = vec![0.0_f32; n];
    for t in 0..t_steps {
        let row = &spike_train[t * n..(t + 1) * n];
        for (acc, &s) in out.iter_mut().zip(row.iter()) {
            *acc += s;
        }
    }
    let inv = 1.0 / (t_steps as f32);
    for v in out.iter_mut() {
        *v *= inv;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_rate_matches_value() {
        let mut rng = LcgRng::new(7);
        let values = vec![0.1_f32, 0.5, 0.9];
        let t_steps = 10_000_usize;
        let n = values.len();
        let mut out = vec![0.0_f32; t_steps * n];
        rate_encode(&values, t_steps, &mut rng, &mut out).expect("encode");
        let estimate = rate_decode(&out, t_steps, n).expect("decode");
        for (&v, &e) in values.iter().zip(estimate.iter()) {
            // Binomial std: sqrt(p(1-p)/T) ≤ 0.005 here; allow ~5σ slack.
            let std = (v * (1.0 - v) / t_steps as f32).sqrt().max(1e-6);
            assert!((v - e).abs() < 5.0 * std, "v={v} estimate={e} std={std}");
        }
    }

    #[test]
    fn shape_and_range_checks() {
        let mut rng = LcgRng::new(1);
        let values = vec![0.3_f32; 4];
        let mut out = vec![0.0_f32; 4 * 5];
        rate_encode(&values, 5, &mut rng, &mut out).expect("encode");
        for &v in &out {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn rejects_invalid_arguments() {
        let mut rng = LcgRng::new(1);
        let mut buf = vec![0.0_f32; 4];
        assert!(matches!(
            rate_encode(&[], 4, &mut rng, &mut buf),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            rate_encode(&[0.5_f32], 0, &mut rng, &mut buf),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            rate_encode(&[1.5_f32], 4, &mut rng, &mut buf),
            Err(SnnError::OutOfRange { .. })
        ));
        let mut wrong = vec![0.0_f32; 3];
        assert!(matches!(
            rate_encode(&[0.5_f32, 0.5], 4, &mut rng, &mut wrong),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn decode_round_trip_zero_value() {
        let mut rng = LcgRng::new(1);
        let values = vec![0.0_f32; 3];
        let t_steps = 32;
        let mut out = vec![0.0_f32; t_steps * values.len()];
        rate_encode(&values, t_steps, &mut rng, &mut out).expect("encode");
        let est = rate_decode(&out, t_steps, values.len()).expect("decode");
        for &v in &est {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }
}
