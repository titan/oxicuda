//! Poisson input encoding wrapper.
//!
//! Repeats [`crate::neuron::poisson::poisson_step`] across `t_steps`, treating
//! each `values[i]` as the firing rate (Hz · dt-units) of input neuron `i`.
//! Compared to [`crate::encoding::rate::rate_encode`] the per-step probability is
//! `rate · dt` rather than `rate` itself, allowing the user to tune the time
//! step independently of the input scale.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::poisson::poisson_step;

/// Generate a Poisson-coded spike train of length `t_steps` using `rates`
/// as per-neuron firing rates and time step `dt`.
///
/// The output buffer is row-major with length `t_steps * values.len()`.
///
/// Errors propagated from [`poisson_step`], plus [`SnnError::EmptyInput`],
/// [`SnnError::BadTimesteps`], [`SnnError::BadShape`].
pub fn poisson_input_encode(
    values: &[f32],
    t_steps: usize,
    dt: f32,
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
    for t in 0..t_steps {
        let row = &mut out[t * n..(t + 1) * n];
        poisson_step(values, dt, rng, row)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_rate_matches_target() {
        let mut rng = LcgRng::new(101);
        let rates = vec![0.05_f32, 0.10, 0.20];
        let t_steps = 10_000_usize;
        let n = rates.len();
        let mut out = vec![0.0_f32; t_steps * n];
        poisson_input_encode(&rates, t_steps, 1.0, &mut rng, &mut out).expect("encode");
        let mut totals = vec![0.0_f32; n];
        for t in 0..t_steps {
            for i in 0..n {
                totals[i] += out[t * n + i];
            }
        }
        for (&r, &count) in rates.iter().zip(totals.iter()) {
            let mean = count / t_steps as f32;
            let std = (r * (1.0 - r) / t_steps as f32).sqrt().max(1e-6);
            assert!(
                (mean - r).abs() < 5.0 * std,
                "rate={r} mean={mean} std={std}"
            );
        }
    }

    #[test]
    fn shape_and_validation() {
        let mut rng = LcgRng::new(0);
        let mut buf = vec![0.0_f32; 12];
        assert!(matches!(
            poisson_input_encode(&[], 4, 1.0, &mut rng, &mut buf),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            poisson_input_encode(&[0.1_f32; 3], 0, 1.0, &mut rng, &mut buf),
            Err(SnnError::BadTimesteps { .. })
        ));
        let mut bad = vec![0.0_f32; 5];
        assert!(matches!(
            poisson_input_encode(&[0.1_f32; 3], 4, 1.0, &mut rng, &mut bad),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn output_is_binary() {
        let mut rng = LcgRng::new(11);
        let rates = vec![0.5_f32; 4];
        let t_steps = 32_usize;
        let n = rates.len();
        let mut out = vec![0.0_f32; t_steps * n];
        poisson_input_encode(&rates, t_steps, 1.0, &mut rng, &mut out).expect("encode");
        for &v in &out {
            assert!(v == 0.0 || v == 1.0);
        }
    }
}
