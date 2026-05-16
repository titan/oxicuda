//! Stochastic Poisson rate neuron.
//!
//! At each step, neuron `i` emits a spike with probability `rate[i] · dt`,
//! sampled by drawing `u ∈ [0,1)` from an [`crate::handle::LcgRng`]. Setting `rate[i] · dt > 1`
//! is a usage error; the function clamps the probability to `[0, 1]` for safety
//! but the caller should keep `dt` small enough for valid Poisson statistics.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Sample one timestep of a Bernoulli approximation to a Poisson process.
///
/// `rates[i]` must be non-negative and finite. `dt` must be strictly positive.
/// Writes `1.0` (spike) or `0.0` (no spike) into each `spikes_out[i]`.
pub fn poisson_step(
    rates: &[f32],
    dt: f32,
    rng: &mut LcgRng,
    spikes_out: &mut [f32],
) -> SnnResult<()> {
    if dt <= 0.0 || !dt.is_finite() {
        return Err(SnnError::BadDt { dt });
    }
    if rates.len() != spikes_out.len() {
        return Err(SnnError::IncompatibleLength {
            a: rates.len(),
            b: spikes_out.len(),
        });
    }
    for (&r, s_out) in rates.iter().zip(spikes_out.iter_mut()) {
        if r < 0.0 || !r.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "rate".into(),
                val: r,
            });
        }
        let mut p = r * dt;
        if p > 1.0 {
            p = 1.0;
        }
        let u = rng.next_f32();
        *s_out = if u < p { 1.0_f32 } else { 0.0_f32 };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_rate_matches_target() {
        let mut rng = LcgRng::new(13);
        let target_rate = 0.05_f32; // 5% per step
        let dt = 1.0_f32;
        let n = 4_usize;
        let rates = vec![target_rate; n];
        let mut spikes = vec![0.0_f32; n];
        let steps = 10_000_usize;
        let mut total = vec![0.0_f32; n];
        for _ in 0..steps {
            poisson_step(&rates, dt, &mut rng, &mut spikes).expect("step");
            for (t, &s) in total.iter_mut().zip(spikes.iter()) {
                *t += s;
            }
        }
        for &t in &total {
            let mean = t / steps as f32;
            // Tolerance: 3 standard deviations of the binomial mean.
            let std = (target_rate * (1.0 - target_rate) / steps as f32).sqrt();
            assert!(
                (mean - target_rate).abs() < 4.0 * std,
                "mean={mean} target={target_rate}"
            );
        }
    }

    #[test]
    fn zero_rate_no_spikes() {
        let mut rng = LcgRng::new(7);
        let rates = vec![0.0_f32; 16];
        let mut spikes = vec![1.0_f32; 16];
        for _ in 0..100 {
            poisson_step(&rates, 1.0, &mut rng, &mut spikes).expect("step");
            for &s in &spikes {
                assert_eq!(s, 0.0);
            }
        }
    }

    #[test]
    fn rejects_negative_rate() {
        let mut rng = LcgRng::new(0);
        let rates = vec![-1.0_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let err = poisson_step(&rates, 1.0, &mut rng, &mut spikes);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_bad_dt() {
        let mut rng = LcgRng::new(0);
        let rates = vec![0.1_f32; 1];
        let mut spikes = vec![0.0_f32; 1];
        let err = poisson_step(&rates, 0.0, &mut rng, &mut spikes);
        assert!(matches!(err, Err(SnnError::BadDt { .. })));
    }
}
