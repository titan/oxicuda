//! Population-coded output readout: rate decoding and winner-take-all.
//!
//! Output-layer spiking networks typically represent a class label or analogue
//! target by the *firing rate* of a population of output neurons over the
//! simulation window. This module provides the standard readouts:
//!
//! * [`crate::metrics::population_coding::rate_decode`] — per-neuron firing rate (spike count divided by the
//!   number of timesteps), the canonical "rate code" readout.
//! * [`winner_take_all`] — the index of the most-active output neuron, i.e. the
//!   `argmax` over spike counts (ties resolved to the lowest index).
//! * [`softmax_decode`] — a numerically-stable softmax over spike counts,
//!   yielding a categorical probability distribution over output neurons.
//! * [`population_mean_decode`] — a value-weighted population average, mapping a
//!   bank of neurons (each with a preferred scalar value) and their firing
//!   rates to a single decoded analogue estimate (Pouget, Dayan & Zemel 2000).
//!
//! Spike-train layout: a flat `(t_steps × n)` **time-major** buffer where the
//! value at `spikes[t*n + i]` is the spike of output neuron `i` at timestep
//! `t`, matching the encoding modules' convention. Any strictly-positive value
//! counts as a spike.

use crate::error::{SnnError, SnnResult};

/// Validate a time-major spike buffer of shape `(t_steps × n)`.
fn validate_spikes(spikes: &[f32], t_steps: usize, n: usize) -> SnnResult<()> {
    if n == 0 {
        return Err(SnnError::BadDim { got: n });
    }
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    if spikes.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: spikes.len(),
        });
    }
    for &s in spikes {
        if !s.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "spike".into(),
                val: s,
            });
        }
    }
    Ok(())
}

/// Count spikes per output neuron over the `(t_steps × n)` window.
///
/// Returns a length-`n` vector of integer spike counts (any value `> 0` at a
/// `(t, i)` slot counts as one spike).
///
/// # Errors
/// [`SnnError::BadDim`], [`SnnError::BadTimesteps`], [`SnnError::BadShape`],
/// [`SnnError::OutOfRange`] for non-finite entries.
pub fn spike_counts(spikes: &[f32], t_steps: usize, n: usize) -> SnnResult<Vec<usize>> {
    validate_spikes(spikes, t_steps, n)?;
    let mut counts = vec![0_usize; n];
    for t in 0..t_steps {
        let row = t * n;
        for (i, c) in counts.iter_mut().enumerate() {
            if spikes[row + i] > 0.0 {
                *c += 1;
            }
        }
    }
    Ok(counts)
}

/// Per-neuron firing rate = spike count / `t_steps`, one value per output neuron.
///
/// Rates lie in `[0, 1]` (spikes per timestep). Multiply by `1/dt` to obtain a
/// rate in physical (Hz) units.
///
/// # Errors
/// As [`spike_counts`].
pub fn rate_decode(spikes: &[f32], t_steps: usize, n: usize) -> SnnResult<Vec<f32>> {
    let counts = spike_counts(spikes, t_steps, n)?;
    let inv = 1.0 / t_steps as f32;
    Ok(counts.iter().map(|&c| c as f32 * inv).collect())
}

/// Winner-take-all readout: index of the output neuron with the most spikes.
///
/// Ties are resolved deterministically to the **lowest** index. This is the
/// standard classification decision rule for rate-coded SNN outputs.
///
/// # Errors
/// As [`spike_counts`].
pub fn winner_take_all(spikes: &[f32], t_steps: usize, n: usize) -> SnnResult<usize> {
    let counts = spike_counts(spikes, t_steps, n)?;
    let mut best_idx = 0usize;
    let mut best = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > best {
            best = c;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// Numerically-stable softmax over per-neuron spike counts.
///
/// Computes `p_i = exp(β·c_i − m) / Σ_j exp(β·c_j − m)` with `m = β·max_j c_j`,
/// where `c_i` is the spike count of neuron `i` and `β > 0` is an inverse-
/// temperature that sharpens the distribution. Returns a length-`n` probability
/// vector that sums to 1.
///
/// # Errors
/// As [`spike_counts`], plus [`SnnError::OutOfRange`] if `beta ≤ 0` or
/// non-finite.
pub fn softmax_decode(spikes: &[f32], t_steps: usize, n: usize, beta: f32) -> SnnResult<Vec<f32>> {
    if beta <= 0.0 || !beta.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "beta".into(),
            val: beta,
        });
    }
    let counts = spike_counts(spikes, t_steps, n)?;
    let max_logit = counts
        .iter()
        .map(|&c| beta * c as f32)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = counts
        .iter()
        .map(|&c| (beta * c as f32 - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // Degenerate (cannot happen for finite inputs since at least the max
        // term equals exp(0)=1) — fall back to a uniform distribution.
        let u = 1.0 / n as f32;
        return Ok(vec![u; n]);
    }
    let inv = 1.0 / sum;
    for p in &mut probs {
        *p *= inv;
    }
    Ok(probs)
}

/// Value-weighted population decode of an analogue estimate.
///
/// Each output neuron `i` has a preferred scalar value `prefs[i]`; the decoded
/// estimate is the firing-rate-weighted mean of the preferred values
/// `Σ_i r_i·prefs[i] / Σ_i r_i`, where `r_i` is the firing rate of neuron `i`
/// (Pouget, Dayan & Zemel 2000, "population code" centre-of-mass estimator).
/// When no neuron fires, the unweighted mean of the preferred values is
/// returned so the estimate stays well-defined.
///
/// # Errors
/// As [`spike_counts`], plus [`SnnError::BadShape`] if `prefs.len() != n`.
pub fn population_mean_decode(
    spikes: &[f32],
    prefs: &[f32],
    t_steps: usize,
    n: usize,
) -> SnnResult<f32> {
    if prefs.len() != n {
        return Err(SnnError::BadShape {
            expected: n,
            got: prefs.len(),
        });
    }
    for &p in prefs {
        if !p.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "pref".into(),
                val: p,
            });
        }
    }
    let rates = rate_decode(spikes, t_steps, n)?;
    let weight_sum: f32 = rates.iter().sum();
    if weight_sum <= 0.0 {
        // No spikes: fall back to the unweighted mean of preferred values.
        return Ok(prefs.iter().sum::<f32>() / n as f32);
    }
    let weighted: f32 = rates.iter().zip(prefs.iter()).map(|(&r, &p)| r * p).sum();
    Ok(weighted / weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a time-major `(t_steps × n)` train where neuron `i` fires on the
    /// first `fire_counts[i]` timesteps.
    fn make_train(fire_counts: &[usize], t_steps: usize) -> (Vec<f32>, usize) {
        let n = fire_counts.len();
        let mut buf = vec![0.0_f32; t_steps * n];
        for (i, &k) in fire_counts.iter().enumerate() {
            for t in 0..k.min(t_steps) {
                buf[t * n + i] = 1.0;
            }
        }
        (buf, n)
    }

    #[test]
    fn rejects_zero_neurons() {
        let buf: Vec<f32> = Vec::new();
        assert!(matches!(
            rate_decode(&buf, 4, 0),
            Err(SnnError::BadDim { .. })
        ));
    }

    #[test]
    fn rejects_zero_timesteps() {
        let buf = vec![0.0_f32; 0];
        assert!(matches!(
            rate_decode(&buf, 0, 2),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    #[test]
    fn rejects_bad_shape() {
        let buf = vec![0.0_f32; 5];
        assert!(matches!(
            rate_decode(&buf, 2, 3),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_spike() {
        let mut buf = vec![0.0_f32; 6];
        buf[2] = f32::NAN;
        assert!(matches!(
            rate_decode(&buf, 3, 2),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn spike_counts_correct() {
        let (buf, n) = make_train(&[2, 5, 0], 8);
        let counts = spike_counts(&buf, 8, n).expect("counts");
        assert_eq!(counts, vec![2, 5, 0]);
    }

    #[test]
    fn rate_decode_is_count_over_steps() {
        let t = 10usize;
        let (buf, n) = make_train(&[2, 5, 0], t);
        let rates = rate_decode(&buf, t, n).expect("rates");
        assert!((rates[0] - 0.2).abs() < 1e-6);
        assert!((rates[1] - 0.5).abs() < 1e-6);
        assert!((rates[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn rates_are_bounded_unit_interval() {
        let t = 7usize;
        let (buf, n) = make_train(&[0, 7, 3, 7], t);
        let rates = rate_decode(&buf, t, n).expect("rates");
        for &r in &rates {
            assert!((0.0..=1.0).contains(&r), "rate out of [0,1]: {r}");
        }
    }

    #[test]
    fn winner_take_all_picks_most_active() {
        let (buf, n) = make_train(&[1, 9, 4, 2], 10);
        let w = winner_take_all(&buf, 10, n).expect("wta");
        assert_eq!(w, 1);
    }

    #[test]
    fn winner_take_all_breaks_ties_to_lowest_index() {
        // Neurons 1 and 3 both fire 5 times; the lowest index (1) must win.
        let (buf, n) = make_train(&[0, 5, 0, 5], 8);
        let w = winner_take_all(&buf, 8, n).expect("wta");
        assert_eq!(w, 1);
    }

    #[test]
    fn softmax_rejects_bad_beta() {
        let (buf, n) = make_train(&[1, 2], 4);
        assert!(matches!(
            softmax_decode(&buf, 4, n, 0.0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            softmax_decode(&buf, 4, n, -1.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn softmax_sums_to_one_and_peaks_at_winner() {
        let (buf, n) = make_train(&[1, 8, 2], 10);
        let probs = softmax_decode(&buf, 10, n, 1.0).expect("softmax");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {sum}");
        // Neuron 1 has the most spikes ⇒ highest probability.
        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .map(|(i, _)| i)
            .expect("nonempty");
        assert_eq!(argmax, 1);
    }

    #[test]
    fn softmax_beta_sharpens_distribution() {
        let (buf, n) = make_train(&[2, 6, 3], 10);
        let soft = softmax_decode(&buf, 10, n, 0.2).expect("soft");
        let sharp = softmax_decode(&buf, 10, n, 5.0).expect("sharp");
        // Larger beta concentrates mass on the winner (neuron 1).
        assert!(
            sharp[1] > soft[1],
            "beta should sharpen: soft={} sharp={}",
            soft[1],
            sharp[1]
        );
        assert!(
            sharp[1] > 0.9,
            "high beta should be near one-hot: {}",
            sharp[1]
        );
    }

    #[test]
    fn softmax_uniform_when_no_spikes() {
        let buf = vec![0.0_f32; 12];
        let probs = softmax_decode(&buf, 4, 3, 2.0).expect("softmax");
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-5, "p={p}");
        }
    }

    #[test]
    fn population_mean_decode_centre_of_mass() {
        // Two neurons with prefs 0 and 10; firing rates 0.2 and 0.6 ⇒
        // weighted mean = (0.2*0 + 0.6*10)/(0.8) = 7.5.
        let t = 10usize;
        let (buf, n) = make_train(&[2, 6], t);
        let prefs = [0.0_f32, 10.0];
        let est = population_mean_decode(&buf, &prefs, t, n).expect("decode");
        assert!((est - 7.5).abs() < 1e-5, "est={est}");
    }

    #[test]
    fn population_mean_decode_no_spikes_falls_back_to_mean() {
        let buf = vec![0.0_f32; 9];
        let prefs = [1.0_f32, 4.0, 7.0];
        let est = population_mean_decode(&buf, &prefs, 3, 3).expect("decode");
        // Unweighted mean of prefs = 4.0.
        assert!((est - 4.0).abs() < 1e-6, "est={est}");
    }

    #[test]
    fn population_mean_decode_rejects_bad_prefs_len() {
        let (buf, n) = make_train(&[1, 2], 4);
        let prefs = [0.0_f32; 3]; // wrong length
        assert!(matches!(
            population_mean_decode(&buf, &prefs, 4, n),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn rate_decode_respects_timestep_count() {
        // Same fire counts, different windows ⇒ different rates.
        let (buf16, n) = make_train(&[4, 8], 16);
        let r16 = rate_decode(&buf16, 16, n).expect("r16");
        assert!((r16[0] - 0.25).abs() < 1e-6);
        assert!((r16[1] - 0.5).abs() < 1e-6);
    }
}
