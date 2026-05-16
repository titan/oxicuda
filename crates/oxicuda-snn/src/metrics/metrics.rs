#![allow(clippy::needless_range_loop)]
//! Spike-train analysis metrics: firing rates, ISI, CV, distance functions.

use crate::error::{SnnError, SnnResult};

/// Per-neuron average firing rate (spikes per unit of time).
///
/// `spikes` length must equal `t_steps * n` flattened row-major over time.
pub fn firing_rate(spikes: &[f32], t_steps: usize, n: usize, dt: f32) -> SnnResult<Vec<f32>> {
    if dt <= 0.0 {
        return Err(SnnError::BadDt { dt });
    }
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if spikes.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: spikes.len(),
        });
    }
    let total_time = t_steps as f32 * dt;
    let mut rates = vec![0.0_f32; n];
    for t in 0..t_steps {
        for i in 0..n {
            rates[i] += spikes[t * n + i];
        }
    }
    for r in &mut rates {
        *r /= total_time;
    }
    Ok(rates)
}

/// Inter-spike intervals from a single-neuron binary spike train of length `t_steps`.
#[must_use]
pub fn isi(spike_train_per_neuron: &[f32], dt: f32) -> Vec<f32> {
    let mut intervals = Vec::new();
    let mut prev: Option<usize> = None;
    for (t, &s) in spike_train_per_neuron.iter().enumerate() {
        if s != 0.0 {
            if let Some(p) = prev {
                intervals.push((t - p) as f32 * dt);
            }
            prev = Some(t);
        }
    }
    intervals
}

/// Coefficient of variation `σ/μ` of an interval list. Returns 0 for ≤1 interval.
#[must_use]
pub fn cv_isi(intervals: &[f32]) -> f32 {
    if intervals.len() < 2 {
        return 0.0;
    }
    let mu = intervals.iter().copied().sum::<f32>() / intervals.len() as f32;
    if mu.abs() < 1e-30 {
        return 0.0;
    }
    let var = intervals.iter().map(|x| (x - mu).powi(2)).sum::<f32>() / intervals.len() as f32;
    var.sqrt() / mu
}

/// Van Rossum distance between two spike trains.
///
/// Convolves each train with `exp(−t/τ)·H(t)` then returns the L² norm of the difference.
pub fn van_rossum_distance(
    s_a: &[f32],
    s_b: &[f32],
    t_steps: usize,
    tau: f32,
    dt: f32,
) -> SnnResult<f32> {
    if tau <= 0.0 {
        return Err(SnnError::BadTau { tau });
    }
    if dt <= 0.0 {
        return Err(SnnError::BadDt { dt });
    }
    if s_a.len() != t_steps || s_b.len() != t_steps {
        return Err(SnnError::IncompatibleLength {
            a: s_a.len(),
            b: s_b.len(),
        });
    }
    let decay = (-dt / tau).exp();
    let mut filt_a = 0.0_f32;
    let mut filt_b = 0.0_f32;
    let mut sumsq = 0.0_f32;
    for t in 0..t_steps {
        filt_a = filt_a * decay + s_a[t];
        filt_b = filt_b * decay + s_b[t];
        let d = filt_a - filt_b;
        sumsq += d * d;
    }
    Ok((sumsq * dt).sqrt())
}

/// Victor-Purpura distance (cost-based) between two spike-time lists.
///
/// `q` is the cost per unit of time of moving a spike. `q=0` ignores timing
/// and reduces to the symmetric difference in spike counts.
#[must_use]
pub fn victor_purpura_distance(times_a: &[f32], times_b: &[f32], q: f32) -> f32 {
    let m = times_a.len();
    let n = times_b.len();
    let mut g = vec![0.0_f32; (m + 1) * (n + 1)];
    for i in 0..=m {
        g[i * (n + 1)] = i as f32;
    }
    for j in 0..=n {
        g[j] = j as f32;
    }
    for i in 1..=m {
        for j in 1..=n {
            let del = g[(i - 1) * (n + 1) + j] + 1.0;
            let ins = g[i * (n + 1) + j - 1] + 1.0;
            let mv = g[(i - 1) * (n + 1) + j - 1] + q * (times_a[i - 1] - times_b[j - 1]).abs();
            let best = del.min(ins).min(mv);
            g[i * (n + 1) + j] = best;
        }
    }
    g[m * (n + 1) + n]
}

/// Synchrony index: peak normalised cross-correlation between two equal-length spike trains.
pub fn sync_index(s_a: &[f32], s_b: &[f32], t_steps: usize) -> SnnResult<f32> {
    if s_a.len() != t_steps || s_b.len() != t_steps {
        return Err(SnnError::IncompatibleLength {
            a: s_a.len(),
            b: s_b.len(),
        });
    }
    let na = s_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = s_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-30 || nb < 1e-30 {
        return Ok(0.0);
    }
    let max_lag = (t_steps / 4).max(1);
    let mut peak = 0.0_f32;
    for lag in 0..=max_lag {
        let mut acc_pos = 0.0_f32;
        let mut acc_neg = 0.0_f32;
        for t in 0..t_steps - lag {
            acc_pos += s_a[t] * s_b[t + lag];
            acc_neg += s_a[t + lag] * s_b[t];
        }
        let v = acc_pos.max(acc_neg) / (na * nb);
        if v > peak {
            peak = v;
        }
    }
    Ok(peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_shape_and_value() {
        // 4 neurons, 10 timesteps; neuron 0 fires every step.
        let mut s = vec![0.0_f32; 4 * 10];
        for t in 0..10 {
            s[t * 4] = 1.0;
        }
        let r = firing_rate(&s, 10, 4, 1.0).expect("ok");
        assert_eq!(r.len(), 4);
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
    }

    #[test]
    fn isi_basic() {
        let train = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0];
        let intervals = isi(&train, 1.0);
        assert_eq!(intervals, vec![2.0_f32, 2.0]);
    }

    #[test]
    fn cv_uniform_zero() {
        let intervals = vec![5.0_f32; 10];
        assert!(cv_isi(&intervals).abs() < 1e-6);
    }

    #[test]
    fn van_rossum_self_zero() {
        let s = vec![0.0_f32, 1.0, 0.0, 1.0];
        let d = van_rossum_distance(&s, &s, 4, 5.0, 1.0).expect("ok");
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn victor_purpura_self_zero() {
        let t = vec![1.0_f32, 5.0, 9.0];
        assert!(victor_purpura_distance(&t, &t, 1.0).abs() < 1e-6);
    }

    #[test]
    fn sync_identical_one() {
        let s = vec![0.0_f32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let v = sync_index(&s, &s, s.len()).expect("ok");
        assert!((v - 1.0).abs() < 1e-6);
    }
}
