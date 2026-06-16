#![allow(clippy::needless_range_loop)]
//! Population decoding and spike-triggered analysis.
//!
//! Two families of estimators:
//!
//! * **Population vector decoding** (Georgopoulos et al. 1986): each neuron is
//!   assigned a preferred direction in a `dim`-dimensional feature space; the
//!   population estimate is the firing-rate-weighted vector sum of the preferred
//!   directions. A cosine tuning curve maps a stimulus angle to an expected
//!   firing rate.
//! * **Spike-triggered average / covariance** (de Boer & Kuyper 1968; Bialek et
//!   al. 1991; Schwartz et al. 2006): the mean (STA) and covariance (STC) of the
//!   stimulus segments immediately preceding each spike, characterising the
//!   linear receptive field and its second-order structure.
//!
//! Layout conventions:
//! * preferred directions `prefs` are flat **neuron-major**: neuron `i`'s vector
//!   is `prefs[i*dim .. i*dim + dim]`;
//! * the stimulus `stim` is flat **time-major**: row `t` channel `d` is
//!   `stim[t*stim_dim + d]`;
//! * single-neuron spike trains are `&[f32]` of length `t_steps`, where the
//!   value at a spike is treated as its weight.

use crate::error::{SnnError, SnnResult};

/// Population-vector decode of a feature from per-neuron firing rates.
///
/// Each neuron `i` has a preferred direction `prefs[i*dim .. i*dim + dim]` and a
/// scalar firing rate `rates[i]`. The decoded vector is
/// `P[d] = Σ_i w_i · prefs[i*dim + d]`, with weight `w_i = rates[i]` (or
/// `rates[i] − mean(rates)` when `subtract_mean_rate` is set). Preferred
/// directions are used as supplied and are **not** renormalised.
///
/// # Errors
/// Returns [`SnnError::BadDim`] if `n == 0` or `dim == 0`, and
/// [`SnnError::BadShape`] if `prefs.len() != n*dim` or `rates.len() != n`.
pub fn population_vector(
    prefs: &[f32],
    rates: &[f32],
    n: usize,
    dim: usize,
    subtract_mean_rate: bool,
) -> SnnResult<Vec<f32>> {
    if n == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if dim == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if prefs.len() != n * dim {
        return Err(SnnError::BadShape {
            expected: n * dim,
            got: prefs.len(),
        });
    }
    if rates.len() != n {
        return Err(SnnError::BadShape {
            expected: n,
            got: rates.len(),
        });
    }

    let mean_rate = if subtract_mean_rate {
        rates.iter().sum::<f32>() / n as f32
    } else {
        0.0
    };

    let mut p = vec![0.0_f32; dim];
    for i in 0..n {
        let w = rates[i] - mean_rate;
        let base = i * dim;
        for d in 0..dim {
            p[d] += w * prefs[base + d];
        }
    }
    Ok(p)
}

/// Cosine tuning curve: expected firing rate for a stimulus angle.
///
/// Returns `max(0, baseline + gain · cos(θ − θ_pref))`, the canonical cosine
/// tuning model with the rate rectified at zero.
#[must_use]
pub fn cosine_tuning_rate(theta: f32, theta_pref: f32, baseline: f32, gain: f32) -> f32 {
    (baseline + gain * (theta - theta_pref).cos()).max(0.0)
}

/// Validate the shared STA / STC parameters and slice shapes.
fn validate_sta_params(
    stim: &[f32],
    spikes: &[f32],
    t_steps: usize,
    stim_dim: usize,
    window: usize,
) -> SnnResult<()> {
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if stim_dim == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if window == 0 || window > t_steps {
        return Err(SnnError::OutOfRange {
            name: "window".to_string(),
            val: window as f32,
        });
    }
    if stim.len() != t_steps * stim_dim {
        return Err(SnnError::BadShape {
            expected: t_steps * stim_dim,
            got: stim.len(),
        });
    }
    if spikes.len() != t_steps {
        return Err(SnnError::BadShape {
            expected: t_steps,
            got: spikes.len(),
        });
    }
    Ok(())
}

/// Spike-triggered average of the stimulus.
///
/// A spike at step `t` (its value is the trigger weight) is associated with the
/// stimulus segment spanning rows `t−window+1 ..= t`; spikes with `t < window−1`
/// are skipped (insufficient history). The STA is the weight-normalised mean of
/// those segments, laid out **window-major**:
/// `STA[w*stim_dim + d] = (1/Σweights) · Σ_t weight(t) · stim[(t−window+1+w)*stim_dim + d]`.
/// The result has length `window * stim_dim`. With no qualifying spikes the STA
/// is all zeros.
///
/// # Errors
/// Returns [`SnnError::BadTimesteps`] for `t_steps == 0`, [`SnnError::BadDim`]
/// for `stim_dim == 0`, [`SnnError::OutOfRange`] if `window` is outside
/// `1..=t_steps`, and [`SnnError::BadShape`] on a stimulus or spike length
/// mismatch.
pub fn spike_triggered_average(
    stim: &[f32],
    spikes: &[f32],
    t_steps: usize,
    stim_dim: usize,
    window: usize,
) -> SnnResult<Vec<f32>> {
    validate_sta_params(stim, spikes, t_steps, stim_dim, window)?;
    let len = window * stim_dim;
    let mut sta = vec![0.0_f32; len];
    let mut weight_sum = 0.0_f32;
    for t in (window - 1)..t_steps {
        let weight = spikes[t];
        if weight == 0.0 {
            continue;
        }
        weight_sum += weight;
        let start = t + 1 - window;
        for w in 0..window {
            let row = (start + w) * stim_dim;
            let dst = w * stim_dim;
            for d in 0..stim_dim {
                sta[dst + d] += weight * stim[row + d];
            }
        }
    }
    if weight_sum != 0.0 {
        let inv = 1.0 / weight_sum;
        for v in &mut sta {
            *v *= inv;
        }
    }
    Ok(sta)
}

/// Spike-triggered covariance of the stimulus.
///
/// Using the same triggering windows as [`spike_triggered_average`], with `W`
/// qualifying spikes and the STA subtracted from each segment, returns the
/// `L × L` (row-major, `L = window*stim_dim`) sample covariance
/// `STC[a*L + b] = (1/(W−1)) · Σ (x_t[a]−STA[a])(x_t[b]−STA[b])`. Spikes are
/// counted (not weighted) for `W` and the segments enter unweighted, matching
/// the standard STC estimator. With fewer than two qualifying spikes the STC is
/// all zeros.
///
/// # Errors
/// Same validation as [`spike_triggered_average`].
pub fn spike_triggered_covariance(
    stim: &[f32],
    spikes: &[f32],
    t_steps: usize,
    stim_dim: usize,
    window: usize,
) -> SnnResult<Vec<f32>> {
    validate_sta_params(stim, spikes, t_steps, stim_dim, window)?;
    let len = window * stim_dim;
    let mut stc = vec![0.0_f32; len * len];

    // Collect the triggering (mean-centred) stimulus segments.
    let mut segments: Vec<Vec<f32>> = Vec::new();
    for t in (window - 1)..t_steps {
        if spikes[t] == 0.0 {
            continue;
        }
        let start = t + 1 - window;
        let mut seg = vec![0.0_f32; len];
        for w in 0..window {
            let row = (start + w) * stim_dim;
            let dst = w * stim_dim;
            seg[dst..dst + stim_dim].copy_from_slice(&stim[row..row + stim_dim]);
        }
        segments.push(seg);
    }

    let w_count = segments.len();
    if w_count < 2 {
        return Ok(stc);
    }

    // Unweighted STA over the triggering segments.
    let mut sta = vec![0.0_f32; len];
    for seg in &segments {
        for a in 0..len {
            sta[a] += seg[a];
        }
    }
    let inv_w = 1.0 / w_count as f32;
    for a in 0..len {
        sta[a] *= inv_w;
    }

    // Sample covariance with (W − 1) denominator.
    let inv_dof = 1.0 / (w_count as f32 - 1.0);
    for seg in &segments {
        for a in 0..len {
            let da = seg[a] - sta[a];
            if da == 0.0 {
                continue;
            }
            let row = a * len;
            for b in 0..len {
                let db = seg[b] - sta[b];
                stc[row + b] += da * db;
            }
        }
    }
    for v in &mut stc {
        *v *= inv_dof;
    }
    Ok(stc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn pop_vector_basic_orthogonal() {
        // Neuron 0 prefers +x, neuron 1 prefers +y; rates 2 and 1.
        let prefs = [1.0_f32, 0.0, 0.0, 1.0];
        let rates = [2.0_f32, 1.0];
        let p = population_vector(&prefs, &rates, 2, 2, false).expect("pv");
        assert_eq!(p, vec![2.0, 1.0]);
        let angle = p[1].atan2(p[0]);
        assert!((angle - 0.463_647_6).abs() < 1e-5, "angle={angle}");
    }

    #[test]
    fn pop_vector_equal_rates_diagonal() {
        let prefs = [1.0_f32, 0.0, 0.0, 1.0];
        let rates = [1.0_f32, 1.0];
        let p = population_vector(&prefs, &rates, 2, 2, false).expect("pv");
        assert!((p[0] - 1.0).abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6);
        let angle = p[1].atan2(p[0]);
        assert!((angle - PI / 4.0).abs() < 1e-6, "angle={angle}");
    }

    #[test]
    fn pop_vector_zero_rates() {
        let prefs = [1.0_f32, 0.0, 0.0, 1.0];
        let rates = [0.0_f32, 0.0];
        let p = population_vector(&prefs, &rates, 2, 2, false).expect("pv");
        assert_eq!(p, vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_peak_and_trough() {
        let baseline = 5.0_f32;
        let gain = 10.0_f32;
        let pref = 0.7_f32;
        // Peak at the preferred angle.
        let peak = cosine_tuning_rate(pref, pref, baseline, gain);
        assert!((peak - (baseline + gain)).abs() < 1e-5, "peak={peak}");
        // Clamped to zero half a turn away (baseline - gain < 0).
        let trough = cosine_tuning_rate(pref + PI, pref, baseline, gain);
        assert!(trough.abs() < 1e-5, "trough={trough}");
    }

    #[test]
    fn cosine_population_recovers_angle() {
        // Eight neurons evenly tiling the circle, cosine-tuned to a stimulus.
        let n = 8_usize;
        let stim_angle = PI / 3.0;
        let baseline = 5.0_f32;
        let gain = 10.0_f32;
        let mut prefs = vec![0.0_f32; n * 2];
        let mut rates = vec![0.0_f32; n];
        for i in 0..n {
            let pref = 2.0 * PI * i as f32 / n as f32;
            prefs[i * 2] = pref.cos();
            prefs[i * 2 + 1] = pref.sin();
            rates[i] = cosine_tuning_rate(stim_angle, pref, baseline, gain);
        }
        let p = population_vector(&prefs, &rates, n, 2, false).expect("pv");
        let decoded = p[1].atan2(p[0]);
        assert!(
            (decoded - stim_angle).abs() < 0.05,
            "decoded={decoded} expected={stim_angle}"
        );
    }

    #[test]
    fn sta_recovers_bump() {
        // 1-D stimulus; a 5-sample bump precedes each of several locked spikes.
        let stim_dim = 1_usize;
        let window = 5_usize;
        let bump = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
        // Lay the bump down repeatedly with a quiet gap, spike on the last
        // bump sample so rows (t-4..=t) exactly equal the bump.
        let mut stim: Vec<f32> = Vec::new();
        let mut spikes: Vec<f32> = Vec::new();
        let repeats = 4_usize;
        let gap = 3_usize;
        for _ in 0..repeats {
            for (k, &b) in bump.iter().enumerate() {
                stim.push(b);
                // Spike exactly on the final bump sample.
                spikes.push(if k == bump.len() - 1 { 1.0 } else { 0.0 });
            }
            for _ in 0..gap {
                stim.push(0.0);
                spikes.push(0.0);
            }
        }
        let t_steps = stim.len();
        let sta = spike_triggered_average(&stim, &spikes, t_steps, stim_dim, window).expect("sta");
        for (k, &b) in bump.iter().enumerate() {
            assert!((sta[k] - b).abs() < 1e-5, "sta[{k}]={} != {b}", sta[k]);
        }
    }

    #[test]
    fn sta_averages_two_patterns() {
        // Two distinct 5-sample patterns A and B, one spike each => STA=(A+B)/2.
        let stim_dim = 1_usize;
        let window = 5_usize;
        let a = [0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0, 1.0, 0.0];
        let mut stim: Vec<f32> = Vec::new();
        let mut spikes: Vec<f32> = Vec::new();
        for &x in &a {
            stim.push(x);
        }
        for (k, _) in a.iter().enumerate() {
            spikes.push(if k == a.len() - 1 { 1.0 } else { 0.0 });
        }
        // Gap so B's window does not overlap A.
        for _ in 0..4 {
            stim.push(0.0);
            spikes.push(0.0);
        }
        let b_start = stim.len();
        for &x in &b {
            stim.push(x);
        }
        for (k, _) in b.iter().enumerate() {
            spikes.push(if b_start + k == stim.len().saturating_sub(1) {
                1.0
            } else {
                0.0
            });
        }
        let t_steps = stim.len();
        let sta = spike_triggered_average(&stim, &spikes, t_steps, stim_dim, window).expect("sta");
        for k in 0..window {
            let expected = (a[k] + b[k]) / 2.0;
            assert!((sta[k] - expected).abs() < 1e-5, "sta[{k}]={}", sta[k]);
        }
    }

    #[test]
    fn stc_zero_for_single_repeated_pattern() {
        // The same window triggers every spike => zero covariance.
        let stim_dim = 1_usize;
        let window = 3_usize;
        let pattern = [1.0_f32, 0.5, 0.25];
        let mut stim: Vec<f32> = Vec::new();
        let mut spikes: Vec<f32> = Vec::new();
        for _ in 0..3 {
            for (k, &x) in pattern.iter().enumerate() {
                stim.push(x);
                spikes.push(if k == pattern.len() - 1 { 1.0 } else { 0.0 });
            }
            for _ in 0..2 {
                stim.push(0.0);
                spikes.push(0.0);
            }
        }
        let t_steps = stim.len();
        let stc =
            spike_triggered_covariance(&stim, &spikes, t_steps, stim_dim, window).expect("stc");
        for &v in &stc {
            assert!(v.abs() < 1e-6, "non-zero stc entry {v}");
        }
    }

    #[test]
    fn stc_diagonal_two_patterns() {
        // A=[0,0,1,0,0], B=[0,1,0,1,0]; STA=(A+B)/2; component 2 variance = 0.5
        // with the (W-1)=1 denominator.
        let stim_dim = 1_usize;
        let window = 5_usize;
        let a = [0.0_f32, 0.0, 1.0, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0, 1.0, 0.0];
        let mut stim: Vec<f32> = Vec::new();
        let mut spikes: Vec<f32> = Vec::new();
        for &x in &a {
            stim.push(x);
        }
        for (k, _) in a.iter().enumerate() {
            spikes.push(if k == a.len() - 1 { 1.0 } else { 0.0 });
        }
        for _ in 0..4 {
            stim.push(0.0);
            spikes.push(0.0);
        }
        let len_before_b = stim.len();
        for &x in &b {
            stim.push(x);
        }
        for (k, _) in b.iter().enumerate() {
            spikes.push(if len_before_b + k == stim.len().saturating_sub(1) {
                1.0
            } else {
                0.0
            });
        }
        let t_steps = stim.len();
        let stc =
            spike_triggered_covariance(&stim, &spikes, t_steps, stim_dim, window).expect("stc");
        let len = window * stim_dim;
        // Diagonal entry for component 2 (the third sample).
        let diag2 = stc[2 * len + 2];
        assert!((diag2 - 0.5).abs() < 1e-5, "diag2={diag2}");
    }

    #[test]
    fn no_spikes_zero_sta_and_stc() {
        let stim = vec![0.3_f32; 10];
        let spikes = vec![0.0_f32; 10];
        let sta = spike_triggered_average(&stim, &spikes, 10, 1, 4).expect("sta");
        assert!(sta.iter().all(|&v| v == 0.0));
        let stc = spike_triggered_covariance(&stim, &spikes, 10, 1, 4).expect("stc");
        assert!(stc.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn shape_and_param_errors() {
        // population_vector.
        let prefs = [1.0_f32, 0.0, 0.0, 1.0];
        let rates = [1.0_f32, 1.0];
        assert!(matches!(
            population_vector(&prefs, &rates, 0, 2, false),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            population_vector(&prefs, &rates, 2, 0, false),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            population_vector(&prefs, &rates, 3, 2, false),
            Err(SnnError::BadShape { .. })
        ));
        // STA / STC.
        let stim = vec![0.0_f32; 8];
        let spikes = vec![0.0_f32; 8];
        assert!(matches!(
            spike_triggered_average(&stim, &spikes, 0, 1, 1),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            spike_triggered_average(&stim, &spikes, 8, 0, 1),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            spike_triggered_average(&stim, &spikes, 8, 1, 0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            spike_triggered_average(&stim, &spikes, 8, 1, 9),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            spike_triggered_average(&stim, &spikes, 8, 2, 1),
            Err(SnnError::BadShape { .. })
        ));
    }
}
