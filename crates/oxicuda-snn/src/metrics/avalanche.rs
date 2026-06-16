#![allow(clippy::needless_range_loop)]
//! Neuronal avalanches and criticality analysis (Beggs & Plenz 2003).
//!
//! A population spike raster is coarse-grained into time bins of `bin_steps`
//! steps. The summed population activity in each bin is the *bin amplitude*; a
//! bin is *empty* iff its amplitude is exactly zero. A neuronal avalanche is a
//! maximal run of consecutive non-empty bins bracketed by empty bins (or by the
//! start / end of the recording).
//!
//! For an avalanche, three observables are recorded:
//! * `size`     — total population spike count summed over its bins,
//! * `duration` — number of bins it spans,
//! * `bins`     — per-bin population counts (the temporal profile).
//!
//! From the population of avalanches we derive criticality diagnostics:
//! the branching parameter σ (mean descendants per ancestor) and the power-law
//! exponent of the avalanche-size distribution via the discrete maximum-
//! likelihood estimator of Clauset, Shalizi & Newman (2009).
//!
//! Spike memory layout is **time-major**: neuron `i` at step `t` lives at
//! `spikes[t * n + i]`. A spike event is any value `!= 0.0`; bin amplitudes sum
//! the raw `f32` values (for binary rasters this equals the spike count).

use crate::error::{SnnError, SnnResult};

/// A single detected neuronal avalanche.
#[derive(Debug, Clone)]
pub struct Avalanche {
    /// Total population spike count over the avalanche (rounded bin sum).
    pub size: usize,
    /// Number of consecutive non-empty bins the avalanche spans.
    pub duration: usize,
    /// Per-bin rounded population counts; `bins.len() == duration`.
    pub bins: Vec<usize>,
}

/// Aggregate avalanche statistics for a recording.
#[derive(Debug, Clone)]
pub struct AvalancheStats {
    /// Every detected avalanche, in temporal order.
    pub avalanches: Vec<Avalanche>,
    /// Size histogram as sorted-ascending `(size, count)` pairs.
    pub size_hist: Vec<(usize, usize)>,
    /// Duration histogram as sorted-ascending `(duration, count)` pairs.
    pub duration_hist: Vec<(usize, usize)>,
}

/// Build a sorted-ascending `(key, count)` histogram from a list of keys.
fn histogram(values: &[usize]) -> Vec<(usize, usize)> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mut hist: Vec<(usize, usize)> = Vec::new();
    for v in sorted {
        match hist.last_mut() {
            Some(last) if last.0 == v => last.1 += 1,
            _ => hist.push((v, 1)),
        }
    }
    hist
}

/// Detect neuronal avalanches from a time-major population spike raster.
///
/// Population activity is coarse-grained into `n_bins = ceil(t_steps / bin_steps)`
/// bins (the final bin may be partial). Bin amplitude `a[b]` is the sum of all
/// `f32` spike values over the `bin_steps` rows and `n` neurons of bin `b`. A run
/// of consecutive bins with `a[b] != 0` bracketed by empty bins (or the recording
/// boundary) is one avalanche; its `size` is the rounded sum of those amplitudes,
/// `duration` the bin count, and `bins` the per-bin rounded amplitudes.
///
/// An all-zero raster yields empty statistics (this is **not** an error).
///
/// # Errors
/// Returns [`SnnError::BadTimesteps`] if `t_steps == 0`, [`SnnError::BadDim`] if
/// `n == 0` or `bin_steps == 0`, and [`SnnError::BadShape`] if
/// `spikes.len() != t_steps * n`.
pub fn detect_avalanches(
    spikes: &[f32],
    t_steps: usize,
    n: usize,
    bin_steps: usize,
) -> SnnResult<AvalancheStats> {
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if n == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if bin_steps == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if spikes.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: spikes.len(),
        });
    }

    // Coarse-grain into bin amplitudes (summed population activity per bin).
    let n_bins = t_steps.div_ceil(bin_steps);
    let mut amplitudes = vec![0.0_f32; n_bins];
    for t in 0..t_steps {
        let b = t / bin_steps;
        let mut row_sum = 0.0_f32;
        let base = t * n;
        for i in 0..n {
            row_sum += spikes[base + i];
        }
        amplitudes[b] += row_sum;
    }

    // Segment into avalanches: maximal runs of non-empty bins.
    let mut avalanches: Vec<Avalanche> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut durations: Vec<usize> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for b in 0..n_bins {
        if amplitudes[b] != 0.0 {
            current.push(amplitudes[b].round() as usize);
        } else if !current.is_empty() {
            let size: usize = current.iter().sum();
            let duration = current.len();
            sizes.push(size);
            durations.push(duration);
            avalanches.push(Avalanche {
                size,
                duration,
                bins: std::mem::take(&mut current),
            });
        }
    }
    if !current.is_empty() {
        let size: usize = current.iter().sum();
        let duration = current.len();
        sizes.push(size);
        durations.push(duration);
        avalanches.push(Avalanche {
            size,
            duration,
            bins: current,
        });
    }

    let size_hist = histogram(&sizes);
    let duration_hist = histogram(&durations);
    Ok(AvalancheStats {
        avalanches,
        size_hist,
        duration_hist,
    })
}

/// Per-avalanche branching parameter σ (Beggs & Plenz 2003).
///
/// For each avalanche of duration ≥ 2 with a non-zero first bin, the ratio
/// `bins[1] / bins[0]` estimates the number of descendant events triggered per
/// ancestor; σ is the mean of these ratios. Returns `None` if no avalanche
/// qualifies. σ ≈ 1 signals a critical branching process.
#[must_use]
pub fn branching_parameter(stats: &AvalancheStats) -> Option<f32> {
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for av in &stats.avalanches {
        if av.duration >= 2 && av.bins[0] > 0 {
            sum += av.bins[1] as f32 / av.bins[0] as f32;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

/// Global (pooled) branching parameter.
///
/// Aggregates descendants and ancestors across every avalanche:
/// `σ = (Σ_a Σ_{k≥1} bins[k]) / (Σ_a Σ_{k≤D−2} bins[k])`, i.e. the total count
/// in all but the first bin divided by the total count in all but the last bin.
/// Returns `None` if the denominator is zero.
#[must_use]
pub fn branching_parameter_global(stats: &AvalancheStats) -> Option<f32> {
    let mut numerator = 0_u64;
    let mut denominator = 0_u64;
    for av in &stats.avalanches {
        let d = av.duration;
        for k in 0..d {
            if k >= 1 {
                numerator += av.bins[k] as u64;
            }
            if k + 2 <= d {
                denominator += av.bins[k] as u64;
            }
        }
    }
    if denominator == 0 {
        None
    } else {
        Some(numerator as f32 / denominator as f32)
    }
}

/// Discrete power-law MLE exponent of the avalanche-size distribution.
///
/// Implements the Clauset-Shalizi-Newman (2009) discrete maximum-likelihood
/// estimator on avalanche sizes `S ≥ x_min`:
///
/// ```text
/// α̂ = 1 + N_tail / Σ_{S ≥ x_min} ln(S / (x_min − 0.5))
/// ```
///
/// Returns `None` if fewer than two avalanche sizes satisfy `S ≥ x_min` or if
/// the log-sum is non-positive. The continuity correction `x_min − 0.5` matches
/// the discrete CSN estimator.
#[must_use]
pub fn powerlaw_mle_exponent(stats: &AvalancheStats, x_min: usize) -> Option<f32> {
    let x_ref = x_min as f32 - 0.5;
    if x_ref <= 0.0 {
        return None;
    }
    let mut n_tail = 0_usize;
    let mut log_sum = 0.0_f32;
    for av in &stats.avalanches {
        if av.size >= x_min {
            n_tail += 1;
            log_sum += (av.size as f32 / x_ref).ln();
        }
    }
    if n_tail < 2 || log_sum <= 0.0 {
        return None;
    }
    Some(1.0 + n_tail as f32 / log_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-neuron (n=1) time-major raster from a per-step amplitude
    /// list. Each amplitude becomes one step; bin_steps=1 keeps one step per bin.
    fn raster_from_steps(amps: &[f32]) -> Vec<f32> {
        amps.to_vec()
    }

    #[test]
    fn branching_sigma_one_for_bins_2_2() {
        // Two bins of amplitude 2, then an empty bin to close the avalanche.
        let raster = raster_from_steps(&[2.0, 2.0, 0.0]);
        let stats = detect_avalanches(&raster, 3, 1, 1).expect("detect");
        assert_eq!(stats.avalanches.len(), 1);
        assert_eq!(stats.avalanches[0].bins, vec![2, 2]);
        let sigma = branching_parameter(&stats).expect("sigma");
        assert!((sigma - 1.0).abs() < 1e-6, "sigma={sigma}");
    }

    #[test]
    fn branching_sigma_half_for_bins_4_2() {
        let raster = raster_from_steps(&[4.0, 2.0, 0.0]);
        let stats = detect_avalanches(&raster, 3, 1, 1).expect("detect");
        assert_eq!(stats.avalanches[0].bins, vec![4, 2]);
        let sigma = branching_parameter(&stats).expect("sigma");
        assert!((sigma - 0.5).abs() < 1e-6, "sigma={sigma}");
        let global = branching_parameter_global(&stats).expect("global");
        assert!((global - 0.5).abs() < 1e-6, "global={global}");
    }

    #[test]
    fn branching_two_for_bins_1_2_4() {
        let raster = raster_from_steps(&[1.0, 2.0, 4.0, 0.0]);
        let stats = detect_avalanches(&raster, 4, 1, 1).expect("detect");
        assert_eq!(stats.avalanches[0].bins, vec![1, 2, 4]);
        let per = branching_parameter(&stats).expect("per");
        assert!((per - 2.0).abs() < 1e-6, "per={per}");
        // Global: numerator = 2+4 = 6, denominator = 1+2 = 3 => 2.0.
        let global = branching_parameter_global(&stats).expect("global");
        assert!((global - 2.0).abs() < 1e-6, "global={global}");
    }

    #[test]
    fn branching_none_when_all_duration_one() {
        // Each avalanche is a single non-empty bin separated by empties.
        let raster = raster_from_steps(&[3.0, 0.0, 5.0, 0.0, 1.0]);
        let stats = detect_avalanches(&raster, 5, 1, 1).expect("detect");
        assert_eq!(stats.avalanches.len(), 3);
        for av in &stats.avalanches {
            assert_eq!(av.duration, 1);
        }
        assert!(branching_parameter(&stats).is_none());
        // Global denominator is zero (no bin has a successor within its run).
        assert!(branching_parameter_global(&stats).is_none());
    }

    #[test]
    fn powerlaw_exponent_known_value() {
        // Sizes {2, 2, 4, 4} via four singleton avalanches.
        let raster = raster_from_steps(&[2.0, 0.0, 2.0, 0.0, 4.0, 0.0, 4.0]);
        let stats = detect_avalanches(&raster, 7, 1, 1).expect("detect");
        let sizes: Vec<usize> = stats.avalanches.iter().map(|a| a.size).collect();
        assert_eq!(sizes, vec![2, 2, 4, 4]);
        let alpha = powerlaw_mle_exponent(&stats, 2).expect("alpha");
        // 1 + 4 / (2*ln(2/1.5) + 2*ln(4/1.5)) = 1 + 4/2.537022 = 2.5767.
        assert!((alpha - 2.5767).abs() < 1e-3, "alpha={alpha}");
    }

    #[test]
    fn powerlaw_none_below_two_sizes() {
        let raster = raster_from_steps(&[5.0, 0.0, 1.0]);
        let stats = detect_avalanches(&raster, 3, 1, 1).expect("detect");
        // Only one size (5) is >= x_min=3.
        assert!(powerlaw_mle_exponent(&stats, 3).is_none());
    }

    #[test]
    fn two_distinct_avalanches_with_hists() {
        // Avalanche A: bins [2,1] (size 3, dur 2); gap; avalanche B: bins [3]
        // (size 3, dur 1). Two distinct durations {1,2}, one shared size {3}.
        let raster = raster_from_steps(&[2.0, 1.0, 0.0, 3.0]);
        let stats = detect_avalanches(&raster, 4, 1, 1).expect("detect");
        assert_eq!(stats.avalanches.len(), 2);
        assert_eq!(stats.avalanches[0].size, 3);
        assert_eq!(stats.avalanches[0].duration, 2);
        assert_eq!(stats.avalanches[1].size, 3);
        assert_eq!(stats.avalanches[1].duration, 1);
        // size_hist: {3:2}; duration_hist: {1:1, 2:1} sorted ascending.
        assert_eq!(stats.size_hist, vec![(3, 2)]);
        assert_eq!(stats.duration_hist, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn multi_neuron_binning_population_sum() {
        // 3 neurons, 4 steps; sum across neurons per step then bin by 2 steps.
        // Step amplitudes (pop counts): t0=2, t1=1, t2=0, t3=0.
        // n=3 time-major: [n0,n1,n2] per step.
        let raster = vec![
            1.0, 1.0, 0.0, // t0 -> 2
            1.0, 0.0, 0.0, // t1 -> 1
            0.0, 0.0, 0.0, // t2 -> 0
            0.0, 0.0, 0.0, // t3 -> 0
        ];
        // bin_steps=2 => bin0 = t0+t1 = 3, bin1 = t2+t3 = 0. One avalanche [3].
        let stats = detect_avalanches(&raster, 4, 3, 2).expect("detect");
        assert_eq!(stats.avalanches.len(), 1);
        assert_eq!(stats.avalanches[0].bins, vec![3]);
        assert_eq!(stats.avalanches[0].size, 3);
    }

    #[test]
    fn all_zero_yields_empty_and_nones() {
        let raster = vec![0.0_f32; 5 * 4];
        let stats = detect_avalanches(&raster, 5, 4, 1).expect("detect");
        assert!(stats.avalanches.is_empty());
        assert!(stats.size_hist.is_empty());
        assert!(stats.duration_hist.is_empty());
        assert!(branching_parameter(&stats).is_none());
        assert!(branching_parameter_global(&stats).is_none());
        assert!(powerlaw_mle_exponent(&stats, 1).is_none());
    }

    #[test]
    fn shape_and_param_errors() {
        let raster = vec![0.0_f32; 6];
        assert!(matches!(
            detect_avalanches(&raster, 0, 3, 1),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            detect_avalanches(&raster, 2, 0, 1),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            detect_avalanches(&raster, 2, 3, 0),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            detect_avalanches(&raster, 3, 3, 1),
            Err(SnnError::BadShape { .. })
        ));
    }
}
