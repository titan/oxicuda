//! Time-resolved firing-rate estimation by Kernel Density Estimation (KDE).
//!
//! The classic peri-stimulus time histogram (PSTH) estimates an instantaneous
//! firing rate by binning spikes into fixed windows, which introduces an
//! arbitrary bin edge and a blocky estimate. KDE replaces the histogram by a
//! smooth sum of kernels centred on every spike (Parzen 1962; Shimazaki &
//! Shinomoto, *J. Comput. Neurosci.* 2010, "Kernel bandwidth optimization in
//! spike rate estimation"). For spike times `{t_s}` and bandwidth `h` the
//! estimate at time `t` is
//!
//! ```text
//! r(t) = Σ_s (1/h) · K( (t − t_s) / h )
//! ```
//!
//! where `K` is a normalised kernel (`∫ K(z) dz = 1`). The `1/h` prefactor makes
//! each spike's contribution integrate to one, so the total rate integrates to
//! the spike count: `∫ r(t) dt ≈ N_spikes`, and `r` has units of spikes per unit
//! time.
//!
//! # Kernels
//!
//! | Kernel        | `K(z)`                                            | support     |
//! |---------------|---------------------------------------------------|-------------|
//! | Gaussian      | `exp(−z²/2)/√(2π)`                                 | `(−∞, ∞)`   |
//! | Exponential   | `exp(−z)` for `z ≥ 0`, else `0` (causal)           | `[0, ∞)`    |
//! | Epanechnikov  | `¾(1 − z²)` for `|z| ≤ 1`, else `0`               | `[−1, 1]`   |
//! | Boxcar        | `½` for `|z| ≤ 1`, else `0` (moving average)      | `[−1, 1]`   |
//!
//! Each is normalised so `∫ K = 1` (the exponential kernel is one-sided/causal,
//! convenient for online rate read-out).

use crate::error::{SnnError, SnnResult};

/// Smoothing-kernel family for KDE rate estimation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum KdeKernel {
    /// Gaussian `exp(−z²/2)/√(2π)`, infinite smooth support.
    #[default]
    Gaussian,
    /// One-sided causal exponential `exp(−z)·H(z)` (only past spikes contribute).
    Exponential,
    /// Epanechnikov `¾(1 − z²)` on `|z| ≤ 1` (minimum-variance optimal kernel).
    Epanechnikov,
    /// Boxcar `½` on `|z| ≤ 1`, i.e. a centred moving average of width `2h`.
    Boxcar,
}

impl KdeKernel {
    /// Evaluate the normalised kernel `K(z)` (such that `∫ K(z) dz = 1`).
    #[must_use]
    pub fn eval(self, z: f32) -> f32 {
        match self {
            KdeKernel::Gaussian => {
                const INV_SQRT_2PI: f32 = 0.398_942_3; // 1/√(2π)
                INV_SQRT_2PI * (-0.5 * z * z).exp()
            }
            KdeKernel::Exponential => {
                if z >= 0.0 {
                    (-z).exp()
                } else {
                    0.0
                }
            }
            KdeKernel::Epanechnikov => {
                if z.abs() <= 1.0 {
                    0.75 * (1.0 - z * z)
                } else {
                    0.0
                }
            }
            KdeKernel::Boxcar => {
                if z.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
        }
    }
}

/// Configuration for KDE firing-rate estimation.
#[derive(Debug, Clone, Copy)]
pub struct KdeRateConfig {
    /// Smoothing kernel family.
    pub kernel: KdeKernel,
    /// Kernel bandwidth `h` (in time units); larger `h` is smoother. Must be
    /// strictly positive.
    pub bandwidth_h: f32,
    /// Time step `dt` separating consecutive grid points of a binary spike
    /// train. Must be strictly positive.
    pub dt: f32,
}

impl Default for KdeRateConfig {
    fn default() -> Self {
        Self {
            kernel: KdeKernel::Gaussian,
            bandwidth_h: 5.0,
            dt: 1.0,
        }
    }
}

impl KdeRateConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::OutOfRange`] when `bandwidth_h ≤ 0` or non-finite, and
    /// [`SnnError::BadDt`] when `dt ≤ 0` or non-finite.
    pub fn validate(&self) -> SnnResult<()> {
        if !self.bandwidth_h.is_finite() || self.bandwidth_h <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "bandwidth_h".into(),
                val: self.bandwidth_h,
            });
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(SnnError::BadDt { dt: self.dt });
        }
        Ok(())
    }
}

/// Estimate the instantaneous firing rate of a single-neuron binary spike train
/// at every one of its `T` grid points.
///
/// The train `spike_train[t]` is interpreted on the time grid `t_k = k · dt`,
/// `k = 0 … T−1`; any non-zero entry counts as a spike at `t_k`. The returned
/// vector has length `T` and gives `r(t_k)` in spikes per unit time. The
/// estimate is normalised so that `Σ_k r(t_k) · dt ≈ N_spikes`.
///
/// An all-zero train yields an all-zero rate.
///
/// # Errors
///
/// Returns [`SnnError::EmptyInput`] when `spike_train` is empty, and propagates
/// configuration errors from [`KdeRateConfig::validate`].
pub fn kde_firing_rate(spike_train: &[f32], cfg: KdeRateConfig) -> SnnResult<Vec<f32>> {
    cfg.validate()?;
    if spike_train.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    let t = spike_train.len();

    // Collect spike times t_s = s · dt for every active bin.
    let mut spike_times: Vec<f32> = Vec::new();
    for (k, &s) in spike_train.iter().enumerate() {
        if s != 0.0 {
            spike_times.push(k as f32 * cfg.dt);
        }
    }

    // Query grid coincides with the train's own time points.
    let query_times: Vec<f32> = (0..t).map(|k| k as f32 * cfg.dt).collect();
    kde_firing_rate_from_times(&spike_times, &query_times, cfg)
}

/// Estimate the firing rate at an arbitrary set of `query_times` from an
/// explicit list of `spike_times`.
///
/// Returns `r(t_q)` for each query time, `r(t) = Σ_s (1/h) · K((t − t_s)/h)`,
/// in spikes per unit time (length `query_times.len()`).
///
/// An empty `spike_times` list produces an all-zero result (no spikes ⇒ zero
/// rate); an empty `query_times` is an [`SnnError::EmptyInput`] error.
///
/// # Errors
///
/// Returns [`SnnError::EmptyInput`] when `query_times` is empty, and propagates
/// configuration errors from [`KdeRateConfig::validate`].
pub fn kde_firing_rate_from_times(
    spike_times: &[f32],
    query_times: &[f32],
    cfg: KdeRateConfig,
) -> SnnResult<Vec<f32>> {
    cfg.validate()?;
    if query_times.is_empty() {
        return Err(SnnError::EmptyInput);
    }
    let h = cfg.bandwidth_h;
    let inv_h = 1.0 / h;
    let mut rate = vec![0.0_f32; query_times.len()];
    for (q, &t) in query_times.iter().enumerate() {
        let mut acc = 0.0_f32;
        for &ts in spike_times {
            let z = (t - ts) * inv_h;
            acc += cfg.kernel.eval(z);
        }
        rate[q] = acc * inv_h;
    }
    Ok(rate)
}

/// Silverman's rule-of-thumb bandwidth for a 1-D KDE over `n_spikes` events that
/// occupy a temporal `span`.
///
/// Silverman (1986) gives `h ≈ 1.06 · σ · n^{−1/5}`. With only the count and the
/// total span available we approximate the spike-time spread by the
/// uniform-distribution standard deviation `σ ≈ span / √12`, yielding
///
/// ```text
/// h = 1.06 · (span / √12) · n^{−1/5}.
/// ```
///
/// For `n_spikes < 2` or `span ≤ 0` a positive fallback bandwidth is returned so
/// that downstream `validate()` never sees a non-positive `h`.
#[must_use]
pub fn optimal_bandwidth_silverman(n_spikes: usize, span: f32) -> f32 {
    const INV_SQRT_12: f32 = 0.288_675_13; // 1/√12
    if n_spikes < 2 || !span.is_finite() || span <= 0.0 {
        // Fallback: a small but strictly-positive bandwidth.
        return if span.is_finite() && span > 0.0 {
            (span * 0.1).max(f32::MIN_POSITIVE)
        } else {
            1.0
        };
    }
    let sigma = span * INV_SQRT_12;
    let n = n_spikes as f32;
    let h = 1.06 * sigma * n.powf(-0.2);
    h.max(f32::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trapezoidal integral of a rate sampled on the train's grid: Σ r·dt.
    fn integral(rate: &[f32], dt: f32) -> f32 {
        rate.iter().sum::<f32>() * dt
    }

    // 1. Config validation: bad h, bad dt.
    #[test]
    fn config_validation() {
        assert!(matches!(
            KdeRateConfig {
                bandwidth_h: 0.0,
                ..KdeRateConfig::default()
            }
            .validate(),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            KdeRateConfig {
                dt: -1.0,
                ..KdeRateConfig::default()
            }
            .validate(),
            Err(SnnError::BadDt { .. })
        ));
    }

    // 2. Kernels are normalised: ∫ K(z) dz ≈ 1 (numeric quadrature).
    #[test]
    fn kernels_integrate_to_one() {
        let dz = 0.001_f32;
        for kernel in [
            KdeKernel::Gaussian,
            KdeKernel::Exponential,
            KdeKernel::Epanechnikov,
            KdeKernel::Boxcar,
        ] {
            let mut area = 0.0_f32;
            let mut z = -20.0_f32;
            while z < 20.0 {
                area += kernel.eval(z) * dz;
                z += dz;
            }
            assert!((area - 1.0).abs() < 1e-2, "{kernel:?} integral={area}");
        }
    }

    // 3. Single spike: rate integrates to ≈ 1 and peaks at the spike time.
    #[test]
    fn single_spike_integrates_to_one_gaussian() {
        // One spike at the centre of a long train so the kernel fits fully.
        let t = 201_usize;
        let mut train = vec![0.0_f32; t];
        train[100] = 1.0;
        let cfg = KdeRateConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth_h: 5.0,
            dt: 1.0,
        };
        let rate = kde_firing_rate(&train, cfg).expect("rate");
        assert_eq!(rate.len(), t);
        let area = integral(&rate, cfg.dt);
        assert!((area - 1.0).abs() < 1e-2, "integral={area}");
        // Peak should be at the spike index.
        let mut peak_idx = 0_usize;
        let mut peak_val = f32::MIN;
        for (i, &r) in rate.iter().enumerate() {
            if r > peak_val {
                peak_val = r;
                peak_idx = i;
            }
        }
        assert_eq!(peak_idx, 100, "Gaussian peak should be at the spike");
        assert!(rate.iter().all(|&r| r >= 0.0), "rate must be non-negative");
    }

    // 4. Total integral ≈ spike count for several spikes.
    #[test]
    fn integral_equals_spike_count() {
        let t = 400_usize;
        let mut train = vec![0.0_f32; t];
        let spike_idxs = [50_usize, 120, 200, 260, 330];
        for &i in &spike_idxs {
            train[i] = 1.0;
        }
        let cfg = KdeRateConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth_h: 6.0,
            dt: 1.0,
        };
        let rate = kde_firing_rate(&train, cfg).expect("rate");
        let area = integral(&rate, cfg.dt);
        assert!(
            (area - spike_idxs.len() as f32).abs() < 0.1,
            "integral={area}, expected {}",
            spike_idxs.len()
        );
    }

    // 5. Gaussian rate is smooth and strictly positive everywhere with a spike.
    #[test]
    fn gaussian_positive_everywhere() {
        let t = 60_usize;
        let mut train = vec![0.0_f32; t];
        train[30] = 1.0;
        let cfg = KdeRateConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth_h: 8.0,
            dt: 1.0,
        };
        let rate = kde_firing_rate(&train, cfg).expect("rate");
        // Gaussian has infinite support → every grid point is > 0.
        assert!(
            rate.iter().all(|&r| r > 0.0),
            "Gaussian rate should be > 0 everywhere"
        );
        // Smoothness: neighbouring differences are small relative to the peak.
        let peak = rate.iter().cloned().fold(f32::MIN, f32::max);
        for w in rate.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.5 * peak, "rate should be smooth");
        }
    }

    // 6. Boxcar reduces to a normalised moving sum (shape + convergent integral).
    #[test]
    fn boxcar_is_moving_average() {
        // Spikes at t=10 and t=11; with h=2 the boxcar covers |t−ts| ≤ 2.
        let t = 40_usize;
        let mut train = vec![0.0_f32; t];
        train[10] = 1.0;
        train[11] = 1.0;
        let cfg = KdeRateConfig {
            kernel: KdeKernel::Boxcar,
            bandwidth_h: 2.0,
            dt: 1.0,
        };
        let rate = kde_firing_rate(&train, cfg).expect("rate");
        // Defining moving-sum behaviour: at t=10 both spikes lie within ±2, each
        // contributing (1/h)·½ = 0.25 → total 0.5 (a normalised local average).
        assert!((rate[10] - 0.5).abs() < 1e-5, "rate[10]={}", rate[10]);
        // Far away (t=30): no spike within ±2h ⇒ exactly 0 (compact support).
        assert!(rate[30].abs() < 1e-6, "rate[30]={}", rate[30]);
        // The estimator is a flat plateau between the two spikes (moving sum).
        assert!((rate[10] - rate[11]).abs() < 1e-6, "plateau between spikes");

        // The integral ∫r dt → spike count as the grid is refined: on a coarse
        // grid the closed-interval boxcar's Riemann sum over-counts its two
        // boundary samples, so verify normalisation on a fine query grid where
        // the sum converges to the true continuous value (= 2 spikes).
        let spike_times = [10.0_f32, 11.0];
        let fine_dt = 0.001_f32;
        let n_pts = ((30.0_f32) / fine_dt) as usize;
        let query: Vec<f32> = (0..n_pts).map(|k| k as f32 * fine_dt).collect();
        let fine = kde_firing_rate_from_times(&spike_times, &query, cfg).expect("rate");
        let area = fine.iter().sum::<f32>() * fine_dt;
        assert!((area - 2.0).abs() < 1e-2, "fine-grid integral={area}");
    }

    // 7. Exponential kernel is causal: only future-of-spike times get rate.
    #[test]
    fn exponential_is_causal() {
        let spike_times = [10.0_f32];
        let query = [5.0_f32, 10.0, 15.0];
        let cfg = KdeRateConfig {
            kernel: KdeKernel::Exponential,
            bandwidth_h: 3.0,
            dt: 1.0,
        };
        let rate = kde_firing_rate_from_times(&spike_times, &query, cfg).expect("rate");
        // Before the spike (t=5 < 10): z=(5−10)/3 < 0 ⇒ kernel 0.
        assert!(
            rate[0].abs() < 1e-6,
            "pre-spike rate should be 0, got {}",
            rate[0]
        );
        // At and after the spike: positive.
        assert!(rate[1] > 0.0, "rate at spike should be > 0");
        assert!(rate[2] > 0.0, "rate after spike should be > 0");
        // Decays: rate(15) < rate(10).
        assert!(
            rate[2] < rate[1],
            "exponential should decay after the spike"
        );
    }

    // 8. Empty spike train → all-zero rate (no spikes, valid config).
    #[test]
    fn empty_train_zero_rate() {
        let train = vec![0.0_f32; 50];
        let cfg = KdeRateConfig::default();
        let rate = kde_firing_rate(&train, cfg).expect("rate");
        assert_eq!(rate.len(), 50);
        assert!(rate.iter().all(|&r| r == 0.0), "no spikes ⇒ zero rate");
    }

    // 9. Truly empty slice → EmptyInput.
    #[test]
    fn empty_slice_errors() {
        let cfg = KdeRateConfig::default();
        assert!(matches!(
            kde_firing_rate(&[], cfg),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            kde_firing_rate_from_times(&[1.0], &[], cfg),
            Err(SnnError::EmptyInput)
        ));
    }

    // 10. No spike times but non-empty queries → zero rate, not an error.
    #[test]
    fn no_spike_times_zero_rate() {
        let cfg = KdeRateConfig::default();
        let rate = kde_firing_rate_from_times(&[], &[0.0, 1.0, 2.0], cfg).expect("rate");
        assert_eq!(rate, vec![0.0, 0.0, 0.0]);
    }

    // 11. Silverman bandwidth is positive and shrinks with more spikes.
    #[test]
    fn silverman_bandwidth_properties() {
        let span = 1000.0_f32;
        let h_few = optimal_bandwidth_silverman(10, span);
        let h_many = optimal_bandwidth_silverman(10_000, span);
        assert!(h_few > 0.0 && h_many > 0.0);
        assert!(
            h_many < h_few,
            "more spikes ⇒ smaller bandwidth: {h_few} vs {h_many}"
        );
        // Degenerate inputs still return a positive, finite bandwidth.
        assert!(optimal_bandwidth_silverman(0, span) > 0.0);
        assert!(optimal_bandwidth_silverman(1, span) > 0.0);
        assert!(optimal_bandwidth_silverman(100, -1.0) > 0.0);
        assert!(optimal_bandwidth_silverman(100, f32::NAN) > 0.0);
        // Sanity: a Silverman-bandwidth config validates.
        let cfg = KdeRateConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth_h: optimal_bandwidth_silverman(100, span),
            dt: 1.0,
        };
        assert!(cfg.validate().is_ok());
    }

    // 12. Larger bandwidth produces a lower, broader peak (same integral).
    #[test]
    fn larger_bandwidth_lower_peak() {
        let t = 401_usize;
        let mut train = vec![0.0_f32; t];
        train[200] = 1.0;
        let mk = |h: f32| KdeRateConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth_h: h,
            dt: 1.0,
        };
        let r_narrow = kde_firing_rate(&train, mk(3.0)).expect("r");
        let r_wide = kde_firing_rate(&train, mk(12.0)).expect("r");
        let peak_narrow = r_narrow.iter().cloned().fold(f32::MIN, f32::max);
        let peak_wide = r_wide.iter().cloned().fold(f32::MIN, f32::max);
        assert!(peak_wide < peak_narrow, "wider kernel ⇒ lower peak");
        // Both integrate to ≈ 1.
        assert!((integral(&r_narrow, 1.0) - 1.0).abs() < 1e-2);
        assert!((integral(&r_wide, 1.0) - 1.0).abs() < 1e-2);
    }
}
