#![allow(clippy::needless_range_loop)]
//! SLAYER — Spike LAYer Error Reassignment (Shrestha & Orchard, NeurIPS 2018).
//!
//! SLAYER converts spike trains into smooth signals via convolution with the
//! exponentially-decaying post-synaptic-potential (PSP) kernel
//!
//! ```text
//! ε(t) = (t / τ_s) · exp(1 − t / τ_s)   for t > 0,
//! ε(t) = 0                              for t ≤ 0,
//! ```
//!
//! which peaks at `t = τ_s` with value `1`. The filtered output `y[n,t] = Σ_τ
//! ε(τ)·s[n,t−τ]` is then compared to a similarly filtered target via a simple
//! mean-squared error: `L = ½ · Σ (y − ŷ)²`.

use crate::error::{SnnError, SnnResult};

/// SLAYER PSP-kernel hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct SlayerConfig {
    /// Synaptic time constant `τ_s`; controls peak location of `ε(t)`.
    pub tau_s: f32,
    /// Discretisation time step `dt`.
    pub dt: f32,
}

impl Default for SlayerConfig {
    fn default() -> Self {
        Self {
            tau_s: 2.0,
            dt: 1.0,
        }
    }
}

/// Evaluate the SLAYER ε-kernel at continuous time `t`.
#[must_use]
pub fn epsilon_psp(t: f32, cfg: &SlayerConfig) -> f32 {
    if t > 0.0 && cfg.tau_s > 0.0 {
        let r = t / cfg.tau_s;
        r * (1.0 - r).exp()
    } else {
        0.0
    }
}

/// Compute a discrete kernel of length ⌈5·τ_s/dt⌉ pre-evaluated at sample points.
fn build_kernel(cfg: &SlayerConfig) -> SnnResult<Vec<f32>> {
    if cfg.tau_s <= 0.0 || !cfg.tau_s.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "tau_s".into(),
            val: cfg.tau_s,
        });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    // 5τ_s covers ~99% of the kernel's mass.
    let len_f = (5.0 * cfg.tau_s / cfg.dt).ceil();
    let len = (len_f as usize).max(1);
    let mut k = Vec::with_capacity(len);
    for i in 0..len {
        let t = (i as f32) * cfg.dt;
        k.push(epsilon_psp(t, cfg));
    }
    Ok(k)
}

/// Validate inputs to [`convolve_psp`].
fn validate_convolve(spikes: &[f32], n_neurons: usize, t_steps: usize) -> SnnResult<()> {
    if n_neurons == 0 {
        return Err(SnnError::BadDim { got: n_neurons });
    }
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    let expected = n_neurons * t_steps;
    if spikes.len() != expected {
        return Err(SnnError::BadShape {
            expected,
            got: spikes.len(),
        });
    }
    Ok(())
}

/// Filter a spike train with the SLAYER PSP kernel, layout `[n_neurons × t_steps]` row-major.
///
/// ```text
/// y[n, t] = Σ_{τ=0}^{kernel_len-1} ε(τ) · s[n, t − τ]
/// ```
///
/// where out-of-range indices are treated as zero (causal convolution with no
/// pre-padding).
pub fn convolve_psp(
    spikes: &[f32],
    n_neurons: usize,
    t_steps: usize,
    cfg: &SlayerConfig,
) -> SnnResult<Vec<f32>> {
    validate_convolve(spikes, n_neurons, t_steps)?;
    let kernel = build_kernel(cfg)?;
    let kl = kernel.len();
    let mut out = vec![0.0_f32; n_neurons * t_steps];
    for n in 0..n_neurons {
        let row_off = n * t_steps;
        for t in 0..t_steps {
            let upper = (t + 1).min(kl);
            let mut acc = 0.0_f32;
            for tau in 0..upper {
                acc += kernel[tau] * spikes[row_off + (t - tau)];
            }
            out[row_off + t] = acc;
        }
    }
    Ok(out)
}

/// SLAYER MSE loss `L = ½ · Σ (y − ŷ)²`.
pub fn slayer_loss(filtered_s: &[f32], filtered_target: &[f32]) -> SnnResult<f32> {
    if filtered_s.len() != filtered_target.len() {
        return Err(SnnError::IncompatibleLength {
            a: filtered_s.len(),
            b: filtered_target.len(),
        });
    }
    let mut acc = 0.0_f32;
    for (&y, &t) in filtered_s.iter().zip(filtered_target.iter()) {
        let d = y - t;
        acc += d * d;
    }
    Ok(0.5 * acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_at_zero_is_zero() {
        let cfg = SlayerConfig {
            tau_s: 2.0,
            dt: 1.0,
        };
        assert!(epsilon_psp(0.0, &cfg).abs() < 1e-9);
        assert!(epsilon_psp(-1.0, &cfg).abs() < 1e-9);
    }

    #[test]
    fn epsilon_peaks_at_tau_s_equals_one() {
        let cfg = SlayerConfig {
            tau_s: 3.5,
            dt: 1.0,
        };
        let v = epsilon_psp(cfg.tau_s, &cfg);
        assert!((v - 1.0).abs() < 1e-6, "ε(τ_s) = {v}, expected 1.0");
    }

    #[test]
    fn epsilon_monotone_after_peak() {
        let cfg = SlayerConfig {
            tau_s: 2.0,
            dt: 1.0,
        };
        // Past the peak the kernel must decrease monotonically.
        let mut prev = epsilon_psp(cfg.tau_s, &cfg);
        for k in 1..50 {
            let t = cfg.tau_s + (k as f32) * 0.1;
            let v = epsilon_psp(t, &cfg);
            assert!(v <= prev + 1e-6, "v={v} > prev={prev} at t={t}");
            prev = v;
        }
    }

    #[test]
    fn epsilon_increasing_before_peak() {
        let cfg = SlayerConfig {
            tau_s: 2.0,
            dt: 1.0,
        };
        let mut prev = 0.0_f32;
        // From t=0+ up to the peak: must increase monotonically.
        for k in 1..20 {
            let t = (k as f32) * 0.1; // 0.1, 0.2, ..., 1.9 (< τ_s = 2.0)
            let v = epsilon_psp(t, &cfg);
            assert!(v >= prev - 1e-6, "v={v} < prev={prev} at t={t}");
            prev = v;
        }
    }

    #[test]
    fn convolve_preserves_shape() {
        let cfg = SlayerConfig {
            tau_s: 2.0,
            dt: 1.0,
        };
        let n = 3_usize;
        let t = 8_usize;
        let spikes = vec![0.0_f32; n * t];
        let out = convolve_psp(&spikes, n, t, &cfg).expect("ok");
        assert_eq!(out.len(), n * t);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn convolve_impulse_response_matches_kernel() {
        let cfg = SlayerConfig {
            tau_s: 2.0,
            dt: 1.0,
        };
        let n = 1_usize;
        // Stay strictly within the truncated kernel support so the impulse
        // response equals the discrete kernel sample-by-sample.
        let kernel_len = (5.0 * cfg.tau_s / cfg.dt).ceil() as usize;
        let t = kernel_len;
        let mut spikes = vec![0.0_f32; n * t];
        spikes[0] = 1.0; // impulse at t=0
        let out = convolve_psp(&spikes, n, t, &cfg).expect("ok");
        for k in 0..t {
            let expected = epsilon_psp((k as f32) * cfg.dt, &cfg);
            assert!(
                (out[k] - expected).abs() < 1e-5,
                "out[{k}]={}, expected={expected}",
                out[k]
            );
        }
    }

    #[test]
    fn loss_non_negative_and_zero_when_equal() {
        let y = vec![0.1_f32, 0.2, 0.3];
        let t = vec![0.1_f32, 0.2, 0.3];
        let l = slayer_loss(&y, &t).expect("ok");
        assert!(l.abs() < 1e-9);
        let t2 = vec![0.0_f32, 0.0, 0.0];
        let l2 = slayer_loss(&y, &t2).expect("ok");
        assert!(l2 > 0.0);
    }

    #[test]
    fn loss_rejects_length_mismatch() {
        let y = vec![0.0_f32; 3];
        let t = vec![0.0_f32; 4];
        assert!(matches!(
            slayer_loss(&y, &t),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn convolve_rejects_bad_shape() {
        let cfg = SlayerConfig::default();
        let spikes = vec![0.0_f32; 5];
        let err = convolve_psp(&spikes, 2, 3, &cfg);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn rejects_bad_tau_s() {
        let cfg = SlayerConfig {
            tau_s: -1.0,
            dt: 1.0,
        };
        let spikes = vec![0.0_f32; 4];
        let err = convolve_psp(&spikes, 2, 2, &cfg);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
