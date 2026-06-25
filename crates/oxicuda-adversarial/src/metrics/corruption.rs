//! Common-corruption robustness evaluation (CIFAR-10-C / ImageNet-C style).
//!
//! Reference: Hendrycks & Dietterich (2019),
//! *"Benchmarking Neural Network Robustness to Common Corruptions and
//! Perturbations"*, ICLR.
//!
//! Adversarial robustness measures worst-case behaviour under a tiny, *crafted*
//! perturbation; **corruption robustness** measures *average* behaviour under
//! naturally-occurring distortions (sensor noise, blur, weather, digital
//! artefacts) applied at several severity levels. The two are complementary
//! and the CIFAR-10-C / ImageNet-C protocol has become the standard corruption
//! benchmark.
//!
//! This module provides two things:
//!
//! 1. A small library of **synthetic corruption operators** that map a clean
//!    `[0, 1]`-normalised input to a corrupted one at an integer severity
//!    `1..=5` (the canonical Hendrycks severities). They are deliberately
//!    light-weight, deterministic-given-RNG reference implementations:
//!    Gaussian / shot / impulse noise, a separable box blur, and brightness /
//!    contrast / saturation-style multiplicative shifts. Each takes the
//!    flattened input plus its `[h, w, c]` shape where spatial structure
//!    matters (blur), and a scalar otherwise.
//!
//! 2. The Hendrycks **aggregate metrics** computed on top of
//!    [`crate::metrics::robust_accuracy::robust_accuracy`]:
//!    * per-`(corruption, severity)` **corruption error** `CE = 1 − accuracy`;
//!    * per-corruption error averaged over severities;
//!    * **unnormalised mean Corruption Error (mCE)** — the mean over all
//!      corruptions of the severity-averaged error;
//!    * **normalised mCE** — each corruption's error divided by a baseline
//!      (e.g. AlexNet) classifier's error on the same corruption, then
//!      averaged (the headline ImageNet-C number);
//!    * **relative mCE** — degradation *relative to clean error*,
//!      `(CE_corrupt − CE_clean) / (CE_baseline_corrupt − CE_baseline_clean)`.
//!
//! The corruption operators draw their randomness from the crate's
//! [`LcgRng`], so an evaluation is fully reproducible from a seed.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── Corruption operators ─────────────────────────────────────────────────────

/// The canonical maximum severity in the Hendrycks protocol.
pub const MAX_SEVERITY: usize = 5;

/// Validate a severity is in `1..=MAX_SEVERITY`.
fn check_severity(severity: usize) -> AdvResult<()> {
    if severity == 0 || severity > MAX_SEVERITY {
        return Err(AdvError::Internal(format!(
            "severity must be in 1..={MAX_SEVERITY}, got {severity}"
        )));
    }
    Ok(())
}

fn check_input(x: &[f32], where_: &'static str) -> AdvResult<()> {
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered { location: where_ });
    }
    Ok(())
}

/// Per-severity standard deviation for additive Gaussian noise (Hendrycks
/// CIFAR-10-C scale, applied to `[0,1]` inputs).
const GAUSSIAN_SIGMA: [f32; MAX_SEVERITY] = [0.04, 0.06, 0.08, 0.09, 0.10];

/// Add zero-mean Gaussian noise of severity-dependent scale, then clip to
/// `[0, 1]`.
///
/// # Errors
/// * [`AdvError::EmptyInput`] / [`AdvError::NanEncountered`] on bad input.
/// * [`AdvError::Internal`] on out-of-range severity.
pub fn gaussian_noise(x: &[f32], severity: usize, rng: &mut LcgRng) -> AdvResult<Vec<f32>> {
    check_input(x, "gaussian_noise:x")?;
    check_severity(severity)?;
    let sigma = GAUSSIAN_SIGMA[severity - 1];
    let mut noise = vec![0.0_f32; x.len()];
    rng.fill_normal(&mut noise);
    Ok(x.iter()
        .zip(noise.iter())
        .map(|(&v, &z)| (v + sigma * z).clamp(0.0, 1.0))
        .collect())
}

/// Per-severity Poisson rate scaler for shot (Poisson) noise.
const SHOT_LAMBDA: [f32; MAX_SEVERITY] = [60.0, 25.0, 12.0, 5.0, 3.0];

/// Shot (Poisson) noise: model each pixel as a Poisson count with rate
/// `λ·pixel`, then renormalise by `λ`. Larger severity ⇒ smaller `λ` ⇒ more
/// relative noise. Poisson sampling uses Knuth's algorithm driven by the
/// crate RNG.
///
/// # Errors
/// As [`gaussian_noise`].
pub fn shot_noise(x: &[f32], severity: usize, rng: &mut LcgRng) -> AdvResult<Vec<f32>> {
    check_input(x, "shot_noise:x")?;
    check_severity(severity)?;
    let lam = SHOT_LAMBDA[severity - 1];
    Ok(x.iter()
        .map(|&v| {
            let rate = (v.clamp(0.0, 1.0) * lam).max(0.0);
            let k = poisson_knuth(rate, rng);
            (k as f32 / lam).clamp(0.0, 1.0)
        })
        .collect())
}

/// Sample a Poisson(`lambda`) variate via Knuth's multiplicative algorithm.
/// Adequate for the small rates used by shot noise (`λ·pixel <= 60`).
fn poisson_knuth(lambda: f32, rng: &mut LcgRng) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0_u32;
    let mut p = 1.0_f32;
    loop {
        p *= rng.next_f32();
        if p <= l {
            return k;
        }
        k += 1;
        if k > 10_000 {
            // Safety cap; unreachable for the rates used here.
            return k;
        }
    }
}

/// Per-severity flip probability for impulse (salt-and-pepper) noise.
const IMPULSE_PROB: [f32; MAX_SEVERITY] = [0.01, 0.02, 0.05, 0.08, 0.12];

/// Impulse (salt-and-pepper) noise: with severity-dependent probability replace
/// a pixel with `0.0` (pepper) or `1.0` (salt), each half the time.
///
/// # Errors
/// As [`gaussian_noise`].
pub fn impulse_noise(x: &[f32], severity: usize, rng: &mut LcgRng) -> AdvResult<Vec<f32>> {
    check_input(x, "impulse_noise:x")?;
    check_severity(severity)?;
    let prob = IMPULSE_PROB[severity - 1];
    Ok(x.iter()
        .map(|&v| {
            if rng.next_f32() < prob {
                if rng.next_f32() < 0.5 { 0.0 } else { 1.0 }
            } else {
                v.clamp(0.0, 1.0)
            }
        })
        .collect())
}

/// Per-severity box-blur radius (in pixels).
const BLUR_RADIUS: [usize; MAX_SEVERITY] = [1, 1, 2, 2, 3];

/// Separable box blur over an `[h, w, c]` image (channels-last, row-major).
///
/// The kernel is a `(2r+1)×(2r+1)` averaging window applied first along width
/// then along height (separable), with edge replication at the borders. This
/// is the cheapest stand-in for the Gaussian / defocus blur in ImageNet-C and
/// is the only spatial corruption here, hence the explicit shape.
///
/// # Errors
/// * [`AdvError::EmptyInput`] / [`AdvError::NanEncountered`] on bad input.
/// * [`AdvError::Internal`] on out-of-range severity.
/// * [`AdvError::DimensionMismatch`] if `h*w*c != x.len()` or any dim is 0.
pub fn box_blur(x: &[f32], h: usize, w: usize, c: usize, severity: usize) -> AdvResult<Vec<f32>> {
    check_input(x, "box_blur:x")?;
    check_severity(severity)?;
    if h == 0 || w == 0 || c == 0 || h * w * c != x.len() {
        return Err(AdvError::DimensionMismatch {
            expected: h * w * c,
            got: x.len(),
        });
    }
    let r = BLUR_RADIUS[severity - 1] as isize;
    let idx = |row: usize, col: usize, ch: usize| (row * w + col) * c + ch;

    // Horizontal pass.
    let mut tmp = vec![0.0_f32; x.len()];
    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0_f32;
                let mut cnt = 0.0_f32;
                for dc in -r..=r {
                    let cc = (col as isize + dc).clamp(0, w as isize - 1) as usize;
                    acc += x[idx(row, cc, ch)];
                    cnt += 1.0;
                }
                tmp[idx(row, col, ch)] = acc / cnt;
            }
        }
    }
    // Vertical pass.
    let mut out = vec![0.0_f32; x.len()];
    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let mut acc = 0.0_f32;
                let mut cnt = 0.0_f32;
                for dr in -r..=r {
                    let rr = (row as isize + dr).clamp(0, h as isize - 1) as usize;
                    acc += tmp[idx(rr, col, ch)];
                    cnt += 1.0;
                }
                out[idx(row, col, ch)] = (acc / cnt).clamp(0.0, 1.0);
            }
        }
    }
    Ok(out)
}

/// Per-severity additive brightness shift.
const BRIGHTNESS_SHIFT: [f32; MAX_SEVERITY] = [0.05, 0.10, 0.15, 0.20, 0.30];

/// Brightness corruption: add a constant offset and clip to `[0, 1]`.
///
/// # Errors
/// As [`gaussian_noise`] (minus RNG; deterministic).
pub fn brightness(x: &[f32], severity: usize) -> AdvResult<Vec<f32>> {
    check_input(x, "brightness:x")?;
    check_severity(severity)?;
    let s = BRIGHTNESS_SHIFT[severity - 1];
    Ok(x.iter().map(|&v| (v + s).clamp(0.0, 1.0)).collect())
}

/// Per-severity contrast multiplier (`< 1` reduces contrast around 0.5).
const CONTRAST_FACTOR: [f32; MAX_SEVERITY] = [0.75, 0.60, 0.50, 0.40, 0.30];

/// Contrast corruption: scale each pixel toward the mid-grey point `0.5` by the
/// severity-dependent factor, then clip.
///
/// # Errors
/// As [`brightness`].
pub fn contrast(x: &[f32], severity: usize) -> AdvResult<Vec<f32>> {
    check_input(x, "contrast:x")?;
    check_severity(severity)?;
    let f = CONTRAST_FACTOR[severity - 1];
    Ok(x.iter()
        .map(|&v| (0.5 + (v - 0.5) * f).clamp(0.0, 1.0))
        .collect())
}

/// Standard library of corruptions in canonical order. Spatial ones (blur)
/// require shape and are excluded; see [`box_blur`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corruption {
    /// Additive Gaussian noise.
    GaussianNoise,
    /// Poisson (shot) noise.
    ShotNoise,
    /// Salt-and-pepper (impulse) noise.
    ImpulseNoise,
    /// Brightness shift.
    Brightness,
    /// Contrast reduction.
    Contrast,
}

impl Corruption {
    /// Apply this (non-spatial) corruption to `x` at `severity`.
    ///
    /// # Errors
    /// As the underlying operator.
    pub fn apply(self, x: &[f32], severity: usize, rng: &mut LcgRng) -> AdvResult<Vec<f32>> {
        match self {
            Corruption::GaussianNoise => gaussian_noise(x, severity, rng),
            Corruption::ShotNoise => shot_noise(x, severity, rng),
            Corruption::ImpulseNoise => impulse_noise(x, severity, rng),
            Corruption::Brightness => brightness(x, severity),
            Corruption::Contrast => contrast(x, severity),
        }
    }

    /// Stable human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Corruption::GaussianNoise => "gaussian_noise",
            Corruption::ShotNoise => "shot_noise",
            Corruption::ImpulseNoise => "impulse_noise",
            Corruption::Brightness => "brightness",
            Corruption::Contrast => "contrast",
        }
    }
}

// ─── Aggregate metrics ────────────────────────────────────────────────────────

/// Corruption error (`1 − accuracy`) for one corruption across severities.
#[derive(Debug, Clone, PartialEq)]
pub struct CorruptionErrors {
    /// Human-readable corruption name.
    pub name: String,
    /// `errors[s]` is the error at severity `s+1` (length = number of
    /// severities reported).
    pub per_severity: Vec<f32>,
    /// Unweighted mean of `per_severity` (the per-corruption error used by mCE).
    pub avg_error: f32,
}

impl CorruptionErrors {
    /// Build from `(prediction, label)` slices per severity.
    ///
    /// `preds_per_severity[s]` and `labels_per_severity[s]` are the predictions
    /// / labels for that severity. All severities must be non-empty and
    /// length-matched.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`] — no severities, or any empty severity slice.
    /// * [`AdvError::DimensionMismatch`] — pred/label length mismatch within a
    ///   severity, or `preds`/`labels` outer-length mismatch.
    pub fn from_predictions(
        name: impl Into<String>,
        preds_per_severity: &[Vec<usize>],
        labels_per_severity: &[Vec<usize>],
    ) -> AdvResult<Self> {
        if preds_per_severity.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if preds_per_severity.len() != labels_per_severity.len() {
            return Err(AdvError::DimensionMismatch {
                expected: labels_per_severity.len(),
                got: preds_per_severity.len(),
            });
        }
        let mut per_severity = Vec::with_capacity(preds_per_severity.len());
        for (p, y) in preds_per_severity.iter().zip(labels_per_severity.iter()) {
            let acc = crate::metrics::robust_accuracy::robust_accuracy(p, y)?;
            per_severity.push(1.0 - acc);
        }
        let avg_error = per_severity.iter().sum::<f32>() / per_severity.len() as f32;
        Ok(Self {
            name: name.into(),
            per_severity,
            avg_error,
        })
    }
}

/// Aggregate corruption-robustness summary across a suite of corruptions.
#[derive(Debug, Clone, PartialEq)]
pub struct CorruptionSummary {
    /// Per-corruption error breakdown.
    pub per_corruption: Vec<CorruptionErrors>,
    /// Unweighted mean over corruptions of their `avg_error` — the
    /// **unnormalised** mean Corruption Error.
    pub mce_unnormalized: f32,
}

impl CorruptionSummary {
    /// Build a summary from a list of per-corruption error breakdowns.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`] — empty corruption list.
    pub fn new(per_corruption: Vec<CorruptionErrors>) -> AdvResult<Self> {
        if per_corruption.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        let mce_unnormalized =
            per_corruption.iter().map(|c| c.avg_error).sum::<f32>() / per_corruption.len() as f32;
        Ok(Self {
            per_corruption,
            mce_unnormalized,
        })
    }

    /// Hendrycks **normalised mCE**: each corruption's severity-averaged error
    /// is divided by a baseline classifier's error on the *same* corruption,
    /// then the ratios are averaged. `baseline_avg_errors[i]` corresponds to
    /// `per_corruption[i]`.
    ///
    /// ```text
    /// mCE = (1/N) Σ_c  E_c / E_c^{baseline}.
    /// ```
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — length mismatch with `per_corruption`.
    /// * [`AdvError::InvalidLossWeight`] — a baseline error is `<= 0` or
    ///   non-finite (division would be undefined).
    pub fn normalized_mce(&self, baseline_avg_errors: &[f32]) -> AdvResult<f32> {
        if baseline_avg_errors.len() != self.per_corruption.len() {
            return Err(AdvError::DimensionMismatch {
                expected: self.per_corruption.len(),
                got: baseline_avg_errors.len(),
            });
        }
        let mut sum = 0.0_f32;
        for (c, &b) in self.per_corruption.iter().zip(baseline_avg_errors.iter()) {
            if !(b.is_finite() && b > 0.0) {
                return Err(AdvError::InvalidLossWeight { weight: b });
            }
            sum += c.avg_error / b;
        }
        Ok(sum / self.per_corruption.len() as f32)
    }

    /// Hendrycks **relative mCE**: measures degradation relative to clean
    /// accuracy, normalised by the baseline's degradation.
    ///
    /// ```text
    /// rel_mCE = (1/N) Σ_c  (E_c − E_clean) / (E_c^{base} − E_clean^{base}).
    /// ```
    ///
    /// `clean_error` is this model's error on uncorrupted data;
    /// `baseline_clean_error` and `baseline_avg_errors` are the baseline's.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — length mismatch with `per_corruption`.
    /// * [`AdvError::InvalidLossWeight`] — a baseline denominator
    ///   `E_c^{base} − E_clean^{base}` is `<= 0` or non-finite.
    pub fn relative_mce(
        &self,
        clean_error: f32,
        baseline_clean_error: f32,
        baseline_avg_errors: &[f32],
    ) -> AdvResult<f32> {
        if baseline_avg_errors.len() != self.per_corruption.len() {
            return Err(AdvError::DimensionMismatch {
                expected: self.per_corruption.len(),
                got: baseline_avg_errors.len(),
            });
        }
        if !(clean_error.is_finite() && baseline_clean_error.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "relative_mce:clean_error",
            });
        }
        let mut sum = 0.0_f32;
        for (c, &b) in self.per_corruption.iter().zip(baseline_avg_errors.iter()) {
            let denom = b - baseline_clean_error;
            if !(denom.is_finite() && denom > 0.0) {
                return Err(AdvError::InvalidLossWeight { weight: denom });
            }
            sum += (c.avg_error - clean_error) / denom;
        }
        Ok(sum / self.per_corruption.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) / (n as f32 - 1.0)).collect()
    }

    #[test]
    fn severity_and_input_validation() {
        let mut rng = LcgRng::new(0);
        let x = vec![0.5_f32; 4];
        assert!(matches!(
            gaussian_noise(&x, 0, &mut rng).unwrap_err(),
            AdvError::Internal(_)
        ));
        assert!(matches!(
            gaussian_noise(&x, 6, &mut rng).unwrap_err(),
            AdvError::Internal(_)
        ));
        assert_eq!(
            gaussian_noise(&[], 1, &mut rng).unwrap_err(),
            AdvError::EmptyInput
        );
        assert!(matches!(
            gaussian_noise(&[f32::NAN], 1, &mut rng).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn corruptions_stay_in_unit_box_and_change_input() {
        let mut rng = LcgRng::new(7);
        let x = ramp(64);
        for sev in 1..=MAX_SEVERITY {
            for corr in [
                Corruption::GaussianNoise,
                Corruption::ShotNoise,
                Corruption::ImpulseNoise,
                Corruption::Brightness,
                Corruption::Contrast,
            ] {
                let y = corr.apply(&x, sev, &mut rng).expect("apply");
                assert_eq!(y.len(), x.len());
                assert!(
                    y.iter().all(|&v| (0.0..=1.0).contains(&v)),
                    "{} out of box",
                    corr.name()
                );
            }
        }
    }

    #[test]
    fn gaussian_noise_severity_monotone_in_perturbation() {
        // Higher severity ⇒ larger expected L2 distance from clean input.
        let x = vec![0.5_f32; 256];
        let mut rng = LcgRng::new(3);
        let y1 = gaussian_noise(&x, 1, &mut rng).expect("s1");
        let mut rng2 = LcgRng::new(3);
        let y5 = gaussian_noise(&x, 5, &mut rng2).expect("s5");
        let d1: f32 = x
            .iter()
            .zip(y1.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let d5: f32 = x
            .iter()
            .zip(y5.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(
            d5 > d1,
            "severity 5 distance {d5} should exceed severity 1 {d1}"
        );
    }

    #[test]
    fn brightness_and_contrast_are_deterministic() {
        let x = ramp(16);
        let b1 = brightness(&x, 3).expect("b1");
        let b2 = brightness(&x, 3).expect("b2");
        assert_eq!(b1, b2);
        // Contrast at 0.5 leaves the mid-grey point fixed.
        let mid = vec![0.5_f32; 8];
        let c = contrast(&mid, 4).expect("c");
        assert!(c.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn box_blur_preserves_constant_image() {
        // A constant image is a fixed point of any averaging blur.
        let h = 4;
        let w = 4;
        let c = 3;
        let x = vec![0.3_f32; h * w * c];
        let y = box_blur(&x, h, w, c, 5).expect("blur");
        assert!(y.iter().all(|&v| (v - 0.3).abs() < 1e-6));
    }

    #[test]
    fn box_blur_reduces_variance() {
        // Blurring a high-frequency checkerboard reduces its variance.
        let h = 8;
        let w = 8;
        let c = 1;
        let mut x = vec![0.0_f32; h * w * c];
        for row in 0..h {
            for col in 0..w {
                x[row * w + col] = if (row + col) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var_in = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>();
        let y = box_blur(&x, h, w, c, 3).expect("blur");
        let mean_y = y.iter().sum::<f32>() / y.len() as f32;
        let var_out = y.iter().map(|&v| (v - mean_y) * (v - mean_y)).sum::<f32>();
        assert!(
            var_out < var_in,
            "blur did not reduce variance: {var_out} >= {var_in}"
        );
    }

    #[test]
    fn box_blur_shape_check() {
        let x = vec![0.5_f32; 12];
        assert!(matches!(
            box_blur(&x, 3, 3, 1, 1).unwrap_err(), // 9 != 12
            AdvError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            box_blur(&x, 0, 4, 3, 1).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn corruption_errors_from_predictions() {
        // Three severities; accuracy 1.0, 0.5, 0.0 ⇒ errors 0, 0.5, 1.0.
        let labels = vec![0_usize, 1, 2, 3];
        let preds = [
            vec![0_usize, 1, 2, 3], // all correct ⇒ err 0
            vec![0_usize, 1, 9, 9], // half correct ⇒ err 0.5
            vec![9_usize, 9, 9, 9], // all wrong ⇒ err 1.0
        ];
        let labels_per = vec![labels.clone(), labels.clone(), labels.clone()];
        let ce =
            CorruptionErrors::from_predictions("gaussian_noise", &preds, &labels_per).expect("ce");
        assert_eq!(ce.per_severity.len(), 3);
        assert!((ce.per_severity[0] - 0.0).abs() < 1e-6);
        assert!((ce.per_severity[1] - 0.5).abs() < 1e-6);
        assert!((ce.per_severity[2] - 1.0).abs() < 1e-6);
        assert!((ce.avg_error - 0.5).abs() < 1e-6);
    }

    #[test]
    fn corruption_errors_rejects_bad_shape() {
        let labels = vec![0_usize, 1];
        let preds: Vec<Vec<usize>> = vec![vec![0, 1]];
        // Outer-length mismatch (1 pred severity vs 2 label severities).
        let bad =
            CorruptionErrors::from_predictions("x", &preds, &[labels.clone(), labels.clone()]);
        assert!(matches!(
            bad.unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
        // No severities.
        assert_eq!(
            CorruptionErrors::from_predictions("x", &[], &[]).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn summary_unnormalized_mce() {
        let labels = vec![0_usize, 1, 2, 3];
        let labels_per = vec![labels.clone(), labels.clone()];
        // Corruption A: errors 0.0 and 0.5 ⇒ avg 0.25.
        let a = CorruptionErrors::from_predictions(
            "a",
            &[vec![0, 1, 2, 3], vec![0, 1, 9, 9]],
            &labels_per,
        )
        .expect("a");
        // Corruption B: errors 0.5 and 1.0 ⇒ avg 0.75.
        let b = CorruptionErrors::from_predictions(
            "b",
            &[vec![0, 1, 9, 9], vec![9, 9, 9, 9]],
            &labels_per,
        )
        .expect("b");
        let summary = CorruptionSummary::new(vec![a, b]).expect("summary");
        // mCE = mean(0.25, 0.75) = 0.5.
        assert!((summary.mce_unnormalized - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalized_and_relative_mce() {
        let labels = vec![0_usize, 1, 2, 3];
        let labels_per = vec![labels.clone(), labels.clone()];
        let a = CorruptionErrors::from_predictions(
            "a",
            &[vec![0, 1, 2, 3], vec![0, 1, 9, 9]],
            &labels_per,
        )
        .expect("a"); // avg 0.25
        let b = CorruptionErrors::from_predictions(
            "b",
            &[vec![0, 1, 9, 9], vec![9, 9, 9, 9]],
            &labels_per,
        )
        .expect("b"); // avg 0.75
        let summary = CorruptionSummary::new(vec![a, b]).expect("summary");
        // Baseline errors: A=0.5, B=1.0 ⇒ normalised mCE = mean(0.25/0.5, 0.75/1.0)
        //                = mean(0.5, 0.75) = 0.625.
        let mce = summary.normalized_mce(&[0.5, 1.0]).expect("mce");
        assert!((mce - 0.625).abs() < 1e-6);

        // Relative mCE with clean_error=0.1, baseline_clean=0.2:
        //   A: (0.25 − 0.1)/(0.5 − 0.2) = 0.15/0.3 = 0.5
        //   B: (0.75 − 0.1)/(1.0 − 0.2) = 0.65/0.8 = 0.8125
        //   mean = 0.65625.
        let rel = summary.relative_mce(0.1, 0.2, &[0.5, 1.0]).expect("rel");
        assert!((rel - 0.656_25).abs() < 1e-5);
    }

    #[test]
    fn mce_rejects_bad_baseline() {
        let labels = vec![0_usize, 1];
        let a = CorruptionErrors::from_predictions("a", &[vec![0, 1]], &[labels]).expect("a");
        let summary = CorruptionSummary::new(vec![a]).expect("summary");
        // Wrong length.
        assert!(matches!(
            summary.normalized_mce(&[0.5, 0.5]).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
        // Zero baseline ⇒ division undefined.
        assert!(matches!(
            summary.normalized_mce(&[0.0]).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
        // Relative mCE: zero denominator (baseline_clean == baseline_err).
        assert!(matches!(
            summary.relative_mce(0.1, 0.5, &[0.5]).unwrap_err(),
            AdvError::InvalidLossWeight { .. }
        ));
    }

    #[test]
    fn summary_rejects_empty() {
        assert_eq!(
            CorruptionSummary::new(vec![]).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn impulse_noise_at_high_severity_flips_some_pixels() {
        // With a large flip probability some pixels become exactly 0 or 1.
        let x = vec![0.5_f32; 4096];
        let mut rng = LcgRng::new(11);
        let y = impulse_noise(&x, 5, &mut rng).expect("impulse");
        let flipped = y.iter().filter(|&&v| v == 0.0 || v == 1.0).count();
        assert!(flipped > 0, "expected some salt/pepper flips");
        // Roughly within an order of magnitude of the nominal probability.
        let frac = flipped as f32 / x.len() as f32;
        assert!(
            frac > 0.04 && frac < 0.30,
            "flip fraction {frac} implausible"
        );
    }
}
