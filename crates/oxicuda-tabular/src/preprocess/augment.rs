//! CutMix and Mixup data augmentation primitives for tabular learning.
//!
//! # Mixup (Zhang et al. 2018)
//!
//! Interpolates two samples `(x_i, y_i)` and `(x_j, y_j)` with a mixing
//! coefficient `λ ~ Beta(α, α)`:
//!
//! ```text
//! x̃ = λ·x_i + (1−λ)·x_j
//! ỹ = λ·y_i + (1−λ)·y_j   (soft target)
//! ```
//!
//! # CutMix (tabular variant, Yun et al. 2019 adapted)
//!
//! Unlike the image version (which cuts a rectangular patch), the tabular
//! adaptation draws a random subset of feature indices (size proportional to
//! `1 − λ`) and replaces them with the corresponding values from a second
//! sample.  Labels are mixed by the fraction `λ` of *retained* features.
//!
//! Both augmentations operate on row-major `[n_samples × n_features]` arrays
//! and return new `(features, soft_labels)` pairs.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Beta-distribution sampler ─────────────────────────────────────────────────

/// Sample from `Beta(α, α)` via the Johnk method (valid for `α > 0`).
///
/// Uses the LCG RNG to generate exponential variates through the log
/// transformation `X = -ln(U)`, then normalises.
fn beta_symmetric(alpha: f32, rng: &mut LcgRng) -> f32 {
    if (alpha - 1.0).abs() < 1e-6 {
        // Degenerate β(1,1) = Uniform(0,1).
        return rng.next_f32();
    }
    // Johnk's method: X ~ Gamma(α,1), Y ~ Gamma(α,1), return X/(X+Y).
    // The result is always in [0, 1] since both Gamma samples are non-negative.
    let x = gamma_sample(alpha, rng);
    let y = gamma_sample(alpha, rng);
    let sum = x + y;
    if sum < 1e-30 {
        return 0.5;
    }
    (x / sum).clamp(0.0, 1.0)
}

/// Sample `Gamma(α, 1)` via the Marsaglia-Tsang squeeze (α ≥ 1).
/// For `α < 1` we use the Ahrens-Dieter reduction: `Gamma(α) = Gamma(α+1) · U^(1/α)`.
fn gamma_sample(alpha: f32, rng: &mut LcgRng) -> f32 {
    if alpha <= 0.0 {
        return 0.0;
    }
    if alpha < 1.0 {
        // Ahrens-Dieter: Gamma(α) = Gamma(α+1) · U^{1/α}
        let u = rng.next_f32().max(1e-7);
        return gamma_sample(alpha + 1.0, rng) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (z, _) = rng.next_normal_pair();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.next_f32().max(1e-7);
        if u < 1.0 - 0.0331 * (z * z) * (z * z) {
            return d * v3;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

// ─── Mixup ────────────────────────────────────────────────────────────────────

/// Configuration for Mixup augmentation.
#[derive(Debug, Clone, Copy)]
pub struct MixupConfig {
    /// Concentration parameter of the symmetric `Beta(α, α)` distribution.
    /// Typical range: 0.1–2.0.  Use `α = 1.0` for uniform mixing.
    pub alpha: f32,
}

impl Default for MixupConfig {
    fn default() -> Self {
        Self { alpha: 0.4 }
    }
}

/// Apply Mixup to a single pair of samples and their soft labels.
///
/// Both `x_a` and `x_b` must have length `n_features`.
/// `y_a` and `y_b` must have the same length (1 for regression, K for
/// classification with K-dimensional one-hot / soft targets).
///
/// Returns `(x_mix, y_mix)`.
///
/// # Errors
/// - [`TabularError::DimensionMismatch`] if shapes disagree.
/// - [`TabularError::InvalidParameter`] if `alpha ≤ 0`.
pub fn mixup_pair(
    x_a: &[f32],
    y_a: &[f32],
    x_b: &[f32],
    y_b: &[f32],
    cfg: MixupConfig,
    rng: &mut LcgRng,
) -> TabularResult<(Vec<f32>, Vec<f32>)> {
    if x_a.len() != x_b.len() {
        return Err(TabularError::DimensionMismatch {
            expected: x_a.len(),
            got: x_b.len(),
        });
    }
    if y_a.len() != y_b.len() {
        return Err(TabularError::DimensionMismatch {
            expected: y_a.len(),
            got: y_b.len(),
        });
    }
    if cfg.alpha <= 0.0 {
        return Err(TabularError::InvalidParameter {
            name: "alpha".into(),
            msg: "must be > 0".into(),
        });
    }
    let lambda = beta_symmetric(cfg.alpha, rng);
    let x_mix: Vec<f32> = x_a
        .iter()
        .zip(x_b.iter())
        .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
        .collect();
    let y_mix: Vec<f32> = y_a
        .iter()
        .zip(y_b.iter())
        .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
        .collect();
    Ok((x_mix, y_mix))
}

/// Apply Mixup to an entire batch by pairing each sample `i` with sample
/// `(i + 1) % n_samples` (sequential pairing).
///
/// `data` is a `[n_samples × n_features]` row-major matrix.
/// `labels` is a `[n_samples × n_label_dim]` row-major matrix.
///
/// Returns `(x_mix, y_mix)` with the same shapes.
///
/// # Errors
/// - [`TabularError::DimensionMismatch`] if sizes are inconsistent.
/// - [`TabularError::EmptyInput`] if `n_samples == 0`.
pub fn mixup_batch(
    data: &[f32],
    labels: &[f32],
    n_samples: usize,
    n_features: usize,
    n_label_dim: usize,
    cfg: MixupConfig,
    rng: &mut LcgRng,
) -> TabularResult<(Vec<f32>, Vec<f32>)> {
    if n_samples == 0 {
        return Err(TabularError::EmptyInput);
    }
    if data.len() != n_samples * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_features,
            got: data.len(),
        });
    }
    if labels.len() != n_samples * n_label_dim {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_label_dim,
            got: labels.len(),
        });
    }
    let mut x_out = vec![0.0_f32; n_samples * n_features];
    let mut y_out = vec![0.0_f32; n_samples * n_label_dim];
    for i in 0..n_samples {
        let j = (i + 1) % n_samples;
        let x_a = &data[i * n_features..(i + 1) * n_features];
        let x_b = &data[j * n_features..(j + 1) * n_features];
        let y_a = &labels[i * n_label_dim..(i + 1) * n_label_dim];
        let y_b = &labels[j * n_label_dim..(j + 1) * n_label_dim];
        let (x_mix, y_mix) = mixup_pair(x_a, y_a, x_b, y_b, cfg, rng)?;
        x_out[i * n_features..(i + 1) * n_features].copy_from_slice(&x_mix);
        y_out[i * n_label_dim..(i + 1) * n_label_dim].copy_from_slice(&y_mix);
    }
    Ok((x_out, y_out))
}

// ─── CutMix (tabular) ─────────────────────────────────────────────────────────

/// Configuration for tabular CutMix augmentation.
#[derive(Debug, Clone, Copy)]
pub struct CutMixConfig {
    /// Concentration parameter for `Beta(α, α)` mixing ratio.
    pub alpha: f32,
}

impl Default for CutMixConfig {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

/// Apply tabular CutMix to a single pair of samples and their soft labels.
///
/// A fraction `(1 − λ)` of feature indices are randomly replaced with values
/// from `x_b`.  The resulting label is `λ · y_a + (1 − λ) · y_b`.
///
/// Returns `(x_cut, y_cut)`.
///
/// # Errors
/// - [`TabularError::DimensionMismatch`] if shapes disagree.
/// - [`TabularError::InvalidParameter`] if `alpha ≤ 0`.
pub fn cutmix_pair(
    x_a: &[f32],
    y_a: &[f32],
    x_b: &[f32],
    y_b: &[f32],
    cfg: CutMixConfig,
    rng: &mut LcgRng,
) -> TabularResult<(Vec<f32>, Vec<f32>)> {
    if x_a.len() != x_b.len() {
        return Err(TabularError::DimensionMismatch {
            expected: x_a.len(),
            got: x_b.len(),
        });
    }
    if y_a.len() != y_b.len() {
        return Err(TabularError::DimensionMismatch {
            expected: y_a.len(),
            got: y_b.len(),
        });
    }
    if cfg.alpha <= 0.0 {
        return Err(TabularError::InvalidParameter {
            name: "alpha".into(),
            msg: "must be > 0".into(),
        });
    }
    let n_feat = x_a.len();
    let lambda = beta_symmetric(cfg.alpha, rng);
    // Number of features to keep from x_a.
    let n_keep = ((lambda * n_feat as f32).round() as usize).clamp(0, n_feat);

    // Build a shuffled index set and keep the first n_keep from x_a.
    let mut indices: Vec<usize> = (0..n_feat).collect();
    // Fisher-Yates shuffle using LCG.
    for k in (1..n_feat).rev() {
        let j = (rng.next_u32() as usize) % (k + 1);
        indices.swap(k, j);
    }
    let mut x_cut = vec![0.0_f32; n_feat];
    for (pos, &idx) in indices.iter().enumerate() {
        x_cut[idx] = if pos < n_keep { x_a[idx] } else { x_b[idx] };
    }
    // Label mixing is proportional to the fraction of retained features.
    let actual_lambda = n_keep as f32 / n_feat.max(1) as f32;
    let y_cut: Vec<f32> = y_a
        .iter()
        .zip(y_b.iter())
        .map(|(&a, &b)| actual_lambda * a + (1.0 - actual_lambda) * b)
        .collect();
    Ok((x_cut, y_cut))
}

/// Apply tabular CutMix to an entire batch using sequential pairing.
///
/// `data` is a `[n_samples × n_features]` row-major matrix.
/// `labels` is a `[n_samples × n_label_dim]` row-major matrix.
///
/// # Errors
/// Propagates errors from [`cutmix_pair`].
pub fn cutmix_batch(
    data: &[f32],
    labels: &[f32],
    n_samples: usize,
    n_features: usize,
    n_label_dim: usize,
    cfg: CutMixConfig,
    rng: &mut LcgRng,
) -> TabularResult<(Vec<f32>, Vec<f32>)> {
    if n_samples == 0 {
        return Err(TabularError::EmptyInput);
    }
    if data.len() != n_samples * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_features,
            got: data.len(),
        });
    }
    if labels.len() != n_samples * n_label_dim {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_label_dim,
            got: labels.len(),
        });
    }
    let mut x_out = vec![0.0_f32; n_samples * n_features];
    let mut y_out = vec![0.0_f32; n_samples * n_label_dim];
    for i in 0..n_samples {
        let j = (i + 1) % n_samples;
        let x_a = &data[i * n_features..(i + 1) * n_features];
        let x_b = &data[j * n_features..(j + 1) * n_features];
        let y_a = &labels[i * n_label_dim..(i + 1) * n_label_dim];
        let y_b = &labels[j * n_label_dim..(j + 1) * n_label_dim];
        let (x_cut, y_cut) = cutmix_pair(x_a, y_a, x_b, y_b, cfg, rng)?;
        x_out[i * n_features..(i + 1) * n_features].copy_from_slice(&x_cut);
        y_out[i * n_label_dim..(i + 1) * n_label_dim].copy_from_slice(&y_cut);
    }
    Ok((x_out, y_out))
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── 1. Mixup output shape matches inputs ──────────────────────────────────
    #[test]
    fn mixup_pair_shape() {
        let mut rng = make_rng();
        let x_a = vec![1.0_f32; 8];
        let x_b = vec![0.0_f32; 8];
        let y_a = vec![1.0_f32, 0.0];
        let y_b = vec![0.0_f32, 1.0];
        let (xm, ym) = mixup_pair(&x_a, &y_a, &x_b, &y_b, MixupConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(xm.len(), 8);
        assert_eq!(ym.len(), 2);
    }

    // ── 2. Mixup: mixed x is convex combination ───────────────────────────────
    #[test]
    fn mixup_convex_combination() {
        let mut rng = make_rng();
        let x_a = vec![2.0_f32; 4];
        let x_b = vec![0.0_f32; 4];
        let y_a = vec![1.0_f32];
        let y_b = vec![0.0_f32];
        for _ in 0..100 {
            let (xm, _) = mixup_pair(&x_a, &y_a, &x_b, &y_b, MixupConfig { alpha: 0.4 }, &mut rng)
                .expect("mixup_pair should succeed");
            for &v in &xm {
                assert!((0.0..=2.0).contains(&v), "v={v}");
            }
        }
    }

    // ── 3. Mixup soft label sums to 1 for one-hot inputs ─────────────────────
    #[test]
    fn mixup_soft_label_sum() {
        let mut rng = make_rng();
        let y_a = vec![1.0_f32, 0.0, 0.0];
        let y_b = vec![0.0_f32, 1.0, 0.0];
        let x_a = vec![0.0_f32; 4];
        let x_b = vec![1.0_f32; 4];
        for _ in 0..50 {
            let (_, ym) = mixup_pair(&x_a, &y_a, &x_b, &y_b, MixupConfig { alpha: 0.5 }, &mut rng)
                .expect("mixup_pair should succeed");
            let s: f32 = ym.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "sum={s}");
        }
    }

    // ── 4. Mixup batch preserves shape ────────────────────────────────────────
    #[test]
    fn mixup_batch_shape() {
        let mut rng = make_rng();
        let n_s = 8_usize;
        let n_f = 4_usize;
        let n_l = 3_usize;
        let data = vec![0.5_f32; n_s * n_f];
        let labels = vec![1.0_f32 / 3.0; n_s * n_l];
        let (xm, ym) = mixup_batch(
            &data,
            &labels,
            n_s,
            n_f,
            n_l,
            MixupConfig::default(),
            &mut rng,
        )
        .expect("value should be present");
        assert_eq!(xm.len(), n_s * n_f);
        assert_eq!(ym.len(), n_s * n_l);
    }

    // ── 5. CutMix pair: feature values come from x_a or x_b ─────────────────
    #[test]
    fn cutmix_pair_values_from_inputs() {
        let mut rng = make_rng();
        let x_a = vec![1.0_f32; 8];
        let x_b = vec![0.0_f32; 8];
        let y_a = vec![1.0_f32, 0.0];
        let y_b = vec![0.0_f32, 1.0];
        for _ in 0..50 {
            let (xc, _) = cutmix_pair(&x_a, &y_a, &x_b, &y_b, CutMixConfig::default(), &mut rng)
                .expect("value should be present");
            // Each feature must be exactly 0 or 1 (from x_a or x_b).
            for &v in &xc {
                assert!(v == 0.0 || v == 1.0, "v={v}");
            }
        }
    }

    // ── 6. CutMix label is in [0, 1] ─────────────────────────────────────────
    #[test]
    fn cutmix_label_range() {
        let mut rng = make_rng();
        let x_a = vec![1.0_f32; 6];
        let x_b = vec![0.0_f32; 6];
        let y_a = vec![1.0_f32, 0.0];
        let y_b = vec![0.0_f32, 1.0];
        for _ in 0..100 {
            let (_, yc) = cutmix_pair(&x_a, &y_a, &x_b, &y_b, CutMixConfig::default(), &mut rng)
                .expect("value should be present");
            for &v in &yc {
                assert!((0.0..=1.0).contains(&v), "v={v}");
            }
        }
    }

    // ── 7. CutMix batch shape ─────────────────────────────────────────────────
    #[test]
    fn cutmix_batch_shape() {
        let mut rng = make_rng();
        let n_s = 10_usize;
        let n_f = 5_usize;
        let n_l = 2_usize;
        let data: Vec<f32> = (0..n_s * n_f).map(|i| i as f32).collect();
        let labels = vec![0.5_f32; n_s * n_l];
        let (xc, yc) = cutmix_batch(
            &data,
            &labels,
            n_s,
            n_f,
            n_l,
            CutMixConfig::default(),
            &mut rng,
        )
        .expect("value should be present");
        assert_eq!(xc.len(), n_s * n_f);
        assert_eq!(yc.len(), n_s * n_l);
    }

    // ── 8. Dimension mismatch returns error ────────────────────────────────────
    #[test]
    fn dimension_mismatch_error() {
        let mut rng = make_rng();
        let x_a = vec![1.0_f32; 4];
        let x_b = vec![0.0_f32; 5]; // wrong length
        let y_a = vec![1.0_f32];
        let y_b = vec![0.0_f32];
        assert!(mixup_pair(&x_a, &y_a, &x_b, &y_b, MixupConfig::default(), &mut rng).is_err());
        assert!(cutmix_pair(&x_a, &y_a, &x_b, &y_b, CutMixConfig::default(), &mut rng).is_err());
    }

    // ── 9. Invalid alpha returns error ─────────────────────────────────────────
    #[test]
    fn invalid_alpha_error() {
        let mut rng = make_rng();
        let x = vec![0.0_f32; 4];
        let y = vec![1.0_f32];
        assert!(mixup_pair(&x, &y, &x, &y, MixupConfig { alpha: -0.1 }, &mut rng).is_err());
        assert!(cutmix_pair(&x, &y, &x, &y, CutMixConfig { alpha: 0.0 }, &mut rng).is_err());
    }

    // ── 10. Beta samples are in [0, 1] ────────────────────────────────────────
    #[test]
    fn beta_samples_in_range() {
        let mut rng = make_rng();
        for _ in 0..1000 {
            let b = beta_symmetric(0.5, &mut rng);
            assert!((0.0..=1.0).contains(&b), "b={b}");
        }
    }

    // ── 11. Mixup with alpha=1 (uniform) produces lambda in [0,1] via labels ─
    #[test]
    fn mixup_uniform_lambda() {
        let mut rng = make_rng();
        let x_a = vec![0.0_f32; 1];
        let x_b = vec![1.0_f32; 1];
        let y_a = vec![0.0_f32];
        let y_b = vec![1.0_f32];
        for _ in 0..200 {
            let (xm, ym) = mixup_pair(&x_a, &y_a, &x_b, &y_b, MixupConfig { alpha: 1.0 }, &mut rng)
                .expect("mixup_pair should succeed");
            assert!((0.0..=1.0).contains(&xm[0]));
            assert!((0.0..=1.0).contains(&ym[0]));
        }
    }
}
