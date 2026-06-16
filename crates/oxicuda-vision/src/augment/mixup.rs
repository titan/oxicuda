//! MixUp and CutMix regularising augmentations for image classification.
//!
//! Both operate on a *batch* of CHW images and their one-hot (or soft) label
//! vectors, producing a convex / spatial mixture of two samples together with
//! the correspondingly mixed targets. They are among the strongest
//! regularisers for modern image classifiers.
//!
//! * **MixUp** (Zhang et al. 2018): pixel-wise convex combination
//!   `x̃ = λ·x_i + (1−λ)·x_j`, label `ỹ = λ·y_i + (1−λ)·y_j`.
//! * **CutMix** (Yun et al. 2019): a rectangular patch of `x_j` is pasted into
//!   `x_i`; the label mixing coefficient is the *area fraction* of the surviving
//!   region of `x_i`, `λ = 1 − (patch_area / image_area)`.
//!
//! Each image is paired with another sample drawn from the same batch via a
//! random permutation (a sample may be paired with itself, in which case the
//! mixture is the identity — exactly as in the reference implementations).
//!
//! The mixing coefficient `λ` is drawn from a `Beta(α, α)` distribution. Beta
//! sampling is performed without external crates via two Gamma(α) draws
//! (Marsaglia–Tsang for α ≥ 1, with the Johnk boost for α < 1) using the
//! crate-local [`LcgRng`].

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

/// Output of a batch mix operation.
#[derive(Debug, Clone)]
pub struct MixOutput {
    /// Mixed images, shape `[batch × channels × h × w]` (flat row-major).
    pub images: Vec<f32>,
    /// Mixed labels, shape `[batch × n_classes]` (flat row-major).
    pub labels: Vec<f32>,
    /// Per-sample label mixing coefficient `λ`, length `batch`.
    pub lambdas: Vec<f32>,
    /// Per-sample partner index (into the original batch), length `batch`.
    pub partners: Vec<usize>,
}

// ─── Beta / Gamma sampling ──────────────────────────────────────────────────

/// Sample `Gamma(shape, 1)` using Marsaglia–Tsang (shape ≥ 1) with the
/// Johnk boosting transform for `shape < 1`.
fn sample_gamma(shape: f32, rng: &mut LcgRng) -> f32 {
    if shape < 1.0 {
        // Boost: Gamma(a) = Gamma(a+1) · U^{1/a}.
        let g = sample_gamma(shape + 1.0, rng);
        let u = rng.next_f32().max(1e-12);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal via Box–Muller.
        let (z, _) = rng.next_normal_pair();
        let v0 = 1.0 + c * z;
        if v0 <= 0.0 {
            continue;
        }
        let v = v0 * v0 * v0;
        let u = rng.next_f32().max(1e-12);
        if u < 1.0 - 0.0331 * z * z * z * z {
            return d * v;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample `λ ~ Beta(alpha, alpha)`. For `alpha <= 0` returns `1.0` (no mixing,
/// matching the common convention where α=0 disables MixUp).
fn sample_beta_symmetric(alpha: f32, rng: &mut LcgRng) -> f32 {
    if !alpha.is_finite() || alpha <= 0.0 {
        return 1.0;
    }
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(alpha, rng);
    let s = x + y;
    if s <= 1e-12 { 0.5 } else { x / s }
}

#[inline]
fn validate_batch(
    images: &[f32],
    labels: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    n_classes: usize,
) -> VisionResult<()> {
    if batch == 0 {
        return Err(VisionError::EmptyInput("mixup batch"));
    }
    if channels == 0 || h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels,
        });
    }
    if n_classes == 0 {
        return Err(VisionError::InvalidNumClasses(n_classes));
    }
    let img_expected = batch * channels * h * w;
    if images.len() != img_expected {
        return Err(VisionError::DimensionMismatch {
            expected: img_expected,
            got: images.len(),
        });
    }
    let lbl_expected = batch * n_classes;
    if labels.len() != lbl_expected {
        return Err(VisionError::DimensionMismatch {
            expected: lbl_expected,
            got: labels.len(),
        });
    }
    Ok(())
}

/// Build a random partner permutation (a derangement is *not* required; a
/// sample may pair with itself, exactly as in the reference code).
fn random_partners(batch: usize, rng: &mut LcgRng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..batch).collect();
    rng.shuffle(&mut perm);
    perm
}

/// Mix `labels[i]` and `labels[j]` as `λ·y_i + (1−λ)·y_j` into `out[i]`.
fn mix_labels_into(
    out: &mut [f32],
    labels: &[f32],
    i: usize,
    j: usize,
    n_classes: usize,
    lambda: f32,
) {
    let oi = i * n_classes;
    let li = i * n_classes;
    let lj = j * n_classes;
    for c in 0..n_classes {
        out[oi + c] = lambda * labels[li + c] + (1.0 - lambda) * labels[lj + c];
    }
}

/// Apply **MixUp** to a batch.
///
/// # Errors
/// Returns [`VisionError`] for empty / mismatched inputs.
pub fn mixup(
    images: &[f32],
    labels: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    n_classes: usize,
    alpha: f32,
    rng: &mut LcgRng,
) -> VisionResult<MixOutput> {
    validate_batch(images, labels, batch, channels, h, w, n_classes)?;
    let chw = channels * h * w;
    let partners = random_partners(batch, rng);

    let mut out_images = vec![0.0_f32; images.len()];
    let mut out_labels = vec![0.0_f32; labels.len()];
    let mut lambdas = vec![0.0_f32; batch];

    for i in 0..batch {
        let j = partners[i];
        let lambda = sample_beta_symmetric(alpha, rng);
        lambdas[i] = lambda;
        let bi = i * chw;
        let bj = j * chw;
        for p in 0..chw {
            out_images[bi + p] = lambda * images[bi + p] + (1.0 - lambda) * images[bj + p];
        }
        mix_labels_into(&mut out_labels, labels, i, j, n_classes, lambda);
    }

    Ok(MixOutput {
        images: out_images,
        labels: out_labels,
        lambdas,
        partners,
    })
}

/// Sample a CutMix bounding box for a `λ` area ratio.
///
/// The patch has side `√(1−λ)` of the image; its centre is uniform. Returns
/// `(x1, y1, x2, y2)` clamped to the image bounds, plus the *corrected* area
/// fraction actually pasted (the box may be clipped at the border).
fn cutmix_bbox(h: usize, w: usize, lambda: f32, rng: &mut LcgRng) -> (usize, usize, usize, usize) {
    let cut_ratio = (1.0 - lambda).max(0.0).sqrt();
    let cut_h = ((h as f32) * cut_ratio).round() as usize;
    let cut_w = ((w as f32) * cut_ratio).round() as usize;
    let cy = rng.next_usize(h);
    let cx = rng.next_usize(w);
    let y1 = cy.saturating_sub(cut_h / 2);
    let x1 = cx.saturating_sub(cut_w / 2);
    let y2 = (cy + cut_h.div_ceil(2)).min(h);
    let x2 = (cx + cut_w.div_ceil(2)).min(w);
    (x1, y1, x2, y2)
}

/// Apply **CutMix** to a batch.
///
/// A rectangular patch from the partner image is pasted over each image; the
/// label coefficient `λ` is corrected to the *true* surviving area fraction
/// `1 − (patch_area / image_area)` after border clipping, as in the reference.
///
/// # Errors
/// Returns [`VisionError`] for empty / mismatched inputs.
pub fn cutmix(
    images: &[f32],
    labels: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    n_classes: usize,
    alpha: f32,
    rng: &mut LcgRng,
) -> VisionResult<MixOutput> {
    validate_batch(images, labels, batch, channels, h, w, n_classes)?;
    let chw = channels * h * w;
    let partners = random_partners(batch, rng);
    let area = (h * w) as f32;

    let mut out_images = images.to_vec();
    let mut out_labels = vec![0.0_f32; labels.len()];
    let mut lambdas = vec![0.0_f32; batch];

    for i in 0..batch {
        let j = partners[i];
        let lambda0 = sample_beta_symmetric(alpha, rng);
        let (x1, y1, x2, y2) = cutmix_bbox(h, w, lambda0, rng);
        let patch_area = ((x2 - x1) * (y2 - y1)) as f32;
        // True coefficient after border clipping.
        let lambda = 1.0 - patch_area / area;
        lambdas[i] = lambda;

        let bi = i * chw;
        let bj = j * chw;
        for c in 0..channels {
            let ci = bi + c * h * w;
            let cj = bj + c * h * w;
            for y in y1..y2 {
                for x in x1..x2 {
                    out_images[ci + y * w + x] = images[cj + y * w + x];
                }
            }
        }
        mix_labels_into(&mut out_labels, labels, i, j, n_classes, lambda);
    }

    Ok(MixOutput {
        images: out_images,
        labels: out_labels,
        lambdas,
        partners,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot_batch(batch: usize, n_classes: usize) -> Vec<f32> {
        let mut labels = vec![0.0_f32; batch * n_classes];
        for i in 0..batch {
            labels[i * n_classes + (i % n_classes)] = 1.0;
        }
        labels
    }

    #[test]
    fn beta_symmetric_in_unit_interval() {
        let mut rng = LcgRng::new(1);
        for _ in 0..1000 {
            let l = sample_beta_symmetric(0.4, &mut rng);
            assert!((0.0..=1.0).contains(&l), "beta sample out of [0,1]: {l}");
        }
    }

    #[test]
    fn beta_alpha_nonpositive_is_one() {
        let mut rng = LcgRng::new(2);
        assert_eq!(sample_beta_symmetric(0.0, &mut rng), 1.0);
        assert_eq!(sample_beta_symmetric(-1.0, &mut rng), 1.0);
    }

    #[test]
    fn gamma_samples_positive() {
        let mut rng = LcgRng::new(3);
        for a in [0.3_f32, 1.0, 2.5, 5.0] {
            for _ in 0..200 {
                let g = sample_gamma(a, &mut rng);
                assert!(g > 0.0 && g.is_finite(), "gamma({a})={g}");
            }
        }
    }

    #[test]
    fn mixup_output_shapes() {
        let batch = 4;
        let (c, h, w, k) = (3, 8, 8, 5);
        let images = vec![0.5_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(4);
        let out = mixup(&images, &labels, batch, c, h, w, k, 0.4, &mut rng).expect("ok");
        assert_eq!(out.images.len(), batch * c * h * w);
        assert_eq!(out.labels.len(), batch * k);
        assert_eq!(out.lambdas.len(), batch);
        assert_eq!(out.partners.len(), batch);
    }

    #[test]
    fn mixup_labels_sum_preserved() {
        // One-hot labels each sum to 1 → any convex mix also sums to 1.
        let batch = 6;
        let (c, h, w, k) = (1, 4, 4, 4);
        let images = vec![0.3_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(5);
        let out = mixup(&images, &labels, batch, c, h, w, k, 0.5, &mut rng).expect("ok");
        for i in 0..batch {
            let s: f32 = out.labels[i * k..(i + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} label sum {s} != 1");
        }
    }

    #[test]
    fn mixup_constant_images_value_preserved() {
        // Both images constant 0.5 → any λ leaves the value 0.5.
        let batch = 3;
        let (c, h, w, k) = (3, 4, 4, 3);
        let images = vec![0.5_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(6);
        let out = mixup(&images, &labels, batch, c, h, w, k, 0.4, &mut rng).expect("ok");
        assert!(out.images.iter().all(|&v| (v - 0.5).abs() < 1e-5));
    }

    #[test]
    fn mixup_output_finite() {
        let batch = 4;
        let (c, h, w, k) = (3, 8, 8, 10);
        let mut rng = LcgRng::new(7);
        let mut images = vec![0.0_f32; batch * c * h * w];
        rng.fill_normal(&mut images);
        let labels = one_hot_batch(batch, k);
        let out = mixup(&images, &labels, batch, c, h, w, k, 0.2, &mut rng).expect("ok");
        assert!(out.images.iter().all(|v| v.is_finite()));
        assert!(out.labels.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mixup_deterministic_with_seed() {
        let batch = 5;
        let (c, h, w, k) = (3, 8, 8, 4);
        let images = vec![0.4_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        let o1 = mixup(&images, &labels, batch, c, h, w, k, 0.5, &mut r1).expect("ok");
        let o2 = mixup(&images, &labels, batch, c, h, w, k, 0.5, &mut r2).expect("ok");
        assert_eq!(o1.partners, o2.partners);
        assert_eq!(o1.lambdas, o2.lambdas);
        assert_eq!(o1.images, o2.images);
    }

    #[test]
    fn mixup_empty_batch_errors() {
        let mut rng = LcgRng::new(8);
        let r = mixup(&[], &[], 0, 3, 8, 8, 5, 0.4, &mut rng);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn mixup_label_size_mismatch_errors() {
        let batch = 4;
        let images = vec![0.5_f32; batch * 3 * 8 * 8];
        let labels = vec![0.0_f32; batch * 4]; // claim k=5 below
        let mut rng = LcgRng::new(9);
        let r = mixup(&images, &labels, batch, 3, 8, 8, 5, 0.4, &mut rng);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn cutmix_output_shapes() {
        let batch = 4;
        let (c, h, w, k) = (3, 16, 16, 5);
        let images = vec![0.5_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(10);
        let out = cutmix(&images, &labels, batch, c, h, w, k, 1.0, &mut rng).expect("ok");
        assert_eq!(out.images.len(), batch * c * h * w);
        assert_eq!(out.labels.len(), batch * k);
        assert_eq!(out.lambdas.len(), batch);
    }

    #[test]
    fn cutmix_labels_sum_to_one() {
        let batch = 6;
        let (c, h, w, k) = (3, 16, 16, 4);
        let images = vec![0.5_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(11);
        let out = cutmix(&images, &labels, batch, c, h, w, k, 1.0, &mut rng).expect("ok");
        for i in 0..batch {
            let s: f32 = out.labels[i * k..(i + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sum {s}");
        }
    }

    #[test]
    fn cutmix_lambda_matches_area() {
        // Build distinguishable images: sample 0 all 0.0, others all 1.0; pair
        // forcing is not possible, but we can at least check the pasted area is
        // consistent with the reported λ for every sample.
        let batch = 4;
        let (c, h, w, k) = (1, 16, 16, 4);
        let images: Vec<f32> = (0..batch).flat_map(|i| vec![i as f32; c * h * w]).collect();
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(12);
        let out = cutmix(&images, &labels, batch, c, h, w, k, 1.0, &mut rng).expect("ok");
        let area = (h * w) as f32;
        for i in 0..batch {
            let j = out.partners[i];
            let vi = i as f32;
            let vj = j as f32;
            if (vi - vj).abs() < 1e-6 {
                continue; // self-paste — undetectable
            }
            // Count pixels that changed to the partner's constant value.
            let base = i * c * h * w;
            let changed = (0..h * w)
                .filter(|&p| (out.images[base + p] - vj).abs() < 1e-5)
                .count() as f32;
            let observed_lambda = 1.0 - changed / area;
            assert!(
                (observed_lambda - out.lambdas[i]).abs() < 1e-4,
                "sample {i}: observed λ {observed_lambda} vs reported {}",
                out.lambdas[i]
            );
        }
    }

    #[test]
    fn cutmix_lambda_in_unit_range() {
        let batch = 5;
        let (c, h, w, k) = (3, 16, 16, 4);
        let images = vec![0.5_f32; batch * c * h * w];
        let labels = one_hot_batch(batch, k);
        let mut rng = LcgRng::new(13);
        let out = cutmix(&images, &labels, batch, c, h, w, k, 0.5, &mut rng).expect("ok");
        for &l in &out.lambdas {
            assert!((0.0..=1.0).contains(&l), "cutmix λ out of range: {l}");
        }
    }

    #[test]
    fn cutmix_self_paste_identity_when_partner_equal() {
        // Single-sample batch: the only partner is itself → output == input.
        let batch = 1;
        let (c, h, w, k) = (3, 16, 16, 3);
        let mut rng = LcgRng::new(14);
        let mut images = vec![0.0_f32; batch * c * h * w];
        rng.fill_normal(&mut images);
        let labels = one_hot_batch(batch, k);
        let out = cutmix(&images, &labels, batch, c, h, w, k, 1.0, &mut rng).expect("ok");
        assert_eq!(out.images, images, "self-paste must be identity");
    }

    #[test]
    fn cutmix_output_finite() {
        let batch = 4;
        let (c, h, w, k) = (3, 16, 16, 10);
        let mut rng = LcgRng::new(15);
        let mut images = vec![0.0_f32; batch * c * h * w];
        rng.fill_normal(&mut images);
        let labels = one_hot_batch(batch, k);
        let out = cutmix(&images, &labels, batch, c, h, w, k, 0.3, &mut rng).expect("ok");
        assert!(out.images.iter().all(|v| v.is_finite()));
        assert!(out.labels.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cutmix_bbox_within_bounds() {
        let mut rng = LcgRng::new(16);
        for _ in 0..200 {
            let (x1, y1, x2, y2) = cutmix_bbox(16, 16, 0.3, &mut rng);
            assert!(x1 <= x2 && y1 <= y2);
            assert!(x2 <= 16 && y2 <= 16);
        }
    }

    #[test]
    fn cutmix_empty_errors() {
        let mut rng = LcgRng::new(17);
        let r = cutmix(&[], &[], 0, 3, 8, 8, 5, 0.4, &mut rng);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }
}
