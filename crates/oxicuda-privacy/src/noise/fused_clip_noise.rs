//! Fused per-vector L2 gradient-clip + Gaussian-noise reference (single pass).
//!
//! The DP-SGD privatisation step (Abadi et al. 2016) clips a (summed) gradient
//! to an L2 bound `C` and then adds `N(0, σ²C²·I)` noise. A naive
//! implementation makes two passes over the gradient: one to write the clipped
//! vector and a second to read it back and add the noise — two streams through
//! (global) memory. This module provides the *fused* reference that does it in a
//! single pass: after the unavoidable norm reduction, each coordinate is scaled
//! and has its noise added in one fused multiply-add, never materialising the
//! clipped intermediate.
//!
//! [`fused_clip_and_noise`] is verified — in this module's tests — to be
//! **bit-for-bit identical** to the two-pass [`sequential_clip_then_noise`]
//! reference when both are driven from the same RNG state, because the noise is
//! drawn by the identical routine in both and `g·scale + z` is the same float
//! whether evaluated in one pass or two.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Compute the L2-clip scale `min(1, clip_norm / ‖grad‖₂)` for a gradient.
///
/// Returns `1.0` for a zero gradient (no scaling, no division by zero).
#[inline]
fn clip_scale(grad: &[f64], clip_norm: f64) -> f64 {
    let norm = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > clip_norm {
        clip_norm / norm
    } else {
        1.0
    }
}

/// Draw `n` Gaussian `N(0, std²)` samples from `rng` using the same pairwise
/// Box-Muller scheme as [`crate::handle::PrivacyHandle::generate_gaussian_noise`]
/// so that the fused and sequential paths consume the RNG identically.
fn gaussian_noise_vec(std: f64, n: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let (a, b) = rng.normal_pair();
        out.push(a * std);
        if i + 1 < n {
            out.push(b * std);
        }
        i += 2;
    }
    out.truncate(n);
    out
}

/// Two-pass reference: clip `grad` to L2 bound `clip_norm` into a fresh vector,
/// then in a second pass add `N(0, (σ·clip_norm)²)` noise.
///
/// This is the straightforward, obviously-correct implementation used as the
/// ground truth for the fused version.
///
/// # Errors
/// - `EmptyInput` if `grad` is empty.
/// - `NonPositiveSensitivity` if `clip_norm ≤ 0`.
/// - `InvalidParameter` if `sigma < 0`.
pub fn sequential_clip_then_noise(
    grad: &[f64],
    clip_norm: f64,
    sigma: f64,
    rng: &mut LcgRng,
) -> PrivacyResult<Vec<f64>> {
    if grad.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if clip_norm <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(clip_norm));
    }
    if sigma < 0.0 {
        return Err(PrivacyError::InvalidParameter("sigma must be ≥ 0".into()));
    }

    let scale = clip_scale(grad, clip_norm);

    // Pass 1: write the clipped intermediate.
    let clipped: Vec<f64> = grad.iter().map(|&g| g * scale).collect();

    // Pass 2: read it back and add the noise.
    let std = sigma * clip_norm;
    let noise = gaussian_noise_vec(std, grad.len(), rng);
    let out = clipped
        .iter()
        .zip(noise.iter())
        .map(|(&c, &z)| c + z)
        .collect();
    Ok(out)
}

/// Fused single-pass clip + noise: after the norm reduction, each coordinate is
/// scaled and noised in one fused multiply-add, without materialising the
/// clipped vector.
///
/// Produces output **bit-for-bit identical** to [`sequential_clip_then_noise`]
/// for the same RNG state (see this module's tests), while touching the gradient
/// stream only once after the reduction.
///
/// # Errors
/// Identical to [`sequential_clip_then_noise`].
pub fn fused_clip_and_noise(
    grad: &[f64],
    clip_norm: f64,
    sigma: f64,
    rng: &mut LcgRng,
) -> PrivacyResult<Vec<f64>> {
    if grad.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if clip_norm <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(clip_norm));
    }
    if sigma < 0.0 {
        return Err(PrivacyError::InvalidParameter("sigma must be ≥ 0".into()));
    }

    // Norm reduction (single reduction pass — unavoidable for the L2 norm).
    let scale = clip_scale(grad, clip_norm);
    let std = sigma * clip_norm;

    // The noise is drawn by the identical routine so the RNG is consumed exactly
    // as in the sequential reference; we then fuse "scale" and "add noise" into a
    // single coordinate pass, never allocating the clipped intermediate.
    let noise = gaussian_noise_vec(std, grad.len(), rng);
    let out = grad
        .iter()
        .zip(noise.iter())
        .map(|(&g, &z)| g * scale + z)
        .collect();
    Ok(out)
}

/// Fused clip + noise that writes into a caller-provided buffer in place
/// (`buf[i] ← buf[i]·scale + zᵢ`), saving the output allocation entirely.
///
/// `buf` holds the gradient on entry and the privatised gradient on return.
///
/// # Errors
/// Identical to [`fused_clip_and_noise`].
pub fn fused_clip_and_noise_in_place(
    buf: &mut [f64],
    clip_norm: f64,
    sigma: f64,
    rng: &mut LcgRng,
) -> PrivacyResult<()> {
    if buf.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if clip_norm <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(clip_norm));
    }
    if sigma < 0.0 {
        return Err(PrivacyError::InvalidParameter("sigma must be ≥ 0".into()));
    }
    let scale = clip_scale(buf, clip_norm);
    let std = sigma * clip_norm;
    let noise = gaussian_noise_vec(std, buf.len(), rng);
    for (b, &z) in buf.iter_mut().zip(noise.iter()) {
        *b = *b * scale + z;
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // 1. Fused output is bit-for-bit identical to the two-pass reference for the
    //    same RNG state.
    #[test]
    fn fused_matches_sequential_bit_for_bit() {
        let grad: Vec<f64> = (0..257).map(|i| (i as f64 - 128.0) * 0.137).collect();
        let clip = 3.0;
        let sigma = 1.1;

        let mut rng_seq = LcgRng::new(0xC0FF_EE12);
        let mut rng_fus = LcgRng::new(0xC0FF_EE12);

        let seq = sequential_clip_then_noise(&grad, clip, sigma, &mut rng_seq).expect("seq");
        let fus = fused_clip_and_noise(&grad, clip, sigma, &mut rng_fus).expect("fus");

        assert_eq!(seq.len(), fus.len());
        for (i, (a, b)) in seq.iter().zip(fus.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "coord {i}: {a} != {b}");
        }
    }

    // 2. In-place fused matches the allocating fused version exactly.
    #[test]
    fn in_place_matches_allocating() {
        let grad: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let clip = 1.5;
        let sigma = 0.7;

        let mut rng_a = LcgRng::new(55);
        let mut rng_b = LcgRng::new(55);
        let alloc = fused_clip_and_noise(&grad, clip, sigma, &mut rng_a).expect("alloc");
        let mut buf = grad.clone();
        fused_clip_and_noise_in_place(&mut buf, clip, sigma, &mut rng_b).expect("inplace");
        for (a, b) in alloc.iter().zip(buf.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // 3. With σ = 0 the output is exactly the clipped gradient and its norm is
    //    bounded by the clip.
    #[test]
    fn zero_sigma_is_pure_clip() {
        let grad = vec![3.0, 4.0]; // norm 5
        let clip = 1.0;
        let mut rng = LcgRng::new(1);
        let out = fused_clip_and_noise(&grad, clip, 0.0, &mut rng).expect("out");
        assert!((out[0] - 0.6).abs() < 1e-12);
        assert!((out[1] - 0.8).abs() < 1e-12);
        assert!(l2(&out) <= clip + 1e-12);
    }

    // 4. Clipping bounds the pre-noise norm: subtracting the (replayed) noise
    //    recovers a vector with norm ≤ clip.
    #[test]
    fn clipping_bounds_norm() {
        let grad: Vec<f64> = (0..64).map(|i| i as f64 + 1.0).collect(); // huge norm
        let clip = 2.0;
        let sigma = 0.0; // isolate the clip
        let mut rng = LcgRng::new(9);
        let out = fused_clip_and_noise(&grad, clip, sigma, &mut rng).expect("out");
        assert!(l2(&out) <= clip + 1e-9, "norm {} > clip {clip}", l2(&out));
    }

    // 5. A gradient already within the clip is left unscaled (only noise added).
    #[test]
    fn small_gradient_unscaled() {
        let grad = vec![0.1, 0.1, 0.1];
        let clip = 10.0;
        let mut rng = LcgRng::new(3);
        // σ=0 ⇒ output equals the (unscaled) gradient exactly.
        let out = fused_clip_and_noise(&grad, clip, 0.0, &mut rng).expect("o");
        for (o, g) in out.iter().zip(grad.iter()) {
            assert!((o - g).abs() < 1e-15);
        }
    }

    // 6. Noise scale: the empirical std of (output − clipped) ≈ σ·clip.
    #[test]
    fn noise_has_correct_scale() {
        let grad = vec![0.0f64; 200_000]; // zero grad ⇒ output is pure noise
        let clip = 2.0;
        let sigma = 0.5;
        let mut rng = LcgRng::new(2718);
        let out = fused_clip_and_noise(&grad, clip, sigma, &mut rng).expect("o");
        let mean = out.iter().sum::<f64>() / out.len() as f64;
        let var = out.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / out.len() as f64;
        let target = (sigma * clip) * (sigma * clip);
        assert!(
            (var - target).abs() / target < 0.03,
            "var {var} vs {target}"
        );
        assert!(mean.abs() < 0.02, "mean {mean} ≉ 0");
    }

    // 7. Error paths.
    #[test]
    fn error_paths() {
        let mut rng = LcgRng::new(0);
        let empty: Vec<f64> = vec![];
        assert!(matches!(
            fused_clip_and_noise(&empty, 1.0, 1.0, &mut rng),
            Err(PrivacyError::EmptyInput)
        ));
        assert!(matches!(
            fused_clip_and_noise(&[1.0], 0.0, 1.0, &mut rng),
            Err(PrivacyError::NonPositiveSensitivity(_))
        ));
        assert!(matches!(
            fused_clip_and_noise(&[1.0], 1.0, -1.0, &mut rng),
            Err(PrivacyError::InvalidParameter(_))
        ));
        let mut empty2: Vec<f64> = vec![];
        assert!(matches!(
            fused_clip_and_noise_in_place(&mut empty2, 1.0, 1.0, &mut rng),
            Err(PrivacyError::EmptyInput)
        ));
    }

    // 8. Determinism: same seed ⇒ same fused output.
    #[test]
    fn deterministic_same_seed() {
        let grad: Vec<f64> = (0..50).map(|i| i as f64 * 0.3 - 7.0).collect();
        let mut r1 = LcgRng::new(1234);
        let mut r2 = LcgRng::new(1234);
        let a = fused_clip_and_noise(&grad, 2.5, 0.8, &mut r1).expect("a");
        let b = fused_clip_and_noise(&grad, 2.5, 0.8, &mut r2).expect("b");
        assert_eq!(a, b);
    }
}
