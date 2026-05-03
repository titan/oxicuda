//! Time-axis warping for log-mel spectrograms (SpecAugment).
//!
//! Implements the time-warping step from Park et al. (2019):
//! a random interior anchor point is selected and bilinearly resampled
//! with a random displacement, independently stretching/compressing the
//! left and right halves of the time axis.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Sample a single output frame at fractional source position `t_frac`.
///
/// `src` — source spectrogram `[src_t, f]` row-major.
/// `t_frac` — fractional frame index in `[0, src_t - 1]` (clamped).
/// `frame_out` — output slice of length `f`.
///
/// Linear interpolation is used between the two bracketing integer frames.
fn bilinear_sample_time(src: &[f32], src_t: usize, f: usize, t_frac: f32, frame_out: &mut [f32]) {
    let t_frac = t_frac.clamp(0.0, (src_t as f32) - 1.0);
    let t0 = t_frac.floor() as usize;
    let t1 = (t0 + 1).min(src_t - 1);
    let alpha = t_frac - t0 as f32; // fractional part in [0, 1)

    let row0 = &src[t0 * f..(t0 + 1) * f];
    let row1 = &src[t1 * f..(t1 + 1) * f];
    for (out_bin, (&v0, &v1)) in frame_out.iter_mut().zip(row0.iter().zip(row1.iter())) {
        *out_bin = v0 * (1.0 - alpha) + v1 * alpha;
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Warp the time axis of a `[T, F]` log-mel spectrogram.
///
/// Selects a random anchor point `w ∈ [max_w, T - max_w)`, draws a random
/// displacement `d ∈ [-max_w, max_w]`, then bilinearly resamples:
/// - Left half  `[0, w)` → `[0, w + d)` in the output.
/// - Right half `[w, T)` → `[w + d, T)` in the output.
///
/// When `T <= 2 * max_w` there are not enough frames to choose an interior
/// anchor, so the function is a no-op.
///
/// # Errors
///
/// - [`AudioError::InvalidSequenceLength`] when `t == 0`.
/// - [`AudioError::InvalidNumMels`] when `f == 0`.
/// - [`AudioError::DimensionMismatch`] when `mel.len() != t * f`.
pub fn time_warp(
    mel: &mut Vec<f32>,
    t: usize,
    f: usize,
    max_w: usize,
    rng: &mut LcgRng,
) -> AudioResult<()> {
    if t == 0 {
        return Err(AudioError::InvalidSequenceLength(0));
    }
    if f == 0 {
        return Err(AudioError::InvalidNumMels(0));
    }
    if mel.len() != t * f {
        return Err(AudioError::DimensionMismatch {
            expected: t * f,
            got: mel.len(),
        });
    }

    // Not enough frames for a meaningful warp.
    if max_w == 0 || t <= 2 * max_w {
        return Ok(());
    }

    // Anchor: w ∈ [max_w, T - max_w)
    let anchor_range = t - 2 * max_w; // width of the valid anchor window
    let w = max_w + rng.next_usize(anchor_range);

    // Displacement: d ∈ [-(max_w), max_w] as integer.
    // Encode as unsigned [0, 2*max_w] then subtract max_w.
    let d_raw = rng.next_usize(2 * max_w + 1) as isize;
    let d = d_raw - max_w as isize;

    let w_prime = (w as isize + d) as usize; // new anchor position after displacement

    // Both w and w_prime must be in (0, T) exclusive; the clamping of
    // bilinear_sample_time handles edge cases numerically.

    let src = mel.clone();
    let out = mel.as_mut_slice();

    // ── Left half: output frames 0..w_prime are resampled from src 0..w ──
    // Linear map: out_frame t' ∈ [0, w_prime) ← src frame t' * w / w_prime
    if w_prime > 0 {
        let w_f = w as f32;
        let wp_f = w_prime as f32;
        for t_out in 0..w_prime {
            let t_src = t_out as f32 * w_f / wp_f;
            bilinear_sample_time(&src, t, f, t_src, &mut out[t_out * f..(t_out + 1) * f]);
        }
    }

    // ── Right half: output frames w_prime..T resampled from src w..T ──
    // Linear map: out_frame t' ∈ [w_prime, T) ← src frame w + (t' - w_prime)*(T-w)/(T-w_prime)
    let right_src_len = t - w; // number of source frames in the right half
    let right_dst_len = t - w_prime; // number of destination frames in the right half

    if right_dst_len > 0 && right_src_len > 0 {
        let scale = right_src_len as f32 / right_dst_len as f32;
        for t_out in w_prime..t {
            let t_src = w as f32 + (t_out - w_prime) as f32 * scale;
            bilinear_sample_time(&src, t, f, t_src, &mut out[t_out * f..(t_out + 1) * f]);
        }
    }

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn time_warp_output_length_unchanged() {
        let t = 20_usize;
        let f = 8_usize;
        let max_w = 4;
        let mut mel: Vec<f32> = (0..t * f).map(|i| i as f32).collect();
        let mut rng = make_rng();
        time_warp(&mut mel, t, f, max_w, &mut rng).expect("ok");
        assert_eq!(mel.len(), t * f);
    }

    #[test]
    fn time_warp_small_t_no_op() {
        // T <= 2 * max_w → must be no-op
        let t = 4_usize;
        let f = 4_usize;
        let max_w = 3; // 2*3=6 > 4
        let original: Vec<f32> = (0..t * f).map(|i| i as f32).collect();
        let mut mel = original.clone();
        let mut rng = make_rng();
        time_warp(&mut mel, t, f, max_w, &mut rng).expect("ok");
        assert_eq!(mel, original, "no-op expected when T <= 2*max_w");
    }

    #[test]
    fn time_warp_output_finite() {
        let t = 30_usize;
        let f = 16_usize;
        let max_w = 5;
        let mut rng = make_rng();
        let mut mel = vec![0.0_f32; t * f];
        rng.fill_normal(&mut mel);
        time_warp(&mut mel, t, f, max_w, &mut rng).expect("ok");
        assert!(
            mel.iter().all(|v| v.is_finite()),
            "non-finite value found after time warp"
        );
    }

    #[test]
    fn time_warp_deterministic() {
        let t = 20_usize;
        let f = 8_usize;
        let max_w = 4;
        let original: Vec<f32> = (0..t * f).map(|i| i as f32).collect();

        let mut mel1 = original.clone();
        let mut rng1 = LcgRng::new(77);
        time_warp(&mut mel1, t, f, max_w, &mut rng1).expect("ok");

        let mut mel2 = original.clone();
        let mut rng2 = LcgRng::new(77);
        time_warp(&mut mel2, t, f, max_w, &mut rng2).expect("ok");

        assert_eq!(mel1, mel2, "same seed must produce identical warp");
    }

    #[test]
    fn bilinear_sample_at_integer() {
        // Sampling at exactly frame index 2 should return the frame-2 data unchanged.
        let t_src = 4_usize;
        let f = 3_usize;
        let src: Vec<f32> = (0..t_src * f).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; f];
        bilinear_sample_time(&src, t_src, f, 2.0, &mut out);
        let expected = &src[2 * f..3 * f];
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "got={got}, exp={exp}");
        }
    }

    #[test]
    fn bilinear_sample_midpoint() {
        // Sampling at 0.5 should give the average of frames 0 and 1.
        let t_src = 3_usize;
        let f = 2_usize;
        let src = vec![0.0_f32, 0.0, 4.0, 6.0, 8.0, 10.0];
        let mut out = vec![0.0_f32; f];
        bilinear_sample_time(&src, t_src, f, 0.5, &mut out);
        // frame0 = [0, 0], frame1 = [4, 6] → average = [2, 3]
        assert!((out[0] - 2.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-6, "out[1]={}", out[1]);
    }

    #[test]
    fn time_warp_zero_t_error() {
        let mut mel = vec![1.0_f32; 8];
        let mut rng = make_rng();
        let err = time_warp(&mut mel, 0, 8, 2, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidSequenceLength(0)));
    }

    #[test]
    fn time_warp_zero_f_error() {
        let mut mel = vec![];
        let mut rng = make_rng();
        let err = time_warp(&mut mel, 8, 0, 2, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidNumMels(0)));
    }
}
