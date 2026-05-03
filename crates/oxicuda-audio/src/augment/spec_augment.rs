//! SpecAugment masking operations for log-mel spectrograms.
//!
//! Implements the time-masking and frequency-masking policies from
//! Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method
//! for Automatic Speech Recognition".

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── time_mask ───────────────────────────────────────────────────────────────

/// Zero out `n_masks` random time bands in a `[T, F]` log-mel tensor.
///
/// For each mask a width `w ∈ [0, max_t]` and a start frame
/// `s ∈ [0, T - w]` are sampled uniformly.  All frequency bins for frames
/// `s..s+w` are zeroed.
///
/// When `n_masks == 0` the function is a no-op.
///
/// # Errors
///
/// - [`AudioError::InvalidSequenceLength`] when `t == 0`.
/// - [`AudioError::InvalidNumMels`] when `f == 0`.
/// - [`AudioError::DimensionMismatch`] when `mel.len() != t * f`.
pub fn time_mask(
    mel: &mut [f32],
    t: usize,
    f: usize,
    max_t: usize,
    n_masks: usize,
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
    if n_masks == 0 || max_t == 0 {
        return Ok(());
    }

    for _ in 0..n_masks {
        // Width in [0, min(max_t, t)]
        let clipped_max = max_t.min(t);
        let width = rng.next_usize(clipped_max + 1);
        if width == 0 {
            continue;
        }
        // Start in [0, t - width]
        let range = t - width + 1;
        let start = rng.next_usize(range);
        for frame in start..start + width {
            let row_start = frame * f;
            let row_end = row_start + f;
            for bin in &mut mel[row_start..row_end] {
                *bin = 0.0;
            }
        }
    }
    Ok(())
}

// ─── freq_mask ───────────────────────────────────────────────────────────────

/// Zero out `n_masks` random frequency bands in a `[T, F]` log-mel tensor.
///
/// For each mask a width `w ∈ [0, max_f]` and a start bin
/// `fs ∈ [0, F - w]` are sampled uniformly.  All time frames for bins
/// `fs..fs+w` are zeroed.
///
/// When `n_masks == 0` the function is a no-op.
///
/// # Errors
///
/// - [`AudioError::InvalidSequenceLength`] when `t == 0`.
/// - [`AudioError::InvalidNumMels`] when `f == 0`.
/// - [`AudioError::DimensionMismatch`] when `mel.len() != t * f`.
pub fn freq_mask(
    mel: &mut [f32],
    t: usize,
    f: usize,
    max_f: usize,
    n_masks: usize,
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
    if n_masks == 0 || max_f == 0 {
        return Ok(());
    }

    for _ in 0..n_masks {
        let clipped_max = max_f.min(f);
        let width = rng.next_usize(clipped_max + 1);
        if width == 0 {
            continue;
        }
        let range = f - width + 1;
        let fs = rng.next_usize(range);
        // Zero the frequency band across all frames
        for frame in 0..t {
            let row_start = frame * f;
            for bin in fs..fs + width {
                mel[row_start + bin] = 0.0;
            }
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    fn ones_mel(t: usize, f: usize) -> Vec<f32> {
        vec![1.0_f32; t * f]
    }

    // ── time_mask ─────────────────────────────────────────────────────────

    #[test]
    fn time_mask_zeros_correct_range() {
        let t = 20_usize;
        let f = 8_usize;
        let max_t = 5;
        let mut mel = ones_mel(t, f);
        let mut rng = make_rng();

        // Apply a large number of masks so at least one region is definitely masked.
        time_mask(&mut mel, t, f, max_t, 10, &mut rng).expect("ok");

        // For any zeroed frame, the entire frequency row must be zero.
        for frame in 0..t {
            let row = &mel[frame * f..(frame + 1) * f];
            if row[0] == 0.0 {
                // All bins in that frame should be zero.
                assert!(
                    row.iter().all(|v| *v == 0.0),
                    "partial row zero in frame {frame}"
                );
            }
        }
    }

    #[test]
    fn time_mask_zero_masks_no_change() {
        let t = 10_usize;
        let f = 4_usize;
        let original = ones_mel(t, f);
        let mut mel = original.clone();
        let mut rng = make_rng();
        time_mask(&mut mel, t, f, 3, 0, &mut rng).expect("ok");
        assert_eq!(mel, original);
    }

    #[test]
    fn time_mask_full_mask_all_zero() {
        // Set max_t = t, single mask: should zero the entire spectrogram
        // (probabilistically, but with a fixed LCG we verify all-zero is possible).
        // Instead we set a very large n_masks to force coverage.
        let t = 5_usize;
        let f = 3_usize;
        // Use a deterministic seed and large n_masks.
        let mut mel = ones_mel(t, f);
        // With max_t == t and 100 masks it is virtually guaranteed everything is zero.
        let mut rng = LcgRng::new(7);
        time_mask(&mut mel, t, f, t, 100, &mut rng).expect("ok");
        assert!(mel.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn time_mask_invalid_t_error() {
        let mut mel = vec![1.0_f32; 8];
        let mut rng = make_rng();
        let err = time_mask(&mut mel, 0, 8, 2, 1, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidSequenceLength(0)));
    }

    #[test]
    fn time_mask_invalid_f_error() {
        let mut mel = vec![1.0_f32; 8];
        let mut rng = make_rng();
        let err = time_mask(&mut mel, 8, 0, 2, 1, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidNumMels(0)));
    }

    // ── freq_mask ─────────────────────────────────────────────────────────

    #[test]
    fn freq_mask_zeros_correct_range() {
        let t = 10_usize;
        let f = 16_usize;
        let max_f = 4;
        let mut mel = ones_mel(t, f);
        let mut rng = LcgRng::new(13);
        freq_mask(&mut mel, t, f, max_f, 5, &mut rng).expect("ok");

        // If a frequency bin is zeroed in one frame, it must be zero in every frame.
        for bin in 0..f {
            let first_frame_val = mel[bin];
            for frame in 1..t {
                assert_eq!(
                    mel[frame * f + bin],
                    first_frame_val,
                    "bin {bin} not uniform across time frames"
                );
            }
        }
    }

    #[test]
    fn freq_mask_zero_masks_no_change() {
        let t = 6_usize;
        let f = 8_usize;
        let original = ones_mel(t, f);
        let mut mel = original.clone();
        let mut rng = make_rng();
        freq_mask(&mut mel, t, f, 3, 0, &mut rng).expect("ok");
        assert_eq!(mel, original);
    }

    #[test]
    fn freq_mask_invalid_t_error() {
        let mut mel = vec![1.0_f32; 8];
        let mut rng = make_rng();
        let err = freq_mask(&mut mel, 0, 8, 2, 1, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidSequenceLength(0)));
    }

    #[test]
    fn freq_mask_invalid_f_error() {
        let mut mel = vec![1.0_f32; 8];
        let mut rng = make_rng();
        let err = freq_mask(&mut mel, 8, 0, 2, 1, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidNumMels(0)));
    }
}
