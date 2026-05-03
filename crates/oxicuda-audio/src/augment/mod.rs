//! Data augmentation operations for log-mel spectrograms.
//!
//! Provides SpecAugment (Park et al., 2019) masking and time-warping
//! transformations as both free functions and a composable pipeline.

pub mod spec_augment;
pub mod time_warp;

pub use spec_augment::{freq_mask, time_mask};
pub use time_warp::time_warp;

use crate::error::AudioResult;
use crate::handle::LcgRng;

// ─── SpecAugOp ───────────────────────────────────────────────────────────────

/// A single SpecAugment operation.
///
/// Enum-dispatched so no `dyn` allocations are needed in the pipeline hot path.
#[derive(Debug, Clone)]
pub enum SpecAugOp {
    /// Zero out `n_masks` random time bands of max width `max_t`.
    TimeMask {
        /// Maximum mask width in frames.
        max_t: usize,
        /// Number of independent masks to apply.
        n_masks: usize,
    },
    /// Zero out `n_masks` random frequency bands of max width `max_f`.
    FreqMask {
        /// Maximum mask width in mel bins.
        max_f: usize,
        /// Number of independent masks to apply.
        n_masks: usize,
    },
    /// Warp the time axis by up to `max_w` frames.
    TimeWarp {
        /// Maximum warp displacement in frames.
        max_w: usize,
    },
}

impl SpecAugOp {
    /// Apply this operation to a `[T, F]` log-mel tensor in-place.
    ///
    /// # Errors
    ///
    /// Propagates any [`AudioError`](crate::error::AudioError) from the
    /// underlying operation.
    pub fn apply(
        &self,
        mel: &mut Vec<f32>,
        t: usize,
        f: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<()> {
        match self {
            SpecAugOp::TimeMask { max_t, n_masks } => {
                time_mask(mel.as_mut_slice(), t, f, *max_t, *n_masks, rng)
            }
            SpecAugOp::FreqMask { max_f, n_masks } => {
                freq_mask(mel.as_mut_slice(), t, f, *max_f, *n_masks, rng)
            }
            SpecAugOp::TimeWarp { max_w } => time_warp(mel, t, f, *max_w, rng),
        }
    }
}

// ─── SpecAugPipeline ─────────────────────────────────────────────────────────

/// A composable pipeline of [`SpecAugOp`] operations applied sequentially.
///
/// Constructed with a builder pattern; each call to [`push`](Self::push)
/// appends one operation and returns `self` so calls can be chained.
#[derive(Debug, Clone, Default)]
pub struct SpecAugPipeline {
    ops: Vec<SpecAugOp>,
}

impl SpecAugPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append an operation to the pipeline.
    ///
    /// Returns `self` to allow builder-style chaining.
    pub fn push(mut self, op: SpecAugOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Apply all operations in insertion order to the given spectrogram.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered; subsequent operations are skipped.
    pub fn apply(
        &self,
        mel: &mut Vec<f32>,
        t: usize,
        f: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<()> {
        for op in &self.ops {
            op.apply(mel, t, f, rng)?;
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn spec_aug_pipeline_new_empty() {
        let p = SpecAugPipeline::new();
        // An empty pipeline applied to any valid spectrogram must be a no-op.
        let t = 10_usize;
        let f = 8_usize;
        let original = vec![1.0_f32; t * f];
        let mut mel = original.clone();
        let mut rng = make_rng();
        p.apply(&mut mel, t, f, &mut rng)
            .expect("empty pipeline ok");
        assert_eq!(mel, original);
    }

    #[test]
    fn spec_aug_pipeline_chain_applies_both() {
        let t = 30_usize;
        let f = 16_usize;
        let pipeline = SpecAugPipeline::new()
            .push(SpecAugOp::TimeMask {
                max_t: 5,
                n_masks: 2,
            })
            .push(SpecAugOp::FreqMask {
                max_f: 4,
                n_masks: 2,
            });
        let mut mel = vec![1.0_f32; t * f];
        let mut rng = make_rng();
        pipeline
            .apply(&mut mel, t, f, &mut rng)
            .expect("pipeline ok");
        // After applying both masks, at least some entries should be zero.
        assert!(mel.contains(&0.0_f32), "expected some zeros after masking");
    }

    #[test]
    fn spec_aug_op_apply_time_mask() {
        let t = 20_usize;
        let f = 8_usize;
        let op = SpecAugOp::TimeMask {
            max_t: 4,
            n_masks: 3,
        };
        let mut mel = vec![1.0_f32; t * f];
        let mut rng = make_rng();
        op.apply(&mut mel, t, f, &mut rng).expect("apply ok");
        assert_eq!(mel.len(), t * f);
    }

    #[test]
    fn spec_aug_op_apply_freq_mask() {
        let t = 15_usize;
        let f = 12_usize;
        let op = SpecAugOp::FreqMask {
            max_f: 3,
            n_masks: 2,
        };
        let mut mel = vec![1.0_f32; t * f];
        let mut rng = make_rng();
        op.apply(&mut mel, t, f, &mut rng).expect("apply ok");
        assert_eq!(mel.len(), t * f);
    }

    #[test]
    fn spec_aug_op_apply_time_warp() {
        let t = 40_usize;
        let f = 8_usize;
        let op = SpecAugOp::TimeWarp { max_w: 5 };
        let mut mel: Vec<f32> = (0..t * f).map(|i| i as f32).collect();
        let mut rng = make_rng();
        op.apply(&mut mel, t, f, &mut rng).expect("apply ok");
        assert_eq!(mel.len(), t * f);
    }

    #[test]
    fn spec_aug_pipeline_default_is_empty() {
        let p = SpecAugPipeline::default();
        let t = 5_usize;
        let f = 4_usize;
        let original = vec![2.0_f32; t * f];
        let mut mel = original.clone();
        let mut rng = make_rng();
        p.apply(&mut mel, t, f, &mut rng)
            .expect("default pipeline ok");
        assert_eq!(mel, original);
    }

    #[test]
    fn spec_aug_pipeline_with_warp() {
        let t = 50_usize;
        let f = 8_usize;
        let pipeline = SpecAugPipeline::new()
            .push(SpecAugOp::TimeWarp { max_w: 8 })
            .push(SpecAugOp::TimeMask {
                max_t: 5,
                n_masks: 1,
            });
        let mut mel: Vec<f32> = (0..t * f).map(|i| i as f32).collect();
        let mut rng = LcgRng::new(123);
        pipeline
            .apply(&mut mel, t, f, &mut rng)
            .expect("warp+mask pipeline ok");
        assert_eq!(mel.len(), t * f);
        assert!(mel.iter().all(|v| v.is_finite()));
    }
}
