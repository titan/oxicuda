//! DropPath (stochastic depth) regularisation for residual transformer blocks.
//!
//! Stochastic depth (Huang 2016, "Deep Networks with Stochastic Depth";
//! popularised for ViTs by DeiT/BEiT/DINO) randomly drops the *residual branch*
//! of a block for an entire sample during training, replacing it with the
//! identity. Surviving branches are rescaled by `1 / keep_prob` so the expected
//! activation magnitude is preserved (the "inverted" formulation), making the
//! operation a no-op in expectation and therefore an identity at inference.
//!
//! Unlike ordinary dropout, the drop decision is taken **per sample** (per row
//! of a `[batch, …]` tensor): every element of a dropped sample's residual
//! branch is zeroed together, never element-wise. This matches the semantics of
//! `timm.layers.DropPath` and `torchvision.ops.stochastic_depth(mode="row")`.
//!
//! A linearly-increasing drop-probability *schedule* across depth
//! ([`drop_path_schedule`]) reproduces the common recipe where shallow blocks
//! are kept almost always and the deepest block is dropped with the configured
//! maximum probability.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a single DropPath (stochastic-depth) operator.
#[derive(Debug, Clone, PartialEq)]
pub struct DropPathConfig {
    /// Probability of dropping the residual branch of a sample, in `[0, 1)`.
    ///
    /// `0.0` disables DropPath (pure identity passthrough).
    pub drop_prob: f32,
    /// Whether surviving branches are rescaled by `1 / (1 - drop_prob)`.
    ///
    /// `true` (the default recipe) keeps the expected value unchanged so that
    /// inference — which never drops — matches training in expectation.
    pub scale_by_keep: bool,
}

impl DropPathConfig {
    /// Create and validate a `DropPathConfig`.
    ///
    /// # Errors
    /// - [`VisionError::Internal`] if `drop_prob` is non-finite or outside
    ///   `[0, 1)` (a drop probability of exactly `1.0` would zero every sample
    ///   and divide by zero under `scale_by_keep`).
    pub fn new(drop_prob: f32, scale_by_keep: bool) -> VisionResult<Self> {
        if !drop_prob.is_finite() {
            return Err(VisionError::NonFinite("drop_path drop_prob"));
        }
        if !(0.0..1.0).contains(&drop_prob) {
            return Err(VisionError::Internal(
                "drop_path drop_prob must lie in [0, 1)".into(),
            ));
        }
        Ok(Self {
            drop_prob,
            scale_by_keep,
        })
    }

    /// Probability that a sample's residual branch survives.
    #[must_use]
    #[inline]
    pub fn keep_prob(&self) -> f32 {
        1.0 - self.drop_prob
    }
}

// ─── DropPath operator ─────────────────────────────────────────────────────────

/// Per-sample stochastic-depth (DropPath) operator over a residual branch.
///
/// The operator is applied to the *output of a residual branch* (e.g. the MHSA
/// or MLP sub-layer **before** the skip add). In training mode each of the
/// `batch` rows is independently kept (and rescaled) or zeroed; in eval mode the
/// input is returned unchanged.
#[derive(Debug, Clone)]
pub struct DropPath {
    config: DropPathConfig,
}

impl DropPath {
    /// Construct a DropPath operator from a validated config.
    #[must_use]
    pub fn new(config: DropPathConfig) -> Self {
        Self { config }
    }

    /// Convenience constructor from a raw drop probability (scale-by-keep on).
    ///
    /// # Errors
    /// Propagates [`DropPathConfig::new`] validation.
    pub fn from_prob(drop_prob: f32) -> VisionResult<Self> {
        Ok(Self::new(DropPathConfig::new(drop_prob, true)?))
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &DropPathConfig {
        &self.config
    }

    /// Apply DropPath in **training** mode to a `[batch, feat]` tensor.
    ///
    /// Each of the `batch` rows is, independently:
    /// - kept with probability `keep_prob` and (if `scale_by_keep`) multiplied by
    ///   `1 / keep_prob`; or
    /// - zeroed entirely with probability `drop_prob`.
    ///
    /// When `drop_prob == 0.0` the input is returned verbatim (no RNG draw).
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `x` is empty or `batch == 0`.
    /// - [`VisionError::DimensionMismatch`] if `x.len() != batch * feat`.
    pub fn forward_train(
        &self,
        x: &[f32],
        batch: usize,
        feat: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<Vec<f32>> {
        if x.is_empty() {
            return Err(VisionError::EmptyInput("drop_path input"));
        }
        if batch == 0 {
            return Err(VisionError::EmptyInput("drop_path batch"));
        }
        if x.len() != batch * feat {
            return Err(VisionError::DimensionMismatch {
                expected: batch * feat,
                got: x.len(),
            });
        }

        if self.config.drop_prob == 0.0 {
            return Ok(x.to_vec());
        }

        let keep = self.config.keep_prob();
        let scale = if self.config.scale_by_keep {
            1.0 / keep
        } else {
            1.0
        };

        let mut out = vec![0.0f32; x.len()];
        for row in 0..batch {
            // Full-range uniform in [0, 1): keep when draw < keep_prob.
            let draw = rng.next_u32() as f64 / 2f64.powi(32);
            let src = &x[row * feat..(row + 1) * feat];
            if (draw as f32) < keep {
                let dst = &mut out[row * feat..(row + 1) * feat];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s * scale;
                }
            }
            // else: row stays zero (branch dropped).
        }
        Ok(out)
    }

    /// Apply DropPath in **eval** mode: identity passthrough.
    ///
    /// Stochastic depth is disabled at inference; the residual branch is always
    /// taken at its un-scaled value (the `scale_by_keep` rescale exactly cancels
    /// the training-time expectation).
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `x` is empty.
    pub fn forward_eval(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        if x.is_empty() {
            return Err(VisionError::EmptyInput("drop_path input"));
        }
        Ok(x.to_vec())
    }

    /// Apply a residual update `out = identity + drop_path(branch)` in training.
    ///
    /// This is the canonical use inside a transformer block: `identity` is the
    /// block input (the skip path) and `branch` is the sub-layer output that may
    /// be stochastically dropped. Both tensors are `[batch, feat]`.
    ///
    /// # Errors
    /// - [`VisionError::ShapeMismatch`] if `identity` and `branch` differ in
    ///   length.
    /// - Propagates [`Self::forward_train`] validation.
    pub fn residual_add_train(
        &self,
        identity: &[f32],
        branch: &[f32],
        batch: usize,
        feat: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<Vec<f32>> {
        if identity.len() != branch.len() {
            return Err(VisionError::ShapeMismatch {
                lhs: vec![identity.len()],
                rhs: vec![branch.len()],
            });
        }
        let dropped = self.forward_train(branch, batch, feat, rng)?;
        let mut out = identity.to_vec();
        for (o, d) in out.iter_mut().zip(dropped.iter()) {
            *o += d;
        }
        Ok(out)
    }
}

// ─── Depth schedule ────────────────────────────────────────────────────────────

/// Linearly-increasing per-block drop-probability schedule.
///
/// Returns a length-`depth` vector where block `i` is assigned
/// `max_drop_prob * i / (depth - 1)`, i.e. the first block is never dropped and
/// the last block is dropped with `max_drop_prob`. This is the standard
/// "linear stochastic-depth decay rule" used by DeiT, BEiT, Swin and ConvNeXt.
///
/// For `depth == 1` the single block gets probability `0.0` (no decay range).
///
/// # Errors
/// - [`VisionError::Internal`] if `depth == 0`.
/// - [`VisionError::NonFinite`] / [`VisionError::Internal`] if `max_drop_prob`
///   is not a finite value in `[0, 1)`.
pub fn drop_path_schedule(depth: usize, max_drop_prob: f32) -> VisionResult<Vec<f32>> {
    if depth == 0 {
        return Err(VisionError::Internal(
            "drop_path schedule depth == 0".into(),
        ));
    }
    if !max_drop_prob.is_finite() {
        return Err(VisionError::NonFinite("drop_path max_drop_prob"));
    }
    if !(0.0..1.0).contains(&max_drop_prob) {
        return Err(VisionError::Internal(
            "drop_path max_drop_prob must lie in [0, 1)".into(),
        ));
    }
    if depth == 1 {
        return Ok(vec![0.0]);
    }
    let denom = (depth - 1) as f32;
    Ok((0..depth)
        .map(|i| max_drop_prob * (i as f32) / denom)
        .collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_out_of_range() {
        assert!(DropPathConfig::new(1.0, true).is_err());
        assert!(DropPathConfig::new(-0.1, true).is_err());
        assert!(DropPathConfig::new(f32::NAN, true).is_err());
        assert!(DropPathConfig::new(0.0, true).is_ok());
        assert!(DropPathConfig::new(0.5, false).is_ok());
    }

    #[test]
    fn config_keep_prob() {
        let cfg = DropPathConfig::new(0.25, true).expect("ok");
        assert!((cfg.keep_prob() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn zero_prob_is_identity() {
        let dp = DropPath::from_prob(0.0).expect("ok");
        let mut rng = LcgRng::new(1);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = dp.forward_train(&x, 2, 3, &mut rng).expect("ok");
        assert_eq!(out, x);
    }

    #[test]
    fn eval_is_identity() {
        let dp = DropPath::from_prob(0.5).expect("ok");
        let x = vec![1.0, -2.0, 3.5];
        let out = dp.forward_eval(&x).expect("ok");
        assert_eq!(out, x);
    }

    #[test]
    fn dropped_rows_are_zero_kept_rows_are_scaled() {
        // With a fixed seed, classify each row as kept (scaled) or dropped (zero).
        let drop_prob = 0.5f32;
        let dp = DropPath::from_prob(drop_prob).expect("ok");
        let keep = 1.0 - drop_prob;
        let scale = 1.0 / keep;
        let batch = 64;
        let feat = 4;
        // Each row is a constant value equal to its row index + 1.
        let mut x = vec![0.0f32; batch * feat];
        for r in 0..batch {
            for c in 0..feat {
                x[r * feat + c] = (r as f32) + 1.0;
            }
        }
        let mut rng = LcgRng::new(2024);
        let out = dp.forward_train(&x, batch, feat, &mut rng).expect("ok");
        let mut n_kept = 0usize;
        for r in 0..batch {
            let v = out[r * feat]; // first element of the row
            let expected_kept = ((r as f32) + 1.0) * scale;
            if v.abs() < 1e-9 {
                // Dropped: every element of the row must be zero.
                assert!(
                    out[r * feat..(r + 1) * feat].iter().all(|&e| e == 0.0),
                    "row {r} partially dropped"
                );
            } else {
                assert!(
                    (v - expected_kept).abs() < 1e-4,
                    "row {r}: expected scaled {expected_kept}, got {v}"
                );
                n_kept += 1;
            }
        }
        // Roughly keep_prob fraction survive; allow a wide tolerance band.
        let frac = n_kept as f32 / batch as f32;
        assert!(
            (0.3..0.7).contains(&frac),
            "kept fraction {frac} far from keep_prob {keep}"
        );
    }

    #[test]
    fn expected_value_preserved_under_scaling() {
        // Averaging many independent draws of a single sample should recover the
        // un-dropped value (inverted-dropout expectation).
        let drop_prob = 0.3f32;
        let dp = DropPath::from_prob(drop_prob).expect("ok");
        let feat = 8;
        let x: Vec<f32> = (0..feat).map(|i| (i as f32) - 3.0).collect();
        let trials = 20_000;
        let mut acc = vec![0.0f64; feat];
        let mut rng = LcgRng::new(7);
        for _ in 0..trials {
            let out = dp.forward_train(&x, 1, feat, &mut rng).expect("ok");
            for (a, &o) in acc.iter_mut().zip(out.iter()) {
                *a += o as f64;
            }
        }
        for (i, &a) in acc.iter().enumerate() {
            let mean = a / trials as f64;
            let expected = x[i] as f64;
            assert!(
                (mean - expected).abs() < 0.1,
                "feature {i}: mean {mean} vs expected {expected}"
            );
        }
    }

    #[test]
    fn no_scale_keeps_unscaled_values() {
        let dp = DropPath::new(DropPathConfig::new(0.5, false).expect("ok"));
        let batch = 32;
        let feat = 2;
        let x = vec![3.0f32; batch * feat];
        let mut rng = LcgRng::new(11);
        let out = dp.forward_train(&x, batch, feat, &mut rng).expect("ok");
        for r in 0..batch {
            let v = out[r * feat];
            assert!(v == 0.0 || (v - 3.0).abs() < 1e-6, "unexpected value {v}");
        }
    }

    #[test]
    fn residual_add_shape_and_skip() {
        let dp = DropPath::from_prob(0.0).expect("ok");
        let mut rng = LcgRng::new(3);
        let identity = vec![1.0, 1.0, 1.0, 1.0];
        let branch = vec![0.5, 0.5, 0.5, 0.5];
        // drop_prob = 0 → branch passes through unchanged.
        let out = dp
            .residual_add_train(&identity, &branch, 2, 2, &mut rng)
            .expect("ok");
        for &v in &out {
            assert!((v - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn residual_add_shape_mismatch_errors() {
        let dp = DropPath::from_prob(0.1).expect("ok");
        let mut rng = LcgRng::new(4);
        let identity = vec![1.0, 1.0];
        let branch = vec![1.0, 1.0, 1.0];
        let r = dp.residual_add_train(&identity, &branch, 1, 2, &mut rng);
        assert!(matches!(r, Err(VisionError::ShapeMismatch { .. })));
    }

    #[test]
    fn forward_train_dim_mismatch_errors() {
        let dp = DropPath::from_prob(0.2).expect("ok");
        let mut rng = LcgRng::new(5);
        let x = vec![1.0f32; 6];
        let r = dp.forward_train(&x, 2, 4, &mut rng); // 2*4 != 6
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_train_empty_errors() {
        let dp = DropPath::from_prob(0.2).expect("ok");
        let mut rng = LcgRng::new(6);
        let r = dp.forward_train(&[], 0, 4, &mut rng);
        assert!(matches!(r, Err(VisionError::EmptyInput(_))));
    }

    #[test]
    fn schedule_linear_increasing() {
        let s = drop_path_schedule(5, 0.4).expect("ok");
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[4] - 0.4).abs() < 1e-6);
        // strictly non-decreasing
        for w in s.windows(2) {
            assert!(w[1] >= w[0]);
        }
        // midpoint
        assert!((s[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn schedule_depth_one_is_zero() {
        let s = drop_path_schedule(1, 0.5).expect("ok");
        assert_eq!(s, vec![0.0]);
    }

    #[test]
    fn schedule_invalid_inputs_error() {
        assert!(drop_path_schedule(0, 0.3).is_err());
        assert!(drop_path_schedule(4, 1.0).is_err());
        assert!(drop_path_schedule(4, f32::INFINITY).is_err());
    }

    #[test]
    fn deterministic_with_seed() {
        let dp = DropPath::from_prob(0.4).expect("ok");
        let x: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let mut r1 = LcgRng::new(99);
        let mut r2 = LcgRng::new(99);
        let a = dp.forward_train(&x, 10, 4, &mut r1).expect("ok");
        let b = dp.forward_train(&x, 10, 4, &mut r2).expect("ok");
        assert_eq!(a, b);
    }
}
