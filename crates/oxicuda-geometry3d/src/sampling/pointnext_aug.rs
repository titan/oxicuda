//! Training-time data augmentations for point clouds (PointNeXt-style).
//!
//! Implements the four augmentations used in
//! Qian et al. 2022 NeurIPS — *PointNeXt: Revisiting PointNet++ with Improved
//! Training and Scaling Strategies* — applied during the data pipeline of a
//! point-cloud classifier:
//!
//! 1. **Random scale** — multiply all coordinates of all points by a single
//!    scalar drawn uniformly from `[scale_low, scale_high]`. The scale is
//!    applied isotropically and globally, preserving shape ratios.
//! 2. **Random jitter** — add per-point per-coordinate Gaussian noise with
//!    standard deviation `jitter_sigma`, clipped to `[-jitter_clip,
//!    jitter_clip]` so a single very-tail draw cannot dominate the augmented
//!    cloud (the original PointNet `jitter_point_cloud` used `np.clip` on a
//!    `np.random.normal`).
//! 3. **Random drop** — randomly remove a fraction `drop_ratio ∈ [0, 1)` of
//!    points, retaining `n_kept = n − floor(n · drop_ratio)` points. Drops are
//!    selected via a partial Fisher-Yates shuffle over the index set so each
//!    point has the same drop probability and no point is drawn twice.
//! 4. **Random yaw rotation** — rotate the cloud about the world up-axis
//!    (y-axis in this crate's convention; many point-cloud datasets also call
//!    this the "up rotation"). The rotation angle is drawn uniformly from
//!    `[0, 2π)` so the network sees the same shape at every azimuth.
//!
//! All randomness flows through the crate's deterministic [`LcgRng`], so a
//! given seed produces a reproducible augmented batch — the same input + same
//! seed always yields the same output.
//!
//! # Distance / coordinate conventions
//!
//! * Points are stored flat row-major as `[n × 3]` with `(x, y, z)` triples.
//! * The yaw rotation rotates about the **y-axis**; only `x` and `z` change.
//! * Gaussian draws come from [`LcgRng::next_normal_pair`] (Box-Muller).

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

/// Configuration for [`PointNextAug`].
///
/// Validation rules (checked once in [`PointNextAug::new`]):
///
/// * `scale_low > 0` and `scale_low <= scale_high` — a degenerate or inverted
///   range would either collapse the cloud (`scale = 0`) or be ill-formed.
/// * `jitter_sigma >= 0` — non-negative standard deviation; `0` disables
///   jitter.
/// * `jitter_clip > 0` — the clipped range must have positive width.
/// * `drop_ratio ∈ [0, 1)` — `1.0` would erase the entire cloud; `0.0`
///   disables dropping.
#[derive(Debug, Clone, PartialEq)]
pub struct PointNextAugConfig {
    /// Lower bound of the random isotropic scale (exclusive of zero).
    pub scale_low: f32,
    /// Upper bound of the random isotropic scale.
    pub scale_high: f32,
    /// Standard deviation of the per-coordinate Gaussian jitter.
    pub jitter_sigma: f32,
    /// Symmetric clip applied to the jitter draw before adding.
    pub jitter_clip: f32,
    /// Fraction of points to drop (e.g. `0.1` drops 10 %).
    pub drop_ratio: f32,
    /// Whether to apply a random yaw rotation in [`PointNextAug::apply`].
    pub rotation_yaw: bool,
}

/// Stateless training-time augmenter for point clouds.
///
/// Holds the validated configuration only; randomness flows through the
/// caller-supplied [`LcgRng`] so the same RNG state can be threaded through
/// every augmentation in a training step.
#[derive(Debug, Clone)]
pub struct PointNextAug {
    cfg: PointNextAugConfig,
}

impl PointNextAug {
    /// Validate the configuration and construct a new augmenter.
    ///
    /// # Errors
    ///
    /// [`Geom3dError::InvalidTopology`] when any of the [`PointNextAugConfig`]
    /// validation rules is violated.
    pub fn new(cfg: PointNextAugConfig) -> Geom3dResult<Self> {
        if cfg.scale_low <= 0.0 || !cfg.scale_low.is_finite() {
            return Err(Geom3dError::InvalidTopology {
                reason: "scale_low must be > 0 and finite",
            });
        }
        if !cfg.scale_high.is_finite() || cfg.scale_low > cfg.scale_high {
            return Err(Geom3dError::InvalidTopology {
                reason: "scale_low must be <= scale_high and finite",
            });
        }
        if cfg.jitter_sigma < 0.0 || !cfg.jitter_sigma.is_finite() {
            return Err(Geom3dError::InvalidTopology {
                reason: "jitter_sigma must be >= 0 and finite",
            });
        }
        if cfg.jitter_clip <= 0.0 || !cfg.jitter_clip.is_finite() {
            return Err(Geom3dError::InvalidTopology {
                reason: "jitter_clip must be > 0 and finite",
            });
        }
        if !(0.0..1.0).contains(&cfg.drop_ratio) || !cfg.drop_ratio.is_finite() {
            return Err(Geom3dError::InvalidTopology {
                reason: "drop_ratio must be in [0, 1) and finite",
            });
        }
        Ok(Self { cfg })
    }

    /// Return a read-only view of the validated configuration.
    #[must_use]
    pub fn config(&self) -> &PointNextAugConfig {
        &self.cfg
    }

    /// Apply a single random scalar scale in `[scale_low, scale_high]` to
    /// every coordinate of every point.
    ///
    /// The same scalar is shared across all `n × 3` coordinates so the cloud
    /// is rescaled isotropically (the *ratio* between any two coordinates is
    /// invariant). This matches the PointNeXt augmentation policy
    /// (`PointWOLF / PointAugment` follow-ups also use a global isotropic
    /// scale).
    ///
    /// # Errors
    ///
    /// [`Geom3dError::DimensionMismatch`] when `points.len() != n * 3`.
    pub fn random_scale(&self, points: &mut [f32], n: usize, rng: &mut LcgRng) -> Geom3dResult<()> {
        check_points_len(points, n)?;
        let u = rng.next_f32();
        let scale = self.cfg.scale_low + u * (self.cfg.scale_high - self.cfg.scale_low);
        for v in points.iter_mut().take(n * 3) {
            *v *= scale;
        }
        Ok(())
    }

    /// Add per-coordinate clipped Gaussian jitter to every point in place.
    ///
    /// Each of the `n × 3` coordinates is offset by an independent
    /// `N(0, jitter_sigma²)` draw, then clipped to `[-jitter_clip,
    /// jitter_clip]` so a tail draw cannot translate a single point far from
    /// the cloud.
    ///
    /// # Errors
    ///
    /// [`Geom3dError::DimensionMismatch`] when `points.len() != n * 3`.
    pub fn random_jitter(
        &self,
        points: &mut [f32],
        n: usize,
        rng: &mut LcgRng,
    ) -> Geom3dResult<()> {
        check_points_len(points, n)?;
        let sigma = self.cfg.jitter_sigma;
        let clip = self.cfg.jitter_clip;
        if sigma == 0.0 {
            // sigma == 0 leaves points unchanged but is a valid configuration.
            return Ok(());
        }
        // Box-Muller produces a normal pair; fill two coords per draw.
        let total = n * 3;
        let mut i = 0;
        while i + 1 < total {
            let (a, b) = rng.next_normal_pair();
            let da = (a * sigma).clamp(-clip, clip);
            let db = (b * sigma).clamp(-clip, clip);
            points[i] += da;
            points[i + 1] += db;
            i += 2;
        }
        if i < total {
            let (a, _) = rng.next_normal_pair();
            let da = (a * sigma).clamp(-clip, clip);
            points[i] += da;
        }
        Ok(())
    }

    /// Drop a random `floor(n · drop_ratio)` fraction of points, returning the
    /// kept points as `(kept × 3, n_kept)`.
    ///
    /// The number of kept points is `n_kept = n − floor(n · drop_ratio)` so a
    /// fixed `n` and `drop_ratio` always produces the same output length. The
    /// identities of the kept points are chosen by a partial Fisher-Yates
    /// shuffle (each point has equal probability of being kept).
    ///
    /// # Errors
    ///
    /// [`Geom3dError::DimensionMismatch`] when `points.len() != n * 3`.
    pub fn random_drop(
        &self,
        points: &[f32],
        n: usize,
        rng: &mut LcgRng,
    ) -> Geom3dResult<(Vec<f32>, usize)> {
        check_points_len(points, n)?;
        if n == 0 {
            return Ok((Vec::new(), 0));
        }
        let n_drop = ((n as f32) * self.cfg.drop_ratio).floor() as usize;
        let n_drop = n_drop.min(n);
        let n_kept = n - n_drop;

        // Partial Fisher-Yates: select the first `n_kept` indices uniformly
        // without replacement.
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..n_kept {
            let j = i + rng.next_usize(n - i);
            indices.swap(i, j);
        }
        // Sort the kept prefix to preserve a stable, deterministic layout.
        let kept_slice = match indices.get_mut(..n_kept) {
            Some(s) => s,
            None => {
                return Err(Geom3dError::Internal(
                    "internal slice access failed in random_drop".to_string(),
                ));
            }
        };
        kept_slice.sort_unstable();

        let mut out = Vec::with_capacity(n_kept * 3);
        for &idx in kept_slice.iter() {
            let base = idx * 3;
            let x = match points.get(base) {
                Some(v) => *v,
                None => {
                    return Err(Geom3dError::Internal(
                        "point index out of range in random_drop".to_string(),
                    ));
                }
            };
            let y = match points.get(base + 1) {
                Some(v) => *v,
                None => {
                    return Err(Geom3dError::Internal(
                        "point index out of range in random_drop".to_string(),
                    ));
                }
            };
            let z = match points.get(base + 2) {
                Some(v) => *v,
                None => {
                    return Err(Geom3dError::Internal(
                        "point index out of range in random_drop".to_string(),
                    ));
                }
            };
            out.push(x);
            out.push(y);
            out.push(z);
        }
        Ok((out, n_kept))
    }

    /// Rotate the cloud about the world y-axis (yaw) by a uniformly random
    /// angle drawn from `[0, 2π)`.
    ///
    /// Only the `x` and `z` coordinates are modified; `y` is preserved.
    /// Euclidean distances between any two points are preserved exactly (up
    /// to floating-point rounding).
    ///
    /// # Errors
    ///
    /// [`Geom3dError::DimensionMismatch`] when `points.len() != n * 3`.
    pub fn random_rotation_yaw(
        &self,
        points: &mut [f32],
        n: usize,
        rng: &mut LcgRng,
    ) -> Geom3dResult<()> {
        check_points_len(points, n)?;
        let theta = rng.next_f32() * 2.0 * std::f32::consts::PI;
        let c = theta.cos();
        let s = theta.sin();
        for i in 0..n {
            let base = i * 3;
            let x = points[base];
            let z = points[base + 2];
            // R_y(θ): x' = c·x + s·z ; z' = -s·x + c·z (right-handed, looking
            // down −y the cloud rotates counter-clockwise as θ grows).
            points[base] = c * x + s * z;
            points[base + 2] = -s * x + c * z;
        }
        Ok(())
    }

    /// Apply the full training-time augmentation pipeline in the canonical
    /// order: **scale → jitter → drop → rotation**.
    ///
    /// `rotation_yaw == false` skips the rotation step. The intermediate
    /// buffers are allocated only when the pipeline cannot operate in place
    /// (`random_drop` always reallocates because the output length changes).
    ///
    /// # Errors
    ///
    /// [`Geom3dError::DimensionMismatch`] when `points.len() != n * 3`.
    pub fn apply(
        &self,
        points: &[f32],
        n: usize,
        rng: &mut LcgRng,
    ) -> Geom3dResult<(Vec<f32>, usize)> {
        check_points_len(points, n)?;
        let mut buf: Vec<f32> = points.to_vec();
        self.random_scale(&mut buf, n, rng)?;
        self.random_jitter(&mut buf, n, rng)?;
        let (mut dropped, n_kept) = self.random_drop(&buf, n, rng)?;
        if self.cfg.rotation_yaw {
            self.random_rotation_yaw(&mut dropped, n_kept, rng)?;
        }
        Ok((dropped, n_kept))
    }
}

/// Sanity-check the flat `[n × 3]` layout.
#[inline]
fn check_points_len(points: &[f32], n: usize) -> Geom3dResult<()> {
    if points.len() != n * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * 3,
            got: points.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference configuration covering all augmentations with realistic
    /// PointNeXt hyperparameters.
    fn reference_cfg() -> PointNextAugConfig {
        PointNextAugConfig {
            scale_low: 0.8,
            scale_high: 1.25,
            jitter_sigma: 0.01,
            jitter_clip: 0.05,
            drop_ratio: 0.1,
            rotation_yaw: true,
        }
    }

    /// Deterministic small cloud spread inside `[-1, 1]`.
    fn make_cloud(n: usize) -> Vec<f32> {
        let mut pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            let a = ((i.wrapping_mul(2_654_435_761)) % 1000) as f32 / 500.0 - 1.0;
            let b = ((i.wrapping_mul(40_503).wrapping_add(7)) % 1000) as f32 / 500.0 - 1.0;
            let c = ((i.wrapping_mul(2_246_822_519)) % 1000) as f32 / 500.0 - 1.0;
            pts.push(a);
            pts.push(b);
            pts.push(c);
        }
        pts
    }

    #[test]
    fn new_validates_scale_low_positive() {
        let mut cfg = reference_cfg();
        cfg.scale_low = 0.0;
        assert!(PointNextAug::new(cfg.clone()).is_err());
        cfg.scale_low = -0.1;
        assert!(PointNextAug::new(cfg).is_err());
    }

    #[test]
    fn new_validates_scale_range_ordered() {
        let mut cfg = reference_cfg();
        cfg.scale_low = 1.5;
        cfg.scale_high = 1.0;
        assert!(PointNextAug::new(cfg).is_err());
    }

    #[test]
    fn new_validates_jitter_sigma_nonneg() {
        let mut cfg = reference_cfg();
        cfg.jitter_sigma = -0.001;
        assert!(PointNextAug::new(cfg).is_err());
    }

    #[test]
    fn new_validates_jitter_clip_positive() {
        let mut cfg = reference_cfg();
        cfg.jitter_clip = 0.0;
        assert!(PointNextAug::new(cfg.clone()).is_err());
        cfg.jitter_clip = -0.1;
        assert!(PointNextAug::new(cfg).is_err());
    }

    #[test]
    fn new_validates_drop_ratio_range() {
        let mut cfg = reference_cfg();
        cfg.drop_ratio = -0.1;
        assert!(PointNextAug::new(cfg.clone()).is_err());
        cfg.drop_ratio = 1.0;
        assert!(PointNextAug::new(cfg.clone()).is_err());
        cfg.drop_ratio = 1.5;
        assert!(PointNextAug::new(cfg).is_err());
    }

    #[test]
    fn random_scale_preserves_shape() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let mut pts = make_cloud(16);
        let original = pts.clone();
        let mut rng = LcgRng::new(7);
        aug.random_scale(&mut pts, 16, &mut rng).unwrap();
        assert_eq!(pts.len(), original.len(), "shape must be n*3");
    }

    #[test]
    fn random_scale_same_factor_everywhere() {
        // Pick a cloud with no zeros so we can divide and check the ratio.
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let mut pts = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let original = pts.clone();
        let mut rng = LcgRng::new(11);
        aug.random_scale(&mut pts, 3, &mut rng).unwrap();
        // All ratios scaled/original must equal the same scalar.
        let s0 = pts[0] / original[0];
        for (a, b) in pts.iter().zip(original.iter()) {
            assert!(
                (a / b - s0).abs() < 1e-5,
                "scale must be uniform: {} vs {}",
                a / b,
                s0
            );
        }
        assert!(s0 >= reference_cfg().scale_low - 1e-5);
        assert!(s0 <= reference_cfg().scale_high + 1e-5);
    }

    #[test]
    fn random_scale_dim_mismatch_err() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let mut pts = vec![0.0_f32; 5];
        let mut rng = LcgRng::new(0);
        assert!(aug.random_scale(&mut pts, 2, &mut rng).is_err());
    }

    #[test]
    fn random_jitter_bounded_by_clip() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 64;
        let original = vec![0.0_f32; n * 3];
        let mut pts = original.clone();
        let mut rng = LcgRng::new(31);
        aug.random_jitter(&mut pts, n, &mut rng).unwrap();
        let clip = reference_cfg().jitter_clip;
        for (a, b) in pts.iter().zip(original.iter()) {
            assert!(
                (*a - *b).abs() <= clip + 1e-6,
                "jitter must be bounded by clip: |{}-{}|>{}",
                a,
                b,
                clip
            );
        }
    }

    #[test]
    fn random_jitter_sigma_zero_is_identity() {
        let mut cfg = reference_cfg();
        cfg.jitter_sigma = 0.0;
        let aug = PointNextAug::new(cfg).unwrap();
        let n = 8;
        let mut pts = make_cloud(n);
        let original = pts.clone();
        let mut rng = LcgRng::new(99);
        aug.random_jitter(&mut pts, n, &mut rng).unwrap();
        assert_eq!(pts, original, "sigma=0 must leave points untouched");
    }

    #[test]
    fn random_drop_kept_count_matches_formula() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 100;
        let pts = make_cloud(n);
        let mut rng = LcgRng::new(13);
        let (kept, n_kept) = aug.random_drop(&pts, n, &mut rng).unwrap();
        let expected = n - ((n as f32) * reference_cfg().drop_ratio).floor() as usize;
        assert_eq!(n_kept, expected, "n_kept formula mismatch");
        assert_eq!(kept.len(), n_kept * 3, "kept buffer length mismatch");
    }

    #[test]
    fn random_drop_reproducible_with_same_seed() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 40;
        let pts = make_cloud(n);
        let mut rng_a = LcgRng::new(2024);
        let mut rng_b = LcgRng::new(2024);
        let a = aug.random_drop(&pts, n, &mut rng_a).unwrap();
        let b = aug.random_drop(&pts, n, &mut rng_b).unwrap();
        assert_eq!(a.0, b.0, "kept points must be identical for same seed");
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn random_drop_ratio_zero_keeps_all() {
        let mut cfg = reference_cfg();
        cfg.drop_ratio = 0.0;
        let aug = PointNextAug::new(cfg).unwrap();
        let n = 17;
        let pts = make_cloud(n);
        let mut rng = LcgRng::new(0);
        let (kept, n_kept) = aug.random_drop(&pts, n, &mut rng).unwrap();
        assert_eq!(n_kept, n, "drop_ratio=0 must keep every point");
        assert_eq!(kept.len(), n * 3);
        // With drop_ratio == 0 the indices are sorted = (0..n) so the output
        // is exactly the input.
        assert_eq!(kept, pts);
    }

    #[test]
    fn random_drop_single_point() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let pts = vec![3.0_f32, 4.0, 5.0];
        let mut rng = LcgRng::new(0);
        let (kept, n_kept) = aug.random_drop(&pts, 1, &mut rng).unwrap();
        // floor(1 * 0.1) = 0 so the point is preserved.
        assert_eq!(n_kept, 1);
        assert_eq!(kept, pts);
    }

    #[test]
    fn random_drop_dim_mismatch_err() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let pts = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut rng = LcgRng::new(0);
        assert!(aug.random_drop(&pts, 2, &mut rng).is_err());
    }

    #[test]
    fn random_rotation_yaw_preserves_distances() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 5;
        let mut pts = vec![
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
        ];
        // Compute pairwise distances before rotation.
        let mut before = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pts[i * 3] - pts[j * 3];
                let dy = pts[i * 3 + 1] - pts[j * 3 + 1];
                let dz = pts[i * 3 + 2] - pts[j * 3 + 2];
                before.push((dx * dx + dy * dy + dz * dz).sqrt());
            }
        }
        let mut rng = LcgRng::new(123);
        aug.random_rotation_yaw(&mut pts, n, &mut rng).unwrap();
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pts[i * 3] - pts[j * 3];
                let dy = pts[i * 3 + 1] - pts[j * 3 + 1];
                let dz = pts[i * 3 + 2] - pts[j * 3 + 2];
                let after = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!(
                    (after - before[k]).abs() < 1e-4,
                    "yaw must preserve distances: before={} after={}",
                    before[k],
                    after
                );
                k += 1;
            }
        }
    }

    #[test]
    fn random_rotation_yaw_only_touches_x_and_z() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 6;
        let mut pts = make_cloud(n);
        let original = pts.clone();
        let mut rng = LcgRng::new(7);
        aug.random_rotation_yaw(&mut pts, n, &mut rng).unwrap();
        for i in 0..n {
            assert!(
                (pts[i * 3 + 1] - original[i * 3 + 1]).abs() < 1e-6,
                "y coordinate must be preserved: before={} after={}",
                original[i * 3 + 1],
                pts[i * 3 + 1]
            );
        }
    }

    #[test]
    fn apply_returns_valid_output_size() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 64;
        let pts = make_cloud(n);
        let mut rng = LcgRng::new(0);
        let (out, n_kept) = aug.apply(&pts, n, &mut rng).unwrap();
        let expected_drop = ((n as f32) * reference_cfg().drop_ratio).floor() as usize;
        assert_eq!(n_kept, n - expected_drop);
        assert_eq!(out.len(), n_kept * 3);
    }

    #[test]
    fn apply_deterministic_with_same_seed() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let n = 32;
        let pts = make_cloud(n);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let a = aug.apply(&pts, n, &mut rng_a).unwrap();
        let b = aug.apply(&pts, n, &mut rng_b).unwrap();
        assert_eq!(a.1, b.1);
        assert_eq!(a.0.len(), b.0.len());
        for (x, y) in a.0.iter().zip(b.0.iter()) {
            assert!((x - y).abs() < 1e-6, "apply must be deterministic");
        }
    }

    #[test]
    fn apply_rotation_disabled_skips_rotation() {
        // With rotation disabled, scale_low == scale_high == 1, sigma == 0 and
        // drop_ratio == 0 the output must equal the input.
        let cfg = PointNextAugConfig {
            scale_low: 1.0,
            scale_high: 1.0,
            jitter_sigma: 0.0,
            jitter_clip: 0.1,
            drop_ratio: 0.0,
            rotation_yaw: false,
        };
        let aug = PointNextAug::new(cfg).unwrap();
        let n = 5;
        let pts = make_cloud(n);
        let mut rng = LcgRng::new(7);
        let (out, n_kept) = aug.apply(&pts, n, &mut rng).unwrap();
        assert_eq!(n_kept, n);
        assert_eq!(out.len(), pts.len());
        for (a, b) in out.iter().zip(pts.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn random_jitter_dim_mismatch_err() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let mut pts = vec![0.0_f32; 7];
        let mut rng = LcgRng::new(0);
        assert!(aug.random_jitter(&mut pts, 3, &mut rng).is_err());
    }

    #[test]
    fn random_rotation_yaw_dim_mismatch_err() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let mut pts = vec![0.0_f32; 8];
        let mut rng = LcgRng::new(0);
        assert!(aug.random_rotation_yaw(&mut pts, 3, &mut rng).is_err());
    }

    #[test]
    fn apply_dim_mismatch_err() {
        let aug = PointNextAug::new(reference_cfg()).unwrap();
        let pts = vec![0.0_f32; 7];
        let mut rng = LcgRng::new(0);
        assert!(aug.apply(&pts, 3, &mut rng).is_err());
    }

    #[test]
    fn config_accessor_roundtrip() {
        let cfg = reference_cfg();
        let aug = PointNextAug::new(cfg.clone()).unwrap();
        assert_eq!(aug.config(), &cfg);
    }
}
