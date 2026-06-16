//! ReviewKD — Distilling Knowledge via Knowledge Review (Chen et al. 2021).
//!
//! Reference: Chen, P., Liu, S., Zhao, H., & Jia, J. (2021). *Distilling
//! Knowledge via Knowledge Review*. CVPR 2021.
//! <https://arxiv.org/abs/2104.09044>
//!
//! # Idea
//!
//! Conventional feature distillation matches the student and teacher *at the same
//! stage*. Knowledge Review instead uses a **cross-stage** connection: a
//! lower-level student stage is supervised by **higher-level** teacher features.
//! Two components implement this:
//!
//! 1. **ABF — Attention-Based Fusion.** Fuses the current (lower-level) student
//!    feature map with the already-reviewed higher-level map. The two maps are
//!    concatenated channel-wise to produce two **spatial attention maps**
//!    (per-pixel weights, normalised so the two weights at each pixel sum to 1);
//!    the fused map is the per-pixel convex combination of the two inputs.
//!
//! 2. **HCL — Hierarchical Context Loss.** A multi-scale pyramid-pooled `L2`:
//!    the loss is the sum of squared differences computed at the full resolution
//!    *and* at several coarser average-pooled scales, capturing context at
//!    multiple granularities.
//!
//! All feature maps here are single-channel `H × W` spatial maps stored as flat
//! row-major `Vec<f32>` (the ABF channel mixing across many channels reduces, for
//! the purpose of the fusion weights, to the two spatial attention maps the paper
//! learns; we model the single-channel fused spatial map directly).

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// A single-channel spatial feature map of shape `H × W` (row-major).
#[derive(Debug, Clone)]
pub struct FeatureMap {
    /// Height.
    pub height: usize,
    /// Width.
    pub width: usize,
    /// Flat `height * width` row-major data.
    pub data: Vec<f32>,
}

impl FeatureMap {
    /// Construct a feature map, validating that `data.len() == height * width`.
    ///
    /// # Errors
    ///
    /// - [`DistillError::EmptyInput`] if `height == 0` or `width == 0`.
    /// - [`DistillError::DimensionMismatch`] if `data.len() != height * width`.
    pub fn new(height: usize, width: usize, data: Vec<f32>) -> DistillResult<Self> {
        if height == 0 || width == 0 {
            return Err(DistillError::EmptyInput);
        }
        if data.len() != height * width {
            return Err(DistillError::DimensionMismatch {
                expected: height * width,
                got: data.len(),
            });
        }
        Ok(Self {
            height,
            width,
            data,
        })
    }

    /// Number of spatial locations (`height * width`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the map has no spatial locations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Attention-Based Fusion module.
///
/// Produces, from the channel-wise concatenation of the low-level student map and
/// the reviewed higher-level map, two spatial attention maps. The paper produces
/// the two attention maps from a `1×1` conv over the **channel-wise concatenation**
/// of the two inputs. We model that with, per branch, a learnable linear
/// combination of *both* pixel values plus a bias: branch `b` has score
/// `s_b = a_b · low + c_b · high + d_b`. A per-pixel softmax over the two branch
/// scores yields normalised attention weights `α_low + α_high = 1`; the fused
/// output is `α_low ⊙ low + α_high ⊙ high`. When the two branches share identical
/// parameters (symmetric initialisation, [`AbfModule::default`]) the two scores
/// coincide at every pixel, so `α = 0.5` everywhere regardless of content.
#[derive(Debug, Clone)]
pub struct AbfModule {
    /// Low-branch weight on the low-level input.
    pub low_on_low: f32,
    /// Low-branch weight on the high-level input.
    pub low_on_high: f32,
    /// Low-branch bias.
    pub low_bias: f32,
    /// High-branch weight on the low-level input.
    pub high_on_low: f32,
    /// High-branch weight on the high-level input.
    pub high_on_high: f32,
    /// High-branch bias.
    pub high_bias: f32,
}

impl Default for AbfModule {
    fn default() -> Self {
        // Symmetric initialisation: both branches share identical parameters ⇒
        // s_low == s_high at every pixel ⇒ α = 0.5 each, for any input content.
        Self {
            low_on_low: 1.0,
            low_on_high: 1.0,
            low_bias: 0.0,
            high_on_low: 1.0,
            high_on_high: 1.0,
            high_bias: 0.0,
        }
    }
}

impl AbfModule {
    /// Construct an ABF module with small random learnable weights around the
    /// symmetric initialisation.
    #[must_use]
    pub fn new(rng: &mut LcgRng) -> Self {
        Self {
            low_on_low: 1.0 + 0.1 * rng.next_normal(),
            low_on_high: 1.0 + 0.1 * rng.next_normal(),
            low_bias: 0.05 * rng.next_normal(),
            high_on_low: 1.0 + 0.1 * rng.next_normal(),
            high_on_high: 1.0 + 0.1 * rng.next_normal(),
            high_bias: 0.05 * rng.next_normal(),
        }
    }

    /// Compute the two per-pixel attention maps `(α_low, α_high)` from the inputs.
    ///
    /// At each spatial location each branch score is a learnable linear
    /// combination of both input values plus a bias, and
    /// `(α_low, α_high) = softmax(s_low, s_high)`, so `α_low + α_high = 1`
    /// everywhere.
    ///
    /// # Errors
    ///
    /// [`DistillError::DimensionMismatch`] if the two maps differ in shape.
    pub fn attention(
        &self,
        low: &FeatureMap,
        high: &FeatureMap,
    ) -> DistillResult<(Vec<f32>, Vec<f32>)> {
        if low.height != high.height || low.width != high.width {
            return Err(DistillError::DimensionMismatch {
                expected: low.len(),
                got: high.len(),
            });
        }
        let n = low.len();
        let mut a_low = vec![0.0_f32; n];
        let mut a_high = vec![0.0_f32; n];
        for p in 0..n {
            let s_low =
                self.low_on_low * low.data[p] + self.low_on_high * high.data[p] + self.low_bias;
            let s_high =
                self.high_on_low * low.data[p] + self.high_on_high * high.data[p] + self.high_bias;
            let m = s_low.max(s_high);
            let e_low = (s_low - m).exp();
            let e_high = (s_high - m).exp();
            let denom = e_low + e_high;
            a_low[p] = e_low / denom;
            a_high[p] = e_high / denom;
        }
        Ok((a_low, a_high))
    }

    /// Fuse the low-level and high-level maps via per-pixel attention.
    ///
    /// `fused = α_low ⊙ low + α_high ⊙ high`.
    ///
    /// # Errors
    ///
    /// [`DistillError::DimensionMismatch`] if the two maps differ in shape.
    pub fn fuse(&self, low: &FeatureMap, high: &FeatureMap) -> DistillResult<FeatureMap> {
        let (a_low, a_high) = self.attention(low, high)?;
        let mut data = vec![0.0_f32; low.len()];
        for p in 0..low.len() {
            data[p] = a_low[p] * low.data[p] + a_high[p] * high.data[p];
        }
        FeatureMap::new(low.height, low.width, data)
    }
}

/// Average-pool a single-channel map by an integer `factor` along both axes.
///
/// The output has dimensions `ceil(H / factor) × ceil(W / factor)`; partial
/// border windows are averaged over the cells they actually cover. A `factor` of
/// 1 returns the input unchanged.
///
/// # Errors
///
/// [`DistillError::InvalidConfig`] if `factor == 0`.
pub fn avg_pool(map: &FeatureMap, factor: usize) -> DistillResult<FeatureMap> {
    if factor == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "avg_pool factor must be >= 1".into(),
        });
    }
    if factor == 1 {
        return Ok(map.clone());
    }
    let out_h = map.height.div_ceil(factor);
    let out_w = map.width.div_ceil(factor);
    let mut data = vec![0.0_f32; out_h * out_w];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = 0.0_f32;
            let mut count = 0u32;
            for dy in 0..factor {
                let y = oy * factor + dy;
                if y >= map.height {
                    break;
                }
                for dx in 0..factor {
                    let x = ox * factor + dx;
                    if x >= map.width {
                        break;
                    }
                    sum += map.data[y * map.width + x];
                    count += 1;
                }
            }
            data[oy * out_w + ox] = sum / count.max(1) as f32;
        }
    }
    FeatureMap::new(out_h, out_w, data)
}

/// Mean squared error between two equal-length flat maps.
#[must_use]
fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        / a.len() as f32
}

/// Hierarchical Context Loss: a multi-scale pyramid-pooled L2 between the student
/// and teacher feature maps.
///
/// For each pooling factor in `pool_factors`, both maps are average-pooled by that
/// factor and the (mean) squared error is accumulated; the result is the sum over
/// scales. A factor of `1` corresponds to the full-resolution term, which the
/// caller should normally include. Larger factors capture coarser context.
///
/// # Errors
///
/// - [`DistillError::DimensionMismatch`] if the two maps differ in shape.
/// - [`DistillError::EmptyInput`] if `pool_factors` is empty.
/// - [`DistillError::NumericalError`] if the result is non-finite.
pub fn hcl_loss(
    student: &FeatureMap,
    teacher: &FeatureMap,
    pool_factors: &[usize],
) -> DistillResult<f32> {
    if student.height != teacher.height || student.width != teacher.width {
        return Err(DistillError::DimensionMismatch {
            expected: student.len(),
            got: teacher.len(),
        });
    }
    if pool_factors.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let mut total = 0.0_f32;
    for &factor in pool_factors {
        let s = avg_pool(student, factor)?;
        let t = avg_pool(teacher, factor)?;
        total += mse(&s.data, &t.data);
    }
    if !total.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "HCL loss produced a non-finite value".into(),
        });
    }
    Ok(total)
}

/// A review connection: fuse a low-level student stage with reviewed higher-level
/// teacher features (via ABF), then score the fused map against the teacher with
/// the multi-scale HCL.
///
/// This is the data-flow assertion of Knowledge Review: the **higher-level**
/// teacher map is what supervises the **lower** student stage.
///
/// Returns `(fused_map, hcl)`.
///
/// # Errors
///
/// Propagates shape errors from [`AbfModule::fuse`] / [`hcl_loss`].
pub fn review_connection(
    abf: &AbfModule,
    student_low: &FeatureMap,
    teacher_high: &FeatureMap,
    pool_factors: &[usize],
) -> DistillResult<(FeatureMap, f32)> {
    // The fused map combines the student low-level map with the *higher-level*
    // teacher map (the "review" of higher-stage knowledge flowing down).
    let fused = abf.fuse(student_low, teacher_high)?;
    let hcl = hcl_loss(&fused, teacher_high, pool_factors)?;
    Ok((fused, hcl))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_map(h: usize, w: usize, scale: f32) -> FeatureMap {
        let data: Vec<f32> = (0..h * w).map(|i| i as f32 * scale).collect();
        FeatureMap::new(h, w, data).expect("valid map")
    }

    // (a) ABF fuses two maps producing a map of correct shape, attention weights
    //     sum to 1 at every pixel.
    #[test]
    fn abf_fusion_shape_and_attention_sum() {
        let abf = AbfModule::default();
        let low = ramp_map(3, 4, 0.5);
        let high = ramp_map(3, 4, -0.3);
        let (a_low, a_high) = abf.attention(&low, &high).expect("ok");
        for p in 0..low.len() {
            let s = a_low[p] + a_high[p];
            assert!(
                (s - 1.0).abs() < 1e-5,
                "attention weights must sum to 1 at pixel {p}, got {s}"
            );
            assert!(a_low[p] >= 0.0 && a_high[p] >= 0.0, "weights non-negative");
        }
        let fused = abf.fuse(&low, &high).expect("ok");
        assert_eq!(fused.height, 3);
        assert_eq!(fused.width, 4);
        assert_eq!(fused.len(), 12, "fused map shape must match inputs");
    }

    // Default (symmetric) ABF gives exactly α = 0.5 ⇒ fused == mean of the maps.
    #[test]
    fn abf_default_is_mean() {
        let abf = AbfModule::default();
        let low = ramp_map(2, 2, 1.0);
        let high = ramp_map(2, 2, 3.0);
        let (a_low, a_high) = abf.attention(&low, &high).expect("ok");
        for p in 0..low.len() {
            assert!((a_low[p] - 0.5).abs() < 1e-6);
            assert!((a_high[p] - 0.5).abs() < 1e-6);
        }
        let fused = abf.fuse(&low, &high).expect("ok");
        for p in 0..fused.len() {
            let expect = 0.5 * low.data[p] + 0.5 * high.data[p];
            assert!((fused.data[p] - expect).abs() < 1e-5);
        }
    }

    // (b) HCL computes a multi-scale L2 loss and is ≥ 0.
    #[test]
    fn hcl_multiscale_nonneg() {
        let s = ramp_map(4, 4, 0.7);
        let t = ramp_map(4, 4, 0.9);
        let loss = hcl_loss(&s, &t, &[1, 2, 4]).expect("ok");
        assert!(loss >= 0.0 && loss.is_finite(), "loss={loss}");
    }

    // (c) Perfect match (student == teacher at all scales) → ~0 HCL.
    #[test]
    fn hcl_perfect_match_zero() {
        let s = ramp_map(4, 6, 0.4);
        let t = s.clone();
        let loss = hcl_loss(&s, &t, &[1, 2, 3]).expect("ok");
        assert!(loss < 1e-10, "identical maps must give ~0 HCL, got {loss}");
    }

    // (d) The review connection feeds higher-level teacher features into the lower
    //     student stage — assert the data flow.
    #[test]
    fn review_connection_uses_teacher_high() {
        let abf = AbfModule::default();
        let student_low = ramp_map(3, 3, 1.0);
        let teacher_high = ramp_map(3, 3, 5.0);
        let (fused, hcl) =
            review_connection(&abf, &student_low, &teacher_high, &[1, 2]).expect("ok");
        // With symmetric ABF the fused map is the mean of student-low and
        // teacher-high; thus it must be strictly pulled toward teacher-high
        // (i.e. differ from the pure student map wherever the two inputs differ).
        let mut moved_toward_teacher = false;
        for p in 0..fused.len() {
            let mean = 0.5 * student_low.data[p] + 0.5 * teacher_high.data[p];
            assert!(
                (fused.data[p] - mean).abs() < 1e-5,
                "fused must be the mean"
            );
            if (teacher_high.data[p] - student_low.data[p]).abs() > 1e-6 {
                // Fused moved away from the student value toward the teacher value.
                let toward = (fused.data[p] - student_low.data[p])
                    * (teacher_high.data[p] - student_low.data[p]);
                assert!(
                    toward > 0.0,
                    "fused must move toward teacher-high at pixel {p}"
                );
                moved_toward_teacher = true;
            }
        }
        assert!(
            moved_toward_teacher,
            "review must inject higher-level teacher info"
        );
        assert!(hcl.is_finite() && hcl >= 0.0);
    }

    // (e) Shapes align after fusion (low/high with matching shape → same shape out).
    #[test]
    fn shapes_align_after_fusion() {
        let abf = AbfModule::new(&mut LcgRng::new(5));
        let low = ramp_map(5, 7, 0.2);
        let high = ramp_map(5, 7, -0.1);
        let fused = abf.fuse(&low, &high).expect("ok");
        assert_eq!((fused.height, fused.width), (5, 7));
        // Mismatched shapes must error.
        let bad = ramp_map(5, 6, 0.0);
        assert!(matches!(
            abf.fuse(&low, &bad),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    // (f) The multi-scale loss strictly includes the coarse-pool term: changing
    //     only a coarse-scale mismatch changes the loss.
    #[test]
    fn coarse_term_changes_loss() {
        // Build a student/teacher pair whose full-resolution MEAN over each 2×2
        // block is identical (so the full-res MSE is unchanged across the two
        // teacher variants) but whose coarse 2×-pooled values differ. Then the
        // pooled term must be the discriminator.
        //
        // Teacher A: a 2×2 block summing to S in one layout.
        // Teacher B: same elementwise SET of values but rearranged so the 2×2
        //            average is the same — instead we perturb the *coarse* mean.
        let student = FeatureMap::new(2, 2, vec![0.0, 0.0, 0.0, 0.0]).expect("ok");
        // teacher_a has block mean 0; teacher_b has block mean shifted ⇒ the pool=2
        // term differs even though we compare to the same zero student.
        let teacher_a = FeatureMap::new(2, 2, vec![1.0, -1.0, 1.0, -1.0]).expect("ok"); // mean 0
        let teacher_b = FeatureMap::new(2, 2, vec![2.0, 0.0, 2.0, 0.0]).expect("ok"); // mean 1

        // Full-resolution term: ‖student - teacher‖²/4. Compare with pool=2 added.
        let full_a = hcl_loss(&student, &teacher_a, &[1]).expect("ok");
        let full_b = hcl_loss(&student, &teacher_b, &[1]).expect("ok");

        let multi_a = hcl_loss(&student, &teacher_a, &[1, 2]).expect("ok");
        let multi_b = hcl_loss(&student, &teacher_b, &[1, 2]).expect("ok");

        // The coarse (pool=2) contribution is multi − full.
        let coarse_a = multi_a - full_a;
        let coarse_b = multi_b - full_b;
        // teacher_a pooled mean is 0 (matches student 0) ⇒ coarse term 0.
        assert!(
            coarse_a < 1e-6,
            "teacher_a coarse term should be ~0, got {coarse_a}"
        );
        // teacher_b pooled mean is 1 ⇒ coarse term is 1² = 1 (single coarse cell).
        assert!(
            coarse_b > 1e-3,
            "teacher_b coarse term must be positive, got {coarse_b}"
        );
        assert!(
            (coarse_b - coarse_a).abs() > 1e-3,
            "changing only a coarse-scale mismatch must change the multi-scale loss"
        );
    }

    // avg_pool sanity: pooling a constant map yields the same constant.
    #[test]
    fn avg_pool_constant() {
        let m = FeatureMap::new(4, 4, vec![2.5_f32; 16]).expect("ok");
        let p = avg_pool(&m, 2).expect("ok");
        assert_eq!((p.height, p.width), (2, 2));
        for &v in &p.data {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    // avg_pool handles non-divisible borders.
    #[test]
    fn avg_pool_partial_border() {
        // 3×3 pooled by 2 ⇒ 2×2; bottom-right cell is a single element.
        let m = ramp_map(3, 3, 1.0); // values 0..8
        let p = avg_pool(&m, 2).expect("ok");
        assert_eq!((p.height, p.width), (2, 2));
        // Bottom-right window is just element (2,2) = 8.
        assert!(
            (p.data[3] - 8.0).abs() < 1e-5,
            "corner cell = {}",
            p.data[3]
        );
    }

    #[test]
    fn map_construction_errors() {
        assert!(matches!(
            FeatureMap::new(0, 3, vec![]),
            Err(DistillError::EmptyInput)
        ));
        assert!(matches!(
            FeatureMap::new(2, 2, vec![1.0; 3]),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn hcl_empty_factors_errors() {
        let s = ramp_map(2, 2, 1.0);
        let t = ramp_map(2, 2, 1.0);
        assert!(matches!(
            hcl_loss(&s, &t, &[]),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn avg_pool_zero_factor_errors() {
        let m = ramp_map(2, 2, 1.0);
        assert!(matches!(
            avg_pool(&m, 0),
            Err(DistillError::InvalidConfig { .. })
        ));
    }
}
