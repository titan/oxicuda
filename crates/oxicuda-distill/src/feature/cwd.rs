//! CWD — Channel-Wise Knowledge Distillation (Shu et al. 2021).
//!
//! Reference: Shu, C., Liu, Y., Gao, J., Yan, Z., & Shen, C. (2021).
//! *Channel-wise Knowledge Distillation for Dense Prediction*. ICCV 2021.
//! <https://arxiv.org/abs/2011.13256>
//!
//! Classical feature distillation (e.g. FitNets, AT) aligns the *spatial* activation maps
//! of teacher and student. For dense-prediction tasks (segmentation, detection) Shu et al.
//! show that aligning the **per-channel probability distribution** over spatial locations is
//! far more effective: each channel is softmax-normalised across its `H · W` spatial
//! positions (with a temperature `T`), and the student is trained to match the teacher's
//! resulting categorical distributions via the asymmetric KL divergence.
//!
//! For a feature map `F ∈ ℝ^{C × H × W}` (stored channel-major, i.e. channel `c` occupies the
//! contiguous block `F[c · HW .. (c+1) · HW]`), the per-channel soft activation is
//!
//! ```text
//!   φ(F)_{c,i} = softmax_i( F_{c,i} / T )            for spatial index i ∈ [0, H·W)
//! ```
//!
//! and the channel-wise distillation loss aggregates the KL divergence of the teacher's
//! distribution from the student's, scaled by `T²` (so the gradient magnitude is
//! temperature-independent, matching Hinton-style KD):
//!
//! ```text
//!   L_CWD = (T² / C) · Σ_c  KL( φ(F^t)_c ‖ φ(F^s)_c )
//! ```
//!
//! When teacher and student channel counts differ, an optional 1×1 linear projection
//! ([`ChannelProjector`]) maps the student's `C_s` channels onto the teacher's `C_t`
//! channels before the spatial softmax.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-10;

/// Configuration for channel-wise distillation.
#[derive(Debug, Clone)]
pub struct CwdConfig {
    /// Number of spatial channels `C` in each feature map.
    pub channels: usize,
    /// Spatial height `H`.
    pub height: usize,
    /// Spatial width `W`.
    pub width: usize,
    /// Softmax temperature `T` (> 0). Larger values soften the spatial distribution.
    pub temperature: f32,
}

impl CwdConfig {
    /// Number of spatial positions `H · W` per channel.
    #[must_use]
    #[inline]
    pub fn spatial(&self) -> usize {
        self.height * self.width
    }

    fn validate(&self) -> DistillResult<()> {
        if self.channels == 0 || self.height == 0 || self.width == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "CWD channels/height/width must all be > 0".into(),
            });
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "CWD temperature must be finite and > 0, got {}",
                    self.temperature
                ),
            });
        }
        Ok(())
    }
}

/// Numerically-stable spatial softmax of one channel.
///
/// `channel` holds the `H·W` activations of a single channel; the result is a probability
/// distribution over spatial positions after dividing by `temperature`.
#[must_use]
pub fn spatial_softmax(channel: &[f32], temperature: f32) -> Vec<f32> {
    let t = if temperature.abs() < EPS {
        EPS
    } else {
        temperature
    };
    let max_val = channel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = channel.iter().map(|&x| ((x - max_val) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let sum_safe = if sum < EPS { EPS } else { sum };
    exps.iter().map(|&e| e / sum_safe).collect()
}

/// Asymmetric KL divergence `KL(p ‖ q) = Σ p_i · ln(p_i / q_i)`.
///
/// Terms with `p_i == 0` contribute zero (the `0 · ln 0` convention).
#[must_use]
pub fn channel_kl(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else {
                pi * (pi / (qi + EPS)).ln()
            }
        })
        .sum()
}

/// Compute the channel-wise distillation loss between teacher and student feature maps.
///
/// Both `student` and `teacher` are flat slices of length `C · H · W` laid out channel-major.
/// The teacher distribution is treated as the target: `KL(teacher ‖ student)`.
///
/// # Errors
/// - [`DistillError::EmptyInput`] if either slice is empty.
/// - [`DistillError::InvalidConfig`] if the config is invalid.
/// - [`DistillError::DimensionMismatch`] if either slice length differs from `C · H · W`.
pub fn cwd_loss(student: &[f32], teacher: &[f32], cfg: &CwdConfig) -> DistillResult<f32> {
    cfg.validate()?;
    if student.is_empty() || teacher.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let hw = cfg.spatial();
    let expected = cfg.channels * hw;
    if student.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: student.len(),
        });
    }
    if teacher.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: teacher.len(),
        });
    }
    let t = cfg.temperature;
    let mut total = 0.0_f32;
    for c in 0..cfg.channels {
        let s_chan = &student[c * hw..(c + 1) * hw];
        let t_chan = &teacher[c * hw..(c + 1) * hw];
        let p_s = spatial_softmax(s_chan, t);
        let p_t = spatial_softmax(t_chan, t);
        total += channel_kl(&p_t, &p_s);
    }
    let loss = t * t * total / cfg.channels as f32;
    if !loss.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "CWD loss is not finite".into(),
        });
    }
    Ok(loss)
}

/// A 1×1 linear projection that maps `in_channels` student channels to `out_channels`
/// (typically the teacher's channel count) before the spatial softmax.
///
/// The same projection weight is applied identically at every spatial position, exactly as a
/// 1×1 convolution would be.
#[derive(Debug, Clone)]
pub struct ChannelProjector {
    /// Number of input (student) channels.
    pub in_channels: usize,
    /// Number of output (teacher) channels.
    pub out_channels: usize,
    /// Projection weight, shape `[out_channels × in_channels]` (row-major).
    pub w: Vec<f32>,
}

impl ChannelProjector {
    /// Construct a projector with weights drawn from `N(0, 1/√in_channels)`.
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, rng: &mut LcgRng) -> Self {
        let scale = if in_channels == 0 {
            1.0
        } else {
            1.0 / (in_channels as f32).sqrt()
        };
        let mut w = vec![0.0_f32; out_channels * in_channels];
        for wi in w.iter_mut() {
            *wi = rng.next_normal() * scale;
        }
        Self {
            in_channels,
            out_channels,
            w,
        }
    }

    /// Project a student feature map `[in_channels × hw]` to `[out_channels × hw]`.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `in_channels == 0`.
    /// - [`DistillError::DimensionMismatch`] if `feat.len() != in_channels · hw`.
    pub fn project(&self, feat: &[f32], hw: usize) -> DistillResult<Vec<f32>> {
        if self.in_channels == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "ChannelProjector in_channels is zero".into(),
            });
        }
        let expected = self.in_channels * hw;
        if feat.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: feat.len(),
            });
        }
        let mut out = vec![0.0_f32; self.out_channels * hw];
        for oc in 0..self.out_channels {
            let w_row = &self.w[oc * self.in_channels..(oc + 1) * self.in_channels];
            for i in 0..hw {
                let mut acc = 0.0_f32;
                for (ic, &w_ic) in w_row.iter().enumerate() {
                    acc += w_ic * feat[ic * hw + i];
                }
                out[oc * hw + i] = acc;
            }
        }
        Ok(out)
    }
}

/// Channel-wise distillation loss with a learnable 1×1 projection of the student feature map.
///
/// The student map `[C_s × H × W]` is first projected to the teacher channel count via
/// `projector`, then [`cwd_loss`] is applied. `cfg.channels` must equal the teacher channel
/// count (i.e. `projector.out_channels`).
///
/// # Errors
/// Propagates errors from [`ChannelProjector::project`] and [`cwd_loss`].
pub fn cwd_loss_projected(
    student: &[f32],
    teacher: &[f32],
    projector: &ChannelProjector,
    cfg: &CwdConfig,
) -> DistillResult<f32> {
    cfg.validate()?;
    if projector.out_channels != cfg.channels {
        return Err(DistillError::DimensionMismatch {
            expected: cfg.channels,
            got: projector.out_channels,
        });
    }
    let hw = cfg.spatial();
    let projected = projector.project(student, hw)?;
    cwd_loss(&projected, teacher, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(c: usize, h: usize, w: usize, t: f32) -> CwdConfig {
        CwdConfig {
            channels: c,
            height: h,
            width: w,
            temperature: t,
        }
    }

    #[test]
    fn spatial_softmax_sums_to_one() {
        let chan = vec![1.0_f32, 2.0, 3.0, 0.5];
        let p = spatial_softmax(&chan, 2.0);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn spatial_softmax_uniform_for_constant_channel() {
        let chan = vec![4.0_f32; 6];
        let p = spatial_softmax(&chan, 3.0);
        for &v in &p {
            assert!(
                (v - 1.0 / 6.0).abs() < 1e-5,
                "constant channel → uniform, got {v}"
            );
        }
    }

    #[test]
    fn channel_kl_identical_is_zero() {
        let p = vec![0.25_f32, 0.25, 0.25, 0.25];
        assert!(channel_kl(&p, &p) < 1e-6, "KL(p‖p) must be ~0");
    }

    #[test]
    fn channel_kl_nonneg() {
        let p = vec![0.7_f32, 0.2, 0.1];
        let q = vec![0.2_f32, 0.3, 0.5];
        assert!(channel_kl(&p, &q) >= 0.0, "KL must be non-negative");
    }

    #[test]
    fn cwd_loss_identical_maps_is_zero() {
        let c = cfg(3, 2, 2, 4.0);
        let feat: Vec<f32> = (0..3 * 4).map(|i| (i as f32) * 0.3).collect();
        let loss = cwd_loss(&feat, &feat, &c).expect("cwd_loss should succeed");
        assert!(loss < 1e-5, "identical maps → ~0 loss, got {loss}");
    }

    #[test]
    fn cwd_loss_finite_and_nonneg() {
        let c = cfg(4, 3, 3, 2.0);
        let hw = c.spatial();
        let s: Vec<f32> = (0..4 * hw).map(|i| (i as f32).sin()).collect();
        let t: Vec<f32> = (0..4 * hw).map(|i| (i as f32 * 0.7).cos()).collect();
        let loss = cwd_loss(&s, &t, &c).expect("cwd_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn cwd_loss_increases_with_divergence() {
        let c = cfg(2, 2, 2, 1.0);
        let hw = c.spatial();
        let teacher: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // close student
        let close: Vec<f32> = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0];
        // far student (peaks shifted)
        let far: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let _ = hw;
        let l_close = cwd_loss(&close, &teacher, &c).expect("cwd_loss should succeed");
        let l_far = cwd_loss(&far, &teacher, &c).expect("cwd_loss should succeed");
        assert!(
            l_far > l_close,
            "far student must have larger loss: close={l_close}, far={l_far}"
        );
    }

    #[test]
    fn cwd_loss_empty_input_errors() {
        let c = cfg(2, 2, 2, 1.0);
        assert!(matches!(
            cwd_loss(&[], &[1.0; 8], &c),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn cwd_loss_dimension_mismatch_errors() {
        let c = cfg(2, 2, 2, 1.0);
        let s = vec![0.0_f32; 7];
        let t = vec![0.0_f32; 8];
        assert!(matches!(
            cwd_loss(&s, &t, &c),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn cwd_loss_invalid_temperature_errors() {
        let c = cfg(2, 2, 2, -1.0);
        let s = vec![0.0_f32; 8];
        assert!(matches!(
            cwd_loss(&s, &s, &c),
            Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn cwd_loss_zero_channels_errors() {
        let c = cfg(0, 2, 2, 1.0);
        let s = vec![0.0_f32; 0];
        assert!(matches!(
            cwd_loss(&s, &s, &c),
            Err(DistillError::EmptyInput) | Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn projector_output_shape() {
        let mut rng = LcgRng::new(3);
        let proj = ChannelProjector::new(4, 6, &mut rng);
        let hw = 5;
        let feat: Vec<f32> = (0..4 * hw).map(|i| i as f32).collect();
        let out = proj.project(&feat, hw).expect("project should succeed");
        assert_eq!(out.len(), 6 * hw);
    }

    #[test]
    fn projector_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(4);
        let proj = ChannelProjector::new(4, 6, &mut rng);
        let feat = vec![0.0_f32; 4 * 5 - 1];
        assert!(matches!(
            proj.project(&feat, 5),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn cwd_loss_projected_matches_channels() {
        let mut rng = LcgRng::new(5);
        let proj = ChannelProjector::new(3, 4, &mut rng);
        let c = cfg(4, 2, 2, 2.0);
        let hw = c.spatial();
        let student: Vec<f32> = (0..3 * hw).map(|i| (i as f32) * 0.1).collect();
        let teacher: Vec<f32> = (0..4 * hw).map(|i| (i as f32) * 0.2).collect();
        let loss = cwd_loss_projected(&student, &teacher, &proj, &c)
            .expect("cwd_loss_projected should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "projected loss={loss}");
    }

    #[test]
    fn cwd_loss_projected_channel_mismatch_errors() {
        let mut rng = LcgRng::new(6);
        let proj = ChannelProjector::new(3, 5, &mut rng); // out=5
        let c = cfg(4, 2, 2, 2.0); // teacher channels=4 != 5
        let student = vec![0.0_f32; 3 * 4];
        let teacher = vec![0.0_f32; 4 * 4];
        assert!(matches!(
            cwd_loss_projected(&student, &teacher, &proj, &c),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn temperature_t_squared_keeps_loss_stable() {
        // The `T²` prefactor compensates for the `~1/T²` shrinkage of KL on softened
        // distributions, so the magnitude of `T²·KL` stays bounded (and finite) as T grows.
        // This is the intended gradient-stabilising behaviour of temperature scaling.
        let c1 = cfg(2, 2, 2, 1.0);
        let c4 = cfg(2, 2, 2, 4.0);
        let hw = c1.spatial();
        let s: Vec<f32> = (0..2 * hw).map(|i| (i as f32) * 0.5).collect();
        let t: Vec<f32> = (0..2 * hw).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let l1 = cwd_loss(&s, &t, &c1).expect("cwd_loss should succeed");
        let l4 = cwd_loss(&s, &t, &c4).expect("cwd_loss should succeed");
        assert!(l1.is_finite() && l4.is_finite(), "losses must be finite");
        assert!(l1 >= 0.0 && l4 >= 0.0, "losses must be non-negative");
        // Both should remain on the same order of magnitude (no blow-up).
        assert!(
            l4 <= 10.0 * (l1 + 1e-6),
            "T²·KL must stay bounded: l1={l1}, l4={l4}"
        );
    }
}
