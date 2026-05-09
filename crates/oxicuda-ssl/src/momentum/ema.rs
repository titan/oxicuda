//! Exponential moving average (EMA) updater for momentum-encoder schemes.
//!
//! BYOL, MoCo, and DINO all maintain a *target* network whose weights are an
//! EMA of the *online* network's weights:
//! ```text
//!     θ_target ← m·θ_target + (1 − m)·θ_online
//! ```
//! with momentum `m` typically scheduled with a cosine ramp from `m_base` to
//! `m_end` over training (e.g. 0.996 → 1.0 in BYOL).

use crate::error::{SslError, SslResult};

/// EMA updater operating on flat parameter buffers.
#[derive(Debug, Clone)]
pub struct EmaUpdater {
    /// Last-applied momentum value.
    pub last_momentum: f32,
}

impl Default for EmaUpdater {
    fn default() -> Self {
        Self {
            last_momentum: 0.99,
        }
    }
}

impl EmaUpdater {
    /// Construct a new updater. The momentum value is supplied per call to
    /// [`Self::update`] so the caller controls scheduling.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply `θ_target = m·θ_target + (1−m)·θ_online` element-wise.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when buffers differ in length.
    /// - [`SslError::InvalidMomentum`] for `m` outside `[0, 1]` or non-finite.
    pub fn update(&mut self, target: &mut [f32], online: &[f32], momentum: f32) -> SslResult<()> {
        if !(momentum.is_finite() && (0.0..=1.0).contains(&momentum)) {
            return Err(SslError::InvalidMomentum { momentum });
        }
        if target.len() != online.len() {
            return Err(SslError::DimensionMismatch {
                expected: target.len(),
                got: online.len(),
            });
        }
        let one_minus_m = 1.0 - momentum;
        for (t, &o) in target.iter_mut().zip(online.iter()) {
            *t = momentum * *t + one_minus_m * o;
        }
        self.last_momentum = momentum;
        Ok(())
    }
}

/// Cosine momentum schedule used by BYOL / DINO:
/// `m(t) = m_end − (m_end − m_base) · cos(π t / T) / 2 − (m_end − m_base) / 2`.
///
/// Equivalently: ramps from `m_base` (at t=0) to `m_end` (at t=T) following a
/// half-cosine curve.
///
/// # Errors
/// - [`SslError::InvalidMomentum`] if either bound is outside `[0, 1]`.
/// - [`SslError::Internal`] if `max_steps == 0`.
pub fn cosine_momentum(step: usize, max_steps: usize, m_base: f32, m_end: f32) -> SslResult<f32> {
    if !(m_base.is_finite() && (0.0..=1.0).contains(&m_base)) {
        return Err(SslError::InvalidMomentum { momentum: m_base });
    }
    if !(m_end.is_finite() && (0.0..=1.0).contains(&m_end)) {
        return Err(SslError::InvalidMomentum { momentum: m_end });
    }
    if max_steps == 0 {
        return Err(SslError::Internal("cosine_momentum: max_steps == 0".into()));
    }
    let t = step.min(max_steps) as f32 / max_steps as f32;
    let theta = std::f32::consts::PI * t;
    Ok(m_end - (m_end - m_base) * (theta.cos() + 1.0) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_zero_momentum_replaces_target() {
        let mut updater = EmaUpdater::new();
        let mut target = vec![0.0_f32, 1.0, 2.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        updater.update(&mut target, &online, 0.0).unwrap();
        assert_eq!(target, online);
        assert!((updater.last_momentum - 0.0).abs() < 1e-7);
    }

    #[test]
    fn ema_full_momentum_keeps_target() {
        let mut updater = EmaUpdater::new();
        let mut target = vec![0.0_f32, 1.0, 2.0];
        let online = vec![10.0_f32, 20.0, 30.0];
        updater.update(&mut target, &online, 1.0).unwrap();
        assert_eq!(target, vec![0.0_f32, 1.0, 2.0]);
    }

    #[test]
    fn ema_half_momentum_averages() {
        let mut updater = EmaUpdater::new();
        let mut target = vec![0.0_f32, 0.0, 0.0];
        let online = vec![2.0_f32, 4.0, 6.0];
        updater.update(&mut target, &online, 0.5).unwrap();
        assert_eq!(target, vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn ema_rejects_invalid_momentum() {
        let mut updater = EmaUpdater::new();
        let mut target = vec![0.0_f32; 3];
        let online = vec![1.0_f32; 3];
        assert!(updater.update(&mut target, &online, -0.1).is_err());
        assert!(updater.update(&mut target, &online, 1.5).is_err());
        assert!(updater.update(&mut target, &online, f32::NAN).is_err());
    }

    #[test]
    fn ema_rejects_dim_mismatch() {
        let mut updater = EmaUpdater::new();
        let mut target = vec![0.0_f32; 3];
        let online = vec![1.0_f32; 5];
        assert!(updater.update(&mut target, &online, 0.5).is_err());
    }

    #[test]
    fn cosine_momentum_at_step_zero_equals_base() {
        let m = cosine_momentum(0, 100, 0.5, 1.0).unwrap();
        assert!((m - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cosine_momentum_at_max_step_equals_end() {
        let m = cosine_momentum(100, 100, 0.5, 1.0).unwrap();
        assert!((m - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_momentum_monotone() {
        let m1 = cosine_momentum(10, 100, 0.5, 1.0).unwrap();
        let m2 = cosine_momentum(50, 100, 0.5, 1.0).unwrap();
        let m3 = cosine_momentum(90, 100, 0.5, 1.0).unwrap();
        assert!(m1 < m2);
        assert!(m2 < m3);
    }

    #[test]
    fn cosine_momentum_rejects_invalid_bounds() {
        assert!(cosine_momentum(0, 100, -0.1, 1.0).is_err());
        assert!(cosine_momentum(0, 100, 0.5, 1.5).is_err());
    }

    #[test]
    fn cosine_momentum_rejects_zero_steps() {
        assert!(cosine_momentum(0, 0, 0.5, 1.0).is_err());
    }
}
