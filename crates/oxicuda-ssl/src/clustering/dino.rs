//! DINO — Caron et al. 2021 — self-distillation with no labels.
//!
//! A student network mimics the soft predictions of a teacher (a momentum-EMA
//! version of the student). The teacher output is *centred* (running batch
//! mean subtracted) and *sharpened* (low temperature), preventing collapse:
//!
//! ```text
//!     L = - Σ_k softmax(t/τ_t)_k · log softmax(s/τ_s)_k
//! ```
//!
//! where `t` is the teacher logit, `s` the student logit, `τ_t < τ_s` (e.g.
//! 0.04 vs 0.1).

use crate::error::{SslError, SslResult};

/// DINO loss configuration.
#[derive(Debug, Clone)]
pub struct DinoConfig {
    /// Student temperature (default 0.1).
    pub student_temperature: f32,
    /// Teacher temperature (default 0.04).
    pub teacher_temperature: f32,
    /// Centre EMA momentum used by the caller (default 0.9).
    pub center_momentum: f32,
}

impl Default for DinoConfig {
    fn default() -> Self {
        Self {
            student_temperature: 0.1,
            teacher_temperature: 0.04,
            center_momentum: 0.9,
        }
    }
}

impl DinoConfig {
    /// Validated config.
    ///
    /// # Errors
    /// - [`SslError::InvalidTemperature`] for non-positive temperatures.
    /// - [`SslError::InvalidMomentum`] when `center_momentum ∉ [0, 1]`.
    pub fn new(
        student_temperature: f32,
        teacher_temperature: f32,
        center_momentum: f32,
    ) -> SslResult<Self> {
        for t in [student_temperature, teacher_temperature] {
            if !(t.is_finite() && t > 0.0) {
                return Err(SslError::InvalidTemperature { temp: t });
            }
        }
        if !(center_momentum.is_finite() && (0.0..=1.0).contains(&center_momentum)) {
            return Err(SslError::InvalidMomentum {
                momentum: center_momentum,
            });
        }
        Ok(Self {
            student_temperature,
            teacher_temperature,
            center_momentum,
        })
    }
}

/// Stable per-row softmax of `[N × K]` matrix at temperature `t`.
fn row_softmax_t(scores: &[f32], n: usize, k: usize, t: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        let row = &scores[i * k..(i + 1) * k];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row {
            if v / t > max_v {
                max_v = v / t;
            }
        }
        let mut s = 0.0_f64;
        let mut tmp = Vec::with_capacity(k);
        for &v in row {
            let e = ((v / t - max_v) as f64).exp();
            tmp.push(e);
            s += e;
        }
        let inv = 1.0_f64 / s.max(1e-30);
        for v in &tmp {
            out.push((*v * inv) as f32);
        }
    }
    out
}

/// Compute the DINO student-teacher cross-entropy with centred + sharpened
/// teacher outputs.
///
/// `student_logits` and `teacher_logits` are `[N × K]` row-major. `centre` is
/// `[K]`: the running EMA of teacher means. The teacher output is centred
/// `t̂ = teacher − centre` then sharpened with `τ_t`, while the student is
/// sharpened with `τ_s` (no centring).
///
/// # Errors
/// - [`SslError::DimensionMismatch`] when shapes disagree.
/// - [`SslError::EmptyInput`] when `n == 0` or `k == 0`.
pub fn dino_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    centre: &[f32],
    n: usize,
    k: usize,
    cfg: &DinoConfig,
) -> SslResult<f32> {
    if n == 0 || k == 0 {
        return Err(SslError::EmptyInput);
    }
    if student_logits.len() != n * k {
        return Err(SslError::DimensionMismatch {
            expected: n * k,
            got: student_logits.len(),
        });
    }
    if teacher_logits.len() != n * k {
        return Err(SslError::DimensionMismatch {
            expected: n * k,
            got: teacher_logits.len(),
        });
    }
    if centre.len() != k {
        return Err(SslError::DimensionMismatch {
            expected: k,
            got: centre.len(),
        });
    }
    // Centre teacher logits.
    let mut t_centred = teacher_logits.to_vec();
    for i in 0..n {
        for j in 0..k {
            t_centred[i * k + j] -= centre[j];
        }
    }
    let p_t = row_softmax_t(&t_centred, n, k, cfg.teacher_temperature);
    let p_s = row_softmax_t(student_logits, n, k, cfg.student_temperature);
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let log_p_s = p_s[i * k + j].max(1e-12).ln();
            total += -(p_t[i * k + j] as f64) * (log_p_s as f64);
        }
    }
    Ok((total / n as f64) as f32)
}

/// Update the running centre with the EMA rule `c = m·c + (1-m)·mean(teacher)`.
///
/// `teacher_logits` is `[N × K]`; `centre` is `[K]` (modified in place).
///
/// # Errors
/// - [`SslError::DimensionMismatch`] when shapes disagree.
/// - [`SslError::InvalidMomentum`] for momentum outside `[0, 1]`.
pub fn update_dino_centre(
    centre: &mut [f32],
    teacher_logits: &[f32],
    n: usize,
    k: usize,
    momentum: f32,
) -> SslResult<()> {
    if !(momentum.is_finite() && (0.0..=1.0).contains(&momentum)) {
        return Err(SslError::InvalidMomentum { momentum });
    }
    if centre.len() != k || teacher_logits.len() != n * k || n == 0 {
        return Err(SslError::DimensionMismatch {
            expected: n * k,
            got: teacher_logits.len(),
        });
    }
    let inv_n = 1.0_f32 / n as f32;
    for j in 0..k {
        let mut mean_j = 0.0_f32;
        for i in 0..n {
            mean_j += teacher_logits[i * k + j];
        }
        mean_j *= inv_n;
        centre[j] = momentum * centre[j] + (1.0 - momentum) * mean_j;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dino_default_temperatures() {
        let cfg = DinoConfig::default();
        assert!(cfg.teacher_temperature < cfg.student_temperature);
    }

    #[test]
    fn dino_rejects_invalid_temperature() {
        assert!(DinoConfig::new(0.0, 0.04, 0.9).is_err());
        assert!(DinoConfig::new(0.1, -1.0, 0.9).is_err());
    }

    #[test]
    fn dino_rejects_invalid_momentum() {
        assert!(DinoConfig::new(0.1, 0.04, 1.5).is_err());
        assert!(DinoConfig::new(0.1, 0.04, -0.1).is_err());
    }

    #[test]
    fn dino_loss_finite_for_random_inputs() {
        let n = 4;
        let k = 8;
        let s: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let t: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.029).cos()).collect();
        let centre = vec![0.0_f32; k];
        let cfg = DinoConfig::default();
        let l = dino_loss(&s, &t, &centre, n, k, &cfg).unwrap();
        assert!(l.is_finite() && l > 0.0);
    }

    #[test]
    fn dino_loss_low_for_aligned_predictions() {
        // Sharp peaks at the same index → low cross-entropy.
        let n = 4;
        let k = 4;
        let mut s = vec![0.0_f32; n * k];
        let mut t = vec![0.0_f32; n * k];
        for i in 0..n {
            s[i * k + i] = 10.0;
            t[i * k + i] = 10.0;
        }
        let centre = vec![0.0_f32; k];
        let cfg = DinoConfig::default();
        let l = dino_loss(&s, &t, &centre, n, k, &cfg).unwrap();
        assert!(l < 1.0, "l = {l}");
    }

    #[test]
    fn dino_loss_centre_subtracts_correctly() {
        let n = 4;
        let k = 4;
        let s = vec![0.0_f32; n * k];
        let t = vec![5.0_f32; n * k];
        let centre = vec![5.0_f32; k]; // Subtracting centre → uniform teacher.
        let cfg = DinoConfig::default();
        let l = dino_loss(&s, &t, &centre, n, k, &cfg).unwrap();
        // Uniform teacher × uniform student → CE = ln(K)
        let expected = (k as f32).ln();
        assert!((l - expected).abs() < 1e-3, "l = {l}, expected {expected}");
    }

    #[test]
    fn dino_loss_dim_mismatch() {
        let s = vec![0.0_f32; 8];
        let t = vec![0.0_f32; 6];
        let c = vec![0.0_f32; 4];
        let cfg = DinoConfig::default();
        assert!(dino_loss(&s, &t, &c, 2, 4, &cfg).is_err());
    }

    #[test]
    fn update_centre_zero_momentum_replaces_with_mean() {
        let mut centre = vec![1.0_f32; 4];
        let teacher = vec![5.0_f32; 8];
        update_dino_centre(&mut centre, &teacher, 2, 4, 0.0).unwrap();
        for &v in &centre {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn update_centre_full_momentum_keeps_old_value() {
        let mut centre = vec![1.0_f32; 4];
        let teacher = vec![10.0_f32; 8];
        update_dino_centre(&mut centre, &teacher, 2, 4, 1.0).unwrap();
        for &v in &centre {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn update_centre_rejects_invalid_momentum() {
        let mut centre = vec![0.0_f32; 4];
        let teacher = vec![0.0_f32; 8];
        assert!(update_dino_centre(&mut centre, &teacher, 2, 4, 1.5).is_err());
    }
}
