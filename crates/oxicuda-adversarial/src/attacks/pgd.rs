//! Projected Gradient Descent (PGD) attack.
//!
//! Iterative L∞ / L2 attack from
//! Madry, Makelov, Schmidt, Tsipras & Vladu (2018),
//! *"Towards Deep Learning Models Resistant to Adversarial Attacks"*, ICLR.
//!
//! Each step takes a signed (L∞) or normalised (L2) gradient ascent step of
//! size `α`, then projects back onto the ε-ball around the original input
//! and clamps to the box `[lo, hi]`. With `random_start = true` the iterate
//! is initialised uniformly at random inside the ε-ball, mirroring the
//! "PGD-rand" baseline that has become the de-facto robustness benchmark.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;
use crate::threat_model::lp_ball::{l2_norm, project_l_inf, project_l2};

/// Hyperparameters for PGD.
///
/// Construct with [`PgdConfig::new`] which validates all fields up-front.
#[derive(Debug, Clone, Copy)]
pub struct PgdConfig {
    /// L∞ / L2 perturbation budget (positive, finite).
    pub eps: f32,
    /// Per-step size (positive, finite).
    pub alpha: f32,
    /// Number of iterations (≥ 1).
    pub n_steps: usize,
    /// If `true`, draw a uniform initial perturbation inside the ε-ball.
    pub random_start: bool,
}

impl PgdConfig {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]  — non-finite or non-positive `eps`.
    /// * [`AdvError::InvalidAlpha`]    — non-finite or non-positive `alpha`.
    /// * [`AdvError::InvalidNumSteps`] — `n_steps == 0`.
    pub fn new(eps: f32, alpha: f32, n_steps: usize, random_start: bool) -> AdvResult<Self> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps });
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(AdvError::InvalidAlpha { alpha });
        }
        if n_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        Ok(Self {
            eps,
            alpha,
            n_steps,
            random_start,
        })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Validate common inputs for both PGD variants.
fn validate(x: &[f32], lo: f32, hi: f32) -> AdvResult<()> {
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }
    Ok(())
}

/// Validate the gradient returned from a closure.
fn check_grad(g: &[f32], expected_len: usize, where_: &'static str) -> AdvResult<()> {
    if g.len() != expected_len {
        return Err(AdvError::DimensionMismatch {
            expected: expected_len,
            got: g.len(),
        });
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered { location: where_ });
    }
    Ok(())
}

/// Element-wise sign with `0.0` for zeros (avoids `f32::signum` returning
/// `±0` and silently flipping sign).
#[inline]
fn sign_inplace(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = if *v > 0.0 {
            1.0
        } else if *v < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

/// Uniform L∞ random init inside `[-eps, +eps]`.
fn rand_init_l_inf(x: &[f32], eps: f32, lo: f32, hi: f32, rng: &mut LcgRng) -> Vec<f32> {
    x.iter()
        .map(|&xi| (xi + (2.0 * rng.next_f32() - 1.0) * eps).clamp(lo, hi))
        .collect()
}

/// Uniform L2 random init inside the ε-ball (sphere-uniform · radius^(1/n)).
fn rand_init_l2(x: &[f32], eps: f32, lo: f32, hi: f32, rng: &mut LcgRng) -> Vec<f32> {
    let n = x.len();
    let mut delta = vec![0.0_f32; n];
    rng.fill_normal(&mut delta);
    let nrm = l2_norm(&delta).max(1e-12);
    // Radius scaled by U^(1/n) for uniformity in the ball volume.
    let u = rng.next_f32().max(1e-12);
    let r = eps * u.powf(1.0 / (n as f32));
    let scale = r / nrm;
    x.iter()
        .zip(delta.iter())
        .map(|(&xi, &di)| (xi + scale * di).clamp(lo, hi))
        .collect()
}

// ─── L∞ PGD ──────────────────────────────────────────────────────────────────

/// Run L∞ PGD.
///
/// # Errors
/// Mirrors the validation errors of [`PgdConfig::new`] plus
/// [`AdvError::EmptyInput`], [`AdvError::DimensionMismatch`] and
/// [`AdvError::NanEncountered`] on invalid gradient outputs.
pub fn pgd_attack_l_inf<F>(
    x: &[f32],
    lo: f32,
    hi: f32,
    cfg: &PgdConfig,
    rng: &mut LcgRng,
    loss_grad: F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    validate(x, lo, hi)?;
    let n = x.len();

    let mut adv = if cfg.random_start {
        rand_init_l_inf(x, cfg.eps, lo, hi, rng)
    } else {
        x.to_vec()
    };
    // Always project the (possibly-randomised) start onto the ε-ball.
    adv = project_l_inf(&adv, x, cfg.eps, lo, hi)?;

    for _ in 0..cfg.n_steps {
        let mut g = loss_grad(&adv)?;
        check_grad(&g, n, "pgd_attack_l_inf:loss_grad")?;
        sign_inplace(&mut g);
        let stepped: Vec<f32> = adv
            .iter()
            .zip(g.iter())
            .map(|(&xi, &gi)| xi + cfg.alpha * gi)
            .collect();
        adv = project_l_inf(&stepped, x, cfg.eps, lo, hi)?;
    }
    Ok(adv)
}

// ─── L2 PGD ──────────────────────────────────────────────────────────────────

/// Run L2 PGD.
///
/// At every step the gradient is L2-normalised before scaling by `α`,
/// matching the canonical Madry et al. formulation.
///
/// # Errors
/// Same as [`pgd_attack_l_inf`].
pub fn pgd_attack_l2<F>(
    x: &[f32],
    lo: f32,
    hi: f32,
    cfg: &PgdConfig,
    rng: &mut LcgRng,
    loss_grad: F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    validate(x, lo, hi)?;
    let n = x.len();

    let mut adv = if cfg.random_start {
        rand_init_l2(x, cfg.eps, lo, hi, rng)
    } else {
        x.to_vec()
    };
    adv = project_l2(&adv, x, cfg.eps, lo, hi)?;

    for _ in 0..cfg.n_steps {
        let g = loss_grad(&adv)?;
        check_grad(&g, n, "pgd_attack_l2:loss_grad")?;
        let nrm = l2_norm(&g).max(1e-12);
        let stepped: Vec<f32> = adv
            .iter()
            .zip(g.iter())
            .map(|(&xi, &gi)| xi + cfg.alpha * gi / nrm)
            .collect();
        adv = project_l2(&stepped, x, cfg.eps, lo, hi)?;
    }
    Ok(adv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threat_model::lp_ball::{l_inf_norm, l2_norm};

    fn quad_grad(target: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| Ok(x.iter().zip(target.iter()).map(|(a, b)| a - b).collect())
    }

    fn const_grad(g: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |_x: &[f32]| Ok(g.clone())
    }

    #[test]
    fn config_new_validates() {
        assert!(PgdConfig::new(0.1, 0.01, 10, true).is_ok());
        assert!(PgdConfig::new(-0.1, 0.01, 10, true).is_err());
        assert!(PgdConfig::new(0.1, 0.0, 10, true).is_err());
        assert!(PgdConfig::new(0.1, 0.01, 0, true).is_err());
        assert!(PgdConfig::new(f32::NAN, 0.01, 10, true).is_err());
    }

    #[test]
    fn smoke_l_inf_no_random_start() {
        let target = vec![0.5_f32; 5];
        let x = vec![0.5_f32; 5];
        let mut rng = LcgRng::new(0);
        let cfg = PgdConfig::new(0.1, 0.02, 10, false).expect("new should succeed");
        let y = pgd_attack_l_inf(&x, -10.0, 10.0, &cfg, &mut rng, quad_grad(target.clone()))
            .expect("value should be present");
        // x == target ⇒ zero grad ⇒ y == x.
        for v in &y {
            assert!((*v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn n_steps_one_l_inf_matches_fgsm() {
        let x = vec![0.5_f32; 4];
        let g = vec![1.0_f32, -1.0, 0.5, -0.5];
        let cfg = PgdConfig::new(0.05, 0.05, 1, false).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let y = pgd_attack_l_inf(&x, -10.0, 10.0, &cfg, &mut rng, const_grad(g))
            .expect("value should be present");
        // n=1, alpha=eps, no random start ⇒ pure FGSM step.
        let expected = [0.55_f32, 0.45, 0.55, 0.45];
        for (a, b) in y.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn random_start_changes_iterate_l_inf() {
        let x = vec![0.5_f32; 16];
        let cfg = PgdConfig::new(0.2, 0.01, 1, true).expect("new should succeed");
        let mut rng = LcgRng::new(123);
        // Zero gradient ⇒ output equals the (random) starting iterate.
        let y = pgd_attack_l_inf(&x, -10.0, 10.0, &cfg, &mut rng, const_grad(vec![0.0; 16]))
            .expect("value should be present");
        let any_diff = y.iter().zip(x.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff);
    }

    #[test]
    fn projection_enforced_l_inf() {
        let target = vec![10.0_f32; 8]; // huge gradients
        let x = vec![0.5_f32; 8];
        let cfg = PgdConfig::new(0.1, 0.05, 50, true).expect("new should succeed");
        let mut rng = LcgRng::new(7);
        let y = pgd_attack_l_inf(&x, 0.0, 1.0, &cfg, &mut rng, quad_grad(target))
            .expect("value should be present");
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l_inf_norm(&delta) <= 0.1 + 1e-5);
        for v in &y {
            assert!(*v >= 0.0 - 1e-6 && *v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn smoke_l2_random_start() {
        let target = vec![1.0_f32; 6];
        let x = vec![0.0_f32; 6];
        let cfg = PgdConfig::new(0.5, 0.1, 8, true).expect("new should succeed");
        let mut rng = LcgRng::new(99);
        let y = pgd_attack_l2(&x, -10.0, 10.0, &cfg, &mut rng, quad_grad(target))
            .expect("value should be present");
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l2_norm(&delta) <= 0.5 + 1e-5);
    }

    #[test]
    fn projection_enforced_l2() {
        let target = vec![10.0_f32; 12];
        let x = vec![0.5_f32; 12];
        let cfg = PgdConfig::new(0.3, 0.2, 30, true).expect("new should succeed");
        let mut rng = LcgRng::new(11);
        let y = pgd_attack_l2(&x, 0.0, 1.0, &cfg, &mut rng, quad_grad(target))
            .expect("value should be present");
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l2_norm(&delta) <= 0.3 + 1e-4);
        for v in &y {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn dim_mismatch_in_grad() {
        let x = vec![0.0_f32; 4];
        let cfg = PgdConfig::new(0.1, 0.05, 3, false).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let bad = const_grad(vec![1.0_f32; 3]);
        assert!(matches!(
            pgd_attack_l_inf(&x, -1.0, 1.0, &cfg, &mut rng, bad).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn empty_input_rejected() {
        let x: Vec<f32> = vec![];
        let cfg = PgdConfig::new(0.1, 0.05, 3, false).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        assert_eq!(
            pgd_attack_l_inf(&x, -1.0, 1.0, &cfg, &mut rng, const_grad(vec![])).unwrap_err(),
            AdvError::EmptyInput
        );
        assert_eq!(
            pgd_attack_l2(&x, -1.0, 1.0, &cfg, &mut rng, const_grad(vec![])).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn nan_grad_caught_l_inf() {
        let x = vec![0.0_f32; 3];
        let cfg = PgdConfig::new(0.1, 0.05, 1, false).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let bad = const_grad(vec![1.0, f32::INFINITY, 1.0]);
        assert!(matches!(
            pgd_attack_l_inf(&x, -1.0, 1.0, &cfg, &mut rng, bad).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let x = vec![0.5_f32; 8];
        let target = vec![1.0_f32; 8];
        let cfg = PgdConfig::new(0.2, 0.05, 10, true).expect("new should succeed");
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let y1 = pgd_attack_l_inf(&x, 0.0, 1.0, &cfg, &mut r1, quad_grad(target.clone()))
            .expect("value should be present");
        let y2 = pgd_attack_l_inf(&x, 0.0, 1.0, &cfg, &mut r2, quad_grad(target))
            .expect("value should be present");
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
