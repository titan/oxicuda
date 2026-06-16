//! Auto-PGD (APGD) — parameter-free PGD with adaptive step size.
//!
//! Reference: Croce & Hein (2020), *"Reliable Evaluation of Adversarial
//! Robustness with an Ensemble of Diverse Parameter-free Attacks"*, ICML.
//!
//! Key features that distinguish AutoPGD from vanilla PGD:
//!
//! 1. **Random L∞ start** uniformly in the ε-ball.
//! 2. **Initial step** `α₀ = 2 · ε`.
//! 3. **Checkpoint schedule** `W = {w_0, w_1, …}` with `w_0 = 0` and
//!    `w_j = ⌈p_j · n_steps⌉` where `p_j` decays geometrically:
//!    `p_{j+1} = max(p_j − 0.03, 0.06)` (Croce & Hein §3.1).
//! 4. **Step-size halving** at every checkpoint when, since the previous
//!    checkpoint, the loss has improved on fewer than `ρ · (w_j − w_{j−1})`
//!    iterations (`ρ = 0.75`). On a halve, the iterate is reset to the
//!    best-so-far point.
//!
//! The implementation here is the L∞ variant. The closure returns
//! `(loss, gradient)` so APGD can monitor improvement at each step.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;
use crate::threat_model::lp_ball::project_l_inf;

/// Hyperparameters for AutoPGD-L∞.
#[derive(Debug, Clone, Copy)]
pub struct AutoPgdConfig {
    /// L∞ perturbation budget.
    pub eps: f32,
    /// Total number of iterations (≥ 2 for any halving to take effect).
    pub n_steps: usize,
    /// Initial checkpoint ratio `p₀`. Croce & Hein use `0.22`.
    pub checkpoint_ratio: f32,
}

impl AutoPgdConfig {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]    — non-finite or non-positive `eps`.
    /// * [`AdvError::InvalidNumSteps`]   — `n_steps == 0`.
    /// * [`AdvError::InvalidLossWeight`] — `checkpoint_ratio` not in `(0, 1)`.
    pub fn new(eps: f32, n_steps: usize, checkpoint_ratio: f32) -> AdvResult<Self> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps });
        }
        if n_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        if !(checkpoint_ratio.is_finite() && checkpoint_ratio > 0.0 && checkpoint_ratio < 1.0) {
            return Err(AdvError::InvalidLossWeight {
                weight: checkpoint_ratio,
            });
        }
        Ok(Self {
            eps,
            n_steps,
            checkpoint_ratio,
        })
    }
}

impl Default for AutoPgdConfig {
    fn default() -> Self {
        Self {
            eps: 8.0 / 255.0,
            n_steps: 100,
            checkpoint_ratio: 0.22,
        }
    }
}

/// Build the Croce & Hein checkpoint schedule from the configuration.
fn build_checkpoints(n_steps: usize, p0: f32) -> Vec<usize> {
    let mut points = vec![0_usize];
    let mut p = p0;
    loop {
        let next = ((p as f64) * (n_steps as f64)).ceil() as usize;
        let next = next.max(*points.last().unwrap_or(&0) + 1);
        if next >= n_steps {
            points.push(n_steps);
            break;
        }
        points.push(next);
        p = (p - 0.03).max(0.06);
    }
    points
}

/// Run AutoPGD-L∞.
///
/// # Errors
/// * [`AdvError::EmptyInput`] — empty input.
/// * [`AdvError::InvalidLossWeight`] — degenerate box.
/// * [`AdvError::DimensionMismatch`] — bad gradient size.
/// * [`AdvError::NanEncountered`] — non-finite loss or gradient.
pub fn auto_pgd_attack<F>(
    x: &[f32],
    lo: f32,
    hi: f32,
    cfg: &AutoPgdConfig,
    rng: &mut LcgRng,
    loss_grad: F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<(f32, Vec<f32>)>,
{
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }

    let n = x.len();
    let rho = 0.75_f32;

    // Random L∞ start.
    let mut adv: Vec<f32> = x
        .iter()
        .map(|&xi| (xi + (2.0 * rng.next_f32() - 1.0) * cfg.eps).clamp(lo, hi))
        .collect();
    adv = project_l_inf(&adv, x, cfg.eps, lo, hi)?;

    // Initial step size.
    let mut alpha = 2.0 * cfg.eps;

    // Evaluate initial point.
    let (l0, g0) = loss_grad(&adv)?;
    if g0.len() != n {
        return Err(AdvError::DimensionMismatch {
            expected: n,
            got: g0.len(),
        });
    }
    if !l0.is_finite() || g0.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "auto_pgd_attack:initial",
        });
    }

    let mut best = adv.clone();
    let mut best_loss = l0;
    let mut prev_loss = l0;
    let mut grad = g0;

    let checkpoints = build_checkpoints(cfg.n_steps, cfg.checkpoint_ratio);
    let mut next_ckpt_idx = 1_usize;
    let mut improvements_in_window: usize = 0;

    for k in 0..cfg.n_steps {
        // Take a signed step using the current gradient.
        let stepped: Vec<f32> = adv
            .iter()
            .zip(grad.iter())
            .map(|(&xi, &gi)| {
                let s = if gi > 0.0 {
                    1.0_f32
                } else if gi < 0.0 {
                    -1.0_f32
                } else {
                    0.0_f32
                };
                xi + alpha * s
            })
            .collect();
        let candidate = project_l_inf(&stepped, x, cfg.eps, lo, hi)?;

        // Evaluate the candidate.
        let (l_new, g_new) = loss_grad(&candidate)?;
        if g_new.len() != n {
            return Err(AdvError::DimensionMismatch {
                expected: n,
                got: g_new.len(),
            });
        }
        if !l_new.is_finite() || g_new.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "auto_pgd_attack:step",
            });
        }

        // Track improvement (we MAXIMISE the loss for an attack).
        if l_new > prev_loss {
            improvements_in_window += 1;
        }
        if l_new > best_loss {
            best_loss = l_new;
            best = candidate.clone();
        }
        prev_loss = l_new;
        adv = candidate;
        grad = g_new;

        // Step k just completed. Iteration index for checkpoint comparison
        // is k + 1 (i.e. number of steps performed).
        let steps_done = k + 1;
        if next_ckpt_idx < checkpoints.len() && steps_done == checkpoints[next_ckpt_idx] {
            let window = checkpoints[next_ckpt_idx] - checkpoints[next_ckpt_idx - 1];
            // Halve step size if the loss did not improve sufficiently.
            let threshold = (rho * (window as f32)).floor() as usize;
            if improvements_in_window < threshold {
                alpha *= 0.5;
                // Reset to best-so-far on halving.
                adv = best.clone();
                let (l_reset, g_reset) = loss_grad(&adv)?;
                if g_reset.len() != n {
                    return Err(AdvError::DimensionMismatch {
                        expected: n,
                        got: g_reset.len(),
                    });
                }
                if !l_reset.is_finite() || g_reset.iter().any(|v| !v.is_finite()) {
                    return Err(AdvError::NanEncountered {
                        location: "auto_pgd_attack:reset",
                    });
                }
                prev_loss = l_reset;
                grad = g_reset;
            }
            improvements_in_window = 0;
            next_ckpt_idx += 1;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threat_model::lp_ball::l_inf_norm;

    /// Toy negative-quadratic loss: `L(x) = −½‖x − target‖²`. Gradient
    /// `−(x − target)`. *Maximising* this drives `x` toward `target`.
    fn neg_quad(target: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<(f32, Vec<f32>)> {
        move |x: &[f32]| {
            let l: f32 = x
                .iter()
                .zip(target.iter())
                .map(|(a, b)| -0.5 * (a - b).powi(2))
                .sum();
            let g: Vec<f32> = x.iter().zip(target.iter()).map(|(a, b)| -(a - b)).collect();
            Ok((l, g))
        }
    }

    /// Quadratic loss `L(x) = ½‖x − target‖²` to be MAXIMISED — gradient
    /// `x − target`. The attack should move x AWAY from target.
    fn quad(target: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<(f32, Vec<f32>)> {
        move |x: &[f32]| {
            let l: f32 = x
                .iter()
                .zip(target.iter())
                .map(|(a, b)| 0.5 * (a - b).powi(2))
                .sum();
            let g: Vec<f32> = x.iter().zip(target.iter()).map(|(a, b)| a - b).collect();
            Ok((l, g))
        }
    }

    #[test]
    fn config_validation() {
        assert!(AutoPgdConfig::new(0.1, 10, 0.22).is_ok());
        assert!(AutoPgdConfig::new(-0.1, 10, 0.22).is_err());
        assert!(AutoPgdConfig::new(0.1, 0, 0.22).is_err());
        assert!(AutoPgdConfig::new(0.1, 10, 0.0).is_err());
        assert!(AutoPgdConfig::new(0.1, 10, 1.0).is_err());
        assert!(AutoPgdConfig::new(f32::NAN, 10, 0.22).is_err());
    }

    #[test]
    fn checkpoints_monotone_and_bounded() {
        let cps = build_checkpoints(100, 0.22);
        assert_eq!(cps[0], 0);
        for w in cps.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert!(*cps.last().expect("last should succeed") == 100);
    }

    #[test]
    fn smoke_increases_loss() {
        let target = vec![1.0_f32; 4];
        let x = vec![0.5_f32; 4];
        let cfg = AutoPgdConfig::new(0.2, 30, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(7);
        let l_initial = quad(target.clone())(&x).expect("value should be present").0;
        let y = auto_pgd_attack(&x, 0.0, 1.0, &cfg, &mut rng, quad(target.clone()))
            .expect("value should be present");
        let l_final = quad(target)(&y).expect("value should be present").0;
        // Loss must strictly increase (we maximise).
        assert!(l_final >= l_initial);
    }

    #[test]
    fn projection_enforced() {
        let target = vec![10.0_f32; 6];
        let x = vec![0.5_f32; 6];
        let cfg = AutoPgdConfig::new(0.05, 60, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(99);
        let y = auto_pgd_attack(&x, 0.0, 1.0, &cfg, &mut rng, quad(target))
            .expect("value should be present");
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l_inf_norm(&delta) <= 0.05 + 1e-5);
        for v in &y {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn empty_input_rejected() {
        let x: Vec<f32> = vec![];
        let cfg = AutoPgdConfig::new(0.1, 5, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let cls = |_x: &[f32]| Ok((0.0_f32, vec![]));
        assert_eq!(
            auto_pgd_attack(&x, -1.0, 1.0, &cfg, &mut rng, cls).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn nan_loss_caught() {
        let x = vec![0.0_f32; 3];
        let cfg = AutoPgdConfig::new(0.1, 5, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let cls = |_x: &[f32]| Ok((f32::NAN, vec![1.0_f32; 3]));
        assert!(matches!(
            auto_pgd_attack(&x, -1.0, 1.0, &cfg, &mut rng, cls).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn dim_mismatch_caught() {
        let x = vec![0.0_f32; 4];
        let cfg = AutoPgdConfig::new(0.1, 5, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(0);
        let cls = |_x: &[f32]| Ok((0.0_f32, vec![1.0_f32; 3]));
        assert!(matches!(
            auto_pgd_attack(&x, -1.0, 1.0, &cfg, &mut rng, cls).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn best_loss_monotone() {
        // The returned point's loss must be ≥ the initial random-start loss
        // since we always track best-so-far.
        let target = vec![1.0_f32; 5];
        let x = vec![0.0_f32; 5];
        let cfg = AutoPgdConfig::new(0.3, 20, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(123);
        // Initial random start loss.
        let mut rng_init = LcgRng::new(123);
        let init: Vec<f32> = x
            .iter()
            .map(|&xi| (xi + (2.0 * rng_init.next_f32() - 1.0) * cfg.eps).clamp(-10.0, 10.0))
            .collect();
        let l_init = quad(target.clone())(&init)
            .expect("value should be present")
            .0;
        let y = auto_pgd_attack(&x, -10.0, 10.0, &cfg, &mut rng, quad(target.clone()))
            .expect("value should be present");
        let l_final = quad(target)(&y).expect("value should be present").0;
        assert!(l_final >= l_init - 1e-5);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let target = vec![1.0_f32; 4];
        let x = vec![0.5_f32; 4];
        let cfg = AutoPgdConfig::new(0.1, 15, 0.22).expect("new should succeed");
        let mut r1 = LcgRng::new(2024);
        let mut r2 = LcgRng::new(2024);
        let y1 = auto_pgd_attack(&x, 0.0, 1.0, &cfg, &mut r1, quad(target.clone()))
            .expect("value should be present");
        let y2 = auto_pgd_attack(&x, 0.0, 1.0, &cfg, &mut r2, quad(target))
            .expect("value should be present");
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn neg_quad_attracts_target() {
        // Gradient ascent on −½‖x − target‖² ⇒ should move toward target.
        let target = vec![0.7_f32; 3];
        let x = vec![0.5_f32; 3];
        let cfg = AutoPgdConfig::new(0.3, 30, 0.22).expect("new should succeed");
        let mut rng = LcgRng::new(7);
        let y = auto_pgd_attack(&x, 0.0, 1.0, &cfg, &mut rng, neg_quad(target.clone()))
            .expect("value should be present");
        let dist_before: f32 = x
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let dist_after: f32 = y
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(dist_after <= dist_before + 1e-5);
    }
}
