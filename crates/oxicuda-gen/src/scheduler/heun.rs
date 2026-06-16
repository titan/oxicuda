//! Heun / ancestral-Euler samplers on a Karras σ schedule.
//!
//! Provides the *k-diffusion* style standalone samplers that operate directly
//! on a denoiser `D(x, σ)` returning the predicted clean sample `x₀`:
//!
//! - **`sample_euler`** — 1st-order explicit Euler over the σ schedule.
//! - **`sample_heun`** — 2nd-order Heun (improved Euler) with a trapezoidal
//!   correction; falls back to Euler at the final (σ_next = 0) step.
//! - **`sample_euler_ancestral`** — stochastic ("ancestral") Euler that injects
//!   fresh noise at each step using the `σ_up` / `σ_down` decomposition.
//!
//! These complement [`crate::scheduler::edm::EdmScheduler`]: that struct couples
//! the σ schedule with EDM *preconditioning* (c_skip/c_out/c_in/c_noise), while
//! this module exposes the bare ODE/SDE integrators over a raw `D(x, σ)` map,
//! matching the `sample_euler` / `sample_heun` / `sample_euler_ancestral`
//! routines of the reference `k-diffusion` implementation (Karras et al. 2022,
//! Algorithm 2 for the ancestral variant).
//!
//! # Karras σ schedule
//!
//! ```text
//! σᵢ = (σ_max^{1/ρ} + i/(N−1) · (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ ,  i = 0..N−1
//! ```
//!
//! with a trailing `σ = 0` appended so the trajectory terminates at the clean
//! sample. The returned schedule therefore has `N + 1` entries.
//!
//! # Reference
//! Karras et al., "Elucidating the Design Space of Diffusion-Based Generative
//! Models", NeurIPS 2022 (Algorithm 2, Eq. 5).

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── KarrasSigmas ──────────────────────────────────────────────────────────────

/// Builder for the Karras et al. 2022 power-law σ schedule used by the
/// k-diffusion samplers.
///
/// Unlike [`crate::scheduler::edm::EdmConfig`], this carries *only* the schedule
/// parameters (no preconditioning), and the produced schedule has a trailing
/// `σ = 0` to drive the final denoising step.
#[derive(Debug, Clone)]
pub struct KarrasSigmas {
    /// Lower bound σ_min (> 0).
    pub sigma_min: f32,
    /// Upper bound σ_max (> σ_min).
    pub sigma_max: f32,
    /// Curvature exponent ρ (> 0; 7.0 in the paper).
    pub rho: f32,
    /// Number of *active* σ levels N (≥ 1). The full schedule has `N + 1`
    /// entries (the final one is 0).
    pub n_steps: usize,
}

impl Default for KarrasSigmas {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            rho: 7.0,
            n_steps: 18,
        }
    }
}

impl KarrasSigmas {
    /// Build the σ schedule with `n_steps + 1` entries: a monotonically
    /// decreasing sequence from σ_max down to σ_min, followed by `0.0`.
    ///
    /// # Errors
    /// - [`GenError::InvalidGuidanceScale`] if any σ/ρ parameter is invalid.
    /// - [`GenError::EmptyInput`] if `n_steps == 0`.
    pub fn build(&self) -> GenResult<Vec<f32>> {
        if self.sigma_min <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(self.sigma_min));
        }
        if self.sigma_max <= self.sigma_min {
            return Err(GenError::InvalidGuidanceScale(self.sigma_max));
        }
        if self.rho <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(self.rho));
        }
        if self.n_steps == 0 {
            return Err(GenError::EmptyInput("n_steps must be >= 1"));
        }

        let inv_rho = 1.0 / self.rho;
        let max_pow = self.sigma_max.powf(inv_rho);
        let min_pow = self.sigma_min.powf(inv_rho);
        let n = self.n_steps;

        let mut sigmas = Vec::with_capacity(n + 1);
        for i in 0..n {
            // Guard division by zero when n == 1: single active level = σ_max.
            let frac = if n == 1 {
                0.0
            } else {
                i as f32 / (n - 1) as f32
            };
            let inner = max_pow + frac * (min_pow - max_pow);
            sigmas.push(inner.powf(self.rho));
        }
        sigmas.push(0.0);
        Ok(sigmas)
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────────

/// `d = (x − D(x, σ)) / σ` — the score-implied ODE derivative (Karras Eq. 4).
fn derivative(x: &[f32], denoised: &[f32], sigma: f32) -> Vec<f32> {
    x.iter()
        .zip(denoised)
        .map(|(&xi, &di)| (xi - di) / sigma)
        .collect()
}

/// Split σ into `(σ_down, σ_up)` for ancestral sampling (Karras Algorithm 2).
///
/// ```text
/// σ_up   = min(σ_next, η · √((σ_next² (σ_cur² − σ_next²)) / σ_cur²))
/// σ_down = √(σ_next² − σ_up²)
/// ```
fn ancestral_step_sigmas(sigma_cur: f32, sigma_next: f32, eta: f32) -> (f32, f32) {
    if sigma_next <= 0.0 {
        return (0.0, 0.0);
    }
    let s_cur2 = sigma_cur * sigma_cur;
    let s_next2 = sigma_next * sigma_next;
    let inner = (s_next2 * (s_cur2 - s_next2) / s_cur2).max(0.0);
    let sigma_up = (eta * inner.sqrt()).min(sigma_next);
    let sigma_down = (s_next2 - sigma_up * sigma_up).max(0.0).sqrt();
    (sigma_down, sigma_up)
}

// ─── Public samplers ────────────────────────────────────────────────────────────

/// Deterministic explicit-Euler sampler over the σ schedule.
///
/// At each interval `[σᵢ, σᵢ₊₁]`:
/// ```text
/// d   = (x − D(x, σᵢ)) / σᵢ
/// x ← x + (σᵢ₊₁ − σᵢ) · d
/// ```
///
/// # Arguments
/// - `x_init` — initial sample at `sigmas[0]` (= σ_max).
/// - `sigmas` — schedule from [`KarrasSigmas::build`] (`N + 1` entries, last 0).
/// - `denoiser` — `D(x, σ) → x₀` predicted-clean closure.
///
/// # Errors
/// - [`GenError::EmptyInput`] if `x_init` is empty.
/// - [`GenError::DimensionMismatch`] if `sigmas.len() < 2`.
pub fn sample_euler<F>(x_init: &[f32], sigmas: &[f32], mut denoiser: F) -> GenResult<Vec<f32>>
where
    F: FnMut(&[f32], f32) -> Vec<f32>,
{
    if x_init.is_empty() {
        return Err(GenError::EmptyInput("x_init is empty"));
    }
    if sigmas.len() < 2 {
        return Err(GenError::DimensionMismatch {
            expected: 2,
            got: sigmas.len(),
        });
    }

    let mut x = x_init.to_vec();
    for w in sigmas.windows(2) {
        let (sigma_cur, sigma_next) = (w[0], w[1]);
        if sigma_cur <= 0.0 {
            break;
        }
        let denoised = denoiser(&x, sigma_cur);
        let d = derivative(&x, &denoised, sigma_cur);
        let dt = sigma_next - sigma_cur;
        for (xi, &di) in x.iter_mut().zip(&d) {
            *xi += dt * di;
        }
    }
    Ok(x)
}

/// Deterministic 2nd-order Heun (improved-Euler) sampler.
///
/// Performs an Euler predictor then a trapezoidal corrector using the
/// derivative evaluated at the predicted point. At the terminal step
/// (`σ_next == 0`) it degrades to plain Euler (the corrector derivative is
/// undefined at σ = 0).
///
/// # Errors
/// - [`GenError::EmptyInput`] if `x_init` is empty.
/// - [`GenError::DimensionMismatch`] if `sigmas.len() < 2`.
pub fn sample_heun<F>(x_init: &[f32], sigmas: &[f32], mut denoiser: F) -> GenResult<Vec<f32>>
where
    F: FnMut(&[f32], f32) -> Vec<f32>,
{
    if x_init.is_empty() {
        return Err(GenError::EmptyInput("x_init is empty"));
    }
    if sigmas.len() < 2 {
        return Err(GenError::DimensionMismatch {
            expected: 2,
            got: sigmas.len(),
        });
    }

    let mut x = x_init.to_vec();
    for w in sigmas.windows(2) {
        let (sigma_cur, sigma_next) = (w[0], w[1]);
        if sigma_cur <= 0.0 {
            break;
        }
        let denoised = denoiser(&x, sigma_cur);
        let d = derivative(&x, &denoised, sigma_cur);
        let dt = sigma_next - sigma_cur;

        // Euler predictor.
        let x_pred: Vec<f32> = x.iter().zip(&d).map(|(&xi, &di)| xi + dt * di).collect();

        if sigma_next > 0.0 {
            // Heun corrector (trapezoidal).
            let denoised_pred = denoiser(&x_pred, sigma_next);
            let d_pred = derivative(&x_pred, &denoised_pred, sigma_next);
            for ((xi, &di), &dpi) in x.iter_mut().zip(&d).zip(&d_pred) {
                *xi += dt * 0.5 * (di + dpi);
            }
        } else {
            x = x_pred;
        }
    }
    Ok(x)
}

/// Stochastic ("ancestral") Euler sampler (Karras Algorithm 2).
///
/// Each step splits `σ_next` into `(σ_down, σ_up)`, takes a deterministic Euler
/// step to `σ_down`, then re-injects Gaussian noise scaled by `σ_up`:
/// ```text
/// x ← x_euler(σ_down) + σ_up · z ,  z ~ N(0, I)
/// ```
/// With `eta = 0` it is identical to [`sample_euler`].
///
/// # Arguments
/// - `eta` — stochasticity in `[0, 1]`; 0 = deterministic, 1 = full ancestral.
///
/// # Errors
/// - [`GenError::EmptyInput`] if `x_init` is empty.
/// - [`GenError::DimensionMismatch`] if `sigmas.len() < 2`.
/// - [`GenError::InvalidFlowTime`] if `eta` is outside `[0, 1]`.
pub fn sample_euler_ancestral<F>(
    x_init: &[f32],
    sigmas: &[f32],
    eta: f32,
    rng: &mut LcgRng,
    mut denoiser: F,
) -> GenResult<Vec<f32>>
where
    F: FnMut(&[f32], f32) -> Vec<f32>,
{
    if x_init.is_empty() {
        return Err(GenError::EmptyInput("x_init is empty"));
    }
    if sigmas.len() < 2 {
        return Err(GenError::DimensionMismatch {
            expected: 2,
            got: sigmas.len(),
        });
    }
    if !(0.0..=1.0).contains(&eta) {
        return Err(GenError::InvalidFlowTime(eta));
    }

    let mut x = x_init.to_vec();
    let mut noise = vec![0.0_f32; x.len()];
    for w in sigmas.windows(2) {
        let (sigma_cur, sigma_next) = (w[0], w[1]);
        if sigma_cur <= 0.0 {
            break;
        }
        let denoised = denoiser(&x, sigma_cur);
        let (sigma_down, sigma_up) = ancestral_step_sigmas(sigma_cur, sigma_next, eta);

        let d = derivative(&x, &denoised, sigma_cur);
        let dt = sigma_down - sigma_cur;
        for (xi, &di) in x.iter_mut().zip(&d) {
            *xi += dt * di;
        }

        if sigma_up > 0.0 {
            rng.fill_normal(&mut noise);
            for (xi, &zi) in x.iter_mut().zip(&noise) {
                *xi += sigma_up * zi;
            }
        }
    }
    Ok(x)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity denoiser `D(x, σ) = x` ⇒ derivative is 0 everywhere.
    fn identity_denoiser(x: &[f32], _sigma: f32) -> Vec<f32> {
        x.to_vec()
    }

    /// Constant-target denoiser `D(x, σ) = target` (pulls x toward target).
    fn const_denoiser(target: Vec<f32>) -> impl FnMut(&[f32], f32) -> Vec<f32> {
        move |x: &[f32], _sigma: f32| {
            // Ensure correct length irrespective of probe.
            let mut out = target.clone();
            out.truncate(x.len());
            out
        }
    }

    fn default_schedule() -> Vec<f32> {
        KarrasSigmas {
            sigma_min: 0.01,
            sigma_max: 10.0,
            rho: 7.0,
            n_steps: 12,
        }
        .build()
        .expect("valid schedule")
    }

    #[test]
    fn schedule_has_n_plus_one_entries() {
        let sigmas = KarrasSigmas {
            n_steps: 8,
            ..KarrasSigmas::default()
        }
        .build()
        .expect("valid");
        assert_eq!(sigmas.len(), 9, "schedule must have n_steps + 1 entries");
    }

    #[test]
    fn schedule_endpoints_and_terminal_zero() {
        let cfg = KarrasSigmas {
            sigma_min: 0.02,
            sigma_max: 50.0,
            rho: 7.0,
            n_steps: 20,
        };
        let sigmas = cfg.build().expect("valid");
        assert!(
            (sigmas[0] - cfg.sigma_max).abs() < 1e-3,
            "first σ should be σ_max, got {}",
            sigmas[0]
        );
        assert!(
            (sigmas[cfg.n_steps - 1] - cfg.sigma_min).abs() < 1e-3,
            "last active σ should be σ_min, got {}",
            sigmas[cfg.n_steps - 1]
        );
        assert_eq!(*sigmas.last().expect("non-empty"), 0.0);
    }

    #[test]
    fn schedule_is_monotonically_decreasing() {
        let sigmas = default_schedule();
        for w in sigmas.windows(2) {
            assert!(w[0] >= w[1], "schedule not decreasing: {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn schedule_single_step_ok() {
        // n_steps=1 must not divide by zero.
        let sigmas = KarrasSigmas {
            sigma_min: 0.1,
            sigma_max: 5.0,
            rho: 7.0,
            n_steps: 1,
        }
        .build()
        .expect("n=1 valid");
        assert_eq!(sigmas.len(), 2);
        assert!((sigmas[0] - 5.0).abs() < 1e-4);
        assert_eq!(sigmas[1], 0.0);
    }

    #[test]
    fn euler_identity_denoiser_no_change() {
        // D(x,σ)=x ⇒ d=0 ⇒ x unchanged.
        let sigmas = default_schedule();
        let x0 = vec![1.0_f32, -2.0, 3.5, 0.25];
        let out = sample_euler(&x0, &sigmas, identity_denoiser).expect("euler ok");
        for (&o, &i) in out.iter().zip(&x0) {
            assert!((o - i).abs() < 1e-5, "expected unchanged: {o} vs {i}");
        }
    }

    #[test]
    fn euler_finite_output() {
        let sigmas = default_schedule();
        let x0 = vec![0.5_f32; 16];
        let out = sample_euler(&x0, &sigmas, const_denoiser(vec![0.0_f32; 16])).expect("euler ok");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn heun_identity_denoiser_no_change() {
        let sigmas = default_schedule();
        let x0 = vec![2.0_f32, 0.0, -1.5];
        let out = sample_heun(&x0, &sigmas, identity_denoiser).expect("heun ok");
        for (&o, &i) in out.iter().zip(&x0) {
            assert!((o - i).abs() < 1e-5, "expected unchanged: {o} vs {i}");
        }
    }

    #[test]
    fn heun_converges_toward_target() {
        // With a constant target denoiser the trajectory should converge to it.
        let sigmas = KarrasSigmas {
            sigma_min: 0.002,
            sigma_max: 80.0,
            rho: 7.0,
            n_steps: 40,
        }
        .build()
        .expect("valid");
        let target = vec![1.0_f32, -1.0, 0.5, 2.0];
        let x0 = vec![20.0_f32, 20.0, 20.0, 20.0];
        let out = sample_heun(&x0, &sigmas, const_denoiser(target.clone())).expect("heun ok");
        for (&o, &t) in out.iter().zip(&target) {
            assert!(
                (o - t).abs() < 0.5,
                "Heun should converge near target: got {o}, target {t}"
            );
        }
    }

    #[test]
    fn heun_more_accurate_than_euler_on_curved_field() {
        // For a const-target denoiser the exact solution at σ→0 is the target.
        // Heun (2nd order) should be at least as accurate as Euler (1st order).
        let sigmas = KarrasSigmas {
            sigma_min: 0.01,
            sigma_max: 10.0,
            rho: 7.0,
            n_steps: 6,
        }
        .build()
        .expect("valid");
        let target = vec![0.0_f32, 0.0, 0.0, 0.0];
        let x0 = vec![5.0_f32, -5.0, 3.0, -3.0];
        let euler = sample_euler(&x0, &sigmas, const_denoiser(target.clone())).expect("ok");
        let heun = sample_heun(&x0, &sigmas, const_denoiser(target.clone())).expect("ok");
        let err_euler: f32 = euler.iter().map(|v| v.abs()).sum();
        let err_heun: f32 = heun.iter().map(|v| v.abs()).sum();
        assert!(
            err_heun <= err_euler + 1e-3,
            "Heun err {err_heun} should not exceed Euler err {err_euler}"
        );
    }

    #[test]
    fn ancestral_eta_zero_matches_euler() {
        let sigmas = default_schedule();
        let x0 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut rng = LcgRng::new(123);
        let ancestral = sample_euler_ancestral(&x0, &sigmas, 0.0, &mut rng, identity_denoiser)
            .expect("ancestral ok");
        let euler = sample_euler(&x0, &sigmas, identity_denoiser).expect("euler ok");
        for (&a, &e) in ancestral.iter().zip(&euler) {
            assert!(
                (a - e).abs() < 1e-5,
                "eta=0 ancestral should equal Euler: {a} vs {e}"
            );
        }
    }

    #[test]
    fn ancestral_injects_noise_when_eta_positive() {
        let sigmas = default_schedule();
        let x0 = vec![0.0_f32; 32];
        let mut rng1 = LcgRng::new(7);
        let mut rng2 = LcgRng::new(99);
        let out1 =
            sample_euler_ancestral(&x0, &sigmas, 1.0, &mut rng1, identity_denoiser).expect("ok");
        let out2 =
            sample_euler_ancestral(&x0, &sigmas, 1.0, &mut rng2, identity_denoiser).expect("ok");
        // Different RNG seeds ⇒ different stochastic trajectories.
        let diff: f32 = out1
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff > 1e-4, "ancestral noise should differ across seeds");
    }

    #[test]
    fn ancestral_finite_output() {
        let sigmas = default_schedule();
        let x0 = vec![1.0_f32; 16];
        let mut rng = LcgRng::new(5);
        let out =
            sample_euler_ancestral(&x0, &sigmas, 0.5, &mut rng, const_denoiser(vec![0.0; 16]))
                .expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn err_empty_x_init() {
        let sigmas = default_schedule();
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            sample_euler(&[], &sigmas, identity_denoiser),
            Err(GenError::EmptyInput(_))
        ));
        assert!(matches!(
            sample_heun(&[], &sigmas, identity_denoiser),
            Err(GenError::EmptyInput(_))
        ));
        assert!(matches!(
            sample_euler_ancestral(&[], &sigmas, 0.5, &mut rng, identity_denoiser),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_short_schedule() {
        let x0 = vec![1.0_f32; 4];
        let one = vec![1.0_f32];
        assert!(matches!(
            sample_euler(&x0, &one, identity_denoiser),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_eta_out_of_range() {
        let sigmas = default_schedule();
        let x0 = vec![1.0_f32; 4];
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            sample_euler_ancestral(&x0, &sigmas, 1.5, &mut rng, identity_denoiser),
            Err(GenError::InvalidFlowTime(_))
        ));
        assert!(matches!(
            sample_euler_ancestral(&x0, &sigmas, -0.1, &mut rng, identity_denoiser),
            Err(GenError::InvalidFlowTime(_))
        ));
    }

    #[test]
    fn err_invalid_schedule_params() {
        assert!(matches!(
            KarrasSigmas {
                sigma_min: 0.0,
                ..KarrasSigmas::default()
            }
            .build(),
            Err(GenError::InvalidGuidanceScale(_))
        ));
        assert!(matches!(
            KarrasSigmas {
                sigma_min: 5.0,
                sigma_max: 1.0,
                ..KarrasSigmas::default()
            }
            .build(),
            Err(GenError::InvalidGuidanceScale(_))
        ));
        assert!(matches!(
            KarrasSigmas {
                n_steps: 0,
                ..KarrasSigmas::default()
            }
            .build(),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn ancestral_sigma_split_invariant() {
        // σ_down² + σ_up² == σ_next²  (within float tolerance).
        let (down, up) = ancestral_step_sigmas(5.0, 3.0, 1.0);
        let lhs = down * down + up * up;
        assert!((lhs - 9.0).abs() < 1e-3, "σ_down²+σ_up²={lhs}, expected 9");
    }
}
