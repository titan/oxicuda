//! Stochastic Interpolants scheduler.
//!
//! Implements the generalisation of flow matching of Albergo &
//! Vanden-Eijnden 2023, "Stochastic Interpolants: A Unifying Framework for
//! Flows and Diffusions". A stochastic interpolant connects two endpoint
//! distributions through a time-dependent coupling
//!
//! ```text
//! X_t = α(t) · x0 + β(t) · x1 + σ(t) · z ,  t ∈ [0, 1] ,  z ~ N(0, I) .
//! ```
//!
//! The deterministic part of the corresponding velocity field is
//!
//! ```text
//! v⋆(X_t, t) = α'(t) · x0 + β'(t) · x1 .
//! ```
//!
//! The sampling SDE/ODE drift is learned as a regression target against
//! `v⋆`. This scheduler returns the deterministic velocity target and
//! integrates the generative ODE `dX/dt = v(X, t)` from `t = 0` to `t = 1`
//! with a forward Euler scheme.
//!
//! The framework specialises to known schemes by choosing `(α, β, σ)`:
//!
//! * `LinearFlow`: `α = 1 − t`, `β = t`, `σ = 0` — recovers Rectified
//!   Flow / linear flow matching.
//! * `TrigInterpolant`: `α = cos(π t / 2)`, `β = sin(π t / 2)`, `σ = 0` —
//!   trigonometric interpolant with the orthonormal identity
//!   `α² + β² ≡ 1`.
//! * `NoisyLinear { sigma_scale }`: `α = 1 − t`, `β = t`,
//!   `σ = sigma_scale · √(t · (1 − t))` — a linear interpolant with a
//!   noise channel that vanishes at the endpoints (a minimal diffusion
//!   bridge).
//!
//! # Reference
//! Albergo & Vanden-Eijnden, "Stochastic Interpolants: A Unifying
//! Framework for Flows and Diffusions", 2023.

use crate::error::{GenError, GenResult};

// ─── InterpolantKind ─────────────────────────────────────────────────────────

/// Choice of the interpolant coefficient triple `(α(t), β(t), σ(t))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolantKind {
    /// Linear flow: `α = 1 − t`, `β = t`, `σ = 0`. Recovers Rectified Flow.
    LinearFlow,
    /// Trigonometric interpolant: `α = cos(π t / 2)`, `β = sin(π t / 2)`,
    /// `σ = 0`. Satisfies the orthonormal identity `α² + β² ≡ 1`.
    TrigInterpolant,
    /// Noisy linear interpolant: `α = 1 − t`, `β = t`, and
    /// `σ(t) = sigma_scale · √(t · (1 − t))`. `σ` vanishes at both
    /// endpoints and peaks at `t = 0.5`.
    NoisyLinear {
        /// Multiplicative scale of the noise channel; `sigma_scale ≥ 0`.
        sigma_scale: f32,
    },
}

// ─── InterpolantConfig ───────────────────────────────────────────────────────

/// Configuration for [`StochasticInterpolant`].
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolantConfig {
    /// Dimensionality of the state vectors. Must be ≥ 1.
    pub dim: usize,
    /// Number of Euler integration steps used by [`StochasticInterpolant::sample_ode`].
    /// Must be ≥ 1.
    pub n_steps: usize,
    /// Choice of `(α, β, σ)`.
    pub kind: InterpolantKind,
}

// ─── StochasticInterpolant ───────────────────────────────────────────────────

/// Scheduler for stochastic interpolants.
///
/// Provides the interpolant coefficients `α, β, σ` and their derivatives
/// `α', β'`, the interpolant sample `X_t = α x0 + β x1 + σ z`, the
/// deterministic velocity target `v⋆ = α' x0 + β' x1`, and an Euler ODE
/// sampler for the generative path.
///
/// # Reference
/// Albergo & Vanden-Eijnden, "Stochastic Interpolants: A Unifying
/// Framework for Flows and Diffusions", 2023.
#[derive(Debug, Clone)]
pub struct StochasticInterpolant {
    /// Configuration controlling dimensionality, step count, and the
    /// interpolant family.
    cfg: InterpolantConfig,
}

impl StochasticInterpolant {
    /// Build a new [`StochasticInterpolant`] from the given configuration.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `dim == 0` or `n_steps == 0`.
    /// - [`GenError::Internal`] if a `NoisyLinear { sigma_scale }` carries a
    ///   negative `sigma_scale`.
    pub fn new(cfg: InterpolantConfig) -> GenResult<Self> {
        if cfg.dim == 0 {
            return Err(GenError::EmptyInput("dim must be >= 1"));
        }
        if cfg.n_steps == 0 {
            return Err(GenError::EmptyInput("n_steps must be >= 1"));
        }
        if let InterpolantKind::NoisyLinear { sigma_scale } = cfg.kind {
            if sigma_scale < 0.0 {
                return Err(GenError::Internal(format!(
                    "NoisyLinear sigma_scale must be >= 0, got {sigma_scale}"
                )));
            }
        }
        Ok(Self { cfg })
    }

    /// Clamp `t` to `[0, 1]` and report an error if it lies outside this range.
    ///
    /// All public per-time methods (`alpha`, `beta`, `sigma`, derivatives,
    /// `interpolate`, `target_velocity`) consume a checked `t` so that
    /// callers see consistent error reporting.
    #[inline]
    fn check_t(t: f32) -> GenResult<f32> {
        if !(0.0..=1.0).contains(&t) {
            return Err(GenError::InvalidFlowTime(t));
        }
        Ok(t)
    }

    /// Validate that a slice has length equal to the configured `dim`.
    #[inline]
    fn check_len(&self, slice: &[f32]) -> GenResult<()> {
        if slice.len() != self.cfg.dim {
            return Err(GenError::DimensionMismatch {
                expected: self.cfg.dim,
                got: slice.len(),
            });
        }
        Ok(())
    }

    /// Coefficient `α(t)` of the `x0` channel.
    pub fn alpha(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.cfg.kind {
            InterpolantKind::LinearFlow | InterpolantKind::NoisyLinear { .. } => 1.0 - t,
            InterpolantKind::TrigInterpolant => (std::f32::consts::FRAC_PI_2 * t).cos(),
        }
    }

    /// Coefficient `β(t)` of the `x1` channel.
    pub fn beta(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.cfg.kind {
            InterpolantKind::LinearFlow | InterpolantKind::NoisyLinear { .. } => t,
            InterpolantKind::TrigInterpolant => (std::f32::consts::FRAC_PI_2 * t).sin(),
        }
    }

    /// Coefficient `σ(t)` of the Gaussian noise channel.
    pub fn sigma(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.cfg.kind {
            InterpolantKind::LinearFlow | InterpolantKind::TrigInterpolant => 0.0,
            InterpolantKind::NoisyLinear { sigma_scale } => {
                // √(t · (1 − t)); peaks at t = 0.5 and vanishes at the
                // endpoints. Clamp the argument so f32 round-off cannot
                // produce a negative value before the square root.
                let arg = (t * (1.0 - t)).max(0.0);
                sigma_scale * arg.sqrt()
            }
        }
    }

    /// Derivative `α'(t)`.
    pub fn dalpha(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.cfg.kind {
            InterpolantKind::LinearFlow | InterpolantKind::NoisyLinear { .. } => -1.0,
            InterpolantKind::TrigInterpolant => {
                let k = std::f32::consts::FRAC_PI_2;
                -k * (k * t).sin()
            }
        }
    }

    /// Derivative `β'(t)`.
    pub fn dbeta(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.cfg.kind {
            InterpolantKind::LinearFlow | InterpolantKind::NoisyLinear { .. } => 1.0,
            InterpolantKind::TrigInterpolant => {
                let k = std::f32::consts::FRAC_PI_2;
                k * (k * t).cos()
            }
        }
    }

    /// Interpolant sample at time `t`:
    /// `X_t = α(t) · x0 + β(t) · x1 + σ(t) · z`.
    ///
    /// `z` is provided externally; for noise-free interpolants
    /// (`LinearFlow`, `TrigInterpolant`) it is multiplied by `σ(t) = 0` and
    /// therefore does not influence the output, but it is still validated
    /// for length.
    ///
    /// # Errors
    /// - [`GenError::InvalidFlowTime`] if `t ∉ [0, 1]`.
    /// - [`GenError::DimensionMismatch`] if any of `x0`, `x1`, `z` differs
    ///   from `dim`.
    pub fn interpolate(&self, x0: &[f32], x1: &[f32], z: &[f32], t: f32) -> GenResult<Vec<f32>> {
        let t = Self::check_t(t)?;
        self.check_len(x0)?;
        self.check_len(x1)?;
        self.check_len(z)?;
        let a = self.alpha(t);
        let b = self.beta(t);
        let s = self.sigma(t);
        let out = x0
            .iter()
            .zip(x1)
            .zip(z)
            .map(|((&a0, &b0), &z0)| a * a0 + b * b0 + s * z0)
            .collect();
        Ok(out)
    }

    /// Deterministic velocity target `α'(t) · x0 + β'(t) · x1`.
    ///
    /// This is the regression target for the velocity network in the
    /// stochastic-interpolants framework.
    ///
    /// # Errors
    /// - [`GenError::InvalidFlowTime`] if `t ∉ [0, 1]`.
    /// - [`GenError::DimensionMismatch`] if `x0` or `x1` differs from `dim`.
    pub fn target_velocity(&self, x0: &[f32], x1: &[f32], t: f32) -> GenResult<Vec<f32>> {
        let t = Self::check_t(t)?;
        self.check_len(x0)?;
        self.check_len(x1)?;
        let da = self.dalpha(t);
        let db = self.dbeta(t);
        let out = x0.iter().zip(x1).map(|(&a, &b)| da * a + db * b).collect();
        Ok(out)
    }

    /// Integrate the generative ODE `dX/dt = velocity(X, t)` from `t = 0`
    /// to `t = 1` with a forward Euler scheme of `n_steps` steps starting
    /// from `x_init`, and return the terminal state.
    ///
    /// The closure `velocity` is queried at each integration time
    /// `t_i = i · dt` with the current state and must return a vector of
    /// length `dim`. The integration corresponds to
    ///
    /// ```text
    /// dt = 1 / n_steps
    /// for i in 0..n_steps:
    ///     t = i · dt
    ///     v = velocity(X, t)
    ///     X += dt · v
    /// ```
    ///
    /// For the LinearFlow kind, supplying the exact `target_velocity` (which
    /// is constant in `t`) recovers `x1` from `x0` in a single Euler step.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `x_init` or any closure output
    ///   differs from `dim`.
    pub fn sample_ode<V>(&self, x_init: &[f32], velocity: V) -> GenResult<Vec<f32>>
    where
        V: Fn(&[f32], f32) -> Vec<f32>,
    {
        self.check_len(x_init)?;
        let dt = 1.0 / self.cfg.n_steps as f32;
        let mut x = x_init.to_vec();
        for i in 0..self.cfg.n_steps {
            let t = i as f32 * dt;
            let v = velocity(&x, t);
            self.check_len(&v)?;
            for (xi, &vi) in x.iter_mut().zip(&v) {
                *xi += dt * vi;
            }
        }
        Ok(x)
    }

    /// Return the configuration.
    pub fn config(&self) -> &InterpolantConfig {
        &self.cfg
    }

    /// Convenience accessor for the dimensionality.
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Convenience accessor for the number of Euler steps.
    pub fn n_steps(&self) -> usize {
        self.cfg.n_steps
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TINY: f32 = 1e-6;

    fn make(dim: usize, n_steps: usize, kind: InterpolantKind) -> StochasticInterpolant {
        StochasticInterpolant::new(InterpolantConfig { dim, n_steps, kind })
            .expect("new should succeed")
    }

    #[test]
    fn linear_flow_alpha_beta_boundary_values() {
        let si = make(3, 8, InterpolantKind::LinearFlow);
        assert!((si.alpha(0.0) - 1.0).abs() < EPS);
        assert!((si.alpha(1.0) - 0.0).abs() < EPS);
        assert!((si.beta(0.0) - 0.0).abs() < EPS);
        assert!((si.beta(1.0) - 1.0).abs() < EPS);
        assert!(si.sigma(0.0).abs() < EPS);
        assert!(si.sigma(0.5).abs() < EPS);
        assert!(si.sigma(1.0).abs() < EPS);
    }

    #[test]
    fn linear_flow_derivatives_are_constant() {
        // α'(t) = −1, β'(t) = +1.
        let si = make(3, 8, InterpolantKind::LinearFlow);
        for &t in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                (si.dalpha(t) - -1.0).abs() < EPS,
                "α'({t})={}",
                si.dalpha(t)
            );
            assert!((si.dbeta(t) - 1.0).abs() < EPS, "β'({t})={}", si.dbeta(t));
        }
    }

    #[test]
    fn trig_alpha_beta_boundary_values() {
        let si = make(3, 8, InterpolantKind::TrigInterpolant);
        assert!((si.alpha(0.0) - 1.0).abs() < EPS);
        assert!(si.alpha(1.0).abs() < EPS);
        assert!(si.beta(0.0).abs() < EPS);
        assert!((si.beta(1.0) - 1.0).abs() < EPS);
        // σ ≡ 0 for the trigonometric interpolant.
        assert!(si.sigma(0.0).abs() < EPS);
        assert!(si.sigma(0.5).abs() < EPS);
        assert!(si.sigma(1.0).abs() < EPS);
    }

    #[test]
    fn trig_orthonormal_identity() {
        // α(t)² + β(t)² ≡ 1 for the trigonometric interpolant.
        let si = make(3, 8, InterpolantKind::TrigInterpolant);
        for k in 0..=10 {
            let t = k as f32 / 10.0;
            let a = si.alpha(t);
            let b = si.beta(t);
            let s = a * a + b * b;
            assert!((s - 1.0).abs() < 1e-5, "α²+β² at t={t} = {s}");
        }
    }

    #[test]
    fn trig_derivatives_hand_check() {
        // β'(t) = (π/2) · cos(π t / 2) ⇒ β'(0) = π/2, β'(1) = 0.
        // α'(t) = −(π/2) · sin(π t / 2) ⇒ α'(0) = 0, α'(1) = −π/2.
        let si = make(3, 8, InterpolantKind::TrigInterpolant);
        let half_pi = std::f32::consts::FRAC_PI_2;
        assert!(si.dalpha(0.0).abs() < EPS, "α'(0)={}", si.dalpha(0.0));
        assert!(
            (si.dalpha(1.0) - (-half_pi)).abs() < 1e-5,
            "α'(1)={}",
            si.dalpha(1.0)
        );
        assert!(
            (si.dbeta(0.0) - half_pi).abs() < 1e-5,
            "β'(0)={}",
            si.dbeta(0.0)
        );
        assert!(si.dbeta(1.0).abs() < EPS, "β'(1)={}", si.dbeta(1.0));
    }

    #[test]
    fn noisy_linear_sigma_endpoints_and_peak() {
        let si = make(2, 8, InterpolantKind::NoisyLinear { sigma_scale: 1.5 });
        assert!(si.sigma(0.0).abs() < EPS, "σ(0)={}", si.sigma(0.0));
        assert!(si.sigma(1.0).abs() < EPS, "σ(1)={}", si.sigma(1.0));
        let peak = si.sigma(0.5);
        let mid_quarter = si.sigma(0.25);
        // σ(0.5) = 1.5·√0.25 = 0.75; σ(0.25) = 1.5·√(0.25·0.75) ≈ 1.5·0.433 ≈ 0.65.
        assert!((peak - 0.75).abs() < 1e-4, "σ(0.5)={peak}");
        assert!(
            peak > mid_quarter,
            "peak {peak} should exceed σ(0.25) {mid_quarter}"
        );
        assert!(peak > si.sigma(0.75));
    }

    #[test]
    fn noisy_linear_sigma_scale_zero_matches_linear_flow() {
        // sigma_scale=0 ⇒ NoisyLinear ≡ LinearFlow on every coefficient.
        let nl = make(3, 4, InterpolantKind::NoisyLinear { sigma_scale: 0.0 });
        let lf = make(3, 4, InterpolantKind::LinearFlow);
        for k in 0..=10 {
            let t = k as f32 / 10.0;
            assert!((nl.alpha(t) - lf.alpha(t)).abs() < EPS);
            assert!((nl.beta(t) - lf.beta(t)).abs() < EPS);
            assert!((nl.sigma(t) - lf.sigma(t)).abs() < EPS);
            assert!((nl.dalpha(t) - lf.dalpha(t)).abs() < EPS);
            assert!((nl.dbeta(t) - lf.dbeta(t)).abs() < EPS);
        }
    }

    #[test]
    fn interpolate_linear_at_t0_gives_x0() {
        let si = make(4, 4, InterpolantKind::LinearFlow);
        let x0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x1 = vec![10.0_f32, 11.0, 12.0, 13.0];
        let z = vec![100.0_f32, 200.0, 300.0, 400.0]; // σ=0: should be ignored
        let out = si
            .interpolate(&x0, &x1, &z, 0.0)
            .expect("interpolate should succeed");
        for (&o, &a) in out.iter().zip(&x0) {
            assert!((o - a).abs() < EPS, "{o} != {a}");
        }
    }

    #[test]
    fn interpolate_linear_at_t1_gives_x1() {
        let si = make(4, 4, InterpolantKind::LinearFlow);
        let x0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x1 = vec![10.0_f32, 11.0, 12.0, 13.0];
        let z = vec![100.0_f32, 200.0, 300.0, 400.0];
        let out = si
            .interpolate(&x0, &x1, &z, 1.0)
            .expect("interpolate should succeed");
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "{o} != {b}");
        }
    }

    #[test]
    fn target_velocity_linear_is_x1_minus_x0() {
        let si = make(3, 5, InterpolantKind::LinearFlow);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 7.0, 0.0];
        // α'·x0 + β'·x1 = (−1)·x0 + 1·x1 = x1 − x0.
        let v = si
            .target_velocity(&x0, &x1, 0.4)
            .expect("target_velocity should succeed");
        let expected = [3.0_f32, 5.0, -3.0];
        for (&vi, &e) in v.iter().zip(&expected) {
            assert!((vi - e).abs() < EPS, "{vi} != {e}");
        }
    }

    #[test]
    fn sample_ode_output_length_equals_dim() {
        let si = make(7, 6, InterpolantKind::LinearFlow);
        let x_init = vec![0.5_f32; 7];
        let out = si
            .sample_ode(&x_init, |x, _| x.to_vec())
            .expect("value should be present");
        assert_eq!(out.len(), 7);
    }

    #[test]
    fn sample_ode_linear_flow_recovers_x1_exactly() {
        // For LinearFlow the exact velocity is constant v = x1 − x0; one or
        // more Euler steps recover x1 exactly from x0.
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let x0 = vec![0.0_f32, -1.0, 2.0];
        let x1 = vec![3.0_f32, 5.0, -2.0];
        let v = si
            .target_velocity(&x0, &x1, 0.0)
            .expect("target_velocity should succeed");
        let out = si
            .sample_ode(&x0, |_, _| v.clone())
            .expect("value should be present");
        for (&o, &b) in out.iter().zip(&x1) {
            assert!((o - b).abs() < EPS, "sample_ode {o} != {b}");
        }
    }

    #[test]
    fn sample_ode_zero_velocity_returns_x_init() {
        let si = make(4, 10, InterpolantKind::LinearFlow);
        let x_init = vec![0.7_f32, -0.3, 1.1, 2.5];
        let out = si
            .sample_ode(&x_init, |_, _| vec![0.0_f32; 4])
            .expect("sample_ode should succeed");
        for (&o, &i) in out.iter().zip(&x_init) {
            assert!((o - i).abs() < EPS, "zero velocity moved x: {o} vs {i}");
        }
    }

    #[test]
    fn sample_ode_deterministic_given_closure() {
        let si = make(3, 6, InterpolantKind::LinearFlow);
        let x0 = vec![0.1_f32, 0.2, 0.3];
        let vel = |x: &[f32], t: f32| x.iter().map(|&xi| 0.5 * xi + t).collect::<Vec<_>>();
        let a = si.sample_ode(&x0, vel).expect("sample_ode should succeed");
        let b = si.sample_ode(&x0, vel).expect("sample_ode should succeed");
        for (&ai, &bi) in a.iter().zip(&b) {
            assert!((ai - bi).abs() < TINY, "non-deterministic: {ai} vs {bi}");
        }
    }

    #[test]
    fn err_dim_zero() {
        let cfg = InterpolantConfig {
            dim: 0,
            n_steps: 4,
            kind: InterpolantKind::LinearFlow,
        };
        assert!(matches!(
            StochasticInterpolant::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_n_steps_zero() {
        let cfg = InterpolantConfig {
            dim: 3,
            n_steps: 0,
            kind: InterpolantKind::LinearFlow,
        };
        assert!(matches!(
            StochasticInterpolant::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_interpolate_t_out_of_range() {
        let si = make(2, 4, InterpolantKind::LinearFlow);
        let x0 = vec![0.0_f32, 0.0];
        let x1 = vec![1.0_f32, 1.0];
        let z = vec![0.0_f32, 0.0];
        assert!(matches!(
            si.interpolate(&x0, &x1, &z, -0.01),
            Err(GenError::InvalidFlowTime(_))
        ));
        assert!(matches!(
            si.interpolate(&x0, &x1, &z, 1.01),
            Err(GenError::InvalidFlowTime(_))
        ));
    }

    #[test]
    fn err_target_velocity_t_out_of_range() {
        let si = make(2, 4, InterpolantKind::TrigInterpolant);
        let x0 = vec![0.0_f32, 0.0];
        let x1 = vec![1.0_f32, 1.0];
        assert!(matches!(
            si.target_velocity(&x0, &x1, -1.0),
            Err(GenError::InvalidFlowTime(_))
        ));
        assert!(matches!(
            si.target_velocity(&x0, &x1, 2.0),
            Err(GenError::InvalidFlowTime(_))
        ));
    }

    #[test]
    fn err_interpolate_x0_wrong_length() {
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let bad_x0 = vec![1.0_f32, 2.0];
        let x1 = vec![1.0_f32, 2.0, 3.0];
        let z = vec![0.0_f32, 0.0, 0.0];
        assert!(matches!(
            si.interpolate(&bad_x0, &x1, &z, 0.5),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_interpolate_x1_wrong_length() {
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let bad_x1 = vec![1.0_f32, 2.0];
        let z = vec![0.0_f32, 0.0, 0.0];
        assert!(matches!(
            si.interpolate(&x0, &bad_x1, &z, 0.5),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_interpolate_z_wrong_length() {
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let bad_z = vec![0.0_f32, 0.0];
        assert!(matches!(
            si.interpolate(&x0, &x1, &bad_z, 0.5),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_sample_ode_x_init_wrong_length() {
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let bad_init = vec![0.0_f32; 2];
        assert!(matches!(
            si.sample_ode(&bad_init, |x, _| x.to_vec()),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_sample_ode_closure_wrong_length() {
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let x_init = vec![0.0_f32; 3];
        assert!(matches!(
            si.sample_ode(&x_init, |_, _| vec![0.0_f32; 2]),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_target_velocity_x_wrong_length() {
        let si = make(3, 4, InterpolantKind::LinearFlow);
        let bad_x = vec![0.0_f32; 2];
        let good_x = vec![1.0_f32; 3];
        assert!(matches!(
            si.target_velocity(&bad_x, &good_x, 0.3),
            Err(GenError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            si.target_velocity(&good_x, &bad_x, 0.3),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_noisy_linear_negative_scale() {
        let cfg = InterpolantConfig {
            dim: 2,
            n_steps: 4,
            kind: InterpolantKind::NoisyLinear { sigma_scale: -0.1 },
        };
        assert!(matches!(
            StochasticInterpolant::new(cfg),
            Err(GenError::Internal(_))
        ));
    }

    #[test]
    fn interpolate_uses_sigma_when_nonzero() {
        // For NoisyLinear at t=0.5 with sigma_scale=2, σ = 2·√0.25 = 1.0,
        // and α=β=0.5. Therefore X_0.5 = 0.5(x0+x1) + 1·z.
        let si = make(3, 4, InterpolantKind::NoisyLinear { sigma_scale: 2.0 });
        let x0 = vec![0.0_f32, 0.0, 0.0];
        let x1 = vec![0.0_f32, 0.0, 0.0];
        let z = vec![1.0_f32, -1.0, 0.5];
        let out = si
            .interpolate(&x0, &x1, &z, 0.5)
            .expect("interpolate should succeed");
        for (&o, &zi) in out.iter().zip(&z) {
            assert!(
                (o - zi).abs() < 1e-5,
                "interpolate should reduce to σ·z: {o} vs {zi}"
            );
        }
    }
}
