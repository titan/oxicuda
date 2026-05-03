//! Beta schedules for diffusion models.
//!
//! Provides linear, cosine, scaled-cosine, and sigmoid beta schedules
//! for diffusion model training and inference.

use crate::error::{GenError, GenResult};

// ─── BetaScheduleType ─────────────────────────────────────────────────────────

/// The type of beta schedule to use for diffusion.
#[derive(Debug, Clone)]
pub enum BetaScheduleType {
    /// Linear interpolation from `beta_start` to `beta_end`.
    Linear { beta_start: f32, beta_end: f32 },
    /// Cosine schedule with offset `s` (Nichol & Dhariwal 2021).
    Cosine { s: f32 },
    /// Cosine schedule with additional rescaling.
    ScaledCosine { s: f32, scale: f32 },
    /// Sigmoid schedule for continuous-time diffusion.
    Sigmoid { start: f32, end: f32, tau: f32 },
}

// ─── BetaSchedule ─────────────────────────────────────────────────────────────

/// Precomputed beta schedule with all derived quantities.
///
/// Stores `β_t`, `α_t = 1 - β_t`, `ᾱ_t = ∏α_i`, `√ᾱ_t`, and `√(1-ᾱ_t)`
/// for all timesteps `t = 0..T`.
#[derive(Debug, Clone)]
pub struct BetaSchedule {
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alphas_bar: Vec<f32>,
    sqrt_alphas_bar: Vec<f32>,
    sqrt_one_minus_alphas_bar: Vec<f32>,
}

impl BetaSchedule {
    /// Create a new beta schedule from the given type and number of steps.
    pub fn new(num_steps: usize, schedule: BetaScheduleType) -> GenResult<Self> {
        if num_steps == 0 {
            return Err(GenError::EmptyInput("num_steps must be > 0"));
        }
        match schedule {
            BetaScheduleType::Linear {
                beta_start,
                beta_end,
            } => Self::linear(num_steps, beta_start, beta_end),
            BetaScheduleType::Cosine { s } => Self::cosine(num_steps, s),
            BetaScheduleType::ScaledCosine { s, scale } => {
                let mut sched = Self::cosine(num_steps, s)?;
                // Rescale betas by scale factor
                for b in &mut sched.betas {
                    *b = (*b * scale).clamp(0.0, 0.999);
                }
                sched.recompute_derived();
                Ok(sched)
            }
            BetaScheduleType::Sigmoid { start, end, tau } => {
                Self::sigmoid(num_steps, start, end, tau)
            }
        }
    }

    /// Linear beta schedule: `β_t = β_start + t/(T-1) * (β_end - β_start)`.
    pub fn linear(num_steps: usize, beta_start: f32, beta_end: f32) -> GenResult<Self> {
        if num_steps == 0 {
            return Err(GenError::EmptyInput("num_steps must be > 0"));
        }
        if !(0.0..1.0).contains(&beta_start) || !(0.0..1.0).contains(&beta_end) {
            return Err(GenError::InvalidBetaSchedule);
        }
        let betas: Vec<f32> = if num_steps == 1 {
            vec![(beta_start + beta_end) * 0.5]
        } else {
            (0..num_steps)
                .map(|t| {
                    let frac = t as f32 / (num_steps - 1) as f32;
                    beta_start + frac * (beta_end - beta_start)
                })
                .collect()
        };
        Self::from_betas(betas)
    }

    /// Cosine beta schedule (Nichol & Dhariwal 2021).
    ///
    /// `ᾱ_t = cos²((t/T + s) / (1+s) * π/2) / ᾱ_0`
    /// `β_t = 1 - ᾱ_t / ᾱ_{t-1}`, clipped to `(0, 0.999]`.
    pub fn cosine(num_steps: usize, s: f32) -> GenResult<Self> {
        if num_steps == 0 {
            return Err(GenError::EmptyInput("num_steps must be > 0"));
        }
        if s <= 0.0 {
            return Err(GenError::InvalidBetaSchedule);
        }
        let t_to_alpha_bar = |t: f32| -> f32 {
            let frac = (t / num_steps as f32 + s) / (1.0 + s);
            (frac * std::f32::consts::FRAC_PI_2).cos().powi(2)
        };
        let alpha_bar_0 = t_to_alpha_bar(0.0);

        let mut betas = Vec::with_capacity(num_steps);
        let mut prev_alpha_bar = alpha_bar_0;
        for t in 1..=num_steps {
            let curr_alpha_bar = t_to_alpha_bar(t as f32) / alpha_bar_0;
            let prev_ab = if t == 1 {
                1.0
            } else {
                t_to_alpha_bar((t - 1) as f32) / alpha_bar_0
            };
            let beta = (1.0 - curr_alpha_bar / prev_ab).clamp(0.0001, 0.999);
            let _ = prev_alpha_bar;
            prev_alpha_bar = curr_alpha_bar;
            betas.push(beta);
        }
        Self::from_betas(betas)
    }

    /// Sigmoid beta schedule.
    ///
    /// `β_t = sigmoid(start + t/(T-1) * (end - start))^tau / Z`
    /// where Z is a normalisation constant to keep betas in (0, 0.999].
    pub fn sigmoid(num_steps: usize, start: f32, end: f32, tau: f32) -> GenResult<Self> {
        if num_steps == 0 {
            return Err(GenError::EmptyInput("num_steps must be > 0"));
        }
        if tau <= 0.0 {
            return Err(GenError::InvalidBetaSchedule);
        }
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let betas: Vec<f32> = (0..num_steps)
            .map(|t| {
                let frac = if num_steps == 1 {
                    0.5
                } else {
                    t as f32 / (num_steps - 1) as f32
                };
                let x = start + frac * (end - start);
                sigmoid(x).powf(tau).clamp(0.0001, 0.999)
            })
            .collect();
        Self::from_betas(betas)
    }

    /// Build a schedule from raw beta values.
    pub fn from_betas(betas: Vec<f32>) -> GenResult<Self> {
        if betas.is_empty() {
            return Err(GenError::EmptyInput("betas vector is empty"));
        }
        for &b in &betas {
            if !(b > 0.0 && b < 1.0) {
                return Err(GenError::InvalidBetaSchedule);
            }
        }
        let mut sched = Self {
            betas,
            alphas: Vec::new(),
            alphas_bar: Vec::new(),
            sqrt_alphas_bar: Vec::new(),
            sqrt_one_minus_alphas_bar: Vec::new(),
        };
        sched.recompute_derived();
        Ok(sched)
    }

    /// Recompute `alphas`, `alphas_bar`, and derived square roots.
    fn recompute_derived(&mut self) {
        let n = self.betas.len();
        self.alphas = self.betas.iter().map(|&b| 1.0 - b).collect();
        self.alphas_bar = Vec::with_capacity(n);
        let mut product = 1.0_f32;
        for &a in &self.alphas {
            product *= a;
            self.alphas_bar.push(product);
        }
        self.sqrt_alphas_bar = self.alphas_bar.iter().map(|&ab| ab.sqrt()).collect();
        self.sqrt_one_minus_alphas_bar = self
            .alphas_bar
            .iter()
            .map(|&ab| (1.0 - ab).max(0.0).sqrt())
            .collect();
    }

    /// Return the beta values `β_t` for each timestep.
    pub fn betas(&self) -> &[f32] {
        &self.betas
    }

    /// Return the alpha values `α_t = 1 - β_t` for each timestep.
    pub fn alphas(&self) -> &[f32] {
        &self.alphas
    }

    /// Return the cumulative product `ᾱ_t = ∏α_i` for each timestep.
    pub fn alphas_bar(&self) -> &[f32] {
        &self.alphas_bar
    }

    /// Return `√ᾱ_t` for each timestep.
    pub fn sqrt_alphas_bar(&self) -> &[f32] {
        &self.sqrt_alphas_bar
    }

    /// Return `√(1 - ᾱ_t)` for each timestep.
    pub fn sqrt_one_minus_alphas_bar(&self) -> &[f32] {
        &self.sqrt_one_minus_alphas_bar
    }

    /// Return the number of timesteps `T`.
    pub fn num_steps(&self) -> usize {
        self.betas.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn linear_schedule_monotone_increasing() {
        let sched = BetaSchedule::linear(1000, 0.0001, 0.02).unwrap();
        let betas = sched.betas();
        for w in betas.windows(2) {
            assert!(w[1] >= w[0] - EPS, "not monotone: {} < {}", w[1], w[0]);
        }
    }

    #[test]
    fn linear_schedule_boundary_values() {
        let sched = BetaSchedule::linear(1000, 0.0001, 0.02).unwrap();
        assert!((sched.betas()[0] - 0.0001).abs() < EPS);
        assert!((sched.betas()[999] - 0.02).abs() < EPS);
    }

    #[test]
    fn linear_schedule_alphas_sum() {
        let sched = BetaSchedule::linear(100, 0.001, 0.01).unwrap();
        for (&a, &b) in sched.alphas().iter().zip(sched.betas()) {
            assert!((a + b - 1.0).abs() < EPS, "alpha + beta != 1");
        }
    }

    #[test]
    fn alphas_bar_monotone_decreasing() {
        let sched = BetaSchedule::linear(100, 0.001, 0.02).unwrap();
        let ab = sched.alphas_bar();
        for w in ab.windows(2) {
            assert!(w[1] <= w[0] + EPS, "alphas_bar not monotone decreasing");
        }
    }

    #[test]
    fn sqrt_alphas_bar_consistency() {
        let sched = BetaSchedule::linear(100, 0.001, 0.02).unwrap();
        for (&sab, &ab) in sched.sqrt_alphas_bar().iter().zip(sched.alphas_bar()) {
            assert!((sab * sab - ab).abs() < EPS, "sqrt(ab)^2 != ab");
        }
    }

    #[test]
    fn sqrt_one_minus_alphas_bar_consistency() {
        let sched = BetaSchedule::linear(100, 0.001, 0.02).unwrap();
        for (&smab, &ab) in sched
            .sqrt_one_minus_alphas_bar()
            .iter()
            .zip(sched.alphas_bar())
        {
            let expected = (1.0 - ab).max(0.0).sqrt();
            assert!(
                (smab - expected).abs() < EPS,
                "sqrt_one_minus_alphas_bar inconsistent"
            );
        }
    }

    #[test]
    fn cosine_schedule_betas_in_range() {
        let sched = BetaSchedule::cosine(1000, 0.008).unwrap();
        for &b in sched.betas() {
            assert!(b > 0.0 && b < 1.0, "beta out of range: {b}");
        }
    }

    #[test]
    fn cosine_schedule_alphas_bar_decreasing() {
        let sched = BetaSchedule::cosine(100, 0.008).unwrap();
        let ab = sched.alphas_bar();
        for w in ab.windows(2) {
            assert!(w[1] <= w[0] + 1e-4, "cosine alphas_bar not decreasing");
        }
    }

    #[test]
    fn num_steps_matches() {
        let sched = BetaSchedule::linear(500, 0.0001, 0.02).unwrap();
        assert_eq!(sched.num_steps(), 500);
    }

    #[test]
    fn from_betas_rejects_out_of_range() {
        let bad = vec![0.0_f32, 0.01, 0.5]; // 0.0 is invalid
        assert!(BetaSchedule::from_betas(bad).is_err());
    }

    #[test]
    fn sigmoid_schedule_valid() {
        let sched = BetaSchedule::sigmoid(100, -3.0, 3.0, 1.0).unwrap();
        assert_eq!(sched.num_steps(), 100);
        for &b in sched.betas() {
            assert!(b > 0.0 && b <= 0.999, "sigmoid beta out of range: {b}");
        }
    }

    #[test]
    fn linear_single_step() {
        let sched = BetaSchedule::linear(1, 0.01, 0.02).unwrap();
        assert_eq!(sched.num_steps(), 1);
        let b = sched.betas()[0];
        assert!(b > 0.0 && b < 1.0, "single step beta: {b}");
    }

    #[test]
    fn scaled_cosine_schedule_valid() {
        let sched = BetaSchedule::new(
            100,
            BetaScheduleType::ScaledCosine {
                s: 0.008,
                scale: 0.5,
            },
        )
        .unwrap();
        assert_eq!(sched.num_steps(), 100);
        for &b in sched.betas() {
            assert!(
                b > 0.0 && b <= 0.999,
                "scaled cosine beta out of range: {b}"
            );
        }
    }
}
