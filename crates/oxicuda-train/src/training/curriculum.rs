//! Curriculum learning — competence-based example pacing (Platanios et al., 2019).
//!
//! "Competence-based Curriculum Learning for Neural Machine Translation"
//! (NAACL 2019, arXiv:1903.09848).
//!
//! Curriculum learning presents training examples in an easy-to-hard order.
//! A **competence** function `c(t) ∈ [0, 1]` grows with the training step `t`
//! and gates which fraction of the (difficulty-sorted) dataset the model is
//! allowed to sample from: at step `t` the learner may draw only from examples
//! whose cumulative-difficulty rank `≤ c(t)`.
//!
//! This module provides:
//!
//! * [`Pacing`] — the standard competence schedules (linear, square-root,
//!   geometric, and step).
//! * [`Curriculum`] — wraps a pacing function with the total curriculum length
//!   and an initial competence `c₀`, exposing the current competence and the
//!   index window `[0, k)` of difficulty-sorted examples currently admissible.
//!
//! The square-root schedule `c(t) = √(t/T·(1 − c₀²) + c₀²)` is the paper's
//! recommendation: it spends *more* of the budget on harder examples (the
//! admissible fraction grows fast early, slow late), which empirically
//! outperforms the linear schedule.

use crate::error::{TrainError, TrainResult};

// ─── Pacing functions ─────────────────────────────────────────────────────────

/// Competence-pacing schedule `c(t)` mapping normalised progress to `[c₀, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pacing {
    /// Linear growth: `c(t) = c₀ + (1 − c₀)·t/T`.
    Linear,
    /// Square-root growth (Platanios default):
    /// `c(t) = √(t/T·(1 − c₀²) + c₀²)`.
    Sqrt,
    /// Geometric / exponential growth controlled by `rate > 0`; larger `rate`
    /// delays admitting harder examples:
    /// `c(t) = c₀ + (1 − c₀)·(eᵏ·ᵗ/ᵀ − 1)/(eᵏ − 1)`.
    Geometric {
        /// Exponential curvature `k` (must be > 0).
        rate: f64,
    },
    /// Discrete step schedule: competence jumps to `1` at progress `≥ threshold`
    /// and stays at `c₀` before that.
    Step {
        /// Progress fraction in `(0, 1]` at which full competence is granted.
        threshold: f64,
    },
}

impl Pacing {
    /// Evaluate the competence at normalised progress `frac ∈ [0, 1]` given the
    /// initial competence `c0 ∈ (0, 1]`.  The result is clamped to `[c0, 1]`.
    #[must_use]
    pub fn competence(self, frac: f64, c0: f64) -> f64 {
        let frac = frac.clamp(0.0, 1.0);
        let c = match self {
            Pacing::Linear => c0 + (1.0 - c0) * frac,
            Pacing::Sqrt => (frac * (1.0 - c0 * c0) + c0 * c0).sqrt(),
            Pacing::Geometric { rate } => {
                let denom = rate.exp_m1(); // e^rate − 1
                let num = (rate * frac).exp_m1(); // e^{rate·frac} − 1
                c0 + (1.0 - c0) * (num / denom)
            }
            Pacing::Step { threshold } => {
                if frac >= threshold {
                    1.0
                } else {
                    c0
                }
            }
        };
        c.clamp(c0, 1.0)
    }
}

// ─── Curriculum ───────────────────────────────────────────────────────────────

/// Competence-based curriculum over a difficulty-sorted dataset.
#[derive(Debug, Clone)]
pub struct Curriculum {
    pacing: Pacing,
    /// Total number of steps over which competence ramps to 1.
    total_steps: u64,
    /// Initial competence `c₀ ∈ (0, 1]`.
    c0: f64,
    /// Number of (difficulty-sorted) examples in the dataset.
    dataset_size: usize,
    step: u64,
}

impl Curriculum {
    /// Create a curriculum.
    ///
    /// * `pacing` – competence schedule.
    /// * `total_steps` – steps over which competence reaches 1 (≥ 1).
    /// * `c0` – initial competence in `(0, 1]` (the minimum admissible fraction).
    /// * `dataset_size` – number of difficulty-sorted examples (≥ 1).
    ///
    /// # Errors
    ///
    /// * [`TrainError::Internal`] for `total_steps == 0`, `c0 ∉ (0, 1]`, or an
    ///   invalid `Geometric`/`Step` parameter.
    /// * [`TrainError::EmptyParams`] if `dataset_size == 0`.
    pub fn new(
        pacing: Pacing,
        total_steps: u64,
        c0: f64,
        dataset_size: usize,
    ) -> TrainResult<Self> {
        if dataset_size == 0 {
            return Err(TrainError::EmptyParams);
        }
        if total_steps == 0 {
            return Err(TrainError::Internal {
                msg: "total_steps must be >= 1".into(),
            });
        }
        if !(c0 > 0.0 && c0 <= 1.0) {
            return Err(TrainError::Internal {
                msg: format!("c0 must be in (0, 1], got {c0}"),
            });
        }
        match pacing {
            Pacing::Geometric { rate } if rate <= 0.0 || rate.is_nan() => {
                return Err(TrainError::Internal {
                    msg: format!("Geometric rate must be > 0, got {rate}"),
                });
            }
            Pacing::Step { threshold } if !(threshold > 0.0 && threshold <= 1.0) => {
                return Err(TrainError::Internal {
                    msg: format!("Step threshold must be in (0, 1], got {threshold}"),
                });
            }
            _ => {}
        }
        Ok(Self {
            pacing,
            total_steps,
            c0,
            dataset_size,
            step: 0,
        })
    }

    /// Competence at the current step.
    #[must_use]
    pub fn competence(&self) -> f64 {
        let frac = self.step as f64 / self.total_steps as f64;
        self.pacing.competence(frac, self.c0)
    }

    /// Number of admissible (easiest) examples at the current step:
    /// `⌈c(t)·N⌉`, always at least 1 and at most `N`.
    #[must_use]
    pub fn admissible(&self) -> usize {
        let k = (self.competence() * self.dataset_size as f64).ceil() as usize;
        k.clamp(1, self.dataset_size)
    }

    /// Advance the curriculum by one step and return the new admissible count.
    pub fn step(&mut self) -> usize {
        if self.step < self.total_steps {
            self.step += 1;
        }
        self.admissible()
    }

    /// Map a uniform random draw `u ∈ [0, 1)` to an admissible example index in
    /// `[0, admissible())`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::Internal`] if `u ∉ [0, 1)`.
    pub fn sample_index(&self, u: f64) -> TrainResult<usize> {
        if !(0.0..1.0).contains(&u) {
            return Err(TrainError::Internal {
                msg: format!("uniform draw must be in [0, 1), got {u}"),
            });
        }
        let k = self.admissible();
        let idx = (u * k as f64) as usize;
        Ok(idx.min(k - 1))
    }

    /// Current step index.
    #[must_use]
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Reset the curriculum to step 0.
    pub fn reset(&mut self) {
        self.step = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn rejects_bad_config() {
        assert!(matches!(
            Curriculum::new(Pacing::Linear, 0, 0.1, 10),
            Err(TrainError::Internal { .. })
        ));
        assert!(matches!(
            Curriculum::new(Pacing::Linear, 10, 0.0, 10),
            Err(TrainError::Internal { .. })
        ));
        assert!(matches!(
            Curriculum::new(Pacing::Linear, 10, 0.1, 0),
            Err(TrainError::EmptyParams)
        ));
        assert!(matches!(
            Curriculum::new(Pacing::Geometric { rate: 0.0 }, 10, 0.1, 10),
            Err(TrainError::Internal { .. })
        ));
        assert!(matches!(
            Curriculum::new(Pacing::Step { threshold: 1.5 }, 10, 0.1, 10),
            Err(TrainError::Internal { .. })
        ));
    }

    /// Linear competence matches the closed form at sampled progress points.
    #[test]
    fn linear_closed_form() {
        let c0 = 0.2;
        let p = Pacing::Linear;
        for &frac in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let expect = c0 + (1.0 - c0) * frac;
            assert!((p.competence(frac, c0) - expect).abs() < 1e-12);
        }
    }

    /// Sqrt competence matches its closed form and reaches 1 at the end.
    #[test]
    fn sqrt_closed_form() {
        let c0 = 0.1_f64;
        let p = Pacing::Sqrt;
        for &frac in &[0.0, 0.3, 0.6, 1.0] {
            let expect = (frac * (1.0 - c0 * c0) + c0 * c0).sqrt();
            assert!((p.competence(frac, c0) - expect).abs() < 1e-12);
        }
        assert!((p.competence(1.0, c0) - 1.0).abs() < 1e-12);
    }

    /// Endpoints: every pacing starts at c0 and ends at 1.
    #[test]
    fn endpoints_all_modes() {
        let c0 = 0.15;
        let modes = [
            Pacing::Linear,
            Pacing::Sqrt,
            Pacing::Geometric { rate: 3.0 },
            Pacing::Step { threshold: 0.5 },
        ];
        for m in modes {
            assert!((m.competence(0.0, c0) - c0).abs() < 1e-9, "{m:?} start");
            assert!((m.competence(1.0, c0) - 1.0).abs() < 1e-9, "{m:?} end");
        }
    }

    /// Competence is monotonically non-decreasing in progress.
    #[test]
    fn monotone_non_decreasing() {
        let c0 = 0.1;
        for m in [
            Pacing::Linear,
            Pacing::Sqrt,
            Pacing::Geometric { rate: 2.5 },
        ] {
            let mut prev = 0.0;
            for i in 0..=100 {
                let frac = i as f64 / 100.0;
                let c = m.competence(frac, c0);
                assert!(c + 1e-12 >= prev, "{m:?} not monotone at {frac}");
                prev = c;
            }
        }
    }

    /// Sqrt admits harder examples earlier than linear (its competence is
    /// strictly larger in the interior).
    #[test]
    fn sqrt_admits_faster_than_linear() {
        let c0 = 0.1;
        let frac = 0.4;
        let lin = Pacing::Linear.competence(frac, c0);
        let sq = Pacing::Sqrt.competence(frac, c0);
        assert!(
            sq > lin,
            "sqrt {sq} should exceed linear {lin} mid-curriculum"
        );
    }

    /// Step pacing jumps at the threshold.
    #[test]
    fn step_jumps_at_threshold() {
        let p = Pacing::Step { threshold: 0.5 };
        assert!((p.competence(0.49, 0.2) - 0.2).abs() < 1e-12);
        assert!((p.competence(0.5, 0.2) - 1.0).abs() < 1e-12);
    }

    /// The admissible window grows from a small set to the full dataset.
    #[test]
    fn admissible_grows_to_full() {
        let n = 100;
        let mut cur = Curriculum::new(Pacing::Linear, 50, 0.1, n).expect("valid");
        let start = cur.admissible();
        assert!(start >= 1 && start < n);
        for _ in 0..50 {
            cur.step();
        }
        assert_eq!(cur.admissible(), n, "should admit all examples at the end");
    }

    /// Sampled indices always fall within the admissible window.
    #[test]
    fn samples_within_window() {
        let n = 64;
        let mut cur = Curriculum::new(Pacing::Sqrt, 30, 0.2, n).expect("valid");
        let mut rng = LcgRng::new(5);
        for _ in 0..30 {
            cur.step();
            let k = cur.admissible();
            for _ in 0..20 {
                let idx = cur.sample_index(f64::from(rng.next_f32())).expect("ok");
                assert!(idx < k, "sampled {idx} outside window {k}");
            }
        }
    }

    #[test]
    fn sample_index_rejects_out_of_range() {
        let cur = Curriculum::new(Pacing::Linear, 10, 0.5, 10).expect("valid");
        assert!(matches!(
            cur.sample_index(1.0),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn reset_returns_to_start() {
        let mut cur = Curriculum::new(Pacing::Linear, 10, 0.3, 10).expect("valid");
        cur.step();
        cur.step();
        cur.reset();
        assert_eq!(cur.current_step(), 0);
    }
}
