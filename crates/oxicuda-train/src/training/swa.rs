//! Stochastic Weight Averaging (SWA) — Izmailov et al., 2018.
//!
//! "Averaging Weights Leads to Wider Optima and Better Generalization"
//! (arXiv:1803.05407).
//!
//! SWA maintains an **equal-weight running average** of the model parameters
//! collected at the end of each SWA cycle.  Unlike an exponential moving
//! average (which weights recent snapshots more heavily — see
//! [`crate::ema`]), SWA gives every captured snapshot identical weight:
//!
//! ```text
//! W_SWA ← (n · W_SWA + W) / (n + 1)        (after the n-th snapshot)
//! ```
//!
//! Averaging points along the SGD trajectory finds flatter regions of the loss
//! surface that generalise better than the final SGD iterate.  Snapshots are
//! typically taken at a constant or cyclic learning rate during the tail of
//! training; [`SwaLr`] provides the companion learning-rate schedule.
//!
//! Because SWA changes the running statistics of any batch-norm layers, the
//! caller should re-estimate BN statistics with a forward pass over the data
//! after calling [`Swa::finalise`] — that step is model-specific and lives
//! outside this crate.

use crate::error::{TrainError, TrainResult};
use crate::lr_scheduler::LrScheduler;

// ─── SWA averager ─────────────────────────────────────────────────────────────

/// Equal-weight running average of model parameters.
#[derive(Debug, Clone)]
pub struct Swa {
    average: Vec<f64>,
    /// Number of snapshots averaged so far.
    n_averaged: u64,
    dim: usize,
}

impl Swa {
    /// Create an SWA averager for a `dim`-element flat parameter vector.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `dim == 0`.
    pub fn new(dim: usize) -> TrainResult<Self> {
        if dim == 0 {
            return Err(TrainError::EmptyParams);
        }
        Ok(Self {
            average: vec![0.0; dim],
            n_averaged: 0,
            dim,
        })
    }

    /// Incorporate a new parameter snapshot into the equal-weight average.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ShapeMismatch`] if `params.len() != dim`.
    pub fn update(&mut self, params: &[f32]) -> TrainResult<()> {
        if params.len() != self.dim {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![params.len()],
            });
        }
        let n = self.n_averaged as f64;
        for (avg, &p) in self.average.iter_mut().zip(params.iter()) {
            // W_SWA ← W_SWA + (W − W_SWA) / (n + 1) == (n·avg + p)/(n+1).
            *avg += (f64::from(p) - *avg) / (n + 1.0);
        }
        self.n_averaged += 1;
        Ok(())
    }

    /// Number of snapshots averaged so far.
    #[must_use]
    pub fn n_averaged(&self) -> u64 {
        self.n_averaged
    }

    /// Immutable view of the (`f64`) averaged weights.
    #[must_use]
    pub fn average(&self) -> &[f64] {
        &self.average
    }

    /// Write the averaged weights into `out` as `f32`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::StateNotInitialised`] if no snapshot has been recorded.
    /// * [`TrainError::ShapeMismatch`] if `out.len() != dim`.
    pub fn finalise(&self, out: &mut [f32]) -> TrainResult<()> {
        if self.n_averaged == 0 {
            return Err(TrainError::StateNotInitialised);
        }
        if out.len() != self.dim {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![out.len()],
            });
        }
        for (o, &a) in out.iter_mut().zip(self.average.iter()) {
            *o = a as f32;
        }
        Ok(())
    }

    /// Reset the averager to its initial (empty) state.
    pub fn reset(&mut self) {
        self.average.iter_mut().for_each(|x| *x = 0.0);
        self.n_averaged = 0;
    }
}

// ─── SWALR schedule ───────────────────────────────────────────────────────────

/// Strategy used by [`SwaLr`] to move from the current LR to the SWA LR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwaLrMode {
    /// Linearly anneal from the start LR to `swa_lr` over the anneal window,
    /// then hold constant.
    Linear,
    /// Cosine anneal from the start LR to `swa_lr` over the anneal window, then
    /// hold constant.
    Cosine,
}

/// The SWA learning-rate schedule.
///
/// During the SWA phase the LR is annealed from a starting value down (or up)
/// to a fixed `swa_lr` over `anneal_steps` steps and then held constant — the
/// constant tail is what lets SWA collect comparable snapshots.
#[derive(Debug, Clone)]
pub struct SwaLr {
    start_lr: f64,
    swa_lr: f64,
    anneal_steps: u64,
    mode: SwaLrMode,
    current_lr: f64,
    steps: u64,
}

impl SwaLr {
    /// Create an SWALR schedule.
    ///
    /// * `start_lr` – LR at the start of the SWA phase (typically the final
    ///   training LR).
    /// * `swa_lr` – constant SWA LR to anneal toward.
    /// * `anneal_steps` – number of steps over which to anneal.
    /// * `mode` – linear or cosine anneal.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidLearningRate`] if `start_lr` or `swa_lr` ≤ 0.
    /// * [`TrainError::Internal`] if `anneal_steps == 0`.
    pub fn new(
        start_lr: f64,
        swa_lr: f64,
        anneal_steps: u64,
        mode: SwaLrMode,
    ) -> TrainResult<Self> {
        if start_lr <= 0.0 || start_lr.is_nan() {
            return Err(TrainError::InvalidLearningRate { lr: start_lr });
        }
        if swa_lr <= 0.0 || swa_lr.is_nan() {
            return Err(TrainError::InvalidLearningRate { lr: swa_lr });
        }
        if anneal_steps == 0 {
            return Err(TrainError::Internal {
                msg: "anneal_steps must be >= 1".into(),
            });
        }
        Ok(Self {
            start_lr,
            swa_lr,
            anneal_steps,
            mode,
            current_lr: start_lr,
            steps: 0,
        })
    }

    /// Constant SWA learning rate the schedule converges to.
    #[must_use]
    pub fn swa_lr(&self) -> f64 {
        self.swa_lr
    }
}

impl LrScheduler for SwaLr {
    fn step(&mut self) -> f64 {
        self.steps += 1;
        if self.steps >= self.anneal_steps {
            self.current_lr = self.swa_lr;
        } else {
            let frac = self.steps as f64 / self.anneal_steps as f64;
            let blend = match self.mode {
                SwaLrMode::Linear => frac,
                // Cosine interpolation: 0 → 1 as frac 0 → 1.
                SwaLrMode::Cosine => 0.5 * (1.0 - (std::f64::consts::PI * frac).cos()),
            };
            self.current_lr = self.start_lr + (self.swa_lr - self.start_lr) * blend;
        }
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "SwaLr"
    }

    fn steps_done(&self) -> u64 {
        self.steps
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn rejects_zero_dim() {
        assert!(matches!(Swa::new(0), Err(TrainError::EmptyParams)));
    }

    #[test]
    fn finalise_before_update_errors() {
        let swa = Swa::new(3).expect("valid");
        let mut out = vec![0.0_f32; 3];
        assert!(matches!(
            swa.finalise(&mut out),
            Err(TrainError::StateNotInitialised)
        ));
    }

    #[test]
    fn wrong_len_update_errors() {
        let mut swa = Swa::new(3).expect("valid");
        assert!(matches!(
            swa.update(&[1.0, 2.0]),
            Err(TrainError::ShapeMismatch { .. })
        ));
    }

    /// The SWA average of a sequence of snapshots equals their arithmetic mean.
    #[test]
    fn equal_weight_mean() {
        let mut swa = Swa::new(2).expect("valid");
        let snaps = [[1.0_f32, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]];
        for s in &snaps {
            swa.update(s).expect("ok");
        }
        assert_eq!(swa.n_averaged(), 4);
        let mut out = vec![0.0_f32; 2];
        swa.finalise(&mut out).expect("ok");
        // means: (1+3+5+7)/4 = 4 ; (10+20+30+40)/4 = 25.
        assert!((out[0] - 4.0).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - 25.0).abs() < 1e-6, "got {}", out[1]);
    }

    /// SWA matches an independently computed running mean over random snapshots.
    #[test]
    fn matches_reference_mean() {
        let dim = 5;
        let mut swa = Swa::new(dim).expect("valid");
        let mut rng = LcgRng::new(99);
        let mut sums = vec![0.0_f64; dim];
        let n = 50;
        for _ in 0..n {
            let snap: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
            for (s, &v) in sums.iter_mut().zip(snap.iter()) {
                *s += f64::from(v);
            }
            swa.update(&snap).expect("ok");
        }
        let mut out = vec![0.0_f32; dim];
        swa.finalise(&mut out).expect("ok");
        for i in 0..dim {
            let ref_mean = (sums[i] / n as f64) as f32;
            assert!(
                (out[i] - ref_mean).abs() < 1e-5,
                "idx {i}: {} vs {ref_mean}",
                out[i]
            );
        }
    }

    #[test]
    fn reset_clears() {
        let mut swa = Swa::new(2).expect("valid");
        swa.update(&[1.0, 2.0]).expect("ok");
        swa.reset();
        assert_eq!(swa.n_averaged(), 0);
        assert!(swa.average().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn swalr_rejects_bad() {
        assert!(matches!(
            SwaLr::new(0.0, 1e-3, 10, SwaLrMode::Linear),
            Err(TrainError::InvalidLearningRate { .. })
        ));
        assert!(matches!(
            SwaLr::new(1e-2, 1e-3, 0, SwaLrMode::Linear),
            Err(TrainError::Internal { .. })
        ));
    }

    /// Linear SWALR anneals start_lr → swa_lr over the window, matching the
    /// closed-form linear interpolation, then holds constant.
    #[test]
    fn swalr_linear_closed_form() {
        let start = 1e-1;
        let swa = 1e-3;
        let steps = 10;
        let mut s = SwaLr::new(start, swa, steps, SwaLrMode::Linear).expect("valid");
        for k in 1..steps {
            let lr = s.step();
            let frac = k as f64 / steps as f64;
            let expect = start + (swa - start) * frac;
            assert!((lr - expect).abs() < 1e-12, "step {k}: {lr} vs {expect}");
        }
        // Beyond the window the LR is held at swa_lr.
        let held = s.step();
        assert!(
            (held - swa).abs() < 1e-12,
            "tail should hold swa_lr, got {held}"
        );
    }

    /// Cosine SWALR reaches swa_lr at the end of the anneal window and holds.
    #[test]
    fn swalr_cosine_reaches_target() {
        let mut s = SwaLr::new(1e-1, 1e-3, 8, SwaLrMode::Cosine).expect("valid");
        let mut last = 0.0;
        for _ in 0..8 {
            last = s.step();
        }
        assert!(
            (last - 1e-3).abs() < 1e-9,
            "cosine should reach swa_lr, got {last}"
        );
        assert_eq!(s.swa_lr(), 1e-3);
    }
}
