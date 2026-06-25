//! DP-Adam convergence harness with an empirical (ε, δ) report.
//!
//! A small, fully self-contained CPU training harness that exercises
//! [`super::dp_adam::DpAdamState`] end-to-end on a synthetic regression task and
//! reports the spent `(ε, δ)` privacy budget through the crate's moments
//! accountant ([`crate::mechanism::sampled_gaussian::SampledGaussianMechanism`],
//! the Sampled-Gaussian RDP accountant used by DP-SGD).
//!
//! The dataset is generated *in process* with the crate [`LcgRng`] — there is no
//! external download (no MNIST/CIFAR). Each example is `(x, y)` with
//! `y = w⋆·x + b⋆ + ξ` for a fixed ground-truth `(w⋆, b⋆)` and small label noise
//! `ξ`. The model is linear least-squares, so the per-sample gradient of the
//! squared-error loss `½(ŷ − y)²` is `(ŷ − y)·[x ; 1]` — cheap, exact, and a
//! faithful smooth convex objective on which DP-Adam should make monotone
//! progress despite the injected noise.
//!
//! Training uses minibatches subsampled uniformly at random each step (rate
//! `q = batch / n`), and the accountant composes one Sampled-Gaussian RDP term
//! per step, exactly as a moments-accountant-tracked DP-SGD/DP-Adam loop would.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;
use crate::mechanism::sampled_gaussian::{SampledGaussianConfig, SampledGaussianMechanism};
use crate::optimizer::dp_adam::{DpAdamConfig, DpAdamState};

/// Configuration of the synthetic DP-Adam convergence harness.
#[derive(Debug, Clone)]
pub struct DpAdamHarnessConfig {
    /// Number of synthetic training examples `n` (a few hundred).
    pub n_samples: usize,
    /// Feature dimension `d` (model has `d + 1` parameters incl. bias).
    pub n_features: usize,
    /// Minibatch size `B` (gives subsampling rate `q = B / n`).
    pub batch_size: usize,
    /// Number of passes over the dataset.
    pub epochs: usize,
    /// Standard deviation of the label noise `ξ`.
    pub label_noise: f64,
    /// DP-Adam configuration (clip, σ, lr, β₁, β₂, ε_adam).
    pub adam: DpAdamConfig,
    /// Maximum integer Rényi order tracked by the accountant (≥ 2).
    pub max_order: usize,
}

impl DpAdamHarnessConfig {
    /// A sensible default: 256 examples, 4 features, batch 32, 12 epochs, with a
    /// moderate noise multiplier that still permits convergence.
    #[must_use]
    pub fn small() -> Self {
        Self {
            n_samples: 256,
            n_features: 4,
            batch_size: 32,
            epochs: 12,
            label_noise: 0.05,
            adam: DpAdamConfig {
                sigma: 0.8,
                grad_clip: 4.0,
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon_adam: 1e-8,
            },
            max_order: 32,
        }
    }
}

/// One synthetic regression dataset: `x` rows (flattened) + targets `y`.
#[derive(Debug, Clone)]
pub struct SyntheticDataset {
    /// Flattened design matrix, shape `[n_samples × n_features]` row-major.
    pub x: Vec<f64>,
    /// Targets, length `n_samples`.
    pub y: Vec<f64>,
    /// Number of examples.
    pub n_samples: usize,
    /// Feature dimension.
    pub n_features: usize,
    /// Ground-truth weights used to generate `y`.
    pub true_w: Vec<f64>,
    /// Ground-truth bias used to generate `y`.
    pub true_b: f64,
}

impl SyntheticDataset {
    /// Generate a synthetic linear-regression dataset with the crate LCG RNG.
    ///
    /// Features are drawn `N(0,1)`; ground-truth weights are a fixed bounded
    /// pattern; targets are `w⋆·x + b⋆ + ξ` with `ξ ~ N(0, label_noise²)`.
    fn generate(n_samples: usize, n_features: usize, label_noise: f64, rng: &mut LcgRng) -> Self {
        // Fixed, well-conditioned ground truth in [-1, 1].
        let true_w: Vec<f64> = (0..n_features)
            .map(|j| (((j % 5) as f64) - 2.0) / 2.0)
            .collect();
        let true_b = 0.3;

        let mut x = Vec::with_capacity(n_samples * n_features);
        let mut y = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut dot = true_b;
            for &w in &true_w {
                let (xi, _) = rng.normal_pair();
                x.push(xi);
                dot += w * xi;
            }
            let (xi_noise, _) = rng.normal_pair();
            y.push(dot + label_noise * xi_noise);
        }
        Self {
            x,
            y,
            n_samples,
            n_features,
            true_w,
            true_b,
        }
    }

    /// Per-sample gradient of `½(ŷ − y)²` for sample `idx` under the current
    /// parameters `θ = [w₀ … w_{d−1}, b]`, written into `out` (length `d + 1`).
    fn sample_gradient(&self, idx: usize, theta: &[f64], out: &mut [f64]) {
        let d = self.n_features;
        let row = &self.x[idx * d..idx * d + d];
        // Prediction ŷ = w·x + b.
        let mut pred = theta[d]; // bias is the last parameter
        for j in 0..d {
            pred += theta[j] * row[j];
        }
        let residual = pred - self.y[idx];
        for j in 0..d {
            out[j] = residual * row[j];
        }
        out[d] = residual; // ∂/∂b
    }

    /// Mean squared-error loss `(1/n)·Σ ½(ŷ − y)²` over the whole dataset.
    fn full_loss(&self, theta: &[f64]) -> f64 {
        let d = self.n_features;
        let mut total = 0.0;
        for i in 0..self.n_samples {
            let row = &self.x[i * d..i * d + d];
            let mut pred = theta[d];
            for j in 0..d {
                pred += theta[j] * row[j];
            }
            let r = pred - self.y[i];
            total += 0.5 * r * r;
        }
        total / self.n_samples as f64
    }
}

/// Result of running the DP-Adam convergence harness.
#[derive(Debug, Clone)]
pub struct DpAdamHarnessReport {
    /// Full-dataset loss measured before training and after each epoch
    /// (length `epochs + 1`).
    pub epoch_losses: Vec<f64>,
    /// Spent ε at the target δ measured after each epoch (length `epochs`),
    /// monotonically non-decreasing.
    pub epoch_epsilons: Vec<f64>,
    /// Total number of optimisation steps taken.
    pub steps: usize,
    /// Final spent ε at the target δ.
    pub final_epsilon: f64,
    /// The target δ used for the (ε, δ) report.
    pub delta: f64,
    /// Learned parameters `[w₀ … w_{d−1}, b]`.
    pub theta: Vec<f64>,
}

/// The synthetic DP-Adam convergence harness.
#[derive(Debug, Clone)]
pub struct DpAdamHarness {
    cfg: DpAdamHarnessConfig,
}

impl DpAdamHarness {
    /// Construct a harness, validating the configuration.
    ///
    /// # Errors
    /// - `InvalidParameter` for a zero sample count / feature count / batch size
    ///   / epoch count, a batch larger than the dataset, or `max_order < 2`.
    /// - Propagates Adam-config validation via [`DpAdamConfig::new`] re-check.
    pub fn new(cfg: DpAdamHarnessConfig) -> PrivacyResult<Self> {
        if cfg.n_samples == 0 || cfg.n_features == 0 {
            return Err(PrivacyError::InvalidParameter(
                "n_samples and n_features must be ≥ 1".into(),
            ));
        }
        if cfg.batch_size == 0 || cfg.batch_size > cfg.n_samples {
            return Err(PrivacyError::InvalidParameter(format!(
                "batch_size must be in 1..=n_samples, got {}",
                cfg.batch_size
            )));
        }
        if cfg.epochs == 0 {
            return Err(PrivacyError::InvalidParameter("epochs must be ≥ 1".into()));
        }
        if cfg.max_order < 2 {
            return Err(PrivacyError::InvalidParameter(
                "max_order must be ≥ 2".into(),
            ));
        }
        // Re-validate the Adam config (round-trips its invariants).
        let a = &cfg.adam;
        DpAdamConfig::new(
            a.sigma,
            a.grad_clip,
            a.learning_rate,
            a.beta1,
            a.beta2,
            a.epsilon_adam,
        )?;
        Ok(Self { cfg })
    }

    /// Read-only access to the configuration.
    #[must_use]
    pub fn config(&self) -> &DpAdamHarnessConfig {
        &self.cfg
    }

    /// Run the full DP-Adam training loop and produce a convergence + (ε, δ)
    /// report.
    ///
    /// `seed` drives both the synthetic-data generation and the DP-Adam noise /
    /// minibatch sampling, so the run is fully reproducible. `delta` is the
    /// target δ for the empirical ε report.
    ///
    /// # Errors
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    /// - Propagates accountant / optimiser errors.
    pub fn run(&self, seed: u64, delta: f64) -> PrivacyResult<DpAdamHarnessReport> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        let cfg = &self.cfg;
        let d = cfg.n_features;
        let n_params = d + 1;

        // Separate RNG streams (different seeds) for data, minibatch sampling,
        // and DP-Adam noise, so they do not alias.
        let mut data_rng = LcgRng::new(seed);
        let dataset = SyntheticDataset::generate(cfg.n_samples, d, cfg.label_noise, &mut data_rng);
        let mut sample_rng = LcgRng::new(seed ^ 0xA5A5_A5A5_5A5A_5A5A);
        let mut noise_rng = LcgRng::new(seed ^ 0x1357_9BDF_2468_ACE0);

        let mut state = DpAdamState::new(n_params);

        // Subsampling rate q = B / n for the moments accountant.
        let q = cfg.batch_size as f64 / cfg.n_samples as f64;
        let acct_cfg = SampledGaussianConfig::new(q, cfg.adam.sigma, cfg.max_order)?;
        let mut accountant = SampledGaussianMechanism::new(&acct_cfg)?;

        let steps_per_epoch = cfg.n_samples / cfg.batch_size;
        let mut epoch_losses = Vec::with_capacity(cfg.epochs + 1);
        let mut epoch_epsilons = Vec::with_capacity(cfg.epochs);
        epoch_losses.push(dataset.full_loss(&state.params));

        let mut grad_buf = vec![0.0f64; n_params];
        let mut batch_grads = vec![0.0f64; cfg.batch_size * n_params];
        let mut total_steps = 0usize;

        for _epoch in 0..cfg.epochs {
            for _ in 0..steps_per_epoch {
                // Assemble a minibatch of per-sample gradients (flat
                // [batch × n_params]) at the current parameters.
                for b in 0..cfg.batch_size {
                    let idx = (sample_rng.next_u64() as usize) % cfg.n_samples;
                    dataset.sample_gradient(idx, &state.params, &mut grad_buf);
                    let dst = &mut batch_grads[b * n_params..b * n_params + n_params];
                    dst.copy_from_slice(&grad_buf);
                }

                // One DP-Adam step (per-sample clip + Gaussian noise + Adam).
                state.step(&batch_grads, cfg.batch_size, &cfg.adam, &mut noise_rng)?;

                // Compose one Sampled-Gaussian RDP term in the accountant.
                accountant.compose(1)?;
                total_steps += 1;
            }
            epoch_losses.push(dataset.full_loss(&state.params));
            epoch_epsilons.push(accountant.get_epsilon(delta)?);
        }

        let final_epsilon = accountant.get_epsilon(delta)?;
        Ok(DpAdamHarnessReport {
            epoch_losses,
            epoch_epsilons,
            steps: total_steps,
            final_epsilon,
            delta,
            theta: state.params.clone(),
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. End-to-end: DP-Adam converges (loss drops) AND the accountant reports a
    //    finite ε at the target δ that grows monotonically across epochs.
    #[test]
    fn dp_adam_converges_and_reports_finite_growing_epsilon() {
        let harness = DpAdamHarness::new(DpAdamHarnessConfig::small()).expect("harness");
        let delta = 1e-5;
        let report = harness.run(20240621, delta).expect("run");

        // (a) Loss decreases over training.
        let first = report.epoch_losses.first().copied().expect("first");
        let last = report.epoch_losses.last().copied().expect("last");
        assert!(
            last < first,
            "DP-Adam should reduce the loss: start {first:.5} → end {last:.5}"
        );
        // Substantial progress, not just a hair.
        assert!(
            last < 0.5 * first,
            "expected the loss to at least halve: {first:.5} → {last:.5}"
        );

        // (b) Final ε is finite and positive at the target δ.
        assert!(
            report.final_epsilon.is_finite() && report.final_epsilon > 0.0,
            "ε must be finite and positive: {}",
            report.final_epsilon
        );

        // (c) Per-epoch ε grows monotonically (more steps → more spent budget).
        for w in report.epoch_epsilons.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "ε must be monotone non-decreasing: {} → {}",
                w[0],
                w[1]
            );
        }
        // Strictly larger from first epoch to last (steps strictly increased).
        let eps_first = report.epoch_epsilons.first().copied().expect("e0");
        let eps_last = report.epoch_epsilons.last().copied().expect("eN");
        assert!(
            eps_last > eps_first,
            "ε should strictly grow over epochs: {eps_first} → {eps_last}"
        );

        // (d) Bookkeeping: total steps = epochs × steps_per_epoch.
        let cfg = harness.config();
        assert_eq!(report.steps, cfg.epochs * (cfg.n_samples / cfg.batch_size));
        assert_eq!(report.theta.len(), cfg.n_features + 1);
    }

    // 2. The run is deterministic for a fixed seed.
    #[test]
    fn run_is_deterministic() {
        let harness = DpAdamHarness::new(DpAdamHarnessConfig::small()).expect("harness");
        let a = harness.run(7, 1e-5).expect("a");
        let b = harness.run(7, 1e-5).expect("b");
        assert_eq!(a.theta, b.theta);
        assert_eq!(a.epoch_losses, b.epoch_losses);
        assert_eq!(a.epoch_epsilons, b.epoch_epsilons);
        assert!((a.final_epsilon - b.final_epsilon).abs() < 1e-15);
    }

    // 3. Different seeds give different trained parameters (data + noise differ).
    #[test]
    fn different_seeds_differ() {
        let harness = DpAdamHarness::new(DpAdamHarnessConfig::small()).expect("harness");
        let a = harness.run(1, 1e-5).expect("a");
        let b = harness.run(2, 1e-5).expect("b");
        assert_ne!(a.theta, b.theta);
    }

    // 4. Learned parameters approach the ground truth (utility under DP noise).
    #[test]
    fn recovers_ground_truth_approximately() {
        let harness = DpAdamHarness::new(DpAdamHarnessConfig::small()).expect("harness");
        let report = harness.run(123456, 1e-5).expect("run");
        // Reconstruct the ground truth deterministically (same data seed).
        let mut data_rng = LcgRng::new(123456);
        let cfg = harness.config();
        let ds = SyntheticDataset::generate(
            cfg.n_samples,
            cfg.n_features,
            cfg.label_noise,
            &mut data_rng,
        );
        // Weights should be in the right neighbourhood (DP noise loosens this).
        for (j, &w_true) in ds.true_w.iter().enumerate() {
            assert!(
                (report.theta[j] - w_true).abs() < 0.4,
                "w[{j}] = {} far from truth {w_true}",
                report.theta[j]
            );
        }
        assert!(
            (report.theta[cfg.n_features] - ds.true_b).abs() < 0.4,
            "bias {} far from truth {}",
            report.theta[cfg.n_features],
            ds.true_b
        );
    }

    // 5. Smaller δ ⇒ larger reported ε for the same training run.
    #[test]
    fn smaller_delta_larger_epsilon() {
        let harness = DpAdamHarness::new(DpAdamHarnessConfig::small()).expect("harness");
        let tight = harness.run(99, 1e-7).expect("tight");
        let loose = harness.run(99, 1e-3).expect("loose");
        assert!(
            tight.final_epsilon > loose.final_epsilon,
            "smaller δ → larger ε: {} > {}",
            tight.final_epsilon,
            loose.final_epsilon
        );
    }

    // 6. Validation of bad configs and bad δ.
    #[test]
    fn config_and_delta_validation() {
        let mut bad = DpAdamHarnessConfig::small();
        bad.batch_size = bad.n_samples + 1;
        assert!(DpAdamHarness::new(bad).is_err());

        let mut bad2 = DpAdamHarnessConfig::small();
        bad2.epochs = 0;
        assert!(DpAdamHarness::new(bad2).is_err());

        let mut bad3 = DpAdamHarnessConfig::small();
        bad3.max_order = 1;
        assert!(DpAdamHarness::new(bad3).is_err());

        let harness = DpAdamHarness::new(DpAdamHarnessConfig::small()).expect("ok");
        assert!(matches!(
            harness.run(1, 0.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(matches!(
            harness.run(1, 1.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
    }
}
