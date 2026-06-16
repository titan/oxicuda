//! Bayesian MAML (BMAML) — Yoon, Kim, Dia, Kim, Bengio & Ahn (NeurIPS 2018).
//!
//! # Background
//!
//! BMAML ("Bayesian Model-Agnostic Meta-Learning") replaces MAML's single
//! point-estimate of the task parameters with an **ensemble of `M` particles**
//! `{θ⁽¹⁾, …, θ⁽ᴹ⁾}` that together approximate the task posterior.  The inner
//! loop adapts the particles with **Stein Variational Gradient Descent**
//! (SVGD), which transports the ensemble toward the posterior while a repulsive
//! kernel term keeps the particles diverse:
//!
//! ```text
//! φᵢ ← φᵢ + ε · 1/M Σⱼ [ k(φⱼ, φᵢ) ∇φⱼ log p(φⱼ)  +  ∇φⱼ k(φⱼ, φᵢ) ]
//! ```
//!
//! Here `∇ log p = −∇L_train` (negative loss gradient, treating the loss as a
//! negative log-posterior) and `k` is the RBF kernel
//! `k(a, b) = exp(−‖a−b‖² / h)` with the **median-heuristic** bandwidth `h`.
//! The first term drives particles down the loss; the second term is the
//! repulsion that prevents mode collapse.
//!
//! The meta-update follows the paper's **Chaser Loss**: after `n_inner` SVGD
//! steps on the support set (the "leader"), a few extra steps on support ∪ query
//! give a "chaser"; the meta-gradient drives the leader ensemble toward the
//! chaser, `½ Σᵢ ‖θ_leaderᵢ − θ_chaserᵢ‖²`.  We implement the standard
//! simplification where the meta-direction is the leader→chaser displacement.
//!
//! All operations are on flat `Vec<f32>` parameter buffers with gradient
//! closures `&[f32] → Vec<f32>`, matching the rest of `oxicuda-meta`.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Boxed gradient closure mapping parameters to the loss gradient `∇L`.
pub type GradFn = Box<dyn Fn(&[f32]) -> Vec<f32>>;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Hyper-parameters for Bayesian MAML.
#[derive(Debug, Clone)]
pub struct BayesianMamlConfig {
    /// Number of particles `M` in the ensemble.
    pub n_particles: usize,
    /// SVGD inner step size `ε`.
    pub inner_lr: f32,
    /// Number of SVGD inner steps for the leader ensemble.
    pub n_inner: usize,
    /// Number of extra SVGD steps producing the chaser ensemble.
    pub n_chaser: usize,
    /// Outer (meta) learning rate.
    pub outer_lr: f32,
    /// Standard deviation of the Gaussian perturbation used to spread the
    /// initial particle cloud around the meta-init.
    pub init_spread: f32,
}

impl Default for BayesianMamlConfig {
    fn default() -> Self {
        Self {
            n_particles: 5,
            inner_lr: 0.05,
            n_inner: 4,
            n_chaser: 2,
            outer_lr: 0.1,
            init_spread: 0.05,
        }
    }
}

impl BayesianMamlConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`MetaError::InvalidEpisodeConfig`] — if `n_particles < 2`.
    /// * [`MetaError::InvalidLr`] — if `inner_lr` or `outer_lr` is
    ///   non-positive / non-finite.
    pub fn validate(&self) -> MetaResult<()> {
        if self.n_particles < 2 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_particles must be ≥ 2 for SVGD repulsion".into(),
            });
        }
        if self.inner_lr <= 0.0 || !self.inner_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: self.inner_lr });
        }
        if self.outer_lr <= 0.0 || !self.outer_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: self.outer_lr });
        }
        Ok(())
    }
}

// ─── SVGD primitives ─────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length vectors.
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Median-heuristic RBF bandwidth: `h = med(pairwise sq dists) / ln(M)`.
///
/// Returns a strictly positive bandwidth (floored).
pub fn median_bandwidth(particles: &[Vec<f32>]) -> f32 {
    let m = particles.len();
    if m < 2 {
        return 1.0;
    }
    let mut dists = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            dists.push(sq_dist(&particles[i], &particles[j]));
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = dists[dists.len() / 2];
    let denom = (m as f32).ln().max(1e-3);
    (med / denom).max(1e-6)
}

/// Perform one SVGD update step on `particles` in place.
///
/// `grad_logp(θ)` must return `∇ log p(θ)` (i.e. the *negative* loss gradient).
///
/// # Errors
///
/// [`MetaError::DimensionMismatch`] if a gradient has the wrong length.
pub fn svgd_step(
    particles: &mut [Vec<f32>],
    grad_logp: &dyn Fn(&[f32]) -> Vec<f32>,
    lr: f32,
) -> MetaResult<()> {
    let m = particles.len();
    if m == 0 {
        return Ok(());
    }
    let dim = particles[0].len();
    let h = median_bandwidth(particles);

    // Precompute ∇log p for every particle.
    let mut grads = Vec::with_capacity(m);
    for p in particles.iter() {
        let g = grad_logp(p);
        if g.len() != dim {
            return Err(MetaError::DimensionMismatch {
                expected: dim,
                got: g.len(),
            });
        }
        grads.push(g);
    }

    // φᵢ_update = 1/M Σⱼ [ k(j,i) ∇logp(j) + ∇φⱼ k(j,i) ]
    // with RBF kernel k = exp(−‖φⱼ−φᵢ‖²/h):  ∇φⱼ k = k · (−2/h)(φⱼ−φᵢ).
    let mut updates = vec![vec![0.0_f32; dim]; m];
    for i in 0..m {
        for j in 0..m {
            let d2 = sq_dist(&particles[j], &particles[i]);
            let k = (-d2 / h).exp();
            let coef = -2.0 / h * k;
            let gj = &grads[j];
            for t in 0..dim {
                let repulse = coef * (particles[j][t] - particles[i][t]);
                updates[i][t] += k * gj[t] + repulse;
            }
        }
    }
    let inv_m = 1.0 / m as f32;
    for i in 0..m {
        for t in 0..dim {
            particles[i][t] += lr * inv_m * updates[i][t];
        }
    }
    Ok(())
}

// ─── Particle ensemble ───────────────────────────────────────────────────────

/// Spread `M` particles around `theta` with Gaussian noise of std `spread`.
fn init_particles(theta: &[f32], m: usize, spread: f32, rng: &mut LcgRng) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(m);
    for _ in 0..m {
        let mut p = theta.to_vec();
        for v in p.iter_mut() {
            // next_f32 ∈ [0,1) → centred noise in [−0.5, 0.5)·2·spread.
            *v += (rng.next_f32() - 0.5) * 2.0 * spread;
        }
        out.push(p);
    }
    out
}

/// Run `steps` SVGD updates on a particle ensemble using `−∇L` as `∇log p`.
fn run_svgd(
    particles: &mut [Vec<f32>],
    loss_grad: &dyn Fn(&[f32]) -> Vec<f32>,
    lr: f32,
    steps: usize,
) -> MetaResult<()> {
    let grad_logp = |theta: &[f32]| -> Vec<f32> { loss_grad(theta).iter().map(|&g| -g).collect() };
    for _ in 0..steps {
        svgd_step(particles, &grad_logp, lr)?;
    }
    Ok(())
}

// ─── Learner ─────────────────────────────────────────────────────────────────

/// Bayesian MAML meta-learner holding the shared particle-init `θ_meta`.
pub struct BayesianMaml {
    theta: Vec<f32>,
    config: BayesianMamlConfig,
    rng: LcgRng,
}

/// One task for a BMAML meta-update.
pub struct BayesianMamlTask {
    /// Gradient of the support (train) loss.
    pub support_grad: GradFn,
    /// Gradient of the support ∪ query loss (used for the chaser).
    pub joint_grad: GradFn,
}

impl BayesianMaml {
    /// Create a BMAML learner over `dim` parameters initialised to `init`.
    ///
    /// # Errors
    ///
    /// * [`MetaError::InvalidEpisodeConfig`] — if `init` is empty.
    /// * Propagates [`BayesianMamlConfig::validate`].
    pub fn new(init: Vec<f32>, config: BayesianMamlConfig, seed: u64) -> MetaResult<Self> {
        if init.is_empty() {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "init parameter vector must be non-empty".into(),
            });
        }
        config.validate()?;
        Ok(Self {
            theta: init,
            config,
            rng: LcgRng::new(seed),
        })
    }

    /// Read-only view of the meta-parameters.
    #[inline]
    #[must_use]
    pub fn theta(&self) -> &[f32] {
        &self.theta
    }

    /// Adapt an ensemble on a task's support set and return the leader particles.
    ///
    /// This is the inference path: it produces an `M`-particle posterior
    /// approximation for a single task.
    ///
    /// # Errors
    ///
    /// Propagates SVGD gradient errors.
    pub fn adapt(
        &mut self,
        support_grad: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> MetaResult<Vec<Vec<f32>>> {
        let mut particles = init_particles(
            &self.theta,
            self.config.n_particles,
            self.config.init_spread,
            &mut self.rng,
        );
        run_svgd(
            &mut particles,
            support_grad,
            self.config.inner_lr,
            self.config.n_inner,
        )?;
        Ok(particles)
    }

    /// One meta-update over a batch of tasks using the chaser loss.
    ///
    /// For each task: build a leader ensemble (`n_inner` SVGD steps on support),
    /// then a chaser (`n_chaser` further steps on support ∪ query); the
    /// meta-gradient is the mean leader→chaser displacement.  `θ_meta` moves
    /// along the averaged displacement.
    ///
    /// Returns the mean squared leader→chaser displacement (chaser loss proxy).
    ///
    /// # Errors
    ///
    /// * [`MetaError::EmptySupport`] — if `tasks` is empty.
    /// * Propagates SVGD errors.
    pub fn meta_step(&mut self, tasks: &[BayesianMamlTask]) -> MetaResult<f32> {
        if tasks.is_empty() {
            return Err(MetaError::EmptySupport);
        }
        let dim = self.theta.len();
        let m = self.config.n_particles;
        let mut meta_dir = vec![0.0_f32; dim];
        let mut total_chaser = 0.0_f32;

        for task in tasks {
            // Leader: n_inner SVGD steps on the support set.
            let mut leader = init_particles(&self.theta, m, self.config.init_spread, &mut self.rng);
            run_svgd(
                &mut leader,
                task.support_grad.as_ref(),
                self.config.inner_lr,
                self.config.n_inner,
            )?;

            // Chaser: continue from the leader with extra steps on the joint
            // (support ∪ query) loss.
            let mut chaser = leader.clone();
            run_svgd(
                &mut chaser,
                task.joint_grad.as_ref(),
                self.config.inner_lr,
                self.config.n_chaser,
            )?;

            // Accumulate mean leader→chaser displacement and chaser loss.
            for p in 0..m {
                for t in 0..dim {
                    let d = chaser[p][t] - leader[p][t];
                    meta_dir[t] += d;
                    total_chaser += d * d;
                }
            }
        }

        let scale = 1.0 / (tasks.len() as f32 * m as f32);
        let mut moved = false;
        for (theta_t, &dir) in self.theta.iter_mut().zip(meta_dir.iter()) {
            let step = self.config.outer_lr * dir * scale;
            *theta_t += step;
            if step.abs() > 0.0 {
                moved = true;
            }
        }
        let _ = moved;
        let chaser_loss = 0.5 * total_chaser * scale;
        if !chaser_loss.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "bmaml meta_step chaser loss non-finite".into(),
            });
        }
        Ok(chaser_loss)
    }

    /// Ensemble-mean prediction utility: average a set of particle parameter
    /// vectors into a single point estimate.
    ///
    /// # Errors
    ///
    /// * [`MetaError::EmptySupport`] — if `particles` is empty.
    /// * [`MetaError::DimensionMismatch`] — if particle lengths differ.
    pub fn ensemble_mean(particles: &[Vec<f32>]) -> MetaResult<Vec<f32>> {
        if particles.is_empty() {
            return Err(MetaError::EmptySupport);
        }
        let dim = particles[0].len();
        let mut mean = vec![0.0_f32; dim];
        for p in particles {
            if p.len() != dim {
                return Err(MetaError::DimensionMismatch {
                    expected: dim,
                    got: p.len(),
                });
            }
            for (m_, &v) in mean.iter_mut().zip(p.iter()) {
                *m_ += v;
            }
        }
        let inv = 1.0 / particles.len() as f32;
        for m_ in mean.iter_mut() {
            *m_ *= inv;
        }
        Ok(mean)
    }

    /// Empirical per-dimension variance across a particle ensemble (uncertainty).
    ///
    /// # Errors
    ///
    /// Propagates [`BayesianMaml::ensemble_mean`] errors.
    pub fn ensemble_variance(particles: &[Vec<f32>]) -> MetaResult<Vec<f32>> {
        let mean = Self::ensemble_mean(particles)?;
        let dim = mean.len();
        let mut var = vec![0.0_f32; dim];
        for p in particles {
            for t in 0..dim {
                let d = p[t] - mean[t];
                var[t] += d * d;
            }
        }
        let inv = 1.0 / particles.len() as f32;
        for v in var.iter_mut() {
            *v *= inv;
        }
        Ok(var)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradient of `½ Σ (θ_i − c_i)²` → `θ − c`.
    fn quad_grad(c: Vec<f32>) -> GradFn {
        Box::new(move |theta: &[f32]| theta.iter().zip(c.iter()).map(|(&t, &ci)| t - ci).collect())
    }

    #[test]
    fn config_default_valid() {
        assert!(BayesianMamlConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_one_particle() {
        let c = BayesianMamlConfig {
            n_particles: 1,
            ..BayesianMamlConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn config_rejects_bad_lr() {
        let c = BayesianMamlConfig {
            inner_lr: -0.1,
            ..BayesianMamlConfig::default()
        };
        assert!(matches!(c.validate(), Err(MetaError::InvalidLr { .. })));
    }

    #[test]
    fn bandwidth_positive_and_scales() {
        let ps = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let h = median_bandwidth(&ps);
        assert!(h > 0.0);
        // Spreading particles further apart increases the bandwidth.
        let ps2 = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]];
        assert!(median_bandwidth(&ps2) > h);
    }

    #[test]
    fn bandwidth_single_particle_defaults() {
        assert_eq!(median_bandwidth(&[vec![1.0, 2.0]]), 1.0);
    }

    #[test]
    fn svgd_step_moves_particles_toward_target() {
        // ∇log p = −(θ − c) points toward c = [4, 4]; the ensemble mean should
        // move closer to c after several steps.
        let c = vec![4.0_f32, 4.0];
        let logp = {
            let c = c.clone();
            move |theta: &[f32]| -> Vec<f32> {
                theta
                    .iter()
                    .zip(c.iter())
                    .map(|(&t, &ci)| -(t - ci))
                    .collect()
            }
        };
        let mut ps = vec![vec![0.0_f32, 0.0], vec![0.1, -0.1], vec![-0.1, 0.1]];
        let mean0 = BayesianMaml::ensemble_mean(&ps).expect("m0");
        for _ in 0..50 {
            svgd_step(&mut ps, &logp, 0.1).expect("step");
        }
        let mean1 = BayesianMaml::ensemble_mean(&ps).expect("m1");
        let d0 = sq_dist(&mean0, &c);
        let d1 = sq_dist(&mean1, &c);
        assert!(d1 < d0, "ensemble mean should approach target: {d0} → {d1}");
    }

    #[test]
    fn svgd_keeps_particles_distinct() {
        // The repulsive term must prevent total collapse: variance stays > 0.
        let c = vec![1.0_f32, 1.0];
        let logp = {
            let c = c.clone();
            move |theta: &[f32]| -> Vec<f32> {
                theta
                    .iter()
                    .zip(c.iter())
                    .map(|(&t, &ci)| -(t - ci))
                    .collect()
            }
        };
        let mut ps = vec![
            vec![0.0_f32, 0.0],
            vec![0.5, 0.2],
            vec![-0.3, 0.4],
            vec![0.1, -0.5],
        ];
        for _ in 0..40 {
            svgd_step(&mut ps, &logp, 0.05).expect("step");
        }
        let var = BayesianMaml::ensemble_variance(&ps).expect("var");
        assert!(var.iter().any(|&v| v > 1e-6), "particles must stay diverse");
    }

    #[test]
    fn svgd_step_rejects_bad_grad_len() {
        let bad = |_theta: &[f32]| -> Vec<f32> { vec![0.0; 5] };
        let mut ps = vec![vec![0.0_f32, 0.0], vec![1.0, 1.0]];
        assert!(svgd_step(&mut ps, &bad, 0.1).is_err());
    }

    #[test]
    fn new_rejects_empty_init() {
        assert!(matches!(
            BayesianMaml::new(vec![], BayesianMamlConfig::default(), 0),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn adapt_produces_correct_ensemble_size() {
        let cfg = BayesianMamlConfig {
            n_particles: 6,
            ..BayesianMamlConfig::default()
        };
        let mut m = BayesianMaml::new(vec![0.0, 0.0, 0.0], cfg, 7).expect("m");
        let g = quad_grad(vec![1.0, 2.0, 3.0]);
        let ens = m.adapt(g.as_ref()).expect("adapt");
        assert_eq!(ens.len(), 6);
        for p in &ens {
            assert_eq!(p.len(), 3);
            assert!(p.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn adapt_moves_ensemble_toward_support_min() {
        let cfg = BayesianMamlConfig {
            n_particles: 5,
            inner_lr: 0.1,
            n_inner: 30,
            init_spread: 0.05,
            ..BayesianMamlConfig::default()
        };
        let mut m = BayesianMaml::new(vec![0.0, 0.0], cfg, 11).expect("m");
        let target = vec![3.0_f32, -2.0];
        let g = quad_grad(target.clone());
        let ens = m.adapt(g.as_ref()).expect("adapt");
        let mean = BayesianMaml::ensemble_mean(&ens).expect("mean");
        // The ensemble mean should be much closer to the support minimum than
        // the meta-init at the origin.
        assert!(sq_dist(&mean, &target) < sq_dist(&[0.0, 0.0], &target));
    }

    #[test]
    fn meta_step_rejects_empty() {
        let mut m = BayesianMaml::new(vec![0.0, 0.0], BayesianMamlConfig::default(), 1).expect("m");
        assert!(matches!(m.meta_step(&[]), Err(MetaError::EmptySupport)));
    }

    #[test]
    fn meta_step_returns_finite_chaser_loss() {
        let mut m = BayesianMaml::new(vec![0.0, 0.0], BayesianMamlConfig::default(), 3).expect("m");
        let task = BayesianMamlTask {
            support_grad: quad_grad(vec![1.0, 1.0]),
            joint_grad: quad_grad(vec![1.5, 1.5]),
        };
        let loss = m.meta_step(std::slice::from_ref(&task)).expect("step");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn meta_step_moves_theta_when_chaser_differs() {
        // Support and joint minima differ, so leader and chaser disagree and the
        // meta-init must move.
        let cfg = BayesianMamlConfig {
            outer_lr: 1.0,
            n_inner: 3,
            n_chaser: 3,
            ..BayesianMamlConfig::default()
        };
        let mut m = BayesianMaml::new(vec![0.0, 0.0], cfg, 5).expect("m");
        let before = m.theta().to_vec();
        for _ in 0..5 {
            let task = BayesianMamlTask {
                support_grad: quad_grad(vec![0.0, 0.0]),
                joint_grad: quad_grad(vec![5.0, 5.0]),
            };
            m.meta_step(std::slice::from_ref(&task)).expect("step");
        }
        let after = m.theta();
        let moved: f32 = before
            .iter()
            .zip(after.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(moved > 1e-4, "θ should move toward the chaser target");
    }

    #[test]
    fn ensemble_mean_and_variance_basic() {
        let ps = vec![vec![0.0_f32, 2.0], vec![2.0, 2.0], vec![4.0, 2.0]];
        let mean = BayesianMaml::ensemble_mean(&ps).expect("mean");
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!((mean[1] - 2.0).abs() < 1e-6);
        let var = BayesianMaml::ensemble_variance(&ps).expect("var");
        // var of {0,2,4} = 8/3; feature 1 is constant → 0.
        assert!((var[0] - 8.0 / 3.0).abs() < 1e-5);
        assert!(var[1].abs() < 1e-6);
    }

    #[test]
    fn ensemble_mean_rejects_empty() {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert!(matches!(
            BayesianMaml::ensemble_mean(&empty),
            Err(MetaError::EmptySupport)
        ));
    }
}
