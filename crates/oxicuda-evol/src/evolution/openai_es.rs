//! OpenAI Evolution Strategies (OpenAI-ES) — a scalable, gradient-free policy optimiser.
//!
//! Reference: T. Salimans, J. Ho, X. Chen, S. Sidor, I. Sutskever,
//! "Evolution Strategies as a Scalable Alternative to Reinforcement Learning",
//! arXiv:1703.03864, 2017. <https://arxiv.org/abs/1703.03864>
//!
//! ## Overview
//! OpenAI-ES treats the objective as a black box and estimates a *search gradient* of
//! the Gaussian-smoothed objective
//!
//! ```text
//! J(θ) = E_{ε ~ N(0, I)} [ f(θ + σ·ε) ]
//! ```
//!
//! by sampling a population of perturbations `ε_i` and forming the Monte-Carlo estimate
//!
//! ```text
//! ∇_θ J(θ) ≈ (1 / (n·σ)) · Σ_i  u_i · ε_i
//! ```
//!
//! where `u_i` is a *rank-normalised* shaping of the raw fitness `f(θ + σ·ε_i)` (centred
//! ranks in `[-0.5, 0.5]`). The estimate is fed to a momentum/Adam-style optimiser to
//! ascend `J`.
//!
//! Two refinements from the paper are implemented:
//! * **Antithetic (mirrored) sampling** — perturbations are drawn in `±ε` pairs, which
//!   halves the variance of the gradient estimate at no extra parameter count.
//! * **Fitness rank normalisation** — the raw returns are replaced by their centred ranks,
//!   making the update invariant to monotone rescaling of `f` and robust to outliers.
//!
//! ## Minimisation convention
//! The rest of `oxicuda-evol` *minimises*. OpenAI-ES natively *maximises* a reward, so this
//! module exposes [`OpenAiEs::minimize`] (internally maximises `-f`) alongside
//! [`OpenAiEs::maximize`]. The per-step API ([`OpenAiEs::ask`] / [`OpenAiEs::tell`]) follows
//! the maximisation convention (higher fitness is better).

use crate::{EvolError, EvolResult, handle::LcgRng};

/// First-order optimiser used to apply the estimated search gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EsOptimizer {
    /// Plain stochastic gradient ascent: `θ ← θ + α·g`.
    Sgd,
    /// SGD with classical (heavy-ball) momentum.
    Momentum,
    /// Adam (Kingma & Ba 2015) — the recommended optimiser in the OpenAI-ES paper.
    Adam,
}

/// Hyper-parameters for an OpenAI-ES run.
#[derive(Debug, Clone)]
pub struct OpenAiEsConfig {
    /// Number of parameters (problem dimension n).
    pub n_dims: usize,
    /// Population size λ. With antithetic sampling this must be **even**.
    pub pop_size: usize,
    /// Perturbation standard deviation σ.
    pub sigma: f64,
    /// Learning rate α of the parameter optimiser.
    pub learning_rate: f64,
    /// Use antithetic (mirrored `±ε`) sampling.
    pub antithetic: bool,
    /// Apply centred-rank fitness shaping before forming the gradient.
    pub rank_normalize: bool,
    /// L2 weight-decay coefficient applied to θ each step (0 disables it).
    pub weight_decay: f64,
    /// First-order optimiser.
    pub optimizer: EsOptimizer,
    /// Momentum coefficient (used by [`EsOptimizer::Momentum`]) / β₁ for Adam.
    pub beta1: f64,
    /// β₂ for Adam.
    pub beta2: f64,
    /// Numerical epsilon for Adam.
    pub epsilon: f64,
    /// Maximum number of generations.
    pub max_iters: usize,
}

impl OpenAiEsConfig {
    /// Build a sensible default configuration for an `n`-dimensional problem.
    ///
    /// Uses Adam with α = 0.01, σ = 0.1, antithetic sampling, rank normalisation, and a
    /// population size of `4 + ⌊3·ln n⌋` rounded up to the next even number.
    pub fn new(n_dims: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        let base = 4 + (3.0 * (n_dims as f64).ln()).floor() as usize;
        let pop_size = if base.is_multiple_of(2) {
            base
        } else {
            base + 1
        };
        Ok(Self {
            n_dims,
            pop_size,
            sigma: 0.1,
            learning_rate: 0.01,
            antithetic: true,
            rank_normalize: true,
            weight_decay: 0.0,
            optimizer: EsOptimizer::Adam,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iters: 1000,
        })
    }

    fn validate(&self) -> EvolResult<()> {
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        if self.pop_size < 2 {
            return Err(EvolError::PopulationTooSmall {
                size: self.pop_size,
                op: "OpenAI-ES",
            });
        }
        if self.antithetic && !self.pop_size.is_multiple_of(2) {
            return Err(EvolError::InvalidParameter(
                "pop_size must be even when antithetic sampling is enabled".to_owned(),
            ));
        }
        if self.sigma <= 0.0 {
            return Err(EvolError::InvalidParameter("sigma must be > 0".to_owned()));
        }
        if self.learning_rate <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "learning_rate must be > 0".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Mutable state for an OpenAI-ES optimiser (online / ask-tell interface).
pub struct OpenAiEs {
    /// Current parameter vector θ.
    pub theta: Vec<f64>,
    /// Configuration.
    cfg: OpenAiEsConfig,
    /// Per-parameter momentum / Adam first-moment estimate m.
    velocity: Vec<f64>,
    /// Per-parameter Adam second-moment estimate v.
    second_moment: Vec<f64>,
    /// Perturbations drawn by the most recent [`ask`](Self::ask) call (n_pairs entries).
    last_epsilons: Vec<Vec<f64>>,
    /// Generation counter (used for Adam bias correction).
    pub generation: usize,
}

impl OpenAiEs {
    /// Create a new optimiser starting at `theta_init`.
    pub fn new(theta_init: Vec<f64>, cfg: OpenAiEsConfig) -> EvolResult<Self> {
        cfg.validate()?;
        if theta_init.len() != cfg.n_dims {
            return Err(EvolError::DimensionMismatch {
                expected: cfg.n_dims,
                got: theta_init.len(),
            });
        }
        let n = cfg.n_dims;
        Ok(Self {
            theta: theta_init,
            cfg,
            velocity: vec![0.0; n],
            second_moment: vec![0.0; n],
            last_epsilons: Vec::new(),
            generation: 0,
        })
    }

    /// Sample the population of candidate parameter vectors `θ + σ·ε`.
    ///
    /// When antithetic sampling is on, `pop_size/2` base perturbations are drawn and each
    /// is returned as a `(+ε, −ε)` pair, yielding `pop_size` candidates in total. The raw
    /// perturbations are cached internally so [`tell`](Self::tell) can form the gradient.
    pub fn ask(&mut self, rng: &mut LcgRng) -> Vec<Vec<f64>> {
        let n = self.cfg.n_dims;
        let sigma = self.cfg.sigma;
        let n_pairs = if self.cfg.antithetic {
            self.cfg.pop_size / 2
        } else {
            self.cfg.pop_size
        };

        self.last_epsilons = (0..n_pairs)
            .map(|_| (0..n).map(|_| rng.next_normal()).collect::<Vec<f64>>())
            .collect();

        let mut candidates = Vec::with_capacity(self.cfg.pop_size);
        for eps in &self.last_epsilons {
            let plus: Vec<f64> = (0..n).map(|i| self.theta[i] + sigma * eps[i]).collect();
            candidates.push(plus);
            if self.cfg.antithetic {
                let minus: Vec<f64> = (0..n).map(|i| self.theta[i] - sigma * eps[i]).collect();
                candidates.push(minus);
            }
        }
        candidates
    }

    /// Feed the fitnesses of the candidates returned by the last [`ask`](Self::ask) and take
    /// one gradient-ascent step.
    ///
    /// `fitnesses` must be ordered exactly as the candidates were returned (so for
    /// antithetic sampling the layout is `[f(+ε₀), f(−ε₀), f(+ε₁), f(−ε₁), …]`). Higher is
    /// better.
    pub fn tell(&mut self, fitnesses: &[f64]) -> EvolResult<()> {
        if fitnesses.len() != self.cfg.pop_size {
            return Err(EvolError::DimensionMismatch {
                expected: self.cfg.pop_size,
                got: fitnesses.len(),
            });
        }
        let n = self.cfg.n_dims;
        let lambda = self.cfg.pop_size as f64;

        // ── Fitness shaping ──────────────────────────────────────────────────
        let shaped = if self.cfg.rank_normalize {
            centered_ranks(fitnesses)
        } else {
            // Standardise to zero-mean / unit-std for a scale-stable gradient.
            standardize(fitnesses)
        };

        // ── Gradient estimate g = (1/(λ·σ)) Σ u_i · ε_i ──────────────────────
        let mut grad = vec![0.0f64; n];
        if self.cfg.antithetic {
            // shaped layout: [u(+ε₀), u(−ε₀), …]; the antithetic gradient uses the
            // difference (u_plus − u_minus) which cancels the even part of f.
            for (pair_idx, eps) in self.last_epsilons.iter().enumerate() {
                let u_plus = shaped[2 * pair_idx];
                let u_minus = shaped[2 * pair_idx + 1];
                let coeff = u_plus - u_minus;
                for i in 0..n {
                    grad[i] += coeff * eps[i];
                }
            }
        } else {
            for (idx, eps) in self.last_epsilons.iter().enumerate() {
                let u = shaped[idx];
                for i in 0..n {
                    grad[i] += u * eps[i];
                }
            }
        }
        let scale = 1.0 / (lambda * self.cfg.sigma);
        for gi in grad.iter_mut() {
            *gi *= scale;
        }

        // ── Optimiser step (ascend J) ─────────────────────────────────────────
        self.generation += 1;
        self.apply_update(&grad);

        // ── Decoupled weight decay ───────────────────────────────────────────
        if self.cfg.weight_decay > 0.0 {
            let wd = self.cfg.weight_decay * self.cfg.learning_rate;
            for i in 0..n {
                self.theta[i] -= wd * self.theta[i];
            }
        }
        Ok(())
    }

    /// Apply the parameter update for the configured optimiser given the (ascent) gradient.
    fn apply_update(&mut self, grad: &[f64]) {
        let n = self.cfg.n_dims;
        let lr = self.cfg.learning_rate;
        match self.cfg.optimizer {
            EsOptimizer::Sgd => {
                for (i, &g) in grad.iter().enumerate().take(n) {
                    self.theta[i] += lr * g;
                }
            }
            EsOptimizer::Momentum => {
                let beta = self.cfg.beta1;
                for (i, &g) in grad.iter().enumerate().take(n) {
                    self.velocity[i] = beta * self.velocity[i] + (1.0 - beta) * g;
                    self.theta[i] += lr * self.velocity[i];
                }
            }
            EsOptimizer::Adam => {
                let b1 = self.cfg.beta1;
                let b2 = self.cfg.beta2;
                let eps = self.cfg.epsilon;
                let t = self.generation as f64;
                let bias1 = 1.0 - b1.powf(t);
                let bias2 = 1.0 - b2.powf(t);
                for (i, &g) in grad.iter().enumerate().take(n) {
                    self.velocity[i] = b1 * self.velocity[i] + (1.0 - b1) * g;
                    self.second_moment[i] = b2 * self.second_moment[i] + (1.0 - b2) * g * g;
                    let m_hat = self.velocity[i] / bias1;
                    let v_hat = self.second_moment[i] / bias2;
                    self.theta[i] += lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
    }

    /// Run the optimiser to **maximise** `reward` for `cfg.max_iters` generations.
    ///
    /// Returns `(best_theta, best_reward)` over all candidates ever evaluated.
    pub fn maximize<F>(&mut self, reward: F, rng: &mut LcgRng) -> EvolResult<(Vec<f64>, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut best_theta = self.theta.clone();
        let mut best_reward = reward(&best_theta);
        for _ in 0..self.cfg.max_iters {
            let candidates = self.ask(rng);
            let fits: Vec<f64> = candidates.iter().map(|c| reward(c)).collect();
            for (c, &f) in candidates.iter().zip(fits.iter()) {
                if f > best_reward {
                    best_reward = f;
                    best_theta = c.clone();
                }
            }
            self.tell(&fits)?;
            // Track the centre too: ES converges the *mean* toward the optimum.
            let centre_reward = reward(&self.theta);
            if centre_reward > best_reward {
                best_reward = centre_reward;
                best_theta = self.theta.clone();
            }
        }
        Ok((best_theta, best_reward))
    }

    /// Run the optimiser to **minimise** `objective` (lower is better).
    ///
    /// Internally maximises `-objective`. Returns `(best_theta, best_objective)`.
    pub fn minimize<F>(&mut self, objective: F, rng: &mut LcgRng) -> EvolResult<(Vec<f64>, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        let (theta, neg) = self.maximize(|x| -objective(x), rng)?;
        Ok((theta, -neg))
    }
}

/// Compute centred ranks of `xs` mapped to `[-0.5, 0.5]` (higher input → higher rank).
///
/// Ties are broken by index (stable), and the average of the produced ranks is ~0 so the
/// gradient estimate is unbiased w.r.t. the location of the fitnesses.
fn centered_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // ranks[i] = position of element i in ascending order, in [0, n-1].
    let mut ranks = vec![0.0f64; n];
    for (pos, &idx) in order.iter().enumerate() {
        ranks[idx] = pos as f64;
    }
    let denom = (n - 1).max(1) as f64;
    ranks.iter().map(|&r| r / denom - 0.5).collect()
}

/// Standardise `xs` to zero mean and unit standard deviation.
fn standardize(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-12);
    xs.iter().map(|&x| (x - mean) / std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    #[test]
    fn config_default_pop_even_with_antithetic() {
        let cfg = OpenAiEsConfig::new(10).expect("ok");
        assert!(cfg.antithetic);
        assert_eq!(cfg.pop_size % 2, 0, "pop must be even for antithetic");
    }

    #[test]
    fn config_rejects_odd_pop_with_antithetic() {
        let mut cfg = OpenAiEsConfig::new(3).expect("ok");
        cfg.pop_size = 7;
        cfg.antithetic = true;
        let es = OpenAiEs::new(vec![0.0; 3], cfg);
        assert!(es.is_err());
    }

    #[test]
    fn config_rejects_zero_dim() {
        assert!(OpenAiEsConfig::new(0).is_err());
    }

    #[test]
    fn centered_ranks_are_symmetric() {
        let r = centered_ranks(&[3.0, 1.0, 2.0, 0.0]);
        // Sum of [-0.5, 0.5] symmetric ranks for 4 evenly-spaced positions = 0.
        let s: f64 = r.iter().sum();
        assert!(s.abs() < 1e-12, "centred ranks must sum to ~0, got {s}");
        // Largest input (3.0 at idx 0) → +0.5.
        assert!((r[0] - 0.5).abs() < 1e-12);
        // Smallest input (0.0 at idx 3) → -0.5.
        assert!((r[3] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn standardize_zero_mean_unit_std() {
        let s = standardize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean: f64 = s.iter().sum::<f64>() / s.len() as f64;
        assert!(mean.abs() < 1e-12);
        let var = s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64;
        assert!((var - 1.0).abs() < 1e-9);
    }

    #[test]
    fn minimizes_sphere_5d_adam() {
        let mut cfg = OpenAiEsConfig::new(5).expect("ok");
        cfg.pop_size = 40;
        cfg.sigma = 0.2;
        cfg.learning_rate = 0.05;
        cfg.max_iters = 600;
        let mut rng = LcgRng::new(7);
        let theta0 = vec![3.0, -2.5, 2.0, -3.0, 1.5];
        let mut es = OpenAiEs::new(theta0, cfg).expect("ok");
        let (best_x, best_f) = es.minimize(sphere, &mut rng).expect("ok");
        assert!(
            best_f < 1e-2,
            "OpenAI-ES (Adam) should minimise 5-D sphere below 1e-2, got {best_f} at {best_x:?}"
        );
    }

    #[test]
    fn minimizes_sphere_non_antithetic_sgd() {
        let mut cfg = OpenAiEsConfig::new(3).expect("ok");
        cfg.pop_size = 50;
        cfg.antithetic = false;
        cfg.optimizer = EsOptimizer::Sgd;
        cfg.sigma = 0.15;
        cfg.learning_rate = 0.2;
        cfg.max_iters = 800;
        let mut rng = LcgRng::new(123);
        let mut es = OpenAiEs::new(vec![2.0, -2.0, 1.0], cfg).expect("ok");
        let (_x, best_f) = es.minimize(sphere, &mut rng).expect("ok");
        assert!(
            best_f < 5e-2,
            "OpenAI-ES (SGD, no antithetic) should minimise sphere, got {best_f}"
        );
    }

    #[test]
    fn maximize_negative_sphere_momentum() {
        let mut cfg = OpenAiEsConfig::new(4).expect("ok");
        cfg.pop_size = 40;
        cfg.optimizer = EsOptimizer::Momentum;
        cfg.sigma = 0.2;
        cfg.learning_rate = 0.1;
        cfg.max_iters = 500;
        let mut rng = LcgRng::new(99);
        let mut es = OpenAiEs::new(vec![2.0, 2.0, -2.0, -2.0], cfg).expect("ok");
        let (_x, best_r) = es.maximize(|x| -sphere(x), &mut rng).expect("ok");
        assert!(
            best_r > -1e-1,
            "maximising -sphere should approach 0, got {best_r}"
        );
    }

    #[test]
    fn ask_returns_pop_size_candidates() {
        let cfg = OpenAiEsConfig::new(4).expect("ok");
        let pop = cfg.pop_size;
        let mut rng = LcgRng::new(1);
        let mut es = OpenAiEs::new(vec![0.0; 4], cfg).expect("ok");
        let cands = es.ask(&mut rng);
        assert_eq!(cands.len(), pop);
        // Antithetic: candidate 1 is the mirror of candidate 0 about theta (=0).
        for (a, b) in cands[0].iter().zip(&cands[1]) {
            assert!((a + b).abs() < 1e-12);
        }
    }

    #[test]
    fn tell_rejects_wrong_length() {
        let cfg = OpenAiEsConfig::new(3).expect("ok");
        let mut rng = LcgRng::new(2);
        let mut es = OpenAiEs::new(vec![0.0; 3], cfg).expect("ok");
        let _ = es.ask(&mut rng);
        assert!(es.tell(&[0.0, 1.0]).is_err());
    }
}
