//! Natural Evolution Strategies (NES) — gradient ascent on expected fitness using the
//! *natural* gradient of a parametric search distribution.
//!
//! Reference: D. Wierstra, T. Schaul, T. Glasmachers, Y. Sun, J. Peters, J. Schmidhuber,
//! "Natural Evolution Strategies", Journal of Machine Learning Research 15:949-980, 2014.
//! <https://jmlr.org/papers/v15/wierstra14a.html>
//!
//! ## Overview
//! NES maintains an isotropic-or-diagonal Gaussian search distribution
//! `N(μ, diag(σ²))` and maximises the expected fitness
//!
//! ```text
//! J(θ) = E_{x ~ N(μ, diag(σ²))} [ f(x) ]
//! ```
//!
//! by following the **natural gradient** `F⁻¹ ∇_θ J`, where `F` is the Fisher information
//! matrix of the search distribution. For the Gaussian search distribution parameterised by
//! `(μ, log σ)` the Fisher matrix is *diagonal and constant*, and the natural-gradient
//! update collapses to the closed form used here (the "Separable NES" / SNES instance,
//! Schaul et al. 2011):
//!
//! ```text
//! z_i  = (x_i − μ) / σ                         (standardised samples)
//! μ   ← μ + η_μ · σ ⊙ Σ_i u_i · z_i
//! σ   ← σ · exp( (η_σ / 2) · Σ_i u_i · (z_i² − 1) )
//! ```
//!
//! where `u_i` are **utility weights** computed from the *ranks* of the fitnesses
//! (Wierstra eq. 11). This rank-based fitness shaping makes NES invariant to monotone
//! transformations of `f` and is the key to its robustness on rugged landscapes.
//!
//! ## Minimisation convention
//! The rest of `oxicuda-evol` *minimises*. NES natively maximises, so this module exposes
//! [`NaturalEvolutionStrategies::minimize`] (internally maximises `-f`) alongside
//! [`NaturalEvolutionStrategies::maximize`].

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for a Separable NES run.
#[derive(Debug, Clone)]
pub struct NesConfig {
    /// Number of parameters (problem dimension n).
    pub n_dims: usize,
    /// Population size λ.
    pub pop_size: usize,
    /// Initial per-coordinate standard deviation σ₀.
    pub sigma_init: f64,
    /// Learning rate for the mean μ.
    pub lr_mean: f64,
    /// Learning rate for the (log) standard deviation σ.
    pub lr_sigma: f64,
    /// Maximum number of generations.
    pub max_iters: usize,
}

impl NesConfig {
    /// Build a default configuration for an `n`-dimensional problem using the SNES default
    /// schedule (Schaul et al. 2011):
    ///
    /// ```text
    /// λ       = 4 + ⌊3·ln n⌋
    /// η_μ     = 1
    /// η_σ     = (3 + ln n) / (5·√n)
    /// ```
    pub fn new(n_dims: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        let nf = n_dims as f64;
        let pop_size = 4 + (3.0 * nf.ln()).floor() as usize;
        let lr_sigma = (3.0 + nf.ln()) / (5.0 * nf.sqrt());
        Ok(Self {
            n_dims,
            pop_size,
            sigma_init: 1.0,
            lr_mean: 1.0,
            lr_sigma,
            max_iters: 2000,
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
                op: "NES",
            });
        }
        if self.sigma_init <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "sigma_init must be > 0".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Mutable state for a Separable Natural Evolution Strategy (ask-tell interface).
pub struct NaturalEvolutionStrategies {
    /// Current distribution mean μ.
    pub mean: Vec<f64>,
    /// Current per-coordinate standard deviation σ.
    pub sigma: Vec<f64>,
    /// Configuration.
    cfg: NesConfig,
    /// Precomputed utility weights u_i (length = pop_size, descending).
    utilities: Vec<f64>,
    /// Standardised samples `z_i` from the most recent [`ask`](Self::ask).
    last_z: Vec<Vec<f64>>,
    /// Generation counter.
    pub generation: usize,
}

impl NaturalEvolutionStrategies {
    /// Create a new NES optimiser centred at `mean_init` with isotropic σ₀.
    pub fn new(mean_init: Vec<f64>, cfg: NesConfig) -> EvolResult<Self> {
        cfg.validate()?;
        if mean_init.len() != cfg.n_dims {
            return Err(EvolError::DimensionMismatch {
                expected: cfg.n_dims,
                got: mean_init.len(),
            });
        }
        let sigma = vec![cfg.sigma_init; cfg.n_dims];
        let utilities = utility_weights(cfg.pop_size);
        Ok(Self {
            mean: mean_init,
            sigma,
            cfg,
            utilities,
            last_z: Vec::new(),
            generation: 0,
        })
    }

    /// Sample λ candidate solutions `x_i = μ + σ ⊙ z_i`, `z_i ~ N(0, I)`.
    ///
    /// The standardised draws `z_i` are cached for [`tell`](Self::tell).
    pub fn ask(&mut self, rng: &mut LcgRng) -> Vec<Vec<f64>> {
        let n = self.cfg.n_dims;
        self.last_z = (0..self.cfg.pop_size)
            .map(|_| (0..n).map(|_| rng.next_normal()).collect::<Vec<f64>>())
            .collect();
        self.last_z
            .iter()
            .map(|z| {
                (0..n)
                    .map(|i| self.mean[i] + self.sigma[i] * z[i])
                    .collect()
            })
            .collect()
    }

    /// Feed the fitnesses of the candidates from the last [`ask`](Self::ask) and take one
    /// natural-gradient ascent step. Higher fitness is better.
    pub fn tell(&mut self, fitnesses: &[f64]) -> EvolResult<()> {
        if fitnesses.len() != self.cfg.pop_size {
            return Err(EvolError::DimensionMismatch {
                expected: self.cfg.pop_size,
                got: fitnesses.len(),
            });
        }
        let n = self.cfg.n_dims;
        let lambda = self.cfg.pop_size;

        // ── Order samples best→worst and assign utility weights by rank ───────
        let mut order: Vec<usize> = (0..lambda).collect();
        // Descending: best (highest fitness) first.
        order.sort_by(|&a, &b| {
            fitnesses[b]
                .partial_cmp(&fitnesses[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Natural-gradient estimates for μ and (log) σ.
        // grad_mu[i]    = Σ_k u_k · z_{(k)}[i]
        // grad_logsig[i]= Σ_k u_k · (z_{(k)}[i]² − 1)
        let mut grad_mu = vec![0.0f64; n];
        let mut grad_logsig = vec![0.0f64; n];
        for (rank, &sample_idx) in order.iter().enumerate() {
            let u = self.utilities[rank];
            let z = &self.last_z[sample_idx];
            for i in 0..n {
                grad_mu[i] += u * z[i];
                grad_logsig[i] += u * (z[i] * z[i] - 1.0);
            }
        }

        // ── Natural-gradient updates ──────────────────────────────────────────
        let eta_mu = self.cfg.lr_mean;
        let eta_sigma = self.cfg.lr_sigma;
        for i in 0..n {
            self.mean[i] += eta_mu * self.sigma[i] * grad_mu[i];
            self.sigma[i] *= (0.5 * eta_sigma * grad_logsig[i]).exp();
            // Guard against numerical collapse / blow-up.
            self.sigma[i] = self.sigma[i].clamp(1e-20, 1e10);
        }

        self.generation += 1;
        Ok(())
    }

    /// Run NES to **maximise** `reward` for `cfg.max_iters` generations.
    ///
    /// Returns `(best_x, best_reward)` across every candidate evaluated (and the final mean).
    pub fn maximize<F>(&mut self, reward: F, rng: &mut LcgRng) -> EvolResult<(Vec<f64>, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut best_x = self.mean.clone();
        let mut best_r = reward(&best_x);
        for _ in 0..self.cfg.max_iters {
            let candidates = self.ask(rng);
            let fits: Vec<f64> = candidates.iter().map(|c| reward(c)).collect();
            for (c, &f) in candidates.iter().zip(fits.iter()) {
                if f > best_r {
                    best_r = f;
                    best_x = c.clone();
                }
            }
            self.tell(&fits)?;
            let centre = reward(&self.mean);
            if centre > best_r {
                best_r = centre;
                best_x = self.mean.clone();
            }
            // Convergence: σ has collapsed → no further progress possible.
            if self.sigma.iter().all(|&s| s < 1e-14) {
                break;
            }
        }
        Ok((best_x, best_r))
    }

    /// Run NES to **minimise** `objective` (lower is better).
    ///
    /// Internally maximises `-objective`. Returns `(best_x, best_objective)`.
    pub fn minimize<F>(&mut self, objective: F, rng: &mut LcgRng) -> EvolResult<(Vec<f64>, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        let (x, neg) = self.maximize(|p| -objective(p), rng)?;
        Ok((x, -neg))
    }
}

/// Compute the NES utility weights (Wierstra 2014, eq. 11), ordered best→worst.
///
/// ```text
/// u_k = max(0, ln(λ/2 + 1) − ln k) / Σ_j max(0, ln(λ/2 + 1) − ln j)  −  1/λ
/// ```
///
/// for `k = 1 … λ`. The trailing `−1/λ` makes the weights sum to **zero**, which is what
/// turns the gradient estimate into a *baseline-subtracted* (hence lower-variance and
/// translation-invariant) natural gradient. Roughly the top half of the population gets
/// positive weight and the bottom half negative.
fn utility_weights(lambda: usize) -> Vec<f64> {
    let lam = lambda as f64;
    let log_term = (lam / 2.0 + 1.0).ln();
    let raw: Vec<f64> = (1..=lambda)
        .map(|k| (log_term - (k as f64).ln()).max(0.0))
        .collect();
    let sum: f64 = raw.iter().sum();
    if sum <= 0.0 {
        // Degenerate (λ very small): fall back to a uniform zero-sum baseline.
        return vec![0.0; lambda];
    }
    raw.iter().map(|&w| w / sum - 1.0 / lam).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (w[0] - 1.0).powi(2))
            .sum()
    }

    #[test]
    fn config_default_schedule() {
        let cfg = NesConfig::new(10).expect("ok");
        assert_eq!(cfg.pop_size, 4 + (3.0 * 10f64.ln()).floor() as usize);
        assert!(cfg.lr_sigma > 0.0 && cfg.lr_sigma < 1.0);
    }

    #[test]
    fn config_rejects_zero_dim() {
        assert!(NesConfig::new(0).is_err());
    }

    #[test]
    fn utility_weights_sum_to_zero() {
        for &lam in &[4usize, 8, 16, 32, 50] {
            let u = utility_weights(lam);
            let s: f64 = u.iter().sum();
            assert!(
                s.abs() < 1e-12,
                "utilities must sum to ~0 for λ={lam}, got {s}"
            );
            // The best individual gets the largest weight, the worst the smallest.
            assert!(u[0] > u[lam - 1], "weights must be descending by rank");
            assert!(
                u[lam - 1] <= 0.0,
                "worst individuals get non-positive weight"
            );
        }
    }

    #[test]
    fn minimizes_sphere_5d() {
        let mut cfg = NesConfig::new(5).expect("ok");
        cfg.sigma_init = 1.0;
        cfg.max_iters = 2000;
        let mut rng = LcgRng::new(42);
        let mut nes =
            NaturalEvolutionStrategies::new(vec![3.0, -2.0, 2.5, -1.5, 1.0], cfg).expect("ok");
        let (best_x, best_f) = nes.minimize(sphere, &mut rng).expect("ok");
        assert!(
            best_f < 1e-8,
            "NES should minimise 5-D sphere below 1e-8, got {best_f} at {best_x:?}"
        );
    }

    #[test]
    fn minimizes_rosenbrock_2d() {
        let mut cfg = NesConfig::new(2).expect("ok");
        cfg.pop_size = 20;
        cfg.sigma_init = 0.5;
        cfg.max_iters = 6000;
        let mut rng = LcgRng::new(11);
        let mut nes = NaturalEvolutionStrategies::new(vec![-1.0, 1.0], cfg).expect("ok");
        let (_x, best_f) = nes.minimize(rosenbrock, &mut rng).expect("ok");
        assert!(
            best_f < 1e-2,
            "NES should reach near optimum on Rosenbrock 2D, got {best_f}"
        );
    }

    #[test]
    fn sigma_shrinks_as_it_converges() {
        let mut cfg = NesConfig::new(3).expect("ok");
        cfg.sigma_init = 1.0;
        cfg.max_iters = 1500;
        let mut rng = LcgRng::new(5);
        let mut nes = NaturalEvolutionStrategies::new(vec![2.0, 2.0, 2.0], cfg).expect("ok");
        let _ = nes.minimize(sphere, &mut rng).expect("ok");
        // After converging on the sphere the step-size must have contracted well below σ₀.
        assert!(
            nes.sigma.iter().all(|&s| s < 0.1),
            "sigma should contract on the sphere, got {:?}",
            nes.sigma
        );
    }

    #[test]
    fn ask_caches_z_and_returns_pop_size() {
        let cfg = NesConfig::new(4).expect("ok");
        let pop = cfg.pop_size;
        let mut rng = LcgRng::new(1);
        let mut nes = NaturalEvolutionStrategies::new(vec![0.0; 4], cfg).expect("ok");
        let cands = nes.ask(&mut rng);
        assert_eq!(cands.len(), pop);
        assert_eq!(nes.last_z.len(), pop);
    }

    #[test]
    fn tell_rejects_wrong_length() {
        let cfg = NesConfig::new(3).expect("ok");
        let mut rng = LcgRng::new(2);
        let mut nes = NaturalEvolutionStrategies::new(vec![0.0; 3], cfg).expect("ok");
        let _ = nes.ask(&mut rng);
        assert!(nes.tell(&[1.0]).is_err());
    }
}
