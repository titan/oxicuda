//! Particle Swarm Optimization (PSO) with linearly decaying inertia weight.
//!
//! Reference: J. Kennedy & R. Eberhart, "Particle Swarm Optimization",
//! Proceedings of ICNN'95, 1995.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// PSO hyper-parameters.
#[derive(Debug, Clone)]
pub struct PsoConfig {
    /// Problem dimension.
    pub n_dims: usize,
    /// Number of particles.
    pub pop_size: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Inertia weight at iteration 0.
    pub w_max: f64,
    /// Inertia weight at final iteration.
    pub w_min: f64,
    /// Cognitive acceleration coefficient.
    pub c1: f64,
    /// Social acceleration coefficient.
    pub c2: f64,
    /// Maximum absolute velocity (clamping bound).
    pub v_max: f64,
    /// Shared position bounds (lb, ub).
    pub bounds: (f64, f64),
}

impl PsoConfig {
    /// Build a default PSO config for `n_dims`-dimensional problems.
    pub fn new(n_dims: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_dims,
            pop_size: 30,
            max_iter: 500,
            w_max: 0.9,
            w_min: 0.4,
            c1: 2.0,
            c2: 2.0,
            v_max: 0.1 * 2.0, // will be set relative to range at construction
            bounds: (-5.0, 5.0),
        })
    }
}

/// A single PSO particle.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current position.
    pub pos: Vec<f64>,
    /// Current velocity.
    pub vel: Vec<f64>,
    /// Personal best position.
    pub pbest_pos: Vec<f64>,
    /// Personal best fitness.
    pub pbest_fit: f64,
}

/// Mutable PSO state.
pub struct PsoState {
    /// All particles.
    pub particles: Vec<Particle>,
    /// Global best position.
    pub gbest_pos: Vec<f64>,
    /// Global best fitness.
    pub gbest_fit: f64,
    /// Current iteration.
    pub iter: usize,
}

impl PsoState {
    /// Initialise particles randomly within bounds and evaluate initial fitness.
    pub fn new(cfg: &PsoConfig, rng: &mut LcgRng) -> EvolResult<Self> {
        if cfg.pop_size == 0 {
            return Err(EvolError::SwarmEmpty);
        }
        let (lb, ub) = cfg.bounds;
        let range = ub - lb;
        let v_max = cfg.v_max.max(0.01 * range);

        let particles: Vec<Particle> = (0..cfg.pop_size)
            .map(|_| {
                let pos: Vec<f64> = (0..cfg.n_dims)
                    .map(|_| lb + rng.next_f64() * range)
                    .collect();
                let vel: Vec<f64> = (0..cfg.n_dims)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * v_max)
                    .collect();
                Particle {
                    pbest_pos: pos.clone(),
                    pbest_fit: f64::INFINITY,
                    pos,
                    vel,
                }
            })
            .collect();

        Ok(Self {
            gbest_pos: vec![lb; cfg.n_dims],
            gbest_fit: f64::INFINITY,
            particles,
            iter: 0,
        })
    }

    /// Execute one PSO iteration: update velocities and positions, evaluate, update bests.
    pub fn step<F: Fn(&[f64]) -> f64>(&mut self, f: &F, cfg: &PsoConfig, rng: &mut LcgRng) {
        let (lb, ub) = cfg.bounds;
        let range = ub - lb;
        let v_max = cfg.v_max.max(0.01 * range);

        // Linearly decaying inertia weight
        let w = cfg.w_max - (cfg.w_max - cfg.w_min) * self.iter as f64 / cfg.max_iter.max(1) as f64;

        for particle in &mut self.particles {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            for d in 0..cfg.n_dims {
                // Velocity update
                let v_new = w * particle.vel[d]
                    + cfg.c1 * r1 * (particle.pbest_pos[d] - particle.pos[d])
                    + cfg.c2 * r2 * (self.gbest_pos[d] - particle.pos[d]);
                particle.vel[d] = v_new.max(-v_max).min(v_max);
                // Position update
                particle.pos[d] = (particle.pos[d] + particle.vel[d]).max(lb).min(ub);
            }
            // Evaluate
            let fit = f(&particle.pos);
            if fit < particle.pbest_fit {
                particle.pbest_fit = fit;
                particle.pbest_pos = particle.pos.clone();
            }
            if fit < self.gbest_fit {
                self.gbest_fit = fit;
                self.gbest_pos = particle.pos.clone();
            }
        }
        self.iter += 1;
    }

    /// Run PSO to convergence.
    pub fn run<F: Fn(&[f64]) -> f64>(
        &mut self,
        f: F,
        cfg: &PsoConfig,
        rng: &mut LcgRng,
    ) -> EvolResult<(Vec<f64>, f64)> {
        if self.particles.is_empty() {
            return Err(EvolError::SwarmEmpty);
        }
        // Evaluate initial positions
        for particle in &mut self.particles {
            let fit = f(&particle.pos);
            particle.pbest_fit = fit;
            if fit < self.gbest_fit {
                self.gbest_fit = fit;
                self.gbest_pos = particle.pos.clone();
            }
        }
        for _ in 0..cfg.max_iter {
            self.step(&f, cfg, rng);
        }
        Ok((self.gbest_pos.clone(), self.gbest_fit))
    }
}
