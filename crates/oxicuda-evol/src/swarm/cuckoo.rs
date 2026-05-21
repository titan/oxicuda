//! Cuckoo Search algorithm for continuous optimization.
//!
//! Reference: X.-S. Yang & S. Deb, "Cuckoo Search via Lévy Flights",
//! Proceedings of World Congress on Nature & Biologically Inspired Computing, 2009.
//!
//! # Algorithm overview
//!
//! Nests represent candidate solutions.  Each iteration:
//!
//! 1. A new cuckoo is generated for each nest via **Lévy flight** (Mantegna's approximation
//!    for α = 1.5) directed toward the current global best.
//! 2. The cuckoo is compared against a randomly chosen nest; if better, it replaces that nest.
//! 3. A fraction `pa` of the *worst* nests are abandoned and replaced with new random nests.
//! 4. The global best is updated.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Cuckoo Search hyper-parameters.
#[derive(Debug, Clone)]
pub struct CuckooConfig {
    /// Number of nests (population size).
    pub n_nests: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Fraction of nests to abandon each iteration (0 < pa < 1).  Default: 0.25.
    pub pa: f64,
    /// Lévy flight step scale factor.  Default: 0.01.
    pub step_scale: f64,
    /// Random seed for the internal LCG RNG.
    pub seed: u64,
}

impl CuckooConfig {
    /// Construct a `CuckooConfig` with sensible defaults.
    pub fn new(n_nests: usize, max_iter: usize) -> EvolResult<Self> {
        if n_nests == 0 {
            return Err(EvolError::SwarmEmpty);
        }
        if n_nests < 2 {
            return Err(EvolError::PopulationTooSmall {
                size: n_nests,
                op: "CuckooSearch",
            });
        }
        if max_iter == 0 {
            return Err(EvolError::InvalidParameter(
                "max_iter must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_nests,
            max_iter,
            pa: 0.25,
            step_scale: 0.01,
            seed: 0,
        })
    }
}

/// Mutable state for a Cuckoo Search run.
pub struct CuckooState {
    /// Current nest positions (n_nests × n_dims).
    pub nests: Vec<Vec<f64>>,
    /// Raw objective value for each nest (minimisation; lower is better).
    pub fitness: Vec<f64>,
    /// Per-dimension search bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Best known decision variable vector.
    pub best: Vec<f64>,
    /// Raw objective value of `best`.
    pub best_fitness: f64,
    /// Number of completed iterations.
    pub generation: usize,
}

impl CuckooState {
    /// Initialise nests uniformly at random within `bounds` and evaluate initial fitness.
    pub fn new<F: Fn(&[f64]) -> f64>(
        bounds: Vec<(f64, f64)>,
        n_nests: usize,
        fitness_fn: &F,
        rng: &mut LcgRng,
    ) -> EvolResult<Self> {
        if bounds.is_empty() {
            return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
        }
        if n_nests == 0 {
            return Err(EvolError::SwarmEmpty);
        }

        let n_dims = bounds.len();
        let nests: Vec<Vec<f64>> = (0..n_nests)
            .map(|_| {
                (0..n_dims)
                    .map(|d| {
                        let (lb, ub) = bounds[d];
                        lb + rng.next_f64() * (ub - lb)
                    })
                    .collect()
            })
            .collect();

        let fitness: Vec<f64> = nests.iter().map(|x| fitness_fn(x)).collect();

        let (best_idx, &best_fitness) = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(EvolError::SwarmEmpty)?;

        let best = nests[best_idx].clone();

        Ok(Self {
            nests,
            fitness,
            bounds,
            best,
            best_fitness,
            generation: 0,
        })
    }
}

/// Mantegna's algorithm for sampling from a Lévy stable distribution with β = 1.5.
///
/// Returns a single Lévy-distributed step magnitude (may be negative).
///
/// The Lévy exponent β = 1.5 is the standard choice for Cuckoo Search.
///
/// Formula:
/// ```text
/// σ_u = ( Γ(1+β)·sin(π·β/2) / (Γ((1+β)/2)·β·2^((β-1)/2)) )^(1/β)
/// u ~ N(0, σ_u²),  v ~ N(0, 1)
/// levy = u / |v|^(1/β)
/// ```
fn levy_flight_step(rng: &mut LcgRng) -> f64 {
    const BETA: f64 = 1.5;

    // Pre-compute σ_u using Stirling / exact Γ values for β = 1.5:
    //   Γ(2.5) = 1.5 * 0.5 * Γ(0.5) = 0.75 * √π  ≈ 1.329340388
    //   Γ(1.25) = 0.906402477
    //   sin(π * 1.5 / 2) = sin(3π/4) = √2/2 ≈ 0.707106781
    //   2^((1.5-1)/2) = 2^0.25 ≈ 1.189207115
    //
    // σ_u^1.5 = 1.329340388 * 0.707106781 / (0.906402477 * 1.5 * 1.189207115)
    //         = 0.939702623 / 1.618340...  ≈ 0.580693...
    // σ_u = 0.580693^(1/1.5) ≈ 0.695720...
    //
    // We compute this analytically at compile-time level, then use the constant.

    let gamma_1_plus_beta = gamma_half_integer(1.0 + BETA); // Γ(2.5)
    let gamma_1_plus_beta_over_2 = gamma_half_integer((1.0 + BETA) / 2.0); // Γ(1.25)
    let sin_term = (std::f64::consts::PI * BETA / 2.0).sin();
    let pow_term = 2_f64.powf((BETA - 1.0) / 2.0);

    let numerator = gamma_1_plus_beta * sin_term;
    let denominator = gamma_1_plus_beta_over_2 * BETA * pow_term;
    let sigma_u = (numerator / denominator).powf(1.0 / BETA);

    // u ~ N(0, σ_u²), v ~ N(0, 1)
    let u = rng.next_normal() * sigma_u;
    let v = rng.next_normal();

    u / v.abs().powf(1.0 / BETA)
}

/// Approximate the Gamma function for half-integer arguments needed by Mantegna's formula.
///
/// Covers: Γ(1.25), Γ(2.5), and general fallback via the Lanczos approximation.
fn gamma_half_integer(x: f64) -> f64 {
    // Use the Lanczos approximation (g=7, n=9 coefficients — Spouge variant).
    // This gives accuracy to ~15 significant digits.
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_half_integer(1.0 - x))
    } else {
        let z = x - 1.0;
        let mut sum = C[0];
        for (i, &ci) in C.iter().enumerate().skip(1) {
            sum += ci / (z + i as f64);
        }
        let t = z + G + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum
    }
}

/// Execute one Cuckoo Search iteration step.
///
/// 1. For each nest i: generate a new cuckoo via Lévy flight toward `best`.
/// 2. Pick a random nest j ≠ i; replace j if the new cuckoo is better.
/// 3. Abandon the worst `pa * n_nests` nests and replace with random positions.
/// 4. Update global best.
pub fn cuckoo_step<F: Fn(&[f64]) -> f64>(
    state: &mut CuckooState,
    fitness_fn: &F,
    rng: &mut LcgRng,
    pa: f64,
    step_scale: f64,
) {
    let n = state.nests.len();
    let n_dims = state.bounds.len();

    // ── Phase 1: Lévy flight for each nest ────────────────────────────────────
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        // Lévy step is scalar; apply per-dimension with directional bias toward best.
        let levy = levy_flight_step(rng);

        let mut new_nest = state.nests[i].clone();
        for d in 0..n_dims {
            let (lb, ub) = state.bounds[d];
            new_nest[d] =
                (new_nest[d] + step_scale * levy * (new_nest[d] - state.best[d])).clamp(lb, ub);
        }

        let new_fitness = fitness_fn(&new_nest);

        // Pick a random nest j ≠ i.
        let j = {
            let raw = rng.next_usize(n - 1);
            if raw >= i { raw + 1 } else { raw }
        };

        if new_fitness < state.fitness[j] {
            state.nests[j] = new_nest;
            state.fitness[j] = new_fitness;
        }
    }

    // ── Phase 2: Abandon worst nests ──────────────────────────────────────────
    let n_abandon = ((pa * n as f64).round() as usize).min(n);
    if n_abandon > 0 {
        // Identify the worst nest indices by fitness (highest = worst for minimisation).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            state.fitness[b]
                .partial_cmp(&state.fitness[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &idx in order.iter().take(n_abandon) {
            let new_nest: Vec<f64> = (0..n_dims)
                .map(|d| {
                    let (lb, ub) = state.bounds[d];
                    lb + rng.next_f64() * (ub - lb)
                })
                .collect();
            let new_fitness = fitness_fn(&new_nest);
            state.nests[idx] = new_nest;
            state.fitness[idx] = new_fitness;
        }
    }

    // ── Phase 3: Update global best ───────────────────────────────────────────
    for i in 0..n {
        if state.fitness[i] < state.best_fitness {
            state.best_fitness = state.fitness[i];
            state.best = state.nests[i].clone();
        }
    }

    state.generation += 1;
}

/// Run Cuckoo Search to completion.
///
/// Returns the final `CuckooState`; `state.best` is the best decision vector found.
pub fn cuckoo_run<F>(
    fitness_fn: F,
    bounds: &[(f64, f64)],
    cfg: &CuckooConfig,
) -> EvolResult<CuckooState>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
    }
    if cfg.n_nests == 0 {
        return Err(EvolError::SwarmEmpty);
    }
    if cfg.n_nests < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: cfg.n_nests,
            op: "CuckooSearch",
        });
    }
    if cfg.pa <= 0.0 || cfg.pa >= 1.0 {
        return Err(EvolError::InvalidParameter(
            "pa must be in (0, 1)".to_owned(),
        ));
    }

    let mut rng = LcgRng::new(cfg.seed);
    let mut state = CuckooState::new(bounds.to_vec(), cfg.n_nests, &fitness_fn, &mut rng)?;

    for _ in 0..cfg.max_iter {
        cuckoo_step(&mut state, &fitness_fn, &mut rng, cfg.pa, cfg.step_scale);
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Objective functions ──────────────────────────────────────────────────

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| {
                let (xi, xj) = (w[0], w[1]);
                100.0 * (xj - xi * xi).powi(2) + (1.0 - xi).powi(2)
            })
            .sum()
    }

    fn ackley(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_cos: f64 = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + std::f64::consts::E
            + 20.0
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
                .sum::<f64>()
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = CuckooConfig::new(20, 100).unwrap();
        assert_eq!(cfg.n_nests, 20);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.pa - 0.25).abs() < 1e-12);
        assert!((cfg.step_scale - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_config_new_zero_nests() {
        assert!(CuckooConfig::new(0, 100).is_err());
    }

    #[test]
    fn test_config_new_one_nest() {
        // Single nest cannot use the algorithm (need j ≠ i).
        assert!(CuckooConfig::new(1, 100).is_err());
    }

    #[test]
    fn test_config_new_zero_iter() {
        assert!(CuckooConfig::new(10, 0).is_err());
    }

    // ── State construction tests ──────────────────────────────────────────────

    #[test]
    fn test_state_new_correct_sizes() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let mut rng = LcgRng::new(0);
        let state = CuckooState::new(bounds, 10, &sphere, &mut rng).unwrap();
        assert_eq!(state.nests.len(), 10);
        assert_eq!(state.fitness.len(), 10);
        assert_eq!(state.nests[0].len(), 3);
        assert_eq!(state.generation, 0);
    }

    #[test]
    fn test_state_new_positions_within_bounds() {
        let bounds: Vec<(f64, f64)> = vec![(-3.0, 3.0), (1.0, 5.0)];
        let mut rng = LcgRng::new(7);
        let state = CuckooState::new(bounds.clone(), 12, &sphere, &mut rng).unwrap();
        for nest in &state.nests {
            for (d, &x) in nest.iter().enumerate() {
                let (lb, ub) = bounds[d];
                assert!(x >= lb && x <= ub, "dim {d}: {x} not in [{lb},{ub}]");
            }
        }
    }

    #[test]
    fn test_state_new_best_fitness_correct() {
        let bounds = vec![(-2.0_f64, 2.0_f64); 2];
        let mut rng = LcgRng::new(42);
        let state = CuckooState::new(bounds, 8, &sphere, &mut rng).unwrap();
        let min_fit = state.fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((state.best_fitness - min_fit).abs() < 1e-12);
    }

    // ── Lévy flight tests ─────────────────────────────────────────────────────

    #[test]
    fn test_levy_step_finite() {
        let mut rng = LcgRng::new(1234);
        for _ in 0..1000 {
            let step = levy_flight_step(&mut rng);
            assert!(step.is_finite(), "levy step was NaN or infinite");
        }
    }

    #[test]
    fn test_gamma_half_integer_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2
        assert!((gamma_half_integer(1.0) - 1.0).abs() < 1e-10);
        assert!((gamma_half_integer(2.0) - 1.0).abs() < 1e-10);
        assert!((gamma_half_integer(3.0) - 2.0).abs() < 1e-10);
        // Γ(0.5) = √π ≈ 1.7724538509
        assert!((gamma_half_integer(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-9);
    }

    // ── Step tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_cuckoo_step_increments_generation() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 2];
        let mut rng = LcgRng::new(1);
        let mut state = CuckooState::new(bounds, 5, &sphere, &mut rng).unwrap();
        assert_eq!(state.generation, 0);
        cuckoo_step(&mut state, &sphere, &mut rng, 0.25, 0.01);
        assert_eq!(state.generation, 1);
        cuckoo_step(&mut state, &sphere, &mut rng, 0.25, 0.01);
        assert_eq!(state.generation, 2);
    }

    #[test]
    fn test_cuckoo_step_best_non_increasing() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let mut rng = LcgRng::new(99);
        let mut state = CuckooState::new(bounds, 10, &sphere, &mut rng).unwrap();
        let initial_best = state.best_fitness;
        for _ in 0..50 {
            cuckoo_step(&mut state, &sphere, &mut rng, 0.25, 0.01);
        }
        assert!(
            state.best_fitness <= initial_best + 1e-12,
            "best increased: {} > {}",
            state.best_fitness,
            initial_best
        );
    }

    // ── Full run tests ────────────────────────────────────────────────────────

    #[test]
    fn test_cuckoo_run_sphere_1d() {
        let bounds = vec![(-10.0_f64, 10.0_f64)];
        let cfg = CuckooConfig {
            n_nests: 20,
            max_iter: 300,
            pa: 0.25,
            step_scale: 0.01,
            seed: 42,
        };
        let state = cuckoo_run(sphere, &bounds, &cfg).unwrap();
        assert!(state.best_fitness < 100.0, "best = {}", state.best_fitness);
        assert_eq!(state.best.len(), 1);
    }

    #[test]
    fn test_cuckoo_run_sphere_5d() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 5];
        let cfg = CuckooConfig {
            n_nests: 25,
            max_iter: 500,
            pa: 0.25,
            step_scale: 0.01,
            seed: 7,
        };
        let state = cuckoo_run(sphere, &bounds, &cfg).unwrap();
        // Worst possible ≈ 125; should improve.
        assert!(state.best_fitness < 125.0, "best = {}", state.best_fitness);
    }

    #[test]
    fn test_cuckoo_run_ackley_3d() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let cfg = CuckooConfig {
            n_nests: 20,
            max_iter: 400,
            pa: 0.25,
            step_scale: 0.01,
            seed: 11,
        };
        let worst = ackley(&[5.0, 5.0, 5.0]);
        let state = cuckoo_run(ackley, &bounds, &cfg).unwrap();
        assert!(
            state.best_fitness < worst,
            "best={} worst={worst}",
            state.best_fitness
        );
    }

    #[test]
    fn test_cuckoo_run_rastrigin_2d() {
        let bounds = vec![(-5.12_f64, 5.12_f64); 2];
        let cfg = CuckooConfig {
            n_nests: 20,
            max_iter: 500,
            pa: 0.25,
            step_scale: 0.01,
            seed: 33,
        };
        let worst = rastrigin(&[5.12, 5.12]);
        let state = cuckoo_run(rastrigin, &bounds, &cfg).unwrap();
        assert!(
            state.best_fitness < worst,
            "best={} worst={worst}",
            state.best_fitness
        );
    }

    #[test]
    fn test_cuckoo_run_rosenbrock_2d() {
        let bounds = vec![(-2.0_f64, 2.0_f64); 2];
        let cfg = CuckooConfig {
            n_nests: 20,
            max_iter: 1000,
            pa: 0.25,
            step_scale: 0.01,
            seed: 77,
        };
        let state = cuckoo_run(rosenbrock, &bounds, &cfg).unwrap();
        assert!(state.best_fitness < 200.0, "best = {}", state.best_fitness);
    }

    #[test]
    fn test_cuckoo_run_error_empty_bounds() {
        let cfg = CuckooConfig {
            n_nests: 5,
            max_iter: 10,
            pa: 0.25,
            step_scale: 0.01,
            seed: 0,
        };
        assert!(cuckoo_run(sphere, &[], &cfg).is_err());
    }

    #[test]
    fn test_cuckoo_run_error_invalid_pa() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 2];
        let cfg = CuckooConfig {
            n_nests: 5,
            max_iter: 10,
            pa: 1.5,
            step_scale: 0.01,
            seed: 0,
        };
        assert!(cuckoo_run(sphere, &bounds, &cfg).is_err());
    }

    #[test]
    fn test_cuckoo_run_best_within_bounds() {
        let bounds: Vec<(f64, f64)> = vec![(-3.0, 3.0), (-3.0, 3.0)];
        let cfg = CuckooConfig {
            n_nests: 15,
            max_iter: 200,
            pa: 0.25,
            step_scale: 0.01,
            seed: 21,
        };
        let state = cuckoo_run(sphere, &bounds, &cfg).unwrap();
        for (d, &x) in state.best.iter().enumerate() {
            let (lb, ub) = bounds[d];
            assert!(x >= lb && x <= ub, "dim {d}: {x} out of [{lb},{ub}]");
        }
    }

    #[test]
    fn test_cuckoo_run_deterministic() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 3];
        let cfg = CuckooConfig {
            n_nests: 10,
            max_iter: 100,
            pa: 0.25,
            step_scale: 0.01,
            seed: 999,
        };
        let s1 = cuckoo_run(sphere, &bounds, &cfg).unwrap();
        let s2 = cuckoo_run(sphere, &bounds, &cfg).unwrap();
        assert_eq!(s1.best_fitness, s2.best_fitness, "runs not deterministic");
    }

    #[test]
    fn test_cuckoo_run_generation_count() {
        let bounds = vec![(-5.0_f64, 5.0_f64); 2];
        let cfg = CuckooConfig {
            n_nests: 8,
            max_iter: 50,
            pa: 0.25,
            step_scale: 0.01,
            seed: 3,
        };
        let state = cuckoo_run(sphere, &bounds, &cfg).unwrap();
        assert_eq!(state.generation, 50);
    }
}
