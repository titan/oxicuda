//! CMA-ES on the 10-dimensional Rosenbrock function — a worked example.
//!
//! # What this example demonstrates
//!
//! This example drives the public [`CmaEsState`] API generation-by-generation
//! (rather than calling [`CmaEsState::run`] in one shot) so that we can:
//!
//! 1. Construct a [`CmaEsConfig`] with custom **σ₀**, **λ** (population) and
//!    **μ** (parents) rather than the defaults `CmaEsConfig::new(n)` produces.
//! 2. Pass a fixed seed into [`LcgRng`] for **bit-reproducible** output —
//!    running the example twice yields identical numbers.
//! 3. Log per-generation diagnostics: the best & mean fitness within the
//!    population, the current step size σ, and the **condition number** of
//!    the covariance matrix `C` computed as `(maxᵢ Dᵢ² / minᵢ Dᵢ²)` from the
//!    internal eigenvalues `Dᵢ²` cached in `state.d_vector`.
//! 4. Print a final convergence summary containing the best `x*`, the
//!    evaluated `f(x*)`, the total number of objective evaluations and the
//!    mean wall-clock time per generation.
//!
//! # Test problem
//!
//! Generalised Rosenbrock in 10 dimensions:
//!
//! ```text
//!     f(x) = Σ_{i=1}^{n-1} [100·(x_{i+1} − x_i²)² + (1 − x_i)²]
//! ```
//!
//! with global minimum `x* = (1, 1, …, 1)` and `f(x*) = 0`. The function has a
//! very narrow curved valley which makes it a classic CMA-ES smoke test.
//!
//! # Reference
//!
//! N. Hansen, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772, 2016.
//! <https://arxiv.org/abs/1604.00772>
//!
//! # API note
//!
//! The public [`CmaEsState`] does not currently expose a "best-so-far" tracker
//! when driving it manually, so this example maintains one externally.
//!
//! # Running
//!
//! ```bash
//! cargo run --example cmaes_blackbox -p oxicuda-evol --all-features
//! ```

use std::time::Instant;

use oxicuda_evol::benchmarks::bbob::rosenbrock;
use oxicuda_evol::evolution::cmaes::cmaes::{CmaEsConfig, CmaEsState};
use oxicuda_evol::handle::LcgRng;
use oxicuda_evol::{EvolError, EvolResult};

/// Problem dimension for the Rosenbrock test.
const N_DIMS: usize = 10;

/// Initial step size σ₀. Larger than the `CmaEsConfig::new` default (0.3) to
/// promote initial exploration over a fairly wide basin.
const SIGMA_INIT: f64 = 0.5;

/// Population size λ. The CMA-ES default for n=10 is 4 + ⌊3·ln 10⌋ = 10;
/// we double it here to demonstrate a larger non-default value.
const POP_SIZE: usize = 20;

/// Number of parents μ used in the weighted recombination. We pick ⌊λ/2⌋.
const MU: usize = POP_SIZE / 2;

/// Maximum number of generations to run before declaring failure.
///
/// The reference paper estimates ~250 generations for 10-D Rosenbrock at the
/// CMA-ES default population size; with our larger λ=20 we double the budget
/// to keep the *number of function evaluations* in the same ballpark while
/// still producing a long enough convergence trace to be informative.
const MAX_GEN: usize = 500;

/// Generation-by-generation logging stride (every N generations).
const LOG_STRIDE: usize = 50;

/// Convergence target on the Rosenbrock objective.
const TARGET_FITNESS: f64 = 1.0e-8;

/// Fixed RNG seed for reproducibility — identical output on every run.
const SEED: u64 = 0x01_cdae_7014_d033;

fn main() -> EvolResult<()> {
    // Title banner.
    println!("CMA-ES on 10-D Rosenbrock — oxicuda-evol worked example");
    println!("========================================================");
    println!(
        "config: n_dims={N_DIMS}, pop_size(λ)={POP_SIZE}, mu={MU}, sigma_init={SIGMA_INIT}, \
         max_gen={MAX_GEN}, target_fitness={TARGET_FITNESS:.1e}, seed=0x{SEED:016x}"
    );
    println!();

    // ── Configure CMA-ES with non-default σ₀, λ, μ ────────────────────────
    let cfg = CmaEsConfig {
        n_dims: N_DIMS,
        pop_size: POP_SIZE,
        mu: MU,
        sigma_init: SIGMA_INIT,
        // Allow plenty of evaluation budget so termination is driven by
        // the generation loop below or by `tol_fun`.
        max_evals: MAX_GEN * POP_SIZE + POP_SIZE,
        tol_fun: TARGET_FITNESS * 1.0e-2,
        tol_x: 1.0e-11,
    };

    // ── Start the search away from x* = (1, …, 1) ────────────────────────
    // Starting from the origin (default for `run_cmaes_benchmark`) is the
    // canonical CMA-ES initialisation for Rosenbrock — the algorithm must
    // still cross the curved valley to reach the optimum.
    let mean_init: Vec<f64> = vec![0.0_f64; N_DIMS];
    let mut state = CmaEsState::new(mean_init.clone(), &cfg)?;
    let mut rng = LcgRng::new(SEED);

    // ── External best-so-far tracker (the public API does not expose one) ─
    let mut best_x = mean_init;
    let mut best_fit = rosenbrock(&best_x);
    let mut total_evals: usize = 1; // counted the centre evaluation

    println!(
        "{:>5} | {:>14} | {:>14} | {:>10} | {:>14} | {:>14}",
        "gen", "best_fitness", "mean_fitness", "sigma", "cond(C)", "elapsed_ms"
    );
    println!(
        "{:->5}-+-{:->14}-+-{:->14}-+-{:->10}-+-{:->14}-+-{:->14}",
        "", "", "", "", "", ""
    );

    let t_start = Instant::now();
    let mut gens_elapsed: usize = 0;
    let mut converged = false;

    for generation in 1..=MAX_GEN {
        let t_gen = Instant::now();
        let samples = state.sample(&cfg, &mut rng);
        let fitnesses: Vec<f64> = samples.iter().map(|x| rosenbrock(x)).collect();
        total_evals += fitnesses.len();

        // Update the external best-so-far tracker.
        for (x, &f) in samples.iter().zip(fitnesses.iter()) {
            if f < best_fit {
                best_fit = f;
                best_x = x.clone();
            }
        }

        state.update(&samples, &fitnesses, &cfg)?;
        gens_elapsed = generation;
        let elapsed_ms = t_gen.elapsed().as_secs_f64() * 1.0e3;

        // Logging at a reasonable cadence (and always at gen 1 / final gen).
        let is_log_gen =
            generation == 1 || generation.is_multiple_of(LOG_STRIDE) || best_fit < TARGET_FITNESS;
        if is_log_gen {
            let mean_fit: f64 = fitnesses.iter().copied().sum::<f64>() / fitnesses.len() as f64;
            let cond_c = covariance_condition_number(&state.d_vector);
            println!(
                "{:>5} | {:>14.6e} | {:>14.6e} | {:>10.4e} | {:>14.6e} | {:>14.3}",
                generation, best_fit, mean_fit, state.sigma, cond_c, elapsed_ms
            );
        }

        if best_fit < TARGET_FITNESS {
            converged = true;
            break;
        }
        if state.sigma < cfg.tol_x {
            // Sigma collapsed before reaching target: stop early.
            break;
        }
    }

    let wall_total = t_start.elapsed();
    let wall_per_gen_ms = wall_total.as_secs_f64() * 1.0e3 / gens_elapsed.max(1) as f64;

    println!();
    println!("--- Convergence summary ---");
    println!("generations run  : {}", gens_elapsed);
    println!("objective evals  : {}", total_evals);
    println!(
        "wall total       : {:.3} ms ({:.3} ms / gen)",
        wall_total.as_secs_f64() * 1.0e3,
        wall_per_gen_ms
    );
    println!("best fitness     : {:.6e}", best_fit);
    println!("target           : {:.6e}", TARGET_FITNESS);
    println!("converged        : {}", converged);

    // Print best x* component-by-component (each should be near 1.0).
    print!("best x*          : [");
    for (i, &xi) in best_x.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:+.6}", xi);
    }
    println!("]");

    // Distance to known global optimum (1, …, 1).
    let dist_to_opt: f64 = best_x
        .iter()
        .map(|&xi| (xi - 1.0).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("||x* − 1||₂      : {:.6e}", dist_to_opt);

    // Acceptance criterion: f(x*) < 1.0e-2 (the wave brief floor).
    if best_fit >= 1.0e-2 {
        return Err(EvolError::ConvergenceFailed { iter: gens_elapsed });
    }

    Ok(())
}

/// Compute the condition number of the covariance matrix `C` from the cached
/// eigenvalues. `CmaEsState::d_vector[i] = √λᵢ`, so the condition number is
/// `(maxᵢ λᵢ) / (minᵢ λᵢ) = (max Dᵢ / min Dᵢ)²`.
fn covariance_condition_number(d_vector: &[f64]) -> f64 {
    let mut d_min = f64::INFINITY;
    let mut d_max = 0.0_f64;
    for &d in d_vector {
        let d_abs = d.abs();
        if d_abs < d_min {
            d_min = d_abs;
        }
        if d_abs > d_max {
            d_max = d_abs;
        }
    }
    if d_min <= 0.0 || !d_min.is_finite() {
        f64::INFINITY
    } else {
        (d_max / d_min).powi(2)
    }
}
