//! NSGA-II on the constrained 2-objective **BNH** (Binh-Korn) problem.
//!
//! # What this example demonstrates
//!
//! 1. Encoding a **constrained** multi-objective problem for the current
//!    [`Nsga2Config`] API, which only supports box bounds and unconstrained
//!    objectives. We use the static **death-penalty** wrap: any individual
//!    violating one of the two BNH constraints has both of its objectives
//!    pushed to a very large value, which guarantees that NSGA-II's
//!    non-dominated sort relegates infeasible solutions to the worst fronts.
//! 2. Running NSGA-II with `pop_size = 100` for `200` generations from a
//!    fixed seed for **bit-reproducible** output.
//! 3. Filtering the final population to its feasible Pareto-front members
//!    and printing a sorted sample as `(f_1, f_2)` pairs.
//! 4. A tiny 40-row ASCII scatter plot of the front to give an at-a-glance
//!    feel for its shape (the BNH front is the well-known partial circle
//!    arc).
//! 5. Reporting the 2-D **hypervolume** of the final front against the
//!    reference point `(140, 50)` using
//!    [`oxicuda_evol::metrics::metrics::hypervolume_2d`].
//!
//! # Test problem — BNH (Binh & Korn, 1997)
//!
//! ```text
//!   minimise  f_1(x) = 4·x_1² + 4·x_2²
//!   minimise  f_2(x) = (x_1 − 5)² + (x_2 − 5)²
//!   s.t.      g_1(x) = (x_1 − 5)² + x_2² − 25      ≤ 0
//!             g_2(x) = (x_1 − 8)² + (x_2 + 3)² − 7.7 ≥ 0
//!             x_1 ∈ [0, 5],  x_2 ∈ [0, 3]
//! ```
//!
//! The true Pareto front is the smooth arc traced by:
//!  - `x_2 = x_1` for `x_1 ∈ [0, 3]`, plus
//!  - `x_2 = 3` for `x_1 ∈ [3, 5]`.
//!
//! # API note
//!
//! `Nsga2Config` carries a single shared `(lb, ub)` box; BNH actually has
//! different upper bounds for `x_1` and `x_2`. We pass `(0.0, 5.0)` and
//! clamp `x_2` to `[0, 3]` inside the wrapped objective so that infeasible
//! offspring along that axis are coerced back into the box before the
//! BNH constraints are checked.
//!
//! # References
//!
//! - K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, *A Fast and Elitist
//!   Multiobjective Genetic Algorithm: NSGA-II*, IEEE TEC 6(2), 2002.
//! - T. T. Binh, U. Korn, *MOBES: A Multiobjective Evolution Strategy for
//!   Constrained Optimization Problems*, MENDEL 1997.
//!
//! # Running
//!
//! ```bash
//! cargo run --example nsga2_constrained -p oxicuda-evol --all-features
//! ```

use std::time::Instant;

use oxicuda_evol::handle::LcgRng;
use oxicuda_evol::metrics::metrics::hypervolume_2d;
use oxicuda_evol::multiobjective::nsga2::{MultiObjectiveIndividual, Nsga2Config, nsga2_run};
use oxicuda_evol::{EvolError, EvolResult};

/// BNH decision variable count.
const N_DIMS: usize = 2;

/// BNH has two minimisation objectives.
const N_OBJECTIVES: usize = 2;

/// NSGA-II population size (even, as required by `Nsga2Config`).
const POP_SIZE: usize = 100;

/// NSGA-II generation budget.
const MAX_GENERATIONS: usize = 200;

/// SBX (Simulated Binary Crossover) distribution index — standard Deb default.
const CROSSOVER_ETA: f64 = 15.0;

/// Polynomial mutation distribution index — standard Deb default.
const MUTATION_ETA: f64 = 20.0;

/// Per-gene mutation probability — Deb recommends `1 / n_dims`.
const MUTATION_PROB: f64 = 1.0 / N_DIMS as f64;

/// Upper bound for `x_2`. The wider `(0, 5)` box is shared by NSGA-II's
/// internal variation operators; `x_2` is clamped to `[0, X2_UB]` inside the
/// fitness wrapper to enforce the genuine BNH bounds.
const X2_UB: f64 = 3.0;

/// Penalty value applied to **both** objectives when a candidate violates
/// either BNH constraint. Picked large enough to be strictly dominated by any
/// feasible solution but finite (so that `hypervolume_2d`'s `f < ref` filter
/// still functions sensibly when very early generations are all infeasible).
const PENALTY_VALUE: f64 = 1.0e9;

/// Hypervolume reference point — slightly above the worst feasible
/// `(f_1, f_2)` corner so the front is fully enclosed.
const HV_REFERENCE: (f64, f64) = (140.0, 50.0);

/// Number of representative front points to print as a sorted table.
const N_FRONT_SAMPLES: usize = 12;

/// ASCII scatter plot height (rows of characters).
const PLOT_HEIGHT: usize = 18;

/// ASCII scatter plot width (columns of characters, excluding axis labels).
const PLOT_WIDTH: usize = 60;

/// Fixed RNG seed for reproducibility — identical output on every run.
const SEED: u64 = 0x0b_8b00_2ec0_a4d3;

fn main() -> EvolResult<()> {
    println!("NSGA-II on BNH (Binh-Korn) — oxicuda-evol worked example");
    println!("=========================================================");
    println!(
        "config: n_dims={N_DIMS}, n_obj={N_OBJECTIVES}, pop={POP_SIZE}, \
         max_gen={MAX_GENERATIONS}, sbx_eta={CROSSOVER_ETA}, mut_eta={MUTATION_ETA}, \
         mut_prob={MUTATION_PROB:.4}, seed=0x{SEED:016x}"
    );
    println!();

    let cfg = Nsga2Config {
        n_dims: N_DIMS,
        n_objectives: N_OBJECTIVES,
        pop_size: POP_SIZE,
        max_generations: MAX_GENERATIONS,
        crossover_eta: CROSSOVER_ETA,
        mutation_eta: MUTATION_ETA,
        mutation_prob: MUTATION_PROB,
        bounds: (0.0, 5.0),
    };
    let mut rng = LcgRng::new(SEED);

    let t_start = Instant::now();
    let population = nsga2_run(bnh_penalised, &cfg, &mut rng)?;
    let elapsed = t_start.elapsed();

    // ── Filter to feasible Pareto-front members ──────────────────────────
    let feasible_front: Vec<&MultiObjectiveIndividual> = population
        .iter()
        .filter(|ind| ind.rank == 0 && bnh_is_feasible(&ind.genome))
        .collect();

    println!(
        "wall total       : {:.3} ms  ({:>3} generations × {} pop)",
        elapsed.as_secs_f64() * 1.0e3,
        MAX_GENERATIONS,
        POP_SIZE
    );
    println!(
        "population size  : {} (feasible Pareto front: {})",
        population.len(),
        feasible_front.len()
    );

    if feasible_front.is_empty() {
        // Defensive: should not happen with this seed/config — but report the
        // best feasible fitnesses if any solution is feasible at all.
        let any_feasible = population
            .iter()
            .filter(|i| bnh_is_feasible(&i.genome))
            .count();
        println!(
            "WARN: feasible Pareto-front is empty (total feasible individuals: {any_feasible})."
        );
        return Err(EvolError::EmptyPopulation);
    }

    // ── Sample of front sorted by f_1 ascending ──────────────────────────
    let mut front_pts: Vec<(f64, f64)> = feasible_front
        .iter()
        .map(|ind| (ind.objectives[0], ind.objectives[1]))
        .collect();
    front_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("--- Pareto-front sample (sorted by f_1) ---");
    println!("{:>6} | {:>14} | {:>14}", "#", "f_1", "f_2");
    println!("{:->6}-+-{:->14}-+-{:->14}", "", "", "");
    let stride = (front_pts.len().saturating_sub(1)).max(1) / N_FRONT_SAMPLES.max(1);
    let stride = stride.max(1);
    let mut shown = 0usize;
    for (i, &(f1, f2)) in front_pts.iter().enumerate() {
        if i % stride == 0 || i + 1 == front_pts.len() {
            println!("{:>6} | {:>14.6} | {:>14.6}", i, f1, f2);
            shown += 1;
            if shown > N_FRONT_SAMPLES {
                break;
            }
        }
    }

    // ── ASCII scatter plot of the front ──────────────────────────────────
    println!();
    println!(
        "--- ASCII scatter ({} × {}, '*'=feasible front point) ---",
        PLOT_WIDTH, PLOT_HEIGHT
    );
    render_ascii_scatter(&front_pts);

    // ── Hypervolume of the final front ───────────────────────────────────
    let hv = hypervolume_2d(&front_pts, HV_REFERENCE)?;
    println!();
    println!(
        "--- Hypervolume (reference {:?}) ---  HV = {:.4}",
        HV_REFERENCE, hv
    );

    // ── Acceptance gate: the BNH analytic HV w.r.t. (140, 50) is ~5300; we
    //    require at least 80% as a generous floor so that any future
    //    regression in NSGA-II is caught loudly. ──────────────────────────
    if hv < 4_240.0 {
        eprintln!(
            "ERROR: hypervolume {hv:.2} is below the acceptance floor (4240). \
             NSGA-II may have regressed."
        );
        return Err(EvolError::ConvergenceFailed {
            iter: MAX_GENERATIONS,
        });
    }

    println!();
    println!("NSGA-II BNH worked example complete.");
    Ok(())
}

/// Wrapped BNH objective applying a death-penalty for constraint violations.
/// Decision variables are first clamped into the true BNH box, then the two
/// objectives are computed; if either constraint is violated, both objectives
/// are replaced by `PENALTY_VALUE`.
fn bnh_penalised(x: &[f64]) -> Vec<f64> {
    let (x1, x2) = clamp_to_box(x);
    let g1 = (x1 - 5.0).powi(2) + x2 * x2 - 25.0;
    let g2 = (x1 - 8.0).powi(2) + (x2 + 3.0).powi(2) - 7.7;
    let feasible = g1 <= 0.0 && g2 >= 0.0;
    if feasible {
        vec![
            4.0 * x1 * x1 + 4.0 * x2 * x2,
            (x1 - 5.0).powi(2) + (x2 - 5.0).powi(2),
        ]
    } else {
        vec![PENALTY_VALUE, PENALTY_VALUE]
    }
}

/// Feasibility test that operates on the **clamped** genome (the same data
/// the objective wrapper sees) so reporting is consistent.
fn bnh_is_feasible(x: &[f64]) -> bool {
    let (x1, x2) = clamp_to_box(x);
    let g1 = (x1 - 5.0).powi(2) + x2 * x2 - 25.0;
    let g2 = (x1 - 8.0).powi(2) + (x2 + 3.0).powi(2) - 7.7;
    g1 <= 0.0 && g2 >= 0.0
}

/// Clamp a 2-D candidate into the true BNH box `[0, 5] × [0, 3]`.
fn clamp_to_box(x: &[f64]) -> (f64, f64) {
    let x1 = x[0].clamp(0.0, 5.0);
    let x2 = x[1].clamp(0.0, X2_UB);
    (x1, x2)
}

/// Render a fixed-size ASCII scatter plot of the front. Axes are autoscaled
/// to the front's bounding box with a small padding on each side.
fn render_ascii_scatter(front_pts: &[(f64, f64)]) {
    if front_pts.is_empty() {
        println!("(empty front)");
        return;
    }
    let (mut f1_min, mut f1_max, mut f2_min, mut f2_max) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &(f1, f2) in front_pts {
        if f1 < f1_min {
            f1_min = f1;
        }
        if f1 > f1_max {
            f1_max = f1;
        }
        if f2 < f2_min {
            f2_min = f2;
        }
        if f2 > f2_max {
            f2_max = f2;
        }
    }
    let f1_span = (f1_max - f1_min).max(1.0e-12);
    let f2_span = (f2_max - f2_min).max(1.0e-12);

    // Grid: rows × cols of bytes.
    let mut grid = vec![vec![b' '; PLOT_WIDTH]; PLOT_HEIGHT];
    for &(f1, f2) in front_pts {
        let col = ((f1 - f1_min) / f1_span * (PLOT_WIDTH as f64 - 1.0)).round() as isize;
        // f_2 axis: smaller is up (visual axes have +y upwards).
        let row = ((f2_max - f2) / f2_span * (PLOT_HEIGHT as f64 - 1.0)).round() as isize;
        if (0..PLOT_HEIGHT as isize).contains(&row) && (0..PLOT_WIDTH as isize).contains(&col) {
            grid[row as usize][col as usize] = b'*';
        }
    }
    // Top axis label.
    println!("  f_2 max = {:>10.3}  |", f2_max);
    for (i, row) in grid.iter().enumerate() {
        let row_str: String = row.iter().map(|&b| b as char).collect();
        if i == 0 {
            println!("    |{row_str}|  <- f_2 = {:>10.3}", f2_max);
        } else if i + 1 == PLOT_HEIGHT {
            println!("    |{row_str}|  <- f_2 = {:>10.3}", f2_min);
        } else {
            println!("    |{row_str}|");
        }
    }
    let footer: String = std::iter::repeat_n('-', PLOT_WIDTH).collect();
    println!("    +{footer}+");
    println!(
        "     f_1 = {:>8.3}  →  {:>8.3}   (rows = f_2 small at top, large at bottom)",
        f1_min, f1_max
    );
}
