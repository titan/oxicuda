//! BBOB-style black-box benchmark test functions and algorithm harness.
//!
//! Provides standard benchmark functions (sphere, Rosenbrock, Rastrigin, etc.)
//! and driver routines to evaluate CMA-ES and NSGA-II on canonical test problems.

pub mod bbob;
pub mod wfg;

pub use bbob::{
    BenchmarkResult, MoBenchmarkResult, ackley, dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7,
    ellipsoid, griewank, rastrigin, rosenbrock, run_cmaes_benchmark, run_nsga2_benchmark, schwefel,
    sphere, zdt1, zdt1_pareto_front_f2, zdt2, zdt2_pareto_front_f2, zdt3, zdt3_pareto_front_f2,
    zdt4, zdt4_pareto_front_f2, zdt6, zdt6_pareto_front_f2,
};
pub use wfg::{
    WfgParams, wfg_optimum_objectives, wfg1, wfg2, wfg3, wfg4, wfg5, wfg6, wfg7, wfg8, wfg9,
};
