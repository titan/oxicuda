//! BBOB-style black-box benchmark test functions and algorithm harness.
//!
//! Provides standard benchmark functions (sphere, Rosenbrock, Rastrigin, etc.)
//! and driver routines to evaluate CMA-ES and NSGA-II on canonical test problems.

pub mod bbob;

pub use bbob::{
    BenchmarkResult, MoBenchmarkResult, ackley, dtlz1, ellipsoid, griewank, rastrigin, rosenbrock,
    run_cmaes_benchmark, run_nsga2_benchmark, schwefel, sphere, zdt1, zdt2,
};
