//! `oxicuda-evol` — Evolutionary & Genetic Algorithms for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-evol
//! ├── genetic/        — Canonical GA: individuals, population, selection, crossover, mutation
//! ├── evolution/
//! │   ├── cmaes/      — CMA-ES: full covariance matrix adaptation evolution strategy
//! │   └── de/         — Differential Evolution: DE/rand/1, DE/best/1, jDE adaptive
//! ├── multiobjective/ — NSGA-II (fast non-dominated sort + crowding), MOEA/D (Tchebycheff)
//! ├── neuroevolution/ — NEAT: topology evolution, innovation tracking, speciation
//! ├── swarm/          — PSO (inertia weight), ACO (Elitist, TSP)
//! └── metrics/        — Hypervolume (2D), IGD, GD, spacing, Pareto front extraction
//! ```

pub mod benchmarks;
pub mod error;
pub mod evolution;
pub mod genetic;
pub mod handle;
pub mod metrics;
pub mod multiobjective;
pub mod neuroevolution;
pub mod ptx_kernels;
pub mod qd;
pub mod swarm;

pub use benchmarks::bbob::{
    BenchmarkResult, MoBenchmarkResult, ackley, dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7,
    ellipsoid, griewank, rastrigin, rosenbrock, run_cmaes_benchmark, run_nsga2_benchmark, schwefel,
    sphere, zdt1, zdt1_pareto_front_f2, zdt2, zdt2_pareto_front_f2, zdt3, zdt3_pareto_front_f2,
    zdt4, zdt4_pareto_front_f2, zdt6, zdt6_pareto_front_f2,
};
pub use benchmarks::wfg::{
    WfgParams, wfg_optimum_objectives, wfg1, wfg2, wfg3, wfg4, wfg5, wfg6, wfg7, wfg8, wfg9,
};
pub use error::{EvolError, EvolResult};
pub use evolution::coevolution::{CoevolConfig, CoevolMode, CoevolResult, coevolve};
pub use evolution::de::de_variants::{De as DeSimple, DeConfig as DeSimpleConfig, DeVariant};
pub use evolution::island::{IslandConfig, IslandResult, Topology, island_model_run};
pub use evolution::memetic::{Inheritance, MemeticConfig, MemeticResult, memetic_run};
pub use genetic::encoding::{
    GrayEncoder, cx_crossover, inversion_mutation, ox_crossover, pmx_crossover, two_opt_improve,
};
pub use genetic::parallel::{
    CellularGaConfig, CellularGaResult, MasterSlaveConfig, MasterSlaveResult, Neighbourhood,
    cellular_ga, master_slave_ga,
};
pub use metrics::hypervolume_nd::{
    dominates, hypervolume_contributions, hypervolume_nd, nondominated_filter,
};
pub use multiobjective::preference::{
    PrefMoeadConfig, PrefMoeadResult, RNsga2Config, RNsga2Result, pref_moead_run, r_nsga2_run,
};

#[cfg(test)]
mod e2e_tests;

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;
