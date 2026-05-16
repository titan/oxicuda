//! Multi-objective evolutionary algorithms: NSGA-II and MOEA/D.

pub mod moead;
pub mod nsga2;

pub use moead::{MoeadConfig, generate_weight_vectors, moead_run};
pub use nsga2::{
    MultiObjectiveIndividual, Nsga2Config, crowding_distance, fast_nondominated_sort, nsga2_run,
    nsga2_tournament,
};
