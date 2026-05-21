//! Multi-objective evolutionary algorithms: NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, MOPSO, and Preference methods.

pub mod moead;
pub mod mopso;
pub mod nsga2;
pub mod nsga3;
pub mod preference;
pub mod sms_emoa;

pub use moead::{MoeadConfig, generate_weight_vectors, moead_run};
pub use mopso::{MopsoConfig, MopsoState, mopso_pareto_front, mopso_run, update_archive};
pub use nsga2::{
    MultiObjectiveIndividual, Nsga2Config, crowding_distance, fast_nondominated_sort, nsga2_run,
    nsga2_tournament,
};
pub use nsga3::{Nsga3Config, Nsga3State, generate_reference_points, nsga3_run};
pub use preference::{
    PrefMoeadConfig, PrefMoeadResult, RNsga2Config, RNsga2Result, pref_moead_run, r_nsga2_run,
};
pub use sms_emoa::{
    SmsEmoaConfig, SmsEmoaState, hv_contribution_2d, sms_emoa_pareto_front, sms_emoa_run,
};
