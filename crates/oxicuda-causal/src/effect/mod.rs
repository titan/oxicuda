pub mod bart;
pub mod double_ml;
pub mod doubly_robust;
pub mod dragonnet;
pub mod g_computation;
pub mod ipw;
pub mod mediation;
pub mod mediation_formula;
#[cfg(test)]
mod mediation_tests;
pub mod meta_learners;
pub mod propensity;
pub mod r_learner;
pub mod rdd;
#[cfg(test)]
mod rdd_tests;
pub mod sequential_g;
pub mod synthetic_control;
pub mod tmle;
#[cfg(test)]
mod tmle_tests;

pub use bart::{Bart, BartConfig, BartNode, BartTree};
pub use g_computation::{GComputationConfig, GComputationResult, g_computation};
pub use mediation::{Mediation, MediationConfig, MediationResult};
pub use r_learner::{RLearnerConfig, RLearnerResult, r_learner};
pub use rdd::{Rdd, RddConfig, RddKernel, RddResult};
pub use sequential_g::{BlipFunction, SequentialGConfig, SequentialGEstimator, SequentialGResult};
pub use synthetic_control::{SyntheticControlConfig, SyntheticControlResult, synthetic_control};
pub use tmle::{Tmle, TmleConfig, TmleResult};
pub mod local_centering;
pub use local_centering::{LocalCentering, LocalCenteringConfig, LocalCenteringResult};
pub mod ope;
pub use ope::{OpeInput, OpeResult, ope_evaluate};
pub mod subsampled_bootstrap;
pub use subsampled_bootstrap::{
    SubsampledBootstrapConfig, SubsampledBootstrapResult, subsampled_bootstrap,
    subsampled_bootstrap_vec,
};
pub mod cevae;
pub use cevae::{Cevae, CevaeConfig};
pub use mediation_formula::{MediationFormulaConfig, MediationFormulaResult, mediation_formula};
pub mod staggered_did;
pub use staggered_did::{
    Aggregation, EventStudyPoint, GroupTimeAtt, NEVER_TREATED, StaggeredDidConfig,
    StaggeredDidResult, callaway_santanna,
};
pub mod psm;
pub use psm::{MatchMetric, MatchedPair, PsmConfig, PsmResult, psm_att};
pub mod entropy_balancing;
pub use entropy_balancing::{EntropyBalancingConfig, EntropyBalancingResult, entropy_balancing};
pub mod stabilized_ipw;
pub use stabilized_ipw::{StabilizedIpwConfig, StabilizedIpwResult, stabilized_ipw};
