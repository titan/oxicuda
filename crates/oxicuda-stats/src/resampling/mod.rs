//! Resampling-based statistical inference: bootstrap, jackknife, permutation tests.

pub mod bootstrap;
pub mod empirical_likelihood;
pub mod jackknife;
pub mod multilevel_bootstrap;
pub mod permutation;
pub mod vectorised_permutation;

pub use bootstrap::{BootstrapResult, bootstrap};
pub use empirical_likelihood::{
    ElConfig, ElResult, el_confidence_interval, el_mean_test, el_ratio_test,
};
pub use jackknife::{JackknifeResult, jackknife};
pub use multilevel_bootstrap::{
    ClusterBootstrapConfig, ClusterBootstrapResult, cluster_bootstrap, jackknife_cluster,
    two_level_bootstrap,
};
pub use permutation::{PermutationResult, permutation_test};
pub use vectorised_permutation::{
    VecPermConfig, VecPermResult, batch_two_sample_stats, permutation_matrix,
    vectorised_permutation_test,
};
