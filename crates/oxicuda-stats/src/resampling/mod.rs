//! Resampling-based statistical inference: bootstrap, jackknife, permutation tests.

pub mod bootstrap;
pub mod jackknife;
pub mod permutation;

pub use bootstrap::{BootstrapResult, bootstrap};
pub use jackknife::{JackknifeResult, jackknife};
pub use permutation::{PermutationResult, permutation_test};
