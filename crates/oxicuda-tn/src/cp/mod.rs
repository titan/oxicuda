//! CANDECOMP/PARAFAC (CP) decomposition.

pub mod als;
pub mod non_negative;

pub use als::{CpResult, cp_als};
pub use non_negative::{NnCpConfig, NnCpResult, nn_cp_decomp, nn_cp_reconstruct};
