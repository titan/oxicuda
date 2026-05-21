//! Boundary matrix, column reduction, and persistence pair extraction.

pub mod boundary;
pub mod cohomology;
pub mod cohomology_z;
pub mod persistent;
pub mod reduction;
pub mod twist;

pub use boundary::BoundaryMatrix;
pub use cohomology::{
    CohomologyConfig, CohomologyPair, CohomologyResult, coboundary_matrix, euler_characteristic,
    persistent_cohomology, reduce_coboundary_matrix, verify_cohomology_homology_agreement,
};
pub use cohomology_z::{CohomologyZ, CohomologyZConfig, CohomologyZResult};
pub use persistent::{PersistencePair, extract_persistence_pairs};
pub use reduction::reduce_boundary_matrix;
pub use twist::{TwistConfig, TwistReduction, TwistResult};
