//! Boundary matrix, column reduction, and persistence pair extraction.

pub mod boundary;
pub mod cohomology;
pub mod cohomology_z;
pub mod gpu_reduction;
pub mod multi_parameter;
pub mod persistent;
pub mod reduction;
pub mod twist;
pub mod zigzag;

pub use boundary::BoundaryMatrix;
pub use cohomology::{
    CohomologyConfig, CohomologyPair, CohomologyResult, coboundary_matrix, euler_characteristic,
    persistent_cohomology, reduce_coboundary_matrix, verify_cohomology_homology_agreement,
};
pub use cohomology_z::{CohomologyZ, CohomologyZConfig, CohomologyZResult};
pub use gpu_reduction::{
    ChunkReductionStats, GpuReductionPlan, batched_column_reduce_ptx, chunked_parallel_reduce,
    vietoris_rips_edges_ptx, wasserstein_auction_ptx,
};
pub use multi_parameter::{
    BiFiltration, BigradedSimplex, HilbertFunction, MultiParameterPersistence,
};
pub use persistent::{PersistencePair, extract_persistence_pairs};
pub use reduction::reduce_boundary_matrix;
pub use twist::{TwistConfig, TwistReduction, TwistResult};
pub use zigzag::{
    ZigzagArrow, ZigzagBar, ZigzagBarcode, ZigzagComplex, ZigzagInput, zigzag_persistence,
};
