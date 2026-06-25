//! Empirical characterisation of HDC scaling laws, measured on the crate's real primitives.
//!
//! * [`capacity`] — associative-memory capacity vs dimension `D` (linear law) and bundling SNR
//!   vs `k` (`√(D/k)` law), measured by actually running the bind-and-superpose memory, the
//!   cleanup recall, and the majority-vote bundle.

pub mod capacity;

pub use capacity::{
    CapacityConfig, CapacityPoint, SnrConfig, SnrPoint, bundle_snr_curve, bundle_snr_point,
    hopfield_capacity_curve, recall_accuracy,
};
