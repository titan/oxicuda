//! Persistence diagram, barcode, and diagram distance metrics.

pub mod barcode;
pub mod diagram;
pub mod distance;

pub use barcode::{Bar, Barcode};
pub use diagram::PersistenceDiagram;
pub use distance::{bottleneck_distance, wasserstein_1};
