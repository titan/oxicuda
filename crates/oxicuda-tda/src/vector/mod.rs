//! Vectorised representations of persistence diagrams and barcodes.
pub mod betti_curve;
pub mod persistence_statistics;
pub use betti_curve::{BettiCurve, betti_curve, betti_curve_from_barcode, betti_curves_all_dims};
pub use persistence_statistics::{PersistenceStatistics, persistence_statistics};
