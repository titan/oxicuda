//! HD classifier modules: HD classifier with prototype-per-class and online update.

pub mod hd_classifier;
pub mod hd_ensemble;
pub mod prototype;

pub use hd_ensemble::{HdEnsemble, HdEnsembleConfig};
