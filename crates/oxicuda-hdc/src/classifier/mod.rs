//! HD classifier modules: HD classifier with prototype-per-class and online update.

pub mod hd_classifier;
pub mod hd_ensemble;
pub mod persistence;
pub mod platt;
pub mod prototype;
pub mod rare_class;

pub use hd_ensemble::{HdEnsemble, HdEnsembleConfig};
pub use persistence::HdModel;
pub use platt::{PlattConfig, PlattScaler, fit as platt_fit};
pub use rare_class::{RareClassClassifier, RareClassConfig};
