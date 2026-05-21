pub mod autoint;
pub mod cin;
#[allow(clippy::module_inception)]
pub mod deepfm;
pub mod wide_deep;

pub use cin::{CinConfig, CinLayer, XDeepFm};
