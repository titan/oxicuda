//! 2-D pairwise CRF + mean-field variational inference.

pub mod grid_crf;
pub mod mean_field;

pub use grid_crf::GridCrf;
pub use mean_field::{MeanFieldConfig, mean_field_inference};
