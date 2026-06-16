//! 2-D pairwise CRF + mean-field variational inference + loopy BP.

pub mod grid_crf;
pub mod loopy_bp;
pub mod mean_field;

pub use grid_crf::GridCrf;
pub use loopy_bp::{LoopyBp, LoopyBpConfig, LoopyBpResult};
pub use mean_field::{MeanFieldConfig, mean_field_inference};
