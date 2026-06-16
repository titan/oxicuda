//! Dynamic time warping distance and alignment.
pub mod dtw;
pub use dtw::{
    DtwConfig, DtwResult, dtw, dtw_barycenter, dtw_cost_matrix, dtw_distance,
    dtw_distance_matrix,
};
