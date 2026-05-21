//! Isolation-based anomaly scoring.
pub mod iforest_score;
pub mod iforest_tree;

pub use iforest_tree::{
    IforConfig, IforFit, IforTree, ifor_c_factor, ifor_fit, ifor_path_length, ifor_predict,
    ifor_score,
};
