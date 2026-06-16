//! Offline change-point detection: PELT, binary segmentation, and CUSUM for
//! mean-shift detection in univariate series.

pub mod changepoint;

pub use changepoint::{
    BinSegConfig, CusumResult, PeltConfig, binary_segmentation, cusum, pelt, segment_means,
};
