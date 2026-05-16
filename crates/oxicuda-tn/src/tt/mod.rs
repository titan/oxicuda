//! Tensor-Train (TT) decomposition.

pub mod tt;
pub mod tt_cross;
pub mod tt_svd;

pub use tt::{TtCore, TtTensor};
pub use tt_cross::tt_cross;
pub use tt_svd::tt_svd;
