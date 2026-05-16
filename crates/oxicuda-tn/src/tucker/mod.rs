//! Tucker decomposition: HOSVD and HOOI.

pub mod hooi;
pub mod hosvd;

pub use hooi::hooi;
pub use hosvd::{TuckerResult, hosvd};
