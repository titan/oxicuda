//! Tucker decomposition: HOSVD, HOOI, and ST-HOSVD.

pub mod hooi;
pub mod hosvd;
pub mod sthosvd;

pub use hooi::hooi;
pub use hosvd::{TuckerResult, hosvd, tucker_reconstruct};
pub use sthosvd::{SthosvdConfig, SthosvdResult, sthosvd, sthosvd_reconstruct};
