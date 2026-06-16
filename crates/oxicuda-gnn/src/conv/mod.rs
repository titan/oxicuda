//! GNN convolution primitives (self-contained, weight-bearing layers).

pub mod gcnii;
pub mod gin;

pub use gcnii::{Gcnii, GcniiConfig, gcnii_beta};
pub use gin::{GinConfig, GinConv, GnnRng};
