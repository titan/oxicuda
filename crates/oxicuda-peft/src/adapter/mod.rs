/// AdapterFusion: attention-based composition of multiple task adapter outputs.
pub mod adapter_fusion;
/// Compacter: PHM-parameterized hypercomplex bottleneck adapter.
pub mod compacter;
/// Houlsby adapter: bottleneck FFN with LayerNorm and residual.
pub mod houlsby;
/// QuaternionAdapter: Hamilton-product hypercomplex bottleneck adapter.
pub mod hypercomplex;
/// LST: Ladder Side-Tuning — side-network without back-prop through frozen trunk.
pub mod lst;
/// Parallel adapter: bottleneck branch summed with the main FFN output.
pub mod parallel_adapter;
/// Pfeiffer adapter: simpler bottleneck FFN without LayerNorm.
pub mod pfeiffer;

#[cfg(test)]
mod hypercomplex_tests;
#[cfg(test)]
mod lst_tests;

pub use adapter_fusion::{AdapterFusion, AdapterFusionConfig};
pub use hypercomplex::{Quat, QuatMatrix, QuaternionAdapter, QuaternionAdapterConfig};
pub use lst::{LadderSideTuning, LstBlock, LstConfig};
