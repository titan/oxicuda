/// Compacter: PHM-parameterized hypercomplex bottleneck adapter.
pub mod compacter;
/// Houlsby adapter: bottleneck FFN with LayerNorm and residual.
pub mod houlsby;
/// Parallel adapter: bottleneck branch summed with the main FFN output.
pub mod parallel_adapter;
/// Pfeiffer adapter: simpler bottleneck FFN without LayerNorm.
pub mod pfeiffer;
