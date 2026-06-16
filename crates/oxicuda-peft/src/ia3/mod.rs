/// IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) scaling vectors.
pub mod ia3;
/// IA³ combined with bottleneck adapter for hybrid scaling (He et al. 2022).
pub mod ia3_adapter;
pub use ia3_adapter::{Ia3AdapterConfig, Ia3AdapterLayer};
pub mod ia3_layer;
pub use ia3_layer::{Ia3Config, Ia3Layer};
