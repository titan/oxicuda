//! Neural network building blocks: MLP and coordinate MLP with Fourier features.

pub mod coordinate_mlp;
pub mod fbpinn;
pub mod hard_bc;
pub mod mlp;

pub use fbpinn::{Fbpinn, FbpinnConfig, Subdomain};
pub use hard_bc::{BoundaryDomain, HardBc, HardBcConfig};
