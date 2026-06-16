//! Neural network building blocks: MLP and coordinate MLP with Fourier features.

pub mod coordinate_mlp;
pub mod fbpinn;
pub mod hard_bc;
pub mod mlp;
pub mod rbf_features;
pub mod reservoir_computing;
pub mod xpinn;

pub use fbpinn::{Fbpinn, FbpinnConfig, Subdomain};
pub use hard_bc::{BoundaryDomain, HardBc, HardBcConfig};
pub use rbf_features::{RbfFeatureConfig, RbfFeatureNetwork, RbfFeatures, RbfKind};
pub use reservoir_computing::{EchoStateNetwork, EsnConfig, spectral_radius};
pub use xpinn::{XPinn, XPinnConfig, XPinnInterfaceLoss, XPinnSubdomain};
