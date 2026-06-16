//! Adversarial defence methods: training-time losses (TRADES, MART) and
//! certified inference (randomized smoothing, interval bound propagation,
//! Lipschitz-based radius).

pub mod awp;
pub mod certified_bounds;
pub mod crown;
pub mod laplace_smoothing;
pub mod lp_relaxation;
pub mod macer;
pub mod mart;
pub mod randomized_smoothing;
pub mod smoothing_lp;
pub mod trades;

pub use awp::{AwpConfig, AwpDefense, AwpWeightDelta};
pub use crown::{AlphaBound, CrownConfig, CrownVerifier, LinearLayer, NeuronBound};
pub use laplace_smoothing::{LaplaceSmoothing, LaplaceSmoothingConfig};
pub use lp_relaxation::{AffineLayer, LpRelaxConfig, LpRelaxVerifier, VerifiedBound};
pub use macer::{MacerConfig, MacerLoss};
pub use smoothing_lp::LpSmoothingCertifier;
