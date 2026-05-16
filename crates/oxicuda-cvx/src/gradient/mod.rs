//! First-order gradient methods (projected GD, accelerated GD, heavy-ball).

pub mod accelerated_gd;
pub mod momentum_gd;
pub mod projected_gradient;

pub use accelerated_gd::nesterov_accelerated;
pub use momentum_gd::heavy_ball;
pub use projected_gradient::projected_gradient;
