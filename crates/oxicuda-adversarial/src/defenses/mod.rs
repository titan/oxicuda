//! Adversarial defence methods: training-time losses (TRADES, MART) and
//! certified inference (randomized smoothing, interval bound propagation,
//! Lipschitz-based radius).

pub mod certified_bounds;
pub mod mart;
pub mod randomized_smoothing;
pub mod trades;
