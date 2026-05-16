//! Quantile sketches: KLL, t-Digest, Greenwald-Khanna, P-square.

pub mod gk_quantile;
pub mod kll;
pub mod p_square;
pub mod t_digest;

pub use gk_quantile::GreenwaldKhanna;
pub use kll::KllSketch;
pub use p_square::PSquare;
pub use t_digest::TDigest;
