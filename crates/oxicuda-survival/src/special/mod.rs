//! Special mathematical functions used in survival analysis (gammaln, digamma).

pub mod digamma;
pub mod gammaln;

pub use digamma::digamma;
pub use gammaln::gammaln;
