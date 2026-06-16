//! Exact (un-regularised) Optimal Transport via the network-simplex method.

/// Auction algorithm for the linear assignment problem (Bertsekas 1988).
pub mod auction_alg;
/// Earth Mover's Distance entry point.
pub mod emd;
/// Network-simplex algorithm for exact OT.
pub mod network_simplex;

pub use auction_alg::{AuctionConfig, AuctionResult, auction_assignment, auction_min_cost};
