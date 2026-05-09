//! Architecture-based continual learning methods.
//!
//! These methods prevent catastrophic forgetting by modifying the network
//! architecture, either by pruning, masking, or expanding the network.

pub mod packnet;
pub mod piggyback;
pub mod progressive;
