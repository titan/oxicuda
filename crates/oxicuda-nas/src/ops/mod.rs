//! NAS operation building blocks.
//!
//! - [`primitives`] — The 8 canonical DARTS operations and their CPU reference implementations.
//! - [`mixed_op`] — `MixedOp`: softmax-weighted combination of candidate ops.
//! - [`search_space`] — `SearchSpace`, `CellSpace`, `NetworkSpace`.

pub mod mixed_op;
pub mod primitives;
pub mod search_space;

pub use mixed_op::MixedOp;
pub use primitives::{OpKind, OpWeights};
pub use search_space::{CellSpace, NetworkSpace, SearchSpace};
