//! NAS operation building blocks.
//!
//! - [`primitives`] — The 8 canonical DARTS operations and their CPU reference implementations.
//! - [`mixed_op`] — `MixedOp`: softmax-weighted combination of candidate ops.
//! - [`search_space`] — `SearchSpace`, `CellSpace`, `NetworkSpace`.
//! - [`proxyless_nas`] — `BinaryGate` / `BinaryGateConfig` for ProxylessNAS binary path sampling.
//! - [`mbconv_ops`] — MBConv MAC and parameter cost estimation.
//! - [`transformer_nas`] — `BlockSpec` / `TransformerArch` / `TransformerSearchSpace`
//!   elastic transformer (AutoFormer / V-MoE) search axes: embed-dim, head-count,
//!   FFN width, depth, and MoE expert layout, with MAC / parameter accounting.

pub mod mbconv_ops;
pub mod mixed_op;
pub mod primitives;
pub mod proxyless_nas;
pub mod search_space;
pub mod transformer_nas;

pub use mbconv_ops::{MbConvSpec, mbconv_mac_count, mbconv_param_count};
pub use mixed_op::MixedOp;
pub use primitives::{OpKind, OpWeights};
pub use proxyless_nas::{BinaryGate, BinaryGateConfig};
pub use search_space::{CellSpace, NetworkSpace, SearchSpace};
pub use transformer_nas::{BlockSpec, TransformerArch, TransformerSearchSpace};
