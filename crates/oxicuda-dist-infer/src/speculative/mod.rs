//! Speculative decoding strategies for distributed inference.
//!
//! Speculative decoding accelerates autoregressive generation by *drafting*
//! several future tokens cheaply and then *verifying* them with the base model
//! in a single forward pass, accepting the longest correct prefix.
//!
//! | Module | Strategy |
//! |--------|----------|
//! | [`medusa`] | Multiple lightweight decoding heads predict `t+1, t+2, …` (Cai 2024) |

pub mod medusa;

pub use medusa::{DistInferRng, MedusaConfig, MedusaHeads};
