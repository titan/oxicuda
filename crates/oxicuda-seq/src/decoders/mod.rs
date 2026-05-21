//! Stochastic decoders for autoregressive sequence generation.
//!
//! This module collects sampling-based decoding strategies that are
//! complementary to the greedy/beam-search decoders in [`crate::beam`].
//! Each submodule implements a different truncation criterion on the
//! softmax distribution before drawing a single token:
//!
//! * [`top_k`] — Fan, Lewis & Dauphin (2018), *Hierarchical Neural Story
//!   Generation*: restrict the support to the `k` most-likely tokens.
//! * [`nucleus`] — Holtzman, Buys, Du, Forbes & Choi (2020), *The Curious
//!   Case of Neural Text Degeneration*: restrict the support to the
//!   smallest set whose cumulative probability exceeds `p`.
//! * [`typical`] — Meister, Pimentel, Wiher & Cotterell (2022), *Typical
//!   Decoding for Natural Language Generation*: restrict the support to
//!   tokens whose negative log-probability is closest to the conditional
//!   entropy of the distribution.
//!
//! All decoders are deterministic given a seeded [`crate::handle::LcgRng`]
//! and return [`crate::error::SeqResult`].

pub mod contrastive;
pub mod nucleus;
pub mod top_k;
pub mod typical;

pub use contrastive::{ContrastiveConfig, ContrastiveSearcher};
pub use nucleus::*;
pub use top_k::*;
pub use typical::*;
