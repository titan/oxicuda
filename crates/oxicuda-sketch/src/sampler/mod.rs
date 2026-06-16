//! Stream samplers that draw coordinates from a turnstile-streamed vector.
//!
//! - [`lp_sampler`]: L0 / Lp sampler (Jowhari–Sağlam–Tardos 2011,
//!   Cormode–Firmani 2014) — returns a near-uniform random non-zero coordinate
//!   via geometric subsampling levels with per-level 1-sparse fingerprinted
//!   recovery.

pub mod lp_sampler;

pub use lp_sampler::LpSampler;
