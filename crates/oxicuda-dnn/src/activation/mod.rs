//! CPU-reference gated-activation primitives.
//!
//! - [`swiglu`] — SwiGLU gated feed-forward network (Shazeer 2020), the
//!   gated-MLP variant used by PaLM, LLaMA, and Mistral.
//!
//! All tensors use flat row-major `Vec<f32>` / `[f32]` layouts and run on the
//! CPU. Random initialisation uses [`crate::position::DnnRng`].

pub mod swiglu;

pub use swiglu::{SwiGlu, SwiGluConfig};
