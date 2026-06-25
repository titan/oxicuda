//! SSM core submodules.
//!
//! - [`discretize`] — Convert continuous-time `(A, B)` to discrete `(Ā, B̄)`.
//! - [`parallel_scan`] — Associative prefix scan for efficient state computation.
//! - [`ssm_kernel`] — Full forward-pass SSM kernel operating on discrete parameters.
//! - [`hippo_variants`] — HiPPO-LegT and HiPPO-FOUT alternative polynomial projection matrices.
//! - [`liquid`] — Liquid-S4: input-modulated-`Δ` diagonal SSM with per-neuron `τ`.
//! - [`selective_scan_backward`] — Reverse-mode gradients for the linear scan recurrence.
//! - [`state_cache`] — Streaming SSM state cache (KV-cache analogue) with
//!   checkpoint / restore for long-context inference.

pub mod discretize;
pub mod hippo_variants;
pub mod liquid;
pub mod parallel_scan;
pub mod selective_scan_backward;
pub mod ssm_kernel;
pub mod state_cache;

pub use hippo_variants::{
    HippoFou, HippoFouConfig, HippoLegT, HippoLegTConfig, HippoMatrix, compare_hippo_variants,
    hippo_legs_matrix,
};
pub use liquid::{LiquidS4Config, LiquidS4Layer};
pub use selective_scan_backward::{
    BatchedScanGrads, ScanGrads, scan_backward, scan_backward_batched, scan_forward,
};
