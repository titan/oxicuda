//! BLAS Level 1 — vector-vector operations.
//!
//! This module provides GPU-accelerated implementations of the classic BLAS
//! Level 1 routines. Each operation launches one or more PTX kernels via the
//! [`BlasHandle`](crate::handle::BlasHandle).
//!
//! | Function | Operation                          |
//! |----------|------------------------------------|
//! | [`fn@axpy`] | y = alpha * x + y                  |
//! | [`fn@scal`] | x = alpha * x                      |
//! | [`fn@dot`]  | result = x . y (dot product)       |
//! | [`fn@nrm2`] | result = ||x||_2 (L2 norm)         |
//! | [`fn@asum`] | result = sum |x_i| (L1 norm)       |
//! | [`fn@iamax`]| result = argmax |x_i|              |
//! | [`fn@copy_vec`] | y = x (vector copy)            |
//! | [`fn@swap`] | x <-> y (swap two vectors)         |

pub mod asum;
pub mod axpy;
pub mod copy_vec;
pub mod dot;
pub mod iamax;
pub mod nrm2;
pub mod scal;
pub mod swap;

// Re-export the primary entry-point functions.
pub use asum::asum;
pub use axpy::axpy;
pub use copy_vec::copy_vec;
pub use dot::dot;
pub use iamax::iamax;
pub use nrm2::nrm2;
pub use scal::scal;
pub use swap::swap;

/// Computes the minimum number of elements a buffer must hold for a vector
/// of `n` logical elements with the given stride (increment).
///
/// The last accessed element is at index `(n - 1) * |inc|`, so the buffer
/// needs at least `1 + (n - 1) * |inc|` elements.
#[inline]
pub(crate) fn required_elements(n: u32, inc: i32) -> usize {
    if n == 0 {
        return 0;
    }
    1 + (n as usize - 1) * inc.unsigned_abs() as usize
}

/// Default thread-block size for Level-1 element-wise kernels.
///
/// This is a deliberate tuning constant, not an arbitrary default. 256 is a
/// power of two and an exact multiple of the 32-lane warp, yielding 8 warps
/// per block — enough to hide global-memory latency on every supported SM
/// (sm_70…sm_90) while keeping register pressure and the tail (partial final
/// block) small for the bandwidth-bound Level-1 workloads, which is also the
/// block size cuBLAS uses for these kernels.
///
/// The Level-1 kernels bake this value into their generated PTX (via
/// `.maxntid` and the grid computation), so it is fixed per build rather than
/// chosen per launch from the driver occupancy API. An occupancy-driven,
/// per-`(kernel, sm)` block-size sweep would require regenerating and caching a
/// PTX variant per block size; that is deferred until an on-device benchmark
/// demonstrates a win over this constant.
pub(crate) const L1_BLOCK_SIZE: u32 = 256;
