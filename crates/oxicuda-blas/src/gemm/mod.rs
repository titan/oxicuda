//! Specialised GEMM variants and host-reference Level-3 kernels.
//!
//! This module provides alternative GEMM implementations beyond the core
//! GPU-dispatch path in [`crate::level3`], plus portable CPU reference
//! implementations of common Level-3 BLAS operations used to validate the
//! GPU kernels:
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`bf16_gemm`]    | Software BF16 (Brain Float 16) GEMM for precision experiments |
//! | [`mod@strassen`] | Strassen O(n^2.807) divide-and-conquer matrix multiplication   |
//! | [`batched_gemm`] | Batched SGEMM — many small `C_i = αA_iB_i + βC_i` in one call   |
//! | [`syrk`]         | Symmetric rank-k update `C = αAAᵀ + βC` (single triangle)       |
//! | [`trsm`]         | Triangular solve with multiple RHS `AX = αB`                    |

pub mod batched_gemm;
pub mod bf16_gemm;
pub mod strassen;
pub mod syrk;
pub mod trsm;

pub use batched_gemm::batched_sgemm;
pub use bf16_gemm::{Bf16, bf16_gemm_error, sgemm_bf16};
pub use strassen::{naive_dgemm, strassen, strassen_with_threshold};
pub use syrk::ssyrk;
pub use trsm::strsm;
