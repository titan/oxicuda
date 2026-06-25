//! Sparse linear algebra: structured-sparse GEMM.
//!
//! Provides the Ampere+ **2:4 structured sparsity** matrix multiply (sparse
//! Tensor Cores) — compression of a dense operand to the 2:4 format with 2-bit
//! lane metadata, a numerically-validated CPU reference SpGEMM, and PTX-string
//! generation for the `mma.sp`-based sparse kernel.
//!
//! | Item | Description |
//! |------|-------------|
//! | [`compress_2to4`] / [`decompress_2to4`] | Dense ↔ 2:4 compressed conversion |
//! | [`pack_metadata`] / [`unpack_metadata`] | 2-bit lane metadata packing |
//! | [`spgemm_2to4`]                         | CPU reference sparse GEMM        |
//! | [`SparseGemmConfig`] + [`generate_sparse_gemm_ptx`] | Sparse Tensor-Core PTX |

pub mod sparse_gemm;

pub use sparse_gemm::{
    Compressed2to4, GROUP, KEPT, SparseGemmConfig, TwoFourMeta, compress_2to4, decompress_2to4,
    generate_sparse_gemm_ptx, pack_metadata, spgemm_2to4, unpack_metadata,
};
