//! Drop-in compatibility shims mirroring vendor sparse-library APIs.
//!
//! The submodules here expose function and handle signatures shaped after
//! well-known sparse libraries so that code written against those APIs can be
//! ported to OxiCUDA with minimal churn. They are thin, idiomatic-Rust wrappers
//! over the crate's own host-resident sparse kernels: every routine returns a
//! [`SparseResult`](crate::error::SparseResult), validates its arguments, and
//! never exposes raw pointers or `unsafe`.
//!
//! - [`cusparse_compat`] -- a cuSPARSE-flavoured surface (`cusparse_spmv`,
//!   `cusparse_spgemm`, `cusparse_spsv`) over [`HostCsr`](crate::host_csr::HostCsr).

pub mod cusparse_compat;

pub use cusparse_compat::{
    CusparseCompatHandle, CusparsePointerMode, cusparse_spgemm, cusparse_spmv, cusparse_spsv,
};
