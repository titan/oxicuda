//! Internal linear-algebra helpers used by Cox and AFT solvers.
//!
//! NOT exported beyond the crate; private helpers for survival math only.

#![allow(dead_code)]

pub(crate) mod cholesky;
pub(crate) mod inverse;
pub(crate) mod matmul;
pub(crate) mod solve;
