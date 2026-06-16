//! Disciplined Convex Programming (DCP) expression trees.
//!
//! Curvature inference, evaluation, gradients and a DCP validity checker for
//! convex-program expression trees, following Grant & Boyd (2008),
//! "Graph Implementations for Nonsmooth Convex Programs".

pub mod expr_tree;

pub use expr_tree::{Constraint, ConstraintKind, Curvature, Expr, Monotonicity, is_dcp};
