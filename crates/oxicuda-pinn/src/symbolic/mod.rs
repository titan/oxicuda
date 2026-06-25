//! Symbolic / interpretable scientific-ML: genetic-programming symbolic
//! regression over expression trees.

pub mod regression;

pub use regression::{Expr, Individual, SymbolicConfig, SymbolicRegressor};
