//! Kelley cutting-plane method.
//!
//! Outer-approximation minimisation of a convex function over a box (or
//! polytope), following Kelley (1960), "The Cutting-Plane Method for Solving
//! Convex Programs".

pub mod cutting_plane;

pub use cutting_plane::{
    CuttingPlaneConfig, CuttingPlaneResult, CuttingPlaneStatus, kelley_cutting_plane,
};
