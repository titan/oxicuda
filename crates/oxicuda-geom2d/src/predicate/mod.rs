//! Geometric predicates: orientation, in-circle, robust signs.

pub mod dot_cross;
pub mod in_circle;
pub mod orientation;
pub mod robust_signs;

pub use dot_cross::{cross2, dot2};
pub use in_circle::{in_circle, in_circle_signed};
pub use orientation::{Orientation, orient, orient_with_eps};
pub use robust_signs::{sign_strict, sign_with_eps};
