//! Graph separators.
//!
//! Balanced vertex separators for (planar) graphs: removing a small separator
//! set splits the graph into two balanced parts with no edges between them.

pub mod planar_separator;

pub use planar_separator::{PlanarSeparator, SeparatorResult};
