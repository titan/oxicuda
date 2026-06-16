//! Quality-Diversity (QD) algorithms.
//!
//! These algorithms aim not just to find a single optimum, but to map out a rich
//! diversity of high-performing solutions across a descriptor (behavior) space.
//!
//! ## Algorithms
//! - [`mod@map_elites`] — MAP-Elites (Mouret & Clune 2015): illuminates behavior spaces
//!   by maintaining a grid of elite solutions, one per behavior cell.
//!
//! ## Maximization note
//! Unlike the rest of the `oxicuda-evol` crate which **minimizes** fitness,
//! the QD algorithms in this module **maximize** fitness (higher is better).
//! This follows the original MAP-Elites framing where fitness reflects a solution's
//! quality and is to be maximized within each cell.

pub mod map_elites;
pub use map_elites::{Elite, MapElitesArchive, MapElitesConfig, map_elites};
