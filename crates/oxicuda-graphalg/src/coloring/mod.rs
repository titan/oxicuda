//! Graph coloring.

pub mod dsatur;
pub mod greedy_coloring;
pub mod welsh_powell;

pub use dsatur::dsatur_coloring;
pub use greedy_coloring::greedy_coloring;
pub use welsh_powell::welsh_powell_coloring;
