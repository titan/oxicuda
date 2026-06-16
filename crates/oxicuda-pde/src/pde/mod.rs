//! Interface-tracking PDE methods.
//!
//! * [`level_set`] — level-set method for implicit interface tracking
//!   (upwind advection, Osher–Sethian normal motion, signed-distance
//!   reinitialisation).

pub mod level_set;

pub use level_set::LevelSet;
