//! Polygon and line clipping algorithms.

pub mod greiner_hormann;
pub mod liang_barsky;
pub mod line_clip_cohen_sutherland;
pub mod sutherland_hodgman;
pub mod weiler_atherton;

pub use greiner_hormann::{
    BooleanOp, clip_polygons, difference, filled_area_of_rings, intersection, signed_area_of_rings,
    union, xor,
};
pub use liang_barsky::liang_barsky;
pub use line_clip_cohen_sutherland::cohen_sutherland;
pub use sutherland_hodgman::sutherland_hodgman;
pub use weiler_atherton::weiler_atherton;
