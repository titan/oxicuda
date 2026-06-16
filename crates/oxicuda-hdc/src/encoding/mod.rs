//! Encoding modules: record-based, n-gram, spatial pattern, continuous-value (level /
//! thermometer), and graph-structure encoding.

pub mod graph_hd;
pub mod level;
pub mod ngram;
pub mod pattern;
pub mod record;
pub mod sequence_hd;
pub mod spatial_hd;

pub use graph_hd::GraphHdEncoder;
pub use level::{LevelEncoder, thermometer_encode};
pub use sequence_hd::SequenceHdEncoder;
pub use spatial_hd::SpatialHdEncoder;
