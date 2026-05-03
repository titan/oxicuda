//! Multi-head attention with relative-position bias.

pub mod rel_pos_attention;
pub mod rel_pos_encoding;

pub use rel_pos_attention::RelPosAttention;
pub use rel_pos_encoding::RelPosEncoding;
