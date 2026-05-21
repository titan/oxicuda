//! Geometric transforms: rigid body, quaternions, and ICP.

pub mod icp;
pub mod quaternion;
pub mod range_image;
pub mod rigid;

pub use range_image::{RangeImage, RangeImageConfig, RangeImageProjector};
