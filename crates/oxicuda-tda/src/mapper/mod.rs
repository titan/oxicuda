//! Mapper algorithm for topological data analysis.

pub mod ball_mapper;
pub mod mapper;

pub use ball_mapper::{BallMapper, BallMapperConfig, BallMapperResult, BallNode, LandmarkStrategy};
pub use mapper::{MapperConfig, MapperGraph, MapperNode, build_mapper};
