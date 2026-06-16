//! Mapper algorithm for topological data analysis.

pub mod ball_mapper;
pub mod mapper;
pub mod multiscale;
pub mod stable;

pub use ball_mapper::{BallMapper, BallMapperConfig, BallMapperResult, BallNode, LandmarkStrategy};
pub use mapper::{MapperConfig, MapperGraph, MapperNode, build_mapper};
pub use multiscale::{
    MultiscaleMapperConfig, MultiscaleMapperResult, ScaleLevel, multiscale_mapper,
};
pub use stable::{StableMapperConfig, StableMapperResult, stable_mapper};
