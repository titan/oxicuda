//! Neural surface reconstruction.
//!
//! - `neuralangelo`: Neuralangelo numerical-gradient SDF with coarse-to-fine
//!   hash-grid level scheduling and curvature regularisation.
//! - `marching_cubes`: Lorensen–Cline marching cubes isosurface / mesh export.

pub mod marching_cubes;
pub mod neuralangelo;

pub use marching_cubes::{
    CORNER_OFFSETS, EDGE_CORNERS, EDGE_TABLE, GridSpec, TRI_TABLE, TriMesh, marching_cubes,
    polygonize_cube, vertex_interp,
};
pub use neuralangelo::{
    Neuralangelo, NeuralangeloConfig, eikonal_residual_of, laplacian_of, numerical_gradient_of,
};
