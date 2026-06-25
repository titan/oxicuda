//! Mesh and point-cloud distance metrics, plus classical computational- and
//! differential-geometry operators (Delaunay, ray/triangle, curvature).

pub mod barycentric;
pub mod chamfer_distance;
pub mod convex_hull;
pub mod curvature;
pub mod delaunay3d;
pub mod earth_movers;
pub mod marching_cubes;
pub mod normal_estimate;
pub mod obb;
pub mod plane_ransac;
pub mod ray_triangle;
pub mod simplify;
pub mod smoothing;
pub mod topology;

pub use barycentric::{
    barycentric_tetrahedron, barycentric_triangle, interpolate_triangle, point_in_tetrahedron,
    point_in_triangle,
};
pub use convex_hull::{ConvexHull3d, HullFace, convex_hull_2d, convex_hull_3d};
pub use curvature::{VertexCurvature, discrete_curvature, icosphere};
pub use delaunay3d::{Delaunay3d, in_sphere, orient3d, tetrahedralize};
pub use marching_cubes::{MarchingCubesConfig, MarchingCubesResult, marching_cubes};
pub use obb::{Aabb, Obb};
pub use plane_ransac::{Plane, PlaneFitResult, fit_plane_ransac};
pub use ray_triangle::{
    RayHit, closest_point_on_mesh, closest_point_on_triangle, ray_aabb_intersect,
    ray_mesh_intersect, ray_triangle_intersect,
};
pub use simplify::{SimplifyResult, simplify_mesh};
pub use smoothing::{laplacian_smooth, taubin_smooth};
pub use topology::{TopologyReport, analyze_topology, validate_mesh};
