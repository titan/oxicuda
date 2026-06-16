//! `oxicuda-geometry3d` — 3D geometry, point-cloud, mesh, and Gaussian-splatting
//! primitives for OxiCUDA.
//!
//! Pure-Rust implementation of 3D geometry operations including FPS, kNN,
//! KD-tree, PointNet/PointNet++/DGCNN/Point-Transformer, sparse 3D conv,
//! Chamfer/EMD distance, ICP, and 3D Gaussian splatting.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-geometry3d
//! ├── sampling/      — FPS, random sampling, voxel downsampling
//! ├── neighborhood/  — kNN, ball query, KD-tree
//! ├── pointops/      — gather, group, interpolate features
//! ├── arch/          — PointNet, PointNet++, DGCNN, Point Transformer, EGNN
//! ├── voxel/         — voxel grid scatter, sparse 3D conv
//! ├── mesh/          — Chamfer, EMD, normal estimation
//! ├── gaussian/      — 3DGS primitives, projection, 2D/3D rasterization
//! ├── transform/     — rigid body, quaternion, ICP
//! ├── io/            — ASCII PLY / PCD point-cloud readers (pure std)
//! ├── error          — Geom3dError / Geom3dResult
//! ├── handle         — Geom3dHandle (SmVersion + LcgRng)
//! └── ptx_kernels    — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

#![forbid(unsafe_code)]

pub mod arch;
pub mod error;
pub mod gaussian;
pub mod handle;
pub mod io;
pub mod mesh;
pub mod neighborhood;
pub mod pointops;
pub mod ptx_kernels;
pub mod sampling;
pub mod transform;
pub mod voxel;

/// Convenience re-exports for common geometry types.
pub mod prelude {
    pub use crate::arch::dgcnn::{EdgeConv, EdgeConvConfig};
    pub use crate::arch::egnn::{Egnn, EgnnConfig, EgnnLayer};
    pub use crate::arch::point_transformer::{PointTransformerConfig, PointTransformerLayer};
    pub use crate::arch::pointnet::{PointNet, PointNetConfig};
    pub use crate::arch::pointnet_pp::{FeaturePropagation, SetAbstraction, SetAbstractionConfig};
    pub use crate::error::{Geom3dError, Geom3dResult};
    pub use crate::gaussian::gaussian::Gaussian3d;
    pub use crate::gaussian::gaussian_2d::{Gaussian2d, rasterize_gaussians_2d};
    pub use crate::gaussian::project::{CameraIntrinsics, ProjectedGaussian, project_gaussian};
    pub use crate::gaussian::rasterize::{RasterConfig, rasterize_gaussians};
    pub use crate::handle::{Geom3dHandle, LcgRng, SmVersion};
    pub use crate::io::{PointCloud, parse_pcd_str, parse_ply_str, read_pcd, read_ply};
    pub use crate::mesh::barycentric::{
        barycentric_tetrahedron, barycentric_triangle, interpolate_triangle, point_in_tetrahedron,
        point_in_triangle,
    };
    pub use crate::mesh::chamfer_distance::{chamfer_distance, chamfer_distance_grad};
    pub use crate::mesh::convex_hull::{ConvexHull3d, HullFace, convex_hull_2d, convex_hull_3d};
    pub use crate::mesh::curvature::{VertexCurvature, discrete_curvature, icosphere};
    pub use crate::mesh::delaunay3d::{Delaunay3d, in_sphere, orient3d, tetrahedralize};
    pub use crate::mesh::earth_movers::{SinkhornConfig, earth_movers_distance};
    pub use crate::mesh::marching_cubes::{
        MarchingCubesConfig, MarchingCubesResult, marching_cubes,
    };
    pub use crate::mesh::normal_estimate::estimate_normals;
    pub use crate::mesh::obb::{Aabb, Obb};
    pub use crate::mesh::plane_ransac::{Plane, PlaneFitResult, fit_plane_ransac};
    pub use crate::mesh::ray_triangle::{
        RayHit, closest_point_on_mesh, closest_point_on_triangle, ray_aabb_intersect,
        ray_mesh_intersect, ray_triangle_intersect,
    };
    pub use crate::mesh::simplify::{SimplifyResult, simplify_mesh};
    pub use crate::mesh::smoothing::{laplacian_smooth, taubin_smooth};
    pub use crate::neighborhood::ball_query::ball_query;
    pub use crate::neighborhood::grid_knn::{GridKnnConfig, SpatialHashGrid};
    pub use crate::neighborhood::kd_tree::KdTree;
    pub use crate::neighborhood::knn::knn;
    pub use crate::pointops::gather_points::gather_points;
    pub use crate::pointops::group_features::group_features;
    pub use crate::pointops::interp_features::interp_features;
    pub use crate::ptx_kernels::*;
    pub use crate::sampling::farthest_point_sample::farthest_point_sample;
    pub use crate::sampling::pointnext_aug::{PointNextAug, PointNextAugConfig};
    pub use crate::sampling::random_sample::random_sample;
    pub use crate::sampling::voxel_downsample::voxel_downsample;
    pub use crate::transform::icp::{IcpConfig, IcpResult, icp};
    pub use crate::transform::quaternion::Quat;
    pub use crate::transform::range_image::{RangeImage, RangeImageConfig, RangeImageProjector};
    pub use crate::transform::rigid::RigidTransform;
    pub use crate::voxel::octree::{Octree, OctreeConfig, OctreeNode};
    pub use crate::voxel::sparse_conv3d::{SparseConv3d, SparseConv3dConfig, SparseTensor};
    pub use crate::voxel::voxelize::{VoxelGrid, VoxelPoolMode};
}

#[cfg(test)]
mod e2e_tests {
    use super::prelude::*;

    fn make_points_rng(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut pts = vec![0.0_f32; n * 3];
        for v in &mut pts {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        pts
    }

    #[test]
    fn e2e_fps_selects_m_distinct_points() {
        let pts = make_points_rng(64, 1);
        let selected =
            farthest_point_sample(&pts, 64, 16).expect("farthest_point_sample should succeed");
        assert_eq!(selected.len(), 16);
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 16, "FPS must select distinct indices");
    }

    #[test]
    fn e2e_pointnet_forward_valid_logits() {
        let cfg = PointNetConfig {
            n_points: 16,
            n_classes: 5,
        };
        let mut rng = LcgRng::new(42);
        let net = PointNet::new(cfg, &mut rng);
        let pts = make_points_rng(16, 2);
        let logits = net.forward(&pts).expect("forward should succeed");
        assert_eq!(logits.len(), 5);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_set_abstraction_reduces_points() {
        let n = 32;
        let npoint = 8;
        let mut rng = LcgRng::new(42);
        let xyz = make_points_rng(n, 3);
        let feat = vec![1.0_f32; n * 4];
        let cfg = SetAbstractionConfig {
            npoint,
            radius: 0.8,
            nsample: 8,
            mlp_channels: vec![8, 16],
        };
        let sa = SetAbstraction::new(cfg, 4, &mut rng);
        let (out_xyz, out_feat) = sa
            .forward(&xyz, n, &feat, 4)
            .expect("forward should succeed");
        assert_eq!(out_xyz.len(), npoint * 3);
        assert_eq!(out_feat.len(), npoint * 16);
    }

    #[test]
    fn e2e_dgcnn_output_shape() {
        let n = 8;
        let c_in = 3;
        let c_out = 16;
        let mut rng = LcgRng::new(42);
        let cfg = EdgeConvConfig {
            k: 3,
            mlp_channels: vec![8, c_out],
        };
        let ec = EdgeConv::new(cfg, c_in, &mut rng);
        let feat = make_points_rng(n, 10);
        let out = ec.forward(&feat, n, c_in).expect("forward should succeed");
        assert_eq!(out.len(), n * c_out);
    }

    #[test]
    fn e2e_chamfer_self_distance_zero() {
        let pts: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let cd = chamfer_distance(&pts, 10, &pts, 10).expect("chamfer_distance should succeed");
        assert!(cd.abs() < 1e-5, "CD(A,A) must be 0, got {cd}");
    }

    #[test]
    fn e2e_icp_identity_convergence() {
        let pts: Vec<f32> = (0..27)
            .flat_map(|i| {
                vec![
                    (i % 3) as f32 * 0.1,
                    ((i / 3) % 3) as f32 * 0.1,
                    (i / 9) as f32 * 0.1,
                ]
            })
            .collect();
        let cfg = IcpConfig {
            max_iter: 20,
            tol: 1e-5,
        };
        let result = icp(&pts, 27, &pts, 27, &cfg).expect("icp should succeed");
        assert!(
            result.residual < 1e-3,
            "ICP on identity should have near-zero residual"
        );
    }

    #[test]
    fn e2e_voxelize_roundtrip() {
        let pts = vec![0.5_f32, 0.5, 0.5, 1.5_f32, 0.5, 0.5, 2.5_f32, 0.5, 0.5];
        let feat = vec![1.0_f32, 2.0, 3.0];
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 1.0, [4, 4, 4], 1);
        grid.scatter(&pts, 3, &feat, VoxelPoolMode::Sum)
            .expect("scatter should succeed");
        let (coords, feats) = grid
            .occupied_centroids()
            .expect("occupied_centroids should succeed");
        assert_eq!(coords.len(), 9); // 3 voxels × 3 coords
        assert_eq!(feats.len(), 3); // 3 voxels × 1 channel
    }

    #[test]
    fn e2e_gaussian_project_valid_depth() {
        let g = Gaussian3d::new_unit([0.0, 0.0, 5.0]);
        let view = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let cam = CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            near: 0.1,
        };
        let pg = project_gaussian(&g, &view, &cam).expect("project_gaussian should succeed");
        assert!(pg.valid, "Gaussian at z=5 should be valid");
        assert!((pg.depth - 5.0).abs() < 1e-3, "Depth should be 5.0");
    }

    #[test]
    fn e2e_kdtree_nearest_correctness() {
        let pts: Vec<f32> = (0..20).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let tree = KdTree::build(&pts, 20).expect("build should succeed");
        let q = [9.9_f32, 0.0, 0.0];
        let (idx, _) = tree.nearest(q).expect("nearest should succeed");
        assert_eq!(idx, 10, "Nearest to 9.9 should be 10");
    }

    #[test]
    fn e2e_knn_vs_brute_force() {
        let pts: Vec<f32> = (0..20).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let q = vec![7.3_f32, 0.0, 0.0];
        let (idx, dists) = knn(&q, 1, &pts, 20, 3).expect("knn should succeed");
        // Nearest 3 to 7.3 should be 7, 8, and 6 or 9
        assert!(dists[0] <= dists[1] && dists[1] <= dists[2]);
        assert!(idx[0] == 7 || idx[0] == 8);
    }

    #[test]
    fn e2e_lcg_rng_determinism() {
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("fps_kernel", farthest_point_sample_ptx),
            ("ball_query_kernel", ball_query_ptx),
            ("gather_kernel", gather_points_ptx),
            ("voxelize_kernel", voxelize_ptx),
            ("chamfer_kernel", chamfer_distance_ptx),
            ("project_kernel", gaussian_project_ptx),
            ("sh_eval_kernel", sh_eval_ptx),
        ];
        for sm in sm_versions {
            for (kernel_name, gen_fn) in kernel_fns {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} sm={sm} missing sm target"
                );
                assert!(ptx.contains(".version"), "PTX missing .version");
                assert!(
                    ptx.contains(kernel_name),
                    "PTX missing kernel name {kernel_name}"
                );
            }
        }
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn e2e_icosphere_curvature_matches_unit_sphere() {
        // Build a unit icosphere mesh and verify discrete curvature recovers
        // the analytic sphere values (K ≈ 1, H ≈ 1).
        let (verts, tris) = icosphere(3);
        let n = verts.len() / 3;
        let curv = discrete_curvature(&verts, n, &tris).expect("discrete_curvature should succeed");
        let mean_k: f64 = curv.gaussian.iter().sum::<f64>() / n as f64;
        let mean_h: f64 = curv.mean.iter().sum::<f64>() / n as f64;
        assert!((mean_k - 1.0).abs() < 0.12, "unit-sphere K≈1, got {mean_k}");
        assert!((mean_h - 1.0).abs() < 0.12, "unit-sphere H≈1, got {mean_h}");
    }

    #[test]
    fn e2e_delaunay_hull_faces_ray_intersect() {
        // Delaunay-tetrahedralize a cube, take its convex-hull faces as a
        // mesh, and shoot a ray through the center to confirm a hit.
        let mut pts = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    pts.push(x as f64);
                    pts.push(y as f64);
                    pts.push(z as f64);
                }
            }
        }
        let delaunay = tetrahedralize(&pts, 8).expect("tetrahedralize should succeed");
        assert!((delaunay.total_volume() - 1.0).abs() < 1e-9);
        let faces = delaunay.convex_hull_faces();
        assert!(!faces.is_empty(), "cube hull must have faces");
        let verts: Vec<f64> = pts.clone();
        // Ray straight down through the cube center hits the top face.
        let hit = ray_mesh_intersect(&[0.5, 0.5, 5.0], &[0.0, 0.0, -1.0], &verts, &faces);
        assert!(
            hit.is_some(),
            "ray through cube center must hit a hull face"
        );
        let (_, rayhit) = hit.expect("hit should be present");
        // First surface crossing is the z=1 top face at t = 4.
        assert!(
            (rayhit.t - 4.0).abs() < 1e-6,
            "expected t≈4, got {}",
            rayhit.t
        );
    }

    #[test]
    fn e2e_marching_cubes_mesh_closest_point() {
        // Extract a sphere isosurface via marching cubes, convert its flat
        // triangle buffer to index triples, and query the closest surface
        // point — it should be within the grid's bounding box.
        let (nx, ny, nz) = (16usize, 16, 16);
        let mut sdf = vec![0.0_f32; nx * ny * nz];
        let center = [7.5_f32, 7.5, 7.5];
        let radius = 4.0_f32;
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let dx = x as f32 - center[0];
                    let dy = y as f32 - center[1];
                    let dz = z as f32 - center[2];
                    sdf[x * ny * nz + y * nz + z] = (dx * dx + dy * dy + dz * dz).sqrt() - radius;
                }
            }
        }
        let cfg = MarchingCubesConfig {
            nx,
            ny,
            nz,
            dx: 1.0,
            dy: 1.0,
            dz: 1.0,
            origin: [0.0, 0.0, 0.0],
            isovalue: 0.0,
        };
        let mesh = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert!(mesh.n_triangles > 0);
        let verts64: Vec<f64> = mesh.vertices.iter().map(|&v| v as f64).collect();
        let tris: Vec<[usize; 3]> = mesh
            .triangles
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        // Query point at the sphere center: nearest surface point should be
        // ~radius away.
        let (idx, _, sq) = closest_point_on_mesh(
            &[center[0] as f64, center[1] as f64, center[2] as f64],
            &verts64,
            &tris,
        );
        assert_ne!(idx, usize::MAX);
        let dist = sq.sqrt();
        assert!(
            (dist - radius as f64).abs() < 1.0,
            "closest surface point ≈ radius {radius}, got {dist}"
        );
    }
}
