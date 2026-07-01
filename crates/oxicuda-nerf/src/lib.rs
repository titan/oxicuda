//! `oxicuda-nerf` — Neural Radiance Fields and neural rendering primitives for OxiCUDA.
//!
//! Pure-Rust implementation of canonical NeRF algorithms with GPU PTX kernel generation.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-nerf
//! ├── camera/       — Pinhole camera model and ray generation
//! ├── encoding/     — Positional encoding, Instant-NGP hash grid (+ trainable
//! │                   backward), Mip-NeRF IPE, spherical harmonics
//! ├── error         — NerfError / NerfResult
//! ├── field/        — TensoRF, Instant-NGP hash field, K-Planes, Plenoxel, VM
//! ├── generative/   — pi-GAN FiLM-SIREN generative radiance field
//! ├── handle        — NerfHandle (SmVersion + LcgRng)
//! ├── metrics/      — PSNR, MSE, SSIM, LPIPS perceptual metric
//! ├── network/      — NeRF MLP, TinyNeRF, NeRF-W, HumanNeRF / InstantAvatar
//! ├── ptx_kernels   — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ├── rendering/    — Ray, sampling, volume rendering, Zip-NeRF, Ref-NeRF,
//! │                   EmerNeRF, 3DGS (+ deformable), Block-NeRF, NSVF octree
//! └── surface/      — Neuralangelo neural SDF + marching-cubes mesh export
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod camera;
pub mod encoding;
pub mod error;
pub mod field;
pub mod generative;
pub mod handle;
pub mod metrics;
pub mod network;
pub mod ptx_kernels;
pub mod rendering;
pub mod surface;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common NeRF types and functions.
pub mod prelude {
    pub use crate::camera::pinhole::PinholeCamera;
    pub use crate::encoding::hash_grid::{HashGrid, HashGridConfig};
    pub use crate::encoding::hash_grid_grad::{GridCache, TrainableHashGrid};
    pub use crate::encoding::integrated_pe::{IpeConfig, integrated_pe};
    pub use crate::encoding::positional::{PosEncConfig, positional_encode};
    pub use crate::error::{NerfError, NerfResult};
    pub use crate::field::hash_field::HashField;
    pub use crate::field::kplanes::{KPlanes, KPlanesConfig};
    pub use crate::field::plenoxel::{PlenoxelConfig, PlenoxelGrid};
    pub use crate::field::tensorf::{TensorRf, TensorRfConfig};
    pub use crate::field::vm_field::{VmField, VmFieldConfig};
    pub use crate::generative::pi_gan::{FilmParams, PiGan, PiGanConfig};
    pub use crate::handle::{LcgRng, NerfHandle, SmVersion};
    pub use crate::metrics::image_quality::{ImageMetrics, compute_image_metrics, psnr};
    pub use crate::metrics::lpips::{Lpips, LpipsConfig};
    pub use crate::metrics::ssim::{SsimConfig, ssim_gray, ssim_image};
    pub use crate::network::human_nerf::{HumanNerf, HumanNerfConfig, NO_PARENT, Rigid, Skeleton};
    pub use crate::network::nerf_mlp::{NerfMlp, NerfMlpConfig};
    pub use crate::network::tiny_nerf::TinyNerf;
    pub use crate::ptx_kernels::{
        f32_hex, hash_grid_lookup_ptx, importance_resample_ptx, occupancy_update_ptx,
        positional_encoding_ptx, ray_march_ptx, sh_to_rgb_ptx, volume_render_ptx,
    };
    pub use crate::rendering::aabb::{Aabb as RayAabb, AabbHit};
    pub use crate::rendering::block_nerf::{Block, BlockNerfConfig, BlockNerfScene};
    pub use crate::rendering::deformable_3dgs::{
        DeformableGaussians, DeformationConfig, DeformationField, GaussianDelta,
    };
    pub use crate::rendering::emernerf::{
        Composite, DynamicOutput, EmerNerf, EmerNerfConfig, warp,
    };
    pub use crate::rendering::gaussian_splat_3d::{
        Gaussian3d, Splat2d, SplatCamera, SplatImage, SplatPixelGrad, backward_pixel,
        project_gaussian, project_scene, quat_to_rotation, rasterize, rasterize_pixel,
    };
    pub use crate::rendering::ndc::{ndc_depth_to_world, ndc_ray};
    pub use crate::rendering::occupancy::OccupancyGrid;
    pub use crate::rendering::ray::{PinholeCamera as RayCamera, Ray};
    pub use crate::rendering::ref_nerf::{
        RefNerf, RefNerfConfig, SpatialOutputs, attenuation_per_degree, ide_encode, reflect,
    };
    pub use crate::rendering::sampling::{importance_sample, merge_samples, stratified_sample};
    pub use crate::rendering::volume_render::{RenderResult, volume_render, volume_render_batch};
    pub use crate::rendering::zip_nerf::{Multisample, ZipNerf, ZipNerfConfig};
    pub use crate::surface::marching_cubes::{
        GridSpec, TriMesh, marching_cubes, polygonize_cube, vertex_interp,
    };
    pub use crate::surface::neuralangelo::{
        Neuralangelo, NeuralangeloConfig, eikonal_residual_of, laplacian_of, numerical_gradient_of,
    };
}

// ─── On-device GPU PTX validation ────────────────────────────────────────────

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    // ── Test 1: Positional encoding shape ────────────────────────────────────

    #[test]
    fn e2e_positional_encoding_shape() {
        let cfg = PosEncConfig {
            n_freq: 10,
            include_input: true,
            input_dim: 3,
        };
        let n_pts = 16;
        let input = vec![0.5_f32; n_pts * 3];
        let out = positional_encode(&input, &cfg).expect("positional_encode should succeed");
        assert_eq!(
            out.len(),
            n_pts * cfg.output_dim(),
            "E2E: positional encoding output shape mismatch"
        );
    }

    // ── Test 2: Positional encoding determinism ───────────────────────────────

    #[test]
    fn e2e_positional_encoding_deterministic() {
        let cfg = PosEncConfig {
            n_freq: 4,
            include_input: false,
            input_dim: 3,
        };
        let input = vec![0.1_f32, 0.5, -0.3, 0.0, 1.0, 0.7];
        let out1 = positional_encode(&input, &cfg).expect("positional_encode should succeed");
        let out2 = positional_encode(&input, &cfg).expect("positional_encode should succeed");
        assert_eq!(out1, out2, "E2E: positional encoding must be deterministic");
    }

    // ── Test 3: Hash grid query shape ─────────────────────────────────────────

    #[test]
    fn e2e_hash_grid_query_shape() {
        let cfg = HashGridConfig {
            n_levels: 8,
            n_features_per_level: 2,
            log2_hashmap_size: 10,
            base_resolution: 8,
            max_resolution: 256,
        };
        let mut rng = LcgRng::new(42);
        let grid = HashGrid::new(cfg, &mut rng).expect("new should succeed");
        let feat = grid.query([0.3, 0.7, 0.5]).expect("query should succeed");
        assert_eq!(
            feat.len(),
            grid.output_dim(),
            "E2E: hash grid output dim should be n_levels * n_feat"
        );
        assert_eq!(grid.output_dim(), 16);
    }

    // ── Test 4: Hash grid corner values differ ────────────────────────────────

    #[test]
    fn e2e_hash_grid_trilinear_corner() {
        let cfg = HashGridConfig {
            n_levels: 4,
            n_features_per_level: 2,
            log2_hashmap_size: 8,
            base_resolution: 4,
            max_resolution: 32,
        };
        let mut rng = LcgRng::new(1234);
        let grid = HashGrid::new(cfg, &mut rng).expect("new should succeed");
        let feat_origin = grid.query([0.0, 0.0, 0.0]).expect("query should succeed");
        let feat_far = grid.query([1.0, 1.0, 1.0]).expect("query should succeed");
        // With random initialization, corners almost certainly differ
        let are_different = feat_origin
            .iter()
            .zip(feat_far.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(
            are_different,
            "E2E: corner queries should return different values"
        );
    }

    // ── Test 5: Volume render empty scene ─────────────────────────────────────

    #[test]
    fn e2e_volume_render_empty_scene() {
        let n = 64;
        let sigma = vec![0.0_f32; n];
        let color = vec![0.5_f32; n * 3];
        let t: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 * 0.1).collect();
        let res = volume_render(&sigma, &color, &t).expect("volume_render should succeed");
        assert!(
            res.opacity < 1e-6,
            "E2E: empty scene (zero density) should have near-zero opacity, got {}",
            res.opacity
        );
    }

    // ── Test 6: Volume render opaque first sample ─────────────────────────────

    #[test]
    fn e2e_volume_render_opaque_first_sample() {
        let n = 16;
        let mut sigma = vec![0.0_f32; n];
        sigma[0] = 1e8_f32; // Extremely dense first sample
        let mut color = vec![0.0_f32; n * 3];
        color[0] = 1.0; // First sample: red
        color[1] = 0.0;
        color[2] = 0.0;
        let t: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 * 0.2).collect();
        let res = volume_render(&sigma, &color, &t).expect("volume_render should succeed");
        assert!(
            res.rgb[0] > 0.99,
            "E2E: opaque red first sample, expected R≈1, got {}",
            res.rgb[0]
        );
        assert!(
            res.opacity > 0.99,
            "E2E: opaque first sample should have opacity≈1, got {}",
            res.opacity
        );
    }

    // ── Test 7: Stratified sampling count ─────────────────────────────────────

    #[test]
    fn e2e_stratified_sampling_count() {
        let mut rng = LcgRng::new(99);
        let t_near = 0.1_f32;
        let t_far = 5.0_f32;
        let n = 128;
        let samples = stratified_sample(t_near, t_far, n, &mut rng)
            .expect("stratified_sample should succeed");
        assert_eq!(
            samples.len(),
            n,
            "E2E: stratified_sample must return exactly n_samples"
        );
        for &t in &samples {
            assert!(
                t >= t_near && t <= t_far,
                "E2E: sample {t} out of bounds [{t_near}, {t_far}]"
            );
        }
    }

    // ── Test 8: Importance sampling count ────────────────────────────────────

    #[test]
    fn e2e_importance_sampling_count() {
        let mut rng = LcgRng::new(77);
        let coarse_t = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let weights = vec![0.01, 0.1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.02];
        let n_fine = 32;
        let fine = importance_sample(&coarse_t, &weights, n_fine, &mut rng)
            .expect("importance_sample should succeed");
        assert_eq!(
            fine.len(),
            n_fine,
            "E2E: importance_sample must return n_fine samples"
        );
    }

    // ── Test 9: TensoRF density non-negative ──────────────────────────────────

    #[test]
    fn e2e_tensorf_density_nonneg() {
        let cfg = TensorRfConfig {
            rank: 8,
            grid_dim: 16,
            n_color_feat: 3,
        };
        let mut rng = LcgRng::new(2024);
        let tf = TensorRf::new(cfg, &mut rng).expect("new should succeed");

        let test_pts: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.3, -0.7, 0.9],
        ];
        for &xyz in test_pts {
            let d = tf.query_density(xyz).expect("query_density should succeed");
            assert!(
                d >= 0.0,
                "E2E: TensoRF density should be >= 0 (got {d}) at {:?}",
                xyz
            );
        }
    }

    // ── Test 10: TinyNerf forward finite ─────────────────────────────────────

    #[test]
    fn e2e_tiny_nerf_forward_finite() {
        let mut rng = LcgRng::new(314);
        let net = TinyNerf::new(24, 64, &mut rng);
        let x = vec![0.1_f32; 24];
        let (sigma, rgb) = net.forward(&x).expect("forward should succeed");
        assert!(
            sigma.is_finite(),
            "E2E: TinyNerf sigma must be finite, got {sigma}"
        );
        assert!(sigma >= 0.0, "E2E: TinyNerf sigma must be >= 0");
        for (i, &c) in rgb.iter().enumerate() {
            assert!(
                c.is_finite(),
                "E2E: TinyNerf RGB[{i}] must be finite, got {c}"
            );
            assert!(
                (0.0..=1.0).contains(&c),
                "E2E: TinyNerf RGB[{i}]={c} must be in [0, 1]"
            );
        }
    }

    // ── Test 11: PSNR on identical images ────────────────────────────────────

    #[test]
    fn e2e_psnr_identity() {
        let img = vec![0.5_f32; 256 * 256 * 3];
        let p = psnr(&img, &img).expect("psnr should succeed");
        assert!(
            p.is_infinite() || p > 100.0,
            "E2E: psnr(x, x) should be Inf or very large, got {p}"
        );
    }

    // ── Test 12: All 7 PTX kernels × 6 SM versions ───────────────────────────

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("pe_kernel", positional_encoding_ptx),
            ("volume_render_kernel", volume_render_ptx),
            ("hash_grid_kernel", hash_grid_lookup_ptx),
            ("ray_march_kernel", ray_march_ptx),
            ("sh_eval_nerf_kernel", sh_to_rgb_ptx),
            ("occupancy_update_kernel", occupancy_update_ptx),
            ("importance_resample_kernel", importance_resample_ptx),
        ];
        for sm in sm_versions {
            for (kernel_name, gen_fn) in kernel_fns {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} sm={sm} missing sm target"
                );
                assert!(
                    ptx.contains(".version"),
                    "PTX for {kernel_name} sm={sm} missing .version"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "PTX for {kernel_name} sm={sm} missing .visible .entry"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "PTX for {kernel_name} sm={sm} missing kernel name"
                );
            }
        }
        // Smoke-test f32_hex
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
