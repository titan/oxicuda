//! PointNet++ Set Abstraction and Feature Propagation modules.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;
use crate::neighborhood::stochastic_ball::stochastic_ball_query;
use crate::pointops::gather_points::gather_points;
use crate::pointops::group_features::group_features;
use crate::sampling::farthest_point_sample::farthest_point_sample;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn shared_mlp_layer(
    input: &[f32],
    n: usize,
    in_dim: usize,
    out_dim: usize,
    w: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * out_dim];
    for p in 0..n {
        for i in 0..out_dim {
            let mut acc = b[i];
            for j in 0..in_dim {
                acc += w[i * in_dim + j] * input[p * in_dim + j];
            }
            out[p * out_dim + i] = acc.max(0.0);
        }
    }
    out
}

fn group_max_pool(group_feat: &[f32], s: usize, c: usize) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; c];
    for si in 0..s {
        for ch in 0..c {
            let v = group_feat[si * c + ch];
            if v > out[ch] {
                out[ch] = v;
            }
        }
    }
    for v in &mut out {
        if *v == f32::NEG_INFINITY {
            *v = 0.0;
        }
    }
    out
}

fn sq_dist_3d(a: &[f32], ai: usize, b: &[f32], bi: usize) -> f32 {
    let dx = a[ai * 3] - b[bi * 3];
    let dy = a[ai * 3 + 1] - b[bi * 3 + 1];
    let dz = a[ai * 3 + 2] - b[bi * 3 + 2];
    dx * dx + dy * dy + dz * dz
}

// ─── Set Abstraction ─────────────────────────────────────────────────────────

/// Configuration for a Set Abstraction layer.
#[derive(Debug, Clone)]
pub struct SetAbstractionConfig {
    pub npoint: usize,
    pub radius: f32,
    pub nsample: usize,
    pub mlp_channels: Vec<usize>,
}

/// Set Abstraction layer (PointNet++ MSG).
pub struct SetAbstraction {
    config: SetAbstractionConfig,
    mlp_weights: Vec<Vec<f32>>,
    mlp_biases: Vec<Vec<f32>>,
}

impl SetAbstraction {
    /// Create a new Set Abstraction layer.
    pub fn new(config: SetAbstractionConfig, in_channels: usize, rng: &mut LcgRng) -> Self {
        let mut mlp_weights = Vec::new();
        let mut mlp_biases = Vec::new();

        let mut prev = in_channels + 3; // concatenate relative xyz
        for &out_ch in &config.mlp_channels {
            let mut w = vec![0.0_f32; out_ch * prev];
            rng.fill_xavier_uniform(&mut w, prev, out_ch);
            mlp_weights.push(w);
            mlp_biases.push(vec![0.0_f32; out_ch]);
            prev = out_ch;
        }

        Self {
            config,
            mlp_weights,
            mlp_biases,
        }
    }

    /// Forward pass.
    ///
    /// Input: `xyz [n×3]`, `feat [n×c_in]`.
    /// Output: `xyz' [npoint×3]`, `feat' [npoint×c_out]`.
    pub fn forward(
        &self,
        xyz: &[f32],
        n: usize,
        feat: &[f32],
        c_in: usize,
    ) -> Geom3dResult<(Vec<f32>, Vec<f32>)> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if xyz.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: xyz.len(),
            });
        }
        if feat.len() != n * c_in {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * c_in,
                got: feat.len(),
            });
        }

        let npoint = self.config.npoint.min(n);
        let seed_indices = farthest_point_sample(xyz, n, npoint)?;

        let c_out = self.config.mlp_channels.last().copied().unwrap_or(c_in);

        let mut out_xyz = vec![0.0_f32; npoint * 3];
        let mut out_feat = vec![0.0_f32; npoint * c_out];

        for (si, &seed_idx) in seed_indices.iter().enumerate() {
            // Store seed xyz
            out_xyz[si * 3..si * 3 + 3].copy_from_slice(&xyz[seed_idx * 3..seed_idx * 3 + 3]);

            let sx = xyz[seed_idx * 3];
            let sy = xyz[seed_idx * 3 + 1];
            let sz = xyz[seed_idx * 3 + 2];

            // Ball query: find up to nsample neighbors
            let r_sq = self.config.radius * self.config.radius;
            let mut neighbors: Vec<usize> = (0..n)
                .filter(|&pi| sq_dist_3d(xyz, pi, xyz, seed_idx) < r_sq)
                .take(self.config.nsample)
                .collect();

            if neighbors.is_empty() {
                neighbors.push(seed_idx);
            }

            let s = neighbors.len();
            let in_dim = c_in + 3;

            // Build grouped input: relative xyz + features
            let mut group_in = vec![0.0_f32; s * in_dim];
            for (j, &ni) in neighbors.iter().enumerate() {
                group_in[j * in_dim] = xyz[ni * 3] - sx;
                group_in[j * in_dim + 1] = xyz[ni * 3 + 1] - sy;
                group_in[j * in_dim + 2] = xyz[ni * 3 + 2] - sz;
                if c_in > 0 {
                    group_in[j * in_dim + 3..j * in_dim + in_dim]
                        .copy_from_slice(&feat[ni * c_in..(ni + 1) * c_in]);
                }
            }

            // Apply MLP layers
            let mut h = group_in;
            let mut cur_in = in_dim;
            for (layer_idx, &out_ch) in self.config.mlp_channels.iter().enumerate() {
                h = shared_mlp_layer(
                    &h,
                    s,
                    cur_in,
                    out_ch,
                    &self.mlp_weights[layer_idx],
                    &self.mlp_biases[layer_idx],
                );
                cur_in = out_ch;
            }

            // Max pool over neighbors
            let pooled = group_max_pool(&h, s, c_out);
            out_feat[si * c_out..(si + 1) * c_out].copy_from_slice(&pooled);
        }

        Ok((out_xyz, out_feat))
    }
}

// ─── Fused FPS → Ball-query → Gather ──────────────────────────────────────────

/// Parameters for the fused [`fps_ball_gather`] grouping primitive.
#[derive(Debug, Clone)]
pub struct FpsBallGatherConfig {
    /// Number of farthest-point-sampled centroids (`npoint`).
    pub npoint: usize,
    /// Neighbours gathered per centroid (`nsample`); the stochastic ball query
    /// cycles (pads) or subsamples-without-replacement to exactly this width.
    pub nsample: usize,
    /// Ball-query radius around each centroid.
    pub radius: f32,
    /// Deterministic seed for the stochastic in-radius subsample.
    pub seed: u64,
}

/// Output of [`fps_ball_gather`].
#[derive(Debug, Clone)]
pub struct FpsBallGather {
    /// FPS centroid indices into the original cloud, length `npoint`.
    pub centroids: Vec<usize>,
    /// Ball-query neighbour indices, row-major `[npoint × nsample]`. The
    /// sentinel `usize::MAX` marks empty slots for centroids with no in-radius
    /// point.
    pub neighbor_indices: Vec<usize>,
    /// Number of distinct in-radius neighbours per centroid (before padding),
    /// length `npoint`.
    pub counts: Vec<usize>,
    /// Grouped neighbour features, row-major `[npoint × nsample × c]`. Sentinel
    /// (`usize::MAX`) slots are left as zeros.
    pub grouped: Vec<f32>,
}

/// Fused PointNet++ grouping: FPS centroids → ball-query neighbourhoods →
/// gather grouped features, in a single call.
///
/// This *composes* three existing primitives rather than re-deriving their
/// maths:
///
/// 1. [`farthest_point_sample`] selects `config.npoint` centroid indices.
/// 2. [`stochastic_ball_query`] finds (and seed-subsamples) up to
///    `config.nsample` in-radius neighbours per centroid.
/// 3. [`group_features`] gathers the neighbour features into a contiguous
///    `[npoint × nsample × c]` tensor.
///
/// The centroid coordinates handed to the ball query are themselves produced by
/// reusing [`gather_points`] on the FPS indices, so no coordinate slicing is
/// duplicated either. The result is identical to running those stages
/// separately (locked in by the `fused_equals_unfused_pipeline` test) but is
/// exposed as one reusable Set-Abstraction grouping op.
///
/// # Errors
///
/// Propagates any error from the composed stages: an empty cloud or `npoint == 0`
/// ([`Geom3dError::EmptyPointCloud`]), a mis-sized `xyz`/`feat` buffer
/// ([`Geom3dError::DimensionMismatch`]), `npoint > n`
/// ([`Geom3dError::InvalidSampleCount`]), `nsample == 0`
/// ([`Geom3dError::InvalidK`]), or a non-positive / non-finite `radius`
/// ([`Geom3dError::InvalidRadius`]).
pub fn fps_ball_gather(
    xyz: &[f32],
    n: usize,
    feat: &[f32],
    c: usize,
    config: &FpsBallGatherConfig,
) -> Geom3dResult<FpsBallGather> {
    // Stage 1 — farthest-point-sample the centroid indices.
    let centroids = farthest_point_sample(xyz, n, config.npoint)?;
    let npoint = centroids.len();

    // Stage 2 — gather the centroid coordinates [npoint × 3] (reuse the indexed
    // gather instead of re-slicing xyz by hand).
    let centroid_xyz = gather_points(xyz, n, 3, &centroids)?;

    // Stage 3 — stochastic in-radius neighbourhoods [npoint × nsample].
    let (neighbor_indices, counts) = stochastic_ball_query(
        &centroid_xyz,
        npoint,
        xyz,
        n,
        config.nsample,
        config.radius,
        config.seed,
    )?;

    // Stage 4 — gather the neighbour features [npoint × nsample × c].
    let grouped = group_features(feat, n, c, &neighbor_indices, npoint, config.nsample)?;

    Ok(FpsBallGather {
        centroids,
        neighbor_indices,
        counts,
        grouped,
    })
}

// ─── Feature Propagation ─────────────────────────────────────────────────────

/// Feature Propagation layer (PointNet++ upsampling).
pub struct FeaturePropagation {
    mlp_weights: Vec<Vec<f32>>,
    mlp_biases: Vec<Vec<f32>>,
    mlp_channels: Vec<usize>,
}

impl FeaturePropagation {
    /// Create a new Feature Propagation layer.
    pub fn new(in_channels: usize, mlp_channels: Vec<usize>, rng: &mut LcgRng) -> Self {
        let mut mlp_weights = Vec::new();
        let mut mlp_biases = Vec::new();

        let mut prev = in_channels;
        for &out_ch in &mlp_channels {
            let mut w = vec![0.0_f32; out_ch * prev];
            rng.fill_xavier_uniform(&mut w, prev, out_ch);
            mlp_weights.push(w);
            mlp_biases.push(vec![0.0_f32; out_ch]);
            prev = out_ch;
        }

        Self {
            mlp_weights,
            mlp_biases,
            mlp_channels,
        }
    }

    /// Upsample: xyz1 is sparse [n1×3], feat1 [n1×c1]; xyz2 is dense [n2×3], feat2 [n2×c2].
    /// Returns interpolated [n2×c_out].
    pub fn forward(
        &self,
        xyz1: &[f32],
        n1: usize,
        feat1: &[f32],
        c1: usize,
        xyz2: &[f32],
        n2: usize,
        feat2: &[f32],
        c2: usize,
    ) -> Geom3dResult<Vec<f32>> {
        if n1 == 0 || n2 == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }

        // 3-NN interpolation from feat1 onto xyz2 positions
        let k = 3.min(n1);
        let c_interp = c1;
        let mut interp_feat = vec![0.0_f32; n2 * c_interp];

        for ti in 0..n2 {
            let tx = xyz2[ti * 3];
            let ty = xyz2[ti * 3 + 1];
            let tz = xyz2[ti * 3 + 2];

            let mut dists: Vec<(f32, usize)> = (0..n1)
                .map(|si| {
                    let dx = xyz1[si * 3] - tx;
                    let dy = xyz1[si * 3 + 1] - ty;
                    let dz = xyz1[si * 3 + 2] - tz;
                    (dx * dx + dy * dy + dz * dz, si)
                })
                .collect();

            dists.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            let k_nn = &dists[..k];
            let weights: Vec<f32> = k_nn.iter().map(|&(d, _)| 1.0 / (d + 1e-10)).collect();
            let w_sum: f32 = weights.iter().sum();

            for ch in 0..c_interp {
                let mut val = 0.0_f32;
                for (j, &(_, si)) in k_nn.iter().enumerate() {
                    val += (weights[j] / w_sum) * feat1[si * c1 + ch];
                }
                interp_feat[ti * c_interp + ch] = val;
            }
        }

        // Concat with feat2
        let c_concat = c_interp + c2;
        let mut concat = vec![0.0_f32; n2 * c_concat];
        for ti in 0..n2 {
            concat[ti * c_concat..ti * c_concat + c_interp]
                .copy_from_slice(&interp_feat[ti * c_interp..(ti + 1) * c_interp]);
            if c2 > 0 {
                concat[ti * c_concat + c_interp..ti * c_concat + c_concat]
                    .copy_from_slice(&feat2[ti * c2..(ti + 1) * c2]);
            }
        }

        // Apply MLP
        let mut h = concat;
        let mut cur_in = c_concat;
        for (layer_idx, &out_ch) in self.mlp_channels.iter().enumerate() {
            h = shared_mlp_layer(
                &h,
                n2,
                cur_in,
                out_ch,
                &self.mlp_weights[layer_idx],
                &self.mlp_biases[layer_idx],
            );
            cur_in = out_ch;
        }

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xyz(n: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut pts = vec![0.0_f32; n * 3];
        for v in &mut pts {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        pts
    }

    #[test]
    fn set_abstraction_reduces_points() {
        let n = 32;
        let npoint = 8;
        let mut rng = LcgRng::new(42);
        let xyz = make_xyz(n, &mut rng);
        let feat: Vec<f32> = vec![1.0; n * 4];
        let cfg = SetAbstractionConfig {
            npoint,
            radius: 0.5,
            nsample: 8,
            mlp_channels: vec![8, 16],
        };
        let sa = SetAbstraction::new(cfg, 4, &mut rng);
        let (out_xyz, out_feat) = sa
            .forward(&xyz, n, &feat, 4)
            .expect("forward should succeed");
        assert_eq!(out_xyz.len(), npoint * 3);
        assert_eq!(out_feat.len(), npoint * 16);
        assert!(out_feat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn set_abstraction_empty_error() {
        let mut rng = LcgRng::new(0);
        let cfg = SetAbstractionConfig {
            npoint: 4,
            radius: 0.5,
            nsample: 4,
            mlp_channels: vec![8],
        };
        let sa = SetAbstraction::new(cfg, 3, &mut rng);
        assert!(sa.forward(&[], 0, &[], 3).is_err());
    }

    #[test]
    fn feature_propagation_upsamples() {
        let n1 = 4;
        let n2 = 16;
        let c1 = 8;
        let c2 = 4;
        let mut rng = LcgRng::new(42);
        let xyz1 = make_xyz(n1, &mut rng);
        let xyz2 = make_xyz(n2, &mut rng);
        let feat1: Vec<f32> = vec![1.0; n1 * c1];
        let feat2: Vec<f32> = vec![0.5; n2 * c2];
        let fp = FeaturePropagation::new(c1 + c2, vec![16, 8], &mut rng);
        let out = fp
            .forward(&xyz1, n1, &feat1, c1, &xyz2, n2, &feat2, c2)
            .expect("value should be present");
        assert_eq!(out.len(), n2 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod fps_ball_gather_tests {
    use super::*;

    /// Deterministic cloud: a 1-D line of `n` points on the x-axis.
    fn line_cloud(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| [i as f32, 0.0, 0.0]).collect()
    }

    /// Per-point features derived from the point index so a gather is easy to
    /// verify by hand: channel `ch` of point `i` is `i * 10 + ch`.
    fn ramp_feat(n: usize, c: usize) -> Vec<f32> {
        let mut f = vec![0.0_f32; n * c];
        for i in 0..n {
            for ch in 0..c {
                f[i * c + ch] = (i * 10 + ch) as f32;
            }
        }
        f
    }

    #[test]
    fn fused_output_shapes_correct() {
        let (n, c) = (16, 4);
        let xyz = line_cloud(n);
        let feat = ramp_feat(n, c);
        let cfg = FpsBallGatherConfig {
            npoint: 5,
            nsample: 3,
            radius: 4.0,
            seed: 7,
        };
        let out = fps_ball_gather(&xyz, n, &feat, c, &cfg).expect("fused grouping should succeed");
        assert_eq!(out.centroids.len(), cfg.npoint);
        assert_eq!(out.counts.len(), cfg.npoint);
        assert_eq!(out.neighbor_indices.len(), cfg.npoint * cfg.nsample);
        assert_eq!(out.grouped.len(), cfg.npoint * cfg.nsample * c);
    }

    #[test]
    fn fused_equals_unfused_pipeline() {
        // The fused helper must equal the composition of the three existing
        // primitives run separately, elementwise, on a deterministic cloud.
        let (n, c) = (20, 5);
        let xyz = line_cloud(n);
        let feat = ramp_feat(n, c);
        let cfg = FpsBallGatherConfig {
            npoint: 6,
            nsample: 4,
            radius: 3.5,
            seed: 123,
        };

        let fused = fps_ball_gather(&xyz, n, &feat, c, &cfg).expect("fused should succeed");

        // Unfused reference pipeline.
        let centroids = farthest_point_sample(&xyz, n, cfg.npoint).expect("fps should succeed");
        let centroid_xyz =
            gather_points(&xyz, n, 3, &centroids).expect("centroid gather should succeed");
        let (nbr_idx, counts) = stochastic_ball_query(
            &centroid_xyz,
            centroids.len(),
            &xyz,
            n,
            cfg.nsample,
            cfg.radius,
            cfg.seed,
        )
        .expect("ball query should succeed");
        let grouped = group_features(&feat, n, c, &nbr_idx, centroids.len(), cfg.nsample)
            .expect("group should succeed");

        assert_eq!(fused.centroids, centroids);
        assert_eq!(fused.neighbor_indices, nbr_idx);
        assert_eq!(fused.counts, counts);
        assert_eq!(
            fused.grouped, grouped,
            "fused tensor must match the unfused pipeline elementwise"
        );
    }

    #[test]
    fn grouped_features_match_raw_gather() {
        // Independent correctness check: every grouped row equals the raw
        // feature row at its neighbour index (or zeros for a sentinel slot).
        let (n, c) = (24, 3);
        let xyz = line_cloud(n);
        let feat = ramp_feat(n, c);
        let cfg = FpsBallGatherConfig {
            npoint: 5,
            nsample: 4,
            radius: 5.0,
            seed: 42,
        };
        let out = fps_ball_gather(&xyz, n, &feat, c, &cfg).expect("fused should succeed");

        for slot in 0..cfg.npoint * cfg.nsample {
            let ni = out.neighbor_indices[slot];
            let row = &out.grouped[slot * c..(slot + 1) * c];
            if ni == usize::MAX {
                assert!(row.iter().all(|&v| v == 0.0), "sentinel slot must be zero");
            } else {
                assert_eq!(
                    row,
                    &feat[ni * c..(ni + 1) * c],
                    "gathered row must equal the source feature row"
                );
            }
        }
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let (n, c) = (30, 2);
        let xyz = line_cloud(n);
        let feat = ramp_feat(n, c);
        let cfg = FpsBallGatherConfig {
            npoint: 8,
            nsample: 5,
            radius: 6.0,
            seed: 99,
        };
        let a = fps_ball_gather(&xyz, n, &feat, c, &cfg).expect("first call should succeed");
        let b = fps_ball_gather(&xyz, n, &feat, c, &cfg).expect("second call should succeed");
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.neighbor_indices, b.neighbor_indices);
        assert_eq!(a.counts, b.counts);
        assert_eq!(a.grouped, b.grouped);
    }

    #[test]
    fn different_seed_changes_subsample() {
        // With many more in-radius candidates than `nsample`, the stochastic
        // subsample is active and the seed must influence which neighbours win.
        let (n, c) = (30, 1);
        let xyz = line_cloud(n);
        let feat = ramp_feat(n, c);
        let cfg_a = FpsBallGatherConfig {
            npoint: 4,
            nsample: 4,
            radius: 12.0,
            seed: 1,
        };
        let cfg_b = FpsBallGatherConfig {
            seed: 2,
            ..cfg_a.clone()
        };
        let a = fps_ball_gather(&xyz, n, &feat, c, &cfg_a).expect("call a should succeed");
        let b = fps_ball_gather(&xyz, n, &feat, c, &cfg_b).expect("call b should succeed");
        // FPS is seed-independent, so the centroids must match …
        assert_eq!(a.centroids, b.centroids);
        // … but the seeded neighbour draw must differ.
        assert_ne!(
            a.neighbor_indices, b.neighbor_indices,
            "a different seed must change the subsample"
        );
    }

    #[test]
    fn errors_propagate_from_stages() {
        let (n, c) = (8, 2);
        let xyz = line_cloud(n);
        let feat = ramp_feat(n, c);
        // npoint > n → FPS InvalidSampleCount.
        assert!(
            fps_ball_gather(
                &xyz,
                n,
                &feat,
                c,
                &FpsBallGatherConfig {
                    npoint: n + 1,
                    nsample: 3,
                    radius: 1.0,
                    seed: 0,
                },
            )
            .is_err()
        );
        // nsample == 0 → ball-query InvalidK.
        assert!(
            fps_ball_gather(
                &xyz,
                n,
                &feat,
                c,
                &FpsBallGatherConfig {
                    npoint: 4,
                    nsample: 0,
                    radius: 1.0,
                    seed: 0,
                },
            )
            .is_err()
        );
        // radius <= 0 → ball-query InvalidRadius.
        assert!(
            fps_ball_gather(
                &xyz,
                n,
                &feat,
                c,
                &FpsBallGatherConfig {
                    npoint: 4,
                    nsample: 3,
                    radius: 0.0,
                    seed: 0,
                },
            )
            .is_err()
        );
        // Mis-sized feature buffer → group_features DimensionMismatch.
        let bad_feat = vec![0.0_f32; n * c - 1];
        assert!(
            fps_ball_gather(
                &xyz,
                n,
                &bad_feat,
                c,
                &FpsBallGatherConfig {
                    npoint: 4,
                    nsample: 3,
                    radius: 1.0,
                    seed: 0,
                },
            )
            .is_err()
        );
    }
}
