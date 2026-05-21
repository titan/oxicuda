//! KPConv — Kernel Point Convolution (Thomas et al., ICCV 2019).
//!
//! A continuous convolution over irregular point clouds using K learnable
//! kernel points arranged in 3D space. Each kernel point contributes to
//! the output via a linear correlation weight based on geometric distance.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for KPConv.
#[derive(Debug, Clone)]
pub struct KPConvConfig {
    /// Number of input feature channels.
    pub in_channels: usize,
    /// Number of output feature channels.
    pub out_channels: usize,
    /// Number of kernel points K. Default 15.
    pub n_kernel_points: usize,
    /// Neighborhood radius r: consider support points within this distance. > 0.
    pub radius: f32,
    /// Kernel correlation radius σ: influence width of each kernel point. > 0.
    /// Typically σ = radius * 2/3.
    pub sigma: f32,
}

// ─── KPConv ──────────────────────────────────────────────────────────────────

/// KPConv layer: continuous convolution for point clouds.
///
/// Uses a set of K learnable kernel points arranged in 3D space.
/// Each kernel point influences nearby support points via a linear correlation
/// function, yielding a continuous convolution over irregular point clouds.
pub struct KPConv {
    cfg: KPConvConfig,
    /// Kernel point positions: K × 3 (row-major, scaled to fit in ball of radius sigma).
    kernel_points: Vec<f32>,
    /// Weight matrices: K × in_channels × out_channels (row-major).
    weights: Vec<f32>,
}

impl KPConv {
    /// Create a new KPConv layer with Kaiming-uniform weight init and
    /// Fibonacci-sphere kernel point placement.
    pub fn new(cfg: KPConvConfig, rng: &mut LcgRng) -> Geom3dResult<Self> {
        if !cfg.radius.is_finite() || cfg.radius <= 0.0 {
            return Err(Geom3dError::InvalidRadius { radius: cfg.radius });
        }
        if !cfg.sigma.is_finite() || cfg.sigma <= 0.0 {
            return Err(Geom3dError::InvalidRadius { radius: cfg.sigma });
        }
        let k = cfg.n_kernel_points;
        let sigma = cfg.sigma;

        // ── Kernel point placement via Fibonacci spiral on sphere of radius σ ──
        let kernel_points = if k == 1 {
            vec![0.0_f32; 3]
        } else {
            let golden_ratio = (1.0_f32 + 5.0_f32.sqrt()) / 2.0;
            let mut kp = Vec::with_capacity(k * 3);
            for i in 0..k {
                let fi = i as f32;
                let fk = k as f32;
                let theta = (1.0 - 2.0 * (fi + 0.5) / fk).clamp(-1.0, 1.0).acos();
                let phi = 2.0 * std::f32::consts::PI * fi / golden_ratio;
                kp.push(theta.sin() * phi.cos() * sigma);
                kp.push(theta.sin() * phi.sin() * sigma);
                kp.push(theta.cos() * sigma);
            }
            kp
        };

        // ── Kaiming uniform weight init: U(-√(6/in_channels), +√(6/in_channels)) ──
        let n_weights = k * cfg.in_channels * cfg.out_channels;
        let limit = (6.0_f32 / cfg.in_channels as f32).sqrt();
        let mut weights = vec![0.0_f32; n_weights];
        for w in weights.iter_mut() {
            *w = (rng.next_f32() * 2.0 - 1.0) * limit;
        }

        Ok(Self {
            cfg,
            kernel_points,
            weights,
        })
    }

    /// Forward pass: compute output features for each center point.
    ///
    /// - `center_pts`: N center points, shape `[N, 3]` flat row-major.
    /// - `n_centers`: N.
    /// - `support_pts`: M support (neighbor) points, shape `[M, 3]`.
    /// - `n_support`: M.
    /// - `support_feats`: M × in_channels features of support points.
    /// - `neighbor_idx`: N × max_neighbors neighbor indices into support_pts.
    ///   Use value `i64::MAX` to indicate padding (no neighbor). Valid indices ∈ [0, M).
    /// - `max_neighbors`: number of columns in neighbor_idx.
    ///
    /// Returns N × out_channels output features.
    ///
    /// # Errors
    /// - `EmptyPointCloud` if N == 0 or M == 0.
    /// - `DimensionMismatch` if shapes don't match.
    /// - `NanEncountered` if any feature is non-finite.
    pub fn forward(
        &self,
        center_pts: &[f32],
        n_centers: usize,
        support_pts: &[f32],
        n_support: usize,
        support_feats: &[f32],
        neighbor_idx: &[i64],
        max_neighbors: usize,
    ) -> Geom3dResult<Vec<f32>> {
        // ── Validate inputs ──────────────────────────────────────────────────
        if n_centers == 0 || center_pts.is_empty() {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if n_support == 0 || support_pts.is_empty() {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if center_pts.len() != n_centers * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n_centers * 3,
                got: center_pts.len(),
            });
        }
        if support_pts.len() != n_support * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n_support * 3,
                got: support_pts.len(),
            });
        }
        if support_feats.len() != n_support * self.cfg.in_channels {
            return Err(Geom3dError::DimensionMismatch {
                expected: n_support * self.cfg.in_channels,
                got: support_feats.len(),
            });
        }
        if neighbor_idx.len() != n_centers * max_neighbors {
            return Err(Geom3dError::DimensionMismatch {
                expected: n_centers * max_neighbors,
                got: neighbor_idx.len(),
            });
        }
        // Check for NaN in support features
        for &v in support_feats {
            if !v.is_finite() {
                return Err(Geom3dError::NanEncountered {
                    location: "kpconv::forward::support_feats",
                });
            }
        }

        let k = self.cfg.n_kernel_points;
        let in_c = self.cfg.in_channels;
        let out_c = self.cfg.out_channels;
        let sigma = self.cfg.sigma;

        let mut output = vec![0.0_f32; n_centers * out_c];

        // ── Per-center forward pass ──────────────────────────────────────────
        for n in 0..n_centers {
            let pn_x = center_pts[n * 3];
            let pn_y = center_pts[n * 3 + 1];
            let pn_z = center_pts[n * 3 + 2];

            let out_slice = &mut output[n * out_c..(n + 1) * out_c];

            let row_start = n * max_neighbors;
            for nb in 0..max_neighbors {
                let j_raw = neighbor_idx[row_start + nb];
                if j_raw == i64::MAX {
                    continue;
                }
                let j = j_raw as usize;

                // Relative position r_j = p_j - p_n
                let rx = support_pts[j * 3] - pn_x;
                let ry = support_pts[j * 3 + 1] - pn_y;
                let rz = support_pts[j * 3 + 2] - pn_z;

                // Per kernel point contribution
                for ki in 0..k {
                    let kp_x = self.kernel_points[ki * 3];
                    let kp_y = self.kernel_points[ki * 3 + 1];
                    let kp_z = self.kernel_points[ki * 3 + 2];

                    let dx = rx - kp_x;
                    let dy = ry - kp_y;
                    let dz = rz - kp_z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    // Linear influence function: max(0, 1 - dist/sigma)
                    let h = (1.0 - dist / sigma).max(0.0);
                    if h <= 0.0 {
                        continue;
                    }

                    // W_k: [in_channels × out_channels] row-major
                    // w[ki * in_c * out_c + row * out_c + col]
                    // weighted_feat = h * (W_k^T · feat_j)
                    let w_base = ki * in_c * out_c;
                    let feat_base = j * in_c;

                    for (oc, out_val) in out_slice.iter_mut().enumerate() {
                        let mut acc = 0.0_f32;
                        for ic in 0..in_c {
                            // W_k[ic, oc] = weights[w_base + ic * out_c + oc]
                            acc += self.weights[w_base + ic * out_c + oc]
                                * support_feats[feat_base + ic];
                        }
                        *out_val += h * acc;
                    }
                }
            }
        }

        // ── Check output for NaN ─────────────────────────────────────────────
        for &v in output.iter() {
            if !v.is_finite() {
                return Err(Geom3dError::NanEncountered {
                    location: "kpconv::forward::output",
                });
            }
        }

        Ok(output)
    }

    /// Number of learnable parameters (kernel weights only; no bias).
    pub fn n_params(&self) -> usize {
        self.cfg.n_kernel_points * self.cfg.in_channels * self.cfg.out_channels
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kpconv(k: usize, in_c: usize, out_c: usize, r: f32) -> KPConv {
        let cfg = KPConvConfig {
            in_channels: in_c,
            out_channels: out_c,
            n_kernel_points: k,
            radius: r,
            sigma: r * 2.0 / 3.0,
        };
        let mut rng = LcgRng::new(42);
        KPConv::new(cfg, &mut rng).unwrap()
    }

    fn make_centers(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n * 3).map(|_| rng.next_f32() * 0.5).collect()
    }

    fn make_support(m: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..m * 3).map(|_| rng.next_f32() * 0.5).collect()
    }

    fn make_feats(m: usize, in_c: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..m * in_c).map(|_| rng.next_f32()).collect()
    }

    /// Build a dense neighbor_idx: every center gets all support as neighbors.
    fn dense_neighbors(n: usize, m: usize) -> Vec<i64> {
        let mut idx = vec![i64::MAX; n * m];
        for ni in 0..n {
            for mi in 0..m {
                idx[ni * m + mi] = mi as i64;
            }
        }
        idx
    }

    #[test]
    fn kpconv_output_shape() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 10;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = make_feats(m, 4, 3);
        let idx = dense_neighbors(n, m);
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        assert_eq!(out.len(), n * 8);
    }

    #[test]
    fn kpconv_output_finite() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 10;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = make_feats(m, 4, 3);
        let idx = dense_neighbors(n, m);
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "output must be finite");
    }

    #[test]
    fn kpconv_zero_features_zero_output() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 10;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = vec![0.0_f32; m * 4];
        let idx = dense_neighbors(n, m);
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "zero features must yield zero output"
        );
    }

    #[test]
    fn kpconv_kernel_points_on_sphere() {
        let k = 15;
        let sigma = 2.0_f32 / 3.0;
        let kp = make_kpconv(k, 4, 8, 1.0);
        for ki in 0..k {
            let x = kp.kernel_points[ki * 3];
            let y = kp.kernel_points[ki * 3 + 1];
            let z = kp.kernel_points[ki * 3 + 2];
            let norm = (x * x + y * y + z * z).sqrt();
            assert!(
                (norm - sigma).abs() < 1e-4,
                "kernel point {ki} norm={norm} expected {sigma}"
            );
        }
    }

    #[test]
    fn kpconv_kernel_points_count() {
        let k = 15;
        let kp = make_kpconv(k, 4, 8, 1.0);
        assert_eq!(kp.kernel_points.len(), k * 3);
    }

    #[test]
    fn kpconv_n_params() {
        let k = 15;
        let in_c = 4;
        let out_c = 8;
        let kp = make_kpconv(k, in_c, out_c, 1.0);
        assert_eq!(kp.n_params(), k * in_c * out_c);
    }

    #[test]
    fn kpconv_single_neighbor() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 3;
        let m = 5;
        let centers = vec![0.0_f32; n * 3];
        let support = vec![0.1_f32; m * 3]; // all near origin
        let feats = make_feats(m, 4, 10);

        // Each center has exactly 1 neighbor (index 0)
        let idx = vec![0_i64; n];
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, 1)
            .unwrap();
        // Non-zero features with non-zero influence -> output should not all be zero
        let all_zero = out.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "single non-zero neighbor should produce non-zero output"
        );
    }

    #[test]
    fn kpconv_no_neighbor() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 10;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = make_feats(m, 4, 3);
        let idx = vec![i64::MAX; n * m]; // all padding
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "all-padding neighbor idx must yield zero output"
        );
    }

    #[test]
    fn kpconv_influence_decreases_with_dist() {
        // K=1, kernel point at origin. Center at origin.
        // Neighbor A at (0,0,0) = center itself -> max influence.
        // Neighbor B at (sigma, sigma, sigma) = far from kernel point -> low influence.
        let sigma = 1.0_f32;
        let cfg = KPConvConfig {
            in_channels: 1,
            out_channels: 1,
            n_kernel_points: 1,
            radius: 1.5 * sigma,
            sigma,
        };
        let mut rng = LcgRng::new(0);
        let kp = KPConv::new(cfg, &mut rng).unwrap();

        let center = vec![0.0_f32, 0.0, 0.0];
        let support = vec![
            0.0_f32, 0.0, 0.0, // neighbor A: at center
            sigma, sigma, sigma, // neighbor B: far away
        ];
        let feats = vec![1.0_f32, 1.0]; // both feats = 1

        // Neighbor A: dist from kp (at origin) = 0 -> h = 1.0
        // Neighbor B: dist from kp = sqrt(3)*sigma > sigma -> h = 0
        let idx_a = vec![0_i64];
        let idx_b = vec![1_i64];

        let out_a = kp
            .forward(&center, 1, &support, 2, &feats, &idx_a, 1)
            .unwrap();
        let out_b = kp
            .forward(&center, 1, &support, 2, &feats, &idx_b, 1)
            .unwrap();

        // out_a should have larger magnitude than out_b
        assert!(
            out_a[0].abs() > out_b[0].abs(),
            "neighbor at center should have higher influence: out_a={} out_b={}",
            out_a[0],
            out_b[0]
        );
    }

    #[test]
    fn kpconv_k1_kernel_at_origin() {
        let sigma = 1.0_f32;
        let cfg = KPConvConfig {
            in_channels: 1,
            out_channels: 1,
            n_kernel_points: 1,
            radius: 2.0,
            sigma,
        };
        let mut rng = LcgRng::new(7);
        let kp = KPConv::new(cfg, &mut rng).unwrap();
        // K=1 -> kernel_points = [0,0,0]
        assert_eq!(kp.kernel_points.len(), 3);
        assert_eq!(kp.kernel_points[0], 0.0);
        assert_eq!(kp.kernel_points[1], 0.0);
        assert_eq!(kp.kernel_points[2], 0.0);

        // Neighbor at center p_n = p_j -> r_j = [0,0,0], dist to kp = 0 -> h = 1.0
        let center = vec![1.0_f32, 0.0, 0.0];
        let support = vec![1.0_f32, 0.0, 0.0]; // same as center
        let feats = vec![1.0_f32];
        let idx = vec![0_i64];
        let out = kp
            .forward(&center, 1, &support, 1, &feats, &idx, 1)
            .unwrap();
        // h=1.0, feat=1.0, weight=w -> out = w
        let expected = kp.weights[0]; // only one weight
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "k=1 at center: out={} expected={}",
            out[0],
            expected
        );
    }

    #[test]
    fn kpconv_radius_effect() {
        // Neighbor placed so it is far from ALL kernel points on the sphere (h=0 for all k).
        // Place neighbor extremely far so dist > sigma for all kernel points.
        let sigma = 1.0_f32;
        let cfg = KPConvConfig {
            in_channels: 2,
            out_channels: 3,
            n_kernel_points: 15,
            radius: 2.0,
            sigma,
        };
        let mut rng = LcgRng::new(5);
        let kp = KPConv::new(cfg, &mut rng).unwrap();

        let center = vec![0.0_f32, 0.0, 0.0];
        // Place neighbor at (100, 100, 100) so r_j = (100,100,100) and dist from any kp
        // on unit sphere >= 100*sqrt(3) - 1 >> sigma -> h=0 for all kernel points
        let support = vec![100.0_f32, 100.0, 100.0];
        let feats = vec![1.0_f32, 1.0];
        let idx = vec![0_i64];
        let out = kp
            .forward(&center, 1, &support, 1, &feats, &idx, 1)
            .unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "neighbor far from all kernel points must contribute 0"
        );
    }

    #[test]
    fn kpconv_err_empty_centers() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let result = kp.forward(&[], 0, &[0.0; 30], 10, &[0.0; 40], &[], 0);
        assert_eq!(result, Err(Geom3dError::EmptyPointCloud));
    }

    #[test]
    fn kpconv_err_empty_support() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let centers = make_centers(5, 1);
        let result = kp.forward(&centers, 5, &[], 0, &[], &[], 0);
        assert_eq!(result, Err(Geom3dError::EmptyPointCloud));
    }

    #[test]
    fn kpconv_err_dim_mismatch_feats() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 10;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        // Wrong feature length: should be m * in_c = 40
        let feats = vec![0.0_f32; 30];
        let idx = dense_neighbors(n, m);
        let result = kp.forward(&centers, n, &support, m, &feats, &idx, m);
        assert!(
            matches!(result, Err(Geom3dError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    #[test]
    fn kpconv_err_dim_mismatch_neighbor() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 10;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = make_feats(m, 4, 3);
        // Wrong neighbor_idx length: n * max_neighbors = 5*10=50 but we pass 30
        let idx = vec![0_i64; 30];
        let result = kp.forward(&centers, n, &support, m, &feats, &idx, 10);
        assert!(
            matches!(result, Err(Geom3dError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    #[test]
    fn kpconv_small_config() {
        let cfg = KPConvConfig {
            in_channels: 2,
            out_channels: 4,
            n_kernel_points: 5,
            radius: 1.0,
            sigma: 0.667,
        };
        let mut rng = LcgRng::new(0);
        let kp = KPConv::new(cfg, &mut rng).unwrap();
        let n = 3;
        let m = 5;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = make_feats(m, 2, 3);
        let idx = dense_neighbors(n, m);
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        assert_eq!(out.len(), n * 4);
    }

    #[test]
    fn kpconv_large_neighborhood() {
        let kp = make_kpconv(15, 4, 8, 1.0);
        let n = 5;
        let m = 30;
        let max_nb = 20;
        let centers = make_centers(n, 1);
        let support = make_support(m, 2);
        let feats = make_feats(m, 4, 3);
        // 20 neighbors per center, indices into m=30 points
        let mut idx = vec![i64::MAX; n * max_nb];
        let mut rng = LcgRng::new(99);
        for ni in 0..n {
            for nb in 0..max_nb {
                idx[ni * max_nb + nb] = rng.next_usize(m) as i64;
            }
        }
        let out = kp
            .forward(&centers, n, &support, m, &feats, &idx, max_nb)
            .unwrap();
        assert_eq!(out.len(), n * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kpconv_deterministic() {
        let cfg = KPConvConfig {
            in_channels: 4,
            out_channels: 8,
            n_kernel_points: 15,
            radius: 1.0,
            sigma: 2.0 / 3.0,
        };
        let mut rng1 = LcgRng::new(77);
        let kp = KPConv::new(cfg, &mut rng1).unwrap();

        let n = 5;
        let m = 10;
        let centers = make_centers(n, 11);
        let support = make_support(m, 22);
        let feats = make_feats(m, 4, 33);
        let idx = dense_neighbors(n, m);

        let out1 = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        let out2 = kp
            .forward(&centers, n, &support, m, &feats, &idx, m)
            .unwrap();
        assert_eq!(out1, out2, "forward must be deterministic");
    }
}
