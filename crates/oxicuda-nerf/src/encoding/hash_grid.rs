//! Instant-NGP multi-resolution hash grid encoding.
//!
//! L levels, T buckets per level, F features per entry.
//! Resolution at level l: `N_l = floor(N_min * b^l)` where `b = exp(ln(N_max/N_min)/(L-1))`.
//! Hash: `h(x1,x2,x3) = (x1 XOR x2*pi2 XOR x3*pi3) % T`
//! with pi1=1, pi2=2654435761, pi3=805459861.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

const PI2: u64 = 2_654_435_761;
const PI3: u64 = 805_459_861;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the multi-resolution hash grid.
#[derive(Debug, Clone)]
pub struct HashGridConfig {
    /// L — number of resolution levels (typical: 16).
    pub n_levels: usize,
    /// F — number of features per hash-table entry (typical: 2).
    pub n_features_per_level: usize,
    /// log2(T) where T = 2^this is the number of hash buckets (typical: 19 → T=524288).
    pub log2_hashmap_size: usize,
    /// N_min — base (coarsest) grid resolution (typical: 16).
    pub base_resolution: usize,
    /// N_max — finest grid resolution (typical: 2048).
    pub max_resolution: usize,
}

// ─── HashGrid ────────────────────────────────────────────────────────────────

/// Multi-resolution hash grid with trilinear interpolation.
#[derive(Debug, Clone)]
pub struct HashGrid {
    /// Grid configuration.
    pub config: HashGridConfig,
    /// Flat feature storage: `[n_levels * T * F]`.
    pub data: Vec<f32>,
    /// Per-level grid resolution N_l.
    level_resolutions: Vec<usize>,
}

impl HashGrid {
    /// Create a new hash grid with parameters initialized to U(-0.0001, 0.0001).
    ///
    /// # Errors
    ///
    /// Returns `InvalidHashConfig` for invalid configuration.
    pub fn new(cfg: HashGridConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.n_levels == 0 {
            return Err(NerfError::InvalidHashConfig {
                msg: "n_levels must be > 0".into(),
            });
        }
        if cfg.n_features_per_level == 0 {
            return Err(NerfError::InvalidHashConfig {
                msg: "n_features_per_level must be > 0".into(),
            });
        }
        if cfg.log2_hashmap_size == 0 || cfg.log2_hashmap_size > 32 {
            return Err(NerfError::InvalidHashConfig {
                msg: "log2_hashmap_size must be in 1..=32".into(),
            });
        }
        if cfg.base_resolution == 0 {
            return Err(NerfError::InvalidHashConfig {
                msg: "base_resolution must be > 0".into(),
            });
        }
        if cfg.max_resolution < cfg.base_resolution {
            return Err(NerfError::InvalidHashConfig {
                msg: "max_resolution must be >= base_resolution".into(),
            });
        }

        let t = 1_usize << cfg.log2_hashmap_size;

        // Compute per-level resolutions
        let level_resolutions = if cfg.n_levels == 1 {
            vec![cfg.base_resolution]
        } else {
            let b = ((cfg.max_resolution as f64) / (cfg.base_resolution as f64)).ln()
                / (cfg.n_levels - 1) as f64;
            (0..cfg.n_levels)
                .map(|l| {
                    let n_l = (cfg.base_resolution as f64 * (b * l as f64).exp()).floor() as usize;
                    n_l.max(1)
                })
                .collect()
        };

        let total = cfg.n_levels * t * cfg.n_features_per_level;
        let mut data = vec![0.0_f32; total];
        for v in data.iter_mut() {
            *v = rng.next_f32_range(-0.0001, 0.0001);
        }

        Ok(Self {
            config: cfg,
            data,
            level_resolutions,
        })
    }

    /// Total output dimension: `n_levels * n_features_per_level`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.config.n_levels * self.config.n_features_per_level
    }

    /// Query a single 3D point in `[0, 1]^3`.
    ///
    /// Returns a feature vector of length `output_dim`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` for wrong input size.
    pub fn query(&self, xyz: [f32; 3]) -> NerfResult<Vec<f32>> {
        let t = 1_usize << self.config.log2_hashmap_size;
        let f = self.config.n_features_per_level;
        let mut out = vec![0.0_f32; self.output_dim()];

        for (level, &n_l) in self.level_resolutions.iter().enumerate() {
            // Scale xyz to level resolution [0, N_l]
            let sx = xyz[0].clamp(0.0, 1.0) * (n_l as f32);
            let sy = xyz[1].clamp(0.0, 1.0) * (n_l as f32);
            let sz = xyz[2].clamp(0.0, 1.0) * (n_l as f32);

            let ix = sx.floor() as i64;
            let iy = sy.floor() as i64;
            let iz = sz.floor() as i64;
            let fx = sx - ix as f32;
            let fy = sy - iy as f32;
            let fz = sz - iz as f32;

            let level_offset = level * t * f;

            // Trilinear interpolation over 8 corners
            for cx in 0_u8..=1 {
                for cy in 0_u8..=1 {
                    for cz in 0_u8..=1 {
                        let xi = ix + i64::from(cx);
                        let yi = iy + i64::from(cy);
                        let zi = iz + i64::from(cz);

                        let bucket = hash_coord(xi, yi, zi, t);
                        let w = trilinear_weight(fx, fy, fz, cx, cy, cz);
                        let base = level_offset + bucket * f;

                        let out_base = level * f;
                        for feat in 0..f {
                            out[out_base + feat] += w * self.data[base + feat];
                        }
                    }
                }
            }
        }

        Ok(out)
    }

    /// Batch query: `xyz_batch` is a flat `[N * 3]` array.
    ///
    /// Returns `[N * output_dim]`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `xyz_batch.len() != n * 3`.
    pub fn query_batch(&self, xyz_batch: &[f32], n: usize) -> NerfResult<Vec<f32>> {
        if xyz_batch.len() != n * 3 {
            return Err(NerfError::DimensionMismatch {
                expected: n * 3,
                got: xyz_batch.len(),
            });
        }
        let out_dim = self.output_dim();
        let mut out = vec![0.0_f32; n * out_dim];

        for (i, out_chunk) in out.chunks_mut(out_dim).enumerate() {
            let x = xyz_batch[i * 3];
            let y = xyz_batch[i * 3 + 1];
            let z = xyz_batch[i * 3 + 2];
            let feat = self.query([x, y, z])?;
            out_chunk.copy_from_slice(&feat);
        }

        Ok(out)
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Hash a grid cell coordinate to a bucket index in `[0, t)`.
#[inline]
fn hash_coord(xi: i64, yi: i64, zi: i64, t: usize) -> usize {
    let hx = xi as u64;
    let hy = (yi as u64).wrapping_mul(PI2);
    let hz = (zi as u64).wrapping_mul(PI3);
    (hx ^ hy ^ hz) as usize % t
}

/// Trilinear interpolation weight for corner (cx, cy, cz) given fractional (fx, fy, fz).
#[inline]
fn trilinear_weight(fx: f32, fy: f32, fz: f32, cx: u8, cy: u8, cz: u8) -> f32 {
    let wx = if cx == 1 { fx } else { 1.0 - fx };
    let wy = if cy == 1 { fy } else { 1.0 - fy };
    let wz = if cz == 1 { fz } else { 1.0 - fz };
    wx * wy * wz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(seed: u64) -> HashGrid {
        let cfg = HashGridConfig {
            n_levels: 4,
            n_features_per_level: 2,
            log2_hashmap_size: 8,
            base_resolution: 4,
            max_resolution: 32,
        };
        let mut rng = LcgRng::new(seed);
        HashGrid::new(cfg, &mut rng).unwrap()
    }

    #[test]
    fn query_output_shape() {
        let grid = make_grid(1);
        let feat = grid.query([0.5, 0.5, 0.5]).unwrap();
        assert_eq!(feat.len(), grid.output_dim());
    }

    #[test]
    fn batch_output_shape() {
        let grid = make_grid(2);
        let pts: Vec<f32> = (0..5).flat_map(|i| [i as f32 * 0.2; 3]).collect();
        let out = grid.query_batch(&pts, 5).unwrap();
        assert_eq!(out.len(), 5 * grid.output_dim());
    }

    #[test]
    fn hash_coord_deterministic() {
        assert_eq!(hash_coord(1, 2, 3, 256), hash_coord(1, 2, 3, 256));
    }

    #[test]
    fn trilinear_weights_sum_to_one() {
        let (fx, fy, fz) = (0.3, 0.7, 0.1);
        let mut sum = 0.0_f32;
        for cx in 0_u8..=1 {
            for cy in 0_u8..=1 {
                for cz in 0_u8..=1 {
                    sum += trilinear_weight(fx, fy, fz, cx, cy, cz);
                }
            }
        }
        assert!((sum - 1.0).abs() < 1e-6, "weights sum={sum}");
    }
}
