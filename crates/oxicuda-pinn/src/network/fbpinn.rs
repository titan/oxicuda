//! Finite Basis Physics-Informed Neural Networks (FBPINNs).
//!
//! Moseley, Markham & Nissen-Meyer (2023) "Finite Basis Physics-Informed Neural
//! Networks (FBPINNs): a scalable domain decomposition approach for solving
//! differential equations", Advances in Computational Mathematics.
//!
//! FBPINNs decompose the domain into overlapping subdomains, each equipped with a
//! small neural network that operates on locally-normalised inputs. Smooth window
//! functions form a **partition of unity** so that the global solution is a
//! weighted sum of the local networks:
//!
//! ```text
//! u(x) = Σ_j ω̂_j(x) · u_j( normalize_local_j(x) ) ,      Σ_j ω̂_j(x) = 1 .
//! ```
//!
//! The window functions are separable products of 1-D Hann (raised-cosine) windows.
//! Because each subdomain's support is enlarged by a fractional `overlap`, adjacent
//! windows overlap and the normalised windows `ω̂_j = ω_j / Σ_k ω_k` partition unity
//! everywhere inside the domain. Local input normalisation keeps each subdomain MLP
//! operating on `~[-1, 1]` inputs, which (per the paper) is key to the method's
//! scalability and conditioning.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;
use crate::network::mlp::{Activation, Mlp, MlpConfig};
use core::cmp::Ordering;

/// A single rectangular subdomain described by its centre and per-dimension half-width.
#[derive(Debug, Clone)]
pub struct Subdomain {
    /// Centre coordinate of the subdomain, length `dim`.
    pub center: Vec<f32>,
    /// Half-width in each dimension (window support radius), length `dim`.
    pub half_width: Vec<f32>,
}

/// Configuration for an FBPINN domain decomposition.
#[derive(Debug, Clone)]
pub struct FbpinnConfig {
    /// Spatial dimension of the domain.
    pub dim: usize,
    /// Number of subdomains along each dimension, length `dim`.
    pub n_subdomains_per_dim: Vec<usize>,
    /// Lower corner of the global domain, length `dim`.
    pub domain_lo: Vec<f32>,
    /// Upper corner of the global domain, length `dim`.
    pub domain_hi: Vec<f32>,
    /// Fractional overlap of adjacent windows (e.g. `0.25`).
    pub overlap: f32,
    /// Hidden width of each subdomain MLP.
    pub hidden: usize,
}

/// Finite Basis PINN: overlapping subdomains, per-subdomain MLPs, partition of unity.
pub struct Fbpinn {
    /// The subdomains in row-major order over the per-dimension grid indices.
    pub subdomains: Vec<Subdomain>,
    /// One MLP per subdomain (`dim → hidden → 1`, `tanh` activation).
    pub networks: Vec<Mlp>,
    /// Spatial dimension.
    pub dim: usize,
}

/// 1-D Hann window centred at `c` with half-width `h`.
///
/// `ω(t) = ½(1 + cos(π (t − c)/h)) = cos²((π/2)(t − c)/h)` for `|t − c| ≤ h`, else `0`.
/// It is `C¹`-continuous, equals `1` at the centre, and `0` at `|t − c| = h`.
#[inline]
fn hann_1d(t: f32, c: f32, h: f32) -> f32 {
    if h <= 0.0 {
        return 0.0;
    }
    let r = (t - c) / h;
    if r.abs() > 1.0 {
        0.0
    } else {
        0.5 * (1.0 + (std::f32::consts::PI * r).cos())
    }
}

impl Fbpinn {
    /// Construct an FBPINN from a configuration.
    ///
    /// Subdomain centres are placed on an even grid spanning `[domain_lo, domain_hi]`:
    /// for dimension `d` with `n_d` cells, cell size `c_d = (hi_d − lo_d)/n_d`, the
    /// centre of cell `k` is `lo_d + (k + ½)·c_d`, and the half-width is
    /// `(c_d/2)·(1 + overlap)` so adjacent windows overlap. The total number of
    /// subdomains is `Π_d n_d` (cartesian product, iterated row-major).
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] for `dim == 0` or mismatched lengths.
    /// - [`PinnError::InvalidGridResolution`] if any `n_subdomains_per_dim[d] == 0`.
    /// - [`PinnError::InvalidTimeInterval`] if any `domain_lo[d] >= domain_hi[d]`.
    /// - [`PinnError::InvalidWeight`] if `overlap < 0` or not finite.
    /// - [`PinnError::InvalidLayerWidth`] if `hidden == 0`.
    pub fn new(cfg: &FbpinnConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if cfg.dim == 0 {
            return Err(PinnError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.n_subdomains_per_dim.len() != cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: cfg.dim,
                got: cfg.n_subdomains_per_dim.len(),
            });
        }
        if cfg.domain_lo.len() != cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: cfg.dim,
                got: cfg.domain_lo.len(),
            });
        }
        if cfg.domain_hi.len() != cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: cfg.dim,
                got: cfg.domain_hi.len(),
            });
        }
        if !cfg.overlap.is_finite() || cfg.overlap < 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: cfg.overlap,
            });
        }
        if cfg.hidden == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        for d in 0..cfg.dim {
            if cfg.n_subdomains_per_dim[d] == 0 {
                return Err(PinnError::InvalidGridResolution {
                    n: cfg.n_subdomains_per_dim[d],
                });
            }
            if cfg.domain_lo[d].partial_cmp(&cfg.domain_hi[d]) != Some(Ordering::Less) {
                return Err(PinnError::InvalidTimeInterval {
                    t0: cfg.domain_lo[d],
                    t1: cfg.domain_hi[d],
                });
            }
        }

        // Per-dimension centres and half-widths.
        let mut centers_per_dim: Vec<Vec<f32>> = Vec::with_capacity(cfg.dim);
        let mut halves_per_dim: Vec<f32> = Vec::with_capacity(cfg.dim);
        for d in 0..cfg.dim {
            let n_d = cfg.n_subdomains_per_dim[d];
            let cell = (cfg.domain_hi[d] - cfg.domain_lo[d]) / n_d as f32;
            let half = (cell / 2.0) * (1.0 + cfg.overlap);
            let centers: Vec<f32> = (0..n_d)
                .map(|k| cfg.domain_lo[d] + (k as f32 + 0.5) * cell)
                .collect();
            centers_per_dim.push(centers);
            halves_per_dim.push(half);
        }

        // Cartesian product over per-dimension indices, row-major.
        let total: usize = cfg.n_subdomains_per_dim.iter().product();
        let mut subdomains = Vec::with_capacity(total);
        let mut idx = vec![0usize; cfg.dim];
        for _ in 0..total {
            let center: Vec<f32> = (0..cfg.dim).map(|d| centers_per_dim[d][idx[d]]).collect();
            let half_width: Vec<f32> = (0..cfg.dim).map(|d| halves_per_dim[d]).collect();
            subdomains.push(Subdomain { center, half_width });
            // Increment the mixed-radix index (last dimension fastest → row-major).
            for d in (0..cfg.dim).rev() {
                idx[d] += 1;
                if idx[d] < cfg.n_subdomains_per_dim[d] {
                    break;
                }
                idx[d] = 0;
            }
        }

        // One MLP per subdomain: dim → hidden → 1, tanh activation.
        let mut networks = Vec::with_capacity(total);
        for _ in 0..total {
            let mlp_cfg = MlpConfig {
                layer_widths: vec![cfg.dim, cfg.hidden, 1],
                activation: Activation::Tanh,
                omega_0: 1.0,
            };
            networks.push(Mlp::new(mlp_cfg, rng)?);
        }

        Ok(Self {
            subdomains,
            networks,
            dim: cfg.dim,
        })
    }

    /// Number of subdomains.
    #[inline]
    pub fn n_subdomains(&self) -> usize {
        self.subdomains.len()
    }

    /// Raw (un-normalised) separable Hann window of subdomain `j` at point `x`.
    ///
    /// `ω_j(x) = Π_d ω_1d(x_d; center_jd, half_width_jd)`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`PinnError::InvalidGridResolution`] if `j >= n_subdomains` (index out of range).
    pub fn window(&self, j: usize, x: &[f32]) -> PinnResult<f32> {
        if x.len() != self.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let sub = self
            .subdomains
            .get(j)
            .ok_or(PinnError::InvalidGridResolution { n: j })?;
        let mut prod = 1.0_f32;
        for (d, &xd) in x.iter().enumerate() {
            prod *= hann_1d(xd, sub.center[d], sub.half_width[d]);
            if prod == 0.0 {
                break;
            }
        }
        Ok(prod)
    }

    /// Normalised partition-of-unity weights `ω̂_j(x)`, length `n_subdomains`.
    ///
    /// If the denominator `Σ_k ω_k(x) ≤ 1e-12` (point outside every support) all
    /// weights are returned as `0`. Otherwise `ω̂_j = ω_j / Σ_k ω_k`, which sums to
    /// `1` for any interior point (each lies in `≥ 1` support).
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    pub fn partition_of_unity(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let raw: Vec<f32> = (0..self.n_subdomains())
            .map(|j| self.window(j, x))
            .collect::<PinnResult<Vec<f32>>>()?;
        let denom: f32 = raw.iter().sum();
        if denom <= 1e-12 {
            return Ok(vec![0.0_f32; self.n_subdomains()]);
        }
        Ok(raw.into_iter().map(|w| w / denom).collect())
    }

    /// Locally-normalised input for subdomain `j`: `(x − center)/half_width`, clamped to `[-1, 1]`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`PinnError::InvalidGridResolution`] if `j >= n_subdomains`.
    pub fn normalize_local(&self, j: usize, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let sub = self
            .subdomains
            .get(j)
            .ok_or(PinnError::InvalidGridResolution { n: j })?;
        let out: Vec<f32> = (0..self.dim)
            .map(|d| {
                let hw = sub.half_width[d];
                let v = if hw > 0.0 {
                    (x[d] - sub.center[d]) / hw
                } else {
                    0.0
                };
                v.clamp(-1.0, 1.0)
            })
            .collect();
        Ok(out)
    }

    /// Global solution `u(x) = Σ_j ω̂_j(x) · u_j(normalize_local_j(x))`.
    ///
    /// Returns `0.0` at points outside every subdomain support (where the partition
    /// of unity is all zeros).
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`PinnError::NanEncountered`] if a non-finite value is produced.
    pub fn forward(&self, x: &[f32]) -> PinnResult<f32> {
        if x.len() != self.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let weights = self.partition_of_unity(x)?;
        let mut acc = 0.0_f32;
        for (j, &w) in weights.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let local = self.normalize_local(j, x)?;
            let u_j = self.networks[j].forward(&local)?;
            acc += w * u_j[0];
        }
        if !acc.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "fbpinn_forward",
            });
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_1d(n: usize) -> FbpinnConfig {
        FbpinnConfig {
            dim: 1,
            n_subdomains_per_dim: vec![n],
            domain_lo: vec![0.0],
            domain_hi: vec![1.0],
            overlap: 0.25,
            hidden: 8,
        }
    }

    fn cfg_2d(nx: usize, ny: usize) -> FbpinnConfig {
        FbpinnConfig {
            dim: 2,
            n_subdomains_per_dim: vec![nx, ny],
            domain_lo: vec![0.0, 0.0],
            domain_hi: vec![1.0, 1.0],
            overlap: 0.3,
            hidden: 6,
        }
    }

    #[test]
    fn n_subdomains_product() {
        let mut rng = LcgRng::new(1);
        let fb = Fbpinn::new(&cfg_2d(2, 3), &mut rng).unwrap();
        assert_eq!(fb.n_subdomains(), 6);
        let fb1 = Fbpinn::new(&cfg_1d(4), &mut rng).unwrap();
        assert_eq!(fb1.n_subdomains(), 4);
    }

    #[test]
    fn window_at_center_is_one() {
        let mut rng = LcgRng::new(2);
        let fb = Fbpinn::new(&cfg_1d(3), &mut rng).unwrap();
        for j in 0..fb.n_subdomains() {
            let c = fb.subdomains[j].center.clone();
            let w = fb.window(j, &c).unwrap();
            assert!((w - 1.0).abs() < 1e-5, "window at center should be 1: {w}");
        }
    }

    #[test]
    fn window_zero_outside_support() {
        let mut rng = LcgRng::new(3);
        let fb = Fbpinn::new(&cfg_1d(3), &mut rng).unwrap();
        // A point far outside the domain is outside every support.
        let w = fb.window(0, &[10.0]).unwrap();
        assert_eq!(w, 0.0);
    }

    #[test]
    fn partition_of_unity_sums_to_one_1d() {
        let mut rng = LcgRng::new(4);
        let fb = Fbpinn::new(&cfg_1d(3), &mut rng).unwrap();
        for i in 1..10 {
            let x = i as f32 / 10.0;
            let p = fb.partition_of_unity(&[x]).unwrap();
            let s: f32 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "PoU sum at x={x} = {s}");
        }
    }

    #[test]
    fn partition_of_unity_sums_to_one_2d() {
        let mut rng = LcgRng::new(5);
        let fb = Fbpinn::new(&cfg_2d(2, 2), &mut rng).unwrap();
        for ix in 1..5 {
            for iy in 1..5 {
                let x = [ix as f32 / 5.0, iy as f32 / 5.0];
                let p = fb.partition_of_unity(&x).unwrap();
                let s: f32 = p.iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "2D PoU sum at {x:?} = {s}");
            }
        }
    }

    #[test]
    fn partition_of_unity_nonneg() {
        let mut rng = LcgRng::new(6);
        let fb = Fbpinn::new(&cfg_1d(4), &mut rng).unwrap();
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let p = fb.partition_of_unity(&[x]).unwrap();
            assert!(p.iter().all(|&w| w >= 0.0), "PoU has a negative weight");
        }
    }

    #[test]
    fn normalize_local_center_and_edge() {
        let mut rng = LcgRng::new(7);
        let fb = Fbpinn::new(&cfg_1d(3), &mut rng).unwrap();
        let j = 1;
        let c = fb.subdomains[j].center.clone();
        let hw = fb.subdomains[j].half_width.clone();
        let at_center = fb.normalize_local(j, &c).unwrap();
        assert!(at_center[0].abs() < 1e-6);
        // Far beyond the edge → clamped to ±1.
        let beyond = [c[0] + 5.0 * hw[0]];
        let nl = fb.normalize_local(j, &beyond).unwrap();
        assert!((nl[0] - 1.0).abs() < 1e-6);
        let below = [c[0] - 5.0 * hw[0]];
        let nl2 = fb.normalize_local(j, &below).unwrap();
        assert!((nl2[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_subdomain_pou_one() {
        let mut rng = LcgRng::new(8);
        let fb = Fbpinn::new(&cfg_1d(1), &mut rng).unwrap();
        assert_eq!(fb.n_subdomains(), 1);
        for i in 1..10 {
            let x = i as f32 / 10.0;
            let p = fb.partition_of_unity(&[x]).unwrap();
            assert!(
                (p[0] - 1.0).abs() < 1e-5,
                "single-subdomain PoU at x={x}={}",
                p[0]
            );
        }
    }

    #[test]
    fn forward_finite_1d() {
        let mut rng = LcgRng::new(9);
        let fb = Fbpinn::new(&cfg_1d(3), &mut rng).unwrap();
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let u = fb.forward(&[x]).unwrap();
            assert!(u.is_finite(), "forward not finite at x={x}");
        }
    }

    #[test]
    fn forward_deterministic_given_seed() {
        let mut rng_a = LcgRng::new(321);
        let mut rng_b = LcgRng::new(321);
        let fb_a = Fbpinn::new(&cfg_2d(2, 2), &mut rng_a).unwrap();
        let fb_b = Fbpinn::new(&cfg_2d(2, 2), &mut rng_b).unwrap();
        let ua = fb_a.forward(&[0.4, 0.6]).unwrap();
        let ub = fb_b.forward(&[0.4, 0.6]).unwrap();
        assert!((ua - ub).abs() < 1e-9);
    }

    #[test]
    fn networks_len_matches_subdomains() {
        let mut rng = LcgRng::new(10);
        let fb = Fbpinn::new(&cfg_2d(3, 2), &mut rng).unwrap();
        assert_eq!(fb.networks.len(), fb.n_subdomains());
    }

    #[test]
    fn centers_within_domain() {
        let mut rng = LcgRng::new(11);
        let fb = Fbpinn::new(&cfg_2d(2, 3), &mut rng).unwrap();
        for sub in &fb.subdomains {
            assert!(sub.center[0] >= 0.0 && sub.center[0] <= 1.0);
            assert!(sub.center[1] >= 0.0 && sub.center[1] <= 1.0);
        }
    }

    #[test]
    fn forward_2d_runs() {
        let mut rng = LcgRng::new(12);
        let fb = Fbpinn::new(&cfg_2d(2, 2), &mut rng).unwrap();
        let u = fb.forward(&[0.5, 0.5]).unwrap();
        assert!(u.is_finite());
    }

    #[test]
    fn err_dim_mismatch_in_x() {
        let mut rng = LcgRng::new(13);
        let fb = Fbpinn::new(&cfg_2d(2, 2), &mut rng).unwrap();
        assert!(matches!(
            fb.forward(&[0.5]),
            Err(PinnError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            fb.partition_of_unity(&[0.5, 0.5, 0.5]),
            Err(PinnError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            fb.window(0, &[0.5]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_npd_len_mismatch() {
        let mut rng = LcgRng::new(14);
        let cfg = FbpinnConfig {
            dim: 2,
            n_subdomains_per_dim: vec![2], // wrong length
            domain_lo: vec![0.0, 0.0],
            domain_hi: vec![1.0, 1.0],
            overlap: 0.25,
            hidden: 4,
        };
        assert!(matches!(
            Fbpinn::new(&cfg, &mut rng),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_zero_in_npd() {
        let mut rng = LcgRng::new(15);
        let cfg = FbpinnConfig {
            dim: 2,
            n_subdomains_per_dim: vec![2, 0],
            domain_lo: vec![0.0, 0.0],
            domain_hi: vec![1.0, 1.0],
            overlap: 0.25,
            hidden: 4,
        };
        assert!(matches!(
            Fbpinn::new(&cfg, &mut rng),
            Err(PinnError::InvalidGridResolution { .. })
        ));
    }

    #[test]
    fn err_lo_ge_hi() {
        let mut rng = LcgRng::new(16);
        let cfg = FbpinnConfig {
            dim: 1,
            n_subdomains_per_dim: vec![2],
            domain_lo: vec![1.0],
            domain_hi: vec![1.0],
            overlap: 0.25,
            hidden: 4,
        };
        assert!(matches!(
            Fbpinn::new(&cfg, &mut rng),
            Err(PinnError::InvalidTimeInterval { .. })
        ));
    }

    #[test]
    fn err_hidden_zero() {
        let mut rng = LcgRng::new(17);
        let cfg = FbpinnConfig {
            dim: 1,
            n_subdomains_per_dim: vec![2],
            domain_lo: vec![0.0],
            domain_hi: vec![1.0],
            overlap: 0.25,
            hidden: 0,
        };
        assert!(matches!(
            Fbpinn::new(&cfg, &mut rng),
            Err(PinnError::InvalidLayerWidth)
        ));
    }

    #[test]
    fn err_j_out_of_range() {
        let mut rng = LcgRng::new(18);
        let fb = Fbpinn::new(&cfg_1d(2), &mut rng).unwrap();
        assert!(matches!(
            fb.window(99, &[0.5]),
            Err(PinnError::InvalidGridResolution { .. })
        ));
        assert!(matches!(
            fb.normalize_local(99, &[0.5]),
            Err(PinnError::InvalidGridResolution { .. })
        ));
    }

    #[test]
    fn err_dim_zero() {
        let mut rng = LcgRng::new(19);
        let cfg = FbpinnConfig {
            dim: 0,
            n_subdomains_per_dim: vec![],
            domain_lo: vec![],
            domain_hi: vec![],
            overlap: 0.25,
            hidden: 4,
        };
        assert!(matches!(
            Fbpinn::new(&cfg, &mut rng),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_negative_overlap() {
        let mut rng = LcgRng::new(20);
        let cfg = FbpinnConfig {
            dim: 1,
            n_subdomains_per_dim: vec![2],
            domain_lo: vec![0.0],
            domain_hi: vec![1.0],
            overlap: -0.1,
            hidden: 4,
        };
        assert!(matches!(
            Fbpinn::new(&cfg, &mut rng),
            Err(PinnError::InvalidWeight { .. })
        ));
    }
}
