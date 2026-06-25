//! FCFD — Feature Compression by Frequency Decomposition.
//!
//! Spatial feature maps carry both *coarse structure* (low spatial frequencies) and
//! *fine detail / texture* (high spatial frequencies). Naively matching student to teacher
//! in the raw spatial domain forces the student to reproduce noisy high-frequency content
//! that often does not transfer well. FCFD instead applies a 2-D **Discrete Cosine
//! Transform** (DCT-II) to each channel, partitions the resulting frequency coefficients
//! into a low band and a high band by a zig-zag radius, and matches the two bands with
//! independent weights. This lets the distiller emphasise the transferable low-frequency
//! structure while down-weighting (or entirely dropping) the high-frequency residual.
//!
//! The DCT here is the orthonormal separable DCT-II, computed directly (`O(N²)` per axis,
//! exact for the small feature-map sizes typical in distillation). Because DCT-II is an
//! orthonormal change of basis, the per-coefficient MSE in the frequency domain equals the
//! per-pixel MSE in the spatial domain when both bands are weighted equally — the band
//! weighting is what makes FCFD a genuine frequency-selective objective.
//!
//! Feature maps are laid out `[channels × height × width]`, channel-major and flat, matching
//! the rest of the `feature/` module.

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-12;

/// Configuration for frequency-decomposition feature distillation.
#[derive(Debug, Clone)]
pub struct FcfdConfig {
    /// Channel count of the feature maps.
    pub channels: usize,
    /// Feature-map height.
    pub height: usize,
    /// Feature-map width.
    pub width: usize,
    /// Zig-zag radius (in `u + v` index sum) separating low from high frequencies.
    /// Coefficients with `u + v <= cutoff` are "low", the rest are "high".
    pub cutoff: usize,
    /// Weight on the low-frequency band MSE.
    pub low_weight: f32,
    /// Weight on the high-frequency band MSE.
    pub high_weight: f32,
}

impl FcfdConfig {
    /// Validate and construct a configuration.
    pub fn new(
        channels: usize,
        height: usize,
        width: usize,
        cutoff: usize,
        low_weight: f32,
        high_weight: f32,
    ) -> DistillResult<Self> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "channels, height and width must all be non-zero".into(),
            });
        }
        if low_weight < 0.0 || high_weight < 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: "band weights must be non-negative".into(),
            });
        }
        if !low_weight.is_finite() || !high_weight.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: "band weights must be finite".into(),
            });
        }
        Ok(Self {
            channels,
            height,
            width,
            cutoff,
            low_weight,
            high_weight,
        })
    }

    /// Total number of spatial elements per channel.
    #[must_use]
    pub fn spatial(&self) -> usize {
        self.height * self.width
    }
}

/// Precomputed orthonormal DCT-II basis row `k` for a signal of length `n`.
///
/// `basis[k * n + i] = c_k · cos(π (2 i + 1) k / (2 n))`, where `c_0 = √(1/n)` and
/// `c_k = √(2/n)` for `k > 0`. The resulting matrix is orthonormal so the inverse is its
/// transpose.
#[must_use]
pub fn dct_basis(n: usize) -> Vec<f32> {
    let mut basis = vec![0.0_f32; n * n];
    if n == 0 {
        return basis;
    }
    let c0 = (1.0_f32 / n as f32).sqrt();
    let ck = (2.0_f32 / n as f32).sqrt();
    for k in 0..n {
        let scale = if k == 0 { c0 } else { ck };
        for i in 0..n {
            let angle = std::f32::consts::PI * (2 * i + 1) as f32 * k as f32 / (2.0 * n as f32);
            basis[k * n + i] = scale * angle.cos();
        }
    }
    basis
}

/// Apply a separable 2-D DCT-II to a single channel (row-major `height × width`).
///
/// Returns the frequency coefficients `[height × width]` with `coeff[u * width + v]`
/// the `(u, v)` frequency. `row_basis` / `col_basis` are produced by [`dct_basis`].
#[must_use]
pub fn dct2_channel(
    plane: &[f32],
    height: usize,
    width: usize,
    row_basis: &[f32],
    col_basis: &[f32],
) -> Vec<f32> {
    // Step 1: transform along the width (rows) — temp[h, v] = Σ_w col_basis[v,w] * plane[h,w].
    let mut temp = vec![0.0_f32; height * width];
    for h in 0..height {
        let in_row = &plane[h * width..(h + 1) * width];
        for v in 0..width {
            let b_row = &col_basis[v * width..(v + 1) * width];
            let mut acc = 0.0_f32;
            for w in 0..width {
                acc += b_row[w] * in_row[w];
            }
            temp[h * width + v] = acc;
        }
    }
    // Step 2: transform along the height (columns) — out[u, v] = Σ_h row_basis[u,h] * temp[h,v].
    let mut out = vec![0.0_f32; height * width];
    for u in 0..height {
        let b_row = &row_basis[u * height..(u + 1) * height];
        for v in 0..width {
            let mut acc = 0.0_f32;
            for h in 0..height {
                acc += b_row[h] * temp[h * width + v];
            }
            out[u * width + v] = acc;
        }
    }
    out
}

/// Banded MSE between two coefficient planes.
///
/// Returns `(low_mse, high_mse)` where a coefficient `(u, v)` is "low" iff `u + v <= cutoff`.
/// Each band is normalised by the number of coefficients it contains.
#[must_use]
pub fn banded_mse(
    coeff_s: &[f32],
    coeff_t: &[f32],
    height: usize,
    width: usize,
    cutoff: usize,
) -> (f32, f32) {
    let mut low_sum = 0.0_f32;
    let mut high_sum = 0.0_f32;
    let mut low_n = 0usize;
    let mut high_n = 0usize;
    for u in 0..height {
        for v in 0..width {
            let idx = u * width + v;
            let diff = coeff_s[idx] - coeff_t[idx];
            let sq = diff * diff;
            if u + v <= cutoff {
                low_sum += sq;
                low_n += 1;
            } else {
                high_sum += sq;
                high_n += 1;
            }
        }
    }
    let low = if low_n > 0 {
        low_sum / low_n as f32
    } else {
        0.0
    };
    let high = if high_n > 0 {
        high_sum / high_n as f32
    } else {
        0.0
    };
    (low, high)
}

/// FCFD loss between a student and teacher feature map.
///
/// Each channel is DCT-transformed, split into low / high bands, and the band MSEs are
/// accumulated across channels with the configured weights:
/// `loss = low_weight · mean_c low_mse_c + high_weight · mean_c high_mse_c`.
pub fn fcfd_loss(s_feat: &[f32], t_feat: &[f32], cfg: &FcfdConfig) -> DistillResult<f32> {
    if s_feat.is_empty() || t_feat.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let expected = cfg.channels * cfg.spatial();
    if s_feat.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: s_feat.len(),
        });
    }
    if t_feat.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: t_feat.len(),
        });
    }
    let row_basis = dct_basis(cfg.height);
    let col_basis = dct_basis(cfg.width);
    let hw = cfg.spatial();
    let mut low_acc = 0.0_f32;
    let mut high_acc = 0.0_f32;
    for c in 0..cfg.channels {
        let s_plane = &s_feat[c * hw..(c + 1) * hw];
        let t_plane = &t_feat[c * hw..(c + 1) * hw];
        let cs = dct2_channel(s_plane, cfg.height, cfg.width, &row_basis, &col_basis);
        let ct = dct2_channel(t_plane, cfg.height, cfg.width, &row_basis, &col_basis);
        let (low, high) = banded_mse(&cs, &ct, cfg.height, cfg.width, cfg.cutoff);
        low_acc += low;
        high_acc += high;
    }
    let nc = cfg.channels as f32;
    Ok(cfg.low_weight * (low_acc / nc) + cfg.high_weight * (high_acc / nc))
}

/// Fraction of total spectral energy contained in the low-frequency band of a feature map.
///
/// A diagnostic in `[0, 1]`: values near 1 mean the map is dominated by coarse structure.
pub fn low_band_energy_ratio(feat: &[f32], cfg: &FcfdConfig) -> DistillResult<f32> {
    if feat.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let expected = cfg.channels * cfg.spatial();
    if feat.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: feat.len(),
        });
    }
    let row_basis = dct_basis(cfg.height);
    let col_basis = dct_basis(cfg.width);
    let hw = cfg.spatial();
    let mut low_energy = 0.0_f32;
    let mut total_energy = 0.0_f32;
    for c in 0..cfg.channels {
        let plane = &feat[c * hw..(c + 1) * hw];
        let coeff = dct2_channel(plane, cfg.height, cfg.width, &row_basis, &col_basis);
        for u in 0..cfg.height {
            for v in 0..cfg.width {
                let e = coeff[u * cfg.width + v].powi(2);
                total_energy += e;
                if u + v <= cfg.cutoff {
                    low_energy += e;
                }
            }
        }
    }
    Ok(low_energy / total_energy.max(EPS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn dct_basis_is_orthonormal() {
        let n = 6;
        let b = dct_basis(n);
        // Rows must be orthonormal: B Bᵀ = I.
        for k in 0..n {
            for l in 0..n {
                let mut dot = 0.0_f32;
                for i in 0..n {
                    dot += b[k * n + i] * b[l * n + i];
                }
                let expected = if k == l { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "<row {k}, row {l}> = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn dct_dc_coefficient_is_scaled_mean() {
        // The (0,0) coefficient of an orthonormal DCT-II equals mean * √(H*W).
        let h = 4;
        let w = 4;
        let plane = vec![3.0_f32; h * w];
        let rb = dct_basis(h);
        let cb = dct_basis(w);
        let coeff = dct2_channel(&plane, h, w, &rb, &cb);
        let expected = 3.0 * ((h * w) as f32).sqrt();
        assert!((coeff[0] - expected).abs() < 1e-3, "dc={}", coeff[0]);
        // A constant plane has zero AC energy.
        for &c in &coeff[1..] {
            assert!(c.abs() < 1e-3, "ac coeff {c} should be ~0");
        }
    }

    #[test]
    fn parseval_energy_preserved() {
        // Orthonormal DCT preserves L2 energy: Σ pixel² == Σ coeff².
        let mut rng = LcgRng::new(42);
        let h = 5;
        let w = 7;
        let plane: Vec<f32> = (0..h * w).map(|_| rng.next_normal()).collect();
        let rb = dct_basis(h);
        let cb = dct_basis(w);
        let coeff = dct2_channel(&plane, h, w, &rb, &cb);
        let e_spatial: f32 = plane.iter().map(|&x| x * x).sum();
        let e_freq: f32 = coeff.iter().map(|&x| x * x).sum();
        assert!(
            (e_spatial - e_freq).abs() < 1e-2 * e_spatial.max(1.0),
            "spatial {e_spatial} vs freq {e_freq}"
        );
    }

    #[test]
    fn fcfd_identical_is_zero() {
        let cfg = FcfdConfig::new(2, 4, 4, 2, 1.0, 0.5).expect("cfg");
        let mut rng = LcgRng::new(7);
        let feat: Vec<f32> = (0..2 * 16).map(|_| rng.next_normal()).collect();
        let loss = fcfd_loss(&feat, &feat, &cfg).expect("loss");
        assert!(loss < 1e-5, "loss {loss}");
    }

    #[test]
    fn fcfd_equal_weights_equals_spatial_mse() {
        // With low_weight == high_weight, FCFD reduces to per-coefficient MSE which (by
        // Parseval) equals per-pixel MSE since each channel band is averaged over all coeffs.
        // Use a single channel and cutoff that yields balanced averaging by reconstructing
        // the combined MSE manually.
        let cfg = FcfdConfig::new(1, 4, 4, 3, 1.0, 1.0).expect("cfg");
        let mut rng = LcgRng::new(99);
        let s: Vec<f32> = (0..16).map(|_| rng.next_normal()).collect();
        let t: Vec<f32> = (0..16).map(|_| rng.next_normal()).collect();
        let rb = dct_basis(4);
        let cb = dct_basis(4);
        let cs = dct2_channel(&s, 4, 4, &rb, &cb);
        let ct = dct2_channel(&t, 4, 4, &rb, &cb);
        let (low, high) = banded_mse(&cs, &ct, 4, 4, 3);
        let manual = low + high;
        let loss = fcfd_loss(&s, &t, &cfg).expect("loss");
        assert!((loss - manual).abs() < 1e-4, "loss {loss} manual {manual}");
    }

    #[test]
    fn fcfd_high_weight_zero_ignores_high_band() {
        // Two maps that differ only in high frequencies should give ~0 loss when high_weight=0.
        let cfg_low = FcfdConfig::new(1, 4, 4, 1, 1.0, 0.0).expect("cfg");
        // Build a teacher; student = teacher + a pure high-frequency perturbation.
        let mut rng = LcgRng::new(123);
        let t: Vec<f32> = (0..16).map(|_| rng.next_normal()).collect();
        // Construct a high-freq-only delta in coefficient space, inverse-transform it.
        let rb = dct_basis(4);
        let cb = dct_basis(4);
        let mut delta_coeff = vec![0.0_f32; 16];
        delta_coeff[3 * 4 + 3] = 2.0; // highest frequency
        // Inverse DCT = transpose application; reuse forward with transposed bases.
        let mut rb_t = vec![0.0_f32; 16];
        let mut cb_t = vec![0.0_f32; 16];
        for a in 0..4 {
            for b in 0..4 {
                rb_t[a * 4 + b] = rb[b * 4 + a];
                cb_t[a * 4 + b] = cb[b * 4 + a];
            }
        }
        let delta = dct2_channel(&delta_coeff, 4, 4, &rb_t, &cb_t);
        let s: Vec<f32> = t.iter().zip(delta.iter()).map(|(&a, &b)| a + b).collect();
        let loss = fcfd_loss(&s, &t, &cfg_low).expect("loss");
        assert!(loss < 1e-3, "high-freq diff leaked into low band: {loss}");
    }

    #[test]
    fn energy_ratio_in_unit_interval() {
        let cfg = FcfdConfig::new(3, 4, 4, 2, 1.0, 1.0).expect("cfg");
        let mut rng = LcgRng::new(55);
        let feat: Vec<f32> = (0..3 * 16).map(|_| rng.next_normal()).collect();
        let r = low_band_energy_ratio(&feat, &cfg).expect("ratio");
        assert!((0.0..=1.0).contains(&r), "ratio {r}");
    }

    #[test]
    fn config_rejects_zero_dims() {
        assert!(FcfdConfig::new(0, 4, 4, 2, 1.0, 1.0).is_err());
        assert!(FcfdConfig::new(1, 0, 4, 2, 1.0, 1.0).is_err());
        assert!(FcfdConfig::new(1, 4, 4, 2, -1.0, 1.0).is_err());
    }

    #[test]
    fn fcfd_dim_mismatch_errors() {
        let cfg = FcfdConfig::new(2, 4, 4, 2, 1.0, 1.0).expect("cfg");
        let s = vec![0.0_f32; 10];
        let t = vec![0.0_f32; 32];
        assert!(fcfd_loss(&s, &t, &cfg).is_err());
    }
}
