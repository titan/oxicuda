//! LPIPS — Learned Perceptual Image Patch Similarity.
//!
//! Zhang, Isola, Efros, Shechtman & Wang (2018), "The Unreasonable
//! Effectiveness of Deep Features as a Perceptual Metric", CVPR.
//!
//! LPIPS measures perceptual distance between two images by comparing deep CNN
//! features rather than raw pixels, matching human similarity judgements far
//! better than MSE/PSNR. The recipe is:
//!
//! 1. Push both images through a fixed CNN backbone, collecting activations at a
//!    handful of layers.
//! 2. **Unit-normalise** each activation across the channel dimension at every
//!    spatial location.
//! 3. Take the per-channel-weighted squared difference of the two normalised
//!    feature stacks, average over spatial positions, and sum over layers.
//!
//! The published metric uses a pretrained VGG/AlexNet backbone with learned
//! linear layer weights. This compact CPU reference substitutes a small, fixed
//! convolutional stack whose filters are drawn from a seeded LCG (so the metric
//! is fully deterministic and dependency-free) and uses unit linear weights.
//! The structural properties LPIPS relies on — non-negativity, symmetry,
//! identity-of-indiscernibles and monotonicity under growing perturbations — are
//! preserved and unit-tested.
//!
//! Image layout is channel-major: a `[channels, height, width]` tensor flattened
//! as `data[(c * height + y) * width + x]`.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

/// Numerical floor for the per-location channel-normalisation denominator.
const NORM_EPS: f32 = 1e-10;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration of the fixed LPIPS backbone.
#[derive(Debug, Clone)]
pub struct LpipsConfig {
    /// Number of input image channels (e.g. 3 for RGB).
    pub in_channels: usize,
    /// Output channels of each successive 3×3 convolution layer. The first layer
    /// keeps resolution (stride 1); every later layer halves it (stride 2).
    pub channels: Vec<usize>,
}

impl Default for LpipsConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            channels: vec![8, 16, 16],
        }
    }
}

// ─── Convolution layer ───────────────────────────────────────────────────────

/// A single 3×3 convolution followed by ReLU (channel-major tensors).
#[derive(Debug, Clone)]
struct ConvLayer {
    weight: Vec<f32>, // [out_c * in_c * k * k]
    bias: Vec<f32>,   // [out_c]
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl ConvLayer {
    fn new(
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        pad: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let fan_in = in_c * k * k;
        let scale = (2.0_f32 / fan_in.max(1) as f32).sqrt();
        let mut weight = vec![0.0_f32; out_c * fan_in];
        let mut idx = 0;
        while idx + 1 < weight.len() {
            let (a, b) = rng.next_normal_pair();
            weight[idx] = a * scale;
            weight[idx + 1] = b * scale;
            idx += 2;
        }
        if idx < weight.len() {
            let (a, _) = rng.next_normal_pair();
            weight[idx] = a * scale;
        }
        Self {
            weight,
            bias: vec![0.0_f32; out_c],
            in_c,
            out_c,
            k,
            stride,
            pad,
        }
    }

    /// Convolve `input` (`[in_c, in_h, in_w]`, channel-major) and apply ReLU.
    /// Returns `(output, out_h, out_w)`.
    fn forward(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
    ) -> NerfResult<(Vec<f32>, usize, usize)> {
        if input.len() != self.in_c * in_h * in_w {
            return Err(NerfError::DimensionMismatch {
                expected: self.in_c * in_h * in_w,
                got: input.len(),
            });
        }
        if in_h + 2 * self.pad < self.k || in_w + 2 * self.pad < self.k {
            return Err(NerfError::InvalidGridResolution {
                res: in_h.min(in_w),
            });
        }
        let out_h = (in_h + 2 * self.pad - self.k) / self.stride + 1;
        let out_w = (in_w + 2 * self.pad - self.k) / self.stride + 1;
        let mut out = vec![0.0_f32; self.out_c * out_h * out_w];

        for oc in 0..self.out_c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = self.bias[oc];
                    for ic in 0..self.in_c {
                        for ky in 0..self.k {
                            let iy = (oy * self.stride + ky) as isize - self.pad as isize;
                            if iy < 0 || iy >= in_h as isize {
                                continue;
                            }
                            let iy_u = iy as usize;
                            for kx in 0..self.k {
                                let ix = (ox * self.stride + kx) as isize - self.pad as isize;
                                if ix < 0 || ix >= in_w as isize {
                                    continue;
                                }
                                let ix_u = ix as usize;
                                let w_idx = ((oc * self.in_c + ic) * self.k + ky) * self.k + kx;
                                let in_idx = (ic * in_h + iy_u) * in_w + ix_u;
                                acc += self.weight[w_idx] * input[in_idx];
                            }
                        }
                    }
                    out[(oc * out_h + oy) * out_w + ox] = acc.max(0.0);
                }
            }
        }
        Ok((out, out_h, out_w))
    }
}

// ─── Feature map ─────────────────────────────────────────────────────────────

/// A collected backbone activation.
#[derive(Debug, Clone)]
struct FeatureMap {
    /// Channel-major activation data `[channels * locations]`.
    data: Vec<f32>,
    /// Number of channels.
    channels: usize,
    /// Number of spatial locations (`height * width`).
    locations: usize,
}

/// Unit-normalise a channel-major feature `[channels * locations]` across the
/// channel dimension at each spatial location.
fn unit_normalize_channels(data: &[f32], channels: usize, locations: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; data.len()];
    for p in 0..locations {
        let mut sum_sq = 0.0_f32;
        for c in 0..channels {
            let v = data[c * locations + p];
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq.sqrt() + NORM_EPS);
        for c in 0..channels {
            out[c * locations + p] = data[c * locations + p] * inv;
        }
    }
    out
}

// ─── Lpips ───────────────────────────────────────────────────────────────────

/// LPIPS perceptual metric over a small fixed convolutional backbone.
#[derive(Debug, Clone)]
pub struct Lpips {
    layers: Vec<ConvLayer>,
    /// Per-layer, per-channel linear weights (all 1.0 in this reference).
    layer_weights: Vec<Vec<f32>>,
    in_channels: usize,
}

impl Lpips {
    /// Build an LPIPS metric with the default backbone using `rng` for the fixed
    /// filters.
    ///
    /// # Errors
    ///
    /// Propagates configuration errors from [`Lpips::with_config`].
    pub fn new(rng: &mut LcgRng) -> NerfResult<Self> {
        Self::with_config(&LpipsConfig::default(), rng)
    }

    /// Build a deterministic reference metric (fixed seed, default backbone).
    ///
    /// # Errors
    ///
    /// Propagates configuration errors from [`Lpips::with_config`].
    pub fn reference() -> NerfResult<Self> {
        let mut rng = LcgRng::new(0x_4C50_4950);
        Self::new(&mut rng)
    }

    /// Build an LPIPS metric from an explicit backbone configuration.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidFeatureDim`] for empty / zero dimensions.
    pub fn with_config(cfg: &LpipsConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.in_channels == 0 || cfg.channels.is_empty() || cfg.channels.contains(&0) {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        let mut layers = Vec::with_capacity(cfg.channels.len());
        let mut prev = cfg.in_channels;
        for (i, &out_c) in cfg.channels.iter().enumerate() {
            let stride = if i == 0 { 1 } else { 2 };
            layers.push(ConvLayer::new(prev, out_c, 3, stride, 1, rng));
            prev = out_c;
        }
        let layer_weights = cfg.channels.iter().map(|&c| vec![1.0_f32; c]).collect();
        Ok(Self {
            layers,
            layer_weights,
            in_channels: cfg.in_channels,
        })
    }

    /// Number of input channels expected by [`Lpips::distance`].
    #[must_use]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Run the backbone, collecting the ReLU activation after every layer.
    fn features(&self, img: &[f32], h: usize, w: usize) -> NerfResult<Vec<FeatureMap>> {
        let mut maps = Vec::with_capacity(self.layers.len());
        let mut cur = img.to_vec();
        let mut cur_h = h;
        let mut cur_w = w;
        for layer in &self.layers {
            let (out, out_h, out_w) = layer.forward(&cur, cur_h, cur_w)?;
            cur = out;
            cur_h = out_h;
            cur_w = out_w;
            maps.push(FeatureMap {
                data: cur.clone(),
                channels: layer.out_c,
                locations: cur_h * cur_w,
            });
        }
        Ok(maps)
    }

    /// Perceptual distance between two channel-major images `[c, h, w]`.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::EmptyInput`] for empty images,
    /// [`NerfError::DimensionMismatch`] if the channel count differs from the
    /// configured backbone or the buffer length is wrong, and
    /// [`NerfError::NanEncountered`] on non-finite inputs.
    pub fn distance(
        &self,
        img_a: &[f32],
        img_b: &[f32],
        c: usize,
        h: usize,
        w: usize,
    ) -> NerfResult<f32> {
        if img_a.is_empty() || img_b.is_empty() {
            return Err(NerfError::EmptyInput);
        }
        if c != self.in_channels {
            return Err(NerfError::DimensionMismatch {
                expected: self.in_channels,
                got: c,
            });
        }
        let expected = c * h * w;
        if img_a.len() != expected || img_b.len() != expected {
            return Err(NerfError::DimensionMismatch {
                expected,
                got: img_a.len().min(img_b.len()),
            });
        }
        if img_a.iter().chain(img_b.iter()).any(|v| !v.is_finite()) {
            return Err(NerfError::NanEncountered {
                context: "lpips: non-finite pixel".into(),
            });
        }

        let feats_a = self.features(img_a, h, w)?;
        let feats_b = self.features(img_b, h, w)?;

        let mut total = 0.0_f32;
        for ((map_a, map_b), weights) in feats_a.iter().zip(feats_b.iter()).zip(&self.layer_weights)
        {
            let norm_a = unit_normalize_channels(&map_a.data, map_a.channels, map_a.locations);
            let norm_b = unit_normalize_channels(&map_b.data, map_b.channels, map_b.locations);
            let locations = map_a.locations;
            let mut layer_sum = 0.0_f32;
            for p in 0..locations {
                let mut location_sum = 0.0_f32;
                for ch in 0..map_a.channels {
                    let diff = norm_a[ch * locations + p] - norm_b[ch * locations + p];
                    location_sum += weights[ch] * diff * diff;
                }
                layer_sum += location_sum;
            }
            if locations > 0 {
                total += layer_sum / locations as f32;
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_image(c: usize, h: usize, w: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..c * h * w).map(|_| rng.next_f32()).collect()
    }

    #[test]
    fn identical_images_zero_distance() {
        let lpips = Lpips::reference().expect("reference should succeed");
        let img = ramp_image(3, 16, 16, 11);
        let d = lpips
            .distance(&img, &img, 3, 16, 16)
            .expect("distance should succeed");
        assert!(d.abs() < 1e-5, "LPIPS(x,x) should be 0, got {d}");
    }

    #[test]
    fn distance_is_symmetric() {
        let lpips = Lpips::reference().expect("reference should succeed");
        let a = ramp_image(3, 16, 16, 1);
        let b = ramp_image(3, 16, 16, 2);
        let d_ab = lpips
            .distance(&a, &b, 3, 16, 16)
            .expect("distance should succeed");
        let d_ba = lpips
            .distance(&b, &a, 3, 16, 16)
            .expect("distance should succeed");
        assert!((d_ab - d_ba).abs() < 1e-5, "asymmetry: {d_ab} vs {d_ba}");
    }

    #[test]
    fn distance_is_non_negative() {
        let lpips = Lpips::reference().expect("reference should succeed");
        for seed in 0..6 {
            let a = ramp_image(3, 16, 16, seed);
            let b = ramp_image(3, 16, 16, seed + 100);
            let d = lpips
                .distance(&a, &b, 3, 16, 16)
                .expect("distance should succeed");
            assert!(
                d >= 0.0 && d.is_finite(),
                "distance must be ≥0 & finite: {d}"
            );
        }
    }

    #[test]
    fn larger_perturbation_larger_distance() {
        let lpips = Lpips::reference().expect("reference should succeed");
        let base = ramp_image(3, 16, 16, 7);
        let mut noise_rng = LcgRng::new(123);
        let noise: Vec<f32> = (0..base.len())
            .map(|_| noise_rng.next_f32_range(-1.0, 1.0))
            .collect();
        let small: Vec<f32> = base
            .iter()
            .zip(noise.iter())
            .map(|(&v, &n)| (v + 0.03 * n).clamp(0.0, 1.0))
            .collect();
        let large: Vec<f32> = base
            .iter()
            .zip(noise.iter())
            .map(|(&v, &n)| (v + 0.4 * n).clamp(0.0, 1.0))
            .collect();
        let d_small = lpips
            .distance(&base, &small, 3, 16, 16)
            .expect("distance should succeed");
        let d_large = lpips
            .distance(&base, &large, 3, 16, 16)
            .expect("distance should succeed");
        assert!(
            d_large > d_small,
            "bigger perturbation should give bigger LPIPS: small={d_small} large={d_large}"
        );
    }

    #[test]
    fn channel_normalization_is_unit_norm() {
        let channels = 3;
        let locations = 4;
        // Every location has a strictly positive channel vector.
        let mut data = vec![0.0_f32; channels * locations];
        for c in 0..channels {
            for p in 0..locations {
                data[c * locations + p] = (c + 1) as f32 * (p + 2) as f32;
            }
        }
        let normed = unit_normalize_channels(&data, channels, locations);
        for p in 0..locations {
            let mag: f32 = (0..channels)
                .map(|c| normed[c * locations + p].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "location {p} norm = {mag}");
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let lpips = Lpips::reference().expect("reference should succeed");
        let img = ramp_image(3, 16, 16, 5);
        // Wrong channel count.
        assert!(matches!(
            lpips.distance(&img, &img, 2, 16, 16),
            Err(NerfError::DimensionMismatch { .. })
        ));
        // Buffer too small for declared shape.
        let small = vec![0.0_f32; 10];
        assert!(lpips.distance(&small, &small, 3, 16, 16).is_err());
        // Empty.
        assert!(matches!(
            lpips.distance(&[], &[], 3, 16, 16),
            Err(NerfError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_non_finite() {
        let lpips = Lpips::reference().expect("reference should succeed");
        let mut a = ramp_image(3, 16, 16, 9);
        a[0] = f32::NAN;
        let b = ramp_image(3, 16, 16, 9);
        assert!(matches!(
            lpips.distance(&a, &b, 3, 16, 16),
            Err(NerfError::NanEncountered { .. })
        ));
    }

    #[test]
    fn with_config_rejects_zero_dims() {
        let mut rng = LcgRng::new(1);
        let cfg = LpipsConfig {
            in_channels: 3,
            channels: vec![],
        };
        assert!(matches!(
            Lpips::with_config(&cfg, &mut rng),
            Err(NerfError::InvalidFeatureDim { .. })
        ));
    }
}
