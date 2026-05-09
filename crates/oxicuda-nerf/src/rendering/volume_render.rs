//! Volume rendering equations for NeRF.
//!
//! Standard NeRF volume rendering:
//! - `alpha_i = 1 - exp(-max(0, sigma_i) * delta_i)`
//! - `T_i = exp(-sum_{j<i} sigma_j * delta_j) = prod_{j<i}(1 - alpha_j)`
//! - `C(r) = sum_i T_i * alpha_i * c_i`
//! - `D(r) = sum_i T_i * alpha_i * t_i`
//! - `O(r) = sum_i T_i * alpha_i = 1 - T_N`

use crate::error::{NerfError, NerfResult};

// ─── RenderResult ─────────────────────────────────────────────────────────────

/// Output of volume rendering for a single ray.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderResult {
    /// Rendered RGB color.
    pub rgb: [f32; 3],
    /// Expected depth (weighted by transmittance × alpha).
    pub depth: f32,
    /// Accumulated opacity (0 = transparent, 1 = fully opaque).
    pub opacity: f32,
}

// ─── Volume render (single ray) ───────────────────────────────────────────────

/// Volume render a single ray.
///
/// - `sigma`: density values `[N]` (non-negative; negatives are clamped to 0).
/// - `color`: RGB values `[N * 3]` in [0, 1].
/// - `t_vals`: sample positions `[N]` along the ray.
///
/// Last segment uses delta = 1e10 (infinity).
///
/// # Errors
///
/// Returns `DimensionMismatch` if sizes don't match,
/// `EmptyInput` if N == 0,
/// `VolumeRenderError` for invalid configurations.
pub fn volume_render(sigma: &[f32], color: &[f32], t_vals: &[f32]) -> NerfResult<RenderResult> {
    let n = sigma.len();
    if n == 0 {
        return Err(NerfError::EmptyInput);
    }
    if color.len() != n * 3 {
        return Err(NerfError::DimensionMismatch {
            expected: n * 3,
            got: color.len(),
        });
    }
    if t_vals.len() != n {
        return Err(NerfError::DimensionMismatch {
            expected: n,
            got: t_vals.len(),
        });
    }

    let mut rgb = [0.0_f32; 3];
    let mut depth = 0.0_f32;
    let mut opacity = 0.0_f32;
    let mut transmittance = 1.0_f32;

    for i in 0..n {
        // Delta: distance to next sample (last gets "infinity")
        let delta = if i + 1 < n {
            (t_vals[i + 1] - t_vals[i]).max(0.0)
        } else {
            1e10_f32
        };

        let sigma_i = sigma[i].max(0.0);
        let alpha = 1.0 - (-sigma_i * delta).exp();
        let weight = transmittance * alpha;

        rgb[0] += weight * color[i * 3];
        rgb[1] += weight * color[i * 3 + 1];
        rgb[2] += weight * color[i * 3 + 2];
        depth += weight * t_vals[i];
        opacity += weight;

        transmittance *= 1.0 - alpha;

        // Early termination when transmittance is essentially zero
        if transmittance < 1e-4 {
            break;
        }
    }

    Ok(RenderResult {
        rgb,
        depth,
        opacity,
    })
}

// ─── Volume render (batch) ────────────────────────────────────────────────────

/// Volume render a batch of M rays.
///
/// - `sigma`: `[M * N]` densities, row-major (ray-major).
/// - `color`: `[M * N * 3]` RGB colors.
/// - `t_vals`: `[M * N]` sample positions (same layout as sigma).
/// - `n_rays`: M
/// - `n_samples`: N
///
/// Returns `(rgb: [M*3], depth: [M], opacity: [M])`.
///
/// # Errors
///
/// Returns `DimensionMismatch` or `EmptyInput` for inconsistent inputs.
pub fn volume_render_batch(
    sigma: &[f32],
    color: &[f32],
    t_vals: &[f32],
    n_rays: usize,
    n_samples: usize,
) -> NerfResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if n_rays == 0 || n_samples == 0 {
        return Err(NerfError::EmptyInput);
    }
    if sigma.len() != n_rays * n_samples {
        return Err(NerfError::DimensionMismatch {
            expected: n_rays * n_samples,
            got: sigma.len(),
        });
    }
    if color.len() != n_rays * n_samples * 3 {
        return Err(NerfError::DimensionMismatch {
            expected: n_rays * n_samples * 3,
            got: color.len(),
        });
    }
    if t_vals.len() != n_rays * n_samples {
        return Err(NerfError::DimensionMismatch {
            expected: n_rays * n_samples,
            got: t_vals.len(),
        });
    }

    let mut rgb_out = vec![0.0_f32; n_rays * 3];
    let mut depth_out = vec![0.0_f32; n_rays];
    let mut opacity_out = vec![0.0_f32; n_rays];

    for ray_idx in 0..n_rays {
        let s_off = ray_idx * n_samples;
        let c_off = ray_idx * n_samples * 3;

        let ray_sigma = &sigma[s_off..s_off + n_samples];
        let ray_color = &color[c_off..c_off + n_samples * 3];
        let ray_t = &t_vals[s_off..s_off + n_samples];

        let res = volume_render(ray_sigma, ray_color, ray_t)?;

        rgb_out[ray_idx * 3] = res.rgb[0];
        rgb_out[ray_idx * 3 + 1] = res.rgb[1];
        rgb_out[ray_idx * 3 + 2] = res.rgb[2];
        depth_out[ray_idx] = res.depth;
        opacity_out[ray_idx] = res.opacity;
    }

    Ok((rgb_out, depth_out, opacity_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_scene_zero_opacity() {
        let sigma = vec![0.0_f32; 8];
        let color = vec![1.0_f32; 24]; // white
        let t = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let res = volume_render(&sigma, &color, &t).unwrap();
        assert!(res.opacity < 1e-6, "zero density → zero opacity");
    }

    #[test]
    fn opaque_first_sample() {
        let mut sigma = vec![0.0_f32; 8];
        sigma[0] = 1e6_f32; // extremely dense first sample
        let mut color = vec![0.0_f32; 24];
        // First sample is red
        color[0] = 1.0;
        let t = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let res = volume_render(&sigma, &color, &t).unwrap();
        assert!(
            res.rgb[0] > 0.99,
            "red first sample dominant, got {}",
            res.rgb[0]
        );
        assert!(res.opacity > 0.99, "high opacity, got {}", res.opacity);
    }

    #[test]
    fn batch_render_shape() {
        let n_rays = 4;
        let n_samp = 8;
        let sigma = vec![0.1_f32; n_rays * n_samp];
        let color = vec![0.5_f32; n_rays * n_samp * 3];
        let t: Vec<f32> = (0..n_rays * n_samp).map(|i| i as f32 * 0.1 + 0.1).collect();
        let (rgb, depth, opacity) =
            volume_render_batch(&sigma, &color, &t, n_rays, n_samp).unwrap();
        assert_eq!(rgb.len(), n_rays * 3);
        assert_eq!(depth.len(), n_rays);
        assert_eq!(opacity.len(), n_rays);
    }
}
