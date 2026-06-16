//! 3D Gaussian Splatting — differentiable CPU rasterizer.
//!
//! Kerbl, Kopanas, Leimkühler & Drettakis (2023), "3D Gaussian Splatting for
//! Real-Time Radiance Field Rendering", SIGGRAPH.
//!
//! A radiance field is represented by a set of anisotropic 3D Gaussians, each
//! carrying a position `μ`, an anisotropic covariance (factorised as a scale
//! `s ∈ R³` and a rotation quaternion `q`), an opacity and a colour. Rendering
//! is a *rasterisation*, not a ray march:
//!
//! 1. **Projection (EWA splatting).** Every 3D Gaussian is projected to a 2D
//!    screen-space Gaussian. The mean projects through the pinhole model
//!    `(u, v) = (f_x x/z + c_x, f_y y/z + c_y)`. The covariance is transported by
//!    `Σ₂ᴅ = J W Σ₃ᴅ Wᵀ Jᵀ`, where `W` is the world→camera rotation and `J` is
//!    the affine (Jacobian) approximation of the perspective projection at the
//!    Gaussian centre — this is the Zwicker et al. *Elliptical Weighted Average*
//!    splatting Jacobian. A small low-pass term (`DILATION`) is added to the 2D
//!    covariance diagonal so sub-pixel splats stay invertible (anti-aliasing).
//! 2. **Depth sort.** Splats are sorted front-to-back by camera-space depth.
//! 3. **Alpha compositing.** Per pixel, `C = Σ_i c_i α_i Π_{j<i}(1 − α_j)` with
//!    `α_i = o_i · exp(−½ dᵀ Σ₂ᴅ⁻¹ d)` and `d` the pixel offset from the splat
//!    centre. This is the same front-to-back integral as volume rendering, but
//!    over a depth-sorted list of 2D Gaussians instead of ray samples.
//!
//! The forward pass is differentiable; [`backward_pixel`] returns the exact
//! analytic gradient of a pixel's colour with respect to every contributing
//! Gaussian's opacity and colour (the parameters that pass through the splat
//! unchanged), which is the load-bearing part of the 3DGS optimiser.

use crate::error::{NerfError, NerfResult};
use crate::rendering::ray::PinholeCamera;

/// Low-pass dilation added to the projected covariance diagonal (in pixel²).
///
/// Keeps the 2D covariance strictly positive-definite — and therefore
/// invertible — even when a Gaussian projects to a sub-pixel footprint, exactly
/// the EWA anti-aliasing filter used by the reference rasteriser.
pub const DILATION: f32 = 0.3;

/// Maximum per-splat alpha, clamped for numerical stability of the
/// front-to-back recurrence (keeps `1 − α > 0`).
pub const MAX_ALPHA: f32 = 0.999;

/// Transmittance below which the front-to-back composite terminates early.
const MIN_TRANSMITTANCE: f32 = 1.0e-4;

// ─── Gaussian3d ──────────────────────────────────────────────────────────────

/// A single anisotropic 3D Gaussian primitive.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian3d {
    /// World-space mean `μ`.
    pub position: [f32; 3],
    /// Per-axis standard deviations `s` (positive) of the principal axes.
    pub scale: [f32; 3],
    /// Orientation quaternion `q = (w, x, y, z)` of the principal axes.
    pub quaternion: [f32; 4],
    /// Opacity in `[0, 1]`.
    pub opacity: f32,
    /// RGB colour.
    pub color: [f32; 3],
}

impl Gaussian3d {
    /// Construct a Gaussian, validating finiteness and positive scale.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::Internal`] for non-finite fields or a non-positive
    /// scale component.
    pub fn new(
        position: [f32; 3],
        scale: [f32; 3],
        quaternion: [f32; 4],
        opacity: f32,
        color: [f32; 3],
    ) -> NerfResult<Self> {
        let finite = position.iter().all(|v| v.is_finite())
            && scale.iter().all(|v| v.is_finite())
            && quaternion.iter().all(|v| v.is_finite())
            && opacity.is_finite()
            && color.iter().all(|v| v.is_finite());
        if !finite {
            return Err(NerfError::Internal {
                msg: "Gaussian3d fields must be finite".into(),
            });
        }
        if scale.iter().any(|&v| v <= 0.0) {
            return Err(NerfError::Internal {
                msg: "Gaussian3d scale components must be strictly positive".into(),
            });
        }
        Ok(Self {
            position,
            scale,
            quaternion,
            opacity,
            color,
        })
    }

    /// World-space 3D covariance `Σ₃ᴅ = R diag(s²) Rᵀ` (row-major 3×3).
    #[must_use]
    pub fn covariance_3d(&self) -> [f32; 9] {
        covariance_3d(self.scale, self.quaternion)
    }
}

// ─── Quaternion / matrix utilities ───────────────────────────────────────────

/// Convert a quaternion `q = (w, x, y, z)` to a row-major 3×3 rotation matrix.
///
/// The quaternion is normalised first, so the result is orthonormal with
/// determinant `+1`. A zero quaternion maps to the identity.
#[must_use]
pub fn quat_to_rotation(q: [f32; 4]) -> [f32; 9] {
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let inv = if norm_sq > 1.0e-20 {
        1.0 / norm_sq.sqrt()
    } else {
        0.0
    };
    let w = q[0] * inv;
    let x = q[1] * inv;
    let y = q[2] * inv;
    let z = q[3] * inv;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// World-space 3D covariance `Σ = R diag(s²) Rᵀ` from a scale and quaternion.
#[must_use]
pub fn covariance_3d(scale: [f32; 3], quat: [f32; 4]) -> [f32; 9] {
    let r = quat_to_rotation(quat);
    // M = R · diag(scale) — scale the columns of R.
    let m = [
        r[0] * scale[0],
        r[1] * scale[1],
        r[2] * scale[2],
        r[3] * scale[0],
        r[4] * scale[1],
        r[5] * scale[2],
        r[6] * scale[0],
        r[7] * scale[1],
        r[8] * scale[2],
    ];
    // Σ = M Mᵀ.
    mat3_mul(&m, &mat3_transpose(&m))
}

/// Row-major 3×3 matrix product `a · b`.
#[must_use]
fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
        }
    }
    out
}

/// Transpose of a row-major 3×3 matrix.
#[must_use]
fn mat3_transpose(a: &[f32; 9]) -> [f32; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

// ─── SplatCamera ─────────────────────────────────────────────────────────────

/// Camera for splatting: pinhole intrinsics plus a world→camera transform.
///
/// A world point `p` is mapped to camera space by `p_cam = view_rot · p +
/// view_trans`, then projected with the [`PinholeCamera`] intrinsics.
#[derive(Debug, Clone, Copy)]
pub struct SplatCamera {
    /// Pinhole intrinsics (focal lengths, principal point, image size).
    pub intrinsics: PinholeCamera,
    /// World→camera rotation `W` (row-major 3×3).
    pub view_rot: [f32; 9],
    /// World→camera translation `t`.
    pub view_trans: [f32; 3],
    /// Near plane; Gaussians with camera-space `z ≤ near` are culled.
    pub near: f32,
}

impl SplatCamera {
    /// Build a splatting camera.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::Internal`] for non-finite extrinsics or a
    /// non-positive near plane.
    pub fn new(
        intrinsics: PinholeCamera,
        view_rot: [f32; 9],
        view_trans: [f32; 3],
        near: f32,
    ) -> NerfResult<Self> {
        if !view_rot.iter().all(|v| v.is_finite())
            || !view_trans.iter().all(|v| v.is_finite())
            || !near.is_finite()
            || near <= 0.0
        {
            return Err(NerfError::Internal {
                msg: "SplatCamera extrinsics must be finite and near > 0".into(),
            });
        }
        Ok(Self {
            intrinsics,
            view_rot,
            view_trans,
            near,
        })
    }

    /// Identity-pose camera (`W = I`, `t = 0`) looking down `+z`.
    ///
    /// # Errors
    ///
    /// Propagates [`SplatCamera::new`] validation.
    pub fn identity(intrinsics: PinholeCamera, near: f32) -> NerfResult<Self> {
        Self::new(
            intrinsics,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            near,
        )
    }

    /// Transform a world point into camera space `p_cam = W p + t`.
    #[must_use]
    pub fn to_camera(&self, p: [f32; 3]) -> [f32; 3] {
        let w = &self.view_rot;
        [
            w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + self.view_trans[0],
            w[3] * p[0] + w[4] * p[1] + w[5] * p[2] + self.view_trans[1],
            w[6] * p[0] + w[7] * p[1] + w[8] * p[2] + self.view_trans[2],
        ]
    }
}

// ─── Splat2d ─────────────────────────────────────────────────────────────────

/// A screen-space 2D Gaussian produced by projecting a [`Gaussian3d`].
#[derive(Debug, Clone, Copy)]
pub struct Splat2d {
    /// Projected pixel-space mean `(u, v)`.
    pub mean: [f32; 2],
    /// 2D covariance `Σ₂ᴅ = [a, b, c]` for `[[a, b], [b, c]]` (with dilation).
    pub cov2d: [f32; 3],
    /// Inverse covariance (conic) `Σ₂ᴅ⁻¹ = [A, B, C]` for `[[A, B], [B, C]]`.
    pub conic: [f32; 3],
    /// Camera-space depth (`z`), used for front-to-back ordering.
    pub depth: f32,
    /// Opacity carried through from the source Gaussian.
    pub opacity: f32,
    /// Colour carried through from the source Gaussian.
    pub color: [f32; 3],
}

impl Splat2d {
    /// Evaluate the unnormalised Gaussian weight `exp(−½ dᵀ Σ⁻¹ d)` at a pixel.
    #[must_use]
    pub fn gaussian_weight(&self, px: f32, py: f32) -> f32 {
        let dx = px - self.mean[0];
        let dy = py - self.mean[1];
        let power = -0.5
            * (self.conic[0] * dx * dx + 2.0 * self.conic[1] * dx * dy + self.conic[2] * dy * dy);
        if power > 0.0 {
            // Numerical guard: the quadratic form is PSD so power ≤ 0 in theory.
            1.0
        } else {
            power.exp()
        }
    }
}

// ─── Projection ──────────────────────────────────────────────────────────────

/// Project a single 3D Gaussian into screen space via EWA splatting.
///
/// Returns `None` when the Gaussian is at/behind the near plane or the
/// projected covariance is degenerate (zero determinant).
#[must_use]
pub fn project_gaussian(g: &Gaussian3d, cam: &SplatCamera) -> Option<Splat2d> {
    let t = cam.to_camera(g.position);
    if !t[2].is_finite() || t[2] <= cam.near {
        return None;
    }
    let fx = cam.intrinsics.fx;
    let fy = cam.intrinsics.fy;
    let inv_z = 1.0 / t[2];

    // Mean projection (pinhole).
    let mean = [
        fx * t[0] * inv_z + cam.intrinsics.cx,
        fy * t[1] * inv_z + cam.intrinsics.cy,
    ];

    // Camera-space covariance Σ_cam = W Σ₃ᴅ Wᵀ.
    let cov3d = g.covariance_3d();
    let cov_cam = mat3_mul(
        &mat3_mul(&cam.view_rot, &cov3d),
        &mat3_transpose(&cam.view_rot),
    );

    // EWA Jacobian J (2×3) of the perspective projection at the centre.
    let inv_z2 = inv_z * inv_z;
    let j00 = fx * inv_z;
    let j02 = -fx * t[0] * inv_z2;
    let j11 = fy * inv_z;
    let j12 = -fy * t[1] * inv_z2;
    // Row 0 = (j00, 0, j02); Row 1 = (0, j11, j12).

    // M = J · Σ_cam (2×3). Row 0 of J is (j00, 0, j02), row 1 is (0, j11, j12).
    let cc = &cov_cam;
    let m00 = j00 * cc[0] + j02 * cc[6];
    let m01 = j00 * cc[1] + j02 * cc[7];
    let m02 = j00 * cc[2] + j02 * cc[8];
    let m11 = j11 * cc[4] + j12 * cc[7];
    let m12 = j11 * cc[5] + j12 * cc[8];

    // Σ₂ᴅ = M · Jᵀ (2×2), symmetric, plus low-pass dilation on the diagonal.
    let a = m00 * j00 + m02 * j02 + DILATION;
    let b = m01 * j11 + m02 * j12;
    let c = m11 * j11 + m12 * j12 + DILATION;

    let det = a * c - b * b;
    if det <= 1.0e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    // Σ⁻¹ = (1/det) [[c, −b], [−b, a]].
    let conic = [c * inv_det, -b * inv_det, a * inv_det];

    Some(Splat2d {
        mean,
        cov2d: [a, b, c],
        conic,
        depth: t[2],
        opacity: g.opacity,
        color: g.color,
    })
}

/// Project a scene of Gaussians and sort the surviving splats front-to-back.
#[must_use]
pub fn project_scene(scene: &[Gaussian3d], cam: &SplatCamera) -> Vec<Splat2d> {
    let mut splats: Vec<Splat2d> = scene
        .iter()
        .filter_map(|g| project_gaussian(g, cam))
        .collect();
    splats.sort_by(|p, q| p.depth.total_cmp(&q.depth));
    splats
}

// ─── Per-pixel forward / backward ────────────────────────────────────────────

/// Composite a depth-sorted splat list at one pixel.
///
/// Returns the rendered RGB plus the accumulated alpha (`1 − T`, the sum of the
/// front-to-back weights, which is non-negative and `≤ 1`).
#[must_use]
pub fn rasterize_pixel(splats: &[Splat2d], px: f32, py: f32) -> ([f32; 3], f32) {
    let mut rgb = [0.0_f32; 3];
    let mut transmittance = 1.0_f32;
    for s in splats {
        let alpha = (s.opacity * s.gaussian_weight(px, py)).clamp(0.0, MAX_ALPHA);
        if alpha < 1.0e-6 {
            continue;
        }
        let weight = transmittance * alpha;
        rgb[0] += weight * s.color[0];
        rgb[1] += weight * s.color[1];
        rgb[2] += weight * s.color[2];
        transmittance *= 1.0 - alpha;
        if transmittance < MIN_TRANSMITTANCE {
            break;
        }
    }
    (rgb, 1.0 - transmittance)
}

/// Gradient of one pixel's colour w.r.t. a single contributing splat.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplatPixelGrad {
    /// `∂(loss)/∂opacity` for this splat.
    pub d_opacity: f32,
    /// `∂(loss)/∂color` for this splat (per channel).
    pub d_color: [f32; 3],
}

/// Analytic backward pass of [`rasterize_pixel`] for one pixel.
///
/// Given the upstream gradient `grad_pixel = ∂loss/∂C` (a 3-vector), returns the
/// gradient of the loss w.r.t. each splat's opacity and colour. The returned
/// vector is index-aligned with `splats` (depth-sorted order). Opacity and
/// colour pass through projection unchanged, so these are also the gradients
/// w.r.t. the source Gaussians' opacity/colour.
///
/// Colour gradient: `∂C/∂c_i = w_i = T_i α_i`.
/// Opacity gradient: with `α_i = o_i G_i`,
/// `∂C/∂α_i = c_i T_i − (Σ_{k>i} w_k c_k)/(1 − α_i)` and
/// `∂C/∂o_i = G_i ∂C/∂α_i` (zero where `α_i` saturates at [`MAX_ALPHA`]).
#[must_use]
pub fn backward_pixel(
    splats: &[Splat2d],
    px: f32,
    py: f32,
    grad_pixel: [f32; 3],
) -> Vec<SplatPixelGrad> {
    let n = splats.len();
    let mut grads = vec![SplatPixelGrad::default(); n];

    // Forward sweep: record per-splat (G, alpha, weight, transmittance-before).
    let mut g_val = vec![0.0_f32; n];
    let mut alpha_v = vec![0.0_f32; n];
    let mut weight_v = vec![0.0_f32; n];
    let mut t_before = vec![0.0_f32; n];
    let mut transmittance = 1.0_f32;
    let mut last = 0usize;
    for i in 0..n {
        let g = splats[i].gaussian_weight(px, py);
        let raw = splats[i].opacity * g;
        let alpha = raw.clamp(0.0, MAX_ALPHA);
        t_before[i] = transmittance;
        g_val[i] = g;
        alpha_v[i] = alpha;
        weight_v[i] = transmittance * alpha;
        transmittance *= 1.0 - alpha;
        last = i + 1;
        if transmittance < MIN_TRANSMITTANCE {
            break;
        }
    }

    // Backward sweep: accumulate the colour behind each splat.
    let mut suffix = [0.0_f32; 3];
    for i in (0..last).rev() {
        let alpha = alpha_v[i];
        if alpha < 1.0e-6 {
            continue;
        }
        let weight = weight_v[i];
        let color = splats[i].color;

        // dL/dcolor = weight * grad_pixel.
        grads[i].d_color = [
            weight * grad_pixel[0],
            weight * grad_pixel[1],
            weight * grad_pixel[2],
        ];

        // dC/dalpha_i = color * T_i - suffix/(1 - alpha_i).
        let one_minus = 1.0 - alpha;
        let inv_one_minus = if one_minus > 1.0e-6 {
            1.0 / one_minus
        } else {
            0.0
        };
        let tb = t_before[i];
        let mut d_alpha = 0.0_f32;
        for ((&gp, &col), &suf) in grad_pixel.iter().zip(color.iter()).zip(suffix.iter()) {
            d_alpha += gp * (col * tb - suf * inv_one_minus);
        }

        // dalpha/dopacity = G (zero when alpha is clamped at the cap).
        let saturated = splats[i].opacity * g_val[i] > MAX_ALPHA;
        grads[i].d_opacity = if saturated { 0.0 } else { d_alpha * g_val[i] };

        // Fold this splat into the suffix colour for the next (nearer) splat.
        suffix[0] += weight * color[0];
        suffix[1] += weight * color[1];
        suffix[2] += weight * color[2];
    }

    grads
}

// ─── Image rasterizer ────────────────────────────────────────────────────────

/// Rendered output of [`rasterize`].
#[derive(Debug, Clone)]
pub struct SplatImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major RGB buffer `[H·W·3]`.
    pub rgb: Vec<f32>,
    /// Row-major accumulated alpha `[H·W]` (sum of weights, in `[0, 1]`).
    pub alpha: Vec<f32>,
    /// Row-major expected depth `[H·W]` (weight-averaged camera-space `z`).
    pub depth: Vec<f32>,
}

/// Rasterise a scene of 3D Gaussians into an image.
///
/// Projects and depth-sorts the Gaussians once, then composites every pixel
/// front-to-back. The pixel sample point is the pixel centre `(col+0.5,
/// row+0.5)`, matching the ray-generation convention of [`PinholeCamera`].
///
/// # Errors
///
/// Returns [`NerfError::EmptyInput`] if the camera image size is zero.
pub fn rasterize(scene: &[Gaussian3d], cam: &SplatCamera) -> NerfResult<SplatImage> {
    let width = cam.intrinsics.width;
    let height = cam.intrinsics.height;
    if width == 0 || height == 0 {
        return Err(NerfError::EmptyInput);
    }

    let splats = project_scene(scene, cam);

    let n_pixels = (width as usize) * (height as usize);
    let mut rgb = vec![0.0_f32; n_pixels * 3];
    let mut alpha = vec![0.0_f32; n_pixels];
    let mut depth = vec![0.0_f32; n_pixels];

    for row in 0..height {
        for col in 0..width {
            let px = col as f32 + 0.5;
            let py = row as f32 + 0.5;
            let pix = (row as usize) * (width as usize) + col as usize;

            // Composite, also tracking weight-averaged depth.
            let mut transmittance = 1.0_f32;
            let mut acc = [0.0_f32; 3];
            let mut acc_depth = 0.0_f32;
            for s in &splats {
                let a = (s.opacity * s.gaussian_weight(px, py)).clamp(0.0, MAX_ALPHA);
                if a < 1.0e-6 {
                    continue;
                }
                let weight = transmittance * a;
                acc[0] += weight * s.color[0];
                acc[1] += weight * s.color[1];
                acc[2] += weight * s.color[2];
                acc_depth += weight * s.depth;
                transmittance *= 1.0 - a;
                if transmittance < MIN_TRANSMITTANCE {
                    break;
                }
            }
            rgb[pix * 3] = acc[0];
            rgb[pix * 3 + 1] = acc[1];
            rgb[pix * 3 + 2] = acc[2];
            alpha[pix] = 1.0 - transmittance;
            depth[pix] = acc_depth;
        }
    }

    Ok(SplatImage {
        width,
        height,
        rgb,
        alpha,
        depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> SplatCamera {
        // 16×16 image, principal point at the centre.
        let intr = PinholeCamera::new(16.0, 16.0, 8.0, 8.0, 16, 16).expect("new should succeed");
        SplatCamera::identity(intr, 0.01).expect("identity should succeed")
    }

    fn unit_quat() -> [f32; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    fn norm3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn quaternion_to_rotation_is_orthonormal() {
        // A non-trivial rotation: 0.6 rad about a tilted axis.
        let half = 0.3_f32;
        let axis = {
            let a = [1.0_f32, -2.0, 0.5];
            let n = norm3(a);
            [a[0] / n, a[1] / n, a[2] / n]
        };
        let q = [
            half.cos(),
            axis[0] * half.sin(),
            axis[1] * half.sin(),
            axis[2] * half.sin(),
        ];
        let r = quat_to_rotation(q);
        // RᵀR = I.
        for i in 0..3 {
            for j in 0..3 {
                let dot = r[i] * r[j] + r[3 + i] * r[3 + j] + r[6 + i] * r[6 + j];
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-5,
                    "RᵀR[{i}][{j}] = {dot}, expected {expect}"
                );
            }
        }
        // det(R) = +1.
        let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);
        assert!((det - 1.0).abs() < 1e-5, "det(R) = {det}");
    }

    #[test]
    fn identity_quaternion_is_identity_matrix() {
        let r = quat_to_rotation(unit_quat());
        let expect = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (val, exp) in r.iter().zip(expect.iter()) {
            assert!((val - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn on_axis_gaussian_projects_to_image_point() {
        let cam = test_camera();
        // Place a Gaussian off-axis at depth 4; expected projection is analytic.
        let g = Gaussian3d::new(
            [0.5, -0.25, 4.0],
            [0.1, 0.1, 0.1],
            unit_quat(),
            0.8,
            [1.0, 0.0, 0.0],
        )
        .expect("value should be present");
        let s = project_gaussian(&g, &cam).expect("Gaussian in front of camera projects");
        let expect_u = 16.0 * 0.5 / 4.0 + 8.0;
        let expect_v = 16.0 * (-0.25) / 4.0 + 8.0;
        assert!((s.mean[0] - expect_u).abs() < 1e-4, "u = {}", s.mean[0]);
        assert!((s.mean[1] - expect_v).abs() < 1e-4, "v = {}", s.mean[1]);
        assert!((s.depth - 4.0).abs() < 1e-5);
    }

    #[test]
    fn centered_gaussian_projects_to_principal_point() {
        let cam = test_camera();
        let g = Gaussian3d::new(
            [0.0, 0.0, 3.0],
            [0.2, 0.2, 0.2],
            unit_quat(),
            0.9,
            [0.2, 0.4, 0.6],
        )
        .expect("value should be present");
        let s = project_gaussian(&g, &cam).expect("project_gaussian should succeed");
        assert!((s.mean[0] - 8.0).abs() < 1e-4);
        assert!((s.mean[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn projected_covariance_is_spd() {
        let cam = test_camera();
        // Anisotropic, rotated Gaussian.
        let q = {
            let h = 0.4_f32;
            [h.cos(), 0.0, h.sin(), 0.0]
        };
        let g = Gaussian3d::new([0.3, 0.2, 2.5], [0.3, 0.1, 0.2], q, 0.7, [0.5, 0.5, 0.5])
            .expect("new should succeed");
        let s = project_gaussian(&g, &cam).expect("project_gaussian should succeed");
        let (a, b, c) = (s.cov2d[0], s.cov2d[1], s.cov2d[2]);
        assert!(a > 0.0, "Σ[0][0] = {a}");
        assert!(c > 0.0, "Σ[1][1] = {c}");
        let det = a * c - b * b;
        assert!(det > 0.0, "det(Σ₂ᴅ) = {det}");
        // Inverse should be consistent (conic · cov ≈ I).
        let id00 = s.conic[0] * a + s.conic[1] * b;
        let id11 = s.conic[1] * b + s.conic[2] * c;
        assert!((id00 - 1.0).abs() < 1e-3, "conic·cov[0][0] = {id00}");
        assert!((id11 - 1.0).abs() < 1e-3, "conic·cov[1][1] = {id11}");
    }

    #[test]
    fn behind_camera_is_culled() {
        let cam = test_camera();
        let g = Gaussian3d::new(
            [0.0, 0.0, -1.0],
            [0.1, 0.1, 0.1],
            unit_quat(),
            0.9,
            [1.0, 1.0, 1.0],
        )
        .expect("value should be present");
        assert!(project_gaussian(&g, &cam).is_none());
    }

    #[test]
    fn opacity_zero_contributes_nothing() {
        let cam = test_camera();
        let g = Gaussian3d::new(
            [0.0, 0.0, 3.0],
            [0.5, 0.5, 0.5],
            unit_quat(),
            0.0,
            [1.0, 1.0, 1.0],
        )
        .expect("value should be present");
        let img = rasterize(&[g], &cam).expect("rasterize should succeed");
        for v in &img.rgb {
            assert_eq!(*v, 0.0, "transparent Gaussian must not paint");
        }
        for a in &img.alpha {
            assert_eq!(*a, 0.0);
        }
    }

    #[test]
    fn depth_ordering_nearer_dominates() {
        let cam = test_camera();
        // Two large, near-opaque, overlapping splats at the image centre,
        // one nearer (red) than the other (blue).
        let near = Gaussian3d::new(
            [0.0, 0.0, 2.0],
            [0.6, 0.6, 0.6],
            unit_quat(),
            0.99,
            [1.0, 0.0, 0.0],
        )
        .expect("value should be present");
        let far = Gaussian3d::new(
            [0.0, 0.0, 6.0],
            [1.8, 1.8, 1.8],
            unit_quat(),
            0.99,
            [0.0, 0.0, 1.0],
        )
        .expect("value should be present");
        let img = rasterize(&[far, near], &cam).expect("rasterize should succeed");
        let center = (8usize * 16 + 8) * 3;
        let r = img.rgb[center];
        let b = img.rgb[center + 2];
        assert!(r > b, "nearer red splat should dominate: r={r}, b={b}");
        assert!(r > 0.5, "near splat should be clearly visible, r={r}");
    }

    #[test]
    fn render_weights_are_bounded_and_nonneg() {
        let cam = test_camera();
        let scene = [
            Gaussian3d::new(
                [0.0, 0.0, 2.0],
                [0.4, 0.4, 0.4],
                unit_quat(),
                0.8,
                [1.0, 0.2, 0.1],
            )
            .expect("value should be present"),
            Gaussian3d::new(
                [0.2, 0.1, 3.0],
                [0.5, 0.3, 0.4],
                unit_quat(),
                0.6,
                [0.1, 0.9, 0.2],
            )
            .expect("value should be present"),
            Gaussian3d::new(
                [-0.2, -0.1, 4.0],
                [0.3, 0.6, 0.3],
                unit_quat(),
                0.7,
                [0.2, 0.3, 0.9],
            )
            .expect("value should be present"),
        ];
        let img = rasterize(&scene, &cam).expect("rasterize should succeed");
        for &a in &img.alpha {
            assert!(
                (0.0..=1.0).contains(&a),
                "accumulated alpha {a} out of [0,1]"
            );
        }
        for &v in &img.rgb {
            assert!(
                v.is_finite() && v >= 0.0,
                "pixel value {v} must be finite, non-negative"
            );
        }
        for &d in &img.depth {
            assert!(d.is_finite() && d >= 0.0);
        }
    }

    #[test]
    fn changing_color_changes_pixel() {
        let cam = test_camera();
        let g1 = Gaussian3d::new(
            [0.0, 0.0, 3.0],
            [0.5, 0.5, 0.5],
            unit_quat(),
            0.9,
            [0.9, 0.1, 0.1],
        )
        .expect("value should be present");
        let mut g2 = g1;
        g2.color = [0.1, 0.1, 0.9];
        let img1 = rasterize(&[g1], &cam).expect("rasterize should succeed");
        let img2 = rasterize(&[g2], &cam).expect("rasterize should succeed");
        let center = (8usize * 16 + 8) * 3;
        let diff = (img1.rgb[center] - img2.rgb[center]).abs()
            + (img1.rgb[center + 2] - img2.rgb[center + 2]).abs();
        assert!(
            diff > 0.1,
            "varying colour must change the pixel, diff={diff}"
        );
    }

    #[test]
    fn analytic_color_gradient_matches_finite_difference() {
        let cam = test_camera();
        let scene = [
            Gaussian3d::new(
                [0.1, 0.0, 2.0],
                [0.5, 0.5, 0.5],
                unit_quat(),
                0.6,
                [0.7, 0.2, 0.3],
            )
            .expect("value should be present"),
            Gaussian3d::new(
                [-0.1, 0.1, 3.0],
                [0.5, 0.5, 0.5],
                unit_quat(),
                0.5,
                [0.2, 0.6, 0.4],
            )
            .expect("value should be present"),
        ];
        let splats = project_scene(&scene, &cam);
        let (px, py) = (8.5_f32, 8.5);
        // Gradient of the red channel: grad_pixel = (1, 0, 0).
        let grads = backward_pixel(&splats, px, py, [1.0, 0.0, 0.0]);
        let h = 1e-3_f32;
        for (i, s) in splats.iter().enumerate() {
            let mut sp = splats.clone();
            sp[i].color[0] = s.color[0] + h;
            let cp = rasterize_pixel(&sp, px, py).0[0];
            sp[i].color[0] = s.color[0] - h;
            let cm = rasterize_pixel(&sp, px, py).0[0];
            let fd = (cp - cm) / (2.0 * h);
            assert!(
                (grads[i].d_color[0] - fd).abs() < 1e-3,
                "splat {i}: analytic dC/dcolor = {}, finite diff = {fd}",
                grads[i].d_color[0]
            );
        }
    }

    #[test]
    fn analytic_opacity_gradient_matches_finite_difference() {
        let cam = test_camera();
        let scene = [
            Gaussian3d::new(
                [0.1, 0.0, 2.0],
                [0.5, 0.5, 0.5],
                unit_quat(),
                0.55,
                [0.7, 0.2, 0.3],
            )
            .expect("value should be present"),
            Gaussian3d::new(
                [-0.05, 0.05, 3.0],
                [0.5, 0.5, 0.5],
                unit_quat(),
                0.45,
                [0.2, 0.6, 0.4],
            )
            .expect("value should be present"),
        ];
        let splats = project_scene(&scene, &cam);
        let (px, py) = (8.5_f32, 8.5);
        let grads = backward_pixel(&splats, px, py, [1.0, 0.0, 0.0]);
        let h = 1e-3_f32;
        for (i, s) in splats.iter().enumerate() {
            let mut sp = splats.clone();
            sp[i].opacity = s.opacity + h;
            let cp = rasterize_pixel(&sp, px, py).0[0];
            sp[i].opacity = s.opacity - h;
            let cm = rasterize_pixel(&sp, px, py).0[0];
            let fd = (cp - cm) / (2.0 * h);
            assert!(
                (grads[i].d_opacity - fd).abs() < 2e-3,
                "splat {i}: analytic dC/dopacity = {}, finite diff = {fd}",
                grads[i].d_opacity
            );
        }
    }

    #[test]
    fn covariance_3d_is_symmetric_psd() {
        let q = {
            let h = 0.5_f32;
            [h.cos(), h.sin() * 0.5, h.sin() * 0.5, h.sin() * 0.6]
        };
        let cov = covariance_3d([0.4, 0.2, 0.6], q);
        // Symmetric.
        assert!((cov[1] - cov[3]).abs() < 1e-6);
        assert!((cov[2] - cov[6]).abs() < 1e-6);
        assert!((cov[5] - cov[7]).abs() < 1e-6);
        // Positive trace and positive determinant (PD).
        let trace = cov[0] + cov[4] + cov[8];
        assert!(trace > 0.0);
        let det = cov[0] * (cov[4] * cov[8] - cov[5] * cov[7])
            - cov[1] * (cov[3] * cov[8] - cov[5] * cov[6])
            + cov[2] * (cov[3] * cov[7] - cov[4] * cov[6]);
        assert!(
            det > 0.0,
            "covariance should be positive-definite, det={det}"
        );
    }

    #[test]
    fn rasterize_output_is_finite() {
        let cam = test_camera();
        let scene: Vec<Gaussian3d> = (0..6)
            .map(|i| {
                let f = i as f32;
                Gaussian3d::new(
                    [0.1 * f - 0.3, 0.05 * f, 2.0 + 0.5 * f],
                    [0.3, 0.4, 0.2],
                    unit_quat(),
                    0.5,
                    [0.3, 0.5, 0.7],
                )
                .expect("value should be present")
            })
            .collect();
        let img = rasterize(&scene, &cam).expect("rasterize should succeed");
        assert_eq!(img.rgb.len(), 16 * 16 * 3);
        assert!(img.rgb.iter().all(|v| v.is_finite()));
        assert!(img.alpha.iter().all(|v| v.is_finite()));
        assert!(img.depth.iter().all(|v| v.is_finite()));
    }
}
