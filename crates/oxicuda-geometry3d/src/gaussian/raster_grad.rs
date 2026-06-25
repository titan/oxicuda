//! Differentiable 2D Gaussian rasterization (forward + analytic backward).
//!
//! Reference: Kerbl et al., SIGGRAPH 2023, supplementary — the analytic
//! gradients of the alpha-compositing splatting renderer ("de-rendering").
//!
//! The forward composites a depth-ordered list of 2D splats front-to-back:
//!
//! ```text
//! G_i = exp(-½ · d_i^T Σ_i⁻¹ d_i),   d_i = p - μ_i
//! w_i = α_i · G_i                     (in [0, 1) since α_i ∈ [0,1], G_i ≤ 1)
//! C   = Σ_i T_i · w_i · c_i,          T_i = Π_{j<i} (1 - w_j)
//! ```
//!
//! [`rasterize_forward_2d`] returns the image and the saved final transmittance
//! per pixel. [`rasterize_backward_2d`] consumes an upstream image gradient and
//! produces, per splat, gradients w.r.t. color, opacity `α`, 2D mean `μ`, and
//! the 2D covariance `Σ` (via the conic `Σ⁻¹`). The backward sweep runs
//! back-to-front, reconstructing the "suffix" color the standard Inria way.
//!
//! This forward intentionally does **not** clamp `w_i`, so it is smooth and the
//! analytic gradients match finite differences (verified in the unit tests).

use crate::error::{Geom3dError, Geom3dResult};

/// A differentiable 2D Gaussian splat (conic parameterisation).
#[derive(Debug, Clone, Copy)]
pub struct DiffSplat2d {
    /// Center `μ = (μx, μy)` in pixel coordinates.
    pub mean: [f32; 2],
    /// Symmetric positive-definite 2×2 covariance `Σ`, row-major `[a, b, b, d]`.
    pub cov: [f32; 4],
    /// RGB color.
    pub color: [f32; 3],
    /// Opacity `α ∈ [0, 1]`.
    pub opacity: f32,
    /// Depth used for front-to-back ordering (smaller = nearer).
    pub depth: f32,
}

/// Per-splat gradients produced by the backward pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplatGrad {
    /// dL/d(color), one entry per RGB channel.
    pub d_color: [f32; 3],
    /// dL/d(opacity).
    pub d_opacity: f32,
    /// dL/d(mean), pixel-space.
    pub d_mean: [f32; 2],
    /// dL/d(cov), row-major 2×2 (symmetric: `[0]`, `[1]==[2]`, `[3]`).
    pub d_cov: [f32; 4],
}

/// Saved forward state needed for the backward pass.
#[derive(Debug, Clone)]
pub struct ForwardCache {
    /// Final per-pixel transmittance `T_final` after all splats.
    pub final_t: Vec<f32>,
    /// Depth-sorted splat order (front-to-back) used by the forward.
    pub order: Vec<usize>,
    /// Image width / height.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

fn invert2x2(m: &[f32; 4]) -> Option<[f32; 4]> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv])
}

fn depth_order(splats: &[DiffSplat2d]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..splats.len()).collect();
    order.sort_by(|&i, &j| {
        splats[i]
            .depth
            .partial_cmp(&splats[j].depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Forward rasterization. Returns `(image, cache)` where `image` is an
/// interleaved channels-last RGB buffer of length `width·height·3`.
///
/// # Errors
///
/// Returns [`Geom3dError::InvalidCovariance`] if any splat covariance is
/// singular (non-invertible).
pub fn rasterize_forward_2d(
    splats: &[DiffSplat2d],
    width: usize,
    height: usize,
) -> Geom3dResult<(Vec<f32>, ForwardCache)> {
    for s in splats {
        if invert2x2(&s.cov).is_none() {
            return Err(Geom3dError::InvalidCovariance {
                reason: "splat covariance is singular",
            });
        }
    }
    let mut image = vec![0.0_f32; width * height * 3];
    let mut t_buf = vec![1.0_f32; width.max(1) * height.max(1)];
    let order = depth_order(splats);
    if width == 0 || height == 0 {
        return Ok((
            image,
            ForwardCache {
                final_t: vec![1.0; width * height],
                order,
                width,
                height,
            },
        ));
    }

    for &si in &order {
        let s = &splats[si];
        let inv = match invert2x2(&s.cov) {
            Some(m) => m,
            None => continue,
        };
        for py in 0..height {
            let dy = py as f32 + 0.5 - s.mean[1];
            for px in 0..width {
                let pix = py * width + px;
                let t = t_buf[pix];
                if t < 1e-6 {
                    continue;
                }
                let dx = px as f32 + 0.5 - s.mean[0];
                let power = inv[0] * dx * dx + (inv[1] + inv[2]) * dx * dy + inv[3] * dy * dy;
                let g = (-0.5 * power).exp();
                let w = s.opacity * g;
                let base = pix * 3;
                image[base] += t * w * s.color[0];
                image[base + 1] += t * w * s.color[1];
                image[base + 2] += t * w * s.color[2];
                t_buf[pix] = t * (1.0 - w);
            }
        }
    }

    let cache = ForwardCache {
        final_t: t_buf,
        order,
        width,
        height,
    };
    Ok((image, cache))
}

/// Backward pass. Given the upstream gradient `d_image` (same layout as the
/// forward image), returns one [`SplatGrad`] per input splat (indexed exactly
/// like `splats`).
///
/// The sweep runs back-to-front, maintaining the suffix accumulated color
/// `acc_c` so that `dC/dw_i = T_i·(c_i − acc_c_after_i)` matches the Inria
/// derivation.
///
/// # Errors
///
/// Returns [`Geom3dError::DimensionMismatch`] if `d_image` is the wrong length
/// or [`Geom3dError::InvalidCovariance`] if a covariance is singular.
pub fn rasterize_backward_2d(
    splats: &[DiffSplat2d],
    d_image: &[f32],
    cache: &ForwardCache,
) -> Geom3dResult<Vec<SplatGrad>> {
    let (w, h) = (cache.width, cache.height);
    if d_image.len() != w * h * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: w * h * 3,
            got: d_image.len(),
        });
    }
    let mut grads = vec![SplatGrad::default(); splats.len()];
    if w == 0 || h == 0 {
        return Ok(grads);
    }

    // Pre-compute inverse covariances.
    let mut inv_covs = vec![[0.0_f32; 4]; splats.len()];
    for (i, s) in splats.iter().enumerate() {
        inv_covs[i] = invert2x2(&s.cov).ok_or(Geom3dError::InvalidCovariance {
            reason: "splat covariance is singular",
        })?;
    }

    // Per-pixel suffix color accumulator (color contributed by splats *behind*
    // the current one) and the running transmittance reconstructed forward.
    // We process pixels independently; for each pixel we walk the order once
    // forward to recover T_i, then once backward to accumulate.
    let order = &cache.order;
    let m = order.len();

    // Scratch reused per pixel.
    let mut t_at = vec![0.0_f32; m]; // transmittance entering splat order[k]
    let mut w_at = vec![0.0_f32; m]; // blend weight of splat order[k]

    for py in 0..h {
        for px in 0..w {
            let pix = py * w + px;
            let base = pix * 3;
            let dl = [d_image[base], d_image[base + 1], d_image[base + 2]];
            if dl[0] == 0.0 && dl[1] == 0.0 && dl[2] == 0.0 {
                continue;
            }

            // Forward to recover T and w at each splat for this pixel.
            let mut t = 1.0_f32;
            for (k, &si) in order.iter().enumerate() {
                let s = &splats[si];
                let inv = &inv_covs[si];
                let dx = px as f32 + 0.5 - s.mean[0];
                let dy = py as f32 + 0.5 - s.mean[1];
                let power = inv[0] * dx * dx + (inv[1] + inv[2]) * dx * dy + inv[3] * dy * dy;
                let g = (-0.5 * power).exp();
                let weight = s.opacity * g;
                t_at[k] = t;
                w_at[k] = weight;
                t *= 1.0 - weight;
            }

            // Backward: suffix color (colors of splats strictly behind k).
            let mut acc_c = [0.0_f32; 3];
            for k in (0..m).rev() {
                let si = order[k];
                let s = &splats[si];
                let inv = &inv_covs[si];
                let t_i = t_at[k];
                let weight = w_at[k];
                let g = if s.opacity.abs() > 1e-20 {
                    weight / s.opacity
                } else {
                    let dx = px as f32 + 0.5 - s.mean[0];
                    let dy = py as f32 + 0.5 - s.mean[1];
                    let power = inv[0] * dx * dx + (inv[1] + inv[2]) * dx * dy + inv[3] * dy * dy;
                    (-0.5 * power).exp()
                };

                // dL/dcolor_i = dl · T_i · w_i
                let g_color = &mut grads[si];
                g_color.d_color[0] += dl[0] * t_i * weight;
                g_color.d_color[1] += dl[1] * t_i * weight;
                g_color.d_color[2] += dl[2] * t_i * weight;

                // dC/dw_i = T_i·(c_i − acc_c)  ⇒  dL/dw_i = dl·that
                let dchan = [
                    t_i * (s.color[0] - acc_c[0]),
                    t_i * (s.color[1] - acc_c[1]),
                    t_i * (s.color[2] - acc_c[2]),
                ];
                let dl_dw = dl[0] * dchan[0] + dl[1] * dchan[1] + dl[2] * dchan[2];

                // w = α·G  ⇒  dL/dα = dl_dw·G ; dL/dG = dl_dw·α
                g_color.d_opacity += dl_dw * g;
                let dl_dg = dl_dw * s.opacity;

                // G = exp(-½ power) ⇒ dG = -½·G·d(power)
                let dl_dpower = dl_dg * (-0.5) * g;
                let dx = px as f32 + 0.5 - s.mean[0];
                let dy = py as f32 + 0.5 - s.mean[1];
                let (a, b, d) = (inv[0], inv[1], inv[3]); // inv symmetric, inv[1]==inv[2]
                // power = a·dx² + 2b·dx·dy + d·dy²
                // d(power)/dμx = -(2a·dx + 2b·dy);  d/dμy = -(2b·dx + 2d·dy)
                g_color.d_mean[0] += dl_dpower * (-(2.0 * a * dx + 2.0 * b * dy));
                g_color.d_mean[1] += dl_dpower * (-(2.0 * b * dx + 2.0 * d * dy));

                // d(power)/d(inv): power = inv·outer(d,d) with symmetry.
                // ∂power/∂inv00 = dx²; ∂/∂inv11 = dy²; ∂/∂inv01 = 2·dx·dy
                let dpow_dinv = [dx * dx, 2.0 * dx * dy, dy * dy]; // [a, b(off), d]
                let dl_dinv_a = dl_dpower * dpow_dinv[0];
                let dl_dinv_b = dl_dpower * dpow_dinv[1];
                let dl_dinv_d = dl_dpower * dpow_dinv[2];

                // Chain through Σ⁻¹ = adj(Σ)/det(Σ) to Σ.
                // For symmetric 2×2 Σ=[[A,B],[B,D]], inv=[[D,-B],[-B,A]]/det,
                // det=AD-B². The closed-form ∂inv/∂Σ contracted with the
                // upstream conic gradient yields:
                let (cov_a, cov_b, cov_d) = (s.cov[0], s.cov[1], s.cov[3]);
                let det = cov_a * cov_d - cov_b * cov_b;
                if det.abs() > 1e-20 {
                    let inv_det = 1.0 / det;
                    // Using d(inv) = -inv · d(Σ) · inv (matrix identity),
                    // contract <dL_dinv, d(inv)> = -<inv·dL_dinv·inv, dΣ>.
                    // Build symmetric dL_dinv matrix M = [[a, b],[b, d]].
                    let m00 = dl_dinv_a;
                    let m01 = dl_dinv_b * 0.5; // split off-diagonal across (0,1),(1,0)
                    let m11 = dl_dinv_d;
                    // P = inv · M · inv, inv = [[D,-B],[-B,A]]·inv_det
                    let i00 = cov_d * inv_det;
                    let i01 = -cov_b * inv_det;
                    let i11 = cov_a * inv_det;
                    // tmp = M · inv
                    let t00 = m00 * i00 + m01 * i01;
                    let t01 = m00 * i01 + m01 * i11;
                    let t10 = m01 * i00 + m11 * i01;
                    let t11 = m01 * i01 + m11 * i11;
                    // P = inv · tmp
                    let p00 = i00 * t00 + i01 * t10;
                    let p01 = i00 * t01 + i01 * t11;
                    let p11 = i01 * t01 + i11 * t11;
                    // dL/dΣ = -P  (symmetric: store [0],[1]=[2],[3])
                    g_color.d_cov[0] += -p00;
                    g_color.d_cov[1] += -p01;
                    g_color.d_cov[2] += -p01;
                    g_color.d_cov[3] += -p11;
                }

                // Fold this splat's contribution into the suffix color so the
                // splat in front sees the correct (c_i − acc_c) difference.
                acc_c[0] = weight * s.color[0] + (1.0 - weight) * acc_c[0];
                acc_c[1] = weight * s.color[1] + (1.0 - weight) * acc_c[1];
                acc_c[2] = weight * s.color[2] + (1.0 - weight) * acc_c[2];
            }
        }
    }

    Ok(grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_splats() -> Vec<DiffSplat2d> {
        vec![
            DiffSplat2d {
                mean: [3.4, 4.1],
                cov: [3.0, 0.6, 0.6, 2.2],
                color: [0.7, 0.2, 0.4],
                opacity: 0.6,
                depth: 1.0,
            },
            DiffSplat2d {
                mean: [4.6, 3.7],
                cov: [2.5, -0.4, -0.4, 3.1],
                color: [0.1, 0.8, 0.3],
                opacity: 0.5,
                depth: 2.0,
            },
            DiffSplat2d {
                mean: [4.0, 4.0],
                cov: [4.0, 0.0, 0.0, 4.0],
                color: [0.3, 0.3, 0.9],
                opacity: 0.4,
                depth: 3.0,
            },
        ]
    }

    fn scalar_loss(img: &[f32], weights: &[f32]) -> f32 {
        img.iter().zip(weights.iter()).map(|(a, b)| a * b).sum()
    }

    /// Build a deterministic but irregular upstream gradient.
    fn loss_weights(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.37).sin() * 0.5 + 0.5) + 0.1)
            .collect()
    }

    #[test]
    fn forward_shapes_and_finite() {
        let splats = sample_splats();
        let (img, cache) = rasterize_forward_2d(&splats, 8, 8).expect("forward should succeed");
        assert_eq!(img.len(), 8 * 8 * 3);
        assert_eq!(cache.final_t.len(), 8 * 8);
        assert!(img.iter().all(|v| v.is_finite()));
        assert!(cache.final_t.iter().all(|t| (0.0..=1.0).contains(t)));
    }

    #[test]
    fn singular_cov_rejected() {
        let mut s = sample_splats();
        s[0].cov = [1.0, 1.0, 1.0, 1.0]; // det 0
        assert!(rasterize_forward_2d(&s, 4, 4).is_err());
    }

    #[test]
    fn grad_color_matches_numeric() {
        let (w, h) = (8usize, 8usize);
        let splats = sample_splats();
        let dl = loss_weights(w * h * 3);
        let (_, cache) = rasterize_forward_2d(&splats, w, h).expect("forward should succeed");
        let grads = rasterize_backward_2d(&splats, &dl, &cache).expect("backward should succeed");

        let eps = 1e-3_f32;
        for si in 0..splats.len() {
            for ch in 0..3 {
                let mut sp = splats.clone();
                sp[si].color[ch] += eps;
                let (ip, _) = rasterize_forward_2d(&sp, w, h).expect("forward should succeed");
                let mut sm = splats.clone();
                sm[si].color[ch] -= eps;
                let (im, _) = rasterize_forward_2d(&sm, w, h).expect("forward should succeed");
                let num = (scalar_loss(&ip, &dl) - scalar_loss(&im, &dl)) / (2.0 * eps);
                let ana = grads[si].d_color[ch];
                assert!(
                    (num - ana).abs() < 1e-2 * (1.0 + num.abs()),
                    "d_color[{si}][{ch}]: num {num} vs ana {ana}"
                );
            }
        }
    }

    #[test]
    fn grad_opacity_matches_numeric() {
        let (w, h) = (8usize, 8usize);
        let splats = sample_splats();
        let dl = loss_weights(w * h * 3);
        let (_, cache) = rasterize_forward_2d(&splats, w, h).expect("forward should succeed");
        let grads = rasterize_backward_2d(&splats, &dl, &cache).expect("backward should succeed");

        let eps = 1e-3_f32;
        for si in 0..splats.len() {
            let mut sp = splats.clone();
            sp[si].opacity += eps;
            let (ip, _) = rasterize_forward_2d(&sp, w, h).expect("forward should succeed");
            let mut sm = splats.clone();
            sm[si].opacity -= eps;
            let (im, _) = rasterize_forward_2d(&sm, w, h).expect("forward should succeed");
            let num = (scalar_loss(&ip, &dl) - scalar_loss(&im, &dl)) / (2.0 * eps);
            let ana = grads[si].d_opacity;
            assert!(
                (num - ana).abs() < 2e-2 * (1.0 + num.abs()),
                "d_opacity[{si}]: num {num} vs ana {ana}"
            );
        }
    }

    #[test]
    fn grad_mean_matches_numeric() {
        let (w, h) = (10usize, 10usize);
        let splats = sample_splats();
        let dl = loss_weights(w * h * 3);
        let (_, cache) = rasterize_forward_2d(&splats, w, h).expect("forward should succeed");
        let grads = rasterize_backward_2d(&splats, &dl, &cache).expect("backward should succeed");

        let eps = 1e-3_f32;
        for si in 0..splats.len() {
            for axis in 0..2 {
                let mut sp = splats.clone();
                sp[si].mean[axis] += eps;
                let (ip, _) = rasterize_forward_2d(&sp, w, h).expect("forward should succeed");
                let mut sm = splats.clone();
                sm[si].mean[axis] -= eps;
                let (im, _) = rasterize_forward_2d(&sm, w, h).expect("forward should succeed");
                let num = (scalar_loss(&ip, &dl) - scalar_loss(&im, &dl)) / (2.0 * eps);
                let ana = grads[si].d_mean[axis];
                assert!(
                    (num - ana).abs() < 3e-2 * (1.0 + num.abs()),
                    "d_mean[{si}][{axis}]: num {num} vs ana {ana}"
                );
            }
        }
    }

    #[test]
    fn grad_cov_matches_numeric() {
        let (w, h) = (10usize, 10usize);
        let splats = sample_splats();
        let dl = loss_weights(w * h * 3);
        let (_, cache) = rasterize_forward_2d(&splats, w, h).expect("forward should succeed");
        let grads = rasterize_backward_2d(&splats, &dl, &cache).expect("backward should succeed");

        let eps = 2e-3_f32;
        // Perturb the symmetric covariance entries (A, B, D) and compare.
        for si in 0..splats.len() {
            // Diagonal A.
            for &(ia, ib) in &[(0usize, 0usize), (1, 2), (3, 3)] {
                let mut sp = splats.clone();
                sp[si].cov[ia] += eps;
                if ib != ia {
                    sp[si].cov[ib] += eps;
                }
                let (ip, _) = rasterize_forward_2d(&sp, w, h).expect("forward should succeed");
                let mut sm = splats.clone();
                sm[si].cov[ia] -= eps;
                if ib != ia {
                    sm[si].cov[ib] -= eps;
                }
                let (im, _) = rasterize_forward_2d(&sm, w, h).expect("forward should succeed");
                let num = (scalar_loss(&ip, &dl) - scalar_loss(&im, &dl)) / (2.0 * eps);
                // Analytic grad summed over the perturbed entries.
                let ana = if ia == ib {
                    grads[si].d_cov[ia]
                } else {
                    grads[si].d_cov[ia] + grads[si].d_cov[ib]
                };
                assert!(
                    (num - ana).abs() < 5e-2 * (1.0 + num.abs()),
                    "d_cov[{si}] entry ({ia},{ib}): num {num} vs ana {ana}"
                );
            }
        }
    }

    #[test]
    fn zero_upstream_zero_grad() {
        let (w, h) = (6usize, 6usize);
        let splats = sample_splats();
        let dl = vec![0.0_f32; w * h * 3];
        let (_, cache) = rasterize_forward_2d(&splats, w, h).expect("forward should succeed");
        let grads = rasterize_backward_2d(&splats, &dl, &cache).expect("backward should succeed");
        for g in &grads {
            assert!(g.d_opacity.abs() < 1e-12);
            assert!(g.d_mean.iter().all(|v| v.abs() < 1e-12));
            assert!(g.d_color.iter().all(|v| v.abs() < 1e-12));
            assert!(g.d_cov.iter().all(|v| v.abs() < 1e-12));
        }
    }

    #[test]
    fn backward_wrong_dimension_errors() {
        let splats = sample_splats();
        let (_, cache) = rasterize_forward_2d(&splats, 4, 4).expect("forward should succeed");
        let bad = vec![0.0_f32; 10];
        assert!(rasterize_backward_2d(&splats, &bad, &cache).is_err());
    }
}
