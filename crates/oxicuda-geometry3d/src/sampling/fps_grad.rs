//! Differentiable Farthest-Point Sampling via a straight-through estimator.
//!
//! FPS selects `m` of `n` points by an iterated hard `argmax` over distances,
//! which has zero gradient almost everywhere. The straight-through estimator
//! (STE) keeps the discrete selection in the **forward** pass but substitutes a
//! smooth surrogate Jacobian in the **backward** pass:
//!
//! * Forward — run ordinary [`farthest_point_sample`] and hard-gather the
//!   selected rows: `Y[k] = X[idx[k]]`.
//! * Backward — replace the one-hot selection matrix with a soft assignment
//!   `S[k, j] = softmax_j(-β · ‖p_j − p_{idx[k]}‖²)`, a temperature-`β`
//!   Gaussian kernel centred on the hard-selected point. Gradients flow as
//!   `dL/dX = Sᵀ · dL/dY`.
//!
//! At `β → ∞`, `S` collapses to the one-hot selection and the STE gradient
//! becomes the exact (sub-)gradient of the hard gather; for finite `β` it is a
//! smooth relaxation whose backward matches finite differences of the *soft*
//! gather `Y_soft = S · X` (verified in the unit tests).

use crate::error::{Geom3dError, Geom3dResult};
use crate::sampling::farthest_point_sample::farthest_point_sample;

/// Result of differentiable FPS: the hard indices plus the soft assignment used
/// for the straight-through backward pass.
#[derive(Debug, Clone)]
pub struct FpsSteResult {
    /// Hard-selected indices (length `m`), identical to plain FPS.
    pub indices: Vec<usize>,
    /// Row-major soft assignment `S` of shape `[m × n]`. Each row sums to 1 and
    /// peaks at the corresponding hard index.
    pub soft_weights: Vec<f32>,
    /// Number of selected points `m`.
    pub m: usize,
    /// Number of input points `n`.
    pub n: usize,
}

fn sq_dist(points: &[f32], i: usize, j: usize) -> f32 {
    let dx = points[i * 3] - points[j * 3];
    let dy = points[i * 3 + 1] - points[j * 3 + 1];
    let dz = points[i * 3 + 2] - points[j * 3 + 2];
    dx * dx + dy * dy + dz * dz
}

/// Run FPS and build the soft straight-through assignment.
///
/// `beta` is the inverse temperature of the soft kernel (must be `>= 0`,
/// finite). Larger `beta` → sharper (closer to one-hot).
///
/// # Errors
///
/// Propagates the errors of [`farthest_point_sample`], and returns
/// [`Geom3dError::InvalidRadius`] (re-used as a generic scalar-range error) if
/// `beta` is negative or non-finite.
pub fn fps_sample_with_grad(
    points: &[f32],
    n: usize,
    m: usize,
    beta: f32,
) -> Geom3dResult<FpsSteResult> {
    if !(beta >= 0.0 && beta.is_finite()) {
        return Err(Geom3dError::InvalidRadius { radius: beta });
    }
    let indices = farthest_point_sample(points, n, m)?;
    let mut soft = vec![0.0_f32; m * n];

    for (k, &center) in indices.iter().enumerate() {
        // Numerically-stable softmax of -β·d² over all n points.
        let mut max_logit = f32::NEG_INFINITY;
        for j in 0..n {
            let logit = -beta * sq_dist(points, center, j);
            if logit > max_logit {
                max_logit = logit;
            }
        }
        let mut sum = 0.0_f32;
        for j in 0..n {
            let e = (-beta * sq_dist(points, center, j) - max_logit).exp();
            soft[k * n + j] = e;
            sum += e;
        }
        let inv = 1.0 / (sum + 1e-20);
        for j in 0..n {
            soft[k * n + j] *= inv;
        }
    }

    Ok(FpsSteResult {
        indices,
        soft_weights: soft,
        m,
        n,
    })
}

/// Forward hard gather of `c`-dimensional features at the FPS indices.
///
/// `features` is row-major `[n × c]`. Returns `[m × c]` with
/// `Y[k] = features[idx[k]]`.
///
/// # Errors
///
/// Returns [`Geom3dError::DimensionMismatch`] if `features.len() != n · c`.
pub fn gather_ste_forward(
    result: &FpsSteResult,
    features: &[f32],
    c: usize,
) -> Geom3dResult<Vec<f32>> {
    if features.len() != result.n * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: result.n * c,
            got: features.len(),
        });
    }
    let mut out = vec![0.0_f32; result.m * c];
    for (k, &idx) in result.indices.iter().enumerate() {
        out[k * c..(k + 1) * c].copy_from_slice(&features[idx * c..(idx + 1) * c]);
    }
    Ok(out)
}

/// Soft gather `Y_soft = S · X` — the surrogate whose Jacobian the STE uses.
///
/// `features` is `[n × c]`, output is `[m × c]`. This is what the backward pass
/// differentiates; in a real training loop the forward would use
/// [`gather_ste_forward`] but the autograd Jacobian would be this one.
///
/// # Errors
///
/// Returns [`Geom3dError::DimensionMismatch`] on a feature-length mismatch.
pub fn gather_ste_soft(
    result: &FpsSteResult,
    features: &[f32],
    c: usize,
) -> Geom3dResult<Vec<f32>> {
    if features.len() != result.n * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: result.n * c,
            got: features.len(),
        });
    }
    let mut out = vec![0.0_f32; result.m * c];
    for k in 0..result.m {
        for j in 0..result.n {
            let w = result.soft_weights[k * result.n + j];
            if w == 0.0 {
                continue;
            }
            for ch in 0..c {
                out[k * c + ch] += w * features[j * c + ch];
            }
        }
    }
    Ok(out)
}

/// Straight-through backward: `dL/dX = Sᵀ · dL/dY`.
///
/// `d_output` is the upstream gradient on the gathered features `[m × c]`.
/// Returns `dL/dfeatures` `[n × c]`.
///
/// # Errors
///
/// Returns [`Geom3dError::DimensionMismatch`] if `d_output.len() != m · c`.
pub fn gather_ste_backward(
    result: &FpsSteResult,
    d_output: &[f32],
    c: usize,
) -> Geom3dResult<Vec<f32>> {
    if d_output.len() != result.m * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: result.m * c,
            got: d_output.len(),
        });
    }
    let mut d_features = vec![0.0_f32; result.n * c];
    for k in 0..result.m {
        for j in 0..result.n {
            let w = result.soft_weights[k * result.n + j];
            if w == 0.0 {
                continue;
            }
            for ch in 0..c {
                d_features[j * c + ch] += w * d_output[k * c + ch];
            }
        }
    }
    Ok(d_features)
}

/// Canonical (hard) straight-through backward for the **hard** gather
/// `Y[k] = X[idx[k]]`.
///
/// This is the textbook straight-through estimator (Bengio et al. 2013): the
/// forward keeps the non-differentiable hard selection, while the backward
/// treats the gather as the *identity* on the selected rows. Concretely it
/// scatters each upstream row `dL/dY[k]` straight back onto the input row
/// `idx[k]` and leaves every unselected input row at zero gradient:
///
/// ```text
/// dL/dX[idx[k]] = dL/dY[k]          (identity on the m selected rows)
/// dL/dX[j]      = 0   for j ∉ idx   (zero on unselected rows)
/// ```
///
/// Equivalently this is the `β → ∞` limit of [`gather_ste_backward`], where the
/// soft assignment `S` collapses to the one-hot selection matrix `Sᵀ`. If FPS
/// ever returned a duplicate index the contributions accumulate (`+=`), which
/// is the correct gradient of a gather that reads the same row twice; plain
/// [`farthest_point_sample`] returns distinct indices so in practice each
/// selected row receives exactly one upstream row.
///
/// # Errors
///
/// Returns [`Geom3dError::DimensionMismatch`] if `d_output.len() != m · c`.
pub fn gather_ste_hard_backward(
    result: &FpsSteResult,
    d_output: &[f32],
    c: usize,
) -> Geom3dResult<Vec<f32>> {
    if d_output.len() != result.m * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: result.m * c,
            got: d_output.len(),
        });
    }
    let mut d_features = vec![0.0_f32; result.n * c];
    for (k, &idx) in result.indices.iter().enumerate() {
        for ch in 0..c {
            d_features[idx * c + ch] += d_output[k * c + ch];
        }
    }
    Ok(d_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn grid_points(n: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(13);
        let mut pts = vec![0.0_f32; n * 3];
        for v in &mut pts {
            *v = rng.next_u32() as f32 / 4_294_967_296.0 * 4.0 - 2.0;
        }
        pts
    }

    fn feats(n: usize, c: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut f = vec![0.0_f32; n * c];
        for v in &mut f {
            *v = rng.next_u32() as f32 / 4_294_967_296.0 * 2.0 - 1.0;
        }
        f
    }

    #[test]
    fn indices_match_plain_fps() {
        let pts = grid_points(40);
        let res = fps_sample_with_grad(&pts, 40, 8, 5.0).expect("fps should succeed");
        let plain = farthest_point_sample(&pts, 40, 8).expect("fps should succeed");
        assert_eq!(res.indices, plain);
    }

    #[test]
    fn soft_rows_sum_to_one_and_peak_at_index() {
        let pts = grid_points(30);
        let res = fps_sample_with_grad(&pts, 30, 6, 8.0).expect("fps should succeed");
        for k in 0..res.m {
            let row = &res.soft_weights[k * res.n..(k + 1) * res.n];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {k} sums to {sum}");
            // The hard index must have the largest weight (its own distance 0).
            let idx = res.indices[k];
            let peak = row[idx];
            assert!(
                row.iter().all(|&w| w <= peak + 1e-6),
                "peak must be at hard index"
            );
        }
    }

    #[test]
    fn high_beta_approaches_one_hot() {
        let pts = grid_points(25);
        let res = fps_sample_with_grad(&pts, 25, 5, 1e4).expect("fps should succeed");
        for k in 0..res.m {
            let idx = res.indices[k];
            let w = res.soft_weights[k * res.n + idx];
            assert!(w > 0.99, "row {k} should be ~one-hot at high β, got {w}");
        }
    }

    #[test]
    fn forward_hard_gather_matches_indices() {
        let (n, c) = (20, 4);
        let pts = grid_points(n);
        let f = feats(n, c, 3);
        let res = fps_sample_with_grad(&pts, n, 5, 6.0).expect("fps should succeed");
        let y = gather_ste_forward(&res, &f, c).expect("gather should succeed");
        for (k, &idx) in res.indices.iter().enumerate() {
            for ch in 0..c {
                assert!((y[k * c + ch] - f[idx * c + ch]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ste_backward_matches_numeric_soft() {
        // The STE Jacobian equals the Jacobian of the soft gather Y_soft = S·X.
        let (n, c) = (16, 3);
        let pts = grid_points(n);
        let f = feats(n, c, 11);
        let res = fps_sample_with_grad(&pts, n, 4, 3.0).expect("fps should succeed");

        // Upstream gradient on the [m × c] output.
        let dout: Vec<f32> = (0..res.m * c)
            .map(|i| ((i as f32 * 0.7).cos() * 0.5 + 0.5) + 0.1)
            .collect();
        let dfeat = gather_ste_backward(&res, &dout, c).expect("backward should succeed");

        let loss = |feat: &[f32]| -> f32 {
            let y = gather_ste_soft(&res, feat, c).expect("soft gather should succeed");
            y.iter().zip(dout.iter()).map(|(a, b)| a * b).sum::<f32>()
        };
        let eps = 1e-3_f32;
        for i in 0..n * c {
            let mut fp = f.clone();
            fp[i] += eps;
            let mut fm = f.clone();
            fm[i] -= eps;
            let num = (loss(&fp) - loss(&fm)) / (2.0 * eps);
            let ana = dfeat[i];
            assert!(
                (num - ana).abs() < 1e-2 * (1.0 + num.abs()),
                "dfeat[{i}]: num {num} vs ana {ana}"
            );
        }
    }

    #[test]
    fn hard_ste_backward_is_identity_on_selected_zero_elsewhere() {
        // The canonical straight-through estimator: the gather is treated as the
        // identity on selected rows in the backward pass. So the gradient w.r.t.
        // each input point equals the upstream gradient of whichever output row
        // selected it, and is exactly zero for every unselected point.
        let (n, c) = (24, 3);
        let pts = grid_points(n);
        let m = 6;
        let res = fps_sample_with_grad(&pts, n, m, 4.0).expect("fps should succeed");

        // Distinct upstream gradient per output row so we can match rows ↔ inputs.
        let dout: Vec<f32> = (0..m * c).map(|i| (i as f32 + 1.0) * 0.5 - 1.0).collect();
        let dfeat = gather_ste_hard_backward(&res, &dout, c).expect("backward should succeed");

        // Shape matches the input point/feature set [n × c].
        assert_eq!(dfeat.len(), n * c);

        // Which input rows are selected (FPS returns distinct indices).
        let mut selected = res.indices.clone();
        selected.sort_unstable();
        selected.dedup();
        assert_eq!(selected.len(), m, "FPS indices must be distinct");

        // Identity on selected rows: dfeat[idx[k]] == dout[k] channel-wise.
        for (k, &idx) in res.indices.iter().enumerate() {
            for ch in 0..c {
                assert!(
                    (dfeat[idx * c + ch] - dout[k * c + ch]).abs() < 1e-9,
                    "selected row {idx} ch {ch}: {} != {}",
                    dfeat[idx * c + ch],
                    dout[k * c + ch]
                );
            }
        }

        // Zero on every unselected row.
        for j in 0..n {
            if res.indices.contains(&j) {
                continue;
            }
            for ch in 0..c {
                assert!(
                    dfeat[j * c + ch].abs() < 1e-12,
                    "unselected row {j} ch {ch} must be zero, got {}",
                    dfeat[j * c + ch]
                );
            }
        }
    }

    #[test]
    fn hard_ste_is_high_beta_limit_of_soft() {
        // As β → ∞ the soft-STE backward converges to the hard-STE backward.
        let (n, c) = (18, 2);
        let pts = grid_points(n);
        let m = 5;
        let res = fps_sample_with_grad(&pts, n, m, 1e5).expect("fps should succeed");
        let dout: Vec<f32> = (0..m * c).map(|i| (i as f32 * 0.31).sin()).collect();
        let soft = gather_ste_backward(&res, &dout, c).expect("soft backward should succeed");
        let hard = gather_ste_hard_backward(&res, &dout, c).expect("hard backward should succeed");
        for (s, h) in soft.iter().zip(hard.iter()) {
            assert!((s - h).abs() < 1e-2, "soft {s} should approach hard {h}");
        }
    }

    #[test]
    fn hard_ste_dimension_mismatch_errors() {
        let pts = grid_points(10);
        let res = fps_sample_with_grad(&pts, 10, 3, 2.0).expect("fps should succeed");
        assert!(gather_ste_hard_backward(&res, &[0.0; 5], 4).is_err());
    }

    #[test]
    fn invalid_beta_errors() {
        let pts = grid_points(10);
        assert!(fps_sample_with_grad(&pts, 10, 3, -1.0).is_err());
        assert!(fps_sample_with_grad(&pts, 10, 3, f32::NAN).is_err());
    }

    #[test]
    fn dimension_mismatch_errors() {
        let pts = grid_points(10);
        let res = fps_sample_with_grad(&pts, 10, 3, 2.0).expect("fps should succeed");
        assert!(gather_ste_forward(&res, &[0.0; 5], 4).is_err());
        assert!(gather_ste_backward(&res, &[0.0; 5], 4).is_err());
    }
}
