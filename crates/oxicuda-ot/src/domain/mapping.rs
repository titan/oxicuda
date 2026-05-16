//! Barycentric mapping for OT-based domain adaptation.
//!
//! Given a transport plan `P ∈ ℝ^{m × n}` and a target dataset
//! `Y ∈ ℝ^{n × dim}`, the barycentric map sends each source point `x_i` to
//!
//! ```text
//! T(x_i) = Σ_j (P_ij / Σ_k P_ik) · y_j .
//! ```
//!
//! This is the conditional expectation of the target under the row-`i` slice
//! of the joint plan and is the natural way to convert a soft OT coupling
//! back into a deterministic map.

use crate::error::{OtError, OtResult};
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Output of `ot_adapt`: the mapped source dataset and the OT plan that
/// produced it.
#[derive(Debug, Clone)]
pub struct OtAdaptResult {
    /// Mapped source samples in the target space, length `m · dim`, row-major.
    pub mapped_source: Vec<f32>,
    /// Transport plan, length `m · n`, row-major.
    pub plan: Vec<f32>,
}

/// Threshold below which a row sum is treated as numerically zero.
const TINY_ROW_SUM: f32 = 1e-12;

/// Apply the barycentric map induced by `plan` to a target dataset `target_y`.
///
/// `plan.len()` must equal `m · n`; `target_y.len()` must equal `n · dim`.
/// Returns the mapped source coordinates of shape `[m × dim]` row-major.
/// If a row of the plan has near-zero mass, the corresponding mapped row is
/// set to the mean of `target_y` so the output remains well-defined.
pub fn barycentric_map(
    plan: &[f32],
    target_y: &[f32],
    m: usize,
    n: usize,
    dim: usize,
) -> OtResult<Vec<f32>> {
    if m == 0 || n == 0 || dim == 0 {
        return Err(OtError::EmptyInput);
    }
    if plan.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: plan.len(),
            b_len: m * n,
        });
    }
    if target_y.len() != n * dim {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: dim,
            a_len: target_y.len(),
            b_len: n * dim,
        });
    }
    for &p in plan {
        if !p.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite plan entry".to_string(),
            });
        }
        if p < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    for &y in target_y {
        if !y.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite target sample".to_string(),
            });
        }
    }

    // Pre-compute mean of `target_y` (used as fallback for empty rows).
    let mut y_mean = vec![0.0_f32; dim];
    for j in 0..n {
        for d in 0..dim {
            y_mean[d] += target_y[j * dim + d];
        }
    }
    for v in y_mean.iter_mut() {
        *v /= n as f32;
    }

    let mut mapped = vec![0.0_f32; m * dim];
    for i in 0..m {
        let row_off = i * n;
        let mut row_sum = 0.0_f32;
        for j in 0..n {
            row_sum += plan[row_off + j];
        }
        if row_sum <= TINY_ROW_SUM {
            for d in 0..dim {
                mapped[i * dim + d] = y_mean[d];
            }
            continue;
        }
        let inv = 1.0 / row_sum;
        for j in 0..n {
            let w = plan[row_off + j] * inv;
            if w == 0.0 {
                continue;
            }
            let y_off = j * dim;
            let map_off = i * dim;
            for d in 0..dim {
                mapped[map_off + d] += w * target_y[y_off + d];
            }
        }
    }
    Ok(mapped)
}

/// Run Sinkhorn-Knopp OT between source `X` and target `Y` and apply the
/// barycentric mapping to obtain the adapted source.
///
/// `source_x` has length `m · dim`, `target_y` length `n · dim`, both
/// row-major. `a` and `b` are the source/target empirical weight histograms
/// (typically uniform `1/m` and `1/n`).
pub fn ot_adapt(
    source_x: &[f32],
    target_y: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    dim: usize,
    sinkhorn_cfg: &SinkhornConfig,
) -> OtResult<OtAdaptResult> {
    if m == 0 || n == 0 || dim == 0 {
        return Err(OtError::EmptyInput);
    }
    if source_x.len() != m * dim {
        return Err(OtError::MarginalMismatch {
            m,
            n: dim,
            a_len: source_x.len(),
            b_len: m * dim,
        });
    }
    if target_y.len() != n * dim {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: dim,
            a_len: target_y.len(),
            b_len: n * dim,
        });
    }

    // Build squared Euclidean cost C_ij = ‖x_i − y_j‖².
    let mut cost = vec![0.0_f32; m * n];
    for i in 0..m {
        let xi = i * dim;
        let row_off = i * n;
        for j in 0..n {
            let yj = j * dim;
            let mut acc = 0.0_f32;
            for d in 0..dim {
                let diff = source_x[xi + d] - target_y[yj + d];
                acc += diff * diff;
            }
            cost[row_off + j] = acc;
        }
    }

    let result = sinkhorn(&cost, a, b, m, n, sinkhorn_cfg)?;
    let mapped_source = barycentric_map(&result.plan, target_y, m, n, dim)?;
    Ok(OtAdaptResult {
        mapped_source,
        plan: result.plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn identity_plan_returns_target() {
        // m == n diagonal plan with row mass 1/m should map x_i → y_i.
        let m = 3;
        let n = 3;
        let dim = 2;
        let mut plan = vec![0.0_f32; m * n];
        for i in 0..m {
            plan[i * n + i] = 1.0 / m as f32;
        }
        let target_y = vec![0.0_f32, 0.0, 1.0, 2.0, -1.0, 4.0];
        let mapped = barycentric_map(&plan, &target_y, m, n, dim).expect("ok");
        for (k, (m_v, t_v)) in mapped.iter().zip(target_y.iter()).enumerate() {
            assert!(
                approx(*m_v, *t_v, 1e-5),
                "entry {k}: mapped {m_v} target {t_v}"
            );
        }
    }

    #[test]
    fn uniform_plan_returns_mean_of_target() {
        let m = 2;
        let n = 4;
        let dim = 2;
        let plan = vec![1.0_f32 / (m * n) as f32; m * n];
        let target_y = vec![1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let mapped = barycentric_map(&plan, &target_y, m, n, dim).expect("ok");
        for i in 0..m {
            assert!(approx(mapped[i * dim], 0.0, 1e-5));
            assert!(approx(mapped[i * dim + 1], 0.0, 1e-5));
        }
    }

    #[test]
    fn output_shape_correct() {
        let m = 5;
        let n = 7;
        let dim = 3;
        let plan = vec![1.0_f32 / (m * n) as f32; m * n];
        let target_y = vec![0.5_f32; n * dim];
        let mapped = barycentric_map(&plan, &target_y, m, n, dim).expect("ok");
        assert_eq!(mapped.len(), m * dim);
    }

    #[test]
    fn empty_row_falls_back_to_mean() {
        let m = 2;
        let n = 2;
        let dim = 1;
        let plan = vec![0.0_f32, 0.0, 0.5, 0.5];
        let target_y = vec![0.0_f32, 4.0];
        let mapped = barycentric_map(&plan, &target_y, m, n, dim).expect("ok");
        assert!(approx(mapped[0], 2.0, 1e-5));
        assert!(approx(mapped[1], 2.0, 1e-5));
    }

    #[test]
    fn rejects_shape_mismatch() {
        let target_y = vec![0.0_f32; 4];
        let plan = vec![0.0_f32; 5];
        let res = barycentric_map(&plan, &target_y, 2, 2, 2);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn rejects_empty_input() {
        let res = barycentric_map(&[], &[], 0, 0, 0);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn rejects_negative_plan_entry() {
        let plan = vec![0.5_f32, -0.1, 0.5, 0.1];
        let target_y = vec![0.0_f32, 1.0];
        let res = barycentric_map(&plan, &target_y, 2, 2, 1);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn ot_adapt_translation_recovery() {
        // Translate target by (+5, 0) — the mapping should approximately move
        // every source x toward x + 5 along axis 0.
        let m = 3;
        let n = 3;
        let dim = 2;
        let source_x = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let target_y = vec![5.0_f32, 0.0, 6.0, 0.0, 5.0, 1.0];
        let a = vec![1.0_f32 / m as f32; m];
        let b = vec![1.0_f32 / n as f32; n];
        let cfg = SinkhornConfig {
            eps: 0.05,
            max_iter: 5000,
            tol: 1e-4,
        };
        let res = ot_adapt(&source_x, &target_y, &a, &b, m, n, dim, &cfg).expect("ok");
        assert_eq!(res.mapped_source.len(), m * dim);
        // Each mapped point should land near the matching translated target.
        for i in 0..m {
            let mx = res.mapped_source[i * dim];
            let my = res.mapped_source[i * dim + 1];
            assert!(mx > 4.5 && mx < 6.5, "mapped x[{i}] = {mx}");
            assert!(my.abs() <= 1.05, "mapped y[{i}] = {my}");
        }
    }

    #[test]
    fn ot_adapt_rejects_shape_mismatch() {
        let cfg = SinkhornConfig::default();
        let res = ot_adapt(
            &[0.0_f32; 5],
            &[0.0_f32; 6],
            &[0.5_f32; 2],
            &[1.0_f32 / 3.0; 3],
            2,
            3,
            2,
            &cfg,
        );
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }
}
