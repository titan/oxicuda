//! Earth Mover's Distance (Sinkhorn optimal transport) between point clouds.

use crate::error::{Geom3dError, Geom3dResult};

/// Configuration for the Sinkhorn algorithm.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    pub epsilon: f32,
    pub n_iter: usize,
    pub tol: f32,
}

/// Log-sum-exp over a row of a matrix.
fn log_sum_exp_row(log_m: &[f32], row: usize, ncols: usize, log_v: &[f32]) -> f32 {
    let mut vals: Vec<f32> = (0..ncols)
        .map(|j| log_m[row * ncols + j] + log_v[j])
        .collect();
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return f32::NEG_INFINITY;
    }
    for v in &mut vals {
        *v -= max_val;
    }
    let sum: f32 = vals.iter().map(|&v| v.exp()).sum();
    max_val + sum.ln()
}

/// Log-sum-exp over a column of a matrix.
fn log_sum_exp_col(log_m: &[f32], col: usize, nrows: usize, ncols: usize, log_u: &[f32]) -> f32 {
    let mut vals: Vec<f32> = (0..nrows)
        .map(|i| log_m[i * ncols + col] + log_u[i])
        .collect();
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return f32::NEG_INFINITY;
    }
    for v in &mut vals {
        *v -= max_val;
    }
    let sum: f32 = vals.iter().map(|&v| v.exp()).sum();
    max_val + sum.ln()
}

/// Entropy-regularized optimal transport (Sinkhorn) distance.
///
/// Builds cost matrix `C[i,j] = ||a_i - b_j||²`.
/// Log-domain Sinkhorn iteration.
/// Returns `EmdDidNotConverge` if any NaN/Inf detected.
pub fn earth_movers_distance(
    a: &[f32],
    na: usize,
    b: &[f32],
    nb: usize,
    cfg: &SinkhornConfig,
) -> Geom3dResult<f32> {
    if na == 0 || nb == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if a.len() != na * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: na * 3,
            got: a.len(),
        });
    }
    if b.len() != nb * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: nb * 3,
            got: b.len(),
        });
    }

    let eps = cfg.epsilon;
    let clamp = 50.0_f32;

    // Build log-kernel: log K[i,j] = -C[i,j] / eps
    let mut log_k = vec![0.0_f32; na * nb];
    for i in 0..na {
        for j in 0..nb {
            let dx = a[i * 3] - b[j * 3];
            let dy = a[i * 3 + 1] - b[j * 3 + 1];
            let dz = a[i * 3 + 2] - b[j * 3 + 2];
            let c_ij = dx * dx + dy * dy + dz * dz;
            log_k[i * nb + j] = (-c_ij / eps).clamp(-clamp, clamp);
        }
    }

    // Initialize log-domain dual variables
    let log_inv_na = -(na as f32).ln();
    let log_inv_nb = -(nb as f32).ln();

    let mut log_u = vec![0.0_f32; na];
    let mut log_v = vec![0.0_f32; nb];

    for iter in 0..cfg.n_iter {
        let log_u_prev = log_u.clone();

        // Update log_u: log(1/na) - logsumexp(log_K + log_v)
        for (i, val) in log_u.iter_mut().enumerate() {
            let lse = log_sum_exp_row(&log_k, i, nb, &log_v);
            *val = (log_inv_na - lse).clamp(-clamp, clamp);
        }

        // Update log_v: log(1/nb) - logsumexp(log_Kt + log_u)
        for (j, val) in log_v.iter_mut().enumerate() {
            let lse = log_sum_exp_col(&log_k, j, na, nb, &log_u);
            *val = (log_inv_nb - lse).clamp(-clamp, clamp);
        }

        // Check for NaN/Inf
        let has_nan = log_u.iter().any(|v| !v.is_finite()) || log_v.iter().any(|v| !v.is_finite());
        if has_nan {
            return Err(Geom3dError::EmdDidNotConverge { iterations: iter });
        }

        // Convergence check
        let delta: f32 = log_u
            .iter()
            .zip(log_u_prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if delta < cfg.tol {
            break;
        }
    }

    // Compute transport plan and cost
    // W = Σ_ij T[i,j] * C[i,j], T[i,j] = exp(log_u[i] + log_K[i,j] + log_v[j])
    let mut total_cost = 0.0_f64;
    for i in 0..na {
        for j in 0..nb {
            let dx = a[i * 3] - b[j * 3];
            let dy = a[i * 3 + 1] - b[j * 3 + 1];
            let dz = a[i * 3 + 2] - b[j * 3 + 2];
            let c_ij = dx * dx + dy * dy + dz * dz;
            let log_t = log_u[i] + log_k[i * nb + j] + log_v[j];
            let t_ij = log_t.exp();
            total_cost += (t_ij * c_ij) as f64;
        }
    }

    let result = total_cost as f32;
    if !result.is_finite() {
        return Err(Geom3dError::EmdDidNotConverge {
            iterations: cfg.n_iter,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> SinkhornConfig {
        SinkhornConfig {
            epsilon: 0.1,
            n_iter: 100,
            tol: 1e-6,
        }
    }

    #[test]
    fn emd_self_near_zero() {
        let pts: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let cfg = default_cfg();
        let d = earth_movers_distance(&pts, 3, &pts, 3, &cfg)
            .expect("earth_movers_distance should succeed");
        // With entropy regularization, self-distance should be small but may not be exactly 0
        assert!(
            d >= 0.0 && d.is_finite(),
            "EMD self should be finite and >=0, got {d}"
        );
    }

    #[test]
    fn emd_nonnegative() {
        let a: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let b: Vec<f32> = vec![0.5, 0.0, 0.0, 2.0, 0.0, 0.0];
        let cfg = default_cfg();
        let d = earth_movers_distance(&a, 2, &b, 2, &cfg)
            .expect("earth_movers_distance should succeed");
        assert!(d >= 0.0);
    }

    #[test]
    fn emd_empty_error() {
        let cfg = default_cfg();
        assert!(earth_movers_distance(&[], 0, &[1.0, 0.0, 0.0], 1, &cfg).is_err());
    }

    #[test]
    fn emd_dim_mismatch_error() {
        let a: Vec<f32> = vec![0.0, 0.0]; // Wrong: not n*3
        let b: Vec<f32> = vec![1.0, 0.0, 0.0];
        let cfg = default_cfg();
        assert!(earth_movers_distance(&a, 1, &b, 1, &cfg).is_err());
    }

    #[test]
    fn emd_increases_with_separation() {
        let a: Vec<f32> = vec![0.0, 0.0, 0.0];
        let b_near: Vec<f32> = vec![0.1, 0.0, 0.0];
        let b_far: Vec<f32> = vec![5.0, 0.0, 0.0];
        let cfg = default_cfg();
        let d_near = earth_movers_distance(&a, 1, &b_near, 1, &cfg)
            .expect("earth_movers_distance should succeed");
        let d_far = earth_movers_distance(&a, 1, &b_far, 1, &cfg)
            .expect("earth_movers_distance should succeed");
        assert!(d_far > d_near, "EMD should increase with separation");
    }

    // ── Self-consistency suite ──────────────────────────────────────────────
    // Verifies the metric-like properties of the CPU Sinkhorn EMD on small,
    // deterministic problems WITHOUT depending on the external Python POT
    // library. (The "vs POT" check is intentionally left out — see TODO.md.)

    /// Deterministic seeded point cloud in [-1, 1]³ via the crate LCG. We avoid
    /// `rand`/`getrandom` per the SciRS2 policy and use a full-range ÷2³² map.
    fn lcg_cloud(n: usize, seed: u64) -> Vec<f32> {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(seed);
        let mut p = vec![0.0_f32; n * 3];
        for v in &mut p {
            *v = rng.next_u32() as f32 / 4_294_967_296.0 * 2.0 - 1.0;
        }
        p
    }

    fn translate(cloud: &[f32], d: [f32; 3]) -> Vec<f32> {
        cloud
            .chunks_exact(3)
            .flat_map(|p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]])
            .collect()
    }

    #[test]
    fn emd_is_symmetric() {
        // C[i,j] = ‖a_i − b_j‖² is symmetric under a↔b (the cost transposes and
        // the uniform marginals are equal), so the Sinkhorn objective must be
        // too. Check on several seed pairs of unequal sizes.
        let cfg = SinkhornConfig {
            epsilon: 0.05,
            n_iter: 400,
            tol: 1e-7,
        };
        for &(na, nb, sa, sb) in &[(5usize, 5usize, 1u64, 2u64), (4, 7, 11, 23), (6, 3, 5, 9)] {
            let a = lcg_cloud(na, sa);
            let b = lcg_cloud(nb, sb);
            let ab = earth_movers_distance(&a, na, &b, nb, &cfg).expect("ab should succeed");
            let ba = earth_movers_distance(&b, nb, &a, na, &cfg).expect("ba should succeed");
            assert!(
                (ab - ba).abs() < 1e-3 * (1.0 + ab.abs()),
                "EMD(a,b)={ab} must equal EMD(b,a)={ba}"
            );
        }
    }

    #[test]
    fn emd_self_is_near_zero_and_below_any_shift() {
        // With entropic regularisation EMD(a,a) is not *exactly* zero (the plan
        // spreads a little mass off the diagonal), but it must be small and,
        // crucially, strictly smaller than the distance to any translated copy
        // (identity-of-indiscernibles, relaxed). Shrinking ε drives it toward 0.
        let n = 8;
        let a = lcg_cloud(n, 7);
        for &eps in &[0.1_f32, 0.03, 0.01] {
            let cfg = SinkhornConfig {
                epsilon: eps,
                n_iter: 600,
                tol: 1e-8,
            };
            let self_d = earth_movers_distance(&a, n, &a, n, &cfg).expect("self should succeed");
            assert!(self_d >= 0.0 && self_d.is_finite());
            // Self-transport cost must stay below the cloud's coordinate spread²
            // (a loose but real upper bound on the entropic blur).
            assert!(
                self_d < 0.5,
                "EMD(a,a)={self_d} at ε={eps} should be near zero"
            );
            // Strictly below the distance to a shifted copy.
            let shifted = translate(&a, [1.0, 0.0, 0.0]);
            let shift_d =
                earth_movers_distance(&a, n, &shifted, n, &cfg).expect("shift should succeed");
            assert!(
                self_d < shift_d,
                "EMD(a,a)={self_d} must be < EMD(a,a+δ)={shift_d}"
            );
        }
    }

    #[test]
    fn emd_self_shrinks_with_epsilon() {
        // The entropic blur in EMD(a,a) is monotone in ε: smaller regularisation
        // ⇒ sharper (more diagonal) plan ⇒ smaller self-distance. This pins down
        // that the residual self-cost is the regulariser, not a bug.
        let n = 8;
        let a = lcg_cloud(n, 31);
        let big = SinkhornConfig {
            epsilon: 0.2,
            n_iter: 600,
            tol: 1e-8,
        };
        let small = SinkhornConfig {
            epsilon: 0.02,
            n_iter: 600,
            tol: 1e-8,
        };
        let d_big = earth_movers_distance(&a, n, &a, n, &big).expect("big should succeed");
        let d_small = earth_movers_distance(&a, n, &a, n, &small).expect("small should succeed");
        assert!(
            d_small <= d_big + 1e-6,
            "self EMD should shrink as ε falls: ε=0.02→{d_small}, ε=0.2→{d_big}"
        );
    }

    #[test]
    fn emd_monotone_under_growing_translation() {
        // Triangle-inequality-flavoured monotonicity: moving b further from a
        // (along a fixed direction, in steps) can only increase the transport
        // cost. Uses a multi-point cloud so the plan is non-trivial.
        let n = 6;
        let a = lcg_cloud(n, 3);
        let cfg = SinkhornConfig {
            epsilon: 0.05,
            n_iter: 500,
            tol: 1e-7,
        };
        let mut prev = f32::NEG_INFINITY;
        for step in 0..5 {
            let shift = step as f32 * 0.8;
            let b = translate(&a, [shift, 0.0, 0.0]);
            let d = earth_movers_distance(&a, n, &b, n, &cfg).expect("emd should succeed");
            assert!(
                d >= prev - 1e-3,
                "EMD must be non-decreasing as the shift grows: step {step} d={d} prev={prev}"
            );
            prev = d;
        }
    }
}
