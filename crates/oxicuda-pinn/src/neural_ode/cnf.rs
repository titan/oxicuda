//! Continuous Normalizing Flows (CNF) with Hutchinson trace estimation.

use crate::error::PinnResult;
use crate::handle::LcgRng;
use crate::neural_ode::solvers::{OdeRhsFn, integrate_fixed};

/// Forward integrate CNF ODE and compute the log-determinant change.
///
/// Returns `(z1, delta_log_p)` where:
/// - `z1` is the transformed point at time `t1`.
/// - `delta_log_p = -∫ tr(∂f/∂z) dt` (log-density change from t0 to t1).
///
/// The trace is computed via finite-difference dense estimation (small dim).
pub fn cnf_forward(
    rhs: OdeRhsFn,
    z0: &[f32],
    t0: f32,
    t1: f32,
    h: f32,
) -> PinnResult<(Vec<f32>, f32)> {
    let dim = z0.len();

    // Integrate state
    let (_, states) = integrate_fixed(rhs, t0, t1, z0, h)?;
    let z1 = states.last().cloned().unwrap_or_else(|| z0.to_vec());

    // Approximate delta_log_p using trapezoidal quadrature of -tr(∂f/∂z)
    let times: Vec<f32> = {
        let mut ts = Vec::with_capacity(states.len());
        let n = states.len();
        for i in 0..n {
            ts.push(t0 + (t1 - t0) * i as f32 / (n - 1).max(1) as f32);
        }
        ts
    };

    let mut delta_log_p = 0.0_f32;
    for (i, (t, y)) in times.iter().zip(states.iter()).enumerate() {
        let tr = dense_trace(rhs, *t, y);
        let weight = if i == 0 || i == states.len() - 1 {
            0.5
        } else {
            1.0
        };
        let dt = if states.len() > 1 {
            (t1 - t0) / (states.len() - 1) as f32
        } else {
            0.0
        };
        delta_log_p -= weight * tr * dt;
    }
    let _ = dim;

    Ok((z1, delta_log_p))
}

/// Hutchinson trace estimator: `tr(J) ≈ (1/n_v) Σ εᵀ J ε` for ε ~ Rademacher.
///
/// Uses finite differences: `J·ε ≈ (f(t, z+δε) - f(t, z-δε)) / (2δ)`.
pub fn hutchinson_trace(rhs: OdeRhsFn, t: f32, z: &[f32], n_v: usize, rng: &mut LcgRng) -> f32 {
    let dim = z.len();
    let delta = 1e-4_f32;
    let mut trace_est = 0.0_f32;

    for _ in 0..n_v {
        // Sample Rademacher vector ε ∈ {-1, +1}^dim
        let eps: Vec<f32> = (0..dim)
            .map(|_| {
                if rng.next_u32() & 1 == 0 {
                    1.0_f32
                } else {
                    -1.0_f32
                }
            })
            .collect();

        // z + δε and z - δε
        let z_plus: Vec<f32> = z
            .iter()
            .zip(eps.iter())
            .map(|(&zi, &ei)| zi + delta * ei)
            .collect();
        let z_minus: Vec<f32> = z
            .iter()
            .zip(eps.iter())
            .map(|(&zi, &ei)| zi - delta * ei)
            .collect();

        let mut f_plus = vec![0.0_f32; dim];
        let mut f_minus = vec![0.0_f32; dim];
        rhs(t, &z_plus, &mut f_plus);
        rhs(t, &z_minus, &mut f_minus);

        // J·ε ≈ (f_plus - f_minus) / (2δ)
        let jv: Vec<f32> = f_plus
            .iter()
            .zip(f_minus.iter())
            .map(|(&fp, &fm)| (fp - fm) / (2.0 * delta))
            .collect();

        // εᵀ J ε
        let etjv: f32 = eps.iter().zip(jv.iter()).map(|(&e, &jvi)| e * jvi).sum();
        trace_est += etjv;
    }

    trace_est / n_v as f32
}

/// Dense trace estimator (for reference / small dimensions).
///
/// Computes each column of J via finite differences, sums diagonal.
pub fn dense_trace(rhs: OdeRhsFn, t: f32, z: &[f32]) -> f32 {
    let dim = z.len();
    let delta = 1e-4_f32;
    let mut tr = 0.0_f32;

    let mut f0 = vec![0.0_f32; dim];
    rhs(t, z, &mut f0);

    for j in 0..dim {
        let mut z_p = z.to_vec();
        z_p[j] += delta;
        let mut f_p = vec![0.0_f32; dim];
        rhs(t, &z_p, &mut f_p);
        // J[j, j] ≈ (f_p[j] - f0[j]) / delta
        tr += (f_p[j] - f0[j]) / delta;
    }

    tr
}

#[cfg(test)]
mod tests {
    use super::*;

    // Linear: f(t, z) = A * z, tr(A) = sum of diagonal
    fn linear_expand(_t: f32, z: &[f32], dz: &mut [f32]) {
        // A = [[0.1, 0], [0, 0.2]] → tr(A) = 0.3
        dz[0] = 0.1 * z[0];
        dz[1] = 0.2 * z[1];
    }

    fn const_zero(_t: f32, z: &[f32], dz: &mut [f32]) {
        let _ = z;
        for d in dz.iter_mut() {
            *d = 0.0;
        }
    }

    #[test]
    fn dense_trace_linear_known() {
        // tr([[0.1, 0], [0, 0.2]]) = 0.3
        let z = vec![1.0_f32, 1.0];
        let tr = dense_trace(&linear_expand, 0.0, &z);
        assert!(
            (tr - 0.3).abs() < 1e-3,
            "dense_trace should be ~0.3, got {tr}"
        );
    }

    #[test]
    fn hutchinson_trace_estimate() {
        let z = vec![1.0_f32, 1.0];
        let mut rng = LcgRng::new(42);
        let tr_hutchinson = hutchinson_trace(&linear_expand, 0.0, &z, 64, &mut rng);
        // Should be within 10% of 0.3
        assert!(
            (tr_hutchinson - 0.3).abs() < 0.1,
            "Hutchinson estimate {} not close to 0.3",
            tr_hutchinson
        );
    }

    #[test]
    fn cnf_forward_zero_flow_no_log_det() {
        // f(z) = 0 → no change in z, tr = 0, delta_log_p = 0
        let z0 = vec![1.0_f32, 2.0];
        let (z1, delta_log_p) = cnf_forward(&const_zero, &z0, 0.0, 1.0, 0.1)
            .expect("CNF forward pass with zero flow should succeed and produce no log-det change");
        assert!((z1[0] - z0[0]).abs() < 1e-5);
        assert!((z1[1] - z0[1]).abs() < 1e-5);
        assert!(delta_log_p.abs() < 1e-3);
    }

    #[test]
    fn cnf_forward_expanding_flow_negative_log_p() {
        // Expanding flow: positive divergence → delta_log_p should be negative
        let z0 = vec![1.0_f32, 1.0];
        let (_, delta_log_p) = cnf_forward(&linear_expand, &z0, 0.0, 1.0, 0.01)
            .expect("CNF forward pass with expanding linear flow should succeed");
        // tr(A) = 0.3 > 0, so -∫ tr dt < 0
        assert!(
            delta_log_p < 0.0,
            "Expected negative delta_log_p, got {delta_log_p}"
        );
    }

    #[test]
    fn cnf_forward_shape_correct() {
        let z0 = vec![0.5_f32, -0.3, 1.2];
        let (z1, dlp) = cnf_forward(&const_zero, &z0, 0.0, 0.5, 0.1)
            .expect("CNF forward pass with zero flow should succeed");
        assert_eq!(z1.len(), 3);
        assert!(dlp.is_finite());
    }

    #[test]
    fn dense_trace_zero_flow() {
        let z = vec![1.0_f32, 1.0];
        let tr = dense_trace(&const_zero, 0.0, &z);
        assert!(
            tr.abs() < 1e-3,
            "zero flow should have zero trace, got {tr}"
        );
    }

    #[test]
    fn hutchinson_trace_zero_flow() {
        let z = vec![1.0_f32, 1.0];
        let mut rng = LcgRng::new(7);
        let tr = hutchinson_trace(&const_zero, 0.0, &z, 32, &mut rng);
        assert!(
            tr.abs() < 0.1,
            "zero flow trace estimate should be ~0, got {tr}"
        );
    }
}
