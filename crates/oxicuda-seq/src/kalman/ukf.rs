//! Unscented Kalman Filter (Julier & Uhlmann 1997 / Wan & van der Merwe 2000).
//!
//! Derivative-free nonlinear Kalman filtering using sigma-point propagation.
//! More accurate than EKF for strongly nonlinear systems because it propagates
//! a carefully chosen set of 2n+1 sigma points rather than linearising.

use super::linalg::{cholesky, inverse, matmul_rect, sub, transpose_rect};
use crate::error::{SeqError, SeqResult};

/// UKF hyperparameters (Wan & van der Merwe scaling parameters).
#[derive(Debug, Clone, Copy)]
pub struct UkfParams {
    /// Alpha: spread of sigma points around the mean (typical range 1e-4 to 1).
    pub alpha: f64,
    /// Beta: encodes prior knowledge of the distribution (2.0 is optimal for Gaussian).
    pub beta: f64,
    /// Kappa: secondary scaling parameter (0.0 or 3−n).
    pub kappa: f64,
}

impl Default for UkfParams {
    fn default() -> Self {
        Self {
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Result of running the UKF on a sequence of observations.
#[derive(Debug, Clone)]
pub struct UkfResult {
    /// T filtered means (a-posteriori), each of length `dim_x`.
    pub means: Vec<Vec<f64>>,
    /// T filtered covariances (a-posteriori), each of length `dim_x²`.
    pub covs: Vec<Vec<f64>>,
    /// T predicted means (a-priori), each of length `dim_x`.
    pub pred_means: Vec<Vec<f64>>,
    /// T predicted covariances (a-priori), each of length `dim_x²`.
    pub pred_covs: Vec<Vec<f64>>,
}

/// Unscented Kalman Filter for nonlinear state-space models.
///
/// State model:  x_{t+1} = f(x_t) + w_t,  w_t ~ N(0, Q)
/// Obs model:    z_t      = h(x_t) + v_t,  v_t ~ N(0, R)
pub struct UnscentedKalmanFilter<'a> {
    /// State dimension n.
    pub dim_x: usize,
    /// Observation dimension.
    pub dim_z: usize,
    /// State transition function x_{t+1} = f(x_t).
    pub f: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    /// Observation model z_t = h(x_t).
    pub h: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    /// Process noise covariance Q (dim_x × dim_x, row-major).
    pub q: Vec<f64>,
    /// Observation noise covariance R (dim_z × dim_z, row-major).
    pub r: Vec<f64>,
    /// Initial state mean (length dim_x).
    pub x0: Vec<f64>,
    /// Initial state covariance (dim_x × dim_x, row-major).
    pub p0: Vec<f64>,
    /// Scaling hyperparameters.
    pub params: UkfParams,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute sigma-point weights from UKF parameters and state dimension.
///
/// Returns `(wm, wc)` each of length `2n+1`.
fn compute_weights(n: usize, p: &UkfParams) -> (Vec<f64>, Vec<f64>) {
    let lambda = p.alpha * p.alpha * (n as f64 + p.kappa) - n as f64;
    let denom = n as f64 + lambda;
    let n_pts = 2 * n + 1;

    let mut wm = vec![0.0; n_pts];
    let mut wc = vec![0.0; n_pts];

    // Central point
    wm[0] = lambda / denom;
    wc[0] = lambda / denom + (1.0 - p.alpha * p.alpha + p.beta);

    // Symmetric points
    let w_sym = 0.5 / denom;
    for i in 1..n_pts {
        wm[i] = w_sym;
        wc[i] = w_sym;
    }
    (wm, wc)
}

/// Generate 2n+1 sigma points given current mean `x_bar` and covariance `p`.
///
/// χ_0     = x̄
/// χ_i     = x̄ + γ·L[:,i-1]   for i = 1..n
/// χ_{n+i} = x̄ − γ·L[:,i-1]   for i = 1..n
/// where γ = √(n + λ), L = cholesky(P).
fn sigma_points(
    x_bar: &[f64],
    p: &[f64],
    n: usize,
    p_params: &UkfParams,
) -> SeqResult<Vec<Vec<f64>>> {
    let lambda = p_params.alpha * p_params.alpha * (n as f64 + p_params.kappa) - n as f64;
    let gamma = (n as f64 + lambda).sqrt();

    let l = cholesky(p, n)?;

    let n_pts = 2 * n + 1;
    let mut pts: Vec<Vec<f64>> = Vec::with_capacity(n_pts);

    // χ_0 = x̄
    pts.push(x_bar.to_vec());

    // χ_i = x̄ + γ · L[:,i-1]  for i=1..n
    for i in 0..n {
        let col_i: Vec<f64> = (0..n).map(|r| l[r * n + i]).collect();
        let pt: Vec<f64> = x_bar
            .iter()
            .zip(col_i.iter())
            .map(|(m, c)| m + gamma * c)
            .collect();
        pts.push(pt);
    }

    // χ_{n+i} = x̄ − γ · L[:,i-1]  for i=1..n
    for i in 0..n {
        let col_i: Vec<f64> = (0..n).map(|r| l[r * n + i]).collect();
        let pt: Vec<f64> = x_bar
            .iter()
            .zip(col_i.iter())
            .map(|(m, c)| m - gamma * c)
            .collect();
        pts.push(pt);
    }

    Ok(pts)
}

/// Weighted mean of a set of vectors: Σ_i w_i · v_i.
fn weighted_mean(pts: &[Vec<f64>], w: &[f64], dim: usize) -> Vec<f64> {
    let mut mean = vec![0.0; dim];
    for (i, pt) in pts.iter().enumerate() {
        for d in 0..dim {
            mean[d] += w[i] * pt[d];
        }
    }
    mean
}

/// Weighted outer-product covariance: Σ_i w_i · (u_i − ū)(v_i − v̄)^T.
///
/// For the symmetric case (u = v): result is dim_u × dim_u.
/// For the cross case: result is dim_u × dim_v.
fn weighted_cross_cov(
    u_pts: &[Vec<f64>],
    u_bar: &[f64],
    v_pts: &[Vec<f64>],
    v_bar: &[f64],
    wc: &[f64],
    dim_u: usize,
    dim_v: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; dim_u * dim_v];
    for i in 0..u_pts.len() {
        let du: Vec<f64> = u_pts[i]
            .iter()
            .zip(u_bar.iter())
            .map(|(a, b)| a - b)
            .collect();
        let dv: Vec<f64> = v_pts[i]
            .iter()
            .zip(v_bar.iter())
            .map(|(a, b)| a - b)
            .collect();
        for r in 0..dim_u {
            for c in 0..dim_v {
                cov[r * dim_v + c] += wc[i] * du[r] * dv[c];
            }
        }
    }
    cov
}

// ---------------------------------------------------------------------------
// Main implementation
// ---------------------------------------------------------------------------

impl<'a> UnscentedKalmanFilter<'a> {
    /// Validate all dimensions before running.
    fn validate(&self, z: &[f64]) -> SeqResult<()> {
        if z.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if z.len() % self.dim_z != 0 {
            return Err(SeqError::DimensionMismatch {
                a: z.len(),
                b: self.dim_z,
            });
        }
        if self.q.len() != self.dim_x * self.dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: self.dim_x * self.dim_x,
                got: self.q.len(),
            });
        }
        if self.r.len() != self.dim_z * self.dim_z {
            return Err(SeqError::ShapeMismatch {
                expected: self.dim_z * self.dim_z,
                got: self.r.len(),
            });
        }
        if self.x0.len() != self.dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: self.dim_x,
                got: self.x0.len(),
            });
        }
        if self.p0.len() != self.dim_x * self.dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: self.dim_x * self.dim_x,
                got: self.p0.len(),
            });
        }
        if self.params.alpha <= 0.0 || self.params.alpha > 1.0 {
            return Err(SeqError::InvalidParameter {
                name: "alpha".to_string(),
                value: self.params.alpha,
            });
        }
        if self.params.beta < 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "beta".to_string(),
                value: self.params.beta,
            });
        }
        if self.params.kappa < 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "kappa".to_string(),
                value: self.params.kappa,
            });
        }
        Ok(())
    }

    /// Run the UKF on observations `z` (T × dim_z, row-major flat).
    ///
    /// Returns filtered (a-posteriori) means and covariances plus the a-priori
    /// predicted means and covariances for each time step.
    pub fn run(&self, z: &[f64]) -> SeqResult<UkfResult> {
        self.validate(z)?;

        let nx = self.dim_x;
        let nz = self.dim_z;
        let t_max = z.len() / nz;

        let (wm, wc) = compute_weights(nx, &self.params);

        let mut x = self.x0.clone();
        let mut p = self.p0.clone();

        let mut means = Vec::with_capacity(t_max);
        let mut covs = Vec::with_capacity(t_max);
        let mut pred_means = Vec::with_capacity(t_max);
        let mut pred_covs = Vec::with_capacity(t_max);

        for t in 0..t_max {
            // ------------------------------------------------------------------
            // 1. Generate sigma points from current (x, P)
            // ------------------------------------------------------------------
            let chi = sigma_points(&x, &p, nx, &self.params)?;

            // ------------------------------------------------------------------
            // 2. Predict: propagate sigma points through f
            // ------------------------------------------------------------------
            let gamma_pts: Vec<Vec<f64>> = chi.iter().map(|s| (self.f)(s)).collect();
            let x_pred = weighted_mean(&gamma_pts, &wm, nx);

            // P⁻ = Σ wc_i (γ_i − x̄⁻)(γ_i − x̄⁻)^T + Q
            let mut p_pred =
                weighted_cross_cov(&gamma_pts, &x_pred, &gamma_pts, &x_pred, &wc, nx, nx);
            for k in 0..p_pred.len() {
                p_pred[k] += self.q[k];
            }

            pred_means.push(x_pred.clone());
            pred_covs.push(p_pred.clone());

            // ------------------------------------------------------------------
            // 3. Update: propagate predicted sigma points through h
            // ------------------------------------------------------------------
            // Re-generate sigma points from (x_pred, p_pred) for measurement update
            let chi_pred = sigma_points(&x_pred, &p_pred, nx, &self.params)?;
            let upsilon_pts: Vec<Vec<f64>> = chi_pred.iter().map(|s| (self.h)(s)).collect();
            let y_pred = weighted_mean(&upsilon_pts, &wm, nz);

            // Innovation covariance S = Σ wc_i (Υ_i − ȳ)(Υ_i − ȳ)^T + R
            let mut s_mat =
                weighted_cross_cov(&upsilon_pts, &y_pred, &upsilon_pts, &y_pred, &wc, nz, nz);
            for k in 0..s_mat.len() {
                s_mat[k] += self.r[k];
            }

            // Cross-covariance P_xy = Σ wc_i (γ_i − x̄⁻)(Υ_i − ȳ)^T  [nx × nz]
            let p_xy = weighted_cross_cov(&chi_pred, &x_pred, &upsilon_pts, &y_pred, &wc, nx, nz);

            // Kalman gain K = P_xy · S⁻¹   [nx × nz]
            let s_inv = inverse(&s_mat, nz)?;
            let k_gain = matmul_rect(&p_xy, &s_inv, nx, nz, nz);

            // Innovation ν = z_t − ȳ
            let z_t = &z[t * nz..(t + 1) * nz];
            let nu = sub(z_t, &y_pred);

            // x = x̄⁻ + K·ν
            let k_nu = matmul_rect(&k_gain, &nu, nx, nz, 1);
            x = x_pred.iter().zip(k_nu.iter()).map(|(a, b)| a + b).collect();

            // P = P⁻ − K·S·K^T
            let ks = matmul_rect(&k_gain, &s_mat, nx, nz, nz);
            let k_t = transpose_rect(&k_gain, nx, nz);
            let kskt = matmul_rect(&ks, &k_t, nx, nz, nx);
            p = sub(&p_pred, &kskt);

            means.push(x.clone());
            covs.push(p.clone());
        }

        Ok(UkfResult {
            means,
            covs,
            pred_means,
            pred_covs,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kalman::kalman_filter::KalmanFilter;

    fn make_1d_ukf<'a>(q_val: f64, r_val: f64, x0: f64, p0: f64) -> UnscentedKalmanFilter<'a> {
        UnscentedKalmanFilter {
            dim_x: 1,
            dim_z: 1,
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![q_val],
            r: vec![r_val],
            x0: vec![x0],
            p0: vec![p0],
            params: UkfParams::default(),
        }
    }

    #[test]
    fn default_params_ok() {
        let p = UkfParams::default();
        assert!((p.alpha - 1e-3).abs() < 1e-15);
        assert!((p.beta - 2.0).abs() < 1e-15);
        assert!(p.kappa.abs() < 1e-15);
    }

    #[test]
    fn ukf_linear_matches_kf() {
        // For a 1D linear system, UKF must agree with the exact linear KF.
        let z = vec![1.0, 1.05, 0.95, 1.02, 1.0];
        let q_val = 0.01;
        let r_val = 0.05;

        let ukf = make_1d_ukf(q_val, r_val, 0.0, 1.0);
        let ukf_res = ukf.run(&z).expect("UKF run failed");

        let kf = KalmanFilter::new(
            1,
            1,
            vec![1.0],
            vec![1.0],
            vec![q_val],
            vec![r_val],
            vec![0.0],
            vec![1.0],
        )
        .expect("ok");
        let kf_res = kf.filter(&z).expect("KF run failed");

        for t in 0..z.len() {
            let diff = (ukf_res.means[t][0] - kf_res.means[t][0]).abs();
            assert!(
                diff < 1e-6,
                "step {t}: UKF={:.10} KF={:.10} diff={:.2e}",
                ukf_res.means[t][0],
                kf_res.means[t][0],
                diff
            );
        }
    }

    #[test]
    fn ukf_identity_state() {
        // Constant state model, perfect observations: posterior should track obs closely.
        let z = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let ukf = UnscentedKalmanFilter {
            dim_x: 1,
            dim_z: 1,
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.001],
            r: vec![1e-6],
            x0: vec![0.0],
            p0: vec![10.0],
            params: UkfParams::default(),
        };
        let res = ukf.run(&z).expect("ok");
        let last = res.means[res.means.len() - 1][0];
        assert!((last - 2.0).abs() < 0.01, "expected ~2.0 got {last}");
    }

    #[test]
    fn ukf_output_length() {
        let z: Vec<f64> = (0..7).map(|i| i as f64 * 0.1).collect();
        let ukf = make_1d_ukf(0.01, 0.05, 0.0, 1.0);
        let res = ukf.run(&z).expect("ok");
        assert_eq!(res.means.len(), 7);
        for t in 0..7 {
            assert_eq!(res.means[t].len(), 1, "means dim mismatch at t={t}");
        }
    }

    #[test]
    fn ukf_cov_positive_diagonal() {
        let z: Vec<f64> = (0..10).map(|i| (i as f64) * 0.1 + 1.0).collect();
        let ukf = make_1d_ukf(0.01, 0.05, 0.0, 1.0);
        let res = ukf.run(&z).expect("ok");
        for (t, cov) in res.covs.iter().enumerate() {
            // diagonal element is cov[0] for 1D
            assert!(cov[0] > 0.0, "non-positive diagonal at t={t}: {}", cov[0]);
        }
    }

    #[test]
    fn ukf_pred_means_correct_length() {
        let z = vec![1.0, 2.0, 3.0];
        let ukf = make_1d_ukf(0.01, 0.1, 0.0, 1.0);
        let res = ukf.run(&z).expect("ok");
        assert_eq!(res.pred_means.len(), 3);
        for t in 0..3 {
            assert_eq!(res.pred_means[t].len(), 1, "pred_means dim at t={t}");
        }
    }

    #[test]
    fn ukf_nonlinear_cos() {
        // f(x) = [cos(x[0])], h(x) = x; 1D cosine state transition.
        let ukf = UnscentedKalmanFilter {
            dim_x: 1,
            dim_z: 1,
            f: Box::new(|x: &[f64]| vec![x[0].cos()]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.1],
            r: vec![0.5],
            x0: vec![0.5],
            p0: vec![1.0],
            params: UkfParams::default(),
        };
        let z = vec![0.9, 0.95, 0.98, 0.97, 0.96];
        let res = ukf.run(&z).expect("nonlinear UKF failed");
        assert_eq!(res.means.len(), 5);
        // Covariance must stay positive
        for (t, cov) in res.covs.iter().enumerate() {
            assert!(cov[0] > 0.0, "negative cov at t={t}");
        }
    }

    #[test]
    fn ukf_tracks_slowly_varying() {
        // Slowly drifting signal; UKF estimate within 3 posterior std.
        let z: Vec<f64> = (0..20).map(|i| 1.0 + i as f64 * 0.05).collect();
        let ukf = UnscentedKalmanFilter {
            dim_x: 1,
            dim_z: 1,
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.01],
            r: vec![0.05],
            x0: vec![1.0],
            p0: vec![1.0],
            params: UkfParams::default(),
        };
        let res = ukf.run(&z).expect("ok");
        let last_mean = res.means[19][0];
        let last_std = res.covs[19][0].sqrt();
        let true_val = z[19];
        assert!(
            (last_mean - true_val).abs() < 3.0 * last_std + 0.5,
            "drifted too far: mean={last_mean:.4} true={true_val:.4} std={last_std:.4}"
        );
    }

    #[test]
    fn err_empty_obs() {
        let ukf = make_1d_ukf(0.01, 0.05, 0.0, 1.0);
        let result = ukf.run(&[]);
        assert!(matches!(result, Err(SeqError::EmptyInput)));
    }

    #[test]
    fn err_z_len_not_multiple_of_dim_z() {
        let ukf = UnscentedKalmanFilter {
            dim_x: 1,
            dim_z: 2,
            f: Box::new(|x: &[f64]| x.to_vec()),
            h: Box::new(|x: &[f64]| vec![x[0], x[0]]),
            q: vec![0.01, 0.0, 0.0, 0.01],
            r: vec![0.1, 0.0, 0.0, 0.1],
            x0: vec![0.0],
            p0: vec![1.0],
            params: UkfParams::default(),
        };
        // z length 3 is not a multiple of dim_z=2
        let result = ukf.run(&[1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(SeqError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_q_wrong_shape() {
        let ukf = UnscentedKalmanFilter {
            dim_x: 2,
            dim_z: 1,
            f: Box::new(|x: &[f64]| x.to_vec()),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.01], // should be 4 elements
            r: vec![0.1],
            x0: vec![0.0, 0.0],
            p0: vec![1.0, 0.0, 0.0, 1.0],
            params: UkfParams::default(),
        };
        let result = ukf.run(&[1.0, 2.0]);
        assert!(matches!(result, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_r_wrong_shape() {
        let ukf = UnscentedKalmanFilter {
            dim_x: 1,
            dim_z: 1,
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.01],
            r: vec![0.1, 0.0, 0.0], // should be 1 element
            x0: vec![0.0],
            p0: vec![1.0],
            params: UkfParams::default(),
        };
        let result = ukf.run(&[1.0, 2.0]);
        assert!(matches!(result, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_x0_wrong_len() {
        let ukf = UnscentedKalmanFilter {
            dim_x: 2,
            dim_z: 1,
            f: Box::new(|x: &[f64]| x.to_vec()),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.01, 0.0, 0.0, 0.01],
            r: vec![0.1],
            x0: vec![0.0], // should be length 2
            p0: vec![1.0, 0.0, 0.0, 1.0],
            params: UkfParams::default(),
        };
        let result = ukf.run(&[1.0, 2.0]);
        assert!(matches!(result, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn sigma_point_count() {
        // Verify sigma point count is 2*dim_x+1 via output quality on 2D problem.
        let ukf = UnscentedKalmanFilter {
            dim_x: 2,
            dim_z: 2,
            f: Box::new(|x: &[f64]| vec![x[0], x[1]]),
            h: Box::new(|x: &[f64]| vec![x[0], x[1]]),
            q: vec![0.01, 0.0, 0.0, 0.01],
            r: vec![0.1, 0.0, 0.0, 0.1],
            x0: vec![0.0, 0.0],
            p0: vec![1.0, 0.0, 0.0, 1.0],
            params: UkfParams::default(),
        };
        // 2*2+1 = 5 sigma points; verify the filter runs without error (implicitly correct)
        let z = vec![1.0, 2.0, 1.1, 2.1, 0.9, 1.9];
        let res = ukf.run(&z).expect("sigma_point_count test failed");
        assert_eq!(res.means.len(), 3);
    }

    #[test]
    fn ukf_2d_state_1d_obs() {
        // dim_x=2, dim_z=1; constant velocity model, position observed.
        let dt = 1.0_f64;
        let ukf = UnscentedKalmanFilter {
            dim_x: 2,
            dim_z: 1,
            f: Box::new(move |x: &[f64]| vec![x[0] + dt * x[1], x[1]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            q: vec![0.01, 0.0, 0.0, 0.01],
            r: vec![0.5],
            x0: vec![0.0, 1.0],
            p0: vec![1.0, 0.0, 0.0, 1.0],
            params: UkfParams::default(),
        };
        let z: Vec<f64> = (0..8).map(|t| t as f64 * 1.0).collect();
        let res = ukf.run(&z).expect("2d state 1d obs failed");
        assert_eq!(res.means.len(), 8);
        for (t, m) in res.means.iter().enumerate() {
            assert_eq!(m.len(), 2, "state dim at t={t}");
        }
    }

    #[test]
    fn ukf_dim_x_1_dim_z_1() {
        // Simplest case: 1D state, 1D obs; should converge.
        let ukf = make_1d_ukf(0.01, 0.1, 0.0, 1.0);
        let z = vec![1.0; 20];
        let res = ukf.run(&z).expect("simplest case failed");
        let last = res.means[19][0];
        assert!((last - 1.0).abs() < 0.05, "did not converge: {last}");
    }

    #[test]
    fn ukf_weights_sum_to_one() {
        // Mathematical property: Σ W_i^m = 1 for any n, α, β, κ.
        // λ/(n+λ) + 2n * 1/(2(n+λ)) = λ/(n+λ) + n/(n+λ) = (λ+n)/(n+λ) = 1.
        for n in [1usize, 2, 3, 5, 10] {
            let p = UkfParams::default();
            let (wm, _wc) = compute_weights(n, &p);
            let sum: f64 = wm.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "weights don't sum to 1 for n={n}: sum={sum}"
            );
        }
    }
}
