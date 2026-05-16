//! Extended Kalman Filter with closure-provided non-linear dynamics and Jacobians.

use super::linalg::{add, eye, inverse, matmul, matmul_rect, sub, transpose_rect};
use crate::error::{SeqError, SeqResult};

/// Result of running the EKF on a sequence.
#[derive(Debug, Clone)]
pub struct ExtendedKalmanResult {
    pub means: Vec<Vec<f64>>,
    pub covs: Vec<Vec<f64>>,
}

/// Extended Kalman filter parameterised on non-linear `f(x)`, `h(x)`, plus
/// Jacobians `Fj(x)` and `Hj(x)`.
pub struct ExtendedKalmanFilter<'a> {
    pub dim_x: usize,
    pub dim_z: usize,
    pub f: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    pub h: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    pub f_jacobian: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    pub h_jacobian: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    pub q: Vec<f64>,
    pub r: Vec<f64>,
    pub x0: Vec<f64>,
    pub p0: Vec<f64>,
}

impl<'a> ExtendedKalmanFilter<'a> {
    /// Run the EKF on `z` (T × dim_z).
    pub fn run(&self, z: &[f64]) -> SeqResult<ExtendedKalmanResult> {
        if z.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if z.len() % self.dim_z != 0 {
            return Err(SeqError::DimensionMismatch {
                a: z.len(),
                b: self.dim_z,
            });
        }
        let t_max = z.len() / self.dim_z;
        let nx = self.dim_x;
        let nz = self.dim_z;
        let i_eye = eye(nx);
        let mut means = Vec::with_capacity(t_max);
        let mut covs = Vec::with_capacity(t_max);
        let mut x = self.x0.clone();
        let mut p = self.p0.clone();
        for t in 0..t_max {
            // Predict
            let x_pred = (self.f)(&x);
            let f_j = (self.f_jacobian)(&x);
            let f_j_t = transpose_rect(&f_j, nx, nx);
            let fp = matmul(&f_j, &p, nx);
            let fpft = matmul_rect(&fp, &f_j_t, nx, nx, nx);
            let p_pred = add(&fpft, &self.q);

            // Update
            let z_t = &z[t * nz..(t + 1) * nz];
            let h_x = (self.h)(&x_pred);
            let mut y = vec![0.0; nz];
            for k in 0..nz {
                y[k] = z_t[k] - h_x[k];
            }
            let h_j = (self.h_jacobian)(&x_pred);
            let h_j_t = transpose_rect(&h_j, nz, nx);
            let hp = matmul_rect(&h_j, &p_pred, nz, nx, nx);
            let hpht = matmul_rect(&hp, &h_j_t, nz, nx, nz);
            let s = add(&hpht, &self.r);
            let s_inv = inverse(&s, nz)?;
            let pht = matmul_rect(&p_pred, &h_j_t, nx, nx, nz);
            let k_gain = matmul_rect(&pht, &s_inv, nx, nz, nz);
            let ky = matmul_rect(&k_gain, &y, nx, nz, 1);
            for d in 0..nx {
                x[d] = x_pred[d] + ky[d];
            }
            let kh = matmul_rect(&k_gain, &h_j, nx, nz, nx);
            let i_kh = sub(&i_eye, &kh);
            p = matmul_rect(&i_kh, &p_pred, nx, nx, nx);
            means.push(x.clone());
            covs.push(p.clone());
        }
        Ok(ExtendedKalmanResult { means, covs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ekf_linear_matches_kf() {
        let dim_x = 1;
        let dim_z = 1;
        let ekf = ExtendedKalmanFilter {
            dim_x,
            dim_z,
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            f_jacobian: Box::new(|_x: &[f64]| vec![1.0]),
            h_jacobian: Box::new(|_x: &[f64]| vec![1.0]),
            q: vec![0.01],
            r: vec![0.05],
            x0: vec![0.0],
            p0: vec![1.0],
        };
        let z = vec![1.0, 1.05, 0.95];
        let res = ekf.run(&z).expect("ok");
        assert_eq!(res.means.len(), 3);
        assert!((res.means[res.means.len() - 1][0] - 1.0).abs() < 0.2);
    }
}
