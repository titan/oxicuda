//! Linear Kalman filter.

use super::linalg::{add, eye, inverse, matmul, matmul_rect, matvec, sub, transpose_rect};
use crate::error::{SeqError, SeqResult};

/// Result of running a Kalman filter on a sequence of observations.
#[derive(Debug, Clone)]
pub struct KalmanResult {
    /// `T` filtered means, each of length `dim_x`.
    pub means: Vec<Vec<f64>>,
    /// `T` filtered covariances, each of length `dim_x²`.
    pub covs: Vec<Vec<f64>>,
    /// Predicted means (a-priori) `x⁻_t`.
    pub pred_means: Vec<Vec<f64>>,
    /// Predicted covariances (a-priori) `P⁻_t`.
    pub pred_covs: Vec<Vec<f64>>,
}

/// Linear (time-invariant) Kalman filter.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    pub dim_x: usize,
    pub dim_z: usize,
    pub f: Vec<f64>, // dim_x × dim_x
    pub h: Vec<f64>, // dim_z × dim_x
    pub q: Vec<f64>, // dim_x × dim_x
    pub r: Vec<f64>, // dim_z × dim_z
    pub x0: Vec<f64>,
    pub p0: Vec<f64>,
}

impl KalmanFilter {
    /// Construct a Kalman filter validating shapes.
    pub fn new(
        dim_x: usize,
        dim_z: usize,
        f: Vec<f64>,
        h: Vec<f64>,
        q: Vec<f64>,
        r: Vec<f64>,
        x0: Vec<f64>,
        p0: Vec<f64>,
    ) -> SeqResult<Self> {
        if f.len() != dim_x * dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: dim_x * dim_x,
                got: f.len(),
            });
        }
        if h.len() != dim_z * dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: dim_z * dim_x,
                got: h.len(),
            });
        }
        if q.len() != dim_x * dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: dim_x * dim_x,
                got: q.len(),
            });
        }
        if r.len() != dim_z * dim_z {
            return Err(SeqError::ShapeMismatch {
                expected: dim_z * dim_z,
                got: r.len(),
            });
        }
        if x0.len() != dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: dim_x,
                got: x0.len(),
            });
        }
        if p0.len() != dim_x * dim_x {
            return Err(SeqError::ShapeMismatch {
                expected: dim_x * dim_x,
                got: p0.len(),
            });
        }
        Ok(Self {
            dim_x,
            dim_z,
            f,
            h,
            q,
            r,
            x0,
            p0,
        })
    }

    /// Run the filter on a sequence of observations `z` (T × dim_z, row-major).
    pub fn filter(&self, z: &[f64]) -> SeqResult<KalmanResult> {
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
        let f_t = transpose_rect(&self.f, nx, nx);
        let h_t = transpose_rect(&self.h, nz, nx);
        let i_eye = eye(nx);

        let mut means = Vec::with_capacity(t_max);
        let mut covs = Vec::with_capacity(t_max);
        let mut pred_means = Vec::with_capacity(t_max);
        let mut pred_covs = Vec::with_capacity(t_max);

        let mut x = self.x0.clone();
        let mut p = self.p0.clone();
        for t in 0..t_max {
            // Predict: x⁻ = F x; P⁻ = F P Fᵀ + Q
            let x_pred = matvec(&self.f, &x, nx);
            let fp = matmul(&self.f, &p, nx);
            let fpft = matmul_rect(&fp, &f_t, nx, nx, nx);
            let p_pred = add(&fpft, &self.q);
            pred_means.push(x_pred.clone());
            pred_covs.push(p_pred.clone());

            // Innovation: y = z − H x⁻
            let z_t = &z[t * nz..(t + 1) * nz];
            let hx = matmul_rect(&self.h, &x_pred, nz, nx, 1);
            let mut y = vec![0.0; nz];
            for k in 0..nz {
                y[k] = z_t[k] - hx[k];
            }
            // S = H P⁻ Hᵀ + R
            let hp = matmul_rect(&self.h, &p_pred, nz, nx, nx);
            let hpht = matmul_rect(&hp, &h_t, nz, nx, nz);
            let s = add(&hpht, &self.r);
            let s_inv = inverse(&s, nz)?;
            // K = P⁻ Hᵀ S⁻¹
            let pht = matmul_rect(&p_pred, &h_t, nx, nx, nz);
            let k_gain = matmul_rect(&pht, &s_inv, nx, nz, nz);
            // x = x⁻ + K y
            let ky = matmul_rect(&k_gain, &y, nx, nz, 1);
            for d in 0..nx {
                x[d] = x_pred[d] + ky[d];
            }
            // P = (I − K H) P⁻
            let kh = matmul_rect(&k_gain, &self.h, nx, nz, nx);
            let i_kh = sub(&i_eye, &kh);
            p = matmul_rect(&i_kh, &p_pred, nx, nx, nx);

            means.push(x.clone());
            covs.push(p.clone());
        }
        Ok(KalmanResult {
            means,
            covs,
            pred_means,
            pred_covs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_walk_track() {
        // x_t = x_{t-1} + noise, z_t = x_t + noise
        let kf = KalmanFilter::new(
            1,
            1,
            vec![1.0],
            vec![1.0],
            vec![0.01],
            vec![0.05],
            vec![0.0],
            vec![1.0],
        )
        .expect("ok");
        let z = vec![1.0, 1.05, 0.95, 1.02, 1.0];
        let res = kf.filter(&z).expect("ok");
        // Estimated mean should track the data closely after a few steps.
        let last = res.means[res.means.len() - 1][0];
        assert!((last - 1.0).abs() < 0.2, "mean drifted: {last}");
    }
}
