//! Rauch-Tung-Striebel (RTS) backward smoother.

use super::kalman_filter::{KalmanFilter, KalmanResult};
use super::linalg::{add, inverse, matmul_rect, matvec, sub, transpose_rect};
use crate::error::{SeqError, SeqResult};

/// Smoothed result: posterior means and covariances over the entire trajectory.
#[derive(Debug, Clone)]
pub struct RtsResult {
    pub means: Vec<Vec<f64>>,
    pub covs: Vec<Vec<f64>>,
}

/// Run RTS smoother given filter output.
pub fn rts_smoother(kf: &KalmanFilter, filt: &KalmanResult) -> SeqResult<RtsResult> {
    if filt.means.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = filt.means.len();
    let nx = kf.dim_x;
    let f_t = transpose_rect(&kf.f, nx, nx);

    let mut means = filt.means.clone();
    let mut covs = filt.covs.clone();
    for t in (0..t_max - 1).rev() {
        let p_t = &filt.covs[t];
        let p_pred = &filt.pred_covs[t + 1];
        let p_pred_inv = inverse(p_pred, nx)?;
        // C = P_t · Fᵀ · (P⁻_{t+1})⁻¹
        let pft = matmul_rect(p_t, &f_t, nx, nx, nx);
        let c = matmul_rect(&pft, &p_pred_inv, nx, nx, nx);

        // x_s_t = x_t + C (x_s_{t+1} − x⁻_{t+1})
        let dx = sub(&means[t + 1], &filt.pred_means[t + 1]);
        let c_dx = matvec(&c, &dx, nx);
        let mut x_s = filt.means[t].clone();
        for d in 0..nx {
            x_s[d] += c_dx[d];
        }
        // P_s_t = P_t + C (P_s_{t+1} − P⁻_{t+1}) Cᵀ
        let dp = sub(&covs[t + 1], p_pred);
        let c_dp = matmul_rect(&c, &dp, nx, nx, nx);
        let c_t = transpose_rect(&c, nx, nx);
        let cdpct = matmul_rect(&c_dp, &c_t, nx, nx, nx);
        let p_s = add(&filt.covs[t], &cdpct);
        means[t] = x_s;
        covs[t] = p_s;
    }
    Ok(RtsResult { means, covs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rts_variance_le_filter() {
        let kf = KalmanFilter::new(
            1,
            1,
            vec![1.0],
            vec![1.0],
            vec![0.01],
            vec![0.1],
            vec![0.0],
            vec![1.0],
        )
        .expect("ok");
        let z = vec![1.0, 0.95, 1.1, 1.05, 0.9, 1.0];
        let f = kf.filter(&z).expect("ok");
        let s = rts_smoother(&kf, &f).expect("ok");
        // Smoother variance at t should be <= filter variance (except at last step).
        for t in 0..z.len() - 1 {
            let var_f = f.covs[t][0];
            let var_s = s.covs[t][0];
            assert!(
                var_s <= var_f + 1e-9,
                "smoother var {var_s} > filter var {var_f} at t={t}"
            );
        }
    }
}
