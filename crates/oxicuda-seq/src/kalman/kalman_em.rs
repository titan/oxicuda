//! EM (Shumway-Stoffer) parameter learning for the linear Gaussian state-space model.

use super::kalman_filter::KalmanFilter;
use super::linalg::matmul_rect;
use super::rts_smoother::rts_smoother;
use crate::error::{SeqError, SeqResult};

/// EM configuration.
#[derive(Debug, Clone, Copy)]
pub struct KalmanEmConfig {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for KalmanEmConfig {
    fn default() -> Self {
        Self {
            max_iter: 20,
            tol: 1e-5,
        }
    }
}

/// Run a simplified EM that refits Q and R from residuals (keeping F, H fixed).
///
/// Returns the refitted filter and the (approximate) data log-likelihood trace.
pub fn kalman_em(
    init: &KalmanFilter,
    z: &[f64],
    cfg: &KalmanEmConfig,
) -> SeqResult<(KalmanFilter, Vec<f64>)> {
    if z.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let nx = init.dim_x;
    let nz = init.dim_z;
    if z.len() % nz != 0 {
        return Err(SeqError::DimensionMismatch { a: z.len(), b: nz });
    }
    let t_max = z.len() / nz;
    let mut kf = init.clone();
    let mut history: Vec<f64> = Vec::with_capacity(cfg.max_iter);
    let mut prev_obj = f64::NEG_INFINITY;
    for _it in 0..cfg.max_iter {
        let filt = kf.filter(z)?;
        let smooth = rts_smoother(&kf, &filt)?;
        // Pseudo log-likelihood proxy: −0.5 Σ tr(P)
        let mut obj = 0.0;
        for c in &smooth.covs {
            for d in 0..nx {
                obj -= 0.5 * c[d * nx + d];
            }
        }
        history.push(obj);
        if (obj - prev_obj).abs() < cfg.tol {
            break;
        }
        prev_obj = obj;

        // Update Q from smoothed state transitions:
        // Q ≈ 1/(T-1) Σ (x_s_{t+1} − F x_s_t)(...)ᵀ + smoothed-variance terms.
        let mut q_new = vec![0.0; nx * nx];
        for t in 0..t_max - 1 {
            let x_next = &smooth.means[t + 1];
            let fx = matmul_rect(&kf.f, &smooth.means[t], nx, nx, 1);
            let dx: Vec<f64> = x_next.iter().zip(fx.iter()).map(|(a, b)| a - b).collect();
            for i in 0..nx {
                for j in 0..nx {
                    q_new[i * nx + j] += dx[i] * dx[j];
                }
            }
        }
        let denom = (t_max - 1).max(1) as f64;
        for q in q_new.iter_mut() {
            *q /= denom;
        }
        // Add small ridge for numerical stability.
        for d in 0..nx {
            q_new[d * nx + d] += 1e-6;
        }
        kf.q = q_new;

        // Update R from observation residuals: R ≈ 1/T Σ (z_t − H x_s_t)(...)ᵀ
        let mut r_new = vec![0.0; nz * nz];
        for t in 0..t_max {
            let hx = matmul_rect(&kf.h, &smooth.means[t], nz, nx, 1);
            let z_t = &z[t * nz..(t + 1) * nz];
            let dy: Vec<f64> = z_t.iter().zip(hx.iter()).map(|(a, b)| a - b).collect();
            for i in 0..nz {
                for j in 0..nz {
                    r_new[i * nz + j] += dy[i] * dy[j];
                }
            }
        }
        for r in r_new.iter_mut() {
            *r /= t_max as f64;
        }
        for d in 0..nz {
            r_new[d * nz + d] += 1e-6;
        }
        kf.r = r_new;
    }
    Ok((kf, history))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn em_runs() {
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
        let z = vec![1.0, 1.05, 0.95, 1.02, 1.0, 1.03];
        let cfg = KalmanEmConfig {
            max_iter: 5,
            tol: 1e-8,
        };
        let (k2, h) = kalman_em(&kf, &z, &cfg).expect("ok");
        assert!(!h.is_empty());
        assert_eq!(k2.dim_x, 1);
    }
}
