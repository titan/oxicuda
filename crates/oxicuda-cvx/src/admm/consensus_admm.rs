//! Consensus ADMM for separable problems `min Σ_i f_i(x_i)` with constraint `x_i = z`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Consensus ADMM for N agents.
///
/// `x_updates[i](z, u_i)` produces agent i's new x_i.
/// All x_i and z share dimension `n`.
pub fn consensus_admm<F>(
    n_agents: usize,
    n: usize,
    rho: f64,
    x_updates: &[F],
    max_iter: usize,
    tol: f64,
) -> CvxResult<ConsensusResult>
where
    F: Fn(&[f64], &[f64]) -> CvxResult<Vec<f64>>,
{
    if n_agents == 0 {
        return Err(CvxError::InvalidParameter("0 agents".into()));
    }
    if x_updates.len() != n_agents {
        return Err(CvxError::DimensionMismatch {
            a: x_updates.len(),
            b: n_agents,
        });
    }
    if rho <= 0.0 || !rho.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "consensus ADMM rho > 0 required, got {rho}"
        )));
    }
    let mut xs: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n_agents];
    let mut us: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n_agents];
    let mut z = vec![0.0_f64; n];
    let mut iters = 0usize;
    let mut pri_norm = 0.0_f64;
    for it in 0..max_iter {
        // x_i updates.
        for i in 0..n_agents {
            let xi_new = x_updates[i](&z, &us[i])?;
            if xi_new.len() != n {
                return Err(CvxError::DimensionMismatch {
                    a: xi_new.len(),
                    b: n,
                });
            }
            xs[i] = xi_new;
        }
        // z update: average of (x_i + u_i).
        let mut z_new = vec![0.0_f64; n];
        for i in 0..n_agents {
            for j in 0..n {
                z_new[j] += xs[i][j] + us[i][j];
            }
        }
        for v in z_new.iter_mut().take(n) {
            *v /= n_agents as f64;
        }
        // u_i update.
        for (xs_i, us_i) in xs.iter().zip(us.iter_mut()).take(n_agents) {
            for j in 0..n {
                us_i[j] += xs_i[j] - z_new[j];
            }
        }
        // Primal residual: ||x_i - z||.
        let mut r_sq = 0.0_f64;
        for xs_i in xs.iter().take(n_agents) {
            for j in 0..n {
                let d = xs_i[j] - z_new[j];
                r_sq += d * d;
            }
        }
        let dz: Vec<f64> = z_new.iter().zip(z.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&dz);
        z = z_new;
        pri_norm = r_sq.sqrt();
        iters = it + 1;
        if pri_norm < tol && d_nrm < tol {
            break;
        }
    }
    Ok(ConsensusResult {
        x: xs,
        z,
        iter: iters,
        pri_residual: pri_norm,
    })
}

/// Result of consensus ADMM.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub x: Vec<Vec<f64>>,
    pub z: Vec<f64>,
    pub iter: usize,
    pub pri_residual: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CvxResult;

    #[test]
    fn consensus_average_minimisation() {
        // min Σ_i 0.5 ||x_i - b_i||² s.t. x_i = z → solution z = mean(b_i).
        let bs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let rho = 1.0_f64;
        let updates: Vec<_> = bs
            .iter()
            .map(|b| {
                let b_owned = b.clone();
                move |z: &[f64], u: &[f64]| -> CvxResult<Vec<f64>> {
                    Ok((0..b_owned.len())
                        .map(|i| (b_owned[i] + rho * (z[i] - u[i])) / (1.0 + rho))
                        .collect())
                }
            })
            .collect();
        let res = consensus_admm(3, 2, rho, &updates, 500, 1.0e-9).expect("ok");
        assert!((res.z[0] - 3.0).abs() < 1.0e-4);
        assert!((res.z[1] - 4.0).abs() < 1.0e-4);
    }
}
