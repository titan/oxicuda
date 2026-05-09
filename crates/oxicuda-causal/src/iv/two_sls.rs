use crate::discovery::notears::gauss_jordan_inv;
use crate::error::{CausalError, CausalResult};

fn mat_mul_rect(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|l| a[i * k + l] * b[l * n + j]).sum();
        }
    }
    c
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut at = vec![0.0_f32; cols * rows];
    for i in 0..rows {
        for j in 0..cols {
            at[j * rows + i] = a[i * cols + j];
        }
    }
    at
}

/// Two-Stage Least Squares (2SLS) instrumental variable estimator.
pub struct TwoSls {
    pub coef: Vec<f32>,
    pub n_instruments: usize,
    pub n_covariates: usize,
}

impl TwoSls {
    pub fn fit(y: &[f32], t: &[f32], z: &[f32], n: usize, n_z: usize) -> CausalResult<Self> {
        if y.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if y.len() != n || t.len() != n || z.len() != n * n_z {
            return Err(CausalError::IncompatibleData);
        }

        // Stage 1: T_hat = Z(Z^TZ)^{-1}Z^TT
        let zt = transpose(z, n, n_z); // [n_z * n]
        // Z^TZ [n_z * n_z]
        let ztz = mat_mul_rect(&zt, z, n_z, n, n_z);
        let ztz_inv = gauss_jordan_inv(&ztz, n_z, 1e-4)?;
        // Z^TT [n_z]
        let ztt: Vec<f32> = (0..n_z)
            .map(|i| (0..n).map(|k| z[k * n_z + i] * t[k]).sum())
            .collect();
        // (Z^TZ)^{-1} Z^TT [n_z]
        let beta1: Vec<f32> = (0..n_z)
            .map(|i| (0..n_z).map(|j| ztz_inv[i * n_z + j] * ztt[j]).sum())
            .collect();
        // T_hat = Z * beta1 [n]
        let t_hat: Vec<f32> = (0..n)
            .map(|i| (0..n_z).map(|j| z[i * n_z + j] * beta1[j]).sum())
            .collect();

        // Stage 2: beta = (T_hat^T T_hat)^{-1} T_hat^T Y
        let t_hat_sq: f32 = t_hat.iter().map(|&v| v * v).sum();
        if t_hat_sq.abs() < 1e-10 {
            return Err(CausalError::MatrixSingular);
        }
        let t_hat_y: f32 = t_hat.iter().zip(y.iter()).map(|(&th, &yi)| th * yi).sum();
        let beta2 = t_hat_y / t_hat_sq;

        Ok(Self {
            coef: vec![beta2],
            n_instruments: n_z,
            n_covariates: 1,
        })
    }

    pub fn predict(&self, t_hat: &[f32]) -> Vec<f32> {
        let beta = self.coef[0];
        t_hat.iter().map(|&th| th * beta).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sls_basic() {
        let n = 30;
        let n_z = 2;
        // Z -> T -> Y with Z as instrument
        let z: Vec<f32> = (0..n * n_z).map(|i| i as f32 / (n * n_z) as f32).collect();
        let t: Vec<f32> = (0..n).map(|i| z[i * n_z] + 0.1).collect();
        let y: Vec<f32> = (0..n).map(|i| t[i] * 2.0 + 0.5).collect();
        let result = TwoSls::fit(&y, &t, &z, n, n_z).unwrap();
        assert!(result.coef[0].is_finite());
    }
}
