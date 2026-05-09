use crate::error::{CausalError, CausalResult};

fn mat_mul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum();
        }
    }
}

fn mat_trace(a: &[f32], n: usize) -> f32 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Padé(3,3) approximation of matrix exponential for small matrices.
/// expm(A) ≈ (I + A/2 + A²/12)(I - A/2 + A²/12)^{-1}
fn expm_pade(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
    let mut a2 = vec![0.0_f32; n * n];
    mat_mul(a, a, &mut a2, n);

    let mut u = vec![0.0_f32; n * n];
    let mut v = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let identity = if i == j { 1.0_f32 } else { 0.0_f32 };
            u[i * n + j] = identity + a[i * n + j] * 0.5 + a2[i * n + j] / 12.0;
            v[i * n + j] = identity - a[i * n + j] * 0.5 + a2[i * n + j] / 12.0;
        }
    }
    // expm ≈ U * V^{-1}: solve V * result = U via Gauss-Jordan
    let v_inv = gauss_jordan_inv(&v, n, 0.0)?;
    let mut result = vec![0.0_f32; n * n];
    mat_mul(&u, &v_inv, &mut result, n);
    Ok(result)
}

pub(crate) fn gauss_jordan_inv(a: &[f32], n: usize, ridge: f32) -> CausalResult<Vec<f32>> {
    let mut m = a.to_vec();
    for i in 0..n {
        m[i * n + i] += ridge;
    }
    let mut inv = vec![0.0_f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                m[r1 * n + col]
                    .abs()
                    .partial_cmp(&m[r2 * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(CausalError::MatrixSingular)?;

        if m[pivot_row * n + col].abs() < 1e-12 {
            return Err(CausalError::MatrixSingular);
        }

        if pivot_row != col {
            for k in 0..n {
                m.swap(col * n + k, pivot_row * n + k);
                inv.swap(col * n + k, pivot_row * n + k);
            }
        }

        let pivot_val = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= pivot_val;
            inv[col * n + k] /= pivot_val;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row * n + col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for k in 0..n {
                let m_val = m[col * n + k];
                m[row * n + k] -= factor * m_val;
                let inv_val = inv[col * n + k];
                inv[row * n + k] -= factor * inv_val;
            }
        }
    }
    Ok(inv)
}

pub(crate) fn ols(x_mat: &[f32], y: &[f32], n: usize, d: usize) -> CausalResult<Vec<f32>> {
    // beta = (X^TX + ridge*I)^{-1} X^T y
    let mut xtx = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            xtx[i * d + j] = (0..n).map(|k| x_mat[k * d + i] * x_mat[k * d + j]).sum();
        }
    }
    let mut xty = vec![0.0_f32; d];
    for i in 0..d {
        xty[i] = (0..n).map(|k| x_mat[k * d + i] * y[k]).sum();
    }
    let inv = gauss_jordan_inv(&xtx, d, 1e-3)?;
    let mut beta = vec![0.0_f32; d];
    for i in 0..d {
        beta[i] = (0..d).map(|j| inv[i * d + j] * xty[j]).sum();
    }
    Ok(beta)
}

/// NOTEARS linear SEM (Zheng et al. 2018).
pub struct NotearsSem {
    pub w: Vec<f32>,
    pub d: usize,
}

impl NotearsSem {
    pub fn new(d: usize) -> Self {
        Self {
            w: vec![0.0_f32; d * d],
            d,
        }
    }

    fn h_func(&self) -> CausalResult<f32> {
        let d = self.d;
        // A = W ⊙ W (elementwise square)
        let a: Vec<f32> = self.w.iter().map(|&v| v * v).collect();
        let expm = expm_pade(&a, d)?;
        Ok(mat_trace(&expm, d) - d as f32)
    }

    fn compute_gradient(&self, x: &[f32], n: usize) -> Vec<f32> {
        let d = self.d;
        // grad_loss[i,j] = (1/n) * X^T(XW - X)[i,j]
        // = (1/n) * sum_k X[k,i] * (sum_l X[k,l]*W[l,j] - X[k,j])
        let mut grad = vec![0.0_f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut val = 0.0_f32;
                for k in 0..n {
                    let xw_kj: f32 = (0..d).map(|l| x[k * d + l] * self.w[l * d + j]).sum();
                    let resid = xw_kj - x[k * d + j];
                    val += x[k * d + i] * resid;
                }
                grad[i * d + j] = val / n as f32;
            }
        }
        grad
    }

    fn h_gradient(&self) -> CausalResult<Vec<f32>> {
        let d = self.d;
        // d/dW h(W) = 2 * W ⊙ (expm(W⊙W))^T
        let a: Vec<f32> = self.w.iter().map(|&v| v * v).collect();
        let expm = expm_pade(&a, d)?;
        let mut grad = vec![0.0_f32; d * d];
        for i in 0..d {
            for j in 0..d {
                grad[i * d + j] = 2.0 * self.w[i * d + j] * expm[j * d + i];
            }
        }
        Ok(grad)
    }

    pub fn fit(&mut self, x: &[f32], n: usize, lambda: f32, max_iter: usize) -> CausalResult<()> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        let d = self.d;
        if x.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: x.len(),
            });
        }

        let mut rho = 1.0_f32;
        let mut alpha = 0.0_f32;
        let lr = 0.001_f32;

        for iter in 0..max_iter {
            let h = self.h_func()?;
            if h.abs() < 1e-8 {
                return Ok(());
            }

            let grad_loss = self.compute_gradient(x, n);
            let grad_h = self.h_gradient()?;

            for (idx, w_val) in self.w.iter_mut().enumerate() {
                let i = idx / d;
                let j = idx % d;
                if i == j {
                    *w_val = 0.0;
                    continue;
                }
                // Gradient of augmented Lagrangian
                let gl = grad_loss[idx];
                let gh = grad_h[idx];
                let aug_grad = gl + (rho * h + alpha) * gh;

                // Gradient step
                *w_val -= lr * aug_grad;

                // Proximal operator for L1 (soft threshold)
                let sign = w_val.signum();
                *w_val = sign * (*w_val).abs().max(0.0) - lr * lambda;
                if w_val.abs() < lr * lambda {
                    *w_val = 0.0;
                }
            }

            // Update dual variable and penalty every few iterations
            if (iter + 1).is_multiple_of(10) {
                alpha += rho * h;
                if h.abs() > 0.25 * h.abs().max(1e-6) {
                    rho *= 2.0;
                }
            }
        }

        let h_final = self.h_func()?;
        if h_final.abs() < 1e-4 {
            Ok(())
        } else {
            Err(CausalError::NotearsDidNotConverge { iter: max_iter })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notears_new_zero() {
        let sem = NotearsSem::new(3);
        assert!(sem.w.iter().all(|&v| v == 0.0));
        assert_eq!(sem.d, 3);
    }

    #[test]
    fn gauss_jordan_identity() {
        let id = vec![1.0_f32, 0.0, 0.0, 1.0];
        let inv = gauss_jordan_inv(&id, 2, 0.0).unwrap();
        assert!((inv[0] - 1.0).abs() < 1e-5);
        assert!((inv[3] - 1.0).abs() < 1e-5);
    }
}
