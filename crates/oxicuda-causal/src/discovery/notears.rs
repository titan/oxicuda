use crate::error::{CausalError, CausalResult};

pub(crate) fn mat_mul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum();
        }
    }
}

fn mat_trace(a: &[f32], n: usize) -> f32 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Padé(1,1) scaling threshold: `‖A/2^s‖∞` is reduced below this before the
/// rational approximation is formed, keeping the truncation error small.
pub(crate) const EXPM_PADE_THETA: f32 = 0.5;

/// Infinity norm (maximum absolute row sum) of an `n × n` row-major matrix.
pub(crate) fn mat_inf_norm(a: &[f32], n: usize) -> f32 {
    let mut norm = 0.0_f32;
    for i in 0..n {
        let row_sum: f32 = (0..n).map(|j| a[i * n + j].abs()).sum();
        if row_sum > norm {
            norm = row_sum;
        }
    }
    norm
}

/// Number of halvings `s` so that `‖A/2^s‖∞ <= EXPM_PADE_THETA`.
pub(crate) fn expm_scaling_exponent(a: &[f32], n: usize) -> u32 {
    let norm = mat_inf_norm(a, n);
    if norm <= EXPM_PADE_THETA || !norm.is_finite() {
        return 0;
    }
    // s = ceil(log2(norm / theta)), capped at 63 so the subsequent `1u64 << s`
    // (and the `s` squaring steps) never overflow. Beyond 63 halvings the scaled
    // matrix is already vanishingly small, so the cap does not affect accuracy.
    let ratio = norm / EXPM_PADE_THETA;
    (ratio.log2().ceil().max(0.0) as u32).min(63)
}

/// Padé(1,1) rational approximation `(I + A/2 + A²/12)(I - A/2 + A²/12)^{-1}`.
/// Assumes `A` is already small in norm; callers scale beforehand.
fn pade11(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
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
    // expm ≈ U * V^{-1}: invert V via Gauss-Jordan (partial pivoting) then GEMM.
    let v_inv = gauss_jordan_inv(&v, n, 0.0)?;
    let mut result = vec![0.0_f32; n * n];
    mat_mul(&u, &v_inv, &mut result, n);
    Ok(result)
}

/// Padé(1,1) matrix exponential with scaling-and-squaring.
///
/// `expm(A) = (expm(A / 2^s))^(2^s)` where `s` is chosen so `‖A/2^s‖∞` is
/// small enough for the bare Padé(1,1) approximant to be accurate. The scaled
/// exponential is then squared `s` times to recover `expm(A)`.
pub(crate) fn expm_pade(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
    let s = expm_scaling_exponent(a, n);
    let scale = 1.0_f32 / (1u64 << s) as f32;
    let scaled: Vec<f32> = a.iter().map(|&v| v * scale).collect();
    let mut result = pade11(&scaled, n)?;
    for _ in 0..s {
        let mut squared = vec![0.0_f32; n * n];
        mat_mul(&result, &result, &mut squared, n);
        result = squared;
    }
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
        // grad_loss = (1/n) · Xᵀ(XW − X).
        //
        // Computed in two O(n·d²) passes (residual then Xᵀ·residual) rather than
        // the naive O(n·d³) form that recomputes (XW)[k,j] inside the (i,j) loop.
        let inv_n = 1.0 / n as f32;
        // R = XW − X, row-major n × d.
        let mut resid = vec![0.0_f32; n * d];
        for k in 0..n {
            let xrow = &x[k * d..(k + 1) * d];
            let rrow = &mut resid[k * d..(k + 1) * d];
            for j in 0..d {
                let mut xw = 0.0_f32;
                for (l, &xl) in xrow.iter().enumerate() {
                    xw += xl * self.w[l * d + j];
                }
                rrow[j] = xw - xrow[j];
            }
        }
        // grad = (1/n) · Xᵀ · R.
        let mut grad = vec![0.0_f32; d * d];
        for k in 0..n {
            let xrow = &x[k * d..(k + 1) * d];
            let rrow = &resid[k * d..(k + 1) * d];
            for i in 0..d {
                let xi = xrow[i];
                if xi == 0.0 {
                    continue;
                }
                let grow = &mut grad[i * d..(i + 1) * d];
                for j in 0..d {
                    grow[j] += xi * rrow[j];
                }
            }
        }
        for g in grad.iter_mut() {
            *g *= inv_n;
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

        // Augmented-Lagrangian NOTEARS (Zheng et al. 2018).
        //
        // We solve a sequence of unconstrained sub-problems
        //   min_W  L(W) + (rho/2)·h(W)² + alpha·h(W) + lambda·‖W‖₁
        // by proximal gradient descent, then ascend the dual variable `alpha`
        // and grow the penalty `rho` whenever the acyclicity residual `h(W)`
        // fails to shrink by the required factor. The outer loop terminates once
        // `h(W)` is below tolerance (the constraint is satisfied) — crucially we
        // never short-circuit on the *initial* W = 0 iterate, which is trivially
        // acyclic but fits nothing.
        let mut rho = 1.0_f32;
        let mut alpha = 0.0_f32;
        let base_lr = 0.01_f32;
        let h_tol = 1e-6_f32;
        // Cap the penalty modestly: with fixed-step proximal GD a runaway `rho`
        // only destabilises the iteration (and inflates ‖W‖ until the matrix
        // exponential overflows). 1e6 is ample to enforce acyclicity here.
        let max_rho = 1e6_f32;
        let outer_rounds = 20_usize;
        let inner_iters = (max_iter / outer_rounds).max(20);
        let mut h_prev = f32::INFINITY;

        for _round in 0..outer_rounds {
            // Inner proximal-gradient descent on the augmented objective.
            for _inner in 0..inner_iters {
                let h = self.h_func()?;
                let grad_loss = self.compute_gradient(x, n);
                let grad_h = self.h_gradient()?;
                // Coefficient on the acyclicity gradient: d/dW[(rho/2)h² + alpha·h]
                //                                        = (rho·h + alpha)·∂h/∂W.
                let h_coef = rho * h + alpha;
                // Adaptive step: shrink the learning rate as the penalty
                // coefficient grows so the constraint term cannot blow the
                // iterate up; keeps `eff_lr·|h_coef|` ≈ O(base_lr).
                let eff_lr = base_lr / (1.0 + base_lr * h_coef.abs());
                let thresh = eff_lr * lambda;
                for (idx, w_val) in self.w.iter_mut().enumerate() {
                    let i = idx / d;
                    let j = idx % d;
                    if i == j {
                        *w_val = 0.0;
                        continue;
                    }
                    let aug_grad = grad_loss[idx] + h_coef * grad_h[idx];
                    // Gradient step on the smooth part.
                    let stepped = *w_val - eff_lr * aug_grad;
                    // Proximal operator of lambda·‖·‖₁ : soft-threshold toward 0,
                    //   prox(z) = sign(z)·max(|z| − eff_lr·lambda, 0).
                    *w_val = stepped.signum() * (stepped.abs() - thresh).max(0.0);
                }
            }

            let h_now = self.h_func()?;
            if h_now.abs() <= h_tol {
                return Ok(());
            }
            // Dual ascent.
            alpha += rho * h_now;
            // Penalty growth when the constraint did not shrink fast enough.
            if h_now.abs() > 0.25 * h_prev.abs() {
                rho = (rho * 10.0).min(max_rho);
            }
            h_prev = h_now;
        }

        let h_final = self.h_func()?;
        if h_final.abs() < 1e-3 {
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
        let inv = gauss_jordan_inv(&id, 2, 0.0).expect("gauss_jordan_inv should succeed");
        assert!((inv[0] - 1.0).abs() < 1e-5);
        assert!((inv[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn expm_pade_zero_is_identity() {
        // expm(0) = I.
        let a = vec![0.0_f32; 9];
        let e = expm_pade(&a, 3).expect("expm_pade should succeed");
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((e[i * 3 + j] - want).abs() < 1e-5, "expm(0) wrong");
            }
        }
    }

    #[test]
    fn expm_pade_diagonal_matches_scalar_exp() {
        // expm(diag(x)) = diag(exp(x)); covers the scaling-and-squaring path.
        let xs = [0.3_f32, -0.7, 1.6];
        let mut a = vec![0.0_f32; 9];
        for (i, &x) in xs.iter().enumerate() {
            a[i * 3 + i] = x;
        }
        let e = expm_pade(&a, 3).expect("expm_pade should succeed");
        for (i, &x) in xs.iter().enumerate() {
            assert!(
                (e[i * 3 + i] - x.exp()).abs() < 2e-3,
                "diag exp mismatch at {i}: got {}, want {}",
                e[i * 3 + i],
                x.exp()
            );
        }
        // off-diagonal must stay zero
        assert!((e[1]).abs() < 1e-4 && (e[3]).abs() < 1e-4);
    }

    #[test]
    fn expm_pade_nilpotent_is_exact_series() {
        // Strictly-upper nilpotent N (N^2 = 0): expm(N) = I + N exactly.
        let n = vec![0.0_f32, 0.4, 0.0, 0.0];
        let e = expm_pade(&n, 2).expect("expm_pade should succeed");
        assert!((e[0] - 1.0).abs() < 1e-4);
        assert!((e[1] - 0.4).abs() < 1e-3);
        assert!((e[2]).abs() < 1e-4);
        assert!((e[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn expm_scaling_exponent_grows_with_norm() {
        // Small norm needs no scaling; large norm needs s >= 1.
        let small = vec![0.1_f32, 0.0, 0.0, 0.1];
        assert_eq!(expm_scaling_exponent(&small, 2), 0);
        let large = vec![4.0_f32, 0.0, 0.0, 4.0];
        let s = expm_scaling_exponent(&large, 2);
        assert!(s >= 3, "norm 4 should scale down by >= 2^3, got s={s}");
        // After scaling, the scaled norm must fall at/below the threshold.
        let scale = 1.0_f32 / (1u64 << s) as f32;
        let scaled: Vec<f32> = large.iter().map(|&v| v * scale).collect();
        assert!(mat_inf_norm(&scaled, 2) <= EXPM_PADE_THETA + 1e-6);
    }

    #[test]
    fn expm_pade_large_norm_accurate() {
        // Without scaling-and-squaring the bare Padé(1,1) is badly wrong here;
        // the scaled path must still recover diag(exp(x)) accurately.
        let a = vec![3.0_f32, 0.0, 0.0, -2.5];
        let e = expm_pade(&a, 2).expect("expm_pade should succeed");
        assert!(
            (e[0] - 3.0_f32.exp()).abs() / 3.0_f32.exp() < 5e-3,
            "exp(3) mismatch: got {}",
            e[0]
        );
        assert!(
            (e[3] - (-2.5_f32).exp()).abs() < 5e-3,
            "exp(-2.5) mismatch: got {}",
            e[3]
        );
    }
}
