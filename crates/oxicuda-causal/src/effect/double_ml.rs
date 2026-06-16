use crate::discovery::notears::gauss_jordan_inv;
use crate::error::{CausalError, CausalResult};

fn linear_predict(w: &[f32], b: f32, x: &[f32], _d: usize) -> f32 {
    let dot: f32 = w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
    dot + b
}

fn fit_ols_intercept(
    x_mat: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
) -> CausalResult<(Vec<f32>, f32)> {
    let d1 = d + 1;
    let mut x_aug = vec![0.0_f32; n * d1];
    for i in 0..n {
        for j in 0..d {
            x_aug[i * d1 + j] = x_mat[i * d + j];
        }
        x_aug[i * d1 + d] = 1.0;
    }
    let mut xtx = vec![0.0_f32; d1 * d1];
    let mut xty = vec![0.0_f32; d1];
    for i in 0..n {
        for r in 0..d1 {
            for c in 0..d1 {
                xtx[r * d1 + c] += x_aug[i * d1 + r] * x_aug[i * d1 + c];
            }
            xty[r] += x_aug[i * d1 + r] * y[i];
        }
    }
    let inv = gauss_jordan_inv(&xtx, d1, 1e-3)?;
    let beta: Vec<f32> = (0..d1)
        .map(|i| (0..d1).map(|j| inv[i * d1 + j] * xty[j]).sum())
        .collect();
    Ok((beta[..d].to_vec(), beta[d]))
}

pub struct DoubleML {
    pub ate: f32,
    pub std_error: f32,
}

impl DoubleML {
    pub fn fit(
        y: &[f32],
        t: &[f32],
        x: &[f32],
        n: usize,
        n_features: usize,
        n_folds: usize,
    ) -> CausalResult<Self> {
        if n == 0 || y.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        if n_folds < 2 {
            return Err(CausalError::InvalidNumFolds { k: n_folds });
        }
        if y.len() != n || t.len() != n || x.len() != n * n_features {
            return Err(CausalError::IncompatibleData);
        }

        let fold_size = n / n_folds;
        if fold_size < 2 {
            return Err(CausalError::InvalidNumFolds { k: n_folds });
        }

        let mut y_tilde = vec![0.0_f32; n];
        let mut t_tilde = vec![0.0_f32; n];

        for fold in 0..n_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == n_folds - 1 {
                n
            } else {
                (fold + 1) * fold_size
            };
            let _n_test = test_end - test_start;

            // Train indices
            let train_idx: Vec<usize> = (0..n)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();
            let n_train = train_idx.len();
            if n_train < 2 {
                continue;
            }

            // Build train data
            let mut x_train = vec![0.0_f32; n_train * n_features];
            let mut y_train = vec![0.0_f32; n_train];
            let mut t_train = vec![0.0_f32; n_train];
            for (k, &i) in train_idx.iter().enumerate() {
                x_train[k * n_features..(k + 1) * n_features]
                    .copy_from_slice(&x[i * n_features..(i + 1) * n_features]);
                y_train[k] = y[i];
                t_train[k] = t[i];
            }

            // Fit g: E[Y|X]
            let (g_w, g_b) = fit_ols_intercept(&x_train, &y_train, n_train, n_features)?;
            // Fit m: E[T|X]
            let (m_w, m_b) = fit_ols_intercept(&x_train, &t_train, n_train, n_features)?;

            // Residuals on test set
            for i in test_start..test_end {
                let xi = &x[i * n_features..(i + 1) * n_features];
                let g_pred = linear_predict(&g_w, g_b, xi, n_features);
                let m_pred = linear_predict(&m_w, m_b, xi, n_features);
                y_tilde[i] = y[i] - g_pred;
                t_tilde[i] = t[i] - m_pred;
            }
        }

        // theta = mean(y_tilde * t_tilde) / mean(t_tilde^2)
        let num: f32 = y_tilde
            .iter()
            .zip(t_tilde.iter())
            .map(|(&yt, &tt)| yt * tt)
            .sum::<f32>()
            / n as f32;
        let denom: f32 = t_tilde.iter().map(|&tt| tt * tt).sum::<f32>() / n as f32;
        if denom.abs() < 1e-10 {
            return Err(CausalError::MatrixSingular);
        }
        let theta = num / denom;

        // Standard error via influence function
        // psi_i = (y_tilde[i] - theta * t_tilde[i]) * t_tilde[i] / denom
        let se_sq: f32 = y_tilde
            .iter()
            .zip(t_tilde.iter())
            .map(|(&yt, &tt)| {
                let psi = (yt - theta * tt) * tt / denom;
                psi * psi
            })
            .sum::<f32>()
            / (n as f32 * n as f32);

        let std_error = se_sq.sqrt();

        Ok(Self {
            ate: theta,
            std_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_ml_basic() {
        let n = 40;
        let d = 2;
        let x: Vec<f32> = (0..n * d).map(|i| i as f32 / (n * d) as f32).collect();
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f32> = (0..n).map(|i| x[i * d] * 0.5 + t[i] * 2.0).collect();
        let result = DoubleML::fit(&y, &t, &x, n, d, 4)
            .expect("DoubleML::fit should succeed for valid inputs");
        assert!(result.ate.is_finite());
        assert!(result.std_error >= 0.0);
    }
}
