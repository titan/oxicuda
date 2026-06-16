use crate::discovery::notears::gauss_jordan_inv;
use crate::effect::propensity::PropensityModel;
use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

fn fit_outcome_model(
    x: &[f32],
    t: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
) -> CausalResult<Vec<f32>> {
    // Outcome model mu(X,T) = linear regression on [X, T, intercept]
    let d_aug = d + 2;
    let mut x_aug = vec![0.0_f32; n * d_aug];
    for i in 0..n {
        for j in 0..d {
            x_aug[i * d_aug + j] = x[i * d + j];
        }
        x_aug[i * d_aug + d] = t[i];
        x_aug[i * d_aug + d + 1] = 1.0;
    }
    let mut xtx = vec![0.0_f32; d_aug * d_aug];
    let mut xty = vec![0.0_f32; d_aug];
    for i in 0..n {
        for r in 0..d_aug {
            for c in 0..d_aug {
                xtx[r * d_aug + c] += x_aug[i * d_aug + r] * x_aug[i * d_aug + c];
            }
            xty[r] += x_aug[i * d_aug + r] * y[i];
        }
    }
    let inv = gauss_jordan_inv(&xtx, d_aug, 1e-3)?;
    let beta: Vec<f32> = (0..d_aug)
        .map(|i| (0..d_aug).map(|j| inv[i * d_aug + j] * xty[j]).sum())
        .collect();
    Ok(beta)
}

fn predict_outcome(beta: &[f32], x: &[f32], t: f32, d: usize) -> f32 {
    let dot: f32 = (0..d).map(|j| beta[j] * x[j]).sum();
    dot + beta[d] * t + beta[d + 1]
}

/// Augmented Inverse Probability Weighting (AIPW) ATE estimator.
pub fn aipw_ate(
    y: &[f32],
    t: &[f32],
    x: &[f32],
    n: usize,
    n_features: usize,
    lr: f32,
) -> CausalResult<f32> {
    if y.is_empty() || n == 0 {
        return Err(CausalError::EmptyInput);
    }
    if y.len() != n || t.len() != n || x.len() != n * n_features {
        return Err(CausalError::IncompatibleData);
    }

    // Fit outcome model
    let beta = fit_outcome_model(x, t, y, n, n_features)?;

    // Fit propensity model
    let mut rng = LcgRng::new(12345);
    let mut prop_model = PropensityModel::new(n_features, &mut rng);
    prop_model.fit(x, t, n, lr, 200)?;
    let pi = prop_model.predict(x, n)?;

    // AIPW estimator
    let tau: f32 = (0..n)
        .map(|i| {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let mu1 = predict_outcome(&beta, xi, 1.0, n_features);
            let mu0 = predict_outcome(&beta, xi, 0.0, n_features);
            let pi_i = pi[i].clamp(0.05, 0.95);
            (mu1 - mu0) + t[i] * (y[i] - mu1) / pi_i - (1.0 - t[i]) * (y[i] - mu0) / (1.0 - pi_i)
        })
        .sum::<f32>()
        / n as f32;

    Ok(tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aipw_finite() {
        let n = 20;
        let d = 2;
        let x: Vec<f32> = (0..n * d).map(|i| i as f32 / (n * d) as f32).collect();
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f32> = (0..n).map(|i| x[i * d] * 0.5 + t[i] * 1.0 + 0.1).collect();
        let ate =
            aipw_ate(&y, &t, &x, n, d, 0.01).expect("aipw_ate should succeed for valid inputs");
        assert!(ate.is_finite());
    }
}
