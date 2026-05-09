use crate::discovery::notears::ols;
use crate::error::{CausalError, CausalResult};

pub struct SLearner {
    model_w: Vec<f32>,
    model_b: f32,
    pub n_features: usize,
}

pub struct TLearner {
    w0: Vec<f32>,
    b0: f32,
    w1: Vec<f32>,
    b1: f32,
    pub n_features: usize,
}

pub struct XLearner {
    pub s_learner: SLearner,
    d0_w: Vec<f32>,
    d0_b: f32,
    d1_w: Vec<f32>,
    d1_b: f32,
}

fn build_augmented_x(x: &[f32], t: &[f32], n: usize, d: usize) -> Vec<f32> {
    // Augmented matrix [X | T]: n rows, d+1 cols
    let mut aug = vec![0.0_f32; n * (d + 1)];
    for i in 0..n {
        for j in 0..d {
            aug[i * (d + 1) + j] = x[i * d + j];
        }
        aug[i * (d + 1) + d] = t[i];
    }
    aug
}

fn fit_linear(x_mat: &[f32], y: &[f32], n: usize, d: usize) -> CausalResult<(Vec<f32>, f32)> {
    // Add intercept column
    let d1 = d + 1;
    let mut x_aug = vec![0.0_f32; n * d1];
    for i in 0..n {
        for j in 0..d {
            x_aug[i * d1 + j] = x_mat[i * d + j];
        }
        x_aug[i * d1 + d] = 1.0;
    }
    let beta = ols(&x_aug, y, n, d1)?;
    let w = beta[..d].to_vec();
    let b = beta[d];
    Ok((w, b))
}

fn predict_linear(w: &[f32], b: f32, x: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let dot: f32 = w
                .iter()
                .zip(x[i * d..(i + 1) * d].iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum();
            dot + b
        })
        .collect()
}

impl SLearner {
    pub fn fit(x: &[f32], t: &[f32], y: &[f32], n: usize, d: usize) -> CausalResult<Self> {
        if n == 0 || x.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        let aug = build_augmented_x(x, t, n, d);
        let (model_w, model_b) = fit_linear(&aug, y, n, d + 1)?;
        Ok(Self {
            model_w,
            model_b,
            n_features: d,
        })
    }
}

impl TLearner {
    pub fn fit(x: &[f32], t: &[f32], y: &[f32], n: usize, d: usize) -> CausalResult<Self> {
        if n == 0 || x.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        // Split into treated and control
        let control_idx: Vec<usize> = (0..n).filter(|&i| t[i] < 0.5).collect();
        let treated_idx: Vec<usize> = (0..n).filter(|&i| t[i] >= 0.5).collect();

        if control_idx.is_empty() || treated_idx.is_empty() {
            return Err(CausalError::IncompatibleData);
        }

        let n0 = control_idx.len();
        let n1 = treated_idx.len();

        let mut x0 = vec![0.0_f32; n0 * d];
        let mut y0 = vec![0.0_f32; n0];
        for (k, &i) in control_idx.iter().enumerate() {
            x0[k * d..(k + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
            y0[k] = y[i];
        }

        let mut x1 = vec![0.0_f32; n1 * d];
        let mut y1 = vec![0.0_f32; n1];
        for (k, &i) in treated_idx.iter().enumerate() {
            x1[k * d..(k + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
            y1[k] = y[i];
        }

        let (w0, b0) = fit_linear(&x0, &y0, n0, d)?;
        let (w1, b1) = fit_linear(&x1, &y1, n1, d)?;

        Ok(Self {
            w0,
            b0,
            w1,
            b1,
            n_features: d,
        })
    }
}

impl XLearner {
    pub fn fit(x: &[f32], t: &[f32], y: &[f32], n: usize, d: usize) -> CausalResult<Self> {
        let t_learner_s = TLearner::fit(x, t, y, n, d)?;

        // Stage 2: pseudo-outcomes
        let mu0_all = predict_linear(&t_learner_s.w0, t_learner_s.b0, x, n, d);
        let mu1_all = predict_linear(&t_learner_s.w1, t_learner_s.b1, x, n, d);

        let treated_idx: Vec<usize> = (0..n).filter(|&i| t[i] >= 0.5).collect();
        let control_idx: Vec<usize> = (0..n).filter(|&i| t[i] < 0.5).collect();

        let n1 = treated_idx.len();
        let n0 = control_idx.len();

        if n0 == 0 || n1 == 0 {
            return Err(CausalError::IncompatibleData);
        }

        // D1[i] = Y[i] - mu0[i] for treated
        let mut x1 = vec![0.0_f32; n1 * d];
        let mut d1_outcomes = vec![0.0_f32; n1];
        for (k, &i) in treated_idx.iter().enumerate() {
            x1[k * d..(k + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
            d1_outcomes[k] = y[i] - mu0_all[i];
        }

        // D0[i] = mu1[i] - Y[i] for control
        let mut x0 = vec![0.0_f32; n0 * d];
        let mut d0_outcomes = vec![0.0_f32; n0];
        for (k, &i) in control_idx.iter().enumerate() {
            x0[k * d..(k + 1) * d].copy_from_slice(&x[i * d..(i + 1) * d]);
            d0_outcomes[k] = mu1_all[i] - y[i];
        }

        let (d1_w, d1_b) = fit_linear(&x1, &d1_outcomes, n1, d)?;
        let (d0_w, d0_b) = fit_linear(&x0, &d0_outcomes, n0, d)?;

        // Build S-learner for combining via propensity
        let s_learner = SLearner::fit(x, t, y, n, d)?;

        Ok(Self {
            s_learner,
            d0_w,
            d0_b,
            d1_w,
            d1_b,
        })
    }
}

pub fn cate_slearner(model: &SLearner, x: &[f32], n: usize) -> Vec<f32> {
    let d = model.n_features;
    let t1 = vec![1.0_f32; n];
    let t0 = vec![0.0_f32; n];
    let aug1 = build_augmented_x(x, &t1, n, d);
    let aug0 = build_augmented_x(x, &t0, n, d);
    let mu1 = predict_linear(&model.model_w, model.model_b, &aug1, n, d + 1);
    let mu0 = predict_linear(&model.model_w, model.model_b, &aug0, n, d + 1);
    mu1.iter()
        .zip(mu0.iter())
        .map(|(&m1, &m0)| m1 - m0)
        .collect()
}

pub fn cate_tlearner(model: &TLearner, x: &[f32], n: usize) -> Vec<f32> {
    let d = model.n_features;
    let mu1 = predict_linear(&model.w1, model.b1, x, n, d);
    let mu0 = predict_linear(&model.w0, model.b0, x, n, d);
    mu1.iter()
        .zip(mu0.iter())
        .map(|(&m1, &m0)| m1 - m0)
        .collect()
}

pub fn cate_xlearner(model: &XLearner, x: &[f32], propensity: &[f32], n: usize) -> Vec<f32> {
    let d = model.s_learner.n_features;
    let tau1 = predict_linear(&model.d1_w, model.d1_b, x, n, d);
    let tau0 = predict_linear(&model.d0_w, model.d0_b, x, n, d);
    (0..n)
        .map(|i| {
            let pi = propensity[i].clamp(0.05, 0.95);
            // Combine: tau = pi * tau0 + (1-pi) * tau1
            pi * tau0[i] + (1.0 - pi) * tau1[i]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slearner_cate_shape() {
        let n = 20;
        let d = 2;
        let x: Vec<f32> = (0..n * d).map(|i| i as f32 / 40.0).collect();
        let t: Vec<f32> = (0..n)
            .map(|i: usize| if i.is_multiple_of(2) { 1.0 } else { 0.0 })
            .collect();
        let y: Vec<f32> = (0..n).map(|i| x[i * d] + t[i] * 0.5).collect();
        let model = SLearner::fit(&x, &t, &y, n, d).unwrap();
        let cate = cate_slearner(&model, &x, n);
        assert_eq!(cate.len(), n);
    }
}
