use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

pub struct PropensityModel {
    w: Vec<f32>,
    b: f32,
    pub n_features: usize,
}

impl PropensityModel {
    pub fn new(n_features: usize, rng: &mut LcgRng) -> Self {
        let scale = (1.0_f32 / n_features as f32).sqrt();
        let w = (0..n_features).map(|_| rng.next_normal() * scale).collect();
        Self {
            w,
            b: 0.0,
            n_features,
        }
    }

    fn logit(&self, x: &[f32]) -> f32 {
        let dot: f32 = x.iter().zip(self.w.iter()).map(|(&xi, &wi)| xi * wi).sum();
        dot + self.b
    }

    pub fn fit(
        &mut self,
        x: &[f32],
        t: &[f32],
        n: usize,
        lr: f32,
        n_epochs: usize,
    ) -> CausalResult<()> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        let d = self.n_features;
        if x.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: x.len(),
            });
        }
        if t.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: t.len(),
            });
        }

        for _ in 0..n_epochs {
            let mut grad_w = vec![0.0_f32; d];
            let mut grad_b = 0.0_f32;
            for i in 0..n {
                let xi = &x[i * d..(i + 1) * d];
                let pred = sigmoid(self.logit(xi));
                let err = (pred - t[i]) / n as f32;
                for j in 0..d {
                    grad_w[j] += err * xi[j];
                }
                grad_b += err;
            }
            for (j, gw) in grad_w.iter().enumerate() {
                self.w[j] -= lr * gw;
            }
            self.b -= lr * grad_b;
        }
        Ok(())
    }

    pub fn predict(&self, x: &[f32], n: usize) -> CausalResult<Vec<f32>> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        let d = self.n_features;
        if x.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: x.len(),
            });
        }
        Ok((0..n)
            .map(|i| {
                let xi = &x[i * d..(i + 1) * d];
                sigmoid(self.logit(xi)).clamp(0.05, 0.95)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propensity_output_range() {
        let mut rng = LcgRng::new(42);
        let mut model = PropensityModel::new(3, &mut rng);
        let x = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let t = vec![1.0_f32, 0.0, 1.0];
        model.fit(&x, &t, 3, 0.01, 100).expect("fit should succeed");
        let preds = model.predict(&x, 3).expect("predict should succeed");
        for &p in &preds {
            assert!((0.05..=0.95).contains(&p));
        }
    }
}
