use super::notears::{expm_scaling_exponent, gauss_jordan_inv};
use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn mat_mul_nn(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum();
        }
    }
}

fn mat_trace(a: &[f32], n: usize) -> f32 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Padé(1,1) rational approximant `(I + A/2 + A²/12)(I - A/2 + A²/12)^{-1}`.
fn pade11(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
    let mut a2 = vec![0.0_f32; n * n];
    mat_mul_nn(a, a, &mut a2, n);
    let mut u = vec![0.0_f32; n * n];
    let mut v = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let id = if i == j { 1.0_f32 } else { 0.0_f32 };
            u[i * n + j] = id + a[i * n + j] * 0.5 + a2[i * n + j] / 12.0;
            v[i * n + j] = id - a[i * n + j] * 0.5 + a2[i * n + j] / 12.0;
        }
    }
    let v_inv = gauss_jordan_inv(&v, n, 0.0)?;
    let mut result = vec![0.0_f32; n * n];
    mat_mul_nn(&u, &v_inv, &mut result, n);
    Ok(result)
}

/// Padé(1,1) matrix exponential with scaling-and-squaring for acyclicity.
fn expm_pade(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
    let s = expm_scaling_exponent(a, n);
    let scale = 1.0_f32 / (1u64 << s) as f32;
    let scaled: Vec<f32> = a.iter().map(|&v| v * scale).collect();
    let mut result = pade11(&scaled, n)?;
    for _ in 0..s {
        let mut squared = vec![0.0_f32; n * n];
        mat_mul_nn(&result, &result, &mut squared, n);
        result = squared;
    }
    Ok(result)
}

/// NOTEARS-MLP: nonlinear structural equation model with acyclicity constraint.
pub struct NotearsNlp {
    pub d: usize,
    /// First layer weights: [d * d * h], block (i,j) = w1[i*d*h + j*h .. j*h+h]
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// Second layer: [d * h]
    w2: Vec<f32>,
    b2: Vec<f32>,
    h_size: usize,
}

impl NotearsNlp {
    pub fn new(d: usize, hidden: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / d as f32).sqrt();
        let w1: Vec<f32> = (0..d * d * hidden)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let b1 = vec![0.0_f32; d * hidden];
        let w2: Vec<f32> = (0..d * hidden).map(|_| rng.next_normal() * scale).collect();
        let b2 = vec![0.0_f32; d];
        Self {
            d,
            w1,
            b1,
            w2,
            b2,
            h_size: hidden,
        }
    }

    /// Forward pass for one sample x (length d), predicting x_hat (length d).
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let d = self.d;
        let h = self.h_size;
        // For each output variable j:
        // hidden_j[k] = relu(sum_i x[i] * w1[i*d*h + j*h + k] + b1[j*h + k])
        // out_j = sum_k hidden_j[k] * w2[j*h + k] + b2[j]
        (0..d)
            .map(|j| {
                let hidden: Vec<f32> = (0..h)
                    .map(|k| {
                        let pre: f32 = (0..d).map(|i| x[i] * self.w1[i * d * h + j * h + k]).sum();
                        relu(pre + self.b1[j * h + k])
                    })
                    .collect();
                let out: f32 = (0..h).map(|k| hidden[k] * self.w2[j * h + k]).sum();
                out + self.b2[j]
            })
            .collect()
    }

    /// Column norms of W1 blocks: A\[i,j\] = ||W1\[i,j,:\]||^2
    pub fn adjacency(&self) -> Vec<f32> {
        let d = self.d;
        let h = self.h_size;
        let mut a = vec![0.0_f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let norm_sq: f32 = (0..h).map(|k| self.w1[i * d * h + j * h + k].powi(2)).sum();
                a[i * d + j] = norm_sq;
            }
        }
        a
    }

    fn h_func(&self) -> CausalResult<f32> {
        let a = self.adjacency();
        let expm = expm_pade(&a, self.d)?;
        Ok(mat_trace(&expm, self.d) - self.d as f32)
    }

    pub fn fit(&mut self, x: &[f32], n: usize, lambda: f32, max_iter: usize) -> CausalResult<()> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        let d = self.d;
        let h = self.h_size;
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
            let h_val = self.h_func()?;
            if h_val.abs() < 1e-6 {
                return Ok(());
            }

            // Gradient of reconstruction loss w.r.t. w1, w2, b1, b2
            let mut grad_w1 = vec![0.0_f32; d * d * h];
            let mut grad_b1 = vec![0.0_f32; d * h];
            let mut grad_w2 = vec![0.0_f32; d * h];
            let mut grad_b2 = vec![0.0_f32; d];

            for sample in 0..n {
                let xi = &x[sample * d..(sample + 1) * d];
                let x_hat = self.forward(xi);
                for j in 0..d {
                    let err = (x_hat[j] - xi[j]) / n as f32;
                    grad_b2[j] += err;
                    // Hidden activations
                    let hidden: Vec<f32> = (0..h)
                        .map(|k| {
                            let pre: f32 =
                                (0..d).map(|i| xi[i] * self.w1[i * d * h + j * h + k]).sum();
                            relu(pre + self.b1[j * h + k])
                        })
                        .collect();
                    for k in 0..h {
                        grad_w2[j * h + k] += err * hidden[k];
                        let delta_h = err * self.w2[j * h + k];
                        let pre: f32 = (0..d).map(|i| xi[i] * self.w1[i * d * h + j * h + k]).sum();
                        let gate = relu_grad(pre + self.b1[j * h + k]);
                        grad_b1[j * h + k] += delta_h * gate;
                        for i in 0..d {
                            grad_w1[i * d * h + j * h + k] += delta_h * gate * xi[i];
                        }
                    }
                }
            }

            // Gradient of acyclicity constraint w.r.t. w1
            // dh/dA[i,j] = expm(A)^T[i,j], dA[i,j]/dw1[i,j,:] = 2*w1[i,j,:]
            let a = self.adjacency();
            if let Ok(expm_a) = expm_pade(&a, d) {
                let h_coeff = rho * h_val + alpha;
                for i in 0..d {
                    for j in 0..d {
                        let dh_da = expm_a[j * d + i]; // expm^T[i,j]
                        for k in 0..h {
                            let dw = 2.0 * self.w1[i * d * h + j * h + k] * dh_da;
                            grad_w1[i * d * h + j * h + k] += h_coeff * dw;
                        }
                    }
                }
            }

            // L1 proximal on w1
            for (idx, gw) in grad_w1.iter().enumerate() {
                self.w1[idx] -= lr * gw;
                let sign = self.w1[idx].signum();
                let mag = (self.w1[idx].abs() - lr * lambda).max(0.0);
                self.w1[idx] = sign * mag;
            }
            for (idx, gw) in grad_w2.iter().enumerate() {
                self.w2[idx] -= lr * gw;
            }
            for (idx, gb) in grad_b1.iter().enumerate() {
                self.b1[idx] -= lr * gb;
            }
            for (idx, gb) in grad_b2.iter().enumerate() {
                self.b2[idx] -= lr * gb;
            }

            if (iter + 1).is_multiple_of(20) {
                alpha += rho * h_val;
                if h_val.abs() > 0.5 {
                    rho = (rho * 2.0).min(1e6);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notears_nlp_adjacency_shape() {
        let mut rng = LcgRng::new(42);
        let model = NotearsNlp::new(3, 4, &mut rng);
        let adj = model.adjacency();
        assert_eq!(adj.len(), 9);
        assert!(adj.iter().all(|&v| v >= 0.0));
    }
}
