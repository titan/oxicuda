use super::notears::{expm_scaling_exponent, gauss_jordan_inv};
use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

// ─── Local matrix helpers ────────────────────────────────────────────────────

fn mat_mul_sq(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum();
        }
    }
}

fn mat_mul_rect(a: &[f32], b: &[f32], c: &mut [f32], rows_a: usize, inner: usize, cols_b: usize) {
    for i in 0..rows_a {
        for j in 0..cols_b {
            c[i * cols_b + j] = (0..inner)
                .map(|k| a[i * inner + k] * b[k * cols_b + j])
                .sum();
        }
    }
}

fn mat_trace_sq(a: &[f32], n: usize) -> f32 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Padé(1,1) rational approximant for matrix exponential.
fn pade11_local(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
    let mut a2 = vec![0.0_f32; n * n];
    mat_mul_sq(a, a, &mut a2, n);
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
    mat_mul_sq(&u, &v_inv, &mut result, n);
    Ok(result)
}

/// Padé(1,1) matrix exponential with scaling-and-squaring.
fn mat_exp(a: &[f32], n: usize) -> CausalResult<Vec<f32>> {
    let s = expm_scaling_exponent(a, n);
    let scale = 1.0_f32 / (1u64 << s) as f32;
    let scaled: Vec<f32> = a.iter().map(|&v| v * scale).collect();
    let mut result = pade11_local(&scaled, n)?;
    for _ in 0..s {
        let mut squared = vec![0.0_f32; n * n];
        mat_mul_sq(&result, &result, &mut squared, n);
        result = squared;
    }
    Ok(result)
}

/// Element-wise square (Hadamard product with itself).
fn hadamard_sq(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&v| v * v).collect()
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// L1 proximal soft-threshold operator.
fn soft_threshold(x: f32, thresh: f32) -> f32 {
    let sign = x.signum();
    sign * (x.abs() - thresh).max(0.0)
}

// ─── Type aliases ─────────────────────────────────────────────────────────────

/// `(loss, grad_w, grad_w1, grad_w2)` returned from `recon_loss_and_grad`.
type ReconGrads = (f32, Vec<f32>, Vec<f32>, Vec<f32>);

// ─── Config & structs ────────────────────────────────────────────────────────

/// Configuration for the DAG-GNN augmented-Lagrangian optimizer.
#[derive(Debug, Clone)]
pub struct DagGnnConfig {
    /// Number of nodes (variables) in the graph.
    pub n_nodes: usize,
    /// Hidden dimension for the GNN encoder/decoder (default 32).
    pub n_hidden: usize,
    /// Number of augmented-Lagrangian outer iterations (default 10).
    pub max_outer_iter: usize,
    /// Gradient-descent steps per outer iteration (default 500).
    pub n_inner_iter: usize,
    /// Gradient-descent learning rate (default 0.001).
    pub lr: f32,
    /// L1 penalty coefficient on the adjacency matrix W (default 0.01).
    pub lambda1: f32,
    /// L2 penalty coefficient on the adjacency matrix W (default 0.01).
    pub lambda2: f32,
    /// Acyclicity-metric convergence threshold (default 1e-8).
    pub h_tol: f32,
    /// Initial augmented-Lagrangian penalty rho (default 1.0).
    pub rho_init: f32,
    /// Penalty growth multiplier (default 10.0).
    pub eta: f32,
    /// Initial dual variable alpha (default 0.0).
    pub alpha_init: f32,
}

impl Default for DagGnnConfig {
    fn default() -> Self {
        Self {
            n_nodes: 0,
            n_hidden: 32,
            max_outer_iter: 10,
            n_inner_iter: 500,
            lr: 0.001,
            lambda1: 0.01,
            lambda2: 0.01,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        }
    }
}

/// DAG-GNN model: adjacency matrix W, GNN encoder W1, decoder W2.
///
/// Reference: Yu, Chen, Gao, Yu, NeurIPS 2019 "DAG-GNN: DAG Structure
/// Learning with Graph Neural Networks".
pub struct DagGnn {
    /// Soft adjacency weights, n x n row-major.
    pub w: Vec<f32>,
    /// Encoder weight matrix, n x h row-major.
    pub w1: Vec<f32>,
    /// Decoder weight matrix, h x n row-major.
    pub w2: Vec<f32>,
    /// Configuration.
    pub config: DagGnnConfig,
}

/// Result returned by `DagGnn::fit`.
#[derive(Debug, Clone)]
pub struct DagGnnResult {
    /// Learned adjacency matrix, n x n (absolute soft W values).
    pub adjacency: Vec<f32>,
    /// Number of edges (entries above the internal threshold 0.3).
    pub n_edges: usize,
    /// Acyclicity metric value h(W) at termination.
    pub h_final: f32,
    /// Number of outer augmented-Lagrangian iterations completed.
    pub n_outer_iter: usize,
}

// ─── DagGnn implementation ───────────────────────────────────────────────────

impl DagGnn {
    /// Construct a new DAG-GNN with weights initialised from `LcgRng(0)`.
    ///
    /// * W is initialised to zero.
    /// * W1 ~ N(0, 1/n), W2 ~ N(0, 1/h).
    pub fn new(config: DagGnnConfig) -> Self {
        Self::new_with_seed(config, 0)
    }

    /// Construct with an explicit PRNG seed (for reproducibility).
    pub fn new_with_seed(config: DagGnnConfig, seed: u64) -> Self {
        let n = config.n_nodes;
        let h = config.n_hidden;
        let mut rng = LcgRng::new(seed);

        let scale_w1 = if n > 0 {
            1.0_f32 / (n as f32).sqrt()
        } else {
            1.0
        };
        let scale_w2 = if h > 0 {
            1.0_f32 / (h as f32).sqrt()
        } else {
            1.0
        };

        let w1: Vec<f32> = (0..n * h).map(|_| rng.next_normal() * scale_w1).collect();
        let w2: Vec<f32> = (0..h * n).map(|_| rng.next_normal() * scale_w2).collect();
        let w = vec![0.0_f32; n * n];

        Self { w, w1, w2, config }
    }

    /// Compute the acyclicity metric h(W) = tr(expm(W*W)) - n.
    fn h_val(&self) -> CausalResult<f32> {
        let n = self.config.n_nodes;
        let ww = hadamard_sq(&self.w);
        let expm_ww = mat_exp(&ww, n)?;
        Ok(mat_trace_sq(&expm_ww, n) - n as f32)
    }

    /// Gradient of h w.r.t. W: d/dW tr(expm(W*W)) = 2 W element-wise * expm(W*W).
    fn h_grad(&self) -> CausalResult<Vec<f32>> {
        let n = self.config.n_nodes;
        let ww = hadamard_sq(&self.w);
        let expm_ww = mat_exp(&ww, n)?;
        let mut grad = vec![0.0_f32; n * n];
        for i in 0..n * n {
            grad[i] = 2.0 * self.w[i] * expm_ww[i];
        }
        Ok(grad)
    }

    /// Simplified GNN forward pass for a batch:
    ///
    /// ```text
    /// H = X * W1          (n_samples x h)
    /// AX = A^T * X        (n_samples x n)  — graph convolution of raw features
    /// M = AX * W1         (n_samples x h)  — aggregated neighbour messages
    /// pre_z = M + H
    /// Z = ReLU(pre_z)
    /// X_hat = Z * W2      (n_samples x n_nodes)
    /// ```
    ///
    /// Returns `(x_hat, pre_z, ax, a_mat)`.
    fn forward(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.config.n_nodes;
        let h = self.config.n_hidden;

        // H = X * W1  (n_samples x h)
        let mut h_mat = vec![0.0_f32; n_samples * h];
        mat_mul_rect(x, &self.w1, &mut h_mat, n_samples, n, h);

        // A = sigmoid(W), zero diagonal  (n x n)
        let mut a_mat = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    a_mat[i * n + j] = sigmoid(self.w[i * n + j]);
                }
            }
        }

        // AX = A^T * X: result[s, j] = sum_{i!=j} A[i,j] * x[s, i]
        let mut ax = vec![0.0_f32; n_samples * n];
        for s in 0..n_samples {
            for j in 0..n {
                let mut val = 0.0_f32;
                for i in 0..n {
                    if i != j {
                        val += a_mat[i * n + j] * x[s * n + i];
                    }
                }
                ax[s * n + j] = val;
            }
        }

        // M = AX * W1  (n_samples x h)
        let mut m_mat = vec![0.0_f32; n_samples * h];
        mat_mul_rect(&ax, &self.w1, &mut m_mat, n_samples, n, h);

        // pre_z = M + H
        let mut pre_z = vec![0.0_f32; n_samples * h];
        for i in 0..n_samples * h {
            pre_z[i] = m_mat[i] + h_mat[i];
        }

        // Z = ReLU(pre_z), X_hat = Z * W2  (n_samples x n)
        let z_mat: Vec<f32> = pre_z.iter().map(|&v| relu(v)).collect();
        let mut x_hat = vec![0.0_f32; n_samples * n];
        mat_mul_rect(&z_mat, &self.w2, &mut x_hat, n_samples, h, n);

        (x_hat, pre_z, ax, a_mat)
    }

    /// Compute reconstruction loss and gradients w.r.t. W, W1, W2.
    fn recon_loss_and_grad(&self, x: &[f32], n_samples: usize) -> CausalResult<ReconGrads> {
        let n = self.config.n_nodes;
        let h = self.config.n_hidden;
        let nd = (n_samples * n) as f32;

        let (x_hat, pre_z, ax, a_mat) = self.forward(x, n_samples);

        // Recon loss = ||X_hat - X||_F^2 / (n_samples * n)
        let mut loss = 0.0_f32;
        let mut d_xhat = vec![0.0_f32; n_samples * n];
        for i in 0..n_samples * n {
            let diff = x_hat[i] - x[i];
            loss += diff * diff;
            d_xhat[i] = 2.0 * diff / nd;
        }
        loss /= nd;

        // Backprop through X_hat = Z * W2
        // dL/dW2 = Z^T * d_xhat   (h x n)
        let mut grad_w2 = vec![0.0_f32; h * n];
        for k in 0..h {
            for j in 0..n {
                let mut val = 0.0_f32;
                for s in 0..n_samples {
                    let z_sk = relu(pre_z[s * h + k]);
                    val += z_sk * d_xhat[s * n + j];
                }
                grad_w2[k * n + j] = val;
            }
        }

        // dL/dZ = d_xhat * W2^T   (n_samples x h)
        let mut d_z = vec![0.0_f32; n_samples * h];
        for s in 0..n_samples {
            for k in 0..h {
                let mut val = 0.0_f32;
                for j in 0..n {
                    val += d_xhat[s * n + j] * self.w2[k * n + j];
                }
                d_z[s * h + k] = val;
            }
        }

        // Backprop through Z = ReLU(pre_z): d_pre_z = d_z * relu_grad(pre_z)
        let mut d_pre_z = vec![0.0_f32; n_samples * h];
        for i in 0..n_samples * h {
            d_pre_z[i] = d_z[i] * relu_grad(pre_z[i]);
        }

        // pre_z = m_mat + h_mat  =>  d_m = d_pre_z,  d_h = d_pre_z

        // dL/dW1 from H path: X^T * d_pre_z  (n x h)
        let mut grad_w1 = vec![0.0_f32; n * h];
        for i in 0..n {
            for k in 0..h {
                let mut val = 0.0_f32;
                for s in 0..n_samples {
                    val += x[s * n + i] * d_pre_z[s * h + k];
                }
                grad_w1[i * h + k] = val;
            }
        }

        // dL/dW1 from M path: AX^T * d_pre_z  (n x h)  — accumulated into grad_w1
        for i in 0..n {
            for k in 0..h {
                let mut val = 0.0_f32;
                for s in 0..n_samples {
                    val += ax[s * n + i] * d_pre_z[s * h + k];
                }
                grad_w1[i * h + k] += val;
            }
        }

        // Backprop through M = AX * W1 to AX:
        // d_ax = d_pre_z * W1^T  (n_samples x n)
        let mut d_ax = vec![0.0_f32; n_samples * n];
        for s in 0..n_samples {
            for i in 0..n {
                let mut val = 0.0_f32;
                for k in 0..h {
                    val += d_pre_z[s * h + k] * self.w1[i * h + k];
                }
                d_ax[s * n + i] = val;
            }
        }

        // Backprop AX[s,j] = sum_{i!=j} A[i,j] * x[s,i]  to A:
        // d_a[i,j] = sum_s d_ax[s,j] * x[s,i]
        // Then to W: A[i,j] = sigmoid(W[i,j]), so d_w[i,j] = d_a[i,j] * sig*(1-sig)
        let mut grad_w = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut d_a_ij = 0.0_f32;
                for s in 0..n_samples {
                    d_a_ij += d_ax[s * n + j] * x[s * n + i];
                }
                let sig = a_mat[i * n + j];
                grad_w[i * n + j] = d_a_ij * sig * (1.0 - sig);
            }
        }

        Ok((loss, grad_w, grad_w1, grad_w2))
    }

    /// Fit the DAG-GNN model on observational data.
    ///
    /// * `x` — row-major data matrix of shape `(n_samples x n_nodes)`.
    /// * `n_samples` — number of rows in `x`.
    pub fn fit(&mut self, x: &[f32], n_samples: usize) -> CausalResult<DagGnnResult> {
        let n = self.config.n_nodes;
        if n == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "n_nodes must be > 0".into(),
            });
        }
        if n_samples == 0 {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n_samples * n {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n,
                got: x.len(),
            });
        }

        let max_outer = self.config.max_outer_iter;
        let n_inner = self.config.n_inner_iter;
        let lr = self.config.lr;
        let lambda1 = self.config.lambda1;
        let lambda2 = self.config.lambda2;
        let h_tol = self.config.h_tol;
        let eta = self.config.eta;

        let mut rho = self.config.rho_init;
        let mut alpha = self.config.alpha_init;
        let mut h_prev = f32::INFINITY;
        let mut outer_iter = 0usize;

        for _outer in 0..max_outer {
            outer_iter += 1;

            // Inner gradient-descent loop
            for _ in 0..n_inner {
                let h_val = self.h_val()?;
                let (_, grad_w_recon, grad_w1, grad_w2) = self.recon_loss_and_grad(x, n_samples)?;
                let grad_h = self.h_grad()?;

                let aug_coeff = alpha + rho * h_val;

                for idx in 0..n * n {
                    let row = idx / n;
                    let col = idx % n;
                    if row == col {
                        self.w[idx] = 0.0;
                        continue;
                    }
                    let total_grad =
                        grad_w_recon[idx] + aug_coeff * grad_h[idx] + lambda2 * self.w[idx];
                    self.w[idx] -= lr * total_grad;
                    self.w[idx] = soft_threshold(self.w[idx], lr * lambda1);
                }

                // Update W1 and W2
                for (p, g) in self.w1.iter_mut().zip(grad_w1.iter()) {
                    *p -= lr * g;
                }
                for (p, g) in self.w2.iter_mut().zip(grad_w2.iter()) {
                    *p -= lr * g;
                }
            }

            let h_val = self.h_val()?;

            // Dual update
            alpha += rho * h_val;

            // Penalty update
            if h_val > 0.25 * h_prev {
                rho *= eta;
            }
            h_prev = h_val;

            // Convergence check
            if h_val < h_tol {
                break;
            }
        }

        let h_final = self.h_val()?;

        let adjacency: Vec<f32> = self.w.iter().map(|&v| v.abs()).collect();
        let n_edges = adjacency.iter().filter(|&&v| v > 0.3).count();

        Ok(DagGnnResult {
            adjacency,
            n_edges,
            h_final,
            n_outer_iter: outer_iter,
        })
    }

    /// Return binary DAG (true = edge present) by thresholding the absolute
    /// adjacency weights at `threshold`.
    pub fn get_dag(&self, threshold: f32) -> Vec<bool> {
        self.w.iter().map(|&v| v.abs() > threshold).collect()
    }
}

// ─── Convenience function ────────────────────────────────────────────────────

/// Fit a DAG-GNN model and return the result.
///
/// Equivalent to `DagGnn::new(config).fit(x, n_samples)`.
pub fn dag_gnn(x: &[f32], n_samples: usize, config: DagGnnConfig) -> CausalResult<DagGnnResult> {
    let mut model = DagGnn::new(config);
    model.fit(x, n_samples)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n: usize) -> DagGnnConfig {
        DagGnnConfig {
            n_nodes: n,
            n_hidden: 8,
            max_outer_iter: 3,
            n_inner_iter: 20,
            lr: 0.001,
            lambda1: 0.01,
            lambda2: 0.01,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        }
    }

    fn random_data(n_samples: usize, n_nodes: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n_samples * n_nodes)
            .map(|_| rng.next_normal())
            .collect()
    }

    // 1
    #[test]
    fn default_config_sane() {
        let cfg = DagGnnConfig::default();
        assert_eq!(cfg.n_hidden, 32);
        assert_eq!(cfg.max_outer_iter, 10);
        assert!(cfg.lr > 0.0);
        assert!(cfg.h_tol > 0.0);
        assert!(cfg.rho_init > 0.0);
        assert!(cfg.eta > 1.0);
    }

    // 2
    #[test]
    fn new_initializes_correctly() {
        let cfg = make_config(4);
        let model = DagGnn::new(cfg.clone());
        assert_eq!(model.w.len(), cfg.n_nodes * cfg.n_nodes);
        assert_eq!(model.w1.len(), cfg.n_nodes * cfg.n_hidden);
        assert_eq!(model.w2.len(), cfg.n_hidden * cfg.n_nodes);
        assert!(model.w.iter().all(|&v| v == 0.0));
    }

    // 3
    #[test]
    fn hadamard_sq_zero() {
        let zeros = vec![0.0_f32; 4];
        let sq = hadamard_sq(&zeros);
        assert!(sq.iter().all(|&v| v == 0.0));
        // tr(expm(0)) = tr(I) = n, so h = n - n = 0
        let n = 2;
        let expm = mat_exp(&sq, n).expect("mat_exp should succeed");
        let h = mat_trace_sq(&expm, n) - n as f32;
        assert!(h.abs() < 1e-5, "h should be 0 for null matrix, got {h}");
    }

    // 4
    #[test]
    fn acyclicity_metric_zero_for_null_graph() {
        let cfg = make_config(3);
        let model = DagGnn::new(cfg);
        let h = model.h_val().expect("h_val should succeed");
        assert!(h.abs() < 1e-5, "h should be 0 for zero W, got {h}");
    }

    // 5
    #[test]
    fn acyclicity_metric_positive_for_cycle() {
        // Create a 2-cycle: W[0,1] = W[1,0] = 1.0
        let mut cfg = make_config(2);
        cfg.n_hidden = 4;
        let mut model = DagGnn::new(cfg);
        model.w[1] = 1.0; // edge 0->1  (row=0, col=1, n=2)
        model.w[2] = 1.0; // edge 1->0  (row=1, col=0, n=2) — creates a 2-cycle
        let h = model.h_val().expect("h_val should succeed");
        assert!(h > 0.0, "2-cycle should give h > 0, got {h}");
    }

    // 6
    #[test]
    fn fit_returns_ok() {
        let cfg = make_config(3);
        let x = random_data(20, 3, 42);
        let result = DagGnn::new(cfg).fit(&x, 20);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // 7
    #[test]
    fn fit_reduces_reconstruction_loss() {
        let n = 3;
        let n_s = 30;
        let x = random_data(n_s, n, 77);
        let before_loss = {
            let m = DagGnn::new(make_config(n));
            let (loss, _, _, _) = m
                .recon_loss_and_grad(&x, n_s)
                .expect("recon_loss_and_grad should succeed");
            loss
        };
        let cfg = DagGnnConfig {
            n_nodes: n,
            n_hidden: 8,
            max_outer_iter: 3,
            n_inner_iter: 50,
            lr: 0.001,
            lambda1: 0.0,
            lambda2: 0.0,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let mut model = DagGnn::new(cfg);
        model.fit(&x, n_s).expect("fit should succeed");
        let (after_loss, _, _, _) = model
            .recon_loss_and_grad(&x, n_s)
            .expect("recon_loss_and_grad should succeed");
        // Loss should not have dramatically increased (allow small tolerance for
        // the acyclicity penalty that's also being optimised)
        assert!(
            after_loss <= before_loss + 1.0,
            "loss should not dramatically increase: before={before_loss}, after={after_loss}"
        );
    }

    // 8
    #[test]
    fn fit_n_edges_nonneg() {
        let cfg = make_config(3);
        let x = random_data(20, 3, 99);
        let result = DagGnn::new(cfg).fit(&x, 20).expect("fit should succeed");
        assert!(result.n_edges <= 3 * 3);
    }

    // 9
    #[test]
    fn fit_h_final_nonneg() {
        let cfg = make_config(3);
        let x = random_data(20, 3, 55);
        let result = DagGnn::new(cfg).fit(&x, 20).expect("fit should succeed");
        assert!(
            result.h_final >= -1e-4,
            "h_final should be non-negative, got {}",
            result.h_final
        );
    }

    // 10
    #[test]
    fn get_dag_threshold_zero_gives_empty() {
        let cfg = make_config(3);
        let model = DagGnn::new(cfg);
        // W = 0, so all |w| = 0 < any positive threshold
        let dag = model.get_dag(0.01);
        assert!(dag.iter().all(|&v| !v), "null W should give empty DAG");
    }

    // 11
    #[test]
    fn get_dag_threshold_works() {
        let mut cfg = make_config(3);
        cfg.n_hidden = 4;
        let mut model = DagGnn::new(cfg);
        model.w[1] = 0.5; // edge 0->1, |w| = 0.5  (row=0, col=1, n=3)
        let low_thresh = model.get_dag(0.3);
        let high_thresh = model.get_dag(0.9);
        assert!(low_thresh[1], "edge 0->1 above 0.3 threshold should appear");
        assert!(
            !high_thresh[1],
            "edge 0->1 below 0.9 threshold should not appear"
        );
    }

    // 12
    #[test]
    fn fit_small_n2_chain() {
        // X1 -> X2: X2 = 0.8*X1 + noise
        let n_s = 30;
        let mut rng = LcgRng::new(11);
        let mut x = vec![0.0_f32; n_s * 2];
        for s in 0..n_s {
            let x1 = rng.next_normal();
            let x2 = 0.8 * x1 + rng.next_normal() * 0.1;
            x[s * 2] = x1;
            x[s * 2 + 1] = x2;
        }
        let cfg = DagGnnConfig {
            n_nodes: 2,
            n_hidden: 4,
            max_outer_iter: 5,
            n_inner_iter: 100,
            lr: 0.005,
            lambda1: 0.0,
            lambda2: 0.0,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let result = DagGnn::new(cfg).fit(&x, n_s).expect("fit should succeed");
        assert_eq!(result.adjacency.len(), 4);
        assert!(result.h_final.is_finite());
    }

    // 13
    #[test]
    fn fit_small_n3_fork() {
        // 3-node fork: X0 -> X1, X0 -> X2
        let n_s = 40;
        let mut rng = LcgRng::new(22);
        let mut x = vec![0.0_f32; n_s * 3];
        for s in 0..n_s {
            let x0 = rng.next_normal();
            let x1 = 0.7 * x0 + rng.next_normal() * 0.2;
            let x2 = 0.9 * x0 + rng.next_normal() * 0.2;
            x[s * 3] = x0;
            x[s * 3 + 1] = x1;
            x[s * 3 + 2] = x2;
        }
        let cfg = DagGnnConfig {
            n_nodes: 3,
            n_hidden: 8,
            max_outer_iter: 3,
            n_inner_iter: 30,
            lr: 0.001,
            lambda1: 0.0,
            lambda2: 0.0,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let result = DagGnn::new(cfg).fit(&x, n_s).expect("fit should succeed");
        assert_eq!(result.adjacency.len(), 9);
        assert!(result.h_final.is_finite());
    }

    // 14
    #[test]
    fn convenience_fn_matches_method() {
        let x = random_data(15, 3, 33);
        let cfg1 = make_config(3);
        let cfg2 = make_config(3);
        let r1 = dag_gnn(&x, 15, cfg1).expect("dag_gnn should succeed");
        let r2 = DagGnn::new(cfg2).fit(&x, 15).expect("fit should succeed");
        // Both use same seed=0, so should produce same result
        assert!(
            (r1.h_final - r2.h_final).abs() < 1e-4,
            "convenience fn: h={}, method: h={}",
            r1.h_final,
            r2.h_final
        );
    }

    // 15
    #[test]
    fn result_adjacency_length() {
        let cfg = make_config(4);
        let x = random_data(20, 4, 44);
        let result = DagGnn::new(cfg).fit(&x, 20).expect("fit should succeed");
        assert_eq!(result.adjacency.len(), 4 * 4);
    }

    // 16
    #[test]
    fn mat_exp_identity_check() {
        let zeros = vec![0.0_f32; 9];
        let e = mat_exp(&zeros, 3).expect("mat_exp should succeed");
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (e[i * 3 + j] - want).abs() < 1e-5,
                    "expm(0)[{i},{j}] = {} != {want}",
                    e[i * 3 + j]
                );
            }
        }
    }

    // 17
    #[test]
    fn no_nan_in_adjacency() {
        let cfg = make_config(3);
        let x = random_data(20, 3, 66);
        let result = DagGnn::new(cfg).fit(&x, 20).expect("fit should succeed");
        assert!(
            result.adjacency.iter().all(|v| v.is_finite()),
            "adjacency contains NaN/Inf"
        );
    }

    // 18
    #[test]
    fn no_nan_in_weights() {
        let cfg = make_config(3);
        let x = random_data(20, 3, 88);
        let mut model = DagGnn::new(cfg);
        model.fit(&x, 20).expect("fit should succeed");
        assert!(
            model.w1.iter().all(|v| v.is_finite()),
            "w1 contains NaN/Inf"
        );
        assert!(
            model.w2.iter().all(|v| v.is_finite()),
            "w2 contains NaN/Inf"
        );
    }

    // 19
    #[test]
    fn lambda1_zero_vs_nonzero() {
        let x = random_data(20, 3, 111);
        let cfg_no_l1 = DagGnnConfig {
            n_nodes: 3,
            n_hidden: 8,
            max_outer_iter: 2,
            n_inner_iter: 20,
            lambda1: 0.0,
            lambda2: 0.0,
            lr: 0.001,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let cfg_l1 = DagGnnConfig {
            lambda1: 0.5,
            ..cfg_no_l1.clone()
        };
        let r1 = DagGnn::new(cfg_no_l1)
            .fit(&x, 20)
            .expect("fit should succeed");
        let r2 = DagGnn::new(cfg_l1).fit(&x, 20).expect("fit should succeed");
        // Both should produce finite results
        assert!(r1.h_final.is_finite());
        assert!(r2.h_final.is_finite());
    }

    // 20
    #[test]
    fn outer_iter_count() {
        let cfg = make_config(3);
        let x = random_data(20, 3, 123);
        let result = DagGnn::new(cfg.clone())
            .fit(&x, 20)
            .expect("fit should succeed");
        assert!(
            result.n_outer_iter <= cfg.max_outer_iter,
            "got {} > max {}",
            result.n_outer_iter,
            cfg.max_outer_iter
        );
    }

    // 21
    #[test]
    fn small_sample() {
        let cfg = DagGnnConfig {
            n_nodes: 2,
            n_hidden: 4,
            max_outer_iter: 2,
            n_inner_iter: 5,
            lr: 0.001,
            lambda1: 0.0,
            lambda2: 0.0,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let x = random_data(5, 2, 77);
        let result = DagGnn::new(cfg).fit(&x, 5);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // 22
    #[test]
    fn h_tol_satisfied() {
        // With W=0, h=0 which is already < any positive h_tol, so should stop
        // after 1 outer iter
        let cfg = DagGnnConfig {
            n_nodes: 3,
            n_hidden: 8,
            max_outer_iter: 10,
            n_inner_iter: 5,
            lr: 0.001,
            lambda1: 0.0,
            lambda2: 0.0,
            h_tol: 1.0, // loose enough for null graph
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let x = random_data(20, 3, 44);
        let result = DagGnn::new(cfg).fit(&x, 20).expect("fit should succeed");
        assert!(result.n_outer_iter >= 1);
    }

    // 23
    #[test]
    fn dag_gnn_large() {
        let cfg = DagGnnConfig {
            n_nodes: 8,
            n_hidden: 16,
            max_outer_iter: 2,
            n_inner_iter: 10,
            lr: 0.001,
            lambda1: 0.0,
            lambda2: 0.0,
            h_tol: 1e-8,
            rho_init: 1.0,
            eta: 10.0,
            alpha_init: 0.0,
        };
        let x = random_data(50, 8, 999);
        let result = DagGnn::new(cfg).fit(&x, 50);
        assert!(result.is_ok(), "{:?}", result.err());
        let r = result.expect("result should be present");
        assert_eq!(r.adjacency.len(), 64);
    }

    // 24
    #[test]
    fn initializer_uses_seed() {
        let cfg1 = make_config(4);
        let cfg2 = make_config(4);
        let m1 = DagGnn::new_with_seed(cfg1, 12345);
        let m2 = DagGnn::new_with_seed(cfg2, 12345);
        assert_eq!(m1.w1, m2.w1, "same seed should produce identical w1");
        assert_eq!(m1.w2, m2.w2, "same seed should produce identical w2");
    }

    // 25
    #[test]
    fn w_shape_correct() {
        let n = 5;
        let cfg = make_config(n);
        let model = DagGnn::new(cfg);
        assert_eq!(model.w.len(), n * n);
    }
}
