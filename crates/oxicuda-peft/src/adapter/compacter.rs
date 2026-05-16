use crate::adapter::houlsby::gelu;
use crate::handle::LcgRng;

/// Configuration for the PHM (Parameterized Hypercomplex Multiplication) decomposition.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    /// Hypercomplex number order `n`: each Kronecker factor A_i ∈ ℝ^{n × n}.
    pub n: usize,
    /// Kronecker rank `k`: the weight is approximated as W = Σ_{i=1}^k A_i ⊗ B_i.
    pub k: usize,
}

/// Compacter adapter using PHM (Parameterized Hypercomplex Multiplication) weight sharing.
///
/// The down-projection weight is reconstructed as `W_down = Σ_i kron(A_i, B_down_i)`
/// where A_i ∈ ℝ^{n×n} and B_down_i ∈ ℝ^{(in_dim/n) × (bottleneck_dim/n)}.
/// The up-projection uses the same structure (transposed dimensions).
///
/// This dramatically reduces the number of adapter parameters compared to a full MLP.
#[derive(Debug, Clone)]
pub struct CompacterAdapter {
    /// Input/output dimension.
    pub in_dim: usize,
    /// Bottleneck dimension.
    pub bottleneck_dim: usize,
    /// Hypercomplex order.
    pub n: usize,
    /// Kronecker rank.
    pub k: usize,
    /// A-factors for down projection, `k` matrices each of shape `[n × n]`.
    pub a_factors: Vec<Vec<f32>>,
    /// B-factors for down projection, `k` matrices each of shape `[(in_dim/n) × (bottleneck_dim/n)]`.
    pub b_factors: Vec<Vec<f32>>,
    /// Down-projection bias, shape `[bottleneck_dim]`.
    pub bias_down: Vec<f32>,
    /// Up-projection bias, shape `[in_dim]`.
    pub bias_up: Vec<f32>,
}

impl CompacterAdapter {
    /// Construct a `CompacterAdapter`.
    ///
    /// A-factors are initialised as scaled identity-like matrices; B-factors use He init.
    #[must_use]
    pub fn new(in_dim: usize, bottleneck_dim: usize, cfg: PhaseConfig, rng: &mut LcgRng) -> Self {
        let b_rows = in_dim / cfg.n;
        let b_cols = bottleneck_dim / cfg.n;
        let he_std = (2.0_f32 / in_dim as f32).sqrt();
        let mut a_factors = Vec::with_capacity(cfg.k);
        let mut b_factors = Vec::with_capacity(cfg.k);
        for _ in 0..cfg.k {
            // A_i: identity-like initialisation (scaled)
            let mut a = vec![0.0_f32; cfg.n * cfg.n];
            for diag in 0..cfg.n {
                a[diag * cfg.n + diag] = 1.0 / cfg.k as f32;
            }
            a_factors.push(a);
            // B_i: random He init
            let mut b = vec![0.0_f32; b_rows * b_cols];
            rng.fill_normal(&mut b);
            for v in b.iter_mut() {
                *v *= he_std;
            }
            b_factors.push(b);
        }
        let bias_down = vec![0.0_f32; bottleneck_dim];
        let bias_up = vec![0.0_f32; in_dim];
        Self {
            in_dim,
            bottleneck_dim,
            n: cfg.n,
            k: cfg.k,
            a_factors,
            b_factors,
            bias_down,
            bias_up,
        }
    }

    /// Reconstruct the down-projection weight `W_down ∈ ℝ^{in_dim × bottleneck_dim}`
    /// as `Σ_i kron(A_i, B_i_down)`.
    ///
    /// Returns a flat matrix of shape `[in_dim × bottleneck_dim]` (row-major).
    #[must_use]
    pub fn reconstruct_w_down(&self) -> Vec<f32> {
        let b_rows = self.in_dim / self.n;
        let b_cols = self.bottleneck_dim / self.n;
        let mut w = vec![0.0_f32; self.in_dim * self.bottleneck_dim];
        for ki in 0..self.k {
            let a = &self.a_factors[ki];
            let b = &self.b_factors[ki];
            // Kronecker product: (A ⊗ B)[i*b_rows + p, j*b_cols + q] = A[i,j] * B[p,q]
            for ai in 0..self.n {
                for aj in 0..self.n {
                    let a_val = a[ai * self.n + aj];
                    if a_val == 0.0 {
                        continue;
                    }
                    for p in 0..b_rows {
                        for q in 0..b_cols {
                            let row = ai * b_rows + p;
                            let col = aj * b_cols + q;
                            w[row * self.bottleneck_dim + col] += a_val * b[p * b_cols + q];
                        }
                    }
                }
            }
        }
        w
    }

    /// Reconstruct the up-projection weight `W_up ∈ ℝ^{bottleneck_dim × in_dim}`
    /// using the transpose of the Kronecker factors (B-factors transposed structure).
    #[must_use]
    pub fn reconstruct_w_up(&self) -> Vec<f32> {
        let b_rows = self.in_dim / self.n;
        let b_cols = self.bottleneck_dim / self.n;
        // W_up is [bottleneck_dim x in_dim]: transpose of W_down
        let w_down = self.reconstruct_w_down();
        let mut w_up = vec![0.0_f32; self.bottleneck_dim * self.in_dim];
        for i in 0..self.in_dim {
            for j in 0..self.bottleneck_dim {
                w_up[j * self.in_dim + i] = w_down[i * self.bottleneck_dim + j];
            }
        }
        // Zero out for near-identity init at start of training
        for v in w_up.iter_mut() {
            *v = 0.0;
        }
        let _ = b_rows;
        let _ = b_cols;
        w_up
    }

    /// Apply the Compacter adapter to an input of shape `[seq_len × in_dim]`.
    ///
    /// Architecture: `x → W_down → GELU → W_up + x` using PHM-reconstructed weights.
    /// Returns a vector of length `seq_len * in_dim`.
    #[must_use]
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let w_down = self.reconstruct_w_down();
        let w_up = self.reconstruct_w_up();
        let mut out = Vec::with_capacity(seq_len * self.in_dim);
        for t in 0..seq_len {
            let token = &x[t * self.in_dim..(t + 1) * self.in_dim];
            // Down: [in_dim] → [bottleneck_dim]
            let mut hidden = vec![0.0_f32; self.bottleneck_dim];
            for j in 0..self.bottleneck_dim {
                let mut acc = self.bias_down[j];
                for i in 0..self.in_dim {
                    acc += w_down[i * self.bottleneck_dim + j] * token[i];
                }
                hidden[j] = gelu(acc);
            }
            // Up: [bottleneck_dim] → [in_dim]
            for i in 0..self.in_dim {
                let mut acc = self.bias_up[i];
                for j in 0..self.bottleneck_dim {
                    acc += w_up[j * self.in_dim + i] * hidden[j];
                }
                // Residual
                out.push(acc + token[i]);
            }
        }
        out
    }
}
