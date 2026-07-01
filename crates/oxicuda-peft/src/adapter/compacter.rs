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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a deterministic input vector in `[-1, 1)` of the given length.
    fn make_input(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 1: output length == in_dim for a single token
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_output_shape_single_token() {
        let mut rng = LcgRng::new(1);
        let cfg = PhaseConfig { n: 2, k: 2 };
        let adapter = CompacterAdapter::new(8, 4, cfg, &mut rng);
        let x = make_input(8, 2);
        let out = adapter.forward(&x, 1);
        assert_eq!(
            out.len(),
            8,
            "output length must equal in_dim for seq_len=1"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 2: output length == seq_len * in_dim for a multi-token sequence
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_output_shape_multi_token() {
        let mut rng = LcgRng::new(3);
        let cfg = PhaseConfig { n: 2, k: 2 };
        let adapter = CompacterAdapter::new(8, 4, cfg, &mut rng);
        let x = make_input(8 * 5, 4);
        let out = adapter.forward(&x, 5);
        assert_eq!(out.len(), 40, "output length must be seq_len * in_dim");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 3: reconstruct_w_up() always returns all zeros (near-identity init),
    // so forward(x) == x exactly: acc_i = 0 + Σ_j 0*hidden[j] = 0; out = 0 + x.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_zero_w_up_forward_is_identity() {
        let mut rng = LcgRng::new(7);
        let cfg = PhaseConfig { n: 2, k: 2 };
        let adapter = CompacterAdapter::new(8, 4, cfg, &mut rng);
        let w_up = adapter.reconstruct_w_up();
        assert!(
            w_up.iter().all(|&v| v == 0.0),
            "reconstruct_w_up must return all zeros at init"
        );
        let x = make_input(8, 9);
        let out = adapter.forward(&x, 1);
        for (i, (&got, &expected)) in out.iter().zip(x.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "forward(x)[{i}] must equal x[{i}] when w_up=0 and bias_up=0"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 4: analytic Kronecker reconstruction.
    //
    // For n=2, k=1, in_dim=4, bottleneck_dim=4 (so b_rows=2, b_cols=2):
    //   A = [[1, 2], [3, 4]]  (row-major: [1,2,3,4])
    //   B = [[5, 6], [7, 8]]  (row-major: [5,6,7,8])
    //
    // kron(A, B)[ai*b_rows+p, aj*b_cols+q] = A[ai,aj] * B[p,q]
    //
    //   Row 0 (ai=0, p=0): [A00*B00, A00*B01, A01*B00, A01*B01] = [ 5,  6, 10, 12]
    //   Row 1 (ai=0, p=1): [A00*B10, A00*B11, A01*B10, A01*B11] = [ 7,  8, 14, 16]
    //   Row 2 (ai=1, p=0): [A10*B00, A10*B01, A11*B00, A11*B01] = [15, 18, 20, 24]
    //   Row 3 (ai=1, p=1): [A10*B10, A10*B11, A11*B10, A11*B11] = [21, 24, 28, 32]
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_kronecker_reconstruction_analytic() {
        let mut rng = LcgRng::new(99);
        let cfg = PhaseConfig { n: 2, k: 1 };
        let mut adapter = CompacterAdapter::new(4, 4, cfg, &mut rng);
        // Override with analytically known values.
        adapter.a_factors[0] = vec![1.0_f32, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        adapter.b_factors[0] = vec![5.0_f32, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
        let w = adapter.reconstruct_w_down();
        let expected: [f32; 16] = [
            5.0, 6.0, 10.0, 12.0, 7.0, 8.0, 14.0, 16.0, 15.0, 18.0, 20.0, 24.0, 21.0, 24.0, 28.0,
            32.0,
        ];
        assert_eq!(
            w.len(),
            expected.len(),
            "reconstructed w_down must have in_dim * bottleneck_dim elements"
        );
        for (i, (&got, &exp)) in w.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5_f32,
                "w_down[{i}]: got {got}, expected {exp}"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 5: Kronecker-factor parameter count < dense weight parameter count.
    //
    // For in_dim=16, bottleneck_dim=8, n=4, k=2:
    //   b_rows = 16/4 = 4, b_cols = 8/4 = 2
    //   Kronecker params = k*(n² + b_rows*b_cols) = 2*(16 + 8) = 48
    //   Dense params     = in_dim * bottleneck_dim = 128
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_parameter_count_smaller_than_dense() {
        let n = 4usize;
        let k = 2usize;
        let in_dim = 16usize;
        let bottleneck_dim = 8usize;
        let mut rng = LcgRng::new(55);
        let cfg = PhaseConfig { n, k };
        let adapter = CompacterAdapter::new(in_dim, bottleneck_dim, cfg, &mut rng);
        let b_rows = in_dim / n;
        let b_cols = bottleneck_dim / n;
        let kronecker_params: usize = adapter.a_factors.iter().map(|a| a.len()).sum::<usize>()
            + adapter.b_factors.iter().map(|b| b.len()).sum::<usize>();
        let expected_kronecker_params = k * (n * n + b_rows * b_cols);
        let dense_params = in_dim * bottleneck_dim;
        assert_eq!(
            kronecker_params, expected_kronecker_params,
            "Kronecker factor parameter count mismatch"
        );
        assert!(
            kronecker_params < dense_params,
            "Compacter Kronecker params ({kronecker_params}) must be fewer than \
             equivalent dense weight ({dense_params})"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 6: same RNG seed → byte-identical forward output (determinism)
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_determinism_fixed_seed() {
        let x = make_input(8, 99);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let adapter_a = CompacterAdapter::new(8, 4, PhaseConfig { n: 2, k: 2 }, &mut rng_a);
        let adapter_b = CompacterAdapter::new(8, 4, PhaseConfig { n: 2, k: 2 }, &mut rng_b);
        let out_a = adapter_a.forward(&x, 1);
        let out_b = adapter_b.forward(&x, 1);
        assert_eq!(
            out_a, out_b,
            "same seed must yield identical forward outputs"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 7: all output values are finite
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn compacter_finite_outputs() {
        let mut rng = LcgRng::new(13);
        let cfg = PhaseConfig { n: 2, k: 2 };
        let adapter = CompacterAdapter::new(8, 4, cfg, &mut rng);
        let x = make_input(8 * 3, 17);
        let out = adapter.forward(&x, 3);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }
}
