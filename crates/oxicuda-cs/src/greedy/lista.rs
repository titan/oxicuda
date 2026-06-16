//! LISTA — Learned ISTA (Gregor-LeCun 2010).
//!
//! "Learning Fast Approximations of Sparse Coding."
//!
//! T-layer unrolled ISTA with shared learned weight matrices W and S:
//!
//! ```text
//! x_0 = 0
//! for t in 0..T:
//!   x_{t+1} = h_λ( W * y + S * x_t )
//! where:
//!   h_λ(v) = sign(v) * max(|v| - λ, 0)   [soft threshold]
//!   W ≈ (1/L) * D^T                       [learned encoder weight, m × n]
//!   S ≈ I - (1/L) * D * D^T               [learned recurrent weight, m × m]
//!   L = ||D||_F^2
//! ```

use crate::error::{CsError, CsResult};

/// Configuration for LISTA.
#[derive(Debug, Clone)]
pub struct ListaConfig {
    /// Signal / measurement dimension (n).
    pub n_measurements: usize,
    /// Dictionary atom / code dimension (m).
    pub n_atoms: usize,
    /// Soft-threshold value (λ).
    pub threshold: f64,
    /// Number of unrolled ISTA layers (T).
    pub n_layers: usize,
}

/// LISTA encoder: T-layer unrolled ISTA with fixed learned matrices derived from a dictionary.
#[derive(Debug, Clone)]
pub struct Lista {
    /// Encoder weight W: [m × n] row-major.
    w: Vec<f64>,
    /// Recurrent weight S: [m × m] row-major.
    s: Vec<f64>,
    config: ListaConfig,
}

impl Lista {
    /// Build a LISTA encoder from dictionary `dict` of shape [m × n] (row-major).
    ///
    /// Computes:
    /// - `L = ||dict||_F^2`
    /// - `W = (1/L) * dict`  (same shape [m × n], since W encodes y∈ℝⁿ → code∈ℝᵐ)
    /// - `S = I_m - (1/L) * dict * dict^T`  shape [m × m]
    ///
    /// `dict` must have length `m * n`.
    pub fn from_dict(dict: &[f64], config: ListaConfig) -> CsResult<Self> {
        let m = config.n_atoms;
        let n = config.n_measurements;
        if dict.len() != m * n {
            return Err(CsError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![dict.len()],
            });
        }
        if m == 0 || n == 0 {
            return Err(CsError::InvalidParameter(
                "n_atoms and n_measurements must be > 0".into(),
            ));
        }

        // L = ||dict||_F^2
        let l: f64 = dict.iter().map(|&v| v * v).sum();
        let inv_l = if l > 0.0 { 1.0 / l } else { 1.0 };

        // W = (1/L) * dict,  shape [m × n].
        let w: Vec<f64> = dict.iter().map(|&v| inv_l * v).collect();

        // S = I_m - (1/L) * dict * dict^T,  shape [m × m].
        // (dict * dict^T)[i, k] = Σ_j dict[i*n+j] * dict[k*n+j]
        let mut s = vec![0.0_f64; m * m];
        for i in 0..m {
            for k in 0..m {
                let mut dot = 0.0_f64;
                for j in 0..n {
                    dot += dict[i * n + j] * dict[k * n + j];
                }
                let identity_val = if i == k { 1.0 } else { 0.0 };
                s[i * m + k] = identity_val - inv_l * dot;
            }
        }

        Ok(Self { w, s, config })
    }

    /// Encode a single measurement vector `y` of length `n_measurements`.
    ///
    /// Runs T unrolled ISTA steps and returns a sparse code of length `n_atoms`.
    pub fn encode(&self, y: &[f64]) -> CsResult<Vec<f64>> {
        let m = self.config.n_atoms;
        let n = self.config.n_measurements;
        if y.len() != n {
            return Err(CsError::DimensionMismatch { a: y.len(), b: n });
        }
        let lam = self.config.threshold;
        let t = self.config.n_layers;

        // x_0 = 0
        let mut x = vec![0.0_f64; m];

        // Precompute W * y (constant across all layers).
        let wy = matvec(&self.w, y, m, n);

        for _ in 0..t {
            // u = W*y + S*x_t
            let sx = matvec(&self.s, &x, m, m);
            for i in 0..m {
                x[i] = Self::soft_threshold(wy[i] + sx[i], lam);
            }
        }
        Ok(x)
    }

    /// Encode a batch of `n_batch` measurement vectors stored row-major in `ys`
    /// (length `n_batch * n_measurements`).
    ///
    /// Returns a flat Vec of length `n_batch * n_atoms` (row-major).
    pub fn encode_batch(&self, ys: &[f64], n_batch: usize) -> CsResult<Vec<f64>> {
        let n = self.config.n_measurements;
        let m = self.config.n_atoms;
        if ys.len() != n_batch * n {
            return Err(CsError::ShapeMismatch {
                expected: vec![n_batch, n],
                got: vec![ys.len()],
            });
        }
        let mut out = Vec::with_capacity(n_batch * m);
        for b in 0..n_batch {
            let y_b = &ys[b * n..(b + 1) * n];
            let code = self.encode(y_b)?;
            out.extend_from_slice(&code);
        }
        Ok(out)
    }

    /// Element-wise soft-threshold operator: `sign(v) * max(|v| - λ, 0)`.
    #[inline]
    #[must_use]
    pub fn soft_threshold(v: f64, lambda: f64) -> f64 {
        if v > lambda {
            v - lambda
        } else if v < -lambda {
            v + lambda
        } else {
            0.0
        }
    }

    /// Reconstruct an approximate signal from code: computes `D * code`.
    ///
    /// `dict` is [m × n] row-major; `code` has length `m`; output has length `n`.
    pub fn reconstruct(dict: &[f64], code: &[f64], n: usize, m: usize) -> CsResult<Vec<f64>> {
        if dict.len() != m * n {
            return Err(CsError::ShapeMismatch {
                expected: vec![m, n],
                got: vec![dict.len()],
            });
        }
        if code.len() != m {
            return Err(CsError::DimensionMismatch {
                a: code.len(),
                b: m,
            });
        }
        // y_hat[j] = Σ_i dict[i*n + j] * code[i]
        let mut y_hat = vec![0.0_f64; n];
        for i in 0..m {
            for j in 0..n {
                y_hat[j] += dict[i * n + j] * code[i];
            }
        }
        Ok(y_hat)
    }

    /// Return the number of dictionary atoms (m).
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.config.n_atoms
    }

    /// Return the signal / measurement dimension (n).
    #[must_use]
    pub fn n_measurements(&self) -> usize {
        self.config.n_measurements
    }

    /// Return the soft-threshold value (λ).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.config.threshold
    }
}

/// Matrix-vector product `A * x` where `A` is row-major `[out_dim × in_dim]`.
fn matvec(a: &[f64], x: &[f64], out_dim: usize, in_dim: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; out_dim];
    for i in 0..out_dim {
        let mut s = 0.0_f64;
        for j in 0..in_dim {
            s += a[i * in_dim + j] * x[j];
        }
        y[i] = s;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dict(m: usize, n: usize) -> Vec<f64> {
        (0..m * n)
            .map(|k| {
                let i = k / n;
                let j = k % n;
                if (i + j) % 3 == 0 { 0.5 } else { -0.1 }
            })
            .collect()
    }

    fn make_config(n: usize, m: usize, lambda: f64, layers: usize) -> ListaConfig {
        ListaConfig {
            n_measurements: n,
            n_atoms: m,
            threshold: lambda,
            n_layers: layers,
        }
    }

    // Test 1
    #[test]
    fn encode_output_shape() {
        let n = 8;
        let m = 12;
        let dict = make_dict(m, n);
        let cfg = make_config(n, m, 0.1, 10);
        let lista = Lista::from_dict(&dict, cfg).expect("ok");
        let y = vec![0.5_f64; n];
        let code = lista.encode(&y).expect("ok");
        assert_eq!(code.len(), m);
    }

    // Test 2
    #[test]
    fn encode_sparse() {
        let n = 8;
        let m = 12;
        let dict = make_dict(m, n);
        let cfg = make_config(n, m, 0.5, 20);
        let lista = Lista::from_dict(&dict, cfg).expect("ok");
        let y = vec![0.3_f64; n];
        let code = lista.encode(&y).expect("ok");
        let nnz = code.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(nnz < m, "expected sparse code, got {nnz}/{m} nonzeros");
    }

    // Test 3
    #[test]
    fn soft_threshold_positive() {
        let v = Lista::soft_threshold(1.5, 0.5);
        assert!((v - 1.0).abs() < 1.0e-12);
    }

    // Test 4
    #[test]
    fn soft_threshold_negative() {
        let v = Lista::soft_threshold(-1.5, 0.5);
        assert!((v + 1.0).abs() < 1.0e-12);
    }

    // Test 5
    #[test]
    fn soft_threshold_within_band() {
        let v = Lista::soft_threshold(0.3, 0.5);
        assert!(v.abs() < 1.0e-12);
    }

    // Test 6
    #[test]
    fn encode_all_finite() {
        let n = 10;
        let m = 15;
        let dict = make_dict(m, n);
        let cfg = make_config(n, m, 0.05, 15);
        let lista = Lista::from_dict(&dict, cfg).expect("ok");
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 - 0.5).collect();
        let code = lista.encode(&y).expect("ok");
        assert!(
            code.iter().all(|v| v.is_finite()),
            "code has non-finite value"
        );
    }

    // Test 7
    #[test]
    fn reconstruct_shape() {
        let n = 8;
        let m = 12;
        let dict = make_dict(m, n);
        let code = vec![0.1_f64; m];
        let y_hat = Lista::reconstruct(&dict, &code, n, m).expect("ok");
        assert_eq!(y_hat.len(), n);
    }

    // Test 8
    #[test]
    fn encode_batch_shape() {
        let n = 8;
        let m = 12;
        let n_batch = 5;
        let dict = make_dict(m, n);
        let cfg = make_config(n, m, 0.1, 10);
        let lista = Lista::from_dict(&dict, cfg).expect("ok");
        let ys = vec![0.4_f64; n_batch * n];
        let codes = lista.encode_batch(&ys, n_batch).expect("ok");
        assert_eq!(codes.len(), n_batch * m);
    }

    // Test 9
    #[test]
    fn larger_lambda_sparser() {
        let n = 8;
        let m = 12;
        let dict = make_dict(m, n);
        let y = vec![0.5_f64; n];

        let cfg_small = make_config(n, m, 0.05, 10);
        let lista_small = Lista::from_dict(&dict, cfg_small).expect("ok");
        let code_small = lista_small.encode(&y).expect("ok");

        let cfg_large = make_config(n, m, 0.5, 10);
        let lista_large = Lista::from_dict(&dict, cfg_large).expect("ok");
        let code_large = lista_large.encode(&y).expect("ok");

        let nnz_small = code_small.iter().filter(|&&v| v.abs() > 1e-10).count();
        let nnz_large = code_large.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(
            nnz_large <= nnz_small,
            "larger lambda should produce sparser code: nnz_small={nnz_small}, nnz_large={nnz_large}"
        );
    }

    // Test 10
    #[test]
    fn from_dict_dim_mismatch() {
        let n = 8;
        let m = 12;
        let bad_dict = vec![0.0_f64; m * n + 1];
        let cfg = make_config(n, m, 0.1, 10);
        let result = Lista::from_dict(&bad_dict, cfg);
        assert!(result.is_err(), "expected Err for dim mismatch");
    }

    // Test 11
    #[test]
    fn zero_layers_returns_zeros() {
        let n = 8;
        let m = 12;
        let dict = make_dict(m, n);
        let cfg = make_config(n, m, 0.1, 0);
        let lista = Lista::from_dict(&dict, cfg).expect("ok");
        let y = vec![1.0_f64; n];
        let code = lista.encode(&y).expect("ok");
        assert!(
            code.iter().all(|&v| v == 0.0),
            "zero layers must return all zeros"
        );
    }
}
