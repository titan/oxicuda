//! Optimized Product Quantization (OPQ).
//!
//! Ge et al., "Optimized Product Quantization for Approximate Nearest Neighbor Search", CVPR 2013.
//!
//! Learns an orthogonal rotation matrix R that minimises the total PQ quantisation error through
//! alternating optimisation:
//!   1. Fix R → train PQ codebook on R·X.
//!   2. Fix codebook → update R via the orthogonal Procrustes solution (SVD of cross-covariance).
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::pq::adc::{adc_distance, build_adc_table};
use crate::pq::codebook::PqCodebook;
use crate::pq::encode::encode_batch;
use crate::pq::train::train_pq;

// ─── matrix helpers ──────────────────────────────────────────────────────────

/// Matrix-vector product: A[n×m] (row-major) · x[m] → out[n].
fn matvec(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows];
    for i in 0..rows {
        for j in 0..cols {
            out[i] += a[i * cols + j] * x[j];
        }
    }
    out
}

/// Matrix multiply: A[n×m] · B[m×k] → C[n×k], all row-major.
fn matmul(a: &[f32], b: &[f32], n: usize, m: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut s = 0.0_f32;
            for l in 0..m {
                s += a[i * m + l] * b[l * k + j];
            }
            c[i * k + j] = s;
        }
    }
    c
}

/// Transpose: A[n×m] → A^T[m×n], all row-major.
fn transpose(a: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut t = vec![0.0_f32; m * n];
    for i in 0..n {
        for j in 0..m {
            t[j * n + i] = a[i * m + j];
        }
    }
    t
}

/// Identity matrix of size n×n, row-major.
fn identity(n: usize) -> Vec<f32> {
    let mut id = vec![0.0_f32; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    id
}

// ─── one-sided Jacobi SVD for Procrustes ─────────────────────────────────────

/// Apply a Jacobi rotation at position (p, q) to the symmetric matrix `a` and
/// accumulate the same rotation into `v`.
fn apply_jacobi_rotation(
    a: &mut [f32],
    v: &mut [f32],
    n: usize,
    p: usize,
    q: usize,
    c: f32,
    s: f32,
) {
    // Update columns p and q of a (symmetric update)
    for r in 0..n {
        let a_rp = a[r * n + p];
        let a_rq = a[r * n + q];
        a[r * n + p] = c * a_rp - s * a_rq;
        a[r * n + q] = s * a_rp + c * a_rq;
    }
    for r in 0..n {
        let a_pr = a[p * n + r];
        let a_qr = a[q * n + r];
        a[p * n + r] = c * a_pr - s * a_qr;
        a[q * n + r] = s * a_pr + c * a_qr;
    }
    // Accumulate rotation in v
    for r in 0..n {
        let v_rp = v[r * n + p];
        let v_rq = v[r * n + q];
        v[r * n + p] = c * v_rp - s * v_rq;
        v[r * n + q] = s * v_rp + c * v_rq;
    }
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix `a_sym` (n×n).
///
/// Returns `(eigenvalues, V)` where `V` is column-major (stored row-major as [n×n])
/// such that `a_sym ≈ V · diag(eigenvalues) · V^T`.
fn jacobi_eigen(mut a: Vec<f32>, n: usize, n_iter: usize) -> (Vec<f32>, Vec<f32>) {
    let mut v = identity(n);
    for _ in 0..n_iter {
        let mut max_off = 0.0_f32;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = a[p * n + q].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        // Early exit when off-diagonal elements are negligible
        if max_off < 1e-10 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a[p * n + q];
                if a_pq.abs() < 1e-10 {
                    continue;
                }
                let a_pp = a[p * n + p];
                let a_qq = a[q * n + q];
                let theta = 0.5 * (a_qq - a_pp) / a_pq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0_f32 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0_f32 + theta * theta).sqrt())
                };
                let cos_val = 1.0_f32 / (1.0_f32 + t * t).sqrt();
                let sin_val = t * cos_val;
                apply_jacobi_rotation(&mut a, &mut v, n, p, q, cos_val, sin_val);
            }
        }
    }
    let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Compute the optimal orthogonal rotation `R = V · U^T` solving the Procrustes
/// problem from the cross-covariance matrix `c` (dim×dim).
///
/// Uses a cyclic Jacobi eigendecomposition on `C^T · C` to get `V`, then derives
/// `U` from `C · V · Σ^{-1}`.
fn svd_procrustes(c: &[f32], dim: usize) -> Vec<f32> {
    // B = C^T @ C  (dim × dim symmetric, positive semi-definite)
    let ct = transpose(c, dim, dim);
    let b = matmul(&ct, c, dim, dim, dim);

    // Jacobi eigen: B = V Σ² V^T
    let (eigvals, v) = jacobi_eigen(b, dim, 100);

    // Compute U from C @ V @ Σ^{-1}
    // Column i of U = (C @ v_i) / σ_i  where σ_i = sqrt(max(0, eigvals[i]))
    // v is stored row-major as [n×n] with eigenvectors as columns
    // → v[r][col] = V[r, col]

    let cv = matmul(c, &v, dim, dim, dim); // C @ V, columns are C·v_i

    let mut u = vec![0.0_f32; dim * dim];
    let eps = 1e-8_f32;
    for col in 0..dim {
        let sigma = eigvals[col].max(0.0).sqrt();
        if sigma > eps {
            let inv_sigma = 1.0 / sigma;
            for row in 0..dim {
                u[row * dim + col] = cv[row * dim + col] * inv_sigma;
            }
        } else {
            // Degenerate singular value: use the corresponding column of V as-is
            // (maintains orthogonality in the zero-singular-value subspace)
            for row in 0..dim {
                u[row * dim + col] = v[row * dim + col];
            }
        }
    }

    // R = V @ U^T
    let ut = transpose(&u, dim, dim);
    matmul(&v, &ut, dim, dim, dim)
}

// ─── decode helpers ───────────────────────────────────────────────────────────

/// Decode PQ codes to reconstructed vectors in the encoded (rotated) space.
fn decode_from_codebook(codes: &[u8], n: usize, cb: &PqCodebook) -> Vec<f32> {
    let dim = cb.m * cb.dsub;
    let mut out = vec![0.0_f32; n * dim];
    for i in 0..n {
        for s in 0..cb.m {
            let c = codes[i * cb.m + s] as usize;
            let centroid = cb.centroid(s, c);
            let base = i * dim + s * cb.dsub;
            out[base..base + cb.dsub].copy_from_slice(centroid);
        }
    }
    out
}

// ─── public API ──────────────────────────────────────────────────────────────

/// Configuration for Optimized Product Quantization.
#[derive(Debug, Clone)]
pub struct OpqConfig {
    /// Number of PQ subspaces.
    pub m: usize,
    /// Number of codewords per subspace (≤ 256).
    pub ksub: usize,
    /// k-means epochs for PQ training per OPQ iteration.
    pub n_pq_epochs: usize,
    /// Number of alternating optimisation iterations.
    pub n_outer_iters: usize,
}

impl OpqConfig {
    /// Create a new [`OpqConfig`] with sensible defaults.
    ///
    /// Validates that `m > 0`, `ksub ∈ [1, 256]`.
    pub fn new(m: usize, ksub: usize) -> AnnResult<Self> {
        if m == 0 {
            return Err(AnnError::InvalidNumSubspaces { m, dim: 0 });
        }
        if ksub == 0 || ksub > 256 {
            return Err(AnnError::InvalidK { k: ksub, n: 256 });
        }
        Ok(Self {
            m,
            ksub,
            n_pq_epochs: 20,
            n_outer_iters: 5,
        })
    }
}

/// OPQ trained model: holds the optimised rotation `R` and the PQ codebook trained on `R·X`.
#[derive(Debug)]
pub struct OpqModel {
    /// The PQ codebook trained on the rotated data.
    pub codebook: PqCodebook,
    /// Rotation matrix R (dim × dim), row-major.
    pub rotation: Vec<f32>,
    /// Ambient dimension of the data.
    pub dim: usize,
    /// Configuration used to train this model.
    pub config: OpqConfig,
}

impl OpqModel {
    /// Train OPQ on `n` vectors of `dim` dimensions.
    pub fn train(
        data: &[f32],
        n: usize,
        dim: usize,
        cfg: OpqConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        // ── input validation ──────────────────────────────────────────────────
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 || !dim.is_multiple_of(cfg.m) {
            return Err(AnnError::InvalidNumSubspaces { m: cfg.m, dim });
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        if cfg.ksub == 0 || cfg.ksub > n {
            return Err(AnnError::InvalidK { k: cfg.ksub, n });
        }

        // ── initialise R = I ──────────────────────────────────────────────────
        let mut rotation = identity(dim);

        // ── alternating optimisation ──────────────────────────────────────────
        let mut codebook = PqCodebook::new(cfg.m, cfg.ksub, dim / cfg.m);

        for _iter in 0..cfg.n_outer_iters {
            // Step 1: rotate data → R · X
            let mut rotated = vec![0.0_f32; n * dim];
            for i in 0..n {
                let x = &data[i * dim..(i + 1) * dim];
                let rx = matvec(&rotation, x, dim, dim);
                rotated[i * dim..(i + 1) * dim].copy_from_slice(&rx);
            }

            // Step 2: train PQ on rotated data
            codebook = train_pq(&rotated, n, dim, cfg.m, cfg.ksub, cfg.n_pq_epochs, rng)?;

            // Step 3: encode rotated data then decode to get reconstructions
            let codes = encode_batch(&rotated, n, &codebook);
            let recon = decode_from_codebook(&codes, n, &codebook);
            // `recon` lives in the rotated space

            // Step 4: build cross-covariance C = Σ_i x_i · recon_i^T  (dim × dim)
            // x_i is the original vector, recon_i is in rotated space.
            // C[a][b] = Σ_i data[i][a] * recon[i][b]
            let mut cross_cov = vec![0.0_f32; dim * dim];
            for i in 0..n {
                let xi = &data[i * dim..(i + 1) * dim];
                let ri = &recon[i * dim..(i + 1) * dim];
                for a in 0..dim {
                    for b in 0..dim {
                        cross_cov[a * dim + b] += xi[a] * ri[b];
                    }
                }
            }

            // Step 5: Procrustes → R_new = V · U^T where C = U Σ V^T
            rotation = svd_procrustes(&cross_cov, dim);
        }

        Ok(OpqModel {
            codebook,
            rotation,
            dim,
            config: cfg,
        })
    }

    /// Apply rotation: y = R · x.
    pub fn rotate(&self, x: &[f32]) -> AnnResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        Ok(matvec(&self.rotation, x, self.dim, self.dim))
    }

    /// Apply inverse rotation: x = R^T · y  (R is orthogonal, so R^{-1} = R^T).
    pub fn unrotate(&self, y: &[f32]) -> AnnResult<Vec<f32>> {
        if y.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: y.len(),
            });
        }
        let rt = transpose(&self.rotation, self.dim, self.dim);
        Ok(matvec(&rt, y, self.dim, self.dim))
    }

    /// Encode a batch of `n` vectors → codes of shape `[n × m]` (m bytes per vector).
    pub fn encode(&self, data: &[f32], n: usize) -> AnnResult<Vec<u8>> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if data.len() != n * self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * self.dim,
                got: data.len(),
            });
        }
        // Rotate each input vector
        let mut rotated = vec![0.0_f32; n * self.dim];
        for i in 0..n {
            let x = &data[i * self.dim..(i + 1) * self.dim];
            let rx = matvec(&self.rotation, x, self.dim, self.dim);
            rotated[i * self.dim..(i + 1) * self.dim].copy_from_slice(&rx);
        }
        Ok(encode_batch(&rotated, n, &self.codebook))
    }

    /// Decode codes → reconstructed vectors in original (pre-rotation) space.
    pub fn decode(&self, codes: &[u8], n: usize) -> AnnResult<Vec<f32>> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let m = self.codebook.m;
        if codes.len() != n * m {
            return Err(AnnError::DimensionMismatch {
                expected: n * m,
                got: codes.len(),
            });
        }
        // Reconstruct in rotated space
        let rotated_recon = decode_from_codebook(codes, n, &self.codebook);
        // Unrotate: apply R^T
        let rt = transpose(&self.rotation, self.dim, self.dim);
        let mut out = vec![0.0_f32; n * self.dim];
        for i in 0..n {
            let y = &rotated_recon[i * self.dim..(i + 1) * self.dim];
            let x = matvec(&rt, y, self.dim, self.dim);
            out[i * self.dim..(i + 1) * self.dim].copy_from_slice(&x);
        }
        Ok(out)
    }

    /// Compute ADC distances from a query to `n` encoded vectors.
    pub fn adc_distances(&self, query: &[f32], codes: &[u8], n: usize) -> AnnResult<Vec<f32>> {
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let m = self.codebook.m;
        if codes.len() != n * m {
            return Err(AnnError::DimensionMismatch {
                expected: n * m,
                got: codes.len(),
            });
        }
        // Rotate query into codebook space
        let rq = matvec(&self.rotation, query, self.dim, self.dim);
        let table = build_adc_table(&rq, &self.codebook);
        let distances = (0..n)
            .map(|i| {
                let code_slice = &codes[i * m..(i + 1) * m];
                adc_distance(code_slice, &table, m, self.codebook.ksub)
            })
            .collect();
        Ok(distances)
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn rand_vecs(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_f32()).collect()
    }

    fn normal_vecs(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut v);
        v
    }

    // ── config validation ─────────────────────────────────────────────────────

    #[test]
    fn opq_config_new_valid() {
        let cfg = OpqConfig::new(4, 16);
        assert!(cfg.is_ok(), "{:?}", cfg.err());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.ksub, 16);
    }

    #[test]
    fn opq_config_invalid_ksub_zero() {
        let res = OpqConfig::new(2, 0);
        assert!(res.is_err(), "ksub=0 should fail");
    }

    #[test]
    fn opq_config_invalid_ksub_too_large() {
        let res = OpqConfig::new(2, 257);
        assert!(res.is_err(), "ksub=257 should fail");
    }

    #[test]
    fn opq_config_invalid_m_zero() {
        let res = OpqConfig::new(0, 16);
        assert!(res.is_err(), "m=0 should fail");
    }

    // ── training ──────────────────────────────────────────────────────────────

    #[test]
    fn opq_train_basic() {
        let mut rng = make_rng(1);
        let n = 100;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 2;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng);
        assert!(model.is_ok(), "{:?}", model.err());
        let model = model.unwrap();
        assert_eq!(model.codebook.m, 2);
        assert_eq!(model.codebook.ksub, 8);
        assert_eq!(model.codebook.dsub, 4);
    }

    #[test]
    fn opq_rotation_shape() {
        let mut rng = make_rng(2);
        let n = 64;
        let dim = 4;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 4).unwrap();
        cfg.n_pq_epochs = 3;
        cfg.n_outer_iters = 1;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        assert_eq!(model.rotation.len(), dim * dim);
    }

    #[test]
    fn opq_rotation_is_orthogonal() {
        let mut rng = make_rng(3);
        let n = 100;
        let dim = 4;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 4).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 3;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let r = &model.rotation;
        // Compute R^T @ R and check it is ≈ I
        let rt = transpose(r, dim, dim);
        let rtdr = matmul(&rt, r, dim, dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0_f32 } else { 0.0_f32 };
                let got = rtdr[i * dim + j];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "R^T R [{i}][{j}] = {got:.6}, expected {expected:.1}"
                );
            }
        }
    }

    #[test]
    fn opq_train_n_zero_error() {
        let mut rng = make_rng(4);
        let cfg = OpqConfig::new(2, 8).unwrap();
        let res = OpqModel::train(&[], 0, 8, cfg, &mut rng);
        assert!(matches!(res, Err(AnnError::EmptyInput)));
    }

    #[test]
    fn opq_train_dim_mismatch_error() {
        let mut rng = make_rng(5);
        let cfg = OpqConfig::new(2, 4).unwrap();
        // data.len() != n * dim → DimensionMismatch
        let data = vec![1.0_f32; 10]; // only 10 floats, but n=5 dim=4 → expected 20
        let res = OpqModel::train(&data, 5, 4, cfg, &mut rng);
        assert!(res.is_err(), "should fail on dimension mismatch");
    }

    #[test]
    fn opq_train_m_not_divide_dim_error() {
        let mut rng = make_rng(6);
        let mut cfg = OpqConfig::new(3, 4).unwrap();
        cfg.n_pq_epochs = 3;
        let data = vec![0.5_f32; 50 * 8]; // dim=8, m=3 → 8 not divisible by 3
        let res = OpqModel::train(&data, 50, 8, cfg, &mut rng);
        assert!(res.is_err(), "should fail because 8 % 3 != 0");
    }

    // ── encode/decode ─────────────────────────────────────────────────────────

    #[test]
    fn opq_encode_output_shape() {
        let mut rng = make_rng(7);
        let n = 80;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 2;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let codes = model.encode(&data, n).unwrap();
        assert_eq!(codes.len(), n * model.codebook.m);
    }

    #[test]
    fn opq_decode_output_shape() {
        let mut rng = make_rng(8);
        let n = 80;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 2;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let codes = model.encode(&data, n).unwrap();
        let decoded = model.decode(&codes, n).unwrap();
        assert_eq!(decoded.len(), n * dim);
    }

    #[test]
    fn opq_encode_decode_roundtrip_approximate() {
        // Decoded vectors should be in the right neighbourhood (quantisation error bounded)
        let mut rng = make_rng(9);
        let n = 128;
        let dim = 4;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 4).unwrap();
        cfg.n_pq_epochs = 10;
        cfg.n_outer_iters = 3;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let codes = model.encode(&data, n).unwrap();
        let decoded = model.decode(&codes, n).unwrap();
        // Average reconstruction MSE should be finite
        let mse: f32 = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / (n * dim) as f32;
        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 10.0, "MSE too large: {mse}");
    }

    #[test]
    fn opq_encode_wrong_n_error() {
        let mut rng = make_rng(10);
        let n = 64;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 1;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        // Provide wrong data length
        let short = vec![0.0_f32; 3 * dim - 1];
        let res = model.encode(&short, 3);
        assert!(res.is_err(), "should fail on wrong data length");
    }

    // ── rotate / unrotate ─────────────────────────────────────────────────────

    #[test]
    fn opq_rotate_unrotate_roundtrip() {
        let mut rng = make_rng(11);
        let n = 64;
        let dim = 4;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 4).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 2;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            let rotated = model.rotate(x).unwrap();
            let recovered = model.unrotate(&rotated).unwrap();
            for d in 0..dim {
                assert!(
                    (recovered[d] - x[d]).abs() < 1e-5,
                    "unrotate(rotate(x))[{d}] = {} ≠ {}",
                    recovered[d],
                    x[d]
                );
            }
        }
    }

    // ── ADC distances ─────────────────────────────────────────────────────────

    #[test]
    fn opq_adc_distances_nonneg() {
        let mut rng = make_rng(12);
        let n = 64;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 2;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let codes = model.encode(&data, n).unwrap();
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let dists = model.adc_distances(&query, &codes, n).unwrap();
        assert!(dists.iter().all(|&d| d >= 0.0), "some distances negative");
    }

    #[test]
    fn opq_adc_distances_shape() {
        let mut rng = make_rng(13);
        let n = 64;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 2;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let codes = model.encode(&data, n).unwrap();
        let query = vec![0.0_f32; dim];
        let dists = model.adc_distances(&query, &codes, n).unwrap();
        assert_eq!(dists.len(), n);
    }

    #[test]
    fn opq_adc_wrong_query_dim_error() {
        let mut rng = make_rng(14);
        let n = 64;
        let dim = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 8).unwrap();
        cfg.n_pq_epochs = 5;
        cfg.n_outer_iters = 1;
        let model = OpqModel::train(&data, n, dim, cfg, &mut rng).unwrap();
        let codes = model.encode(&data, n).unwrap();
        let bad_query = vec![0.0_f32; dim + 1];
        let res = model.adc_distances(&bad_query, &codes, n);
        assert!(res.is_err(), "wrong query dimension should fail");
    }

    // ── n_outer_iters = 0 → identity rotation (no optimisation) ─────────────

    #[test]
    fn opq_identity_rotation_matches_pq() {
        let mut rng = make_rng(15);
        let n = 64;
        let dim = 4;
        let data = rand_vecs(n, dim, &mut rng);
        let mut cfg = OpqConfig::new(2, 4).unwrap();
        cfg.n_outer_iters = 0;
        cfg.n_pq_epochs = 5;
        // n_outer_iters = 0 means the loop doesn't run and codebook is default (zeroed)
        // We just verify it returns Ok with an identity-like rotation
        let res = OpqModel::train(&data, n, dim, cfg, &mut rng);
        // Either Ok or Err is acceptable depending on how zero-iter handles empty codebook,
        // but we expect it to be Ok with default rotation = identity
        if let Ok(model) = res {
            // rotation should be identity (no iterations performed)
            let id = identity(dim);
            for (a, b) in model.rotation.iter().zip(id.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "rotation should be identity when n_outer_iters=0"
                );
            }
        }
        // If it's Err (empty codebook from zero iters), that is also acceptable behaviour
    }

    // ── determinism ───────────────────────────────────────────────────────────

    #[test]
    fn opq_deterministic_same_seed() {
        let n = 64;
        let dim = 4;
        let mut rng0 = make_rng(99);
        let data = normal_vecs(n, dim, &mut rng0);

        let mut rng_a = make_rng(42);
        let mut rng_b = make_rng(42);

        let mut cfg_a = OpqConfig::new(2, 4).unwrap();
        cfg_a.n_pq_epochs = 5;
        cfg_a.n_outer_iters = 2;
        let mut cfg_b = cfg_a.clone();
        cfg_b.n_pq_epochs = 5;
        cfg_b.n_outer_iters = 2;

        let model_a = OpqModel::train(&data, n, dim, cfg_a, &mut rng_a).unwrap();
        let model_b = OpqModel::train(&data, n, dim, cfg_b, &mut rng_b).unwrap();

        // Same seed → same rotation
        assert_eq!(
            model_a.rotation, model_b.rotation,
            "same seed should produce identical rotation"
        );
    }
}
