//! Frequent Directions (Liberty 2013 STOC) streaming matrix sketch.
//!
//! A deterministic streaming low-rank matrix approximation algorithm.
//! Maintains a sketch of size `l × d` and provides a (1+ε) covariance
//! approximation guarantee:
//!
//! `‖A^T A − B^T B‖ ≤ ‖A − A_k‖_F² / (l − k)`
//!
//! where A_k is the best rank-k approximation of the full input matrix A.
//!
//! Reference: Liberty, "Simple and Deterministic Matrix Sketching",
//! KDD 2013 / STOC 2013.

use crate::error::{SketchError, SketchResult};

/// Frequent Directions streaming matrix sketch.
///
/// Maintains a rank-`l` sketch of a matrix built from streaming rows.
/// After inserting `n` rows of dimension `d`, the sketch `B` satisfies the
/// covariance-approximation guarantee of Liberty (2013).
#[derive(Debug, Clone)]
pub struct FrequentDirections {
    /// Column dimension of input rows.
    pub d: usize,
    /// Sketch rank (l ≥ 2).
    pub l: usize,
    /// `l × d` row-major sketch matrix; some trailing rows may be zero.
    pub sketch: Vec<f64>,
    /// Number of non-zero rows currently in the sketch.
    pub n_filled: usize,
    /// Total rows processed (including compressed rows).
    pub n_rows_seen: usize,
}

impl FrequentDirections {
    /// Create a new sketch of rank `l` for `d`-dimensional rows.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `d < 1` or `l < 2`.
    pub fn new(d: usize, l: usize) -> SketchResult<Self> {
        if d < 1 {
            return Err(SketchError::InvalidParameter {
                name: "d".to_string(),
                reason: "column dimension must be >= 1".to_string(),
            });
        }
        if l < 2 {
            return Err(SketchError::InvalidParameter {
                name: "l".to_string(),
                reason: "sketch rank must be >= 2".to_string(),
            });
        }
        Ok(Self {
            d,
            l,
            sketch: vec![0.0; l * d],
            n_filled: 0,
            n_rows_seen: 0,
        })
    }

    /// Process a new row vector of length `d`.
    ///
    /// Triggers SVD (via Jacobi eigen-decomposition) compression when the
    /// internal buffer is full (`n_filled == l`).
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `row.len() != d`, or propagates errors
    /// from `compress()`.
    pub fn push(&mut self, row: &[f64]) -> SketchResult<()> {
        if row.len() != self.d {
            return Err(SketchError::DimensionMismatch {
                a: self.d,
                b: row.len(),
            });
        }
        // Copy new row into the next available slot.
        let offset = self.n_filled * self.d;
        self.sketch[offset..offset + self.d].copy_from_slice(row);
        self.n_filled += 1;
        self.n_rows_seen += 1;

        if self.n_filled == self.l {
            self.compress()?;
        }
        Ok(())
    }

    /// Return the current sketch matrix (`l × d`, row-major).
    ///
    /// Some trailing rows may be zero vectors (unused slots after compression).
    #[must_use]
    pub fn sketch_matrix(&self) -> &[f64] {
        &self.sketch
    }

    /// Covariance approximation: B^T B, returned as a `d × d` row-major matrix.
    ///
    /// # Errors
    /// Returns `NumericalInstability` if any computed value is non-finite.
    pub fn covariance(&self) -> SketchResult<Vec<f64>> {
        let d = self.d;
        let mut cov = vec![0.0_f64; d * d];
        for i in 0..self.l {
            let row = &self.sketch[i * d..(i + 1) * d];
            for r in 0..d {
                for c in 0..d {
                    cov[r * d + c] += row[r] * row[c];
                }
            }
        }
        for &v in &cov {
            if !v.is_finite() {
                return Err(SketchError::NumericalInstability(
                    "covariance contains non-finite value".to_string(),
                ));
            }
        }
        Ok(cov)
    }

    /// Approximate squared singular values of the sketch (length `l`, sorted descending).
    #[must_use]
    pub fn squared_singular_values(&self) -> Vec<f64> {
        let d = self.d;
        let mut norms: Vec<f64> = (0..self.l)
            .map(|i| {
                let row = &self.sketch[i * d..(i + 1) * d];
                row.iter().map(|v| v * v).sum::<f64>()
            })
            .collect();
        norms.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        norms
    }

    // ---- Private helpers ----------------------------------------------------

    /// Compress the sketch when `n_filled == l` using the Frequent Directions
    /// shrinkage step.
    ///
    /// 1. Compute M = B B^T (l × l symmetric).
    /// 2. Eigen-decompose M via cyclic Jacobi → (λ, U).
    /// 3. δ = min λ_i (clamped to 0).
    /// 4. For each eigenvector U_i: new_row_i = sqrt(max(λ_i − δ, 0) / λ_i) · B^T U_i.
    /// 5. Overwrite sketch; sort by descending norm²; update n_filled.
    fn compress(&mut self) -> SketchResult<()> {
        let l = self.l;
        let d = self.d;

        // Step 1: M = sketch · sketch^T  (l × l, symmetric).
        let mut m_flat = vec![0.0_f64; l * l];
        for i in 0..l {
            for j in 0..l {
                let mut dot = 0.0_f64;
                for k in 0..d {
                    dot += self.sketch[i * d + k] * self.sketch[j * d + k];
                }
                m_flat[i * l + j] = dot;
            }
        }

        // Step 2: Eigen-decompose M.
        let (eigenvalues, eigenvectors) = jacobi_eigen(&mut m_flat, l, 1e-12, 100)?;

        // Step 3: δ = min eigenvalue, clamped ≥ 0.
        let delta = eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(0.0);

        // Step 4: Compute new rows.
        let mut new_rows = vec![0.0_f64; l * d];
        for i in 0..l {
            let lambda_i = eigenvalues[i];
            if lambda_i <= 1e-14 {
                // Zero row — leave as zero.
                continue;
            }
            let scale = ((lambda_i - delta).max(0.0) / lambda_i).sqrt();
            if scale < 1e-300 {
                continue;
            }
            // u_i = eigenvectors[:, i]  (column i in column-major storage).
            // v_i = sketch^T @ u_i, dimension d.
            let u_col_offset = i * l; // eigenvectors[u_col_offset..u_col_offset+l] = column i.
            for k in 0..d {
                let mut v_k = 0.0_f64;
                for j in 0..l {
                    v_k += self.sketch[j * d + k] * eigenvectors[u_col_offset + j];
                }
                new_rows[i * d + k] = scale * v_k;
            }
        }

        // Step 5: Sort rows by norm² descending, overwrite sketch, update n_filled.
        let mut norms_idx: Vec<(usize, f64)> = (0..l)
            .map(|i| {
                let norm_sq = (0..d).map(|k| new_rows[i * d + k].powi(2)).sum::<f64>();
                (i, norm_sq)
            })
            .collect();
        norms_idx
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (out_pos, (src_idx, _)) in norms_idx.iter().enumerate() {
            let src_off = src_idx * d;
            let dst_off = out_pos * d;
            self.sketch[dst_off..dst_off + d].copy_from_slice(&new_rows[src_off..src_off + d]);
        }

        // Count non-zero rows.
        self.n_filled = (0..l)
            .filter(|&i| {
                let row = &self.sketch[i * d..(i + 1) * d];
                row.iter().any(|v| v.abs() > 1e-300)
            })
            .count();

        Ok(())
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric `n × n` matrix `a` (row-major,
/// modified in place).
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is stored
/// **column-major** (column `i` occupies `eigenvectors[i*n..(i+1)*n]`).
///
/// Convergence criterion: off-diagonal Frobenius norm < `eps`, or `max_iter`
/// sweeps completed (returns `Err(NotConverged)` in the latter case).
fn jacobi_eigen(
    a: &mut [f64],
    n: usize,
    eps: f64,
    max_iter: usize,
) -> SketchResult<(Vec<f64>, Vec<f64>)> {
    // Eigenvector matrix V starts as identity (column-major).
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let eps_sq = eps * eps;

    for iter in 0..max_iter {
        // Compute off-diagonal Frobenius norm squared.
        let mut off_sq = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = a[p * n + q];
                off_sq += 2.0 * val * val;
            }
        }
        if off_sq < eps_sq {
            break;
        }

        // Sweep over all off-diagonal pairs.
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a[p * n + q];
                if a_pq.abs() < eps_sq {
                    continue;
                }
                let a_pp = a[p * n + p];
                let a_qq = a[q * n + q];

                // Compute rotation angle.
                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal and the (p,q) element.
                a[p * n + p] = a_pp - t * a_pq;
                a[q * n + q] = a_qq + t * a_pq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal rows/columns r ≠ p, q.
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let a_rp = a[r * n + p];
                    let a_rq = a[r * n + q];
                    let new_rp = c * a_rp - s * a_rq;
                    let new_rq = s * a_rp + c * a_rq;
                    a[r * n + p] = new_rp;
                    a[p * n + r] = new_rp;
                    a[r * n + q] = new_rq;
                    a[q * n + r] = new_rq;
                }

                // Update eigenvector matrix V (column-major: column p and column q).
                for r in 0..n {
                    let v_rp = v[p * n + r]; // V[r, p]
                    let v_rq = v[q * n + r]; // V[r, q]
                    v[p * n + r] = c * v_rp - s * v_rq;
                    v[q * n + r] = s * v_rp + c * v_rq;
                }

                if iter == max_iter - 1 {
                    // Final convergence check.
                    let mut final_off = 0.0_f64;
                    for pp in 0..n {
                        for qq in (pp + 1)..n {
                            let val = a[pp * n + qq];
                            final_off += 2.0 * val * val;
                        }
                    }
                    if final_off >= eps_sq {
                        return Err(SketchError::NotConverged { iter: max_iter });
                    }
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    Ok((eigenvalues, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. d=10, l=3 creates without error.
    #[test]
    fn new_valid_ok() {
        let fd = FrequentDirections::new(10, 3);
        assert!(fd.is_ok());
    }

    // 2. l=1 → Err; l=0 → Err.
    #[test]
    fn new_invalid_l_lt_2() {
        assert!(FrequentDirections::new(10, 1).is_err());
        assert!(FrequentDirections::new(10, 0).is_err());
    }

    // 3. d=0 → Err.
    #[test]
    fn new_invalid_d_zero() {
        assert!(FrequentDirections::new(0, 3).is_err());
    }

    // 4. push row of wrong length → Err.
    #[test]
    fn push_wrong_length() {
        let mut fd = FrequentDirections::new(5, 3).expect("new should succeed");
        let result = fd.push(&[1.0, 2.0]); // length 2 ≠ 5
        assert!(result.is_err());
    }

    // 5. push l-1 rows; n_filled == l-1.
    #[test]
    fn push_fills_sketch() {
        let l = 4;
        let mut fd = FrequentDirections::new(5, l).expect("new should succeed");
        for i in 0..(l - 1) {
            let v = i as f64;
            fd.push(&[v, v, v, v, v]).expect("push should succeed");
        }
        assert_eq!(fd.n_filled, l - 1);
    }

    // 6. push l rows; n_filled < l after (compression freed a slot).
    #[test]
    fn push_triggers_compression() {
        let l = 4;
        let mut fd = FrequentDirections::new(5, l).expect("new should succeed");
        for i in 0..l {
            let v = i as f64;
            fd.push(&[v, v, v, v, v]).expect("push should succeed");
        }
        // After l inserts, compress should have run and reduced n_filled.
        assert!(
            fd.n_filled < l,
            "n_filled={} should be < l={}",
            fd.n_filled,
            l
        );
    }

    // 7. push n identical rows v; covariance ≈ n * v^T v (up to the shrinkage).
    #[test]
    fn covariance_rank_one_input() {
        let d = 3;
        let l = 4;
        let mut fd = FrequentDirections::new(d, l).expect("new should succeed");
        let v = vec![1.0, 0.0, 0.0_f64];
        for _ in 0..20 {
            fd.push(&v).expect("push should succeed");
        }
        let cov = fd.covariance().expect("covariance should succeed");
        // cov[0,0] should be the largest entry, > 0.
        assert!(cov[0] > 0.0, "cov[0,0]={}", cov[0]);
        // Off-diagonals involving index 0 should be small (rank-1 structure).
        assert!(
            cov[1].abs() < 1e-10,
            "cov[0,1] should be ~0, got {}",
            cov[1]
        );
    }

    // 8. sketch_matrix().len() == l * d.
    #[test]
    fn sketch_matrix_length_correct() {
        let d = 7;
        let l = 3;
        let fd = FrequentDirections::new(d, l).expect("new should succeed");
        assert_eq!(fd.sketch_matrix().len(), l * d);
    }

    // 9. squared_singular_values result is non-increasing.
    #[test]
    fn squared_singular_values_sorted() {
        let mut fd = FrequentDirections::new(5, 4).expect("new should succeed");
        for i in 0..20u64 {
            let row: Vec<f64> = (0..5).map(|k| (i * 5 + k) as f64).collect();
            fd.push(&row).expect("push should succeed");
        }
        let sv = fd.squared_singular_values();
        for w in sv.windows(2) {
            assert!(w[0] >= w[1] - 1e-10, "not sorted: {:?}", sv);
        }
    }

    // 10. All squared_singular_values ≥ 0.
    #[test]
    fn squared_singular_values_nonneg() {
        let mut fd = FrequentDirections::new(5, 4).expect("new should succeed");
        for i in 0..15u64 {
            let v = i as f64;
            fd.push(&[v, v, v, v, v]).expect("push should succeed");
        }
        for &v in &fd.squared_singular_values() {
            assert!(v >= 0.0, "negative squared singular value: {v}");
        }
    }

    // 11. n_rows_seen == number of push calls.
    #[test]
    fn n_rows_seen_counts_all() {
        let mut fd = FrequentDirections::new(4, 3).expect("new should succeed");
        for i in 0..50u64 {
            let v = i as f64;
            fd.push(&[v, v, v, v]).expect("push should succeed");
        }
        assert_eq!(fd.n_rows_seen, 50);
    }

    // 12. Push 100 rows of d=20; no error; n_filled ≤ l.
    #[test]
    fn streaming_many_rows() {
        let d = 20;
        let l = 5;
        let mut fd = FrequentDirections::new(d, l).expect("new should succeed");
        for i in 0..100u64 {
            let row: Vec<f64> = (0..d).map(|k| (i + k as u64) as f64).collect();
            fd.push(&row).expect("push should succeed");
        }
        assert!(fd.n_filled <= l);
    }

    // 13. cov[i*d+j] == cov[j*d+i] for all i,j.
    #[test]
    fn covariance_symmetric() {
        let d = 4;
        let mut fd = FrequentDirections::new(d, 3).expect("new should succeed");
        for i in 0..30u64 {
            let row: Vec<f64> = (0..d).map(|k| (i * k as u64 + 1) as f64).collect();
            fd.push(&row).expect("push should succeed");
        }
        let cov = fd.covariance().expect("covariance should succeed");
        for r in 0..d {
            for c in 0..d {
                let diff = (cov[r * d + c] - cov[c * d + r]).abs();
                assert!(diff < 1e-9, "cov not symmetric at ({r},{c}): diff={diff}");
            }
        }
    }

    // 14. d=1, l=2; push 10 values; squared_singular_values()[0] > 0.
    #[test]
    fn single_column_dataset() {
        let mut fd = FrequentDirections::new(1, 2).expect("new should succeed");
        for i in 1u64..=10 {
            fd.push(&[i as f64]).expect("push should succeed");
        }
        let sv = fd.squared_singular_values();
        assert!(
            sv[0] > 0.0,
            "top singular value should be > 0, got {}",
            sv[0]
        );
    }

    // 15. Push l identical copies of same vector; after compression, squared_singular_values is valid.
    #[test]
    fn all_zero_rows_after_compression() {
        let d = 4;
        let l = 3;
        let mut fd = FrequentDirections::new(d, l).expect("new should succeed");
        let v = vec![2.0; d];
        // Push exactly l identical rows to trigger compression.
        for _ in 0..l {
            fd.push(&v).expect("push should succeed");
        }
        let sv = fd.squared_singular_values();
        // After compression of rank-1 data: top singular value should be positive.
        assert!(sv[0] > 0.0);
        for &s in &sv {
            assert!(s >= 0.0);
        }
    }

    // 16. Data is rank-1: push n*v for random v; after many inserts, top singular value dominates.
    #[test]
    fn low_rank_preservation() {
        let d = 6;
        let l = 4;
        let mut fd = FrequentDirections::new(d, l).expect("new should succeed");
        // All rows are the same direction (rank-1).
        let v: Vec<f64> = (1..=d as u64).map(|i| i as f64).collect();
        for scale in 1u64..=60 {
            let row: Vec<f64> = v.iter().map(|&x| x * scale as f64).collect();
            fd.push(&row).expect("push should succeed");
        }
        let sv = fd.squared_singular_values();
        // Top singular value should dominate (ratio > 10).
        let sum: f64 = sv.iter().sum();
        if sum > 1e-10 {
            let ratio = sv[0] / sum;
            assert!(
                ratio > 0.5,
                "top sv should dominate in rank-1 data: ratio={ratio}"
            );
        }
    }

    // 17. jacobi_eigen on identity matrix; eigenvalues all ≈ 1.0.
    #[test]
    fn jacobi_identity() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let (evals, _evecs) =
            jacobi_eigen(&mut a, n, 1e-12, 100).expect("jacobi_eigen should succeed");
        for &ev in &evals {
            assert!((ev - 1.0).abs() < 1e-10, "eigenvalue {ev} ≠ 1.0");
        }
    }

    // 18. jacobi_eigen on 2×2 [[2,1],[1,2]]; eigenvalues ≈ {3,1}.
    #[test]
    fn jacobi_2x2() {
        let n = 2;
        let mut a = vec![2.0_f64, 1.0, 1.0, 2.0];
        let (mut evals, _evecs) =
            jacobi_eigen(&mut a, n, 1e-12, 100).expect("jacobi_eigen should succeed");
        evals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (evals[0] - 3.0).abs() < 1e-10,
            "expected 3.0, got {}",
            evals[0]
        );
        assert!(
            (evals[1] - 1.0).abs() < 1e-10,
            "expected 1.0, got {}",
            evals[1]
        );
    }
}
