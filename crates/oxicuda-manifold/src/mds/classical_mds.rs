//! Classical (Torgerson) MDS.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// Result of a classical-MDS fit.
pub struct ClassicalMdsResult {
    /// Embedded coordinates (`n x n_components`).
    pub embedding: Vec<f64>,
    /// Top-`n_components` eigenvalues.
    pub eigenvalues: Vec<f64>,
}

/// Classical MDS on an `n x n` distance matrix.
///
/// 1. Build `D^2` (squared distances).
/// 2. Double-centre: `B = -1/2 * J D^2 J` with `J = I - (1/n) 1 1^T`.
/// 3. Eigendecompose `B = V Lambda V^T`.
/// 4. Embedding `Y = V_+ sqrt(Lambda_+)` where positive eigenvalues only.
pub fn classical_mds(
    distances: &[f64],
    n: usize,
    n_components: usize,
) -> ManifoldResult<ClassicalMdsResult> {
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if distances.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![distances.len()],
        });
    }
    if n_components == 0 || n_components >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n}, got {n_components}"),
        });
    }
    // D^2
    let d2: Vec<f64> = distances.iter().map(|d| d * d).collect();
    // Means
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            row_mean[i] += d2[i * n + j];
            col_mean[j] += d2[i * n + j];
            total += d2[i * n + j];
        }
    }
    for v in &mut row_mean {
        *v /= n as f64;
    }
    for v in &mut col_mean {
        *v /= n as f64;
    }
    total /= (n * n) as f64;
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d2[i * n + j] - row_mean[i] - col_mean[j] + total);
        }
    }
    let (mut w, mut v) = jacobi_eigh(&b, n)?;
    sort_eigen_descending(&mut w, &mut v, n);
    let mut embedding = vec![0.0; n * n_components];
    let mut eigenvalues = vec![0.0; n_components];
    for c in 0..n_components {
        let lam = w[c].max(0.0);
        eigenvalues[c] = lam;
        let s = lam.sqrt();
        for r in 0..n {
            embedding[r * n_components + c] = v[r * n + c] * s;
        }
    }
    Ok(ClassicalMdsResult {
        embedding,
        eigenvalues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_line() {
        // 3 collinear points at x = 0, 1, 3
        let pts: [f64; 3] = [0.0, 1.0, 3.0];
        let mut d = vec![0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                d[i * 3 + j] = (pts[i] - pts[j]).abs();
            }
        }
        let r = classical_mds(&d, 3, 1).expect("ok");
        // After embedding, pairwise distances must match.
        for i in 0..3 {
            for j in 0..3 {
                let de = (r.embedding[i] - r.embedding[j]).abs();
                assert!((de - d[i * 3 + j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn negative_components_clipped() {
        // Provide a distance matrix that yields some negative eigenvalues.
        let n = 4;
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    d[i * n + j] = 1.0;
                }
            }
        }
        let r = classical_mds(&d, n, 2).expect("ok");
        for v in r.eigenvalues.iter() {
            assert!(*v >= 0.0);
        }
    }
}
