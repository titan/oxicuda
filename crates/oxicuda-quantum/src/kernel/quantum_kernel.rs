use crate::embedding::zz_feature::zz_feature_map;
use crate::error::{QuantumError, QuantumResult};

/// Overlap kernel k(x, y) = |⟨ψ(x)|ψ(y)⟩|².
///
/// Both x and y are encoded via the ZZ feature map with 1 repetition.
pub fn overlap_kernel(x: &[f32], y: &[f32]) -> QuantumResult<f32> {
    if x.len() != y.len() {
        return Err(QuantumError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    if x.is_empty() {
        return Err(QuantumError::EmptyInput);
    }

    let psi_x = zz_feature_map(x, 1)?;
    let psi_y = zz_feature_map(y, 1)?;

    let ip = psi_x.inner_product(&psi_y)?;
    Ok(ip.norm_sqr())
}

/// Compute the full kernel matrix K\[i,j\] = k(xs\[i\], xs\[j\]).
pub fn kernel_matrix(xs: &[Vec<f32>]) -> QuantumResult<Vec<Vec<f32>>> {
    if xs.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    let n = xs.len();
    let mut mat = vec![vec![0.0_f32; n]; n];

    for i in 0..n {
        for j in i..n {
            let k = overlap_kernel(&xs[i], &xs[j])?;
            mat[i][j] = k;
            mat[j][i] = k;
        }
    }

    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_kernel_is_one() {
        let x = vec![0.5_f32, 1.0];
        let k = overlap_kernel(&x, &x)
            .expect("self-overlap of a non-empty, same-length vector pair cannot fail");
        assert!((k - 1.0).abs() < 1e-4, "k={k}");
    }

    #[test]
    fn kernel_matrix_diagonal_is_one() {
        let xs = vec![vec![0.3_f32, 0.7], vec![1.0_f32, -0.5]];
        let mat = kernel_matrix(&xs)
            .expect("kernel matrix of non-empty, equal-length vectors cannot fail");
        assert!((mat[0][0] - 1.0).abs() < 1e-4);
        assert!((mat[1][1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gram_matrix_is_psd() {
        // A fidelity/overlap kernel Gram matrix K(x,y)=|⟨ψ(x)|ψ(y)⟩|² is positive
        // semidefinite (it is the entrywise modulus-squared of a Gram matrix of
        // unit vectors, hence a valid PSD kernel). Verify vᵀKv ≥ 0 for random v.
        use crate::handle::LcgRng;
        let xs = vec![
            vec![0.3_f32, 0.7],
            vec![1.0_f32, -0.5],
            vec![-0.2_f32, 0.9],
            vec![0.6_f32, 0.1],
            vec![-1.1_f32, 0.4],
        ];
        let mat = kernel_matrix(&xs).expect("kernel matrix");
        let m = xs.len();
        // Symmetric.
        for (i, row) in mat.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!((v - mat[j][i]).abs() < 1e-5, "asymmetry at ({i},{j})");
            }
        }
        // PSD via many random test vectors.
        let mut rng = LcgRng::new(2024);
        for _ in 0..50 {
            let v: Vec<f32> = (0..m).map(|_| rng.next_normal()).collect();
            let mut quad = 0.0_f32;
            for (i, row) in mat.iter().enumerate() {
                for (j, &kij) in row.iter().enumerate() {
                    quad += v[i] * kij * v[j];
                }
            }
            assert!(quad >= -1e-3, "quadratic form negative: {quad}");
        }
    }
}
