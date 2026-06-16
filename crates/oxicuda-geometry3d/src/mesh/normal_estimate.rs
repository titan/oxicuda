//! Per-point normal estimation via PCA on kNN neighborhoods.

use crate::error::{Geom3dError, Geom3dResult};

/// Compute the 3×3 covariance matrix from a neighborhood (row-major).
fn covariance3x3(pts: &[&[f32]]) -> [f32; 9] {
    let n = pts.len();
    if n == 0 {
        return [0.0; 9];
    }

    // Mean
    let mut mean = [0.0_f32; 3];
    for p in pts.iter() {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    mean[0] /= n as f32;
    mean[1] /= n as f32;
    mean[2] /= n as f32;

    // Covariance
    let mut cov = [0.0_f32; 9];
    for p in pts.iter() {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += d[i] * d[j];
            }
        }
    }
    for v in &mut cov {
        *v /= n as f32;
    }
    cov
}

/// Compute eigenvalues of 3×3 symmetric matrix analytically (Cardano's formula).
fn eigenvalues_sym3(m: &[f32; 9]) -> [f32; 3] {
    // Following the method from: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3×3_matrices
    let a00 = m[0];
    let a01 = m[1];
    let a02 = m[2];
    let a11 = m[4];
    let a12 = m[5];
    let a22 = m[8];

    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if p1.abs() < 1e-10 {
        // Diagonal matrix
        let mut eigs = [a00, a11, a22];
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        return eigs;
    }

    let q = (a00 + a11 + a22) / 3.0;
    let p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    let b00 = (a00 - q) / p;
    let b11 = (a11 - q) / p;
    let b22 = (a22 - q) / p;
    let b01 = a01 / p;
    let b02 = a02 / p;
    let b12 = a12 / p;

    // det(B)/2
    let r = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02))
        / 2.0;

    let phi = if r <= -1.0 {
        std::f32::consts::PI / 3.0
    } else if r >= 1.0 {
        0.0_f32
    } else {
        r.acos() / 3.0
    };

    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f32::consts::PI / 3.0).cos();
    let eig2 = 3.0 * q - eig1 - eig3;

    let mut eigs = [eig1, eig2, eig3];
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

/// Find the eigenvector corresponding to the smallest eigenvalue via
/// the cross-product method on the two larger-eigenvalue eigenvectors,
/// or via power iteration on `(max_eig * I - A)` when eigenvalues are distinct.
fn eigenvector_smallest(cov: &[f32; 9], eigenvalues: &[f32; 3]) -> [f32; 3] {
    let lambda_max = eigenvalues[2];
    let lambda_min = eigenvalues[0];
    let lambda_mid = eigenvalues[1];

    // If all eigenvalues are nearly equal, return z-axis
    if (lambda_max - lambda_min).abs() < 1e-8 {
        return [0.0, 0.0, 1.0];
    }

    // Use multiple starting vectors for robustness via power iteration
    // on B = (λ_max * I - A), which maps the smallest-eigval direction
    // to the largest eigenvalue (λ_max - λ_min)
    let b = [
        lambda_max - cov[0],
        -cov[1],
        -cov[2],
        -cov[3],
        lambda_max - cov[4],
        -cov[5],
        -cov[6],
        -cov[7],
        lambda_max - cov[8],
    ];

    // Special case: if lambda_min ≈ 0 (flat surface), the smallest eigenvector
    // is in the null space of A. We can find it by solving (A - 0*I)*v = 0,
    // i.e., A*v ≈ 0. Use the two largest eigenvalue eigenvectors' cross product.
    // But for robustness, try all 3 canonical starting vectors.
    let starts = [
        [1.0_f32, 0.0, 0.0],
        [0.0_f32, 1.0, 0.0],
        [0.0_f32, 0.0, 1.0],
    ];

    let mut best_v = [0.0_f32, 0.0, 1.0];
    let mut best_rayleigh = f32::INFINITY;

    for start in &starts {
        let mut v = *start;
        for _ in 0..100 {
            let vn = [
                b[0] * v[0] + b[1] * v[1] + b[2] * v[2],
                b[3] * v[0] + b[4] * v[1] + b[5] * v[2],
                b[6] * v[0] + b[7] * v[1] + b[8] * v[2],
            ];
            let norm = (vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]).sqrt();
            if norm < 1e-14 {
                break;
            }
            v = [vn[0] / norm, vn[1] / norm, vn[2] / norm];
        }

        // Rayleigh quotient: v^T A v — smallest eigenvec minimizes this
        let av = [
            cov[0] * v[0] + cov[1] * v[1] + cov[2] * v[2],
            cov[3] * v[0] + cov[4] * v[1] + cov[5] * v[2],
            cov[6] * v[0] + cov[7] * v[1] + cov[8] * v[2],
        ];
        let rq = v[0] * av[0] + v[1] * av[1] + v[2] * av[2];

        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if norm > 0.5 && rq < best_rayleigh {
            best_rayleigh = rq;
            best_v = v;
        }
    }

    // Additional check: if lambda_mid and lambda_max are both >> 0 but lambda_min ≈ 0,
    // try using null-space approach: v must satisfy cov*v ≈ lambda_min * v
    // For near-zero lambda_min, iterate on A itself (converges to max, but we want min)
    // So use deflation: iterate on (lambda_max+lambda_mid)*I - A  ≈ subspace of lambda_min
    let shift = lambda_max + lambda_mid + 0.01;
    let b2 = [
        shift - cov[0],
        -cov[1],
        -cov[2],
        -cov[3],
        shift - cov[4],
        -cov[5],
        -cov[6],
        -cov[7],
        shift - cov[8],
    ];

    let mut v2 = [0.0_f32, 0.0, 1.0];
    for _ in 0..100 {
        let vn = [
            b2[0] * v2[0] + b2[1] * v2[1] + b2[2] * v2[2],
            b2[3] * v2[0] + b2[4] * v2[1] + b2[5] * v2[2],
            b2[6] * v2[0] + b2[7] * v2[1] + b2[8] * v2[2],
        ];
        let norm = (vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]).sqrt();
        if norm < 1e-14 {
            break;
        }
        v2 = [vn[0] / norm, vn[1] / norm, vn[2] / norm];
    }

    let av2 = [
        cov[0] * v2[0] + cov[1] * v2[1] + cov[2] * v2[2],
        cov[3] * v2[0] + cov[4] * v2[1] + cov[5] * v2[2],
        cov[6] * v2[0] + cov[7] * v2[1] + cov[8] * v2[2],
    ];
    let rq2 = v2[0] * av2[0] + v2[1] * av2[1] + v2[2] * av2[2];
    let norm2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if norm2 > 0.5 && rq2 < best_rayleigh {
        best_v = v2;
    }

    best_v
}

/// Estimate per-point normals via PCA on kNN neighborhood.
///
/// Returns `normals [n×3]`. Normals are oriented toward +z by default
/// (flip if dot(normal, +z) < 0).
pub fn estimate_normals(points: &[f32], n: usize, k: usize) -> Geom3dResult<Vec<f32>> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if points.len() != n * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * 3,
            got: points.len(),
        });
    }
    if k == 0 {
        return Err(Geom3dError::InvalidK { k: 0, n });
    }

    let actual_k = k.min(n);
    let mut normals = vec![0.0_f32; n * 3];

    for i in 0..n {
        // Find k nearest neighbors
        let mut dists: Vec<(f32, usize)> = (0..n)
            .map(|j| {
                let dx = points[i * 3] - points[j * 3];
                let dy = points[i * 3 + 1] - points[j * 3 + 1];
                let dz = points[i * 3 + 2] - points[j * 3 + 2];
                (dx * dx + dy * dy + dz * dz, j)
            })
            .collect();

        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors: Vec<usize> = dists.iter().take(actual_k).map(|&(_, j)| j).collect();
        let neighbor_pts: Vec<&[f32]> = neighbors
            .iter()
            .map(|&j| &points[j * 3..j * 3 + 3])
            .collect();

        let cov = covariance3x3(&neighbor_pts);
        let eigs = eigenvalues_sym3(&cov);
        let normal = eigenvector_smallest(&cov, &eigs);

        // Orient toward +z
        let dot_z = normal[2]; // dot with [0,0,1]
        let sign = if dot_z < 0.0 { -1.0_f32 } else { 1.0 };

        normals[i * 3] = sign * normal[0];
        normals[i * 3 + 1] = sign * normal[1];
        normals[i * 3 + 2] = sign * normal[2];
    }

    Ok(normals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_estimate_flat_xy_plane() {
        // Points on z=0 plane: normal should be (0,0,1)
        let pts = vec![
            0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0,
        ];
        let normals = estimate_normals(&pts, 5, 4).expect("estimate_normals should succeed");
        assert_eq!(normals.len(), 15);
        // For a flat plane, normal should point in z direction
        for i in 0..5 {
            let nz = normals[i * 3 + 2].abs();
            assert!(
                nz > 0.7,
                "Normal z component should be dominant, got {}",
                normals[i * 3 + 2]
            );
        }
    }

    #[test]
    fn normal_estimate_output_shape() {
        let pts: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let normals = estimate_normals(&pts, 10, 3).expect("estimate_normals should succeed");
        assert_eq!(normals.len(), 10 * 3);
    }

    #[test]
    fn normal_estimate_unit_length() {
        let pts: Vec<f32> = (0..10)
            .flat_map(|i| vec![i as f32, (i as f32).sin(), 0.0])
            .collect();
        let normals = estimate_normals(&pts, 10, 5).expect("estimate_normals should succeed");
        for i in 0..10 {
            let nx = normals[i * 3];
            let ny = normals[i * 3 + 1];
            let nz = normals[i * 3 + 2];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (len - 1.0).abs() < 0.1,
                "Normal should be unit length, got len={len}"
            );
        }
    }

    #[test]
    fn normal_estimate_empty_error() {
        assert_eq!(
            estimate_normals(&[], 0, 3),
            Err(Geom3dError::EmptyPointCloud)
        );
    }
}
