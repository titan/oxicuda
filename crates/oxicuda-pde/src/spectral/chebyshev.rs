//! Chebyshev collocation: nodes, differentiation matrix, Poisson solver.
//!
//! Reference: Trefethen, "Spectral Methods in MATLAB" (2000), Chapter 6.

use crate::error::{PdeError, PdeResult};

/// Chebyshev-Gauss-Lobatto nodes on `[-1, 1]`: `x_j = cos(j*pi/n)` for `j = 0..=n`.
/// Returns `n+1` nodes (so `n` is the spectral order).
pub fn cheb_nodes(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![0.0];
    }
    (0..=n)
        .map(|j| (std::f64::consts::PI * j as f64 / n as f64).cos())
        .collect()
}

/// Chebyshev differentiation matrix D of size `(n+1) x (n+1)`.
///
/// Formula (Trefethen 2000):
/// `D[i,j] = c_i / c_j * (-1)^(i+j) / (x_i - x_j)`  for `i != j`
/// `D[j,j] = -x_j / (2*(1 - x_j^2))`  for interior `j`
/// `D[0,0] = (2*n^2+1)/6`,  `D[n,n] = -(2*n^2+1)/6`
/// where `c_0 = c_n = 2`, otherwise `c_j = 1`.
pub fn cheb_diff_matrix(n: usize) -> PdeResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(PdeError::InvalidOrder {
            order: n,
            reason: "n>=1 required".into(),
        });
    }
    let x = cheb_nodes(n);
    let n1 = n + 1;
    let mut d = vec![0.0_f64; n1 * n1];
    let c = |i: usize| if i == 0 || i == n { 2.0 } else { 1.0 };
    for i in 0..n1 {
        for j in 0..n1 {
            if i == j {
                continue;
            }
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            d[i * n1 + j] = (c(i) / c(j)) * sign / (x[i] - x[j]);
        }
    }
    // Diagonal: D[i,i] = -sum_{j != i} D[i,j]
    for i in 0..n1 {
        let mut s = 0.0;
        for j in 0..n1 {
            if i != j {
                s += d[i * n1 + j];
            }
        }
        d[i * n1 + i] = -s;
    }
    Ok((x, d))
}

/// Solve `-u''(x) = f(x)` on `[-1, 1]` with `u(-1)=u(1)=0` using Chebyshev collocation.
///
/// Returns `u` on Chebyshev nodes (with `u[0]=u[n]=0`).
pub fn solve_poisson_chebyshev(n: usize, f_at_nodes: &[f64]) -> PdeResult<Vec<f64>> {
    let n1 = n + 1;
    if f_at_nodes.len() != n1 {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n1],
            got: vec![f_at_nodes.len()],
        });
    }
    if n < 2 {
        return Err(PdeError::InvalidOrder {
            order: n,
            reason: "n>=2 required".into(),
        });
    }
    let (_x, d) = cheb_diff_matrix(n)?;
    // D2 = D * D
    let mut d2 = vec![0.0; n1 * n1];
    for i in 0..n1 {
        for j in 0..n1 {
            let mut s = 0.0;
            for k in 0..n1 {
                s += d[i * n1 + k] * d[k * n1 + j];
            }
            d2[i * n1 + j] = s;
        }
    }
    // We solve interior: -D2_int * u_int = f_int  (size n-1 x n-1)
    let m = n - 1;
    let mut a = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..m {
            a[i * m + j] = -d2[(i + 1) * n1 + (j + 1)];
        }
    }
    let mut rhs = vec![0.0; m];
    rhs[..m].copy_from_slice(&f_at_nodes[1..=m]);
    let u_int = gauss_solve_dense(&mut a, &mut rhs, m)?;
    let mut u = vec![0.0; n1];
    u[1..n].copy_from_slice(&u_int);
    Ok(u)
}

/// Gaussian elimination with partial pivoting for a dense `m x m` system.
pub fn gauss_solve_dense(a: &mut [f64], b: &mut [f64], m: usize) -> PdeResult<Vec<f64>> {
    if a.len() != m * m || b.len() != m {
        return Err(PdeError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    for k in 0..m {
        // partial pivot
        let mut p = k;
        let mut pv = a[k * m + k].abs();
        for i in k + 1..m {
            let v = a[i * m + k].abs();
            if v > pv {
                pv = v;
                p = i;
            }
        }
        if pv < 1.0e-300 {
            return Err(PdeError::SingularMatrix(format!(
                "gauss: zero pivot at column {k}"
            )));
        }
        if p != k {
            for j in 0..m {
                a.swap(k * m + j, p * m + j);
            }
            b.swap(k, p);
        }
        // eliminate
        for i in k + 1..m {
            let factor = a[i * m + k] / a[k * m + k];
            a[i * m + k] = 0.0;
            for j in k + 1..m {
                a[i * m + j] -= factor * a[k * m + j];
            }
            b[i] -= factor * b[k];
        }
    }
    // back substitution
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = b[i];
        for j in i + 1..m {
            s -= a[i * m + j] * x[j];
        }
        x[i] = s / a[i * m + i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheb_nodes_count() {
        let x = cheb_nodes(5);
        assert_eq!(x.len(), 6);
        assert!((x[0] - 1.0).abs() < 1.0e-12);
        assert!((x[5] + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn cheb_diff_matrix_rows_sum_zero() {
        let (_, d) = cheb_diff_matrix(8).expect("ok");
        let n1 = 9;
        for i in 0..n1 {
            let s: f64 = (0..n1).map(|j| d[i * n1 + j]).sum();
            assert!(s.abs() < 1.0e-10, "row {i} sum = {s}");
        }
    }

    #[test]
    fn cheb_diff_matrix_differentiates_constant() {
        // D * (1, 1, ..., 1)^T = 0
        let (_, d) = cheb_diff_matrix(10).expect("ok");
        let n1 = 11;
        let v = vec![1.0; n1];
        for i in 0..n1 {
            let mut s = 0.0;
            for j in 0..n1 {
                s += d[i * n1 + j] * v[j];
            }
            assert!(s.abs() < 1.0e-9);
        }
    }

    #[test]
    fn cheb_diff_differentiates_x_squared() {
        // d/dx x^2 = 2x
        let n = 12;
        let n1 = n + 1;
        let (x, d) = cheb_diff_matrix(n).expect("ok");
        let v: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let mut du = vec![0.0; n1];
        for i in 0..n1 {
            for j in 0..n1 {
                du[i] += d[i * n1 + j] * v[j];
            }
        }
        for i in 0..n1 {
            assert!((du[i] - 2.0 * x[i]).abs() < 1.0e-9);
        }
    }

    #[test]
    fn cheb_poisson_exact_polynomial() {
        // -u''(x) = 2 on [-1,1], u(-1)=u(1)=0 => u(x) = 1 - x^2
        let n = 16;
        let x = cheb_nodes(n);
        let f: Vec<f64> = x.iter().map(|_| 2.0).collect();
        let u = solve_poisson_chebyshev(n, &f).expect("ok");
        for i in 0..=n {
            let expected = 1.0 - x[i] * x[i];
            assert!((u[i] - expected).abs() < 1.0e-9);
        }
    }

    #[test]
    fn gauss_solve_2x2() {
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        let x = gauss_solve_dense(&mut a, &mut b, 2).expect("ok");
        // 2x+y=5, x+3y=10 => x=1, y=3
        assert!((x[0] - 1.0).abs() < 1.0e-12);
        assert!((x[1] - 3.0).abs() < 1.0e-12);
    }
}
