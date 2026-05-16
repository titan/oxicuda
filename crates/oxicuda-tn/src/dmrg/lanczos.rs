//! Lanczos algorithm for the smallest eigenvalue/eigenvector of a symmetric linear
//! operator, with full reorthogonalisation and tridiagonal QL eigensolver.
//!
//! The operator is provided as a closure `apply: Fn(&[f64]) -> Vec<f64>`. This lets the
//! same routine be used for explicit `n × n` matrices or for the DMRG effective
//! Hamiltonian (matrix-free).

use crate::{TnError, TnResult};

/// Output of the Lanczos algorithm.
#[derive(Debug, Clone)]
pub struct LanczosResult {
    /// The smallest computed eigenvalue.
    pub eigenvalue: f64,
    /// The corresponding eigenvector (length `n`).
    pub eigenvector: Vec<f64>,
    /// Actual number of iterations performed.
    pub iter: usize,
}

/// Compute the smallest eigenvalue / eigenvector via Lanczos with full
/// reorthogonalisation.
///
/// `apply(v)` should return `H v` for a symmetric operator `H`.
/// `n` is the dimension. `v0` is the starting vector (need not be normalised).
/// Caller controls maximum iterations and convergence tolerance on eigenvalue.
pub fn lanczos_smallest<F>(
    apply: F,
    n: usize,
    v0: &[f64],
    max_iter: usize,
    tol: f64,
) -> TnResult<LanczosResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    if v0.len() != n {
        return Err(TnError::ShapeMismatch {
            expected: vec![n],
            got: vec![v0.len()],
        });
    }
    let max_iter = max_iter.min(n);
    let max_iter = max_iter.max(1);

    // Normalise the starting vector
    let mut q0 = v0.to_vec();
    let norm0 = norm(&q0);
    if norm0 < 1e-300 {
        return Err(TnError::NumericalInstability(
            "lanczos zero start vector".into(),
        ));
    }
    for x in &mut q0 {
        *x /= norm0;
    }

    let mut q_vecs: Vec<Vec<f64>> = vec![q0];
    let mut alphas: Vec<f64> = Vec::new();
    let mut betas: Vec<f64> = vec![0.0]; // β_0 is unused
    let mut prev_smallest = f64::INFINITY;
    let mut converged_iter = max_iter;

    for j in 0..max_iter {
        let qj = q_vecs[j].clone();
        let mut w = apply(&qj);
        if w.len() != n {
            return Err(TnError::ShapeMismatch {
                expected: vec![n],
                got: vec![w.len()],
            });
        }
        // α_j = <q_j, w>
        let alpha = dot(&qj, &w);
        alphas.push(alpha);
        // w := w - α_j q_j - β_j q_{j-1}
        axpy(&mut w, &qj, -alpha);
        if j > 0 {
            let prev = q_vecs[j - 1].clone();
            let beta_j = betas[j];
            axpy(&mut w, &prev, -beta_j);
        }
        // Full reorthogonalisation (twice)
        for _ in 0..2 {
            for q in &q_vecs {
                let c = dot(q, &w);
                axpy(&mut w, q, -c);
            }
        }
        let beta_new = norm(&w);
        betas.push(beta_new);

        // Eigenvalues of the j+1 × j+1 tridiagonal so far
        let eigs = tridiagonal_eigenvalues(&alphas, &betas[1..]);
        let smallest = *eigs
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&f64::INFINITY);

        if (smallest - prev_smallest).abs() < tol && j > 0 {
            converged_iter = j + 1;
            break;
        }
        prev_smallest = smallest;

        if beta_new < 1e-14 {
            // Invariant subspace found
            converged_iter = j + 1;
            break;
        }
        // q_{j+1} = w / β_{j+1}
        let mut q_new = w;
        for x in &mut q_new {
            *x /= beta_new;
        }
        q_vecs.push(q_new);
    }

    // Now build dense tridiagonal eigensolver (Jacobi-symmetric on T) for vector recovery.
    let k = alphas.len();
    let mut t = vec![0.0; k * k];
    for i in 0..k {
        t[i * k + i] = alphas[i];
    }
    for i in 0..k - 1 {
        let b = betas[i + 1];
        t[i * k + i + 1] = b;
        t[(i + 1) * k + i] = b;
    }
    let (vals, vecs) = jacobi_symm(&mut t, k)?;
    // Find smallest eigenvalue index
    let (idx, &smallest) = vals
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| TnError::NumericalInstability("lanczos empty eigvals".into()))?;
    // Ritz vector: y_i = sum_j v_{j, idx} q_j
    let mut y = vec![0.0; n];
    for j in 0..k {
        let c = vecs[j * k + idx];
        for r in 0..n {
            y[r] += c * q_vecs[j][r];
        }
    }
    let yn = norm(&y);
    if yn > 1e-300 {
        for v in &mut y {
            *v /= yn;
        }
    }
    Ok(LanczosResult {
        eigenvalue: smallest,
        eigenvector: y,
        iter: converged_iter,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn axpy(y: &mut [f64], x: &[f64], a: f64) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// Eigenvalues of a symmetric tridiagonal matrix (alphas on diagonal, betas on off).
fn tridiagonal_eigenvalues(alphas: &[f64], betas: &[f64]) -> Vec<f64> {
    let k = alphas.len();
    if k == 0 {
        return Vec::new();
    }
    // Build dense and Jacobi-diagonalise
    let mut t = vec![0.0; k * k];
    for i in 0..k {
        t[i * k + i] = alphas[i];
    }
    for i in 0..k - 1 {
        let b = if i < betas.len() { betas[i] } else { 0.0 };
        t[i * k + i + 1] = b;
        t[(i + 1) * k + i] = b;
    }
    let (vals, _) = jacobi_symm(&mut t, k).unwrap_or_else(|_| (Vec::new(), Vec::new()));
    vals
}

/// Symmetric Jacobi eigendecomposition of an `n × n` matrix (row-major in-place).
/// Returns `(eigenvalues_ascending, V)` where columns of `V` are eigenvectors.
fn jacobi_symm(a: &mut [f64], n: usize) -> TnResult<(Vec<f64>, Vec<f64>)> {
    const MAX_SWEEPS: usize = 200;
    const TOL: f64 = 1.0e-14;
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for sweep in 0..MAX_SWEEPS {
        let mut max_off: f64 = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let av = a[p * n + q].abs();
                if av > max_off {
                    max_off = av;
                }
            }
        }
        if max_off < TOL {
            break;
        }
        if sweep == MAX_SWEEPS - 1 {
            return Err(TnError::NotConverged { iter: MAX_SWEEPS });
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < TOL {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta.abs() > 1.0e10 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let h = t * apq;
                a[p * n + p] = app - h;
                a[q * n + q] = aqq + h;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = c * arp - s * arq;
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = s * arp + c * arq;
                    a[q * n + r] = a[r * n + q];
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    // Sort ascending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        eigs[i]
            .partial_cmp(&eigs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_eigs: Vec<f64> = order.iter().map(|&i| eigs[i]).collect();
    let mut sorted_v = vec![0.0; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            sorted_v[row * n + new_col] = v[row * n + old_col];
        }
    }
    Ok((sorted_eigs, sorted_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanczos_identity_smallest() {
        let n = 5;
        let apply = |v: &[f64]| v.to_vec();
        let v0 = vec![1.0; n];
        let r = lanczos_smallest(apply, n, &v0, 10, 1e-12).expect("ok");
        assert!((r.eigenvalue - 1.0).abs() < 1e-8);
    }

    #[test]
    fn lanczos_diag_recovers() {
        let n = 5;
        let diag = [3.0, 1.0, 4.0, 1.5, 9.0];
        let apply = |v: &[f64]| {
            v.iter()
                .zip(diag.iter())
                .map(|(x, d)| x * d)
                .collect::<Vec<f64>>()
        };
        let v0 = vec![1.0; n];
        let r = lanczos_smallest(apply, n, &v0, n, 1e-12).expect("ok");
        assert!((r.eigenvalue - 1.0).abs() < 1e-8);
    }

    #[test]
    fn lanczos_dense_5x5() {
        // Build a known symmetric matrix from its spectral decomposition: H = sum_i λ_i |u_i><u_i|
        // We pick H = diag(1, 2, 3, 4, 5) for simplicity.
        let n = 5;
        let h: Vec<f64> = {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                m[i * n + i] = (i + 1) as f64;
            }
            m
        };
        let apply = |v: &[f64]| {
            let mut out = vec![0.0; n];
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += h[i * n + j] * v[j];
                }
                out[i] = acc;
            }
            out
        };
        let v0 = vec![1.0; n];
        let r = lanczos_smallest(apply, n, &v0, n, 1e-12).expect("ok");
        assert!((r.eigenvalue - 1.0).abs() < 1e-8);
    }
}
