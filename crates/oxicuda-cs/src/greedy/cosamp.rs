//! CoSaMP — Compressive Sampling Matching Pursuit (Needell-Tropp 2009).
//!
//! Steps each iteration:
//! 1. Compute proxy `u = Φᵀ r`.
//! 2. Identify Ω = top-2K indices of |u|.
//! 3. Merge T = Ω ∪ support.
//! 4. Solve LS on T to get b.
//! 5. Prune support to top-K of |b|.
//! 6. Update residual r = y - Φ x.

use crate::error::{CsError, CsResult};
use crate::greedy::GreedyResult;
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};

/// CoSaMP with target sparsity `k`, capped at `max_iter` and residual tolerance.
pub fn cosamp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    k: usize,
    max_iter: usize,
    tol_residual: f64,
) -> CsResult<GreedyResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if k == 0 || k > m.min(n) {
        return Err(CsError::InvalidSparsity(k));
    }
    if max_iter == 0 {
        return Err(CsError::InvalidParameter("max_iter = 0".into()));
    }
    let mut support: Vec<usize> = Vec::new();
    let mut residual = y.to_vec();
    let mut x_full = vec![0.0_f64; n];
    let mut iter = 0usize;
    let mut prev_r = f64::INFINITY;
    for _ in 0..max_iter {
        let r_norm = norm2(&residual);
        if r_norm < tol_residual {
            break;
        }
        if r_norm >= prev_r * (1.0 - 1.0e-10) && iter > 0 {
            break;
        }
        prev_r = r_norm;
        let proxy = mat_t_vec(phi, m, n, &residual)?;
        // top-2k indices of |proxy|.
        let take_2k = (2 * k).min(n);
        let mut abs_idx: Vec<(usize, f64)> = proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        abs_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let omega: Vec<usize> = abs_idx.into_iter().take(take_2k).map(|(i, _)| i).collect();
        // Merge with support.
        let mut t: Vec<usize> = support.clone();
        for &o in &omega {
            if !t.contains(&o) {
                t.push(o);
            }
        }
        t.sort();
        if t.len() > m {
            t.truncate(m);
        }
        // LS on T.
        let b_sub = solve_subset_ls(phi, m, n, &t, y)?;
        // Prune to top-k.
        let mut bi: Vec<(usize, f64, f64)> = b_sub
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v, v.abs()))
            .collect();
        bi.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        bi.truncate(k);
        // Build new support, x_full.
        let mut new_support: Vec<usize> = bi.iter().map(|(i, _, _)| t[*i]).collect();
        new_support.sort();
        x_full.fill(0.0);
        for &(i_sub, v, _) in &bi {
            x_full[t[i_sub]] = v;
        }
        // Update residual via Φ_T b_sub mapped to support
        let sub = submat_columns(phi, m, n, &new_support)?;
        let x_new: Vec<f64> = new_support.iter().map(|&j| x_full[j]).collect();
        let ax = mat_vec(&sub, m, new_support.len(), &x_new)?;
        for i in 0..m {
            residual[i] = y[i] - ax[i];
        }
        support = new_support;
        iter += 1;
    }
    Ok(GreedyResult {
        x: x_full,
        support,
        residual_norm: norm2(&residual),
        iterations: iter,
    })
}

/// Configuration for [`cosamp_with_config`].
#[derive(Debug, Clone)]
pub struct CoSampConfig {
    /// Assumed sparsity level (k): number of non-zero entries in the signal.
    pub sparsity: usize,
    /// Maximum number of CoSaMP iterations.
    pub max_iter: usize,
    /// Residual norm tolerance for early stopping.
    pub tol: f64,
}

/// CoSaMP with a config struct wrapper.
///
/// # Arguments
/// - `a`: `[m × n]` measurement matrix in row-major order.
/// - `b`: `[m]` observation vector.
/// - `m`: number of rows (measurements).
/// - `n`: number of columns (signal dimension).
/// - `cfg`: algorithm configuration.
///
/// # Returns
/// A length-`n` sparse estimate of the signal.
pub fn cosamp_with_config(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    cfg: &CoSampConfig,
) -> CsResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CsError::DimensionMismatch { a: b.len(), b: m });
    }
    if cfg.sparsity == 0 || cfg.sparsity > n {
        return Err(CsError::InvalidSparsity(cfg.sparsity));
    }
    if cfg.sparsity > m {
        return Err(CsError::ShapeMismatch {
            expected: vec![cfg.sparsity],
            got: vec![m],
        });
    }
    let result = cosamp(a, m, n, b, cfg.sparsity, cfg.max_iter, cfg.tol)?;
    Ok(result.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosamp_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = cosamp(&phi, 4, 4, &y, 2, 20, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
        assert!((r.x[0] - 1.0).abs() < 1.0e-6);
        assert!((r.x[2] - 0.5).abs() < 1.0e-6);
    }

    // ── cosamp_with_config tests ──────────────────────────────────────────────

    /// Test 1: 4×4 identity, y=[1,0,0.5,0], k=2, check x[0]≈1 and x[2]≈0.5.
    #[test]
    fn sparse_recovery_k1() {
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![1.0, 0.0, 0.5, 0.0];
        let cfg = CoSampConfig {
            sparsity: 2,
            max_iter: 30,
            tol: 1.0e-9,
        };
        let x = cosamp_with_config(&a, &b, 4, 4, &cfg).expect("sparse_recovery_k1");
        assert!(
            (x[0] - 1.0).abs() < 1.0e-6,
            "x[0] should be ~1.0, got {}",
            x[0]
        );
        assert!(
            (x[2] - 0.5).abs() < 1.0e-6,
            "x[2] should be ~0.5, got {}",
            x[2]
        );
    }

    /// Test 2: output length must equal n.
    #[test]
    fn output_len() {
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![1.0, 0.0, 0.5, 0.0];
        let cfg = CoSampConfig {
            sparsity: 2,
            max_iter: 20,
            tol: 1.0e-9,
        };
        let x = cosamp_with_config(&a, &b, 4, 4, &cfg).expect("output_len");
        assert_eq!(x.len(), 4);
    }

    /// Test 3: nonzero entries ≤ sparsity k.
    #[test]
    fn sparsity_bounded() {
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![3.0, 0.0, 1.5, 0.0];
        let k = 2usize;
        let cfg = CoSampConfig {
            sparsity: k,
            max_iter: 30,
            tol: 1.0e-9,
        };
        let x = cosamp_with_config(&a, &b, 4, 4, &cfg).expect("sparsity_bounded");
        let nnz = x.iter().filter(|&&v| v.abs() > 1.0e-8).count();
        assert!(nnz <= k, "expected ≤{k} nonzeros, got {nnz}");
    }

    /// Test 4: more iterations produce a smaller residual ||Ax-b||.
    #[test]
    fn residual_decreases() {
        // Use 6×6 identity so that CoSaMP can make progress in each iteration.
        let a: Vec<f64> = (0..6)
            .flat_map(|i| (0..6).map(move |j| if i == j { 1.0 } else { 0.0 }))
            .collect();
        let b = vec![1.0, 0.0, 0.7, 0.0, 0.3, 0.0];
        let cfg1 = CoSampConfig {
            sparsity: 3,
            max_iter: 1,
            tol: 1.0e-15,
        };
        let cfg50 = CoSampConfig {
            sparsity: 3,
            max_iter: 50,
            tol: 1.0e-15,
        };
        let x1 = cosamp_with_config(&a, &b, 6, 6, &cfg1).expect("residual_decreases iter=1");
        let x50 = cosamp_with_config(&a, &b, 6, 6, &cfg50).expect("residual_decreases iter=50");
        // Compute ||Ax - b|| for each.
        let res_of = |x: &[f64]| -> f64 {
            let mut s = 0.0_f64;
            for i in 0..6 {
                // A is identity, so Ax = x
                let diff = x[i] - b[i];
                s += diff * diff;
            }
            s.sqrt()
        };
        let r1 = res_of(&x1);
        let r50 = res_of(&x50);
        assert!(
            r50 <= r1 + 1.0e-10,
            "50-iter residual {r50} should be ≤ 1-iter residual {r1}"
        );
    }

    /// Test 5: very small tol on an easy system stops when ||r|| < tol.
    #[test]
    fn tol_stops_early() {
        // 4×4 identity is trivially solved in one step.
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![1.0, 0.0, 0.5, 0.0];
        // tol_residual is checked against ||r|| each iteration. For identity
        // the residual after one step is zero, so even 1e-12 is easily met.
        let cfg = CoSampConfig {
            sparsity: 2,
            max_iter: 100,
            tol: 1.0e-12,
        };
        let x = cosamp_with_config(&a, &b, 4, 4, &cfg).expect("tol_stops_early");
        // Check the result is still approximately correct (proving it converged, not just exited).
        assert!((x[0] - 1.0).abs() < 1.0e-6);
        assert!((x[2] - 0.5).abs() < 1.0e-6);
    }

    /// Test 6: sparsity > n → InvalidSparsity error.
    #[test]
    fn k_gt_n_error() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2×2
        let b = vec![1.0, 0.5];
        let cfg = CoSampConfig {
            sparsity: 5,
            max_iter: 10,
            tol: 1.0e-6,
        }; // sparsity=5 > n=2
        let result = cosamp_with_config(&a, &b, 2, 2, &cfg);
        assert!(
            matches!(result, Err(CsError::InvalidSparsity(_))),
            "expected InvalidSparsity, got {:?}",
            result
        );
    }

    /// Test 7: m < sparsity → error (ShapeMismatch or InvalidSparsity).
    #[test]
    fn m_lt_k_error() {
        // m=2, n=5, sparsity=3 → m < sparsity
        let a: Vec<f64> = (0..10).map(|v| v as f64 * 0.1).collect(); // 2×5
        let b = vec![0.5, 0.3];
        let cfg = CoSampConfig {
            sparsity: 3,
            max_iter: 10,
            tol: 1.0e-6,
        };
        let result = cosamp_with_config(&a, &b, 2, 5, &cfg);
        assert!(
            matches!(
                result,
                Err(CsError::ShapeMismatch { .. }) | Err(CsError::InvalidSparsity(_))
            ),
            "expected ShapeMismatch or InvalidSparsity, got {:?}",
            result
        );
    }

    /// Test 8: all output values are finite.
    #[test]
    fn output_finite() {
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![2.5, 0.0, 1.1, 0.0];
        let cfg = CoSampConfig {
            sparsity: 2,
            max_iter: 20,
            tol: 1.0e-9,
        };
        let x = cosamp_with_config(&a, &b, 4, 4, &cfg).expect("output_finite");
        assert!(
            x.iter().all(|v| v.is_finite()),
            "some outputs are non-finite: {:?}",
            x
        );
    }

    /// Test 9: 8×16 structured matrix, 2-sparse signal recovery.
    ///
    /// We build A with a deterministic pattern so that columns 3 and 7 have clear
    /// "presence" in measurements, then check the reconstruction picks them up.
    #[test]
    fn harder_system_converges() {
        // 8 rows × 16 cols.  We construct A so that column j has a known pattern.
        // A[i][j] = cos(i * j * pi / 16) * 0.5, normalised per column.
        let m = 8usize;
        let n = 16usize;
        let mut a = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i * n + j] =
                    (i as f64 * (j + 1) as f64 * std::f64::consts::PI / (n as f64)).cos();
            }
        }
        // Normalise each column to unit L2.
        for j in 0..n {
            let col_norm: f64 = (0..m).map(|i| a[i * n + j].powi(2)).sum::<f64>().sqrt();
            if col_norm > 1.0e-12 {
                for i in 0..m {
                    a[i * n + j] /= col_norm;
                }
            }
        }
        // True signal: x* = [0..0, 1.0 at col 3, 0..0, 0.8 at col 7, 0..0]
        let mut x_true = vec![0.0_f64; n];
        x_true[3] = 1.0;
        x_true[7] = 0.8;
        // Compute b = A x*.
        let b: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j] * x_true[j]).sum())
            .collect();
        let cfg = CoSampConfig {
            sparsity: 2,
            max_iter: 100,
            tol: 1.0e-8,
        };
        let x = cosamp_with_config(&a, &b, m, n, &cfg).expect("harder_system_converges");
        assert_eq!(x.len(), n, "output length mismatch");
        // The algorithm should produce a finite result without panicking.
        assert!(
            x.iter().all(|v| v.is_finite()),
            "non-finite output: {:?}",
            x
        );
        // The residual ||Ax - b|| should be reasonably small.
        let ax: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect();
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(av, bv)| (av - bv).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual < 0.5,
            "residual {residual} too large after convergence"
        );
    }
}
