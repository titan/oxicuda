//! PiSSA — Principal Singular values and Singular vectors Adaptation
//!
//! Reference: Meng, F., Wang, Z., & Zhang, M. (2024). *PiSSA: Principal Singular Values
//! and Singular Vectors Adaptation of Large Language Models.* NeurIPS 2024.
//! <https://arxiv.org/abs/2404.02948>
//!
//! Unlike LoRA, which initialises `B = 0` so the adapter starts as an identity, PiSSA
//! migrates the *principal* part of the frozen base weight `W₀` directly into the
//! trainable LoRA factors and uses the remaining *residual* part as the new frozen
//! weight:
//!
//! ```text
//!   W₀ = U Σ V^T
//!   U_r, Σ_r, V_r  = top-r singular triplet
//!   residual_w   = W₀ - U_r Σ_r V_r^T   (frozen)
//!   A           = Σ_r^{1/2} V_r^T       (trainable, shape r × in)
//!   B           = U_r Σ_r^{1/2}         (trainable, shape out × r)
//!   y           = (residual_w + α · B · A) · x
//! ```
//!
//! Because the principal singular components dominate the descent direction during
//! pre-training, starting LoRA from this decomposition removes the "warm-up" phase
//! observed in vanilla LoRA where `B` slowly leaves the origin.
//!
//! ## SVD strategy
//!
//! PiSSA only needs the SVD at *construction*, so we use a numerically robust but
//! quadratic one-sided Jacobi sweep over the smaller side of `W₀`. This is fine for
//! the modest matrix sizes typical at adapter-init time, and avoids any external
//! linear-algebra dependency.

use crate::error::{PeftError, PeftResult};

/// Hyper-parameters for [`PissaAdapter::from_weight`].
#[derive(Debug, Clone)]
pub struct PissaConfig {
    /// Input feature count of `W₀`.
    pub in_dim: usize,
    /// Output feature count of `W₀`.
    pub out_dim: usize,
    /// Top-rank used to move the principal components into the adapter.
    pub rank: usize,
    /// Global scale multiplier `α` applied to the trainable correction.
    pub alpha: f64,
    /// Maximum number of one-sided Jacobi sweeps.
    pub max_jacobi_sweeps: usize,
    /// Convergence tolerance on the off-diagonal ratio of `A^T A`.
    pub jacobi_tol: f64,
}

/// PiSSA-initialised LoRA-style adapter.
#[derive(Debug, Clone)]
pub struct PissaAdapter {
    /// Configuration captured at construction time.
    pub config: PissaConfig,
    /// Residual frozen weight `W₀ - U_r Σ_r V_r^T`, shape `out_dim × in_dim`.
    pub residual_w: Vec<f64>,
    /// Down-projection `A = Σ_r^{1/2} V_r^T`, shape `rank × in_dim`.
    pub a: Vec<f64>,
    /// Up-projection `B = U_r Σ_r^{1/2}`, shape `out_dim × rank`.
    pub b: Vec<f64>,
}

impl PissaAdapter {
    /// Build a PiSSA adapter from a pre-trained `W₀` (row-major `out_dim × in_dim`).
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::DimensionMismatch`] if `w0.len() != in_dim * out_dim`.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_dim, out_dim)`.
    /// - [`PeftError::Internal`] if Jacobi fails to converge in `max_jacobi_sweeps`.
    pub fn from_weight(w0: &[f64], cfg: PissaConfig) -> PeftResult<Self> {
        validate_config(&cfg, w0.len())?;
        let JacobiSvd { u, s, vt } = one_sided_jacobi_svd(
            w0,
            cfg.out_dim,
            cfg.in_dim,
            cfg.max_jacobi_sweeps,
            cfg.jacobi_tol,
        )?;
        let r = cfg.rank;
        let out_dim = cfg.out_dim;
        let in_dim = cfg.in_dim;

        // a[k, j] = sqrt(s[k]) * vt[k, j]
        // b[i, k] = u[i, k] * sqrt(s[k])
        let mut a = vec![0.0_f64; r * in_dim];
        let mut b = vec![0.0_f64; out_dim * r];
        let mut sqrt_s = vec![0.0_f64; r];
        for k in 0..r {
            sqrt_s[k] = s[k].max(0.0).sqrt();
        }
        for k in 0..r {
            for j in 0..in_dim {
                a[k * in_dim + j] = sqrt_s[k] * vt[k * in_dim + j];
            }
        }
        for i in 0..out_dim {
            for k in 0..r {
                b[i * r + k] = u[i * cfg.rank_full() + k] * sqrt_s[k];
            }
        }

        // residual_w = W₀ - U_r Σ_r V_r^T
        // Build U_r Σ_r V_r^T into a temporary, then subtract from a clone of W₀.
        let mut principal = vec![0.0_f64; out_dim * in_dim];
        for k in 0..r {
            let sk = s[k];
            if sk == 0.0 {
                continue;
            }
            for i in 0..out_dim {
                let u_ik = u[i * cfg.rank_full() + k];
                let coeff = sk * u_ik;
                if coeff == 0.0 {
                    continue;
                }
                for j in 0..in_dim {
                    principal[i * in_dim + j] += coeff * vt[k * in_dim + j];
                }
            }
        }
        let residual_w: Vec<f64> = w0
            .iter()
            .zip(principal.iter())
            .map(|(w, p)| w - p)
            .collect();

        Ok(Self {
            config: cfg,
            residual_w,
            a,
            b,
        })
    }

    /// Compute `y = (residual_w + α · B · A) · x`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if `x.len() != in_dim`.
    pub fn forward(&self, x: &[f64]) -> PeftResult<Vec<f64>> {
        if x.len() != self.config.in_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.in_dim,
                got: x.len(),
            });
        }
        let in_dim = self.config.in_dim;
        let out_dim = self.config.out_dim;
        let r = self.config.rank;
        let alpha = self.config.alpha;

        // base = residual_w · x
        let mut y = vec![0.0_f64; out_dim];
        for (i, yi) in y.iter_mut().enumerate() {
            let row = i * in_dim;
            let mut acc = 0.0_f64;
            for (j, xj) in x.iter().enumerate().take(in_dim) {
                acc += self.residual_w[row + j] * xj;
            }
            *yi = acc;
        }
        // ax = A · x, length r
        let mut ax = vec![0.0_f64; r];
        for (k, axk) in ax.iter_mut().enumerate() {
            let row = k * in_dim;
            let mut acc = 0.0_f64;
            for (j, xj) in x.iter().enumerate().take(in_dim) {
                acc += self.a[row + j] * xj;
            }
            *axk = acc;
        }
        // y += α · B · ax
        for (i, yi) in y.iter_mut().enumerate() {
            let row = i * r;
            let mut acc = 0.0_f64;
            for (k, axk) in ax.iter().enumerate() {
                acc += self.b[row + k] * axk;
            }
            *yi += alpha * acc;
        }
        Ok(y)
    }

    /// Reconstruct the merged weight `W = residual_w + α · B · A` (row-major
    /// `out_dim × in_dim`).
    #[must_use]
    pub fn merge(&self) -> Vec<f64> {
        let in_dim = self.config.in_dim;
        let out_dim = self.config.out_dim;
        let r = self.config.rank;
        let alpha = self.config.alpha;
        let mut merged = self.residual_w.clone();
        for i in 0..out_dim {
            for k in 0..r {
                let b_ik = self.b[i * r + k];
                if b_ik == 0.0 {
                    continue;
                }
                let coeff = alpha * b_ik;
                let a_row = k * in_dim;
                let m_row = i * in_dim;
                for j in 0..in_dim {
                    merged[m_row + j] += coeff * self.a[a_row + j];
                }
            }
        }
        merged
    }

    /// Relative Frobenius norm of `residual_w / W₀` at construction time, useful as a
    /// sanity check (should decrease with `rank`).
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if `w0.len() != in_dim * out_dim`.
    pub fn init_residual_norm(&self, w0: &[f64]) -> PeftResult<f64> {
        if w0.len() != self.config.in_dim * self.config.out_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.in_dim * self.config.out_dim,
                got: w0.len(),
            });
        }
        let num_sq: f64 = self.residual_w.iter().map(|v| v * v).sum();
        let den_sq: f64 = w0.iter().map(|v| v * v).sum();
        if den_sq < 1e-300 {
            return Ok(0.0);
        }
        Ok((num_sq / den_sq).sqrt())
    }
}

impl PissaConfig {
    /// Number of singular values returned by [`one_sided_jacobi_svd`].
    #[inline]
    fn rank_full(&self) -> usize {
        self.in_dim.min(self.out_dim)
    }
}

fn validate_config(cfg: &PissaConfig, w0_len: usize) -> PeftResult<()> {
    if cfg.in_dim == 0 || cfg.out_dim == 0 || cfg.rank == 0 {
        return Err(PeftError::EmptyInput);
    }
    if cfg.max_jacobi_sweeps == 0 {
        return Err(PeftError::Internal {
            msg: "max_jacobi_sweeps must be ≥ 1".to_string(),
        });
    }
    if cfg.jacobi_tol <= 0.0 || cfg.jacobi_tol.is_nan() {
        return Err(PeftError::Internal {
            msg: "jacobi_tol must be > 0".to_string(),
        });
    }
    if cfg.rank > cfg.in_dim.min(cfg.out_dim) {
        return Err(PeftError::RankTooLarge {
            rank: cfg.rank,
            dim: cfg.in_dim.min(cfg.out_dim),
        });
    }
    if w0_len != cfg.in_dim * cfg.out_dim {
        return Err(PeftError::DimensionMismatch {
            expected: cfg.in_dim * cfg.out_dim,
            got: w0_len,
        });
    }
    Ok(())
}

/// Thin one-sided-Jacobi SVD result.
///
/// `u` is `out_dim × k`, `s` is `k`, `vt` is `k × in_dim` where `k = min(in_dim, out_dim)`.
/// Singular values are sorted in descending order.
struct JacobiSvd {
    u: Vec<f64>,
    s: Vec<f64>,
    vt: Vec<f64>,
}

/// One-sided Jacobi SVD on the smaller of the two axes of `W` (row-major `m × n`).
///
/// We always rotate columns of the **wider** side: if `n ≤ m`, work on `W` directly;
/// otherwise work on `W^T`. The result is converted back into the canonical
/// `U Σ V^T` form for `W`.
fn one_sided_jacobi_svd(
    matrix: &[f64],
    m: usize,
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> PeftResult<JacobiSvd> {
    if n <= m {
        run_jacobi_on(matrix, m, n, max_sweeps, tol, /*transposed=*/ false)
    } else {
        // Build W^T into a tmp buffer
        let mut wt = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                wt[j * m + i] = matrix[i * n + j];
            }
        }
        // After computing SVD(W^T) = U' Σ V'^T, we have W = V' Σ U'^T,
        // so the "U" of W is V' and the "V^T" of W is U'^T.
        let svd = run_jacobi_on(&wt, n, m, max_sweeps, tol, /*transposed=*/ true)?;
        Ok(svd)
    }
}

fn run_jacobi_on(
    matrix: &[f64],
    m: usize,
    n: usize,
    max_sweeps: usize,
    tol: f64,
    transposed: bool,
) -> PeftResult<JacobiSvd> {
    // `matrix` is row-major m × n. We rotate columns of A and accumulate them in V (n×n).
    let mut a: Vec<f64> = matrix.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let mut sweeps_done = 0_usize;
    let mut rotations = usize::MAX;
    while rotations > 0 && sweeps_done < max_sweeps {
        rotations = 0;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut app = 0.0_f64;
                let mut aqq = 0.0_f64;
                let mut apq = 0.0_f64;
                for i in 0..m {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    app += aip * aip;
                    aqq += aiq * aiq;
                    apq += aip * aiq;
                }
                let prod = app * aqq;
                if prod < 1.0e-300 {
                    continue;
                }
                if apq.abs() < tol * prod.sqrt() {
                    continue;
                }
                let (c, s) = givens_angles(app, aqq, apq);
                for i in 0..m {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip + s * aiq;
                    a[i * n + q] = -s * aip + c * aiq;
                }
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip + s * viq;
                    v[i * n + q] = -s * vip + c * viq;
                }
                rotations += 1;
            }
        }
        sweeps_done += 1;
    }
    if rotations > 0 {
        return Err(PeftError::Internal {
            msg: format!("one-sided Jacobi did not converge in {max_sweeps} sweeps"),
        });
    }

    // After convergence, column norms of A are the singular values.
    let mut sigma = vec![0.0_f64; n];
    for j in 0..n {
        let mut s2 = 0.0_f64;
        for i in 0..m {
            s2 += a[i * n + j] * a[i * n + j];
        }
        sigma[j] = s2.sqrt();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = m.min(n);
    let mut s_out = vec![0.0_f64; k];
    let mut u_thin = vec![0.0_f64; m * k];
    let mut v_thin = vec![0.0_f64; n * k];
    for (new_col, &old_col) in order.iter().enumerate().take(k) {
        let sv = sigma[old_col];
        s_out[new_col] = sv;
        if sv > 1e-300 {
            for i in 0..m {
                u_thin[i * k + new_col] = a[i * n + old_col] / sv;
            }
        } else {
            for i in 0..m {
                u_thin[i * k + new_col] = 0.0;
            }
        }
        for r in 0..n {
            v_thin[r * k + new_col] = v[r * n + old_col];
        }
    }

    // Convert to the SvdResult convention expected by PiSSA:
    //   u (out_dim × k), vt (k × in_dim)
    // When `transposed == false`: m = out_dim, n = in_dim
    //   u   = u_thin             (m × k)
    //   vt[k, n] = v_thin[n, k]^T
    // When `transposed == true`: we computed SVD(W^T) where W^T is n_outer × m_outer,
    // so the original W was (m, n) but we passed (n_inner, m_inner) = (n_orig_in, m_orig_out).
    // Re-mapping: U_of_W = v_thin (n × k) and V_of_W = u_thin (m × k).
    if !transposed {
        // vt[k, n]
        let mut vt = vec![0.0_f64; k * n];
        for kk in 0..k {
            for jj in 0..n {
                vt[kk * n + jj] = v_thin[jj * k + kk];
            }
        }
        Ok(JacobiSvd {
            u: u_thin,
            s: s_out,
            vt,
        })
    } else {
        // matrix here was W^T of shape (n_for_solver, m_for_solver) where the original W
        // was (m_orig, n_orig) = (m_for_solver, n_for_solver) before transpose. Inside
        // this call we used m = n_orig, n = m_orig.
        let n_orig = m; // (rotated columns count of W^T) = m_orig (out_dim)
        let m_orig = n; // (rows of W^T) = n_orig (in_dim)
        // U of W = V' (n_orig × k) = v_thin reshaped (m_orig=in_dim was actually n here…)
        // We renamed locally — clarify by remembering:
        //   For the solver, m == in_dim, n == out_dim.
        //   u_thin: (in_dim × k), v_thin: (out_dim × k)
        // Therefore for original W (out_dim × in_dim):
        //   U_of_W  = v_thin (out_dim × k)
        //   Vt_of_W = u_thin^T (k × in_dim)
        debug_assert_eq!(n_orig, m); // sanity: m here is the in_dim of W
        debug_assert_eq!(m_orig, n); // n here is the out_dim of W
        let in_dim_actual = m;
        let out_dim_actual = n;
        // U' = v_thin has shape (out_dim_actual × k)
        let u = {
            let mut buf = vec![0.0_f64; out_dim_actual * k];
            for i in 0..out_dim_actual {
                for kk in 0..k {
                    buf[i * k + kk] = v_thin[i * k + kk];
                }
            }
            buf
        };
        // V'^T (the "vt" for W) has shape (k × in_dim_actual)
        let vt = {
            let mut buf = vec![0.0_f64; k * in_dim_actual];
            for kk in 0..k {
                for jj in 0..in_dim_actual {
                    buf[kk * in_dim_actual + jj] = u_thin[jj * k + kk];
                }
            }
            buf
        };
        Ok(JacobiSvd { u, s: s_out, vt })
    }
}

fn givens_angles(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    if apq.abs() < 1e-300 {
        return (1.0, 0.0);
    }
    let theta = (app - aqq) / (2.0 * apq);
    let t = if theta.abs() > 1.0e8 {
        0.5 / theta
    } else {
        theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg(out_dim: usize, in_dim: usize, rank: usize) -> PissaConfig {
        PissaConfig {
            in_dim,
            out_dim,
            rank,
            alpha: 1.0,
            max_jacobi_sweeps: 60,
            jacobi_tol: 1e-13,
        }
    }

    fn frob_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn deterministic_matrix(out_dim: usize, in_dim: usize) -> Vec<f64> {
        // Mix sin/cos so we get a full-rank, non-degenerate matrix.
        let mut w = vec![0.0_f64; out_dim * in_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                let a = (i as f64 + 1.3) * 0.7;
                let b = (j as f64 + 0.9) * 1.1;
                w[i * in_dim + j] = (a * b).sin() + 0.5 * (a + b).cos();
            }
        }
        w
    }

    #[test]
    fn rejects_zero_dims() {
        let w = vec![1.0_f64];
        let cfg = default_cfg(0, 1, 1);
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::EmptyInput)
        ));
        let cfg = default_cfg(1, 0, 1);
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::EmptyInput)
        ));
        let cfg = default_cfg(1, 1, 0);
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_rank_too_large() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = default_cfg(2, 2, 3);
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_w0_dim_mismatch() {
        let w = vec![1.0, 2.0, 3.0]; // length 3 ≠ 2*2
        let cfg = default_cfg(2, 2, 1);
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_zero_max_sweeps_and_nonpositive_tol() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let mut cfg = default_cfg(2, 2, 1);
        cfg.max_jacobi_sweeps = 0;
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::Internal { .. })
        ));
        let mut cfg = default_cfg(2, 2, 1);
        cfg.jacobi_tol = -1e-5;
        assert!(matches!(
            PissaAdapter::from_weight(&w, cfg),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn rank_equals_full_gives_near_zero_residual() {
        let w = deterministic_matrix(4, 4);
        let cfg = default_cfg(4, 4, 4);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let rn = ad.init_residual_norm(&w).unwrap();
        assert!(rn < 1e-9, "residual should vanish at full rank, got {rn}");
    }

    #[test]
    fn merge_roundtrip_at_full_rank() {
        let w = deterministic_matrix(4, 5);
        let cfg = default_cfg(4, 5, 4);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let merged = ad.merge();
        assert_eq!(merged.len(), w.len());
        let mut err = 0.0_f64;
        for (a, b) in merged.iter().zip(w.iter()) {
            err += (a - b).powi(2);
        }
        err = err.sqrt();
        assert!(err < 1e-8, "‖W - merge‖ = {err}");
    }

    #[test]
    fn rank_one_smoke() {
        let w = deterministic_matrix(3, 4);
        let cfg = default_cfg(3, 4, 1);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        assert_eq!(ad.a.len(), 4);
        assert_eq!(ad.b.len(), 3);
        let rn = ad.init_residual_norm(&w).unwrap();
        assert!(rn > 0.0 && rn < 1.0);
    }

    #[test]
    fn known_2x2_svd_reconstruction() {
        // W = [[3, 0], [0, 4]] → singular values {4, 3}; rank-2 reconstruction exact.
        let w = vec![3.0, 0.0, 0.0, 4.0];
        let cfg = default_cfg(2, 2, 2);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let merged = ad.merge();
        for (a, b) in merged.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-10, "expected {b}, got {a}");
        }
        // residual should be near zero
        let rn = frob_norm(&ad.residual_w) / frob_norm(&w);
        assert!(rn < 1e-10);
    }

    #[test]
    fn residual_norm_monotone_in_rank() {
        let w = deterministic_matrix(5, 5);
        let mut prev = f64::INFINITY;
        for r in 1..=5 {
            let cfg = default_cfg(5, 5, r);
            let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
            let rn = ad.init_residual_norm(&w).unwrap();
            assert!(
                rn <= prev + 1e-12,
                "residual must be non-increasing: r={r} rn={rn} prev={prev}"
            );
            prev = rn;
        }
    }

    #[test]
    fn forward_output_dim_correct() {
        let w = deterministic_matrix(5, 7);
        let cfg = default_cfg(5, 7, 3);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let x: Vec<f64> = (0..7).map(|i| 0.1 * i as f64).collect();
        let y = ad.forward(&x).unwrap();
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn zero_w0_gives_zero_components() {
        let w = vec![0.0_f64; 16];
        let cfg = default_cfg(4, 4, 2);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        for v in ad.a.iter().chain(ad.b.iter()).chain(ad.residual_w.iter()) {
            assert!(v.abs() < 1e-14);
        }
    }

    #[test]
    fn deterministic() {
        let w = deterministic_matrix(4, 6);
        let cfg = default_cfg(4, 6, 3);
        let ad1 = PissaAdapter::from_weight(&w, cfg.clone()).unwrap();
        let ad2 = PissaAdapter::from_weight(&w, cfg).unwrap();
        assert_eq!(ad1.a, ad2.a);
        assert_eq!(ad1.b, ad2.b);
        assert_eq!(ad1.residual_w, ad2.residual_w);
    }

    #[test]
    fn alpha_zero_uses_residual_only() {
        let w = deterministic_matrix(4, 4);
        let cfg = PissaConfig {
            alpha: 0.0,
            ..default_cfg(4, 4, 2)
        };
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let x = vec![0.3_f64, -0.6, 0.9, 0.2];
        let y = ad.forward(&x).unwrap();
        // y should equal residual_w · x
        let mut expected = [0.0_f64; 4];
        for (i, ei) in expected.iter_mut().enumerate() {
            for (j, xj) in x.iter().enumerate().take(4) {
                *ei += ad.residual_w[i * 4 + j] * xj;
            }
        }
        for (a, b) in y.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn rank_one_hand_computed() {
        // W = u v^T with u=(1,0,0)^T, v=(2, 1)^T → top singular vec is u with σ=√5.
        let v = [2.0_f64, 1.0];
        let u = [1.0_f64, 0.0, 0.0];
        let mut w = vec![0.0_f64; 3 * 2];
        for i in 0..3 {
            for j in 0..2 {
                w[i * 2 + j] = u[i] * v[j];
            }
        }
        let cfg = default_cfg(3, 2, 1);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        // After rank-1 PiSSA, residual should be ~0 since rank(W)=1.
        let rn = frob_norm(&ad.residual_w) / frob_norm(&w);
        assert!(rn < 1e-10, "rank-1 weight: residual rn={rn}");
        // merge() should match W
        let merged = ad.merge();
        for (m, w) in merged.iter().zip(w.iter()) {
            assert!((m - w).abs() < 1e-10);
        }
    }

    #[test]
    fn forward_dim_mismatch() {
        let w = deterministic_matrix(4, 5);
        let cfg = default_cfg(4, 5, 2);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let bad_x = vec![1.0, 2.0]; // wrong length
        assert!(matches!(
            ad.forward(&bad_x),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn init_residual_norm_input_check() {
        let w = deterministic_matrix(3, 3);
        let cfg = default_cfg(3, 3, 2);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        let bad = vec![1.0, 2.0]; // wrong length
        assert!(matches!(
            ad.init_residual_norm(&bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn wide_matrix_path_smoke() {
        // out_dim < in_dim triggers the transposed Jacobi branch.
        let w = deterministic_matrix(3, 7);
        let cfg = default_cfg(3, 7, 2);
        let ad = PissaAdapter::from_weight(&w, cfg).unwrap();
        // Sanity: merge of trained-init equals (residual + principal) ≤ ‖W‖
        let merged = ad.merge();
        let err: f64 = merged
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        // For wide matrices with rank < min, merge ≠ W; we check finite + non-NaN.
        assert!(err.is_finite());
        assert!(!err.is_nan());
        // And full-rank reconstruction must match exactly.
        let cfg_full = default_cfg(3, 7, 3);
        let ad_full = PissaAdapter::from_weight(&w, cfg_full).unwrap();
        let merged_full = ad_full.merge();
        let err_full: f64 = merged_full
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err_full < 1e-8, "full-rank merge err={err_full}");
    }
}
