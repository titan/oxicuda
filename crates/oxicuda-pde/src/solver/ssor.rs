//! Symmetric Successive Over-Relaxation (SSOR) preconditioner.
//!
//! Given `A = D + L + U`, the SSOR preconditioner is
//! `M = (D/ω + L) · (D/ω)^{-1} · (D/ω + U)`.
//!
//! `ssor_apply` solves `M z = r` for `z` (used as preconditioning step inside PCG).

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::SparseCsr;

/// Apply the SSOR preconditioner to a residual `r`, returning `z = M^{-1} r`.
pub fn ssor_apply(a: &SparseCsr, r: &[f64], omega: f64) -> PdeResult<Vec<f64>> {
    let n = a.n_rows;
    if r.len() != n {
        return Err(PdeError::DimensionMismatch { a: r.len(), b: n });
    }
    if !(omega > 0.0 && omega < 2.0) {
        return Err(PdeError::InvalidParameter {
            name: "omega".into(),
            reason: "must satisfy 0 < omega < 2".into(),
        });
    }
    let diag = a.diagonal()?;
    for &d in &diag {
        if d.abs() < 1.0e-300 {
            return Err(PdeError::SingularMatrix("ssor: zero diagonal".into()));
        }
    }
    // Forward sweep: (D/omega + L) y = r
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        let mut s = r[i];
        for k in row_lo..row_hi {
            let j = a.cols[k];
            if j < i {
                s -= a.vals[k] * y[j];
            }
        }
        y[i] = s * omega / diag[i];
    }
    // Middle: scale by D / omega (i.e. multiply by D/omega^2 -- since we have y = (D/omega)^-1 * stuff)
    // We just multiply y by diag[i]/omega
    let mut z_mid = vec![0.0_f64; n];
    for i in 0..n {
        z_mid[i] = y[i] * diag[i] / omega;
    }
    // Backward sweep: (D/omega + U) z = z_mid
    let mut z = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        let mut s = z_mid[i];
        for k in row_lo..row_hi {
            let j = a.cols[k];
            if j > i {
                s -= a.vals[k] * z[j];
            }
        }
        z[i] = s * omega / diag[i];
    }
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssor_identity_is_identity() {
        let a =
            SparseCsr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).expect("ok");
        let r = vec![1.0, 2.0, 3.0];
        let z = ssor_apply(&a, &r, 1.0).expect("ok");
        for i in 0..3 {
            assert!((z[i] - r[i]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn ssor_omega_out_of_range_errors() {
        let a = SparseCsr::new(1, 1, vec![0, 1], vec![0], vec![1.0]).expect("ok");
        assert!(ssor_apply(&a, &[0.0], 0.0).is_err());
        assert!(ssor_apply(&a, &[0.0], 2.5).is_err());
    }
}
