//! Nuclear-norm matrix completion `min ||X||_* s.t. P_Ω(X) = P_Ω(M)` via SVT iteration.

use crate::error::CsResult;
use crate::matrix_completion::CompletionResult;
use crate::matrix_completion::svt::svt;

/// Nuclear-norm minimisation by SVT. Selects `tau` automatically as `tau = √(h * w) * scale` where
/// `scale` defaults to 1 if not specified.
pub fn nuclear_norm_minimization(
    m: &[f64],
    mask: &[bool],
    h: usize,
    w: usize,
    tau: Option<f64>,
    max_iter: usize,
    tol: f64,
) -> CsResult<CompletionResult> {
    let tau_val = tau.unwrap_or(((h * w) as f64).sqrt());
    let delta = 1.2_f64 * (h * w) as f64 / mask.iter().filter(|x| **x).count().max(1) as f64;
    svt(m, mask, h, w, tau_val, delta, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nuclear_norm_runs() {
        let m = vec![1.0, 2.0, 2.0, 4.0];
        let mask = vec![true, true, true, false];
        let r = nuclear_norm_minimization(&m, &mask, 2, 2, Some(1.0), 200, 1.0e-7).expect("ok");
        assert!(r.iterations > 0);
    }
}
