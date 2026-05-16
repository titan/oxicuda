//! Generic backtracking line search (with simple sufficient-decrease test).

use crate::error::CvxResult;
use crate::linesearch::armijo::armijo_search;

/// Backtracking with default parameters (alpha0=1, rho=0.5, c1=1e-4, max_iter=50).
pub fn backtracking_search<F>(x: &[f64], d: &[f64], grad: &[f64], f: F) -> CvxResult<f64>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
{
    armijo_search(x, d, grad, f, 1.0, 0.5, 1.0e-4, 50)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CvxResult;

    #[test]
    fn backtracking_basic() {
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(x.iter().map(|v| v * v).sum::<f64>()) };
        let x = vec![3.0];
        let grad = vec![6.0];
        let d = vec![-6.0];
        let alpha = backtracking_search(&x, &d, &grad, f).expect("ok");
        assert!(alpha > 0.0);
    }
}
